"""SY-* checks: what the symbol itself carries about its version, level and mask.

``SY-VERSION-MIN`` reads the version off the sampled grid, ``SY-EC-MATCH`` the level from two
independent sources, ``SY-MASK`` the mask out of the format information, and
``SY-MATRIX-EQUIV`` compares the whole grid with the golden encoder at pinned parameters. A
target's ``declared_*`` metadata is evidence only and never reaches a verdict.

Counts: 240 corpus cases for each of the first three checks, 8 synthetic pinned cases for the
fourth -- 728 cases, 728 adapter invocations and 728 primary decodes, which is what
:data:`EXPECTED_ENCODES` declares.

Reasons this family emits: ``sampling_failed`` when no grid, level or mask could be read;
``value_mismatch`` when a version, level, mask or module differs from the expectation;
``format_info_disagreement`` when the decoder's level and the format information disagree.
Rejection, unusable response and unsupported request are translated by
:func:`qrcheck.checks.support.case_status_for_error`; ``TargetUnavailable`` and
``TargetTimeout`` propagate so the runner owns the one-error-result-per-check policy.

Layering: :mod:`qrcheck.checks.support`, the registry and result-model vocabulary, the
decoded-evidence package, the module-count arithmetic and :mod:`qrcheck.targets.base` only.
Nothing here reads a random source, ``hash()``, ``id()`` or the filesystem, or asserts a
duration; cases are yielded in record-``id`` order and the matrix set by ``case_id``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import segno

import qrcheck.checks.support as support
from qrcheck.decode import DecodeResult, Matrix, matrix_sha256
from qrcheck.decode.matrix import module_count_for_version, version_from_dimension
from qrcheck.registry.check import DEPENDENCY_PRIMARY_DECODER, Case, RunContext, check
from qrcheck.report.model import CaseResult, Reason
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Type-only: the corpus builder imports numpy for its payload generators.
    from qrcheck.corpus.builder import CorpusRecord

__all__ = [
    "EXPECTED_ENCODES",
    "SY_EC_MATCH",
    "SY_MASK",
    "SY_MATRIX_EQUIV",
    "SY_VERSION_MIN",
]

# The four stable identifiers, spelled once and derived from nowhere: the registry owns the
# literal tuple of all twenty-one (``CHECK_IDS``) and validates these, so a typo fails the import
# and the decorator, the case identifiers and EXPECTED_ENCODES cannot disagree.
_VERSION_MIN_ID: Final[str] = "SY-VERSION-MIN"
_EC_MATCH_ID: Final[str] = "SY-EC-MATCH"
_MASK_ID: Final[str] = "SY-MASK"
_MATRIX_EQUIV_ID: Final[str] = "SY-MATRIX-EQUIV"

# The family that owns the SY- prefix.  The registry checks the pair, and the name is also
# the JUnit ``classname`` suffix, so it is a stable output surface.
_FAMILY: Final[str] = "symbol"

# Case-identifier prefixes.  AAP §0.3.4 fixes both: a corpus-driven case is named after its
# record, a constructed one after the name its family gave it.  The corpus form is repeated here
# because a case's callable is not handed its own identifier.
_CORPUS_PREFIX: Final[str] = "corpus:"
_SYNTHETIC_PREFIX: Final[str] = "synthetic:"

# ISO/IEC 18004 §7.8 defines exactly eight mask patterns, numbered 0-7.  The range is a
# property of the standard rather than a threshold this product chooses, which is why it
# lives here and not in ``qrcheck.toml``: it is what an unpinned request is held to.
_MASK_MINIMUM: Final[int] = 0
_MASK_MAXIMUM: Final[int] = 7

# The eight synthetic matrix cases: four versions x two payloads, each paired with one of two
# distinct legal masks.  Pinning the version, the error level and the mask is what makes
# module-by-module equality legitimate (AAP §0.3.5); nothing else in the product compares matrices.
# The payloads are two to five bytes so they fit Version 1 at every error level -- at EC H,
# Version 1 holds seven bytes in byte mode and ten characters in alphanumeric mode -- and stay
# clear of the 64-byte window of the defective target's truncation injection.
_MATRIX_VERSIONS: Final[tuple[int, ...]] = (1, 2, 7, 10)
_MATRIX_PAYLOADS: Final[tuple[tuple[str, bytes], ...]] = (("a", b"QR7"), ("b", b"1A2-B"))
_MATRIX_MASKS: Final[tuple[int, int]] = (2, 5)


class _MatrixCase(NamedTuple):
    """One pinned matrix case: the pins it sends, its payload and its name.

    A record rather than a loop body, so the case count, the identifiers and the expectations are
    all read off one table.
    """

    version: int
    label: str
    payload: bytes
    mask: int

    @property
    def case_id(self) -> str:
        """Return the ``synthetic:`` identifier AAP §0.3.4 fixes for this case."""
        return f"{_SYNTHETIC_PREFIX}matrix-v{self.version}-{self.label}"


_MATRIX_CASES: Final[tuple[_MatrixCase, ...]] = tuple(
    _MatrixCase(version=version, label=label, payload=payload, mask=mask)
    for version in _MATRIX_VERSIONS
    for (label, payload), mask in zip(_MATRIX_PAYLOADS, _MATRIX_MASKS, strict=True)
)

# The corpus's record total: the fixed workload AAP §0.8.1 defines for every corpus-driven
# check, and one of the two numbers EXPECTED_ENCODES is made of.
# :data:`qrcheck.corpus.builder.TOTAL_RECORDS` owns this count, and this is the same 240 rather
# than a second definition: importing
# the builder would pull its numpy-bearing payload generators into ``qrcheck list``.  The
# integration gate compares the executed case count against EXPECTED_ENCODES.
_CORPUS_RECORDS: Final[int] = 240

# The adapter-invocation count each check is expected to make in one full run, read by the
# invocation-count gate through a counting stub: a retry registers as one more and fails it (AAP
# §0.4.3, §0.5.6).  The corpus counts are that record total, the matrix count the case table's
# size, so neither number is typed twice.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        _EC_MATCH_ID: _CORPUS_RECORDS,
        _MASK_ID: _CORPUS_RECORDS,
        _MATRIX_EQUIV_ID: len(_MATRIX_CASES),
        _VERSION_MIN_ID: _CORPUS_RECORDS,
    }
)


# A judge reads one decoded symbol and returns the failure reason (``None`` for a pass) with the
# measurements the family adds to the evidence -- kept apart from result construction so no judge
# decides a duration, an evidence path or a status the model does not own.
_Judge = Callable[[DecodeResult, GenerateRequest], tuple[Reason | None, dict[str, Any]]]

# What a case does with the context and the argument its `cases` generator bound to it.
_CaseHandler = Callable[[RunContext, Any], CaseResult]


def _oracle(
    payload: bytes,
    ec_level: str,
    *,
    version: int | None = None,
    mask: int | None = None,
) -> segno.QRCode:
    """Return the golden-encoder symbol an expectation is read from.

    The oracle decides what a conforming target should have emitted and never produces an artefact:
    ``boost_error`` is always ``False`` (a boosted level would fail a conforming target) and the
    call is always ``make_qr``, Model 2 by construction, so no Micro QR symbol can reach a geometry
    assertion. ``version`` and ``mask`` are the case's pins, or ``None`` for segno's own choice.

    Raises:
        ValueError: segno refuses the payload or a parameter -- a case-definition defect here rather
            than a target's fault.
    """
    return segno.make_qr(
        payload,
        error=ec_level,
        version=version,
        mask=mask,
        boost_error=False,
    )


def _oracle_version(payload: bytes, ec_level: str) -> int:
    """Return the smallest Model 2 version that holds `payload` at `ec_level`.

    segno's ascending version search, which ``tests/unit/test_capacity_oracle.py`` holds equal to
    ``segno.consts`` for Versions 1-40.

    Raises:
        ValueError: Exactly as :func:`_oracle`.
    """
    return _oracle(payload, ec_level).version


def _oracle_grid(symbol: segno.QRCode) -> tuple[tuple[bool, ...], ...]:
    """Return the oracle symbol's module matrix as immutable rows of booleans.

    The same shape :func:`_grid` gives a sampled grid -- ``N`` rows of ``N`` booleans,
    ``N = 4V + 17``, a dark module being ``True`` -- so the two compare in plain Python.
    """
    return _grid(symbol.matrix)


def _grid(matrix: Matrix | Sequence[Sequence[int]]) -> tuple[tuple[bool, ...], ...]:
    """Return `matrix` as immutable rows of booleans.

    Accepts a sampled grid (numpy ``bool_``) or an oracle matrix (rows of ``0``/``1``) and
    normalises both by truthiness, so the comparison never depends on which integer a producer
    used for "dark"; normalising here also keeps the array stack out of this module.
    """
    return tuple(tuple(bool(module) for module in row) for row in matrix)


def _difference(
    sampled: Sequence[Sequence[bool]],
    oracle: Sequence[Sequence[bool]],
) -> tuple[int, tuple[int, int] | None]:
    """Return how many modules differ and where the first difference is.

    Both grids must have the same dimensions; every caller checks that first, because a difference
    in dimension is a different verdict from a difference in a module.
    """
    differing = 0
    first: tuple[int, int] | None = None
    for row_index, (sampled_row, oracle_row) in enumerate(zip(sampled, oracle)):
        for column_index, (sampled_module, oracle_module) in enumerate(
            zip(sampled_row, oracle_row)
        ):
            if sampled_module != oracle_module:
                differing += 1
                if first is None:
                    first = (row_index, column_index)
    return differing, first


def _bound(handler: _CaseHandler, argument: Any) -> Callable[[RunContext], CaseResult]:
    """Return the case callable that runs `handler` with `argument` bound to it.

    A case's callable receives only the run's context, so a family closes over its subject here
    rather than through a nested closure per check.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run the case `handler` describes against the run's context."""
        return handler(ctx, argument)

    return execute


def _elapsed_ms(started: float) -> float:
    """Return milliseconds elapsed since `started`, to three decimals.

    Recorded on every result and asserted by nothing here: AAP §0.5.5 excludes durations from
    every correctness assertion, and ``PF-*`` alone reads time.
    """
    return round((time.perf_counter() - started) * 1000.0, 3)


def _evidence(decoded: DecodeResult, **measurements: Any) -> dict[str, Any]:
    """Return the decoded-evidence mapping: the projection plus this check's findings.

    ``decoded.json`` receives it verbatim, so every entry is JSON-safe: the projection carries
    ``per_decoder``, ``polarity_applied``, ``agreement``, ``ec_level_decoder``, ``ec_level_format``,
    ``version``, ``mask`` and ``matrix_sha256`` -- the evidence table AAP §0.4.5 requires -- and
    the measurements on top are scalars, lists or ``None``.
    """
    evidence = decoded.as_dict()
    evidence.update(measurements)
    return evidence


def _run_case(
    ctx: RunContext,
    check_id: str,
    case_id: str,
    request: GenerateRequest,
    *,
    expected: Mapping[str, Any],
    judge: _Judge,
    request_extra: Mapping[str, Any] | None = None,
) -> CaseResult:
    """Generate one symbol, decode it, judge it and persist evidence if it did not pass.

    The single path every check here takes, so "one invocation, one decode" is a property of one
    function. A declared rejection, an unusable response and an unsupported request are translated
    by :func:`qrcheck.checks.support.case_status_for_error`; ``TargetUnavailable`` and
    ``TargetTimeout`` are deliberately not caught, because the runner owns the one-error-result-per-
    check policy.

    Args:
        ctx: The run's context.
        check_id, case_id: The check's and the case's stable identifiers.
        request: The request to send, built through :func:`qrcheck.checks.support.build`.
        expected: What the case asserted, recorded beside the outcome.
        judge: The assertion, returning a failure reason or ``None`` plus its measurements.
        request_extra: Further facts for ``request.json``, such as the oracle digest used.

    Raises:
        TargetUnavailable: The target could not be reached at all; propagates unchanged.
        TargetTimeout: No response within the configured timeout; propagates unchanged.
        ValueError: A field of the result is outside its documented range.
    """
    started = time.perf_counter()
    try:
        symbol, decoded = support.generate_and_decode(ctx, request)
    except (TargetRejected, TargetContractError, TargetUnsupported) as error:
        return support.case_status_for_error(
            check_id,
            case_id,
            error,
            request=request,
            duration_ms=_elapsed_ms(started),
            seed=ctx.seed,
        )
    duration_ms = _elapsed_ms(started)
    reason, measurements = judge(decoded, request)
    evidence = _evidence(decoded, **measurements)
    projection = request.as_dict()
    if reason is None:
        result = support.pass_(
            check_id,
            case_id,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    else:
        result = support.fail(
            check_id,
            case_id,
            reason=reason,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    # The store owns the persistence policy and returns a passing result untouched, so the
    # artefact travels with every case and only a non-passing one is written.
    return support.record_evidence(ctx, result, symbol=symbol, request_extra=request_extra)



def _corpus_case_id(record: CorpusRecord) -> str:
    """Return the ``corpus:<record-id>`` identifier AAP §0.3.4 fixes for this case.

    A case's callable is not handed its own identifier, so the one spelling the registry also
    uses is repeated here rather than guessed at.
    """
    return f"{_CORPUS_PREFIX}{record.id}"


def _expected_version(request: GenerateRequest, oracle_version: int) -> int:
    """Return the version the symbol is required to have.

    An **unpinned** request is held to the oracle's smallest fitting version -- the assertion that
    the generator chose the smallest symbol rather than a larger legal one. A **pinned** request is
    held to the pin, because a target declaring ``version_pin`` is expected to honour it. The 240
    corpus cases carry no pin, so the oracle minimum governs all of them.
    """
    return oracle_version if request.version is None else request.version


@check(
    id=_VERSION_MIN_ID,
    title="Version is the smallest symbol that fits the payload, read from the module grid",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.ec_level}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def SY_VERSION_MIN(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per corpus record: the symbol is the smallest version that fits.

    ``ec_level`` is required because the smallest fitting version depends on the level: a payload
    that fits Version 5 at EC L needs Version 6 at EC M. The version is never taken from the
    target or a decoder -- none reports one -- but from the sampled grid's dimension, the
    product's only version source (AAP IM-1). Cases are ascending by record ``id``.
    """
    return support.corpus_cases(
        ctx,
        _VERSION_MIN_ID,
        ctx.corpus,
        execute_factory=lambda record: _bound(_version_min_case, record),
    )


def _version_min_case(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Assert that one corpus record's symbol is the smallest version that fits it.

    Returns ``value_mismatch`` when the sampled version differs from the expectation,
    ``sampling_failed`` when no grid could be recovered, and a pass otherwise.

    Raises:
        TargetUnavailable, TargetTimeout: Exactly as :func:`_run_case`.
        ValueError: The payload does not fit even Version 40 at the run's level, which would be a
            corpus defect rather than a target's.
    """
    case_id = _corpus_case_id(record)
    payload = record.payload_bytes
    request = support.build(payload, ctx.ec_level)
    oracle_version = _oracle_version(payload, ctx.ec_level)
    expected_version = _expected_version(request, oracle_version)
    expected = {
        "ec_level": request.ec_level,
        "module_count": module_count_for_version(expected_version),
        "oracle_version": oracle_version,
        "pinned_version": request.version,
        "version": expected_version,
    }

    def judge(
        decoded: DecodeResult, request: GenerateRequest
    ) -> tuple[Reason | None, dict[str, Any]]:
        """Compare the symbol's own version with the expectation."""
        measurements: dict[str, Any] = {
            "expected_version": expected_version,
            "module_count_for_expected_version": module_count_for_version(expected_version),
            "oracle_version": oracle_version,
            "request_ec_level": request.ec_level,
            "sampled_version": decoded.version,
        }
        if decoded.version is None:
            # No grid could be recovered, so the symbol has no measured version: a reason
            # rather than a verdict, which is what AAP §0.8.1 requires of this path.
            return Reason.SAMPLING_FAILED, measurements
        if decoded.version != expected_version:
            return Reason.VALUE_MISMATCH, measurements
        return None, measurements

    return _run_case(ctx, _VERSION_MIN_ID, case_id, request, expected=expected, judge=judge)


@check(
    id=_EC_MATCH_ID,
    title=(
        "Error-correction level from the decoder and from the format information "
        "matches the request"
    ),
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.ec_level}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def SY_EC_MATCH(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per corpus record: the symbol carries the level that was requested.

    Both independent readings are asserted: the primary decoder's ``ec_level`` metadata and the
    15-bit format-information field the framework reads itself
    (:mod:`qrcheck.decode.format_info`). A symbol whose level is wrong shows it in both; one the
    decoder mis-reports shows the disagreement instead of a wrong verdict.
    """
    return support.corpus_cases(
        ctx,
        _EC_MATCH_ID,
        ctx.corpus,
        execute_factory=lambda record: _bound(_ec_match_case, record),
    )


def _ec_match_case(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Assert the error-correction level of one corpus record's symbol.

    The two sources are compared **with each other** first, so a disagreement is
    ``format_info_disagreement`` rather than a mismatch against the request (AAP §0.4.5's reason
    order), and only then is the agreed level compared with the request. A missing reading is a
    failure of its own: an absent source can never be read as agreement.

    Raises:
        TargetUnavailable, TargetTimeout: Exactly as :func:`_run_case`.
        ValueError: The payload does not fit even Version 40 at the run's level.
    """
    case_id = _corpus_case_id(record)
    request = support.build(record.payload_bytes, ctx.ec_level)
    expected = {
        "ec_level": request.ec_level,
        "sources": ["decoder", "format_info"],
    }

    def judge(
        decoded: DecodeResult, request: GenerateRequest
    ) -> tuple[Reason | None, dict[str, Any]]:
        """Compare both readings of the level with each other, then with the request."""
        measurements: dict[str, Any] = {
            "ec_level_decoder": decoded.ec_level_decoder,
            "ec_level_format": decoded.ec_level_format,
            "request_ec_level": request.ec_level,
            "sampled_version": decoded.version,
        }
        if decoded.ec_level_decoder is None or decoded.ec_level_format is None:
            return Reason.SAMPLING_FAILED, measurements
        if decoded.ec_level_decoder != decoded.ec_level_format:
            return Reason.FORMAT_INFO_DISAGREEMENT, measurements
        if decoded.ec_level_decoder != request.ec_level:
            return Reason.VALUE_MISMATCH, measurements
        return None, measurements

    return _run_case(ctx, _EC_MATCH_ID, case_id, request, expected=expected, judge=judge)


@check(
    id=_MASK_ID,
    title="Mask pattern from the format information is legal and honours a pin",
    family=_FAMILY,
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def SY_MASK(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per corpus record: the mask in the symbol is a legal mask.

    No decoder in the pinned set reports a mask, so it is read from the format information alone.
    Mask *selection* is encoder-defined (AAP §0.3.5), so an unpinned request is held only to the
    legal range ``0..7``; a request that pins a mask is held to the pin. The 240 corpus cases pin
    nothing, so the pin branch is a guard, and ``SY-MATRIX-EQUIV`` is where a pin is observed.
    """
    return support.corpus_cases(
        ctx,
        _MASK_ID,
        ctx.corpus,
        execute_factory=lambda record: _bound(_mask_case, record),
    )


def _mask_case(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Assert the mask pattern of one corpus record's symbol.

    The mask is only ever the format information's reading: the framework's own sampling, never a
    decoder and never the target's metadata. Reasons: ``sampling_failed`` when it could not be read,
    ``value_mismatch`` when the mask is outside ``0..7`` or contradicts a pin, a pass otherwise.

    Raises:
        TargetUnavailable, TargetTimeout: Exactly as :func:`_run_case`.
        ValueError: The payload does not fit even Version 40 at the run's level.
    """
    case_id = _corpus_case_id(record)
    request = support.build(record.payload_bytes, ctx.ec_level)
    expected = {
        "legal_range": [_MASK_MINIMUM, _MASK_MAXIMUM],
        "mask_pin": request.mask,
    }

    def judge(
        decoded: DecodeResult, request: GenerateRequest
    ) -> tuple[Reason | None, dict[str, Any]]:
        """Compare the format information's mask with the legal range and any pin."""
        measurements: dict[str, Any] = {
            "mask_pin": request.mask,
            "sampled_version": decoded.version,
        }
        if decoded.mask is None:
            return Reason.SAMPLING_FAILED, measurements
        if not _MASK_MINIMUM <= decoded.mask <= _MASK_MAXIMUM:
            return Reason.VALUE_MISMATCH, measurements
        if request.mask is not None and decoded.mask != request.mask:
            return Reason.VALUE_MISMATCH, measurements
        return None, measurements

    return _run_case(ctx, _MASK_ID, case_id, request, expected=expected, judge=judge)


@check(
    id=_MATRIX_EQUIV_ID,
    title="Module matrix equals the oracle at a pinned version, error level and mask",
    family=_FAMILY,
    requires=frozenset(
        {
            Capability.png,
            Capability.ec_level,
            Capability.version_pin,
            Capability.mask_pin,
        }
    ),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def SY_MATRIX_EQUIV(ctx: RunContext) -> Iterator[Case]:
    """Yield the eight synthetic cases that compare a whole module matrix with the oracle.

    The product's **only** matrix comparison, legitimate only because every request pins the
    version, the error level and the mask: at pinned parameters the standard fixes the symbol
    completely, whereas an unpinned mask is the encoder's choice and two conforming encoders may
    legitimately differ (AAP §0.3.5). Ascending by ``case_id``.
    """
    cases = [
        support.synthetic_case(_MATRIX_EQUIV_ID, spec.case_id, _bound(_matrix_case, spec))
        for spec in _MATRIX_CASES
    ]
    return iter(sorted(cases, key=lambda case: case.case_id))


def _matrix_case(ctx: RunContext, spec: _MatrixCase) -> CaseResult:
    """Compare one pinned symbol's sampled grid with the oracle's matrix, module by module.

    Three assertions, in order: the grid's dimension is exactly ``4V + 17``;
    ``version_from_dimension`` reads the pinned version back off it; and every one of the ``N x N``
    modules equals the oracle's. The first differing ``(row, column)`` and the differing module
    count are recorded. Reasons: ``sampling_failed`` when no grid could be recovered,
    ``value_mismatch`` otherwise.

    Raises:
        TargetUnavailable, TargetTimeout: Exactly as :func:`_run_case`.
        ValueError: The payload does not fit the pinned version at the run's level.
    """
    payload = spec.payload
    request = support.build(payload, ctx.ec_level, version=spec.version, mask=spec.mask)
    oracle = _oracle_grid(
        _oracle(payload, ctx.ec_level, version=spec.version, mask=spec.mask)
    )
    oracle_digest = matrix_sha256(oracle)
    expected_module_count = module_count_for_version(spec.version)
    expected = {
        "ec_level": request.ec_level,
        "mask": spec.mask,
        "module_count": expected_module_count,
        "oracle_matrix_sha256": oracle_digest,
        "version": spec.version,
    }
    # The oracle digest and the expected dimension reach request.json, so a reader can
    # regenerate the expectation from what the evidence records without reading this module.
    request_extra = {
        "expected_module_count": expected_module_count,
        "oracle_matrix_sha256": oracle_digest,
    }

    def judge(
        decoded: DecodeResult, request: GenerateRequest
    ) -> tuple[Reason | None, dict[str, Any]]:
        """Compare the sampled grid with the oracle's matrix at the pinned parameters."""
        measurements: dict[str, Any] = {
            "expected_module_count": expected_module_count,
            "expected_version": spec.version,
            "oracle_matrix_sha256": oracle_digest,
            "request_ec_level": request.ec_level,
        }
        if decoded.matrix is None:
            return Reason.SAMPLING_FAILED, measurements
        sampled = _grid(decoded.matrix)
        measurements["sampled_module_count"] = len(sampled)
        measurements["sampled_matrix_sha256"] = matrix_sha256(sampled)
        measurements["sampled_mask"] = decoded.mask
        measurements["sampled_version"] = decoded.version
        measurements["sampled_version_from_dimension"] = version_from_dimension(len(sampled))
        if len(sampled) != expected_module_count:
            return Reason.VALUE_MISMATCH, measurements
        if version_from_dimension(len(sampled)) != spec.version:
            return Reason.VALUE_MISMATCH, measurements
        differing, first = _difference(sampled, oracle)
        measurements["differing_modules"] = differing
        measurements["first_difference"] = None if first is None else [first[0], first[1]]
        if differing:
            return Reason.VALUE_MISMATCH, measurements
        return None, measurements

    return _run_case(
        ctx,
        _MATRIX_EQUIV_ID,
        spec.case_id,
        request,
        expected=expected,
        judge=judge,
        request_extra=request_extra,
    )
