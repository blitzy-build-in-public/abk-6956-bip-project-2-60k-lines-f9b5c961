"""SY-* checks: what the symbol itself says about its version, level and mask.

The four checks here assert what the *symbol* carries, never what the target claims about
it.  ``SY-VERSION-MIN`` reads the version off the sampled module grid, ``SY-EC-MATCH``
reads the error-correction level from two independent sources, ``SY-MASK`` reads the mask
out of the 15-bit format information, and ``SY-MATRIX-EQUIV`` compares the whole sampled
grid against the golden encoder at the one set of parameters where such a comparison is
legitimate.  The target's own ``declared_version``, ``declared_ec`` and ``declared_mask``
are evidence only and are never read by this module: a target that lies about its metadata
is recorded as having lied rather than believed.

Family and case matrix (AAP §0.8.1)
-----------------------------------
===========================  =================================  =====  =======  =======
Check                        Case source                          Cases  Encodes  Decodes
===========================  =================================  =====  =======  =======
``SY-VERSION-MIN``           all 240 corpus records                 240      240      240
``SY-EC-MATCH``              all 240 corpus records                 240      240      240
``SY-MASK``                  all 240 corpus records                 240      240      240
``SY-MATRIX-EQUIV``          8 synthetic pinned cases                 8        8        8
===========================  =================================  =====  =======  =======

Every case invokes the adapter exactly once and decodes the artefact it returned exactly
once -- 728 encodes and 728 primary decodes for the four checks together, which is the
``encodes``/``decodes`` column of the plan's case matrix and the count
:data:`EXPECTED_ENCODES` declares for the invocation-count gate.  No case retries, and
nothing in this module calls the target a second time for the same case.

What each check asserts, and the reasons it can emit
----------------------------------------------------
``SY-VERSION-MIN``
    The version recovered from the sampled grid dimension (``N = 4V + 17``) equals the
    smallest version that holds the payload at the requested level.  The oracle minimum
    governs an unpinned request; a version pin replaces it, because a target that declares
    the ``version_pin`` capability is expected to honour one.  Fails with
    ``sampling_failed`` when no grid could be recovered and ``value_mismatch`` when the
    version differs from the expectation.  This is the smallest-symbol assertion: a
    generator that always emits Version 40 would pass ``RT-ROUNDTRIP`` and fail here.
``SY-EC-MATCH``
    Both independent readings of the error-correction level -- the primary decoder's
    ``ec_level`` metadata and the level decoded from the symbol's own format information --
    equal the requested level.  Fails with ``format_info_disagreement`` when the two
    sources disagree with *each other*, with ``value_mismatch`` when they agree and differ
    from the request, and with ``sampling_failed`` when either reading is absent (an
    absent source is a failure, never a pass).
``SY-MASK``
    The mask read from the format information is a legal mask, and equals the pinned mask
    when the request pinned one.  Fails with ``sampling_failed`` when the format
    information could not be read and ``value_mismatch`` when the mask is outside ``0..7``
    or contradicts a pin.
``SY-MATRIX-EQUIV``
    At a pinned version, error level and mask, the sampled grid equals the golden
    encoder's matrix module for module, and its dimension is exactly ``4V + 17`` while
    ``version_from_dimension`` reads back the pinned version.  Fails with
    ``sampling_failed`` when no grid could be recovered and ``value_mismatch`` when the
    dimension, the version read off it, or any module differs -- the first differing
    ``(row, column)`` is recorded so a reader can see the defect.

The oracle discipline: two segno defaults would corrupt every expectation
-----------------------------------------------------------------------
The golden encoder is used strictly as an oracle, and both of its convenient defaults are
disabled at every call site of this module:

* ``boost_error`` defaults to ``True`` and spends spare capacity on a **higher** error
  level without changing the version, so an oracle built with the default would expect a
  level the target was never asked for and would fail a conforming target (AAP §0.3.5,
  IM-2).  Every call passes ``boost_error=False``.
* ``segno.make()`` returns a **Micro QR** symbol whenever the payload fits, and a Micro
  symbol has no ``4V + 17`` dimension, no Version 1-40 number and a different
  format-information layout, so every geometry assertion here would be wrong on exactly
  the short payloads a corpus is full of.  This module therefore calls
  ``segno.make_qr`` -- Model 2 by construction -- and never ``segno.make``.  The pinned
  release (1.6.6) gives ``make_qr`` no ``micro`` argument to pass, so the discipline is
  the function choice itself rather than a keyword.

Both rules are contract, not style: ``tests/unit/test_capacity_oracle.py`` holds the
capacity tables this module's expectations rest on equal to ``segno.consts`` for Versions
1-40, and the eight-case matrix set is what proves the render -> sample -> compare path
end to end.

What this module deliberately never asserts (AAP §0.3.5)
--------------------------------------------------------
* Matrix equality, a specific mask, or a specific version for an **unpinned** request.
  Mask selection is encoder-defined, so asserting a mask value for an unpinned request
  produces false failures against a conforming target; the unpinned mask assertion is the
  legal range ``0..7``, and matrix comparison happens only inside ``SY-MATRIX-EQUIV``,
  where the request pins the version, the error level and the mask.
* Anything the target declares about itself.  ``declared_*`` fields never reach a status.
* Any duration.  ``duration_ms`` is recorded on every result and asserted by nothing here;
  timing is ``PF-*``'s business alone.
* Anything about the payload's bytes.  ``RT-ROUNDTRIP`` owns round-trip fidelity; a check
  that also compared payloads here would report one defect twice and would blur which
  assertion failed.

Proving the checks can fail: the defective target
-------------------------------------------------
``qrcheck/targets/defective.py`` injects five defects, and AAP §0.4.2 fixes exactly nine
checks that must fail against it.  Three are in this family: ``SY-EC-MATCH`` (D1 encodes
one level below the request), ``SY-VERSION-MIN`` (at a capacity boundary that downshift
fits the same payload in a smaller version) and ``SY-MATRIX-EQUIV`` (a pinned level cannot
produce the requested matrix).  ``SY-MASK`` must keep **passing**: D4 honours an explicit
pin and still yields a legal mask for an unpinned request, so the mask assertion is
satisfied.  No check here is weakened, skipped, retried or relaxed to accommodate that
target, and no assertion was added that would make ``SY-MASK`` fail there: the injections
and their expected failing set are acceptance criteria in their own right.

No project rules exist for this product: the review-rules source returns no user rules and
AAP §0.12 records the same, so the binding constraints are the AAP's MUST/MUST NOT list and
this package's invariants, honoured here as if they were rules.

Layering and import cost
------------------------
The check layer sees the port and nothing behind it.  This module imports
:mod:`qrcheck.checks.support`, the registry's case and context types, the result model's
vocabulary, the decoded-evidence package (:class:`~qrcheck.decode.DecodeResult`,
:func:`~qrcheck.decode.matrix_sha256`, :data:`~qrcheck.decode.Matrix`), the module-count
arithmetic that is the product's only version source, and from ``qrcheck.targets`` only
:mod:`qrcheck.targets.base` -- the protocol, the capability vocabulary and the three adapter
errors the per-case translation table names.  Nothing from the corpus is imported at run
time: the builder's record type is named for the annotations alone, and the 240-record total
is a declared case count, so no payload generator enters this module's import graph.
**No concrete adapter is imported**, no decoder, no image library, no transport client and
no configuration reader: ``tests/unit/test_layering.py`` holds that mechanically for every
module under ``qrcheck/checks/``.

Third-party imports are exactly ``segno``, used as the oracle only.  ``numpy`` is *not*
imported here even though the sampled grid is a numpy array: grids are compared and hashed
through :func:`~qrcheck.decode.matrix_sha256`, which normalises whatever it is given, so no
name from the array stack appears in this module and no array arithmetic is written here.
The decoded-evidence package this module imports does load numpy for the decoders; what is
avoided is a dependency of this module's *logic* on it, not the load itself.  The corpus
builder is likewise reached for a type only, never for its numpy-bearing payload generators.

Determinism
-----------
Nothing here reads a clock, a random source, ``hash()``, ``id()`` or the filesystem, and
nothing sorts by anything but a stable identifier: corpus cases are yielded in record-``id``
order by :func:`qrcheck.checks.support.corpus_cases` and the synthetic cases are sorted by
``case_id`` before they are yielded.  The mask and payload tables below are declared once,
in a fixed order, so the eight matrix cases are the same eight on every machine.  Each
result records the run seed, so a case is reproducible from ``summary.json`` alone.

Evidence
--------
Evidence is persisted only through :func:`qrcheck.checks.support.record_evidence`, with the
artefact's bytes so ``symbol.png`` is captured for a failing case and, for the matrix set,
the oracle's digest and the expected module count are added to ``request.json`` through
``request_extra``.  ``decoded.json`` receives the whole
:meth:`qrcheck.decode.DecodeResult.as_dict` projection -- ``per_decoder``, the polarity,
the agreement, the EC-level pair, the version, the mask and ``matrix_sha256`` -- plus this
family's measurements, which is the evidence table AAP §0.4.5 requires for the
``wrong_ec_level`` defect: ``ec_level_decoder``, ``ec_level_format`` and
``request_ec_level`` side by side.  This module writes no file itself.
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

# The four stable identifiers, spelled once here and derived from nowhere.  The registry
# owns the literal tuple of all twenty-one (``CHECK_IDS``) and validates these against it,
# so a typo fails the import rather than producing a plausible-looking report; these names
# exist so the decorator, the case identifiers and EXPECTED_ENCODES cannot disagree.
_VERSION_MIN_ID: Final[str] = "SY-VERSION-MIN"
_EC_MATCH_ID: Final[str] = "SY-EC-MATCH"
_MASK_ID: Final[str] = "SY-MASK"
_MATRIX_EQUIV_ID: Final[str] = "SY-MATRIX-EQUIV"

# The family that owns the SY- prefix.  The registry checks the pair, and the name is also
# the JUnit ``classname`` suffix, so it is a stable output surface.
_FAMILY: Final[str] = "symbol"

# Case-identifier prefixes.  AAP §0.3.4 fixes both: a corpus-driven case is named after its
# record, a constructed one carries the name this family gave it.  The corpus form is
# repeated here because a case's callable is not handed its own identifier, and the two
# spellings must agree or one case would be one name in the registry and another in a
# result.
_CORPUS_PREFIX: Final[str] = "corpus:"
_SYNTHETIC_PREFIX: Final[str] = "synthetic:"

# ISO/IEC 18004 §7.8 defines exactly eight mask patterns, numbered 0-7.  The range is a
# property of the standard rather than a threshold this product chooses, which is why it
# lives here and not in ``qrcheck.toml``: it is what an unpinned request is held to.
_MASK_MINIMUM: Final[int] = 0
_MASK_MAXIMUM: Final[int] = 7

# The eight synthetic matrix cases: four versions × two payloads, each paired with one of
# two distinct legal masks.  Pinning the version, the error level and the mask is what makes
# module-by-module equality a legitimate assertion (AAP §0.3.5); nothing else in the product
# compares matrices.  The payloads are two to five bytes so they fit Version 1 at every
# error level -- at EC H, Version 1 holds seven bytes in byte mode and ten characters in
# alphanumeric mode -- and so they are shorter than the 64-byte window of the defective
# target's truncation injection, which therefore can never reach them.
_MATRIX_VERSIONS: Final[tuple[int, ...]] = (1, 2, 7, 10)
_MATRIX_PAYLOADS: Final[tuple[tuple[str, bytes], ...]] = (("a", b"QR7"), ("b", b"1A2-B"))
_MATRIX_MASKS: Final[tuple[int, int]] = (2, 5)


class _MatrixCase(NamedTuple):
    """One pinned matrix case: the pins it sends, the payload it sends them with, and its name.

    A record rather than a loop body so the case count, the identifiers and the expectations
    are all read off one table: :data:`_MATRIX_CASES` is built from the three tables above,
    and :data:`EXPECTED_ENCODES` counts it rather than restating ``8``.
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
# check, and one of the two numbers EXPECTED_ENCODES is made of.  ``CorpusError``-guarded
# ``qrcheck.corpus.builder.TOTAL_RECORDS`` is the owner of this count, and this is the same
# 240 rather than a second definition of it: it is declared here as a case count because
# importing the builder would reach the numpy-bearing payload generators, and a check module
# has no business pulling the payload corpus into the import graph of ``qrcheck list``.  The
# integration gate compares the executed case count against EXPECTED_ENCODES, so a corpus
# that changed size would fail there rather than silently here.
_CORPUS_RECORDS: Final[int] = 240

# The adapter-invocation count each check is expected to make in one full run, read by the
# invocation-count gate through a counting stub: a retry registers as one more than this and
# fails the gate, which is how "retry until green" cannot be introduced quietly (AAP §0.4.3,
# §0.5.6).  The corpus-driven counts are that record total and the matrix count is the size of
# the case table above, so neither number is typed twice.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        _EC_MATCH_ID: _CORPUS_RECORDS,
        _MASK_ID: _CORPUS_RECORDS,
        _MATRIX_EQUIV_ID: len(_MATRIX_CASES),
        _VERSION_MIN_ID: _CORPUS_RECORDS,
    }
)


# A judge reads one decoded symbol and returns the reason the case failed for (``None`` for
# a pass) together with the measurements the family adds to the decoded evidence.  Kept
# separate from result construction so each assertion reads as the assertion it is, and so
# no judge can decide a duration, an evidence path or a status the model does not own.
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

    The oracle is independent of the target under test and is never used to produce an
    artefact: it decides what a conforming target should have emitted.  ``boost_error`` is
    always ``False`` -- see the module docstring for why a boosted level would fail a
    conforming target -- and the call is always ``make_qr``, which is Model 2 by
    construction, so no Micro QR symbol can ever reach a geometry assertion.

    ``version`` and ``mask`` default to ``None``, which asks segno to choose them: that is
    what an unpinned request's expectation is made of.  When a case pins them, the pin is
    passed through, so the oracle matrix describes the symbol the request asked for.

    Args:
        payload: The payload bytes the request carried.
        ec_level: The requested error-correction level.
        version: The pinned version, or ``None`` to let segno choose the smallest fitting one.
        mask: The pinned mask, or ``None`` to let segno choose one.

    Returns:
        The oracle symbol, whose ``version``, ``error``, ``mask`` and ``matrix`` the
        expectations are read from.

    Raises:
        ValueError: segno refuses the payload or a parameter -- a payload that does not fit
            the pinned version at the level, for instance.  That is a case-definition defect
            in this module rather than a target's fault, so the runner turns it into an
            ``error`` result carrying the traceback instead of a plausible-looking verdict.
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

    The expectation ``SY-VERSION-MIN`` is built on: segno's own ascending version search,
    which is the same arithmetic the corpus's capacity tables carry and which
    ``tests/unit/test_capacity_oracle.py`` holds equal to ``segno.consts`` for Versions
    1-40.

    Args:
        payload: The payload bytes the request carried.
        ec_level: The requested error-correction level.

    Returns:
        The version in 1-40.

    Raises:
        ValueError: Exactly as :func:`_oracle`.
    """
    return _oracle(payload, ec_level).version


def _oracle_grid(symbol: segno.QRCode) -> tuple[tuple[bool, ...], ...]:
    """Return the oracle symbol's module matrix as immutable rows of booleans.

    The same shape :func:`_grid` gives a sampled matrix, so the two grids compare module for
    module in plain Python; a dark module is ``True``, matching
    :data:`qrcheck.decode.base.Matrix`.

    Args:
        symbol: The oracle symbol.

    Returns:
        ``N`` rows of ``N`` booleans, ``N = 4V + 17``.
    """
    return _grid(symbol.matrix)


def _grid(matrix: Matrix | Sequence[Sequence[int]]) -> tuple[tuple[bool, ...], ...]:
    """Return `matrix` as immutable rows of booleans.

    Accepts either a sampled grid (a numpy array of ``bool_``) or an oracle matrix (a tuple
    of ``bytearray`` rows of ``0``/``1``), and normalises both to the same plain-Python
    shape: truthiness becomes ``True``, so a decoder-independent comparison never depends on
    which integer a producer used for "dark".  Normalising here instead of importing numpy
    keeps this module's import graph free of the image stack.

    Args:
        matrix: The grid to normalise, in either producer's representation.

    Returns:
        The rows of the grid as tuples of booleans.
    """
    return tuple(tuple(bool(module) for module in row) for row in matrix)


def _difference(
    sampled: Sequence[Sequence[bool]],
    oracle: Sequence[Sequence[bool]],
) -> tuple[int, tuple[int, int] | None]:
    """Return how many modules differ and where the first difference is.

    Both grids must have the same dimensions; every caller checks that first, because a
    difference in dimension is a different verdict from a difference in a module.

    Args:
        sampled: The sampled grid, from :func:`_grid`.
        oracle: The oracle grid, from :func:`_oracle_grid`.

    Returns:
        The number of differing modules -- ``0`` when the grids are equal -- and the first
        differing ``(row, column)`` coordinate, or ``None`` when nothing differs.
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

    A case's callable receives only the run's context, so a family closes over what the case
    is about -- a corpus record, or one row of the matrix case table -- through this one
    helper rather than through a nested closure per check.

    Args:
        handler: The function that produces the case's result.
        argument: The case's own subject: a corpus record or a :class:`_MatrixCase`.

    Returns:
        The callable a :class:`~qrcheck.registry.check.Case` holds.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run the case `handler` describes against the run's context."""
        return handler(ctx, argument)

    return execute


def _elapsed_ms(started: float) -> float:
    """Return milliseconds elapsed since `started`, rounded to the report's precision.

    Recorded on every result and asserted by nothing in this module: AAP §0.5.5 excludes
    durations from every correctness assertion, and ``PF-*`` alone reads time.

    Args:
        started: The ``time.perf_counter()`` reading taken before the case began.

    Returns:
        A non-negative duration in milliseconds, to the millisecond.
    """
    return round((time.perf_counter() - started) * 1000.0, 3)


def _evidence(decoded: DecodeResult, **measurements: Any) -> dict[str, Any]:
    """Return the decoded-evidence mapping this family records: the projection plus findings.

    ``decoded.json`` receives the result's own ``decoded`` mapping verbatim, so every entry
    here has to be JSON-safe; :meth:`qrcheck.decode.DecodeResult.as_dict` guarantees that
    for the projection, and the measurements added on top are scalars, lists or ``None``.
    The projection always carries ``per_decoder``, ``polarity_applied``, ``agreement``,
    ``ec_level_decoder``, ``ec_level_format``, ``version``, ``mask`` and ``matrix_sha256``,
    which is the evidence table AAP §0.4.5 requires.

    Args:
        decoded: The decoded evidence for the case.
        **measurements: The check's own findings, added beside the projection.

    Returns:
        A plain, JSON-serialisable mapping.
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
    """Generate one symbol, decode it, judge it and persist the evidence if it did not pass.

    The single path every check in this module takes, so "one invocation, one decode" is a
    property of one function rather than a convention four checks are trusted to follow.

    Three adapter errors are per-case outcomes and are translated by
    :func:`qrcheck.checks.support.case_status_for_error`: a declared rejection, an unusable
    response and an unsupported request each become the failed or skipped result the
    translation table prescribes.  :class:`~qrcheck.targets.base.TargetUnavailable` and
    :class:`~qrcheck.targets.base.TargetTimeout` are deliberately **not** caught: the runner
    owns the one-error-result-per-check policy on target loss, and it can only apply it if it
    sees the error, so converting them here would let a dead target look like a defective
    one.

    Args:
        ctx: The run's context.
        check_id: The check's stable identifier.
        case_id: The case's stable identifier.
        request: The request to send, built through :func:`qrcheck.checks.support.build`.
        expected: What the case asserted, recorded beside the outcome.
        judge: The assertion, returning a failure reason or ``None`` plus its measurements.
        request_extra: Further identity-adjacent facts to record in ``request.json``, such as
            the oracle digest a matrix comparison used.

    Returns:
        The case's result, carrying its evidence directory when anything was persisted.

    Raises:
        TargetUnavailable: The target could not be reached at all; propagates unchanged.
        TargetTimeout: No response within the configured timeout; propagates unchanged.
        ValueError: A field of the result is outside its documented range, which means a case
            definition in this module is wrong.
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

    The registry's case carries the same name, built by
    :func:`qrcheck.checks.support.corpus_cases`; a case's callable is not handed its own
    identifier, so the one spelling is repeated here rather than guessed at.
    """
    return f"{_CORPUS_PREFIX}{record.id}"


def _expected_version(request: GenerateRequest, oracle_version: int) -> int:
    """Return the version the symbol is required to have.

    Two cases, stated explicitly because the difference is the whole point of the check:

    * an **unpinned** request is held to the oracle's smallest fitting version -- the
      assertion that the generator chose the smallest symbol rather than a larger legal
      one;
    * a **pinned** request is held to the pin, because a target that declares the
      ``version_pin`` capability is expected to honour it.  The oracle minimum is then
      irrelevant: a generator legitimately encodes a short payload into a pinned larger
      version by padding it.

    The 240 corpus cases carry no pin, so the oracle minimum governs all of them.

    Args:
        request: The request about to be sent.
        oracle_version: The smallest version that holds the payload at the requested level.

    Returns:
        The version the sampled grid must show.
    """
    return oracle_version if request.version is None else request.version


@check(
    id=_VERSION_MIN_ID,
    title="Version is the smallest symbol that fits the payload, read from the module grid",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.ec_level}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def SY_VERSION_MIN(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per corpus record: the symbol is the smallest version that fits.

    ``ec_level`` is required because the smallest fitting version depends on the level: a
    payload that fits Version 5 at EC L needs Version 6 at EC M, so a target that cannot be
    asked for a level cannot be held to a version.  The version itself is never taken from
    the target or from a decoder -- no pinned decoder reports one -- but from the sampled
    grid's dimension, which is the product's only version source (AAP IM-1).

    Args:
        ctx: The run's context; its corpus is already filtered and sorted by the runner.

    Returns:
        The cases, ascending by record ``id``.
    """
    return support.corpus_cases(
        ctx,
        _VERSION_MIN_ID,
        ctx.corpus,
        execute_factory=lambda record: _bound(_version_min_case, record),
    )


def _version_min_case(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Assert that one corpus record's symbol is the smallest version that fits it.

    Args:
        ctx: The run's context.
        record: The corpus record this case is about.

    Returns:
        The case's result: ``value_mismatch`` when the sampled version differs from the
        expectation, ``sampling_failed`` when no grid could be recovered, a pass otherwise.

    Raises:
        TargetUnavailable, TargetTimeout: Exactly as :func:`_run_case`.
        ValueError: The payload does not fit even Version 40 at the run's level, which would
            be a corpus defect rather than a target's.
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
)
def SY_EC_MATCH(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per corpus record: the symbol carries the level that was requested.

    Both independent readings are asserted.  The primary decoder reports the level from the
    format information it read, and the framework reads the same 15-bit field itself
    (:mod:`qrcheck.decode.format_info`); a symbol whose level is wrong shows it in both, and
    a symbol the decoder mis-reports shows the disagreement rather than a wrong verdict.

    Args:
        ctx: The run's context; its corpus is already filtered and sorted by the runner.

    Returns:
        The cases, ascending by record ``id``.
    """
    return support.corpus_cases(
        ctx,
        _EC_MATCH_ID,
        ctx.corpus,
        execute_factory=lambda record: _bound(_ec_match_case, record),
    )


def _ec_match_case(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Assert the error-correction level of one corpus record's symbol.

    The reason order is the one AAP §0.4.5 needs: the two sources are compared **with each
    other** first, so a symbol whose decoder and format information disagree is reported as
    ``format_info_disagreement`` rather than as a mismatch against the request, and only
    then is the agreed level compared with the request.  A missing reading is a failure on
    its own (``sampling_failed``): an absent source can never be read as agreement.

    Args:
        ctx: The run's context.
        record: The corpus record this case is about.

    Returns:
        The case's result: ``format_info_disagreement`` when the two readings differ,
        ``value_mismatch`` when they agree on a level other than the requested one,
        ``sampling_failed`` when either reading is absent, a pass otherwise.

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
)
def SY_MASK(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per corpus record: the mask in the symbol is a legal mask.

    No decoder in the pinned set reports a mask, so it is read from the format information
    alone.  Mask *selection* is encoder-defined (AAP §0.3.5), so an unpinned request is held
    only to the legal range ``0..7`` -- asserting one value would fail a conforming target
    that scored its eight patterns differently.  A request that pins a mask is held to the
    pin, because a target declaring ``mask_pin`` is expected to honour it.  The 240 corpus
    cases pin nothing, so that branch is a guard rather than a path this case set takes: it
    keeps the pin rule in the same expression as the range rule, so a future case that pins a
    mask cannot bypass it.  ``SY-MATRIX-EQUIV`` is where a pin is actually observed on a
    symbol.

    Args:
        ctx: The run's context; its corpus is already filtered and sorted by the runner.

    Returns:
        The cases, ascending by record ``id``.
    """
    return support.corpus_cases(
        ctx,
        _MASK_ID,
        ctx.corpus,
        execute_factory=lambda record: _bound(_mask_case, record),
    )


def _mask_case(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Assert the mask pattern of one corpus record's symbol.

    The mask is only ever the format information's reading: it comes from the framework's
    own sampling of the symbol, never from a decoder and never from the target's metadata.

    Args:
        ctx: The run's context.
        record: The corpus record this case is about.

    Returns:
        The case's result: ``sampling_failed`` when the format information could not be
        read, ``value_mismatch`` when the mask is outside ``0..7`` or contradicts a pin, a
        pass otherwise.

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

    This is the product's **only** matrix comparison, and it is legitimate only because every
    request pins the version, the error level and the mask: at pinned parameters the standard
    fixes the symbol completely, whereas at unpinned ones the mask is the encoder's choice and
    two conforming encoders may legitimately differ (AAP §0.3.5).  A target that declares
    ``version_pin`` and ``mask_pin`` cannot honour either without emitting exactly the
    matrix the oracle emits.

    Args:
        ctx: The run's context; only its error level and seed are read when a case runs.

    Returns:
        The eight cases, ascending by ``case_id``.
    """
    cases = [
        support.synthetic_case(_MATRIX_EQUIV_ID, spec.case_id, _bound(_matrix_case, spec))
        for spec in _MATRIX_CASES
    ]
    return iter(sorted(cases, key=lambda case: case.case_id))


def _matrix_case(ctx: RunContext, spec: _MatrixCase) -> CaseResult:
    """Compare one pinned symbol's sampled grid with the oracle's matrix, module by module.

    Three assertions, in order, because each answers a different question and a reader of the
    evidence must be able to tell them apart: the sampled grid's dimension is exactly
    ``4V + 17``; ``version_from_dimension`` reads the pinned version back off that dimension;
    and every one of the ``N x N`` modules equals the oracle's.  The first differing
    ``(row, column)`` and the number of differing modules are recorded, so a reader sees the
    defect rather than being told a matrix inequality exists.

    Args:
        ctx: The run's context; the request carries the run's error level with the pins.
        spec: The pinned case this case is about.

    Returns:
        The case's result: ``sampling_failed`` when no grid could be recovered,
        ``value_mismatch`` when a dimension, the version read off it, or a module differs, a
        pass otherwise.

    Raises:
        TargetUnavailable, TargetTimeout: Exactly as :func:`_run_case`.
        ValueError: The payload does not fit the pinned version at the run's level, which
            would be a case-definition defect in this module.
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
