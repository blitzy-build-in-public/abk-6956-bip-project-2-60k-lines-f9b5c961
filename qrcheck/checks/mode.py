"""Mode-family checks: ``MD-MODE-OPTIMAL`` and ``MD-CHARSET``.

AAP §0.8.1's check-family table and its case-matrix rows of 60 and 74 cases, and §0.8.2's
``MD-CHARSET`` coverage mapping.

Neither pinned decoder reports an encoding mode: zxing-cpp 2.2.0 exposes the error level and no
segment metadata, and OpenCV exposes nothing beyond the payload and the corner quadrangle.  A
check that claimed to observe the mode would be reporting a guess, and the target's own
``declared_mode`` is evidence only.  Mode optimality is proven indirectly, through the version the
payload forces: a payload that fits the numeric charset occupies a strictly smaller symbol in
numeric mode than in byte mode at the same level, so an encoder that used the wider mode is
visible as a version above the payload's own-charset minimum, which is what this family asserts.

``MD-MODE-OPTIMAL`` runs 60 cases, one encode and one decode each: the 24 discriminating boundary
records (all 12 ``boundary-numeric`` and all 12 ``boundary-alphanumeric``, in ``id`` order), whose
payloads sit at a capacity boundary where the numeric or alphanumeric minimum is strictly below
the byte-mode minimum, plus a byte-mode sample of :data:`MODE_OPTIMAL_BYTE_SAMPLE_SIZE` records
that states the complementary property -- no larger symbol than the payload's byte-mode minimum
was used.  The assertion is an upper bound on the decoded version rather than equality with the
mode minimum; the inline comment at :func:`execute_mode_optimal` states why, and
:data:`EXPECTED_ENCODES` records the case count the counting-stub test holds this check to.

``MD-CHARSET`` runs 74 cases, one encode and one decode each, from the six coverage groups of
AAP §0.8.2 resolved through :data:`CHARSET_COVERAGE_CLASSES`: ``ascii`` from
``boundary-alphanumeric`` and ``otp``, ``utf8-multibyte``, ``emoji``, ``right-to-left`` from
``rtl``, ``embedded-nul`` from ``nul-byte``, and ``four-byte-code-points`` from ``emoji`` plus
the ``utf8-multibyte`` records whose code points are U+10000 and above -- the predicate, not the
class, decides (see :func:`coverages_for_record`).  That resolves to ``boundary-alphanumeric``
12, ``emoji`` 12, ``nul-byte`` 6, ``otp`` 12, ``rtl`` 8 and ``utf8-multibyte`` 24.
:func:`qrcheck.checks.support.records_by_class` validates every class name against
``qrcheck.corpus.builder.CLASS_NAMES``, that vocabulary's one owner, so a typo in the mapping
fails loudly at case-build time instead of silently selecting nothing.  Each case asserts
byte-exact preservation of the requested payload, and the evidence records both payload digests
and the first divergence offset, which makes the truncating defect's symptom attributable rather
than merely visible.

``MD-MODE-OPTIMAL`` emits ``value_mismatch`` when the symbol is larger than the payload's
own-charset minimum and ``sampling_failed`` when the module grid could not be recovered at all.
``MD-CHARSET`` emits ``value_mismatch`` and, of its own verdicts, nothing else: it compares
payload bytes rather than geometry, so a symbol whose grid could not be sampled still yields the
primary decoder's payload or none at all, and "none at all" is a payload mismatch.

Per-case adapter errors are not classified here.  Both checks route every one through
:func:`qrcheck.checks.support.case_status_for_error`, the product's only translation of an
adapter failure into a status: a declared rejection or an unusable artefact becomes a failure
(``target_rejected``, ``contract_error``), an unsupported capability a skip
(``unsupported_capability``), and target loss (``TargetUnavailable``, ``TargetTimeout``) is
re-raised unchanged for the runner's one-error-result-per-check policy.

Determinism: nothing here reads a clock for an assertion, a random source, ``hash()`` or
``id()``; iteration is by corpus record ``id`` ascending and every mapping is sorted before it
reaches a result, while ``duration_ms`` is measured and recorded, never compared.  The module
writes no file: evidence is persisted through
:func:`qrcheck.checks.support.record_evidence`, and only for a case that did not pass.

It imports no image library, no encoder and no concrete adapter -- only the shared check support,
the registry's contract, the result model, the capacity tables, and from the port its capability
vocabulary and error base class, which lets an adapter failure reach the shared translation
without naming a single error type here -- so it runs against a stub adapter with no decoder, no
network and no symbol in sight.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from qrcheck.checks import support
from qrcheck.corpus.capacity import minimal_version
from qrcheck.registry.check import DEPENDENCY_PRIMARY_DECODER, Case, RunContext, check
from qrcheck.report.model import CaseResult, Reason
from qrcheck.targets.base import Capability, TargetError

if TYPE_CHECKING:  # Imported for the annotations alone; see the module docstring.
    from qrcheck.corpus.builder import CorpusRecord
    from qrcheck.decode import DecodeResult
    from qrcheck.targets.base import GenerateRequest

__all__ = [
    "CHARSET_CLASSES",
    "CHARSET_COVERAGE_CLASSES",
    "EXPECTED_ENCODES",
    "MODE_OPTIMAL_BYTE_SAMPLE_SIZE",
    "MODE_OPTIMAL_DISCRIMINATING_CLASSES",
    "candidate_minimal_versions",
    "charset_cases",
    "coverages_for_record",
    "first_divergence",
    "mode_of",
    "mode_optimal_cases",
    "payload_length_in_mode",
]

# The two check identifiers this module registers, each written once and used for
# registration, for every result it produces, and for the counting contract in
# ``EXPECTED_ENCODES``.  They are the literals of ``qrcheck.registry.check.CHECK_IDS``
# and are never renamed: CI trend lines and the expected-failure table key on them.
MODE_OPTIMAL_CHECK: Final[str] = "MD-MODE-OPTIMAL"
CHARSET_CHECK: Final[str] = "MD-CHARSET"

# How many encode invocations each check makes in one full run over the default
# 240-record corpus: one per case, because neither check ever re-asks the target for a
# symbol (AAP §0.4.3 and §0.5.6 -- a retry would register here as the declared count
# plus one and fail the counting test).  ``tests/unit/test_checks_negative.py`` drives
# each check with a counting stub and asserts the observed count equals these values.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        MODE_OPTIMAL_CHECK: 60,
        CHARSET_CHECK: 74,
    }
)

# The discriminating half of ``MD-MODE-OPTIMAL``'s case set (AAP §0.8.1): the two
# boundary classes whose payloads sit at a capacity boundary, so their numeric or
# alphanumeric minimum is strictly below their byte-mode minimum and a wider mode shows
# up as a larger symbol.  Taken in full, in ``id`` order.
MODE_OPTIMAL_DISCRIMINATING_CLASSES: Final[tuple[str, ...]] = (
    "boundary-alphanumeric",
    "boundary-numeric",
)

# The size of ``MD-MODE-OPTIMAL``'s byte-mode sample (AAP §0.8.1, whose case matrix
# fixes the check at 60 cases: 24 discriminating records plus these 36).  The sample is
# drawn by the declared rule in :func:`mode_optimal_records` -- the first 36 records in
# ``id`` order whose payload is verifiably outside both the numeric and the
# alphanumeric charset -- rather than from an arbitrary list, so the set is reproducible
# from the corpus alone.
MODE_OPTIMAL_BYTE_SAMPLE_SIZE: Final[int] = 36

# Which corpus classes witness which ``MD-CHARSET`` coverage (AAP §0.8.2).  The one
# declaration of the mapping in the product, resolved to the class list below.
CHARSET_COVERAGE_CLASSES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "ascii": ("boundary-alphanumeric", "otp"),
        "utf8-multibyte": ("utf8-multibyte",),
        "emoji": ("emoji",),
        "right-to-left": ("rtl",),
        "embedded-nul": ("nul-byte",),
        "four-byte-code-points": ("emoji", "utf8-multibyte"),
    }
)

# The corpus classes ``MD-CHARSET`` draws its 74 cases from, derived from the mapping
# above rather than listed a second time and sorted so the case set never depends on the
# mapping's declaration order.  It equals ``qrcheck.corpus.builder.MD_CHARSET_CLASSES``,
# the corpus builder's own view of the same six classes; a unit test asserts the
# equality, and :func:`qrcheck.checks.support.records_by_class` validates every name
# against the corpus vocabulary at case-build time.
CHARSET_CLASSES: Final[tuple[str, ...]] = tuple(
    sorted({class_name for names in CHARSET_COVERAGE_CLASSES.values() for class_name in names})
)

# The encoding modes, in increasing order of the character set each admits, mapped to
# the unit a payload length is counted in for that mode (ISO/IEC 18004 §7.4: numeric and
# alphanumeric lengths are character counts, byte mode is a byte count).  Sorted order
# is the declaration order, so the candidate versions this module derives are emitted in
# the same order on every run.
_MODE_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {"numeric": 0, "alphanumeric": 1, "byte": 2}
)

# The two restricted charsets (ISO/IEC 18004 §7.4.3 and §7.4.4).  Kanji mode is never
# generated by the corpus and Micro QR is out of scope, so no third charset exists here.
_NUMERIC_CHARSET: Final[frozenset[str]] = frozenset("0123456789")
_ALPHANUMERIC_CHARSET: Final[frozenset[str]] = frozenset(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
)

# The smallest code point that needs four UTF-8 bytes, which is the boundary the
# ``four-byte-code-points`` coverage asserts (U+10000 and above).
_FOUR_BYTE_CODE_POINT_MIN: Final[int] = 0x10000

# The prefix of a corpus-driven case identifier: ``corpus:<record-id>`` (AAP §0.3.4).
# Repeated from the shared support module because this module composes the identifier
# its own executor reports, and the support module keeps its copy private.
_CORPUS_CASE_PREFIX: Final[str] = "corpus:"


@check(
    id=MODE_OPTIMAL_CHECK,
    title="Encoding mode is optimal: no symbol larger than the payload's own charset requires",
    family="mode",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def mode_optimal_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per record of the discriminating set and the byte-mode sample.

    Sixty cases over the default corpus: the 24 discriminating boundary records plus 36
    byte-mode records, each named ``corpus:<record-id>`` and yielded in ascending ``id``
    order by the shared case builder, so import order and corpus order can never reach a
    report.

    The decorator replaces this name with the registered contract, so the generator is
    reached as ``mode_optimal_cases.cases(ctx)``; ``qrcheck.registry.check`` documents
    that substitution.

    Args:
        ctx: The run's context.  Its corpus is already filtered by the runner's
            ``--class`` selection and its EC level is the level every request asks for.

    Yields:
        One :class:`~qrcheck.registry.check.Case` per selected record.
    """
    yield from support.corpus_cases(
        ctx,
        MODE_OPTIMAL_CHECK,
        mode_optimal_records(ctx),
        execute_factory=_mode_optimal_executor,
    )


def mode_optimal_records(ctx: RunContext) -> tuple[CorpusRecord, ...]:
    """Return ``MD-MODE-OPTIMAL``'s case set: the boundary classes plus the byte sample.

    The discriminating half is the two boundary classes in full.  The complementary half
    is drawn by the declared, reproducible rule of :data:`MODE_OPTIMAL_BYTE_SAMPLE_SIZE`:
    every record whose payload :func:`mode_of` reports as ``byte`` -- verifiably outside
    both restricted charsets -- taken in ascending ``id`` order, first
    :data:`MODE_OPTIMAL_BYTE_SAMPLE_SIZE` of them.  No list of record identifiers is
    transcribed anywhere, so a corpus rebuild re-derives the sample instead of silently
    keeping stale names.

    The union is de-duplicated by record ``id`` and returned in ascending ``id`` order.
    A record cannot in fact be in both halves (a boundary numeric or alphanumeric payload
    is not byte-mode), but the de-duplication is what guarantees one case per record even
    if that ever stopped holding, since two cases sharing an identifier would share one
    evidence directory.

    Args:
        ctx: The run's context; only its corpus is read.

    Returns:
        The selected records, ascending by ``id``.  Fewer than 60 when the run filtered
        the corpus, which is the filter's decision and not this function's.
    """
    discriminating = support.records_by_class(ctx.corpus, MODE_OPTIMAL_DISCRIMINATING_CLASSES)
    byte_mode = tuple(
        record for record in ctx.corpus if mode_of(record.payload_bytes) == "byte"
    )
    sample = support.first_records(byte_mode, MODE_OPTIMAL_BYTE_SAMPLE_SIZE)
    selected = {record.id: record for record in (*discriminating, *sample)}
    return tuple(selected[record_id] for record_id in sorted(selected))


def _mode_optimal_executor(record: CorpusRecord) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs `record`'s ``MD-MODE-OPTIMAL`` case.

    Args:
        record: The corpus record the case is about.

    Returns:
        A callable taking the run's context and returning that case's result.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run the case for `record` against the run's target."""
        return execute_mode_optimal(ctx, record)

    return execute


def execute_mode_optimal(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Ask the target for `record`'s symbol and assert its version is mode-optimal.

    One encode and one decode, through the single shared request builder and the single
    decode path every image check uses.  The expected version is the smallest Version
    1-40 that holds the payload in the mode its own charset admits, computed from the
    capacity tables at the run's error-correction level -- the tables a unit test proves
    equal to the oracle's -- so the expectation is derived rather than transcribed.

    The assertion is an upper bound: the symbol may not be larger than the payload's
    own-charset minimum.  Mode optimality is precisely that claim, and a wider mode shows
    up as a version above the bound.  The inline comment below records why equality is
    not asserted here.

    Args:
        ctx: The run's context: the adapter, the EC level, the seed and the evidence store.
        record: The corpus record to encode.

    Returns:
        ``passed`` when the decoded version is at or below the own-charset minimum;
        ``failed`` with ``value_mismatch`` when it is larger, and ``failed`` with
        ``sampling_failed`` when the module grid could not be recovered from the symbol.
        A failing result carries its evidence directory.

    Raises:
        TargetUnsupported: The target does not declare ``png``.  Normally pre-resolved to
            a ``skipped`` result before execution; if it reaches here it becomes a
            ``skipped`` result with reason ``unsupported_capability``, which is counted
            separately and never as a pass.
        TargetUnavailable, TargetTimeout: Propagated unchanged for the runner's
            one-error-result-per-check policy on target loss.
        TargetRejected, TargetContractError: Translated into a failing result carrying the
            target's own text verbatim.
        ValueError: The record contradicts itself -- its declared mode does not admit its
            own payload, or its payload does not fit any version at this level.  Both are
            corpus defects, recorded by the runner as an ``error`` with a traceback.
    """
    case_id = _corpus_case_id(record)
    payload = record.payload_bytes
    request = support.build(payload, ctx.ec_level)
    expected_version = _own_charset_minimal_version(record, payload, ctx.ec_level)
    started = time.perf_counter()
    try:
        symbol, decoded = support.generate_and_decode(ctx, request)
    except TargetError as error:
        return _translated_failure(ctx, MODE_OPTIMAL_CHECK, case_id, error, request, started)
    duration_ms = _elapsed_ms(started)
    evidence = _mode_evidence(decoded, record, payload, ctx.ec_level)
    expected: Mapping[str, Any] = {
        "assertion": "decoded version at or below the payload's own-charset minimum",
        "ec_level": ctx.ec_level,
        "expected_mode": record.expected_mode,
        "max_version": expected_version,
        "payload_len_in_mode": payload_length_in_mode(payload, record.expected_mode),
    }
    projection = request.as_dict()
    if decoded.version is None:
        result = support.fail(
            MODE_OPTIMAL_CHECK,
            case_id,
            reason=Reason.SAMPLING_FAILED,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    # The assertion is an upper bound, not equality with the mode minimum: the claim is
    # that the encoder used no wider mode than the payload admits, and a wider mode shows
    # up as a version above the own-charset minimum.  A version *below* the bound cannot be
    # a mode defect -- the payload does not fit a smaller symbol at the requested
    # error-correction level -- so an encoder that reaches one has encoded at a lower level,
    # which SY-EC-MATCH and SY-VERSION-MIN assert against.  Equality would report that
    # lower-level fault a second time here and would put MD-MODE-OPTIMAL into the defective
    # target's expected failing set, which declares that it must pass.
    elif decoded.version > expected_version:
        result = support.fail(
            MODE_OPTIMAL_CHECK,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    else:
        return support.pass_(
            MODE_OPTIMAL_CHECK,
            case_id,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    return support.record_evidence(ctx, result, symbol=symbol)


def _mode_evidence(
    decoded: DecodeResult,
    record: CorpusRecord,
    payload: bytes,
    ec_level: str,
) -> dict[str, Any]:
    """Return the decoded evidence plus the three candidate minima for `record`.

    The projection the decode layer produces is kept whole -- both decoders' outcomes,
    agreement, the EC-level pair, version, mask and matrix digest -- and the facts that
    make the discrimination auditable are added beside it: the record's declared mode,
    the payload length in that mode's unit, and the minimal version the payload would
    force in each of the three modes (``None`` where the payload does not fit that
    charset at all).  A reader can then see, from the evidence alone, whether the symbol
    under test is the smallest one the payload admits.

    Args:
        decoded: The decoded evidence for the symbol.
        record: The corpus record the symbol was produced for.
        payload: The requested payload bytes.
        ec_level: The error-correction level the symbol was requested at.

    Returns:
        A JSON-ready mapping: the decode projection plus ``expected_mode``,
        ``payload_len_in_mode`` and ``minimal_versions``.
    """
    evidence = decoded.as_dict()
    evidence["expected_mode"] = record.expected_mode
    evidence["payload_len_in_mode"] = payload_length_in_mode(payload, record.expected_mode)
    evidence["minimal_versions"] = candidate_minimal_versions(payload, ec_level)
    return evidence


def _own_charset_minimal_version(
    record: CorpusRecord,
    payload: bytes,
    ec_level: str,
) -> int:
    """Return the minimal version the payload forces in `record`'s own charset.

    Args:
        record: The corpus record, whose ``expected_mode`` is the mode asserted.
        payload: The record's payload bytes.
        ec_level: The run's error-correction level.

    Returns:
        The smallest Version 1-40 that holds the payload in the record's mode.

    Raises:
        ValueError: The record's declared mode does not admit its own payload, or the
            payload does not fit Version 40 at `ec_level`.  Both mean the corpus
            contradicts itself, which the builder's invariant check exists to prevent, so
            either is a framework defect worth a traceback rather than a verdict about the
            target.
    """
    mode = record.expected_mode
    if not _admits(payload, mode):
        raise ValueError(
            f"corpus record {record.id!r} declares mode {mode!r}, but its payload does not "
            f"fit that charset; the narrowest charset it fits is {mode_of(payload)!r}"
        )
    version = minimal_version(payload_length_in_mode(payload, mode), ec_level, mode)
    if version is None:
        raise ValueError(
            f"corpus record {record.id!r}: {len(payload)} bytes in {mode!r} mode do not fit "
            f"Version 40 at error level {ec_level!r}"
        )
    return version


@check(
    id=CHARSET_CHECK,
    title="Charset preservation: every payload class round-trips byte for byte",
    family="mode",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def charset_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per record of the six charset coverages.

    Seventy-four cases over the default corpus, one per record of
    :data:`CHARSET_CLASSES`, each named ``corpus:<record-id>`` and yielded in ascending
    ``id`` order.  The coverage each record witnesses is recorded in its result's
    evidence, so the six-coverage claim is auditable from a report rather than from the
    class list alone.

    The decorator replaces this name with the registered contract, so the generator is
    reached as ``charset_cases.cases(ctx)``.

    Args:
        ctx: The run's context.  Its corpus is already filtered by the runner's
            ``--class`` selection and its EC level is the level every request asks for.

    Yields:
        One :class:`~qrcheck.registry.check.Case` per selected record.
    """
    yield from support.corpus_cases(
        ctx,
        CHARSET_CHECK,
        support.records_by_class(ctx.corpus, CHARSET_CLASSES),
        execute_factory=_charset_executor,
    )


def _charset_executor(record: CorpusRecord) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs `record`'s ``MD-CHARSET`` case.

    Args:
        record: The corpus record the case is about.

    Returns:
        A callable taking the run's context and returning that case's result.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run the case for `record` against the run's target."""
        return execute_charset(ctx, record)

    return execute


def execute_charset(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Ask the target for `record`'s symbol and assert the payload survives byte-exactly.

    One encode and one decode through the shared path.  The comparison is between the
    payload bytes the record requested and the payload bytes the authoritative decoder
    returned: not the decoded text, which a decoder may re-render, and never the
    secondary decoder's answer, which corroborates but never decides (a symbol the
    primary could not read is a mismatch, not a rescue).

    The evidence records both digests and the offset of the first divergent byte, so a
    shortened payload is diagnosable from the artefact: a truncation shows a decoded
    digest that differs and an offset equal to the decoded length, where a corrupted byte
    shows the offset of that byte.

    Args:
        ctx: The run's context: the adapter, the EC level, the seed and the evidence store.
        record: The corpus record to encode.

    Returns:
        ``passed`` when the decoded payload equals the requested payload byte for byte;
        ``failed`` with ``value_mismatch`` otherwise, carrying its evidence directory.

    Raises:
        TargetUnsupported: The target does not declare ``png``.  Normally pre-resolved to
            a ``skipped`` result before execution; if it reaches here it becomes a
            ``skipped`` result with reason ``unsupported_capability``, counted separately
            and never as a pass.
        TargetUnavailable, TargetTimeout: Propagated unchanged for the runner's
            one-error-result-per-check policy on target loss.
        TargetRejected, TargetContractError: Translated into a failing result carrying the
            target's own text verbatim.
    """
    case_id = _corpus_case_id(record)
    payload = record.payload_bytes
    request = support.build(payload, ctx.ec_level)
    started = time.perf_counter()
    try:
        symbol, decoded = support.generate_and_decode(ctx, request)
    except TargetError as error:
        return _translated_failure(ctx, CHARSET_CHECK, case_id, error, request, started)
    duration_ms = _elapsed_ms(started)
    received = decoded.payload_bytes
    evidence = decoded.as_dict()
    evidence["expected_mode"] = record.expected_mode
    evidence["payload_len_requested"] = len(payload)
    evidence["payload_sha256_requested"] = _digest(payload)
    evidence["payload_sha256_decoded"] = _digest(received)
    evidence["first_divergence_offset"] = (
        None if received is None else first_divergence(payload, received)
    )
    evidence["coverages"] = list(coverages_for_record(record))
    expected: Mapping[str, Any] = {
        "assertion": "decoded payload equals the requested payload byte for byte",
        "ec_level": ctx.ec_level,
        "expected_mode": record.expected_mode,
        "payload_len": len(payload),
        "payload_sha256": _digest(payload),
    }
    projection = request.as_dict()
    if received == payload:
        return support.pass_(
            CHARSET_CHECK,
            case_id,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    result = support.fail(
        CHARSET_CHECK,
        case_id,
        reason=Reason.VALUE_MISMATCH,
        request=projection,
        decoded=evidence,
        expected=expected,
        duration_ms=duration_ms,
        seed=ctx.seed,
    )
    return support.record_evidence(ctx, result, symbol=symbol)


def _translated_failure(
    ctx: RunContext,
    check_id: str,
    case_id: str,
    error: TargetError,
    request: GenerateRequest,
    started: float,
) -> CaseResult:
    """Translate one adapter error into the result the error table prescribes, with evidence.

    :func:`qrcheck.checks.support.case_status_for_error` is the product's only translation
    of an adapter failure into a status, so every per-case adapter error reaches it rather
    than being classified here.  For this family that means a declared rejection or an
    unusable artefact is a *failure* (these checks never expect a rejection, unlike the
    invalid-input probes) and an unsupported capability is a *skip* -- the belt-and-braces
    path behind capability resolution.  ``TargetUnavailable`` and ``TargetTimeout`` are
    re-raised by the translation, unchanged, for the runner's one-error-result-per-check
    policy on target loss.

    The translated result is persisted through the shared evidence call, so a failure
    caused by the target rather than by the symbol still carries ``request.json`` and the
    target's own diagnostic.  There is no artefact to persist: the target produced none.

    Args:
        ctx: The run's context; its evidence store and seed are read.
        check_id: The check the case belongs to.
        case_id: The case's stable identifier.
        error: The adapter error, one of the port's five.
        request: The request that produced it, in its unprojected form.
        started: The ``perf_counter`` reading taken before the target was asked.

    Returns:
        The failed or skipped result, carrying its evidence directory.

    Raises:
        TargetUnavailable: Re-raised unchanged when that is the error supplied.
        TargetTimeout: Re-raised unchanged when that is the error supplied.
        TypeError: `error` is not one of the five adapter errors, which would mean a
            framework defect rather than a target's.
    """
    translated = support.case_status_for_error(
        check_id,
        case_id,
        error,
        request=request,
        duration_ms=_elapsed_ms(started),
        seed=ctx.seed,
    )
    return support.record_evidence(ctx, translated)


def mode_of(payload: bytes) -> str:
    """Return the narrowest encoding mode `payload` fits, as one of the three modes.

    The payload's own answer, read from its bytes: ``numeric`` when every character is a
    digit, ``alphanumeric`` when every character is in the alphanumeric set (ISO/IEC
    18004 §7.4.3 and §7.4.4), and ``byte`` for anything else -- including every payload
    carrying a byte that is not ASCII, which no restricted charset can carry.

    This is what makes the byte-mode sample of ``MD-MODE-OPTIMAL`` reproducible: a record
    is in the sample because its payload verifiably does not fit either restricted
    charset, not because a list says so.

    Args:
        payload: The payload bytes.

    Returns:
        ``numeric``, ``alphanumeric`` or ``byte``.
    """
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError:
        return "byte"
    characters = set(text)
    if characters <= _NUMERIC_CHARSET:
        return "numeric"
    if characters <= _ALPHANUMERIC_CHARSET:
        return "alphanumeric"
    return "byte"


def payload_length_in_mode(payload: bytes, mode: str) -> int:
    """Return `payload`'s length in the unit `mode` counts, as the capacity tables expect.

    Args:
        payload: The payload bytes.
        mode: One of ``numeric``, ``alphanumeric`` or ``byte``.

    Returns:
        The byte count for ``byte`` mode, and the character count for the two character
        modes -- which for their ASCII payloads equals the byte count, but is derived from
        the text so the unit is the one the capacity arithmetic documents.

    Raises:
        ValueError: `mode` is not one of the three modes.
        UnicodeDecodeError: A payload is not ASCII, so it has no character count in a
            restricted charset.  The caller is asserting a charset the payload does not
            fit, which is a corpus defect rather than a target verdict.
    """
    if mode not in _MODE_ORDER:
        raise ValueError(f"unknown mode {mode!r}; the modes are {tuple(_MODE_ORDER)}")
    if mode == "byte":
        return len(payload)
    return len(payload.decode("ascii"))


def candidate_minimal_versions(payload: bytes, ec_level: str) -> dict[str, int | None]:
    """Return the minimal version `payload` forces in each of the three modes.

    The discrimination ``MD-MODE-OPTIMAL`` rests on, made visible: a payload that fits the
    numeric charset needs a strictly smaller symbol in numeric mode than in byte mode at
    the same error-correction level, so a target that used the wider mode produces a
    version above the numeric minimum.  A mode the payload does not fit reports ``None``
    rather than a version, because there is no such thing as "this payload in that
    charset".

    Args:
        payload: The payload bytes.
        ec_level: The error-correction level to compute against.

    Returns:
        A mapping of mode to minimal Version 1-40, in mode order, with ``None`` for the
        charsets the payload does not fit -- and ``None`` for a mode whose payload exceeds
        Version 40 as well.
    """
    versions: dict[str, int | None] = {}
    for mode in _MODE_ORDER:
        if not _admits(payload, mode):
            versions[mode] = None
            continue
        versions[mode] = minimal_version(
            payload_length_in_mode(payload, mode), ec_level, mode
        )
    return versions


def coverages_for_record(record: CorpusRecord) -> tuple[str, ...]:
    """Return the ``MD-CHARSET`` coverages `record` witnesses, ascending.

    The declared mapping decides, with one predicate applied on top: the
    ``four-byte-code-points`` coverage needs a payload that really carries a code point
    of U+10000 or above, so a ``utf8-multibyte`` record whose widest code point is
    two or three bytes wide does not count toward it.  Every other coverage is a
    property of the class the record belongs to.

    Args:
        record: The corpus record.

    Returns:
        The coverage names, ascending; empty for a class the mapping does not list,
        which the class list makes unreachable for the records this check selects.
    """
    covered = [
        coverage
        for coverage, classes in CHARSET_COVERAGE_CLASSES.items()
        if record.class_name in classes
    ]
    if "four-byte-code-points" in covered and not _holds_four_byte_code_point(
        record.payload_bytes
    ):
        covered.remove("four-byte-code-points")
    return tuple(sorted(covered))


def first_divergence(requested: bytes, received: bytes) -> int | None:
    """Return the offset of the first byte at which `received` differs from `requested`.

    Args:
        requested: The payload the case asked for.
        received: The payload the decoder returned.

    Returns:
        ``None`` when the two are equal, the offset of the first differing byte, or the
        length of the shorter payload when one is a prefix of the other -- which is the
        offset a truncation produces.
    """
    shared = min(len(requested), len(received))
    for offset in range(shared):
        if requested[offset] != received[offset]:
            return offset
    return None if len(requested) == len(received) else shared


def _admits(payload: bytes, mode: str) -> bool:
    """Return whether `payload` fits the character set `mode` encodes.

    Args:
        payload: The payload bytes.
        mode: One of ``numeric``, ``alphanumeric`` or ``byte``.

    Returns:
        ``True`` for ``byte`` mode always, and for a restricted mode when the payload's
        narrowest charset is that mode or a narrower one -- numeric payloads fit the
        alphanumeric set too (ISO/IEC 18004 §7.4.4's set contains §7.4.3's).
    """
    if mode not in _MODE_ORDER:
        raise ValueError(f"unknown mode {mode!r}; the modes are {tuple(_MODE_ORDER)}")
    return _MODE_ORDER[mode_of(payload)] <= _MODE_ORDER[mode]


def _holds_four_byte_code_point(payload: bytes) -> bool:
    """Return whether `payload` holds at least one code point needing four UTF-8 bytes.

    A payload that is not valid UTF-8 holds no code point at all, so the answer is
    ``False`` rather than an error: the coverage is about the text a payload carries.

    Args:
        payload: The payload bytes.

    Returns:
        ``True`` when the payload decodes as UTF-8 and some code point is U+10000 or
        above, ``False`` otherwise.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return any(ord(character) >= _FOUR_BYTE_CODE_POINT_MIN for character in text)


def _digest(payload: bytes | None) -> str | None:
    """Return the SHA-256 hex digest of `payload`, or ``None`` when there is no payload.

    A content digest, never ``hash()``: it is stable across processes and machines, which
    is what lets a reader compare two evidence files from two runs.

    Args:
        payload: The payload bytes, or ``None`` when the decoder returned nothing.

    Returns:
        The 64-character hex digest, or ``None``.
    """
    return None if payload is None else hashlib.sha256(payload).hexdigest()


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, a :func:`time.perf_counter` value.

    Measured and recorded, never asserted: a duration is the one volatile quantity this
    family reports, and no comparison in this module reads it.

    Args:
        started: The ``perf_counter`` reading taken before the target was asked.

    Returns:
        The elapsed time in milliseconds, never negative.
    """
    return (time.perf_counter() - started) * 1000.0


def _corpus_case_id(record: CorpusRecord) -> str:
    """Return `record`'s case identifier, ``corpus:<record-id>``.

    Args:
        record: The corpus record.

    Returns:
        The identifier the shared case builder also derives for the same record, so the
        result the executor reports and the case the runner walks share one name.
    """
    return f"{_CORPUS_CASE_PREFIX}{record.id}"
