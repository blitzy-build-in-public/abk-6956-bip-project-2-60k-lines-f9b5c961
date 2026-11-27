"""Mode-family checks: ``MD-MODE-OPTIMAL`` and ``MD-CHARSET``.

Plan item served: the check-family table of AAP §0.8.1 (``checks/mode.py`` owns ``MD-MODE-OPTIMAL``
and ``MD-CHARSET``), that section's case-matrix rows of 60 and 74 cases, and the ``MD-CHARSET``
coverage mapping of §0.8.2.

Why mode optimality is proven through the version: neither pinned decoder reports an encoding mode
-- zxing-cpp 2.2.0 exposes the error level and no segment metadata, OpenCV nothing beyond the
payload and the corner quadrangle -- so a check claiming to observe the mode would report a guess,
and the target's own ``declared_mode`` is evidence only. Optimality is proven indirectly: a numeric
payload occupies a strictly smaller symbol in numeric mode than in byte mode at the same level, so a
wider mode shows up as a version above the payload's own-charset minimum.

``MD-MODE-OPTIMAL``: 60 cases, one encode and one decode each -- the 24 discriminating boundary
records (all 12 ``boundary-numeric`` and all 12 ``boundary-alphanumeric``, in ``id`` order) plus a
36-record byte-mode sample chosen by :data:`MODE_OPTIMAL_BYTE_SAMPLE_SIZE`. The assertion is an
upper bound on the decoded version rather than equality with the mode minimum; the inline comment at
:func:`execute_mode_optimal` states why.

``MD-CHARSET``: 74 cases, one encode and one decode each -- the six coverages of AAP §0.8.2,
resolved through :data:`CHARSET_COVERAGE_CLASSES`: ``ascii`` (``boundary-alphanumeric`` 12, ``otp``
12), ``utf8-multibyte`` 24, ``emoji`` 12, ``right-to-left`` (``rtl``) 8, ``embedded-nul`` (``nul-
byte``) 6, plus ``four-byte-code-points`` from the ``emoji`` and ``utf8-multibyte`` records whose
code points reach U+10000 and above, which the predicate decides rather than the class. Each case
asserts byte-exact preservation of the request.

Reasons emitted: ``decoder_disagreement`` from ``MD-CHARSET`` when the two decoders returned
differing payloads, which it reports ahead of its byte comparison (AAP §0.3.2);
``value_mismatch`` from both checks and ``sampling_failed`` from ``MD-MODE-OPTIMAL`` when no module
grid could be recovered. Adapter errors are not classified here:
both checks route every one through :func:`qrcheck.checks.support.case_status_for_error`, and
``TargetUnavailable``/``TargetTimeout`` are re-raised for the runner's target-loss policy, and a
declared rejection or an unusable artefact becomes a failure with ``target_rejected`` or
``contract_error``. Nothing reads a clock for an assertion, a random source, ``hash()`` or ``id()``,
and no image library, encoder or concrete adapter is imported.
"""

from __future__ import annotations

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
    "mode_of",
    "mode_optimal_cases",
    "payload_length_in_mode",
]

# The two check identifiers this module registers, written once and used for registration, for
# every result and for ``EXPECTED_ENCODES``.  They are literals of ``CHECK_IDS``, never renamed.
MODE_OPTIMAL_CHECK: Final[str] = "MD-MODE-OPTIMAL"
CHARSET_CHECK: Final[str] = "MD-CHARSET"

# How many encode invocations each check makes in one full run over the default 240-record
# corpus: one per case, because neither check ever re-asks the target for a symbol (AAP §0.4.3,
# §0.5.6 -- a retry would register as the declared count plus one and fail the counting test).
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        MODE_OPTIMAL_CHECK: 60,
        CHARSET_CHECK: 74,
    }
)

# The discriminating half of ``MD-MODE-OPTIMAL``'s case set (AAP §0.8.1): the two boundary
# classes, whose numeric or alphanumeric minimum is strictly below their byte-mode minimum.
MODE_OPTIMAL_DISCRIMINATING_CLASSES: Final[tuple[str, ...]] = (
    "boundary-alphanumeric",
    "boundary-numeric",
)

# The size of ``MD-MODE-OPTIMAL``'s byte-mode sample (AAP §0.8.1 fixes the check at 60 cases:
# 24 discriminating records plus these 36).  The sample is drawn by the declared rule in
# :func:`mode_optimal_records` -- the first 36 records in ``id`` order outside both charsets.
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

# The corpus classes ``MD-CHARSET`` draws its 74 cases from, derived from the mapping above and
# sorted so the case set never depends on declaration order.  It equals the corpus builder's own
# ``MD_CHARSET_CLASSES``; a unit test asserts it and ``records_by_class`` validates every name.
CHARSET_CLASSES: Final[tuple[str, ...]] = tuple(
    sorted({class_name for names in CHARSET_COVERAGE_CLASSES.values() for class_name in names})
)

# The encoding modes in increasing order of the charset each admits, mapped to the unit a payload
# length is counted in (ISO/IEC 18004 §7.4: numeric and alphanumeric are character counts, byte
# mode a byte count).  Sorted order is declaration order, so derived versions are emitted the same
# way.
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

# The one consensus classification that is a finding in its own right (AAP §0.3.2): two decoder
# payloads that differ.  ``qrcheck.decode.base`` publishes the vocabulary as a ``Literal``, so the
# value is named here rather than spelled out at the comparison.
_DISAGREE: Final[str] = "disagree"


@check(
    id=MODE_OPTIMAL_CHECK,
    title="Encoding mode is optimal: no symbol larger than the payload's own charset requires",
    family="mode",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def mode_optimal_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per record of the discriminating set and the byte-mode sample.

    Sixty cases over the default corpus: the 24 discriminating boundary records plus 36 byte-mode
    records, each named ``corpus:<record-id>`` and yielded in ascending ``id`` order by the shared
    case builder, so import order and corpus order can never reach a report. The decorator replaces
    this name with the registered contract, so the generator is reached as
    ``mode_optimal_cases.cases(ctx)``.

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

    The discriminating half is the two boundary classes in full. The complementary half is drawn by
    the declared rule of :data:`MODE_OPTIMAL_BYTE_SAMPLE_SIZE`: every record whose payload
    :func:`mode_of` reports as ``byte``, taken in ascending ``id`` order, first
    :data:`MODE_OPTIMAL_BYTE_SAMPLE_SIZE` of them. No list of record identifiers is transcribed, so
    a corpus rebuild re-derives the sample. The union is de-duplicated by record ``id`` and returned
    ascending; fewer than 60 records when the run filtered the corpus.
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

    The callable takes the run's context and returns that case's result.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run the case for `record` against the run's target."""
        return execute_mode_optimal(ctx, record)

    return execute


def execute_mode_optimal(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Ask the target for `record`'s symbol and assert its version is mode-optimal.

    One encode and one decode, through the shared request builder and decode path. The expected
    version is the smallest Version 1-40 that holds the payload in the mode its own charset admits,
    from the capacity tables at the run's error-correction level. The assertion is an upper bound: a
    wider mode shows up as a version above that minimum, and the inline comment records why equality
    is not asserted. Returns ``passed`` at or below the minimum, ``failed`` with ``value_mismatch``
    above it, and ``failed`` with ``sampling_failed`` when no grid could be recovered.

    Raises:
        TargetUnsupported: The target does not declare ``png``; a skip with reason
        ``unsupported_capability`` if it reaches here. TargetUnavailable, TargetTimeout: Propagated
        unchanged for the runner's target-loss policy. TargetRejected, TargetContractError:
        Translated into a failing result carrying the target's own text verbatim. ValueError: The
        record contradicts itself -- its declared mode does not admit its payload, or the payload
        does not fit any version at this level. Both are corpus defects.
    """
    case_id = support.corpus_case_id(record.id)
    payload = record.payload_bytes
    request = support.build(payload, ctx.ec_level)
    expected_version = _own_charset_minimal_version(record, payload, ctx.ec_level)
    started = time.perf_counter()
    try:
        symbol, decoded = support.generate_and_decode(ctx, request)
    except TargetError as error:
        return _translated_failure(ctx, MODE_OPTIMAL_CHECK, case_id, error, request, started)
    duration_ms = support.elapsed_ms(started)
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
    # The assertion is an upper bound, not equality with the mode minimum: the claim is that the
    # encoder used no wider mode, and a wider mode shows up as a version above the own-charset
    # minimum.  A version below the bound cannot be a mode defect -- the payload does not fit a
    # smaller symbol at that level -- and equality would drop the check into the defective
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

    The projection the decode layer produces is kept whole -- both decoders' outcomes, agreement,
    the EC-level pair, version, mask and matrix digest -- and the facts that make the discrimination
    auditable are added beside it: the record's declared mode, the payload length in that mode's
    unit, and the minimal version the payload would force in each of the three modes (``None`` where
    the payload does not fit that charset at all). A reader can then see, from the evidence alone,
    whether the symbol under test is the smallest one the payload admits.
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

    Raises:
        ValueError: The record's declared mode does not admit its own payload, or the payload does
            not fit Version 40 at the run's EC level. Both mean the corpus contradicts itself, which
            the builder's invariant check exists to prevent, so either is a framework defect worth a
            traceback rather than a verdict about the target.
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
    corpus_driven=True,
)
def charset_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield one case per record of the six charset coverages.

    Seventy-four cases over the default corpus, one per record of :data:`CHARSET_CLASSES`, each
    named ``corpus:<record-id>`` and yielded in ascending ``id`` order. The coverage each record
    witnesses is recorded in its result's evidence, so the six-coverage claim is auditable from a
    report rather than from the class list alone. The decorator replaces this name with the
    registered contract.

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

    The callable takes the run's context and returns that case's result.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run the case for `record` against the run's target."""
        return execute_charset(ctx, record)

    return execute


def execute_charset(ctx: RunContext, record: CorpusRecord) -> CaseResult:
    """Ask the target for `record`'s symbol and assert the payload survives byte-exactly.

    One encode and one decode through the shared path. The comparison is between the payload bytes
    the record requested and the bytes the authoritative decoder returned: not the decoded text,
    which a decoder may re-render, and never the secondary decoder's answer, which corroborates but
    never decides. Two payloads the decoders reported that differ are reported first, as
    ``decoder_disagreement`` (AAP §0.3.2), because the consensus classification describes what the
    decoders did before any byte of the payload is compared. The evidence records both digests and
    the offset of the first divergent byte, so a truncation shows an offset equal to the decoded
    length. Returns ``passed`` on an exact match, ``failed`` with ``decoder_disagreement`` or
    ``value_mismatch`` otherwise.

    Raises:
        TargetUnsupported: The target does not declare ``png``; a skip with reason
        ``unsupported_capability`` if it reaches here. TargetUnavailable, TargetTimeout: Propagated
        unchanged for the runner's target-loss policy. TargetRejected, TargetContractError:
        Translated into a failing result carrying the target's own text verbatim.
    """
    case_id = support.corpus_case_id(record.id)
    payload = record.payload_bytes
    request = support.build(payload, ctx.ec_level)
    started = time.perf_counter()
    try:
        symbol, decoded = support.generate_and_decode(ctx, request)
    except TargetError as error:
        return _translated_failure(ctx, CHARSET_CHECK, case_id, error, request, started)
    duration_ms = support.elapsed_ms(started)
    received = decoded.payload_bytes
    evidence = decoded.as_dict()
    evidence["expected_mode"] = record.expected_mode
    evidence["payload_len_requested"] = len(payload)
    evidence["payload_sha256_requested"] = support.sha256_hex(payload)
    evidence["payload_sha256_decoded"] = (
        None if received is None else support.sha256_hex(received)
    )
    evidence["first_divergence_offset"] = support.first_divergence(payload, received)
    evidence["coverages"] = list(coverages_for_record(record))
    expected: Mapping[str, Any] = {
        "assertion": "decoded payload equals the requested payload byte for byte",
        "ec_level": ctx.ec_level,
        "expected_mode": record.expected_mode,
        "payload_len": len(payload),
        "payload_sha256": support.sha256_hex(payload),
    }
    projection = request.as_dict()
    if decoded.agreement == _DISAGREE:
        # Two payloads that differ are a finding in their own right, ahead of the byte comparison
        # (AAP §0.3.2): the consensus classification describes what the decoders did, which is
        # settled before any byte of the payload is compared. The result is persisted through the
        # shared evidence call exactly as the mismatch path persists it, so the case still carries
        # symbol.png, request.json and a decoded.json holding both decoders' payloads verbatim.
        result = support.fail(
            CHARSET_CHECK,
            case_id,
            reason=Reason.DECODER_DISAGREEMENT,
            request=projection,
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
        return support.record_evidence(ctx, result, symbol=symbol)
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

    :func:`qrcheck.checks.support.case_status_for_error` is the product's only translation, so every
    per-case adapter error reaches it rather than being classified here: a declared rejection or an
    unusable artefact is a *failure* -- these checks never expect a rejection -- and an unsupported
    capability a *skip*. The result is persisted through the shared evidence call, so a target-
    caused failure still carries ``request.json`` and the diagnostic; there is no artefact to
    persist.

    Raises:
        TargetUnavailable: Re-raised unchanged when that is the error supplied.
        TargetTimeout: Re-raised unchanged when that is the error supplied.
        TypeError: `error` is not one of the five adapter errors, which would be a framework defect.
    """
    translated = support.case_status_for_error(
        check_id,
        case_id,
        error,
        request=request,
        duration_ms=support.elapsed_ms(started),
        seed=ctx.seed,
    )
    return support.record_evidence(ctx, translated)


def mode_of(payload: bytes) -> str:
    """Return the narrowest encoding mode `payload` fits, as one of the three modes.

    The payload's own answer, read from its bytes: ``numeric`` when every character is a digit,
    ``alphanumeric`` when every character is in the alphanumeric set (ISO/IEC 18004 §7.4.3 and
    §7.4.4), and ``byte`` for anything else -- including every payload carrying a byte that is not
    ASCII. This is what makes the byte-mode sample reproducible: a record is in it because its
    payload verifiably does not fit either restricted charset, not because a list says so.
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

    The byte count for ``byte`` mode, and the character count for the two character modes -- which
    for their ASCII payloads equals the byte count, but is derived from the text so the unit is the
    one the capacity arithmetic documents.

    Raises:
        ValueError: `mode` is not one of the three modes.
        UnicodeDecodeError: A payload is not ASCII, so it has no character count in a restricted
            charset. The caller is asserting a charset the payload does not fit, which is a corpus
            defect rather than a target verdict.
    """
    if mode not in _MODE_ORDER:
        raise ValueError(f"unknown mode {mode!r}; the modes are {tuple(_MODE_ORDER)}")
    if mode == "byte":
        return len(payload)
    return len(payload.decode("ascii"))


def candidate_minimal_versions(payload: bytes, ec_level: str) -> dict[str, int | None]:
    """Return the minimal version `payload` forces in each of the three modes.

    The discrimination ``MD-MODE-OPTIMAL`` rests on, made visible: a payload that fits the numeric
    charset needs a strictly smaller symbol in numeric mode than in byte mode at the same
    error-correction level. A mode the payload does not fit reports ``None`` rather than a version,
    because there is no such thing as "this payload in that charset"; ``None`` also for a mode whose
    payload exceeds Version 40.
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

    The declared mapping decides, with one predicate on top: ``four-byte-code-points`` needs a
    payload that really carries a code point of U+10000 or above, so a ``utf8-multibyte`` record
    whose widest code point is two or three bytes wide does not count toward it. Every other
    coverage is a property
    of the record's class. Empty for a class the mapping does not list, which the class list makes
    unreachable for the records this check selects.
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


def _admits(payload: bytes, mode: str) -> bool:
    """Return whether `payload` fits the character set `mode` encodes.

    ``True`` for ``byte`` mode always, and for a restricted mode when the payload's narrowest
    charset is that mode or a narrower one -- numeric payloads fit the alphanumeric set too (ISO/IEC
    18004 §7.4.4's set contains §7.4.3's).
    """
    if mode not in _MODE_ORDER:
        raise ValueError(f"unknown mode {mode!r}; the modes are {tuple(_MODE_ORDER)}")
    return _MODE_ORDER[mode_of(payload)] <= _MODE_ORDER[mode]


def _holds_four_byte_code_point(payload: bytes) -> bool:
    """Return whether `payload` holds at least one code point needing four UTF-8 bytes.

    A payload that is not valid UTF-8 holds no code point at all, so the answer is ``False`` rather
    than an error: the coverage is about the text a payload carries.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return any(ord(character) >= _FOUR_BYTE_CODE_POINT_MIN for character in text)
