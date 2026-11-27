"""Shared support for the twenty-one checks: case identity, one request path, one error table.

Case iteration over corpus records, request construction through the port's builder, the
deterministic sampling helpers, the documented-recovery budget helper, the ``CaseResult`` factories,
and the single translation of an adapter error into a status. It also owns the four helpers a family
would otherwise copy -- :func:`corpus_case_id`, :func:`first_divergence`, :func:`sha256_hex` and
:func:`elapsed_ms` -- because AAP §0.8.1 makes this module the home of every cross-cutting concern
the check families share and allows no second implementation of one. This module is **not** a check
family: it registers nothing, owns no check ID and asserts nothing about a symbol.

It never owns request validation (the port's :func:`~qrcheck.targets.base.build_request`), the
result vocabulary and failure identifier (:mod:`qrcheck.report.model`), the evidence layout
(:mod:`qrcheck.evidence.naming` and the evidence store), the case type and run context
(:mod:`qrcheck.registry.check`), or any threshold -- each limit lives in ``qrcheck.toml``.

The adapter-error translation, following AAP §0.3.4 exactly (see :func:`case_status_for_error`), is
closed at five: ``TargetRejected``, ``TargetContractError`` and ``TargetUnsupported`` become
results, and ``TargetUnavailable`` and ``TargetTimeout`` are propagated unchanged because
:mod:`qrcheck.runner` owns the one-error-result-per-check target-loss policy. No bare ``except``, no
retry and no swallowed exception exists here (AAP §0.5.6).

Nothing here reads a clock except :func:`elapsed_ms`, whose value is recorded and never asserted;
nothing here reads a random source, ``hash()``, ``id()`` or the filesystem, and nothing sorts by
anything but an identifier. Two names differ from the plan's prose and the owners win: the
evidence store is :attr:`qrcheck.registry.check.RunContext.evidence`, and
:meth:`qrcheck.report.evidence.EvidenceStore.record` returns ``None`` for a passing case, so
:func:`record_evidence` returns a passing result unchanged. :mod:`qrcheck.decode` and
:mod:`qrcheck.corpus.builder` load at the point of use.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from qrcheck.registry.check import DEPENDENCY_SECONDARY_DECODER, Case, RunContext
from qrcheck.report.model import CaseResult, Reason, Status
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    TargetUnsupported,
    build_request,
    hex_prefix,
    require_capabilities,
)

if TYPE_CHECKING:  # Type-only, and both are imported for their names alone.
    from qrcheck.corpus.builder import CorpusRecord
    from qrcheck.decode import DecodeResult

__all__ = [
    "build",
    "case_status_for_error",
    "construct_out_of_contract",
    "corpus_case_id",
    "corpus_cases",
    "decode",
    "elapsed_ms",
    "fail",
    "first_divergence",
    "first_records",
    "generate",
    "generate_and_decode",
    "pass_",
    "record_evidence",
    "records_by_class",
    "recovery_fraction",
    "sha256_hex",
    "skip",
    "synthetic_case",
]

# The prefix that identifies the invalid-input family. The one place this module asks what
# kind of check it is serving, and it asks only because AAP §0.3.4 makes a declared rejection
# the *expected* outcome of an ``IV-*`` probe and a failure everywhere else.
_INVALID_CHECK_PREFIX: Final[str] = "IV-"

# Case-identifier prefixes. AAP §0.3.4 fixes both: a corpus-driven case is named after its
# record, and a constructed one carries the name the family gave it. ``synthetic:target``
# is the runner's own result on target loss and is deliberately not constructible here.
_CORPUS_CASE_PREFIX: Final[str] = "corpus:"
_SYNTHETIC_CASE_PREFIX: Final[str] = "synthetic:"

# The key that marks a deliberately illegal request in ``request.json`` and in a result's
# request projection (AAP §0.8.2). A reader who saw an out-of-contract request without the
# marker would take it for a framework bug.
_OUT_OF_CONTRACT_KEY: Final[str] = "out_of_contract"

# The three keys the evidence store's identity envelope owns. A ``request_extra`` key that
# collides with one of them would be silently overwritten by the envelope, so the collision
# is refused instead.
_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset({"case_id", "check_id", "seed"})

# The port's documented default colours (black on white). A request keeps them unless the
# caller asked for something else, so a pair that differs from them is what makes the
# ``color`` capability required.
_DEFAULT_FOREGROUND: Final[tuple[int, int, int]] = (0, 0, 0)
_DEFAULT_BACKGROUND: Final[tuple[int, int, int]] = (255, 255, 255)

# The documented, approximate recovery capacity of each level, as a fraction of the symbol's
# codewords (ISO/IEC 18004 §7.5; the figures AAP §0.2.1 and §0.6.5 record).  Properties of the
# standard, not thresholds this product chooses, which is why they live here and not in
# ``qrcheck.toml``; a check spends them as a budget and subtracts the configured safety margin.
_RECOVERY_FRACTIONS: Final[Mapping[str, float]] = MappingProxyType(
    {"L": 0.07, "M": 0.15, "Q": 0.25, "H": 0.30}
)


def build(
    payload: bytes,
    ec_level: str = "M",
    *,
    output_format: str = "png",
    module_size_px: int | None = None,
    quiet_zone_modules: int | None = None,
    foreground_rgb: tuple[int, int, int] = _DEFAULT_FOREGROUND,
    background_rgb: tuple[int, int, int] = _DEFAULT_BACKGROUND,
    logo_overlay_path: Path | None = None,
    version: int | None = None,
    mask: int | None = None,
) -> GenerateRequest:
    """Build the one legal request a check may send to a target.

    A thin delegation to :func:`qrcheck.targets.base.build_request`, the single definition of "the
    framework considers this request valid": an error level outside ``L``/``M``/``Q``/``H``, a
    version outside 1..40, a mask outside 0..7, a non-positive module size, a negative quiet zone, a
    colour channel outside 0..255, or a payload that does not fit the symbol is refused there.
    Nothing is re-validated here -- a second gate would be a second definition, and the ``IV-*``
    checks exist to prove the target rejects what only that gate would have stopped.

    Raises:
        ValueError: A field is outside its documented range or the payload does not fit the symbol;
            the message names the offending field. A framework-side defect rather than an adapter
            error, so the runner records an ``error`` result with its traceback.
    """
    return build_request(
        payload,
        ec_level,
        output_format=output_format,
        module_size_px=module_size_px,
        quiet_zone_modules=quiet_zone_modules,
        foreground_rgb=foreground_rgb,
        background_rgb=background_rgb,
        logo_overlay_path=logo_overlay_path,
        version=version,
        mask=mask,
    )


def construct_out_of_contract(
    payload: bytes,
    ec_level: str = "M",
    *,
    output_format: str = "png",
    module_size_px: int | None = None,
    quiet_zone_modules: int | None = None,
    foreground_rgb: tuple[int, int, int] = _DEFAULT_FOREGROUND,
    background_rgb: tuple[int, int, int] = _DEFAULT_BACKGROUND,
    logo_overlay_path: Path | None = None,
    version: int | None = None,
    mask: int | None = None,
) -> GenerateRequest:
    """Build a deliberately illegal request, with no validation at all.

    The documented bypass of :func:`build` (AAP §0.8.2) and the only one in the product: the
    ``IV-*`` probes must prove that a target *rejects* a request breaking exactly the rules the
    builder enforces, so this hands a target the same frozen request with ``ec_level="X"``,
    ``module_size_px=0``, ``quiet_zone_modules=-1``, a below-floor colour pair, or a version or mask
    pin outside its range. ``GenerateRequest`` validates nothing itself precisely so this is
    possible. Calling it for an ordinary case is a defect; pass ``out_of_contract=True`` to
    :func:`record_evidence` for these cases.
    """
    # The deliberately out-of-vocabulary values are exactly what this constructor exists to
    # accept: GenerateRequest has no __post_init__ and coerces nothing (see its docstring),
    # and the field annotations describe the legal contract rather than a runtime gate.
    return GenerateRequest(
        payload=payload,
        ec_level=ec_level,  # type: ignore[arg-type]
        output_format=output_format,  # type: ignore[arg-type]
        module_size_px=module_size_px,
        quiet_zone_modules=quiet_zone_modules,
        foreground_rgb=foreground_rgb,
        background_rgb=background_rgb,
        logo_overlay_path=logo_overlay_path,
        version=version,
        mask=mask,
    )


def corpus_case_id(record_id: str) -> str:
    """Return the case identifier of a corpus record: ``corpus:<record-id>``.

    The single definition of the identifier form AAP §0.3.4 fixes for a corpus-driven case, and the
    single owner of the ``corpus:`` prefix. :func:`corpus_cases` names the case it yields with it,
    and each family's executor names the result it reports with it, so the case the runner walks,
    the result it records and the evidence directory it writes cannot drift apart.

    Args:
        record_id: The stable identifier the corpus builder assigned the record, such as
            ``boundary-numeric-001``. :attr:`qrcheck.corpus.builder.CorpusRecord.id` is the source.

    Returns:
        The case identifier, exactly ``corpus:`` followed by `record_id`.
    """
    return f"{_CORPUS_CASE_PREFIX}{record_id}"


def corpus_cases(
    ctx: RunContext,
    check_id: str,
    records: Iterable[CorpusRecord],
    *,
    execute_factory: Callable[[CorpusRecord], Callable[[RunContext], CaseResult]],
) -> Iterator[Case]:
    """Yield one :class:`~qrcheck.registry.check.Case` per corpus record, in ``id`` order.

    The registry's own ``Case`` type, named ``corpus:<record-id>`` as AAP §0.3.4 fixes. Records are
    sorted by ``id`` here, so the order a family received them in can never reach a report.
    ``execute_factory`` is called once per record, which lets a family close over that record's
    payload and expectations.

    Raises:
        ValueError: ``check_id`` is not a non-empty string, or two records would register the same
            case identifier -- which would make two results share one evidence directory.
        TypeError: ``execute_factory`` (or what it returns) is not callable.
    """
    _require_check_id(check_id)
    if not callable(execute_factory):
        raise TypeError(
            f"check {check_id!r}: execute_factory must be callable, got "
            f"{type(execute_factory).__name__}"
        )
    seen: set[str] = set()
    for record in sorted(records, key=_by_record_id):
        case_id = corpus_case_id(record.id)
        if case_id in seen:
            raise ValueError(
                f"check {check_id!r} would register case {case_id!r} twice; corpus record "
                f"identifiers must be unique within a check"
            )
        seen.add(case_id)
        execute = execute_factory(record)
        if not callable(execute):
            raise TypeError(
                f"check {check_id!r}: execute_factory returned "
                f"{type(execute).__name__} for case {case_id!r}, which is not callable"
            )
        yield Case(case_id=case_id, execute=execute)


def synthetic_case(
    check_id: str,
    case_id: str,
    execute: Callable[[RunContext], CaseResult],
) -> Case:
    """Return the single constructed case `case_id` names.

    For the cases the framework builds rather than draws from the corpus: the invalid-input probes,
    the logo overlay, the pinned-matrix set, the colour pair, the performance probes, the
    determinism pair. Their identifiers are ``synthetic:<name>`` by AAP §0.3.4, and the prefix is
    checked rather than trusted. ``synthetic:target`` is refused: the one-error-result-per-check
    result on target loss belongs to :mod:`qrcheck.runner`.

    Raises:
        ValueError: ``case_id`` does not start with ``synthetic:``, carries no name after the
        prefix, or is the runner's own ``synthetic:target``; or ``check_id`` is not a non-empty
        string. TypeError: ``execute`` is not callable.
    """
    _require_check_id(check_id)
    if not isinstance(case_id, str) or not case_id.startswith(_SYNTHETIC_CASE_PREFIX):
        raise ValueError(
            f"check {check_id!r}: a constructed case identifier must start with "
            f"{_SYNTHETIC_CASE_PREFIX!r}, got {case_id!r}"
        )
    name = case_id[len(_SYNTHETIC_CASE_PREFIX) :].strip()
    if not name:
        raise ValueError(
            f"check {check_id!r}: case identifier {case_id!r} carries no name after "
            f"{_SYNTHETIC_CASE_PREFIX!r}"
        )
    if case_id == f"{_SYNTHETIC_CASE_PREFIX}target":
        raise ValueError(
            f"check {check_id!r}: {case_id!r} is the runner's own result on target loss and "
            f"must not be constructed by a check"
        )
    return Case(case_id=case_id, execute=execute)


def first_records(records: Iterable[CorpusRecord], count: int) -> tuple[CorpusRecord, ...]:
    """Return the first `count` records in ascending ``id`` order.

    AAP §0.8.1 fixes several case samples as "the first N records in ``id`` order", which is
    reproducible only if every family applies the same rule; this is that rule. Fewer records than
    `count` yields all of them rather than raising, because a filtered corpus is a legitimate input
    and the runner decides whether an empty filter is an error.

    Raises:
        TypeError: ``count`` is not an integer. A ``bool`` is refused even though it is an ``int``
            subclass, because ``True`` as a sample size would otherwise validate silently.
        ValueError: ``count`` is negative.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"count must be an integer, got {type(count).__name__}")
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    return tuple(sorted(records, key=_by_record_id)[:count])


def records_by_class(
    records: Iterable[CorpusRecord],
    classes: str | Iterable[str],
) -> tuple[CorpusRecord, ...]:
    """Return the records of `classes`, in ascending ``id`` order.

    The class-name filter behind the fixed samples of AAP §0.8.1. Exact class names are taken, not
    globs: the run-time ``--class`` glob is applied by the runner before a check sees the corpus, so
    a check may rely on exact names and a typo in one is a defect worth naming.

    Raises:
        TypeError: A name is not a string.
        ValueError: A name is outside the corpus's class vocabulary, and the message names it and
            the vocabulary.
    """
    # Read from its owner at call time rather than copied here: the class vocabulary has one
    # definition.  The import stays inside the function because qrcheck.corpus.builder reaches
    # numpy through its payload generators.
    from qrcheck.corpus.builder import CLASS_NAMES

    names = (classes,) if isinstance(classes, str) else tuple(classes)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"corpus class names must be strings, got {type(name).__name__}"
            )
    wanted: frozenset[str] = frozenset(names)
    unknown = sorted(wanted - set(CLASS_NAMES))
    if unknown:
        raise ValueError(
            f"unknown corpus class(es) {unknown}; the corpus classes are {list(CLASS_NAMES)}"
        )
    if not wanted:
        return ()
    return tuple(
        sorted(
            (record for record in records if record.class_name in wanted),
            key=_by_record_id,
        )
    )


def generate(ctx: RunContext, request: GenerateRequest) -> GeneratedSymbol:
    """Ask the run's target for one symbol, after the capability gate.

    :func:`qrcheck.targets.base.require_capabilities` runs first and raises
    :class:`~qrcheck.targets.base.TargetUnsupported` **before any I/O**, so a request the target
    never declared is refused without a process, a socket or a file being touched and without
    consuming the target's ``timeout_s``. No adapter error is caught here: every one of the five
    reaches the check, or the runner, for the two that propagate.

    Raises:
        TargetUnsupported: The target does not declare a capability this request needs.
        TargetUnavailable: The target could not be reached at all.
        TargetTimeout: No response within the configured timeout.
        TargetRejected: The target declared a client-side rejection of the request.
        TargetContractError: A response arrived but could not be used.
    """
    require_capabilities(ctx.adapter, _request_capabilities(request))
    return ctx.adapter.generate(request)


def decode(ctx: RunContext, image_bytes: bytes, request: GenerateRequest) -> DecodeResult:
    """Decode one artefact through the framework's two-decoder pipeline.

    A delegation to :func:`qrcheck.decode.decode_symbol`, called exactly once per symbol: the
    primary decoder is the authority for every check (AAP §0.3.2), so nothing here reads
    ``per_decoder`` to rescue a payload or reconcile the two. The run's resolved fact about the
    secondary decoder is passed through when ``RunContext.availability`` holds one, so the recorded
    degradation and the resolution that decided the run's skips cannot disagree.

    Raises:
            ValueError: ``image_bytes`` is not bytes, or ``request`` carries no colours -- both
            caller defects. Unusable bytes, an SVG artefact, an unreadable symbol and an unsamplable
            grid return a result instead of raising.
    """
    # Imported inside the call: qrcheck.decode loads numpy for its decoders, and importing
    # this module must not start a native image stack in a process that only lists checks.
    from qrcheck.decode import decode_symbol

    return decode_symbol(
        image_bytes,
        request,
        secondary=ctx.availability.get(DEPENDENCY_SECONDARY_DECODER),
    )


def generate_and_decode(
    ctx: RunContext,
    request: GenerateRequest,
) -> tuple[GeneratedSymbol, DecodeResult]:
    """Generate one symbol and decode it: the single path every image check uses.

    Keeps "one invocation, one decode" a property of one function rather than a convention each
    family is trusted to follow.

    Raises:
        TargetUnsupported, TargetUnavailable, TargetTimeout, TargetRejected, TargetContractError:
            Exactly as :func:`generate`.
        ValueError: Exactly as :func:`decode`.
    """
    symbol = generate(ctx, request)
    return symbol, decode(ctx, symbol.image_bytes, request)


def recovery_fraction(ec_level: str) -> float:
    """Return the documented error-correction recovery fraction of `ec_level`.

    The single definition in the product of the recovery capacity each level promises: ``L`` 7 %,
    ``M`` 15 %, ``Q`` 25 %, ``H`` 30 % of the symbol's codewords (AAP §0.2.1, §0.3.4
    ``[checks.damage]``, §0.6.5). ``qrcheck.checks.damage`` and ``qrcheck.checks.render`` both
    budget against it and neither may carry a copy. The figures are approximate -- a 1-L symbol
    corrects at most two of its twenty-six codewords -- so a caller subtracts the configured
    absolute ``occlusion_margin``.

    Raises:
        KeyError: `ec_level` is not one of the four levels, and the message names the value it was
        given and the levels it accepts.
    """
    try:
        return _RECOVERY_FRACTIONS[ec_level]
    except (KeyError, TypeError):
        raise KeyError(
            f"no documented error-correction recovery fraction for error level "
            f"{ec_level!r}; expected one of {', '.join(sorted(_RECOVERY_FRACTIONS))}"
        ) from None


def first_divergence(requested: bytes, received: bytes | None) -> int | None:
    """Return the offset at which `received` stops agreeing with `requested`, or ``None``.

    The single definition of the offset the evidence records as ``first_divergence_offset`` (AAP
    §0.4.5): the point at which the payload the decoder returned departs from the payload that was
    asked for. The requested payload is always the first argument, so every call reads the same way.

    ``None`` when `received` was never read -- there is nothing to compare -- and when the two
    payloads are equal; otherwise the index of the first differing byte; and the length of the
    common prefix when one payload is a prefix of the other, which is the offset a truncation
    produces. That last case is reported as a divergence rather than as ``None`` because it is where
    the two payloads stop agreeing, which is exactly what a reader diagnosing a truncated symbol
    needs.

    The comparison is on bytes, never on rendered text: a payload that is not valid UTF-8 is judged
    on what actually travelled.
    """
    if received is None:
        return None
    shared = min(len(requested), len(received))
    for offset in range(shared):
        if requested[offset] != received[offset]:
            return offset
    return None if len(requested) == len(received) else shared


def sha256_hex(data: bytes) -> str:
    """Return the lower-case hexadecimal SHA-256 digest of `data`.

    The single definition of a payload digest in the check layer, so two families cannot report the
    same bytes under two spellings. ``hashlib`` rather than the builtin ``hash()``: a digest
    recorded in evidence has to be stable across processes and machines, and ``hash()`` of a bytes
    object is salted per process, which would make two evidence files differ for the same payload
    (AAP §0.5.1).
    """
    return hashlib.sha256(data).hexdigest()


def elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, a :func:`time.perf_counter` reading.

    Measurement only. The value becomes a result's ``duration_ms``, which every report masks and
    which no comparison in the check layer reads: AAP §0.5.5 admits a duration into an assertion
    only in the ``PF-*`` thresholds and the run ceiling, and neither is measured here.

    Clamped at ``0.0``. A monotonic ``perf_counter`` makes a negative delta unreachable, so the
    clamp changes no reachable value; it keeps the stored field non-negative by construction.
    """
    return max((time.perf_counter() - started) * 1000.0, 0.0)


def case_status_for_error(
    check_id: str,
    case_id: str,
    error: BaseException,
    *,
    request: GenerateRequest,
    out_of_contract: bool = False,
    duration_ms: float = 0.0,
    seed: int | None = None,
) -> CaseResult:
    """Translate one adapter error into the result AAP §0.3.4 prescribes for it.

    The product's only translation of an adapter failure into a status, closed at five.
    ``TargetRejected`` on an ``IV-*`` check is a **pass** -- a declared rejection is what those
    probes exist to observe -- with the target's text verbatim in ``decoded["rejection"]``, and a
    **failure** with reason ``target_rejected`` on any other check. ``TargetContractError`` is a
    **failure** with reason ``contract_error``, the diagnostic in ``decoded["error_text"]`` and the
    first 256 bytes of the unusable body in ``decoded["hex_prefix"]``. ``TargetUnsupported`` is a
    **skip** with reason ``unsupported_capability``. The request travels with the result, marked
    ``out_of_contract: true`` for a deliberate probe, and no ``error`` result is ever built: the
    runner alone constructs one.

    Raises:
        TargetUnavailable: Re-raised unchanged when that is the error supplied. TargetTimeout: Re-
        raised unchanged when that is the error supplied. TypeError: ``error`` is not one of the
        five adapter errors, ``request`` is not a request record, or a result mapping is not a
        mapping. All are caller defects. ValueError: A field of the result is outside its documented
        range, or ``check_id``/``case_id`` is not a non-empty string.
    """
    _require_identity(check_id, case_id)
    if isinstance(error, (TargetUnavailable, TargetTimeout)):
        # Propagated, not converted: the runner's target-loss policy needs the error itself.
        raise error

    projection = _request_projection(request, out_of_contract)
    if isinstance(error, TargetRejected):
        text = str(error)
        if check_id.startswith(_INVALID_CHECK_PREFIX):
            return pass_(
                check_id,
                case_id,
                request=projection,
                decoded={"rejection": text},
                expected={"outcome": "declared_rejection"},
                duration_ms=duration_ms,
                seed=seed,
            )
        return fail(
            check_id,
            case_id,
            reason=Reason.TARGET_REJECTED,
            request=projection,
            decoded={"rejection": text},
            expected={"outcome": "generated_symbol"},
            duration_ms=duration_ms,
            seed=seed,
        )
    if isinstance(error, TargetContractError):
        text = str(error)
        return fail(
            check_id,
            case_id,
            reason=Reason.CONTRACT_ERROR,
            request=projection,
            decoded={
                "error_text": text,
                "hex_prefix": hex_prefix(error.body_prefix),
            },
            expected={"outcome": "usable_artefact"},
            duration_ms=duration_ms,
            seed=seed,
        )
    if isinstance(error, TargetUnsupported):
        return skip(
            check_id,
            case_id,
            reason=Reason.UNSUPPORTED_CAPABILITY,
            request=projection,
            decoded={"error_text": str(error)},
            expected={"outcome": "declared_capability"},
            duration_ms=duration_ms,
            seed=seed,
        )
    raise TypeError(
        f"check {check_id!r}: {type(error).__name__} is not one of the five adapter errors; "
        f"an adapter raises TargetUnavailable, TargetTimeout, TargetRejected, "
        f"TargetContractError or TargetUnsupported, and nothing else"
    )


def pass_(
    check_id: str,
    case_id: str,
    *,
    request: Mapping[str, Any] | None = None,
    decoded: Mapping[str, Any] | None = None,
    expected: Mapping[str, Any] | None = None,
    duration_ms: float = 0.0,
    seed: int | None = None,
) -> CaseResult:
    """Return a passing result: no reason, no evidence, the case's own record.

    A passing case writes no evidence and the model therefore refuses one, so this factory offers no
    way to attach an evidence directory; :func:`record_evidence` returns a passing result unchanged
    for the same reason.

    Raises:
        TypeError: A record field is not a mapping or a number, or an identifier is not a string.
        ValueError: An identifier is empty, ``duration_ms`` is negative or not finite, or ``seed``
        is negative.
    """
    return _case_result(
        Status.PASSED,
        None,
        check_id,
        case_id,
        request=request,
        decoded=decoded,
        expected=expected,
        duration_ms=duration_ms,
        seed=seed,
    )


def fail(
    check_id: str,
    case_id: str,
    *,
    reason: Reason | str,
    request: Mapping[str, Any] | None = None,
    decoded: Mapping[str, Any] | None = None,
    expected: Mapping[str, Any] | None = None,
    duration_ms: float = 0.0,
    seed: int | None = None,
) -> CaseResult:
    """Return a failing result carrying the reason the check failed for.

    ``reason`` is coerced here and re-validated by the model, so a wrong code fails loudly at
    construction instead of reaching a report as an unknown string.

    Raises:
        TypeError: ``reason`` is not one of the accepted forms, or a record field has the wrong
        type. ValueError: ``reason`` is not a :class:`~qrcheck.report.model.Reason` member, or
        another field is outside its documented range.
    """
    return _case_result(
        Status.FAILED,
        reason,
        check_id,
        case_id,
        request=request,
        decoded=decoded,
        expected=expected,
        duration_ms=duration_ms,
        seed=seed,
    )


def skip(
    check_id: str,
    case_id: str,
    *,
    reason: Reason | str,
    request: Mapping[str, Any] | None = None,
    decoded: Mapping[str, Any] | None = None,
    expected: Mapping[str, Any] | None = None,
    duration_ms: float = 0.0,
    seed: int | None = None,
) -> CaseResult:
    """Return a skipped result carrying the machine-readable reason it did not run.

    A skip is a real outcome and never a flavoured pass: ``counts.skipped`` is its own integer and
    the exit code reads only ``failed`` and ``error``. ``reason`` is ``unsupported_capability``,
    ``missing_dependency`` or ``unsupported_platform``, and ``decoded`` carries any evidence the
    skip rests on, such as the discovery order a rasterizer probe attempted.

    Raises:
        TypeError: ``reason`` is not one of the accepted forms, or a record field has the wrong
        type. ValueError: ``reason`` is not a :class:`~qrcheck.report.model.Reason` member, or
        another field is outside its documented range.
    """
    return _case_result(
        Status.SKIPPED,
        reason,
        check_id,
        case_id,
        request=request,
        decoded=decoded,
        expected=expected,
        duration_ms=duration_ms,
        seed=seed,
    )


def record_evidence(
    ctx: RunContext,
    result: CaseResult,
    *,
    symbol: GeneratedSymbol | None = None,
    repeat_symbol: GeneratedSymbol | None = None,
    damage_variants: Sequence[tuple[str, bytes]] = (),
    logo: bytes | None = None,
    out_of_contract: bool = False,
    request_extra: Mapping[str, Any] | None = None,
) -> CaseResult:
    """Persist this case's evidence and return the result carrying its directory.

    The only call site of :meth:`qrcheck.report.evidence.EvidenceStore.record` outside
    ``qrcheck/report/``. The store owns the policy and the artefact set, the naming module the
    layout, and this function the envelope of ``request.json``: the case's identity plus whatever
    the check resolved for it -- ``out_of_contract: true`` for a deliberate probe, and the
    ``request_extra`` entries. A
    **passing** result is returned unchanged: the store persists nothing for a pass.

    Raises:
            TypeError: A persisted mapping holds a value the writer cannot serialise, or an
            artefact's bytes are not bytes-like. ValueError: ``request_extra`` names one of the keys
            the identity envelope owns (``check_id``, ``case_id``, ``seed``), or an artefact format
            or damage kind is not one the naming module declares. OSError: The case directory cannot
            be created or an artefact cannot be written. Propagated untouched, because
            ``qrcheck.cli.run`` turns an unwritable report directory into exit 3.
    """
    extra = _as_mapping("request_extra", request_extra)
    clashes = sorted(set(extra) & _ENVELOPE_KEYS)
    if clashes:
        raise ValueError(
            f"request_extra names {clashes}, which the evidence envelope owns; the envelope "
            f"always wins, so the value would be lost silently"
        )
    enriched = dict(result.request)
    if out_of_contract:
        enriched[_OUT_OF_CONTRACT_KEY] = True
    enriched.update(extra)
    staged = dataclasses.replace(result, request=enriched)
    evidence_dir = ctx.evidence.record(
        staged,
        symbol=symbol,
        repeat_symbol=repeat_symbol,
        damage_variants=tuple(damage_variants),
        logo=logo,
    )
    if evidence_dir is None:
        return result
    return dataclasses.replace(result, evidence_dir=evidence_dir)


def _case_result(
    status: Status,
    reason: Reason | str | None,
    check_id: str,
    case_id: str,
    *,
    request: Mapping[str, Any] | None,
    decoded: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    duration_ms: float,
    seed: int | None,
) -> CaseResult:
    """Build one result, normalising every field and validating what the model does not.

    A passing result must carry no reason and every other status must.

    Raises:
        TypeError: `reason` is not a :class:`~qrcheck.report.model.Reason`, its string value or
            ``None``; or a record field has the wrong type.
        ValueError: `reason` is not a :class:`~qrcheck.report.model.Reason` member, a status and
            reason disagree, or another field is outside its documented range.
    """
    _require_identity(check_id, case_id)
    outcome = Status(status)
    cause: Reason | None = None
    if outcome is Status.PASSED:
        if reason is not None:
            raise ValueError(
                f"case {case_id!r} passed, so it carries no reason; got {reason!r}"
            )
    else:
        if reason is None:
            raise ValueError(f"case {case_id!r} is {outcome.value!r} and needs a reason")
        if not isinstance(reason, (Reason, str)):
            raise TypeError(
                f"case {case_id!r}: reason must be a Reason or its string value, got "
                f"{type(reason).__name__}"
            )
        try:
            cause = Reason(reason)
        except ValueError:
            raise ValueError(
                f"case {case_id!r}: reason {reason!r} is not one of the reasons "
                f"{[member.value for member in Reason]}"
            ) from None
    return CaseResult(
        check_id=check_id,
        case_id=case_id,
        status=outcome,
        reason=cause,
        duration_ms=_checked_duration(duration_ms),
        request=_as_mapping("request", request),
        decoded=_as_mapping("decoded", decoded),
        expected=None if expected is None else _as_mapping("expected", expected),
        evidence_dir=None,
        seed=_checked_seed(seed),
    )


def _request_projection(request: GenerateRequest, out_of_contract: bool) -> dict[str, Any]:
    """Return the JSON-ready projection of `request`, marked when it is a probe.

    `out_of_contract` marks a request built without validation, so ``request.json`` says so.

    Raises:
        TypeError: `request` is not a :class:`~qrcheck.targets.base.GenerateRequest`.
    """
    if not isinstance(request, GenerateRequest):
        raise TypeError(
            f"request must be a GenerateRequest, got {type(request).__name__}"
        )
    projection = request.as_dict()
    if out_of_contract:
        projection[_OUT_OF_CONTRACT_KEY] = True
    return projection


def _require_identity(check_id: str, case_id: str) -> None:
    """Reject an empty check or case identifier, naming the field that carries the defect.

    The result model validates its status, reason, duration and evidence path but not its two
    identifiers, and an empty identifier would produce an evidence directory named for nothing.

    Raises:
        TypeError: An identifier is not a string.
        ValueError: An identifier holds no visible character.
    """
    _require_check_id(check_id)
    if not isinstance(case_id, str):
        raise TypeError(f"case_id must be a str, got {type(case_id).__name__}")
    if not case_id.strip():
        raise ValueError(f"case_id must not be empty, got {case_id!r}")


def _require_check_id(check_id: str) -> None:
    """Reject a check identifier that is not a non-empty string.

    Raises:
        TypeError: `check_id` is not a string.
        ValueError: `check_id` holds no visible character.
    """
    if not isinstance(check_id, str):
        raise TypeError(f"check_id must be a str, got {type(check_id).__name__}")
    if not check_id.strip():
        raise ValueError(f"check_id must not be empty, got {check_id!r}")


def _as_mapping(field: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return `value` as a plain dict, or an empty one when it is ``None``.

    A plain dict, never the mapping that was passed: the report writers serialise these with
    ``json``, which cannot emit a mapping proxy or any other ``Mapping`` implementation.

    Raises:
        TypeError: `value` is neither a mapping nor ``None``.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field} must be a mapping or None for a CaseResult, got "
            f"{type(value).__name__}"
        )
    return dict(value)


def _checked_duration(duration_ms: float) -> float:
    """Return `duration_ms` as a finite, non-negative float.

    The model already refuses a negative duration; finiteness is added here because a ``nan`` or an
    ``inf`` would reach ``summary.json`` as a token no JSON consumer accepts.

    Raises:
        TypeError: `duration_ms` is not a number. A ``bool`` is refused even though it is an ``int``
            subclass.
        ValueError: `duration_ms` is negative or not finite.
    """
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        raise TypeError(f"duration_ms must be a number, got {type(duration_ms).__name__}")
    value = float(duration_ms)
    if not math.isfinite(value):
        raise ValueError(f"duration_ms must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"duration_ms must not be negative, got {value!r}")
    return value


def _checked_seed(seed: int | None) -> int:
    """Return `seed`, or ``0`` when the case has none, refusing a negative value.

    The model stores the seed without validating it; a negative seed cannot be derived from the run
    seed by :func:`qrcheck.config.seeding.derive_seed`.

    Raises:
        TypeError: `seed` is neither ``None`` nor an integer. A ``bool`` is refused.
        ValueError: `seed` is negative.
    """
    if seed is None:
        return 0
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed}")
    return seed


def _request_capabilities(request: GenerateRequest) -> frozenset[Capability]:
    """Return the capabilities `request` itself asks for, for the gate before any I/O.

    Only a field that positively asks for an operation can be read off a request: the output format,
    and the version pin, mask pin, module size, quiet zone, non-default colour pair and logo
    overlay. ``ec_level`` is deliberately not among them: every request carries a level, so
    requiring the capability here would refuse every request to a target that supports the format
    but not level selection.

    Raises:
        ValueError: ``request.output_format`` is neither ``png`` nor ``svg``, which means the
        request bypassed :func:`build` without being an ``IV-*`` probe.
    """
    try:
        needed: set[Capability] = {Capability(request.output_format)}
    except ValueError:
        raise ValueError(
            "a request may ask for output_format in png, svg; got "
            f"{request.output_format!r}"
        ) from None
    if request.module_size_px is not None:
        needed.add(Capability.module_size)
    if request.quiet_zone_modules is not None:
        needed.add(Capability.quiet_zone)
    if (request.foreground_rgb, request.background_rgb) != (
        _DEFAULT_FOREGROUND,
        _DEFAULT_BACKGROUND,
    ):
        needed.add(Capability.color)
    if request.logo_overlay_path is not None:
        needed.add(Capability.logo_overlay)
    if request.version is not None:
        needed.add(Capability.version_pin)
    if request.mask is not None:
        needed.add(Capability.mask_pin)
    return frozenset(needed)


def _by_record_id(record: CorpusRecord) -> str:
    """Return the sort key of a corpus record: its identifier.

    The one sort key in this module, so every helper that promises ``id`` order promises the same
    order.
    """
    return record.id

