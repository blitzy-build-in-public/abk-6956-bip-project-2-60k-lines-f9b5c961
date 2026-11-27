"""The invalid-input family: ``IV-BAD-OPTION``, ``IV-EMPTY`` and ``IV-OVERCAPACITY``.

The product's contract test of a target's *refusal* behaviour: a generator that silently accepts an
empty payload, an out-of-contract option or a payload larger than the symbol it was pinned to is
broken in a way no round-trip assertion can see.

* ``IV-EMPTY`` sends ``payload=b""`` at error level M in a PNG, with nothing else pinned: the empty
  payload is the only rule the request breaks, so a target that declares only ``png`` can serve it
  and its own defaults decide the geometry. A declared rejection **is** the expected outcome; a
  returned symbol is a *documented success* and passes too -- declared rejection **or** documented
  success, never a crash, never a partially written file. A crash, a ``TargetContractError`` or an
  empty or wrong-format artefact fails with ``Reason.CONTRACT_ERROR``.
* ``IV-OVERCAPACITY`` pins Version 10 at error level M and sends one byte more than Version 10 holds
  at level **L**. A declared rejection passes; a returned symbol is decoded once and fails with
  ``Reason.CAPACITY_VIOLATION``, recording ``requested_len``, ``capacity``, ``decoded_len`` and
  ``first_divergence_offset`` -- the measurements that say how far over capacity the request was and
  what the target returned instead of a rejection.
* ``IV-BAD-OPTION`` runs four probes, each breaking exactly one rule the port's builder enforces:
  ``ec_level="X"``, ``module_size_px=0``, ``quiet_zone_modules=-1``, and a colour pair below the
  target's **declared** ``contrast_floor``. A returned symbol fails with ``Reason.VALUE_MISMATCH``.

Why the over-capacity payload is measured against level L: the probe pins Version 10 at level M and
must send a payload no target can fit by quietly encoding one level lower, which is what
``wrong_ec_level`` does. Level L holds more than level M, so the length is
``max_payload_bytes(10, "L", "byte") + 1`` -- 272 bytes against level M's 213 -- read from
:mod:`qrcheck.corpus.capacity` at call time.

Every request is built by :func:`qrcheck.checks.support.construct_out_of_contract`, the one
documented bypass of the validating builder, and every result is recorded with
``out_of_contract=True``. The workload is six cases and six ``generate`` calls -- 1/1/0 for
``IV-EMPTY``, 1/1/1 for ``IV-OVERCAPACITY``, 4/4/0 for ``IV-BAD-OPTION`` -- which is
:data:`EXPECTED_ENCODES`, with no retry.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final

from qrcheck.checks import support
from qrcheck.config.models import DEFAULT_CONTRAST_FLOOR, TargetConfig
from qrcheck.corpus.capacity import max_payload_bytes
from qrcheck.registry.check import (
    DEPENDENCY_PRIMARY_DECODER,
    Case,
    RunContext,
    check,
)
from qrcheck.report.model import CaseResult, Reason
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetError,
)

__all__ = [
    "EXPECTED_ENCODES",
    "bad_option_cases",
    "empty_cases",
    "overcapacity_cases",
]

# The per-check adapter-invocation counts: one ``generate`` per case, and no second call anywhere;
# ``tests/unit/test_checks_negative.py`` asserts the observed count.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "IV-BAD-OPTION": 4,
        "IV-EMPTY": 1,
        "IV-OVERCAPACITY": 1,
    }
)

# The stable case identifiers: ``synthetic:<name>`` for a case the framework constructs rather than
# draws from the corpus.
_EMPTY_CASE_ID: Final[str] = "synthetic:iv-empty"
_OVERCAPACITY_CASE_ID: Final[str] = "synthetic:iv-overcapacity"

# The legal payload the probes that break something other than the payload carry.  Small
# enough that an unpinned request fits a small symbol at every error level, so the one
# broken rule is always the option under test and never an incidental capacity overflow.
_PROBE_PAYLOAD: Final[bytes] = b"qrcheck invalid-input probe"

# The error level every probe but the invalid-level one requests.  A lawful level, so the
# option under test is the only rule the request breaks.
_PROBE_EC_LEVEL: Final[str] = "M"

# ``IV-OVERCAPACITY``'s pin and levels: the version is pinned so the capacity bound is a fact about
# one symbol rather than a scan; the requested level is M and the payload is sized against level L,
# so no lower encoding level can rescue it.  The capacity unit follows the mode: byte mode counts
# bytes.
_OVERCAPACITY_VERSION: Final[int] = 10
_OVERCAPACITY_REQUESTED_EC: Final[str] = "M"
_OVERCAPACITY_SIZING_EC: Final[str] = "L"
_OVERCAPACITY_MODE: Final[str] = "byte"

# The filler the over-capacity payload is built from: lowercase letters and a colon put the bytes
# outside the numeric and alphanumeric charsets, so every encoder reads it in byte mode.
_OVERCAPACITY_FILLER: Final[bytes] = b"qrcheck-overcapacity-probe:"

# The background the below-floor contrast probe measures against, and the grey ladder it walks from
# darkest to lightest.  Ladder order is lightness order, so the ratio against white strictly
# decreases and the first entry below the declared ``contrast_floor`` is the darkest admissible
# probe.  ``0x88`` is first because it measures ≈3.5:1, the ratio that sits below the configured
# default floor; white on white, ratio 1.0, is below any floor the schema accepts.
_CONTRAST_BACKGROUND_RGB: Final[tuple[int, int, int]] = (0xFF, 0xFF, 0xFF)
_CONTRAST_GREY_LADDER: Final[tuple[int, ...]] = (
    0x88,
    0x99,
    0xAA,
    0xBB,
    0xCC,
    0xD8,
    0xE8,
    0xF0,
    0xF8,
    0xFC,
    0xFF,
)

# The error level ``IV-BAD-OPTION``'s error-level probe sends: a value outside the closed
# ``L``/``M``/``Q``/``H`` vocabulary the port's builder accepts.  Named here so the probe
# and the result's expectation cannot disagree about which illegal value was sent.
_OUT_OF_CONTRACT_EC_LEVEL: Final[str] = "X"

# The key that marks a deliberately illegal request, written into the result's own projection so
# ``summary.json`` and ``request.json`` both say it.  The evidence store sets the same name; a pass
# writes no evidence, and would otherwise carry an unmarked probe request.
_OUT_OF_CONTRACT_KEY: Final[str] = "out_of_contract"


class _OnAcceptedSymbol(enum.StrEnum):
    """What a returned symbol means for the check that received it.

    The target's refusal is the expected outcome of every probe in this family, so the interesting
    case is the one where a symbol comes back anyway, and this is the only place the three checks
    differ. ``DOCUMENTED_SUCCESS`` is ``IV-EMPTY``'s; ``CAPACITY_VIOLATION`` is
    ``IV-OVERCAPACITY``'s, where an accepted over-capacity request is the defect; ``VALUE_MISMATCH``
    is ``IV-BAD-OPTION``'s, where the target has not done what it declared.
    """

    DOCUMENTED_SUCCESS = "documented_success"
    CAPACITY_VIOLATION = "capacity_violation"
    VALUE_MISMATCH = "value_mismatch"


@dataclasses.dataclass(frozen=True)
class _Requested:
    """One probe's request, plus the premise its result records beside the outcome.

    Built per case execution rather than once at case enumeration, so a defect in a probe's
    arithmetic becomes one case's ``error`` result instead of failing the whole check's case list.
    ``expected`` reaches the report's ``expected`` field and ``request_extra`` the resolved premise
    in ``request.json``, so a reader can check the premise -- the measured contrast ratio and
    declared floor, the requested length and capacity -- without consulting this module.
    """

    request: GenerateRequest
    expected: Mapping[str, Any]
    request_extra: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@check(
    id="IV-EMPTY",
    title="An empty payload is refused, or accepted as a documented success",
    family="invalid",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def empty_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield ``IV-EMPTY``'s single case: an empty payload at error level M.

    ``payload=b""`` at error level M in a PNG, and nothing else pinned: the empty payload is the
    only rule the request breaks, so a target declaring nothing beyond ``png`` can serve it and the
    geometry is whatever that target renders by default.

    Yields:
        One case, ``synthetic:iv-empty``.
    """
    yield _case(
        "IV-EMPTY",
        _EMPTY_CASE_ID,
        _OnAcceptedSymbol.DOCUMENTED_SUCCESS,
        _empty_request,
    )


@check(
    id="IV-OVERCAPACITY",
    title="A payload over the pinned version's capacity is refused",
    family="invalid",
    requires=frozenset({Capability.png, Capability.version_pin}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def overcapacity_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield ``IV-OVERCAPACITY``'s single case: a payload over Version 10's capacity.

    Yields:
        One case, ``synthetic:iv-overcapacity``.
    """
    yield _case(
        "IV-OVERCAPACITY",
        _OVERCAPACITY_CASE_ID,
        _OnAcceptedSymbol.CAPACITY_VIOLATION,
        _overcapacity_request,
    )


@check(
    id="IV-BAD-OPTION",
    title="An option outside the request contract is refused",
    family="invalid",
    # Three of the four probes pin a field one of these capabilities owns -- module size 0, quiet
    # zone -1, a below-floor colour pair -- and a target must accept the field before it can be
    # asked to reject an illegal value for it. Declaring them keeps the pre-I/O gate and this
    # table in agreement, so the probes the target can serve are never skipped case by case.
    requires=frozenset(
        {
            Capability.png,
            Capability.ec_level,
            Capability.module_size,
            Capability.quiet_zone,
            Capability.color,
        }
    ),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def bad_option_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield ``IV-BAD-OPTION``'s four cases, ascending by case identifier.

    The order is the identifiers' own rather than the order the probes were thought of, so the
    sequence is the one every report and every evidence directory lists.

    Yields:
        ``synthetic:iv-bad-option-contrast``, ``-ec``, ``-module-size`` and ``-quiet-zone``.
    """
    for probe in _BAD_OPTION_PROBES:
        yield _case(
            "IV-BAD-OPTION",
            probe.case_id,
            _OnAcceptedSymbol.VALUE_MISMATCH,
            probe.build,
        )


def _empty_request(ctx: RunContext) -> _Requested:
    """Return the empty-payload probe's request and premise.

    Nothing but the payload is pinned, so the empty payload is the only rule the request breaks and
    a target that declares only ``png`` can serve it. Both outcomes are acceptable, which is why the
    premise records it: a declared rejection passes, and a documented success passes with
    ``decoded["outcome"] = "documented_success"``. Only a crash or an incomplete artefact fails.
    """
    request = support.construct_out_of_contract(b"", _PROBE_EC_LEVEL)
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "documented_success_accepted": True,
        "payload_len": 0,
    }
    return _Requested(request=request, expected=expected)


def _overcapacity_request(ctx: RunContext) -> _Requested:
    """Return the over-capacity probe's request and premise.

    The payload is one byte longer than the pinned version holds at level **L**, so it exceeds the
    capacity at the requested level M by a wide margin and cannot be rescued by a target that
    quietly encodes one level lower. Both lengths are read from :mod:`qrcheck.corpus.capacity` at
    call time; the requested level is the capacity the assertion is measured against.
    """
    capacity = max_payload_bytes(
        _OVERCAPACITY_VERSION, _OVERCAPACITY_REQUESTED_EC, _OVERCAPACITY_MODE
    )
    sizing_capacity = max_payload_bytes(
        _OVERCAPACITY_VERSION, _OVERCAPACITY_SIZING_EC, _OVERCAPACITY_MODE
    )
    payload = _overcapacity_payload(sizing_capacity + 1)
    request = support.construct_out_of_contract(
        payload,
        _OVERCAPACITY_REQUESTED_EC,
        version=_OVERCAPACITY_VERSION,
    )
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "requested_len": len(payload),
        "capacity": capacity,
        "sizing_capacity": sizing_capacity,
        "version": _OVERCAPACITY_VERSION,
        "ec_level": _OVERCAPACITY_REQUESTED_EC,
    }
    request_extra: dict[str, Any] = {
        "requested_len": len(payload),
        "capacity": capacity,
        "sizing_capacity": sizing_capacity,
    }
    return _Requested(request=request, expected=expected, request_extra=request_extra)


def _ec_request(ctx: RunContext) -> _Requested:
    """Return the invalid-error-level probe's request and premise.

    ``ec_level="X"`` is outside ``L``/``M``/``Q``/``H``, so the port's builder would refuse it.
    """
    request = support.construct_out_of_contract(_PROBE_PAYLOAD, _OUT_OF_CONTRACT_EC_LEVEL)
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "offending_field": "ec_level",
        "value": _OUT_OF_CONTRACT_EC_LEVEL,
    }
    return _Requested(request=request, expected=expected)


def _module_size_request(ctx: RunContext) -> _Requested:
    """Return the zero-module-size probe's request and premise.

    ``module_size_px=0`` is the one rule the request breaks.
    """
    request = support.construct_out_of_contract(
        _PROBE_PAYLOAD, _PROBE_EC_LEVEL, module_size_px=0
    )
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "offending_field": "module_size_px",
        "value": 0,
    }
    return _Requested(request=request, expected=expected)


def _quiet_zone_request(ctx: RunContext) -> _Requested:
    """Return the negative-quiet-zone probe's request and premise.

    ``quiet_zone_modules=-1`` is the one rule the request breaks.
    """
    request = support.construct_out_of_contract(
        _PROBE_PAYLOAD, _PROBE_EC_LEVEL, quiet_zone_modules=-1
    )
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "offending_field": "quiet_zone_modules",
        "value": -1,
    }
    return _Requested(request=request, expected=expected)


def _contrast_request(ctx: RunContext) -> _Requested:
    """Return the below-floor-contrast probe's request and premise.

    The floor is the target's **declared** ``contrast_floor``, read from the resolved configuration
    for the target under test -- never a literal here, because a threshold has one home in
    ``qrcheck.toml``. The request is a grey on white measuring ≈3.5:1 against the configured
    floor; a target that declares a tighter floor gets a lighter grey that is still genuinely
    below it, and the measured ratio is recorded so the premise is auditable from the report alone.
    """
    floor = _declared_contrast_floor(ctx)
    foreground = _below_floor_foreground(floor)
    ratio = _measured_contrast_ratio(foreground, _CONTRAST_BACKGROUND_RGB)
    request = support.construct_out_of_contract(
        _PROBE_PAYLOAD,
        _PROBE_EC_LEVEL,
        foreground_rgb=foreground,
        background_rgb=_CONTRAST_BACKGROUND_RGB,
    )
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "offending_field": "foreground_rgb",
        "contrast_ratio": round(ratio, 3),
        "contrast_floor": floor,
    }
    request_extra: dict[str, Any] = {
        "contrast_floor": floor,
        "contrast_ratio": round(ratio, 3),
    }
    return _Requested(request=request, expected=expected, request_extra=request_extra)


@dataclasses.dataclass(frozen=True)
class _OptionProbe:
    """One ``IV-BAD-OPTION`` case: its stable identifier and its request builder."""

    case_id: str
    build: Callable[[RunContext], _Requested]


# The four probes, ascending by case identifier, which is the order ``bad_option_cases``
# yields them in.  Each breaks exactly one rule of the request contract: the contrast floor
# the target itself declares, the error level, the module size, and the quiet zone.
_BAD_OPTION_PROBES: Final[tuple[_OptionProbe, ...]] = (
    _OptionProbe("synthetic:iv-bad-option-contrast", _contrast_request),
    _OptionProbe("synthetic:iv-bad-option-ec", _ec_request),
    _OptionProbe("synthetic:iv-bad-option-module-size", _module_size_request),
    _OptionProbe("synthetic:iv-bad-option-quiet-zone", _quiet_zone_request),
)


@dataclasses.dataclass(frozen=True)
class _Planned:
    """One case's identity, its expected-outcome rule and the builder of its request.

    The three travel together because every executor needs all three, and the identity is what the
    result, its evidence directory and its failure identifier are filed under.
    """

    check_id: str
    case_id: str
    on_accept: _OnAcceptedSymbol
    build: Callable[[RunContext], _Requested]


def _case(
    check_id: str,
    case_id: str,
    on_accept: _OnAcceptedSymbol,
    build: Callable[[RunContext], _Requested],
) -> Case:
    """Return the registry case that executes one probe.

    The case carries only the identifier and a callable, so the runner decides whether it runs at
    all; the request is built inside the callable, which keeps a defect in one probe's arithmetic
    from failing the enumeration of a check's other cases.

    Raises:
        ValueError: ``case_id`` is not a ``synthetic:`` identifier, or it is the runner's own
            ``synthetic:target``; or an identifier is empty.
    """
    planned = _Planned(
        check_id=check_id, case_id=case_id, on_accept=on_accept, build=build
    )

    def execute(ctx: RunContext) -> CaseResult:
        return _run(ctx, planned)

    return support.synthetic_case(check_id, case_id, execute)


def _run(ctx: RunContext, planned: _Planned) -> CaseResult:
    """Run one probe: send its request and report what the target did with it.

    One execution path for all six cases, so the family cannot treat two probes differently by
    accident. The adapter error table is :func:`qrcheck.checks.support.case_status_for_error`'s and
    is not restated here: a ``TargetRejected`` passes with the target's text verbatim, a
    ``TargetContractError`` fails with ``contract_error``, a ``TargetUnsupported`` skips, and
    ``TargetUnavailable``/``TargetTimeout`` are re-raised by that translator. Exactly one
    ``generate`` call is made per case, which is the ``1``/``4`` of :data:`EXPECTED_ENCODES`.

    Raises:
        TargetUnavailable: The target could not be reached; the runner owns the one-error-result-
        per-check policy. TargetTimeout: No response within the configured timeout, for the same
        reason. ValueError: A probe's own arithmetic is defective -- a colour ladder that cannot
        place a ratio below the declared floor. The runner records an ``error``.
    """
    requested = planned.build(ctx)
    started = time.perf_counter()
    try:
        symbol = support.generate(ctx, requested.request)
    except TargetError as error:
        return support.record_evidence(
            ctx,
            _with_premise(
                support.case_status_for_error(
                    planned.check_id,
                    planned.case_id,
                    error,
                    request=requested.request,
                    out_of_contract=True,
                    duration_ms=_elapsed_ms(started),
                    seed=ctx.seed,
                ),
                requested.expected,
            ),
            out_of_contract=True,
            request_extra=requested.request_extra,
        )
    duration_ms = _elapsed_ms(started)
    if planned.on_accept is _OnAcceptedSymbol.DOCUMENTED_SUCCESS:
        return _documented_success(ctx, planned, requested, symbol, duration_ms)
    if planned.on_accept is _OnAcceptedSymbol.CAPACITY_VIOLATION:
        return _over_capacity_accepted(ctx, planned, requested, symbol, duration_ms)
    return _bad_option_accepted(ctx, planned, requested, symbol, duration_ms)


def _documented_success(
    ctx: RunContext,
    planned: _Planned,
    requested: _Requested,
    symbol: GeneratedSymbol,
    duration_ms: float,
) -> CaseResult:
    """Report an empty payload the target chose to encode anyway.

    Declared rejection **or** documented success is the correct assertion for ``IV-EMPTY``, so a
    returned symbol passes, and the case deliberately does not decode it. What is still asserted is
    that the answer is an artefact at all: an empty body or one of the wrong
    format is a partially written file, reported as ``Reason.CONTRACT_ERROR``.
    """
    complete = bool(symbol.image_bytes) and (
        symbol.output_format == requested.request.output_format
    )
    decoded: dict[str, Any] = {
        "outcome": "documented_success" if complete else "incomplete_artefact",
        "output_format": symbol.output_format,
        "symbol_len": len(symbol.image_bytes),
    }
    projection = _projection(requested.request)
    if complete:
        result = support.pass_(
            planned.check_id,
            planned.case_id,
            request=projection,
            decoded=decoded,
            expected=requested.expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    else:
        result = support.fail(
            planned.check_id,
            planned.case_id,
            reason=Reason.CONTRACT_ERROR,
            request=projection,
            decoded=decoded,
            expected=requested.expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        out_of_contract=True,
        request_extra=requested.request_extra,
    )


def _over_capacity_accepted(
    ctx: RunContext,
    planned: _Planned,
    requested: _Requested,
    symbol: GeneratedSymbol,
    duration_ms: float,
) -> CaseResult:
    """Report a symbol the target produced for an over-capacity request.

    This is the ``over_capacity_acceptance`` defect: the case fails with
    ``Reason.CAPACITY_VIOLATION`` whatever the symbol contains, because a version and level were
    pinned, the payload exceeds what that symbol holds, and the target answered with an artefact
    instead of a rejection. The symbol is decoded once so the evidence can say what it did:
    ``decoded_len`` and ``first_divergence_offset`` beside ``capacity``. A symbol neither decoder
    can read is still a violation; ``decoded_len`` is then ``None``.
    """
    decoded = support.decode(ctx, symbol.image_bytes, requested.request)
    measurements: dict[str, Any] = {
        "capacity": max_payload_bytes(
            _OVERCAPACITY_VERSION, _OVERCAPACITY_REQUESTED_EC, _OVERCAPACITY_MODE
        ),
        "decoded_len": (
            None if decoded.payload_bytes is None else len(decoded.payload_bytes)
        ),
        "first_divergence_offset": _first_divergence_offset(
            requested.request.payload, decoded.payload_bytes
        ),
    }
    result = support.fail(
        planned.check_id,
        planned.case_id,
        reason=Reason.CAPACITY_VIOLATION,
        request=_projection(requested.request),
        decoded={**decoded.as_dict(), **measurements},
        expected=requested.expected,
        duration_ms=duration_ms,
        seed=ctx.seed,
    )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        out_of_contract=True,
        request_extra=requested.request_extra,
    )


def _bad_option_accepted(
    ctx: RunContext,
    planned: _Planned,
    requested: _Requested,
    symbol: GeneratedSymbol,
    duration_ms: float,
) -> CaseResult:
    """Report a symbol the target produced for an out-of-contract option.

    Each probe breaks exactly one rule of the request contract, and a target that answers one with a
    symbol has accepted a request it had to refuse, so the case fails with
    ``Reason.VALUE_MISMATCH``. The symbol travels as evidence so a reader can see what was produced;
    nothing about it is decoded, because the finding is the acceptance, not the payload.
    """
    decoded: dict[str, Any] = {
        "outcome": "accepted_symbol",
        "output_format": symbol.output_format,
        "symbol_len": len(symbol.image_bytes),
    }
    result = support.fail(
        planned.check_id,
        planned.case_id,
        reason=Reason.VALUE_MISMATCH,
        request=_projection(requested.request),
        decoded=decoded,
        expected=requested.expected,
        duration_ms=duration_ms,
        seed=ctx.seed,
    )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        out_of_contract=True,
        request_extra=requested.request_extra,
    )


def _projection(request: GenerateRequest) -> dict[str, Any]:
    """Return the request's JSON-ready projection, marked as deliberately illegal.

    The marker is set on the result's own projection rather than left to the evidence store, because
    a
    *passing* case writes no evidence and would otherwise reach ``summary.json`` unmarked.
    """
    projected = request.as_dict()
    projected[_OUT_OF_CONTRACT_KEY] = True
    return projected


def _with_premise(result: CaseResult, premise: Mapping[str, Any]) -> CaseResult:
    """Return `result` with the probe's measurements recorded beside its expectation.

    The translator's own keys win on a collision, because its ``outcome`` names what the probe
    expected at the point the target failed -- ``usable_artefact`` for a contract error,
    ``declared_rejection`` for a declared rejection -- while the probe's measurements (the requested
    length and capacity, the measured ratio and declared floor, the offending field) are added
    beside it. The result is returned unchanged when it already carries the premise.
    """
    merged: dict[str, Any] = {**premise, **(result.expected or {})}
    if merged == (result.expected or {}):
        return result
    return dataclasses.replace(result, expected=merged)


def _overcapacity_payload(length: int) -> bytes:
    """Return a deterministic byte-mode payload of exactly `length` bytes.

    Byte mode is what the capacity figures are measured in, and the filler holds lowercase letters,
    so no encoder can spend a smaller charset and fit the payload after all.

    Raises:
        ValueError: ``length`` is not a positive integer -- a programming error, since the caller
            derives it from a capacity table that is never smaller than one byte.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"an over-capacity payload needs a positive length, got {length!r}")
    repeats = -(-length // len(_OVERCAPACITY_FILLER))
    return (_OVERCAPACITY_FILLER * repeats)[:length]


def _first_divergence_offset(requested: bytes, decoded: bytes | None) -> int | None:
    """Return the offset at which `decoded` first stops matching `requested`.

    ``None`` when the two are identical or when nothing was decoded at all. When the decoded payload
    is a strict prefix of the requested one -- the signature of a target that truncated instead of
    rejecting -- the offset is the shorter length, which is the truncation point.
    """
    if decoded is None:
        return None
    common = min(len(requested), len(decoded))
    for index in range(common):
        if requested[index] != decoded[index]:
            return index
    if len(requested) == len(decoded):
        return None
    return common


def _declared_contrast_floor(ctx: RunContext) -> float:
    """Return the contrast floor the target under test declares.

    Read from the resolved configuration for the running adapter's own ``[targets.<name>]`` record.
    A target with no record -- a stub in a test, say -- falls back to the declared default, the same
    floor ``RN-CONTRAST`` asserts; the number lives in :mod:`qrcheck.config.models`.
    """
    declared: TargetConfig | None = ctx.config.targets.get(ctx.adapter.name)
    if declared is None:
        return DEFAULT_CONTRAST_FLOOR
    return declared.contrast_floor


def _below_floor_foreground(floor: float) -> tuple[int, int, int]:
    """Return the darkest grey from the fixed ladder whose ratio sits below `floor`.

    The ladder runs from ``0x88`` -- the ≈3.5:1 pair, chosen whenever the target's
    ``contrast_floor`` is the configured default -- towards white, so a target declaring a tighter
    floor still receives a request genuinely below it.

    Raises:
        ValueError: No ladder entry measures below `floor`, which can only happen for a floor at or
            below ``1.0`` -- a value the schema refuses, so it means the record was hand-built.
    """
    for level in _CONTRAST_GREY_LADDER:
        candidate = (level, level, level)
        if _measured_contrast_ratio(candidate, _CONTRAST_BACKGROUND_RGB) < floor:
            return candidate
    raise ValueError(
        f"cannot place a below-floor contrast probe for contrast floor {floor!r}: every "
        f"one of the grey ladder's ratios against {_CONTRAST_BACKGROUND_RGB!r} is at or "
        f"above it, and a floor of 1.0 or less is outside the configuration schema's range"
    )


def _measured_contrast_ratio(
    foreground: tuple[int, int, int], background: tuple[int, int, int]
) -> float:
    """Return the WCAG contrast ratio between two colours, from the one implementation.

    Delegated to :func:`qrcheck.decode.consensus.contrast_ratio`, which the built-in targets refuse
    a request with and ``RN-CONTRAST`` asserts at or above, so the number this probe is constructed
    against and the number a target measures cannot drift apart. Imported at the point of use
    because that module loads ``numpy`` and ``Pillow``, and a process that only lists checks must
    not start an image stack.
    """
    from qrcheck.decode.consensus import contrast_ratio

    return contrast_ratio(foreground, background)


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, never negative.

    Recorded in the result and never asserted.
    """
    return max(0.0, (time.perf_counter() - started) * 1000.0)
