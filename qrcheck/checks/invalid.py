"""The invalid-input family: ``IV-BAD-OPTION``, ``IV-EMPTY`` and ``IV-OVERCAPACITY``.

AAP §0.8.1's component-to-check-family table, its "What the built-in targets must do for the
``IV-*`` cases" table, §0.8.2's validation rules and their documented bypass, and §0.9.1.
These three checks are the product's contract test of a target's *refusal* behaviour: a
generator that silently accepts an empty payload, an out-of-contract option or a payload
larger than the symbol it was pinned to is broken in a way no round-trip assertion can see,
because the symbol it returns may decode perfectly.

``IV-EMPTY`` sends ``payload=b""`` at level M in a PNG with a four-pixel module size and a
four-module quiet zone.  A declared rejection is the expected outcome; a returned symbol is a
*documented success* and passes too, which is AAP §0.3.5's substitute for the wrong assertion
that a target must reject anything the framework dislikes.  A crash, a ``TargetContractError``
or an artefact that is empty or of the wrong format fails with ``Reason.CONTRACT_ERROR``.
``IV-OVERCAPACITY`` pins Version 10 at level M and sends one byte more than Version 10 holds
at level **L**: a declared rejection passes, while a returned symbol is decoded once and fails
with ``Reason.CAPACITY_VIOLATION``, recording ``requested_len``, ``capacity``,
``decoded_len`` and ``first_divergence_offset`` -- the evidence AAP §0.4.5 requires of the
``over_capacity_acceptance`` defect.  ``IV-BAD-OPTION`` runs four probes, each breaking one
rule the port's builder enforces -- ``ec_level="X"``, ``module_size_px=0``,
``quiet_zone_modules=-1``, and a colour pair below the target's **declared** ``contrast_floor``
-- where a declared rejection passes and a returned symbol fails with
``Reason.VALUE_MISMATCH``, because the target accepted a request it must refuse.

The over-capacity length is ``max_payload_bytes(10, "L", "byte") + 1`` -- 272 bytes against
level M's 213 -- rather than §0.8.1's ``max_payload_bytes(10, "M", ...) + 1``, because §0.4.2
also requires the payload to exceed the L-level capacity: an ``M + 1`` payload would fit
comfortably after the ``defective`` target's D1 downshift and the probe would stop proving
anything about over-capacity acceptance.  Both figures come from
:mod:`qrcheck.corpus.capacity` at call time and are never transcribed.

Every request here is built by :func:`qrcheck.checks.support.construct_out_of_contract`,
AAP §0.8.2's one documented bypass of :func:`qrcheck.checks.support.build`, and every result
records ``out_of_contract=True`` so ``request.json`` states that the request was deliberately
illegal.

Six cases, one ``generate`` call each, and a decode only where an accepted symbol has to be
measured: ``IV-EMPTY`` 1 case, 1 encode, 0 decodes; ``IV-OVERCAPACITY`` 1 case, 1 encode, 1
decode; ``IV-BAD-OPTION`` 4 cases, 4 encodes, 0 decodes.  Those counts are
:data:`EXPECTED_ENCODES`, the declaration AAP §0.4.3 makes a counting stub assert, so a retry
added anywhere here would surface as the declared count plus one.  There is no retry, no
re-run-on-failure and no quarantine path.

:func:`qrcheck.checks.support.case_status_for_error` is the single translation of an adapter
error, called rather than reimplemented: a ``TargetRejected`` is a pass with the target's text
verbatim in ``decoded["rejection"]``, a ``TargetContractError`` is a failure with reason
``contract_error``, and a ``TargetUnsupported`` is a skip.  ``TargetUnavailable`` and
``TargetTimeout`` are not caught, because ``qrcheck.runner`` owns the one-error-result-per-check
target-loss policy, and a target that cannot be reached must not look like one that accepted an
illegal request.  There is no bare or broad ``except`` in this file.

This file owns no request validation (the port's builder), no capability pre-gate
(:func:`qrcheck.checks.support.generate`), no result vocabulary or failure identifier
(:mod:`qrcheck.report.model`), no evidence layout or artefact set
(:mod:`qrcheck.report.evidence`), and no threshold: the contrast floor is read from the
resolved :class:`~qrcheck.config.models.TargetConfig`.  The four-module size, the four-module
quiet zone and Version 10 are the request values AAP §0.8.1 tabulates, not thresholds.

Determinism: nothing here reads a clock for an assertion, a random source, ``hash()``,
``id()``, the filesystem or a locale; the six cases are generated in ``case_id`` order and the
contrast probe's colour ladder is a fixed tuple, so ``duration_ms`` is the only measurement
that varies between two runs and it is recorded and never asserted (AAP §0.5.5).
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

# The adapter-invocation counts AAP §0.4.3 declares for this family: one ``generate`` per
# case, and no second call anywhere.  ``tests/unit/test_checks_negative.py`` drives each
# check with a counting stub and asserts the observed count equals this table, so a retry
# introduced anywhere in this module fails that test as the declared count plus one.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "IV-BAD-OPTION": 4,
        "IV-EMPTY": 1,
        "IV-OVERCAPACITY": 1,
    }
)

# The stable case identifiers AAP §0.3.4 fixes: ``synthetic:<name>`` for a case the
# framework constructs rather than draws from the corpus.
_EMPTY_CASE_ID: Final[str] = "synthetic:iv-empty"
_OVERCAPACITY_CASE_ID: Final[str] = "synthetic:iv-overcapacity"

# The legal payload the probes that break something other than the payload carry.  Small
# enough that an unpinned request fits a small symbol at every error level, so the one
# broken rule is always the option under test and never an incidental capacity overflow.
_PROBE_PAYLOAD: Final[bytes] = b"qrcheck invalid-input probe"

# The error level every probe but the invalid-level one requests.  A lawful level, so the
# option under test is the only rule the request breaks.
_PROBE_EC_LEVEL: Final[str] = "M"

# The module size and quiet zone ``IV-EMPTY`` asks for (AAP §0.8.1): both are lawful
# values, so the empty payload is the only thing that request breaks.
_PROBE_MODULE_SIZE_PX: Final[int] = 4
_PROBE_QUIET_ZONE_MODULES: Final[int] = 4

# ``IV-OVERCAPACITY``'s pin and levels.  The version is pinned so the capacity bound is a
# fact about one symbol rather than a scan; the requested level is M, and the payload is
# sized against level L so AAP §0.4.2's cascade containment holds (see the module
# docstring).  The capacity unit follows the mode: byte mode counts bytes.
_OVERCAPACITY_VERSION: Final[int] = 10
_OVERCAPACITY_REQUESTED_EC: Final[str] = "M"
_OVERCAPACITY_SIZING_EC: Final[str] = "L"
_OVERCAPACITY_MODE: Final[str] = "byte"

# The filler the over-capacity payload is built from.  Lowercase letters and a colon put
# the bytes outside the numeric and alphanumeric charsets, so every encoder -- the two
# built-in targets and :func:`qrcheck.corpus.payloads.mode_of` alike -- reads the payload
# in byte mode, which is the mode the capacity figures above are measured in.
_OVERCAPACITY_FILLER: Final[bytes] = b"qrcheck-overcapacity-probe:"

# The background the below-floor contrast probe measures against, and the grey ladder it
# walks from darkest to lightest.  Ladder order is lightness order, so the ratio against a
# white background strictly decreases along it and the first entry below the declared
# floor is the darkest -- that is, the closest to the floor -- admissible probe.  ``0x88``
# is first because it is the ≈3.5:1 pair AAP §0.8.1 tabulates, and it is the entry chosen
# whenever the target declares the documented 7.0 floor; a target that declares a tighter
# floor still gets a genuine below-floor probe, and the last entry (white on white, ratio
# 1.0) is below any floor the configuration schema accepts, since it requires a floor
# greater than 1.0.
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

# The key that marks a deliberately illegal request, written into the result's own
# request projection so ``summary.json`` and ``request.json`` both say it.  The evidence
# store sets the same key from its own constant; the two agree because the name is AAP
# §0.8.2's, and a pass -- which writes no evidence at all -- is the case that would
# otherwise carry an unmarked probe request.
_OUT_OF_CONTRACT_KEY: Final[str] = "out_of_contract"


class _OnAcceptedSymbol(enum.StrEnum):
    """What a returned symbol means for the check that received it.

    The target's refusal is the expected outcome of every probe in this family, so the
    interesting case is the one where a symbol comes back anyway: this decides what the
    check reports then, and it is the only place the three checks differ in that respect.

    ``DOCUMENTED_SUCCESS`` is ``IV-EMPTY``'s -- AAP §0.3.5's substitute for "the target
    must reject anything the framework dislikes" is "declared rejection **or** documented
    success, never a crash, never a partially written file", so an accepted empty payload
    passes while an incomplete artefact still fails.

    ``CAPACITY_VIOLATION`` is ``IV-OVERCAPACITY``'s: an accepted over-capacity request is
    the ``over_capacity_acceptance`` defect, so the symbol is decoded once and the case
    fails with the measurements AAP §0.4.5 names.

    ``VALUE_MISMATCH`` is ``IV-BAD-OPTION``'s: a target that accepts an out-of-contract
    option has not done what it declared it would do.
    """

    DOCUMENTED_SUCCESS = "documented_success"
    CAPACITY_VIOLATION = "capacity_violation"
    VALUE_MISMATCH = "value_mismatch"


@dataclasses.dataclass(frozen=True)
class _Requested:
    """One probe's request, plus the premise its result records beside the outcome.

    Built per case execution rather than once at case enumeration, so a defect in a
    probe's arithmetic becomes one case's ``error`` result instead of failing the whole
    check's case list.

    ``expected`` is what the case asserted and reaches the report's ``expected`` field;
    ``request_extra`` is the resolved premise that belongs in ``request.json`` beside the
    request itself (AAP §0.3.4: the case's resolved parameters).  Both are recorded
    because a reader must be able to check the probe's premise -- the measured contrast
    ratio and the declared floor, the requested length and the capacity -- without
    consulting this module.
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

    The request is the one AAP §0.8.1 tabulates -- ``payload=b""``, error level M, a PNG
    with a four-pixel module size and a four-module quiet zone -- so the empty payload is
    the only rule it breaks.

    Args:
        ctx: The run's context, passed through to the case's executor.

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

    Args:
        ctx: The run's context, passed through to the case's executor.

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
    requires=frozenset({Capability.png, Capability.ec_level}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def bad_option_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield ``IV-BAD-OPTION``'s four cases, ascending by case identifier.

    The order is the identifiers' own rather than the order the probes were thought of, so
    the sequence is the one every report and every evidence directory would list anyway
    (AAP §0.5.3).

    Args:
        ctx: The run's context, passed through to each case's executor.

    Yields:
        Four cases: ``synthetic:iv-bad-option-contrast``, ``-ec``, ``-module-size`` and
        ``-quiet-zone``.
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

    Both outcomes are acceptable, which is why the premise records it: a declared
    rejection passes, and a documented success passes with
    ``decoded["outcome"] = "documented_success"``.  Only a crash or an incomplete
    artefact fails (AAP §0.3.5).

    Args:
        ctx: Unused; the signature is this family's single builder shape, so one executor
            can call every builder.

    Returns:
        The out-of-contract request and the premise ``request.json`` records.
    """
    request = support.construct_out_of_contract(
        b"",
        _PROBE_EC_LEVEL,
        module_size_px=_PROBE_MODULE_SIZE_PX,
        quiet_zone_modules=_PROBE_QUIET_ZONE_MODULES,
    )
    expected: dict[str, Any] = {
        "outcome": "declared_rejection",
        "documented_success_accepted": True,
        "payload_len": 0,
    }
    return _Requested(request=request, expected=expected)


def _overcapacity_request(ctx: RunContext) -> _Requested:
    """Return the over-capacity probe's request and premise.

    The payload is one byte longer than the pinned version holds at level **L**, so it
    exceeds the capacity at the requested level M by a wide margin and cannot be rescued
    by a target that quietly encodes one level lower (AAP §0.4.2).  Both lengths are read
    from :mod:`qrcheck.corpus.capacity` at call time; the requested level is the one the
    request names, and that is the capacity the assertion is measured against.

    Args:
        ctx: Unused; the signature is this family's single builder shape.

    Returns:
        The out-of-contract request and the premise ``request.json`` records.
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

    Args:
        ctx: Unused; the signature is this family's single builder shape.

    Returns:
        The out-of-contract request and the premise ``request.json`` records.
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

    Args:
        ctx: Unused; the signature is this family's single builder shape.

    Returns:
        The out-of-contract request and the premise ``request.json`` records.
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

    Args:
        ctx: Unused; the signature is this family's single builder shape.

    Returns:
        The out-of-contract request and the premise ``request.json`` records.
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

    The floor is the target's **declared** ``contrast_floor``, read from the resolved
    configuration for the target under test -- never a literal here, because a threshold
    has one home and this is not it (AAP §0.6.2).  AAP §0.8.1's request is a grey on
    white measuring ≈3.5:1 against the documented 7.0 floor; a target that declares a
    tighter floor gets a lighter grey that is still genuinely below it, and the measured
    ratio is recorded so the premise is auditable from the report alone.

    Args:
        ctx: The run's context; its configuration names the declared floor.

    Returns:
        The out-of-contract request and the premise ``request.json`` records.
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

    The three travel together because every executor needs all three and the identity is
    what the result, its evidence directory and its failure identifier are filed under.
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

    The case carries only the identifier and a callable, so the runner decides whether it
    runs at all; the request is built inside the callable, which is what keeps a defect in
    one probe's arithmetic from failing the enumeration of a check's other cases.

    Args:
        check_id: The owning check's stable identifier.
        case_id: The case's stable identifier, in ``synthetic:<name>`` form.
        on_accept: What a returned symbol means for this check.
        build: The builder of the request this case sends.

    Returns:
        The case, ready for the runner.

    Raises:
        ValueError: ``case_id`` is not a ``synthetic:`` identifier, or it is the runner's
            own ``synthetic:target``; or an identifier is empty.
    """
    planned = _Planned(
        check_id=check_id, case_id=case_id, on_accept=on_accept, build=build
    )

    def execute(ctx: RunContext) -> CaseResult:
        return _run(ctx, planned)

    return support.synthetic_case(check_id, case_id, execute)


def _run(ctx: RunContext, planned: _Planned) -> CaseResult:
    """Run one probe: send its request and report what the target did with it.

    One execution path for all six cases, so the family cannot treat two probes
    differently by accident.  The adapter error table is
    :func:`qrcheck.checks.support.case_status_for_error`'s and is not restated here: a
    ``TargetRejected`` passes with the target's text verbatim, a ``TargetContractError``
    fails with ``contract_error``, a ``TargetUnsupported`` skips -- and
    ``TargetUnavailable``/``TargetTimeout`` are re-raised by that translator, because a
    target that cannot be reached must not look like one that answered.

    Exactly one ``generate`` call is made per case, which is the ``1``/``4`` of
    :data:`EXPECTED_ENCODES`; nothing here retries.

    Args:
        ctx: The run's context.
        planned: The case's identity, expected-outcome rule and request builder.

    Returns:
        The case's result, carrying its evidence directory when it did not pass.

    Raises:
        TargetUnavailable: The target could not be reached; the runner owns the
            one-error-result-per-check policy.
        TargetTimeout: No response within the configured timeout, for the same reason.
        ValueError: A probe's own arithmetic is defective -- a colour ladder that cannot
            place a ratio below the declared floor.  The runner records an ``error``.
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

    AAP §0.3.5 makes "declared rejection **or** documented success" the correct assertion
    for ``IV-EMPTY``, so a returned symbol passes -- and the case deliberately does not
    decode it, because the product has no assertion to make about what an encoder does
    with an empty payload.  What is still asserted is that the answer is an artefact at
    all: an empty body or one of the wrong format is a partially written file, which is
    the failure the same rule names.

    Args:
        ctx: The run's context.
        planned: The case's identity and expected-outcome rule.
        requested: The request that was sent, with its premise.
        symbol: What the target returned.
        duration_ms: How long the invocation took; recorded, never asserted.

    Returns:
        A passing result, or a failure with ``Reason.CONTRACT_ERROR`` when the artefact is
        incomplete.  No evidence is persisted for a pass, which is the store's policy.
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

    This is the ``over_capacity_acceptance`` defect, and the case fails with
    ``Reason.CAPACITY_VIOLATION`` whatever the symbol contains: a version and level were
    pinned, the payload exceeds what that symbol holds, and the target answered with an
    artefact instead of a rejection.  The symbol is decoded exactly once so the evidence
    can say *what* it did with the request -- AAP §0.4.5 asks for ``decoded_len`` and
    ``first_divergence_offset``, and AAP §0.6.1 reads them from ``decoded.json`` beside
    ``capacity``.

    A symbol neither decoder can read is still a violation; ``decoded_len`` is then
    ``None`` and the decoder's own complaint travels in ``decoded["per_decoder"]``.

    Args:
        ctx: The run's context.
        planned: The case's identity and expected-outcome rule.
        requested: The request that was sent, with its premise.
        symbol: What the target returned.
        duration_ms: How long the invocation took; recorded, never asserted.

    Returns:
        A failing result with ``Reason.CAPACITY_VIOLATION`` and its evidence directory.
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

    Each probe breaks exactly one rule of the request contract, and a target that answers
    one with a symbol has accepted a request it had to refuse, so the case fails with
    ``Reason.VALUE_MISMATCH``.  The symbol travels as evidence so a reader can see what
    was produced for the illegal request; nothing about it is decoded, because the
    finding is the acceptance, not the payload.

    Args:
        ctx: The run's context.
        planned: The case's identity and expected-outcome rule.
        requested: The request that was sent, with its premise.
        symbol: What the target returned.
        duration_ms: How long the invocation took; recorded, never asserted.

    Returns:
        A failing result with ``Reason.VALUE_MISMATCH`` and its evidence directory.
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

    The marker is set on the result's own projection rather than left to the evidence
    store, because a *passing* case writes no evidence at all and would otherwise reach
    ``summary.json`` with an unmarked probe request.

    Args:
        request: The request the case sent.

    Returns:
        The projection ``summary.json`` and ``request.json`` both carry.
    """
    projected = request.as_dict()
    projected[_OUT_OF_CONTRACT_KEY] = True
    return projected


def _with_premise(result: CaseResult, premise: Mapping[str, Any]) -> CaseResult:
    """Return `result` with the probe's measurements recorded beside its expectation.

    The translator's own keys win on a collision, because its ``outcome`` names what the
    probe expected at the point the target failed -- ``usable_artefact`` for a contract
    error, ``declared_rejection`` for a declared rejection -- while the probe's
    measurements (the requested length and capacity, the measured ratio and declared
    floor, the offending field) are added beside it.

    Args:
        result: The result the translator produced.
        premise: The probe's premise, as :class:`_Requested` recorded it.

    Returns:
        The result, unchanged when it already carries the premise.
    """
    merged: dict[str, Any] = {**premise, **(result.expected or {})}
    if merged == (result.expected or {}):
        return result
    return dataclasses.replace(result, expected=merged)


def _overcapacity_payload(length: int) -> bytes:
    """Return a deterministic byte-mode payload of exactly `length` bytes.

    Byte mode is what the capacity figures are measured in, and the filler holds lowercase
    letters, so no encoder can spend a smaller charset and fit the payload after all.

    Args:
        length: The payload length in bytes, at least one.

    Returns:
        The payload.

    Raises:
        ValueError: ``length`` is not a positive integer -- a programming error, since the
            caller derives it from a capacity table that is never smaller than one byte.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"an over-capacity payload needs a positive length, got {length!r}")
    repeats = -(-length // len(_OVERCAPACITY_FILLER))
    return (_OVERCAPACITY_FILLER * repeats)[:length]


def _first_divergence_offset(requested: bytes, decoded: bytes | None) -> int | None:
    """Return the offset at which `decoded` first stops matching `requested`.

    ``None`` when the two are identical or when nothing was decoded at all.  When the
    decoded payload is a strict prefix of the requested one -- the signature of a target
    that truncated instead of rejecting -- the offset is the shorter length, which is the
    truncation point.

    Args:
        requested: The payload the framework asked for.
        decoded: The payload the symbol carried, or ``None`` when it did not decode.

    Returns:
        The zero-based offset, or ``None``.
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

    Read from the resolved configuration for the running adapter's own ``[targets.<name>]``
    record, so the probe measures against the floor the target is actually held to.  A
    target with no record -- a stub in a test, say -- falls back to the declared default,
    which is the same floor ``RN-CONTRAST`` asserts and the same one a target that declares
    nothing is held to (AAP §0.3.4); the number itself lives in
    :mod:`qrcheck.config.models` and is never written here.

    Args:
        ctx: The run's context; only its configuration and the adapter's name are read.

    Returns:
        The floor as a float greater than ``1.0``.
    """
    declared: TargetConfig | None = ctx.config.targets.get(ctx.adapter.name)
    if declared is None:
        return DEFAULT_CONTRAST_FLOOR
    return declared.contrast_floor


def _below_floor_foreground(floor: float) -> tuple[int, int, int]:
    """Return the darkest grey from the fixed ladder whose ratio sits below `floor`.

    The ladder runs from ``0x88`` -- AAP §0.8.1's ≈3.5:1 pair, chosen whenever the target
    declares the documented 7.0 floor -- towards white, so a target declaring a tighter
    floor still receives a request that is genuinely below it rather than a probe that
    silently stopped being one.

    Args:
        floor: The target's declared contrast floor, greater than ``1.0``.

    Returns:
        The grey as a three-channel colour.

    Raises:
        ValueError: No ladder entry measures below `floor`, which can only happen for a
            floor at or below ``1.0`` -- a value the configuration schema refuses, so it
            means the record was hand-built.
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

    Delegated to :func:`qrcheck.decode.consensus.contrast_ratio`, which the built-in
    targets refuse a request with and ``RN-CONTRAST`` asserts at or above, so the number
    this probe is constructed against and the number a target measures cannot drift apart.
    Imported at the point of use because that module loads ``numpy`` and ``Pillow`` when it
    is imported, and a process that only lists checks must not start an image stack --
    the rule :func:`qrcheck.checks.support.decode` follows for the same reason.

    Args:
        foreground: The dark-module colour, three ``0-255`` integers.
        background: The light-module colour, three ``0-255`` integers.

    Returns:
        The ratio, ``>= 1.0``.
    """
    from qrcheck.decode.consensus import contrast_ratio

    return contrast_ratio(foreground, background)


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, never negative.

    Args:
        started: A :func:`time.perf_counter` reading taken before the target was called.

    Returns:
        The duration in milliseconds, recorded in the result and never asserted.
    """
    return max(0.0, (time.perf_counter() - started) * 1000.0)
