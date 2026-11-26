"""Every one of the twenty-one checks must fail on the defect it asserts against.

Plan item served: AAP §0.4.4 -- "``tests/unit/test_checks_negative.py`` is the mechanism that
catches [a check that cannot fail]" -- together with §0.4.3 and §0.5.6 (the declared
adapter-invocation counts, because a retry registers as the declared count plus one) and
§0.8.1's case matrix.  No project rules exist for this product (AAP §0.12), so the binding
constraints here are the AAP's own MUST/MUST NOT list: no check is weakened, skipped,
``xfail``-ed, retried or threshold-relaxed to make a target pass, and this module is the
enforcement of that rule at the check level.

What a negative case is
-----------------------
For each of the twenty-one registered checks this module drives the check's own ``cases``
generator with a protocol-conforming stub adapter and a stubbed decode result carrying the
**one** defect that check exists to detect: a payload differing in one byte, a version one
above the oracle minimum, an error level that disagrees with the format information, a mask
of nine, a grid that is not the oracle's, a quiet zone of one module, a module edge two
pixels wide of the request, a symbol that does not decode under damage, an over-capacity
request the target accepted, a mask that changes between two identical calls, an unreachable
performance floor, a temporary file left behind.  The assertion is the check's exact
:class:`~qrcheck.report.model.Reason`, never merely "it failed": a case that fails for an
unrelated reason -- a stub returning bad bytes, a missing fixture -- would satisfy
``status is failed`` and say nothing about the assertion under test (AAP §0.4.4).

Every check also carries a **control** round, driven through exactly the same harness with
no defect injected, which must pass.  The control is what makes the negative round
attributable: the conforming renderer and the conforming echo decoder below are what a
conforming target looks like to a check, so a negative round that fails proves the failure
came from the injected defect and not from the harness.  A check whose control fails is
reported as a defect of this module's harness, never as a pass.

The conforming stubs, and why they are stubs rather than the real decoders
--------------------------------------------------------------------------
* :func:`_render` is a small segno-backed target: it honours the requested module size,
  quiet zone, colours, version and mask pins, the ``png``/``svg`` format and the logo
  overlay, which is exactly what the five ``RN-*`` checks measure.  ``segno`` is used here
  for the same reason AAP §0.5 sanctions it in the render, damage, symbol and matrix cases:
  those checks genuinely read an image.
* :func:`_echo` is a conforming decoder: it returns the requested payload, the version the
  oracle picks for that payload at the requested level, the pinned mask and the symbol's own
  matrix.  The real decoder pair is stubbed because a negative case needs to *put* a specific
  defect in front of a check, and a real decoder cannot be asked for one; ``patch_decode``
  itself states the same purpose.  Nothing here decodes, encodes over the network, spawns a
  process or touches a host outside ``tmp_path``.

Three names differ from AAP §0.4.4's prose, and the owning module wins
---------------------------------------------------------------------
The plan's suggested table left three reasons to be read off the implementation; each was,
and each is asserted as the implementation documents it:

* ``SY-MASK`` -- an illegal mask is a ``value_mismatch``, not a separate code
  (``qrcheck/checks/symbol.py``: "``value_mismatch`` when the mask is outside ``0..7`` or
  contradicts a pin").
* ``IV-BAD-OPTION`` -- a target that accepts an out-of-contract option is a
  ``value_mismatch``; ``contract_error`` belongs to ``IV-EMPTY``'s unusable artefact
  (``qrcheck/checks/invalid.py``).
* ``CL-ARTIFACTS`` -- a leak is a ``value_mismatch``; its skip is ``unsupported_platform``
  (``qrcheck/checks/cleanup.py``).

Two further refinements the implementation forces, recorded here so a reader does not read
them as weakened assertions:

* ``RN-SVG`` reports ``sampling_failed`` when no module count can be implied from the raster
  and ``value_mismatch`` when the module count is known and the payload is not there.  The
  negative round asserts the latter, so the round carries a version and no payload rather
  than neither.
* ``DM-RESILIENCE`` declares eight adapter invocations in one full run -- one encode per
  sampled record, shared across the seven damage kinds -- so this module expresses its
  expectation as "one invocation per seven cases" rather than as the constant eight.  At full
  scale the expression evaluates to the declared eight, which
  :func:`test_declared_invocations_match_the_family_tables` proves against each family
  module's own ``EXPECTED_ENCODES`` so the two cannot drift.

Declared adapter-invocation counts (AAP §0.4.3, §0.5.6)
-----------------------------------------------------
:data:`EXPECTED_INVOCATIONS` maps every check to the number of ``generate`` calls it makes
for ``n`` executed cases: ``0`` for the three target-independent checks (``DM-SEEDED``,
``PF-DECODE``, ``CL-ARTIFACTS``), ``2 * n`` for ``DT-DETERMINISTIC`` (its defect is only
observable across two calls of the identical request), ``n // 7`` for ``DM-RESILIENCE``,
``warmup + encode_samples`` for ``PF-ENCODE`` (read from the resolved configuration, never a
literal), and ``n`` for the remaining fifteen.  The observed count is asserted **equal** to
the declared one, so a retry -- the one thing the recorded artifact cannot show -- shows up
as the declared count plus one.

Bounded case selection
----------------------
A full run is 1,283 cases (§0.8.1's matrix) and this module is a unit test, so each check
executes a documented prefix of its own cases (:attr:`NegativeSpec.case_limit`): one to four
cases for the synthetic checks, two for the corpus-driven ones, all seven for
``DM-RESILIENCE`` (one record's seven kinds, which is the unit its memo makes cheap) and one
for ``DM-SEEDED``.  The prefix is always taken from the check's own ascending case order, so
it is reproducible from the check's source alone.  The counts are asserted as an exact
equality against :data:`EXPECTED_INVOCATIONS` evaluated at the number of cases that actually
executed, which is why a bounded prefix cannot hide a retry.

Isolation
---------
``DM-SEEDED`` reads the symbol, the pristine reading and the damaged variants that
``DM-RESILIENCE`` memoises on the damage module, so each damage run clears that memo first
and re-derives it through its own prerequisite.  Every artefact is written under the test's
own report directory, every image the harness opens is closed before ``CL-ARTIFACTS``
executes, and the one temporary entry the ``CL-ARTIFACTS`` round injects is removed in a
``finally``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy
import pytest
import segno
from PIL import Image

from qrcheck.checks import FAMILY_MODULE_OBJECTS, load_families
from qrcheck.checks import cleanup as cleanup_family
from qrcheck.checks import damage as damage_family
from qrcheck.config import models
from qrcheck.decode.base import DecodeResult
from qrcheck.registry.check import (
    CHECK_IDS,
    Case,
    RunContext,
    get_check,
    registered_ids,
)
from qrcheck.report.model import CaseResult, Counts, Reason, Status
from qrcheck.targets.base import (
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
)

# Registering the twenty-one checks is this module's first act: every table below is keyed by
# a check identifier and asserted against the registry, so the registry has to be populated
# before the module is collected rather than when a test happens to need it.
REGISTERED: Final[tuple[str, ...]] = load_families()

# ---------------------------------------------------------------------------
# Harness constants: stub geometry and the one recorded count that is not per case
# ---------------------------------------------------------------------------

# The module edge the stub renderer draws when a request pins none. Four pixels is enough for
# a decoder to read the symbol and small enough that a 500-pixel probe image is unnecessary
# here; it is a property of the stub, not a threshold any check reads.
_DEFAULT_SCALE_PX: Final[int] = 4

# The quiet zone the stub renderer draws when a request pins none, in modules: the four
# ISO/IEC 18004 §7.3.6 requires, so a check that measures the margin of an unpinned request
# measures a conforming one.
_DEFAULT_BORDER_MODULES: Final[int] = 4

# How many damage kinds one ``DM-RESILIENCE`` record contributes: ``occluders``, ``blur``,
# ``jpeg``, ``noise``, ``rotation``, ``perspective`` and ``downscale`` (AAP §0.3.4
# ``[checks.damage]``).  The check encodes a record once and re-uses that symbol for all
# seven, so the invocation count is the record count and the case count is seven times it.
_DAMAGE_KINDS_PER_RECORD: Final[int] = 7

# The record count of the corpus the fixed samples and this module's own corpus are drawn
# from, used only to translate "the first N records in id order" into a slice.  It is a case
# parameter, not a threshold: no check reads it.
_RECORDS_PER_CHECK: Final[int] = 2

# The name of the temporary entry the ``CL-ARTIFACTS`` negative round leaves in the system
# temporary directory.  It carries the product-wide ``qrcheck-`` prefix the check scans for,
# which is what makes it a leak rather than stray test debris.
_LEAK_NAME: Final[str] = f"{cleanup_family.TEMP_PREFIX}negative-case-leak"

# ---------------------------------------------------------------------------
# The two tables under test: the negative rounds and the controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NegativeRound:
    """One driven variant of a check: what is injected, and what must happen.

    A round names its defects rather than carrying them, so the table below is data a reader
    can audit without following a callable: each name is resolved per run through
    :data:`DECODE_DEFECTS`, :data:`SYMBOL_DEFECTS` or :data:`PREPARATIONS`, which is also what
    lets a stateful defect (a mask that changes between two calls) keep its state inside the
    run instead of inside the module.

    Args:
        label: What the round is about, quoted in every assertion message.
        status: The status the check must report.  ``failed`` for every negative round;
            ``skipped`` for the one documented skip; ``passed`` for a control.
        reason: The exact :class:`~qrcheck.report.model.Reason` the check must carry, or
            ``None`` for a passing round.
        case_limit: How many of the check's cases to execute, counted from the first in the
            check's own ascending case order.
        decode_defect: A key of :data:`DECODE_DEFECTS`, applied to the conforming echo result.
        symbol_defect: A key of :data:`SYMBOL_DEFECTS`, applied to the conforming artefact.
        adapter_error: A key of :data:`ADAPTER_ERRORS`: the error the stub raises on every
            invocation, in place of returning a symbol.
        adapter_symbol: A key of :data:`ADAPTER_SYMBOLS`: fixed bytes the stub returns instead
            of rendering, for a probe whose request cannot be rendered at all.
        config: ``Config`` field overrides the round needs, splatted into ``config_factory``.
        prepare: A key of :data:`PREPARATIONS`, run after the cases are enumerated and before
            they execute -- the only window in which a leaked temporary entry can be created.
        prerequisite: A check identifier whose control round runs first, for a check that
            reads what another check memoised.
    """

    label: str
    status: Status = Status.FAILED
    reason: Reason | None = None
    case_limit: int = 1
    decode_defect: str | None = None
    symbol_defect: str | None = None
    adapter_error: str | None = None
    adapter_symbol: str | None = None
    config: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    prepare: str | None = None
    prerequisite: str | None = None


@dataclass(frozen=True, slots=True)
class NegativeSpec:
    """Every round one check is driven through: its control first, then its defects."""

    control: NegativeRound
    rounds: tuple[NegativeRound, ...]

    @property
    def all_rounds(self) -> tuple[NegativeRound, ...]:
        """Return the control and the negative rounds in the order they are driven."""
        return (self.control, *self.rounds)


def _control(
    label: str,
    *,
    case_limit: int,
    prerequisite: str | None = None,
    adapter_error: str | None = None,
) -> NegativeRound:
    """Return the passing round every check is driven through before its defects.

    Args:
        label: What the control covers, quoted in the assertion message.
        case_limit: How many of the check's cases the control executes.
        prerequisite: A check identifier whose control must run first.
        adapter_error: The adapter behaviour the control needs to pass, for the three
            invalid-input checks: a declared rejection is the *expected* outcome of those
            probes (AAP §0.3.4), so their control is a target that declares one.

    Returns:
        The control round: no artefact or decode defect, ``passed``, no reason.
    """
    return NegativeRound(
        label=label,
        status=Status.PASSED,
        reason=None,
        case_limit=case_limit,
        adapter_error=adapter_error,
        prerequisite=prerequisite,
    )


# ---------------------------------------------------------------------------
# The declared adapter-invocation contract, one entry per check (AAP §0.4.3, §0.5.6)
# ---------------------------------------------------------------------------

# The performance defaults, read from the configuration model rather than typed: PF-ENCODE
# asks the target for one symbol ``warmup + encode_samples`` times, and that number must be the
# declaration ``qrcheck/config/schema.py`` and the family module both read, not a copy of it.
_PERFORMANCE_DEFAULTS: Final[models.PerformanceLimits] = models.PerformanceLimits()


def _per_case(cases: int) -> int:
    """Return the invocation count of a check that encodes once per case."""
    return cases


def _twice_per_case(cases: int) -> int:
    """Return the count of a check that requests each case's symbol twice and compares them."""
    return 2 * cases


def _per_damage_record(cases: int) -> int:
    """Return the count of ``DM-RESILIENCE``: one encode per record, shared across its kinds."""
    return cases // _DAMAGE_KINDS_PER_RECORD


def _never(cases: int) -> int:
    """Return the count of a check that never asks the target for anything."""
    del cases
    return 0


def _encode_probe(cases: int) -> int:
    """Return the count of ``PF-ENCODE``: its warm-up plus its measured samples, once."""
    del cases
    return _PERFORMANCE_DEFAULTS.warmup + _PERFORMANCE_DEFAULTS.encode_samples


EXPECTED_INVOCATIONS: Final[dict[str, Callable[[int], int]]] = {
    "CL-ARTIFACTS": _never,
    "DM-RESILIENCE": _per_damage_record,
    "DM-SEEDED": _never,
    "DT-DETERMINISTIC": _twice_per_case,
    "IV-BAD-OPTION": _per_case,
    "IV-EMPTY": _per_case,
    "IV-OVERCAPACITY": _per_case,
    "MD-CHARSET": _per_case,
    "MD-MODE-OPTIMAL": _per_case,
    "PF-DECODE": _never,
    "PF-ENCODE": _encode_probe,
    "RN-CONTRAST": _per_case,
    "RN-LOGO": _per_case,
    "RN-MODULE-SIZE": _per_case,
    "RN-QUIET-ZONE": _per_case,
    "RN-SVG": _per_case,
    "RT-ROUNDTRIP": _per_case,
    "SY-EC-MATCH": _per_case,
    "SY-MASK": _per_case,
    "SY-MATRIX-EQUIV": _per_case,
    "SY-VERSION-MIN": _per_case,
}

# ---------------------------------------------------------------------------
# The negative rounds: one entry per check identifier, control first
#
# Each round injects the minimum change the check's own assertion detects, so a failing round
# cannot be failing for a second reason: one field of the decoded evidence, one property of
# the rendered artefact, one adapter behaviour, or one configuration threshold.  The reasons
# are the ones the family modules document; where AAP §0.4.4's prose named a different one,
# the module docstring records the delta.
# ---------------------------------------------------------------------------

NEGATIVE_CASES: Final[dict[str, NegativeSpec]] = {
    "RT-ROUNDTRIP": NegativeSpec(
        control=_control("decoded payload equals the requested payload", case_limit=2),
        rounds=(
            NegativeRound(
                label="decoded payload differs from the request in its last byte",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="payload-last-byte-changed",
            ),
        ),
    ),
    "SY-VERSION-MIN": NegativeSpec(
        control=_control("the symbol is the smallest version that fits", case_limit=2),
        rounds=(
            NegativeRound(
                label="sampled version is one larger than the oracle minimum",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="version-plus-one",
            ),
            NegativeRound(
                label="no module grid was recovered, so there is no measured version",
                reason=Reason.SAMPLING_FAILED,
                case_limit=2,
                decode_defect="grid-not-sampled",
            ),
        ),
    ),
    "SY-EC-MATCH": NegativeSpec(
        control=_control("both readings are the requested level", case_limit=2),
        rounds=(
            NegativeRound(
                label="the format information disagrees with the decoder's level",
                reason=Reason.FORMAT_INFO_DISAGREEMENT,
                case_limit=2,
                decode_defect="ec-format-disagrees",
            ),
            NegativeRound(
                label="both readings agree on a level other than the requested one",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="ec-both-wrong",
            ),
        ),
    ),
    "SY-MASK": NegativeSpec(
        control=_control("the mask from the format information is legal", case_limit=2),
        rounds=(
            NegativeRound(
                label="the reported mask is 9, outside the eight legal patterns",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="mask-out-of-range",
            ),
            NegativeRound(
                label="the format information carried no mask at all",
                reason=Reason.SAMPLING_FAILED,
                case_limit=2,
                decode_defect="mask-not-read",
            ),
        ),
    ),
    "SY-MATRIX-EQUIV": NegativeSpec(
        control=_control("the grid equals the oracle at the pinned parameters", case_limit=1),
        rounds=(
            NegativeRound(
                label="the grid has the pinned dimension and is not the oracle's matrix",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                decode_defect="matrix-differs",
            ),
            NegativeRound(
                label="no module grid was recovered, so there is no matrix to compare",
                reason=Reason.SAMPLING_FAILED,
                case_limit=1,
                decode_defect="grid-not-sampled",
            ),
        ),
    ),
    "MD-MODE-OPTIMAL": NegativeSpec(
        control=_control("the symbol is no larger than the own-charset minimum", case_limit=2),
        rounds=(
            NegativeRound(
                label="the symbol is one version larger than the own-charset minimum",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="version-plus-one",
            ),
        ),
    ),
    "MD-CHARSET": NegativeSpec(
        control=_control("the decoded payload is byte-identical to the request", case_limit=2),
        rounds=(
            NegativeRound(
                label="the decoded payload is missing its last byte",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="payload-truncated-by-one",
            ),
        ),
    ),
    "RN-QUIET-ZONE": NegativeSpec(
        control=_control("the rendered margin is at least the requested modules", case_limit=2),
        rounds=(
            NegativeRound(
                label="one module of margin was drawn where four or more were asked for",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                symbol_defect="short-quiet-zone",
            ),
        ),
    ),
    "RN-MODULE-SIZE": NegativeSpec(
        control=_control("the rendered module edge is the requested pixel size", case_limit=2),
        rounds=(
            NegativeRound(
                label="the rendered module edge is two pixels wider than the request",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                symbol_defect="wrong-module-size",
            ),
        ),
    ),
    "RN-CONTRAST": NegativeSpec(
        control=_control("the symbol carries the requested payload and palette", case_limit=2),
        rounds=(
            NegativeRound(
                label="the payload the symbol carries is not the one requested",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="payload-last-byte-changed",
            ),
        ),
    ),
    "RN-SVG": NegativeSpec(
        control=_control("the canvas carries the module count and rasterizes", case_limit=2),
        rounds=(
            NegativeRound(
                label="the rasterized canvas carries no decodable payload",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                decode_defect="undecodable-but-measured",
            ),
            NegativeRound(
                label="no SVG rasterizer was discovered, so nothing could be rasterized",
                status=Status.SKIPPED,
                reason=Reason.MISSING_DEPENDENCY,
                case_limit=2,
                config={"render": models.RenderConfig(svg_rasterizer="none")},
            ),
        ),
    ),
    "RN-LOGO": NegativeSpec(
        control=_control("the overlay is applied and the symbol survives its budget", case_limit=1),
        rounds=(
            NegativeRound(
                label="the covered fraction is within budget but the symbol does not decode",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                decode_defect="payload-last-byte-changed",
            ),
        ),
    ),
    "DM-RESILIENCE": NegativeSpec(
        control=_control(
            "all seven damages leave the payload readable",
            case_limit=_DAMAGE_KINDS_PER_RECORD,
        ),
        rounds=(
            NegativeRound(
                label="the pristine symbol reads but every damaged variant decodes to nothing",
                reason=Reason.VALUE_MISMATCH,
                case_limit=_DAMAGE_KINDS_PER_RECORD,
                decode_defect="undetected-after-first",
            ),
        ),
    ),
    "DM-SEEDED": NegativeSpec(
        control=_control(
            "the variant regenerated from the recorded seed is byte-identical",
            case_limit=1,
            prerequisite="DM-RESILIENCE",
        ),
        rounds=(
            NegativeRound(
                label="the recorded variant is not the one the recorded seed re-derives",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                prerequisite="DM-RESILIENCE",
                prepare="tamper-recorded-variant",
            ),
        ),
    ),
    "IV-EMPTY": NegativeSpec(
        control=_control(
            "the target declares the empty payload rejected",
            case_limit=1,
            adapter_error="rejected",
        ),
        rounds=(
            NegativeRound(
                label="the target crashes instead of declaring the empty payload rejected",
                reason=Reason.CONTRACT_ERROR,
                case_limit=1,
                adapter_error="contract-error",
            ),
        ),
    ),
    "IV-OVERCAPACITY": NegativeSpec(
        control=_control(
            "the target declares the over-capacity payload rejected",
            case_limit=1,
            adapter_error="rejected",
        ),
        rounds=(
            NegativeRound(
                label="the target accepts an over-capacity payload and returns a symbol",
                reason=Reason.CAPACITY_VIOLATION,
                case_limit=1,
                adapter_symbol="canned",
            ),
        ),
    ),
    "IV-BAD-OPTION": NegativeSpec(
        control=_control(
            "the target declares every bad option rejected",
            case_limit=4,
            adapter_error="rejected",
        ),
        rounds=(
            NegativeRound(
                label="the target accepts every out-of-contract option it must refuse",
                reason=Reason.VALUE_MISMATCH,
                case_limit=4,
                adapter_symbol="canned",
            ),
        ),
    ),
    "DT-DETERMINISTIC": NegativeSpec(
        control=_control("two identical requests produce identical symbols", case_limit=2),
        rounds=(
            NegativeRound(
                label="the mask changes between two calls of the identical request",
                reason=Reason.VALUE_MISMATCH,
                case_limit=2,
                symbol_defect="changing-mask",
            ),
        ),
    ),
    "PF-ENCODE": NegativeSpec(
        control=_control("the encode floors hold for the configured sample counts", case_limit=1),
        rounds=(
            NegativeRound(
                label="the configured throughput floor is unreachable, so the assertion must fail",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                config={
                    "performance": models.PerformanceLimits(
                        min_symbols_per_minute=sys.float_info.max
                    )
                },
            ),
        ),
    ),
    "PF-DECODE": NegativeSpec(
        control=_control("the decode floor holds for the configured sample count", case_limit=1),
        rounds=(
            NegativeRound(
                label="the configured decode floor is the smallest positive double, so it fails",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                config={"performance": models.PerformanceLimits(decode_p95_ms=sys.float_info.min)},
            ),
        ),
    ),
    "CL-ARTIFACTS": NegativeSpec(
        control=_control("the run leaves no temporary entry and no open image", case_limit=1),
        rounds=(
            NegativeRound(
                label="an entry with the framework's prefix appears in the temporary directory",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                prepare="leak-temp-entry",
            ),
        ),
    ),
}



# ---------------------------------------------------------------------------
# The conforming stubs: a renderer that honours a request, a decoder that echoes one
# ---------------------------------------------------------------------------


def _oracle(payload: bytes, request: GenerateRequest) -> segno.QRCode:
    """Return the golden symbol `request` asks for, as the oracle would draw it.

    The two defaults that corrupt every expectation are both off, at every call site: the
    ``boost_error`` default spends spare capacity on a higher level than was asked for, and
    ``segno.make`` returns a Micro QR symbol for a short payload -- which has no ``4V + 17``
    dimension and no Version 1-40 number (AAP §0.3.5).  ``make_qr`` is Model 2 by
    construction and the pinned release gives it no ``micro`` argument.

    Args:
        payload: The bytes to encode.
        request: The request whose level and pins the stub honours.

    Returns:
        The oracle's symbol.
    """
    return segno.make_qr(
        payload,
        error=request.ec_level,
        version=request.version,
        mask=request.mask,
        boost_error=False,
    )


def _grid(symbol: segno.QRCode) -> numpy.ndarray:
    """Return the symbol's module matrix as a boolean grid, dark = ``True``."""
    return numpy.asarray(
        [[bool(cell) for cell in row] for row in symbol.matrix], dtype=numpy.bool_
    )


def _render_with(
    request: GenerateRequest,
    *,
    border: int | None = None,
    scale: int | None = None,
    mask: int | None = None,
) -> GeneratedSymbol:
    """Render `request` as a conforming target would, with optional overrides.

    Args:
        request: The request to answer.
        border: The quiet zone to draw, overriding the request's own when given.
        scale: The module edge in pixels, overriding the request's pin when given.
        mask: The mask pattern to draw, overriding the request's pin when given.

    Returns:
        The artefact, its format matching the request's.
    """
    pinned_mask = request.mask if mask is None else mask
    symbol = segno.make_qr(
        request.payload,
        error=request.ec_level,
        version=request.version,
        mask=pinned_mask,
        boost_error=False,
    )
    module_px = request.module_size_px if scale is None else scale
    modules_across = (
        _DEFAULT_BORDER_MODULES
        if request.quiet_zone_modules is None
        else request.quiet_zone_modules
    )
    quiet_zone = modules_across if border is None else border
    buffer = io.BytesIO()
    if request.output_format == "svg":
        # A vector artefact carries no background rectangle in the pinned encoder, so only the
        # dark colour is drawn; the render layer composites the raster onto white itself.
        symbol.save(
            buffer,
            kind="svg",
            scale=_DEFAULT_SCALE_PX if module_px is None else module_px,
            border=quiet_zone,
            dark=request.foreground_rgb,
        )
    else:
        symbol.save(
            buffer,
            kind="png",
            scale=_DEFAULT_SCALE_PX if module_px is None else module_px,
            border=quiet_zone,
            dark=request.foreground_rgb,
            light=request.background_rgb,
        )
    artefact = buffer.getvalue()
    if request.logo_overlay_path is not None:
        artefact = _composite_logo(artefact, Path(request.logo_overlay_path))
    return GeneratedSymbol(image_bytes=artefact, output_format=request.output_format)


def _render(request: GenerateRequest) -> GeneratedSymbol:
    """Render `request` exactly as a conforming target would."""
    return _render_with(request)


def _composite_logo(artefact: bytes, overlay_path: Path) -> bytes:
    """Composite the request's logo bitmap onto the centre of the rendered symbol.

    A target that declares ``logo_overlay`` is expected to have behaviour behind it, and
    ``RN-LOGO`` asserts exactly that: the rendered image carries colours the two-colour symbol
    did not have.  Both handles are closed before the artefact is returned, because
    ``CL-ARTIFACTS`` fails a run that leaves an image open.

    Args:
        artefact: The rendered PNG.
        overlay_path: The bitmap the request points at.

    Returns:
        The composited PNG.
    """
    with Image.open(io.BytesIO(artefact)) as base_image:
        canvas = base_image.convert("RGB")
    with Image.open(overlay_path) as overlay_image:
        overlay = overlay_image.convert("RGB")
    left = max(0, (canvas.width - overlay.width) // 2)
    top = max(0, (canvas.height - overlay.height) // 2)
    canvas.paste(overlay, (left, top))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _request_key(request: GenerateRequest) -> tuple[Any, ...]:
    """Return the identity of a request: every field a conforming target reads.

    Used as a memo key, so the stub answers an identical request with the identical artefact and
    the identical evidence.  A conforming encoder is deterministic -- ``DT-DETERMINISTIC`` exists
    to assert exactly that -- so memoising cannot change a verdict; it only stops a probe that
    repeats one request two hundred times from paying for two hundred symbol encodes.
    """
    return (
        request.payload,
        request.ec_level,
        request.output_format,
        request.module_size_px,
        request.quiet_zone_modules,
        request.foreground_rgb,
        request.background_rgb,
        request.logo_overlay_path,
        request.version,
        request.mask,
    )


def _memoised(answer: Callable[[GenerateRequest], Any]) -> Callable[[GenerateRequest], Any]:
    """Return `answer` behind a per-run memo keyed on the request's identity.

    The memo lives in the closure, so it dies with the round: nothing carried between rounds,
    checks or tests.  A defect that depends on call order (the mask that changes between two
    calls) wraps *this* function rather than living inside it, so the re-render it forces still
    differs from the memoised answer.
    """
    cache: dict[tuple[Any, ...], Any] = {}

    def memoised(request: GenerateRequest) -> Any:
        """Return the memoised answer for `request`, computing it on first use."""
        key = _request_key(request)
        if key not in cache:
            cache[key] = answer(request)
        return cache[key]

    return memoised


def _echo_factory(
    make_decode_result: Callable[..., DecodeResult],
) -> Callable[[GenerateRequest], DecodeResult]:
    """Return the conforming decoder: the evidence a perfect decoder would report.

    The payload is the requested payload byte for byte, the version is the oracle's for that
    payload at the requested level, the mask and the grid are the artefact's own, and both
    independent readings of the error-correction level are the level that was asked for.  A
    check driven with it must pass, which is what makes the control round meaningful.

    Args:
        make_decode_result: The shared decoded-evidence factory from ``tests/conftest.py``,
            which keeps the per-decoder digests consistent with the payload and the grid.

    Returns:
        A callable producing the conforming result for one request.
    """

    def echo(request: GenerateRequest) -> DecodeResult:
        """Return the conforming decoded evidence for `request`."""
        symbol = _oracle(request.payload, request)
        try:
            text = request.payload.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        return make_decode_result(
            payload_bytes=request.payload,
            text=text,
            version=symbol.version,
            mask=symbol.mask,
            matrix=_grid(symbol),
            ec_level_decoder=request.ec_level,
            ec_level_format=request.ec_level,
        )

    return _memoised(echo)


# The defects a round may name, as tuples so a diagnostic can quote the whole vocabulary and
# the table above can be checked against it: a round naming something neither function
# handles fails with the list rather than with a bare KeyError.
_DECODE_DEFECT_NAMES: Final[tuple[str, ...]] = (
    "ec-both-wrong",
    "ec-format-disagrees",
    "grid-not-sampled",
    "mask-not-read",
    "mask-out-of-range",
    "matrix-differs",
    "payload-last-byte-changed",
    "payload-truncated-by-one",
    "undecodable-but-measured",
    "undetected-after-first",
    "version-plus-one",
)

_SYMBOL_DEFECT_NAMES: Final[tuple[str, ...]] = (
    "changing-mask",
    "short-quiet-zone",
    "wrong-module-size",
)

# A decode defect rewrites the conforming evidence for one request; an artefact defect rewrites
# the conforming artefact.  Both are pure functions of what the conforming stub produced, so
# neither can introduce a second defect by accident.
DecodeDefect = Callable[[GenerateRequest, DecodeResult], DecodeResult]
SymbolDefect = Callable[[GenerateRequest, GeneratedSymbol], GeneratedSymbol]


def _replace(result: DecodeResult, **overrides: Any) -> DecodeResult:
    """Return `result` with `overrides` applied through the record's own constructor.

    ``dataclasses.replace`` re-runs ``DecodeResult.__post_init__``, so a defect that changes the
    grid and forgets the digest is refused rather than quietly recorded beside a stale digest.

    Args:
        result: The conforming decoded evidence.
        **overrides: The fields the defect changes.

    Returns:
        The defective result.
    """
    return dataclasses.replace(result, **overrides)


def _last_byte_changed(payload: bytes) -> bytes:
    """Return `payload` with its last byte altered by one, keeping its length."""
    if not payload:
        return b"\x00"
    return payload[:-1] + bytes([(payload[-1] + 1) % 256])


def _other_level(ec_level: str) -> str:
    """Return an error-correction level other than `ec_level`, from the four legal ones."""
    for candidate in ("L", "M", "Q", "H"):
        if candidate != ec_level:
            return candidate
    return "M"


def _decode_defect(name: str) -> DecodeDefect:
    """Return the decode defect registered under `name`, built fresh for one run.

    Fresh per run because two of them are stateful -- the damaged-variant defect echoes the
    first decode and reports nothing afterwards, which is what models a symbol that reads
    before it is damaged -- and a module-level closure would carry a counter from one run into
    the next.

    Args:
        name: One of the keys this function handles, as a round names it.

    Returns:
        The defect, applied to the conforming result and the request it was built for.

    Raises:
        KeyError: `name` is not a decode defect this module declares.
    """
    if name == "payload-last-byte-changed":
        return lambda request, result: _replace(
            result, payload_bytes=_last_byte_changed(request.payload)
        )
    if name == "payload-truncated-by-one":
        return lambda request, result: _replace(result, payload_bytes=request.payload[:-1])
    if name == "version-plus-one":
        return lambda request, result: _replace(result, version=int(result.version) + 1)
    if name == "grid-not-sampled":
        return lambda request, result: _replace(
            result, matrix=None, matrix_sha256=None, version=None
        )
    if name == "ec-format-disagrees":
        return lambda request, result: _replace(
            result, ec_level_format=_other_level(result.ec_level_decoder)
        )
    if name == "ec-both-wrong":
        return lambda request, result: _replace(
            result,
            ec_level_decoder=_other_level(request.ec_level),
            ec_level_format=_other_level(request.ec_level),
        )
    if name == "mask-out-of-range":
        return lambda request, result: _replace(result, mask=9)
    if name == "mask-not-read":
        return lambda request, result: _replace(result, mask=None)
    if name == "matrix-differs":
        return lambda request, result: _replace(
            result, matrix=numpy.zeros_like(result.matrix), matrix_sha256=None
        )
    if name == "undecodable-but-measured":
        return lambda request, result: _replace(result, payload_bytes=None, text=None)
    if name == "undetected-after-first":
        state = {"calls": 0}

        def undetected_after_first(request: GenerateRequest, result: DecodeResult) -> DecodeResult:
            """Echo the first decode of a run and report nothing for every later one."""
            state["calls"] += 1
            if state["calls"] == 1:
                return result
            return _replace(
                result, payload_bytes=None, text=None, matrix=None, matrix_sha256=None, version=None
            )

        return undetected_after_first
    raise KeyError(
        f"no decode defect is declared under {name!r}; this module declares "
        f"{sorted(_DECODE_DEFECT_NAMES)}"
    )


def _symbol_defect(name: str) -> SymbolDefect:
    """Return the artefact defect registered under `name`, built fresh for one run.

    Args:
        name: One of the keys this function handles, as a round names it.

    Returns:
        The defect, applied to the conforming artefact and the request it answers.

    Raises:
        KeyError: `name` is not an artefact defect this module declares.
    """

    def short_quiet_zone(request: GenerateRequest, symbol: GeneratedSymbol) -> GeneratedSymbol:
        """Draw one module of margin, whatever zone the request asked for."""
        del symbol
        return _render_with(request, border=1)

    def wrong_module_size(request: GenerateRequest, symbol: GeneratedSymbol) -> GeneratedSymbol:
        """Draw a module edge two pixels wider than the row of choices requested."""
        del symbol
        requested = _DEFAULT_SCALE_PX if request.module_size_px is None else request.module_size_px
        return _render_with(request, scale=requested + 2)

    def changing_mask_factory() -> SymbolDefect:
        """Return a defect whose mask differs on every call of the identical request."""
        state = {"calls": 0}

        def changing_mask(request: GenerateRequest, symbol: GeneratedSymbol) -> GeneratedSymbol:
            """Answer the first call faithfully and the next one with a different mask."""
            del symbol
            state["calls"] += 1
            if state["calls"] % 2 == 1:
                return _render(request)
            chosen = _oracle(request.payload, request).mask
            return _render_with(request, mask=(int(chosen) + 1) % 8)

        return changing_mask

    if name == "short-quiet-zone":
        return short_quiet_zone
    if name == "wrong-module-size":
        return wrong_module_size
    if name == "changing-mask":
        return changing_mask_factory()
    raise KeyError(
        f"no artefact defect is declared under {name!r}; this module declares "
        f"{sorted(_SYMBOL_DEFECT_NAMES)}"
    )



# ---------------------------------------------------------------------------
# The adapter behaviours a round may name: a declared rejection, a crash, canned bytes
# ---------------------------------------------------------------------------

# The error each name raises on every invocation, as a factory so no exception instance is
# re-raised after a traceback has been attached to it.  Both are drawn from the port's closed
# five-error vocabulary; ``TargetUnavailable`` and ``TargetTimeout`` are deliberately absent,
# because the runner -- not a check -- owns the one-error-result-per-check policy for those.
ADAPTER_ERRORS: Final[dict[str, Callable[[], BaseException]]] = {
    "rejected": lambda: TargetRejected("declared rejection: qrcheck negative-case probe"),
    "contract-error": lambda: TargetContractError(
        "response body is not an image", body_prefix=b"\x00\x01\x02"
    ),
}

# The bytes a stub returns instead of rendering, for the invalid-input probes whose requests
# cannot be rendered at all -- an over-capacity payload has no symbol, which is the whole point
# of the probe.  The checks that use it read the artefact's length and format and never open it.
_CANNED_ARTEFACT: Final[bytes] = b"\x89PNG\r\n\x1a\nqrcheck-negative-canned"

ADAPTER_SYMBOLS: Final[dict[str, Callable[[], GeneratedSymbol]]] = {
    "canned": lambda: GeneratedSymbol(image_bytes=_CANNED_ARTEFACT, output_format="png"),
}


def _responder(round_spec: NegativeRound) -> Callable[[GenerateRequest], GeneratedSymbol]:
    """Return the stub target's behaviour for one round.

    Args:
        round_spec: The round being driven: it may name an error to raise, canned bytes to
            return, or an artefact defect to apply to the conforming rendering.

    Returns:
        A callable answering one request the way that round's target does.

    Raises:
        BaseException: The configured adapter error, on every invocation, when the round names
            one -- exactly as a target that refuses or breaks would.
    """
    error_factory = ADAPTER_ERRORS.get(round_spec.adapter_error or "")
    symbol_factory = ADAPTER_SYMBOLS.get(round_spec.adapter_symbol or "")
    defect = None if round_spec.symbol_defect is None else _symbol_defect(round_spec.symbol_defect)

    render = _memoised(_render)

    def responder(request: GenerateRequest) -> GeneratedSymbol:
        """Answer `request` as this round's target does."""
        if error_factory is not None:
            raise error_factory()
        symbol = symbol_factory() if symbol_factory is not None else render(request)
        return symbol if defect is None else defect(request, symbol)

    return responder


# ---------------------------------------------------------------------------
# The window between enumerating a check's cases and executing them
# ---------------------------------------------------------------------------

# What a preparation may read: the run's context, the cases about to execute, and the cases a
# prerequisite round enumerated -- which is how the seeded-damage defect finds the variants the
# resilience check recorded without either check naming a damage kind.
@dataclass(frozen=True, slots=True)
class _Preparation:
    ctx: RunContext
    cases: tuple[Case, ...]
    prerequisite_cases: tuple[Case, ...]


# The bytes the seeded-damage defect stores in place of a recorded variant: not a PNG, and
# deliberately not derived from anything, so the regeneration can never match them.
_TAMPERED_VARIANT: Final[bytes] = b"qrcheck negative tampered variant"


@contextlib.contextmanager
def _leak_temp_entry(preparation: _Preparation) -> Iterator[None]:
    """Leave one framework-prefixed entry in the temporary directory, then remove it.

    The entry is created after ``CL-ARTIFACTS`` snapshotted the directory at enumeration and
    before its case executes, which is the only window in which the check's own observation is
    the thing under test rather than a sibling process's debris.
    """
    del preparation
    leaked = Path(tempfile.gettempdir()) / _LEAK_NAME
    if leaked.exists():
        leaked.unlink()
    leaked.write_bytes(b"qrcheck negative case")
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            leaked.unlink()


@contextlib.contextmanager
def _tamper_recorded_variant(preparation: _Preparation) -> Iterator[None]:
    """Replace every variant the prerequisite round recorded with bytes no seed can derive.

    The case identifiers of the prerequisite are ``corpus:<record-id>@<kind>`` (AAP §0.3.4), so
    the record and the damage kind are read off the cases themselves: no damage-kind name is
    transcribed here, and a kind added to the check is covered without a second list.
    """
    for case in preparation.prerequisite_cases:
        record_part, _, kind = case.case_id.partition("@")
        record_id = record_part.split(":", 1)[1]
        damage_family.remember_variant(record_id, kind, _TAMPERED_VARIANT)
    yield


PREPARATIONS: Final[dict[str, Callable[[_Preparation], Iterator[None]]]] = {
    "leak-temp-entry": _leak_temp_entry,
    "tamper-recorded-variant": _tamper_recorded_variant,
}

# ---------------------------------------------------------------------------
# Driving one round: enumerate, install the decode stub, prepare, execute, count
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What one driven round produced.

    Args:
        round: The round that was driven.
        results: One result per executed case, in the check's ascending case order.
        case_objects: The cases themselves, for a preparation or a prerequisite to read.
        invocations: How many times the stub target's ``generate`` was called.
        config: The resolved configuration the round ran with, so a threshold assertion can
            compare against what was actually configured rather than against a literal.
    """

    round: NegativeRound
    results: tuple[CaseResult, ...]
    case_objects: tuple[Case, ...]
    invocations: int
    config: models.Config

    @property
    def cases(self) -> int:
        """Return how many cases the round executed."""
        return len(self.results)


@dataclass(frozen=True, slots=True)
class _CheckRun:
    """Every round one check was driven through: its control, then its negative rounds."""

    check_id: str
    control: _Outcome
    rounds: tuple[_Outcome, ...]

    @property
    def all(self) -> tuple[_Outcome, ...]:
        """Return the control and the negative rounds in the order they were driven."""
        return (self.control, *self.rounds)


@dataclass(frozen=True, slots=True)
class _Harness:
    """The fixtures one driven round needs, bundled so no helper takes six arguments.

    Args:
        config_factory: Builds a fully populated configuration in memory.
        make_run_context: Builds the :class:`~qrcheck.registry.check.RunContext` a check sees.
        patcher: The installed ``patch_decode`` patcher, which holds the constant result the
            invalid-input probes are given and which has already replaced the decode entry in
            every module that binds it.
        monkeypatch: The fixture that owns this module's own decode stub, installed on top of
            the patcher's so the result can depend on the request.
        make_decode_result: The shared decoded-evidence factory.
        counting_adapter: The counting stub's own class, taken from the fixture so this module
            keeps no second copy of it.
        corpus: The records every corpus-driven check is driven over.
    """

    config_factory: Callable[..., models.Config]
    make_run_context: Callable[..., RunContext]
    patcher: Any
    monkeypatch: pytest.MonkeyPatch
    make_decode_result: Callable[..., DecodeResult]
    counting_adapter: Callable[..., Any]
    corpus: tuple[Any, ...]


def _decode_stub(
    harness: _Harness, round_spec: NegativeRound
) -> Callable[..., DecodeResult]:
    """Return the decode entry one round runs against.

    The echo is the conforming answer -- the requested payload, the oracle's version, the
    artefact's mask and grid, the requested level twice -- and a round's decode defect rewrites
    exactly one of those facts.  The two invalid-input probes that return a symbol instead of a
    rejection are given the constant *undetected* result the ``patch_decode`` patcher holds,
    because an over-capacity payload has no symbol to describe: the probe's evidence records
    that the primary decoder read nothing, which is the truth about that artefact.

    Args:
        harness: The fixtures the stub needs.
        round_spec: The round being driven.

    Returns:
        A callable with the real entry's signature.
    """
    echo = _echo_factory(harness.make_decode_result)
    defect = None if round_spec.decode_defect is None else _decode_defect(round_spec.decode_defect)
    if round_spec.adapter_symbol is None:
        base: Callable[[GenerateRequest], DecodeResult] = echo
    else:
        harness.patcher.result = harness.make_decode_result(payload_bytes=None)

        def constant(request: GenerateRequest) -> DecodeResult:
            """Return the patcher's constant undetected result for every artefact."""
            del request
            return harness.patcher.result

        base = constant

    def stub(
        image_bytes: bytes,
        request: GenerateRequest,
        *,
        secondary: bool | None = None,
    ) -> DecodeResult:
        """Return this round's decoded evidence for `request`."""
        del image_bytes, secondary
        result = base(request)
        return result if defect is None else defect(request, result)

    return stub


def _drive_round(
    check_id: str,
    round_spec: NegativeRound,
    harness: _Harness,
    *,
    prerequisite_cases: tuple[Case, ...] = (),
) -> _Outcome:
    """Enumerate, prepare and execute one round of one check, then report what happened.

    Args:
        check_id: The check whose cases the round drives.
        round_spec: What to inject and how many cases to execute.
        harness: The fixtures the round needs.
        prerequisite_cases: The cases a prerequisite round enumerated, empty when there is
            none.  A non-empty value also means the damage memo must be left standing, which
            is how ``DM-SEEDED`` sees the symbols ``DM-RESILIENCE`` encoded.

    Returns:
        The round's results, its case objects, the observed invocation count and the
        configuration it ran with.
    """
    config = harness.config_factory(**dict(round_spec.config))
    adapter = harness.counting_adapter(responder=_responder(round_spec))
    ctx = harness.make_run_context(adapter=adapter, config=config, corpus=harness.corpus)
    if not prerequisite_cases:
        damage_family.clear_symbol_memo()
    # The dotted form, because ``monkeypatch.setattr`` reads a bare string as a whole import
    # path: it resolves the module itself and remembers the previous value, so the patcher's
    # own installation is restored when this test ends.
    harness.monkeypatch.setattr(
        "qrcheck.decode.decode_symbol", _decode_stub(harness, round_spec)
    )
    contract = get_check(check_id)
    enumerated = tuple(contract.cases(ctx))
    selected = enumerated[: round_spec.case_limit]
    preparation = None if round_spec.prepare is None else PREPARATIONS[round_spec.prepare]
    with contextlib.ExitStack() as stack:
        if preparation is not None:
            stack.enter_context(
                preparation(
                    _Preparation(
                        ctx=ctx, cases=selected, prerequisite_cases=prerequisite_cases
                    )
                )
            )
        results = tuple(case.execute(ctx) for case in selected)
    return _Outcome(
        round=round_spec,
        results=results,
        case_objects=selected,
        invocations=adapter.count,
        config=config,
    )


def _execute_round(check_id: str, round_spec: NegativeRound, harness: _Harness) -> _Outcome:
    """Drive one round, running the prerequisite round it names before it.

    A prerequisite is another check's control, driven through the same harness so both rounds
    see the same target and the same configuration.  No prerequisite names a prerequisite, so
    the recursion is one level deep by construction rather than by a depth limit.

    Args:
        check_id: The check whose cases the round drives.
        round_spec: The round to drive.
        harness: The fixtures the round needs.

    Returns:
        The round's outcome.
    """
    prerequisite_cases: tuple[Case, ...] = ()
    if round_spec.prerequisite is not None:
        prerequisite = _execute_round(
            round_spec.prerequisite,
            NEGATIVE_CASES[round_spec.prerequisite].control,
            harness,
        )
        prerequisite_cases = prerequisite.case_objects
    return _drive_round(check_id, round_spec, harness, prerequisite_cases=prerequisite_cases)


def _drive(check_id: str, harness: _Harness) -> _CheckRun:
    """Drive one check through its control and then through every one of its rounds.

    Args:
        check_id: The check to drive.
        harness: The fixtures the rounds need.

    Returns:
        The control's outcome and every negative round's, in table order.
    """
    spec = NEGATIVE_CASES[check_id]
    return _CheckRun(
        check_id=check_id,
        control=_execute_round(check_id, spec.control, harness),
        rounds=tuple(_execute_round(check_id, round_spec, harness) for round_spec in spec.rounds),
    )



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(
    config_factory: Callable[..., models.Config],
    make_run_context: Callable[..., RunContext],
    patch_decode: Any,
    monkeypatch: pytest.MonkeyPatch,
    make_decode_result: Callable[..., DecodeResult],
    counting_adapter: Any,
    committed_corpus: Sequence[Any],
) -> _Harness:
    """Return the fixtures one driven round needs, over a two-record corpus.

    The corpus is the first two committed records in ``id`` order, which one rule satisfies for
    every corpus-driven check: both are ``boundary-alphanumeric`` records, so they are inside
    ``MD-MODE-OPTIMAL``'s discriminating classes and inside ``MD-CHARSET``'s ``ascii``
    coverage, and their 25- and 47-byte payloads are inside the 8-64-byte window
    ``DM-RESILIENCE`` samples from.  Two records rather than one so a check that cycles a
    request parameter over its sample is driven through more than the first choice.

    Args:
        config_factory: The in-memory configuration builder.
        make_run_context: The run-context builder.
        patch_decode: The installed decode patcher, whose constant result the invalid-input
            probes are given and whose installation this module's own stub replaces.
        monkeypatch: The patch owner for this module's request-aware decode stub.
        make_decode_result: The shared decoded-evidence factory.
        counting_adapter: The counting stub, from which this module takes the class it builds
            one adapter per round with.
        committed_corpus: The 240 committed records.

    Returns:
        The bundle every driver below takes.
    """
    return _Harness(
        config_factory=config_factory,
        make_run_context=make_run_context,
        patcher=patch_decode,
        monkeypatch=monkeypatch,
        make_decode_result=make_decode_result,
        counting_adapter=type(counting_adapter),
        corpus=tuple(sorted(committed_corpus, key=lambda record: record.id))[:_RECORDS_PER_CHECK],
    )


# ---------------------------------------------------------------------------
# The guard: no check may exist without a negative round, and no round may name a defect
# that does not exist
# ---------------------------------------------------------------------------


def test_the_tables_cover_every_registered_check() -> None:
    """Both tables name every registered check, and every round names something real.

    This is the gate AAP §0.4.4 needs at the table level: a newly registered check cannot slip
    in without a negative round, an identifier cannot be dropped from the registry without the
    tables failing, and a round that misspells a defect is caught here rather than as a
    ``KeyError`` in the middle of a drive.
    """
    assert REGISTERED == registered_ids(), (
        "loading the families must not change the registered set: "
        f"{REGISTERED} != {registered_ids()}"
    )
    assert set(REGISTERED) == set(CHECK_IDS), (
        "the registry must hold exactly the declared identifiers; "
        f"{sorted(set(CHECK_IDS) ^ set(REGISTERED))} differ"
    )
    assert len(CHECK_IDS) == 21
    assert set(NEGATIVE_CASES) == set(CHECK_IDS), (
        "every check needs a negative round; missing: "
        f"{sorted(set(CHECK_IDS) - set(NEGATIVE_CASES))}, unexpected: "
        f"{sorted(set(NEGATIVE_CASES) - set(CHECK_IDS))}"
    )
    assert set(EXPECTED_INVOCATIONS) == set(CHECK_IDS), (
        "every check needs a declared invocation count; missing: "
        f"{sorted(set(CHECK_IDS) - set(EXPECTED_INVOCATIONS))}"
    )
    for check_id, spec in sorted(NEGATIVE_CASES.items()):
        assert spec.control.status is Status.PASSED, f"{check_id}: a control cannot fail"
        assert spec.control.reason is None, f"{check_id}: a passing control carries no reason"
        assert spec.control.case_limit >= 1, f"{check_id}: the control must execute a case"
        assert spec.rounds, f"{check_id}: a negative round is what proves the check can fail"
        for round_spec in spec.rounds:
            _assert_round_resolves(check_id, round_spec)


def _assert_round_resolves(check_id: str, round_spec: NegativeRound) -> None:
    """Assert one round is well formed and that every defect it names exists.

    Args:
        check_id: The check the round belongs to, quoted in every diagnostic.
        round_spec: The round to check.
    """
    where = f"{check_id} / {round_spec.label!r}"
    assert round_spec.label, f"{check_id}: a round needs a label"
    assert round_spec.case_limit >= 1, f"{where}: a round must execute at least one case"
    assert round_spec.status is not Status.PASSED, f"{where}: a negative round may not pass"
    if round_spec.status is Status.SKIPPED:
        # The one documented skip: a skip is a real outcome, but it is only tolerated for the
        # missing-rasterizer case, and it is asserted as a skip rather than as a failure.
        assert round_spec.reason is Reason.MISSING_DEPENDENCY, (
            f"{where}: the only tolerated skip is a missing dependency, got {round_spec.reason!r}"
        )
    else:
        assert round_spec.reason is not None, f"{where}: a failure needs an exact reason"
    injected = (
        round_spec.decode_defect,
        round_spec.symbol_defect,
        round_spec.adapter_error,
        round_spec.adapter_symbol,
        round_spec.prepare,
    )
    assert any(item is not None for item in injected) or round_spec.config, (
        f"{where}: a negative round must inject something, or it is a second control"
    )
    if round_spec.decode_defect is not None:
        assert round_spec.decode_defect in _DECODE_DEFECT_NAMES, round_spec.decode_defect
        _decode_defect(round_spec.decode_defect)
    if round_spec.symbol_defect is not None:
        assert round_spec.symbol_defect in _SYMBOL_DEFECT_NAMES, round_spec.symbol_defect
        _symbol_defect(round_spec.symbol_defect)
    if round_spec.adapter_error is not None:
        assert round_spec.adapter_error in ADAPTER_ERRORS, round_spec.adapter_error
    if round_spec.adapter_symbol is not None:
        assert round_spec.adapter_symbol in ADAPTER_SYMBOLS, round_spec.adapter_symbol
    if round_spec.prepare is not None:
        assert round_spec.prepare in PREPARATIONS, round_spec.prepare
    if round_spec.prerequisite is not None:
        assert round_spec.prerequisite in NEGATIVE_CASES, round_spec.prerequisite


# ---------------------------------------------------------------------------
# The negative case itself: the exact status and the exact reason, per check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("check_id", sorted(NEGATIVE_CASES), ids=lambda value: value)
def test_every_negative_round_fails_with_the_reason_it_documents(
    check_id: str, harness: _Harness
) -> None:
    """The check passes its control and fails each injected defect for that defect's reason.

    The control is asserted first, because a negative round that fails proves nothing unless the
    same harness, unchanged, makes the same check pass.  Then every round's every case is
    asserted to carry the round's exact status and exact :class:`~qrcheck.report.model.Reason`,
    so "it failed" is never enough and a check that fails for an unrelated reason is caught.
    """
    run = _drive(check_id, harness)
    control = run.control
    assert control.cases >= 1, f"{check_id}: the control executed no case"
    unpredicted = [
        f"{result.case_id}: {result.status.value}/{result.reason!r}"
        for result in control.results
        if result.status is not Status.PASSED
    ]
    assert not unpredicted, (
        f"{check_id}: the control round must pass, or the negative rounds below prove nothing; "
        f"failing cases: {unpredicted}"
    )
    for outcome in run.rounds:
        assert outcome.cases >= 1, f"{check_id} / {outcome.round.label!r}: no case executed"
        for result in outcome.results:
            assert result.status is outcome.round.status, (
                f"{check_id} / {outcome.round.label!r} / {result.case_id}: expected status "
                f"{outcome.round.status.value}, got {result.status.value} "
                f"(reason {result.reason!r})"
            )
            assert result.reason is outcome.round.reason, (
                f"{check_id} / {outcome.round.label!r} / {result.case_id}: expected reason "
                f"{outcome.round.reason!r}, got {result.reason!r}"
            )


# ---------------------------------------------------------------------------
# The declared invocation counts: an exact equality, so a retry is visible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("check_id", sorted(NEGATIVE_CASES), ids=lambda value: value)
def test_adapter_invocations_match_the_declared_count(check_id: str, harness: _Harness) -> None:
    """Every round invokes the target exactly as often as the contract declares.

    Asserted as an equality rather than an upper bound: a retry -- the one thing a recorded
    artefact cannot show, and the reason AAP §0.4.3 forbids one -- registers as the declared
    count plus one and fails here, and a case that quietly skipped its encode registers as the
    declared count minus one and fails here too.
    """
    run = _drive(check_id, harness)
    expected = EXPECTED_INVOCATIONS[check_id]
    for outcome in run.all:
        # A skipped round never reaches the target: the skip is decided before any I/O, which is
        # exactly what ``RN-SVG`` does when no rasterizer was discovered, so its declared count
        # for that round is zero rather than one per case.
        declared = 0 if outcome.round.status is Status.SKIPPED else expected(outcome.cases)
        assert outcome.invocations == declared, (
            f"{check_id} / {outcome.round.label!r}: {outcome.cases} case(s) invoked the target "
            f"{outcome.invocations} time(s); the contract declares {declared}"
        )


def test_declared_invocations_match_the_family_tables(
    harness: _Harness, committed_corpus: Sequence[Any]
) -> None:
    """At full scale every declared count equals the family module's own ``EXPECTED_ENCODES``.

    The unit above is driven over a bounded prefix, so this is the other end of the same
    contract: enumerate every check's cases over the whole committed corpus -- which executes
    nothing, because a case is a closure until the runner calls it -- and check that
    :data:`EXPECTED_INVOCATIONS` evaluated at that full case count reproduces the number the
    owning family module declares.  The two tables therefore cannot drift apart: a kind added to
    the damage matrix, a sample size changed in a family, or a retry baked into one would each
    break this test rather than silently rescale the counts.
    """
    family_counts: dict[str, int] = {}
    for family, module in FAMILY_MODULE_OBJECTS.items():
        declared = getattr(module, "EXPECTED_ENCODES", None)
        assert declared is not None, f"family module {family!r} must declare EXPECTED_ENCODES"
        for check_id, count in dict(declared).items():
            assert check_id not in family_counts, f"{check_id} declared by two family modules"
            family_counts[check_id] = int(count)
    assert set(family_counts) == set(CHECK_IDS), (
        "every check's family module must declare its invocation count; missing: "
        f"{sorted(set(CHECK_IDS) - set(family_counts))}"
    )
    full_ctx = harness.make_run_context(corpus=tuple(committed_corpus))
    for check_id in CHECK_IDS:
        cases = len(tuple(get_check(check_id).cases(full_ctx)))
        computed = EXPECTED_INVOCATIONS[check_id](cases)
        assert computed == family_counts[check_id], (
            f"{check_id}: {cases} case(s) over the committed corpus imply {computed} "
            f"invocation(s), but the family module declares {family_counts[check_id]}"
        )
    assert sum(family_counts.values()) == 1287, (
        "the case matrix of AAP §0.8.1 declares 1,287 encodes for one full run; the family "
        f"modules declare {sum(family_counts.values())}"
    )


# ---------------------------------------------------------------------------
# Thresholds come from the resolved configuration, never from a literal here
# ---------------------------------------------------------------------------


def test_performance_results_copy_the_configured_thresholds(harness: _Harness) -> None:
    """Both performance checks report the thresholds the configuration declares, and their samples.

    The measured durations are recorded and never asserted -- a wall-clock reading in an
    assertion would break AAP §0.5.5 -- so what is asserted is that the result carries the
    *configured* floor it was compared against, read from the configuration the round ran with
    rather than from any number written in this module, together with the sample count the
    probe actually took.  The negative rounds use an unreachable floor, and their results must
    still carry that same configured value, which is what makes the failure arithmetic rather
    than a timing.
    """
    expectations: dict[str, tuple[str, ...]] = {
        "PF-ENCODE": ("encode_p95_ms", "encode_samples", "min_symbols_per_minute", "warmup"),
        "PF-DECODE": ("decode_image_px", "decode_p95_ms", "decode_samples", "warmup"),
    }
    sample_keys: dict[str, str] = {"PF-ENCODE": "encode_samples", "PF-DECODE": "decode_samples"}
    for check_id, keys in expectations.items():
        run = _drive(check_id, harness)
        for outcome in run.all:
            limits = outcome.config.performance
            assert len(outcome.results) == 1, f"{check_id}: one probe case per run"
            result = outcome.results[0]
            assert result.expected is not None, f"{check_id}: the probe records what it asserted"
            for key in keys:
                assert result.expected[key] == getattr(limits, key), (
                    f"{check_id} / {outcome.round.label!r}: expected[{key!r}] is "
                    f"{result.expected.get(key)!r}, but the resolved configuration declares "
                    f"{getattr(limits, key)!r}"
                )
            metrics = result.decoded["metrics"]
            assert metrics["samples"] == getattr(limits, sample_keys[check_id]), (
                f"{check_id} / {outcome.round.label!r}: the probe took {metrics['samples']} "
                f"sample(s); the configuration declares {getattr(limits, sample_keys[check_id])}"
            )


def test_a_skipped_round_is_never_counted_as_a_pass(harness: _Harness) -> None:
    """The one documented skip is tallied in ``skipped`` alone, leaving ``passed`` at zero.

    AAP §0.3.4 makes a skipped result a real outcome rather than a flavoured pass, and the
    four-integer :class:`~qrcheck.report.model.Counts` is what holds that: the missing-rasterizer
    round contributes to ``skipped`` and to nothing else, so a run that skipped a check can
    never look green because of it.
    """
    run = _drive("RN-SVG", harness)
    skipped = [outcome for outcome in run.rounds if outcome.round.status is Status.SKIPPED]
    assert len(skipped) == 1, "RN-SVG declares exactly one skip round"
    counts = Counts.from_results(skipped[0].results)
    assert counts.passed == 0, f"a skip must never be counted as a pass: {counts!r}"
    assert counts.skipped == len(skipped[0].results) > 0, (
        f"every skipped case must be tallied in skipped: {counts!r}"
    )
    assert counts.failed == 0 and counts.error == 0, (
        f"a skip is neither a failure nor an error: {counts!r}"
    )

    control = Counts.from_results(run.control.results)
    assert control.passed == len(run.control.results) > 0, (
        f"with a rasterizer available every case passes: {control!r}"
    )
    assert control.skipped == 0, f"the control must not skip anything: {control!r}"

