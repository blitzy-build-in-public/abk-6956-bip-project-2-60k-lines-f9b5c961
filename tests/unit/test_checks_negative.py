"""Every one of the twenty-one checks must fail on the defect it asserts against.

For each registered check this module drives the check's own ``cases`` generator with a
protocol-conforming stub adapter and a stubbed decode result carrying the **one** defect it
exists to detect, and asserts the check's exact :class:`~qrcheck.report.model.Reason`, never
merely "it failed". Every check also carries a **control** round with no defect injected, which
must pass, so a negative round is attributable to the injected defect and not to the harness.

The decoders are stubs because a negative case must *put* a specific defect in front of a check,
and a real decoder cannot be asked for one -- with one exception. The round carrying the ninth
named defect, an encode that returns non-image bytes, lifts the stub
(:attr:`NegativeRound.real_decode`): the artefact *is* the defect, so ``RT-ROUNDTRIP`` is handed
``bytes`` no image library can open through the real decode entry, and the framework's own mapping
for unusable bytes is what the round asserts. ``RN-SVG`` is stood in for the same way at its own
seam: the rasterizer belongs to the CI e2e job alone (the workflow installs the tool there and in
no other job), so the render family's probe and rasterizer are stubbed here while
:func:`qrcheck.checks.render.discover_svg_rasterizer` keeps its real order, its real ``attempted``
list and its real ``none`` branch.

:data:`EXPECTED_INVOCATIONS` declares the ``generate`` calls each check makes for ``n`` cases under
the round's resolved ``[checks.performance]`` limits, and the observed count is asserted **equal**
to it, so a retry registers as the declared count plus one; each check drives a documented prefix
of its own ascending case order (:attr:`NegativeSpec.case_limit`). The decode half is asserted
the same way: the ``decode_symbol`` calls a performance case makes are held **equal** to the
count :data:`qrcheck.checks.performance.EXPECTED_DECODES` declares for it, so a retried decode is
visible too.

Three reasons are asserted as the owning module documents them: ``SY-MASK``, ``IV-BAD-OPTION``
and ``CL-ARTIFACTS`` report ``value_mismatch``, and ``RN-SVG``'s round too rather than
``sampling_failed``. ``DM-RESILIENCE``'s eight invocations are held against each family module's
own ``EXPECTED_ENCODES``, and ``DM-SEEDED`` clears the memo ``DM-RESILIENCE`` leaves first.

``RN-SVG``'s document guard is additionally driven through the product's own ``rasterize_svg``
rather than the stub: the reference allow-list the rasterizer sink enforces and the byte bound it
captures with are proved against the real function, never against a stand-in for it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import sys
import tempfile
import types
import zlib
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
from qrcheck.checks import performance as performance_family
from qrcheck.checks import render as render_family
from qrcheck.config import models
from qrcheck.decode.base import DecodeResult
from qrcheck.evidence import mux
from qrcheck.registry.check import (
    CHECK_IDS,
    Case,
    RunContext,
    get_check,
    registered_ids,
)
from qrcheck.report.model import CaseResult, Counts, Reason, Status
from qrcheck.targets import cli as cli_adapter
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

# Harness constants: stub geometry and the one recorded count that is not per case

# The module edge the stub renderer draws when a request pins none. Four pixels is enough for
# a decoder to read the symbol and small enough that a 500-pixel probe image is unnecessary
# here; it is a property of the stub, not a threshold any check reads.
_DEFAULT_SCALE_PX: Final[int] = 4

# The quiet zone the stub renderer draws when a request pins none, in modules: the four
# ISO/IEC 18004 §7.3.6 requires, so a check that measures the margin of an unpinned request
# measures a conforming one.
_DEFAULT_BORDER_MODULES: Final[int] = 4

# How many damage kinds one ``DM-RESILIENCE`` record contributes:
# :data:`~qrcheck.evidence.naming.DAMAGE_KINDS` declares seven -- ``occluders``, ``blur``,
# ``jpeg``, ``noise``, ``rotation``, ``perspective`` and ``downscale``.  The check encodes a
# record once and re-uses that symbol for all seven, so the invocation count is the record count
# and the case count is seven times it.
_DAMAGE_KINDS_PER_RECORD: Final[int] = 7

# The record count of the corpus the fixed samples and this module's own corpus are drawn
# from, used only to translate "the first N records in id order" into a slice.  It is a case
# parameter, not a threshold: no check reads it.
_RECORDS_PER_CHECK: Final[int] = 2

# The canvases the artefact-ceiling rounds declare.  Each ceiling is read from the render family's
# own constant rather than restated here, and the margin only has to be large enough that a refusal
# cannot be an off-by-one, so neither number is a threshold any check reads from this module.
_CEILING_MARGIN_PX: Final[int] = 1000
_OVERSIZED_ARTEFACT_PX: Final[int] = render_family._MAX_ARTEFACT_SIDE_PX + _CEILING_MARGIN_PX
_OVERSIZED_RASTER_PX: Final[int] = render_family._MAX_RASTER_SIDE_PX + _CEILING_MARGIN_PX
# The canvas the damage family's decode-ceiling round declares.  Its ceiling is read from that
# family module rather than restated here, and the margin only has to rule out an off-by-one, so
# neither number is a threshold any check reads from this module.
_OVERSIZED_DECODE_ARTEFACT_PX: Final[int] = damage_family._MAX_ARTEFACT_SIDE_PX + _CEILING_MARGIN_PX
# The canvas the library-refusal round declares: above twice ``PIL.Image.MAX_IMAGE_PIXELS``, which
# is where Pillow's own header probe raises instead of answering, derived from that value so the
# round cannot go stale.  It is the library's rule, not a ceiling this framework holds.
_ABOVE_LIBRARY_LIMIT_PX: Final[int] = math.isqrt(2 * int(Image.MAX_IMAGE_PIXELS)) + 1

# The name of the temporary entry the ``CL-ARTIFACTS`` negative round leaves behind.  It carries
# the product-wide ``qrcheck-`` prefix the check scans for, which is what makes it a leak rather
# than stray test debris; the harness points the temporary directory at a directory the test owns,
# so the entry is never written where the whole host can see it.
_LEAK_NAME: Final[str] = f"{cleanup_family.TEMP_PREFIX}negative-case-leak"

# The name the driven target reports as ``ctx.adapter.name``: ``tests/conftest.py``'s
# ``StubAdapter`` defaults it, and a check looks a target's ``[targets.<name>]`` table up by that
# name, so a round that overrides a target setting declares the table under this name.
_STUB_TARGET_NAME: Final[str] = "stub"

# The two tables under test: the negative rounds and the controls


@dataclass(frozen=True, slots=True)
class NegativeRound:
    """One driven variant of a check: what is injected, and what must happen. A round names its
    defects, resolved per run so a stateful defect keeps its state inside the run."""

    label: str
    status: Status = Status.FAILED
    reason: Reason | None = None
    case_limit: int = 1
    decode_defect: str | None = None
    symbol_defect: str | None = None
    adapter_error: str | None = None
    adapter_symbol: str | None = None
    # The one round that lifts this module's decode stub: it hands the check the real decode
    # entry, so the framework's own mapping for unusable bytes is the thing under test rather
    # than a defect scripted in front of the check (the ninth named defect).
    real_decode: bool = False
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
    """Return the passing round every check is driven through before its defects."""
    return NegativeRound(
        label=label,
        status=Status.PASSED,
        reason=None,
        case_limit=case_limit,
        adapter_error=adapter_error,
        prerequisite=prerequisite,
    )


# The declared adapter-invocation contract, one entry per check

# Each helper takes the round's resolved ``[checks.performance]`` limits: the same object
# ``PF-ENCODE`` copies its warm-up and sample counts into the result's ``expected`` from, so one
# configuration source feeds both the declaration here and the count the check reports. A retune
# of a sample count moves both, and neither number is a copy of the other.


def _per_case(cases: int, limits: models.PerformanceLimits) -> int:
    """Return the invocation count of a check that encodes once per case."""
    del limits
    return cases


def _twice_per_case(cases: int, limits: models.PerformanceLimits) -> int:
    """Return the count of a check that requests each case's symbol twice and compares them."""
    del limits
    return 2 * cases


def _per_damage_record(cases: int, limits: models.PerformanceLimits) -> int:
    """Return the count of ``DM-RESILIENCE``: one encode per record, shared across its kinds."""
    del limits
    return cases // _DAMAGE_KINDS_PER_RECORD


def _never(cases: int, limits: models.PerformanceLimits) -> int:
    """Return the count of a check that never asks the target for anything."""
    del cases, limits
    return 0


def _encode_probe(cases: int, limits: models.PerformanceLimits) -> int:
    """Return the count of ``PF-ENCODE``: the round's resolved warm-up plus its resolved
    measured samples, once -- the limits the check also reports in the result's ``expected``."""
    del cases
    return limits.warmup + limits.encode_samples


EXPECTED_INVOCATIONS: Final[dict[str, Callable[[int, models.PerformanceLimits], int]]] = {
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

# The negative rounds: one entry per check identifier, control first. Each round injects the
# minimum change the check's own assertion detects, so a failing round cannot be failing for a
# second reason: one field of the decoded evidence, one property of the artefact, one adapter
# behaviour or one configuration threshold.

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
            NegativeRound(
                label="the target's encode returns bytes that no image library can open",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                adapter_symbol="non-image",
                real_decode=True,
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
            NegativeRound(
                label="the PNG's own header declares a canvas above the measurement ceiling",
                reason=Reason.SAMPLING_FAILED,
                case_limit=1,
                symbol_defect="png-canvas-above-measurement-ceiling",
            ),
            NegativeRound(
                label="the image library refuses the declared canvas for its size",
                reason=Reason.SAMPLING_FAILED,
                case_limit=1,
                symbol_defect="raster-canvas-above-library-limit",
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
            NegativeRound(
                label=(
                    "the rendered palette is one step lighter than the pair requested, so the "
                    "palette assertion must fail"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="lightened-dark-palette",
            ),
            NegativeRound(
                label=(
                    "the declared contrast floor is unreachable, so the ratio assertion must fail"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                config={
                    "targets": {
                        _STUB_TARGET_NAME: models.TargetConfig(
                            name=_STUB_TARGET_NAME,
                            type="reference",
                            contrast_floor=sys.float_info.max,
                        )
                    }
                },
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
            NegativeRound(
                label="the document declares a canvas above the raster ceiling",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="svg-canvas-above-raster-ceiling",
            ),
            NegativeRound(
                label="the document declares an internal entity, which is refused unparsed",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="svg-entity-declaration",
            ),
            NegativeRound(
                label="the same declaration carried in UTF-16, which the scan also refuses",
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="svg-utf16-entity-declaration",
            ),
            NegativeRound(
                label=(
                    "the document carries a stylesheet instruction, which the parser drops and the "
                    "byte scan refuses"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="svg-stylesheet-instruction",
            ),
            NegativeRound(
                label=(
                    "a reference outside the document is refused before any rasterizer runs"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="svg-external-reference",
            ),
            NegativeRound(
                label=(
                    "the vector canvas carries a quiet zone the request did not ask for, so the "
                    "module pitch is not integral"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="svg-canvas-not-module-multiple",
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
            NegativeRound(
                label=(
                    "the declared overlay is never composited, so the overlay-applied assertion "
                    "must fail"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                symbol_defect="logo-not-composited",
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
            NegativeRound(
                label="the artefact declares a canvas above the per-side decode ceiling",
                reason=Reason.SAMPLING_FAILED,
                case_limit=_DAMAGE_KINDS_PER_RECORD,
                symbol_defect="png-canvas-above-decode-ceiling",
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
                label="the target answers an empty payload with an unusable response "
                "instead of a declared rejection",
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
            NegativeRound(
                label=(
                    "only the mask differs between two calls, so the mask comparison must fail"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                decode_defect="mask-changed-on-repeat",
            ),
            NegativeRound(
                label=(
                    "only the sampled matrix differs between two calls, so the matrix comparison "
                    "must fail"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                decode_defect="matrix-changed-on-repeat",
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
            NegativeRound(
                label=(
                    "the configured encode p95 floor is unreachable, so the p95 assertion must fail"
                ),
                reason=Reason.VALUE_MISMATCH,
                case_limit=1,
                config={"performance": models.PerformanceLimits(encode_p95_ms=sys.float_info.min)},
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


# The conforming stubs: a renderer that honours a request, a decoder that echoes one


def _oracle(payload: bytes, request: GenerateRequest) -> segno.QRCode:
    """Return the golden symbol `request` asks for, as the oracle would draw it."""
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
    """Render `request` as a conforming target would, with optional overrides."""
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
    """Composite the request's logo bitmap onto the centre of the rendered symbol: a target that
    declares ``logo_overlay`` must have behaviour behind it, which ``RN-LOGO`` asserts."""
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
    """Return the identity of a request -- every field a conforming target reads -- as a memo key,
    so an identical request gets the identical artefact and the identical evidence."""
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
    """Return `answer` behind a per-run memo keyed on the request's identity; a defect that depends
    on call order wraps *this* function, so the re-render it forces still differs from the memo."""
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
    """Return the conforming decoder: the evidence a perfect decoder would report. A check driven
    with it must pass, which is what makes the control round meaningful."""

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
    "mask-changed-on-repeat",
    "mask-not-read",
    "mask-out-of-range",
    "matrix-changed-on-repeat",
    "matrix-differs",
    "payload-last-byte-changed",
    "payload-truncated-by-one",
    "undecodable-but-measured",
    "undetected-after-first",
    "version-plus-one",
)

_SYMBOL_DEFECT_NAMES: Final[tuple[str, ...]] = (
    "changing-mask",
    "lightened-dark-palette",
    "logo-not-composited",
    "png-canvas-above-decode-ceiling",
    "png-canvas-above-measurement-ceiling",
    "raster-canvas-above-library-limit",
    "short-quiet-zone",
    "svg-canvas-above-raster-ceiling",
    "svg-canvas-not-module-multiple",
    "svg-entity-declaration",
    "svg-external-reference",
    "svg-stylesheet-instruction",
    "svg-utf16-entity-declaration",
    "wrong-module-size",
)

# A decode defect rewrites the conforming evidence for one request; an artefact defect rewrites
# the conforming artefact.  Both are pure functions of what the conforming stub produced, so
# neither can introduce a second defect by accident.
DecodeDefect = Callable[[GenerateRequest, DecodeResult], DecodeResult]
SymbolDefect = Callable[[GenerateRequest, GeneratedSymbol], GeneratedSymbol]


def _replace(result: DecodeResult, **overrides: Any) -> DecodeResult:
    """Return `result` with `overrides` applied through the record's own constructor, whose
    ``__post_init__`` refuses a changed grid beside a stale digest rather than recording it."""
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


# The fixed step, in grey levels per channel, by which the palette defect lightens the darker of a
# requested pair.  One step and no more: the measured ratio of every declared palette stays above
# the 7:1 floor at it, so the round proves the palette assertion alone rather than the ratio one as
# well.
_PALETTE_LIGHTEN_STEP: Final[int] = 24


def _lightened(colour: Sequence[int]) -> tuple[int, int, int]:
    """Return `colour` moved one fixed step lighter, each channel clamped at full intensity."""
    return (
        min(255, int(colour[0]) + _PALETTE_LIGHTEN_STEP),
        min(255, int(colour[1]) + _PALETTE_LIGHTEN_STEP),
        min(255, int(colour[2]) + _PALETTE_LIGHTEN_STEP),
    )


def _decode_defect(name: str) -> DecodeDefect:
    """Return the decode defect registered under `name`, built fresh for one run: a module-level
    closure would carry a stateful defect's counter from one run into the next."""
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
    if name == "mask-changed-on-repeat":
        mask_state = {"calls": 0}

        def mask_changed_on_repeat(request: GenerateRequest, result: DecodeResult) -> DecodeResult:
            """Echo the first decode of a run and change only the mask of every later one, leaving
            the matrix and the payload as the first call reported them."""
            mask_state["calls"] += 1
            if mask_state["calls"] == 1:
                return result
            changed = 0 if result.mask is None else (int(result.mask) + 1) % 8
            return _replace(result, mask=changed)

        return mask_changed_on_repeat
    if name == "matrix-changed-on-repeat":
        matrix_state = {"calls": 0}

        def matrix_changed_on_repeat(
            request: GenerateRequest, result: DecodeResult
        ) -> DecodeResult:
            """Echo the first decode of a run and change only the matrix of every later one,
            leaving the mask and the payload as the first call reported them."""
            matrix_state["calls"] += 1
            if matrix_state["calls"] == 1:
                return result
            return _replace(result, matrix=numpy.zeros_like(result.matrix), matrix_sha256=None)

        return matrix_changed_on_repeat
    raise KeyError(
        f"no decode defect is declared under {name!r}; this module declares "
        f"{sorted(_DECODE_DEFECT_NAMES)}"
    )


# Where a PNG states its own canvas: the IHDR chunk's type, width, height and CRC (ISO/IEC 15948
# §5.6).  A round rewrites the declaration and leaves every pixel as the conforming stub drew it.
_PNG_TYPE_AT: Final[int] = 12
_PNG_WIDTH_AT: Final[int] = 16
_PNG_HEIGHT_AT: Final[int] = 20
_PNG_CRC_AT: Final[int] = 29

# The internal entities a billion-laughs document carries: each level references the one below it
# ten times, so a few hundred bytes of source become a million characters while the parser reads
# them (CWE-776).  The framework refuses the declaration before the parse, so none is expanded here.
_ENTITY_DECLARATION: Final[bytes] = (
    b"<!DOCTYPE svg [\n"
    b'  <!ENTITY a "aaaaaaaaaa">\n'
    b'  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">\n'
    b'  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">\n'
    b'  <!ENTITY d "&c;&c;&c;&c;&c;&c;&c;&c;&c;&c;">\n'
    b"]>\n"
)

# The processing instruction a hostile target splices in: it names a stylesheet a renderer would
# fetch, and ElementTree drops a processing instruction as it parses, so the tree the reference scan
# walks never carries it and only the byte scan can refuse it.
_STYLESHEET_INSTRUCTION: Final[bytes] = (
    b'<?xml-stylesheet type="text/css" href="http://example.invalid/x.css"?>'
)


def _png_declaring_canvas(png_bytes: bytes, side: int) -> bytes:
    """Return `png_bytes` with its IHDR declaring a `side` x `side` canvas.

    The declared canvas is the artefact's own statement about itself, which is what a measurement
    ceiling is read from, so only the header changes: the pixels stay the ones the conforming stub
    drew. The IHDR chunk's CRC is recomputed, so the refusal is about the declared size and never
    about a corrupt header.
    """
    payload = bytearray(png_bytes)
    payload[_PNG_WIDTH_AT : _PNG_WIDTH_AT + 4] = side.to_bytes(4, "big")
    payload[_PNG_HEIGHT_AT : _PNG_HEIGHT_AT + 4] = side.to_bytes(4, "big")
    payload[_PNG_CRC_AT : _PNG_CRC_AT + 4] = (
        zlib.crc32(bytes(payload[_PNG_TYPE_AT : _PNG_CRC_AT])) & 0xFFFFFFFF
    ).to_bytes(4, "big")
    return bytes(payload)


def _pnm_declaring_canvas(side: int) -> bytes:
    """Return a binary Netpbm image whose header declares a `side` x `side` canvas.

    The one artefact form whose canvas this framework never reads for itself: Pillow parses a PNM's
    dimensions out of its header, so the library's own size guard is what meets the declaration, and
    one pixel of body is all the header needs beside it.
    """
    return f"P6\n{side} {side}\n255\n".encode("ascii") + b"\x00\x00\x00"


def _svg_with_bytes_after_declaration(svg_bytes: bytes, spliced: bytes) -> bytes:
    """Return `svg_bytes` carrying `spliced` after its XML declaration, or ahead of a document that
    carries none.

    Spliced exactly where a hostile target puts it -- immediately after the declaration the
    conforming encoder writes -- so the document stays well-formed XML, the spliced bytes are the
    only defect in it, and the refusal observed is the framework's own byte scan rather than a
    parser fault.
    """
    marker = b"?>"
    closing = svg_bytes.find(marker)
    if not svg_bytes.startswith(b"<?xml") or closing < 0:
        return spliced + svg_bytes
    split_at = closing + len(marker)
    return svg_bytes[:split_at] + b"\n" + spliced + svg_bytes[split_at:].lstrip(b"\r\n")


def _svg_with_entity_declaration(svg_bytes: bytes) -> bytes:
    """Return `svg_bytes` carrying :data:`_ENTITY_DECLARATION` after its XML declaration."""
    return _svg_with_bytes_after_declaration(svg_bytes, _ENTITY_DECLARATION)


def _svg_with_stylesheet_instruction(svg_bytes: bytes) -> bytes:
    """Return `svg_bytes` carrying :data:`_STYLESHEET_INSTRUCTION` after its XML declaration."""
    return _svg_with_bytes_after_declaration(svg_bytes, _STYLESHEET_INSTRUCTION)


def _svg_in_utf16_with_entity_declaration(svg_bytes: bytes) -> bytes:
    """Return `svg_bytes` with an entity declaration, encoded as UTF-16 with a byte-order mark.

    A UTF-16 document spells ``<!ENTITY`` two bytes per character, so a scan of the raw bytes that
    knew only the single-byte spelling would hand the declaration to the parser intact; the refusal
    reaches every encoding the parser decodes, which is what this artefact proves.
    """
    text = svg_bytes.decode("utf-8", "replace")
    closing = text.find("?>")
    if text.startswith("<?xml") and closing >= 0:
        text = text[closing + len("?>"):].lstrip()
    document = (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<!DOCTYPE svg [<!ENTITY e0 "AAAAAAAAAA">]>\n'
        f"{text}"
    )
    return document.encode("utf-16")


# The remote image the external-reference defect splices into the conforming document: the form a
# hostile target uses to make a rasterizer fetch a host the framework never chose. It declares the
# ``xlink`` namespace itself, so the document stays well-formed XML and the refusal under test is
# the framework's reference rule rather than a parser fault, and it carries no geometry of its own,
# so the canvas, the payload and the module grid stay the ones the conforming stub drew.
_REMOTE_IMAGE_ELEMENT: Final[bytes] = (
    b'<image xmlns:xlink="http://www.w3.org/1999/xlink"'
    b' xlink:href="http://example.invalid/x.png" width="1" height="1"/>'
)

# The URL that element names, which a refusal must quote back.
_REFUSED_REFERENCE: Final[str] = "http://example.invalid/x.png"

# The closing element a rendered SVG document ends with, which is where the defect splices in.
_SVG_CLOSING_TAG: Final[bytes] = b"</svg>"


def _svg_with_remote_image(svg_bytes: bytes) -> bytes:
    """Return `svg_bytes` carrying :data:`_REMOTE_IMAGE_ELEMENT` before its closing element.

    Inserted rather than substituted, so the payload, the geometry and the canvas are exactly the
    conforming ones and the remote reference is the only defect in the document.

    Raises:
        ValueError: The artefact carries no closing ``</svg>``, so it is not the document this
            module's conforming stub renders and there is nowhere to splice the element in.
    """
    closing = svg_bytes.rfind(_SVG_CLOSING_TAG)
    if closing < 0:
        raise ValueError(
            f"the conforming artefact carries no {_SVG_CLOSING_TAG!r} to splice a reference into"
        )
    return svg_bytes[:closing] + _REMOTE_IMAGE_ELEMENT + svg_bytes[closing:]


def _symbol_defect(name: str) -> SymbolDefect:
    """Return the artefact defect registered under `name`, built fresh for one run."""

    def short_quiet_zone(request: GenerateRequest, symbol: GeneratedSymbol) -> GeneratedSymbol:
        """Draw one module of margin, whatever zone the request asked for."""
        del symbol
        return _render_with(request, border=1)

    def wrong_module_size(request: GenerateRequest, symbol: GeneratedSymbol) -> GeneratedSymbol:
        """Draw a module edge two pixels wider than the row of choices requested."""
        del symbol
        requested = _DEFAULT_SCALE_PX if request.module_size_px is None else request.module_size_px
        return _render_with(request, scale=requested + 2)

    def png_canvas_above_decode_ceiling(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Declare a canvas above the decode ceiling in the artefact's own PNG header."""
        del request
        return GeneratedSymbol(
            image_bytes=_png_declaring_canvas(symbol.image_bytes, _OVERSIZED_DECODE_ARTEFACT_PX),
            output_format=symbol.output_format,
        )

    def png_canvas_above_measurement_ceiling(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Declare a canvas above the measurement ceiling in the artefact's own PNG header."""
        del request
        return GeneratedSymbol(
            image_bytes=_png_declaring_canvas(symbol.image_bytes, _OVERSIZED_ARTEFACT_PX),
            output_format=symbol.output_format,
        )

    def raster_canvas_above_library_limit(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Answer with bytes whose declared canvas the image library refuses to open."""
        del symbol
        return GeneratedSymbol(
            image_bytes=_pnm_declaring_canvas(_ABOVE_LIBRARY_LIMIT_PX),
            output_format=request.output_format,
        )

    def svg_canvas_above_raster_ceiling(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Draw the same document onto a canvas above the ceiling a rasterizer may allocate."""
        del symbol
        return _render_with(request, scale=_OVERSIZED_RASTER_PX)

    def svg_entity_declaration(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Carry an internal entity declaration, which this framework refuses before it parses."""
        del request
        return GeneratedSymbol(
            image_bytes=_svg_with_entity_declaration(symbol.image_bytes),
            output_format=symbol.output_format,
        )

    def svg_utf16_entity_declaration(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Carry the same declaration encoded as UTF-16, which the byte scan must also refuse."""
        del request
        return GeneratedSymbol(
            image_bytes=_svg_in_utf16_with_entity_declaration(symbol.image_bytes),
            output_format=symbol.output_format,
        )

    def svg_stylesheet_instruction(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Carry a ``<?xml-stylesheet`` instruction, which the parser drops and the byte scan
        refuses."""
        del request
        return GeneratedSymbol(
            image_bytes=_svg_with_stylesheet_instruction(symbol.image_bytes),
            output_format=symbol.output_format,
        )

    def svg_external_reference(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Carry an image sourced from a remote URL, which no rasterizer may be sent to fetch."""
        del request
        return GeneratedSymbol(
            image_bytes=_svg_with_remote_image(symbol.image_bytes),
            output_format=symbol.output_format,
        )

    def canvas_not_module_multiple(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Draw one module of margin where the request asked for none, so the canvas is
        ``(modules + 2) * scale`` while the decoded version implies ``modules`` across."""
        del symbol
        return _render_with(request, border=1)

    def lightened_dark_palette(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Draw the symbol with the darker of the two requested colours one step lighter, so the
        measured palette is not the requested one and the pair stays above the floor.

        The channel sum orders the two colours, the same key the check's colour measurement clusters
        the rendered pixels by, so "the darker one" means the same thing here and in the check. A
        swap of the pair would not do: the check compares the two measured colours as a set, which a
        swap leaves unchanged."""
        del symbol
        foreground = tuple(request.foreground_rgb)
        background = tuple(request.background_rgb)
        if sum(foreground) <= sum(background):
            return _render(dataclasses.replace(request, foreground_rgb=_lightened(foreground)))
        return _render(dataclasses.replace(request, background_rgb=_lightened(background)))

    def logo_not_composited(
        request: GenerateRequest, symbol: GeneratedSymbol
    ) -> GeneratedSymbol:
        """Draw the plain symbol the request asks for and ignore the overlay it names, so a declared
        ``logo_overlay`` capability has no behaviour behind it."""
        del symbol
        return _render(dataclasses.replace(request, logo_overlay_path=None))

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
    if name == "lightened-dark-palette":
        return lightened_dark_palette
    if name == "logo-not-composited":
        return logo_not_composited
    if name == "png-canvas-above-decode-ceiling":
        return png_canvas_above_decode_ceiling
    if name == "png-canvas-above-measurement-ceiling":
        return png_canvas_above_measurement_ceiling
    if name == "raster-canvas-above-library-limit":
        return raster_canvas_above_library_limit
    if name == "svg-canvas-above-raster-ceiling":
        return svg_canvas_above_raster_ceiling
    if name == "svg-canvas-not-module-multiple":
        return canvas_not_module_multiple
    if name == "svg-entity-declaration":
        return svg_entity_declaration
    if name == "svg-utf16-entity-declaration":
        return svg_utf16_entity_declaration
    if name == "svg-external-reference":
        return svg_external_reference
    if name == "svg-stylesheet-instruction":
        return svg_stylesheet_instruction
    if name == "changing-mask":
        return changing_mask_factory()
    raise KeyError(
        f"no artefact defect is declared under {name!r}; this module declares "
        f"{sorted(_SYMBOL_DEFECT_NAMES)}"
    )


# The adapter behaviours a round may name: a declared rejection, a crash, canned bytes

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

# The bytes a round hands a decode-backed check instead of a rendered symbol: plain text, with no
# image signature in it at all, so no decoder pipeline can open it.  Deliberately unlike
# ``_CANNED_ARTEFACT`` above, which carries a PNG signature and is only ever measured, never
# opened -- this artefact is opened, by the real decode entry, which is how the ninth named defect
# ("an encode that returns non-image bytes") is proved to reach a check that fails.
_NON_IMAGE_ARTEFACT: Final[bytes] = b"qrcheck negative case: not an image in any format\n"

ADAPTER_SYMBOLS: Final[dict[str, Callable[[], GeneratedSymbol]]] = {
    "canned": lambda: GeneratedSymbol(image_bytes=_CANNED_ARTEFACT, output_format="png"),
    "non-image": lambda: GeneratedSymbol(image_bytes=_NON_IMAGE_ARTEFACT, output_format="png"),
}


def _responder(round_spec: NegativeRound) -> Callable[[GenerateRequest], GeneratedSymbol]:
    """Return the stub target's behaviour for one round."""
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


# The window between enumerating a check's cases and executing them

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
    """Leave one framework-prefixed entry in the temporary directory, then remove it after
    ``CL-ARTIFACTS`` snapshotted it -- the only window where the check's own view is under test.
    The harness points the temporary directory at a directory this test owns, so the entry is
    never written where another process running the same suite could race on it."""
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
    """Replace every recorded variant with bytes no seed can derive: record and damage kind are
    read off the ``corpus:<record-id>@<kind>`` cases, never transcribed.

    The variant memo is module-global, so the original bytes of every key tampered with are
    snapshotted first, one entry per ``(record_id, kind)``, and put back in a ``finally``: the
    helper leaves no tampered state behind for a later round to observe. A key the prerequisite
    round recorded nothing for is not a recorded variant and is left untouched rather than
    invented, so a seeded round whose prerequisite recorded nothing fails on its own subject
    instead of passing on state made up here."""
    originals: dict[tuple[str, str], bytes] = {}
    for case in preparation.prerequisite_cases:
        record_part, _, kind = case.case_id.partition("@")
        record_id = record_part.split(":", 1)[1]
        recorded = damage_family.recall_variant(record_id, kind)
        if recorded is not None:
            originals[(record_id, kind)] = recorded
    try:
        for record_id, kind in originals:
            damage_family.remember_variant(record_id, kind, _TAMPERED_VARIANT)
        yield
    finally:
        for (record_id, kind), recorded in originals.items():
            damage_family.remember_variant(record_id, kind, recorded)


PREPARATIONS: Final[dict[str, Callable[[_Preparation], Iterator[None]]]] = {
    "leak-temp-entry": _leak_temp_entry,
    "tamper-recorded-variant": _tamper_recorded_variant,
}

# Driving one round: enumerate, install the decode stub, prepare, execute, count


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What one driven round produced."""

    round: NegativeRound
    results: tuple[CaseResult, ...]
    case_objects: tuple[Case, ...]
    invocations: int
    decodes: int
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
    """The fixtures one driven round needs, bundled so no helper takes six arguments."""

    config_factory: Callable[..., models.Config]
    make_run_context: Callable[..., RunContext]
    patcher: Any
    monkeypatch: pytest.MonkeyPatch
    make_decode_result: Callable[..., DecodeResult]
    counting_adapter: Callable[..., Any]
    corpus: tuple[Any, ...]


def _decode_stub(
    harness: _Harness,
    round_spec: NegativeRound,
    decode_calls: dict[str, int],
) -> Callable[..., DecodeResult]:
    """Return the decode entry one round runs against; the two invalid-input probes get the constant
    *undetected* result, because an over-capacity payload has no symbol to describe. Every call is
    counted into ``decode_calls``, so the round reports how many decodes its case asked the one
    decode seam for."""
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
        """Return this round's decoded evidence for `request`, counting the decode it answered."""
        del image_bytes, secondary
        decode_calls["calls"] += 1
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
    """Enumerate, prepare and execute one round of one check, then report what happened."""
    config = harness.config_factory(**dict(round_spec.config))
    adapter = harness.counting_adapter(responder=_responder(round_spec))
    ctx = harness.make_run_context(adapter=adapter, config=config, corpus=harness.corpus)
    if not prerequisite_cases:
        damage_family.clear_symbol_memo()
    # What the case asks the decode layer for, counted at the one seam every image check decodes
    # through, so a retried decode registers as one more call.
    decode_calls = {"calls": 0}
    # The dotted form, because ``monkeypatch.setattr`` reads a bare string as a whole import
    # path: it resolves the module itself and remembers the previous value, so the patcher's
    # own installation is restored when this test ends.  The stub exists because a negative case
    # must put a specific defect in front of a check and a real decoder cannot be asked for one.
    # That reason does not hold for a round whose artefact *is* the defect -- bytes no image
    # library can open -- so that round restores the real decode entry instead, and the
    # framework's own mapping for unusable bytes becomes what the check is observed to report
    # (the ninth named defect).  No constant result is installed for it either: the stub sets one
    # only because a canned artefact describes no symbol at all; every other round counts its
    # decodes at the one seam it runs through.
    decode_entry = (
        harness.patcher.real
        if round_spec.real_decode
        else _decode_stub(harness, round_spec, decode_calls)
    )
    harness.monkeypatch.setattr("qrcheck.decode.decode_symbol", decode_entry)
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
        decodes=decode_calls["calls"],
        config=config,
    )


def _execute_round(check_id: str, round_spec: NegativeRound, harness: _Harness) -> _Outcome:
    """Drive one round, running the prerequisite round it names first: no prerequisite names a
    prerequisite, so the recursion is one level deep by construction, not by a depth limit."""
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
    """Drive one check through its control and then through every one of its rounds."""
    spec = NEGATIVE_CASES[check_id]
    return _CheckRun(
        check_id=check_id,
        control=_execute_round(check_id, spec.control, harness),
        rounds=tuple(_execute_round(check_id, round_spec, harness) for round_spec in spec.rounds),
    )


# The SVG rasterizer the unit tier stands in for
# ``RN-SVG`` rasterizes through whatever ``discover_svg_rasterizer`` finds on the host, and the
# tool belongs to the CI e2e job alone (the workflow installs ``librsvg2-bin`` there and in no
# other job), so this module answers that seam with a stub: the real tool's fidelity is the e2e
# tier's proof, and the subject here is the check's own arithmetic and the discovery order it
# reports.  The name is one the product's own order lists, so discovery still walks
# ``SVG_RASTERIZER_AUTO_ORDER`` and still stops at the first candidate its probe accepts.
_STUB_RASTERIZER: Final[str] = "rsvg-convert"

# The product's own rasterization, captured at import time because the autouse fixture below
# replaces the module attribute: the tests that drive the sink's document guard and its byte bound
# must run the real function, not the stub that answers a fixed PNG.
_REAL_RASTERIZE_SVG: Final[Callable[..., bytes]] = render_family.rasterize_svg

# The rasterizer status those tests run against, built directly rather than probed, so no test in
# this module depends on which tools the host happens to provide.
_RASTERIZER_STATUS: Final[render_family.RasterizerStatus] = render_family.RasterizerStatus(
    configured="auto",
    name=_STUB_RASTERIZER,
    available=True,
    attempted=(_STUB_RASTERIZER,),
)

# The side, in pixels, of the image the stub rasterizer answers with: stub geometry, not a
# threshold any check reads.
_STUB_RASTER_SIDE_PX: Final[int] = 8


def _stub_raster_png() -> bytes:
    """Return a small, genuinely valid PNG for the stub rasterizer to answer every SVG with.

    Nothing here decodes these bytes: the decode entry in front of a case is stubbed, so what an
    image *contains* is never this module's subject.  They are a real image rather than an empty
    artefact so the stub hands a check a PNG, exactly as the rasterizer it stands in for does.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (_STUB_RASTER_SIDE_PX, _STUB_RASTER_SIDE_PX), "white").save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


_STUB_RASTER: Final[bytes] = _stub_raster_png()


@pytest.fixture(autouse=True)
def stub_svg_rasterizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the render family's rasterizer seam, so no case here needs a tool from the host.

    Only the two seams are replaced.  ``discover_svg_rasterizer`` keeps its real probe order, its
    real ``attempted`` list and its real ``"none"`` branch -- which answers before any probe runs,
    so the module's skip round still skips -- and ``rasterize_svg`` answers a PNG instead of driving
    third-party code.  The unit tier is the fast, fully-mocked layer the workflow's unit job runs,
    and the real tool remains the e2e tier's proof.
    """

    def probe(name: str) -> bool:
        """Report the one rasterizer this module's stub provides."""
        return name == _STUB_RASTERIZER

    def rasterize(svg_bytes: bytes, status: Any, canvas: Any = None) -> bytes:
        """Answer every vector artefact with the stub's valid PNG.

        `canvas` is the ``(width, height, unit)`` the check already parsed from the document and
        hands to the real `rasterize_svg` so its guard need not parse it again; the stub allocates
        nothing, so it takes and ignores it.
        """
        del svg_bytes, status, canvas
        return _STUB_RASTER

    monkeypatch.setattr(render_family, "_probe_rasterizer", probe)
    monkeypatch.setattr(render_family, "rasterize_svg", rasterize)


# Fixtures


@pytest.fixture
def harness(
    config_factory: Callable[..., models.Config],
    make_run_context: Callable[..., RunContext],
    patch_decode: Any,
    monkeypatch: pytest.MonkeyPatch,
    make_decode_result: Callable[..., DecodeResult],
    counting_adapter: Any,
    committed_corpus: Sequence[Any],
    tmp_path: Path,
) -> _Harness:
    """Return the fixtures one driven round needs, over a two-record corpus: two records so a check
    that cycles a request parameter over its sample is driven through more than the first choice."""
    # ``_leak_temp_entry`` writes the framework-prefixed entry ``CL-ARTIFACTS`` scans for, so the
    # temporary directory has to be one this test owns: the check takes its baseline at case
    # enumeration and its scan at execution, and both must observe this directory rather than the
    # one the whole host shares, where a neighbour's run could race on the same name.
    scratch = tmp_path / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return _Harness(
        config_factory=config_factory,
        make_run_context=make_run_context,
        patcher=patch_decode,
        monkeypatch=monkeypatch,
        make_decode_result=make_decode_result,
        counting_adapter=type(counting_adapter),
        corpus=tuple(sorted(committed_corpus, key=lambda record: record.id))[:_RECORDS_PER_CHECK],
    )


# The guard: no check may exist without a negative round, and no round may name a defect that
# does not exist


def test_the_tables_cover_every_registered_check() -> None:
    """Both tables name every registered check and every round names something real -- the
    table-level gate that keeps the suite non-vacuous, where a misspelt defect fails here."""
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
    """Assert one round is well formed and that every defect it names exists."""
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
    if round_spec.real_decode:
        # The declared shape of a real-decode round: the artefact is what the check is handed,
        # so the defect reaches it through an adapter symbol rather than a scripted decode
        # result, which would need the stub this round lifts.
        assert round_spec.adapter_symbol is not None, (
            f"{where}: a real-decode round must name an adapter_symbol, because the artefact is "
            "the only defect it can put in front of the check"
        )
    if round_spec.prepare is not None:
        assert round_spec.prepare in PREPARATIONS, round_spec.prepare
    if round_spec.prerequisite is not None:
        assert round_spec.prerequisite in NEGATIVE_CASES, round_spec.prerequisite


# The negative case itself: the exact status and the exact reason, per check


@pytest.mark.parametrize("check_id", sorted(NEGATIVE_CASES), ids=lambda value: value)
def test_every_negative_round_fails_with_the_reason_it_documents(
    check_id: str, harness: _Harness
) -> None:
    """The check passes its control and fails each injected defect for that defect's reason, the
    control first: a negative round proves nothing unless the same harness makes the check pass."""
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


# What an unmeasurable artefact records: the ceiling it was refused against, and no invented canvas


def test_an_unmeasurable_artefact_records_the_ceiling_and_never_an_invented_canvas(
    harness: _Harness,
) -> None:
    """A canvas this framework refuses records the dimensions the header declared; one the image
    library refuses for its size records none, because those dimensions were never read out. Neither
    is an ``error`` result, which is what the two rounds exist to prove."""
    run = _drive("RN-QUIET-ZONE", harness)
    by_defect = {
        outcome.round.symbol_defect: outcome
        for outcome in run.rounds
        if outcome.round.symbol_defect is not None
    }
    declared = by_defect["png-canvas-above-measurement-ceiling"].results[0]
    refused = by_defect["raster-canvas-above-library-limit"].results[0]
    for result in (declared, refused):
        assert result.status is Status.FAILED, (
            f"{result.case_id}: a refused canvas must never become an error result, got "
            f"{result.status.value}/{result.reason!r}"
        )
        assert result.reason is Reason.SAMPLING_FAILED, result.reason
        assert result.decoded["reason"] == "artefact_canvas_above_measurement_ceiling"
        assert result.decoded["artefact_side_ceiling_px"] == render_family._MAX_ARTEFACT_SIDE_PX
    assert declared.decoded["artefact_px"] == [_OVERSIZED_ARTEFACT_PX, _OVERSIZED_ARTEFACT_PX]
    assert declared.decoded["artefact_side_px"] == _OVERSIZED_ARTEFACT_PX
    assert refused.decoded["artefact_px"] is None, "a canvas never read out must not be invented"
    assert refused.decoded["artefact_side_px"] is None


# What a refused artefact records: the token the refusal carries, and the text the refused document
# named.  An external reference is a target-side verdict with its own token; an ``<?xml-stylesheet``
# instruction is refused by the pre-parse byte scan, because the parser drops a processing
# instruction and no scan of the parsed tree could see the stylesheet it names, so it carries the
# unreadable-canvas token the two entity rounds carry.  Both are ``value_mismatch``, which is the
# point: the target's own bytes are a verdict about the target rather than a framework error.  Case
# inputs, not thresholds.
_REFUSED_ARTEFACT_ROUNDS: Final[tuple[tuple[str, str, str], ...]] = (
    ("svg-external-reference", "svg_external_reference", _REFUSED_REFERENCE),
    ("svg-stylesheet-instruction", "svg_canvas_unreadable", "<?xml-stylesheet"),
)


@pytest.mark.parametrize(
    ("symbol_defect", "problem", "named"),
    _REFUSED_ARTEFACT_ROUNDS,
    ids=[entry[0] for entry in _REFUSED_ARTEFACT_ROUNDS],
)
def test_a_refused_artefact_is_a_verdict_about_the_target(
    symbol_defect: str, problem: str, named: str, harness: _Harness
) -> None:
    """An artefact that references something outside itself, and one carrying a declaration the
    parser would drop, are both refused before any rasterizer runs: the case is ``value_mismatch``
    carrying `problem`, the recorded error names `named` from the document, and the target's own
    bytes are a verdict about the target, never a framework error."""
    run = _drive("RN-SVG", harness)
    refused = [outcome for outcome in run.rounds if outcome.round.symbol_defect == symbol_defect]
    assert len(refused) == 1, f"RN-SVG declares exactly one {symbol_defect} round"
    result = refused[0].results[0]
    assert result.status is Status.FAILED, (
        f"{result.case_id}: a refused artefact is the target's own and never an error result, "
        f"got {result.status.value}/{result.reason!r}"
    )
    assert result.reason is Reason.VALUE_MISMATCH, result.reason
    assert result.decoded["problems"] == [problem], result.decoded["problems"]
    assert named in str(result.decoded["svg_error"]), (
        f"the recorded error must name {named!r} from the document, got "
        f"{result.decoded['svg_error']!r}"
    )


# What a refused decode canvas records, and that the refusal costs no decode at all


def test_the_decode_ceiling_refuses_a_declared_canvas_before_any_decode(harness: _Harness) -> None:
    """A canvas above the damage family's per-side decode ceiling is refused from the header alone:
    every case of the round fails with ``sampling_failed``, the declared dimensions and the ceiling
    are recorded, and the round asks the decode layer for zero decodes -- which is what proves no
    pixel of the artefact the target returned was ever allocated."""
    run = _drive("DM-RESILIENCE", harness)
    refused = [
        outcome
        for outcome in run.rounds
        if outcome.round.symbol_defect == "png-canvas-above-decode-ceiling"
    ]
    assert len(refused) == 1, "DM-RESILIENCE declares exactly one decode-ceiling round"
    outcome = refused[0]
    assert outcome.results, "the decode-ceiling round executed no case"
    assert outcome.decodes == 0, (
        f"{len(outcome.results)} refused case(s) asked the decode layer for {outcome.decodes} "
        "decode(s); the refusal must happen before any decode"
    )
    for result in outcome.results:
        assert result.status is Status.FAILED, (
            f"{result.case_id}: a refused canvas must never become an error result, got "
            f"{result.status.value}/{result.reason!r}"
        )
        assert result.reason is Reason.SAMPLING_FAILED, result.reason
        assert result.decoded["reason"] == "artefact_canvas_above_decode_ceiling"
        assert result.decoded["artefact_side_ceiling_px"] == damage_family._MAX_ARTEFACT_SIDE_PX
        assert result.decoded["artefact_side_px"] == _OVERSIZED_DECODE_ARTEFACT_PX
        assert result.decoded["artefact_px"] == [
            _OVERSIZED_DECODE_ARTEFACT_PX,
            _OVERSIZED_DECODE_ARTEFACT_PX,
        ]


# The declared invocation counts: an exact equality, so a retry is visible


@pytest.mark.parametrize("check_id", sorted(NEGATIVE_CASES), ids=lambda value: value)
def test_adapter_invocations_match_the_declared_count(check_id: str, harness: _Harness) -> None:
    """Every round invokes the target exactly as often as the contract declares, as an equality
    rather than a bound: a retry registers as one more call and a skipped encode as one fewer."""
    run = _drive(check_id, harness)
    expected = EXPECTED_INVOCATIONS[check_id]
    for outcome in run.all:
        # A skipped round never reaches the target: the skip is decided before any I/O, which is
        # exactly what ``RN-SVG`` does when no rasterizer was discovered, so its declared count
        # for that round is zero rather than one per case.
        declared = (
            0
            if outcome.round.status is Status.SKIPPED
            else expected(outcome.cases, outcome.config.performance)
        )
        assert outcome.invocations == declared, (
            f"{check_id} / {outcome.round.label!r}: {outcome.cases} case(s) invoked the target "
            f"{outcome.invocations} time(s); the contract declares {declared}"
        )


def test_declared_invocations_match_the_family_tables(
    harness: _Harness, committed_corpus: Sequence[Any]
) -> None:
    """At full scale every declared count equals the family module's own ``EXPECTED_ENCODES``: cases
    are enumerated over the whole corpus, which executes nothing, and the count must reproduce."""
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
        computed = EXPECTED_INVOCATIONS[check_id](cases, full_ctx.config.performance)
        assert computed == family_counts[check_id], (
            f"{check_id}: {cases} case(s) over the committed corpus imply {computed} "
            f"invocation(s), but the family module declares {family_counts[check_id]}"
        )
    assert sum(family_counts.values()) == 1287, (
        "the family modules' EXPECTED_ENCODES tables total 1,287 encodes for one full run; "
        f"they declare {sum(family_counts.values())}"
    )


@pytest.mark.parametrize("check_id", sorted(EXPECTED_INVOCATIONS), ids=lambda value: value)
def test_the_declared_count_follows_the_resolved_limits(check_id: str) -> None:
    """The declared count follows the round's resolved configuration, one source for both
    assertions: ``PF-ENCODE`` declares the retuned ``warmup + encode_samples`` its result's
    ``expected`` reports too, while every other check's count is the same under those limits as
    under the schema defaults."""
    retuned = models.PerformanceLimits(warmup=1, encode_samples=4)
    cases = _DAMAGE_KINDS_PER_RECORD
    declared = EXPECTED_INVOCATIONS[check_id](cases, retuned)
    if check_id == "PF-ENCODE":
        assert declared == retuned.warmup + retuned.encode_samples, (
            f"PF-ENCODE: {cases} case(s) declared {declared} invocation(s) under the retuned "
            f"limits {retuned!r}; the declaration must follow them"
        )
        assert declared == 5, f"one warm-up plus four samples is five, not {declared}"
    else:
        assert declared == EXPECTED_INVOCATIONS[check_id](cases, models.PerformanceLimits()), (
            f"{check_id}: {cases} case(s) declared {declared} invocation(s) under retuned "
            f"performance limits; this check's count must not read them"
        )


def test_decode_invocations_match_the_declared_count(harness: _Harness) -> None:
    """The decode half of the accounting, asserted as a strict equality: ``PF-ENCODE`` decodes
    nothing and ``PF-DECODE`` decodes its probe ``warmup + decode_samples`` times, as
    :data:`qrcheck.checks.performance.EXPECTED_DECODES` declares, so a retry on the decode path
    registers as one more call and fails here."""
    for check_id, declared in performance_family.EXPECTED_DECODES.items():
        run = _drive(check_id, harness)
        for outcome in run.all:
            assert outcome.cases == 1, (
                f"{check_id} / {outcome.round.label!r}: {outcome.cases} case(s) executed, but "
                "each performance check is one constructed case"
            )
            assert outcome.decodes == declared, (
                f"{check_id} / {outcome.round.label!r}: the case asked the decode layer for "
                f"{outcome.decodes} decode(s); the family module declares {declared}"
            )


# Thresholds come from the resolved configuration, never from a literal here


def test_performance_results_copy_the_configured_thresholds(harness: _Harness) -> None:
    """Both performance checks report the configured thresholds and their sample count; the measured
    durations are recorded, never asserted, so no wall-clock value reaches an assertion."""
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
    """The one documented skip is tallied in ``skipped`` alone, leaving ``passed`` at zero: a skip
    is a real outcome, and the four-integer ``Counts`` is what holds that."""
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


# The image boundary the decode probe builds its image through


def test_the_image_boundary_declares_the_pillow_module_the_probe_builds_with() -> None:
    """``PF-DECODE``'s probe builds its image through ``mux.Image``, so that module is a declared
    name of the boundary's public surface: renaming the Pillow import inside ``mux`` must fail here
    rather than at a probe's construction."""
    assert "Image" in tuple(mux.__all__), (
        "qrcheck.evidence.mux must declare the Pillow image module PF-DECODE's probe builds with; "
        f"__all__ declares {sorted(mux.__all__)}"
    )
    assert mux.Image is Image, f"mux.Image must be PIL.Image itself, got {mux.Image!r}"


# The rasterization sink itself: its document guard, driven through the real function
# The autouse stub stands in for the tool, never for the guard: the reference allow-list, the byte
# bound and the capture all live in ``rasterize_svg``, so these tests drive the real function with
# the discovered tool replaced by an argv-only command that is this interpreter.

# The square canvas, in pixels, every sink document below declares: one pixel per module of the
# smallest legal symbol, so the document's own canvas guard is satisfied and the reference rule is
# the only thing under test. Case geometry, not a threshold any check reads.
_SINK_CANVAS_PX: Final[int] = 21

# The references a sink document carries outside itself, each one quoted back by the refusal: an
# absolute local path, a remote stylesheet, and the protocol-relative form that inherits the
# fetching host's scheme.
_LOCAL_FILE_REFERENCE: Final[str] = "/etc/passwd"
_REMOTE_STYLESHEET_REFERENCE: Final[str] = "http://example.invalid/p.svg#g"
_PROTOCOL_RELATIVE_REFERENCE: Final[str] = "//example.invalid/x.svg#g"


def _sink_document(body: str) -> bytes:
    """Return an SVG document declaring :data:`_SINK_CANVAS_PX` per side and carrying `body`.

    The ``xlink`` namespace is declared on the root because ``xlink:href`` is one of the reference
    forms under test: without it the document would not be well-formed XML, and the refusal observed
    would be a parser fault rather than the framework's reference rule.
    """
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{_SINK_CANVAS_PX}" height="{_SINK_CANVAS_PX}">{body}</svg>'
    ).encode("utf-8")


# The reference forms a target's document may carry that name something outside it, each with the
# form's own identifier and the text the refusal must quote back. Case inputs, not thresholds.
_EXTERNAL_REFERENCE_DOCUMENTS: Final[tuple[tuple[str, str, bytes], ...]] = (
    (
        "remote-href",
        _REFUSED_REFERENCE,
        _sink_document('<image href="http://example.invalid/x.png" width="1" height="1"/>'),
    ),
    (
        "xlink-href-local-path",
        _LOCAL_FILE_REFERENCE,
        _sink_document('<image xlink:href="/etc/passwd" width="1" height="1"/>'),
    ),
    (
        "style-url-remote",
        _REMOTE_STYLESHEET_REFERENCE,
        _sink_document(
            '<rect width="1" height="1" style="fill:url(http://example.invalid/p.svg#g)"/>'
        ),
    ),
    (
        "style-import",
        "@import",
        _sink_document('<style>@import "x.css";</style>'),
    ),
    (
        "style-protocol-relative",
        _PROTOCOL_RELATIVE_REFERENCE,
        _sink_document("<style>rect{fill:url(//example.invalid/x.svg#g)}</style>"),
    ),
)


# The same hole spelled with a CSS escape, which no scan of the raw value can see: CSS writes any
# character as a backslash and its code point, so ``u\72 l(...)`` and ``\75 rl(...)`` are both
# ``url(...)`` to a renderer while neither reads as one here.  The entries cover the two spellings
# the verifier reproduced and the presentation-attribute form of the same hole, which carries no
# ``style`` attribute at all.  Case inputs, not thresholds.
_CSS_ESCAPE_REFERENCE_DOCUMENTS: Final[tuple[tuple[str, str, bytes], ...]] = (
    (
        "style-escape-code-point",
        "fill:u\\72 l(http://example.invalid/x)",
        _sink_document(
            '<rect width="1" height="1" style="fill:u\\72 l(http://example.invalid/x)"/>'
        ),
    ),
    (
        "style-escape-with-prefix",
        "fill:\\75 rl(http://example.invalid/x)",
        _sink_document(
            '<rect width="1" height="1" style="fill:\\75 rl(http://example.invalid/x)"/>'
        ),
    ),
    (
        "presentation-attribute-escape",
        "\\75 rl(http://example.invalid/x)",
        _sink_document('<rect width="1" height="1" fill="\\75 rl(http://example.invalid/x)"/>'),
    ),
)


# The document a hostile target sends to make a rasterizer fetch a stylesheet, with the instruction
# immediately after the XML declaration where a target puts it: ElementTree drops it, so only the
# byte scan can refuse the resource it names.
_STYLESHEET_DOCUMENT: Final[bytes] = (
    b'<?xml version="1.0"?>'
    + _STYLESHEET_INSTRUCTION
    + _sink_document('<rect width="1" height="1" fill="#000000"/>')
)


# Every document the sink must refuse, one row per form: `label` names the form, `offending` is the
# value the artefact carries from outside itself, and `document` is the artefact carrying it.  The
# forms are the external references above, the same hole spelled as a CSS escape that no scan of a
# value's text can see, and the stylesheet instruction the parser drops before any scan of the tree
# could meet it.  Case inputs, not thresholds.
_REFUSED_SINK_DOCUMENTS: Final[tuple[tuple[str, str, bytes], ...]] = (
    *_EXTERNAL_REFERENCE_DOCUMENTS,
    *_CSS_ESCAPE_REFERENCE_DOCUMENTS,
    ("stylesheet-instruction", "<?xml-stylesheet", _STYLESHEET_DOCUMENT),
)


@pytest.mark.parametrize(
    ("label", "offending", "document"),
    _REFUSED_SINK_DOCUMENTS,
    ids=[entry[0] for entry in _REFUSED_SINK_DOCUMENTS],
)
def test_the_rasterizer_sink_refuses_a_document_reaching_outside_itself(
    label: str,
    offending: str,
    document: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every form of reference outside the document, and the stylesheet instruction no scan of the
    parsed tree can meet, is refused before a command is built, so the discovered tool is never
    spawned: `label` names the form under test, the raise quotes the offending value back in the
    form the scan reports it -- an escape spelling in its escaped form -- and a refused document
    that reached the command seam would fail the case here."""
    argv_requests: list[str] = []

    def command(name: str) -> tuple[str, ...]:
        """Record an argv request: reaching here means a refused document was about to run."""
        argv_requests.append(name)
        return (name,)

    monkeypatch.setattr(render_family, "_rasterizer_command", command)
    with pytest.raises(RuntimeError) as excinfo:
        _REAL_RASTERIZE_SVG(document, _RASTERIZER_STATUS)
    # A value carrying a CSS escape is reported in the escaped form it was written in, which the
    # refusal quotes through ``repr``; every other form is quoted as it stands.
    quoted = repr(offending) if "\\" in offending else offending
    assert quoted in str(excinfo.value), (
        f"{label}: the refusal must quote {quoted}, got {str(excinfo.value)!r}"
    )
    assert not argv_requests, (
        f"{label}: a refused document must never reach a rasterizer, but {argv_requests} was built"
    )


# The document whose only references are same-document fragments: a gradient that the rectangle's
# fill and the use element both name by ``#id``, which is what a conforming encoder emits.
_FRAGMENT_ONLY_DOCUMENT: Final[bytes] = _sink_document(
    '<defs><linearGradient id="g"><stop offset="0" stop-color="#000000"/></linearGradient></defs>'
    '<rect width="1" height="1" fill="url(#g)"/>'
    '<use href="#g"/>'
)


def _answer_with(monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...]) -> None:
    """Answer the rasterizer's argv seam with `argv`, so no tool this host provides is spawned."""
    monkeypatch.setattr(render_family, "_rasterizer_command", lambda name: argv)


def _refusal(monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...]) -> str:
    """Answer the argv seam with `argv` and return the sink's refusal of the fragment-only document.

    The one shape every refusal case drives: the product's own ``rasterize_svg``, never a stand-in
    for it, over a document whose only references are its own ``#fragment`` ones.
    """
    _answer_with(monkeypatch, argv)
    with pytest.raises(RuntimeError) as raised:
        _REAL_RASTERIZE_SVG(_FRAGMENT_ONLY_DOCUMENT, _RASTERIZER_STATUS)
    return str(raised.value)


def _emitting_command(png_bytes: bytes) -> tuple[str, ...]:
    """Return an argv-only command that writes `png_bytes` to its stdout, reading its stdin first.

    The stand-in for the discovered tool: the sink is what these tests drive, so the command it runs
    may be this interpreter as long as both halves of the contract -- an SVG on stdin, a PNG on
    stdout -- are honoured.
    """
    program = (
        "import sys; sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]))"
    )
    return (sys.executable, "-c", program, png_bytes.hex())


def test_the_rasterizer_sink_still_rasterizes_a_document_of_same_document_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conforming artefact's ``#fragment`` references pass the allow-list and the document is
    rasterized: the bytes the tool wrote come back unchanged, so the guard refuses references
    without refusing the artefact a conforming encoder produces."""

    _answer_with(monkeypatch, _emitting_command(_STUB_RASTER))
    assert _REAL_RASTERIZE_SVG(_FRAGMENT_ONLY_DOCUMENT, _RASTERIZER_STATUS) == _STUB_RASTER


def _cairosvg_stand_in(*, accepts_fetcher: bool, seen: list[object]) -> types.ModuleType:
    """Return a stand-in ``cairosvg`` module whose ``svg2png`` records the fetcher it was offered.

    `accepts_fetcher` decides whether the stand-in's ``svg2png`` takes the ``url_fetcher`` keyword,
    which is the whole difference between the cairosvg the framework must restrict and an older one
    it must still run; `seen` receives the fetcher, or ``None`` for the shape that takes none.
    """
    module = types.ModuleType("cairosvg")
    if accepts_fetcher:

        def with_fetcher(
            bytestring: bytes, url_fetcher: Callable[..., bytes] | None = None
        ) -> bytes:
            """Answer the artefact and record the fetcher the caller offered."""
            del bytestring
            seen.append(url_fetcher)
            return _STUB_RASTER

        module.svg2png = with_fetcher
        return module

    def without_fetcher(bytestring: bytes) -> bytes:
        """Answer the artefact without offering a fetcher at all."""
        del bytestring
        seen.append(None)
        return _STUB_RASTER

    module.svg2png = without_fetcher
    return module


@pytest.mark.parametrize(
    "accepts_fetcher", [True, False], ids=["accepts-url-fetcher", "no-url-fetcher"]
)
def test_the_cairosvg_path_restricts_the_fetcher_only_where_the_module_takes_one(
    accepts_fetcher: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cairosvg path hands a fetcher that refuses every URL to a module whose ``svg2png`` takes
    one, and still calls a module whose ``svg2png`` does not: the restriction is never an argument
    an older cairosvg would reject, and the document's own scan guards both."""
    seen: list[object] = []
    monkeypatch.setitem(
        sys.modules,
        "cairosvg",
        _cairosvg_stand_in(accepts_fetcher=accepts_fetcher, seen=seen),
    )
    assert render_family._rasterize_with_cairosvg(_FRAGMENT_ONLY_DOCUMENT) == _STUB_RASTER
    assert len(seen) == 1, "the stand-in's svg2png must be called exactly once"
    fetcher = seen[0]
    if not accepts_fetcher:
        assert fetcher is None, "a module taking no url_fetcher must be called without one"
        return
    assert callable(fetcher), "a module taking url_fetcher must be handed a fetcher"
    with pytest.raises(render_family._ExternalReferenceRefused):
        fetcher(_REFUSED_REFERENCE)


def _flooding_command(stream: str, payload: bytes) -> tuple[str, ...]:
    """Return an argv-only command that writes `payload` to `stream`, reading its stdin first.

    The same stand-in aimed at one stream at a time: `stream` is ``"stdout"`` or ``"stderr"``, named
    as an attribute of :mod:`sys` in the child, so one command shape proves the capture bound on
    either stream and the stream that is not under test is left with nothing in it.
    """
    program = (
        "import sys; sys.stdin.buffer.read(); "
        "flood = getattr(sys, sys.argv[2]).buffer; "
        "flood.write(bytes.fromhex(sys.argv[1])); flood.flush()"
    )
    return (sys.executable, "-c", program, payload.hex(), stream)


# The capture ceiling the over-bound round installs, in bytes: one stub-image side's worth, derived
# from the stub's own geometry so the stub PNG the emitting command writes passes it many times
# over, and so this module restates no byte bound of its own.
_TINY_STREAM_BOUND_BYTES: Final[int] = _STUB_RASTER_SIDE_PX


@pytest.mark.parametrize("stream", ["stdout", "stderr"], ids=["stdout", "stderr"])
def test_the_rasterizer_sink_refuses_a_stream_past_its_byte_ceiling(
    stream: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rasterizer that floods either stream is refused for that alone: `stream` names the stream
    under test, and the raise names the ceiling the capture applied and the stream that passed it,
    so that stream was closed at the bound instead of being buffered whole."""
    monkeypatch.setattr(render_family, "_MAX_CAPTURED_RASTER_BYTES", _TINY_STREAM_BOUND_BYTES)

    message = _refusal(monkeypatch, _flooding_command(stream, _STUB_RASTER))
    assert f"{_TINY_STREAM_BOUND_BYTES}-byte" in message, (
        f"the refusal must name the ceiling it applied, got {message!r}"
    )
    assert f"its {stream}" in message, (
        f"the refusal must name the {stream} that passed the ceiling, got {message!r}"
    )


# The stderr the failing rasterizer writes, in bytes: several times the documented diagnostic tail,
# derived from that constant rather than restated, so a message carrying the stream would show.
_LONG_STDERR_BYTES: Final[int] = render_family._RASTERIZER_DIAGNOSTIC_CHARS * 4

# The last thing that rasterizer writes to stderr, which only a message quoting the tail can carry.
_STDERR_TAIL_MARKER: Final[str] = "qrcheck-rasterizer-stderr-tail"

# The exit status the failing rasterizer leaves, which the refusal must name.
_RASTERIZER_FAILURE_EXIT: Final[int] = 3


def _failing_command() -> tuple[str, ...]:
    """Return an argv-only command that floods stderr and exits non-zero, reading its stdin first.

    The other half of the stand-in contract: a rasterizer that fails is reported by its exit status
    and by what it last said, so this command says a great deal and then fails.
    """
    program = (
        "import sys; sys.stdin.buffer.read(); "
        "sys.stderr.buffer.write(b'x' * int(sys.argv[1])); "
        "sys.stderr.buffer.write(sys.argv[2].encode('utf-8')); "
        "sys.exit(int(sys.argv[3]))"
    )
    return (
        sys.executable,
        "-c",
        program,
        str(_LONG_STDERR_BYTES),
        _STDERR_TAIL_MARKER,
        str(_RASTERIZER_FAILURE_EXIT),
    )


def test_a_failing_rasterizer_reports_the_exit_status_and_only_the_tail_of_its_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rasterizer that exits non-zero is reported by its exit status and by the last
    :data:`render_family._RASTERIZER_DIAGNOSTIC_CHARS` characters of its stderr, so one diagnostic
    never carries a whole captured stream into the report or the evidence."""

    message = _refusal(monkeypatch, _failing_command())
    assert f"exited {_RASTERIZER_FAILURE_EXIT}" in message, message
    assert _STDERR_TAIL_MARKER in message, "the message must carry the end of what the tool said"
    assert len(message) < _LONG_STDERR_BYTES, (
        f"the message must not carry the whole stream: {len(message)} characters for a "
        f"{_LONG_STDERR_BYTES}-byte stderr"
    )


def test_the_rasterizer_capture_ceiling_is_the_cli_adapters_stream_bound() -> None:
    """The rasterizer's per-stream capture ceiling is the cli adapter's per-stream bound -- and so
    the same 16 MiB the http adapter reads from one body -- asserted rather than restated here, so
    the three bounds cannot drift apart."""
    assert render_family._MAX_CAPTURED_RASTER_BYTES == cli_adapter._MAX_CAPTURED_STREAM_BYTES, (
        "the rasterizer capture ceiling must equal the cli adapter's per-stream bound: "
        f"{render_family._MAX_CAPTURED_RASTER_BYTES} != {cli_adapter._MAX_CAPTURED_STREAM_BYTES}"
    )
