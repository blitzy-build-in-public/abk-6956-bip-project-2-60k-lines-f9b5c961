"""The render family: the five ``RN-*`` checks and the product's one SVG rasterizer.

Owns the product's only SVG-rasterizer discovery (:func:`discover_svg_rasterizer`,
:data:`SVG_RASTERIZER_AUTO_ORDER`) and its only rasterization (:func:`rasterize_svg`), so
the tool ``qrcheck doctor`` prints is the tool ``RN-SVG`` uses; owns the five
self-registering ``RN-*`` checks with the capability requirements of the resolution table;
owns the two geometry measurements that need no decoder (:func:`measure_module_px`,
:func:`measure_quiet_zone`), which deliberately do not borrow :mod:`qrcheck.decode.matrix`
because a quiet-zone or module-size defect must be visible even when no decoder can read
the symbol at all; and owns :data:`EXPECTED_ENCODES`, the declared adapter-invocation
counts that make a retry detectable as the declared count plus one.

Fixed case matrix for one full run at the run's error level: ``RN-QUIET-ZONE`` and
``RN-MODULE-SIZE`` take the first 12 corpus records (12 encodes, 0 decodes each),
``RN-CONTRAST`` 6 palettes x 2 polarities (12 encodes, 12 decodes), ``RN-SVG`` the first 12
corpus records (12 encodes, 12 decodes, counted on the rasterized PNG), ``RN-LOGO`` 3
overlay sizes x EC H and L (6 encodes, 6 decodes) -- 54 cases in total.

Restates nothing another module owns: every ratio is
:func:`qrcheck.decode.consensus.contrast_ratio` and the inverted-palette predicate
:func:`qrcheck.decode.consensus.is_inverted_request`; the evidence layout, the recovery
budget and seed derivation stay in :mod:`qrcheck.checks.support` and
:mod:`qrcheck.config.seeding`; every pixel crosses the single image boundary in
:mod:`qrcheck.evidence.mux` with no concrete adapter import and no ``PIL`` import
(``tests/unit/test_layering.py`` holds both); and request validation is the port's
``build_request`` through :func:`qrcheck.checks.support.build`.  No threshold is a literal
here -- the contrast floor comes from the target's ``contrast_floor``, the overlay ceiling
and the occlusion margin from ``[checks.damage]`` -- and no check is weakened, skipped,
retried or relaxed for the ``defective`` target.

Reason codes emitted: ``sampling_failed`` when the module geometry cannot be established at
all (no dark bounding box, no measurable finder run, no implied module count);
``value_mismatch`` when a measurement contradicts the request; ``missing_dependency`` when
no rasterizer was discovered, which skips every ``RN-SVG`` case with the attempted discovery
order and never counts as a pass; and ``target_rejected``, ``contract_error`` and
``unsupported_capability`` as :func:`qrcheck.checks.support.case_status_for_error`
translates them.  ``TargetUnavailable`` and ``TargetTimeout`` are never converted here: they
propagate so the runner can apply its one-error-result-per-check target-loss policy.

Determinism: nothing reads a wall clock for an assertion (``time.perf_counter`` only fills
``duration_ms``), nothing reads ``hash()`` or ``id()``, every iteration is over a sorted
sequence or a declared tuple, and the overlay bitmap's only random source is
:func:`qrcheck.config.seeding.derive_seed` into ``numpy``'s ``PCG64``, so any case is
re-derivable from ``(run seed, check id, case id)``.

Decisions this file had to take: ``RN-SVG`` requests a zero-module quiet zone, because its
assertion is the canvas module count at one pitch and a short margin is ``RN-QUIET-ZONE``'s
finding, so the ``defective`` target cannot fail both; ``RN-SVG`` records the payload the
rasterized symbol decodes to instead of asserting it, since payload fidelity is
``RT-ROUNDTRIP``'s assertion; ``RN-MODULE-SIZE`` asserts a legal-module-count dimension
contract needing no decoder and no version pin; and ``RN-LOGO`` sizes its overlay from the
smallest legal symbol and measures the covered fraction against the symbol's own module
area.  The one temporary file this module writes is the logo bitmap, named with the
product-wide ``qrcheck-`` prefix that ``CL-ARTIFACTS`` scans for and removed in a ``finally``
block, so a run leaves nothing behind even when the target refuses the request.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy

from qrcheck.checks import support
from qrcheck.config.models import (
    DEFAULT_DECODE_IMAGE_PX,
    SVG_RASTERIZERS,
    RenderConfig,
    TargetConfig,
)
from qrcheck.config.seeding import derive_seed
from qrcheck.decode.consensus import contrast_ratio, is_inverted_request
from qrcheck.decode.matrix import module_count_for_version, version_from_dimension
from qrcheck.evidence.mux import decode_image, encode_png
from qrcheck.evidence.naming import LOGO_PNG
from qrcheck.registry.check import (
    DEPENDENCY_PRIMARY_DECODER,
    DEPENDENCY_SVG_RASTERIZER,
    Case,
    RunContext,
    check,
)
from qrcheck.report.model import CaseResult, Reason
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Annotation resolution only: no runtime edge into the corpus builder.
    from qrcheck.corpus.builder import CorpusRecord

__all__ = [
    "EXPECTED_ENCODES",
    "MODULE_SIZE_TOLERANCE_PX",
    "QUIET_ZONE_TOLERANCE_MODULES",
    "RASTERIZER_TIMEOUT_S",
    "SVG_RASTERIZER_AUTO_ORDER",
    "RasterizerStatus",
    "contrast_cases",
    "discover_rasterizer",
    "discover_svg_rasterizer",
    "logo_cases",
    "measure_module_px",
    "measure_quiet_zone",
    "module_size_cases",
    "quiet_zone_cases",
    "rasterize_svg",
    "svg_cases",
    "svg_rasterizer_available",
    "svg_rasterizer_status",
]

# The five checks this module registers, in the identifier order the registry iterates.
RN_QUIET_ZONE: Final[str] = "RN-QUIET-ZONE"
RN_MODULE_SIZE: Final[str] = "RN-MODULE-SIZE"
RN_CONTRAST: Final[str] = "RN-CONTRAST"
RN_SVG: Final[str] = "RN-SVG"
RN_LOGO: Final[str] = "RN-LOGO"

# The one family these checks belong to: the owning module's name, which is also the
# JUnit ``classname`` suffix (``classname="qrcheck.render"``).
_FAMILY: Final[str] = "render"

# The two case-identifier prefixes AAP §0.3.4 fixes: a corpus-driven case is named after
# its record, a constructed one after the name its family gave it.  Written out here
# because a check has to name its own case when it builds a result, and the same two
# strings are what ``qrcheck.report.model.failure_id`` embeds in an evidence path.
_CORPUS_CASE_PREFIX: Final[str] = "corpus:"
_SYNTHETIC_CASE_PREFIX: Final[str] = "synthetic:"

# AAP §0.3.4 and §0.7.2 fix the discovery order: an importable ``cairosvg`` module
# first (never a hard dependency), then the four command-line rasterizers on ``PATH``.
# The order is a contract rather than a preference -- ``doctor`` and ``RN-SVG`` report
# the sequence they attempted, so reordering it changes a report.
SVG_RASTERIZER_AUTO_ORDER: Final[tuple[str, ...]] = (
    "cairosvg",
    "rsvg-convert",
    "resvg",
    "magick",
    "convert",
)

# The ceiling on one rasterizer invocation.  AAP §0.3.4 gives no configuration key for
# it -- the ``[render]`` table holds only ``svg_rasterizer`` -- so the backstop is
# declared here rather than invented as a threshold: it exists to stop a hung rasterizer
# from hanging the run, and it is not a performance budget (that is ``[checks.performance]``).
RASTERIZER_TIMEOUT_S: Final[float] = 60.0

# The largest canvas this framework will ask a rasterizer to render, per side, in pixels:
# the same ceiling the decode pipeline applies to a raster artefact, derived from the same
# single declaration ``[checks.performance] decode_image_px`` that
# :mod:`qrcheck.config.models` owns (AAP §0.6.2).  A target's SVG is untrusted, and
# ``rsvg-convert`` allocates the canvas it is told to draw, so an unbounded ``width`` would
# let a small compressed artefact exhaust the run before any decoder sees a pixel.
_MAX_RASTER_SIDE_PX: Final[int] = 8 * DEFAULT_DECODE_IMAGE_PX

# How many CSS pixels one SVG length unit spans at the 96 dpi every rasterizer defaults
# to, which is what turns a declared canvas into the pixel count it will allocate.  A unit
# outside this table (a percentage, say) has no absolute extent, so no ceiling applies.
_UNIT_TO_PX: Final[Mapping[str, float]] = {
    "": 1.0,
    "px": 1.0,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "mm": 96.0 / 25.4,
    "cm": 96.0 / 2.54,
    "in": 96.0,
    "em": 16.0,
}

# The module-edge tolerance ``RN-MODULE-SIZE`` allows, in pixels.  AAP §0.6.2 names this
# as the owning check's own single named constant: a measurement rounded to whole pixels
# can be off by one from the requested edge on a conforming render, and two would hide a
# real geometry defect.
MODULE_SIZE_TOLERANCE_PX: Final[float] = 1.0

# A quarter of a module of slack on the measured quiet zone, for the same reason at the
# fraction level: the margin is measured in whole pixels and divided by an estimated
# module pitch, so a conforming four-module zone can read as a hair under four.  A
# quarter of a module cannot hide a missing module, and the ``defective`` target draws
# one module where four were requested -- a whole three modules short.
QUIET_ZONE_TOLERANCE_MODULES: Final[float] = 0.25

# The fixed samples AAP §0.8.1 fixes for this family: the first twelve corpus records
# in ``id`` order for each corpus-driven check.  A sample size, not a threshold.
_SAMPLE_RECORDS: Final[int] = 12

# The geometry each corpus-driven check requests, cycled over the sample so a target
# that ignores the field and always renders one value cannot pass.  Both are case
# parameters, never thresholds.
_MODULE_SIZE_CHOICES: Final[tuple[int, ...]] = (4, 6, 8)
_QUIET_ZONE_CHOICES: Final[tuple[int, ...]] = (4, 5, 6)

# The quiet zone ``RN-MODULE-SIZE`` and ``RN-LOGO`` request, in modules: ISO/IEC 18004
# §7.3.6 requires four, and pinning it is what makes the dimension contract and the
# overlay area decidable before the symbol exists.
_PINNED_QUIET_ZONE_MODULES: Final[int] = 4

# The quiet zone ``RN-SVG`` requests, in modules: none.  The vector canvas is asserted to
# carry the module count times one pitch, which is only decidable when the canvas is
# exactly the symbol -- and a vector artefact's margin is not what this check is about.
# Pinning four instead would make the canvas ``(modules + 8) * pitch``, so a target that
# renders a *short* quiet zone would fail this rendering check too, which is
# ``RN-QUIET-ZONE``'s finding and not this one's (AAP §0.4.2's attribution).
_SVG_QUIET_ZONE_MODULES: Final[int] = 0

# The payload length ``RN-LOGO`` requests, in bytes, excluding the case's own label.
# A case parameter, chosen from the capacity tables' consequences rather than as a
# threshold: a payload this long puts the symbol at Version 5 or above, where the
# overlay's contiguous block is spread over several interleaved codeword blocks and so
# behaves as the documented recovery fraction predicts, and it keeps the EC-L cases above
# ``recovery_fraction("L") - occlusion_margin`` so they are recorded rather than asserted.
_LOGO_PAYLOAD_BYTES: Final[int] = 40

# The module edge ``RN-LOGO`` requests, in pixels.  The overlay fraction is
# scale-invariant in this value, so it only decides how large the bitmap under test is.
_LOGO_MODULE_SIZE_PX: Final[int] = 4

# The three overlay sizes of AAP §0.8.1 are fractions of the symbol's *area*: one third,
# two thirds and the whole of ``[checks.damage] logo_area_fraction``.  The largest step
# therefore spends exactly the configured ceiling and no value is written twice.
_LOGO_FRACTION_STEPS: Final[tuple[int, ...]] = (1, 2, 3)
_LOGO_FRACTION_DENOMINATOR: Final[int] = 3

# The two error levels ``RN-LOGO`` contrasts: H has spare correction capacity, L has
# almost none, so the same overlay is asserted to decode at one and only recorded at the
# other (AAP §0.3.5's "wrong assertion" table forbids asserting the EC-L outcome).
_LOGO_EC_LEVELS: Final[tuple[str, str]] = ("H", "L")

# The six colour pairs ``RN-CONTRAST`` measures.  They are case parameters, not
# thresholds: each is measured at or above the documented 7:1 floor, and the assertion
# compares the image's measured ratio against the target's own declared
# ``contrast_floor``.  Written out rather than derived, so a diff shows a changed palette.
_CONTRAST_PALETTES: Final[tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] = (
    ((0, 0, 0), (255, 255, 255)),
    ((32, 32, 32), (255, 255, 255)),
    ((0, 64, 0), (255, 255, 255)),
    ((0, 0, 128), (255, 255, 255)),
    ((128, 0, 0), (255, 255, 255)),
    ((0, 0, 0), (204, 204, 204)),
)

# The two polarities each palette is requested at.  ``inverted`` swaps the pair so the
# modules are the light pixels, which exercises the framework's polarity-normalised
# second decode pass rather than any decoder's inversion support (zxing-cpp 2.2.0 has no
# switch for it and OpenCV 4.10 cannot detect a light-on-dark symbol at all).
_CONTRAST_POLARITIES: Final[tuple[str, ...]] = ("normal", "inverted")

# The prefix of the one temporary file this module writes.  ``qrcheck-`` is the
# product-wide prefix ``CL-ARTIFACTS`` scans ``tempfile.gettempdir()`` for, so a leak of
# this bitmap is exactly what that check reports.
_TEMP_PREFIX: Final[str] = "qrcheck-logo-"

# How many pixel rows and columns beside the top-left finder pattern are probed for its
# dark run.  The finder's outer ring is one module thick, so the first module's worth of
# rows all read the full seven-module width and the longest reading is the finder's.
_FINDER_PROBE_LINES: Final[int] = 3

# ISO/IEC 18004 §7.3.3: a finder pattern is a 7x7 module square, so the dark run across
# its top edge is exactly seven modules wide and the pitch is that run divided by seven.
_FINDER_MODULES: Final[int] = 7

# The finder pattern's middle-row (and middle-column) signature, in modules: one dark, one
# light, three dark, one light, one dark.  Together they span the whole seven modules, and
# each run is what confirms the pitch measured across them.
_FINDER_SIGNATURE: Final[tuple[int, ...]] = (1, 1, 3, 1, 1)

# How far a measured run may sit from its expected width, in modules, before the line is
# declared not to be a finder pattern.  Half a module absorbs the pixel rounding of a
# render whose pitch is not a whole number of pixels; a run outside it is a different
# object, not a noisier one.
_FINDER_RUN_TOLERANCE: Final[float] = 0.5

# The spread below which no threshold can separate a symbol's modules from its
# background: sixteen of the 255 grey levels, because a narrower band means the "dark"
# mask is measuring antialiasing rather than modules.
_MIN_GEOMETRY_SPREAD: Final[float] = 16.0

# A tolerance for comparing a pitch that should be a whole number of user units.  The
# value is a float-comparison epsilon, not a threshold on a symbol.
_INTEGRAL_EPSILON: Final[float] = 1e-6

# The adapter-invocation counts AAP §0.4.3 declares for this family: one ``generate`` per
# case, and no second call anywhere.  ``RN-QUIET-ZONE``, ``RN-MODULE-SIZE``, ``RN-CONTRAST``
# and ``RN-SVG`` draw twelve cases each from the fixed sample and ``RN-LOGO`` six, so the
# totals below are the case counts.  ``tests/unit/test_checks_negative.py`` drives each
# check with a counting stub and asserts the observed count equals this table, so a retry
# introduced anywhere in this module fails that test as the declared count plus one.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        RN_QUIET_ZONE: 12,
        RN_MODULE_SIZE: 12,
        RN_CONTRAST: 12,
        RN_SVG: 12,
        RN_LOGO: 6,
    }
)


# The single SVG-rasterizer discovery


@dataclass(frozen=True, slots=True)
class RasterizerStatus:
    """What the ``[render] svg_rasterizer`` setting resolved to on this host.

    One record is built per run and read by ``qrcheck doctor`` (which prints it), by the runner
    (which maps :attr:`available` onto the
    :data:`~qrcheck.registry.check.DEPENDENCY_SVG_RASTERIZER` token ``qrcheck.registry.resolve``
    consumes) and by ``RN-SVG`` (which rasterizes with :attr:`name`, or skips with
    :attr:`attempted` when there is nothing to use).  ``configured`` echoes the configured value
    verbatim, so a report distinguishes "the run asked for ``cairosvg`` and it was absent" from
    "the run asked for ``auto`` and found ``rsvg-convert``"; ``attempted`` is the discovery order
    actually probed -- one entry for an explicit name, every name tried up to and including the
    one that answered for ``auto``, and empty for ``none``, which probes nothing.
    """

    configured: str
    name: str | None
    available: bool
    attempted: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe projection ``doctor --json`` and the evidence carry.

        ``attempted`` becomes a list because JSON has no tuple, and the four keys are
        emitted in a fixed order so two runs' reports line up field for field.
        """
        return {
            "configured": self.configured,
            "name": self.name,
            "available": self.available,
            "attempted": list(self.attempted),
        }


def _configured_rasterizer(value: RenderConfig | str) -> str:
    """Return the configured rasterizer name from what a caller handed us.

    The ``[render]`` record is the documented argument, and the string form is accepted because a
    caller that already holds the name (a test, a CLI override) has no reason to build a record
    for it; both spellings resolve to the same value, so the two discovery call sites cannot
    disagree about which object to pass.

    Raises:
        ValueError: The value is not one of the declared rasterizers.  The configuration schema
            already rejects that at load time, so a value reaching here can only have been
            constructed by hand.
    """
    name = value if isinstance(value, str) else value.svg_rasterizer
    if name not in SVG_RASTERIZERS:
        raise ValueError(
            f"[render] svg_rasterizer must be one of {list(SVG_RASTERIZERS)}, got {name!r}"
        )
    return name


def _probe_rasterizer(name: str) -> bool:
    """Report whether the rasterizer `name` can actually be used on this host.

    ``cairosvg`` is probed as an importable module and every other name as an executable on
    ``PATH``.  Neither probe imports or runs anything: ``find_spec`` locates a module's loader
    without executing the module and ``shutil.which`` stats the ``PATH`` directories, so
    discovery never runs a third-party binary merely to find out whether it exists.
    """
    if name == "cairosvg":
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A missing parent package raises ImportError and a module whose ``__spec__``
            # was never set raises ValueError; both mean "not usable", and neither may
            # escape a probe.
            return False
    return shutil.which(name) is not None


def discover_svg_rasterizer(value: RenderConfig | str) -> RasterizerStatus:
    """Resolve the configured SVG rasterizer against what this host provides.

    The product's only discovery implementation (AAP §0.3.4, §0.7.2), read by both ``doctor``
    and ``RN-SVG``, so the tool the run reports and the tool the check uses are one value, and a
    host with no rasterizer degrades ``RN-SVG`` to ``skipped``/``missing_dependency`` instead of
    failing it.  An explicit value is honoured exactly -- no fallback follows a name the run
    asked for, because a silent substitution would make ``doctor`` report a tool the
    configuration never named -- with ``none`` reporting ``available=False`` and probing
    nothing; ``auto`` probes :data:`SVG_RASTERIZER_AUTO_ORDER` in order and stops at the first
    hit.  The status carries the configured value, the resolved name (``None`` when nothing was
    found), whether it is usable, and the order that was probed.

    Raises:
        ValueError: The configured value is not one of
            :data:`qrcheck.config.models.SVG_RASTERIZERS`.
    """
    configured = _configured_rasterizer(value)
    if configured == "none":
        return RasterizerStatus(
            configured=configured, name=None, available=False, attempted=()
        )
    if configured != "auto":
        attempted = (configured,)
        found = _probe_rasterizer(configured)
        return RasterizerStatus(
            configured=configured,
            name=configured if found else None,
            available=found,
            attempted=attempted,
        )
    attempted: list[str] = []
    for candidate in SVG_RASTERIZER_AUTO_ORDER:
        attempted.append(candidate)
        if _probe_rasterizer(candidate):
            return RasterizerStatus(
                configured=configured,
                name=candidate,
                available=True,
                attempted=tuple(attempted),
            )
    return RasterizerStatus(
        configured=configured, name=None, available=False, attempted=tuple(attempted)
    )


def svg_rasterizer_available(value: RenderConfig | str) -> bool:
    """Report whether an SVG rasterizer is usable for this run.

    The predicate form of :func:`discover_svg_rasterizer`, for a caller that fills the runner's
    availability mapping and needs nothing else from the status: a projection of the same
    discovery, never a second one.
    """
    return discover_svg_rasterizer(value).available


# The names other modules may reach this discovery by.  ``discover_svg_rasterizer`` is
# the canonical spelling; these are aliases of the one function, not a second
# implementation, so a caller that asks for the status or the rasterizer object by
# another name still gets this file's single discovery.
discover_rasterizer = discover_svg_rasterizer
svg_rasterizer_status = discover_svg_rasterizer


def _rasterizer_command(name: str) -> tuple[str, ...]:
    """Return the argv that rasterizes SVG on stdin to PNG on stdout for `name`.

    Every entry is an argument list, never a shell string: this product's subprocesses are
    argv-only with ``shell=False`` (AAP §0.3.2), and a rasterizer is third-party code reading an
    untrusted artefact, so the rule holds here too.  ``-`` is each tool's own spelling for stdin
    and none of these writes a file, so the argv carries no shell metacharacter and no
    user-controlled text.

    Raises:
        ValueError: `name` is not one of the declared command-line rasterizers, which means the
            status was built without this module's discovery.
    """
    if name == "rsvg-convert":
        return (name, "--format=png", "-")
    if name == "resvg":
        return (name, "--format", "png", "-", "-o", "-")
    if name == "magick":
        return (name, "svg:-", "png:-")
    if name == "convert":
        return (name, "svg:-", "png:-")
    raise ValueError(
        f"{name!r} is not a command-line rasterizer; the discovered ones are "
        f"{[candidate for candidate in SVG_RASTERIZER_AUTO_ORDER if candidate != 'cairosvg']}"
    )


def rasterize_svg(svg_bytes: bytes, status: RasterizerStatus) -> bytes:
    """Rasterize an SVG artefact to PNG bytes with the discovered rasterizer.

    The product's only rasterization: ``cairosvg`` is driven through its module when discovery
    found one, and every other rasterizer is an argv-only subprocess reading the SVG on stdin
    and writing the PNG to stdout, with :data:`RASTERIZER_TIMEOUT_S` as the backstop.  Nothing
    is written to disk and nothing is imported at module scope.

    Raises:
        RuntimeError: No rasterizer is available, the configured one emitted nothing, exited
            non-zero, timed out or could not be executed; the message names the command, its
            exit status and its captured stderr, and the runner records it as an ``error``
            result with the traceback rather than a verdict about the target.
        ValueError: ``svg_bytes`` is not bytes-like, or `status` names a rasterizer this module
            cannot drive.
    """
    if not isinstance(svg_bytes, (bytes, bytearray, memoryview)):
        raise ValueError(f"svg_bytes must be bytes; got {type(svg_bytes).__name__}")
    if status.name is None or not status.available:
        raise RuntimeError(
            f"no SVG rasterizer is available for the configured [render] svg_rasterizer = "
            f"{status.configured!r}; discovery attempted {list(status.attempted)}"
        )
    payload = bytes(svg_bytes)
    oversized = _oversized_canvas(payload)
    if oversized is not None:
        raise RuntimeError(
            f"the SVG declares a canvas of {oversized:g} px per side, above the "
            f"{_MAX_RASTER_SIDE_PX} px per-side ceiling this framework rasterizes: a "
            f"target's vector artefact is untrusted and the rasterizer allocates it"
        )
    name = status.name
    if name == "cairosvg":
        return _rasterize_with_cairosvg(payload)
    argv = _rasterizer_command(name)
    try:
        completed = subprocess.run(
            list(argv),
            input=payload,
            shell=False,
            capture_output=True,
            timeout=RASTERIZER_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"SVG rasterizer {name!r} did not finish within {RASTERIZER_TIMEOUT_S} s: "
            f"command {list(argv)}"
        ) from None
    except OSError as exc:
        raise RuntimeError(
            f"SVG rasterizer {name!r} could not be executed: command {list(argv)} failed "
            f"with {type(exc).__name__}: {exc}"
        ) from None
    if completed.returncode != 0:
        raise RuntimeError(
            f"SVG rasterizer {name!r} failed: command {list(argv)} exited "
            f"{completed.returncode} and reported "
            f"{completed.stderr.decode('utf-8', 'replace')!r}"
        )
    if not completed.stdout:
        raise RuntimeError(
            f"SVG rasterizer {name!r} produced no output: command {list(argv)} exited 0 "
            f"and wrote nothing to stdout, stderr was "
            f"{completed.stderr.decode('utf-8', 'replace')!r}"
        )
    return completed.stdout


def _rasterize_with_cairosvg(svg_bytes: bytes) -> bytes:
    """Rasterize `svg_bytes` through an importable ``cairosvg`` module.

    ``cairosvg`` is never a dependency of this product: it is discovered at run time from the
    host's own environment, so the import happens here, on the one path that needs it.

    Raises:
        RuntimeError: The module cannot be imported, refused the artefact, or produced nothing --
            the same vocabulary a failing binary is reported in.
    """
    try:
        module = importlib.import_module("cairosvg")
        rasterized = module.svg2png(bytestring=svg_bytes)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # The five shapes a rasterizer module can report through this call: an import
        # that fails, a native library that cannot open the input, the library's own
        # error type, a refused argument and a refusal raised by the module itself.  Each
        # is a rasterizer fault, never a verdict about the target, so all of them become
        # the one documented RuntimeError.
        raise RuntimeError(
            f"SVG rasterizer 'cairosvg' failed: {type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(rasterized, (bytes, bytearray)) or not rasterized:
        raise RuntimeError(
            "SVG rasterizer 'cairosvg' produced no output: cairosvg.svg2png returned "
            f"{type(rasterized).__name__}"
        )
    return bytes(rasterized)


# Pixels, through the one image boundary


def _grey_levels(image_bytes: bytes) -> numpy.ndarray:
    """Return the artefact as an ``H x W`` ``uint8`` grey array.

    The reduction is Pillow's own ``convert("L")`` -- the ITU-R BT.601 weighting the decoder path
    also reads -- so this module states no luminance formula of its own.  A fully transparent
    pixel is replaced by the light level, because a transparent background shows the page rather
    than the black its colour bands happen to store; a half-transparent edge pixel is a real
    blend of module and background and is left as measured.

    Raises:
        UnidentifiedImageError: Pillow cannot recognise the bytes.  The adapters convert that
            into ``TargetContractError`` before a check sees them, so a raise here means a
            framework artefact rather than a target's.
    """
    image = decode_image(image_bytes)
    try:
        levels = numpy.array(image.convert("L"), dtype=numpy.uint8, copy=True)
        if "A" in image.getbands():
            alpha = numpy.array(image.getchannel("A"), dtype=numpy.uint8, copy=True)
            levels[alpha == 0] = 255
    finally:
        image.close()
    return levels


def _rgb_pixels(image_bytes: bytes) -> numpy.ndarray:
    """Return the artefact as an ``H x W x 3`` ``uint8`` array, transparency read as the page.

    The colour counterpart of :func:`_grey_levels`, with the same transparency rule, and the
    array the palette measurement and the overlay-visibility measurement read.

    Raises:
        UnidentifiedImageError: Pillow cannot recognise the bytes.
    """
    image = decode_image(image_bytes)
    try:
        pixels = numpy.array(image.convert("RGB"), dtype=numpy.uint8, copy=True)
        if "A" in image.getbands():
            alpha = numpy.array(image.getchannel("A"), dtype=numpy.uint8, copy=True)
            pixels[alpha == 0] = 255
    finally:
        image.close()
    return pixels


def _dark_mask(levels: numpy.ndarray) -> numpy.ndarray | None:
    """Return the boolean dark-module mask of `levels`, or ``None`` when there is none.

    The threshold is the midpoint of the artefact's own darkest and lightest grey level, which
    needs no contrast assumption and is stable for any palette: a black-on-white symbol and a
    navy-on-cream one are separated at their own midpoint.  An artefact whose darkest and
    lightest pixels differ by less than :data:`_MIN_GEOMETRY_SPREAD` grey levels has no such
    threshold, so the geometry is reported as unestablished rather than measured from noise.
    """
    darkest = float(levels.min())
    lightest = float(levels.max())
    if lightest - darkest < _MIN_GEOMETRY_SPREAD:
        return None
    return levels < (darkest + lightest) / 2.0


def _leading_dark_run(line: numpy.ndarray, start: int) -> int:
    """Return the length of the dark run beginning at index `start` of a 1-D line.

    Args:
        line: One row or column of the dark mask.
        start: The index to begin counting from.

    Returns:
        The number of consecutive dark pixels, ``0`` when `start` is light or out of
        range.
    """
    if start < 0 or start >= line.shape[0] or not line[start]:
        return 0
    length = 0
    for value in line[start:]:
        if not value:
            break
        length += 1
    return length


def _finder_runs(line: numpy.ndarray, start: int, count: int) -> tuple[int, ...]:
    """Return up to `count` run lengths of alternating values, beginning at `start`.

    A "run" is a maximal span of equal values, and the alternation is implied by the QR finder
    pattern itself: its middle row and middle column read dark, light, dark, light, dark.  A line
    whose first value at `start` is light yields no runs, because a finder pattern begins at its
    outer ring, and fewer than `count` runs come back when the line ends first.
    """
    if start < 0 or start >= line.shape[0] or not line[start]:
        return ()
    runs: list[int] = []
    current = bool(line[start])
    length = 0
    for value in line[start:]:
        if bool(value) is not current:
            runs.append(length)
            if len(runs) == count:
                return tuple(runs)
            current = bool(value)
            length = 0
        length += 1
    runs.append(length)
    return tuple(runs)


def _finder_pitch(line: numpy.ndarray, start: int) -> float | None:
    """Return the module pitch measured across one finder pattern, or ``None``.

    ISO/IEC 18004 §7.3.3: across a finder pattern's middle row or column the five runs are 1, 1,
    3, 1 and 1 modules wide, so their total is exactly seven modules and each run confirms the
    pitch it implies.  That confirmation is what stops a measurement being taken from something
    that merely looks dark -- noise, a text block, a partially rendered symbol -- because a run
    half a module away from its expected width means this is not a finder pattern at all, and the
    honest answer is then ``None`` rather than a plausible-looking number.
    """
    runs = _finder_runs(line, start, len(_FINDER_SIGNATURE))
    if len(runs) != len(_FINDER_SIGNATURE):
        return None
    pitch = sum(runs) / _FINDER_MODULES
    if pitch <= 0:
        return None
    for expected, observed in zip(_FINDER_SIGNATURE, runs):
        if abs(observed / pitch - expected) > _FINDER_RUN_TOLERANCE:
            return None
    return pitch


def _module_pitch(dark: numpy.ndarray) -> float | None:
    """Return the module edge in pixels measured from the top-left finder pattern.

    ISO/IEC 18004 §7.3.3: a finder pattern is a 7x7 module square -- a dark outer ring, a light
    ring and a 3x3 dark core -- so the dark run along its top edge and down its left edge is
    exactly seven modules wide.  The top-left finder sits at the corner of the dark bounding box,
    which is why the search starts at the first dark row and column: no decoder, no corner
    quadrangle and no version assumption is needed, so the measurement still works on a symbol
    nothing can decode.  The top edge gives a rough pitch that locates the finder's middle row
    and middle column, and the returned pitch is measured across those, where the full 1:1:3:1:1
    signature can be confirmed (:func:`_finder_pitch`); both axes are measured and averaged, so a
    render that is off by a pixel in one axis does not shift the other.
    """
    rows = numpy.flatnonzero(dark.any(axis=1))
    columns = numpy.flatnonzero(dark.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    top = int(rows[0])
    left = int(columns[0])
    rough = max(
        _leading_dark_run(dark[row, :], left)
        for row in range(top, min(top + _FINDER_PROBE_LINES, dark.shape[0]))
    ) / _FINDER_MODULES
    if rough <= 0:
        return None
    estimates: list[float] = []
    middle_row = top + int(round(rough * (_FINDER_MODULES - 1) / 2))
    if middle_row < dark.shape[0]:
        measured = _finder_pitch(dark[middle_row, :], left)
        if measured is not None:
            estimates.append(measured)
    middle_column = left + int(round(rough * (_FINDER_MODULES - 1) / 2))
    if middle_column < dark.shape[1]:
        measured = _finder_pitch(dark[:, middle_column], top)
        if measured is not None:
            estimates.append(measured)
    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def _quiet_zone_from(dark: numpy.ndarray, module_px: float) -> dict[str, float]:
    """Return the all-light margin on each side of the dark bounding box, in modules.

    The dark bounding box is the rectangle the modules occupy, so the rows above its first dark
    row, the rows below its last one and the columns beside its left and right edges are the
    quiet zone as rendered.  A bottom or right margin is a lower bound on the zone the target
    drew -- the last module row may itself end in light modules -- which is the safe direction
    for an assertion of *at least* the requested zone.

    Raises:
        ValueError: `module_px` is not positive, or the artefact holds no dark pixel at all so
            there is no bounding box to measure against.
    """
    if not module_px > 0:
        raise ValueError(f"module_px must be positive; got {module_px!r}")
    rows = numpy.flatnonzero(dark.any(axis=1))
    columns = numpy.flatnonzero(dark.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("the artefact holds no dark pixel, so it has no module bounding box")
    height, width = dark.shape
    return {
        "top": float(rows[0]) / module_px,
        "right": float(width - 1 - int(columns[-1])) / module_px,
        "bottom": float(height - 1 - int(rows[-1])) / module_px,
        "left": float(columns[0]) / module_px,
    }


def measure_module_px(image_bytes: bytes) -> float | None:
    """Return the artefact's module edge in pixels, or ``None`` if it cannot be measured.

    The public form of :func:`_module_pitch`, so a caller that only needs the pitch -- or a test
    asserting the measurement itself -- reaches no private helper.  ``None`` is the honest answer
    when the geometry cannot be established, and the checks report ``sampling_failed`` for it
    rather than a verdict built on a guess.
    """
    dark = _dark_mask(_grey_levels(image_bytes))
    return None if dark is None else _module_pitch(dark)


def measure_quiet_zone(image_bytes: bytes, module_px: float) -> dict[str, float]:
    """Return the artefact's rendered quiet zone on all four sides, in modules.

    The public form of :func:`_quiet_zone_from` for one measurement.

    Raises:
        ValueError: `module_px` is not positive, the artefact has no measurable geometry, or it
            holds no dark pixel at all.
    """
    dark = _dark_mask(_grey_levels(image_bytes))
    if dark is None:
        raise ValueError("the artefact has no dark/light separation to measure a quiet zone in")
    return _quiet_zone_from(dark, module_px)


def _measure_colours(image_bytes: bytes) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the artefact's darkest and lightest rendered colours.

    The two module colours as the image actually carries them: every pixel sharing the minimum
    channel sum is averaged for the dark reading and every pixel sharing the maximum for the light
    one, so a single antialiased outlier cannot move either value.  The channel sum only orders
    the pixels -- it is a clustering key, never a luminance, and the ratio itself is
    :func:`qrcheck.decode.consensus.contrast_ratio`'s -- and both readings are equal when the
    image is one flat colour.
    """
    pixels = _rgb_pixels(image_bytes).reshape(-1, 3).astype(numpy.int32)
    key = pixels.sum(axis=1)
    return (
        _mean_colour(pixels[key == key.min()]),
        _mean_colour(pixels[key == key.max()]),
    )


def _mean_colour(group: numpy.ndarray) -> tuple[int, int, int]:
    """Return the rounded per-channel mean of `group` as an ``0-255`` triple.

    Args:
        group: An ``N x 3`` integer array of pixels.

    Returns:
        The mean colour.
    """
    means = group.mean(axis=0)
    return (
        int(round(float(means[0]))),
        int(round(float(means[1]))),
        int(round(float(means[2]))),
    )


def _distinct_colours(image_bytes: bytes) -> int:
    """Return how many distinct colours the artefact holds.

    A two-colour symbol renders exactly its foreground and background; a symbol carrying a
    composited overlay carries the overlay's own colours as well.  That difference is how
    ``RN-LOGO`` observes that the ``logo_overlay`` capability has behaviour behind it, rather than
    trusting a target that declares the capability and ignores it.
    """
    pixels = _rgb_pixels(image_bytes).reshape(-1, 3)
    return int(numpy.unique(pixels, axis=0).shape[0])


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    """Return the artefact's ``(width, height)`` in pixels.

    Args:
        image_bytes: The artefact exactly as the target returned it.

    Returns:
        The pixel dimensions.
    """
    image = decode_image(image_bytes)
    try:
        width, height = image.size
    finally:
        image.close()
    return int(width), int(height)


# The vector canvas

# A length as SVG writes it: a number with an optional CSS unit suffix, which is how
# ``width="148"`` and ``width="37mm"`` are both expressed.
_SVG_LENGTH: Final[re.Pattern[str]] = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z%]*)\s*$"
)

# The separators an SVG ``viewBox`` may use between its four numbers.
_SVG_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[,\s]+")


def _svg_length(value: str | None, field_name: str) -> tuple[float, str]:
    """Return one SVG length attribute as ``(number, unit)``.

    The unit is ``""`` for a unitless length.

    Raises:
        ValueError: The attribute is absent or is not a length.  SVG authoring is
            encoder-defined, so an unreadable canvas is reported as a mismatch rather than
            assumed.
    """
    if value is None:
        raise ValueError(f"the SVG carries no {field_name!r} attribute")
    match = _SVG_LENGTH.match(value)
    if match is None:
        raise ValueError(f"the SVG {field_name!r} attribute is not a length: {value!r}")
    return float(match.group(1)), match.group(2)


def _svg_canvas(svg_bytes: bytes) -> tuple[float, float, str]:
    """Return the vector canvas as ``(width, height, unit)``.

    ``viewBox`` is preferred when the document has one, because its third and fourth numbers are
    the canvas in the user units the geometry is drawn in; ``width`` and ``height`` are the
    fallback and may carry a physical unit.  Nothing else about the document is inspected:
    element names, attribute sets and byte layout are encoder-defined and AAP §0.3.5 forbids
    asserting them.  The unit is ``""`` for user units.

    Raises:
        ElementTree.ParseError: The bytes are not well-formed XML.
        ValueError: The canvas cannot be read, is not square, or its two sides disagree about
            their unit.
    """
    root = ElementTree.fromstring(svg_bytes)
    view_box = root.get("viewBox")
    if view_box:
        parts = [part for part in _SVG_SEPARATORS.split(view_box.strip()) if part]
        if len(parts) != 4:
            raise ValueError(
                f"the SVG 'viewBox' must hold four numbers; got {view_box!r}"
            )
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            raise ValueError(f"the SVG 'viewBox' is not four numbers: {view_box!r}") from None
        width, height, unit = numbers[2], numbers[3], ""
    else:
        width, unit = _svg_length(root.get("width"), "width")
        height, other_unit = _svg_length(root.get("height"), "height")
        if other_unit != unit:
            raise ValueError(
                f"the SVG canvas mixes units: width in {unit!r}, height in {other_unit!r}"
            )
    if not math.isclose(width, height, rel_tol=0.0, abs_tol=_INTEGRAL_EPSILON):
        raise ValueError(f"the SVG canvas is not square: {width!r} x {height!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"the SVG canvas has no area: {width!r} x {height!r}")
    return width, height, unit


def _oversized_canvas(svg_bytes: bytes) -> float | None:
    """Return the declared canvas side in pixels when it exceeds the raster ceiling.

    A vector artefact is untrusted input and a rasterizer allocates the canvas it is told to
    draw, so the declared side is turned into the pixel count it will cost before any rasterizer
    runs.  A document whose canvas cannot be read, or whose length unit has no absolute extent,
    is left to the rasterizer: this is a resource guard, not a geometry assertion, and the
    geometry belongs to ``RN-SVG``.
    """
    try:
        width, height, unit = _svg_canvas(svg_bytes)
    except (ElementTree.ParseError, ValueError):
        return None
    factor = _UNIT_TO_PX.get(unit)
    if factor is None:
        return None
    side = max(width, height) * factor
    return side if side > _MAX_RASTER_SIDE_PX else None


# The one temporary artefact: the logo overlay bitmap


def _logo_area_fraction(ctx: RunContext, step: int) -> float:
    """Return the symbol-area fraction `step` asks the overlay to cover.

    ``[checks.damage] logo_area_fraction`` is the ceiling and the three steps divide it into
    thirds, so the largest overlay spends exactly the configured value and no number is written
    twice in this module.
    """
    return ctx.config.damage.logo_area_fraction * step / _LOGO_FRACTION_DENOMINATOR


def _logo_side_px(fraction: float) -> int:
    """Return the overlay's pixel side for `fraction` of the symbol's area.

    The symbol's own pixel side is ``(modules + 2 * quiet_zone) * module_size`` and the module
    count is only known once the target has rendered, so the overlay is sized from the
    **smallest** legal Model 2 symbol (Version 1, 21 modules).  Every real symbol is at least that
    large, so the fraction the overlay actually covers can never exceed the configured ceiling,
    whatever version the target chose; the covered fraction is measured from the produced image
    and it is that measurement the assertion reads.  Sizing from an estimate instead could push a
    case past the error-correction budget and fail a conforming target.
    """
    smallest_symbol_px = (
        module_count_for_version(1) + 2 * _PINNED_QUIET_ZONE_MODULES
    ) * _LOGO_MODULE_SIZE_PX
    return max(1, round(math.sqrt(fraction) * smallest_symbol_px))


def _logo_bitmap(seed: int, side_px: int) -> bytes:
    """Return an opaque `side_px` square of seeded noise, encoded as PNG bytes.

    Seeded through :func:`qrcheck.config.seeding.derive_seed` into ``numpy``'s ``PCG64``, so the
    same case always produces the same bitmap and the covered area is re-derivable from the
    recorded seed.  A uniformly random opaque bitmap corrupts roughly half of every module it
    covers, which makes the overlay a genuine error-correction test rather than a cosmetic one.
    The pixels are written as a P6 Netpbm body, which Pillow opens natively, and re-encoded by
    :func:`qrcheck.evidence.mux.encode_png`, so nothing here imports Pillow.

    Raises:
        ValueError: `side_px` is not positive, or the encoded bitmap is empty.
    """
    if side_px < 1:
        raise ValueError(f"side_px must be at least 1; got {side_px}")
    generator = numpy.random.Generator(numpy.random.PCG64(seed))
    pixels = generator.integers(0, 256, size=(side_px, side_px, 3), dtype=numpy.uint8)
    header = f"P6\n{side_px} {side_px}\n255\n".encode("ascii")
    image = decode_image(header + pixels.tobytes())
    try:
        encoded = encode_png(image.convert("RGB"))
    finally:
        image.close()
    if not encoded:
        raise ValueError(f"encoding a {side_px}x{side_px} logo overlay produced no bytes")
    return encoded


@contextlib.contextmanager
def _logo_bitmap_file(bitmap_png: bytes) -> Iterator[Path]:
    """Write `bitmap_png` to a temporary file and remove it when the block ends.

    A request names its overlay by path, so the bitmap has to exist on disk for the duration of
    one ``generate`` call.  The name carries the product-wide ``qrcheck-`` prefix that
    ``CL-ARTIFACTS`` scans ``tempfile.gettempdir()`` for, and the file is removed in a ``finally``
    block, so a target that refuses the request cannot leave an artefact for that check to find.

    Raises:
        OSError: The temporary file cannot be created or written.
    """
    descriptor, raw_path = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=".png")
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(bitmap_png)
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


# Shared per-case plumbing


def _target_config(ctx: RunContext) -> TargetConfig:
    """Return the resolved ``[targets.<name>]`` record for the run's adapter.

    The record owns ``contrast_floor``, the floor ``RN-CONTRAST`` asserts at or above.  A target
    the configuration does not declare -- a stub an ad-hoc test drives directly -- is held to the
    documented default floor rather than to no floor at all, which is the same fallback the
    configuration model declares for a target that states nothing.
    """
    declared = ctx.config.targets.get(ctx.adapter.name)
    if declared is not None:
        return declared
    return TargetConfig(name=ctx.adapter.name, type="reference")


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, a ``time.perf_counter`` reading.

    A monotonic stopwatch fills the recorded ``duration_ms`` only; no assertion in this
    module reads a clock (AAP §0.5.5).

    Args:
        started: The value ``time.perf_counter()`` returned before the case ran.

    Returns:
        The elapsed time in milliseconds, never negative.
    """
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _translate(
    check_id: str,
    case_id: str,
    error: BaseException,
    request: GenerateRequest,
    started: float,
) -> CaseResult:
    """Translate one adapter error through the framework's single translation table.

    :func:`qrcheck.checks.support.case_status_for_error` owns that table: a declared rejection
    fails a rendering check with ``target_rejected``, a contract error with ``contract_error``,
    and an unsupported capability skips it.

    Raises:
        TargetUnavailable: Re-raised unchanged, so the runner can apply its target-loss policy.
        TargetTimeout: Re-raised unchanged, so the runner can apply its target-loss policy.
    """
    return support.case_status_for_error(
        check_id,
        case_id,
        error,
        request=request,
        duration_ms=_elapsed_ms(started),
    )


# RN-QUIET-ZONE


@check(
    id=RN_QUIET_ZONE,
    title="Quiet zone: every rendered margin is at least the requested modules",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.quiet_zone}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def quiet_zone_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the quiet-zone cases: the first twelve corpus records, in ``id`` order.

    Twelve cases, one encode each and no decode at all: the margin is measured from the
    image, so a quiet zone that is too short is reported even on a symbol no decoder can
    read.  The requested zone cycles over :data:`_QUIET_ZONE_CHOICES`, so a target that
    always draws the same zone cannot pass by coincidence.
    """
    sample = support.first_records(ctx.corpus, _SAMPLE_RECORDS)
    positions = {record.id: index for index, record in enumerate(sample)}
    yield from support.corpus_cases(
        ctx,
        RN_QUIET_ZONE,
        sample,
        execute_factory=lambda record: _quiet_zone_case(positions[record.id], record),
    )


def _quiet_zone_case(position: int, record: CorpusRecord) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one quiet-zone case.

    Args:
        position: The record's rank in the sample, which selects the requested zone.
        record: The corpus record the case draws its payload from.

    Returns:
        The case's ``execute`` callable.
    """
    case_id = f"{_CORPUS_CASE_PREFIX}{record.id}"
    requested = _QUIET_ZONE_CHOICES[position % len(_QUIET_ZONE_CHOICES)]
    return lambda ctx: _run_quiet_zone(ctx, case_id, record, requested)


def _run_quiet_zone(
    ctx: RunContext,
    case_id: str,
    record: CorpusRecord,
    requested: int,
) -> CaseResult:
    """Generate one symbol and assert the quiet zone it actually rendered.

    Emits ``sampling_failed`` when the image's module geometry cannot be established,
    ``value_mismatch`` when any side renders less than the requested zone, and the translated
    adapter reasons through :func:`_translate`.  ``decoded`` carries ``quiet_zone_measured``
    beside ``quiet_zone_requested`` and the measured module pitch, which is the evidence AAP
    §0.4.5 requires for the ``missing_quiet_zone`` defect.
    """
    started = time.perf_counter()
    request = support.build(
        record.payload_bytes,
        ctx.ec_level,
        module_size_px=_MODULE_SIZE_CHOICES[0],
        quiet_zone_modules=requested,
    )
    expected: dict[str, object] = {
        "quiet_zone_modules": requested,
        "assertion": "measured_quiet_zone_at_least_requested",
        "decoded": False,
    }
    try:
        symbol = support.generate(ctx, request)
    except (TargetRejected, TargetContractError, TargetUnsupported) as error:
        return _translate(RN_QUIET_ZONE, case_id, error, request, started)

    module_px = measure_module_px(symbol.image_bytes)
    if module_px is None:
        result = support.fail(
            RN_QUIET_ZONE,
            case_id,
            reason=Reason.SAMPLING_FAILED,
            request=request.as_dict(),
            decoded={
                "assertion": "module pitch from the top-left finder pattern",
                "reason": "no finder run could be measured in the artefact",
            },
            expected=expected,
            duration_ms=_elapsed_ms(started),
        )
        return support.record_evidence(
            ctx, result, symbol=symbol, request_extra={"quiet_zone_modules": requested}
        )

    measured = measure_quiet_zone(symbol.image_bytes, module_px)
    width, height = _image_size(symbol.image_bytes)
    short = sorted(
        side
        for side, value in measured.items()
        if value + QUIET_ZONE_TOLERANCE_MODULES < requested
    )
    decoded: dict[str, object] = {
        "image_px": [width, height],
        "module_px_measured": module_px,
        "quiet_zone_measured": dict(measured),
        "quiet_zone_requested": requested,
        "quiet_zone_tolerance_modules": QUIET_ZONE_TOLERANCE_MODULES,
    }
    duration = _elapsed_ms(started)
    if short:
        decoded["short_sides"] = short
        result = support.fail(
            RN_QUIET_ZONE,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    else:
        result = support.pass_(
            RN_QUIET_ZONE,
            case_id,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        request_extra={
            "quiet_zone_modules": requested,
            "quiet_zone_measured": dict(measured),
        },
    )


# RN-MODULE-SIZE


@check(
    id=RN_MODULE_SIZE,
    title="Module size: the rendered module edge matches the requested pixel size",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.module_size}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def module_size_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the module-size cases: the first twelve corpus records, in ``id`` order.

    Twelve cases, one encode and no decode each: the edge is measured from the image, so
    the dimension contract is asserted arithmetically rather than through a decoder.  The
    requested edge cycles over :data:`_MODULE_SIZE_CHOICES`, which is what catches a
    target that renders one fixed scale whatever the request asked for.
    """
    sample = support.first_records(ctx.corpus, _SAMPLE_RECORDS)
    positions = {record.id: index for index, record in enumerate(sample)}
    yield from support.corpus_cases(
        ctx,
        RN_MODULE_SIZE,
        sample,
        execute_factory=lambda record: _module_size_case(positions[record.id], record),
    )


def _module_size_case(
    position: int, record: CorpusRecord
) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one module-size case.

    Args:
        position: The record's rank in the sample, which selects the requested edge.
        record: The corpus record the case draws its payload from.

    Returns:
        The case's ``execute`` callable.
    """
    case_id = f"{_CORPUS_CASE_PREFIX}{record.id}"
    requested = _MODULE_SIZE_CHOICES[position % len(_MODULE_SIZE_CHOICES)]
    return lambda ctx: _run_module_size(ctx, case_id, record, requested)


def _run_module_size(
    ctx: RunContext,
    case_id: str,
    record: CorpusRecord,
    requested: int,
) -> CaseResult:
    """Generate one symbol and assert the module geometry it actually rendered.

    Two arithmetic assertions, both from the image and neither from a decoder: the measured
    module edge is within :data:`MODULE_SIZE_TOLERANCE_PX` of the requested one, and the
    dimension contract ``(modules + 2 * quiet_zone) * module_size`` holds, the canvas counted in
    module pitches being a legal Model 2 module count plus the requested quiet zone on both
    sides.  The second is why a target that renders a short quiet zone fails here as well as in
    ``RN-QUIET-ZONE`` (AAP §0.4.2), and being a legal-count test rather than a comparison against
    a decoder's version it still measures a symbol that cannot be decoded.

    Emits ``sampling_failed`` when the geometry cannot be established and ``value_mismatch``
    when either assertion fails; the measured and requested values travel together in
    ``decoded`` so a reader can see the deviation.
    """
    started = time.perf_counter()
    request = support.build(
        record.payload_bytes,
        ctx.ec_level,
        module_size_px=requested,
        quiet_zone_modules=_PINNED_QUIET_ZONE_MODULES,
    )
    expected: dict[str, object] = {
        "module_size_px": requested,
        "tolerance_px": MODULE_SIZE_TOLERANCE_PX,
        "quiet_zone_modules": _PINNED_QUIET_ZONE_MODULES,
        "assertion": "measured_module_edge_and_dimension_contract",
        "decoded": False,
    }
    try:
        symbol = support.generate(ctx, request)
    except (TargetRejected, TargetContractError, TargetUnsupported) as error:
        return _translate(RN_MODULE_SIZE, case_id, error, request, started)

    module_px = measure_module_px(symbol.image_bytes)
    if module_px is None:
        result = support.fail(
            RN_MODULE_SIZE,
            case_id,
            reason=Reason.SAMPLING_FAILED,
            request=request.as_dict(),
            decoded={
                "assertion": "module pitch from the top-left finder pattern",
                "reason": "no finder run could be measured in the artefact",
            },
            expected=expected,
            duration_ms=_elapsed_ms(started),
        )
        return support.record_evidence(
            ctx, result, symbol=symbol, request_extra={"module_size_px": requested}
        )

    width, height = _image_size(symbol.image_bytes)
    deviation = abs(module_px - requested)
    # The canvas in module pitches, and the module count that leaves once the requested
    # quiet zone is taken off both sides.  A canvas that is not a whole number of pitches
    # is itself a broken dimension, so it is counted before the subtraction is trusted.
    modules_across = int(round(width / module_px))
    implied_module_count = modules_across - 2 * _PINNED_QUIET_ZONE_MODULES
    implied_version = version_from_dimension(implied_module_count)
    decoded: dict[str, object] = {
        "image_px": [width, height],
        "module_px_measured": module_px,
        "module_size_px_requested": requested,
        "module_px_deviation": deviation,
        "module_px_tolerance": MODULE_SIZE_TOLERANCE_PX,
        "modules_across_canvas": modules_across,
        "quiet_zone_modules_requested": _PINNED_QUIET_ZONE_MODULES,
        "implied_module_count": implied_module_count,
        "implied_version": implied_version,
    }
    problems: list[str] = []
    if deviation > MODULE_SIZE_TOLERANCE_PX:
        problems.append("module_edge_outside_tolerance")
    if abs(width - height) > MODULE_SIZE_TOLERANCE_PX:
        problems.append("image_not_square")
    if implied_version is None:
        problems.append("dimension_contract_violated")
    duration = _elapsed_ms(started)
    if problems:
        decoded["problems"] = problems
        result = support.fail(
            RN_MODULE_SIZE,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    else:
        result = support.pass_(
            RN_MODULE_SIZE,
            case_id,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        request_extra={
            "module_size_px": requested,
            "module_px_measured": module_px,
        },
    )


# RN-CONTRAST


@check(
    id=RN_CONTRAST,
    title="Contrast: the rendered palette matches the request and clears the declared floor",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.color}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def contrast_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the contrast cases: six palettes, each at both polarities.

    Twelve cases, one encode and one decode each.  The inverted half is why the check
    exists: the framework owns polarity normalisation (AAP §0.3.2, IM-4), so a
    light-on-dark symbol must decode through the second pass rather than through a
    decoder feature the pinned versions do not have.
    """
    for index, palette in enumerate(_CONTRAST_PALETTES, start=1):
        for polarity in _CONTRAST_POLARITIES:
            case_id = f"{_SYNTHETIC_CASE_PREFIX}contrast-{index}-{polarity}"
            yield support.synthetic_case(
                RN_CONTRAST,
                case_id,
                _contrast_case(case_id, index, palette, polarity),
            )


def _contrast_payload(index: int) -> bytes:
    """Return the payload the `index`-th contrast case requests.

    Lower-case text, so the encoder spends byte mode and the symbol stays the size the palette
    case expects, and a fixed string rather than a random one, so a failing case can be
    reproduced from its identifier alone.
    """
    return f"qrcheck contrast case {index}".encode("utf-8")


def _contrast_case(
    case_id: str,
    index: int,
    palette: tuple[tuple[int, int, int], tuple[int, int, int]],
    polarity: str,
) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one contrast case.

    Args:
        case_id: The case's stable identifier.
        index: The one-based palette number.
        palette: The ``(foreground, background)`` pair for the normal polarity.
        polarity: ``normal`` or ``inverted``.

    Returns:
        The case's ``execute`` callable.
    """
    return lambda ctx: _run_contrast(ctx, case_id, index, palette, polarity)


def _run_contrast(
    ctx: RunContext,
    case_id: str,
    index: int,
    palette: tuple[tuple[int, int, int], tuple[int, int, int]],
    polarity: str,
) -> CaseResult:
    """Generate one coloured symbol and assert what the image actually rendered.

    Three assertions, all measured from the produced image rather than from the request: the
    palette the target drew is the palette that was asked for, the ratio between the two measured
    colours is at or above the target's declared ``contrast_floor``, and the symbol decodes to the
    requested payload -- which for an inverted request means the polarity-normalised second pass
    worked.  Emits ``value_mismatch`` for any of the three and the translated adapter reasons
    through :func:`_translate`.
    """
    started = time.perf_counter()
    if polarity not in _CONTRAST_POLARITIES:
        raise ValueError(f"polarity must be one of {list(_CONTRAST_POLARITIES)}; got {polarity!r}")
    foreground, background = palette
    if polarity == "inverted":
        foreground, background = background, foreground
    request = support.build(
        _contrast_payload(index),
        ctx.ec_level,
        module_size_px=_MODULE_SIZE_CHOICES[0],
        quiet_zone_modules=_PINNED_QUIET_ZONE_MODULES,
        foreground_rgb=foreground,
        background_rgb=background,
    )
    floor = float(_target_config(ctx).contrast_floor)
    expected: dict[str, object] = {
        "foreground_rgb": list(foreground),
        "background_rgb": list(background),
        "contrast_floor": floor,
        "polarity": polarity,
        "assertion": "rendered_palette_and_ratio_and_payload",
    }
    try:
        symbol, decoded_result = support.generate_and_decode(ctx, request)
    except (TargetRejected, TargetContractError, TargetUnsupported) as error:
        return _translate(RN_CONTRAST, case_id, error, request, started)

    measured_dark, measured_light = _measure_colours(symbol.image_bytes)
    ratio = contrast_ratio(measured_dark, measured_light)
    decoded: dict[str, object] = {
        "contrast_ratio_measured": ratio,
        "contrast_floor": floor,
        "palette_measured": {"dark": list(measured_dark), "light": list(measured_light)},
        "palette_requested": {
            "foreground": list(foreground),
            "background": list(background),
        },
        "polarity_requested": polarity,
        "inverted_request": is_inverted_request(request),
    }
    decoded.update(decoded_result.as_dict())
    problems: list[str] = []
    if {measured_dark, measured_light} != {tuple(foreground), tuple(background)}:
        problems.append("palette_not_rendered")
    if ratio < floor:
        problems.append("contrast_below_floor")
    if decoded_result.payload_bytes != request.payload:
        problems.append("payload_not_carried")
    duration = _elapsed_ms(started)
    if problems:
        decoded["problems"] = problems
        result = support.fail(
            RN_CONTRAST,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    else:
        result = support.pass_(
            RN_CONTRAST,
            case_id,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        request_extra={
            "contrast_floor": floor,
            "contrast_ratio_measured": ratio,
            "polarity": polarity,
        },
    )


# RN-SVG


@check(
    id=RN_SVG,
    title="SVG: the vector canvas carries the module count and rasterizes to a decodable symbol",
    family=_FAMILY,
    requires=frozenset({Capability.svg}),
    dependencies=(DEPENDENCY_SVG_RASTERIZER,),
)
def svg_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the SVG cases: the first twelve corpus records, requested as vector output.

    Twelve cases, one encode each and one decode of the rasterized PNG.  Discovery runs
    once here rather than per case, so every case of a run reports the same rasterizer and
    the same attempted order; when nothing was found every case is a skip carrying that
    order, never a pass (AAP §0.3.5, §0.7.2).
    """
    status = discover_svg_rasterizer(ctx.config.render)
    sample = support.first_records(ctx.corpus, _SAMPLE_RECORDS)
    yield from support.corpus_cases(
        ctx,
        RN_SVG,
        sample,
        execute_factory=lambda record: _svg_case(record, status),
    )


def _svg_case(record: CorpusRecord, status: RasterizerStatus) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one SVG case.

    Args:
        record: The corpus record the case draws its payload from.
        status: The run's rasterizer discovery result.

    Returns:
        The case's ``execute`` callable.
    """
    case_id = f"{_CORPUS_CASE_PREFIX}{record.id}"
    return lambda ctx: _run_svg(ctx, case_id, record, status)


def _run_svg(
    ctx: RunContext,
    case_id: str,
    record: CorpusRecord,
    status: RasterizerStatus,
) -> CaseResult:
    """Rasterize one vector artefact and assert the geometry and the symbol it carries.

    Emits ``missing_dependency`` when no rasterizer was discovered, ``value_mismatch`` when the
    document does not parse, its canvas is not a positive square that carries the module count
    at one integral pitch, or the rasterized artefact carries no decodable symbol, and
    ``sampling_failed`` when no module count could be implied.  A rasterizer that fails outright
    raises, so the runner records an ``error`` result with the traceback instead of a verdict
    about the target.

    The payload the rasterized symbol decodes to is recorded rather than asserted: payload
    fidelity is ``RT-ROUNDTRIP``'s assertion, so a defect another check already reports must not
    become a second failure here.
    """
    expected: dict[str, object] = {
        "output_format": "svg",
        "quiet_zone_modules": _SVG_QUIET_ZONE_MODULES,
        "assertion": "canvas_carries_module_count_and_raster_decodes",
        "rasterizer_configured": status.configured,
    }
    if not status.available:
        return support.skip(
            RN_SVG,
            case_id,
            reason=Reason.MISSING_DEPENDENCY,
            expected={
                **expected,
                "rasterizer_attempted": list(status.attempted),
                "rasterizer": status.name,
            },
            decoded={"rasterizer": status.as_dict()},
        )
    started = time.perf_counter()
    request = support.build(
        record.payload_bytes,
        ctx.ec_level,
        output_format="svg",
        quiet_zone_modules=_SVG_QUIET_ZONE_MODULES,
    )
    try:
        symbol = support.generate(ctx, request)
    except (TargetRejected, TargetContractError, TargetUnsupported) as error:
        return _translate(RN_SVG, case_id, error, request, started)

    decoded: dict[str, object] = {
        "output_format": request.output_format,
        "quiet_zone_modules": _SVG_QUIET_ZONE_MODULES,
        "rasterizer": status.as_dict(),
        "svg_bytes": len(symbol.image_bytes),
    }
    expected = {**expected, "rasterizer": status.name}
    try:
        canvas_width, canvas_height, canvas_unit = _svg_canvas(symbol.image_bytes)
    except (ElementTree.ParseError, ValueError) as error:
        decoded["problems"] = ["svg_canvas_unreadable"]
        decoded["svg_error"] = f"{type(error).__name__}: {error}"
        result = support.fail(
            RN_SVG,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=_elapsed_ms(started),
        )
        return support.record_evidence(ctx, result, symbol=symbol)

    decoded["svg_canvas_side"] = canvas_width
    decoded["svg_canvas_unit"] = canvas_unit
    rasterized = rasterize_svg(symbol.image_bytes, status)
    decoded["rasterized_bytes"] = len(rasterized)
    decoded_result = support.decode(ctx, rasterized, request)
    decoded.update(decoded_result.as_dict())
    if decoded_result.version is None:
        decoded["problems"] = ["module_count_not_implied"]
        result = support.fail(
            RN_SVG,
            case_id,
            reason=Reason.SAMPLING_FAILED,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=_elapsed_ms(started),
        )
        return support.record_evidence(ctx, result, symbol=symbol)

    modules = module_count_for_version(decoded_result.version)
    pitch = canvas_width / modules
    decoded["svg_module_count"] = modules
    decoded["svg_module_pitch"] = pitch
    decoded["payload_matches_request"] = decoded_result.payload_bytes == request.payload
    problems: list[str] = []
    if not pitch > 0:
        problems.append("module_pitch_not_positive")
    elif not canvas_unit and abs(pitch - round(pitch)) > _INTEGRAL_EPSILON:
        # A unitless canvas is drawn in plain user units, so a conforming encoder emits a
        # whole number of them per module.  A canvas with a physical unit cannot be held
        # to that -- one module may lawfully be 0.5 mm -- so only the sign is asserted.
        problems.append("module_pitch_not_integral")
    if decoded_result.payload_bytes is None:
        problems.append("raster_does_not_decode")
    duration = _elapsed_ms(started)
    if problems:
        decoded["problems"] = problems
        result = support.fail(
            RN_SVG,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    else:
        result = support.pass_(
            RN_SVG,
            case_id,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
        )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        request_extra={
            "output_format": "svg",
            "rasterizer": status.name,
            "svg_canvas_side": canvas_width,
            "svg_module_pitch": pitch,
        },
    )


# RN-LOGO


@check(
    id=RN_LOGO,
    title="Logo overlay: the composite is applied and survives its error-correction budget",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.logo_overlay, Capability.ec_level}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def logo_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the logo cases: three overlay sizes at each of two error levels.

    Six cases, one encode and one decode each.  The sizes are thirds of
    ``[checks.damage] logo_area_fraction``, so the largest spends exactly the configured
    ceiling, and the two levels are the contrast AAP §0.4.2 requires: the same overlay is
    asserted to decode at H and only recorded at L.
    """
    for level in _LOGO_EC_LEVELS:
        for step in _LOGO_FRACTION_STEPS:
            case_id = f"{_SYNTHETIC_CASE_PREFIX}logo-{step}-{level.lower()}"
            yield support.synthetic_case(
                RN_LOGO,
                case_id,
                _logo_case(case_id, level, step),
            )


def _logo_payload(level: str, step: int) -> bytes:
    """Return the payload one logo case requests.

    Lower-case text padded to :data:`_LOGO_PAYLOAD_BYTES`, so the encoder spends byte mode and
    the symbol lands on the same version at a given level for every run and every machine: the
    covered fraction is measured, but its reproducibility is what makes the recorded evidence
    comparable.
    """
    label = f"qrcheck logo overlay case {level.lower()}{step} "
    return label.ljust(_LOGO_PAYLOAD_BYTES, "x").encode("utf-8")


def _logo_case(case_id: str, level: str, step: int) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one logo case.

    Args:
        case_id: The case's stable identifier.
        level: The requested error-correction level.
        step: The overlay size step, one of :data:`_LOGO_FRACTION_STEPS`.

    Returns:
        The case's ``execute`` callable.
    """
    return lambda ctx: _run_logo(ctx, case_id, level, step)


def _run_logo(ctx: RunContext, case_id: str, level: str, step: int) -> CaseResult:
    """Composite one seeded overlay onto a symbol and assert it survives its budget.

    Asserts that the overlay was really applied -- the rendered image carries colours the
    two-colour symbol did not have, so a declared ``logo_overlay`` capability has behaviour
    behind it -- and, when the covered fraction is at or below
    ``recovery_fraction(decoded level) - occlusion_margin``, that the symbol still decodes to
    the requested payload.  The covered fraction and the outcome of a case that exceeds the
    budget are recorded without assertion: the EC-L overlay is deliberately there to observe,
    as AAP §0.3.5 requires.  The budget reads the error level decoded from the symbol, never the
    requested one, which keeps the ``defective`` target's EC downshift from turning six logo
    cases into six failures of a check that is not about error levels (AAP §0.4.2).

    Emits ``value_mismatch`` for either assertion and the translated adapter reasons through
    :func:`_translate`; evidence is recorded when the case did not pass.
    """
    started = time.perf_counter()
    fraction = _logo_area_fraction(ctx, step)
    seed = derive_seed(ctx.seed, RN_LOGO, case_id, "logo")
    side_px = _logo_side_px(fraction)
    bitmap = _logo_bitmap(seed, side_px)
    margin = ctx.config.damage.occlusion_margin
    expected: dict[str, object] = {
        "error_level": level,
        "logo_area_fraction": fraction,
        "logo_area_fraction_ceiling": ctx.config.damage.logo_area_fraction,
        "logo_side_px": side_px,
        "logo_seed": seed,
        "occlusion_margin": margin,
        "assertion": "decodes_when_covered_area_within_recovery_budget",
    }
    with _logo_bitmap_file(bitmap) as overlay_path:
        request = support.build(
            _logo_payload(level, step),
            level,
            module_size_px=_LOGO_MODULE_SIZE_PX,
            quiet_zone_modules=_PINNED_QUIET_ZONE_MODULES,
            logo_overlay_path=overlay_path,
        )
        try:
            symbol, decoded_result = support.generate_and_decode(ctx, request)
        except (TargetRejected, TargetContractError, TargetUnsupported) as error:
            return _translate(RN_LOGO, case_id, error, request, started)

    width, height = _image_size(symbol.image_bytes)
    module_px = measure_module_px(symbol.image_bytes)
    modules = (
        None
        if decoded_result.version is None
        else module_count_for_version(decoded_result.version)
    )
    covered_image_area = (min(side_px, width) * min(side_px, height)) / float(width * height)
    if modules is not None and module_px:
        # The covered fraction is measured against the symbol's own module area, because
        # that is the area ``logo_area_fraction`` names -- the quiet zone is not part of
        # the symbol.  Measuring against the whole image instead would shrink the fraction
        # by the quiet zone's share, and a target that renders a short margin would then
        # inflate the damage this case believes it is testing.
        symbol_side = modules * module_px
        covered = min(1.0, (side_px / symbol_side) ** 2)
        covered_basis = "symbol_module_area"
    else:
        covered = covered_image_area
        covered_basis = "image_area"
    decoded_ec = (
        decoded_result.ec_level_format or decoded_result.ec_level_decoder or level
    )
    recovery = support.recovery_fraction(decoded_ec)
    budget = recovery - margin
    asserted = covered <= budget
    distinct = _distinct_colours(symbol.image_bytes)
    palette_size = len({tuple(request.foreground_rgb), tuple(request.background_rgb)})
    decoded: dict[str, object] = {
        "assertion_applied": asserted,
        "decoded_ec_level": decoded_ec,
        "error_level_requested": level,
        "logo_area_fraction_basis": covered_basis,
        "logo_area_fraction_measured": covered,
        "logo_area_fraction_measured_image_area": covered_image_area,
        "logo_area_fraction_requested": fraction,
        "logo_side_px": side_px,
        "logo_seed": seed,
        "module_px_measured": module_px,
        "occlusion_margin": margin,
        "recovery_budget": budget,
        "recovery_fraction": recovery,
        "rendered_colour_count": distinct,
        "symbol_module_count": modules,
        "symbol_px": [width, height],
    }
    decoded.update(decoded_result.as_dict())
    problems: list[str] = []
    if distinct <= palette_size:
        problems.append("logo_overlay_not_applied")
    if asserted and decoded_result.payload_bytes != request.payload:
        problems.append("payload_not_carried")
    duration = _elapsed_ms(started)
    projection = request.as_dict()
    # The overlay reached the target by way of a temporary file whose name must not reach
    # an artefact (AAP §0.5.5 forbids absolute paths); the bitmap itself is persisted as
    # the ``logo.png`` evidence artefact, so the projection names that instead.
    projection["logo_overlay_path"] = LOGO_PNG
    if problems:
        decoded["problems"] = problems
        result = support.fail(
            RN_LOGO,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=projection,
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
            seed=seed,
        )
    else:
        result = support.pass_(
            RN_LOGO,
            case_id,
            request=projection,
            decoded=decoded,
            expected=expected,
            duration_ms=duration,
            seed=seed,
        )
    return support.record_evidence(
        ctx,
        result,
        symbol=symbol,
        logo=bitmap,
        request_extra={
            "logo_area_fraction_requested": fraction,
            "logo_area_fraction_measured": covered,
            "logo_seed": seed,
            "logo_side_px": side_px,
            "recovery_budget": budget,
        },
    )
