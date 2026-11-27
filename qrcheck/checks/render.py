"""The render family: the five ``RN-*`` checks and the product's one SVG rasterizer.

Owns the product's only rasterizer discovery (:func:`discover_svg_rasterizer`,
:data:`SVG_RASTERIZER_AUTO_ORDER`) and its only rasterization (:func:`rasterize_svg`), so the tool
``qrcheck doctor`` prints is the tool ``RN-SVG`` uses; the two geometry measurements that need no
decoder (:func:`measure_module_px`, :func:`measure_quiet_zone`), which do not borrow
:mod:`qrcheck.decode.matrix` because a quiet-zone or module-size defect must be visible even when
nothing can decode the symbol; and :data:`EXPECTED_ENCODES`.

Fixed case matrix for one full run: ``RN-QUIET-ZONE`` and ``RN-MODULE-SIZE`` take the first 12
corpus records (12 encodes, 0 decodes each), ``RN-CONTRAST`` 6 palettes x 2 polarities (12 encodes,
12 decodes), ``RN-SVG`` the first 12 corpus records (12 encodes, 12 decodes on the rasterized PNG),
``RN-LOGO`` 3 overlay sizes x EC H and L (6 encodes, 6 decodes) -- 54 cases.

Restates nothing another module owns: contrast and polarity come from
:mod:`qrcheck.decode.consensus`, the evidence layout and seed derivation from
:mod:`qrcheck.checks.support` and :mod:`qrcheck.config.seeding`, request validation from the port's
``build_request``, and every pixel crosses the single image boundary in
:mod:`qrcheck.evidence.mux`. The contrast floor is the target's ``contrast_floor`` and the overlay
ceiling comes from ``[checks.damage]``: no threshold is a literal here.

Reasons emitted: ``sampling_failed`` when the module geometry cannot be established at all;
``value_mismatch`` when a measurement contradicts the request; ``missing_dependency`` when no
rasterizer was discovered, which skips every ``RN-SVG`` case with the attempted order; and
``target_rejected``, ``contract_error`` and ``unsupported_capability`` as
:func:`qrcheck.checks.support.case_status_for_error` translates them. ``TargetUnavailable`` and
``TargetTimeout`` are never converted here: the runner owns the target-loss policy. No clock, no
``hash()``, no ``id()`` reaches an assertion, and the overlay bitmap's only random source is
:func:`qrcheck.config.seeding.derive_seed` into numpy's ``PCG64``.

Decisions this file had to take: ``RN-SVG`` requests a zero-module quiet zone, because a short
margin is ``RN-QUIET-ZONE``'s finding and the ``defective`` target must not fail both; it records
the decoded payload instead of asserting it, since fidelity is ``RT-ROUNDTRIP``'s assertion; ``RN-
MODULE-SIZE`` asserts a legal-module-count dimension contract needing no decoder; ``RN-LOGO`` sizes
its overlay from the smallest legal symbol. The one temporary file written is the logo bitmap, named
with the ``qrcheck-`` prefix ``CL-ARTIFACTS`` scans for and removed in a ``finally`` block.

A vector artefact is a target's bytes and is bounded before the framework spends anything on it: a
document longer than :data:`_MAX_SVG_BYTES`, or one that declares a document type definition, an
entity or an ``<?xml-stylesheet`` instruction, is refused rather than parsed, because the parser
expands an internal entity as it reads (CWE-776) and drops a processing instruction instead of
resolving it, and one whose ``href``, CSS ``url(...)`` or ``@import`` names anything but a
same-document ``#fragment`` -- or that spells such a reference with a CSS escape, the one spelling
a lexical scan cannot see -- is refused before any rasterizer runs, because the tool would
otherwise fetch a host this framework never chose or read a local file. All of those
refusals belong to :func:`_parse_svg`, the one parse path, so no caller can hand a document to a
rasterizer without them. A declared canvas above :data:`_MAX_RASTER_SIDE_PX` per side is then
priced from that parse and reported as ``value_mismatch``, because what an artefact says about
itself is a verdict about the target; only a rasterizer or host fault stays an ``error`` result
, and each of the rasterizer's streams is captured under
:data:`_MAX_CAPTURED_RASTER_BYTES` with only a short tail of it carried into a diagnostic, so
neither a flood nor one long message escapes it.

A raster artefact is bounded the same way before it is read: its header is probed first, and one
declaring more than :data:`_MAX_ARTEFACT_SIDE_PX` per side is reported as ``sampling_failed``
instead of being decoded. A PNG's declared canvas is read from its own IHDR, and a header the image
library itself refuses for its size is reported the same way with no dimensions recorded, so no
artefact a target returns turns a measuring case into an ``error`` result. Each measuring case
opens its artefact once, through one :class:`_ArtefactMeasurements` record, so every arithmetic
measurement that case takes reads the same decoded image.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
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

from qrcheck.capture import BoundedStreamCapture, open_capture_pair
from qrcheck.checks import support
from qrcheck.config.models import (
    DEFAULT_DECODE_IMAGE_PX,
    SVG_RASTERIZERS,
    RenderConfig,
    TargetConfig,
)
from qrcheck.config.seeding import derive_seed
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
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Annotation resolution only: no runtime edge into the corpus builder,
    # and no image library in this module's import graph.
    import numpy

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

# The two case-identifier prefixes a rendered case carries: ``corpus:`` for a corpus-driven case,
# ``synthetic:`` for a constructed one.  ``failure_id`` embeds both in an evidence path.
_CORPUS_CASE_PREFIX: Final[str] = "corpus:"
_SYNTHETIC_CASE_PREFIX: Final[str] = "synthetic:"

# The discovery order: importable ``cairosvg`` first (never a hard dependency), then the four
# command-line rasterizers on ``PATH``.  ``doctor`` and ``RN-SVG`` report the sequence they
# attempted, so reordering it changes a report.
SVG_RASTERIZER_AUTO_ORDER: Final[tuple[str, ...]] = (
    "cairosvg",
    "rsvg-convert",
    "resvg",
    "magick",
    "convert",
)

# The ceiling on one rasterizer invocation.  The configuration sets no timeout for it, so it is a
# backstop against a hung rasterizer rather than a performance budget (that is
# ``[checks.performance]``).
RASTERIZER_TIMEOUT_S: Final[float] = 60.0

# The most this module keeps from either stream of the rasterizer it runs: the cli adapter's bound
# on one stream of a target (``qrcheck.targets.cli._MAX_CAPTURED_STREAM_BYTES``) and the http
# adapter's bound on one response body (``qrcheck.targets.http._MAX_RESPONSE_BODY_BYTES``) -- the
# same 16 MiB, because no legitimate PNG or SVG of a Model 2 symbol approaches any of them. A
# rasterizer reading a target's document is as untrusted as the target is, and a crafted canvas can
# make it emit far more than it declares, so a stream that passes this bound is refused rather than
# buffered.
_MAX_CAPTURED_RASTER_BYTES: Final[int] = 16 * 1024 * 1024

# How many characters of a failed rasterizer's stderr one diagnostic keeps: the cli adapter's
# ``DIAGNOSTIC_TAIL_CHARS``, since a message may not carry a whole captured stream, and the end of
# it is the part that says what went wrong.
_RASTERIZER_DIAGNOSTIC_CHARS: Final[int] = 2000

# How many bytes one read from a captured rasterizer stream asks for: one pipe buffer, so a flood
# costs a bounded number of reads rather than one call per byte.
_RASTERIZER_READ_CHUNK_BYTES: Final[int] = 64 * 1024

# How long a capture waits for its drain thread once the rasterizer has exited: the wait is for the
# bytes already in the pipe, never for the tool, which has already exited.
_RASTERIZER_DRAIN_JOIN_TIMEOUT_S: Final[float] = 5.0

# The largest canvas a rasterizer is asked to render, per side, in pixels: the ceiling the
# decode pipeline applies, from ``[checks.performance] decode_image_px``.  A target's SVG is
# untrusted and ``rsvg-convert`` allocates the canvas it is told to draw.
_MAX_RASTER_SIDE_PX: Final[int] = 8 * DEFAULT_DECODE_IMAGE_PX

# The largest raster artefact this module measures, per side, in pixels: the same per-side ceiling
# :mod:`qrcheck.decode.consensus` applies, derived the same way from the one configuration default
# (``[checks.performance] decode_image_px``) because that module owns the pixel ceiling for the
# decode path.  A target's PNG is the same untrusted bytes whether it is decoded or measured, so it
# is priced from the image header and refused before any pixel is allocated.  The two values are
# equal by construction, and neither is written as a literal.
_MAX_ARTEFACT_SIDE_PX: Final[int] = 8 * DEFAULT_DECODE_IMAGE_PX

# The largest byte length of a target-supplied SVG this module will parse.  A target artefact is
# untrusted twice over: a rasterizer allocates the canvas it is told to draw, and the XML parser
# expands every internal entity as it reads, so a few hundred bytes can become megabytes of text
# before any geometry guard runs (CWE-776).  The length is therefore bounded before the parse.  A
# conforming vector canvas of at most ``_MAX_RASTER_SIDE_PX`` per side is a few kilobytes of path
# data whatever its scale, because the path is drawn with a transform, so this ceiling is generous.
_MAX_SVG_BYTES: Final[int] = 4 * 1024 * 1024

# The XML declarations a target's SVG may not carry.  An internal entity is expanded while the
# document is read, and a document type definition is what gives it something to expand; a
# ``<?xml-stylesheet`` instruction names a resource a renderer would fetch, and the parser drops a
# processing instruction rather than resolving it, so no scan of the parsed tree can see the
# reference in it and it has to be refused from the raw bytes like the other two.  Each is matched
# before the parse, in every spelling a document the XML parser accepts can carry it in -- the
# single-byte form ASCII, Latin-1 and UTF-8 share, and UTF-16 in both byte orders -- because a scan
# of the raw bytes that knew only the single-byte spelling would let a UTF-16 document through to
# the parser with its declaration intact.
_REFUSED_XML_DECLARATIONS: Final[tuple[str, ...]] = ("<!DOCTYPE", "<!ENTITY", "<?xml-stylesheet")


def _declaration_spellings(name: str) -> tuple[bytes, ...]:
    """Return the lower-cased byte spellings of one refused XML declaration.

    One spelling per encoding the parser can decode: the single-byte one (ASCII, Latin-1 and UTF-8
    agree on these characters) and UTF-16 in each byte order. ``bytes.lower()`` folds the ASCII
    letters inside a UTF-16 stream as well, so one lower-cased scan sees all three.
    """
    folded = name.lower()
    return (folded.encode("ascii"), folded.encode("utf-16-le"), folded.encode("utf-16-be"))


_REFUSED_XML_DECLARATION_SPELLINGS: Final[tuple[tuple[str, tuple[bytes, ...]], ...]] = tuple(
    (name, _declaration_spellings(name)) for name in _REFUSED_XML_DECLARATIONS
)

# CSS pixels per SVG length unit at the 96 dpi every rasterizer defaults to; a unit outside this
# table (a percentage, say) has no absolute extent, so no ceiling applies.
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

# The module-edge tolerance ``RN-MODULE-SIZE`` allows, in pixels: a measurement rounded to whole
# pixels can be off by one on a conforming render, and two would hide a real geometry defect.
MODULE_SIZE_TOLERANCE_PX: Final[float] = 1.0

# A quarter of a module of slack on the measured quiet zone: the margin is measured in whole
# pixels and divided by an estimated pitch, so a conforming four-module zone can read as a hair
# under four.  A quarter cannot hide a missing module.
QUIET_ZONE_TOLERANCE_MODULES: Final[float] = 0.25

# The fixed samples this family takes: the first twelve corpus records in ``id`` order.  A sample
# size, not a threshold.
_SAMPLE_RECORDS: Final[int] = 12

# The geometry each corpus-driven check cycles over its sample, so a target that ignores the
# field cannot pass.  Case parameters, never thresholds.
_MODULE_SIZE_CHOICES: Final[tuple[int, ...]] = (4, 6, 8)
_QUIET_ZONE_CHOICES: Final[tuple[int, ...]] = (4, 5, 6)

# The quiet zone ``RN-MODULE-SIZE`` and ``RN-LOGO`` request, in modules: ISO/IEC 18004 §7.3.6
# requires four, and pinning it makes the dimension contract and the overlay area decidable
# before the symbol exists.
_PINNED_QUIET_ZONE_MODULES: Final[int] = 4

# The quiet zone ``RN-SVG`` requests: none.  The canvas is asserted to carry the module count
# times one pitch, which is decidable only when the canvas is exactly the symbol; pinning four
# would make a *short* margin fail this check too, which is ``RN-QUIET-ZONE``'s finding.
_SVG_QUIET_ZONE_MODULES: Final[int] = 0

# The payload length ``RN-LOGO`` requests, in bytes, excluding the case's own label.  A case
# parameter: at this length the symbol is Version 5 or above, where the overlay is spread over
# interleaved codeword blocks, and the EC-L cases stay above
# ``recovery_fraction("L") - occlusion_margin`` so they are recorded rather than asserted.
_LOGO_PAYLOAD_BYTES: Final[int] = 40

# The module edge ``RN-LOGO`` requests, in pixels.  The overlay fraction is
# scale-invariant in this value, so it only decides how large the bitmap under test is.
_LOGO_MODULE_SIZE_PX: Final[int] = 4

# The three overlay sizes as fractions of the symbol's *area*, so the largest step spends exactly
# ``[checks.damage] logo_area_fraction`` and no value is written twice.
_LOGO_FRACTION_STEPS: Final[tuple[int, ...]] = (1, 2, 3)
_LOGO_FRACTION_DENOMINATOR: Final[int] = 3

# The two error levels ``RN-LOGO`` contrasts: H has spare correction capacity, L almost none, so
# the same overlay is asserted to decode at one and only recorded at the other.
_LOGO_EC_LEVELS: Final[tuple[str, str]] = ("H", "L")

# The six colour pairs ``RN-CONTRAST`` measures.  Case parameters, each measured at or above the
# configured floor the target declares in ``contrast_floor``; written out so a diff shows a
# changed palette.
_CONTRAST_PALETTES: Final[tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] = (
    ((0, 0, 0), (255, 255, 255)),
    ((32, 32, 32), (255, 255, 255)),
    ((0, 64, 0), (255, 255, 255)),
    ((0, 0, 128), (255, 255, 255)),
    ((128, 0, 0), (255, 255, 255)),
    ((0, 0, 0), (204, 204, 204)),
)

# The two polarities each palette is requested at.  ``inverted`` exercises the framework's
# polarity-normalised second pass: zxing-cpp 2.2.0 has no switch for it and OpenCV 4.10 cannot
# detect a light-on-dark symbol at all.
_CONTRAST_POLARITIES: Final[tuple[str, ...]] = ("normal", "inverted")

# The prefix of the one temporary file this module writes: the product-wide ``qrcheck-`` that
# ``CL-ARTIFACTS`` scans ``tempfile.gettempdir()`` for.
_TEMP_PREFIX: Final[str] = "qrcheck-logo-"

# How many rows and columns beside the top-left finder are probed for its dark run: the outer
# ring is one module thick, so the longest reading is the finder's.
_FINDER_PROBE_LINES: Final[int] = 3

# ISO/IEC 18004 §7.3.3: a finder pattern is a 7x7 module square, so its dark run is seven modules
# wide and the pitch is that run divided by seven.
_FINDER_MODULES: Final[int] = 7

# The finder pattern's middle-row (and middle-column) signature, in modules: 1 dark, 1 light, 3
# dark, 1 light, 1 dark, spanning the seven modules that confirm the measured pitch.
_FINDER_SIGNATURE: Final[tuple[int, ...]] = (1, 1, 3, 1, 1)

# How far a measured run may sit from its expected width, in modules, before the line is not a
# finder pattern: half a module absorbs pixel rounding, and a run outside it is a different object
# rather than a noisier one.
_FINDER_RUN_TOLERANCE: Final[float] = 0.5

# The spread below which no threshold separates modules from background: sixteen of the 255 grey
# levels, below which the "dark" mask measures antialiasing.
_MIN_GEOMETRY_SPREAD: Final[float] = 16.0

# A tolerance for comparing a pitch that should be a whole number of user units.  The
# value is a float-comparison epsilon, not a threshold on a symbol.
_INTEGRAL_EPSILON: Final[float] = 1e-6

# The per-check adapter-invocation counts: one ``generate`` per case and no second call.  Twelve
# cases each for ``RN-QUIET-ZONE``, ``RN-MODULE-SIZE``, ``RN-CONTRAST`` and ``RN-SVG``, six for
# ``RN-LOGO``, so the totals are the case counts, and ``tests/unit/test_checks_negative.py``
# asserts the observed count against this table.
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

    ``configured`` echoes the configured value verbatim, so a report distinguishes "asked for
    ``cairosvg`` and it was absent" from "asked for ``auto`` and found ``rsvg-convert``";
    ``attempted`` is the discovery order actually probed -- one entry for an explicit name, every
    name tried up to the one that answered for ``auto``, and empty for ``none``, which probes
    nothing. ``name`` is the resolved tool, ``None`` when nothing was found.
    """

    configured: str
    name: str | None
    available: bool
    attempted: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe projection ``doctor --json`` and the evidence carry.

        ``attempted`` becomes a list because JSON has no tuple, and the keys keep a fixed order.
        """
        return {
            "configured": self.configured,
            "name": self.name,
            "available": self.available,
            "attempted": list(self.attempted),
        }


def _configured_rasterizer(value: RenderConfig | str) -> str:
    """Return the configured rasterizer name from what the caller passes.

    The ``[render]`` record is the documented argument; the string form is accepted too, so a
    caller that already holds the name need not build a record and both discovery call sites
    cannot disagree.

    Raises:
        ValueError: The value is not one of the declared rasterizers -- possible only for a value
            constructed by hand, since the configuration schema rejects it at load time.
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
    ``PATH``; ``find_spec`` and ``shutil.which`` neither import nor run anything, so discovery
    never executes a third-party binary to find out whether it exists.
    """
    if name == "cairosvg":
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A missing parent package raises ImportError and an absent ``__spec__`` ValueError;
            # both mean "not usable", and neither may escape a probe.
            return False
    return shutil.which(name) is not None


def discover_svg_rasterizer(value: RenderConfig | str) -> RasterizerStatus:
    """Resolve the configured SVG rasterizer against what this host provides.

    The product's only discovery implementation, read by both ``doctor`` and ``RN-SVG``, so the tool
    the run reports and the tool the check uses are one value, and a host with no rasterizer
    degrades ``RN-SVG`` to ``skipped``/``missing_dependency``. An explicit value is honoured exactly
    -- a silent substitution would make ``doctor`` report a tool the configuration never named --
    and ``auto`` probes :data:`SVG_RASTERIZER_AUTO_ORDER` in order, stopping at the first hit.

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

    The predicate form of :func:`discover_svg_rasterizer`, a projection of the same discovery.
    """
    return discover_svg_rasterizer(value).available


# The names other modules may reach this discovery by: aliases of the one function, never a
# second implementation.
discover_rasterizer = discover_svg_rasterizer
svg_rasterizer_status = discover_svg_rasterizer


def _rasterizer_command(name: str) -> tuple[str, ...]:
    """Return the argv that rasterizes SVG on stdin to PNG on stdout for `name`.

    Every entry is an argument list, never a shell string: subprocesses here are argv-only with
    ``shell=False``, and a rasterizer is third-party code reading an untrusted artefact. ``-`` is
    each tool's own spelling for stdin, so the argv carries no shell metacharacter and no
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


def _raise_for_overbound_stream(
    name: str,
    argv: tuple[str, ...],
    *captures: BoundedStreamCapture,
) -> None:
    """Refuse a rasterizer stream that passed the documented byte bound.

    The decision is taken before the exit status and the empty-output checks, so a tool that flooded
    one stream is refused for that alone: its output is unusable whatever it exited with, which is
    the rule the cli adapter applies to a target's stream.

    Raises:
        RuntimeError: One stream delivered more than :data:`_MAX_CAPTURED_RASTER_BYTES`. The message
            names the stream, the ceiling it passed and how much arrived.
    """
    for capture in captures:
        if not capture.overflowed:
            continue
        raise RuntimeError(
            f"SVG rasterizer {name!r} wrote more than the {_MAX_CAPTURED_RASTER_BYTES}-byte "
            f"ceiling to its {capture.name} ({capture.received} bytes arrived), so its output is "
            f"unusable: command {list(argv)}"
        )


def _diagnostic_tail(capture: BoundedStreamCapture) -> str:
    """Return at most the last :data:`_RASTERIZER_DIAGNOSTIC_CHARS` of a captured stream, decoded.

    A failure message quotes what the tool said, and a tool writing close to the capture ceiling
    would otherwise put megabytes of stderr into one diagnostic.
    """
    return capture.bytes[-_RASTERIZER_DIAGNOSTIC_CHARS:].decode("utf-8", "replace")


def rasterize_svg(
    svg_bytes: bytes,
    status: RasterizerStatus,
    canvas: tuple[float, float, str] | None = None,
) -> bytes:
    """Rasterize an SVG artefact to PNG bytes with the discovered rasterizer.

    The product's only rasterization: ``cairosvg`` is driven through its module when discovery found
    one, and every other rasterizer is an argv-only subprocess reading the SVG on stdin and writing
    the PNG to stdout, with :data:`RASTERIZER_TIMEOUT_S` as the backstop. Nothing is written to
    disk. Both of the child's streams are captured through
    :func:`~qrcheck.capture.open_capture_pair`, so no document can make the tool grow this process
    past :data:`_MAX_CAPTURED_RASTER_BYTES` per stream, and a failure quotes only the tail of what
    the tool said.

    The document is read through :func:`_parse_svg` on every call, whichever rasterizer answers, so
    no caller can hand a document to third-party code without the byte bound, the declaration
    refusal and the same-document reference allow-list running first. ``canvas`` is the
    ``(width, height, unit)`` the caller already parsed from `svg_bytes` with :func:`_svg_canvas`,
    so this call need not read it out of the tree again; ``None`` takes it from this call's own
    parse, and a canvas that cannot be read is left to the rasterizer, since that is a geometry
    question the tool answers.

    Raises:
        RuntimeError: No rasterizer is available, one of its streams passed the capture ceiling, the
            configured one emitted nothing, exited non-zero, timed out, could not be executed or
            could not have its output captured; the message names the command, its exit status and
            the tail of its stderr, and the runner records an ``error`` result rather than a verdict
            about the target. A declared canvas above :data:`_MAX_RASTER_SIDE_PX` and a refused
            external reference are the two RuntimeErrors a target's own artefact can provoke, and
            both are defence in depth: a check prices the canvas and scans the references it parsed
            before calling here and reports that case as ``value_mismatch``, so either raise reaches
            only a caller that skipped the case-level pricing and it is then a framework fault.
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
    try:
        root = _parse_svg(payload)
    except (ElementTree.ParseError, ValueError) as error:
        raise RuntimeError(
            f"the SVG artefact cannot be rasterized: {type(error).__name__}: {error}"
        ) from None
    declared_side_px: float | None = None
    if canvas is not None:
        declared_side_px = _canvas_side_px(*canvas)
    else:
        with contextlib.suppress(ValueError):
            declared_side_px = _canvas_side_px(*_canvas_of(root))
    if declared_side_px is not None and declared_side_px > _MAX_RASTER_SIDE_PX:
        raise RuntimeError(
            f"the SVG declares a canvas of {declared_side_px:g} px per side, above the "
            f"{_MAX_RASTER_SIDE_PX} px per-side ceiling this framework rasterizes: a "
            f"target's vector artefact is untrusted and the rasterizer allocates it"
        )
    name = status.name
    if name == "cairosvg":
        return _rasterize_with_cairosvg(payload)
    argv = _rasterizer_command(name)
    try:
        stdout_capture, stderr_capture = open_capture_pair(
            limit=_MAX_CAPTURED_RASTER_BYTES,
            chunk_bytes=_RASTERIZER_READ_CHUNK_BYTES,
            join_timeout_s=_RASTERIZER_DRAIN_JOIN_TIMEOUT_S,
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"SVG rasterizer {name!r} was not run: its output could not be captured "
            f"({type(exc).__name__}: {exc}); command {list(argv)}"
        ) from None
    try:
        try:
            completed = subprocess.run(
                list(argv),
                input=payload,
                shell=False,
                stdout=stdout_capture.write_fd,
                stderr=stderr_capture.write_fd,
                timeout=RASTERIZER_TIMEOUT_S,
                check=False,
            )
        finally:
            # Both pipes are released on every path -- success, timeout, spawn failure -- before any
            # error is converted, so neither a descriptor nor a drain thread outlives this call.
            stdout_capture.finish()
            stderr_capture.finish()
    except subprocess.TimeoutExpired:
        _raise_for_overbound_stream(name, argv, stdout_capture, stderr_capture)
        raise RuntimeError(
            f"SVG rasterizer {name!r} did not finish within {RASTERIZER_TIMEOUT_S} s: "
            f"command {list(argv)}"
        ) from None
    except OSError as exc:
        # Only a spawn failure reaches here: subprocess.run raises OSError from execvp alone, before
        # any of the child's output is read.
        _raise_for_overbound_stream(name, argv, stdout_capture, stderr_capture)
        raise RuntimeError(
            f"SVG rasterizer {name!r} could not be executed: command {list(argv)} failed "
            f"with {type(exc).__name__}: {exc}"
        ) from None
    _raise_for_overbound_stream(name, argv, stdout_capture, stderr_capture)
    if completed.returncode != 0:
        raise RuntimeError(
            f"SVG rasterizer {name!r} failed: command {list(argv)} exited "
            f"{completed.returncode} and reported {_diagnostic_tail(stderr_capture)!r}"
        )
    if not stdout_capture.bytes:
        raise RuntimeError(
            f"SVG rasterizer {name!r} produced no output: command {list(argv)} exited 0 "
            f"and wrote nothing to stdout, stderr was {_diagnostic_tail(stderr_capture)!r}"
        )
    return stdout_capture.bytes


def _accepts_url_fetcher(svg2png: object) -> bool:
    """Report whether a ``cairosvg`` module's ``svg2png`` takes a ``url_fetcher`` keyword.

    ``cairosvg`` is never a dependency, so the installed version is whatever the host provides and
    the keyword is offered only where the callable accepts it: a ``**kwargs`` catch-all counts, a
    parameter of that name counts unless it is positional-only, and a callable whose signature
    cannot be read counts as not accepting it. An older or a wrapped ``cairosvg`` therefore still
    runs, and is still protected by the document's own reference scan.
    """
    try:
        parameters = inspect.signature(svg2png).parameters
    except (TypeError, ValueError):
        # A C callable or an object without ``__signature__`` has none to read; passing the keyword
        # anyway could turn a working rasterizer into a refused argument, so it is not passed.
        return False
    named = parameters.get("url_fetcher")
    if named is not None:
        return named.kind is not inspect.Parameter.POSITIONAL_ONLY
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _refuse_url(url: str, *args: object, **kwargs: object) -> bytes:
    """Refuse every URL a ``cairosvg`` call asks for: the restricted fetcher itself.

    :func:`_parse_svg` has already refused every reference outside the document, so reaching here
    means the module tried to resolve something that scan did not see, and refusing keeps the
    rasterizer inside the document either way. The trailing parameters absorb whatever
    the caller passes after the URL.

    Raises:
        _ExternalReferenceRefused: For every URL, unconditionally.
    """
    del args, kwargs
    raise _ExternalReferenceRefused(
        f"the SVG artefact made the cairosvg rasterizer fetch {url!r}: a rasterizer may not reach "
        f"a host this framework never chose or read a local file"
    )


def _rasterize_with_cairosvg(svg_bytes: bytes) -> bytes:
    """Rasterize `svg_bytes` through an importable ``cairosvg`` module.

    ``cairosvg`` is never a dependency: it is discovered at run time from the host's environment,
    so the import happens on the one path that needs it. Where the module's ``svg2png`` takes a
    ``url_fetcher``, a fetcher that refuses every URL is passed, because the module's own default
    resolves whatever the document names; a version whose ``svg2png`` does not take the keyword is
    called as it always was.

    Raises:
        RuntimeError: The module cannot be imported, refused the artefact, tried to fetch a URL or
            produced nothing -- the same vocabulary a failing binary is reported in.
    """
    try:
        module = importlib.import_module("cairosvg")
        svg2png = module.svg2png
        arguments: dict[str, object] = {"bytestring": svg_bytes}
        if _accepts_url_fetcher(svg2png):
            arguments["url_fetcher"] = _refuse_url
        rasterized = svg2png(**arguments)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # The five shapes a rasterizer module can report: a failed import, a native library
        # that cannot open the input, its own error type, a refused argument, its own refusal.
        # Each is a rasterizer fault, never a verdict about the target, so all become RuntimeError.
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


class _ArtefactTooLarge(ValueError):
    """A target artefact whose declared canvas this framework will not measure.

    A named member of the :exc:`ValueError` family rather than a bare exception, so a measuring case
    catches exactly this refusal and every other fault still escapes: the artefact is a target's own
    bytes, and refusing to measure it is a verdict about that target rather than a framework defect.
    ``width`` and ``height`` are the canvas the header declares, which is what the case records as
    its evidence, or ``None`` for both when the declaration is one the image library refuses to open
    for its size -- the dimensions of such a canvas are never read out, so none is invented for it.
    """

    def __init__(self, width: int | None, height: int | None) -> None:
        """Hold the refused canvas' declared dimensions, in pixels, when they could be read."""
        self.width = width
        self.height = height
        if width is None or height is None:
            declared = "declares a canvas the image library itself refuses to open for its size"
        else:
            declared = (
                f"declares {width}x{height} pixels, above the {_MAX_ARTEFACT_SIDE_PX} px per-side "
                f"ceiling this framework measures"
            )
        super().__init__(
            f"the artefact {declared}: a target artefact is untrusted and every measurement of it "
            f"would allocate the canvas"
        )

    @property
    def declared_px(self) -> tuple[int, int] | None:
        """Return the ``(width, height)`` the header declared, or ``None`` when it declared none."""
        if self.width is None or self.height is None:
            return None
        return self.width, self.height

    @property
    def longer_side_px(self) -> int | None:
        """Return the artefact's longer declared side in pixels, or ``None`` when it has none."""
        declared = self.declared_px
        return None if declared is None else max(declared)


def _transparent_as_page(image: Any, pixels: numpy.ndarray) -> numpy.ndarray:
    """Return `pixels` with every fully transparent pixel set to the page's light level.

    A transparent background shows the page rather than the colour its bands store, so it measures
    as white; a half-transparent edge pixel is a real blend and is left as measured. One rule, and
    the same one for the grey and the RGB reduction.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    import numpy

    if "A" in image.getbands():
        alpha = numpy.array(image.getchannel("A"), dtype=numpy.uint8, copy=True)
        pixels[alpha == 0] = 255
    return pixels


class _ArtefactMeasurements:
    """One target artefact, opened once, with every measurement derived at most once.

    Every ``RN-*`` assertion is arithmetic over the same produced image -- its pixel size, the grey
    levels the module geometry is measured in, the RGB pixels the palette is read from, the module
    pitch, the quiet zone at that pitch, the darkest and lightest colours and the distinct-colour
    count -- so all of them are derived here from the one image :func:`_measure_artefact` opened,
    rather than from a fresh open and conversion per measurement. Each value is computed on first
    use and kept, so a case that asks twice pays once, and a record is usable only inside the
    ``with`` block that created it: its image is closed when that block ends.

    The reductions are Pillow's own ``convert("L")`` and ``convert("RGB")`` -- the ITU-R BT.601
    weighting the decoder path also reads -- so this module states no luminance formula, and
    transparency is read as the page on both (see :func:`_transparent_as_page`).

    ``image`` is the loaded image :func:`qrcheck.evidence.mux.decode_image` returns. Its type is
    left to Pillow because this module never imports it: :mod:`qrcheck.evidence.mux` is the
    product's only Pillow importer, and every pixel here crosses that boundary once, when the record
    is opened.
    """

    __slots__ = ("_image", "_derived")

    def __init__(self, image: Any) -> None:
        """Hold `image` for measurement; nothing is computed until a measurement asks for it."""
        self._image = image
        # A derived value is kept under its own key, and ``None`` is a real answer here -- an
        # artefact with no measurable geometry has no pitch -- so presence in the mapping, never
        # truth, decides whether a value has already been computed.
        self._derived: dict[object, Any] = {}

    @property
    def size(self) -> tuple[int, int]:
        """Return the artefact's ``(width, height)`` in pixels."""
        if "size" not in self._derived:
            width, height = self._image.size
            self._derived["size"] = (int(width), int(height))
        return self._derived["size"]

    @property
    def grey(self) -> numpy.ndarray:
        """Return the artefact as an ``H x W`` ``uint8`` grey array."""
        # Deferred: importing this module starts no image stack in a process that only lists checks.
        import numpy

        if "grey" not in self._derived:
            levels = numpy.array(self._image.convert("L"), dtype=numpy.uint8, copy=True)
            self._derived["grey"] = _transparent_as_page(self._image, levels)
        return self._derived["grey"]

    @property
    def rgb(self) -> numpy.ndarray:
        """Return the artefact as an ``H x W x 3`` ``uint8`` RGB array."""
        # Deferred: importing this module starts no image stack in a process that only lists checks.
        import numpy

        if "rgb" not in self._derived:
            pixels = numpy.array(self._image.convert("RGB"), dtype=numpy.uint8, copy=True)
            self._derived["rgb"] = _transparent_as_page(self._image, pixels)
        return self._derived["rgb"]

    @property
    def dark(self) -> numpy.ndarray | None:
        """Return the dark-module mask of the grey array, or ``None`` when there is none."""
        if "dark" not in self._derived:
            self._derived["dark"] = _dark_mask(self.grey)
        return self._derived["dark"]

    @property
    def module_px(self) -> float | None:
        """Return the module edge in pixels, or ``None`` when the geometry cannot be established."""
        if "module_px" not in self._derived:
            dark = self.dark
            self._derived["module_px"] = None if dark is None else _module_pitch(dark)
        return self._derived["module_px"]

    def quiet_zone(self, module_px: float) -> dict[str, float]:
        """Return the rendered quiet zone on all four sides, in modules, at `module_px`.

        Raises:
            ValueError: `module_px` is not positive, the artefact has no dark/light separation, or
                it holds no dark pixel at all.
        """
        key = ("quiet_zone", module_px)
        if key not in self._derived:
            dark = self.dark
            if dark is None:
                raise ValueError(
                    "the artefact has no dark/light separation to measure a quiet zone in"
                )
            self._derived[key] = _quiet_zone_from(dark, module_px)
        return dict(self._derived[key])

    @property
    def colours(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return the artefact's darkest and lightest rendered colours.

        Every pixel sharing the minimum channel sum is averaged for the dark reading and every pixel
        sharing the maximum for the light one, so a single antialiased outlier cannot move either
        value. The channel sum only orders the pixels -- it is a clustering key, never a luminance,
        and the ratio itself is :func:`qrcheck.decode.consensus.contrast_ratio`'s -- and both
        readings are equal when the image is one flat colour.
        """
        # Deferred: importing this module starts no image stack in a process that only lists checks.
        import numpy

        if "colours" not in self._derived:
            pixels = self.rgb.reshape(-1, 3).astype(numpy.int32)
            key = pixels.sum(axis=1)
            self._derived["colours"] = (
                _mean_colour(pixels[key == key.min()]),
                _mean_colour(pixels[key == key.max()]),
            )
        return self._derived["colours"]

    @property
    def distinct_colours(self) -> int:
        """Return how many distinct colours the artefact holds.

        A two-colour symbol renders exactly its foreground and background; one carrying a composited
        overlay carries the overlay's colours as well. That difference is how ``RN-LOGO`` observes
        that the ``logo_overlay`` capability has behaviour behind it.
        """
        # Deferred: importing this module starts no image stack in a process that only lists checks.
        import numpy

        if "distinct_colours" not in self._derived:
            self._derived["distinct_colours"] = int(
                numpy.unique(self.rgb.reshape(-1, 3), axis=0).shape[0]
            )
        return self._derived["distinct_colours"]


@contextlib.contextmanager
def _measure_artefact(image_bytes: bytes) -> Iterator[_ArtefactMeasurements]:
    """Open one target artefact for measurement and close it when the block ends.

    The header is probed first, so an artefact that declares more than :data:`_MAX_ARTEFACT_SIDE_PX`
    per side is refused before a pixel is allocated -- the ceiling :mod:`qrcheck.decode.consensus`
    applies to the same untrusted bytes on the decode path. The image itself is then opened exactly
    once, through :func:`qrcheck.evidence.mux.decode_image` -- the product's one image boundary --
    and closed in a ``finally`` block, so every measurement a case takes reads one decoded image and
    no handle outlives the case. A caller that needs a single measurement reaches
    :func:`measure_module_px` or :func:`measure_quiet_zone`, which open a record of their own.

    Raises:
        _ArtefactTooLarge: The header declares a canvas above the per-side ceiling, or declares one
            the image library refuses to open for its size. Either is a fact about the target's own
            output, which every measuring case reports as ``sampling_failed``.
        UnidentifiedImageError: Pillow cannot recognise the bytes; the adapters convert that into
            ``TargetContractError`` first, so a raise here means a framework artefact.
        OSError: The header is truncated or otherwise broken.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.evidence.mux import DeclaredCanvasTooLarge
    from qrcheck.evidence.mux import decode_image
    from qrcheck.evidence.mux import probe_image_size

    try:
        width, height = probe_image_size(image_bytes)
    except DeclaredCanvasTooLarge as error:
        # A PNG is priced from its own IHDR, above, without the library being consulted; any other
        # format is priced by the library's header probe, and a canvas the library refuses for its
        # size is the same fact about the same untrusted bytes, so it is refused the same way
        # instead of escaping the check as a framework fault.  The refused dimensions were never
        # read out, so the refusal carries none rather than an invented pair.
        raise _ArtefactTooLarge(None, None) from error
    if max(width, height) > _MAX_ARTEFACT_SIDE_PX:
        raise _ArtefactTooLarge(width, height)
    image = decode_image(image_bytes)
    try:
        yield _ArtefactMeasurements(image)
    finally:
        image.close()


def _dark_mask(levels: numpy.ndarray) -> numpy.ndarray | None:
    """Return the boolean dark-module mask of `levels`, or ``None`` when there is none.

    The threshold is the midpoint of the artefact's own darkest and lightest grey level, which needs
    no contrast assumption and is stable for any palette. An artefact whose levels differ by less
    than :data:`_MIN_GEOMETRY_SPREAD` has no such threshold, so the geometry is reported as
    unestablished rather than measured from noise.
    """
    darkest = float(levels.min())
    lightest = float(levels.max())
    if lightest - darkest < _MIN_GEOMETRY_SPREAD:
        return None
    return levels < (darkest + lightest) / 2.0


def _leading_dark_run(line: numpy.ndarray, start: int) -> int:
    """Return the length of the dark run beginning at index `start` of a 1-D line.

    Zero when `start` is light or out of range.
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

    The alternation is implied by the QR finder pattern itself. A line whose first value at `start`
    is light yields no runs, because a finder pattern begins at its outer ring, and fewer than
    `count` runs come back when the line ends first.
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

    ISO/IEC 18004 §7.3.3: across a finder pattern's middle row or column the five runs are 1, 1, 3,
    1 and 1 modules wide, so their total is exactly seven modules and each run confirms the pitch it
    implies. That confirmation is what stops a measurement being taken from something that merely
    looks dark -- a run half a module off its expected width means this is not a finder pattern, and
    ``None`` is then the honest answer.
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

    ISO/IEC 18004 §7.3.3: a finder pattern is a 7x7 module square, so the dark run along its top
    edge and down its left edge is exactly seven modules wide. The top-left finder sits at the
    corner of the dark bounding box, which is why the search starts at the first dark row and
    column: no decoder, corner quadrangle or version assumption is needed, so the measurement
    works on a symbol nothing can decode. The top edge gives a rough pitch that locates the
    finder's middle row and column, and the returned pitch is measured across those, where the full
    1:1:3:1:1 signature is confirmed by :func:`_finder_pitch`; both axes are averaged.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    import numpy

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

    The dark bounding box is the rectangle the modules occupy, so the rows above its first dark row,
    the rows below its last and the columns beside its edges are the quiet zone as rendered. A
    bottom or right margin is a lower bound on the zone drawn -- the last module row may itself end
    in light modules -- which is the safe direction for an assertion of *at least* the requested
    zone.

    Raises:
        ValueError: `module_px` is not positive, or the artefact holds no dark pixel at all.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    import numpy

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

    The public form of one :class:`_ArtefactMeasurements` record: the artefact is opened once for
    this call and closed before it returns. ``None`` is the honest answer when the geometry cannot
    be established, and the checks report ``sampling_failed`` for it rather than a verdict built on
    a guess.

    Raises:
        _ArtefactTooLarge: The header declares a canvas this framework refuses to measure, either
            above :data:`_MAX_ARTEFACT_SIDE_PX` per side or one the image library refuses to open
            for its size, so the artefact is not measured at all.
        UnidentifiedImageError: Pillow cannot recognise the bytes.
    """
    with _measure_artefact(image_bytes) as artefact:
        return artefact.module_px


def measure_quiet_zone(image_bytes: bytes, module_px: float) -> dict[str, float]:
    """Return the artefact's rendered quiet zone on all four sides, in modules.

    The public form of one :class:`_ArtefactMeasurements` record for one measurement: the artefact
    is opened once for this call and closed before it returns.

    Raises:
        ValueError: `module_px` is not positive, the artefact has no measurable geometry, or it
            holds no dark pixel at all.
        _ArtefactTooLarge: The header declares a canvas this framework refuses to measure, either
            above :data:`_MAX_ARTEFACT_SIDE_PX` per side or one the image library refuses to open
            for its size, so the artefact is not measured at all.
        UnidentifiedImageError: Pillow cannot recognise the bytes.
    """
    with _measure_artefact(image_bytes) as artefact:
        return artefact.quiet_zone(module_px)


def _mean_colour(group: numpy.ndarray) -> tuple[int, int, int]:
    """Return the rounded per-channel mean of an ``N x 3`` integer array of pixels."""
    means = group.mean(axis=0)
    return (
        int(round(float(means[0]))),
        int(round(float(means[1]))),
        int(round(float(means[2]))),
    )


# The vector canvas

# A length as SVG writes it: a number with an optional CSS unit suffix, which is how
# ``width="148"`` and ``width="37mm"`` are both expressed.
_SVG_LENGTH: Final[re.Pattern[str]] = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z%]*)\s*$"
)

# The separators an SVG ``viewBox`` may use between its four numbers.
_SVG_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[,\s]+")

# The only reference form a vector canvas may carry: a same-document fragment, which names an
# element of the artefact itself -- ``href="#gradient"``, ``fill="url(#gradient)"``.  A conforming
# encoder emits nothing else, so this allow-list costs no legitimate artefact while closing the sink
# a rasterizer would otherwise be: handed ``<image xlink:href="http://host/x.png">`` or
# ``fill="url(file:///etc/passwd)"``, the tool fetches a host this framework never chose or reads a
# local file, and the raster it produces is persisted as evidence and inlined into ``index.html``.
# That breaks the loopback-only guarantee with this product's own binary as the client.
_FRAGMENT_PREFIX: Final[str] = "#"

# The character a CSS escape sequence is written with, and the reason one anywhere in a document is
# refused rather than scanned. CSS spells any character as a backslash and its code point, so both
# ``fill:u\72 l(http://host/x)`` and ``fill:\75 rl(http://host/x)`` are ``url(http://host/x)``
# to a renderer while neither carries the text ``url(`` a lexical scan of the value looks for; a
# presentation attribute such as ``fill="\75 rl(...)"`` is a CSS property value in SVG, so the same
# hole exists with no ``style`` attribute at all. A conforming encoder emits no backslash
# anywhere -- segno's own SVG output carries none -- so refusing a document that carries one costs
# no legitimate artefact, and judging the escape instead would mean re-implementing the renderer's
# own value parser inside this scan.
_CSS_ESCAPE_MARKER: Final[str] = "\\"

# Every CSS ``url(...)`` in an attribute value or in a style element's text, quoted or not, so the
# scan sees the same reference however the document spells it.  Matched case-insensitively, because
# a CSS function name is case-insensitive and ``URL(x)`` names the same resource as ``url(x)``.
_CSS_URL_REFERENCE: Final[re.Pattern[str]] = re.compile(r"url\(\s*([^)]*)\)", re.IGNORECASE)

# An ``@import`` directive, which fetches a whole stylesheet rather than naming an element: refused
# wherever it appears, because there is no same-document form of it to allow.
_CSS_IMPORT_DIRECTIVE: Final[re.Pattern[str]] = re.compile(r"@import\b", re.IGNORECASE)

# The quote characters a CSS ``url(...)`` may wrap its target in, stripped before the target is
# judged, so ``url("http://host/x")`` and ``url(http://host/x)`` are one reference.
_REFERENCE_QUOTES: Final[str] = "\"'"


def _svg_length(value: str | None, field_name: str) -> tuple[float, str]:
    """Return one SVG length attribute as ``(number, unit)``, the unit empty when absent.

    Raises:
        ValueError: The attribute is absent or is not a length. SVG authoring is encoder-defined, so
            an unreadable canvas is reported as a mismatch rather than assumed.
    """
    if value is None:
        raise ValueError(f"the SVG carries no {field_name!r} attribute")
    match = _SVG_LENGTH.match(value)
    if match is None:
        raise ValueError(f"the SVG {field_name!r} attribute is not a length: {value!r}")
    return float(match.group(1)), match.group(2)


def _refused_declaration(svg_bytes: bytes) -> str | None:
    """Return the XML declaration in `svg_bytes` this module refuses, or ``None`` for a clean one.

    The scan is case-insensitive, encoding-aware and runs before the parse, because an internal
    entity is expanded while the document is read: the point of the refusal is to spend neither the
    memory nor the time on the expansion, so a check after the parse would be too late. A
    ``<?xml-stylesheet`` instruction is refused here for the opposite reason -- the parser drops a
    processing instruction rather than resolving it, so the tree the reference scan walks never
    carries the resource it names. Every encoding the parser can decode is covered by
    :data:`_REFUSED_XML_DECLARATION_SPELLINGS`, so a document cannot carry its declaration past this
    scan by choosing UTF-16 over the single-byte form. A declaration spelled inside a comment or a
    CDATA section is refused as well -- a conforming encoder emits none of them, so the conservative
    reading costs no legitimate artefact.
    """
    lowered = svg_bytes.lower()
    found = sorted(
        name
        for name, spellings in _REFUSED_XML_DECLARATION_SPELLINGS
        if any(spelling in lowered for spelling in spellings)
    )
    return found[0] if found else None


class _ExternalReferenceRefused(ValueError):
    """A vector artefact that references a resource outside the document itself.

    A named member of the :exc:`ValueError` family, beside :class:`_ArtefactTooLarge`, so the
    ``RN-SVG`` case can tell this refusal -- a verdict about the target's own artefact -- from a
    canvas it simply could not read, while every other fault still escapes. The document it refuses
    is never rasterized: a rasterizer handed an external reference fetches a host this framework
    never chose or reads a local file, and the raster it produces would be persisted as evidence and
    inlined into ``index.html``.
    """


def _is_same_document_fragment(reference: str) -> bool:
    """Report whether `reference` names an element of the document that carries it.

    The one form allowed: a leading ``#``, optionally after whitespace, exactly as
    ``href="#gradient"`` and ``url(#gradient)`` write it. Everything else -- an ``http`` or
    ``https`` URL, a ``file:`` URI, a ``data:`` URI, a bare relative or absolute name, a
    protocol-relative ``//host/x`` -- names a resource outside the artefact.
    """
    return reference.lstrip().startswith(_FRAGMENT_PREFIX)


def _refused_css_url(reference_text: str) -> str | None:
    """Return the target of the first CSS ``url(...)`` in `reference_text` that leaves the document.

    `reference_text` is an attribute value or a style element's text, so one rule covers
    ``style="fill:url(...)"`` and ``<style>path{fill:url(...)}</style>`` alike. Each target is
    stripped of its surrounding whitespace and quotes before it is judged, so the quoted and bare
    spellings of one reference are refused the same way, and an empty one is not a reference.
    """
    for match in _CSS_URL_REFERENCE.finditer(reference_text):
        target = match.group(1).strip().strip(_REFERENCE_QUOTES).strip()
        if target and not _is_same_document_fragment(target):
            return target
    return None


def _refused_reference(root: ElementTree.Element) -> tuple[str, str] | None:
    """Return the first reference `root` carries that leaves the document, or ``None`` for none.

    The pair is the reference's form and the value the case reports, so the refusal names what the
    artefact tried to reach. Three forms make a rasterizer leave the document, and all three are
    judged over the parsed tree rather than over the raw bytes, so a reference a namespace prefix or
    a character reference spells differently is seen as the parser reads it: an ``href`` attribute
    -- the expanded ``xlink:href`` included -- a CSS ``url(...)`` in any attribute value, and an
    ``@import`` directive in any attribute value or style element's text, which has no
    same-document form at all. An empty ``href`` is no reference and is skipped, and a style
    element is matched by its expanded name, so the document's own prefix is irrelevant.

    Every attribute value and every style element's text is refused when it carries
    :data:`_CSS_ESCAPE_MARKER`, before any of those forms is looked for. A CSS escape sequence
    spells any character as a backslash and its code point, so ``u\\72 l(http://host/x)`` and
    ``\\75 rl(http://host/x)`` are one ``url(http://host/x)`` to a renderer while neither carries
    the text this scan matches -- and a presentation attribute is a CSS property value in SVG, so a
    ``fill="\\75 rl(...)"`` hides the same reference without a ``style`` attribute at all. The
    escape is refused rather than resolved, because resolving it here would be a second
    implementation of the renderer's own value parser; a conforming encoder emits no backslash, so
    no legitimate artefact is refused.
    """
    for element in root.iter():
        for name, value in element.attrib.items():
            if _CSS_ESCAPE_MARKER in value:
                return ("a CSS escape sequence", value)
            if (name == "href" or name.endswith("}href")) and value.strip():
                if not _is_same_document_fragment(value):
                    return ("an 'href' reference", value)
            if _CSS_IMPORT_DIRECTIVE.search(value) is not None:
                return ("a CSS '@import' directive", value)
            target = _refused_css_url(value)
            if target is not None:
                return ("a CSS 'url(...)' reference", target)
        if element.tag == "style" or str(element.tag).endswith("}style"):
            text = "".join(element.itertext())
            if _CSS_ESCAPE_MARKER in text:
                return ("a CSS escape sequence", text.strip())
            if _CSS_IMPORT_DIRECTIVE.search(text) is not None:
                return ("a CSS '@import' directive", text.strip())
            target = _refused_css_url(text)
            if target is not None:
                return ("a CSS 'url(...)' reference", target)
    return None


def _parse_svg(svg_bytes: bytes) -> ElementTree.Element:
    """Return the parsed document of a vector artefact, bounded and reference-checked first.

    The one parse path every reader of a target's SVG takes, so no caller can hand a document to a
    rasterizer without the reference scan. The refusals run in the cheap order: the byte length,
    then the refused XML declarations, then -- after the parse -- any reference that is not a
    same-document fragment.

    Raises:
        ElementTree.ParseError: The bytes are not well-formed XML.
        _ExternalReferenceRefused: The document references a resource outside itself.
        ValueError: The artefact is longer than :data:`_MAX_SVG_BYTES`, or it carries a
            :data:`_REFUSED_XML_DECLARATIONS` declaration.
    """
    payload = bytes(svg_bytes)
    if len(payload) > _MAX_SVG_BYTES:
        raise ValueError(
            f"the SVG artefact is {len(payload)} bytes, above the {_MAX_SVG_BYTES}-byte ceiling "
            f"this framework parses: a target's vector artefact is untrusted and is bounded before "
            f"it is read"
        )
    declaration = _refused_declaration(payload)
    if declaration is not None:
        raise ValueError(
            f"the SVG artefact declares {declaration}, which this framework refuses to parse: an "
            f"internal entity is expanded while the document is read, and a stylesheet instruction "
            f"names a resource the rasterizer would fetch, before any canvas guard applies, so a "
            f"target's vector artefact may not carry one"
        )
    root = ElementTree.fromstring(payload)
    refused = _refused_reference(root)
    if refused is not None:
        form, reference = refused
        raise _ExternalReferenceRefused(
            f"the SVG artefact carries {form} {reference!r}, which names a resource outside the "
            f"document: a rasterizer handed a reference that leaves it fetches a host this "
            f"framework never chose or reads a local file, and a conforming encoder emits only "
            f"same-document '{_FRAGMENT_PREFIX}' fragments"
        )
    return root


def _svg_canvas(svg_bytes: bytes) -> tuple[float, float, str]:
    """Return the vector canvas as ``(width, height, unit)``.

    ``viewBox`` is preferred when the document has one, because its third and fourth numbers are the
    canvas in the user units the geometry is drawn in; ``width`` and ``height`` are the fallback.
    Nothing else about the document is inspected: element names, attribute sets and byte layout are
    encoder-defined and none of them is asserted.

    The artefact is untrusted, so it is read through :func:`_parse_svg` and not around it: a
    document longer than :data:`_MAX_SVG_BYTES`, or one that declares a ``<!DOCTYPE``, an
    ``<!ENTITY`` or an ``<?xml-stylesheet`` instruction, is refused rather than parsed -- an
    internal entity is expanded as the document is read, before any geometry a caller could check
    (CWE-776), and a stylesheet instruction is one the parser drops, so no scan of the tree would
    see it -- and one that references anything outside itself is refused before any rasterizer sees
    it (:class:`_ExternalReferenceRefused`).

    Raises:
        ElementTree.ParseError: The bytes are not well-formed XML.
        _ExternalReferenceRefused: The document references a resource outside itself.
        ValueError: The canvas cannot be read, is not square, or its sides disagree about their
            unit; the artefact is longer than :data:`_MAX_SVG_BYTES`; or it carries a
            :data:`_REFUSED_XML_DECLARATIONS` declaration.
    """
    return _canvas_of(_parse_svg(svg_bytes))


def _canvas_of(root: ElementTree.Element) -> tuple[float, float, str]:
    """Return the canvas `root` declares as ``(width, height, unit)``.

    The parsing half of :func:`_svg_canvas`, split from the byte-level refusals so a caller that
    already holds the parsed document prices its canvas without reading the artefact again.

    Raises:
        ValueError: The canvas cannot be read, is not square, or its sides disagree about their
            unit. SVG authoring is encoder-defined, so an unreadable canvas is reported as a
            mismatch rather than assumed.
    """
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


def _canvas_side_px(width: float, height: float, unit: str) -> float | None:
    """Return the canvas' longer side in pixels, or ``None`` when its unit has no absolute extent.

    A vector artefact is untrusted input and a rasterizer allocates the canvas it is told to draw,
    so the declared side is priced in pixels before any rasterizer runs. This is the pure arithmetic
    half of that guard: a caller that has already parsed the document hands the canvas in, so the
    guard never reads it out of the tree a second time. A unit outside :data:`_UNIT_TO_PX` (a
    percentage, say) has no absolute extent and is left to the rasterizer -- this is a resource
    guard, not a geometry assertion.
    """
    factor = _UNIT_TO_PX.get(unit)
    if factor is None:
        return None
    return max(width, height) * factor


# The one temporary artefact: the logo overlay bitmap


def _logo_area_fraction(ctx: RunContext, step: int) -> float:
    """Return the symbol-area fraction `step` asks the overlay to cover.

    ``[checks.damage] logo_area_fraction`` is the ceiling and the three steps divide it into thirds,
    so the largest overlay spends exactly the configured value.
    """
    return ctx.config.damage.logo_area_fraction * step / _LOGO_FRACTION_DENOMINATOR


def _logo_side_px(fraction: float) -> int:
    """Return the overlay's pixel side for `fraction` of the symbol's area.

    The symbol's pixel side is ``(modules + 2 * quiet_zone) * module_size`` and the module count is
    known only after the target rendered, so the overlay is sized from the **smallest** legal Model
    2 symbol (Version 1, 21 modules). Every real symbol is at least that large, so the fraction the
    overlay covers can never exceed the configured ceiling whatever version the target chose, and
    the covered fraction is measured from the produced image. Sizing from an estimate instead could
    push a case past the error-correction budget and fail a conforming target.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.decode.matrix import module_count_for_version

    smallest_symbol_px = (
        module_count_for_version(1) + 2 * _PINNED_QUIET_ZONE_MODULES
    ) * _LOGO_MODULE_SIZE_PX
    return max(1, round(math.sqrt(fraction) * smallest_symbol_px))


def _logo_bitmap(seed: int, side_px: int) -> bytes:
    """Return an opaque `side_px` square of seeded noise, encoded as PNG bytes.

    Seeded through :func:`qrcheck.config.seeding.derive_seed` into ``numpy``'s ``PCG64``, so the
    same case always produces the same bitmap and the covered area is re-derivable from the recorded
    seed. A uniformly random opaque bitmap corrupts roughly half of every module it covers, which
    makes the overlay a genuine error-correction test. The pixels are written as a P6 Netpbm body,
    which Pillow opens natively, and re-encoded by :func:`qrcheck.evidence.mux.encode_png`.

    Raises:
        ValueError: `side_px` is not positive, or the encoded bitmap is empty.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.evidence.mux import decode_image
    from qrcheck.evidence.mux import encode_png
    import numpy

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

    A request names its overlay by path, so the bitmap has to exist for the duration of one
    ``generate`` call. The name carries the ``qrcheck-`` prefix ``CL-ARTIFACTS`` scans
    ``tempfile.gettempdir()`` for, and the file is removed in a ``finally`` block, so a target that
    refuses the request cannot leave an artefact for that check to find.

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

    The record owns ``contrast_floor``, the floor ``RN-CONTRAST`` asserts at or above. A target the
    configuration does not declare is held to the documented default floor rather than to no floor
    at all, the same fallback the configuration model declares.
    """
    declared = ctx.config.targets.get(ctx.adapter.name)
    if declared is not None:
        return declared
    return TargetConfig(name=ctx.adapter.name, type="reference")


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, a ``time.perf_counter`` reading.

    A monotonic stopwatch fills the recorded ``duration_ms`` only; no assertion here reads a clock.
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

    :func:`qrcheck.checks.support.case_status_for_error` owns that table: a declared rejection fails
    a rendering check with ``target_rejected``, a contract error with ``contract_error``, and an
    unsupported capability skips it.

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


def _unmeasurable(
    ctx: RunContext,
    check_id: str,
    case_id: str,
    request: Mapping[str, object],
    expected: Mapping[str, object],
    symbol: GeneratedSymbol,
    error: _ArtefactTooLarge,
    started: float,
    request_extra: Mapping[str, object],
    seed: int | None = None,
    *,
    logo: bytes | None = None,
) -> CaseResult:
    """Return the documented "this artefact cannot be measured" outcome for one case.

    A target artefact whose header declares a canvas this framework refuses to measure -- one above
    :data:`_MAX_ARTEFACT_SIDE_PX` per side, or one the image library itself refuses to open for its
    size -- is a fact about the target's own output, not a framework fault, so it is reported the
    way the ``module_px is None`` branch already reports an unmeasurable artefact:
    ``sampling_failed``, with the declared side and the ceiling in ``decoded`` and the symbol
    persisted as evidence. The mismatched measurement is neither asserted nor called a
    ``value_mismatch``, because no pixel was read to contradict the request. A refusal the image
    library made never reads the dimensions out, so ``artefact_px`` and ``artefact_side_px`` are
    then recorded as ``null`` rather than guessed at. `logo` is the overlay bitmap ``RN-LOGO``
    composited, which that check persists as ``logo.png`` for every case it records, so a case it
    could not measure keeps the artefact set the evidence layout documents.

    Raises:
        TypeError: A mapping holds a value the result model rejects.
        ValueError: An identifier, a duration or a seed is outside its documented range.
    """
    declared = error.declared_px
    decoded: dict[str, object] = {
        "assertion": "artefact_measurable",
        "reason": "artefact_canvas_above_measurement_ceiling",
        "artefact_px": None if declared is None else list(declared),
        "artefact_side_px": error.longer_side_px,
        "artefact_side_ceiling_px": _MAX_ARTEFACT_SIDE_PX,
    }
    result = support.fail(
        check_id,
        case_id,
        reason=Reason.SAMPLING_FAILED,
        request=request,
        decoded=decoded,
        expected=expected,
        duration_ms=_elapsed_ms(started),
        seed=seed,
    )
    return support.record_evidence(
        ctx, result, symbol=symbol, logo=logo, request_extra=request_extra
    )


# RN-QUIET-ZONE


@check(
    id=RN_QUIET_ZONE,
    title="Quiet zone: every rendered margin is at least the requested modules",
    family=_FAMILY,
    requires=frozenset({Capability.png, Capability.quiet_zone}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def quiet_zone_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the quiet-zone cases: the first twelve corpus records, in ``id`` order.

    Twelve cases, one encode each and no decode: the margin is measured from the image, so a short
    quiet zone is reported even on a symbol no decoder can read. The requested zone cycles over
    :data:`_QUIET_ZONE_CHOICES`, so a target that always draws the same zone cannot pass by
    coincidence.
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

    `position` is the record's rank in the sample, which selects the requested zone; `record` is the
    corpus record the case draws its payload from.
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
    adapter reasons through :func:`_translate`. ``decoded`` carries ``quiet_zone_measured`` beside
    ``quiet_zone_requested`` and the module pitch, which is what a reader needs to diagnose the
    ``missing_quiet_zone`` defect from the report alone.
    """
    started = time.perf_counter()
    # The module size is deliberately left unset, so the request asks for the one thing this
    # check is about: the pitch it measures is whatever the target chose, and pinning an edge
    # would make the request need a capability this check does not declare.
    request = support.build(
        record.payload_bytes,
        ctx.ec_level,
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

    try:
        with _measure_artefact(symbol.image_bytes) as artefact:
            module_px = artefact.module_px
            measured = None if module_px is None else artefact.quiet_zone(module_px)
            width, height = artefact.size
    except _ArtefactTooLarge as error:
        return _unmeasurable(
            ctx,
            RN_QUIET_ZONE,
            case_id,
            request.as_dict(),
            expected,
            symbol,
            error,
            started,
            {"quiet_zone_modules": requested},
        )
    if module_px is None or measured is None:
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
    # The request pins the quiet zone as well as the edge: the dimension contract is stated in the
    # request's own quiet zone, so honouring the pin needs the capability and this declaration must
    # say so, or a target declaring only ``module_size`` is resolved as runnable and then skips
    # every case at the pre-I/O gate.
    requires=frozenset({Capability.png, Capability.module_size, Capability.quiet_zone}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def module_size_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the module-size cases: the first twelve corpus records, in ``id`` order.

    Twelve cases, one encode and no decode each: the edge is measured from the image, so the
    dimension contract is asserted arithmetically rather than through a decoder. The requested edge
    cycles over :data:`_MODULE_SIZE_CHOICES`, which catches a target that renders one fixed scale.
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

    `position` is the record's rank in the sample, which selects the requested edge; `record` is the
    corpus record the case draws its payload from.
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

    Two arithmetic assertions, both from the image: the measured module edge is within
    :data:`MODULE_SIZE_TOLERANCE_PX` of the requested one, and the dimension contract
    ``(modules + 2 * quiet_zone) * module_size`` holds, the canvas counted in module pitches being a
    legal Model 2 module count plus the requested quiet zone on both sides. The second is why a
    short quiet zone fails here as well as in ``RN-QUIET-ZONE``, and being a legal-count test rather
    than a comparison against a decoder's version it still measures a symbol that cannot be
    decoded. Emits ``sampling_failed`` when the geometry cannot be established and
    ``value_mismatch`` when either assertion fails.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.decode.matrix import version_from_dimension

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

    try:
        with _measure_artefact(symbol.image_bytes) as artefact:
            module_px = artefact.module_px
            width, height = artefact.size
    except _ArtefactTooLarge as error:
        return _unmeasurable(
            ctx,
            RN_MODULE_SIZE,
            case_id,
            request.as_dict(),
            expected,
            symbol,
            error,
            started,
            {"module_size_px": requested},
        )
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

    deviation = abs(module_px - requested)
    # The canvas in module pitches, and the module count left once the requested quiet zone is
    # taken off both sides; a canvas that is not a whole number of pitches is already broken.
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

    Twelve cases, one encode and one decode each. The inverted half is why the check exists: the
    framework owns polarity normalisation through a second decode pass, so a light-on-dark symbol
    must decode that way rather than through a decoder feature the pinned versions lack.
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

    Lower-case text, so the encoder spends byte mode and the symbol stays the size the palette case
    expects, and a fixed string, so a failing case can be reproduced from its identifier alone.
    """
    return f"qrcheck contrast case {index}".encode("utf-8")


def _contrast_case(
    case_id: str,
    index: int,
    palette: tuple[tuple[int, int, int], tuple[int, int, int]],
    polarity: str,
) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one contrast case.

    `case_id` is the case's stable identifier, `index` the one-based palette number, `palette` the
    ``(foreground, background)`` pair for the normal polarity and `polarity` ``normal`` or
    ``inverted``.
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

    Three assertions, all measured from the produced image rather than the request: the palette
    drawn is the palette asked for, the ratio between the two measured colours is at or above the
    target's declared ``contrast_floor``, and the symbol decodes to the requested payload -- which
    for an inverted request means the polarity-normalised second pass worked. Emits
    ``value_mismatch`` for any of the three and the translated adapter reasons through
    :func:`_translate`.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.decode.consensus import contrast_ratio
    from qrcheck.decode.consensus import is_inverted_request

    started = time.perf_counter()
    if polarity not in _CONTRAST_POLARITIES:
        raise ValueError(f"polarity must be one of {list(_CONTRAST_POLARITIES)}; got {polarity!r}")
    foreground, background = palette
    if polarity == "inverted":
        foreground, background = background, foreground
    # The colours are the subject of this check, so the geometry is left at the target's own
    # defaults: pinning a module size or a quiet zone would make the request need capabilities
    # this check does not declare, and nothing here measures either.
    request = support.build(
        _contrast_payload(index),
        ctx.ec_level,
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

    try:
        with _measure_artefact(symbol.image_bytes) as artefact:
            measured_dark, measured_light = artefact.colours
    except _ArtefactTooLarge as error:
        return _unmeasurable(
            ctx,
            RN_CONTRAST,
            case_id,
            request.as_dict(),
            expected,
            symbol,
            error,
            started,
            {"contrast_floor": floor, "polarity": polarity},
        )
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
    # The request pins a zero-module quiet zone, which is what lets the canvas imply the module
    # count at one integral pitch, so the capability is genuinely needed: declared here, a target
    # that cannot honour the pin is resolved as skipped instead of skipping case by case.
    requires=frozenset({Capability.svg, Capability.quiet_zone}),
    dependencies=(DEPENDENCY_SVG_RASTERIZER,),
    corpus_driven=True,
)
def svg_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the SVG cases: the first twelve corpus records, requested as vector output.

    Twelve cases, one encode each and one decode of the rasterized PNG. Discovery runs once here
    rather than per case, so every case reports the same rasterizer and attempted order; when
    nothing was found every case is a skip carrying that order, never a pass.
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

    `record` is the corpus record the case draws its payload from and `status` the run's rasterizer
    discovery result.
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
    document does not parse or is refused before the parse, references anything outside itself, its
    declared canvas is above :data:`_MAX_RASTER_SIDE_PX` per side, its canvas is not a positive
    square carrying the module count at one integral pitch, or the raster carries no decodable
    symbol, and ``sampling_failed`` when no module count could be implied. What a target's own
    artefact says is a verdict, so the declared canvas is priced and the document's references are
    checked here from the one parse of it: an oversized canvas is ``value_mismatch`` with
    ``canvas_above_raster_ceiling``, an unreadable canvas ``svg_canvas_unreadable``, and a reference
    that leaves the document ``svg_external_reference``, each leaving the run's exit code at 1 and
    refusing the rasterization entirely. A rasterizer or host fault -- a
    binary that exits non-zero, times out or writes nothing -- raises, and the runner records an
    ``error`` result, which is exit 3. The decoded payload is recorded rather than asserted:
    fidelity is ``RT-ROUNDTRIP``'s assertion.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.decode.matrix import module_count_for_version

    expected: dict[str, object] = {
        "output_format": "svg",
        "quiet_zone_modules": _SVG_QUIET_ZONE_MODULES,
        "assertion": "canvas_carries_module_count_and_raster_decodes",
        "rasterizer_configured": status.configured,
        "svg_canvas_ceiling_px": _MAX_RASTER_SIDE_PX,
    }
    if not status.available:
        # A skipped-with-reason case is persisted like a failure, so the per-case half of the
        # rasterizer gate leaves the same evidence the whole-check skip does rather than only a
        # record in summary.json.
        return support.record_evidence(
            ctx,
            support.skip(
                RN_SVG,
                case_id,
                reason=Reason.MISSING_DEPENDENCY,
                expected={
                    **expected,
                    "rasterizer_attempted": list(status.attempted),
                    "rasterizer": status.name,
                },
                decoded={"rasterizer": status.as_dict()},
            ),
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
        # A reference to anything outside the document is the target's own artefact trying to send
        # this framework's rasterizer to a host or a file of its choosing, so it is a verdict about
        # the target with its own problem token -- exit 1 -- and not the framework fault an
        # unreadable canvas is priced as.
        decoded["problems"] = [
            "svg_external_reference"
            if isinstance(error, _ExternalReferenceRefused)
            else "svg_canvas_unreadable"
        ]
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
    declared_side_px = _canvas_side_px(canvas_width, canvas_height, canvas_unit)
    if declared_side_px is not None and declared_side_px > _MAX_RASTER_SIDE_PX:
        # The declared canvas is what the target's own artefact says about itself, so a canvas no
        # rasterizer may allocate is a verdict about the target -- exit 1 -- and never the framework
        # error exit 3 reserves. The priced side, its unit and the ceiling are the evidence.
        decoded["svg_canvas_side_px"] = declared_side_px
        decoded["svg_canvas_ceiling_px"] = _MAX_RASTER_SIDE_PX
        decoded["problems"] = ["canvas_above_raster_ceiling"]
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

    rasterized = rasterize_svg(
        symbol.image_bytes, status, canvas=(canvas_width, canvas_height, canvas_unit)
    )
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
        # A unitless canvas is drawn in plain user units, so a conforming encoder emits a whole
        # number per module; one with a physical unit may lawfully be 0.5 mm, so only the sign is
        # asserted.
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
    # The request pins a module size and the overlay is sized from it before the target is
    # called, so a target that cannot render at that edge cannot serve this check. The quiet zone
    # is deliberately not pinned -- the covered fraction is measured on the symbol's own module
    # area -- so the capability is not part of what this check asks for.
    requires=frozenset(
        {Capability.png, Capability.logo_overlay, Capability.ec_level, Capability.module_size}
    ),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def logo_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the logo cases: three overlay sizes at each of two error levels.

    Six cases, one encode and one decode each. The sizes are thirds of ``[checks.damage]
    logo_area_fraction``, so the largest spends exactly the configured ceiling, and the two levels
    are the contrast this check draws: the same overlay is asserted to decode at H and only recorded
    at L.
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

    Lower-case text padded to :data:`_LOGO_PAYLOAD_BYTES`, so the encoder spends byte mode and the
    symbol lands on the same version at a given level on every run: the covered fraction is
    measured, and its reproducibility is what makes the recorded evidence comparable.
    """
    label = f"qrcheck logo overlay case {level.lower()}{step} "
    return label.ljust(_LOGO_PAYLOAD_BYTES, "x").encode("utf-8")


def _logo_case(case_id: str, level: str, step: int) -> Callable[[RunContext], CaseResult]:
    """Return the callable that runs one logo case.

    `case_id` is the case's stable identifier, `level` the requested error-correction level and
    `step` the overlay size step, one of :data:`_LOGO_FRACTION_STEPS`.
    """
    return lambda ctx: _run_logo(ctx, case_id, level, step)


def _run_logo(ctx: RunContext, case_id: str, level: str, step: int) -> CaseResult:
    """Composite one seeded overlay onto a symbol and assert it survives its budget.

    Asserts that the overlay was really applied -- the rendered image carries colours the two-colour
    symbol did not have, so a declared ``logo_overlay`` capability has behaviour behind it -- and,
    when the covered fraction is at or below ``recovery_fraction(decoded level) -
    occlusion_margin``, that the symbol still decodes to the requested payload. A case above the
    budget is recorded without a payload assertion, because the covered area may exceed what the
    level can correct, and the budget reads the level decoded from the symbol, never the requested
    one. Emits ``value_mismatch`` for either assertion.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.decode.matrix import module_count_for_version

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
            logo_overlay_path=overlay_path,
        )
        try:
            symbol, decoded_result = support.generate_and_decode(ctx, request)
        except (TargetRejected, TargetContractError, TargetUnsupported) as error:
            return _translate(RN_LOGO, case_id, error, request, started)

    projection = request.as_dict()
    # The overlay reached the target through a temporary file whose name must not reach an artefact
    # (an absolute path may never reach an evidence file); the bitmap is persisted as ``logo.png``.
    projection["logo_overlay_path"] = LOGO_PNG
    try:
        with _measure_artefact(symbol.image_bytes) as artefact:
            width, height = artefact.size
            module_px = artefact.module_px
            distinct = artefact.distinct_colours
    except _ArtefactTooLarge as error:
        return _unmeasurable(
            ctx,
            RN_LOGO,
            case_id,
            projection,
            expected,
            symbol,
            error,
            started,
            {
                "logo_area_fraction_requested": fraction,
                "logo_seed": seed,
                "logo_side_px": side_px,
            },
            seed=seed,
            logo=bitmap,
        )
    modules = (
        None
        if decoded_result.version is None
        else module_count_for_version(decoded_result.version)
    )
    covered_image_area = (min(side_px, width) * min(side_px, height)) / float(width * height)
    if modules is not None and module_px:
        # The covered fraction is measured against the symbol's own module area, which is what
        # ``logo_area_fraction`` names; the whole image would shrink it by the quiet zone's share,
        # inflating the damage a short-margin target appears to survive.
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
