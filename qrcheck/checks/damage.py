"""``DM-RESILIENCE`` and ``DM-SEEDED``: damage one symbol seven ways, then reproduce it.

The damage family and nothing else (AAP §0.8.1, §0.9.1).  Every threshold lives in
``qrcheck.toml`` and is read from the resolved configuration -- this module holds no threshold
of its own -- and no assertion is weakened, skipped or relaxed to keep a target green.

Case matrix: both checks run the same 56 cases, a fixed 8-record sample crossed with the seven
declared damage kinds in their declared order
(:data:`~qrcheck.evidence.naming.DAMAGE_KINDS`).  One case is one variant of one record, and
its case identifier is ``corpus:<record-id>@<kind>`` -- the AAP §0.3.4 prefix plus the variant
suffix, which :func:`qrcheck.report.model.failure_id` sanitizes for the evidence path.  The
sample is the first eight corpus records in ``id`` order whose payload is between
:data:`_MIN_PAYLOAD_BYTES` and :data:`_MAX_PAYLOAD_BYTES`: the upper bound keeps the defective
target's ``truncated_payload`` injection away from them (it truncates only payloads longer than
64 bytes whose truncation keeps the version), and the window keeps them well inside capacity,
so the damage budget is the only thing under test.

Encoder invocations are declared, not incidental (:data:`EXPECTED_ENCODES`, AAP §0.4.3):
``DM-RESILIENCE`` makes 8 -- one per sampled record, shared across that record's seven kinds --
and ``DM-SEEDED`` makes 0.  No retry, re-run or second call exists in this module, so a count of
nine or one is a framework defect.  Decoder invocations are one per case for the damaged
variant (56 each) plus, in ``DM-RESILIENCE``, one memoised pristine reading per sampled record
(8), because the occlusion budget is the recovery capacity of the EC level decoded from the
symbol and that fact exists only after a decode.

What each check asserts: ``DM-RESILIENCE`` applies one damage kind to the symbol the target
returned and asserts that the **primary decoder** still returns the requested payload, while
the secondary decoder's outcome is recorded and never asserted (AAP §0.3.2) -- opencv is
fragile on damaged symbols and a silent second opinion is not a disagreement, so failing on it
would report the corroborating decoder's weakness as the target's defect.  ``DM-SEEDED``
re-applies the same damage to the symbol it already holds, with the same derived seed and the
same resolved thresholds, and asserts the result is **byte-identical** to the variant
``DM-RESILIENCE`` produced, before decoding it and asserting the payload again; it never calls
the adapter, which makes "a damaged image is reproducible from what the report records" (AAP
§0.5.2, criterion 4) a property of the product rather than of a re-run.

The occluder budget is ``recovery_fraction(decoded EC level) - occlusion_margin`` taken from
the level the symbol itself reports, never the requested one -- AAP §0.4.2's cascade
containment, and why the defective target's ``wrong_ec_level`` injection cannot push damage past
the symbol's real capacity and turn eight damage cases into failures.  It buys
``round(budget x eligible module count)`` modules, spent whole codewords at a time: the recovery
fractions are stated per codeword and one occluded module corrupts the entire codeword it sits
in, so spreading the same modules one by one would corrupt up to eight times the codewords the
budget allows and fail a conforming target at EC Q and EC H.  A codeword is
:data:`_MODULES_PER_CODEWORD` modules (ISO/IEC 18004 §7.6), and the codewords are located by
replaying the standard's placement traversal (:func:`_codeword_modules`) over the exact
function-pattern mask.

The run-scoped symbol memo lives here because ``RunContext`` carries no symbol store, and three
invariants keep it honest: it is cleared once, at the start of ``DM-RESILIENCE``'s case
enumeration, which the runner always reaches first because it walks checks by identifier
ascending; a record's symbol is encoded lazily on first touch, since case identifiers sort
alphabetically and nothing may assume an order; and nothing else in the product may read it --
the framework is single-process and sequential (AAP §0.3.6), so it needs no lock.

Never: no retry, no ``xfail``, no threshold literal, no ``except Exception``, no bare
``except``, no assertion on the secondary decoder under damage and none on wall-clock time (the
only clock read produces ``duration_ms``, which every report masks).  ``TargetUnavailable`` and
``TargetTimeout`` propagate to :mod:`qrcheck.runner`, which owns the one-error-result-per-check
target-loss policy; the other three adapter errors are translated by
:func:`qrcheck.checks.support.case_status_for_error`, the single translation in the product.  No
image library is imported: Pillow is reachable only through :mod:`qrcheck.evidence.mux`, so a
raster is built from the image mux handed us (:func:`_raster`) and every raster is encoded by
:func:`qrcheck.evidence.mux.encode_png`.

Reason codes this family emits: ``value_mismatch`` (the damaged symbol no longer returns the
requested payload, or the regenerated variant differs from the recorded one),
``sampling_failed`` (only when the pristine, pre-damage measurement cannot recover the module
grid or the symbol's own EC level, without which no budget exists), ``missing_dependency``
(``DM-SEEDED`` selected without the ``DM-RESILIENCE`` run that holds the symbol), plus whatever
:func:`~qrcheck.checks.support.case_status_for_error` reports for ``target_rejected``,
``contract_error`` and ``unsupported_capability``.

Determinism (AAP §0.5.1, §0.5.2): every random draw comes from
``derive_seed(run_seed, "DM-RESILIENCE", case_id, kind)`` fed only into
:class:`numpy.random.PCG64`; the derived seed is the result's ``seed``, the ``<seed>`` of the
``damage-<kind>-<seed>.png`` artefact name, and the value ``DM-SEEDED`` re-derives.  Identical
inputs therefore produce identical damaged bytes, which is what makes criterion 4's
byte-for-byte comparison meaningful.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import time
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy

from qrcheck.checks import support
from qrcheck.config.models import DamageLimits
from qrcheck.config.seeding import derive_seed
from qrcheck.decode import DecodeResult
from qrcheck.decode.matrix import module_count_for_version, version_from_dimension
from qrcheck.evidence import mux
from qrcheck.evidence.naming import DAMAGE_KINDS
from qrcheck.registry.check import (
    DEPENDENCY_PRIMARY_DECODER,
    Case,
    RunContext,
    check,
)
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
)

class _Record(Protocol):
    """The two members of a corpus record this family reads.

    Declared structurally rather than imported: the corpus builder is not part of this
    file's dependency set, and a check needs nothing from a record but its identifier and its
    payload, so a protocol states the contract without reaching for a module the layering
    does not grant.
    """

    # The record's stable identifier, ``<class>-<3-digit index>``, and the payload it asks
    # the target to encode.  Attribute declarations, not members with bodies: a protocol
    # states a shape and implements nothing.
    id: str
    payload_bytes: bytes


__all__ = [
    "DM_RESILIENCE",
    "DM_SEEDED",
    "EXPECTED_ENCODES",
    "clear_symbol_memo",
    "dm_resilience_cases",
    "dm_seeded_cases",
    "recall_symbol",
    "recall_variant",
    "remember_symbol",
    "remember_variant",
]

# The two check identifiers, spelled once: they appear in the registry table, in the
# expected-failure table, in the seed derivation's namespace and in every diagnostic here.
DM_RESILIENCE: Final[str] = "DM-RESILIENCE"
DM_SEEDED: Final[str] = "DM-SEEDED"

# The declared adapter-invocation counts of AAP §0.4.3 / §0.8.1's case matrix: eight encodes
# for DM-RESILIENCE, one per sampled record and shared across its seven kinds; none at all
# for DM-SEEDED, which re-applies the damage to the symbol it already holds.  The negative
# test asserts the observed count equals this table, so a retry shows up as a ninth call.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {DM_RESILIENCE: 8, DM_SEEDED: 0}
)

# The fixed sample of AAP §0.8.1: the first eight corpus records in id order whose payload
# length is inside this window.  The upper bound is the truncation injection's trigger
# length, so no sampled record can be touched by it; the lower bound keeps every record
# large enough for the symbol to carry a meaningful damage budget at EC L.
_SAMPLE_SIZE: Final[int] = 8
_MIN_PAYLOAD_BYTES: Final[int] = 8
_MAX_PAYLOAD_BYTES: Final[int] = 64

# ISO/IEC 18004 §7.6: one codeword is eight modules, and the error-correction capacity the
# recovery fractions describe is expressed per codeword -- which is why the occlusion budget
# is spent codeword by codeword rather than module by scattered module (one occluded module
# corrupts a whole codeword, so module-level spreading spends the budget eight times over).
_MODULES_PER_CODEWORD: Final[int] = 8

# How many seeded, evenly strided runs of consecutive codewords the occlusion budget is
# spread over.  Consecutive codewords belong to different Reed-Solomon blocks (the symbol
# interleaves them), so a run of consecutive codewords damages every block evenly, while
# several separated runs keep the damaged area from becoming one contiguous hole that would
# test the decoder's detection rather than the symbol's error correction.
_OCCLUDER_RUNS: Final[int] = 4

# The single occluder colour, as an RGB triple: a mid grey that is neither the dark nor the
# light module colour.  It is a pixel value rather than a damage threshold, which is why it is
# a constant here while every threshold comes from ``[checks.damage]``.
_OCCLUDER_RGB: Final[tuple[int, int, int]] = (128, 128, 128)

# The one indexed coordinate the traversal must skip: column 6 carries the vertical timing
# pattern (ISO/IEC 18004 §7.3.4), so the two-module column pairs stop short of it.
_TIMING_LINE: Final[int] = 6

# The finder-plus-separator footprint (ISO/IEC 18004 §7.3.2-7.3.3): seven modules of finder
# plus one of separator, i.e. rows 0..7 by columns 0..7.  The three corner blocks reserved
# below are one module wider because the eighth row and column of each carry the first
# format-information copy, the second copy and the fixed dark module as well.
_FINDER_FOOTPRINT: Final[int] = 8

# Version information (ISO/IEC 18004 §7.10) exists from Version 7 upwards, as two 6x3 blocks.
_FIRST_VERSION_WITH_VERSION_INFO: Final[int] = 7

# What a warp writes where it samples outside the source image: the light module colour, which
# is also the quiet zone's, so a rotated or skewed symbol extends its quiet zone instead of
# gaining a dark border that would be a second, unintended damage kind.
_FILL_VALUE: Final[float] = 255.0


# The run-scoped memo: the symbol, the pristine reading and the damaged variant of each
# sampled record.  DM-SEEDED's zero-encode assertion rests on it, so its lifetime is stated
# in the module docstring and its only writer is DM-RESILIENCE's case executor.

# The symbol the target returned for a record, keyed by record id: one encode per record,
# shared across that record's seven kinds.
_SYMBOLS: dict[str, GeneratedSymbol] = {}

# The pristine decode of that symbol, keyed by record id: the module count and the EC level
# the symbol itself reports, which the occlusion budget is derived from.  Memoised so the
# seven kinds of a record cost one reading rather than seven.
_READINGS: dict[str, DecodeResult] = {}

# The damaged artefact DM-RESILIENCE produced, keyed by (record id, damage kind): the bytes
# DM-SEEDED regenerates and compares against, byte for byte.
_VARIANTS: dict[tuple[str, str], bytes] = {}


def clear_symbol_memo() -> None:
    """Forget every symbol, pristine reading and damaged variant held for a run.

    The runner walks checks by identifier ascending, so ``DM-RESILIENCE`` always enumerates
    its cases before ``DM-SEEDED`` executes, and calling this at the start of that
    enumeration is what stops a second run in the same process from inheriting the first
    run's symbols, readings or variants.
    """
    _SYMBOLS.clear()
    _READINGS.clear()
    _VARIANTS.clear()


def remember_symbol(record_id: str, symbol: GeneratedSymbol) -> None:
    """Record the symbol `symbol` the target returned for corpus record `record_id`.

    Args:
        record_id: The corpus record's identifier, e.g. ``boundary-byte-000``.
        symbol: The artefact the target returned for that record's request.

    Raises:
        ValueError: `record_id` is not a non-empty string, or `symbol` carries no bytes.
    """
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"record id must be a non-empty str, got {record_id!r}")
    if not isinstance(symbol, GeneratedSymbol):
        raise ValueError(
            f"record {record_id!r}: expected a GeneratedSymbol, got "
            f"{type(symbol).__name__}"
        )
    _SYMBOLS[record_id] = symbol


def recall_symbol(record_id: str) -> GeneratedSymbol | None:
    """Return the symbol held for `record_id`, or ``None`` when the memo holds none.

    ``None`` is the normal answer when ``DM-SEEDED`` is selected without
    ``DM-RESILIENCE``: nothing has encoded the record yet, and the case is skipped with
    reason ``missing_dependency`` rather than decoding a symbol this check would have had to
    ask the adapter for.
    """
    return _SYMBOLS.get(record_id)


def remember_variant(record_id: str, kind: str, png_bytes: bytes) -> None:
    """Record the damaged variant `DM-RESILIENCE` produced for one record and damage kind.

    Args:
        record_id: The corpus record's identifier.
        kind: One of :data:`~qrcheck.evidence.naming.DAMAGE_KINDS`.
        png_bytes: The encoded damaged artefact, exactly as it was decoded and as it is
            persisted under ``damage-<kind>-<seed>.png``.

    Raises:
        ValueError: `record_id` is not a non-empty string, `kind` is not a declared damage
            kind, or `png_bytes` is not bytes-like.
    """
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"record id must be a non-empty str, got {record_id!r}")
    if kind not in DAMAGE_KINDS:
        raise ValueError(
            f"unknown damage kind {kind!r}; declared kinds, in order: {', '.join(DAMAGE_KINDS)}"
        )
    if not isinstance(png_bytes, (bytes, bytearray, memoryview)):
        raise ValueError(
            f"record {record_id!r} damage {kind!r}: expected bytes, got "
            f"{type(png_bytes).__name__}"
        )
    _VARIANTS[(record_id, kind)] = bytes(png_bytes)


def recall_variant(record_id: str, kind: str) -> bytes | None:
    """Return the damaged variant held for ``(record_id, kind)``, or ``None`` when absent."""
    return _VARIANTS.get((record_id, kind))


def _remember_reading(record_id: str, reading: DecodeResult) -> None:
    """Record the pristine reading of `record_id`'s symbol; see :data:`_READINGS`."""
    _READINGS[record_id] = reading


def _recall_reading(record_id: str) -> DecodeResult | None:
    """Return the pristine reading held for `record_id`, or ``None`` when the memo has none."""
    return _READINGS.get(record_id)


# The symbol's own structure (ISO/IEC 18004): which modules are function patterns, and which
# eight modules form each codeword.  Both are needed before a single occluder can be placed:
# covering a finder, separator, timing, alignment, format-information or version-information
# module makes a symbol undetectable however much error-correction capacity remains, and a
# budget expressed per codeword needs to know where the codewords are.


def _alignment_centres(version: int) -> tuple[int, ...]:
    """Return the alignment-pattern centre coordinates of `version`.

    ISO/IEC 18004 Annex E: Version 1 has none, and from Version 2 the centres are ``6`` followed by
    an evenly stepped ascending sequence ending at ``4V + 10``, with Version 32's nominal step of 26
    divided differently -- hence the special case.

    Raises:
        ValueError: `version` is not one of the versions this traversal accepts (1-40), which
            :func:`qrcheck.decode.matrix.module_count_for_version` refuses first.
    """
    if version < 2:
        return ()
    count = version // 7 + 2
    step = 26 if version == 32 else ((version * 4 + count * 2 + 1) // (count * 2 - 2)) * 2
    centres = [_TIMING_LINE]
    position = version * 4 + 10
    for _ in range(count - 1):
        centres.insert(1, position)
        position -= step
    return tuple(centres)


def _function_modules(module_count: int, version: int) -> numpy.ndarray:
    """Return the ``(N, N)`` mask of modules that carry no data, ``True`` where reserved.

    The complete function-pattern set of ISO/IEC 18004: the three finder patterns with their
    separators as the three corner blocks -- one module wider than the finder footprint, because
    their eighth row and column also carry both copies of the 15-bit format information and the
    fixed dark module, which live nowhere else; the horizontal and vertical timing patterns; every
    alignment pattern, skipping the three whose centres fall on the finder patterns; and both
    version-information blocks, for Version 7 and above.  ``N = 4V + 17``.

    Raises:
        ValueError: `module_count` and `version` disagree, or `version` is outside 1-40.
    """
    expected = module_count_for_version(version)
    if module_count != expected:
        raise ValueError(
            f"version {version} implies {expected} modules per side, got {module_count}"
        )
    reserved = numpy.zeros((module_count, module_count), dtype=bool)
    corner = _FINDER_FOOTPRINT + 1
    reserved[0:corner, 0:corner] = True
    reserved[0:corner, module_count - _FINDER_FOOTPRINT : module_count] = True
    reserved[module_count - _FINDER_FOOTPRINT : module_count, 0:corner] = True
    reserved[_TIMING_LINE, :] = True
    reserved[:, _TIMING_LINE] = True
    centres = _alignment_centres(version)
    overlapping = {
        (_TIMING_LINE, _TIMING_LINE),
        (_TIMING_LINE, module_count - 7),
        (module_count - 7, _TIMING_LINE),
    }
    for row in centres:
        for column in centres:
            if (row, column) in overlapping:
                continue
            reserved[
                max(row - 2, 0) : row + 3, max(column - 2, 0) : column + 3
            ] = True
    if version >= _FIRST_VERSION_WITH_VERSION_INFO:
        reserved[0:6, module_count - 11 : module_count - 8] = True
        reserved[module_count - 11 : module_count - 8, 0:6] = True
    return reserved


def _codeword_modules(
    module_count: int,
    reserved: numpy.ndarray,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the module coordinates of every codeword, in placement order.

    ISO/IEC 18004 §7.7.3: the symbol is filled from the bottom-right corner in two-module-wide
    columns, alternating upwards and downwards, skipping the vertical timing column and every
    function module, and each group of eight modules is one codeword.  The encoder interleaves the
    Reed-Solomon blocks along exactly this order, so consecutive codewords belong to different
    blocks -- which is what makes a run of consecutive codewords damage every block evenly.  The
    trailing group of fewer than eight modules a symbol may end with is the remainder bits of
    ISO/IEC 18004 Table 1: they belong to no codeword and are dropped, so this returns fewer than
    ``data modules / 8`` groups only when a symbol has remainder bits.

    Raises:
        ValueError: `reserved` is not an ``(N, N)`` boolean array.
    """
    if reserved.shape != (module_count, module_count):
        raise ValueError(
            f"function-module mask of shape {reserved.shape} does not fit "
            f"{module_count} modules per side"
        )
    codewords: list[tuple[tuple[int, int], ...]] = []
    current: list[tuple[int, int]] = []
    for right in range(module_count - 1, 0, -2):
        column_pair = right - 1 if right <= _TIMING_LINE else right
        upward = ((column_pair + 1) & 2) == 0
        for index in range(module_count):
            row = module_count - 1 - index if upward else index
            for column in (column_pair, column_pair - 1):
                if reserved[row, column]:
                    continue
                current.append((row, column))
                if len(current) == _MODULES_PER_CODEWORD:
                    codewords.append(tuple(current))
                    current = []
    return tuple(codewords)


# Pixel geometry: where the modules are on the image, and which of them an occluder may
# cover.  Measured from the pixels rather than read from the request, because a target is
# free to draw a smaller module, a different quiet zone or a different image size.


@dataclasses.dataclass(frozen=True)
class _Frame:
    """The pixel-to-module geometry of one pristine symbol.

    ``top`` and ``left`` are the first pixel of the symbol's own top-left module, and the two
    pitches are the module edge in pixels per axis: the dark-pixel bounding box of a Model 2
    symbol is exactly the symbol's extent, because the three finder patterns touch row 0, row
    ``N - 1``, column 0 and column ``N - 1``.  Measuring that box therefore needs neither the
    requested quiet zone -- which a defective target may not have drawn -- nor the requested
    module size, and a target that draws a non-square module is measured per axis rather than
    averaged into a box that drifts off its last column.
    """

    module_count: int
    version: int
    top: int
    left: int
    pitch_x: float
    pitch_y: float
    reserved: numpy.ndarray
    codewords: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def pitch(self) -> float:
        """Return the module edge in pixels, averaged over the two axes."""
        return (self.pitch_x + self.pitch_y) / 2.0

    @property
    def eligible_modules(self) -> int:
        """Return how many modules a codeword may occupy, i.e. every non-function module."""
        return self.module_count * self.module_count - int(self.reserved.sum())

    def pixel_box(self, row: int, column: int) -> tuple[int, int, int, int]:
        """Return the ``(x0, y0, x1, y1)`` pixel box of the module at `row`, `column`.

        The box is half-open on the right and bottom, which is the form a numpy slice uses,
        so an occluder lands on exactly the modules it names and never overlaps its
        neighbour.
        """
        x0 = int(round(self.left + column * self.pitch_x))
        y0 = int(round(self.top + row * self.pitch_y))
        x1 = int(round(self.left + (column + 1) * self.pitch_x))
        y1 = int(round(self.top + (row + 1) * self.pitch_y))
        return x0, y0, x1, y1


def _luma(pixels: numpy.ndarray) -> numpy.ndarray:
    """Return the single-channel luminance of an ``(H, W, 3)`` or ``(H, W)`` pixel array."""
    values = pixels.astype(numpy.float64)
    if values.ndim == 3:
        return values.mean(axis=2)
    return values


def _frame(version: int, pixels: numpy.ndarray) -> _Frame | None:
    """Measure the module geometry of a symbol, or return ``None`` when it cannot be.

    A measurement failure is a value rather than an exception: the pixels came from a target, so an
    image with no contrast carries no grid to find, and the caller reports that as
    ``sampling_failed`` evidence instead of turning it into a framework traceback.

    Raises:
        ValueError: `version` is outside 1-40, or `pixels` is not a 2- or 3-dimensional array --
            both producer defects rather than evidence about a target.
    """
    module_count = module_count_for_version(version)
    luma = _luma(pixels)
    if luma.ndim != 2:
        raise ValueError(f"symbol pixels must be 1- or 3-channel, got shape {pixels.shape}")
    low = float(luma.min())
    high = float(luma.max())
    if not high > low:
        return None
    dark = luma < (low + high) / 2.0
    rows = numpy.flatnonzero(dark.any(axis=1))
    columns = numpy.flatnonzero(dark.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    top = int(rows[0])
    bottom = int(rows[-1])
    left = int(columns[0])
    right = int(columns[-1])
    reserved = _function_modules(module_count, version)
    return _Frame(
        module_count=module_count,
        version=version,
        top=top,
        left=left,
        pitch_x=(right - left + 1) / module_count,
        pitch_y=(bottom - top + 1) / module_count,
        reserved=reserved,
        codewords=_codeword_modules(module_count, reserved),
    )


def _occluder_count(budget: float, eligible_modules: int) -> int:
    """Return how many codewords `budget` of a symbol's eligible area may lose.

    The recovery fractions of :func:`qrcheck.checks.support.recovery_fraction` are stated per
    codeword and :data:`_MODULES_PER_CODEWORD` modules make one, so a budget spent over the
    eligible module count converts to a codeword count by that divisor.  At least one codeword is
    always damaged: at EC L the budget is a fiftieth of the area, which rounds to nothing on a
    small symbol, and a case that damaged nothing would be vacuous.
    """
    share = budget * eligible_modules / _MODULES_PER_CODEWORD
    return max(1, int(round(share)))


def _occluder_modules(
    codewords: tuple[tuple[tuple[int, int], ...], ...],
    count: int,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    """Return the modules of `count` codewords, chosen at seeded, evenly strided runs.

    Consecutive codewords belong to different Reed-Solomon blocks, so a run of consecutive
    codewords damages every block evenly -- exactly what the documented recovery capacity assumes,
    and what selecting codewords uniformly at random does not achieve, since random selection
    overshoots one block while leaving another untouched.  Several separated runs rather than one
    keep the damaged area from becoming a single contiguous hole, and the run's start is drawn from
    the seeded generator, so the placement is reproducible from
    ``(run seed, check id, case id, kind)`` and differs between records.
    """
    total = len(codewords)
    if total == 0:
        raise ValueError("a symbol with no codeword cannot be damaged")
    count = min(max(count, 1), total)
    runs = min(_OCCLUDER_RUNS, count)
    base, extra = divmod(count, runs)
    stride = max(total // runs, 1)
    generator = numpy.random.Generator(numpy.random.PCG64(seed))
    chosen: set[int] = set()
    for index in range(runs):
        length = base + (1 if index < extra else 0)
        start = (int(generator.integers(0, stride)) + index * stride) % total
        for offset in range(length):
            chosen.add((start + offset) % total)
    return tuple(
        module for codeword in sorted(chosen) for module in codewords[codeword]
    )


# The seven damage kinds.  Each is a pure function of the symbol's pixels and the resolved
# ``[checks.damage]`` values -- no clock, no module state, no filesystem -- and the two that
# draw a random value take it from the case's derived seed only.


def _bilinear(pixels: numpy.ndarray, xs: numpy.ndarray, ys: numpy.ndarray) -> numpy.ndarray:
    """Sample `pixels` bilinearly at the fractional coordinates `xs`, `ys`.

    Coordinates outside the source image read :data:`_FILL_VALUE`, which is the light module
    colour and the quiet zone's colour: a warp must extend the quiet zone, never punch a dark
    border into it, or it would be testing the decoder's handling of a black frame.

    Args:
        pixels: The source pixels, ``(H, W, 3)`` uint8.
        xs: The fractional source column of every destination pixel.
        ys: The fractional source row of every destination pixel.

    Returns:
        The sampled pixels, same shape as the coordinate arrays plus the channel axis.
    """
    values = pixels.astype(numpy.float64)
    height, width = values.shape[0], values.shape[1]
    x0 = numpy.floor(xs).astype(numpy.int64)
    y0 = numpy.floor(ys).astype(numpy.int64)
    inside = (xs >= 0.0) & (xs <= width - 1) & (ys >= 0.0) & (ys <= height - 1)
    fraction_x = (xs - x0)[..., None]
    fraction_y = (ys - y0)[..., None]
    left = numpy.clip(x0, 0, width - 1)
    right = numpy.clip(x0 + 1, 0, width - 1)
    upper = numpy.clip(y0, 0, height - 1)
    lower = numpy.clip(y0 + 1, 0, height - 1)
    top_row = values[upper, left] * (1.0 - fraction_x) + values[upper, right] * fraction_x
    bottom_row = values[lower, left] * (1.0 - fraction_x) + values[lower, right] * fraction_x
    sampled = top_row * (1.0 - fraction_y) + bottom_row * fraction_y
    filled = numpy.where(inside[..., None], sampled, _FILL_VALUE)
    return numpy.clip(numpy.rint(filled), 0, 255).astype(numpy.uint8)


def _resample(pixels: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    """Resample `pixels` to `width` by `height`, sampling at the centre of every pixel.

    Args:
        pixels: The source pixels.
        width: The destination width in pixels, at least 1.
        height: The destination height in pixels, at least 1.

    Returns:
        The resampled pixels.

    Raises:
        ValueError: `width` or `height` is not positive.
    """
    if width < 1 or height < 1:
        raise ValueError(f"resample target must be positive, got {width}x{height}")
    source_height, source_width = pixels.shape[0], pixels.shape[1]
    columns = (numpy.arange(width) + 0.5) * (source_width / width) - 0.5
    rows = (numpy.arange(height) + 0.5) * (source_height / height) - 0.5
    xs, ys = numpy.meshgrid(columns, rows)
    return _bilinear(pixels, xs, ys)


def _rotate_pixels(pixels: numpy.ndarray, degrees: float) -> numpy.ndarray:
    """Rotate `pixels` by `degrees` about its centre onto an expanded canvas.

    The canvas grows to hold the rotated rectangle, so the symbol is never clipped, and the
    corners the rotation exposes are the light colour -- a black wedge over the quiet zone
    would be a second, unintended damage kind.

    Args:
        pixels: The source pixels.
        degrees: The rotation, in degrees; the configured ``rotation_deg``.

    Returns:
        The rotated pixels.
    """
    height, width = pixels.shape[0], pixels.shape[1]
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    out_width = int(math.ceil(abs(width * cosine) + abs(height * sine)))
    out_height = int(math.ceil(abs(height * cosine) + abs(width * sine)))
    columns = numpy.arange(out_width) - (out_width - 1) / 2.0
    rows = numpy.arange(out_height) - (out_height - 1) / 2.0
    xs, ys = numpy.meshgrid(columns, rows)
    source_x = cosine * xs + sine * ys + (width - 1) / 2.0
    source_y = -sine * xs + cosine * ys + (height - 1) / 2.0
    return _bilinear(pixels, source_x, source_y)


def _homography(destination: numpy.ndarray, source: numpy.ndarray) -> numpy.ndarray:
    """Return the eight projective coefficients mapping `destination` onto `source`.

    The standard four-point direct-linear-transform system, solved with
    :func:`numpy.linalg.solve`: for each correspondence ``(x, y) -> (u, v)``,
    ``u = (a x + b y + c) / (g x + h y + 1)`` and ``v = (d x + e y + f) / (g x + h y + 1)``, so
    the result is ``[a, b, c, d, e, f, g, h]``.

    Raises:
        ValueError: the four points are degenerate, so no projective map exists.
    """
    matrix = numpy.zeros((8, 8), dtype=numpy.float64)
    target = numpy.zeros(8, dtype=numpy.float64)
    for index in range(4):
        x, y = float(destination[index][0]), float(destination[index][1])
        u, v = float(source[index][0]), float(source[index][1])
        matrix[2 * index] = (x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y)
        matrix[2 * index + 1] = (0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)
        target[2 * index] = u
        target[2 * index + 1] = v
    try:
        return numpy.linalg.solve(matrix, target)
    except numpy.linalg.LinAlgError:
        raise ValueError("a degenerate quadrangle has no perspective transform") from None


def _perspective_pixels(pixels: numpy.ndarray, degrees: float) -> numpy.ndarray:
    """Warp `pixels` in perspective by `degrees`, keeping the canvas size.

    The top edge of the symbol is narrowed and shifted, which is the shape a symbol takes
    when it is photographed off-axis; the quadrangle is then solved back onto the source
    rectangle, so the sampling stays a plain bilinear read.

    Args:
        pixels: The source pixels.
        degrees: The configured ``perspective_deg``; its tangent sets the skew.

    Returns:
        The warped pixels.
    """
    height, width = pixels.shape[0], pixels.shape[1]
    shift = min(abs(math.tan(math.radians(degrees))), 1.0) * width / 2.0
    destination = numpy.array(
        [
            (shift, 0.0),
            (width - 1 - shift, 0.0),
            (width - 1.0, height - 1.0),
            (0.0, height - 1.0),
        ],
        dtype=numpy.float64,
    )
    source = numpy.array(
        [
            (0.0, 0.0),
            (width - 1.0, 0.0),
            (width - 1.0, height - 1.0),
            (0.0, height - 1.0),
        ],
        dtype=numpy.float64,
    )
    a, b, c, d, e, f, g, h = _homography(destination, source)
    columns = numpy.arange(width).astype(numpy.float64)
    rows = numpy.arange(height).astype(numpy.float64)
    xs, ys = numpy.meshgrid(columns, rows)
    denominator = g * xs + h * ys + 1.0
    denominator = numpy.where(numpy.abs(denominator) < 1e-9, 1e-9, denominator)
    source_x = (a * xs + b * ys + c) / denominator
    source_y = (d * xs + e * ys + f) / denominator
    return _bilinear(pixels, source_x, source_y)


def _downscale_pixels(
    pixels: numpy.ndarray,
    px_per_module: float,
    module_count: int,
) -> numpy.ndarray:
    """Shrink `pixels` to `px_per_module` pixels per module, then restore its size.

    The resolution loss is the damage: a symbol that survives it survives being printed
    small, photographed and enlarged again.  The image keeps its aspect ratio, so the quiet
    zone shrinks with the modules.

    Args:
        pixels: The source pixels.
        px_per_module: The configured ``downscale_px_per_module``.
        module_count: The symbol's modules per side.

    Returns:
        The downscaled-then-restored pixels.
    """
    height, width = pixels.shape[0], pixels.shape[1]
    side = max(1, int(round(px_per_module * module_count)))
    scale = side / float(max(height, width))
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    return _resample(_resample(pixels, small_width, small_height), width, height)


def _blur_pixels(pixels: numpy.ndarray, radius_px: float) -> numpy.ndarray:
    """Blur `pixels` with a Gaussian whose standard deviation is half the radius.

    A separable convolution, so the cost is linear in the kernel width rather than quadratic, and
    an edge-padded source so the quiet zone's border is not darkened by padding with zeros.  The
    kernel's weights are :mod:`numpy` floats, so the result is identical on every run.
    """
    sigma = max(radius_px / 2.0, 1e-6)
    half = max(1, int(math.ceil(2.0 * sigma)))
    offsets = numpy.arange(-half, half + 1, dtype=numpy.float64)
    kernel = numpy.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    blurred = pixels.astype(numpy.float64)
    for axis in (0, 1):
        padding = (
            ((half, half), (0, 0), (0, 0))
            if axis == 0
            else ((0, 0), (half, half), (0, 0))
        )
        padded = numpy.pad(blurred, padding, mode="edge")
        length = blurred.shape[axis]
        total = numpy.zeros_like(blurred)
        for index, weight in enumerate(kernel):
            window = (
                padded[index : index + length]
                if axis == 0
                else padded[:, index : index + length]
            )
            total += weight * window
        blurred = total
    return numpy.clip(numpy.rint(blurred), 0, 255).astype(numpy.uint8)


def _noise_pixels(pixels: numpy.ndarray, sigma: float, seed: int) -> numpy.ndarray:
    """Add Gaussian noise of standard deviation `sigma` to `pixels`.

    The noise pattern comes from :class:`numpy.random.PCG64` seeded with the case's derived
    seed, whose bit stream is stable across numpy releases, so the damaged image is
    reproducible byte for byte inside the pinned lock (AAP §0.5.4).

    Args:
        pixels: The source pixels.
        sigma: The configured ``noise_sigma``.
        seed: The case's derived seed.

    Returns:
        The noisy pixels.
    """
    if sigma <= 0.0:
        return pixels.copy()
    generator = numpy.random.Generator(numpy.random.PCG64(seed))
    noise = generator.normal(0.0, sigma, size=pixels.shape)
    return numpy.clip(numpy.rint(pixels + noise), 0, 255).astype(numpy.uint8)


def _occlude_pixels(
    pixels: numpy.ndarray,
    frame: _Frame,
    modules: tuple[tuple[int, int], ...],
) -> numpy.ndarray:
    """Cover each module of `modules` with the occluder colour.

    The occluders are module-sized blocks rather than one contiguous hole: several small
    opaque patches test the symbol's error correction, whereas a single hole of the same
    area would mostly test the decoder's detection.

    Args:
        pixels: The source pixels.
        frame: The measured geometry the module coordinates belong to.
        modules: The modules to cover.

    Returns:
        The occluded pixels.
    """
    damaged = pixels.copy()
    for row, column in modules:
        x0, y0, x1, y1 = frame.pixel_box(row, column)
        damaged[y0:y1, x0:x1] = _OCCLUDER_RGB
    return damaged


def _raster(pixels: numpy.ndarray, reference: Any) -> Any:
    """Return the reference image's own raster class carrying `pixels`.

    Pillow is reachable only through :mod:`qrcheck.evidence.mux` in this package, and mux offers
    opening and PNG encoding rather than construction from an array -- deliberately, because it is
    the framework's single image boundary.  The image mux handed us is therefore used as the
    canvas: resizing it to the array's dimensions allocates a raster of exactly that size, and
    ``putdata`` then replaces every pixel, so none of the reference's content survives.  The result
    is an ordinary Pillow image, encoded by :func:`qrcheck.evidence.mux.encode_png` like every
    other artefact of this framework.
    """
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    canvas = reference.convert("RGB").resize((width, height))
    canvas.putdata([tuple(pixel) for pixel in pixels.reshape(-1, 3).tolist()])
    return canvas


def _jpeg_pixels(pixels: numpy.ndarray, quality: int, reference: Any) -> numpy.ndarray:
    """Re-encode `pixels` as JPEG at `quality` and read the decoded result back.

    JPEG's lossy transform is the damage: a symbol that survives it survives being stored or
    transmitted in a lossy format.  The re-encode goes through Pillow's own writer, reached
    as a method of the image mux handed us, so this module still imports no image library.

    Args:
        pixels: The source pixels.
        quality: The configured ``jpeg_quality``.
        reference: The symbol's image, used only as the raster canvas.

    Returns:
        The pixels as they come back out of a JPEG round trip.
    """
    buffer = io.BytesIO()
    _raster(pixels, reference).save(buffer, format="JPEG", quality=int(quality))
    return numpy.asarray(mux.decode_image(buffer.getvalue()).convert("RGB"), dtype=numpy.uint8)


def _damaged_pixels(
    kind: str,
    pixels: numpy.ndarray,
    *,
    frame: _Frame,
    limits: DamageLimits,
    seed: int,
    decoded_ec: str,
    reference: Any,
) -> tuple[numpy.ndarray, dict[str, Any]]:
    """Return the damaged pixels of one damage kind, plus the values actually applied.

    Each kind reads its threshold from the resolved ``[checks.damage]`` record and nothing else,
    and the measurements returned here are what ``request.json`` and ``decoded.json`` record, so a
    reader can re-derive the damage without consulting this file.  The occlusion budget comes from
    `decoded_ec`, the level the symbol itself reports.

    Raises:
        ValueError: `kind` is not a declared damage kind -- a caller defect, since the case
            identifiers are built from the same declared tuple.
        KeyError: `decoded_ec` is not one of ``L``/``M``/``Q``/``H``; the caller establishes the
            level before it reaches here.
    """
    if kind == "occluders":
        budget = support.recovery_fraction(decoded_ec) - limits.occlusion_margin
        count = _occluder_count(budget, frame.eligible_modules)
        modules = _occluder_modules(frame.codewords, count, seed)
        applied: dict[str, Any] = {
            "occlusion_budget": round(budget, 6),
            "occlusion_margin": limits.occlusion_margin,
            "occluded_codewords": count,
            "occluded_modules": len(modules),
            "eligible_modules": frame.eligible_modules,
        }
        return _occlude_pixels(pixels, frame, modules), applied
    if kind == "blur":
        radius = limits.blur_radius_per_module * frame.pitch
        return _blur_pixels(pixels, radius), {
            "blur_radius_modules": limits.blur_radius_per_module,
            "blur_radius_px": round(radius, 6),
            "module_pitch_px": round(frame.pitch, 6),
        }
    if kind == "jpeg":
        return _jpeg_pixels(pixels, limits.jpeg_quality, reference), {
            "jpeg_quality": limits.jpeg_quality
        }
    if kind == "noise":
        return _noise_pixels(pixels, limits.noise_sigma, seed), {
            "noise_sigma": limits.noise_sigma
        }
    if kind == "rotation":
        rotated = _rotate_pixels(pixels, limits.rotation_deg)
        return rotated, {
            "rotation_deg": limits.rotation_deg,
            "canvas_width_px": int(rotated.shape[1]),
            "canvas_height_px": int(rotated.shape[0]),
        }
    if kind == "perspective":
        return _perspective_pixels(pixels, limits.perspective_deg), {
            "perspective_deg": limits.perspective_deg
        }
    if kind == "downscale":
        side = max(1, int(round(limits.downscale_px_per_module * frame.module_count)))
        return _downscale_pixels(pixels, limits.downscale_px_per_module, frame.module_count), {
            "downscale_px_per_module": limits.downscale_px_per_module,
            "downscale_side_px": side,
        }
    raise ValueError(
        f"unknown damage kind {kind!r}; declared kinds, in order: {', '.join(DAMAGE_KINDS)}"
    )


# Case machinery: identity, the sample, the request, the memoised symbol and reading, and
# the two executors.  One request path, one case identifier, one assertion.


@dataclasses.dataclass(frozen=True)
class _Measurement:
    """The pristine measurements a damage case needs before it can damage anything.

    ``frame`` is the module geometry, ``reading`` the pristine decode that supplied it and
    ``decoded_ec`` the EC level the symbol itself reports.  All three are memoised per record
    so a record's seven kinds cost one reading rather than seven.
    """

    frame: _Frame
    reading: DecodeResult
    decoded_ec: str


def _case_id(record: _Record, kind: str) -> str:
    """Return the case identifier of one record's variant: ``corpus:<record-id>@<kind>``."""
    return f"corpus:{record.id}@{kind}"


def _case_seed(run_seed: int, case_id: str, kind: str) -> int:
    """Return the derived seed of one case, through the product's only derivation point.

    The namespace is ``DM-RESILIENCE`` for both checks on purpose: ``DM-SEEDED`` re-derives
    the seed ``DM-RESILIENCE`` used for the same case, and a namespace of its own would make
    the two disagree and the byte comparison meaningless.
    """
    return derive_seed(run_seed, DM_RESILIENCE, case_id, kind)


def _sample_records(ctx: RunContext) -> tuple[_Record, ...]:
    """Return the fixed eight-record sample of AAP §0.8.1, in ``id`` order.

    "The first eight records in ``id`` order whose payload length is between 8 and 64 bytes
    at the run's EC level", applied as a declared rule through
    :func:`qrcheck.checks.support.first_records`, which owns the ordering.
    """
    window = (
        record
        for record in ctx.corpus
        if _MIN_PAYLOAD_BYTES <= len(record.payload_bytes) <= _MAX_PAYLOAD_BYTES
    )
    return support.first_records(window, _SAMPLE_SIZE)


def _case_matrix(ctx: RunContext) -> tuple[tuple[_Record, str], ...]:
    """Return every ``(record, kind)`` case of the matrix, sorted by case identifier."""
    pairs = ((record, kind) for record in _sample_records(ctx) for kind in DAMAGE_KINDS)
    return tuple(sorted(pairs, key=lambda pair: _case_id(pair[0], pair[1])))


def _elapsed(started: float) -> float:
    """Return the milliseconds elapsed since `started`, for the result's ``duration_ms``.

    The only clock read in this module, and it feeds a field every report masks: no
    assertion here compares a duration (AAP §0.5.5).
    """
    return max((time.perf_counter() - started) * 1000.0, 0.0)


def _symbol_for(
    ctx: RunContext,
    record: _Record,
    request: GenerateRequest,
) -> GeneratedSymbol:
    """Return the record's symbol, encoding it once and memoising it for the other kinds.

    Case identifiers sort alphabetically, so ``@blur`` executes before ``@occluders``: the
    lazy encode is what keeps the family's declared eight invocations true whatever order the
    runner walks the cases in.
    """
    held = recall_symbol(record.id)
    if held is not None:
        return held
    symbol = support.generate(ctx, request)
    remember_symbol(record.id, symbol)
    return symbol


def _reading_for(
    ctx: RunContext,
    record: _Record,
    symbol: GeneratedSymbol,
    request: GenerateRequest,
) -> DecodeResult:
    """Return the pristine decode of the record's symbol, decoding it at most once.

    ``DM-SEEDED`` never reaches this: it takes the reading from the memo only, which is what
    keeps its invocation count at zero.
    """
    held = _recall_reading(record.id)
    if held is not None:
        return held
    reading = support.decode(ctx, symbol.image_bytes, request)
    _remember_reading(record.id, reading)
    return reading


def _version_of(reading: DecodeResult) -> int | None:
    """Return the version the sampled grid implies, or ``None`` when it implies none.

    :func:`qrcheck.decode.matrix.version_from_dimension` is the product's only version
    source, and the sampled dimension is the only input: a decoder reports no version under
    the pinned zxing-cpp, and a target's declared version is evidence rather than assertion.
    """
    if reading.version is not None:
        return int(reading.version)
    if reading.matrix is not None:
        return version_from_dimension(int(reading.matrix.shape[0]))
    return None


def _decoded_ec(reading: DecodeResult) -> str | None:
    """Return the EC level the symbol itself reports, format information first.

    Two independent readings exist (the decoder's metadata and the 15-bit format
    information); the format information is the symbol's own and is preferred, and the
    decoder's is the fallback.  Neither is compared with the requested level here -- that is
    ``SY-EC-MATCH``'s assertion, and AAP §0.4.2 keeps the two apart so a wrong EC level
    cannot cascade into the damage budget.
    """
    return reading.ec_level_format or reading.ec_level_decoder


def _measure(
    request: GenerateRequest,
    reading: DecodeResult,
    pixels: numpy.ndarray,
    check_id: str,
    case_id: str,
    seed: int,
    started: float,
) -> _Measurement | CaseResult:
    """Measure the pristine symbol, or return the result that reports why it could not be.

    The version and the symbol's own EC level are prerequisites of every kind: the module
    pitch comes from the module count, and the occlusion budget from the EC level.  A symbol
    that supplies neither leaves no budget to spend, so the case fails with
    ``sampling_failed`` -- the one branch in this module that reports a sampling failure, and
    only for the pristine pre-damage measurement.  A *damaged* image that cannot be sampled
    is never a failure here: the payload comparison is the assertion, and sampling a damaged
    grid is allowed to fail.
    """
    version = _version_of(reading)
    ec_level = _decoded_ec(reading)
    frame = None if version is None else _frame(version, pixels)
    if frame is None or ec_level is None:
        expected = {
            "module_grid": "recoverable from the pristine symbol",
            "ec_level": "reported by the symbol's format information or the decoder",
        }
        decoded = {
            "pristine": {
                "version": reading.version,
                "ec_level_decoder": reading.ec_level_decoder,
                "ec_level_format": reading.ec_level_format,
                "matrix_sha256": reading.matrix_sha256,
                "sampling": None if reading.sampling is None else reading.sampling.as_dict(),
            }
        }
        return support.fail(
            check_id,
            case_id,
            reason=Reason.SAMPLING_FAILED,
            request=request.as_dict(),
            decoded=decoded,
            expected=expected,
            duration_ms=_elapsed(started),
            seed=seed,
        )
    return _Measurement(frame=frame, reading=reading, decoded_ec=ec_level)


def _damage_evidence(
    measurement: _Measurement,
    outcome: DecodeResult,
    kind: str,
    seed: int,
    applied: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the ``decoded`` mapping of one damage case.

    The damaged symbol's own decode projection, plus one ``damage`` entry carrying the kind,
    the seed, the consensus classification, each decoder's status and polarity, the decoded
    length and the values actually applied; plus the pristine reading the budget came from,
    so a reader can see which EC level governed the occlusion.
    """
    return {
        **outcome.as_dict(),
        "damage": [
            {
                "kind": kind,
                "seed": seed,
                "agreement": outcome.agreement,
                "decoded_len": (
                    None if outcome.payload_bytes is None else len(outcome.payload_bytes)
                ),
                "per_decoder": {
                    name: {
                        "status": decoder.status,
                        "polarity": decoder.polarity,
                        "reported_ec": decoder.reported_ec,
                        "error_text": decoder.error_text,
                    }
                    for name, decoder in outcome.per_decoder.items()
                },
                "applied": dict(applied),
            }
        ],
        "pristine": {
            "ec_level_decoder": measurement.reading.ec_level_decoder,
            "ec_level_format": measurement.reading.ec_level_format,
            "decoded_ec": measurement.decoded_ec,
            "version": measurement.reading.version,
            "matrix_sha256": measurement.reading.matrix_sha256,
        },
    }


def _expectation(
    request: GenerateRequest,
    kind: str,
    seed: int,
    applied: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the ``expected`` mapping: what the case asserted, and the values it applied."""
    return {
        "payload_sha256": hashlib.sha256(request.payload).hexdigest(),
        "payload_len": len(request.payload),
        "damage_kind": kind,
        "damage_seed": seed,
        "applied": dict(applied),
        "outcome": "the primary decoder returns the requested payload",
    }


def _request_extra(kind: str, seed: int, applied: Mapping[str, Any]) -> dict[str, Any]:
    """Return the facts ``request.json`` records beside the request itself."""
    return {"damage": {"kind": kind, "seed": seed, "applied": dict(applied)}}


def _first_difference(left: bytes, right: bytes) -> int | None:
    """Return the offset of the first differing byte, or ``None`` when the two are equal."""
    for offset, (mine, theirs) in enumerate(zip(left, right)):
        if mine != theirs:
            return offset
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _assert_payload(
    check_id: str,
    case_id: str,
    record: _Record,
    request: GenerateRequest,
    outcome: DecodeResult,
    evidence: dict[str, Any],
    expected: dict[str, Any],
    duration: float,
    seed: int,
) -> CaseResult:
    """Return the pass or the payload mismatch of one damaged symbol.

    The assertion is the primary decoder's payload, exactly: ``outcome.payload_bytes`` is
    the primary decoder's whenever it produced one (AAP §0.3.2's one-authority rule), so a
    symbol the authority could not read -- or read to different bytes -- fails, and the
    secondary decoder is never the one that decides.
    """
    if outcome.payload_bytes == record.payload_bytes:
        return support.pass_(
            check_id,
            case_id,
            request=request.as_dict(),
            decoded=evidence,
            expected=expected,
            duration_ms=duration,
            seed=seed,
        )
    return support.fail(
        check_id,
        case_id,
        reason=Reason.VALUE_MISMATCH,
        request=request.as_dict(),
        decoded=evidence,
        expected=expected,
        duration_ms=duration,
        seed=seed,
    )


def _run_resilience_case(ctx: RunContext, record: _Record, kind: str) -> CaseResult:
    """Damage one record's symbol with `kind` and assert the payload survives it.

    The flow is the family's contract in order: build the one legal request, take the
    record's memoised symbol (encoding it on first touch), take the memoised pristine reading
    for the module count and the symbol's own EC level, apply the kind, record the variant so
    ``DM-SEEDED`` can compare against it, decode the damaged image once, and compare the
    primary decoder's payload with the requested one.

    Raises:
        TargetUnavailable: the target could not be reached; propagated to the runner, which
            owns the one-error-result-per-check policy on target loss.
        TargetTimeout: no response within the configured timeout; propagated for the same
            reason.
    """
    case_id = _case_id(record, kind)
    seed = _case_seed(ctx.seed, case_id, kind)
    request = support.build(record.payload_bytes, ctx.ec_level)
    started = time.perf_counter()
    try:
        symbol = _symbol_for(ctx, record, request)
        reference = mux.decode_image(symbol.image_bytes)
        pixels = numpy.asarray(reference.convert("RGB"), dtype=numpy.uint8)
        measurement = _measure(
            request,
            _reading_for(ctx, record, symbol, request),
            pixels,
            DM_RESILIENCE,
            case_id,
            seed,
            started,
        )
        if isinstance(measurement, CaseResult):
            return support.record_evidence(ctx, measurement, symbol=symbol)
        damaged, applied = _damaged_pixels(
            kind,
            pixels,
            frame=measurement.frame,
            limits=ctx.config.damage,
            seed=seed,
            decoded_ec=measurement.decoded_ec,
            reference=reference,
        )
        variant = mux.encode_png(_raster(damaged, reference))
        remember_variant(record.id, kind, variant)
    except (
        TargetUnavailable,
        TargetTimeout,
        TargetRejected,
        TargetContractError,
        TargetUnsupported,
    ) as error:
        # The single translation of an adapter error into a status: it re-raises the two
        # target-loss errors to the runner and reports the other three as the AAP
        # prescribes.
        return support.record_evidence(
            ctx,
            support.case_status_for_error(
                DM_RESILIENCE,
                case_id,
                error,
                request=request,
                seed=seed,
                duration_ms=_elapsed(started),
            ),
        )
    outcome = support.decode(ctx, variant, request)
    duration = _elapsed(started)
    evidence = _damage_evidence(measurement, outcome, kind, seed, applied)
    expected = _expectation(request, kind, seed, applied)
    result = _assert_payload(
        DM_RESILIENCE, case_id, record, request, outcome, evidence, expected, duration, seed
    )
    if result.status is not Status.PASSED:
        return support.record_evidence(
            ctx,
            result,
            symbol=symbol,
            damage_variants=((kind, variant),),
            request_extra=_request_extra(kind, seed, applied),
        )
    return result


def _run_seeded_case(ctx: RunContext, record: _Record, kind: str) -> CaseResult:
    """Regenerate one recorded variant from its seed and assert it is byte-identical.

    No adapter call: the symbol comes from the memo ``DM-RESILIENCE`` filled, the damage is
    re-applied with the same derived seed and the same resolved thresholds, and the encoded
    result is compared with the bytes ``DM-RESILIENCE`` recorded before decoding it and
    asserting the payload again.  When the memo holds nothing -- ``DM-SEEDED`` selected
    without ``DM-RESILIENCE`` -- the case is skipped with reason ``missing_dependency``,
    which is counted separately and never as a pass.
    """
    case_id = _case_id(record, kind)
    seed = _case_seed(ctx.seed, case_id, kind)
    request = support.build(record.payload_bytes, ctx.ec_level)
    started = time.perf_counter()
    symbol = recall_symbol(record.id)
    reading = _recall_reading(record.id)
    recorded = recall_variant(record.id, kind)
    if symbol is None or reading is None or recorded is None:
        return support.record_evidence(
            ctx,
            support.skip(
                DM_SEEDED,
                case_id,
                reason=Reason.MISSING_DEPENDENCY,
                request=request.as_dict(),
                decoded={
                    "prerequisite": DM_RESILIENCE,
                    "held_symbol": symbol is not None,
                    "held_reading": reading is not None,
                    "held_variant": recorded is not None,
                },
                expected={
                    "outcome": (
                        f"the {kind} variant regenerated from the recorded seed is "
                        f"byte-identical to the one {DM_RESILIENCE} recorded"
                    )
                },
                duration_ms=_elapsed(started),
                seed=seed,
            ),
        )
    reference = mux.decode_image(symbol.image_bytes)
    pixels = numpy.asarray(reference.convert("RGB"), dtype=numpy.uint8)
    measurement = _measure(request, reading, pixels, DM_SEEDED, case_id, seed, started)
    if isinstance(measurement, CaseResult):
        return support.record_evidence(ctx, measurement, symbol=symbol)
    damaged, applied = _damaged_pixels(
        kind,
        pixels,
        frame=measurement.frame,
        limits=ctx.config.damage,
        seed=seed,
        decoded_ec=measurement.decoded_ec,
        reference=reference,
    )
    regenerated = mux.encode_png(_raster(damaged, reference))
    outcome = support.decode(ctx, regenerated, request)
    duration = _elapsed(started)
    evidence = _damage_evidence(measurement, outcome, kind, seed, applied)
    evidence["variant"] = {
        "regenerated_sha256": hashlib.sha256(regenerated).hexdigest(),
        "recorded_sha256": hashlib.sha256(recorded).hexdigest(),
        "regenerated_len": len(regenerated),
        "recorded_len": len(recorded),
        "first_difference": _first_difference(regenerated, recorded),
    }
    expected = _expectation(request, kind, seed, applied)
    expected["variant"] = (
        "byte-identical to the artefact recorded under damage-<kind>-<seed>.png"
    )
    if regenerated != recorded:
        failed = support.fail(
            DM_SEEDED,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded=evidence,
            expected=expected,
            duration_ms=duration,
            seed=seed,
        )
        return support.record_evidence(
            ctx,
            failed,
            symbol=symbol,
            damage_variants=((kind, regenerated),),
            request_extra=_request_extra(kind, seed, applied),
        )
    result = _assert_payload(
        DM_SEEDED, case_id, record, request, outcome, evidence, expected, duration, seed
    )
    if result.status is not Status.PASSED:
        return support.record_evidence(
            ctx,
            result,
            symbol=symbol,
            damage_variants=((kind, regenerated),),
            request_extra=_request_extra(kind, seed, applied),
        )
    return result


def _resilience_case(record: _Record, kind: str) -> Case:
    """Return the case of one ``(record, kind)`` pair of ``DM-RESILIENCE``."""

    def execute(ctx: RunContext) -> CaseResult:
        """Run this case against the run's context."""
        return _run_resilience_case(ctx, record, kind)

    return Case(case_id=_case_id(record, kind), execute=execute)


def _seeded_case(record: _Record, kind: str) -> Case:
    """Return the case of one ``(record, kind)`` pair of ``DM-SEEDED``."""

    def execute(ctx: RunContext) -> CaseResult:
        """Run this case against the run's context."""
        return _run_seeded_case(ctx, record, kind)

    return Case(case_id=_case_id(record, kind), execute=execute)


@check(
    id=DM_RESILIENCE,
    title="Damage resilience: seven kinds of damage leave the payload readable",
    family="damage",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def dm_resilience_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the 56 cases of ``DM-RESILIENCE``: eight records by seven damage kinds.

    Enumerating clears the run-scoped memo first, which is what stops a second run in the
    same process from inheriting the first one's symbols; the cases are yielded in case-
    identifier order, so the walk never depends on the order this generator built them in.
    """
    clear_symbol_memo()
    for record, kind in _case_matrix(ctx):
        yield _resilience_case(record, kind)


@check(
    id=DM_SEEDED,
    title="Damage reproducibility: the recorded seed regenerates the damaged artefact",
    family="damage",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def dm_seeded_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the 56 cases of ``DM-SEEDED``: the same matrix as ``DM-RESILIENCE``.

    The identifiers are the same because the cases *are* the same cases: this check
    re-applies each one's damage to the symbol the earlier check already holds and compares
    the bytes, so its case set must match one for one.  The evidence directory differs, since
    the check identifier is a path segment above the failure identifier.
    """
    for record, kind in _case_matrix(ctx):
        yield _seeded_case(record, kind)
