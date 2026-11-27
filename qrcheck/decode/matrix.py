"""Module-grid sampling: the module count, the grid and the version, from pixels.

No pinned decoder reports a version, yet ``SY-VERSION-MIN``, ``SY-EC-MATCH``, ``SY-MASK``,
``SY-MATRIX-EQUIV`` and the measured quiet zone all need the module grid, so the grid is recovered
from the symbol itself. ISO/IEC 18004:2015(E) §6.3.2.1 gives a Model 2 symbol 21 modules per side
at Version 1 and four more per version, ``4V + 17`` in closed form, and
:func:`version_from_dimension` inverts that rule; it is the product's only version source.

Three acceptance paths and nothing else: the authoritative decoder's corner quadrangle projected by
a planar homography, pitched from the finder pattern, snapped to a legal ``4V + 17`` and confirmed
against the timing patterns; the same procedure on the secondary decoder's rectified
``straight_qrcode``, labelled ``source="straight_qrcode"``; or ``matrix=None`` with
``reason="sampling_failed"``, which a check reports instead of a wrong verdict. Neither guard may be
relaxed: the snapped count's implied pitch must lie within 20% of the measured pitch, and the timing
patterns must show exactly ``N - 16`` alternations on both axes.

Standard library and numpy only -- no ``cv2`` -- and deterministic. A failure on target-derived data
is a value, never an exception: one of ``position_unusable``, ``low_contrast``,
``finder_unreadable``, ``pitch_mismatch``, ``timing_mismatch`` or ``fallback_unusable`` on the
estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy

from qrcheck.decode.base import Matrix, ModuleCountEstimate, SampledMatrix

__all__ = [
    "MAX_MODULE_COUNT",
    "MAX_VERSION",
    "MIN_MODULE_COUNT",
    "MIN_VERSION",
    "estimate_module_count",
    "module_count_for_version",
    "sample_matrix",
    "version_from_dimension",
]

# ISO/IEC 18004:2015(E) §6.3.2.1: a Model 2 symbol is identified by a version in 1..40, and
# its dimension is exactly 4V + 17 modules per side. Micro QR (M1-M4) is out of
# scope, so no second dimension rule exists in this product.
MIN_VERSION: Final = 1
MAX_VERSION: Final = 40
MIN_MODULE_COUNT: Final = 4 * MIN_VERSION + 17  # 21 modules, Version 1
MAX_MODULE_COUNT: Final = 4 * MAX_VERSION + 17  # 177 modules, Version 40

# ISO/IEC 18004:2015(E) §6.3.3.1: each of the three finder patterns is a 7x7 module square (dark
# outer ring, light ring, 3x3 dark core), so the dark run across its middle row and down its middle
# column is exactly 7 modules wide. That is the measured quantity the module pitch is derived from.
FINDER_MODULE_COUNT: Final = 7

# ISO/IEC 18004:2015(E) §6.3.5: the timing patterns are the alternating dark/light lines between the
# finder patterns -- module row 6 horizontally and module column 6 vertically. Counting their
# alternations is the independent confirmation that the chosen module count is right.
TIMING_ROW: Final = 6
TIMING_COLUMN: Final = 6

# The first guard on a snapped module count: the pitch implied by the candidate
# count must agree with the pitch measured from the finder pattern. A count that
# is wrong by a module or more moves the implied pitch by far more than this.
_PITCH_TOLERANCE: Final = 0.20

# The 10th-to-90th luminance percentile spread below which no threshold can separate dark from light
# modules: under 8 grey levels a symbol is unreadable however it is sampled.
_MIN_CONTRAST_SPREAD: Final = 8.0

# A quadrangle whose shoelace area is below one square pixel cannot anchor a
# homography, so its solve is refused before the matrix is inverted.
_MIN_QUAD_AREA: Final = 1.0

# The module pitch, in rectified pixels, below which a 3x3 mean around a module centre would reach
# into a neighbouring module; sampling then uses the centre pixel alone. Only the smaller guards the
# exactness of a 1 px/module image such as the secondary decoder's ``straight_qrcode``.
_MIN_AVERAGING_PITCH: Final = 3.0

# How many parallel lines the finder run is measured on, per axis. The outermost row and column of a
# rectified square sit on the symbol's edge, where a rotated or resampled symbol blends one module
# with the quiet zone; measuring them alone would let a single antialiased pixel row report a zero
# run. Every one of these lines lies inside the finder's own dark square -- its first module row, or
# at one pixel per module a ring row of the finder itself -- so the longest reading is the finder's
# width and cannot come from the data region.
_FINDER_PROBE_LINES: Final = 3


def module_count_for_version(version: int) -> int:
    """Return the module count ``N = 4V + 17`` of a Model 2 symbol
    (ISO/IEC 18004:2015(E) §6.3.2.1): ``21`` for Version 1, ``177`` for Version 40.

    Raises:
        ValueError: ``version`` is not an integer in 1-40, so a Micro QR version has no module
            count here.
    """
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"version must be an integer; got {version!r}")
    if version < MIN_VERSION or version > MAX_VERSION:
        raise ValueError(f"version must be in {MIN_VERSION}..{MAX_VERSION}; got {version}")
    return 4 * version + 17


def version_from_dimension(module_count: int) -> int | None:
    """Return the version implied by a sampled dimension, or ``None`` if none is.

    The product's only version source: ISO/IEC 18004:2015(E) §6.3.2.1's ``N = 4V + 17`` inverted,
    never a decoder -- the pinned ``zxing-cpp`` 2.2.0 reports none -- and never a target's metadata.
    ``None`` covers a ``module_count - 17`` that is not a multiple of four, a version outside 1-40,
    and the negative offsets a Micro QR dimension would produce.
    """
    if isinstance(module_count, bool) or not isinstance(module_count, int):
        return None
    if module_count < MIN_MODULE_COUNT or module_count > MAX_MODULE_COUNT:
        return None
    offset = module_count - 17
    if offset % 4 != 0:
        return None
    return offset // 4


def _rejected(reason: str) -> ModuleCountEstimate:
    """Return the estimate that records nothing but why sampling was refused."""
    return ModuleCountEstimate(
        n_est=None,
        pitch=None,
        snapped_n=None,
        finder_run_px=None,
        timing_transitions=None,
        accepted=False,
        reason=reason,
    )


def _as_luma(image: numpy.ndarray) -> numpy.ndarray:
    """Return ``image`` as a two-dimensional ``float64`` luminance array.

    A colour array is reduced with the ITU-R BT.601 weights ``0.299 R + 0.587 G + 0.114 B`` --
    thresholding weights, deliberately not the WCAG relative luminance the contrast check uses.
    ``ValueError`` when ``image`` is not ``numpy.uint8`` or is neither ``H x W`` nor ``H x W x 3``:
    a producer defect, not a sampling failure.
    """
    array = numpy.asarray(image)
    if array.dtype != numpy.uint8:
        raise ValueError(f"image must be uint8; got dtype {array.dtype}")
    if array.ndim == 2:
        return array.astype(numpy.float64)
    if array.ndim == 3 and array.shape[2] == 3:
        channels = array.astype(numpy.float64)
        return 0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]
    raise ValueError(f"image must have shape (H, W) or (H, W, 3); got {array.shape}")


def _quad_area(quad: numpy.ndarray) -> float:
    """Return the shoelace area of a quadrangle given in traversal order."""
    x = quad[:, 0]
    y = quad[:, 1]
    cross = float(numpy.dot(x, numpy.roll(y, -1)) - numpy.dot(y, numpy.roll(x, -1)))
    return abs(cross) / 2.0


def _canonical_quad(position: Sequence[tuple[float, float]] | None) -> numpy.ndarray | None:
    """Return the four corners clockwise from the top-left, or ``None``.

    A decoder may report its quadrangle from any corner in either direction; ordering it canonically
    -- top-left smallest ``x + y``, bottom-right largest, top-right smallest ``y - x``, bottom-left
    largest -- keeps the rectified orientation stable, so the same symbol always samples to the same
    digest.
    """
    if position is None:
        return None
    try:
        points = numpy.asarray(position, dtype=numpy.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
        return None
    corners = points[:4]
    if not numpy.all(numpy.isfinite(corners)):
        return None
    sums = corners[:, 0] + corners[:, 1]
    differences = corners[:, 1] - corners[:, 0]
    indices = (
        int(numpy.argmin(sums)),
        int(numpy.argmin(differences)),
        int(numpy.argmax(sums)),
        int(numpy.argmax(differences)),
    )
    if len(set(indices)) != 4:
        return None
    quad = corners[list(indices)]
    if _quad_area(quad) < _MIN_QUAD_AREA:
        return None
    return quad


def _square_to_quad(quad: numpy.ndarray, side: float) -> numpy.ndarray | None:
    """Solve the homography mapping a ``side x side`` square onto ``quad``.

    The 8x8 direct-linear-transform system is solved with :func:`numpy.linalg.solve` rather than an
    image library's transform, because ``cv2`` is optional and a run must still sample grids in
    single-decoder mode. ``None`` for a singular or non-finite system: a degenerate quadrangle, not
    a target defect.
    """
    square = numpy.array(
        [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)], dtype=numpy.float64
    )
    coefficients = numpy.zeros((8, 8), dtype=numpy.float64)
    targets = numpy.zeros(8, dtype=numpy.float64)
    for index, ((u, v), (x, y)) in enumerate(zip(square, quad)):
        row = 2 * index
        coefficients[row] = (u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x)
        coefficients[row + 1] = (0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y)
        targets[row] = x
        targets[row + 1] = y
    try:
        solved = numpy.linalg.solve(coefficients, targets)
    except numpy.linalg.LinAlgError:
        return None
    if not numpy.all(numpy.isfinite(solved)):
        return None
    return numpy.append(solved, 1.0).reshape(3, 3)


def _bilinear(luma: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Sample ``luma`` at the continuous coordinates ``(x, y)`` by bilinear interpolation.

    Pixel centres sit at ``index + 0.5``, so each coordinate is shifted by half a pixel before its
    four taps are taken; coordinates outside the image clip to the nearest edge pixel rather than
    wrapping.
    """
    height, width = luma.shape
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    sx = numpy.clip(x - 0.5, 0.0, max_x)
    sy = numpy.clip(y - 0.5, 0.0, max_y)
    x0 = numpy.floor(sx).astype(numpy.intp)
    y0 = numpy.floor(sy).astype(numpy.intp)
    x1 = numpy.minimum(x0 + 1, max_x).astype(numpy.intp)
    y1 = numpy.minimum(y0 + 1, max_y).astype(numpy.intp)
    fraction_x = sx - x0
    fraction_y = sy - y0
    top = luma[y0, x0] * (1.0 - fraction_x) + luma[y0, x1] * fraction_x
    bottom = luma[y1, x0] * (1.0 - fraction_x) + luma[y1, x1] * fraction_x
    return top * (1.0 - fraction_y) + bottom * fraction_y


def _rectify(luma: numpy.ndarray, quad: numpy.ndarray) -> tuple[numpy.ndarray, int] | None:
    """Project the symbol bounded by ``quad`` onto a square luminance image.

    The side is the mean of the quadrangle's four edge lengths, clipped to at least
    :data:`MIN_MODULE_COUNT`. The quadrangle bounds the symbol, not the 4-module quiet zone of
    ISO/IEC 18004:2015(E) §6.3.8, so the square holds the symbol area only. ``None`` when the
    homography is singular, the inputs are empty, or a projected coordinate is not finite.
    """
    if luma.size == 0:
        return None
    edges = numpy.linalg.norm(numpy.roll(quad, -1, axis=0) - quad, axis=1)
    side = max(int(round(float(numpy.mean(edges)))), MIN_MODULE_COUNT)
    homography = _square_to_quad(quad, float(side))
    if homography is None:
        return None
    centres = numpy.arange(side, dtype=numpy.float64) + 0.5
    u, v = numpy.meshgrid(centres, centres)
    denominator = homography[2, 0] * u + homography[2, 1] * v + homography[2, 2]
    if numpy.any(numpy.abs(denominator) < 1e-12):
        return None
    x = (homography[0, 0] * u + homography[0, 1] * v + homography[0, 2]) / denominator
    y = (homography[1, 0] * u + homography[1, 1] * v + homography[1, 2]) / denominator
    if not (numpy.all(numpy.isfinite(x)) and numpy.all(numpy.isfinite(y))):
        return None
    return _bilinear(luma, x, y), side


def _dark_threshold(luma: numpy.ndarray) -> float | None:
    """Return the luminance threshold separating dark modules from light ones.

    The midpoint of the 10th and 90th luminance percentiles, so a stray specular pixel cannot drag
    it. ``None`` when the image is empty or its spread is narrower than :data:`_MIN_CONTRAST_SPREAD`
    grey levels, which no sampling can repair.
    """
    if luma.size == 0:
        return None
    low, high = numpy.percentile(luma, (10.0, 90.0))
    if float(high) - float(low) < _MIN_CONTRAST_SPREAD:
        return None
    return float((float(low) + float(high)) / 2.0)


def _leading_dark_run(pixels: numpy.ndarray) -> int:
    """Return the number of dark pixels at the start of a one-dimensional run."""
    run = 0
    for pixel in pixels:
        if not pixel:
            break
        run += 1
    return run


def _finder_run_px(dark: numpy.ndarray) -> int:
    """Measure the top-left finder pattern's dark run, in pixels.

    The rectified square starts at the symbol's top-left corner, so the dark run from ``(0, 0)``
    along row 0 and column 0 spans ISO/IEC 18004:2015(E) §6.3.3.1's 7x7 finder's first module row
    and column, the 1-module light separator at index 7 terminating it. The first
    :data:`_FINDER_PROBE_LINES` lines of each axis are read and the longest taken, because the
    outermost line lies on the symbol's edge and a blended row would refuse a samplable symbol.
    """
    rows = min(_FINDER_PROBE_LINES, int(dark.shape[0]))
    columns = min(_FINDER_PROBE_LINES, int(dark.shape[1]))
    run_row = max(_leading_dark_run(dark[index, :]) for index in range(rows))
    run_column = max(_leading_dark_run(dark[:, index]) for index in range(columns))
    if run_row == 0 or run_column == 0:
        return 0
    return int(round((run_row + run_column) / 2.0))


def _snap_module_count(n_est: int, pitch: float, side: int) -> int | None:
    """Snap an estimated module count to a legal ``4V + 17``, or refuse to.

    The nearest legal count is accepted only when the pitch it implies, ``side / N``, lies within
    :data:`_PITCH_TOLERANCE` of the pitch measured from the finder pattern; without that guard a
    count off by a module would round to a plausible neighbour and yield a confidently shifted grid.
    """
    # Every legal count is a candidate, so an estimate far outside 1..40 (a broken pitch) still
    # snaps to the closest legal dimension and is then refused by the pitch guard instead of being
    # clipped to a boundary value. A tie keeps the smaller count, which is the one tie-break that
    # does not depend on a rounding mode.
    nearest: int | None = None
    best = 0
    for candidate in range(MIN_MODULE_COUNT, MAX_MODULE_COUNT + 1, 4):
        distance = abs(candidate - n_est)
        if nearest is None or distance < best:
            nearest, best = candidate, distance
    if nearest is None:
        return None
    implied_pitch = side / nearest
    if abs(implied_pitch - pitch) > _PITCH_TOLERANCE * pitch:
        return None
    return nearest


def _sample_grid(
    luma: numpy.ndarray, threshold: float, module_count: int, side: int
) -> Matrix:
    """Sample one luminance value per module centre and threshold it into a grid.

    Module ``(r, c)`` is read at its centre, each sample the mean of a 3x3 neighbourhood that
    absorbs a pixel of quadrangle jitter, shrinking to the centre pixel below
    :data:`_MIN_AVERAGING_PITCH` where that mean would reach into the neighbouring module. Returns a
    C-order ``numpy.bool_`` array of shape ``(N, N)``, ``True`` for a dark module.
    """
    pitch = side / module_count
    centres = numpy.floor((numpy.arange(module_count, dtype=numpy.float64) + 0.5) * pitch)
    centres = numpy.clip(centres, 0, side - 1).astype(numpy.intp)
    radius = 1 if pitch >= _MIN_AVERAGING_PITCH else 0
    total = numpy.zeros((module_count, module_count), dtype=numpy.float64)
    taps = 0
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            rows = numpy.clip(centres + offset_y, 0, side - 1)
            columns = numpy.clip(centres + offset_x, 0, side - 1)
            total += luma[numpy.ix_(rows, columns)]
            taps += 1
    return numpy.ascontiguousarray((total / taps) < threshold)


def _count_changes(values: numpy.ndarray) -> int:
    """Return the number of adjacent pairs in ``values`` that differ."""
    return int(numpy.count_nonzero(values[1:] != values[:-1]))


def _timing_transitions(grid: Matrix) -> tuple[int, int]:
    """Count the horizontal and vertical timing-pattern alternations of ``grid``.

    ISO/IEC 18004:2015(E) §6.3.5 alternates once per module between the finder patterns: over
    columns ``7 .. N-9`` that is ``N - 15`` samples and ``N - 16`` adjacent pairs, each of which
    must alternate, and the same holds down column 6 over rows ``7 .. N-9``. Any other reading
    means a shifted grid and a refused count.
    """
    module_count = int(grid.shape[0])
    stop = module_count - 8
    horizontal = _count_changes(grid[TIMING_ROW, 7:stop])
    vertical = _count_changes(grid[7:stop, TIMING_COLUMN])
    return horizontal, vertical


def _analyse(rectified: numpy.ndarray, side: int) -> tuple[ModuleCountEstimate, Matrix | None]:
    """Measure, snap and confirm a module count for an already-rectified square.

    ``rectified``'s top-left pixel is the symbol's top-left module and ``side`` its length in
    pixels. Returns ``(estimate, grid)``, where ``grid`` is the sampled matrix only when the
    estimate was accepted, the estimate's ``reason`` naming what refused it otherwise.
    """
    threshold = _dark_threshold(rectified)
    if threshold is None:
        return _rejected("low_contrast"), None
    finder_run = _finder_run_px(rectified < threshold)
    if finder_run <= 0:
        return _rejected("finder_unreadable"), None
    pitch = finder_run / FINDER_MODULE_COUNT
    n_est = max(int(round(side / pitch)), 1)
    snapped = _snap_module_count(n_est, pitch, side)
    if snapped is None:
        return (
            ModuleCountEstimate(
                n_est=n_est,
                pitch=pitch,
                snapped_n=None,
                finder_run_px=finder_run,
                timing_transitions=None,
                accepted=False,
                reason="pitch_mismatch",
            ),
            None,
        )
    grid = _sample_grid(rectified, threshold, snapped, side)
    transitions = _timing_transitions(grid)
    expected = snapped - 16
    accepted = transitions[0] == expected and transitions[1] == expected
    estimate = ModuleCountEstimate(
        n_est=n_est,
        pitch=pitch,
        snapped_n=snapped,
        finder_run_px=finder_run,
        timing_transitions=transitions,
        accepted=accepted,
        reason=None if accepted else "timing_mismatch",
    )
    return estimate, (grid if accepted else None)


def _try_luma(array: numpy.ndarray | None) -> numpy.ndarray | None:
    """Return ``array`` as luminance, or ``None`` when it is not a usable image.

    The secondary decoder's rectified image is target-derived data, so an unexpected shape or dtype
    makes that path unusable rather than aborting the sample.
    """
    if array is None:
        return None
    try:
        luma = _as_luma(array)
    except (TypeError, ValueError):
        return None
    return luma


def estimate_module_count(
    image: numpy.ndarray, position: Sequence[tuple[float, float]] | None = None
) -> ModuleCountEstimate:
    """Estimate the module count of the symbol bounded by ``position``.

    The measurement half of :func:`sample_matrix`: the quadrangle is ordered canonically and
    rectified, thresholded between its 10th and 90th luminance percentiles, pitched from the
    top-left finder's dark run (``run / 7``), ``round(side / pitch)`` snapped to a legal ``4V + 17``
    within :data:`_PITCH_TOLERANCE`, and the snap confirmed by ``N - 16`` timing alternations on
    both axes.

    Raises:
        ValueError: ``image`` violates this module's input contract -- a producer defect, not
            evidence about a target.
    """
    luma = _as_luma(image)
    quad = _canonical_quad(position)
    if quad is None:
        return _rejected("position_unusable")
    rectified = _rectify(luma, quad)
    if rectified is None:
        return _rejected("position_unusable")
    estimate, _grid = _analyse(*rectified)
    return estimate


def sample_matrix(
    image: numpy.ndarray,
    position: Sequence[tuple[float, float]] | None = None,
    *,
    straight_qrcode: numpy.ndarray | None = None,
    keep_rectified: bool = False,
) -> SampledMatrix:
    """Sample the module grid of the symbol in ``image``, or report why not.

    The primary path first -- rectify the quadrangle ``position`` bounds and measure, snap and
    confirm as :func:`estimate_module_count` does, labelling ``source`` ``"primary"`` -- then the
    fallback on the secondary decoder's rectified ``straight_qrcode``, labelled
    ``"straight_qrcode"``. Otherwise ``matrix``, ``module_count`` and ``version`` are ``None`` and
    ``reason`` is ``"sampling_failed"``.

    Raises:
        ValueError: ``image`` violates this module's input contract; ``straight_qrcode`` never
            raises, being decoder output derived from target data.
    """
    luma = _as_luma(image)
    rectified_luma: numpy.ndarray | None = None
    estimate = _rejected("position_unusable")
    quad = _canonical_quad(position)
    if quad is not None:
        rectified = _rectify(luma, quad)
        if rectified is not None:
            rectified_luma, side = rectified
            estimate, grid = _analyse(rectified_luma, side)
            if grid is not None:
                module_count = int(grid.shape[0])
                return SampledMatrix(
                    matrix=grid,
                    version=version_from_dimension(module_count),
                    module_count=module_count,
                    reason=None,
                    estimate=estimate,
                    source="primary",
                    rectified_luma=rectified_luma if keep_rectified else None,
                )
    if straight_qrcode is not None:
        fallback = _try_luma(straight_qrcode)
        if fallback is None or fallback.shape[0] != fallback.shape[1] or fallback.size == 0:
            estimate = _rejected("fallback_unusable")
        else:
            rectified_luma = fallback
            estimate, grid = _analyse(fallback, int(fallback.shape[0]))
            if grid is not None:
                module_count = int(grid.shape[0])
                return SampledMatrix(
                    matrix=grid,
                    version=version_from_dimension(module_count),
                    module_count=module_count,
                    reason=None,
                    estimate=estimate,
                    source="straight_qrcode",
                    rectified_luma=rectified_luma if keep_rectified else None,
                )
    return SampledMatrix(
        matrix=None,
        version=None,
        module_count=None,
        reason="sampling_failed",
        estimate=estimate,
        source="straight_qrcode" if straight_qrcode is not None else "primary",
        rectified_luma=rectified_luma if keep_rectified else None,
    )
