"""Grid sampling: ``N = 4V + 17`` read from pixels, the snap, the fallback and ``sampling_failed``.

:mod:`qrcheck.decode.matrix` is the product's **only** version source. The pinned
``zxing-cpp`` 2.2.0 reports no version and no mask, and the secondary decoder reports
neither, yet ``SY-VERSION-MIN``, ``SY-MASK``, ``SY-MATRIX-EQUIV`` and the measured quiet
zone all read the module grid -- so one sampling defect corrupts four checks at once while
each of them keeps returning a confident verdict. That failure mode, described in AAP
§0.8.1 ("How ``N`` is chosen, and what happens when it cannot be"), is what this module
exists to prevent, and it is why the refusal paths below are exercised as carefully as the
happy path: ``sampling_failed`` is a diagnosis, a shifted grid is a wrong answer.

Every image is synthesised in-fixture. :func:`segno.make_qr` supplies the module matrix,
``segno`` renders it at a known module size and quiet zone, and
:mod:`qrcheck.evidence.mux` is the image boundary in both directions
(:func:`~qrcheck.evidence.mux.decode_image` for the pixel array,
:func:`~qrcheck.evidence.mux.encode_png` for the inverted artefact the quiet-zone case
needs). Because the geometry is therefore known exactly, the four symbol corners are
handed to the sampler as its ``position`` quad instead of being obtained from a decoder,
so no decoder is needed for the pixel-level cases and nothing here touches the network, a
subprocess or a file outside ``tmp_path``.

What is asserted, and why each assertion carries weight:

* the sampled grid is element for element the oracle's matrix and its dimension is
  ``4V + 17`` -- Version 40 (177 modules) is included because an implementation that snaps
  to 178 produces a grid one module too wide and every downstream matrix assertion fails
  with a confusing message;
* :func:`~qrcheck.decode.matrix.version_from_dimension` inverts
  :func:`~qrcheck.decode.matrix.module_count_for_version` over Versions 1-40 and refuses
  every illegal dimension instead of silently snapping it;
* the finder pattern's dark run, the pitch it implies, the snap to a legal count and the
  timing-pattern confirmation are all measured, so a pitch estimator that reads seven
  modules as six fails here rather than inside four other checks;
* a count the pitch guard accepts but the timing pattern does not is **refused**, and the
  refusal keeps its numbers, the rectified square and the transition counts, which is what
  the evidence store needs to explain it;
* the ``straight_qrcode`` fallback recovers the grid when no quadrangle is available, and
  marks the path it took;
* when both paths fail the result is ``None`` with ``reason="sampling_failed"``, and the
  matrix-dependent checks report that reason rather than a verdict;
* sampling imports no decoder, so it survives single-decoder mode (AAP §0.3.2).

No threshold of ``qrcheck.toml`` appears in this file (AAP §0.4.3): the rendering
parameters are this module's own fixture constants, and the module-size tolerance belongs
to ``RN-MODULE-SIZE`` rather than to sampling.
"""

from __future__ import annotations

import ast
import importlib
import io
from pathlib import Path
from typing import Any, Final

import numpy
import pytest
import segno

from qrcheck.checks.render import measure_module_px, quiet_zone_cases
from qrcheck.checks.symbol import SY_VERSION_MIN
from qrcheck.decode.base import Matrix, ModuleCountEstimate, SampledMatrix, matrix_sha256
from qrcheck.decode.matrix import (
    FINDER_MODULE_COUNT,
    MAX_MODULE_COUNT,
    MAX_VERSION,
    MIN_MODULE_COUNT,
    MIN_VERSION,
    TIMING_COLUMN,
    TIMING_ROW,
    estimate_module_count,
    module_count_for_version,
    sample_matrix,
    version_from_dimension,
)
from qrcheck.evidence.mux import decode_image, encode_png
from qrcheck.report.model import Reason, Status

# ---------------------------------------------------------------------------
# Fixture constants: the known geometry every case is rendered at
#
# Deterministic by construction -- integer pixels per module and a fixed quiet zone -- so
# the module pitch the sampler measures is known in advance and can be asserted exactly
# rather than compared against a second measurement. The payload is deliberately short
# enough to fit Version 1 at every error level, so one payload serves every sampled pair.
# ---------------------------------------------------------------------------

#: Pixels per module of every rendered symbol. Two is the smallest pitch at which the
#: sampler's 3x3 averaging neighbourhood is still used, so the pixel-level cases exercise
#: the path a real artefact takes rather than the one-pixel-per-module shortcut.
_MODULE_SIZE_PX: Final = 2

#: Quiet zone, in modules, around every rendered symbol. Four is the ISO/IEC 18004 default,
#: and it only places the symbol inside the image: a corner quadrangle bounds the symbol
#: itself, so the sampler never sees this border.
_QUIET_ZONE_MODULES: Final = 4

#: The payload every rendered symbol carries. Three alphanumeric characters fit Version 1
#: at error level H, so the same payload can be pinned into any sampled (version, EC) pair.
_PAYLOAD: Final = b"QR1"

#: The (version, error level) pairs sampled end to end, in sorted order. AAP §0.8.1 names
#: Versions 1, 2, 7 and 10; Version 40 is added because its 177-module dimension is the
#: case that punishes an off-by-one in the ``4V + 17`` snap.
_SAMPLE_VERSIONS: Final[tuple[tuple[int, str], ...]] = (
    (1, "L"),
    (1, "M"),
    (2, "M"),
    (7, "Q"),
    (10, "M"),
    (40, "L"),
)

#: Dimensions that are not a legal Model 2 symbol size, with the reason each is refused.
#: ``0`` and the negative value are below the closed range, ``20`` is one module short of
#: Version 1, ``22``/``23`` are not congruent to 1 modulo 4, ``178`` is the count an
#: off-by-one snap produces and ``181`` is the dimension just past Version 40.
_ILLEGAL_DIMENSIONS: Final[tuple[int, ...]] = (0, -1, -21, 20, 22, 23, 178, 181)

#: The module-import graph the sampler is allowed, in full: the standard library, numpy
#: and the decode types it returns. ``cv2``, ``PIL``, ``zxingcpp`` and ``segno`` must not
#: appear, because grid sampling has to work in single-decoder mode (AAP §0.3.2).
_ALLOWED_SAMPLER_IMPORTS: Final[tuple[str, ...]] = (
    "__future__",
    "collections.abc",
    "numpy",
    "qrcheck.decode.base",
    "typing",
)


# ---------------------------------------------------------------------------
# Rendering helpers: the known-geometry image, and the quadrangle that bounds it
# ---------------------------------------------------------------------------


def _render(
    version: int,
    ec_level: str,
    *,
    payload: bytes = _PAYLOAD,
    scale: int = _MODULE_SIZE_PX,
    border: int = _QUIET_ZONE_MODULES,
) -> tuple[segno.QRCode, bytes]:
    """Render one Model 2 symbol and return it with the PNG bytes of its raster.

    ``segno.make_qr`` is Model 2 by construction and takes no ``micro`` argument, which is
    why it is used instead of ``segno.make`` (that one returns a Micro QR symbol for a
    short payload, and a Micro symbol has no ``4V + 17`` dimension at all). ``boost_error``
    is disabled so the pinned version and error level are the ones the oracle actually
    encodes with, exactly as every other call site in the product does (AAP §0.3.5).

    Args:
        version: The version to pin, 1-40.
        ec_level: The error-correction level to pin.
        payload: The payload to encode.
        scale: Pixels per module.
        border: Quiet zone in modules.

    Returns:
        ``(symbol, png_bytes)``, where ``png_bytes`` is the raster at exactly
        ``(module_count + 2 * border) * scale`` pixels per side.
    """
    symbol = segno.make_qr(payload, version=version, error=ec_level, boost_error=False)
    stream = io.BytesIO()
    symbol.save(stream, kind="png", scale=scale, border=border)
    return symbol, stream.getvalue()


def _luma(png_bytes: bytes) -> numpy.ndarray:
    """Return a rendered PNG as the two-dimensional ``uint8`` luminance array the sampler takes.

    This is the production path for an artefact: the bytes are opened through the
    framework's single image boundary, and only the mode is normalised here, because Pillow
    may hand back a bitonal image whose array is not ``uint8``.

    Args:
        png_bytes: A PNG artefact.

    Returns:
        A ``numpy.uint8`` array of shape ``(H, W)``.
    """
    image = decode_image(png_bytes)
    if image.mode != "L":
        image = image.convert("L")
    return numpy.asarray(image)


def _inverted(png_bytes: bytes) -> bytes:
    """Return the 255-complement of a rendered PNG, encoded back to PNG bytes.

    A light-on-dark symbol is a legal request (``RN-CONTRAST``) but the inverse of what a
    finder-pattern measurement expects, so it is the cheapest way to produce a valid image
    whose module geometry genuinely cannot be established. The transform goes through the
    image object the framework's own decoder returned, so no part of this module opens or
    encodes an image for itself.

    Args:
        png_bytes: A PNG artefact.

    Returns:
        The inverted artefact's PNG bytes.
    """
    image = decode_image(png_bytes)
    if image.mode != "L":
        image = image.convert("L")
    return encode_png(image.point(lambda value: 255 - value))


def _oracle_grid(symbol: segno.QRCode) -> Matrix:
    """Return the oracle's module matrix as the sampler's own matrix type.

    segno exposes the matrix as rows of ``0``/``1`` bytes; the sampler works in
    ``numpy.bool_`` with ``True`` for a dark module, so the conversion is stated once here
    instead of at each call site.

    Args:
        symbol: The rendered symbol.

    Returns:
        A ``(N, N)`` ``numpy.bool_`` array, ``True`` for a dark module.
    """
    return numpy.array(
        [[bool(module) for module in row] for row in symbol.matrix], dtype=numpy.bool_
    )


def _quad_around_symbol(
    version: int,
    *,
    span_modules: float | None = None,
    offset_modules: float = 0.0,
) -> list[tuple[float, float]]:
    """Return four symbol corners in source-image pixels for a known rendered symbol.

    A decoder reports the symbol's own corners, which exclude the quiet zone, so the quad
    starts ``border * scale`` pixels in. The two knobs are what make the refusal cases
    reachable: ``span_modules`` changes how many modules the quad covers (a quad that
    overshoots the symbol is a wrong pitch), and ``offset_modules`` shifts it off the
    symbol's corner (a quad that starts light cannot measure a finder pattern).

    Args:
        version: The version the rendered symbol has, which fixes its module count.
        span_modules: Modules the quad covers per side; defaults to the symbol's own count.
        offset_modules: Displacement in modules, applied to both axes.

    Returns:
        The four corners in traversal order.
    """
    module_count = module_count_for_version(version)
    side = (module_count if span_modules is None else span_modules) * _MODULE_SIZE_PX
    origin = (_QUIET_ZONE_MODULES + offset_modules) * _MODULE_SIZE_PX
    return [
        (origin, origin),
        (origin + side, origin),
        (origin + side, origin + side),
        (origin, origin + side),
    ]


def _one_pixel_per_module(version: int, ec_level: str) -> numpy.ndarray:
    """Return a symbol rendered at exactly one pixel per module and no quiet zone.

    That is the shape the secondary decoder hands over as ``straight_qrcode``, and the
    shape of the fallback path's input.

    Args:
        version: The version to pin.
        ec_level: The error-correction level to pin.

    Returns:
        A ``(N, N)`` ``uint8`` array.
    """
    _, png_bytes = _render(version, ec_level, scale=1, border=0)
    return _luma(png_bytes)


# ---------------------------------------------------------------------------
# The happy path: the grid the sampler recovers is the oracle's matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "ec_level"),
    [
        pytest.param(version, ec_level, id=f"V{version}-{ec_level}")
        for version, ec_level in _SAMPLE_VERSIONS
    ],
)
def test_sampled_matrix_matches_the_oracle(version: int, ec_level: str) -> None:
    """The sampled grid equals segno's matrix, element for element, at the known geometry.

    The whole point of grid sampling: the product has no other version source, so the grid
    it recovers has to be the grid the encoder wrote. The quadrangle is the symbol's own
    bounding box, which is the geometry a decoder reports, and ``straight_qrcode`` is
    omitted entirely -- the primary path must need nothing from the secondary decoder.
    """
    symbol, png_bytes = _render(version, ec_level)
    expected = _oracle_grid(symbol)
    module_count = module_count_for_version(version)

    sampled = sample_matrix(_luma(png_bytes), _quad_around_symbol(version))

    assert isinstance(sampled, SampledMatrix)
    assert sampled.reason is None
    assert sampled.source == "primary", (
        "a quadrangle on the symbol's own corners is the primary path"
    )
    assert sampled.module_count == module_count == 4 * version + 17
    assert sampled.version == version, "the version is read off the dimension, and nothing else"
    assert sampled.matrix is not None
    assert sampled.matrix.shape == (module_count, module_count)
    assert sampled.matrix.dtype == numpy.bool_
    assert numpy.array_equal(sampled.matrix, expected)


def test_sampling_the_same_image_twice_yields_the_same_digest() -> None:
    """Two samplings of one artefact are byte-identical, and hash exactly as the oracle's grid.

    ``matrix_sha256`` is what ``DT-DETERMINISTIC`` compares across two calls of an identical
    request, so a sampler that produced a different grid from identical pixels -- a
    dictionary-ordered tap, a rounding wobble -- would be reported as a non-deterministic
    target. The digest is also compared against the oracle's grid, which is the stronger
    statement: the sampler reproduces the encoder, not merely itself.
    """
    symbol, png_bytes = _render(2, "M")
    image = _luma(png_bytes)
    quad = _quad_around_symbol(2)

    first = sample_matrix(image, quad)
    second = sample_matrix(image, quad)

    assert first.matrix is not None and second.matrix is not None
    assert matrix_sha256(first.matrix) == matrix_sha256(second.matrix)
    assert matrix_sha256(first.matrix) == matrix_sha256(_oracle_grid(symbol))


# ---------------------------------------------------------------------------
# The version, and the dimensions that have no version
# ---------------------------------------------------------------------------


def test_version_from_dimension_inverts_the_module_count() -> None:
    """``4V + 17`` and back, for every Model 2 version -- the product's only version source."""
    assert MIN_VERSION == 1 and MAX_VERSION == 40
    assert MIN_MODULE_COUNT == 21 and MAX_MODULE_COUNT == 177
    for version in range(MIN_VERSION, MAX_VERSION + 1):
        dimension = module_count_for_version(version)
        assert dimension == 4 * version + 17
        assert version_from_dimension(dimension) == version


@pytest.mark.parametrize("dimension", _ILLEGAL_DIMENSIONS)
def test_illegal_dimensions_are_refused_rather_than_snapped(dimension: int) -> None:
    """A dimension that is not a legal ``4V + 17`` yields ``None``; it is never rounded.

    Documented contract of :func:`~qrcheck.decode.matrix.version_from_dimension`: an
    out-of-range count, a negative one and one not congruent to 1 modulo 4 all return
    ``None``. A helper that snapped instead would hand a version to every consumer of a
    garbled grid.
    """
    assert version_from_dimension(dimension) is None


@pytest.mark.parametrize("version", (0, -1, MIN_VERSION - 1, MAX_VERSION + 1, 100))
def test_module_count_for_version_refuses_a_version_outside_the_model_2_range(version: int) -> None:
    """The version vocabulary is closed at Versions 1-40, and the diagnostic names the value."""
    with pytest.raises(ValueError) as excinfo:
        module_count_for_version(version)
    assert str(version) in str(excinfo.value)
    assert version_from_dimension(4 * version + 17) is None


@pytest.mark.parametrize("value", (True, False, 21.0, "21", None))
def test_module_count_for_version_refuses_a_value_that_is_not_an_integer(value: object) -> None:
    """A non-integer (including a ``bool``, which is an ``int`` subclass) is a producer defect."""
    with pytest.raises(ValueError):
        module_count_for_version(value)
    assert version_from_dimension(value) is None


# ---------------------------------------------------------------------------
# The pitch, the snap and the timing confirmation
# ---------------------------------------------------------------------------


def test_estimate_measures_the_finder_pitch_and_confirms_the_timing_pattern() -> None:
    """Each measurement the snapshot of ``N`` rests on is pinned to its exact known value.

    The rendered pitch is known, so the estimator's numbers are asserted rather than
    compared against a second run of the same code: the finder pattern's dark run is seven
    modules wide (ISO/IEC 18004 §7.3.3), the pitch is that run over seven, the estimate is
    the square's side over the pitch, the snap lands on the legal count, and the timing
    pattern shows exactly ``N - 16`` alternations on both axes (§7.3.5). A pitch estimator
    that divided the run by six would move the estimate, the pitch and the snap, and this
    test is what fails first.
    """
    version = 2
    module_count = module_count_for_version(version)
    _, png_bytes = _render(version, "M")

    estimate = estimate_module_count(_luma(png_bytes), _quad_around_symbol(version))

    assert isinstance(estimate, ModuleCountEstimate)
    assert estimate.accepted is True
    assert estimate.reason is None
    assert estimate.finder_run_px == FINDER_MODULE_COUNT * _MODULE_SIZE_PX
    assert estimate.pitch == pytest.approx(float(_MODULE_SIZE_PX))
    assert estimate.n_est == module_count
    assert estimate.snapped_n == module_count
    assert estimate.timing_transitions == (module_count - 16, module_count - 16)
    # The timing pair is read from module row and column 6, which is fixed by the standard.
    assert (TIMING_ROW, TIMING_COLUMN) == (6, 6)


def test_a_quad_that_overshoots_the_symbol_is_refused_by_the_timing_pattern() -> None:
    """A wrong pitch is snapped by the pitch guard and then refused by the timing pattern.

    The quadrangle spans two modules more than the symbol, so the pitch measured from the
    finder pattern is right while the square is too wide: the estimate lands between legal
    counts, snaps to the symbol's own count because the implied pitch is still within
    tolerance, and only the timing pattern can reject it. That is the second guard doing
    the work the first could not, and the refusal has to keep its numbers -- the estimate,
    the transition counts and the rectified square -- because that is everything a reader
    of the evidence has to explain the failure.
    """
    version = 1
    module_count = module_count_for_version(version)
    _, png_bytes = _render(version, "M")
    quad = _quad_around_symbol(version, span_modules=module_count + 2)

    estimate = estimate_module_count(_luma(png_bytes), quad)
    sampled = sample_matrix(_luma(png_bytes), quad, keep_rectified=True)

    assert estimate.accepted is False
    assert estimate.reason == "timing_mismatch"
    assert estimate.snapped_n == module_count, "the pitch guard alone accepted the snap"
    assert estimate.n_est == module_count + 2
    assert estimate.timing_transitions is not None
    assert estimate.timing_transitions != (module_count - 16, module_count - 16)

    assert sampled.matrix is None and sampled.version is None and sampled.module_count is None
    assert sampled.reason == "sampling_failed"
    assert sampled.source == "primary"
    assert sampled.estimate == estimate
    assert sampled.rectified_luma is not None
    assert sampled.rectified_luma.shape == ((module_count + 2) * _MODULE_SIZE_PX,) * 2
    assert sampled.rectified_luma.dtype == numpy.float64
    recorded = sampled.as_dict()
    assert recorded["reason"] == "sampling_failed"
    assert recorded["estimate"]["timing_transitions"] == list(estimate.timing_transitions)


def test_a_quad_that_begins_inside_the_symbol_finds_no_finder_pattern() -> None:
    """A quadrangle offset off the symbol's corner starts light, so no pitch exists at all.

    This is the case the module's own documentation calls out: a quad that does not begin at
    the symbol's corner is *refused*, never silently re-anchored from the rectified image's
    dark extent, because re-anchoring would turn a wrong measurement into a confident one.
    """
    version = 1
    _, png_bytes = _render(version, "M")
    quad = _quad_around_symbol(
        version, span_modules=module_count_for_version(version) - 2, offset_modules=1.0
    )

    estimate = estimate_module_count(_luma(png_bytes), quad)

    assert estimate.accepted is False
    assert estimate.reason == "finder_unreadable"
    assert estimate.finder_run_px is None
    assert estimate.pitch is None
    assert estimate.n_est is None
    assert estimate.snapped_n is None


def test_a_pitch_no_legal_count_can_match_is_refused_before_sampling() -> None:
    """A square far too narrow for the measured pitch snaps to nothing at all.

    With the quadrangle covering twelve modules of a twenty-one-module symbol the estimate
    is twelve, whose nearest legal counts imply a pitch more than the tolerance away from
    the measured one, so the count is refused *before* a grid is sampled and the reason says
    so. This is the first of the two guards, on its own.
    """
    version = 1
    module_count = module_count_for_version(version)
    _, png_bytes = _render(version, "M")
    quad = _quad_around_symbol(version, span_modules=module_count - 9)

    estimate = estimate_module_count(_luma(png_bytes), quad)

    assert estimate.accepted is False
    assert estimate.reason == "pitch_mismatch"
    assert estimate.n_est == module_count - 9
    assert estimate.snapped_n is None
    assert estimate.timing_transitions is None, "no grid is sampled once the snap is refused"
    assert estimate.pitch == pytest.approx(float(_MODULE_SIZE_PX))


# ---------------------------------------------------------------------------
# The rectified-image fallback, and the failure when both paths refuse
# ---------------------------------------------------------------------------


def test_the_rectified_fallback_recovers_the_grid_without_a_quadrangle() -> None:
    """With no ``position`` at all, the secondary decoder's rectified image still yields the grid.

    The fallback exists because the secondary decoder hands over an already-rectified square
    for free, so a symbol whose quadrangle the primary decoder never reported stays
    measurable. What is asserted beyond the grid is that the path is *labelled*: ``source``
    names ``straight_qrcode`` on the result and in its evidence projection, so no reader can
    mistake a fallback sampling for a primary one.
    """
    version = 2
    module_count = module_count_for_version(version)
    symbol, png_bytes = _render(version, "M")
    rectified = _one_pixel_per_module(version, "M")
    assert rectified.shape == (module_count, module_count)

    sampled = sample_matrix(_luma(png_bytes), None, straight_qrcode=rectified)

    assert sampled.reason is None
    assert sampled.source == "straight_qrcode"
    assert sampled.version == version
    assert sampled.module_count == module_count
    assert sampled.matrix is not None
    assert numpy.array_equal(sampled.matrix, _oracle_grid(symbol))
    assert sampled.estimate.accepted is True
    assert sampled.estimate.finder_run_px == FINDER_MODULE_COUNT
    assert sampled.estimate.pitch == pytest.approx(1.0)
    assert sampled.as_dict()["source"] == "straight_qrcode"


def test_both_paths_refusing_yields_sampling_failed() -> None:
    """No quadrangle and no usable rectified image is a failure, and it says so.

    The shape asserted here is the one the checks branch on: ``matrix``, ``version`` and
    ``module_count`` are all ``None`` and ``reason`` is ``sampling_failed``, which is a
    diagnosis a check can report instead of inventing a verdict from an absent grid. The
    estimate carries the last path's reason, so the evidence names what was unusable.
    """
    _, png_bytes = _render(1, "M")
    non_square = numpy.zeros((21, 22), dtype=numpy.uint8)

    sampled = sample_matrix(_luma(png_bytes), None, straight_qrcode=non_square, keep_rectified=True)

    assert sampled.matrix is None
    assert sampled.version is None
    assert sampled.module_count is None
    assert sampled.reason == "sampling_failed"
    assert sampled.source == "straight_qrcode", "the fallback was the path attempted last"
    assert sampled.estimate.accepted is False
    assert sampled.estimate.reason == "fallback_unusable"
    assert sampled.rectified_luma is None


def test_a_refused_primary_with_an_unusable_fallback_still_reports_its_diagnosis() -> None:
    """When every path refuses, the numbers behind the refusal survive into the evidence.

    A quadrangle that was rectified but not accepted leaves a measured estimate behind, and
    the flat fallback that follows it is refused for its own reason. Whichever estimate is
    finally reported, the result is a ``sampling_failed`` with a reason, a numeric record
    and -- because the caller asked for it -- the rectified square, never a wrong grid.
    """
    version = 1
    module_count = module_count_for_version(version)
    _, png_bytes = _render(version, "M")
    quad = _quad_around_symbol(version, span_modules=module_count + 2)
    flat = numpy.full((module_count, module_count), 255, dtype=numpy.uint8)

    sampled = sample_matrix(_luma(png_bytes), quad, straight_qrcode=flat, keep_rectified=True)

    assert sampled.reason == "sampling_failed"
    assert sampled.matrix is None
    assert sampled.estimate.accepted is False
    assert sampled.estimate.reason == "low_contrast", "a flat square cannot be thresholded"
    assert sampled.rectified_luma is not None
    assert sampled.rectified_luma.dtype == numpy.float64
    assert sampled.as_dict()["estimate"]["accepted"] is False


# ---------------------------------------------------------------------------
# The consumers: a check reports the reason, never a verdict, when no grid exists
# ---------------------------------------------------------------------------


def test_version_min_reports_sampling_failed_when_no_grid_is_recovered(
    run_context: Any, patch_decode: Any, make_decode_result: Any
) -> None:
    """``SY-VERSION-MIN`` fails with ``sampling_failed``, not with a version mismatch.

    The check's version comes from the grid and from nowhere else, so a decode result that
    carries no ``version`` must produce the sampling reason. Reporting ``value_mismatch``
    instead would blame the target for a measurement the framework could not take, and
    scoring it as a pass would hide a broken sampler behind a green run.
    """
    patch_decode(make_decode_result(matrix=None, version=None, mask=None))
    cases = list(SY_VERSION_MIN.cases(run_context))
    assert cases, "SY-VERSION-MIN must contribute at least one case for the corpus subset"

    result = cases[0].execute(run_context)

    assert result.status is Status.FAILED
    assert result.reason is Reason.SAMPLING_FAILED
    assert result.decoded["sampled_version"] is None
    assert patch_decode.count == 1, "one case is one encode and one decode"


def test_quiet_zone_reports_sampling_failed_when_no_pitch_can_be_measured(
    make_run_context: Any, stub_adapter: Any
) -> None:
    """``RN-QUIET-ZONE`` fails with ``sampling_failed`` when the artefact defeats measurement.

    The quiet zone is measured arithmetically from the modules' pitch, so an artefact whose
    module geometry cannot be established -- here a valid, decodable symbol in the inverted
    polarity, which no finder-pattern measurement recognises -- has no quiet-zone verdict to
    report. It must therefore report the sampling reason rather than a measured margin.
    """
    _, png_bytes = _render(1, "M")
    inverted = _inverted(png_bytes)
    assert measure_module_px(inverted) is None, "the fixture must defeat the pitch measurement"
    stub_adapter.image_bytes = inverted
    ctx = make_run_context(adapter=stub_adapter)

    cases = list(quiet_zone_cases.cases(ctx))
    assert cases, "RN-QUIET-ZONE must contribute at least one case for the corpus subset"

    result = cases[0].execute(ctx)

    assert result.status is Status.FAILED
    assert result.reason is Reason.SAMPLING_FAILED


# ---------------------------------------------------------------------------
# Independence: sampling needs no decoder
# ---------------------------------------------------------------------------


def test_sampling_imports_no_decoder() -> None:
    """The sampler's import graph is pinned: no ``cv2``, no ``PIL``, no decoder, no oracle.

    Single-decoder mode is a supported degradation (AAP §0.3.2), so grid sampling has to
    work when the secondary decoder is not installed at all -- which means the module may
    not import ``cv2``, and it may not import the primary decoder or Pillow either, because
    it takes arrays and returns arrays. The runtime half of the same property is
    :func:`test_sampled_matrix_matches_the_oracle`, which samples with no
    ``straight_qrcode`` and no decoder involvement whatsoever; this half is the static scan
    that keeps a later convenience import from quietly reintroducing the dependency.
    """
    module = importlib.import_module("qrcheck.decode.matrix")
    source = Path(module.__file__).read_text(encoding="utf-8")

    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    unexpected = sorted({name for name in imported if name not in _ALLOWED_SAMPLER_IMPORTS})
    assert unexpected == [], f"qrcheck.decode.matrix must import only {_ALLOWED_SAMPLER_IMPORTS}"
    assert "numpy" in imported, "the sampler works on numpy arrays, so it imports numpy"

