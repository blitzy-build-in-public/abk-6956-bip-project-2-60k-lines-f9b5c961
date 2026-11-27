"""Consensus: both reference decoders, one authority, one :class:`DecodeResult`.

Every check in the product reads exactly one decoded-evidence object and this module assembles
it, in this order: :mod:`qrcheck.evidence.mux` opens the artefact the target returned, composites
any alpha band over an opaque backdrop and hands back one ``H x W`` grayscale array that both
decoders and the grid sampler read -- the image closes as soon as that array exists, because
``CL-ARTIFACTS`` asserts that nothing the framework touched is still open, and an artefact whose
longer side is above the ceiling this module declares is refused before a pixel is decoded; the
primary decoder runs on the artefact as generated and the secondary too when it is installed,
both a second time on the 255-inverted copy for an inverted request; the module grid is sampled
from the symbol's normalised polarity and its format information read, which is the only source
of the version, the mask and the second, decoder-independent EC-level reading, both copies'
distances and corrected-bit counts recorded so "the grid could not be recovered" and "the grid
was recovered and its format information could not be read" are two pieces of evidence rather
than one missing mask; and the two answers are compared into one ``agreement`` value.

The rules below are contract, not implementation detail (AAP §0.3.2).

* A decoder that returns no payload is ``undetected``, never a disagreement: ``disagree``
  requires two payloads that differ, so a decoder's silence can never fail a conforming target.
* ``DecodeResult.payload_bytes`` is the primary decoder's payload whenever it produced one, so
  "primary failed, secondary succeeded" is ``secondary_only`` and *fails* downstream with
  ``payload_bytes`` left ``None``: no check can pass on the secondary's strength.
* Inversion is handled here, not hoped for from a decoder (§0.3.2, IM-4).  ``zxing-cpp`` 2.2.0
  exposes no inversion switch and OpenCV 4.10 cannot detect a light-on-dark symbol at all, so an
  inverted request -- one whose ``foreground_rgb`` is lighter than its ``background_rgb`` -- is
  decoded twice with both decoders and the grid is sampled from the normalised copy: sampling a
  light-on-dark array measures the finder pattern's light ring as its dark run and reports
  ``finder_unreadable``.  ``polarity_applied`` records which polarity the authoritative payload
  came from, and a non-inverted request is decoded once, in the polarity as generated.
* ``ec_level_decoder`` (the primary decoder's metadata) and ``ec_level_format`` (the symbol's own
  15-bit format information) are two independent readings, never reconciled: a disagreement
  between them is the ``format_info_disagreement`` finding a check reports.
* Damage is not special-cased (§0.4): under ``DM-*`` only the primary decoder's success is
  asserted, and the classification rules above already stop the secondary's silence from
  manufacturing a failure.

One documented refinement, and the only deviation.  The secondary decoder returns text, not
bytes, so for a payload that is not losslessly text-representable -- the ``random-bytes`` corpus
class and every ``payload_b64`` record -- its re-encoded bytes legitimately differ from a binary
primary payload even for a perfectly conforming symbol.  Comparing them byte-exactly would
classify those records ``disagree``, and AAP §0.4.1's clean-run gate demands ``counts.failed ==
0`` for the conforming ``reference`` target across all 240 records, so forty conforming cases
would fail.  ``disagree`` therefore applies only when the primary payload round-trips through
UTF-8 (``primary.text.encode("utf-8") == primary.payload``) and the secondary's payload still
differs; anything else the secondary cannot corroborate is recorded as ``primary_only``, which
never loosens the primary's authority: it only declines to let a corroborator that cannot speak
about bytes contradict bytes.

``DecodeResult.secondary_comparable`` makes that narrowing visible, and it is the only place it
is visible: ``True`` when both decoders answered and their payloads were comparable, ``False``
when both answered with differing payloads the refinement declined to compare, and ``None`` when
no comparison was attempted because the secondary decoder did not run, found nothing or errored,
or the primary decoder produced no payload.  A reader of ``decoded.json`` or ``summary.json`` can
therefore tell "the secondary said nothing" from "the secondary's answer could not be compared",
which the ``primary_only`` classification alone records identically.

Imports: the four decode-layer modules this pipeline joins, ``qrcheck.evidence.mux`` (the owner of
every byte the framework turns into an image, so the pixel decode is delegated rather than
repeated) and ``qrcheck.config.models`` (which declares the reference decode size
``decode_image_px`` exactly once, the budget this module's artefact ceiling derives from -- AAP
§0.6.2).  ``GenerateRequest`` is imported for its annotations only, under ``TYPE_CHECKING``, so
importing this module does not load ``qrcheck.targets`` with its ``click``, ``httpx`` and
``segno`` (AAP §0.7.1).  Nothing here comes from ``qrcheck.checks``, ``qrcheck.report`` or
``qrcheck.registry``.

It raises nothing for target-supplied data: unreadable bytes, an SVG artefact, a symbol neither
decoder can read and a grid that cannot be sampled all come back as a ``DecodeResult``.
``ValueError`` is reserved for a programmer error in an argument -- an object that is not bytes,
or a request that carries no colours.
"""

from __future__ import annotations

import dataclasses
import io
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy
from PIL import Image

from qrcheck.config.models import DEFAULT_DECODE_IMAGE_PX
from qrcheck.decode import base, format_info, matrix, opencv, zxing
from qrcheck.decode.base import (
    PRIMARY_DECODER,
    SECONDARY_DECODER,
    Agreement,
    DecodeResult,
    DecoderOutcome,
    Polarity,
    SampledMatrix,
)
from qrcheck.evidence.mux import decode_image

if TYPE_CHECKING:  # pragma: no cover - annotation resolution only, never executed
    from qrcheck.targets.base import GenerateRequest

__all__ = [
    "contrast_ratio",
    "decode_symbol",
    "invert_image",
    "is_inverted_request",
    "relative_luminance",
]

# WCAG 2.x relative luminance: the channel value is linearised at the 0.03928
# knee and the three channels are weighted for human perception. These are the
# numbers behind the ``contrast_floor`` target key and the ~3.5:1 request that
# ``IV-BAD-OPTION`` expects a target to reject.
_LINEARISATION_KNEE: Final = 0.03928
_LINEARISATION_DIVISOR: Final = 12.92
_LINEARISATION_OFFSET: Final = 0.055
_LINEARISATION_SCALE: Final = 1.055
_LINEARISATION_EXPONENT: Final = 2.4
_RED_WEIGHT: Final = 0.2126
_GREEN_WEIGHT: Final = 0.7152
_BLUE_WEIGHT: Final = 0.0722

#: The contrast ratio's constant offset: ``(lighter + k) / (darker + k)`` with
#: ``k = 0.05`` is WCAG's definition, which keeps the ratio finite for black.
_CONTRAST_OFFSET: Final = 0.05

#: How much of the head of the byte stream is inspected for an SVG signature.
#: An XML declaration or a root ``<svg>`` element appears within the first few
#: bytes of every SVG a generator emits, and nothing else is opened as text.
_SIGNATURE_LENGTH: Final = 64

#: The signatures of an SVG artefact. A vector artefact cannot be decoded: it
#: has to be rasterized first, which is ``RN-SVG``'s job, so this pipeline
#: records the reason rather than pretending the symbol was undetectable.
_SVG_SIGNATURES: Final[tuple[bytes, ...]] = (b"<?xml", b"<svg")

#: The UTF-8 byte-order mark, which can precede an XML declaration.
_BYTE_ORDER_MARK: Final = b"\xef\xbb\xbf"

#: The reason recorded when the artefact is a vector image.
_SVG_REASON: Final = (
    "artefact is SVG, not a raster image: it must be rasterized before it can be "
    "decoded, which is the SVG rendering check's job"
)

#: The opaque backdrop an artefact's alpha band is composited over before its pixels are
#: measured. ``rsvg-convert`` rasterises segno's SVG, which paints no background, into an
#: ``RGBA`` image whose transparent pixels store ``(0, 0, 0)``; dropping that band instead
#: of compositing it would turn a conforming symbol into a solid black image.
_RASTER_BACKDROP: Final = (255, 255, 255, 255)

#: The largest per-side pixel dimension this pipeline decodes from an untrusted artefact:
#: eight times the reference decode size ``[checks.performance] decode_image_px``, whose
#: single declaration ``qrcheck.config.models`` owns (AAP §0.6.2), so 4000 px per side.
#: A Version-40 symbol is 177 modules plus an 8-module quiet zone, which at the 10 px per
#: module the framework's own requests reach is 1850 px, and a rotated copy of it needs a
#: square root of two more, so the ceiling accepts the largest symbol this framework ever
#: asks for while refusing one that would expand the colour, grayscale and float64
#: luminance copies into gigabytes of memory a target dictated (AAP §0.1.3, §0.3.4).
_MAX_ARTEFACT_SIDE_PX: Final = 8 * DEFAULT_DECODE_IMAGE_PX

#: An object address inside a diagnostic. Pillow names the buffer it could not identify by
#: its ``repr`` -- ``cannot identify image file <_io.BytesIO object at 0x7f...>`` -- and that
#: address changes with every run, so it must not reach ``decoded.json``, which is required
#: to be byte-identical for the same case and seed (AAP §0.5.4, §0.5.5). Only the address
#: is replaced; every other character of the library's message is recorded verbatim.
_VOLATILE_ADDRESS: Final = re.compile(r"0x[0-9a-fA-F]+")


def _channels(rgb: object, label: str) -> tuple[int, int, int]:
    """Return ``rgb`` as a validated ``(r, g, b)`` triple of ``0-255`` integers.

    Args:
        rgb: The value to validate, normally a ``GenerateRequest`` colour field.
        label: The field name, used verbatim in the diagnostic so the offending
            key is named.

    Returns:
        The three channels as plain ``int`` values.

    Raises:
        ValueError: ``rgb`` is not a three-element iterable, or a channel is not
            an integer in ``0-255``. A colour reaches this product from a request
            the framework itself built, so a malformed one is a producer defect
            rather than evidence about a target, and it is raised instead of being
            recorded as a decoding outcome.
    """
    if isinstance(rgb, (str, bytes)):
        raise ValueError(f"{label} must be a (r, g, b) triple; got {type(rgb).__name__}")
    try:
        values = tuple(numpy.asarray(rgb, dtype=object).ravel())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a (r, g, b) triple of integers; got {rgb!r}"
        ) from exc
    if len(values) != 3:
        raise ValueError(f"{label} must have exactly three channels; got {len(values)}")
    channels: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, numpy.integer)):
            raise ValueError(f"{label} channels must be integers; got {rgb!r}")
        channel = int(value)
        if not 0 <= channel <= 255:
            raise ValueError(f"{label} channels must be in 0-255; got {channel}")
        channels.append(channel)
    return (channels[0], channels[1], channels[2])


def _linearise(channel: int) -> float:
    """Linearise one ``0-255`` channel as WCAG 2.x relative luminance defines it."""
    scaled = channel / 255.0
    if scaled <= _LINEARISATION_KNEE:
        return scaled / _LINEARISATION_DIVISOR
    return ((scaled + _LINEARISATION_OFFSET) / _LINEARISATION_SCALE) ** _LINEARISATION_EXPONENT


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Return the WCAG 2.x relative luminance of an sRGB colour.

    Each ``0-255`` channel is divided by 255, linearised at the ``0.03928`` knee
    (``c / 12.92`` below it, ``((c + 0.055) / 1.055) ** 2.4`` above it) and
    weighted ``0.2126 R + 0.7152 G + 0.0722 B``. Black is ``0.0`` and white is
    ``1.0``.

    This is the definition behind the ``contrast_floor`` target key and the
    ~``3.5:1`` request ``IV-BAD-OPTION`` builds against a declared floor:
    :func:`contrast_ratio` is the ratio those two numbers are measured in, and it
    is built on this function. ``qrcheck/decode/matrix.py`` reduces colour with
    the ITU-R BT.601 luma weights instead, which is a *different* quantity used
    only for thresholding pixels -- the two must not be substituted for one
    another.

    Args:
        rgb: Three ``0-255`` integers.

    Returns:
        The relative luminance in the closed interval ``0.0-1.0``.

    Raises:
        ValueError: ``rgb`` is not a three-element sequence of ``0-255``
            integers.
    """
    red, green, blue = _channels(rgb, "rgb")
    return (
        _RED_WEIGHT * _linearise(red)
        + _GREEN_WEIGHT * _linearise(green)
        + _BLUE_WEIGHT * _linearise(blue)
    )


def contrast_ratio(
    foreground_rgb: tuple[int, int, int], background_rgb: tuple[int, int, int]
) -> float:
    """Return the WCAG contrast ratio between a foreground and a background colour.

    ``(lighter + 0.05) / (darker + 0.05)`` on the two relative luminances, so
    black on white is ``21.0`` and identical colours are ``1.0``. The order of
    the arguments does not matter: the ratio is symmetric, which is why an
    inverted palette is measured exactly as a normal one.

    This is the single implementation of the ratio. ``RN-CONTRAST`` measures a
    rendered image's colours against it, the built-in targets reject a request
    below the target's declared ``contrast_floor`` with it, and ``IV-BAD-OPTION``
    relies on that rejection -- all three import this function rather than
    re-deriving the formula, so the number a request is refused for and the number
    a report records can never drift apart.

    Args:
        foreground_rgb: The dark-module colour, three ``0-255`` integers.
        background_rgb: The light-module colour, three ``0-255`` integers.

    Returns:
        The ratio, ``>= 1.0``.

    Raises:
        ValueError: Either colour is not a three-element sequence of ``0-255``
            integers.
    """
    foreground = relative_luminance(foreground_rgb)
    background = relative_luminance(background_rgb)
    lighter, darker = (
        (foreground, background) if foreground >= background else (background, foreground)
    )
    return (lighter + _CONTRAST_OFFSET) / (darker + _CONTRAST_OFFSET)


def _request_colours(request: object) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return a request's ``(foreground, background)`` colours, validated.

    Reading the two attributes rather than requiring a ``GenerateRequest``
    instance keeps this usable from a test stub that carries the same two fields,
    while a request that carries neither is still refused by name.

    Raises:
        ValueError: ``request`` has no ``foreground_rgb``/``background_rgb``, or a
            colour is malformed. Both are caller defects.
    """
    try:
        foreground = getattr(request, "foreground_rgb")
        background = getattr(request, "background_rgb")
    except AttributeError as exc:
        raise ValueError(
            "request must carry foreground_rgb and background_rgb; got "
            f"{type(request).__name__}"
        ) from exc
    return (
        _channels(foreground, "foreground_rgb"),
        _channels(background, "background_rgb"),
    )


def is_inverted_request(request: GenerateRequest) -> bool:
    """Report whether a request asks for a light-on-dark (inverted) symbol.

    A request is inverted when its ``foreground_rgb`` is *lighter* than its
    ``background_rgb``: the modules become the light pixels and the quiet zone
    the dark ones. ``RN-CONTRAST``'s inverted cases are why this exists -- they
    exercise a palette the pinned secondary decoder cannot detect at all, so the
    pipeline has to know the polarity was inverted before it can normalise it.

    Args:
        request: The request the artefact was produced for.

    Returns:
        ``True`` for an inverted palette, ``False`` otherwise -- including for
        equal luminances, which is a zero-contrast request rather than an
        inverted one.

    Raises:
        ValueError: ``request`` carries no colours, or a colour is malformed.
    """
    foreground, background = _request_colours(request)
    return relative_luminance(foreground) > relative_luminance(background)


def invert_image(image: numpy.ndarray) -> numpy.ndarray:
    """Return the 255-inverted copy of a ``uint8`` image array.

    Polarity normalisation, not a decoder feature: the framework builds this copy
    itself because ``zxing-cpp`` 2.2.0 exposes no inversion switch and OpenCV
    4.10 cannot detect a light-on-dark symbol, so the second decode pass and the
    grid sampling of an inverted request both read this array. Being a per-pixel
    complement, it preserves every coordinate, so the corner quadrangle a decoder
    reported for one polarity is valid for the other.

    Args:
        image: An ``H x W`` or ``H x W x 3`` ``uint8`` array.

    Returns:
        A C-contiguous ``uint8`` array of the same shape, each pixel replaced by
        ``255 - pixel``.

    Raises:
        ValueError: ``image`` is not a ``uint8`` array -- a caller defect, since
            every array in this pipeline is produced by :func:`_rasterise`.
    """
    array = numpy.asarray(image)
    if array.dtype != numpy.uint8:
        raise ValueError(f"image must have dtype uint8; got {array.dtype}")
    return numpy.ascontiguousarray(255 - array)


class _UnreadableRaster(Exception):
    """Target-supplied bytes that are not a raster image this pipeline can decode.

    A private signal between :func:`_rasterise` and :func:`decode_symbol` only:
    it never escapes the module, and it exists so the failure can be described
    once and then recorded as decoder outcomes rather than raised at a caller.
    """


def _looks_like_svg(raw: bytes) -> bool:
    """Report whether the head of ``raw`` is an XML/SVG signature."""
    head = raw.lstrip()
    if head.startswith(_BYTE_ORDER_MARK):
        head = head[len(_BYTE_ORDER_MARK) :].lstrip()
    return head[:_SIGNATURE_LENGTH].lower().startswith(_SVG_SIGNATURES)


def _grayscale_pixels(image: Image.Image) -> numpy.ndarray:
    """Return ``image`` as the single ``H x W`` ``uint8`` grayscale array decoded from.

    An image that carries an alpha band -- an ``"A"`` among its bands, or a
    ``transparency`` entry in its ``info``, which is how a palette image states its
    alpha -- is composited over :data:`_RASTER_BACKDROP` and only then converted, so
    a transparent pixel is measured as that backdrop rather than as the black zeroes
    its band happens to store. Every other image is converted directly, which is the
    same reduction without the composite.

    One grayscale array is what both decoders read, so neither binding's
    three-channel order is ever exercised: ``zxing-cpp`` documents its three-byte
    input as ``BGR`` and OpenCV reads colour as ``BGR``, and a single convention the
    framework states for itself cannot disagree with either of them.
    """
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        backdrop = Image.new("RGBA", rgba.size, _RASTER_BACKDROP)
        gray_image = Image.alpha_composite(backdrop, rgba).convert("L")
    else:
        gray_image = image.convert("L")
    return numpy.array(gray_image, dtype=numpy.uint8, copy=True)


def _rasterise(image_bytes: bytes) -> numpy.ndarray:
    """Open target bytes as the one grayscale array the decoders and the grid read.

    The pixel decode is delegated to :func:`qrcheck.evidence.mux.decode_image`, the
    framework's owner of the image byte path (AAP §0.8.1), so this pipeline keeps no
    second way to turn an artefact into pixels. The image it returns is closed in a
    ``finally`` once the array exists, so neither the decoded buffer nor a descriptor
    outlives the call: ``CL-ARTIFACTS`` asserts that the framework leaves nothing
    open, and an artefact this pipeline opened is the only kind it could be blamed
    for.

    The artefact's dimensions are read from its header first, in an open that parses
    the header and decodes no pixel, and two refusals happen before anything is
    allocated per pixel. An image with no pixels at all is refused, and so is one
    whose longer side is above :data:`_MAX_ARTEFACT_SIDE_PX`, because a
    target-supplied artefact is untrusted (AAP §0.1.3) and the grayscale and
    ``float64`` luminance copies this pipeline builds grow with its pixel count: an
    unbounded dimension would let a small compressed file exhaust the runner instead
    of producing the recorded contract failure AAP §0.3.4 requires. :func:`_grayscale_pixels`
    then composites any alpha band over an opaque backdrop, so a symbol on a
    transparent background is measured as drawn rather than as its band's zeroes.

    Args:
        image_bytes: The artefact exactly as the target returned it.

    Returns:
        An ``H x W`` ``uint8`` grayscale array -- a private compact copy, so no
        consumer can mutate a buffer another one holds.

    Raises:
        ValueError: ``image_bytes`` is not a bytes-like object -- a caller defect
            rather than a fact about a target.
        _UnreadableRaster: the bytes are an SVG artefact, are not an image Pillow can
            open, describe an image with no pixels, describe one above the per-side
            ceiling, or cannot be reduced to a pixel array. All of these are facts
            about target output and are recorded, never raised at a caller.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError(f"image_bytes must be bytes; got {type(image_bytes).__name__}")
    raw = bytes(image_bytes)
    if _looks_like_svg(raw):
        raise _UnreadableRaster(_SVG_REASON)
    image: Image.Image | None = None
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            width, height = probe.size
        if width <= 0 or height <= 0:
            raise _UnreadableRaster("image bytes describe an image with no pixels")
        if max(width, height) > _MAX_ARTEFACT_SIDE_PX:
            raise _UnreadableRaster(
                f"artefact is {width}x{height} pixels, above the "
                f"{_MAX_ARTEFACT_SIDE_PX} px per-side ceiling this framework decodes"
            )
        image = decode_image(raw)
        return _grayscale_pixels(image)
    except _UnreadableRaster:
        raise
    except Exception as exc:  # noqa: BLE001 - target bytes must never abort a run
        # Deliberately broad, and the only catch of this width in the module.
        # Every way Pillow can refuse hostile or broken bytes -- UnidentifiedImageError,
        # a truncated stream's OSError, a decompression-bomb guard -- must become one
        # recorded outcome, because a check reports on a target's misbehaviour rather
        # than dying of it. The library's own text is preserved so the cause stays
        # diagnosable from decoded.json; only the run-varying object address is
        # replaced, by _VOLATILE_ADDRESS, so the recorded evidence stays
        # byte-identical across runs.
        raise _UnreadableRaster(
            "image bytes are not a raster image Pillow can open: "
            f"{type(exc).__name__}: {_VOLATILE_ADDRESS.sub('0x...', str(exc))}"
        ) from exc
    finally:
        if image is not None:
            image.close()


def _authoritative(
    original: zxing.PrimaryReading, inverted: zxing.PrimaryReading | None
) -> tuple[Polarity, zxing.PrimaryReading]:
    """Return the polarity and the reading the authoritative payload came from.

    The first polarity at which the primary decoder produced a payload wins,
    original before inverted, which is the §0.3.2 authority rule. A payload of
    ``b""`` is a payload: an empty symbol the decoder did read is a different fact
    from a symbol it could not read, and the invalid-input checks depend on that
    distinction. When neither polarity produced one, the original reading is
    returned with ``"original"`` -- nothing was applied, and the field never
    claims a polarity that recovered nothing.
    """
    if original.outcome.payload is not None:
        return ("original", original)
    if inverted is not None and inverted.outcome.payload is not None:
        return ("inverted", inverted)
    return ("original", original)


def _merge_outcome(
    original: DecoderOutcome | None,
    inverted: DecoderOutcome | None,
    preferred: Polarity,
) -> DecoderOutcome | None:
    """Merge one decoder's two polarity passes into the single outcome recorded.

    The outcome recorded in ``per_decoder`` is the one carrying a payload when
    only one polarity produced one. When both produced the same payload the merged
    polarity is ``"both"``, which is what a decoder reading either palette reports
    and what makes an inverted case's evidence self-explanatory. When the two
    payloads differ -- which only a decoder reading two different symbols could
    produce -- the pass at the polarity the authoritative payload came from is
    kept, so the recorded answer never contradicts the authority; the other pass
    remains visible through its own evidence of the polarity it was read at.
    """
    if original is None and inverted is None:
        return None
    if original is None:
        return inverted
    if inverted is None:
        return original
    with_payload = [outcome for outcome in (original, inverted) if outcome.payload is not None]
    if not with_payload:
        return original
    if len(with_payload) == 1:
        return with_payload[0]
    first, second = with_payload[0], with_payload[1]
    if first.payload == second.payload:
        return dataclasses.replace(first, polarity="both")
    for outcome in with_payload:
        if outcome.polarity == preferred:
            return outcome
    return first


def _rectified(
    readings: Sequence[opencv.SecondaryReading | None], preferred: Polarity
) -> numpy.ndarray | None:
    """Return the secondary decoder's rectified grid, preferring the useful pass.

    The rectified image is the fallback module-count source, so a pass that
    produced a payload and the polarity the authoritative payload came from are
    preferred in that order; the ordering is a stable sort, so the choice is
    decided entirely by the passes themselves and never by their arrival order.
    """
    ordered = sorted(
        (reading for reading in readings if reading is not None),
        key=lambda reading: (
            0 if reading.outcome.payload is not None else 1,
            0 if reading.outcome.polarity == preferred else 1,
        ),
    )
    for reading in ordered:
        if reading.straight_qrcode is not None:
            return reading.straight_qrcode
    return None


def _is_text_payload(payload: bytes, text: str | None) -> bool:
    """Report whether ``payload`` round-trips through UTF-8 as ``text``.

    This is the predicate that decides whether the secondary decoder's text can
    contradict the primary decoder's bytes at all; see the module docstring for
    why a payload that fails it is recorded as ``primary_only`` with
    ``secondary_comparable`` ``False``.
    """
    if text is None:
        return False
    try:
        return text.encode("utf-8") == payload
    except UnicodeEncodeError:
        return False


def _secondary_comparable(
    primary: DecoderOutcome, secondary: DecoderOutcome | None, primary_text: str | None
) -> bool | None:
    """Report whether the two answers could be compared, and if not, why not.

    ``None`` means no comparison was attempted: the secondary decoder did not
    run, found nothing or errored, or the primary decoder produced no payload.
    ``True`` means both decoders answered and their payloads are comparable --
    they are equal, or the primary payload round-trips through UTF-8, so the
    secondary decoder's text could have contradicted it. ``False`` means both
    answered with payloads that differ and that the text-only secondary decoder
    cannot be held to, which is the one case in which :func:`_classify` declines
    to report ``disagree``; the module docstring states why that refinement is
    required. This predicate is the single derivation of the recorded
    ``secondary_comparable`` field: :func:`_classify` consumes the same value
    rather than deciding the question a second time.
    """
    if primary.payload is None or secondary is None or secondary.payload is None:
        return None
    if secondary.payload == primary.payload:
        return True
    return _is_text_payload(primary.payload, primary_text)


def _classify(
    primary: DecoderOutcome, secondary: DecoderOutcome | None, primary_text: str | None
) -> tuple[Agreement, bool | None]:
    """Classify the two decoders' payloads, and report whether they were comparable.

    ``undetected`` is a decoder finding no payload, never a disagreement. A
    ``disagree`` verdict requires a secondary decoder that ran, two payloads that
    actually differ, and a comparability of ``True`` -- a primary payload the
    secondary decoder's text could have spoken about in the first place -- which
    is the documented refinement, without which a conforming binary payload would
    be reported as a disagreement.

    Returns:
        The ``agreement`` value and, beside it, the ``secondary_comparable`` value
        :func:`_secondary_comparable` derived. Both are returned together so the
        recorded field and the ``disagree`` decision are one derivation and cannot
        drift apart.
    """
    comparable = _secondary_comparable(primary, secondary, primary_text)
    if primary.payload is not None:
        if secondary is None or secondary.payload is None:
            return "primary_only", comparable
        if secondary.payload == primary.payload:
            return "agree", comparable
        return ("disagree" if comparable else "primary_only"), comparable
    if secondary is not None and secondary.payload is not None:
        return "secondary_only", comparable
    return "undetected", comparable


def _unreadable_result(reason: str, secondary_ok: bool) -> DecodeResult:
    """Record bytes no decoder can be shown as ``undetected``, with the reason.

    One ``error`` outcome is recorded per decoder that would have been run, so
    ``decoded.json`` names the decoders and the reason in the same place a decoded
    case names its payload, and the result's ``agreement`` stays ``undetected``
    because no decoder found anything -- an unreadable artefact is not a
    disagreement between two decoders that both ran.
    """
    names: list[str] = [PRIMARY_DECODER]
    if secondary_ok:
        names.append(SECONDARY_DECODER)
    per_decoder = {
        name: DecoderOutcome(
            decoder=name,
            status="error",
            polarity="original",
            payload_sha256=None,
            reported_ec=None,
            reported_version=None,
            error_text=reason,
        )
        for name in names
    }
    return DecodeResult(
        payload_bytes=None,
        text=None,
        symbology_identifier=None,
        content_type=None,
        ec_level_decoder=None,
        ec_level_format=None,
        version=None,
        mask=None,
        matrix=None,
        matrix_sha256=None,
        per_decoder=per_decoder,
        agreement="undetected",
        polarity_applied="original",
        rejection=None,
        metrics={},
        sampling=None,
        secondary_comparable=None,
    )


def decode_symbol(
    image_bytes: bytes,
    request: GenerateRequest,
    *,
    secondary: bool | None = None,
) -> DecodeResult:
    """Decode one artefact and return the single evidence object every check reads.

    The steps are exactly the pipeline the module docstring describes, and nothing
    else happens here: the bytes become arrays, the decoders run, the primary's
    payload becomes authoritative, the grid and the format information are read,
    the two answers are classified, and one :class:`DecodeResult` is assembled.
    ``rejection`` and ``metrics`` are left empty because they belong to other
    layers -- a declared rejection is recorded by the check that provoked it and
    the performance probes fill ``metrics`` themselves.

    Inversion is decided from the request, not from a decoder: a request whose
    ``foreground_rgb`` is lighter than its ``background_rgb`` gets a second pass
    over the 255-inverted copy with both decoders, and the grid is sampled from
    that normalised copy, because sampling the light-on-dark array measures the
    finder pattern's light ring as its dark run and would report a wrong module
    count rather than no module count.

    Args:
        image_bytes: The artefact exactly as the target returned it. Nothing is
            written and no path is accepted: this module never touches the
            filesystem beyond reading the bytes it was handed.
        request: The request the artefact was produced for. Only its colours are
            read here, to decide the polarity; every other assertion belongs to a
            check.
        secondary: ``True`` to run the secondary decoder, ``False`` to record
            single-decoder mode, ``None`` (the default) to probe whether it is
            installed. A caller passing ``False`` gets a ``per_decoder`` map with
            the primary decoder alone, which is the degraded mode a report shows a
            banner for.

    Returns:
        The assembled :class:`DecodeResult`. ``payload_bytes`` is the primary
        decoder's payload whenever it produced one and ``None`` otherwise, even
        when the secondary decoder succeeded -- that case is ``secondary_only`` and
        must fail downstream rather than pass on the secondary's strength.
        ``ec_level_decoder`` and ``ec_level_format`` are recorded independently and
        never reconciled. ``format_info`` carries the format-information reading
        itself -- both copies' distances and the bit errors corrected -- and is
        ``None`` both when no grid could be sampled and when a sampled grid's two
        copies were both beyond the correction radius; ``sampling.reason`` tells
        those two apart.

    Raises:
        ValueError: ``image_bytes`` is not bytes, or ``request`` carries no
            colours. Both are defects in the caller, and this function raises for
            neither on behalf of a target: unusable bytes, an SVG artefact, an
            unreadable symbol and an unsamplable grid all return a result.
    """
    inverted = is_inverted_request(request)
    secondary_ok = base.secondary_available() if secondary is None else bool(secondary)

    try:
        gray = _rasterise(image_bytes)
    except _UnreadableRaster as exc:
        return _unreadable_result(str(exc), secondary_ok)

    primary_original = zxing.read_primary(gray, polarity="original")
    secondary_original = (
        opencv.read_secondary(gray, polarity="original") if secondary_ok else None
    )

    # An inverted request is decoded twice, with both decoders, on the artefact as
    # generated and on its 255-inverted copy (§0.3.2, IM-4): the framework owns
    # polarity normalisation rather than relying on a decoder's inversion support,
    # and the normalised copy is also what the grid is sampled from below. One
    # inverted copy is built and both decoders and the sampler read it.
    inverted_gray: numpy.ndarray | None = None
    primary_inverted: zxing.PrimaryReading | None = None
    secondary_inverted: opencv.SecondaryReading | None = None
    if inverted:
        inverted_gray = invert_image(gray)
        primary_inverted = zxing.read_primary(inverted_gray, polarity="inverted")
        if secondary_ok:
            secondary_inverted = opencv.read_secondary(inverted_gray, polarity="inverted")

    polarity_applied, authority = _authoritative(primary_original, primary_inverted)
    primary_outcome = _merge_outcome(
        primary_original.outcome,
        None if primary_inverted is None else primary_inverted.outcome,
        polarity_applied,
    )
    secondary_outcome = _merge_outcome(
        None if secondary_original is None else secondary_original.outcome,
        None if secondary_inverted is None else secondary_inverted.outcome,
        polarity_applied,
    )

    position = authority.outcome.position
    if position is None and secondary_outcome is not None:
        position = secondary_outcome.position
    sampling: SampledMatrix = matrix.sample_matrix(
        gray if inverted_gray is None else inverted_gray,
        position,
        straight_qrcode=_rectified((secondary_original, secondary_inverted), polarity_applied),
    )

    mask: int | None = None
    ec_level_format: str | None = None
    format_details: format_info.FormatInfo | None = None
    if sampling.matrix is not None:
        format_details = format_info.decode_format_info(sampling.matrix)
        if format_details is not None:
            mask = format_details.mask
            ec_level_format = format_details.ec_level

    payload = authority.outcome.payload
    per_decoder = {
        outcome.decoder: outcome
        for outcome in (primary_outcome, secondary_outcome)
        if outcome is not None
    }
    agreement, secondary_comparable = _classify(
        primary_outcome if primary_outcome is not None else primary_original.outcome,
        secondary_outcome,
        authority.text,
    )
    return DecodeResult(
        payload_bytes=payload,
        text=authority.text if payload is not None else None,
        symbology_identifier=authority.symbology_identifier if payload is not None else None,
        content_type=authority.content_type if payload is not None else None,
        ec_level_decoder=None if primary_outcome is None else primary_outcome.reported_ec,
        ec_level_format=ec_level_format,
        version=sampling.version,
        mask=mask,
        matrix=sampling.matrix,
        matrix_sha256=None,
        per_decoder=per_decoder,
        agreement=agreement,
        polarity_applied=polarity_applied,
        rejection=None,
        metrics={},
        sampling=sampling,
        format_info=None if format_details is None else format_details.as_dict(),
        secondary_comparable=secondary_comparable,
    )

