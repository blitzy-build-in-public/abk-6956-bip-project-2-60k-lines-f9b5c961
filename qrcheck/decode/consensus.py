"""Consensus: both reference decoders, one authority, one :class:`DecodeResult`.

Every check in the product reads exactly one decoded-evidence object, and this
module is what assembles it. The pipeline it owns, in order:

1. **Bytes to pixels.** Pillow opens the artefact the target returned into an
   ``RGB`` array for the primary decoder and an ``L`` grayscale array for the
   secondary one, which is the form that decoder's own documentation asks for.
   The handle closes before this module returns, because ``CL-ARTIFACTS``
   asserts that nothing the framework touched is still open.
2. **Passes.** The primary decoder is run on the artefact as generated, and the
   secondary decoder too when it is installed. For an inverted request both
   decoders run a second time on the 255-inverted copy.
3. **Authority.** The primary decoder's payload, at the first polarity that
   produced one, is authoritative for all twenty-one checks.
4. **Grid.** The module grid is sampled from the symbol's normalised polarity
   and its format information read, which is the only source of the version, the
   mask and the second, decoder-independent EC-level reading.
5. **Classification.** The two answers are compared into one ``agreement`` value.

The rules below are contract, not implementation detail (AAP §0.3.2).

* **A decoder that returns no payload is ``undetected``, never a
  disagreement.** ``disagree`` requires two payloads that differ, so a decoder's
  silence can never fail a conforming target.
* **One authority for all twenty-one checks.** ``DecodeResult.payload_bytes`` is
  the primary decoder's payload whenever it produced one. The secondary decoder
  corroborates and never becomes the authority: "primary failed, secondary
  succeeded" is ``secondary_only`` and *fails* downstream with
  ``payload_bytes`` left ``None``, so no check can pass on the secondary's
  strength.
* **Inversion is handled here, not hoped for from a decoder** (§0.3.2, IM-4).
  ``zxing-cpp`` 2.2.0 exposes no inversion switch -- it happens to read either
  palette -- while OpenCV 4.10 cannot detect a light-on-dark symbol at all. So an
  inverted request, meaning one whose ``foreground_rgb`` is lighter than its
  ``background_rgb``, is decoded twice with both decoders, and the grid is
  sampled from the normalised copy: sampling a light-on-dark array measures the
  finder pattern's light ring as its dark run and reports ``finder_unreadable``,
  which is exactly the wrong-verdict trap ``format_info`` warns about when it
  says an inverted symbol is normalised by the consensus layer before a grid is
  sampled. ``polarity_applied`` records which polarity the authoritative payload
  came from, and a non-inverted request is decoded once, in the polarity as
  generated.
* **Two independent EC-level readings, never reconciled.** ``ec_level_decoder``
  is the primary decoder's metadata and ``ec_level_format`` is the symbol's own
  15-bit format information. A disagreement between them is the
  ``format_info_disagreement`` finding a check reports, so collapsing one into
  the other here would delete the finding.
* **Damage is not special-cased** (§0.4). Under ``DM-*`` only the primary
  decoder's success is asserted, and the classification rules above already
  guarantee that the secondary decoder's silence cannot manufacture a failure.

**One documented refinement, and why it is the only deviation.** The secondary
decoder returns text, not bytes, so for a payload that is not losslessly
text-representable -- the ``random-bytes`` corpus class and every ``payload_b64``
record -- its re-encoded bytes legitimately differ from a binary primary payload
even for a perfectly conforming symbol. Comparing them byte-exactly would
classify those records as ``disagree``, and AAP §0.4.1's clean-run gate demands
``counts.failed == 0`` for the conforming ``reference`` target across all 240
records, so forty conforming cases would fail and the non-vacuity proof would
collapse. ``disagree`` therefore applies only when the primary payload round-trips
through UTF-8 (``primary.text.encode("utf-8") == primary.payload``) and the
secondary's payload still differs; anything else the secondary cannot corroborate
is recorded as ``primary_only``. This never loosens the primary's authority -- it
only declines to let a corroborator that cannot speak about bytes contradict
bytes.

Imports: the four decode-layer modules this pipeline joins, the single product
dependency ``qrcheck.targets.base`` (for ``GenerateRequest`` and nothing else),
the standard library, ``numpy`` and ``Pillow``. No concrete adapter is imported,
and nothing here comes from ``qrcheck.checks``, ``qrcheck.report``,
``qrcheck.registry``, ``qrcheck.evidence`` or ``qrcheck.config``.

It raises nothing for target-supplied data: unreadable bytes, an SVG artefact, a
symbol neither decoder can read, and a grid that cannot be sampled all come back
as a ``DecodeResult``. ``ValueError`` is reserved for a programmer error in an
argument -- an object that is not bytes, or a request that carries no colours.
"""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Sequence
from typing import Final

import numpy
from PIL import Image

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
# numbers behind ``contrast_floor`` (default 7.0) and the ~3.5:1 request that
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

    This is the definition behind the ``contrast_floor`` target key (default
    ``7.0``) and the ~``3.5:1`` request ``IV-BAD-OPTION`` builds against a
    declared floor: :func:`contrast_ratio` is the ratio those two numbers are
    measured in, and it is built on this function. ``qrcheck/decode/matrix.py``
    reduces colour with the ITU-R BT.601 luma weights instead, which is a
    *different* quantity used only for thresholding pixels -- the two must not be
    substituted for one another.

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


def _rasterise(image_bytes: bytes) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Open target bytes as the two arrays the decoders need.

    Pillow decodes the artefact inside a ``with`` block, so the file handle and
    the decoded buffer are released before this function returns: ``CL-ARTIFACTS``
    asserts that the framework leaves no descriptor open, and an artefact this
    pipeline opened is the only kind it could be blamed for.

    The primary decoder takes the ``RGB`` array as produced, and the secondary
    decoder takes a grayscale array, which is the form its own module asks for --
    a three-channel array reaches OpenCV as BGR, so a coloured symbol would be
    measured through swapped channels. Both are private compact copies, so no
    consumer can mutate a buffer another one holds.

    Args:
        image_bytes: The artefact exactly as the target returned it.

    Returns:
        ``(rgb, gray)``: ``H x W x 3`` and ``H x W`` ``uint8`` arrays.

    Raises:
        ValueError: ``image_bytes`` is not a bytes-like object -- a caller defect
            rather than a fact about a target.
        _UnreadableRaster: the bytes are an SVG artefact, are not an image Pillow
            can open, or describe an image with no pixels. All three are facts
            about target output and are recorded, never raised at a caller.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError(f"image_bytes must be bytes; got {type(image_bytes).__name__}")
    raw = bytes(image_bytes)
    if _looks_like_svg(raw):
        raise _UnreadableRaster(_SVG_REASON)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            rgb = numpy.array(image.convert("RGB"), dtype=numpy.uint8, copy=True)
            gray = numpy.array(image.convert("L"), dtype=numpy.uint8, copy=True)
    except Exception as exc:  # noqa: BLE001 - target bytes must never abort a run
        # Deliberately broad, and the only catch of this width in the module.
        # Every way Pillow can refuse hostile or broken bytes -- UnidentifiedImageError,
        # a truncated stream's OSError, a decompression-bomb guard -- must become one
        # recorded outcome, because a check reports on a target's misbehaviour rather
        # than dying of it. The library's own text is preserved so the cause stays
        # diagnosable from decoded.json.
        raise _UnreadableRaster(
            "image bytes are not a raster image Pillow can open: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if rgb.size == 0 or gray.size == 0:
        raise _UnreadableRaster("image bytes describe an image with no pixels")
    return rgb, gray


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
    why a payload that fails it is recorded as ``primary_only``.
    """
    if text is None:
        return False
    try:
        return text.encode("utf-8") == payload
    except UnicodeEncodeError:
        return False


def _classify(
    primary: DecoderOutcome, secondary: DecoderOutcome | None, primary_text: str | None
) -> Agreement:
    """Classify the two decoders' payloads into the ``agreement`` vocabulary.

    ``undetected`` is a decoder finding no payload, never a disagreement. A
    ``disagree`` verdict requires a secondary decoder that ran, two payloads that
    actually differ, and a primary payload the secondary decoder's text could have
    spoken about in the first place -- the documented refinement, without which a
    conforming binary payload would be reported as a disagreement.
    """
    if primary.payload is not None:
        if secondary is None or secondary.payload is None:
            return "primary_only"
        if secondary.payload == primary.payload:
            return "agree"
        return "disagree" if _is_text_payload(primary.payload, primary_text) else "primary_only"
    if secondary is not None and secondary.payload is not None:
        return "secondary_only"
    return "undetected"


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
        never reconciled.

    Raises:
        ValueError: ``image_bytes`` is not bytes, or ``request`` carries no
            colours. Both are defects in the caller, and this function raises for
            neither on behalf of a target: unusable bytes, an SVG artefact, an
            unreadable symbol and an unsamplable grid all return a result.
    """
    inverted = is_inverted_request(request)
    secondary_ok = base.secondary_available() if secondary is None else bool(secondary)

    try:
        rgb, gray = _rasterise(image_bytes)
    except _UnreadableRaster as exc:
        return _unreadable_result(str(exc), secondary_ok)

    primary_original = zxing.read_primary(rgb, polarity="original")
    secondary_original = (
        opencv.read_secondary(gray, polarity="original") if secondary_ok else None
    )

    # An inverted request is decoded twice, with both decoders, on the artefact as
    # generated and on its 255-inverted copy (§0.3.2, IM-4): the framework owns
    # polarity normalisation rather than relying on a decoder's inversion support,
    # and the normalised copy is also what the grid is sampled from below.
    inverted_rgb: numpy.ndarray | None = None
    primary_inverted: zxing.PrimaryReading | None = None
    secondary_inverted: opencv.SecondaryReading | None = None
    if inverted:
        inverted_rgb = invert_image(rgb)
        primary_inverted = zxing.read_primary(inverted_rgb, polarity="inverted")
        if secondary_ok:
            secondary_inverted = opencv.read_secondary(invert_image(gray), polarity="inverted")

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
        rgb if inverted_rgb is None else inverted_rgb,
        position,
        straight_qrcode=_rectified((secondary_original, secondary_inverted), polarity_applied),
    )

    mask: int | None = None
    ec_level_format: str | None = None
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
        agreement=_classify(
            primary_outcome if primary_outcome is not None else primary_original.outcome,
            secondary_outcome,
            authority.text,
        ),
        polarity_applied=polarity_applied,
        rejection=None,
        metrics={},
        sampling=sampling,
    )

