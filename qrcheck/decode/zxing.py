"""Primary reference decoder: the authority every check compares against.

Wraps the pinned ``zxing-cpp`` 2.2.0 binding, whose payload is authoritative for all twenty-one
checks: a symbol it cannot read is a failed or errored case even when the secondary decoder reads
it, and the secondary never overwrites what this module reported. It targets only 2.2.0's flat
function surface -- ``zxingcpp.read_barcode`` (no ``ReaderOptions``, no ``try_harder`` keyword) and
``zxingcpp.Result``, whose ``symbology_identifier`` is ``]Q1`` without an ECI designator and ``]Q2``
with one.

2.2.0 exposes no version and no mask accessor, so ``reported_version`` is ``None`` here by design:
the version is recovered from the sampled grid (``N = 4V + 17``) and the mask from the symbol's own
format information. A decoder that returns no payload is ``undetected``, never a disagreement, and a
payload of ``b""`` is one the decoder did read -- a different fact from ``None``, which the
invalid-input checks depend on.

Nothing here raises on behalf of a symbol: a missing binding and a binding that throws both become
an ``error`` outcome carrying the text that explains it, and ``ValueError`` is reserved for an
argument that is not an image at all.
"""

from __future__ import annotations

import dataclasses
import hashlib
import types
from typing import Final

import numpy

from qrcheck.decode.base import PRIMARY_DECODER, DecoderOutcome, Polarity

__all__ = [
    "DECODER_NAME",
    "PrimaryReading",
    "decode_primary",
    "read_primary",
]

#: The decoder's canonical name, re-exported from the decode vocabulary so the
#: per-decoder evidence maps, :attr:`DecodeResult.primary` and the reports all
#: key on one literal.
DECODER_NAME: Final = PRIMARY_DECODER

#: The diagnostic recorded when the binding cannot be imported. It names the
#: distribution and the lock file, because the primary decoder is a blocking
#: dependency rather than an optional one: without it no round-trip assertion
#: has an authority, so ``doctor`` and ``run`` refuse to proceed.
_MISSING_BINDING_MESSAGE: Final = (
    "zxing-cpp is not importable; it is a blocking dependency (see requirements.lock)"
)

#: The neutral message recorded when the decoder found no symbol at all and
#: reported no message of its own to preserve.
_NO_SYMBOL_MESSAGE: Final = "no QR symbol found"

#: The four corners of :class:`zxingcpp.Position`, in the decoder's own order.
#: They are read by name rather than by iteration: the ``Position`` type of
#: 2.2.0 is neither iterable nor sized.
_CORNER_NAMES: Final[tuple[str, str, str, str]] = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)


def _load_zxingcpp() -> types.ModuleType | None:
    """Resolve the pinned ``zxingcpp`` binding, or ``None`` when it is absent.

    The import happens per call and is not cached, so the absence path stays exercisable at any
    point in a process's life. An ``ImportError`` raised inside the binding's own import and an
    ``OSError`` from a shared object that cannot be loaded both mean unusable, and the caller gets
    the diagnostic naming the distribution rather than a traceback out of a decoder probe.
    """
    try:
        import zxingcpp
    except (ImportError, OSError):
        return None
    return zxingcpp


def _validate_image(image: numpy.ndarray) -> None:
    """Reject any argument that is not a supported image array.

    The accepted forms are exactly the two the decode layer produces: an ``H x W`` ``uint8``
    grayscale array and an ``H x W x 3`` ``uint8`` BGR array, byte 0 blue, the order the binding
    documents rather than the RGB a Pillow image converts to. No other dtype is accepted, so a float
    image cannot decode at a pixel scale the framework never measured.

    Raises:
        ValueError: ``image`` is not a ``uint8`` ``numpy`` array of either shape -- a caller
            defect, which recorded as an outcome would misattribute a programmer error to the target
            under test.
    """
    if not isinstance(image, numpy.ndarray):
        raise ValueError(f"image must be a numpy.ndarray; got {type(image).__name__}")
    if image.dtype != numpy.uint8:
        raise ValueError(f"image must have dtype uint8; got {image.dtype}")
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] == 3:
        return
    raise ValueError(
        f"image must be HxW grayscale or HxWx3 BGR uint8; got shape {tuple(image.shape)}"
    )


def _optional_text(value: object) -> str | None:
    """Return ``value`` as text, keeping ``None`` and ``""`` distinct: an absent value and
    an empty payload are different evidence.
    """
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _content_type(result: object) -> str | None:
    """Return the decoder's content classification as a stable string.

    The ``ContentType`` member's name is recorded rather than its ``repr``, which names the
    binding's module path and would tie a piece of evidence to an implementation detail.
    """
    value = getattr(result, "content_type", None)
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return value if isinstance(value, str) else str(value)


def _position(result: object) -> tuple[tuple[float, float], ...] | None:
    """Return the symbol's four corners in source-image pixel coordinates.

    The order is the decoder's own -- top-left, top-right, bottom-right, bottom-left -- and the
    coordinates bound the symbol itself, quiet zone excluded. ``None`` when the decoder reported no
    quadrangle, and also when any single corner is missing: a square fabricated from a partial
    reading would rectify the wrong region and produce a matrix that is wrong without looking wrong,
    whereas ``None`` sends sampling to its fallback.
    """
    quadrangle = getattr(result, "position", None)
    if quadrangle is None:
        return None
    corners: list[tuple[float, float]] = []
    for corner_name in _CORNER_NAMES:
        point = getattr(quadrangle, corner_name, None)
        if point is None:
            return None
        corners.append((float(point.x), float(point.y)))
    return tuple(corners)


def _undetected_text(result: object | None) -> str:
    """Return the decoder's own message for a symbol it could not read.

    ``read_barcode`` returns ``None`` when it finds nothing, so a missing result has no message of
    its own; text it did report is preserved verbatim.
    """
    if result is None:
        return _NO_SYMBOL_MESSAGE
    message = _optional_text(getattr(result, "error", None))
    return message or _NO_SYMBOL_MESSAGE


def _failed_outcome(polarity: Polarity, error_text: str) -> DecoderOutcome:
    """Build the ``error`` outcome for a decoder that could not produce a verdict -- an
    absent binding or one that raised -- so both paths record identical metadata.
    """
    return DecoderOutcome(
        decoder=DECODER_NAME,
        status="error",
        polarity=polarity,
        payload_sha256=None,
        reported_ec=None,
        reported_version=None,
        error_text=error_text,
        payload=None,
        position=None,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PrimaryReading:
    """One authoritative reading of one symbol, as the consensus layer needs it.

    ``DecoderOutcome`` carries only what every decoder can answer, so this decoder's rendered
    ``text``, symbology identifier and content classification travel beside the outcome rather than
    widening a type both decoders must fill. ``text`` is the rendering under the default
    ``TextMode.HRI`` and is never what a round-trip assertion compares, because only ``payload`` is
    byte-exact.
    """

    outcome: DecoderOutcome
    text: str | None
    symbology_identifier: str | None
    content_type: str | None


def decode_primary(image: numpy.ndarray, *, polarity: Polarity = "original") -> DecoderOutcome:
    """Decode ``image`` with the primary reference decoder.

    Returns the outcome alone -- ``decoded``, ``undetected`` or ``error``, where an empty payload is
    still a payload -- and never raises for anything the symbol or the binding does. ``polarity`` is
    stamped onto the outcome for the record; this function never inverts anything, the consensus
    layer building the inverted copy itself.

    Raises:
        ValueError: ``image`` is not a supported array -- a defect in the caller rather than the
        symbol.
    """
    return read_primary(image, polarity=polarity).outcome


def read_primary(image: numpy.ndarray, *, polarity: Polarity = "original") -> PrimaryReading:
    """Decode ``image`` and return the outcome together with its text metadata.

    On success the outcome is ``decoded`` with the payload bytes, its SHA-256, the decoder's EC
    level and the corner quadrangle, while the reading carries the rendered text, the symbology
    identifier and the content type. A symbol the decoder could not find is ``undetected`` with its
    message; an absent or raising binding is ``error`` and says why.

    Raises:
        ValueError: ``image`` is not a supported array.
    """
    _validate_image(image)

    binding = _load_zxingcpp()
    if binding is None:
        return PrimaryReading(
            outcome=_failed_outcome(polarity, _MISSING_BINDING_MESSAGE),
            text=None,
            symbology_identifier=None,
            content_type=None,
        )

    try:
        result = binding.read_barcode(
            image,
            formats=binding.BarcodeFormat.QRCode,
            return_errors=True,
        )
    except Exception as exc:
        # Deliberately broad, and the only catch of this width in the module: a decoder that throws
        # on a hostile symbol must leave an evidence record and let the run continue, so every
        # failure of the binding -- a buffer it refuses, a native fault, an unexpected shape --
        # becomes an "error" outcome carrying the exception's own text. Narrowing this would let an
        # unanticipated binding failure abort a run that is supposed to report on misbehaviour
        # rather than die of it.
        return PrimaryReading(
            outcome=_failed_outcome(polarity, f"{type(exc).__name__}: {exc}"),
            text=None,
            symbology_identifier=None,
            content_type=None,
        )

    if result is None or not getattr(result, "valid", False):
        return PrimaryReading(
            outcome=DecoderOutcome(
                decoder=DECODER_NAME,
                status="undetected",
                polarity=polarity,
                payload_sha256=None,
                reported_ec=None,
                reported_version=None,
                error_text=_undetected_text(result),
                payload=None,
                position=None,
            ),
            text=None,
            symbology_identifier=None,
            content_type=None,
        )

    payload = bytes(result.bytes)
    return PrimaryReading(
        outcome=DecoderOutcome(
            decoder=DECODER_NAME,
            status="decoded",
            polarity=polarity,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            reported_ec=_optional_text(result.ec_level),
            # zxing-cpp 2.2.0 exposes neither a version nor a mask, so this is None by contract
            # rather than by omission. Reading it through getattr means a binding that does report
            # one (3.x carries both in Result.extra) populates the field with no consumer change,
            # and nothing is ever synthesised from the payload or the symbol size.
            reported_version=getattr(result, "version", None),
            error_text=None,
            payload=payload,
            position=_position(result),
        ),
        text=_optional_text(result.text),
        symbology_identifier=_optional_text(result.symbology_identifier),
        content_type=_content_type(result),
    )
