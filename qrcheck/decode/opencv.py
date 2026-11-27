"""The secondary reference decoder: OpenCV's ``QRCodeDetector``, corroboration only.

``opencv-python-headless`` 4.10 is the second opinion of the decode pipeline: its payload
corroborates the primary decoder's and its rectified module image is the fallback geometry source
when the primary's corner quadrangle cannot be snapped to a legal module count, but it is never the
authority for any check and its silence is ``undetected``, never a disagreement, so a symbol it
cannot read can never fail a conforming target.

Four limitations of the pinned binding shape how consensus reads it: no metadata at all
(``reported_ec`` and ``reported_version`` are always ``None``, which is evidence rather than a gap);
no inverted detection, which is why consensus decodes the 255-inverted copy itself; fragility under
damage, where a miss and a detection without a decode map to distinct ``error_text`` values; and an
unreliable one-module quiet zone, which produces the ``primary_only`` case consensus records rather
than judges.

It never raises for a decode failure -- an exception from the binding becomes ``status="error"``,
because consensus treats a secondary error as absence and must never manufacture a
``decoder_disagreement``. A binding that cannot be imported is not a decode failure at all:
:func:`binding_problem` lets consensus record it as an unavailable decoder rather than as an outcome
(AAP §0.4.5), while a direct call to :func:`read_secondary` still answers with ``status="error"``.
``cv2`` is resolved lazily, and ``ValueError`` is reserved for a caller defect.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import types
from typing import Final

import numpy

from qrcheck.decode.base import SECONDARY_DECODER, DecoderOutcome, Polarity

__all__ = [
    "DECODER_NAME",
    "RESOLUTION_ABSENT_TEXT",
    "SecondaryReading",
    "binding_problem",
    "decode_secondary",
    "read_secondary",
]

#: The name this decoder reports in every outcome and in ``per_decoder``; it is
#: ``base.SECONDARY_DECODER`` itself, so the two can never drift apart.
DECODER_NAME: Final = SECONDARY_DECODER

#: Reported when the binding cannot be imported at all: the decode pipeline's
#: one degradable dependency, named so the diagnostic says what to install.
_MISSING_DEPENDENCY_TEXT: Final = (
    "opencv-python-headless is not importable; the secondary decoder is a "
    "degradable dependency"
)

#: Recorded when the run's own resolution left the secondary decoder out, so it was
#: never asked: the reason a caller reads in ``DecodeResult.unavailable_decoders``.
RESOLUTION_ABSENT_TEXT: Final = "the run resolved the secondary decoder as unavailable"

#: ``points is None``: the binding found no symbol anywhere in the image.
_NO_SYMBOL_TEXT: Final = "no symbol detected"

#: ``points`` present with empty text: the geometry was found but no payload
#: was recovered. Also what a zero-length payload looks like, which the binding
#: cannot distinguish.
_NOT_DECODED_TEXT: Final = "symbol detected but not decoded"


@dataclasses.dataclass(frozen=True, slots=True)
class SecondaryReading:
    """One call's outcome, plus the rectified module image the binding produced.

    ``straight_qrcode`` is the one artefact :class:`~qrcheck.decode.base.DecoderOutcome` cannot
    carry: OpenCV's rectified square, exactly ``N`` by ``N`` pixels, which ``matrix.py`` uses as its
    fallback module-count source (``SampledMatrix.source == "straight_qrcode"``). It is a private
    C-order copy and ``None`` whenever there was nothing to rectify.
    """

    outcome: DecoderOutcome
    straight_qrcode: numpy.ndarray | None = None


def decode_secondary(image: numpy.ndarray, *, polarity: Polarity = "original") -> DecoderOutcome:
    """Decode ``image`` with OpenCV and return the outcome on its own.

    The interface the decode pipeline consumes; a caller that also needs the rectified module image
    as a geometry fallback calls :func:`read_secondary`. ``polarity`` is recorded verbatim on the
    outcome, and this wrapper never inverts the image itself.

    Raises:
        ValueError: ``image`` is not a non-empty ``numpy.uint8`` grayscale or three-channel
            array -- a caller defect rather than a decode result.
    """
    return read_secondary(image, polarity=polarity).outcome


def read_secondary(
    image: numpy.ndarray, *, polarity: Polarity = "original"
) -> SecondaryReading:
    """Decode ``image`` with OpenCV, returning the outcome and the rectified grid.

    ``QRCodeDetector().detectAndDecode(image)`` returns ``(decoded_info, points, straight_qrcode)``,
    and the mapping is fixed: ``points is None`` is ``undetected`` with ``error_text`` ``"no symbol
    detected"``; ``points`` present without ``decoded_info`` is ``undetected`` with ``"symbol
    detected but not decoded"`` while ``position`` keeps the genuine geometry; non-empty
    ``decoded_info`` is ``decoded`` with the UTF-8 payload, its ``sha256`` digest and the four
    corner pairs. A fresh detector per call shares no state, and ``reported_ec`` with
    ``reported_version`` are always ``None``.

    Raises:
        ValueError: ``image`` is not a non-empty ``numpy.uint8`` array of an accepted shape; a
            missing or failing decoder is an outcome, not a raise.
    """
    decodable = _as_decodable_image(image)
    module = _load_cv2()
    if module is None:
        return SecondaryReading(
            outcome=DecoderOutcome(
                decoder=DECODER_NAME,
                status="error",
                polarity=polarity,
                payload_sha256=None,
                reported_ec=None,
                reported_version=None,
                error_text=_MISSING_DEPENDENCY_TEXT,
            )
        )

    try:
        decoded_info, points, straight_qrcode = module.QRCodeDetector().detectAndDecode(decodable)
    except Exception as exc:  # noqa: BLE001 - the closed error vocabulary of the pipeline
        # Deliberate, and the only broad catch in this module: the consensus
        # layer treats a secondary decoder's failure as absence, and an
        # exception escaping here would abort a check that the primary decoder
        # can still answer. The binding's own message is preserved verbatim so
        # the cause stays diagnosable from evidence.
        return SecondaryReading(
            outcome=DecoderOutcome(
                decoder=DECODER_NAME,
                status="error",
                polarity=polarity,
                payload_sha256=None,
                reported_ec=None,
                reported_version=None,
                error_text=f"{type(exc).__name__}: {exc}",
            )
        )

    # ``reported_ec`` and ``reported_version`` are ``None`` in every outcome
    # below: the binding exposes neither, so the absence is the evidence rather
    # than a gap in the mapping.
    position = _quad(points)
    rectified = _rectified(straight_qrcode)
    payload = _payload_from(decoded_info)
    if payload is None:
        return SecondaryReading(
            outcome=DecoderOutcome(
                decoder=DECODER_NAME,
                status="undetected",
                polarity=polarity,
                payload_sha256=None,
                reported_ec=None,
                reported_version=None,
                error_text=_NOT_DECODED_TEXT if points is not None else _NO_SYMBOL_TEXT,
                position=position,
            ),
            straight_qrcode=rectified,
        )
    return SecondaryReading(
        outcome=DecoderOutcome(
            decoder=DECODER_NAME,
            status="decoded",
            polarity=polarity,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            reported_ec=None,
            reported_version=None,
            error_text=None,
            payload=payload,
            position=position,
        ),
        straight_qrcode=rectified,
    )


def binding_problem() -> str | None:
    """Return why the binding cannot be consulted, or ``None`` when it can.

    The probe the decode pipeline consults *before* it asks this decoder anything: a binding that
    cannot be imported raised nothing about any artefact, so it is recorded as an unavailable
    decoder rather than as an outcome with ``status="error"``, which is what makes single-decoder
    mode's evidence list one decoder (AAP §0.6.6 row 4, §0.4.5). Built on :func:`_load_cv2`, so an
    :exc:`ImportError` from an absent distribution and an :exc:`OSError` from a compiled extension
    that cannot be loaded both mean unavailable, and a successful import is cached by the
    interpreter, so a probe before every decode costs one import for the whole run.

    Never raises. :func:`read_secondary` keeps its own behaviour, so a direct caller that asks for
    a decode still gets an ``error`` outcome when the binding is missing.
    """
    return None if _load_cv2() is not None else _MISSING_DEPENDENCY_TEXT


def _load_cv2() -> types.ModuleType | None:
    """Return the ``cv2`` module, or ``None`` when it is not usable.

    Resolved at call time, so importing the decode package loads no native image stack. An
    ``ImportError`` from an absent distribution and an ``OSError`` from a compiled extension that
    cannot be loaded both mean absent, and neither escapes as a traceback.
    """
    try:
        return importlib.import_module("cv2")
    except (ImportError, OSError):
        return None


def _as_decodable_image(image: numpy.ndarray) -> numpy.ndarray:
    """Validate ``image`` and return it as a contiguous array the binding accepts.

    Only the two shapes the pipeline produces are accepted -- ``H x W`` grayscale and ``H x W x 3``
    colour -- so an image of another dtype or shape cannot decode differently by accident, the
    binding raising an opaque overload error instead of a diagnosable one. ``ValueError`` when
    ``image`` is empty, is not ``numpy.uint8``, or is neither two-dimensional nor three-channel.
    """
    array = numpy.asarray(image)
    if array.dtype != numpy.uint8:
        raise ValueError(
            f"the secondary decoder needs a numpy.uint8 image; got dtype {array.dtype}"
        )
    if array.ndim == 3:
        if array.shape[2] != 3:
            raise ValueError(
                "the secondary decoder needs an HxW grayscale or HxWx3 colour image; "
                f"got shape {array.shape}"
            )
    elif array.ndim != 2:
        raise ValueError(
            "the secondary decoder needs an HxW grayscale or HxWx3 colour image; "
            f"got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError(f"the secondary decoder needs a non-empty image; got shape {array.shape}")
    return numpy.ascontiguousarray(array)


def _payload_from(decoded_info: object) -> bytes | None:
    """Return the payload the binding reported, or ``None`` when it reported none.

    ``""`` is the binding's only signal for no payload -- a symbol detected but not decoded and a
    zero-length payload alike -- so it is never turned into bytes: a zero-length payload would
    otherwise be compared against a legitimate empty request.
    """
    if not isinstance(decoded_info, str) or decoded_info == "":
        return None
    return decoded_info.encode("utf-8")


def _quad(points: object) -> tuple[tuple[float, float], ...] | None:
    """Return the detected corner quadrangle as four ``(x, y)`` float pairs.

    The binding reports a ``(1, 4, 2)`` array of the symbol's corners in source-image pixel
    coordinates; anything that is not exactly four corners yields ``None``, so no consumer ever
    samples a grid from an incomplete quadrangle.
    """
    if points is None:
        return None
    try:
        corners = numpy.asarray(points, dtype=numpy.float64)
    except (TypeError, ValueError):
        return None
    if corners.size != 8:
        return None
    return tuple((float(x), float(y)) for x, y in corners.reshape(4, 2))


def _rectified(straight_qrcode: object) -> numpy.ndarray | None:
    """Return the binding's rectified module image as a private ``uint8`` copy.

    The ``N x N`` array is the fallback module-count source for grid sampling; it is copied into C
    order so no caller mutates the buffer the binding owns, and is ``None`` whenever the binding
    produced no rectified image.
    """
    if straight_qrcode is None:
        return None
    try:
        array = numpy.asarray(straight_qrcode, dtype=numpy.uint8)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.size == 0:
        return None
    return numpy.array(array, dtype=numpy.uint8, order="C", copy=True)
