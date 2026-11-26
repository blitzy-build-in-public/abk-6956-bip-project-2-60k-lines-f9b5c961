"""The secondary reference decoder: OpenCV's ``QRCodeDetector``, corroboration only.

``opencv-python-headless`` 4.10 is the *second opinion* of the decode pipeline.
Its payload corroborates the primary decoder's, and its rectified module image is
the fallback geometry source when the primary decoder's corner quadrangle cannot
be snapped to a legal module count. It is never the authority for any check --
the primary decoder is -- and its silence is recorded as ``undetected``, never as
a disagreement, so a symbol this decoder cannot read can never fail a conforming
target.

Four limitations of the pinned binding are documented here because each one
changes how the consensus layer must read this decoder's answer:

* **No metadata.** ``detectAndDecode`` exposes neither the version, the error
  correction level nor the mask, so ``reported_ec`` and ``reported_version`` are
  always ``None``. That is evidence, not a gap: a future binding that reports
  them populates the fields without a consumer change.
* **No inverted detection.** A light-on-dark symbol is not detected at all
  (verified against 4.10.0), which is why ``consensus.py`` decodes the
  255-inverted copy for an inverted request instead of relying on decoder
  support.
* **Fragile under damage.** Blur, low resolution, distortion and occlusion make
  it either miss the symbol or detect it without decoding. The two cases are
  mapped to distinct ``error_text`` values so ``decoded.json`` shows which
  happened.
* **Narrow quiet zones are unreliable.** Detection is not guaranteed for a
  one-module border, so such a symbol may report ``undetected`` here while the
  primary decoder reads it -- the ``primary_only`` case consensus is built to
  record rather than to judge. Nothing may be concluded from this decoder's
  silence under damage.

Three contracts hold for everything in this module:

* **It never raises for a decode failure.** A missing binding, or an exception
  from it, becomes an outcome with ``status="error"``, because the consensus
  layer treats a secondary error as absence and must never be able to
  manufacture a ``decoder_disagreement`` out of it. ``ValueError`` is reserved
  for a caller defect: an image that is not a ``numpy.uint8`` grayscale or
  three-channel array.
* **Nothing heavy is imported at import time.** ``cv2`` is resolved lazily, so
  ``import qrcheck.decode`` carries no native image stack and the degradation
  path stays exercisable.
* **A decoded payload is text, not bytes.** ``detectAndDecode`` returns ``str``,
  so ``payload`` is the UTF-8 encoding of that text and consensus compares this
  decoder *as text* -- never as the byte-exact authority, which is what keeps
  the binary corpus classes from failing round-trip on a conforming target. A
  zero-length payload is not representable at all: the binding returns ``""``
  for an empty payload exactly as it does for a symbol it detected but could not
  decode, and both map to ``undetected`` here.

The decoder is a degradable dependency: when it is not importable the run
proceeds in single-decoder mode with ``agreement == "primary_only"`` and a
banner in ``index.html``. No check is ever marked passed on the strength of the
missing decoder.
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
    "SecondaryReading",
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

#: ``points is None``: the binding found no symbol anywhere in the image.
_NO_SYMBOL_TEXT: Final = "no symbol detected"

#: ``points`` present with empty text: the geometry was found but no payload
#: was recovered. Also what a zero-length payload looks like, which the binding
#: cannot distinguish.
_NOT_DECODED_TEXT: Final = "symbol detected but not decoded"


@dataclasses.dataclass(frozen=True, slots=True)
class SecondaryReading:
    """One call's outcome, plus the rectified module image the binding produced.

    ``straight_qrcode`` is the only artefact this decoder contributes that
    :class:`~qrcheck.decode.base.DecoderOutcome` cannot carry: OpenCV's
    rectified square, exactly ``N`` by ``N`` pixels for ``N = 4V + 17``, which
    ``qrcheck/decode/matrix.py`` uses as its fallback module-count source when
    the primary decoder's quadrangle cannot be snapped to a legal dimension
    (recorded there as ``SampledMatrix.source == "straight_qrcode"``). The array
    is a private C-order ``numpy.uint8`` copy, so a consumer may read it freely
    and no consumer can mutate the binding's buffer. It is ``None`` whenever
    there was nothing to rectify: no detection, a detection without a decode, a
    missing dependency, or an error.

    ``outcome`` is the answer itself -- payload, payload digest, corner
    quadrangle, polarity and error text -- and is what the consensus layer
    records in ``per_decoder``.
    """

    outcome: DecoderOutcome
    straight_qrcode: numpy.ndarray | None = None


def decode_secondary(image: numpy.ndarray, *, polarity: Polarity = "original") -> DecoderOutcome:
    """Decode ``image`` with OpenCV and return the outcome on its own.

    This is the interface the decode pipeline consumes; a caller that also needs
    the rectified module image as a geometry fallback calls
    :func:`read_secondary` instead and takes its ``outcome``.

    Args:
        image: An ``H x W`` ``numpy.uint8`` grayscale or ``H x W x 3``
            ``numpy.uint8`` colour image. Grayscale is preferred and is what the
            consensus layer passes; for a three-channel array OpenCV reads the
            channels as BGR, which grayscale avoids entirely.
        polarity: The polarity ``image`` was presented at. It is recorded
            verbatim on the outcome so evidence shows which pass produced the
            answer; this wrapper never inverts the image itself.

    Returns:
        The decoder's outcome: ``decoded`` with a UTF-8 payload and its digest,
        ``undetected`` with ``no symbol detected`` or ``symbol detected but not
        decoded`` in ``error_text``, or ``error`` when the binding is missing or
        raised.

    Raises:
        ValueError: ``image`` is not a non-empty ``numpy.uint8`` grayscale or
            three-channel array -- a caller defect rather than a decode result,
            which is why it is the one thing this module raises.
    """
    return read_secondary(image, polarity=polarity).outcome


def read_secondary(
    image: numpy.ndarray, *, polarity: Polarity = "original"
) -> SecondaryReading:
    """Decode ``image`` with OpenCV, returning the outcome and the rectified grid.

    The binding is called as ``QRCodeDetector().detectAndDecode(image)``, which
    returns ``(decoded_info, points, straight_qrcode)``. The mapping is fixed:

    * ``points is None`` -- nothing was detected anywhere: ``status`` is
      ``undetected`` and ``error_text`` is ``"no symbol detected"``.
    * ``points`` present but ``decoded_info`` empty -- the geometry was found and
      the payload was not: ``status`` is ``undetected``, ``error_text`` is
      ``"symbol detected but not decoded"``, and ``position`` carries the real
      quadrangle, because the geometry is genuine even though the payload is
      not. A zero-length payload is indistinguishable from this case and is
      reported the same way.
    * ``decoded_info`` non-empty -- ``status`` is ``decoded``, ``payload`` is the
      UTF-8 encoding of that text, ``payload_sha256`` is its ``sha256`` digest
      and ``position`` is the four corner ``(x, y)`` pairs of the binding's
      ``(1, 4, 2)`` array.

    A fresh ``QRCodeDetector`` is constructed per call, so no mutable state is
    shared between calls and two runs of the same image cannot influence each
    other. ``reported_ec`` and ``reported_version`` are always ``None``: the
    binding reports neither.

    Args:
        image: An ``H x W`` ``numpy.uint8`` grayscale or ``H x W x 3``
            ``numpy.uint8`` colour image (BGR channel order for colour).
        polarity: The polarity ``image`` was presented at, recorded verbatim on
            the outcome. Inversion is the caller's job: this wrapper does not
            invert, because the second pass for an inverted request belongs to
            the consensus layer that also runs the primary decoder on it.

    Returns:
        A :class:`SecondaryReading` whose ``outcome`` never carries a fabricated
        payload and whose ``straight_qrcode`` is the rectified ``N x N`` module
        image when the binding produced one, else ``None``.

    Raises:
        ValueError: ``image`` is not a non-empty ``numpy.uint8`` grayscale or
            three-channel array. This is the only failure this function raises:
            a missing or failing decoder is reported as an outcome.
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


def _load_cv2() -> types.ModuleType | None:
    """Return the ``cv2`` module, or ``None`` when it is not importable.

    Resolved at call time rather than at module import time, so importing the
    decode package never loads a native image stack and ``qrcheck doctor`` can
    report this decoder's absence without paying for its import. An import
    failure is the documented degradation: the caller reports it as an outcome
    rather than an exception.
    """
    try:
        return importlib.import_module("cv2")
    except ImportError:
        return None


def _as_decodable_image(image: numpy.ndarray) -> numpy.ndarray:
    """Validate ``image`` and return it as a contiguous array the binding accepts.

    Only the two shapes the pipeline produces are accepted -- ``H x W``
    grayscale and ``H x W x 3`` colour -- so an image of another dtype or shape
    can never decode differently by accident; the binding would raise an opaque
    overload error for a float or boolean array instead of a diagnosable one.

    Args:
        image: The array a caller presented for decoding.

    Returns:
        The same pixels as a C-contiguous ``numpy.uint8`` array.

    Raises:
        ValueError: ``image`` is empty, is not ``numpy.uint8``, or is neither
            two-dimensional nor three-channel.
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

    ``detectAndDecode`` returns a ``str``, and the empty string is the binding's
    only signal for "no payload": it is what a symbol it detected but could not
    decode produces, and also what a zero-length payload produces. Neither is
    turned into bytes, so a zero-length payload is reported as ``undetected``
    rather than as a decoded empty payload consensus could compare against a
    legitimate empty request.
    """
    if not isinstance(decoded_info, str) or decoded_info == "":
        return None
    return decoded_info.encode("utf-8")


def _quad(points: object) -> tuple[tuple[float, float], ...] | None:
    """Return the detected corner quadrangle as four ``(x, y)`` float pairs.

    The binding reports a ``(1, 4, 2)`` array of the symbol's corners in
    source-image pixel coordinates. Anything that is not exactly four corners
    yields ``None`` rather than a partial geometry, so no consumer ever samples
    a module grid from an incomplete quadrangle.
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

    The array is ``N x N`` for ``N = 4V + 17`` and is the fallback module-count
    source for grid sampling. It is copied into C order so the caller can
    neither mutate the buffer the binding owns nor depend on its strides, and it
    is ``None`` whenever the binding produced no rectified image.
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
