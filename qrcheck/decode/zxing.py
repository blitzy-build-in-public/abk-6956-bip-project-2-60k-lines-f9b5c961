"""Primary reference decoder: the authority every check compares against.

The framework never trusts a single decoder's opinion. This module wraps the
pinned ``zxing-cpp`` 2.2.0 binding as the *primary* reference decoder, whose
payload is authoritative for all twenty-one checks: a symbol the primary
decoder cannot read is a failed or errored case even when the secondary
decoder reads it, and the secondary decoder is never allowed to overwrite
what this module reported.

The binding surface it targets is the flat function surface of 2.2.0, and
nothing beyond it:

* :func:`zxingcpp.read_barcode` -- there is no ``ReaderOptions`` class and no
  ``try_harder`` keyword. The supported keywords are ``formats``,
  ``try_rotate``, ``try_downscale``, ``text_mode``, ``binarizer``, ``is_pure``,
  ``ean_add_on_symbol`` and ``return_errors``, and the image is a
  ``numpy`` array.
* :class:`zxingcpp.Result` -- exposes ``valid``, ``text``, ``bytes``,
  ``format``, ``symbology_identifier`` (``"]Q1"`` without an ECI designator,
  ``"]Q2"`` with one), ``ec_level`` (a ``str`` such as ``"M"``, added in
  2.2.0), ``content_type``, ``position``, ``orientation`` and ``error``.

What the binding does **not** provide shapes the rest of the decode layer.
2.2.0 exposes no version and no mask accessor -- both arrive only in 3.x, via
``Result.extra`` -- so the version is recovered from the sampled module grid
(``N = 4V + 17``) and the mask from the symbol's own format information, and
the corner quadrangle this module publishes is the independent geometry that
sampling needs. ``reported_version`` is therefore ``None`` here by design, not
by omission.

Two behaviours of a decoder's silence are contract, not detail. A decoder that
returns no payload is ``undetected``, which is recorded and is never a
disagreement; only two differing payloads are. And a payload of ``b""`` -- an
empty symbol the decoder did read -- is a different fact from ``None``: the
invalid-input checks depend on that distinction.

Nothing here raises on behalf of a symbol. A missing binding and a binding
that throws both become an ``error`` outcome carrying the text to explain it,
because a hostile or broken symbol must leave evidence behind and let the run
continue rather than abort it. The single exception is an argument that is not
an image at all: that is a defect in the caller, and :func:`read_primary`
raises :class:`ValueError` for it.

The binding is imported lazily, per call, so importing this module never loads
native decoder code: ``doctor``, the capability resolver and the test suite all
import the decode layer on machines where the primary decoder may be absent,
and its absence is reported through
:func:`qrcheck.decode.base.primary_available` rather than as an
``ImportError`` escaping an import of this package.
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

    The import is deliberately performed here instead of at module scope, and
    it is not cached: after the first import the statement is a
    ``sys.modules`` lookup, and resolving afresh keeps the absence path
    exercisable at any point in a process's life -- a cached failure would
    outlive a test that installs the binding, and a cached success would
    outlive one that removes it.

    An ``ImportError`` raised *inside* the binding's own import counts as
    absent too, and so does an ``OSError``: a compiled extension whose shared
    object cannot be loaded -- a wrong architecture, a missing native library
    -- fails that way rather than with ``ImportError``, and it is equally
    unusable. Either way the caller gets the diagnostic naming the distribution
    it has to fix instead of a traceback out of a decoder probe.
    """
    try:
        import zxingcpp
    except (ImportError, OSError):
        return None
    return zxingcpp


def _validate_image(image: numpy.ndarray) -> None:
    """Reject any argument that is not a supported image array.

    The accepted forms are exactly the two the decode layer produces: an
    ``H x W`` ``uint8`` grayscale array and an ``H x W x 3`` ``uint8`` RGB
    array. No other dtype is accepted, so a float image cannot quietly decode
    against a different pixel scale than the ``uint8`` one the framework
    measured and recorded.

    Raises:
        ValueError: ``image`` is not a ``numpy.ndarray``, is not ``uint8``, or
            has a shape that is neither ``H x W`` nor ``H x W x 3``. This is a
            defect in the caller -- the consensus layer only ever passes arrays
            it built -- so it is raised rather than turned into an outcome:
            recording it as a decoder failure would misattribute a programmer
            error to the target under test.
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
        f"image must be HxW grayscale or HxWx3 RGB uint8; got shape {tuple(image.shape)}"
    )


def _optional_text(value: object) -> str | None:
    """Return ``value`` as text, preserving ``None`` and ``""`` as distinct.

    An absent value and an empty value are different evidence: the decoder
    reports nothing for metadata it lacks and an empty string for a payload it
    read as empty, so neither is coerced into the other.
    """
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _content_type(result: object) -> str | None:
    """Return the decoder's content classification as a stable string.

    ``zxingcpp`` reports a ``ContentType`` enum member rather than text, and
    the member's ``repr`` names the binding's module path, which would tie a
    recorded piece of evidence to an implementation detail. The member's name
    (``"Text"``, ``"Binary"``, ``"Mixed"``, ...) is recorded instead: it is
    stable, readable and JSON-safe, which :class:`DecodeResult` requires.
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

    The order is the decoder's own -- top-left, top-right, bottom-right,
    bottom-left -- and the coordinates are the corners of the symbol itself,
    quiet zone excluded. Grid sampling rectifies this quadrangle to recover the
    module matrix, which is the only way the version is knowable at all
    because no pinned decoder reports one.

    ``None`` is returned when the decoder reported no quadrangle, and also when
    any single corner is missing: a square fabricated from a partial reading
    would rectify the wrong region and produce a matrix that is wrong without
    looking wrong, whereas ``None`` sends sampling to its documented fallback.
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

    ``read_barcode`` returns ``None`` when it finds nothing -- ``return_errors``
    only surfaces a symbol it *did* detect but could not trust, such as one
    failing a checksum -- so a missing result has no message of its own. The
    decoder's text is preserved verbatim whenever there is one, because a
    failing check reports it unchanged.
    """
    if result is None:
        return _NO_SYMBOL_MESSAGE
    message = _optional_text(getattr(result, "error", None))
    return message or _NO_SYMBOL_MESSAGE


def _failed_outcome(polarity: Polarity, error_text: str) -> DecoderOutcome:
    """Build the ``error`` outcome for a decoder that could not produce a verdict.

    Used for the two ways the decoder itself can fail to run -- an absent
    binding and a binding that raised -- so both paths record identical
    metadata and differ only in the text that explains them.
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

    :class:`DecoderOutcome` deliberately carries only what every decoder can
    answer -- payload, digest, EC level, error text and geometry. The
    :class:`DecodeResult` this reading feeds into also carries the
    authoritative decoder's rendered text, its symbology identifier and its
    content classification, and those are this decoder's alone, so they travel
    beside the outcome rather than being widened into a type both decoders
    must fill.

    ``text`` is the decoder's rendering of the payload under its default
    ``TextMode.HRI``; it is never what a round-trip assertion compares, because
    the rendering is the decoder's choice and only ``payload`` is byte-exact.
    """

    outcome: DecoderOutcome
    text: str | None
    symbology_identifier: str | None
    content_type: str | None


def decode_primary(image: numpy.ndarray, *, polarity: Polarity = "original") -> DecoderOutcome:
    """Decode ``image`` with the primary reference decoder.

    This is the interface the rest of the decode layer calls. It returns the
    outcome alone -- ``decoded``, ``undetected`` or ``error`` -- and never
    raises for anything the symbol or the binding does.

    Args:
        image: An ``H x W x 3`` ``uint8`` RGB array or an ``H x W`` ``uint8``
            grayscale array. Any other dtype or shape is rejected, so pixels
            cannot be interpreted at a scale the framework never measured.
        polarity: The polarity of the pixels in ``image``, stamped onto the
            outcome for the record. This function never inverts anything: the
            consensus layer runs an inverted request's second pass on its own
            255-inverted copy, so the caller always states which copy it
            passed.

    Returns:
        The decoder's outcome. ``status`` is ``decoded`` when a payload was
        read (an empty payload is a payload), ``undetected`` when no symbol was
        found, and ``error`` when the decoder could not run or raised.

    Raises:
        ValueError: ``image`` is not a supported array, which is a defect in
            the caller rather than a fact about the symbol.
    """
    return read_primary(image, polarity=polarity).outcome


def read_primary(image: numpy.ndarray, *, polarity: Polarity = "original") -> PrimaryReading:
    """Decode ``image`` and return the outcome together with its text metadata.

    Args:
        image: An ``H x W x 3`` ``uint8`` RGB array or an ``H x W`` ``uint8``
            grayscale array.
        polarity: The polarity of the pixels in ``image``, stamped onto the
            outcome. ``"original"`` for the artifact as the target produced it,
            ``"inverted"`` for the 255-inverted copy the consensus layer builds
            for an inverted request.

    Returns:
        The reading. On success the outcome is ``decoded`` with the payload
        bytes, its SHA-256, the decoder's EC level and the corner quadrangle,
        and the reading's own fields carry the rendered text, the symbology
        identifier and the content type. On a symbol the decoder could not find
        the outcome is ``undetected`` with the decoder's message; when the
        binding is absent or raises, the outcome is ``error`` and says why.

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
        # Deliberately broad, and the only catch of this width in the module:
        # a decoder that throws on a hostile symbol must leave an evidence
        # record and let the run continue, so every failure of the binding --
        # a buffer it refuses, a native fault, an unexpected shape -- becomes
        # an "error" outcome carrying the exception's own text. Narrowing this
        # would let an unanticipated binding failure abort a run that is
        # supposed to report on misbehaviour rather than die of it.
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
            # zxing-cpp 2.2.0 exposes neither a version nor a mask, so this is
            # None by contract rather than by omission. Reading it through
            # getattr means a binding that does report one (3.x carries both in
            # Result.extra) populates the field with no consumer change, and
            # nothing is ever synthesised from the payload or the symbol size.
            reported_version=getattr(result, "version", None),
            error_text=None,
            payload=payload,
            position=_position(result),
        ),
        text=_optional_text(result.text),
        symbology_identifier=_optional_text(result.symbology_identifier),
        content_type=_content_type(result),
    )
