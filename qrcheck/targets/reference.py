"""The conforming self-check target: a segno-backed generator that must come back clean.

This target proves the suite is not vacuous: ``qrcheck run --adapter reference`` must report zero
failed, zero errored and zero skipped results with at least one ``passed`` result for every one of
the twenty-one checks, while :mod:`qrcheck.targets.defective` must fail exactly its nine. Weakening
a check to make it pass here is forbidden.

Two segno disciplines are mandatory at the call site. ``boost_error=False``: segno's default spends
spare capacity on a *higher* error-correction level without changing the version. Model 2
only: Micro QR has no ``4V+17`` module count and a different format-information layout, so every
capacity and version computation would be wrong -- hence ``segno.make_qr()`` and never bare
``segno.make()``, which returns Micro QR for short content (``make_qr`` takes no ``micro`` argument
in segno 1.6.6).

All thirteen capabilities are declared unconditionally, because ``capabilities`` is a gate and
narrowing it would let a check skip. ``logo_overlay_path`` is applied through
:func:`qrcheck.evidence.mux.composite_overlay`, the product's single image boundary, and an overlay
on a vector artefact is refused with ``TargetUnsupported`` before anything is encoded. Documented
limitation: segno is also the oracle, so ``SY-MATRIX-EQUIV`` compares it with itself in the encoder
dimension; the independent evidence is the decoder-derived set.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Final, Literal

import segno

from qrcheck.config.models import TargetConfig
from qrcheck.corpus.capacity import EC_LEVELS, max_payload_bytes
from qrcheck.corpus.payloads import mode_of
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
    ensure_image_bytes,
    require_capabilities,
)

__all__ = [
    "DEFAULT_MODULE_SIZE_PX",
    "DEFAULT_QUIET_ZONE_MODULES",
    "ReferenceAdapter",
]

# This adapter's defaults for the two geometry fields a request may leave unset: four
# modules of quiet zone, the ISO/IEC 18004 minimum, and four pixels per module.
DEFAULT_MODULE_SIZE_PX: Final[int] = 4
DEFAULT_QUIET_ZONE_MODULES: Final[int] = 4

# ISO/IEC 18004 Model 2 bounds; outside these is outside the standard.
_MIN_VERSION: Final[int] = 1
_MAX_VERSION: Final[int] = 40
_MIN_MASK: Final[int] = 0
_MAX_MASK: Final[int] = 7

# A colour channel is one unsigned byte.
_MIN_CHANNEL: Final[int] = 0
_MAX_CHANNEL: Final[int] = 255

# The two artefact kinds segno renders for us, and the segno `kind` name of each.
_RENDER_KINDS: Final[tuple[str, ...]] = ("png", "svg")

# WCAG 2.1 sRGB relative-luminance constants. A channel is linearised piecewise -- a
# straight division below the knee, a gamma expansion above it -- and the luminance is
# the weighted sum, which is what makes the ratio below order-independent and therefore
# equally lawful for an inverted (light-on-dark) request.
_SRGB_KNEE: Final[float] = 0.03928
_SRGB_SLOPE: Final[float] = 12.92
_SRGB_OFFSET: Final[float] = 0.055
_SRGB_SCALE: Final[float] = 1.055
_SRGB_GAMMA: Final[float] = 2.4
_LUMINANCE_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

# The additive term of the WCAG formula, so two identical colours give exactly 1.0.
_CONTRAST_OFFSET: Final[float] = 0.05


def _linear_channel(channel: int) -> float:
    """Return one 0..255 sRGB channel as a linearised luminance component.

    ``channel`` has already been validated as an integer in 0..255 by the caller, so this function
    raises nothing.
    """
    value = channel / 255.0
    if value <= _SRGB_KNEE:
        return value / _SRGB_SLOPE
    return ((value + _SRGB_OFFSET) / _SRGB_SCALE) ** _SRGB_GAMMA


def _relative_luminance(colour: tuple[int, int, int]) -> float:
    """Return the WCAG 2.1 relative luminance of a three-channel ``0..255`` colour.

    The result is in 0.0..1.0, black at 0.0 and white at 1.0, and raises nothing for a validated
    colour.
    """
    red, green, blue = (_linear_channel(channel) for channel in colour)
    weight_red, weight_green, weight_blue = _LUMINANCE_WEIGHTS
    return weight_red * red + weight_green * green + weight_blue * blue


def _contrast_ratio(
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
) -> float:
    """Return the WCAG 2.1 contrast ratio between two ``0..255`` colours, from 1.0 up.

    The lighter colour is placed over the darker one, so the ratio is order-independent and an
    inverted request is measured exactly as a normal one is: ``(lighter + 0.05) / (darker + 0.05)``,
    spanning 1.0 (identical colours) to 21.0 (black on white).
    """
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = (first, second) if first >= second else (second, first)
    return (lighter + _CONTRAST_OFFSET) / (darker + _CONTRAST_OFFSET)


def _colour_hex(colour: tuple[int, int, int]) -> str:
    """Return a validated ``0..255`` colour as the ``#rrggbb`` string segno renders.

    Both writers accept this form, which keeps one colour representation for the two artefact kinds;
    it raises nothing for a validated colour.
    """
    red, green, blue = colour
    return f"#{red:02x}{green:02x}{blue:02x}"


class ReferenceAdapter:
    """A conforming generator: segno with the two expectation-corrupting defaults fixed.

    The adapter is stateless, so two identical requests produce identical bytes and identical
    declared metadata, which is what ``DT-DETERMINISTIC`` compares. Every request is validated
    before segno sees it, because segno is a general-purpose encoder: it encodes an empty payload
    without complaint and reports only ``DataOverflowError`` for capacity. The ``IV-*`` checks
    assert a *declared* contract, so these messages are public API. Of the five adapter errors this
    class raises :class:`TargetRejected`, :class:`TargetContractError` and
    :class:`TargetUnsupported`, and never :class:`TargetUnavailable` or :class:`TargetTimeout`.
    """

    def __init__(self, config: TargetConfig, *, name: str | None = None) -> None:
        """Bind the adapter to its resolved ``[targets.<name>]`` record.

        Only ``contrast_floor`` is read: all thirteen capabilities are declared, and ``timeout_s``
        is not read because the encode has no boundary to bound.

        Raises:
        TypeError: ``contrast_floor`` is not a real number -- the record was built by hand.
        """
        self.name: str = config.type if name is None else name
        self.capabilities: frozenset[Capability] = frozenset(Capability)
        self._contrast_floor: float = float(config.contrast_floor)

    def __repr__(self) -> str:
        """Return a diagnostic form naming the target and its capability count."""
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"capabilities={len(self.capabilities)}, "
            f"contrast_floor={self._contrast_floor!r})"
        )

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Produce one symbol for `request`, or raise exactly one adapter error.

        The order is fixed: the capability gate first, then the validation chain, then the artefact
        kind and the vector-overlay refusal, then the encode and the render inside one timing
        window. The four ``declared_*`` fields are the target's claim, evidence only.

        Raises:
        TargetUnsupported: An undeclared capability, or a ``logo_overlay`` on a vector artefact.
        TargetRejected: The request breaks the target's own contract: an error level outside
        L/M/Q/H, a version or mask out of range, a non-positive module size, a negative quiet zone,
        a malformed colour, an empty payload, a payload that does not fit the pinned version, or a
        contrast ratio below the declared floor.
        TargetContractError: segno or the writer failed for a reason the request does not explain,
        the bytes are not the artefact requested, or the named overlay is unreadable.
        """
        require_capabilities(self, self._required_capabilities(request))
        self._validate(request)

        output_format = self._effective_output_format(request)
        if request.logo_overlay_path is not None and output_format != "png":
            # A composite needs a pixel grid, so the refusal precedes the encode and an unsupported
            # request
            # never reaches work it cannot be given.
            raise TargetUnsupported(
                f"adapter {self.name!r} cannot composite a 'logo_overlay' onto a "
                f"{output_format!r} artefact: only a raster symbol has a pixel grid to "
                "place it on"
            )
        started = time.perf_counter()
        code = self._encode(request)
        image_bytes = ensure_image_bytes(self._render(code, request), output_format)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return GeneratedSymbol(
            image_bytes=image_bytes,
            output_format=output_format,
            declared_version=code.version,
            declared_ec=self._declared_ec(request),
            declared_mask=code.mask,
            declared_mode=code.mode,
            elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Release what this adapter holds -- nothing -- and stay safe to call twice.

        The runner closes every adapter once a run ends, so this must never raise. There is no
        process, socket or file to release: each call renders into a fresh in-memory stream.
        """

    def _required_capabilities(self, request: GenerateRequest) -> frozenset[Capability]:
        """Return the capabilities `request` needs, for the gate before any work.

        Only the request is inspected: the format, the error level and the colours are always
        applied, and each pinned or optional field adds its own capability. The gate can never
        refuse for this adapter, which declares all thirteen, but it is still called: the port
        requires every implementation to refuse an undeclared operation before any I/O.
        """
        needed: set[Capability] = {Capability.ec_level, Capability.color}
        if request.output_format in _RENDER_KINDS:
            needed.add(Capability(request.output_format))
        if request.module_size_px is not None:
            needed.add(Capability.module_size)
        if request.quiet_zone_modules is not None:
            needed.add(Capability.quiet_zone)
        if request.logo_overlay_path is not None:
            needed.add(Capability.logo_overlay)
        if request.version is not None:
            needed.add(Capability.version_pin)
        if request.mask is not None:
            needed.add(Capability.mask_pin)
        return frozenset(needed)

    # Validation: the target's declared contract, checked before segno

    def _validate(self, request: GenerateRequest) -> None:
        """Run the three validation steps in their fixed order: shape, payload, contrast.

        The order is deliberately stable, so the message a probe receives never depends on the order
        rules happen to be written in.

        Raises:
        TargetRejected: The request breaks one of the target's own rules; the message names the
        field and is public API.
        """
        self._validate_shape(request)
        self._check_payload(request)
        self._check_contrast(request)

    def _validate_shape(self, request: GenerateRequest) -> None:
        """Refuse a request whose parameters are outside the target's contract.

        The six checks run in the order the contract lists them: error level, version pin, mask pin,
        module size, quiet zone and colours.

        Raises:
        TargetRejected: One parameter is outside its documented range.
        """
        ec_level = request.ec_level
        if ec_level not in EC_LEVELS:
            raise TargetRejected(f"invalid error correction level {ec_level!r}")

        version = request.version
        if version is not None and not _MIN_VERSION <= version <= _MAX_VERSION:
            raise TargetRejected(
                f"version must be in {_MIN_VERSION}..{_MAX_VERSION}; got {version}"
            )

        mask = request.mask
        if mask is not None and not _MIN_MASK <= mask <= _MAX_MASK:
            raise TargetRejected(f"mask must be in {_MIN_MASK}..{_MAX_MASK}; got {mask}")

        module_size = request.module_size_px
        if module_size is not None and module_size < 1:
            raise TargetRejected("module size must be greater than 0")

        quiet_zone = request.quiet_zone_modules
        if quiet_zone is not None and quiet_zone < 0:
            raise TargetRejected("quiet zone must not be negative")

        self._checked_colour("foreground_rgb", request.foreground_rgb)
        self._checked_colour("background_rgb", request.background_rgb)

    def _checked_colour(self, field: str, value: object) -> tuple[int, int, int]:
        """Return `value` as a three-channel ``0..255`` colour, or refuse it.

        ``bool`` is refused even though it is an ``int`` subclass, because ``True`` would silently
        render as 1.

        Raises:
        TargetRejected: `value` is not three integers in 0..255; the message names `field`.
        """
        expected = (
            f"{field} must be a 3-tuple of integers in "
            f"{_MIN_CHANNEL}..{_MAX_CHANNEL}"
        )
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise TargetRejected(f"{expected}; got {value!r}")
        channels: list[int] = []
        for channel in value:
            if (
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not _MIN_CHANNEL <= channel <= _MAX_CHANNEL
            ):
                raise TargetRejected(f"{expected}; got {value!r}")
            channels.append(channel)
        return (channels[0], channels[1], channels[2])

    def _check_payload(self, request: GenerateRequest) -> None:
        """Refuse an empty payload, then a payload the pinned version cannot hold.

        The capacity bound is the table row for the mode the payload's own bytes are limited to,
        because a conforming encoder spends the smallest charset that holds the payload; an unpinned
        request needs no rule here, since ``segno.DataOverflowError`` beyond Version 40 becomes the
        same rejection.

        Raises:
        TargetRejected: The payload is empty, or it does not fit the pinned version.
        """
        payload = self._effective_payload(request)
        if not payload:
            raise TargetRejected("empty payload: qrcheck requires at least one byte")

        version = request.version
        if version is None:
            return
        ec_level = self._effective_ec(request)
        if len(payload) > max_payload_bytes(version, ec_level, mode_of(payload)):
            raise TargetRejected(self._overflow_message(request, payload, ec_level))

    def _check_contrast(self, request: GenerateRequest) -> None:
        """Refuse a colour pair whose contrast is below the target's declared floor.

        The ratio is the WCAG 2.1 relative-luminance ratio, order-independent, so an inverted
        request is measured like a normal one, and the floor is the configured ``contrast_floor``.

        Raises:
        TargetRejected: The measured ratio is below the floor; both values are reported.
        """
        ratio = _contrast_ratio(request.foreground_rgb, request.background_rgb)
        if ratio < self._contrast_floor:
            raise TargetRejected(
                f"contrast ratio {ratio:.1f} is below the declared floor "
                f"{self._contrast_floor:.1f}"
            )

    # Effective values: the seams a derived target changes one at a time

    def _effective_ec(self, request: GenerateRequest) -> str:
        """Return the error-correction level to encode at: the requested one.

        This is the seam a target that quietly encodes at a lower level uses.
        """
        return request.ec_level

    def _effective_payload(self, request: GenerateRequest) -> bytes:
        """Return the bytes to encode: the requested payload, unmodified.

        Every payload decision in this class goes through this method, so a target that alters the
        payload alters it in one place and :meth:`_check_payload` measures the bytes actually
        encoded.
        """
        return request.payload

    def _effective_quiet_zone(self, request: GenerateRequest) -> int:
        """Return the quiet zone to render, in modules: the requested one or the default.

        The image is sized from exactly this value, so ``(modules + 2 * quiet_zone) * module_size``
        holds for whatever a subclass returns here and never drifts from the zone drawn.
        """
        quiet_zone = request.quiet_zone_modules
        return DEFAULT_QUIET_ZONE_MODULES if quiet_zone is None else quiet_zone

    def _effective_module_size(self, request: GenerateRequest) -> int:
        """Return the module edge to render, in pixels: the requested one or the default."""
        module_size = request.module_size_px
        return DEFAULT_MODULE_SIZE_PX if module_size is None else module_size

    def _effective_mask(self, request: GenerateRequest) -> int | None:
        """Return the mask to encode with: the pinned one, or ``None`` to let segno choose.

        A pin is honoured exactly, which keeps ``SY-MATRIX-EQUIV`` comparable at pinned parameters
        and leaves an unpinned request's mask an encoder choice.
        """
        return request.mask

    def _declared_ec(self, request: GenerateRequest) -> str:
        """Return the error-correction level this target *claims* it encoded at.

        For this target the claim is true and is the level segno used. It is a seam separate from
        :meth:`_effective_ec` because a target may encode at one level and declare another, and the
        claim is evidence only -- ``SY-EC-MATCH`` never reads it.
        """
        return request.ec_level

    def _effective_output_format(self, request: GenerateRequest) -> Literal["png", "svg"]:
        """Return the normalised artefact kind to render.

        Raises:
        TargetRejected: The request asks for neither ``png`` nor ``svg`` -- the belt-and-braces path
        for an out-of-contract request.
        """
        requested = request.output_format
        normalised = requested.strip().lower() if isinstance(requested, str) else ""
        if normalised == "png":
            return "png"
        if normalised == "svg":
            return "svg"
        raise TargetRejected(
            f"invalid output format {requested!r}; expected one of "
            f"{', '.join(_RENDER_KINDS)}"
        )

    # Encode and render

    def _encode(self, request: GenerateRequest) -> segno.QRCode:
        """Encode the request with segno, always Model 2 and never boosting the level.

        ``boost_error=False`` and ``make_qr`` are the two disciplines this method exists to hold,
        and the returned symbol's ``version``, ``error``, ``mask`` and ``mode`` are the declared
        metadata.

        Raises:
        TargetRejected: segno reports that the content does not fit or a parameter is illegal -- in
        the pinned release every such complaint is a plain ``ValueError``, so it is a request the
        target should itself have refused first.
        TargetContractError: segno failed for a reason the request does not explain.
        """
        payload = self._effective_payload(request)
        ec_level = self._effective_ec(request)
        try:
            return segno.make_qr(
                payload,
                error=ec_level,
                version=request.version,
                mask=self._effective_mask(request),
                boost_error=False,
            )
        except segno.DataOverflowError:
            raise TargetRejected(self._overflow_message(request, payload, ec_level)) from None
        # segno 1.6.6 defines exactly one dedicated exception, ``DataOverflowError`` above, and
        # reports an
        # illegal level, version or mask as plain ``ValueError`` -- the ``VersionError``,
        # ``ErrorLevelError``,
        # ``ModeError`` and ``MaskError`` the specification names do not exist in the pinned
        # release, so
        # this one clause converts them all to the rejection they are.
        except ValueError as exc:
            raise TargetRejected(str(exc)) from None
        except Exception as exc:
            raise TargetContractError(
                f"encoding {len(payload)} bytes at error level {ec_level} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from None

    def _overflow_message(
        self,
        request: GenerateRequest,
        payload: bytes,
        ec_level: str,
    ) -> str:
        """Return the capacity rejection message for an over-capacity `payload`.

        The version named is the pinned one, or Version 40 for an unpinned request -- the bound it
        actually broke. The capacity and its unit follow the mode the payload's own bytes are
        limited to: characters in numeric and alphanumeric mode, bytes in byte mode. Both numbers
        come from the tables rather than from segno's exception text.
        """
        version = _MAX_VERSION if request.version is None else request.version
        mode = mode_of(payload)
        capacity = max_payload_bytes(version, ec_level, mode)
        unit = "bytes" if mode == "byte" else "characters"
        return (
            f"data too large for version {version} at error level {ec_level}: "
            f"capacity {capacity} {unit}, got {len(payload)}"
        )

    def _render(self, code: segno.QRCode, request: GenerateRequest) -> bytes:
        """Render the symbol to the artefact kind the request asked for.

        Both kinds go into an :class:`io.BytesIO`: segno's SVG writer wraps the stream in a UTF-8
        codec writer, so it needs a *byte* stream and raises ``TypeError`` on a text stream. Colours
        are passed as ``#rrggbb``, the one form both writers accept, and a ``png`` artefact with a
        ``logo_overlay_path`` is composited by :meth:`_composite_overlay`.

        Raises:
        TargetContractError: The writer failed, produced an empty body, or the overlay is
        unreadable.
        """
        kind = self._effective_output_format(request)
        scale = self._effective_module_size(request)
        border = self._effective_quiet_zone(request)
        # The colours are converted outside the guard: the validation chain proved them, so a
        # failure
        # here is a framework defect that must surface as an errored case. The guard covers segno's
        # writer, which is the library.
        # so a
        # failure here is a framework defect that must surface as an errored case rather than a
        # target
        # verdict. The guard covers segno's writer, which is the library.
        dark = _colour_hex(request.foreground_rgb)
        light = _colour_hex(request.background_rgb)
        stream = io.BytesIO()
        try:
            code.save(
                stream,
                kind=kind,
                scale=scale,
                border=border,
                dark=dark,
                light=light,
            )
            data = stream.getvalue()
        except Exception as exc:
            raise TargetContractError(
                f"rendering a {kind!r} symbol failed: {type(exc).__name__}: {exc}"
            ) from None
        if not data:
            raise TargetContractError(f"rendering produced an empty {kind!r} body")
        overlay_path = request.logo_overlay_path
        if overlay_path is not None and kind == "png":
            data = self._composite_overlay(data, overlay_path)
        return data

    def _composite_overlay(self, data: bytes, overlay_path: Path) -> bytes:
        """Composite the request's logo overlay onto the rendered symbol.

        The overlay keeps its pixel size, lands centred and has its alpha channel honoured. The
        transform and the open both belong to :mod:`qrcheck.evidence.mux`, reached at the point of
        use so importing this package never loads Pillow.

        Raises:
        TargetContractError: The path cannot be read, does not hold an image Pillow can open, or the
        rendered bytes no longer decode.
        """
        from qrcheck.evidence.mux import composite_overlay, decode_image, encode_png, open_image

        try:
            with open_image(overlay_path) as opened:
                overlay = opened.convert("RGBA")
            symbol = decode_image(data)
            return encode_png(composite_overlay(symbol, overlay))
        except OSError as exc:
            raise TargetContractError(
                f"compositing overlay {str(overlay_path)!r} onto the rendered symbol "
                f"failed: {type(exc).__name__}: {exc}"
            ) from None

