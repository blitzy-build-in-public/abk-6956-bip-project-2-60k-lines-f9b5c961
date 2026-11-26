"""The conforming self-check target: a segno-backed generator that must come back clean.

This target exists to prove the suite is not vacuous.  ``qrcheck run --adapter
reference`` must report zero failed, zero errored and zero skipped results, with at
least one ``passed`` result for every one of the twenty-one checks; the contrast pair
is :mod:`qrcheck.targets.defective`, which must fail exactly the nine checks its five
injections target.  Nothing here is the product under test -- it is the yardstick the
product is measured against, and a change that makes a check pass against this target
by weakening the check is forbidden.

Two segno disciplines are mandatory at the call site
---------------------------------------------------
``boost_error=False``
    segno's default spends spare capacity on a *higher* error-correction level without
    changing the version, so an oracle built with defaults would expect a different EC
    level than the target was asked for and would fail a conforming target.  Every
    call here passes ``boost_error=False`` explicitly.
Model 2 only
    Micro QR is out of scope: an M-series symbol has no ``4V+17`` module count, no
    Version 1-40 number and a different format-information layout, so ``N = 4V+17``,
    the version-from-dimension mapping, the mask/EC extraction and every capacity
    lookup would be wrong on exactly the short payloads the corpus is full of.
    ``segno.make()`` returns Micro QR for short content, so this module calls
    ``segno.make_qr()`` and never bare ``segno.make()``.

    Recorded discrepancy: the plan writes the call as ``segno.make_qr(..., micro=False)``,
    but in segno 1.6.6 ``make_qr`` has no ``micro`` parameter.  It is Model 2 by
    construction -- its own documentation says "Creates a QR code (never a Micro QR
    code)" and it forwards ``micro=False`` to ``make`` internally -- while bare
    ``segno.make`` accepts ``micro`` and defaults it to ``None``, meaning "prefer Micro
    QR".  The outcome the plan requires is therefore obtained by calling
    ``segno.make_qr``; passing ``micro=False`` to it would be a ``TypeError``.

Why all thirteen capabilities are declared
------------------------------------------
``capabilities`` is a gate, not documentation: the runner resolves a check whose
requirements an adapter does not declare to ``skipped``.  This target declares every
one of the thirteen :class:`~qrcheck.targets.base.Capability` members
unconditionally -- it does not intersect them with the configured set -- because its
whole job is the zero-skip clean-run gate, and narrowing the declared set would let a
check skip instead of run.

Division of labour with the checks
----------------------------------
``logo_overlay_path`` is accepted and honoured as a supported field (that is what the
``logo_overlay`` capability declares), but the overlay is **not** composited here.
The product's only image-library boundary is :mod:`qrcheck.evidence.mux`, so
``RN-LOGO`` in ``qrcheck/checks/render.py`` generates, composites and measures the
overlay bitmap and writes it as ``logo.png`` evidence.  An adapter that opened images
with Pillow would create a second image pipeline and would break the single-owner rule
for the evidence artefact.

``timeout_s`` is read from configuration but is not enforceable in this adapter: it is
an in-process function call, and the product deliberately carries no thread, signal or
scheduler machinery, so there is nothing legitimate to interrupt.  The field matters
for the out-of-process adapters, where it bounds a subprocess or an HTTP request.

Documented limitation, stated here so nobody mistakes it for independence
-----------------------------------------------------------------------
This target is segno-backed, and segno is also the oracle, so ``SY-MATRIX-EQUIV``
against this target compares the oracle to itself in the encoder dimension.  What that
check still proves is the render -> sample -> decode path end to end, and the tautology
is broken by :mod:`qrcheck.targets.defective`, whose module matrix at pinned
parameters differs from the oracle's.  The genuinely independent evidence is the
decoder-derived set (``RT-ROUNDTRIP``, ``MD-CHARSET``, the decoder-side EC level of
``SY-EC-MATCH``, the format-information mask of ``SY-MASK``) plus decoder-versus-decoder
agreement.  No second encoder is added: the dependency set is pinned and names none.

Layering
--------
Only ``io``, ``time`` and :mod:`typing` from the standard library, ``segno``, and the
three modules this file may depend on: :mod:`qrcheck.targets.base`,
:mod:`qrcheck.corpus.capacity` and :mod:`qrcheck.config.models`.  It imports no
decoder, no image library, no transport client and none of ``qrcheck.checks``,
``qrcheck.report``, ``qrcheck.decode`` or ``qrcheck.registry``, which is the property
``tests/unit/test_layering.py`` holds by scanning source.
"""

from __future__ import annotations

import io
import time
from typing import Final, Literal

import segno

from qrcheck.config.models import TargetConfig
from qrcheck.corpus.capacity import EC_LEVELS, max_payload_bytes
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    ensure_image_bytes,
    require_capabilities,
)

__all__ = [
    "DEFAULT_MODULE_SIZE_PX",
    "DEFAULT_QUIET_ZONE_MODULES",
    "ReferenceAdapter",
]

# This adapter's defaults for the two geometry fields a request may leave unset.  They
# only fill in a value the request did not ask for, so no check asserts a number it did
# not request; four modules of quiet zone is the ISO/IEC 18004 minimum, and four pixels
# per module keeps a Version 40 symbol inside a manageable bitmap.
DEFAULT_MODULE_SIZE_PX: Final[int] = 4
DEFAULT_QUIET_ZONE_MODULES: Final[int] = 4

# ISO/IEC 18004 Model 2 bounds.  A pinned value outside these is outside the standard,
# not merely outside this product.
_MIN_VERSION: Final[int] = 1
_MAX_VERSION: Final[int] = 40
_MIN_MASK: Final[int] = 0
_MAX_MASK: Final[int] = 7

# A colour channel is one unsigned byte.
_MIN_CHANNEL: Final[int] = 0
_MAX_CHANNEL: Final[int] = 255

# The two artefact kinds segno renders for us, and the segno `kind` name of each.
_RENDER_KINDS: Final[tuple[str, ...]] = ("png", "svg")

# WCAG 2.1 sRGB relative-luminance constants.  A channel is linearised piecewise -- a
# straight division below the knee, a gamma expansion above it -- and the luminance is
# the weighted sum, which is what makes the ratio below order-independent and therefore
# equally lawful for an inverted (light-on-dark) request.
_SRGB_KNEE: Final[float] = 0.03928
_SRGB_SLOPE: Final[float] = 12.92
_SRGB_OFFSET: Final[float] = 0.055
_SRGB_SCALE: Final[float] = 1.055
_SRGB_GAMMA: Final[float] = 2.4
_LUMINANCE_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

# The additive term of the WCAG contrast formula, so the ratio of two identical colours
# is exactly 1.0 rather than a division by zero.
_CONTRAST_OFFSET: Final[float] = 0.05


def _linear_channel(channel: int) -> float:
    """Return one 0..255 sRGB channel as a linearised luminance component.

    ``channel`` has already been validated as an integer in 0..255 by the caller, so
    this function raises nothing.
    """
    value = channel / 255.0
    if value <= _SRGB_KNEE:
        return value / _SRGB_SLOPE
    return ((value + _SRGB_OFFSET) / _SRGB_SCALE) ** _SRGB_GAMMA


def _relative_luminance(colour: tuple[int, int, int]) -> float:
    """Return the WCAG 2.1 relative luminance of a three-channel ``0..255`` colour.

    The result is in 0.0..1.0, with black at 0.0 and white at 1.0.  Raises nothing for
    a validated colour.
    """
    red, green, blue = (_linear_channel(channel) for channel in colour)
    weight_red, weight_green, weight_blue = _LUMINANCE_WEIGHTS
    return weight_red * red + weight_green * green + weight_blue * blue


def _contrast_ratio(
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
) -> float:
    """Return the WCAG 2.1 contrast ratio between two ``0..255`` colours, from 1.0 up.

    The lighter colour is placed over the darker one, so the ratio is
    order-independent: an inverted request is measured exactly as a normal one is, and
    a colour pair is never rejected merely for being light-on-dark.  The formula is
    ``(lighter + 0.05) / (darker + 0.05)``, which spans 1.0 (identical colours) to 21.0
    (black on white).
    """
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = (first, second) if first >= second else (second, first)
    return (lighter + _CONTRAST_OFFSET) / (darker + _CONTRAST_OFFSET)


def _colour_hex(colour: tuple[int, int, int]) -> str:
    """Return a validated ``0..255`` colour as the ``#rrggbb`` string segno renders.

    Both the PNG and the SVG writer accept this form, which is what keeps one colour
    representation for the two artefact kinds.  Raises nothing for a validated colour.
    """
    red, green, blue = colour
    return f"#{red:02x}{green:02x}{blue:02x}"


class ReferenceAdapter:
    """A conforming generator: segno with the two expectation-corrupting defaults fixed.

    The adapter is stateless -- it holds the configured target name, its declared
    capabilities and the contrast floor it enforces, and nothing that a call mutates --
    so two identical requests produce identical bytes and identical declared metadata.
    That determinism is what ``DT-DETERMINISTIC`` compares across two invocations and
    what the zero-failure clean run depends on.

    Every request is validated before segno sees it.  segno is a general-purpose
    encoder, not a conformance target: it encodes an empty payload without complaint,
    raises a bare ``ValueError`` for an illegal error level and reports nothing but
    ``DataOverflowError`` for capacity.  The ``IV-*`` checks assert a *declared*
    contract, so the messages this class raises are public API and are quoted verbatim
    in ``docs/configuration.md`` and ``docs/targets.md``.

    Which of the five adapter errors this class can raise: :class:`TargetRejected` for
    a request the target's own contract refuses, :class:`TargetContractError` when
    segno or the stream writer fails for a reason that is not the request's fault, and
    :class:`TargetUnsupported` from the capability gate.  :class:`TargetUnavailable` and
    :class:`TargetTimeout` are deliberately never raised: this adapter is an in-process
    call, so there is no process to spawn and nothing to time out.
    """

    def __init__(self, config: TargetConfig, *, name: str | None = None) -> None:
        """Bind the adapter to its resolved ``[targets.<name>]`` record.

        Args:
            config: The resolved target record.  Only ``contrast_floor`` is read: the
                declared capability set is deliberately ignored (this target always
                declares all thirteen) and ``timeout_s`` is not enforceable in-process.
            name: The configured target name, used by every diagnostic.  Defaults to
                ``config.type`` so a caller that holds only the record can still
                construct the adapter.

        Raises:
            TypeError: ``config.contrast_floor`` is not a real number.  The
                configuration schema rejects that at load time, so it means the record
                was hand-built.
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

        The order is fixed: the capability gate first (it performs no I/O), then the
        validation chain, then the encode and the render inside one timing window.  The
        four ``declared_*`` fields carry what segno reports about the symbol it built;
        they are the target's claim and are evidence only -- the checks read the version
        from the sampled module grid, the EC level from the decoder and the format
        information, and the mask from the format information.

        Args:
            request: The request to satisfy, built by ``build_request`` (or, for the
                ``IV-*`` probes, its documented out-of-contract bypass).

        Returns:
            The artefact exactly as segno rendered it, with its declared metadata and the
            encode duration in milliseconds.

        Raises:
            TargetUnsupported: The request needs a capability this adapter did not
                declare.  Unreachable for this target, which declares all thirteen, and
                kept as the belt-and-braces path the port requires.
            TargetRejected: The request breaks the target's own contract -- an error
                level outside L/M/Q/H, a version or mask outside its range, a
                non-positive module size, a negative quiet zone, a malformed colour, an
                empty payload, a payload that does not fit the pinned version, or a
                contrast ratio below the declared floor.  The message is the contract.
            TargetContractError: segno or the stream writer failed for a reason the
                request does not explain, or the rendered bytes are not the artefact the
                request asked for.  Both are framework-side defects rather than target
                verdicts.
        """
        require_capabilities(self, self._required_capabilities(request))
        self._validate(request)

        output_format = self._effective_output_format(request)
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

        The runner closes every adapter once a run ends, including after a failure, so
        this must never raise.  There is no process, socket or file to release: the
        adapter is stateless and each call renders into a fresh in-memory stream.
        """

    def _required_capabilities(self, request: GenerateRequest) -> frozenset[Capability]:
        """Return the capabilities `request` needs, for the gate before any work.

        Only the request is inspected: the artefact format, the error level and the
        colours are always applied, and each pinned or optional field adds its own
        capability.  Metadata capabilities are not derived from a request, because a
        request never asks for metadata -- this target reports it unconditionally.

        The gate can never refuse for this adapter, which declares all thirteen
        members.  It is still called, because the port requires every implementation to
        refuse an undeclared operation before any I/O, and because the day a target
        stops declaring something is not the day to discover that the gate was missing.
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

    # ------------------------------------------------------------------
    # Validation: the target's declared contract, checked before segno
    # ------------------------------------------------------------------

    def _validate(self, request: GenerateRequest) -> None:
        """Run the three validation steps in their fixed order.

        The order is deliberately stable -- shape, then payload, then contrast -- so the
        message a probe receives never depends on the order rules happen to be written
        in.  Each ``IV-*`` probe breaks exactly one rule, so the order is unobservable
        across the gate while still being a contract.

        Raises:
            TargetRejected: The request breaks one of the target's own rules; the
                message names the offending field and is public API.
        """
        self._validate_shape(request)
        self._check_payload(request)
        self._check_contrast(request)

    def _validate_shape(self, request: GenerateRequest) -> None:
        """Refuse a request whose parameters are outside the target's contract.

        The six checks run in the order the contract lists them: error level, version
        pin, mask pin, module size, quiet zone, colours.

        Raises:
            TargetRejected: One parameter is outside its documented range.  The message
                names the offending field and, for a range, the accepted bounds.
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

        ``bool`` is refused even though it is an ``int`` subclass, because ``True`` as a
        channel would silently render as 1 rather than fail.

        Raises:
            TargetRejected: `value` is not three integers in 0..255; the message names
                `field`.
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

        Both bounds are measured against the payload and the error level this adapter is
        about to encode, which for this conforming target are the request's own values.
        An unpinned request needs no capacity rule here: Version 40 is the largest Model
        2 symbol, and ``segno.DataOverflowError`` beyond it is converted to the same
        rejection by :meth:`_encode`.  The capacity figure comes from
        :func:`qrcheck.corpus.capacity.max_payload_bytes` and is never transcribed.

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
        capacity = max_payload_bytes(version, ec_level, "byte")
        if len(payload) > capacity:
            raise TargetRejected(
                f"data too large for version {version} at error level {ec_level}: "
                f"capacity {capacity} bytes, got {len(payload)}"
            )

    def _check_contrast(self, request: GenerateRequest) -> None:
        """Refuse a colour pair whose contrast is below the target's declared floor.

        The ratio is the WCAG 2.1 relative-luminance ratio, order-independent, so an
        inverted (light-on-dark) request is measured exactly like a normal one and is
        never refused merely for its polarity.  The floor is the configured
        ``contrast_floor`` for this target, which is also the value ``RN-CONTRAST``
        asserts at or above, so a target that declares nothing is still held to the
        documented 7:1.

        Raises:
            TargetRejected: The measured ratio is below the floor.  The message reports
                both values to one decimal place.
        """
        ratio = _contrast_ratio(request.foreground_rgb, request.background_rgb)
        if ratio < self._contrast_floor:
            raise TargetRejected(
                f"contrast ratio {ratio:.1f} is below the declared floor "
                f"{self._contrast_floor:.1f}"
            )

    # ------------------------------------------------------------------
    # Effective values: the seams a derived target changes one at a time
    # ------------------------------------------------------------------

    def _effective_ec(self, request: GenerateRequest) -> str:
        """Return the error-correction level to encode at: the requested one.

        This is the seam a target that quietly encodes at a lower level uses; for this
        target the encoded level and the requested level are the same value.
        """
        return request.ec_level

    def _effective_payload(self, request: GenerateRequest) -> bytes:
        """Return the bytes to encode: the requested payload, unmodified.

        Every payload decision in this class is expressed through this method, so a
        target that alters the payload alters it in one place and the capacity check in
        :meth:`_check_payload` measures the bytes actually encoded rather than a
        different quantity.
        """
        return request.payload

    def _effective_quiet_zone(self, request: GenerateRequest) -> int:
        """Return the quiet zone to render, in modules: the requested one or the default.

        The rendered image is sized from exactly this value, so the dimension contract
        ``(modules + 2 * quiet_zone) * module_size`` holds for whatever a subclass
        returns here and never drifts from the zone actually drawn.
        """
        quiet_zone = request.quiet_zone_modules
        return DEFAULT_QUIET_ZONE_MODULES if quiet_zone is None else quiet_zone

    def _effective_module_size(self, request: GenerateRequest) -> int:
        """Return the module edge to render, in pixels: the requested one or the default."""
        module_size = request.module_size_px
        return DEFAULT_MODULE_SIZE_PX if module_size is None else module_size

    def _effective_mask(self, request: GenerateRequest) -> int | None:
        """Return the mask to encode with: the pinned one, or ``None`` to let segno choose.

        A pin is honoured exactly, which is what keeps ``SY-MATRIX-EQUIV`` comparable at
        pinned parameters and what makes an unpinned request's mask an encoder choice
        rather than a framework expectation.
        """
        return request.mask

    def _declared_ec(self, request: GenerateRequest) -> str:
        """Return the error-correction level this target *claims* it encoded at.

        For this target the claim is true, so it is the requested level; it is also the
        level segno actually used.  It is a separate seam from :meth:`_effective_ec`
        because a target may encode at one level and declare another, and the claim is
        evidence only -- ``SY-EC-MATCH`` reads the decoder and the format information,
        never this value.
        """
        return request.ec_level

    def _effective_output_format(self, request: GenerateRequest) -> Literal["png", "svg"]:
        """Return the normalised artefact kind to render.

        Raises:
            TargetRejected: The request asks for neither ``png`` nor ``svg``.  This
                cannot happen through ``build_request``, which refuses any other value;
                it is the belt-and-braces path for an out-of-contract request.
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

    # ------------------------------------------------------------------
    # Encode and render
    # ------------------------------------------------------------------

    def _encode(self, request: GenerateRequest) -> segno.QRCode:
        """Encode the request with segno, always Model 2 and never boosting the level.

        ``boost_error=False`` and ``make_qr`` are the two disciplines this method exists
        to hold: see the module docstring for why each one changes the outcome.

        Args:
            request: The validated request to encode.

        Returns:
            The segno symbol, whose ``version``, ``error``, ``mask`` and ``mode`` are the
            target's declared metadata.

        Raises:
            TargetRejected: segno reports that the content does not fit, or that a
                parameter is illegal.  Every ``ValueError`` from segno in the pinned
                release is a complaint about the content or the parameters -- an illegal
                error level, an unsupported version, an invalid mask, an illegal mode --
                so it is a request the target should itself have refused first.
            TargetContractError: segno failed for a reason the request does not explain.
                A framework-side defect, reported rather than allowed to escape.
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
            raise TargetRejected(self._overflow_message(request, len(payload), ec_level)) from None
        # segno 1.6.6 defines exactly one dedicated exception, ``DataOverflowError``
        # above, and reports an illegal error level, an unsupported version, an invalid
        # mask and an illegal mode as plain ``ValueError``.  The ``VersionError``,
        # ``ErrorLevelError``, ``ModeError`` and ``MaskError`` the specification names do
        # not exist in the pinned release and would each be a ``ValueError`` subclass, so
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
        length: int,
        ec_level: str,
    ) -> str:
        """Return the capacity rejection message for a payload of `length` bytes.

        The version named is the pinned one, or Version 40 for an unpinned request --
        the largest Model 2 symbol, which is the bound an unpinned request actually
        broke.  The capacity is computed from the tables rather than read out of segno's
        own exception text, so the message is the same shape whichever version overflowed.
        """
        version = _MAX_VERSION if request.version is None else request.version
        capacity = max_payload_bytes(version, ec_level, "byte")
        return (
            f"data too large for version {version} at error level {ec_level}: "
            f"capacity {capacity} bytes, got {length}"
        )

    def _render(self, code: segno.QRCode, request: GenerateRequest) -> bytes:
        """Render the symbol to the artefact kind the request asked for.

        Both kinds go into an :class:`io.BytesIO`.  segno's SVG writer wraps the stream
        in a UTF-8 codec writer, so it needs a *byte* stream and raises ``TypeError`` on
        a text stream -- verified against segno 1.6.6 rather than assumed.  Colours are
        passed as ``#rrggbb`` strings, the one form both writers accept.

        ``logo_overlay_path`` is not applied here: the overlay is generated, composited
        and measured by ``RN-LOGO`` through :mod:`qrcheck.evidence.mux`, the product's
        single image-library boundary.  See the module docstring.

        Args:
            code: The encoded symbol.
            request: The validated request, read for format, geometry and colours.

        Returns:
            The artefact bytes, non-empty.

        Raises:
            TargetContractError: The writer failed, or produced an empty body.  Both are
                framework-side defects rather than target verdicts.
        """
        kind = self._effective_output_format(request)
        scale = self._effective_module_size(request)
        border = self._effective_quiet_zone(request)
        # The colours are converted outside the guard: they come from fields the
        # validation chain has already proved, so a failure here is a framework defect
        # that must surface as an errored case with its traceback rather than be masked
        # as a target verdict.  The guard covers segno's writer, which is the library.
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
        return data

