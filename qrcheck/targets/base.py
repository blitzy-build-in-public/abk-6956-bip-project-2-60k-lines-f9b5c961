"""The adapter port every QR generator target is driven through.

This module is the seam between *what is asserted* (``qrcheck/checks/``) and
*what is being asserted against* (``qrcheck/targets/``), and it is deliberately
the thinnest place in the product: the capability vocabulary, the two records a
target exchanges with the framework, the closed five-error vocabulary every
adapter raises, the single request builder every check calls, the capability
gate, and the resolution helpers the concrete adapters share.  A fourth adapter
kind needs no change to any check and a new check needs no change to any
adapter, because neither side of the seam ever sees the other.

What the port guarantees
------------------------
* :class:`Capability` is the complete and closed list of the thirteen operations
  a target may declare, and :func:`require_capabilities` refuses an undeclared
  operation **before any I/O**, so an unsupported request never reaches a target
  and never consumes its ``timeout_s``.
* :func:`build_request` is the only legal path to a :class:`GenerateRequest`.
  Every check builds its request through it, which is what keeps one definition
  of "this request is valid" in the product.
* :func:`decode_returned_image`, :func:`resolve_dotted_path`,
  :func:`apply_metadata_map`, :func:`ensure_image_bytes` and
  :func:`hex_prefix` each exist exactly once here; ``library.py``, ``cli.py``
  and ``http.py`` import them rather than re-deriving them.

The five-error vocabulary
-------------------------
An adapter raises nothing else.  Exactly one trigger produces each error, and
every ``except`` inside ``qrcheck/targets/`` converts a library exception into
one of these five rather than letting it escape unwrapped:

=========================  =====================================================
``TargetUnavailable``      the process cannot be spawned, the connection is
                           refused, the import path is missing
``TargetTimeout``          no response within ``targets.<name>.timeout_s``
``TargetRejected``         the target returned a declared client-side rejection
                           for a request the framework considers valid; its text
                           is reported verbatim
``TargetContractError``    a response arrived but is unusable: empty body,
                           undecodable base64, image bytes no decoder or image
                           library can open, a configured field the response
                           does not carry
``TargetUnsupported``      the request asks for a capability the adapter did not
                           declare; raised before any I/O
=========================  =====================================================

:class:`TargetRejected` carries the target's text unchanged -- ``str(exc)`` is
exactly what the target said, because the ``IV-*`` checks record it in
``decoded.rejection`` and every other check reports it as reason
``target_rejected``.

Why ``GenerateRequest`` validates nothing
-----------------------------------------
:class:`GenerateRequest` has no ``__post_init__``, no coercion and no guard.
That is a contract, not an oversight: ``IV-EMPTY``, ``IV-OVERCAPACITY`` and the
four ``IV-BAD-OPTION`` probes exist to prove that a target rejects requests
breaking exactly the rules :func:`build_request` enforces, so they must be able
to hand a target a deliberately illegal request (``ec_level="X"``,
``module_size_px=0``, ``quiet_zone_modules=-1``, a below-floor colour pair).
They do so through ``qrcheck/checks/support.py:construct_out_of_contract``,
which is a plain constructor call precisely because this type has no gate.
Validation added here would silently make every ``IV-*`` case unconstructable.

Layering
--------
This module imports only the standard library, :mod:`qrcheck.config.models` and
:mod:`qrcheck.corpus.capacity`.  It imports no image library, no decoder and
none of ``qrcheck.checks``, ``qrcheck.report``, ``qrcheck.decode`` or
``qrcheck.registry``, which is what lets ``decode/``, ``registry/``,
``checks/support.py`` and ``runner.py`` all import it without a cycle.  In the
other direction, ``qrcheck/checks/*`` may import this module and nothing else
from ``qrcheck/targets/``, and nothing in ``qrcheck/targets/`` may import
``qrcheck.checks`` -- the property ``tests/unit/test_layering.py`` holds by
scanning source.
"""

from __future__ import annotations

import base64
import binascii
import enum
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, runtime_checkable

from qrcheck.config.models import RETURNS_KINDS, ReturnsKind, TargetConfig
from qrcheck.corpus.capacity import EC_LEVELS, max_payload_bytes

__all__ = [
    "Adapter",
    "Capability",
    "GenerateRequest",
    "GeneratedSymbol",
    "HEX_PREFIX_LIMIT",
    "METADATA_FIELDS",
    "OUTPUT_FORMATS",
    "TargetContractError",
    "TargetError",
    "TargetRejected",
    "TargetTimeout",
    "TargetUnavailable",
    "TargetUnsupported",
    "apply_metadata_map",
    "build_request",
    "declared_capabilities",
    "decode_returned_image",
    "ensure_image_bytes",
    "hex_prefix",
    "require_capabilities",
    "resolve_dotted_path",
]

# The artefact formats a request may ask for.  Both are also capability names,
# because a target declares which of them it can produce.
OUTPUT_FORMATS: Final[tuple[str, ...]] = ("png", "svg")

# The four metadata fields a target may report about a symbol, in the order the
# mapping is projected.  They are the keys of ``metadata_map`` in configuration
# and the keys :func:`apply_metadata_map` returns; every one of them is evidence
# only, never an assertion.
METADATA_FIELDS: Final[tuple[str, ...]] = ("version", "ec", "mask", "mode")

# How many leading bytes :func:`hex_prefix` renders by default: the first 256
# bytes of a body that is not an image, recorded as 512 hex characters so the
# defect is diagnosable from the report alone.
HEX_PREFIX_LIMIT: Final[int] = 256

# ISO/IEC 18004 Model 2 version and mask ranges.  A pinned value outside these
# is outside the standard, not merely outside this product.
_MIN_VERSION: Final[int] = 1
_MAX_VERSION: Final[int] = 40
_MIN_MASK: Final[int] = 0
_MAX_MASK: Final[int] = 7

# A colour channel is one unsigned byte.
_MIN_CHANNEL: Final[int] = 0
_MAX_CHANNEL: Final[int] = 255

# A requested module edge is at least one pixel; a zero-pixel module renders no
# symbol, which is what IV-BAD-OPTION probes for.
_MIN_MODULE_SIZE: Final[int] = 1

# Leading byte sequences that identify the artefact formats the framework
# accepts.  A signature is evidence, not full decodability: whether the bytes
# are a *valid* image is decided by the decode layer and the evidence mux, never
# here, because Pillow is reachable only through ``qrcheck/evidence/mux.py``.
_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE: Final[bytes] = b"\xff\xd8\xff"
_BYTE_ORDER_MARK: Final[bytes] = b"\xef\xbb\xbf"

# How far into a body an SVG or XML marker is looked for.  SVG authored by a
# renderer may open with a BOM, whitespace, an XML declaration or a doctype, so
# the marker is searched rather than required at offset zero.
_SVG_SNIFF_BYTES: Final[int] = 512

# Which sniffed formats satisfy which requested output format.  JPEG is not a
# request format the framework emits, but a target may hand back JPEG bytes for
# a ``png`` request, and that mismatch must be reported as one rather than
# mistaken for an unknown body.
_FORMAT_SNIFFS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "png": frozenset({"png"}),
        "svg": frozenset({"svg"}),
        "jpeg": frozenset({"jpeg"}),
        "jpg": frozenset({"jpeg"}),
    }
)


class Capability(enum.StrEnum):
    """One operation a target may declare that it supports.

    A ``StrEnum`` member *is* its string value, so ``Capability.png == "png"``
    holds and a set of members compares and hashes equal to the
    ``frozenset[str]`` a resolved configuration carries.  That is what lets
    capability intersection and containment work across the configuration seam
    without either side importing the other: ``qrcheck.config.models`` keeps
    member *names* (``CAPABILITY_NAMES``) because it must not import this
    package, and :func:`declared_capabilities` converts to members here.

    The member set is closed at thirteen and is declared in a fixed order.  No
    check, adapter or resolution rule may add, rename, reorder or omit a member:
    ``SY-MATRIX-EQUIV``, the ``RN-*`` checks, ``IV-OVERCAPACITY`` and the
    capability-resolution table all key on these exact names.
    """

    png = "png"
    svg = "svg"
    ec_level = "ec_level"
    module_size = "module_size"
    quiet_zone = "quiet_zone"
    color = "color"
    logo_overlay = "logo_overlay"
    version_pin = "version_pin"
    mask_pin = "mask_pin"
    metadata_version = "metadata_version"
    metadata_ec = "metadata_ec"
    metadata_mask = "metadata_mask"
    metadata_mode = "metadata_mode"


def declared_capabilities(config: TargetConfig) -> frozenset[Capability]:
    """Return the capabilities `config` declares, as members of :class:`Capability`.

    This is the one place the configuration's ``frozenset[str]`` becomes a
    ``frozenset[Capability]``, so no adapter repeats the conversion and no
    adapter can drift from the vocabulary.

    Args:
        config: The resolved ``[targets.<name>]`` record whose ``capabilities``
            holds member names.

    Returns:
        The declared capabilities as members, empty when the target declares
        none.

    Raises:
        ValueError: A declared name is not a :class:`Capability` member.  This is
            a framework defect rather than a user error: the configuration
            schema already rejects an unknown capability member at load time with
            exit 2 naming it, so a name reaching this function has been past that
            gate and can only mean the two lists have diverged.
    """
    try:
        return frozenset(Capability(name) for name in sorted(config.capabilities))
    except ValueError as exc:
        raise ValueError(
            f"target {config.name!r} declares an unknown capability: {exc}"
        ) from None


class TargetError(Exception):
    """Base class of the five errors an adapter may raise.

    Catch this to handle every failure the target layer can report; catch a
    subclass to handle one trigger.  Nothing outside this hierarchy escapes an
    adapter, so a caller catching :class:`TargetError` holds the complete failure
    vocabulary of a target.
    """


class TargetUnavailable(TargetError):
    """The target could not be reached at all.

    The single trigger is that the process cannot be spawned, the connection is
    refused, or the configured import path is missing.  The diagnostic names the
    transport failure and the configured field, because the fix is on the target
    side and the report is all a reader has.
    """


class TargetTimeout(TargetError):
    """The target did not answer within ``targets.<name>.timeout_s``.

    The single trigger is that no response arrived inside the configured timeout.
    The diagnostic names the timeout in seconds and the operation attempted, so a
    reader can tell a slow target from a hung one.
    """


class TargetRejected(TargetError):
    """The target returned a declared client-side rejection of a valid request.

    The single trigger is a target's own refusal of a request the framework
    considers valid.  The message is the target's text, unchanged: ``str(exc)``
    is exactly what the target said, never re-wrapped, prefixed, truncated or
    reformatted, because the ``IV-*`` checks record it verbatim in
    ``decoded.rejection`` and every other check reports it as reason
    ``target_rejected``.
    """


class TargetContractError(TargetError):
    """A response arrived but could not be used.

    The single trigger is a response present but unusable: an empty body,
    undecodable base64, image bytes that no decoder or image library can open, or
    a field named in configuration that the response does not carry.  The
    diagnostic names the configured field path, the ``returns`` kind or the
    leading bytes' hex, so the defect is diagnosable from the report alone.
    """


class TargetUnsupported(TargetError):
    """The request needs a capability the adapter did not declare.

    Raised by :func:`require_capabilities`, and belt-and-braces by an adapter
    about to be asked for something it cannot do, **before any I/O**: an
    unsupported request must never reach a target and never consume its
    ``timeout_s``.  The diagnostic names the missing capability and the adapter.
    """


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """One request the framework makes of a target.

    The ten fields are the request contract in full, and the four values a
    configuration cannot express directly (``foreground_rgb``,
    ``background_rgb``, ``version``, ``mask``) are the ones a target must refuse
    when it does not declare the matching capability.

    ``payload`` is bytes and never text, so byte-exactness is unambiguous: a
    payload that is not valid UTF-8 still round-trips, and nothing here guesses an
    encoding on the caller's behalf.

    The eight fields after ``payload`` carry defaults **for convenience only**, so
    an ordinary request is ``GenerateRequest(payload)``; a caller that cares about
    geometry, colour or a pin passes it explicitly, and a check that needs a
    specific value always does.

    This record validates nothing: no ``__post_init__``, no coercion, no guard.
    The gate is :func:`build_request`, and the documented bypass is
    ``qrcheck/checks/support.py:construct_out_of_contract``, which exists so the
    ``IV-*`` checks can hand a target a deliberately illegal request and prove it
    is refused.  Do not add validation here.
    """

    payload: bytes
    ec_level: Literal["L", "M", "Q", "H"] = "M"
    output_format: Literal["png", "svg"] = "png"
    module_size_px: int | None = None
    quiet_zone_modules: int | None = None
    foreground_rgb: tuple[int, int, int] = (0, 0, 0)
    background_rgb: tuple[int, int, int] = (255, 255, 255)
    logo_overlay_path: Path | None = None
    version: int | None = None
    mask: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Project the request to the JSON-safe mapping the evidence store writes.

        ``payload`` travels as ``payload_b64`` together with ``payload_len``, so a
        reader can compare what a symbol decoded to against what was asked for,
        plus ``payload_text`` when the bytes round-trip through UTF-8.  Colours
        become ``[r, g, b]`` lists, and ``logo_overlay_path`` is the configured
        path text rather than a resolved absolute path: no artefact records an
        absolute location.
        """
        projected: dict[str, Any] = {
            "payload_b64": base64.b64encode(self.payload).decode("ascii"),
            "payload_len": len(self.payload),
            "ec_level": self.ec_level,
            "output_format": self.output_format,
            "module_size_px": self.module_size_px,
            "quiet_zone_modules": self.quiet_zone_modules,
            "foreground_rgb": list(self.foreground_rgb),
            "background_rgb": list(self.background_rgb),
            "logo_overlay_path": (
                None if self.logo_overlay_path is None else str(self.logo_overlay_path)
            ),
            "version": self.version,
            "mask": self.mask,
        }
        text = _utf8_text(self.payload)
        if text is not None:
            projected["payload_text"] = text
        return projected


@dataclass(frozen=True, slots=True)
class GeneratedSymbol:
    """The artefact a target produced, with the metadata it claims about it.

    ``image_bytes`` is the artefact exactly as the target produced it, and
    ``output_format`` is the format actually produced: a target that answers a
    ``png`` request with SVG bytes has broken its contract, and
    :func:`ensure_image_bytes` is where that is caught.

    The four ``declared_*`` fields are the target's own claim and are **evidence
    only, never the assertion**.  ``SY-EC-MATCH`` reads the EC level from the
    primary decoder and from the symbol's format information, ``SY-MASK`` reads
    the mask from the format information, and ``SY-VERSION-MIN`` takes the version
    from the sampled module count, so a target that lies about its metadata is
    recorded as having lied rather than believed.  ``elapsed_ms`` is the
    target-side timing: recorded, reported, and excluded from every determinism
    comparison.

    The trailing fields are defaulted because a target that reports no metadata
    should have to say so only once.
    """

    image_bytes: bytes
    output_format: Literal["png", "svg"]
    declared_version: int | None = None
    declared_ec: str | None = None
    declared_mask: int | None = None
    declared_mode: str | None = None
    elapsed_ms: float = 0.0


@runtime_checkable
class Adapter(Protocol):
    """The port a QR generator target is driven through.

    Exactly these four members, and no fifth: checks, the runner and the registry
    resolve all depend on this shape, and a stub in a test may implement it in four
    lines because nothing else is required of it.

    ``capabilities`` is a ``frozenset`` because it is compared and differenced
    against a check's requirements, and ``name`` is the configured target name
    that every diagnostic quotes.

    Protocol conformance is asserted with ``isinstance(adapter, Adapter)`` and
    never with ``issubclass``: on CPython 3.11 a runtime-checkable protocol with
    non-method members (``name`` and ``capabilities`` here) supports ``isinstance``
    by attribute presence, while ``issubclass`` raises ``TypeError: Protocols with
    non-method members don't support issubclass()``.

    Both directions of the layering rule are enforced at this seam: an
    implementation raises only the five :class:`TargetError` subclasses defined
    above and converts every library exception into one of them, and nothing in
    ``qrcheck/targets/`` may import ``qrcheck.checks`` while
    ``qrcheck/checks/*`` may import this module and nothing else from
    ``qrcheck/targets/``.
    """

    name: str
    capabilities: frozenset[Capability]

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Produce one symbol for `request`, or raise exactly one adapter error.

        Implementations must call :func:`require_capabilities` before doing any
        I/O, so an operation the target never declared is refused without
        touching a process, a socket or the filesystem.

        Raises:
            TargetUnsupported: The request needs a capability the adapter did not
                declare.  Raised before any I/O.
            TargetUnavailable: The target could not be reached at all.
            TargetTimeout: No response within the configured timeout.
            TargetRejected: The target declared a client-side rejection of the
                request.
            TargetContractError: A response arrived but could not be used.
        """
        ...

    def close(self) -> None:
        """Release whatever the adapter holds, and be safe to call twice.

        A closed adapter is released rather than unusable: the runner closes every
        adapter once the run ends, including after a failure, so closing must not
        raise for the ordinary reasons (an already-closed client, a child that has
        already exited).
        """
        ...



def build_request(
    payload: bytes,
    ec_level: str = "M",
    *,
    output_format: str = "png",
    module_size_px: int | None = None,
    quiet_zone_modules: int | None = None,
    foreground_rgb: tuple[int, int, int] = (0, 0, 0),
    background_rgb: tuple[int, int, int] = (255, 255, 255),
    logo_overlay_path: Path | None = None,
    version: int | None = None,
    mask: int | None = None,
) -> GenerateRequest:
    """Build the one legal :class:`GenerateRequest` a check may send to a target.

    This is the single request-construction function in the product, and
    therefore the single definition of "the framework considers this request
    valid": an error level outside ``L``/``M``/``Q``/``H``, a version outside
    1..40, a mask outside 0..7, a non-positive module size, a negative quiet zone,
    a colour channel outside 0..255, or a payload that does not fit the requested
    symbol is refused here, before any target sees it.  Everything the ``IV-*``
    checks send instead goes through the deliberate bypass in
    ``qrcheck/checks/support.py``.

    Args:
        payload: The bytes to encode.  ``bytearray`` and ``memoryview`` are
            accepted and normalised; text is refused, because the caller's
            encoding choice must be visible rather than guessed.
        ec_level: The requested error-correction level.
        output_format: ``png`` or ``svg``, matched case-insensitively.
        module_size_px: Requested module edge in pixels, when pinned.
        quiet_zone_modules: Requested quiet zone in modules, when pinned.
        foreground_rgb: Requested dark-module colour as three 0..255 channels.
        background_rgb: Requested light-module colour as three 0..255 channels.
        logo_overlay_path: Optional logo bitmap to composite onto the symbol.
        version: Optional version pin, 1..40.
        mask: Optional mask pin, 0..7.

    Returns:
        The request, with the format lowered and the payload normalised to
        ``bytes``.

    Raises:
        ValueError: A field is outside its documented range or the payload does
            not fit the symbol.  The message names the offending field.  This is a
            framework-side programming defect, not an adapter error and not a
            configuration error, so the runner converts it into an ``error``
            result carrying its traceback; it is deliberately not
            :class:`TargetRejected`, which would report a framework bug as a
            target defect.
    """
    body = _checked_payload(payload)
    level = _checked_ec_level(ec_level)
    output = _checked_output_format(output_format)
    module_size = _checked_optional_int(
        "module_size_px", module_size_px, minimum=_MIN_MODULE_SIZE
    )
    quiet_zone = _checked_optional_int("quiet_zone_modules", quiet_zone_modules, minimum=0)
    foreground = _checked_rgb("foreground_rgb", foreground_rgb)
    background = _checked_rgb("background_rgb", background_rgb)
    logo = _checked_logo_path(logo_overlay_path)
    pinned_version = _checked_optional_int(
        "version", version, minimum=_MIN_VERSION, maximum=_MAX_VERSION
    )
    pinned_mask = _checked_optional_int("mask", mask, minimum=_MIN_MASK, maximum=_MAX_MASK)

    # The capacity gate is evaluated in byte mode deliberately: a byte is the most
    # expensive way to carry a character, so a payload that fits in byte mode fits
    # in every other mode, and this is the only bound that holds for every corpus
    # class -- random-bytes and nul-byte included, which no smaller charset holds.
    # An unpinned request is bounded by Version 40, the largest Model 2 symbol.
    capacity_version = _MAX_VERSION if pinned_version is None else pinned_version
    capacity = max_payload_bytes(capacity_version, level, "byte")
    if len(body) > capacity:
        symbol = (
            f"version {_MAX_VERSION}, the largest Model 2 symbol,"
            if pinned_version is None
            else f"version {pinned_version}"
        )
        raise ValueError(
            f"payload is {len(body)} bytes but {symbol} at error level {level} holds at "
            f"most {capacity} bytes in byte mode"
        )

    return GenerateRequest(
        payload=body,
        ec_level=level,
        output_format=output,
        module_size_px=module_size,
        quiet_zone_modules=quiet_zone,
        foreground_rgb=foreground,
        background_rgb=background,
        logo_overlay_path=logo,
        version=pinned_version,
        mask=pinned_mask,
    )


def _checked_payload(payload: object) -> bytes:
    """Return `payload` as ``bytes``, refusing text and anything else.

    Raises:
        ValueError: ``payload`` is not bytes-like.  Text is refused rather than
            encoded, so the caller's encoding choice stays visible in the code.
    """
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, (bytearray, memoryview)):
        return bytes(payload)
    raise ValueError(
        "payload must be bytes, bytearray or memoryview so byte-exactness stays "
        f"unambiguous; got {type(payload).__name__} -- encode text before building "
        "a request"
    )


def _checked_ec_level(ec_level: object) -> str:
    """Return `ec_level` when it is one of ``L``, ``M``, ``Q``, ``H``.

    Raises:
        ValueError: ``ec_level`` is not one of the four levels of
            ISO/IEC 18004 §7.5.
    """
    if isinstance(ec_level, str) and ec_level in EC_LEVELS:
        return ec_level
    raise ValueError(f"ec_level must be one of {', '.join(EC_LEVELS)}; got {ec_level!r}")


def _checked_output_format(output_format: object) -> str:
    """Return `output_format` lowered when it is ``png`` or ``svg``.

    Raises:
        ValueError: ``output_format`` is not one of :data:`OUTPUT_FORMATS`.
    """
    if isinstance(output_format, str):
        normalised = output_format.strip().lower()
        if normalised in OUTPUT_FORMATS:
            return normalised
    raise ValueError(
        f"output_format must be one of {', '.join(OUTPUT_FORMATS)}; got {output_format!r}"
    )


def _checked_optional_int(
    field: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int | None:
    """Return `value` when it is ``None`` or an integer inside the given bounds.

    ``bool`` is rejected even though it is an ``int`` subclass, because ``True``
    as a version pin or a module size is a caller mistake that would otherwise
    validate silently.

    Raises:
        ValueError: ``value`` is not an integer or falls outside the bounds, and
            the message names `field`.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or None; got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"in {minimum}..{maximum}"
        raise ValueError(f"{field} must be {bound}; got {value}")
    return value


def _checked_rgb(field: str, value: object) -> tuple[int, int, int]:
    """Return `value` as a three-channel colour tuple of 0..255 integers.

    Raises:
        ValueError: ``value`` is not three integers in range, and the message
            names `field`.  A ``bool`` channel is refused for the same reason a
            ``bool`` integer is.
    """
    expected = (
        f"{field} must be a 3-tuple of integers in "
        f"{_MIN_CHANNEL}..{_MAX_CHANNEL}"
    )
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{expected}; got {value!r}")
    channels: list[int] = []
    for channel in value:
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not _MIN_CHANNEL <= channel <= _MAX_CHANNEL
        ):
            raise ValueError(f"{expected}; got {value!r}")
        channels.append(channel)
    return (channels[0], channels[1], channels[2])


def _checked_logo_path(value: object) -> Path | None:
    """Return `value` as a :class:`~pathlib.Path` when a logo overlay is requested.

    Raises:
        ValueError: ``value`` is neither ``None`` nor a filesystem path.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, (str, os.PathLike)):
        try:
            return Path(os.fspath(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"logo_overlay_path must be a filesystem path; got {value!r} ({exc})"
            ) from None
    raise ValueError(
        f"logo_overlay_path must be a path or None; got {type(value).__name__}"
    )


def _utf8_text(payload: bytes) -> str | None:
    """Return `payload` as text when it round-trips through UTF-8, else ``None``.

    A payload that is not valid UTF-8 has no faithful text form, so evidence
    records only its base64 form rather than a lossy replacement string.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text.encode("utf-8") == payload else None



def require_capabilities(
    adapter: Adapter,
    needed: frozenset[Capability] | Iterable[Capability],
) -> None:
    """Refuse `needed` unless `adapter` declares every one of them.

    The gate is read-only: it reads ``adapter.capabilities`` and nothing else, and
    returns immediately for an empty requirement without touching the adapter at
    all.  It performs no I/O and has no side effect, which is what makes both
    callers safe -- ``registry/resolve.py`` resolving a check to ``skipped`` before
    execution, and an adapter refusing a request it is about to receive.  An
    unsupported request therefore never reaches a target and never consumes its
    ``timeout_s``.

    Args:
        adapter: The target the request is destined for.
        needed: The capabilities the operation requires.  A single member (or its
            name) is accepted for a one-capability operation; a member is itself a
            string, so it would otherwise iterate as characters.

    Raises:
        TargetUnsupported: At least one requirement is not declared.  The message
            names the adapter and the first missing capability in sorted order,
            plus every other missing one, so it is identical on every run.
        ValueError: A member of `needed`, or a value the adapter declares, is not
            a :class:`Capability`.  The configuration schema rejects an unknown
            member at load time, so this is a framework defect rather than a user
            error.
    """
    if isinstance(needed, str):
        required = frozenset({Capability(needed)})
    else:
        required = frozenset(Capability(member) for member in needed)
    if not required:
        return
    declared = frozenset(Capability(member) for member in adapter.capabilities)
    missing = required - declared
    if not missing:
        return
    ordered = sorted(missing, key=str)
    raise TargetUnsupported(
        f"adapter {adapter.name!r} does not declare capability {str(ordered[0])!r}; "
        f"missing: {', '.join(str(member) for member in ordered)}"
    )


def decode_returned_image(
    value: object,
    returns: ReturnsKind,
    *,
    base_dir: Path | None = None,
) -> bytes:
    """Return the artefact bytes a target handed back under the `returns` contract.

    The single implementation of the ``returns`` contract, shared by the
    ``library`` and ``cli`` adapters, so a target's own convention is decoded in
    one place:

    ``raw_bytes``
        `value` is the artefact itself, as ``bytes`` (``bytearray`` and
        ``memoryview`` are normalised).
    ``base64``
        `value` is base64 text, as a ``str`` or as ASCII ``bytes``, decoded with
        ``validate=True`` so a stray character is a failure rather than silently
        skipped.
    ``bytes_path``
        `value` is a path the *target* chose, holding the artefact.
    ``file_path``
        `value` is a path the *framework* supplied (the ``output_arg`` or
        ``kwargs_map`` output path) which the target then wrote.

    A relative path is resolved against `base_dir` when one is given, which is how
    a CLI target's working directory is honoured without the adapter rewriting the
    path the target reported.

    Args:
        value: What the target returned, or the path it wrote.
        returns: The configured ``returns`` kind.
        base_dir: Directory a relative path from the target is resolved against.

    Returns:
        The artefact bytes, never empty.

    Raises:
        TargetContractError: `value` does not match `returns` (wrong type, an
            empty path, undecodable base64, an unreadable file) or the artefact is
            empty.  The message names the ``returns`` kind and the path as given,
            never a resolved absolute path.
        ValueError: `returns` is not one of ``RETURNS_KINDS``.  The configuration
            schema rejects that value at load time, so it is a framework defect.
    """
    if returns not in RETURNS_KINDS:
        raise ValueError(f"returns must be one of {', '.join(RETURNS_KINDS)}; got {returns!r}")

    describe: str | None = None
    if returns == "raw_bytes":
        data = _raw_bytes(value)
    elif returns == "base64":
        data = _base64_bytes(value)
    else:
        data, describe = _path_bytes(value, returns, base_dir)

    if not data:
        where = "" if describe is None else f" at {describe!r}"
        raise TargetContractError(
            f"returns = {returns!r}: the target produced an empty artefact{where}"
        )
    return data


def _raw_bytes(value: object) -> bytes:
    """Return `value` as artefact bytes for the ``raw_bytes`` contract.

    Raises:
        TargetContractError: `value` is not bytes-like.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TargetContractError(
        f"returns = 'raw_bytes' expects bytes; the target returned {type(value).__name__}"
    )


def _base64_bytes(value: object) -> bytes:
    """Decode `value` as base64 text for the ``base64`` contract.

    Raises:
        TargetContractError: `value` is not text, is not ASCII, or is not valid
            base64.
    """
    if isinstance(value, str):
        encoded = value.strip()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            encoded = raw.decode("ascii").strip()
        except UnicodeDecodeError:
            raise TargetContractError(
                "returns = 'base64' expects base64 text; the target returned bytes that "
                f"are not ASCII, starting with {hex_prefix(raw)}"
            ) from None
    else:
        raise TargetContractError(
            f"returns = 'base64' expects base64 text; the target returned "
            f"{type(value).__name__}"
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TargetContractError(
            f"returns = 'base64': the target returned text that is not valid base64 ({exc})"
        ) from None


def _path_bytes(
    value: object,
    returns: ReturnsKind,
    base_dir: Path | None,
) -> tuple[bytes, str]:
    """Read the artefact at the path `value` names, for a path-based contract.

    Returns:
        The artefact bytes and the path exactly as the target gave it, which the
        caller uses in a diagnostic and which is never a resolved absolute path.

    Raises:
        TargetContractError: `value` is not a path, is an empty path, or cannot be
            read.
    """
    if isinstance(value, Path):
        given = value
        shown = str(value)
    elif isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if not isinstance(text, str):
            raise TargetContractError(
                f"returns = {returns!r} expects a text path; the target returned bytes"
            )
        if not text:
            raise TargetContractError(
                f"returns = {returns!r} expects a path; the target returned an empty path"
            )
        given = Path(text)
        shown = text
    else:
        raise TargetContractError(
            f"returns = {returns!r} expects a path; the target returned "
            f"{type(value).__name__}"
        )

    resolved = given if base_dir is None or given.is_absolute() else base_dir / given
    try:
        return resolved.read_bytes(), shown
    except OSError as exc:
        reason = exc.strerror or exc.__class__.__name__
        raise TargetContractError(
            f"returns = {returns!r}: cannot read the artefact at {shown!r} ({reason})"
        ) from None


def resolve_dotted_path(data: object, path: str) -> object:
    """Walk `path` into `data`, supporting mapping keys and sequence indices.

    This is the one dotted-path lookup in the product; the ``cli`` and ``http``
    adapters both import it, so a configured ``image_field`` or ``metadata_map``
    source means the same thing for either transport.  A segment is read as a
    mapping key when the current value is a mapping and as a zero-based index when
    it is a sequence of items -- ``"images.0.png"`` reads key ``images``, then
    element ``0``, then key ``png``.

    The raw response body is *not* reachable through this function: ``_body``
    names it and is handled by the ``http`` adapter, because it is a property of
    that transport rather than of a path into a parsed value.

    Args:
        data: The parsed response or metadata value to walk into.
        path: A non-empty dotted path.

    Returns:
        The value the path names.

    Raises:
        KeyError: A segment is absent, a sequence segment is not a non-negative
            integer index, an index is out of range, or the current value is
            neither a mapping nor a sequence.  ``KeyError`` is a ``LookupError``,
            and the message carries the full path and the failing segment.
        ValueError: `path` is not a non-empty string -- a framework defect, since
            the configuration schema rejects an empty field path at load time.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"dotted path must be a non-empty string; got {path!r}")

    current: object = data
    for segment in path.split("."):
        if not segment:
            raise KeyError(f"dotted path {path!r} has an empty segment")
        if isinstance(current, Mapping):
            if segment not in current:
                available = ", ".join(sorted(str(key) for key in current)) or "none"
                raise KeyError(
                    f"dotted path {path!r}: segment {segment!r} is absent; "
                    f"available keys: {available}"
                )
            current = current[segment]
            continue
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray, memoryview)
        ):
            try:
                index = int(segment, 10)
            except ValueError:
                raise KeyError(
                    f"dotted path {path!r}: segment {segment!r} is not an index into a "
                    f"sequence of {len(current)}"
                ) from None
            if not 0 <= index < len(current):
                raise KeyError(
                    f"dotted path {path!r}: index {index} is out of range for a sequence "
                    f"of {len(current)}"
                )
            current = current[index]
            continue
        raise KeyError(
            f"dotted path {path!r}: segment {segment!r} cannot be read from "
            f"{type(current).__name__}"
        )
    return current


def apply_metadata_map(
    raw: Mapping[str, object],
    metadata_map: Mapping[str, str],
) -> dict[str, object]:
    """Read a target's declared metadata through the configured `metadata_map`.

    The four framework fields are exactly ``version``, ``ec``, ``mask`` and
    ``mode``.  Each configured entry maps one of them to the source key or dotted
    path that carries it in the target's response, so no vendor field name is ever
    hard-coded in an adapter.

    The values returned are the target's claims: evidence for
    ``GeneratedSymbol.declared_*``, never the assertion a check makes.  ``version``
    and ``mask`` are coerced to integers, ``ec`` and ``mode`` to stripped text.

    Args:
        raw: The parsed metadata the target reported.
        metadata_map: Framework field name to source key or dotted path.  An empty
            mapping returns an empty result without touching `raw`, which is the
            normal case for a target that reports nothing.

    Returns:
        A mapping from each *configured* framework field to its coerced value, in
        :data:`METADATA_FIELDS` order.  A field that is not configured is absent
        rather than ``None``.

    Raises:
        TargetContractError: A configured source is absent from `raw`, resolves to
            ``None``, or holds a value that cannot be coerced to the field's type.
            This is the "missing field named in config" contract failure, and the
            message names the framework field and the configured path.
        ValueError: `metadata_map` names a framework field outside
            :data:`METADATA_FIELDS`.  The configuration schema rejects that at load
            time, so it is a framework defect.
    """
    if not metadata_map:
        return {}
    unknown = sorted(set(metadata_map) - set(METADATA_FIELDS))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(
            f"metadata_map names unknown framework field(s) {names}; expected any of "
            f"{', '.join(METADATA_FIELDS)}"
        )

    resolved: dict[str, object] = {}
    for field_name in METADATA_FIELDS:
        if field_name not in metadata_map:
            continue
        source = metadata_map[field_name]
        try:
            value = resolve_dotted_path(raw, source)
        except KeyError as exc:
            detail = exc.args[0] if exc.args else str(exc)
            raise TargetContractError(
                f"metadata_map[{field_name!r}] names {source!r}, which the target's "
                f"response does not provide: {detail}"
            ) from None
        resolved[field_name] = _metadata_value(field_name, source, value)
    return resolved


def _metadata_value(field_name: str, source: str, value: object) -> object:
    """Coerce one configured metadata value to its framework field's type.

    Raises:
        TargetContractError: The value is ``None`` or cannot be coerced.
    """
    if value is None:
        raise TargetContractError(
            f"metadata_map[{field_name!r}] names {source!r}, whose value is null"
        )
    if field_name in ("version", "mask"):
        return _metadata_int(field_name, source, value)
    return _metadata_text(field_name, source, value)


def _metadata_int(field_name: str, source: str, value: object) -> int:
    """Return `value` as an integer for a numeric metadata field.

    Raises:
        TargetContractError: `value` is a bool, a non-integral number, or text that
            is not a base-10 integer.
    """
    invalid = (
        f"metadata_map[{field_name!r}] names {source!r}, whose value {value!r} is not an "
        "integer"
    )
    if isinstance(value, bool):
        raise TargetContractError(invalid)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError:
            raise TargetContractError(invalid) from None
    raise TargetContractError(invalid)


def _metadata_text(field_name: str, source: str, value: object) -> str:
    """Return `value` stripped as text for a textual metadata field.

    Raises:
        TargetContractError: `value` is not text.
    """
    if isinstance(value, str):
        return value.strip()
    raise TargetContractError(
        f"metadata_map[{field_name!r}] names {source!r}, whose value {value!r} is not text"
    )


def ensure_image_bytes(data: bytes, output_format: str) -> bytes:
    """Return `data` after confirming it plausibly is the artefact it claims to be.

    A dependency-free plausibility gate, not a decoder: it sniffs the leading
    bytes for a PNG signature, a JPEG signature, or an SVG/XML marker within the
    first 512 bytes after any byte-order mark and whitespace, and refuses an empty
    body, a body that is none of those, and a body whose format contradicts the
    requested `output_format`.  That keeps the "image bytes that no decoder or
    image library can open" contract failure live without this module importing an
    image library -- Pillow is reachable only through ``qrcheck/evidence/mux.py``.
    Whether the bytes are a *valid* image is decided later, by the decode layer,
    and must not be duplicated here.

    Args:
        data: The artefact the target returned.
        output_format: The format the request asked for.

    Returns:
        The bytes, unchanged, when they pass the gate.

    Raises:
        TargetContractError: The body is empty, unrecognisable, or recognisably a
            different format from the one requested.  The message carries the
            leading bytes' hex so the report shows what actually arrived.
        ValueError: `data` is not bytes-like.  A framework defect rather than a
            target one, since adapters pass the bytes they decoded.
    """
    if isinstance(data, (bytearray, memoryview)):
        body = bytes(data)
    elif isinstance(data, bytes):
        body = data
    else:
        raise ValueError(f"ensure_image_bytes expects bytes; got {type(data).__name__}")

    if not body:
        raise TargetContractError(
            f"empty image body for output format {output_format!r}"
        )

    sniffed = _sniff_image_format(body)
    if sniffed is None:
        raise TargetContractError(
            f"image body for output format {output_format!r} is not PNG, JPEG or SVG; "
            f"it starts with {hex_prefix(body)}"
        )
    expected: frozenset[str] | None = None
    if isinstance(output_format, str):
        expected = _FORMAT_SNIFFS.get(output_format.strip().lower())
    if expected is not None and sniffed not in expected:
        raise TargetContractError(
            f"image body is {sniffed} but the request declared output format "
            f"{output_format!r}; it starts with {hex_prefix(body)}"
        )
    return body


def _sniff_image_format(data: bytes) -> str | None:
    """Return ``png``, ``jpeg``, ``svg`` or ``None`` for the body `data`.

    The SVG test tolerates a byte-order mark, leading whitespace, an XML
    declaration and a doctype, because every one of those is legitimate ahead of
    the root element of a renderer's output.
    """
    if data.startswith(_PNG_SIGNATURE):
        return "png"
    if data.startswith(_JPEG_SIGNATURE):
        return "jpeg"
    window = data[:_SVG_SNIFF_BYTES]
    if window.startswith(_BYTE_ORDER_MARK):
        window = window[len(_BYTE_ORDER_MARK) :]
    lowered = window.lstrip().lower()
    if lowered.startswith(b"<svg") or lowered.startswith(b"<?xml"):
        return "svg"
    if b"<svg" in lowered:
        return "svg"
    return None


def hex_prefix(data: bytes, limit: int = HEX_PREFIX_LIMIT) -> str:
    """Return the first `limit` bytes of `data` as lowercase hexadecimal.

    The record of what a target actually sent when its body was not usable.  At
    the default limit of 256 bytes the result is 512 hex characters -- the length
    ``tests/unit/test_targets_errors.py`` asserts for the non-image-bytes
    contract failure -- so the limit is a contract, not a formatting choice.

    Raises:
        ValueError: `limit` is negative, or `data` is not bytes-like.
    """
    if isinstance(data, (bytearray, memoryview)):
        body = bytes(data)
    elif isinstance(data, bytes):
        body = data
    else:
        raise ValueError(f"hex_prefix expects bytes; got {type(data).__name__}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(f"limit must be a non-negative integer; got {limit!r}")
    return body[:limit].hex()
