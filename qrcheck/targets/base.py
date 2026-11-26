"""The adapter port every QR generator target is driven through.

The seam between what is asserted (``qrcheck/checks/``) and what it is asserted against: the closed
thirteen-member capability vocabulary, the two records a target exchanges with the framework, the
closed five-error vocabulary, the one request builder and the capability gate.

Guarantees: :class:`Capability` is the complete list of operations a target may declare;
:func:`require_capabilities` refuses an undeclared one **before any I/O**, so an unsupported request
never reaches a target nor consumes its ``timeout_s``; :func:`build_request` is the only legal path
to a :class:`GenerateRequest`, which itself validates nothing. The five errors, one trigger each:
``TargetUnavailable`` -- the process cannot be spawned, the connection is refused, the import path
is missing; ``TargetTimeout`` -- no response within ``targets.<name>.timeout_s``; ``TargetRejected``
-- the target returned a declared client-side rejection of a request the framework considers valid,
carried verbatim; ``TargetContractError`` -- a response arrived but is
unusable: empty body, undecodable base64, image bytes no decoder or image library can open, a
configured field the response does not carry; ``TargetUnsupported`` -- the request asks for a
capability the adapter did not declare.

Layering: the standard library, :mod:`qrcheck.config.models`, :mod:`qrcheck.corpus.capacity` and a
deferred ``payloads.mode_of``; nothing here imports an image library, a decoder or
``qrcheck.checks``.
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

from qrcheck.config.models import (
    CAPABILITY_NAMES,
    METADATA_FIELD_NAMES,
    RETURNS_KINDS,
    ReturnsKind,
    TargetConfig,
)
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

# The artefact formats a request may ask for; both are also capability names.
OUTPUT_FORMATS: Final[tuple[str, ...]] = ("png", "svg")

# The four metadata fields a target may report, in projection order: the keys of
# ``metadata_map`` and of :func:`apply_metadata_map`. Every one is evidence only, never an
# assertion; the names are declared once as ``models.METADATA_FIELD_NAMES``.
METADATA_FIELDS: Final[tuple[str, ...]] = METADATA_FIELD_NAMES

# How many leading bytes :func:`hex_prefix` renders: the first 256 bytes of a body that is not
# an image, recorded as 512 hex characters so the defect is diagnosable from the report.
HEX_PREFIX_LIMIT: Final[int] = 256

# ISO/IEC 18004 Model 2 version and mask ranges; outside these is outside the standard.
_MIN_VERSION: Final[int] = 1
_MAX_VERSION: Final[int] = 40
_MIN_MASK: Final[int] = 0
_MAX_MASK: Final[int] = 7

# A colour channel is one unsigned byte.
_MIN_CHANNEL: Final[int] = 0
_MAX_CHANNEL: Final[int] = 255

# A requested module edge is at least one pixel; a zero-pixel module renders no symbol.
_MIN_MODULE_SIZE: Final[int] = 1

# Leading byte sequences that identify the artefact formats. A signature is evidence, not
# decodability: whether the bytes are a *valid* image is the decode layer's decision, because
# Pillow is reachable only through ``qrcheck/evidence/mux.py``.
_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE: Final[bytes] = b"\xff\xd8\xff"
_BYTE_ORDER_MARK: Final[bytes] = b"\xef\xbb\xbf"

# How far into a body an SVG or XML marker is searched for, because renderer output may open
# with a BOM, whitespace, an XML declaration or a doctype.
_SVG_SNIFF_BYTES: Final[int] = 512

# Which sniffed formats satisfy which requested format. JPEG is not a request format, but a
# target answering a ``png`` request with JPEG bytes must be reported as a mismatch rather than
# as an unknown body.
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

    A ``StrEnum`` member *is* its string value, so a set of members compares and hashes equal to the
    ``frozenset[str]`` a resolved configuration carries. The set is closed at thirteen, in a fixed
    order: no check, adapter or resolution rule may add, rename, reorder or omit a member, because
    ``SY-MATRIX-EQUIV``, the ``RN-*`` checks, ``IV-OVERCAPACITY`` and the capability-resolution
    table key on these exact names.
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


def _check_capability_vocabulary() -> bool:
    """Fail the import when the two capability declarations hold different members.

    The thirteen names exist twice -- ``CAPABILITY_NAMES`` there and :class:`Capability` here -- and
    this gate is what makes a rename on either side fail loudly at import.

    Raises:
    RuntimeError: The declarations differ; the message names both sites and every member present in
    only one of them.
    """
    drift = frozenset(Capability) ^ frozenset(CAPABILITY_NAMES)
    if not drift:
        return True
    members = ", ".join(sorted(str(member) for member in drift))
    raise RuntimeError(
        "capability vocabulary drift: 'Capability' in qrcheck/targets/base.py and "
        f"'CAPABILITY_NAMES' in qrcheck/config/models.py disagree on: {members}"
    )


# The gate runs at import; drift between the two declarations fails this import.
_CAPABILITY_DRIFT_GATE: Final[bool] = _check_capability_vocabulary()


def declared_capabilities(config: TargetConfig) -> frozenset[Capability]:
    """Return the capabilities `config` declares, as members of :class:`Capability`.

    The one place the configuration's ``frozenset[str]`` becomes a ``frozenset[Capability]``, and
    the one sort that keeps resolution deterministic.

    Raises:
    ValueError: A declared name is not a :class:`Capability` member -- a framework defect, since the
    schema rejects an unknown member at load.
    """
    try:
        return frozenset(Capability(name) for name in sorted(config.capabilities))
    except ValueError as exc:
        raise ValueError(
            f"target {config.name!r} declares an unknown capability: {exc}"
        ) from None


class TargetError(Exception):
    """Base class of the five errors an adapter may raise.

    Catch this to handle every failure the target layer can report, because nothing outside this
    hierarchy escapes an adapter.
    """


class TargetUnavailable(TargetError):
    """The target could not be reached at all.

    The single trigger is that the process cannot be spawned, the connection is refused, or the
    configured import path is missing; the diagnostic names the transport failure and the configured
    field.
    """


class TargetTimeout(TargetError):
    """The target did not answer within ``targets.<name>.timeout_s``.

    The single trigger is that no response arrived inside the configured timeout; the diagnostic
    names the timeout in seconds and the operation attempted.
    """


class TargetRejected(TargetError):
    """The target returned a declared client-side rejection of a valid request.

    The single trigger is a target's own refusal of a request the framework considers valid, and the
    message is the target's text unchanged: ``str(exc)`` is exactly what the target said, never
    re-wrapped or truncated, because the ``IV-*`` checks record it verbatim in ``decoded.rejection``
    and every other check reports reason ``target_rejected``.
    """


class TargetContractError(TargetError):
    """A response arrived but could not be used.

    The single trigger is a response present but unusable: an empty body, undecodable base64, image
    bytes that no decoder or image library can open, or a field named in configuration that the
    response does not carry. ``body_prefix`` holds the first :data:`HEX_PREFIX_LIMIT` bytes of the
    returned artefact when there are any, so the evidence field ``hex_prefix`` records the body
    itself rather than this message.
    """

    def __init__(self, message: str = "", *, body_prefix: bytes = b"") -> None:
        super().__init__(message)
        self.body_prefix: bytes = bytes(body_prefix[:HEX_PREFIX_LIMIT])


class TargetUnsupported(TargetError):
    """The request needs a capability the adapter did not declare.

    Raised by :func:`require_capabilities`, and belt-and-braces by an adapter about to be asked for
    something it cannot do, **before any I/O**: the request must never reach a target nor consume
    its ``timeout_s``. The diagnostic names the missing capability and the adapter.
    """


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """One request the framework makes of a target.

    The ten fields are the request contract in full; the eight after ``payload`` are defaulted for
    convenience only, and ``payload`` is bytes and never text so byte-exactness is unambiguous. This
    record validates nothing -- no ``__post_init__``, no coercion, no guard: the gate is
    :func:`build_request`, and the documented bypass is
    ``qrcheck/checks/support.py:construct_out_of_contract``. Do not add validation here.
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

        ``payload`` travels as ``payload_b64`` with ``payload_len``, plus ``payload_text`` when the
        bytes round-trip through UTF-8; colours become ``[r, g, b]`` lists and ``logo_overlay_path``
        stays the configured path text, so no artefact records an absolute location.
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

    ``image_bytes`` is the artefact exactly as produced and ``output_format`` the format actually
    produced -- :func:`ensure_image_bytes` catches a mismatch. The four ``declared_*`` fields are
    **evidence only, never the assertion**: the checks read EC level, mask and version from the
    decoders and the symbol's format information, so a target that lies is recorded as having lied.
    ``elapsed_ms`` is recorded and excluded from every determinism comparison.
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

    Exactly these four members, and no fifth, so a test stub implements it in four lines.
    Conformance is asserted with ``isinstance(adapter, Adapter)`` and never ``issubclass``: on
    CPython 3.11 a runtime-checkable protocol with non-method members raises ``TypeError`` for
    ``issubclass``. An implementation raises only the five :class:`TargetError` subclasses and
    converts every library exception into one of them.
    """

    name: str
    capabilities: frozenset[Capability]

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Produce one symbol for `request`, or raise exactly one adapter error.

        Implementations call :func:`require_capabilities` before any I/O, so an undeclared operation
        never touches a process, a socket or the filesystem.

        Raises:
        TargetUnsupported: Undeclared capability; raised before any I/O.
        TargetUnavailable: The target could not be reached at all.
        TargetTimeout: No response within the configured timeout.
        TargetRejected: The target declared a client-side rejection of the request.
        TargetContractError: A response arrived but could not be used.
        """
        ...

    def close(self) -> None:
        """Release whatever the adapter holds, and be safe to call twice.

        The runner closes every adapter once the run ends, including after a failure, so closing
        must not raise for the ordinary reasons.
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

    The single definition of "the framework considers this request valid": an error level outside
    ``L``/``M``/``Q``/``H``, a version outside 1..40, a mask outside 0..7, a non-positive module
    size, a negative quiet zone, a colour channel outside 0..255, or a payload that does not fit the
    requested version and level per the capacity tables in the payload's own mode is refused here.
    Text is refused, because the caller's encoding must stay visible.

    Raises:
    ValueError: A field is outside its documented range or the payload does not fit; the message
    names the field. A framework-side defect, deliberately not :class:`TargetRejected`.
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

    # The capacity gate is measured in the mode the payload's own bytes are limited to:
    # a conforming encoder spends the smallest charset that holds the payload, so the
    # capacity tables' row for that mode is the bound the request is actually held to.
    # The byte row alone is the smallest of the three and would refuse a payload that
    # lawfully fits a pinned version in numeric or alphanumeric mode -- Version 1 at
    # error level L holds 41 digits but only 17 bytes. Numeric and alphanumeric
    # payloads are ASCII by construction, so len(body) is their character count as well
    # as their byte count. An unpinned request is bounded by Version 40, the largest
    # Model 2 symbol, in that same mode.
    #
    # ``mode_of`` is imported at the point of use, never at module scope: its module
    # imports ``numpy`` for the corpus's seeded generators, and this port is imported by
    # the registry, the decoders and every check, so a module-level import would put
    # numpy's cost into each of them. ``mode_of`` itself is pure string classification.
    from qrcheck.corpus.payloads import mode_of

    capacity_version = _MAX_VERSION if pinned_version is None else pinned_version
    mode = mode_of(body)
    capacity = max_payload_bytes(capacity_version, level, mode)
    if len(body) > capacity:
        symbol = (
            f"version {_MAX_VERSION}, the largest Model 2 symbol,"
            if pinned_version is None
            else f"version {pinned_version}"
        )
        unit = "bytes" if mode == "byte" else "characters"
        raise ValueError(
            f"payload is {len(body)} bytes but {symbol} at error level {level} holds at "
            f"most {capacity} {unit} in {mode} mode"
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
    ValueError: ``payload`` is not bytes-like; text is refused, never encoded.
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
    ValueError: It is not one of the four levels of ISO/IEC 18004 clause 7.5.
    """
    if isinstance(ec_level, str) and ec_level in EC_LEVELS:
        return ec_level
    raise ValueError(f"ec_level must be one of {', '.join(EC_LEVELS)}; got {ec_level!r}")


def _checked_output_format(output_format: object) -> str:
    """Return `output_format` lowered when it is ``png`` or ``svg``.

    Raises:
    ValueError: It is not one of :data:`OUTPUT_FORMATS`.
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

    ``bool`` is rejected even though it is an ``int`` subclass, because ``True`` as a version pin
    would otherwise validate silently.

    Raises:
    ValueError: ``value`` is not an integer or is out of bounds; the message names `field`.
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
    ValueError: ``value`` is not three in-range integers; the message names `field`, and a ``bool``
    channel is refused.
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
    """Return `value` as a path when a logo overlay is requested.

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

    A payload that is not valid UTF-8 has no faithful text form, so evidence records only its base64
    form rather than a lossy replacement string.
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

    Read-only and free of I/O, returning immediately for an empty requirement, so an unsupported
    request never reaches a target and never consumes its ``timeout_s``.

    Raises:
    TargetUnsupported: A requirement is not declared; the message names the adapter and every
    missing capability in sorted order.
    ValueError: A member of `needed`, or a name the adapter declares, is not a :class:`Capability`
    -- a framework defect.
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

    The single implementation of ``returns``: ``raw_bytes`` is the artefact itself; ``base64`` is
    base64 text decoded with ``validate=True``, so a stray character fails; ``bytes_path`` and
    ``file_path`` name a path the target chose and one the framework supplied.

    Raises:
    TargetContractError: `value` does not match `returns`, or the artefact is empty.
    ValueError: `returns` is not one of ``RETURNS_KINDS`` -- a framework defect.
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
    TargetContractError: `value` is not text, is not ASCII, or is not valid base64.
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

    Returns: The artefact bytes and the path exactly as the target gave it, which a diagnostic
    quotes and which is never a resolved absolute path.

    Raises:
    TargetContractError: `value` is not a path, is an empty path, or cannot be read.
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

    The one dotted-path lookup in the product, so a configured ``image_field`` or ``metadata_map``
    source means the same thing for either transport. A segment is a mapping key when the current
    value is a mapping and a zero-based index when it is a sequence.

    Raises:
    KeyError: A segment is absent, is not a valid index, is out of range, or the value is neither a
    mapping nor a sequence.
    ValueError: `path` is not a non-empty string -- a framework defect.
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

    ``version``, ``ec``, ``mask`` and ``mode`` are each mapped to the source key or dotted path
    carrying it, so no vendor field name is hard-coded in an adapter, and the values are the
    target's claims -- evidence, never the assertion a check makes.

    Raises:
    TargetContractError: A configured source is absent, resolves to ``None``, or cannot be coerced
    -- the "missing field named in config" contract failure.
    ValueError: `metadata_map` names a field outside :data:`METADATA_FIELDS`.
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
    TargetContractError: `value` is a bool, a non-integral number, or text that is not a base-10
    integer.
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

    A dependency-free plausibility gate, not a decoder: it sniffs the leading bytes for a PNG
    signature, a JPEG signature or an SVG/XML marker, and refuses an empty body, a body that is none
    of those, and one whose format contradicts the requested `output_format`.

    Raises:
    TargetContractError: The body is empty, unrecognisable or the wrong format; the refused body
    travels on ``body_prefix``.
    ValueError: `data` is not bytes-like -- a framework defect.
    """
    if isinstance(data, (bytearray, memoryview)):
        body = bytes(data)
    elif isinstance(data, bytes):
        body = data
    else:
        raise ValueError(f"ensure_image_bytes expects bytes; got {type(data).__name__}")

    if not body:
        raise TargetContractError(
            f"empty image body for output format {output_format!r}",
            body_prefix=body,
        )

    sniffed = _sniff_image_format(body)
    if sniffed is None:
        raise TargetContractError(
            f"image body for output format {output_format!r} is not PNG, JPEG or SVG; "
            f"it starts with {hex_prefix(body)}",
            body_prefix=body,
        )
    expected: frozenset[str] | None = None
    if isinstance(output_format, str):
        expected = _FORMAT_SNIFFS.get(output_format.strip().lower())
    if expected is not None and sniffed not in expected:
        raise TargetContractError(
            f"image body is {sniffed} but the request declared output format "
            f"{output_format!r}; it starts with {hex_prefix(body)}",
            body_prefix=body,
        )
    return body


def _sniff_image_format(data: bytes) -> str | None:
    """Return ``png``, ``jpeg``, ``svg`` or ``None`` for the body `data`.

    The SVG test tolerates a byte-order mark, leading whitespace, an XML declaration and a doctype,
    because every one of those legitimately precedes a renderer's root element.
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

    The record of what a target actually sent when its body was not usable; at the default limit of
    256 bytes the result is 512 hex characters, the length ``tests/unit/test_targets_errors.py``
    asserts.

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
