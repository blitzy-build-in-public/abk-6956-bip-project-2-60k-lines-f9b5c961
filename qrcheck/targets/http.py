"""The loopback-only ``http`` adapter: every request and response shape comes from configuration.

:class:`HttpAdapter` drives a QR generator reachable over loopback HTTP.  Nothing
in this module knows a vendor's endpoint, field name or response key: the URL, the
method, the request body, the path to the image inside the response, the way that
image is carried and the paths to any declared metadata all come from the
``[targets.<name>]`` table (:class:`qrcheck.config.models.HTTPTarget`).  A fourth
vendor is therefore a configuration edit and not a code change, which is the
guarantee ``qrcheck/checks/`` relies on when it sees only the adapter port.

Request templates
=================
``request_template`` is the mapping the adapter sends after substituting the eight
placeholders of :data:`qrcheck.config.models.TEMPLATE_FIELDS`:

======================  ========  ==============================================
``payload_b64``         text      the payload bytes, base64-encoded
``payload_utf8``        text      the payload as UTF-8 text; omitted when the
                                  payload is not valid UTF-8
``ec_level``            text      ``L``, ``M``, ``Q`` or ``H``
``format``              text      ``png`` or ``svg``
``size``                integer   module edge in pixels, when pinned
``margin``              integer   quiet zone in modules, when pinned
``fg``                  text      the dark-module colour as ``#rrggbb``
``bg``                  text      the light-module colour as ``#rrggbb``
======================  ========  ==============================================

A string leaf that is one whole placeholder (``"{size}"``) receives the value with
its own type, so ``{size}`` arrives as a JSON number rather than as text.  A
placeholder embedded in a longer string (``"--ec={ec_level}"``) receives the
textual form.  Everything else in the template -- numbers, booleans, null -- is
sent unchanged.  Conversion and format specifications (``{size:d}``,
``{payload_utf8!s}``) are honoured and ``{{``/``}}`` are literal braces, exactly
as the configuration schema validates them.

**A placeholder whose request value is ``None`` omits the containing key** rather
than sending an empty string.  ``module_size_px`` and ``quiet_zone_modules`` are
``None`` whenever a case lets the target apply its own default, and a key that is
simply absent is the only way to say "unspecified" over a protocol whose values
are otherwise always present.  The same rule covers ``{payload_utf8}`` meeting a
payload that is not valid UTF-8: that key is dropped, and a template carrying
``payload_utf8`` as its only payload channel refuses such a payload with
:class:`TargetUnsupported` instead of sending a request with no payload at all.

``GET`` carries the substituted template as query parameters, so every leaf must
be a scalar and a nested table or array is a :class:`TargetContractError` naming
the template path that holds it.  Passing the mapping as ``params=`` rather than
concatenating a URL is what keeps a base64 payload's ``+``, ``/`` and ``=``
intact through the query string; a hand-built query string would turn ``+`` into a
space and corrupt exactly the binary payloads the corpus is full of.  Every other
method sends the template as a JSON body.

Responses
=========
``image_field`` names the image inside the response: ``_body`` means the response
body *is* the image, and any other value is a dotted path into the parsed JSON
response (``data.0.image``), read by the single dotted-path resolver in the
product, :func:`qrcheck.targets.base.resolve_dotted_path`.  ``image_encoding``
says how that value carries the image: ``base64`` decodes it as base64 text,
``raw`` takes a string's UTF-8 bytes or a byte value unchanged.  Whichever route
the image takes, :func:`qrcheck.targets.base.ensure_image_bytes` is the gate that
refuses an empty, unrecognisable or wrong-format body; whether those bytes are a
*valid* image stays the decode layer's decision and is not repeated here.

``metadata_map`` maps each of the four framework metadata fields -- ``version``,
``ec``, ``mask``, ``mode`` -- to the dotted path carrying it, and the values land
in ``GeneratedSymbol.declared_*`` as evidence a check may compare against, never
as the assertion it makes.  A target whose ``image_field`` is ``_body`` has no
JSON envelope, so it declares no metadata.

Loopback only
=============
The framework is offline: the only hosts it may reach are ``127.0.0.1``,
``localhost`` and ``::1``, and that is enforced when the configuration is loaded
(``qrcheck.config.schema``) with exit 2 naming the offending host.  The
constructor re-checks the URL anyway and raises :class:`ValueError` -- a framework
defect rather than a user diagnostic -- so a caller that builds an ``HTTPTarget``
directly cannot aim the transport off loopback.  The client is created with
``follow_redirects=False`` because a redirect could carry the request to a host
the product forbids, and with ``trust_env=False`` because an ``HTTP_PROXY`` or
``HTTPS_PROXY`` environment variable would route a loopback request through a
remote host.  Between them the offline guarantee holds at the transport layer and
not merely in configuration.  No credential, cookie jar, proxy setting,
authorisation header or telemetry is read, written or sent.

A non-2xx status is always :class:`TargetContractError`, never
:class:`TargetRejected`: the plan gives an HTTP target no declared-rejection
channel (``cli`` and ``library`` have one through ``returns``), so a refusal and a
crash are reported alike and the four ``IV-BAD-OPTION``/``IV-EMPTY`` probes --
which pass only on a declared rejection -- fail against an HTTP target by design
rather than by oversight.

Configuration faults this module refuses itself
===============================================
The schema validates the template's placeholders, the ``image_encoding`` enum and
the loopback URL before any record is built.  Four faults it cannot decide are
reported as :class:`qrcheck.config.schema.ConfigError`, so they leave as exit 2
naming the offending key instead of as exit 3 with a traceback:

* a ``request_template`` given as JSON text that does not parse or that is not a
  JSON object (the schema's own shape is a table, and a JSON-text template is
  still accepted so that a hand-built record cannot silently mean nothing);
* a placeholder that is not one of the eight -- belt-and-braces, because the
  schema rejects one at load time;
* a ``metadata_map`` naming a framework field outside ``version``, ``ec``,
  ``mask`` and ``mode``;
* a template carrying no payload placeholder at all, since every request it could
  build would omit the payload.

Layering
========
The standard library, :mod:`httpx`, and three project modules --
:mod:`qrcheck.config.models`, :mod:`qrcheck.config.schema` and
:mod:`qrcheck.targets.base`.  No check, report, decode or registry module is
imported, no image library is imported, and ``qrcheck/checks/`` may import the
adapter port but never this module; ``tests/unit/test_layering.py`` holds both
directions mechanically.
"""

from __future__ import annotations

import base64
import binascii
import json
import string
import threading
import time
import urllib.parse
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import httpx

from qrcheck.config.models import IMAGE_ENCODINGS, TEMPLATE_FIELDS, HTTPTarget
from qrcheck.config.schema import ConfigError
from qrcheck.targets.base import (
    METADATA_FIELDS,
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetError,
    TargetTimeout,
    TargetUnavailable,
    TargetUnsupported,
    apply_metadata_map,
    declared_capabilities,
    ensure_image_bytes,
    hex_prefix,
    require_capabilities,
    resolve_dotted_path,
)

__all__ = ["HttpAdapter"]

# ---------------------------------------------------------------------------
# Protocol vocabulary
#
# The only string literals this module holds: the URL schemes, the method
# comparison, the JSON content type, the Location header, the `_body` sentinel,
# the two `image_encoding` values, the eight placeholder names of TEMPLATE_FIELDS,
# the framework's own field and capability names (`png`, `svg`, `version`, `ec`,
# `mask`, `mode`, `logo_overlay_path`) and the configuration keys a fault is
# reported on.  Every vendor-specific field name, endpoint and response key arrives
# from configuration, so a grep for a literal here finds protocol and framework
# vocabulary and no vendor shape.
# ---------------------------------------------------------------------------

_GET_METHOD: Final[str] = "GET"
_JSON_CONTENT_TYPE: Final[str] = "application/json"
_CONTENT_TYPE_HEADER: Final[str] = "content-type"
_LOCATION_HEADER: Final[str] = "location"
_BODY_SENTINEL: Final[str] = "_body"
_BASE64_ENCODING: Final[str] = "base64"
_RAW_ENCODING: Final[str] = "raw"

# The loopback hosts and URL schemes the transport may use.  The authoritative
# gate is ``qrcheck.config.schema``; this module re-checks the same rule so a
# directly constructed record cannot bypass it.
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})
_LOOPBACK_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# The request format's capability member, so a request for one format requires
# exactly the capability that names it.
_FORMAT_CAPABILITIES: Final[Mapping[str, Capability]] = MappingProxyType(
    {"png": Capability.png, "svg": Capability.svg}
)

# The framework's default colour pair.  A request carrying it needs no colour
# capability, because honouring "black on white" asks nothing of the target.
_DEFAULT_FOREGROUND_RGB: Final[tuple[int, int, int]] = (0, 0, 0)
_DEFAULT_BACKGROUND_RGB: Final[tuple[int, int, int]] = (255, 255, 255)

# The placeholders that carry the payload.  A template referencing neither can
# never send one, so it is refused when the adapter is constructed.
_PAYLOAD_PLACEHOLDERS: Final[frozenset[str]] = frozenset({"payload_b64", "payload_utf8"})

# Request fields the template cannot convey, paired with the capability that
# governs each.  ``TEMPLATE_FIELDS`` has no placeholder for any of them, so a
# request that pins one is refused before any I/O rather than sent without it.
_INEXPRESSIBLE_FIELDS: Final[tuple[tuple[str, Capability], ...]] = (
    ("version", Capability.version_pin),
    ("mask", Capability.mask_pin),
    ("logo_overlay_path", Capability.logo_overlay),
)

# How much of an unusable response body a diagnostic quotes, so a report stays
# readable while still showing what actually arrived.
_STATUS_BODY_EXCERPT_BYTES: Final[int] = 500
_JSON_BODY_EXCERPT_BYTES: Final[int] = 64

# One formatter for the whole module: ``string.Formatter`` is stateless, and using
# a single instance keeps placeholder parsing identical everywhere.
_FORMATTER: Final[string.Formatter] = string.Formatter()

# Sentinel meaning "the template key holding this value must be dropped", which is
# how the omit-when-``None`` rule travels out of a recursive walk.
_OMIT: Final[object] = object()

# One parsed template leaf: the literal text, the field name (``None`` for literal
# text only), the format specification and the conversion.
_ParsedLeaf = tuple[str, str | None, str, str | None]


# ---------------------------------------------------------------------------
# Template handling
# ---------------------------------------------------------------------------


def _leaf_key(key_prefix: str, suffix: str) -> str:
    """Return the dotted configuration key of one template leaf.

    The key is what a :class:`ConfigError` reports, so it names the exact leaf a
    reader has to edit -- ``targets.example.request_template.image`` or
    ``targets.example.request_template.emitted[0].name`` -- rather than the whole
    table.
    """
    return f"{key_prefix}.request_template{suffix}"


def _parse_string(text: str, key: str) -> list[_ParsedLeaf]:
    """Parse one template string and require every placeholder to be a known field.

    Raises:
        ConfigError: The string is not a well-formed placeholder string, or it
            names a field outside :data:`~qrcheck.config.models.TEMPLATE_FIELDS`.
            The schema rejects both at load time, so this is the belt-and-braces
            path for a record built without passing through the loader.
    """
    try:
        parsed = list(_FORMATTER.parse(text))
    except ValueError as exc:
        raise ConfigError(key, f"malformed placeholder string {text!r}: {exc}") from None
    for _literal, field_name, _format_spec, _conversion in parsed:
        if field_name is None:
            continue
        if field_name not in TEMPLATE_FIELDS:
            raise ConfigError(
                key,
                f"unknown placeholder '{field_name}' in request_template; "
                f"allowed: {sorted(TEMPLATE_FIELDS)}",
            )
    return parsed


def _parse_template(template: object, key_prefix: str) -> dict[str, Any]:
    """Return the request template as a mapping, from a table or from JSON text.

    The configuration contract types ``request_template`` as a table, which is
    what ``qrcheck.config.load`` builds.  JSON text is accepted as well, because
    the documented shape of a template is a JSON object and a caller that stores
    one as text means the same thing.

    Raises:
        ConfigError: The template is not a table, JSON text does not parse, or the
            parsed value is not a JSON object.  A malformed template is a
            configuration fault, so it leaves as exit 2 naming the key rather than
            as an internal error with a traceback.
    """
    key = f"{key_prefix}.request_template"
    body: object
    if isinstance(template, Mapping):
        body = dict(template)
    elif isinstance(template, (str, bytes, bytearray)):
        try:
            text = (
                template.decode("utf-8")
                if isinstance(template, (bytes, bytearray))
                else template
            )
        except UnicodeDecodeError as exc:
            raise ConfigError(key, f"invalid JSON: {exc}") from None
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(key, f"invalid JSON: {exc}") from None
    else:
        raise ConfigError(
            key,
            f"expected a table or a JSON object string; got {type(template).__name__}",
        )
    if not isinstance(body, Mapping):
        raise ConfigError(key, f"must be a JSON object; got {type(body).__name__}")
    return {str(name): value for name, value in body.items()}


def _template_placeholders(
    node: object,
    suffix: str,
    key_prefix: str,
    into: set[str],
) -> None:
    """Record every placeholder the template references, validating each one.

    The walk mirrors the schema's own: tables, arrays and strings carry
    placeholders, and every other value is a literal that passes through.
    """
    if isinstance(node, Mapping):
        for name, child in node.items():
            _template_placeholders(child, f"{suffix}.{name}", key_prefix, into)
        return
    if isinstance(node, (list, tuple)):
        for index, child in enumerate(node):
            _template_placeholders(child, f"{suffix}[{index}]", key_prefix, into)
        return
    if isinstance(node, str):
        for _literal, field_name, _spec, _conversion in _parse_string(
            node, _leaf_key(key_prefix, suffix)
        ):
            if field_name is not None:
                into.add(field_name)


def _template_values(request: GenerateRequest) -> dict[str, object]:
    """Return the value every placeholder stands for in `request`.

    ``None`` for ``payload_utf8``, ``size`` and ``margin`` is meaningful: it is how
    the omit-when-``None`` rule learns that the request leaves those values
    unspecified.  Every key of :data:`~qrcheck.config.models.TEMPLATE_FIELDS` is
    present, so a placeholder never resolves to a missing name.
    """
    return {
        "payload_b64": base64.b64encode(request.payload).decode("ascii"),
        "payload_utf8": _utf8_text(request.payload),
        "ec_level": request.ec_level,
        "format": request.output_format,
        "size": request.module_size_px,
        "margin": request.quiet_zone_modules,
        "fg": _hex_colour(request.foreground_rgb),
        "bg": _hex_colour(request.background_rgb),
    }


def _utf8_text(payload: bytes) -> str | None:
    """Return `payload` as text when it is valid UTF-8, else ``None``.

    A payload that is not valid UTF-8 has no faithful text form, so the template
    key carrying ``{payload_utf8}`` is omitted rather than filled with a lossy
    replacement string that the target would encode as different bytes.
    """
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _hex_colour(rgb: tuple[int, int, int]) -> str:
    """Return one colour as the ``#rrggbb`` text the ``{fg}``/``{bg}`` fields carry."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _render_string(
    text: str,
    values: Mapping[str, object],
    suffix: str,
    key_prefix: str,
) -> object:
    """Return one template string with its placeholders substituted.

    A whole-string single placeholder keeps the value's own type, an embedded
    placeholder renders as text, escaped braces become literal braces, and a
    placeholder whose value is ``None`` yields the omit sentinel so the containing
    key disappears from the request.
    """
    key = _leaf_key(key_prefix, suffix)
    parsed = _parse_string(text, key)
    names = [field_name for _literal, field_name, _spec, _conv in parsed if field_name]
    if not names:
        return _FORMATTER.vformat(text, (), {})
    if any(values[field_name] is None for field_name in names):
        return _OMIT
    literal, field_name, format_spec, conversion = parsed[0]
    if len(parsed) == 1 and not literal and not format_spec and conversion is None:
        return values[str(field_name)]
    return _FORMATTER.vformat(text, (), values)


def _render_node(
    node: object,
    values: Mapping[str, object],
    suffix: str,
    key_prefix: str,
) -> object:
    """Return the template `node` with every placeholder substituted.

    Tables and arrays are rebuilt rather than mutated, so the frozen configuration
    a request is rendered from is never touched, and a leaf that renders to the
    omit sentinel drops out of its parent.
    """
    if isinstance(node, Mapping):
        rendered: dict[str, object] = {}
        for name, child in node.items():
            value = _render_node(child, values, f"{suffix}.{name}", key_prefix)
            if value is _OMIT:
                continue
            rendered[str(name)] = value
        return rendered
    if isinstance(node, (list, tuple)):
        items: list[object] = []
        for index, child in enumerate(node):
            value = _render_node(child, values, f"{suffix}[{index}]", key_prefix)
            if value is _OMIT:
                continue
            items.append(value)
        return items
    if isinstance(node, str):
        return _render_string(node, values, suffix, key_prefix)
    return node


def _query_parameters(body: Mapping[str, Any], key_prefix: str) -> dict[str, Any]:
    """Return the rendered template as query parameters for a ``GET`` request.

    Raises:
        TargetContractError: A parameter is a table or an array.  A query string
            carries scalars only, so a structured template has to be sent with a
            method that has a body; the message names the template path.
    """
    params: dict[str, Any] = {}
    for name, value in body.items():
        if isinstance(value, (Mapping, list, tuple)):
            raise TargetContractError(
                f"{key_prefix}.request_template.{name} holds a "
                f"{type(value).__name__}, but a GET request carries the template as "
                "query parameters, which must be scalars; use a method with a body "
                "for a structured request"
            )
        params[str(name)] = value
    return params


def _json_body(body: Mapping[str, Any], key_prefix: str) -> bytes:
    """Return the rendered template as a canonical JSON request body.

    Keys are sorted so a request's bytes are a function of the configuration
    alone, which keeps two runs' traffic comparable.

    Raises:
        ConfigError: The template holds a value JSON cannot carry.  The schema
            rejects one at load time, so this is the belt-and-braces path, and it
            is a configuration fault rather than a target one.
    """
    try:
        return json.dumps(body, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{key_prefix}.request_template",
            f"cannot be sent as JSON: {exc}",
        ) from None


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def _body_excerpt(body: bytes, limit: int = _STATUS_BODY_EXCERPT_BYTES) -> str:
    """Return a short, printable rendering of an unusable response body.

    UTF-8 text is quoted as text because that is what a JSON error envelope looks
    like; anything else is quoted as hexadecimal so a binary body is still
    identifiable.  An empty body yields an empty string and no clause at all.
    """
    if not body:
        return ""
    window = body[:limit]
    try:
        return repr(window.decode("utf-8"))
    except UnicodeDecodeError:
        return f"0x{hex_prefix(window)}"


def _parse_response_json(body: bytes, key_prefix: str, target: str) -> object:
    """Parse a response body as JSON, naming the target and the key on failure.

    Raises:
        TargetContractError: The body is not UTF-8, or not JSON, although
            ``image_field`` names a field inside it.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TargetContractError(
            f"target {target!r} answered with a body that is not UTF-8 JSON ({exc}) "
            f"although {key_prefix}.image_field names a field inside it; it starts "
            f"with 0x{hex_prefix(body, _JSON_BODY_EXCERPT_BYTES)}"
        ) from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetContractError(
            f"target {target!r} answered with a body that is not JSON ({exc}) although "
            f"{key_prefix}.image_field names a field inside it; it starts with "
            f"0x{hex_prefix(body, _JSON_BODY_EXCERPT_BYTES)}"
        ) from None


def _base64_text(value: object, field: str, key_prefix: str) -> bytes:
    """Decode the base64 text carried by one response field.

    Raises:
        TargetContractError: The field is not base64 text, or holds text that does
            not decode.  ``validate=True`` is deliberate: a stray character is
            reported rather than silently skipped.
    """
    if isinstance(value, str):
        encoded = value.strip()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        try:
            encoded = bytes(value).decode("ascii").strip()
        except UnicodeDecodeError:
            raise TargetContractError(
                f"{key_prefix}.image_encoding is 'base64' but "
                f"{key_prefix}.image_field '{field}' holds bytes that are not ASCII "
                f"base64; they start with 0x{hex_prefix(bytes(value), _JSON_BODY_EXCERPT_BYTES)}"
            ) from None
    else:
        raise TargetContractError(
            f"{key_prefix}.image_encoding is 'base64' but {key_prefix}.image_field "
            f"'{field}' holds a {type(value).__name__}, not base64 text"
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TargetContractError(
            f"{key_prefix}.image_encoding is 'base64' but {key_prefix}.image_field "
            f"'{field}' is not valid base64 ({exc})"
        ) from None


def _raw_field_bytes(value: object, field: str, key_prefix: str) -> bytes:
    """Return one response field's bytes under the ``raw`` encoding.

    Raises:
        TargetContractError: The field is neither text nor bytes: a raw image
            cannot be recovered from a number, a table or an array.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TargetContractError(
        f"{key_prefix}.image_encoding is 'raw' but {key_prefix}.image_field '{field}' "
        f"holds a {type(value).__name__}, not text or bytes"
    )


def _required_capabilities(request: GenerateRequest) -> frozenset[Capability]:
    """Return the capabilities `request` needs a target to have declared.

    Only what the request actually asks for is required: its output format, plus a
    geometry or colour operation only when that value is pinned or non-default.  A
    request that leaves module size and quiet zone unspecified asks for nothing
    there, so a target that cannot be told them is still usable.

    Raises:
        ValueError: ``output_format`` is neither ``png`` nor ``svg``.  The request
            builder refuses that, so it is a framework defect and not an adapter
            error; it is raised before any I/O.
    """
    format_capability = _FORMAT_CAPABILITIES.get(request.output_format)
    if format_capability is None:
        raise ValueError(
            f"output_format must be one of {', '.join(sorted(_FORMAT_CAPABILITIES))}; "
            f"got {request.output_format!r}"
        )
    needed = {format_capability}
    if request.module_size_px is not None:
        needed.add(Capability.module_size)
    if request.quiet_zone_modules is not None:
        needed.add(Capability.quiet_zone)
    if (
        request.foreground_rgb != _DEFAULT_FOREGROUND_RGB
        or request.background_rgb != _DEFAULT_BACKGROUND_RGB
    ):
        needed.add(Capability.color)
    return frozenset(needed)


class HttpAdapter:
    """Drive a loopback HTTP QR generator through the adapter port.

    One instance is one ``[targets.<name>]`` table: the URL, the method, the
    request template, the path to the image in the response, the way that image is
    carried and any metadata paths all come from the configuration, so the same
    class drives any vendor and this module names none of them.  The module
    docstring is the complete contract; what the class adds is the lifecycle.

    The client is created on the first :meth:`generate` and reused, so a run opens
    one set of connections instead of one per symbol, and :meth:`close` releases it.
    An adapter that is closed and then used again creates a fresh client, because
    the runner closes every adapter once a run ends and a closed adapter should be
    released rather than poisoned.

    Constructor faults split by who has to fix them: a configuration fault the
    schema cannot decide raises :class:`~qrcheck.config.schema.ConfigError` (exit 2,
    key named) and a broken loopback invariant raises :class:`ValueError` (a
    framework defect, never a user diagnostic).  Nothing is opened, resolved or
    dialled during construction.
    """

    def __init__(self, config: HTTPTarget, *, name: str | None = None) -> None:
        """Prepare the adapter for `config` without any I/O.

        Args:
            config: The resolved ``[targets.<name>]`` table this adapter drives.
            name: The name reports should use for this target.  It defaults to the
                target's ``type``, and a caller that knows the configured name --
                the adapter factory does -- passes it so every diagnostic quotes the
                name the configuration declares.

        Raises:
            ConfigError: The request template is not a table or a JSON object, does
                not parse, names a placeholder outside
                :data:`~qrcheck.config.models.TEMPLATE_FIELDS`, carries no payload
                placeholder, or the ``metadata_map`` names a framework field outside
                ``version``, ``ec``, ``mask`` and ``mode``.
            ValueError: ``url`` is not an ``http``/``https`` URL on a loopback host.
                The configuration schema already rejects that with exit 2 naming the
                host, so reaching this means a record was built without it.
        """
        self._config = config
        self.name = config.type if name is None else name
        self.capabilities = declared_capabilities(config)
        self._key_prefix = f"targets.{config.name}"
        self._template = _parse_template(config.request_template, self._key_prefix)
        referenced: set[str] = set()
        _template_placeholders(self._template, "", self._key_prefix, referenced)
        if not referenced & _PAYLOAD_PLACEHOLDERS:
            raise ConfigError(
                f"{self._key_prefix}.request_template",
                "carries no payload placeholder, so every request it builds would omit "
                f"the payload; reference one of {sorted(_PAYLOAD_PLACEHOLDERS)}",
            )
        self._check_loopback(config.url)
        self._check_metadata_map(config.metadata_map)
        self._payload_placeholders = frozenset(referenced & _PAYLOAD_PLACEHOLDERS)
        self._client: httpx.Client | None = None
        # Serialises the first request and close() against each other.  The runner
        # is sequential, so this only has to stop two racing callers from each
        # creating a client and leaking one of them.
        self._client_lock = threading.Lock()

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Produce one symbol for `request`, or raise exactly one adapter error.

        The capability gate, the inexpressible-field refusal and the template
        rendering all happen before the client is touched, so a request this
        adapter cannot honour never opens a connection and never consumes the
        configured timeout.

        Raises:
            TargetUnsupported: The request needs a capability the target did not
                declare, or pins a version, a mask or a logo overlay the request
                template has no placeholder for.  Raised before any I/O.
            TargetUnavailable: The connection was refused, the address was
                unusable, or connecting timed out.
            TargetTimeout: No response arrived within ``timeout_s``.
            TargetContractError: A response arrived but was unusable -- a non-2xx
                status, an empty body, a body that is not the JSON the configured
                ``image_field`` needs, a field the configured path does not find,
                undecodable base64, a missing metadata path, or bytes that are not
                the requested artefact.
            ConfigError: The template cannot be rendered or sent as configured.
            ValueError: A framework invariant was broken before any I/O (an
                unbuildable request, an unknown ``image_encoding``).  The request
                builder and the configuration schema prevent both, so this is a
                framework defect rather than a diagnostic for the user.
        """
        require_capabilities(self, _required_capabilities(request))
        self._reject_inexpressible(request)
        body = self._render_body(request)
        params: dict[str, Any] | None = None
        content: bytes | None = None
        headers: dict[str, str] | None = None
        if self._config.method == _GET_METHOD:
            params = _query_parameters(body, self._key_prefix)
        else:
            content = _json_body(body, self._key_prefix)
            headers = {_CONTENT_TYPE_HEADER: _JSON_CONTENT_TYPE}
        try:
            client = self._client_or_create()
            started = time.perf_counter()
            response = client.request(
                self._config.method,
                self._config.url,
                params=params,
                content=content,
                headers=headers,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return self._interpret_response(response, request, elapsed_ms)
        except (json.JSONDecodeError, binascii.Error, UnicodeDecodeError, LookupError) as exc:
            raise TargetContractError(
                f"target {self.name!r} answered {self._config.method} {self._config.url} "
                f"with a body the configured fields do not describe "
                f"({type(exc).__name__}: {exc})"
            ) from None
        except (TargetError, ConfigError, ValueError):
            # Already in the adapter's vocabulary, or a configuration fault, or a
            # framework defect.  Re-wrapping any of them would mislabel it; the
            # body-decoding clause above precedes it because JSONDecodeError and
            # binascii.Error are both ValueErrors.
            raise
        except httpx.ConnectTimeout as exc:
            # A connect timeout means the endpoint never answered, which is the
            # "cannot be reached" trigger rather than a slow response.
            raise TargetUnavailable(
                f"target {self.name!r} accepted no connection to {self._config.url} "
                f"within {self._config.timeout_s} s ({type(exc).__name__}: {exc})"
            ) from None
        except httpx.TimeoutException as exc:
            raise TargetTimeout(
                f"target {self.name!r} gave no response to {self._config.method} "
                f"{self._config.url} within {self._config.timeout_s} s "
                f"({type(exc).__name__})"
            ) from None
        except (httpx.UnsupportedProtocol, httpx.InvalidURL, httpx.NetworkError) as exc:
            # NetworkError covers ConnectError and the read/write/close errors; an
            # unusable address arrives as InvalidURL, which is not an HTTPError.
            raise TargetUnavailable(
                f"target {self.name!r} could not be reached at {self._config.url} "
                f"({type(exc).__name__}: {exc})"
            ) from None
        except httpx.HTTPError as exc:
            raise TargetContractError(
                f"target {self.name!r} failed {self._config.method} {self._config.url} "
                f"({type(exc).__name__}: {exc})"
            ) from None
        except OSError as exc:
            raise TargetContractError(
                f"target {self.name!r} failed {self._config.method} {self._config.url} "
                f"({type(exc).__name__}: {exc})"
            ) from None
        except Exception as exc:
            raise TargetContractError(
                f"target {self.name!r} produced an unusable response to "
                f"{self._config.method} {self._config.url} "
                f"({type(exc).__name__}: {exc})"
            ) from None

    def close(self) -> None:
        """Close the HTTP client, and do nothing at all on a second call.

        The runner closes every adapter once a run ends, including after a failure,
        so closing must not raise for the ordinary reasons; the client reference is
        dropped first, so a concurrent caller cannot observe a client that is being
        closed.
        """
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def _client_or_create(self) -> httpx.Client:
        """Return the shared client, creating it on first use.

        ``follow_redirects=False`` keeps a redirect from carrying the request to a
        host the product forbids, and ``trust_env=False`` keeps a ``HTTP_PROXY`` or
        ``HTTPS_PROXY`` environment variable from routing a loopback request through
        a remote host.  ``httpx.Timeout(timeout_s)`` maps the configured per-target
        ceiling onto every phase of the request, which is why httpx is the client.
        """
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(self._config.timeout_s),
                    follow_redirects=False,
                    trust_env=False,
                )
            return self._client

    def _check_loopback(self, url: str) -> None:
        """Require `url` to be an ``http``/``https`` URL on a loopback host.

        The authoritative check is the configuration schema's, which exits 2 naming
        the host.  This is belt-and-braces for a caller that builds an ``HTTPTarget``
        without the loader, so it raises :class:`ValueError` -- a framework defect --
        and deliberately no user-facing diagnostic, so that no second exit-2 path
        exists for the same mistake.

        Raises:
            ValueError: The URL does not parse, its scheme is not ``http``/``https``,
                its host is not ``127.0.0.1``, ``localhost`` or ``::1``, or it
                carries a username or password.
        """
        try:
            parts = urllib.parse.urlsplit(url)
        except ValueError as exc:
            raise ValueError(f"http target url {url!r} is not parseable: {exc}") from None
        if parts.scheme not in _LOOPBACK_SCHEMES:
            raise ValueError(
                f"http target url {url!r} has scheme {parts.scheme!r}; the framework is "
                f"offline and only {sorted(_LOOPBACK_SCHEMES)} are reachable"
            )
        host = parts.hostname
        if host is None or host.lower() not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"http target url {url!r} names host {host!r}; the framework is offline "
                f"and only {sorted(_LOOPBACK_HOSTS)} are reachable"
            )
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                f"http target url {url!r} carries a username or password; the "
                "configuration holds no credentials, and there is nothing to "
                "authenticate to on loopback"
            )

    def _check_metadata_map(self, metadata_map: Mapping[str, str]) -> None:
        """Require every ``metadata_map`` key to be one of the framework's four fields.

        The schema checks the map's shape but not its keys, so without this the
        first response would fail with a bare ``ValueError`` from the metadata
        mapper and leave as an internal error.  A wrong framework field is a
        configuration fault and has to read as one.

        Raises:
            ConfigError: A key is not one of ``version``, ``ec``, ``mask`` and
                ``mode``; the message names every offending key.
        """
        unknown = sorted(set(metadata_map) - set(METADATA_FIELDS))
        if not unknown:
            return
        names = ", ".join(repr(name) for name in unknown)
        raise ConfigError(
            f"{self._key_prefix}.metadata_map",
            f"names unknown framework field(s) {names}; expected any of "
            f"{', '.join(METADATA_FIELDS)}",
        )

    def _reject_inexpressible(self, request: GenerateRequest) -> None:
        """Refuse a request the configured template cannot convey.

        ``version``, ``mask`` and ``logo_overlay_path`` have no placeholder in
        :data:`~qrcheck.config.models.TEMPLATE_FIELDS`, so no request this adapter
        sends can carry them.  Sending the request anyway would leave the target
        guessing and report its guess as a conformance failure, so the pin is
        refused instead and the affected check reports a skip.  A template whose
        only payload placeholder is ``payload_utf8`` cannot carry a payload that is
        not valid UTF-8 for the same reason: that key is omitted, and the request
        would arrive with no payload at all.

        Raises:
            TargetUnsupported: A pinned or overlaid request coincides with a
                capability the adapter did not declare, or with one it declared but
                cannot honour over this transport; or the payload cannot be encoded
                into the template.  Raised before any I/O.
        """
        if "payload_b64" not in self._payload_placeholders:
            if _utf8_text(request.payload) is None:
                raise TargetUnsupported(
                    f"adapter {self.name!r} cannot carry a payload that is not valid "
                    "UTF-8: its request template references only 'payload_utf8', which "
                    "is omitted for such a payload"
                )
        for field_name, capability in _INEXPRESSIBLE_FIELDS:
            if getattr(request, field_name) is None:
                continue
            require_capabilities(self, capability)
            raise TargetUnsupported(
                f"adapter {self.name!r} cannot express {field_name} over its request "
                f"template: no placeholder in {sorted(TEMPLATE_FIELDS)} carries it"
            )

    def _render_body(self, request: GenerateRequest) -> dict[str, Any]:
        """Return the request mapping the configured template produces.

        Raises:
            ConfigError: The template rendered to something other than a table.  The
                constructor already refuses a non-table template, so this converts a
                future change into a configuration fault rather than a crash.
        """
        rendered = _render_node(
            self._template, _template_values(request), "", self._key_prefix
        )
        if not isinstance(rendered, Mapping):
            raise ConfigError(
                f"{self._key_prefix}.request_template",
                f"must render to a JSON object; got {type(rendered).__name__}",
            )
        return dict(rendered)

    def _interpret_response(
        self,
        response: httpx.Response,
        request: GenerateRequest,
        elapsed_ms: float,
    ) -> GeneratedSymbol:
        """Turn one HTTP response into a :class:`GeneratedSymbol`.

        Raises:
            TargetContractError: The status is outside 2xx, the body is unusable, a
                configured path is absent from it, or the artefact is not the format
                the request asked for.
        """
        status = response.status_code
        if status < 200 or status >= 300:
            raise TargetContractError(self._status_detail(response))
        image_bytes, parsed = self._carried_image(response.content, request)
        metadata = self._declared_metadata(parsed)
        return GeneratedSymbol(
            image_bytes=image_bytes,
            output_format=request.output_format,
            declared_version=metadata.get("version"),
            declared_ec=metadata.get("ec"),
            declared_mask=metadata.get("mask"),
            declared_mode=metadata.get("mode"),
            elapsed_ms=elapsed_ms,
        )

    def _carried_image(
        self,
        body: bytes,
        request: GenerateRequest,
    ) -> tuple[bytes, object | None]:
        """Return the artefact bytes the response carries, plus its parsed JSON.

        The parsed JSON is ``None`` when the body itself was the image, which is
        also why such a target declares no metadata: there is no envelope left to
        read it from.

        Raises:
            TargetContractError: The body is empty, is not the JSON that
                ``image_field`` needs, does not carry the named path, does not carry
                the image in the declared encoding, or does not carry the requested
                artefact format.
        """
        field = self._config.image_field
        if not body:
            raise TargetContractError(
                f"target {self.name!r} answered {self._config.method} "
                f"{self._config.url} with an empty body"
            )
        if field == _BODY_SENTINEL:
            artefact = self._carried_bytes(body, field)
            return ensure_image_bytes(artefact, request.output_format), None
        parsed = _parse_response_json(body, self._key_prefix, self.name)
        try:
            value = resolve_dotted_path(parsed, field)
        except KeyError as exc:
            detail = exc.args[0] if exc.args else str(exc)
            raise TargetContractError(
                f"{self._key_prefix}.image_field names '{field}', which the response "
                f"from target {self.name!r} does not provide: {detail}"
            ) from None
        artefact = self._carried_bytes(value, field)
        return ensure_image_bytes(artefact, request.output_format), parsed

    def _carried_bytes(self, value: object, field: str) -> bytes:
        """Return the image bytes `value` carries under ``image_encoding``.

        Raises:
            TargetContractError: The value does not match the declared encoding.
            ValueError: ``image_encoding`` is neither ``base64`` nor ``raw``.  The
                schema rejects that value at load time, so it is a framework defect.
        """
        encoding = self._config.image_encoding
        if encoding == _BASE64_ENCODING:
            return _base64_text(value, field, self._key_prefix)
        if encoding == _RAW_ENCODING:
            return _raw_field_bytes(value, field, self._key_prefix)
        raise ValueError(
            f"image_encoding must be one of {', '.join(IMAGE_ENCODINGS)}; got {encoding!r}"
        )

    def _declared_metadata(self, parsed: object | None) -> dict[str, Any]:
        """Return the metadata the target declared, through the configured map.

        Raises:
            TargetContractError: A configured path is absent from the response or
                holds a value of the wrong type, or metadata was configured but the
                response is not a JSON object to read it from.
        """
        metadata_map = self._config.metadata_map
        if not metadata_map or parsed is None:
            return {}
        if not isinstance(parsed, Mapping):
            raise TargetContractError(
                f"{self._key_prefix}.metadata_map names metadata in the response, but "
                f"the response from target {self.name!r} is a "
                f"{type(parsed).__name__} rather than a JSON object"
            )
        return apply_metadata_map(parsed, metadata_map)

    def _status_detail(self, response: httpx.Response) -> str:
        """Describe an unusable status in one diagnostic naming what arrived."""
        status = response.status_code
        detail = (
            f"target {self.name!r} answered HTTP {status} to {self._config.method} "
            f"{self._config.url}"
        )
        if 300 <= status < 400:
            location = response.headers.get(_LOCATION_HEADER)
            detail += (
                f"; a redirect to {location!r}, which the adapter does not follow"
                if location
                else "; a redirect without a Location header, which the adapter does not follow"
            )
        excerpt = _body_excerpt(response.content)
        if excerpt:
            detail += f"; body starts {excerpt}"
        return (
            detail
            + "; the framework treats any non-2xx status as a contract error, because "
            "an http target has no declared-rejection channel"
        )

