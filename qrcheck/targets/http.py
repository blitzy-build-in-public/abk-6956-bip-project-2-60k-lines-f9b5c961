"""The loopback-only ``http`` adapter: every request and response shape comes from configuration.

:class:`HttpAdapter` drives a QR generator reachable over loopback HTTP, and nothing here knows a
vendor's endpoint, field name or response key: they all come from the ``[targets.<name>]`` table.

``request_template`` substitutes ``payload_b64``, ``payload_utf8``, ``ec_level``, ``format``,
``size``, ``margin``, ``fg`` and ``bg``; a whole-placeholder leaf keeps the value's own type, and a
placeholder whose request value is ``None`` omits its containing key -- the only way to say
"unspecified", including ``{payload_utf8}`` meeting a non-UTF-8 payload, refused with
``TargetUnsupported`` when it is the only payload channel. ``GET`` sends the template as query
parameters (scalars only), passed as ``params=`` so a base64 payload survives; other methods send a
canonical JSON body. ``image_field`` names the image (``_body`` for the body itself), and
``metadata_map`` projects the four framework fields onto ``GeneratedSymbol.declared_*`` as evidence.

Loopback is enforced at the transport layer: the client uses ``follow_redirects=False`` and
``trust_env=False``, so neither a redirect nor a proxy variable can route a request off loopback. A
non-2xx status is always ``TargetContractError``: an HTTP target has no declared-rejection channel.
The body is read once, streamed, and stops past :data:`_MAX_RESPONSE_BODY_BYTES`, so a misbehaving
target cannot exhaust this process' memory; the ``httpx``, ``OSError`` and decoding families become
adapter errors, and anything else propagates as a framework defect.
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

from qrcheck.config import models
from qrcheck.config.models import IMAGE_ENCODINGS, TEMPLATE_FIELDS, HTTPTarget
from qrcheck.config.schema import ConfigError
from qrcheck.targets.base import (
    HEX_PREFIX_LIMIT,
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

# Protocol vocabulary: the URL schemes, the method comparison, the JSON content type, the
# Location header, the ``_body`` sentinel, the two ``image_encoding`` values, the placeholder
# names and the framework's own field names. Every vendor-specific name arrives from configuration.

_GET_METHOD: Final[str] = "GET"
_JSON_CONTENT_TYPE: Final[str] = "application/json"
_CONTENT_TYPE_HEADER: Final[str] = "content-type"
_LOCATION_HEADER: Final[str] = "location"
_BODY_SENTINEL: Final[str] = "_body"
_BASE64_ENCODING: Final[str] = "base64"
_RAW_ENCODING: Final[str] = "raw"

# The URL schemes the transport may use. ``qrcheck.config.schema`` is the authoritative gate;
# this re-checks the same rule so a directly constructed record cannot bypass it.
_LOOPBACK_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# The request format's capability member: a format request names exactly that capability.
_FORMAT_CAPABILITIES: Final[Mapping[str, Capability]] = MappingProxyType(
    {"png": Capability.png, "svg": Capability.svg}
)

# The framework's default colour pair: honouring "black on white" asks nothing of the target.
_DEFAULT_FOREGROUND_RGB: Final[tuple[int, int, int]] = (0, 0, 0)
_DEFAULT_BACKGROUND_RGB: Final[tuple[int, int, int]] = (255, 255, 255)

# The placeholders that carry the payload; a template referencing neither is refused at
# construction.
_PAYLOAD_PLACEHOLDERS: Final[frozenset[str]] = frozenset({"payload_b64", "payload_utf8"})

# Request fields the template cannot convey, paired with the capability that governs each: a
# request pinning one is refused before any I/O rather than sent without it.
_INEXPRESSIBLE_FIELDS: Final[tuple[tuple[str, Capability], ...]] = (
    ("version", Capability.version_pin),
    ("mask", Capability.mask_pin),
    ("logo_overlay_path", Capability.logo_overlay),
)

# How much of an unusable response body a diagnostic quotes.
_STATUS_BODY_EXCERPT_BYTES: Final[int] = 500
_JSON_BODY_EXCERPT_BYTES: Final[int] = 64

# The bound on a response body. A loopback target is the code under test and is untrusted, so
# an unbounded read would let a misbehaving one exhaust this process' memory. No legitimate
# artefact comes near 16 MiB -- the extreme 1640x1640 px geometry is about 1.6 kB and a Version
# 40 symbol there about 21 kB, because a PNG of flat modules compresses -- so the bound refuses
# a runaway response without refusing a real one.
_MAX_RESPONSE_BODY_BYTES: Final[int] = 16 * 1024 * 1024

# One formatter for the whole module: ``string.Formatter`` is stateless and a single instance keeps
_FORMATTER: Final[string.Formatter] = string.Formatter()

# Sentinel meaning "the template key holding this value must be dropped", carried out of a
_OMIT: Final[object] = object()

# One parsed template leaf: literal text, field name, format specification, conversion.
_ParsedLeaf = tuple[str, str | None, str, str | None]


# Template handling


def _leaf_key(key_prefix: str, suffix: str) -> str:
    """Return the dotted configuration key of one template leaf.

    The key is what a :class:`ConfigError` reports, so it names the exact leaf to edit rather than
    the whole table.
    """
    return f"{key_prefix}.request_template{suffix}"


def _parse_string(text: str, key: str) -> list[_ParsedLeaf]:
    """Parse one template string and require every placeholder to be a known field.

    Raises:
        ConfigError: The string is not well formed, or names a field outside
            :data:`~qrcheck.config.models.TEMPLATE_FIELDS` -- the belt-and-braces path for a record
            built without passing through the loader.
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

    The contract accepts the template as the JSON object it describes, written as a table or as a
    string holding that object's JSON text.

    Raises:
        ConfigError: The template is not a table, its JSON text does not parse, or the parsed value
            is not a JSON object -- so it leaves as exit 2 naming the key.
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

    The walk mirrors the schema's own: tables, arrays and strings carry placeholders, and every
    other value is a literal that passes through.
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

    ``None`` for ``payload_utf8``, ``size`` and ``margin`` is how the omit-when-``None`` rule learns
    those values are unspecified; every key of ``TEMPLATE_FIELDS`` is present, so a placeholder
    never resolves to a missing name.
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

    A payload that is not valid UTF-8 has no faithful text form, so the key carrying
    ``{payload_utf8}`` is omitted rather than filled with a lossy replacement.
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

    A whole-string single placeholder keeps the value's own type, an embedded placeholder renders as
    text, escaped braces become literal braces, and a placeholder whose value is ``None`` yields the
    omit sentinel so the containing key disappears.
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

    Tables and arrays are rebuilt rather than mutated, so the frozen configuration a request is
    rendered from is never touched, and a leaf that renders to the sentinel drops out.
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
        TargetContractError: A parameter is a table or an array. A query string carries scalars
            only; the message names the template path.
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

    Keys are sorted so a request's bytes are a function of the configuration alone.

    Raises:
        ConfigError: The template holds a value JSON cannot carry -- a configuration fault.
    """
    try:
        return json.dumps(body, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{key_prefix}.request_template",
            f"cannot be sent as JSON: {exc}",
        ) from None


# Response handling


def _body_excerpt(body: bytes, limit: int = _STATUS_BODY_EXCERPT_BYTES) -> str:
    """Return a short, printable rendering of an unusable response body.

    UTF-8 text is quoted as text because that is what a JSON error envelope looks like; anything
    else is quoted as hexadecimal. An empty body yields no clause at all.
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

    The refused body travels on ``body_prefix``, so a report records the bytes that were not JSON.

    Raises:
        TargetContractError: The body is not UTF-8, or not JSON, although ``image_field`` names a
            field inside it.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TargetContractError(
            f"target {target!r} answered with a body that is not UTF-8 JSON ({exc}) "
            f"although {key_prefix}.image_field names a field inside it; it starts "
            f"with 0x{hex_prefix(body, _JSON_BODY_EXCERPT_BYTES)}",
            body_prefix=body,
        ) from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetContractError(
            f"target {target!r} answered with a body that is not JSON ({exc}) although "
            f"{key_prefix}.image_field names a field inside it; it starts with "
            f"0x{hex_prefix(body, _JSON_BODY_EXCERPT_BYTES)}",
            body_prefix=body,
        ) from None


def _base64_text(value: object, field: str, key_prefix: str) -> bytes:
    """Decode the base64 text carried by one response field.

    Raises:
        TargetContractError: The field is not base64 text, or holds text that does not decode;
            ``validate=True`` reports a stray character rather than skipping it.
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
        TargetContractError: The field is neither text nor bytes, so no image can be recovered from
            it.
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

    Only what the request asks for is required: its output format, plus a geometry or colour
    operation only when that value is pinned or non-default.

    Raises:
        ValueError: ``output_format`` is neither ``png`` nor ``svg`` -- a framework defect.
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

    One instance is one ``[targets.<name>]`` table, so the same class drives any vendor; the module
    docstring is the complete contract. The client is created on the first :meth:`generate` and
    reused, so a run opens one set of connections, and :meth:`close` releases it; a closed adapter
    used again creates a fresh client. A configuration fault the schema cannot decide raises
    :class:`~qrcheck.config.schema.ConfigError` and a broken loopback invariant raises
    ``ValueError``.
    """

    def __init__(self, config: HTTPTarget, *, name: str | None = None) -> None:
        """Prepare the adapter for `config` without any I/O.

        Raises:
            ConfigError: The template is malformed, names an unknown placeholder, carries no payload
                placeholder, or ``metadata_map`` names an unknown framework field.
            ValueError: ``url`` is not an ``http``/``https`` URL on a loopback host.
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
        # Serialises the first request and close() against each other, so two racing callers cannot
        # each
        # create a client and leak one.
        self._client_lock = threading.Lock()

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Produce one symbol for `request`, or raise exactly one adapter error.

        The capability gate, the inexpressible-field refusal and the template rendering all happen
        before the client is touched, so an unhonourable request never opens a connection.

        Raises:
            TargetUnsupported: An undeclared capability, or a pin or overlay no placeholder carries.
            TargetUnavailable: The connection was refused or the address was unusable.
            TargetTimeout: No response arrived within ``timeout_s``.
            TargetContractError: A response arrived but was unusable -- a non-2xx status, an empty
                or unparseable body, an absent field, undecodable base64, or a body past the read
                bound.
            ConfigError: The template cannot be rendered or sent as configured.
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
            with client.stream(
                self._config.method,
                self._config.url,
                params=params,
                content=content,
                headers=headers,
            ) as response:
                response_body = self._read_body(response)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return self._interpret_response(response, response_body, request, elapsed_ms)
        except (json.JSONDecodeError, binascii.Error, UnicodeDecodeError) as exc:
            raise TargetContractError(
                f"target {self.name!r} answered {self._config.method} {self._config.url} "
                f"with a body the configured fields do not describe "
                f"({type(exc).__name__}: {exc})"
            ) from None
        except (TargetError, ConfigError, ValueError):
            # Already in the vocabulary, a configuration fault or a framework defect: re-wrapping
            # would mislabel it. The decoding clause above precedes this one because
            # ``json.JSONDecodeError`` and ``binascii.Error`` are both ``ValueError`` subclasses.
            # A ``KeyError`` or ``IndexError`` raised by framework code is deliberately left
            # uncaught: it is a framework defect, so it must reach the runner as an ``error``
            # result with its traceback rather than become a verdict about the target.
            raise
        except httpx.ConnectTimeout as exc:
            # A connect timeout means the endpoint never answered: unreachable, not slow.
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
            # NetworkError covers ConnectError and the read/write/close errors; an unusable address
            # arrives
            # as InvalidURL, which is not an HTTPError.
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

    def close(self) -> None:
        """Close the HTTP client, and do nothing at all on a second call.

        The client reference is dropped first, so a concurrent caller cannot observe a client that
        is being closed.
        """
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def _client_or_create(self) -> httpx.Client:
        """Return the shared client, creating it on first use.

        ``follow_redirects=False`` keeps a redirect from carrying the request to a host the product
        forbids, ``trust_env=False`` keeps a proxy variable from routing a loopback request through
        a remote host, and ``httpx.Timeout(timeout_s)`` maps the configured ceiling onto every phase
        of the request.
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

        Belt-and-braces for a caller that builds an ``HTTPTarget`` without the loader; the schema's
        own check exits 2 naming the host.

        Raises:
            ValueError: The URL does not parse, its scheme or host is not loopback, or it carries
                credentials.
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
        if host is None or host.lower() not in models.LOOPBACK_HOSTS:
            raise ValueError(
                f"http target url {url!r} names host {host!r}; the framework is offline "
                f"and only {sorted(models.LOOPBACK_HOSTS)} are reachable"
            )
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                f"http target url {url!r} carries a username or password; the "
                "configuration holds no credentials, and there is nothing to "
                "authenticate to on loopback"
            )

    def _check_metadata_map(self, metadata_map: Mapping[str, str]) -> None:
        """Require every ``metadata_map`` key to be one of the four framework fields.

        Raises:
            ConfigError: A key is outside ``version``, ``ec``, ``mask`` and ``mode``; the message
                names every offending key. The schema's own check is authoritative, and this path
                spares a hand-built record a bare ``ValueError`` from the metadata mapper.
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

        ``version``, ``mask`` and ``logo_overlay_path`` have no placeholder in ``TEMPLATE_FIELDS``,
        and a template whose only payload placeholder is ``payload_utf8`` cannot carry a non-UTF-8
        payload.

        Raises:
            TargetUnsupported: Such a request. Raised before any I/O.
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
            ConfigError: The template rendered to something other than a table, which turns a future
                change into a configuration fault rather than a crash.
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

    def _read_body(self, response: httpx.Response) -> bytes:
        """Read a streamed response body, refusing one that passes the bound.

        The body is never materialised in one piece, so an oversized response costs this process at
        most the bytes up to the bound.

        Raises:
            TargetContractError: The body grew past the bound; the diagnostic names the bound, the
                bytes received, the target, the method and the URL.
        """
        body = bytearray()
        received = 0
        for chunk in response.iter_bytes():
            received += len(chunk)
            if received > _MAX_RESPONSE_BODY_BYTES:
                raise TargetContractError(
                    f"target {self.name!r} answered {self._config.method} "
                    f"{self._config.url} with a body larger than the "
                    f"{_MAX_RESPONSE_BODY_BYTES} bytes this adapter reads from one "
                    f"response: {received} bytes had arrived when the read was "
                    "stopped.  A loopback target is untrusted, so the framework "
                    "bounds the body rather than let a misbehaving one exhaust the "
                    "process",
                    body_prefix=bytes(body[:HEX_PREFIX_LIMIT]),
                )
            body += chunk
        return bytes(body)

    def _interpret_response(
        self,
        response: httpx.Response,
        body: bytes,
        request: GenerateRequest,
        elapsed_ms: float,
    ) -> GeneratedSymbol:
        """Turn one HTTP response and its already-read `body` into a symbol.

        `body` is what :meth:`_read_body` handed back, so the response is never asked for its
        content.

        Raises:
            TargetContractError: The status is outside 2xx, the body is unusable, a configured path
                is absent, or the artefact is not the requested format.
        """
        status = response.status_code
        if status < 200 or status >= 300:
            raise TargetContractError(self._status_detail(response, body), body_prefix=body)
        image_bytes, parsed = self._carried_image(body, request)
        metadata = self._declared_metadata(parsed, body)
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

        The parsed JSON is ``None`` when the body itself was the image, which is why such a target
        declares no metadata.

        Raises:
            TargetContractError: The body is empty, is not the JSON ``image_field`` needs, does not
                carry the named path, or is not the requested artefact format.
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
                f"from target {self.name!r} does not provide: {detail}",
                body_prefix=body,
            ) from None
        artefact = self._carried_bytes(value, field)
        return ensure_image_bytes(artefact, request.output_format), parsed

    def _carried_bytes(self, value: object, field: str) -> bytes:
        """Return the image bytes `value` carries under ``image_encoding``.

        Raises:
            TargetContractError: The value does not match the declared encoding.
            ValueError: ``image_encoding`` is neither ``base64`` nor ``raw`` -- a framework defect,
                since the schema rejects that value at load.
        """
        encoding = self._config.image_encoding
        if encoding == _BASE64_ENCODING:
            return _base64_text(value, field, self._key_prefix)
        if encoding == _RAW_ENCODING:
            return _raw_field_bytes(value, field, self._key_prefix)
        raise ValueError(
            f"image_encoding must be one of {', '.join(IMAGE_ENCODINGS)}; got {encoding!r}"
        )

    def _declared_metadata(self, parsed: object | None, body: bytes) -> dict[str, Any]:
        """Return the metadata the target declared, through the configured map.

        Raises:
            TargetContractError: A configured path is absent or holds the wrong type, or metadata
                was configured but the response is not a JSON object to read it from.
        """
        metadata_map = self._config.metadata_map
        if not metadata_map or parsed is None:
            return {}
        if not isinstance(parsed, Mapping):
            raise TargetContractError(
                f"{self._key_prefix}.metadata_map names metadata in the response, but "
                f"the response from target {self.name!r} is a "
                f"{type(parsed).__name__} rather than a JSON object",
                body_prefix=body,
            )
        return apply_metadata_map(parsed, metadata_map)

    def _status_detail(self, response: httpx.Response, body: bytes) -> str:
        """Describe an unusable status in one diagnostic naming what arrived.

        `body` is the response body already read, so the excerpt comes from the bytes in hand.
        """
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
        excerpt = _body_excerpt(body)
        if excerpt:
            detail += f"; body starts {excerpt}"
        return (
            detail
            + "; the framework treats any non-2xx status as a contract error, because "
            "an http target has no declared-rejection channel"
        )
