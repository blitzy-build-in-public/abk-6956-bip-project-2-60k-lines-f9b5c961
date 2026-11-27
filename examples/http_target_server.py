"""Loopback-only HTTP QR generator: the working counterpart of the ``http`` adapter, one of the
repository's two load-bearing examples. ``tests/integration/conftest.py`` starts it and
``tests/integration/test_http_adapter.py`` drives it through the ``http`` adapter, so the
configurations below are executed rather than described; ``docs/targets.md`` quotes both.

Process contract: run ``python examples/http_target_server.py``. It binds ``127.0.0.1`` on an
ephemeral port, prints that port as the only line it writes to stdout, and logs to stderr. No
option, argument or environment variable changes the address, so the process cannot leave
loopback. Exit status 0 on ``SIGTERM`` or ``SIGINT``; no credential is read, no outbound
connection made and no file written. ``PORT`` below is the port this process printed.

Canonical configurations, both valid TOML and both loaded through the configuration schema:

    # base64 envelope: image_field is a dotted path into the JSON response
    [targets.example_http]
    type = "http"
    url = "http://127.0.0.1:PORT/generate"
    method = "POST"
    image_field = "image_b64"
    image_encoding = "base64"
    timeout_s = 30.0
    # No version_pin or mask_pin: the eight template placeholders below carry no
    # version or mask, so declaring either would make a pin-needing check resolve
    # as runnable and only skip mid-run instead of skipping before execution.
    capabilities = [
        "png", "ec_level", "module_size", "quiet_zone", "color",
        "metadata_version", "metadata_ec", "metadata_mask", "metadata_mode",
    ]

    # The request body: the adapter sends this table as the JSON object its keys
    # spell, one placeholder per value.
    [targets.example_http.request_template]
    payload_b64 = "{payload_b64}"
    ec_level = "{ec_level}"
    format = "{format}"
    module_size = "{size}"
    quiet_zone = "{margin}"
    foreground = "{fg}"
    background = "{bg}"

    # Framework field name -> dotted path into the response.
    [targets.example_http.metadata_map]
    version = "metadata.version"
    ec = "metadata.ec_level"
    mask = "metadata.mask"
    mode = "metadata.mode"

    # raw body: image_field = "_body", image_encoding = "raw"
    [targets.example_http_raw]
    type = "http"
    url = "http://127.0.0.1:PORT/raw"
    method = "POST"
    image_field = "_body"
    image_encoding = "raw"
    # No metadata_* capabilities: the raw route returns the PNG with no JSON
    # envelope, so there is no metadata to map (the block declares no map).
    capabilities = ["png", "ec_level", "module_size", "quiet_zone", "color"]

    [targets.example_http_raw.request_template]
    payload_b64 = "{payload_b64}"
    ec_level = "{ec_level}"
    format = "{format}"
    module_size = "{size}"
    quiet_zone = "{margin}"
    foreground = "{fg}"
    background = "{bg}"

Request fields on both routes: ``payload_b64`` (base64) or ``payload_utf8``, exactly one and
non-empty; ``ec_level`` (default ``M``); ``format`` (png only); ``module_size`` (8),
``quiet_zone`` (4), ``foreground`` and ``background``; and optional ``version`` and ``mask``
pins, which no placeholder conveys. A ``POST`` body's fields win over the query string's.

Wire protocol: ``POST`` and ``GET`` on ``/generate`` answer ``200`` with ``{"image_b64":
<PNG>, "metadata": {...}}``, every metadata value read back from the symbol so a generator
defect stays visible; on ``/raw`` they answer the raw PNG; any other path ``404``. A rejected
request answers ``400`` and the server keeps serving -- an HTTP target has no
declared-rejection channel, so that is ``TargetContractError`` and the ``IV-*`` checks are
never pointed here -- while an unexpected failure answers ``500`` rather than a dead socket."""

import base64
import binascii
import io
import json
import re
import signal
import sys
import threading
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import FrameType

import segno

# The bind address is a literal so no configuration can widen it; the port is
# always 0, which asks the kernel for an ephemeral port reported after bind.
_HOST = "127.0.0.1"
_MAX_BODY_BYTES = 1 << 20

_JSON_CONTENT_TYPE = "application/json"
_PNG_CONTENT_TYPE = "image/png"
_GENERATE_PATH = "/generate"
_RAW_PATH = "/raw"
_ROUTES = frozenset({_GENERATE_PATH, _RAW_PATH})

_DEFAULT_EC_LEVEL = "M"
_DEFAULT_OUTPUT_FORMAT = "png"
_DEFAULT_MODULE_SIZE = 8
_DEFAULT_QUIET_ZONE = 4
_DEFAULT_FOREGROUND = "#000000"
_DEFAULT_BACKGROUND = "#FFFFFF"
_EC_LEVELS = frozenset({"L", "M", "Q", "H"})
_MIN_VERSION, _MAX_VERSION = 1, 40
_MIN_MASK, _MAX_MASK = 0, 7
_COLOUR_PATTERN = re.compile(r"#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})\Z")


class RequestError(ValueError):
    """A request that cannot be honoured; the message is the ``400`` body's error."""


@dataclass(frozen=True)
class _SymbolRequest:
    """One validated request for a single symbol."""

    payload: bytes
    ec_level: str
    module_size: int
    quiet_zone: int
    foreground: str
    background: str
    version: int | None
    mask: int | None


def _json_bytes(payload: object) -> bytes:
    """Serialise ``payload`` canonically, so identical answers are identical bytes."""
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _as_int(value: object, field: str) -> int:
    """Coerce a JSON number or a numeric string to ``int``, rejecting anything else."""
    if isinstance(value, bool):
        raise RequestError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError:
            raise RequestError(f"{field} must be an integer") from None
    raise RequestError(f"{field} must be an integer")


def _optional_int(fields: Mapping[str, object], field: str) -> int | None:
    """Read an optional integer field, treating an absent or null value as unset."""
    value = fields.get(field)
    if value is None:
        return None
    return _as_int(value, field)


def _parse_colour(value: object) -> str:
    """Normalise ``#RGB`` or ``#RRGGBB`` to ``#RRGGBB``, rejecting any other colour."""
    if isinstance(value, str) and _COLOUR_PATTERN.match(value):
        digits = value[1:]
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        return "#" + digits.upper()
    raise RequestError(f"invalid colour {value!r}: expected #RRGGBB or #RGB")


def _decode_payload(fields: Mapping[str, object]) -> bytes:
    """Resolve the payload bytes from exactly one of ``payload_b64``/``payload_utf8``."""
    encoded = fields.get("payload_b64")
    text = fields.get("payload_utf8")
    if encoded is None and text is None:
        raise RequestError("missing payload: send payload_b64 or payload_utf8")
    payloads: list[bytes] = []
    if encoded is not None:
        if not isinstance(encoded, str):
            raise RequestError("payload_b64 must be a base64 string")
        try:
            payloads.append(base64.b64decode(encoded, validate=True))
        except (binascii.Error, ValueError):
            raise RequestError("invalid base64 payload") from None
    if text is not None:
        if not isinstance(text, str):
            raise RequestError("payload_utf8 must be a string")
        try:
            payloads.append(text.encode("utf-8"))
        except UnicodeEncodeError:
            raise RequestError("payload_utf8 is not encodable as UTF-8") from None
    if len(payloads) == 2 and payloads[0] != payloads[1]:
        raise RequestError("ambiguous payload: send exactly one of payload_b64 or payload_utf8")
    payload = payloads[0]
    if not payload:
        raise RequestError("empty payload: qrcheck requires at least one byte")
    return payload


def _decode_fields(fields: Mapping[str, object]) -> _SymbolRequest:
    """Validate every request field, applying the documented defaults."""
    payload = _decode_payload(fields)
    ec_level = fields.get("ec_level", _DEFAULT_EC_LEVEL)
    if not isinstance(ec_level, str) or ec_level not in _EC_LEVELS:
        raise RequestError(f"invalid error correction level {ec_level!r}")
    output_format = fields.get("format", _DEFAULT_OUTPUT_FORMAT)
    if output_format != _DEFAULT_OUTPUT_FORMAT:
        raise RequestError(
            f"unsupported output format {output_format!r}: "
            "examples/http_target_server.py emits png only"
        )
    module_size = _as_int(fields.get("module_size", _DEFAULT_MODULE_SIZE), "module_size")
    if module_size <= 0:
        raise RequestError("module size must be greater than 0")
    quiet_zone = _as_int(fields.get("quiet_zone", _DEFAULT_QUIET_ZONE), "quiet_zone")
    if quiet_zone < 0:
        raise RequestError("quiet zone must not be negative")
    version = _optional_int(fields, "version")
    if version is not None and not _MIN_VERSION <= version <= _MAX_VERSION:
        raise RequestError("version must be between 1 and 40")
    mask = _optional_int(fields, "mask")
    if mask is not None and not _MIN_MASK <= mask <= _MAX_MASK:
        raise RequestError("mask must be between 0 and 7")
    foreground = _parse_colour(fields.get("foreground", _DEFAULT_FOREGROUND))
    background = _parse_colour(fields.get("background", _DEFAULT_BACKGROUND))
    return _SymbolRequest(
        payload=payload,
        ec_level=ec_level,
        module_size=module_size,
        quiet_zone=quiet_zone,
        foreground=foreground,
        background=background,
        version=version,
        mask=mask,
    )


def _render(request: _SymbolRequest) -> tuple[bytes, dict[str, object]]:
    """Encode one symbol and return its PNG bytes plus the metadata it reports."""
    try:
        # make_qr forces Model 2 (never Micro QR), and boost_error=False keeps the
        # requested level instead of silently spending spare capacity on a higher one.
        symbol = segno.make_qr(
            request.payload,
            error=request.ec_level,
            version=request.version,
            mask=request.mask,
            boost_error=False,
        )
    except segno.DataOverflowError:
        where = (
            f"version {request.version}"
            if request.version is not None
            else f"any version up to {_MAX_VERSION}"
        )
        raise RequestError(
            f"data too large for {where} at error level {request.ec_level}: "
            f"{len(request.payload)} bytes requested"
        ) from None
    except (ValueError, TypeError) as exc:
        raise RequestError(str(exc)) from None
    buffer = io.BytesIO()
    symbol.save(
        buffer,
        kind=_DEFAULT_OUTPUT_FORMAT,
        scale=request.module_size,
        border=request.quiet_zone,
        dark=request.foreground,
        light=request.background,
    )
    width, height = symbol.symbol_size(scale=request.module_size, border=request.quiet_zone)
    # Metadata describes the symbol that was actually encoded; echoing the request
    # would make this target unable to reveal a generator defect.
    metadata: dict[str, object] = {
        "version": int(symbol.version),
        "ec_level": str(symbol.error),
        "mask": int(symbol.mask),
        "mode": str(symbol.mode),
        "format": _DEFAULT_OUTPUT_FORMAT,
        "width": int(width),
        "height": int(height),
    }
    return buffer.getvalue(), metadata


class _Handler(BaseHTTPRequestHandler):
    """Answer the wire protocol on one connection per request."""

    protocol_version = "HTTP/1.1"
    server_version = "qrcheck-example"
    sys_version = ""
    # A stalled client must not hold a worker thread: the socket timeout applies to
    # every read on this connection, so a truncated body fails instead of hanging.
    timeout = 30.0

    def log_message(self, format: str, *args: object) -> None:
        """Log to stderr only; stdout carries the port and nothing else."""
        sys.stderr.write("qrcheck-example: " + format % args + "\n")

    def do_GET(self) -> None:
        """Answer a request whose fields arrive in the query string."""
        self._dispatch(read_body=False)

    def do_POST(self) -> None:
        """Answer a request whose JSON body overrides the query string."""
        self._dispatch(read_body=True)

    def _dispatch(self, *, read_body: bool) -> None:
        """Build the whole response first, then send it exactly once."""
        try:
            status, content_type, body = self._build_response(read_body=read_body)
        except RequestError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:  # last-resort net: a reported 500 beats a dead socket
            self._send_json(500, {"error": f"internal error: {type(exc).__name__}: {exc}"})
            return
        self._send_bytes(status, content_type, body)

    def _build_response(self, *, read_body: bool) -> tuple[int, str, bytes]:
        """Validate the request and produce its response triple."""
        path, query = self._split_target()
        fields: dict[str, object] = dict(query)
        if read_body:
            fields.update(self._read_body())
        if path not in _ROUTES:
            return 404, _JSON_CONTENT_TYPE, _json_bytes({"error": f"not found: {path}"})
        symbol_request = _decode_fields(fields)
        png, metadata = _render(symbol_request)
        if path == _GENERATE_PATH:
            encoded = base64.b64encode(png).decode("ascii")
            body = _json_bytes({"image_b64": encoded, "metadata": metadata})
            return 200, _JSON_CONTENT_TYPE, body
        return 200, _PNG_CONTENT_TYPE, png

    def _split_target(self) -> tuple[str, dict[str, str]]:
        """Split the request target into its path and its query-string fields."""
        parts = urllib.parse.urlsplit(self.path)
        query = {
            key: values[-1]
            for key, values in urllib.parse.parse_qs(
                parts.query, keep_blank_values=True
            ).items()
        }
        return parts.path, query

    def _read_body(self) -> dict[str, object]:
        """Read a bounded JSON-object body, or nothing when the request has none."""
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            # A request without Content-Length carries no body, so the query string
            # supplies the fields; nothing is ever read beyond a declared length.
            return {}
        try:
            length = int(raw_length.strip(), 10)
        except ValueError:
            raise RequestError(f"invalid Content-Length {raw_length!r}") from None
        if length < 0 or length > _MAX_BODY_BYTES:
            raise RequestError(f"request body length {length} outside 0..{_MAX_BODY_BYTES}")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("truncated request body")
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RequestError("request body is not valid JSON") from None
        if not isinstance(parsed, dict):
            raise RequestError("request body must be a JSON object")
        return {str(key): value for key, value in parsed.items()}

    def _send_json(self, status: int, payload: object) -> None:
        """Send a canonical JSON object."""
        self._send_bytes(status, _JSON_CONTENT_TYPE, _json_bytes(payload))

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        """Send one complete response and close the connection."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)


class _Server(ThreadingHTTPServer):
    """Serve the example's loopback clients concurrently, with no connect() left queued.

    ``request_queue_size`` covers a burst wider than the ``socketserver`` default of 5,
    which a 20-wide client would overflow because every response closes its connection
    (``Connection: close``) and so arrives on a fresh one. Both attributes are class-level
    because ``listen()`` runs inside the constructor: the kernel is told the queue depth
    before any statement after the constructor could change it, and a handler thread still
    running must not delay the process's exit.
    """

    request_queue_size = 128
    daemon_threads = True


def main() -> int:
    """Serve until ``SIGTERM`` or ``SIGINT``, then return the exit status."""
    server = _Server((_HOST, 0), _Handler)

    def _stop(signum: int, frame: FrameType | None) -> None:
        # shutdown() waits for serve_forever() to return, and a handler runs on the
        # very thread parked inside it, so the call has to go to another thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    # The only stdout write in the process: the fixture reads the port from it.
    print(server.server_address[1], flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
