"""The transport-mapping half of the adapter contract: how an answer is decoded, and how a request
is shaped, all of it from configuration.

:mod:`qrcheck.targets.base` states the ``returns`` contract once and both the ``library`` and
the ``cli`` adapters consume it through :func:`~qrcheck.targets.base.decode_returned_image`; the
dotted ``metadata_map`` lookup, the dotted ``image_field`` and the ``kwargs_map`` projection are
each one implementation too. Three properties are load-bearing: every delivery mode delivers the
same bytes, and the four ways that can go wrong -- a missing file, a path that is a directory,
base64 that does not decode, an empty body -- all arrive as
:class:`~qrcheck.targets.base.TargetContractError` rather than as the ``OSError`` or
``binascii.Error`` underneath them; a response's shape comes only from configuration, with
``_body`` the one sentinel the adapter may know (AAP §0.3.4's MUST NOT on vendor endpoints,
field names and response shapes), whose mechanical half is the AST deny-list this module runs
over ``qrcheck/targets/http.py``; and a subprocess is argv-only -- a list of ``str`` words,
``shell=False`` and the configured ``timeout_s`` handed to ``subprocess.run`` (AAP §0.3.2,
NFR-7) -- because a missing ``timeout`` would let a hung target hang the whole run.

Deliberately not asserted: the ``declared_*`` fields of
:class:`~qrcheck.targets.base.GeneratedSymbol` are the target's own claim, so what is pinned is
that they carry exactly what the configured path holds and stay ``None`` when nothing is
configured. No threshold from ``qrcheck.toml`` is read, nothing is skipped, marked, retried or
timed, no socket is opened (``httpx.MockTransport`` answers every request and ``subprocess.run``
is replaced by a recorder), and no directory is created outside ``tmp_path``.
"""

from __future__ import annotations

import ast
import base64
import json
import re
import subprocess
import sys
import tempfile
import types
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from qrcheck.config.models import (
    CLITarget,
    HTTPTarget,
    LibraryTarget,
    LOOPBACK_HOSTS,
    RETURNS_KINDS,
    TEMPLATE_FIELDS,
)
from qrcheck.evidence import mux
from qrcheck.targets import base as base_module
from qrcheck.targets import cli as cli_module
from qrcheck.targets import http as http_module
from qrcheck.targets.base import (
    GenerateRequest,
    TargetContractError,
    apply_metadata_map,
    build_request,
    decode_returned_image,
    resolve_dotted_path,
)
from qrcheck.targets.cli import CliAdapter
from qrcheck.targets.http import HttpAdapter
from qrcheck.targets.library import LibraryAdapter

# A genuine 1x1 PNG, and the artefact every delivery mode has to reproduce byte for byte.
# It is a literal rather than something built at import time because that keeps the whole
# module free of an image-library import; that it really is a PNG is asserted once, in
# test_the_artefact_fixture_is_a_real_png, so a corrupted literal fails there instead of
# making every adapter test agree with a broken fixture.
_PNG_BYTES: Final[bytes] = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00"
    b"\x007n\xf9$\x00\x00\x00\nIDATx\x9cch\x00\x00\x00\x82\x00\x81w\xcdr\xb6\x00\x00\x00"
    b"\x00IEND\xaeB`\x82"
)

#: The same artefact as the base64 text a target may hand back instead.
_PNG_BASE64: Final[str] = base64.b64encode(_PNG_BYTES).decode("ascii")

# A payload with a non-ASCII character, so ``payload_utf8`` and the CLI's bare ``{}`` slot
# have a real encoding to perform: a payload that happened to be ASCII would let a lossy
# or latin-1 conversion pass these tests.
_PAYLOAD_TEXT: Final[str] = "héllo wörld — qrcheck"
_PAYLOAD: Final[bytes] = _PAYLOAD_TEXT.encode("utf-8")

# The request's two colours, chosen so the expected ``#rrggbb`` text is not the default
# pair: a transport that dropped colour entirely would otherwise still match.
_FOREGROUND: Final[tuple[int, int, int]] = (0x12, 0x34, 0x56)
_BACKGROUND: Final[tuple[int, int, int]] = (0xDD, 0xEE, 0xFF)

# A test-local ``timeout_s``, carried into every transport test that can assert it.  It is
# a value of this module's own configuration records, not a threshold read from
# ``qrcheck.toml``; nothing here asserts a duration, only that the configured number is the
# one handed to the transport.
_TIMEOUT_S: Final[float] = 7.5

# The loopback endpoint an HTTP target record carries.  Nothing dials it: every request in
# this module is answered by ``httpx.MockTransport`` inside the process, so the port is
# never bound and no packet is sent.
_HTTP_URL: Final[str] = "http://127.0.0.1:18000/generate"

#: The sibling configuration record the HTTP adapter is asserted to mirror.
_HTTP_MODULE_PATH: Final[Path] = Path(http_module.__file__).resolve()

# The vendor field names the HTTP adapter may never name.  A literal equal to one of these
# anywhere in ``qrcheck/targets/http.py`` means the module learned a response shape of its
# own, which is the AAP §0.3.4 MUST NOT; the only response-shape vocabulary it may hold is
# the ``_body`` sentinel and the placeholder tokens of
# :data:`qrcheck.config.models.TEMPLATE_FIELDS`.  Sorted, so a failure message lists the
# offenders in one fixed order.
_VENDOR_FIELD_DENYLIST: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "image",
            "qrcode",
            "qr_code",
            "code",
            "data",
            "result",
            "barcode",
            "symbol",
            "payload_image",
        }
    )
)

# A literal shaped like a dotted path is a hard-coded response key by definition -- the
# adapter's own response paths arrive from ``image_field`` and ``metadata_map``, so none
# may appear in its source.
_DOTTED_PATH_LITERAL: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)

_URL_LITERAL: Final[re.Pattern[str]] = re.compile(r"\bhttps?://", re.IGNORECASE)

#: The one sentinel the HTTP adapter may know, restated here rather than imported so the
#: scan does not silently follow a rename in the module it audits.
_BODY_SENTINEL: Final[str] = "_body"

#: How each ``returns`` kind carries the artefact, as a value builder over the PNG and the
#: path it was written to.  Iterated in sorted order and asserted complete against
#: ``RETURNS_KINDS``, so a fifth kind cannot be added to the configuration vocabulary
#: without a case here.
_RETURN_VALUES: Final[Mapping[str, Callable[[bytes, Path], object]]] = {
    "base64": lambda png, path: base64.b64encode(png).decode("ascii"),
    "bytes_path": lambda png, path: path,
    "file_path": lambda png, path: str(path),
    "raw_bytes": lambda png, path: png,
}


def _hex_colour(rgb: tuple[int, int, int]) -> str:
    """Return one colour as the ``#rrggbb`` text the HTTP template documents for it. Derived here,
    from the request's own tuple, rather than imported from the adapter: a test that reused the
    adapter's formatter could not notice the formatter changing the documented textual shape.
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **members: object) -> str:
    """Put a throwaway module named `name` into ``sys.modules`` and return its name. The
    ``library`` adapter resolves its target through ``importlib.import_module(module)``
    followed by attribute lookup, so an in-memory module is what lets a test exercise that path
    with no file on disk and no ``sys.path`` edit.
    """
    module = types.ModuleType(name)
    for attribute in sorted(members):
        setattr(module, attribute, members[attribute])
    monkeypatch.setitem(sys.modules, name, module)
    return name


def _library_target(
    *,
    callable_path: str,
    returns: str,
    kwargs_map: Mapping[str, str] | None = None,
    metadata_map: Mapping[str, str] | None = None,
    capabilities: Iterable[str] = ("png",),
    name: str = "library",
) -> LibraryTarget:
    """Return a resolved ``type = "library"`` target record for `callable_path`."""
    return LibraryTarget(
        name=name,
        type="library",
        capabilities=frozenset(capabilities),
        callable=callable_path,
        kwargs_map=dict(kwargs_map or {}),
        returns=returns,  # type: ignore[arg-type]
        metadata_map=dict(metadata_map or {}),
    )


def _cli_target(
    *,
    payload_arg: str,
    output_arg: str,
    returns: str,
    command: tuple[str, ...] = ("qrcheck-cli-target",),
    capabilities: Iterable[str] = ("png",),
    metadata_map: Mapping[str, str] | None = None,
    name: str = "cli",
) -> CLITarget:
    """Return a resolved ``type = "cli"`` target record for the given command line."""
    return CLITarget(
        name=name,
        type="cli",
        timeout_s=_TIMEOUT_S,
        capabilities=frozenset(capabilities),
        command=command,
        payload_arg=payload_arg,
        output_arg=output_arg,
        returns=returns,  # type: ignore[arg-type]
        metadata_map=dict(metadata_map or {}),
    )


def _http_target(**overrides: Any) -> HTTPTarget:
    """Return a resolved ``type = "http"`` target record, overridden per test. The defaults
    describe the simplest loopback target -- one payload placeholder, the raw body as the image
    -- so a test that cares about the template or the extraction states only what it changes.
    """
    values: dict[str, Any] = {
        "name": "example",
        "type": "http",
        "timeout_s": _TIMEOUT_S,
        "capabilities": frozenset({"png", "color", "module_size", "quiet_zone"}),
        "url": _HTTP_URL,
        "method": "POST",
        "request_template": {"payload_b64": "{payload_b64}"},
        "image_field": _BODY_SENTINEL,
        "image_encoding": "raw",
        "metadata_map": {},
    }
    values.update(overrides)
    return HTTPTarget(**values)


def _install_http_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[list[httpx.Request], list[dict[str, Any]]]:
    """Answer every HTTP request in-process and return what the adapter sent. ``httpx.Client`` is
    replaced inside :mod:`qrcheck.targets.http` with a factory that injects an
    ``httpx.MockTransport`` over the test's `handler`, so the adapter's own construction -- its
    timeout, its redirect and proxy settings -- stays live while no socket is touched.
    """
    requests: list[httpx.Request] = []
    client_kwargs: list[dict[str, Any]] = []
    real_client = httpx.Client

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        client_kwargs.append(dict(kwargs))
        kwargs["transport"] = httpx.MockTransport(record)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(http_module.httpx, "Client", factory)
    return requests, client_kwargs


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    written: bytes,
    stdout: bytes = b"",
    write_at: Callable[[list[str]], str] | None = None,
) -> list[tuple[list[str], dict[str, Any]]]:
    """Replace ``subprocess.run`` in the ``cli`` adapter with a recorder. No child process is
    started: the recorder captures the argument vector and keyword arguments the adapter
    passed, optionally writes the artefact where a path-bearing ``output_arg`` said it would be
    -- which is what a conforming target does -- and returns a zero-status
    ``CompletedProcess``.
    """
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        if write_at is not None:
            Path(write_at(argv)).write_bytes(written)
        return subprocess.CompletedProcess(argv, 0, stdout, b"")

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    return calls


def _assert_argv_only(calls: list[tuple[list[str], dict[str, Any]]]) -> list[str]:
    """Assert the one recorded invocation is argv-only, and return its argument vector. The three
    properties AAP §0.3.2 and NFR-7 require of every subprocess this product starts: the vector
    is a list (never a command string a shell would split), every word is text, no shell is
    interposed, and the configured timeout is handed to ``subprocess.run`` so a hung target is
    killed rather than waited on.
    """
    assert len(calls) == 1, f"expected exactly one invocation, recorded {len(calls)}"
    argv, kwargs = calls[0]
    assert isinstance(argv, list), f"argv must be a list, not {type(argv).__name__}"
    assert argv, "argv must not be empty"
    assert all(isinstance(word, str) for word in argv), f"non-text argv word in {argv!r}"
    assert kwargs["shell"] is False, "the adapter must never interpose a shell"
    assert kwargs["timeout"] == _TIMEOUT_S, (
        f"the configured timeout_s must reach subprocess.run; got {kwargs['timeout']!r}"
    )
    assert kwargs["capture_output"] is True, "the adapter must capture the child's output"
    return argv


@pytest.fixture
def private_tempdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point :mod:`tempfile` at a directory inside the test, and return it. Redirecting
    ``tempfile.tempdir`` keeps that inside ``tmp_path`` rather than the host's shared temporary
    directory, where a sibling checkout's leftovers would otherwise be visible to -- and
    pollute -- the ``CL-ARTIFACTS`` snapshot the suite takes of the same directory.
    """
    directory = tmp_path / "tempdir"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


def _written_path_after(prefix: str) -> Callable[[list[str]], str]:
    """Return an extractor for the artefact path in an ``output_arg`` word."""

    def extract(argv: list[str]) -> str:
        word = argv[-1]
        assert word.startswith(prefix), f"unexpected artefact argument {word!r}"
        return word[len(prefix) :]

    return extract


def _string_literals() -> list[tuple[int, str]]:
    """Return every string literal in ``qrcheck/targets/http.py`` as ``(lineno, text)``. Read from
    the module's own resolved path, never from the working directory, and sorted by line so a
    failure names its offenders in source order.
    """
    source = _HTTP_MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_HTTP_MODULE_PATH))
    return sorted(
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _format_offenders(offenders: Iterable[tuple[int, str]]) -> str:
    """Render offending literals as ``path:lineno:literal``, one clause per offender."""
    return "; ".join(
        f"{_HTTP_MODULE_PATH}:{lineno}:{literal!r}" for lineno, literal in sorted(offenders)
    )


def _host_of(literal: str) -> str | None:
    """Return the host a URL-shaped literal names, or ``None`` when it has none."""
    return urllib.parse.urlsplit(literal).hostname


# ---------------------------------------------------------------------------
# The artefact fixture, and the four kinds of `returns`
# ---------------------------------------------------------------------------


def test_the_artefact_fixture_is_a_real_png() -> None:
    """The literal every delivery mode reproduces really is a 1x1 PNG. Without this, a corrupted
    byte string would make every adapter test below agree with a broken fixture: the adapters
    sniff (or pass through) bytes, so nothing else in this module would notice.
    """
    assert mux.mime_type(_PNG_BYTES) == mux.MIME_PNG
    assert mux.decode_base64(_PNG_BASE64) == _PNG_BYTES
    image = mux.decode_image(_PNG_BYTES)
    assert image.size == (1, 1)


def test_returns_kinds_cover_the_configuration_enum() -> None:
    """Every ``returns`` token the configuration accepts has a case in this module. The
    parametrization below is driven by :data:`_RETURN_VALUES`, so this is what keeps a fifth
    ``returns`` kind from being added to the configuration vocabulary while no test covers it
    -- the delivery contract would otherwise grow silently.
    """
    assert sorted(_RETURN_VALUES) == sorted(RETURNS_KINDS)
    assert sorted(RETURNS_KINDS) == ["base64", "bytes_path", "file_path", "raw_bytes"]


@pytest.mark.parametrize("returns", sorted(_RETURN_VALUES))
def test_every_returns_kind_delivers_the_artefact_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, returns: str
) -> None:
    """Each of the four delivery modes yields the artefact the target wrote, unchanged.
    ``raw_bytes`` hands the artefact back itself, ``base64`` hands back its base64 text,
    ``file_path`` names a path the framework supplied, and ``bytes_path`` a path the target
    chose; all four converge on one ``decode_returned_image`` call in the port, so a mode that
    copied, re-encoded or truncated the bytes would show up here rather than in the check
    layer.
    """
    artefact = tmp_path / "symbol.png"
    artefact.write_bytes(_PNG_BYTES)
    value = _RETURN_VALUES[returns](_PNG_BYTES, artefact)
    module = _install_module(
        monkeypatch, f"transport_fixture_{returns}", generate=lambda: value
    )
    adapter = LibraryAdapter(
        _library_target(callable_path=f"{module}:generate", returns=returns)
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert symbol.image_bytes == _PNG_BYTES
    assert symbol.output_format == "png"
    assert symbol.declared_version is None
    assert symbol.declared_ec is None
    assert symbol.declared_mask is None
    assert symbol.declared_mode is None
    assert symbol.elapsed_ms >= 0.0


def test_a_relative_path_is_resolved_against_the_given_base_dir(tmp_path: Path) -> None:
    """A relative path a target reports is read relative to the base directory it names."""
    (tmp_path / "symbol.png").write_bytes(_PNG_BYTES)

    assert decode_returned_image("symbol.png", "bytes_path", base_dir=tmp_path) == _PNG_BYTES
    assert (
        decode_returned_image(Path("symbol.png"), "file_path", base_dir=tmp_path) == _PNG_BYTES
    )


@pytest.mark.parametrize(
    ("returns", "build_value"),
    [
        ("bytes_path", lambda directory: str(directory / "no-such-artefact.png")),
        ("file_path", lambda directory: str(directory / "a-directory")),
        ("base64", lambda directory: "@@@ not base64 @@@"),
        ("raw_bytes", lambda directory: b""),
    ],
)
def test_a_returns_failure_arrives_as_a_contract_error(
    returns: str, build_value: Callable[[Path], object], tmp_path: Path
) -> None:
    """Every way a delivery mode can fail surfaces as ``TargetContractError``. A missing file, a
    directory instead of a file, text that is not base64 and an empty body are the four the
    adapter must convert: letting the ``OSError``, the ``IsADirectoryError`` or the
    ``binascii.Error`` underneath them escape would report an unusable response as an internal
    framework fault (``error``) instead of a contract failure (``failed``), which is exactly
    the distinction AAP §0.3.4 draws.
    """
    (tmp_path / "a-directory").mkdir()

    with pytest.raises(TargetContractError) as raised:
        decode_returned_image(build_value(tmp_path), returns, base_dir=None)

    assert str(raised.value)


def test_an_empty_base64_body_is_a_contract_error() -> None:
    """Base64 that decodes to nothing is an empty artefact, not a zero-length image."""
    with pytest.raises(TargetContractError):
        decode_returned_image("", "base64")


def test_an_unknown_returns_kind_is_a_framework_defect() -> None:
    """A token outside the four is the schema's to refuse, so it raises ``ValueError``.
    ``qrcheck.config.schema`` rejects an out-of-enum ``returns`` at load time with exit 2, so a
    kind reaching the port is a framework defect rather than a target failure and must not be
    reported as one.
    """
    with pytest.raises(ValueError) as raised:
        decode_returned_image(_PNG_BYTES, "stdout")

    assert "stdout" in str(raised.value)


def test_ensure_image_bytes_refuses_a_body_of_the_wrong_format() -> None:
    """A ``png`` request answered with SVG bytes is a contract failure, not a symbol."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'

    assert base_module.ensure_image_bytes(svg, "svg") == svg
    with pytest.raises(TargetContractError) as raised:
        base_module.ensure_image_bytes(svg, "png")

    assert "png" in str(raised.value)
    assert raised.value.body_prefix == svg


# ---------------------------------------------------------------------------
# metadata_map: the configured dotted path decides, and nothing is defaulted
# ---------------------------------------------------------------------------

# A response shape with one shallow path, one nested path, two different paths to a mask,
# a null, and a sequence: every case the dotted lookup has to distinguish.  Its key names
# are this fixture's own -- the point of the tests below is that the adapter knew none of
# them until configuration named them.
_RESPONSE: Final[Mapping[str, object]] = {
    "images": [{"png": "first"}, {"png": "second"}],
    "metadata": {"deep": {"mask": 3}, "level": " Q ", "null_mask": None, "version": 7},
    "nested": {"deep": {"mask": 5}},
}


@pytest.mark.parametrize(
    ("field", "path", "expected"),
    [
        ("version", "metadata.version", 7),
        ("mask", "metadata.deep.mask", 3),
        ("mask", "nested.deep.mask", 5),
        ("ec", "metadata.level", "Q"),
    ],
)
def test_metadata_map_reads_the_configured_dotted_path(
    field: str, path: str, expected: object
) -> None:
    """The value recorded is the one at the configured path, coerced to the field's type. ``mask``
    is reachable two ways here, so a mapper that guessed a key instead of walking the
    configured one could not return both 3 and 5.
    """
    assert apply_metadata_map(_RESPONSE, {field: path}) == {field: expected}


def test_an_empty_metadata_map_reads_nothing() -> None:
    """A target that reports no metadata is not asked for any, and gets an empty result."""
    assert apply_metadata_map(_RESPONSE, {}) == {}


def test_metadata_map_reports_a_field_the_response_does_not_carry() -> None:
    """A configured path the response lacks is a contract error naming field and path. This is AAP
    §0.3.4's "missing field named in config" trigger: the adapter promised to read
    ``metadata.mask`` and the response does not carry it, so the response is unusable and the
    diagnostic has to say which field and which path a reader should fix.
    """
    with pytest.raises(TargetContractError) as raised:
        apply_metadata_map(_RESPONSE, {"mask": "metadata.mask"})

    message = str(raised.value)
    assert "mask" in message
    assert "metadata.mask" in message


def test_metadata_map_reports_a_configured_path_holding_null() -> None:
    """A configured path whose value is null is unusable, not silently "unknown"."""
    with pytest.raises(TargetContractError) as raised:
        apply_metadata_map(_RESPONSE, {"mask": "metadata.null_mask"})

    assert "metadata.null_mask" in str(raised.value)


def test_an_unknown_framework_metadata_field_is_a_framework_defect() -> None:
    """A key outside the four framework fields is the schema's to refuse (exit 2)."""
    with pytest.raises(ValueError) as raised:
        apply_metadata_map(_RESPONSE, {"version_pin": "metadata.version"})

    assert "version_pin" in str(raised.value)


def test_resolve_dotted_path_reads_mapping_keys_and_sequence_indices() -> None:
    """One walk serves both shapes, so ``images.0.png`` and ``metadata.version`` agree."""
    assert resolve_dotted_path(_RESPONSE, "metadata.version") == 7
    assert resolve_dotted_path(_RESPONSE, "images.0.png") == "first"
    assert resolve_dotted_path(_RESPONSE, "images.1.png") == "second"


def test_the_body_sentinel_is_not_a_dotted_path() -> None:
    """``_body`` names a transport's raw body, so the dotted walk does not reach it. The HTTP
    adapter resolves that one sentinel itself; if this walk also special-cased it, the
    ``library`` and ``cli`` adapters would inherit a notion of "raw body" they have no use for,
    and the sentinel would stop being the single name the HTTP layer owns.
    """
    with pytest.raises(KeyError) as raised:
        resolve_dotted_path(_RESPONSE, _BODY_SENTINEL)

    assert _BODY_SENTINEL in str(raised.value)


def test_metadata_reaches_the_symbol_as_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four ``declared_*`` fields carry the configured paths and nothing else. They are the
    target's own claim: ``SY-EC-MATCH`` reads the primary decoder and the format information,
    ``SY-MASK`` the format information and ``SY-VERSION-MIN`` the sampled module count, so
    believing these values would let a target that lies about its metadata pass.
    """

    def generate() -> tuple[bytes, Mapping[str, object]]:
        return _PNG_BYTES, _RESPONSE

    module = _install_module(monkeypatch, "transport_metadata_target", generate=generate)
    adapter = LibraryAdapter(
        _library_target(
            callable_path=f"{module}:generate",
            returns="raw_bytes",
            metadata_map={"ec": "metadata.level", "version": "metadata.version"},
        )
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert symbol.image_bytes == _PNG_BYTES
    assert symbol.declared_version == 7
    assert symbol.declared_ec == "Q"
    assert symbol.declared_mask is None
    assert symbol.declared_mode is None


def test_a_metadata_map_beside_a_bare_artefact_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``metadata_map`` configured the callable owes an ``(image, metadata)`` pair. Returning
    the artefact alone would leave the adapter reading a PNG as if it were a mapping, so the
    shape is refused explicitly instead of failing somewhere deeper.
    """
    module = _install_module(monkeypatch, "transport_bare_target", generate=lambda: _PNG_BYTES)
    adapter = LibraryAdapter(
        _library_target(
            callable_path=f"{module}:generate",
            returns="raw_bytes",
            metadata_map={"version": "metadata.version"},
        )
    )

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert "metadata_map" in str(raised.value)


# ---------------------------------------------------------------------------
# kwargs_map: the framework's field, the target's keyword argument
# ---------------------------------------------------------------------------


def test_kwargs_map_projects_each_configured_field_onto_its_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each mapped request field arrives as the kwarg the configuration names. The values are the
    request's own: ``payload`` travels as the exact bytes (a transport may not guess an
    encoding for the framework), ``module_size_px`` as the integer the request pinned.
    """
    received: list[dict[str, object]] = []

    def generate(**kwargs: object) -> bytes:
        received.append(dict(kwargs))
        return _PNG_BYTES

    module = _install_module(monkeypatch, "transport_kwargs_target", generate=generate)
    kwargs_map = {
        "ec_level": "ecl",
        "module_size_px": "scale",
        "output_format": "fmt",
        "payload": "text",
    }
    adapter = LibraryAdapter(
        _library_target(
            callable_path=f"{module}:generate",
            returns="raw_bytes",
            kwargs_map=kwargs_map,
            capabilities=("png", "module_size"),
        )
    )
    request = build_request(_PAYLOAD, "Q", output_format="png", module_size_px=6)

    adapter.generate(request)

    assert len(received) == 1
    kwargs_seen = received[0]
    assert sorted(kwargs_seen) == sorted(kwargs_map.values())
    assert kwargs_seen["text"] == _PAYLOAD
    assert kwargs_seen["text"].decode("utf-8") == _PAYLOAD_TEXT  # type: ignore[union-attr]
    assert kwargs_seen["ecl"] == request.ec_level
    assert kwargs_seen["fmt"] == request.output_format
    assert kwargs_seen["scale"] == request.module_size_px
    assert "version" not in kwargs_seen
    assert "quiet_zone_modules" not in kwargs_seen


def test_kwargs_map_names_request_fields_not_template_tokens() -> None:
    """``payload_utf8`` is a template token, not a request field, and is refused. The two
    vocabularies are deliberately different: ``kwargs_map`` keys are :class:`GenerateRequest`
    field names, while ``payload_utf8`` belongs to the CLI and HTTP template vocabulary.
    """
    with pytest.raises(ValueError) as raised:
        LibraryAdapter(
            _library_target(
                callable_path="builtins:str",
                returns="raw_bytes",
                kwargs_map={"payload_utf8": "text"},
            )
        )

    assert "payload_utf8" in str(raised.value)


def test_kwargs_map_refuses_two_fields_on_one_kwarg() -> None:
    """Two framework fields cannot share one keyword argument; the second would win."""
    with pytest.raises(ValueError) as raised:
        LibraryAdapter(
            _library_target(
                callable_path="builtins:str",
                returns="raw_bytes",
                kwargs_map={"payload": "value", "ec_level": "value"},
            )
        )

    assert "value" in str(raised.value)


# ---------------------------------------------------------------------------
# The CLI adapter: both template shapes, an argv list, no shell, the timeout
# ---------------------------------------------------------------------------


def test_a_flag_payload_template_pipes_the_bytes_on_stdin(
    monkeypatch: pytest.MonkeyPatch, private_tempdir: Path
) -> None:
    """A bare flag names the child's stdin convention; the payload travels as raw bytes. A
    template that starts with ``-`` and carries no placeholder is one argv word, and the
    payload is written to the child's standard input -- which is the only way to deliver a
    payload no argv word can hold, since an argument vector cannot carry arbitrary bytes.
    """
    calls = _install_fake_process(
        monkeypatch, written=_PNG_BYTES, write_at=_written_path_after("--out=")
    )
    adapter = CliAdapter(
        _cli_target(payload_arg="--payload-stdin", output_arg="--out={}", returns="file_path")
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    argv = _assert_argv_only(calls)
    assert argv[:2] == ["qrcheck-cli-target", "--payload-stdin"]
    assert argv[2].startswith("--out=")
    artefact = Path(argv[2][len("--out=") :])
    assert artefact.name == "symbol.png"
    assert artefact.parent.parent == private_tempdir
    _, kwargs = calls[0]
    assert kwargs["input"] == _PAYLOAD
    assert kwargs["stdin"] is None, "a stdin payload must be piped, not replaced by DEVNULL"
    assert kwargs["text"] is False, "the child's streams must stay bytes"
    assert symbol.image_bytes == _PNG_BYTES
    assert symbol.output_format == "png"


def test_positional_templates_carry_the_payload_and_the_output_path(
    monkeypatch: pytest.MonkeyPatch, private_tempdir: Path
) -> None:
    """A bare ``{}`` slot renders the payload as UTF-8 text and the path as one word. The other
    permitted shape: both templates are placeholders with no flag around them, so the payload
    is an argv word and the child inherits ``DEVNULL`` on standard input rather than a pipe
    nobody would read.
    """
    calls = _install_fake_process(
        monkeypatch, written=_PNG_BYTES, write_at=_written_path_after("")
    )
    adapter = CliAdapter(
        _cli_target(payload_arg="{}", output_arg="{}", returns="file_path")
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    argv = _assert_argv_only(calls)
    assert argv[1] == _PAYLOAD_TEXT
    assert Path(argv[2]).name == "symbol.png"
    assert Path(argv[2]).parent.parent == private_tempdir
    _, kwargs = calls[0]
    assert kwargs["input"] is None
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert symbol.image_bytes == _PNG_BYTES


def test_a_named_payload_template_renders_the_shared_vocabulary(
    monkeypatch: pytest.MonkeyPatch, private_tempdir: Path
) -> None:
    """The CLI placeholder vocabulary is the HTTP one, rendered as one argv word."""
    calls = _install_fake_process(
        monkeypatch, written=_PNG_BYTES, write_at=_written_path_after("")
    )
    template = "{payload_b64}|{ec_level}|{format}|{size}|{margin}|{fg}|{bg}"
    adapter = CliAdapter(
        _cli_target(
            payload_arg=template,
            output_arg="{}",
            returns="file_path",
            capabilities=("png", "color", "module_size", "quiet_zone"),
        )
    )
    request = build_request(
        _PAYLOAD,
        "Q",
        module_size_px=4,
        quiet_zone_modules=2,
        foreground_rgb=_FOREGROUND,
        background_rgb=_BACKGROUND,
    )

    adapter.generate(request)

    expected = "|".join(
        [
            base64.b64encode(request.payload).decode("ascii"),
            request.ec_level,
            request.output_format,
            str(request.module_size_px),
            str(request.quiet_zone_modules),
            _hex_colour(request.foreground_rgb),
            _hex_colour(request.background_rgb),
        ]
    )
    assert calls[0][0][1] == expected
    artefact = Path(calls[0][0][2])
    assert artefact.name == "symbol.png"
    assert artefact.parent.parent == private_tempdir


def test_an_unpinned_geometry_renders_as_an_empty_word_never_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A geometry the request pins nothing for renders empty, not as the text ``"None"``. This is
    the bug the omission rule exists to prevent, in its argv spelling: a target handed
    ``"None"`` for a margin would parse it as a number and either fail for the wrong reason or
    draw a symbol the framework never asked for.
    """
    calls = _install_fake_process(monkeypatch, written=_PNG_BYTES, stdout=_PNG_BYTES)
    adapter = CliAdapter(
        _cli_target(payload_arg="{size}", output_arg="", returns="raw_bytes")
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    argv = _assert_argv_only(calls)
    assert argv[-1] == ""
    assert "None" not in argv
    assert symbol.image_bytes == _PNG_BYTES


@pytest.mark.parametrize(
    ("returns", "stdout"),
    [("raw_bytes", _PNG_BYTES), ("base64", _PNG_BASE64.encode("ascii"))],
)
def test_a_stdout_returns_kind_reads_the_child_stdout(
    monkeypatch: pytest.MonkeyPatch, returns: str, stdout: bytes
) -> None:
    """With no ``output_arg`` the artefact arrives on stdout, in the configured form. An empty
    ``output_arg`` passes no path, so only the two stdout kinds can read the answer; ``base64``
    arrives as text and is decoded by the port's one base64 decoder, so a target that wrote
    something else is refused rather than half-read.
    """
    calls = _install_fake_process(monkeypatch, written=_PNG_BYTES, stdout=stdout)
    adapter = CliAdapter(_cli_target(payload_arg="{}", output_arg="", returns=returns))

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    argv = _assert_argv_only(calls)
    assert argv == ["qrcheck-cli-target", _PAYLOAD_TEXT], "no artefact path may be passed"
    assert calls[0][1]["input"] is None
    assert symbol.image_bytes == _PNG_BYTES


def test_the_timeout_the_transport_receives_is_the_resolved_configurations(
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., Any],
) -> None:
    """The ``timeout_s`` handed to the transport is the resolved record's, not a literal. This is
    the property NFR-7 and AAP §0.3.2 rest on: an adapter that kept a ceiling of its own would
    kill a slow target the user had allowed more time for, or -- worse -- wait forever on one
    the user had bounded.
    """
    target = CLITarget(
        name="cli",
        type="cli",
        capabilities=frozenset({"png"}),
        command=("qrcheck-cli-target",),
        payload_arg="{}",
        output_arg="",
        returns="raw_bytes",
        metadata_map={},
    )
    resolved = config_factory(targets={"cli": target}).targets["cli"]
    calls = _install_fake_process(monkeypatch, written=_PNG_BYTES, stdout=_PNG_BYTES)
    adapter = CliAdapter(resolved)  # type: ignore[arg-type]

    adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert resolved.timeout_s > 0
    assert calls[0][1]["timeout"] == resolved.timeout_s
    assert calls[0][1]["shell"] is False


# ---------------------------------------------------------------------------
# The HTTP adapter: the eight placeholders, and a shape that comes from configuration
# ---------------------------------------------------------------------------

# A template naming every placeholder the configuration contract defines, plus one
# placeholder embedded in literal text.  The eight-token coverage is asserted before the
# substitution is, so a placeholder quietly dropped from this fixture fails as a fixture
# fault rather than letting the substitution test pass on a shorter vocabulary.
_FULL_TEMPLATE: Final[Mapping[str, object]] = {
    "payload_b64": "{payload_b64}",
    "payload_utf8": "{payload_utf8}",
    "ec": "{ec_level}",
    "format": "{format}",
    "size": "{size}",
    "margin": "{margin}",
    "fg": "{fg}",
    "bg": "{bg}",
    "label": "size={size}px",
}

# The dotted path to the image inside an envelope response, and the envelope itself: this
# fixture's own key names, so nothing in the adapter could have known them.
_IMAGE_FIELD: Final[str] = "envelope.images.0.png"


def _payload_request() -> GenerateRequest:
    """Return the one request every HTTP test sends, built through the single builder."""
    return build_request(
        _PAYLOAD,
        "Q",
        module_size_px=4,
        quiet_zone_modules=2,
        foreground_rgb=_FOREGROUND,
        background_rgb=_BACKGROUND,
    )


def _png_response(request: httpx.Request) -> httpx.Response:
    """Answer every request with a PNG body: the raw-body delivery form."""
    return httpx.Response(200, content=_PNG_BYTES)


def _envelope_response(request: httpx.Request) -> httpx.Response:
    """Answer every request with a JSON envelope carrying the image and its metadata."""
    body = {
        "envelope": {
            "images": [{"png": _PNG_BASE64}],
            "meta": {"level": "M", "version": 3},
        }
    }
    return httpx.Response(200, content=json.dumps(body).encode("utf-8"))


def test_the_post_template_substitutes_all_eight_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every documented placeholder carries the request's own value into the request body. Each
    expectation is computed from the request rather than written down, so the assertion is
    about the mapping and not about a hard-coded string.
    """
    referenced = {
        name for text in _FULL_TEMPLATE.values() for name in re.findall(r"{([a-z0-9_]+)}", text)
    }
    assert sorted(referenced) == sorted(TEMPLATE_FIELDS)

    requests, _ = _install_http_client(monkeypatch, _png_response)
    adapter = HttpAdapter(_http_target(request_template=dict(_FULL_TEMPLATE)), name="example")
    request = _payload_request()

    symbol = adapter.generate(request)

    assert len(requests) == 1
    sent = requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == _HTTP_URL
    assert sent.headers["content-type"] == "application/json"
    assert json.loads(sent.content) == {
        "bg": _hex_colour(request.background_rgb),
        "ec": request.ec_level,
        "fg": _hex_colour(request.foreground_rgb),
        "format": request.output_format,
        "label": f"size={request.module_size_px}px",
        "margin": request.quiet_zone_modules,
        "payload_b64": base64.b64encode(request.payload).decode("ascii"),
        "payload_utf8": _PAYLOAD_TEXT,
        "size": request.module_size_px,
    }
    assert symbol.image_bytes == _PNG_BYTES


def test_a_get_carries_the_template_as_query_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``GET`` sends the same rendered table as query text, in its textual spelling. Query
    parameters are scalars only, so the geometry arrives as text rather than as a number -- the
    one place the two methods differ in what they put on the wire, and the reason the adapter
    refuses a structured template for a ``GET`` instead of stringifying it.
    """
    requests, _ = _install_http_client(monkeypatch, _png_response)
    adapter = HttpAdapter(
        _http_target(method="GET", request_template=dict(_FULL_TEMPLATE)), name="example"
    )
    request = _payload_request()

    adapter.generate(request)

    sent = requests[0]
    assert sent.method == "GET"
    assert not sent.content, "a GET carries no request body"
    assert sent.url.host in LOOPBACK_HOSTS
    params = sent.url.params
    assert params["payload_b64"] == base64.b64encode(request.payload).decode("ascii")
    assert params["payload_utf8"] == _PAYLOAD_TEXT
    assert params["ec"] == request.ec_level
    assert params["format"] == request.output_format
    assert params["size"] == str(request.module_size_px)
    assert params["margin"] == str(request.quiet_zone_modules)
    assert params["fg"] == _hex_colour(request.foreground_rgb)
    assert params["bg"] == _hex_colour(request.background_rgb)
    assert params["label"] == f"size={request.module_size_px}px"


def test_an_unpinned_optional_is_omitted_instead_of_rendering_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A placeholder whose value the request leaves unset drops its key entirely. Rendering
    ``"None"`` would hand a target a literal it would try to parse as a number or a string;
    dropping the key is the documented rule, and it applies to a placeholder embedded in
    literal text exactly as it does to a whole-string one.
    """
    requests, _ = _install_http_client(monkeypatch, _png_response)
    adapter = HttpAdapter(
        _http_target(
            request_template={
                "payload_b64": "{payload_b64}",
                "label": "size={size}px",
                "margin": "{margin}",
                "size": "{size}",
            }
        ),
        name="example",
    )

    adapter.generate(GenerateRequest(payload=_PAYLOAD))

    sent = requests[0]
    assert json.loads(sent.content) == {
        "payload_b64": base64.b64encode(_PAYLOAD).decode("ascii")
    }
    assert b"None" not in sent.content


def test_the_body_sentinel_yields_the_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``image_field = "_body"`` with the ``raw`` encoding returns the body unchanged."""
    requests, _ = _install_http_client(monkeypatch, _png_response)
    adapter = HttpAdapter(
        _http_target(image_field=_BODY_SENTINEL, image_encoding="raw"), name="example"
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert len(requests) == 1
    assert symbol.image_bytes == _PNG_BYTES
    assert symbol.output_format == "png"
    assert symbol.declared_version is None
    assert symbol.declared_mask is None


def test_a_dotted_image_field_yields_the_decoded_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured dotted path plus ``base64`` reaches the image and its envelope. The metadata
    read through the map lands on the ``declared_*`` fields only, so a target's claim is
    recorded and never believed.
    """
    requests, _ = _install_http_client(monkeypatch, _envelope_response)
    adapter = HttpAdapter(
        _http_target(
            image_field=_IMAGE_FIELD,
            image_encoding="base64",
            metadata_map={"ec": "envelope.meta.level", "version": "envelope.meta.version"},
        ),
        name="example",
    )

    symbol = adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert len(requests) == 1
    assert symbol.image_bytes == _PNG_BYTES
    assert symbol.declared_version == 3
    assert symbol.declared_ec == "M"
    assert symbol.declared_mask is None
    assert symbol.declared_mode is None


def test_the_client_carries_the_configured_timeout_and_no_proxy_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client holds three properties of the offline contract only the transport can hold.
    ``timeout_s`` is the only ceiling on a target, ``follow_redirects=False`` keeps a redirect
    from carrying the request to a host the product forbids, and ``trust_env=False`` keeps an
    ``HTTP_PROXY`` variable from routing a loopback call off the machine (AAP §0.3.2, NFR-2).
    """
    _, client_kwargs = _install_http_client(monkeypatch, _png_response)
    adapter = HttpAdapter(_http_target(), name="example")

    adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert len(client_kwargs) == 1
    assert client_kwargs[0]["timeout"] == httpx.Timeout(_TIMEOUT_S)
    assert client_kwargs[0]["follow_redirects"] is False
    assert client_kwargs[0]["trust_env"] is False


def test_a_placeholder_the_configuration_does_not_define_is_a_configuration_error() -> None:
    """A placeholder outside the eight is a configuration fault, not a traceback. The schema
    rejects it at load time with exit 2 naming the key; constructing the adapter from a
    hand-built record must not turn the same mistake into an internal error, so the adapter
    raises the schema's own ``ConfigError`` naming ``targets.<name>.request_template``.
    """
    with pytest.raises(http_module.ConfigError) as raised:
        HttpAdapter(
            _http_target(
                request_template={"payload_b64": "{payload_b64}", "x": "{nope}"}
            ),
            name="example",
        )

    assert "nope" in str(raised.value)


def test_a_template_without_a_payload_placeholder_is_a_configuration_error() -> None:
    """A template that can never send a payload is refused where it is configured."""
    with pytest.raises(http_module.ConfigError) as raised:
        HttpAdapter(_http_target(request_template={"q": "{ec_level}"}), name="example")

    assert "payload" in str(raised.value)


# ---------------------------------------------------------------------------
# The HTTP module's own vocabulary: no vendor field, no hard-coded response path
# ---------------------------------------------------------------------------


def test_the_scan_reads_the_http_module_and_finds_its_vocabulary() -> None:
    """The scan is not vacuous: it parsed the module and sees the vocabulary it permits. Without
    this guard every assertion below would also pass on an empty parse -- a moved file, a
    renamed module, a walk that collected nothing -- and the MUST NOT would look enforced while
    enforcing nothing.
    """
    literals = _string_literals()
    texts = {text for _, text in literals}

    assert _HTTP_MODULE_PATH.is_file()
    assert literals, f"expected string literals in {_HTTP_MODULE_PATH}"
    assert literals == sorted(literals), "the scan must return its literals in source order"
    assert _BODY_SENTINEL in texts, "the one response-shape sentinel must be a literal here"
    assert set(TEMPLATE_FIELDS) <= texts, "the eight placeholder tokens must be literals here"
    assert _DOTTED_PATH_LITERAL.fullmatch("vendor.image.png") is not None
    assert _URL_LITERAL.search("https://example.invalid/qr") is not None
    assert _host_of("http://127.0.0.1:1/x") == "127.0.0.1"


def test_the_http_module_names_no_vendor_field() -> None:
    """No string literal in the HTTP adapter is a vendor-ish response field name. The adapter's
    response shape is configuration: ``image_field`` names the image and ``metadata_map`` names
    the metadata, so a literal equal to one of the deny-listed names anywhere in the module
    would mean the module had learned a vendor's key.
    """
    offenders = [
        (lineno, text) for lineno, text in _string_literals() if text in _VENDOR_FIELD_DENYLIST
    ]

    assert not offenders, (
        "qrcheck/targets/http.py must learn no vendor field name; the deny-list is "
        f"{list(_VENDOR_FIELD_DENYLIST)} and the offenders are: {_format_offenders(offenders)}"
    )


def test_the_http_module_hard_codes_no_response_path() -> None:
    """No literal in the adapter is shaped like a dotted path, so no response key is baked in.
    ``image_field`` and ``metadata_map`` are dotted paths a user writes down; one appearing in
    the source would be a hard-coded response key with extra steps.
    """
    offenders = [
        (lineno, text)
        for lineno, text in _string_literals()
        if _DOTTED_PATH_LITERAL.fullmatch(text) and not text.startswith("targets.")
    ]

    assert not offenders, (
        "qrcheck/targets/http.py must hard-code no response path; the offenders are: "
        + _format_offenders(offenders)
    )


def test_the_http_module_names_no_endpoint_outside_loopback() -> None:
    """A URL literal may only name a loopback host, and the adapter holds none at all. A URL
    literal is a hard-coded endpoint however well it is spelled, so this is the endpoint half
    of the same MUST NOT: the only host the framework may talk to is one of
    :data:`qrcheck.config.LOOPBACK_HOSTS`, and a vendor's address in this module would be a
    remote endpoint with a loopback check layered over it.
    """
    offenders = [
        (lineno, text)
        for lineno, text in _string_literals()
        if _URL_LITERAL.search(text)
        and str(_host_of(text)).lower() not in LOOPBACK_HOSTS
    ]

    assert not offenders, (
        f"qrcheck/targets/http.py must name no endpoint outside {sorted(LOOPBACK_HOSTS)}; "
        f"the offenders are: {_format_offenders(offenders)}"
    )

