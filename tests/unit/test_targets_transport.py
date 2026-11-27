"""The transport-mapping half of the adapter contract: how an answer is decoded, and how a request
is shaped, all of it from configuration.

:mod:`qrcheck.targets.base` states the ``returns`` contract once and the ``library`` and ``cli``
adapters consume it through :func:`~qrcheck.targets.base.decode_returned_image`; the dotted
``metadata_map`` lookup, the dotted ``image_field`` and the ``kwargs_map`` projection are each
one implementation too. Three properties are load-bearing: every delivery mode delivers the same
bytes, and the four ways that can go wrong (a missing file, a path that is a directory, base64
that does not decode, an empty body) all arrive as
:class:`~qrcheck.targets.base.TargetContractError`, never as the ``OSError`` or
``binascii.Error`` underneath; a response's shape comes only from configuration, ``_body`` being
the one sentinel the adapter may know, whose mechanical half is the AST deny-list over
``qrcheck/targets/http.py``; and a subprocess is argv-only -- ``str`` words, ``shell=False``, the
configured ``timeout_s`` handed to ``subprocess.run`` and both of the child's streams captured
through the adapter's own bounded pipes rather than ``capture_output``, because a missing timeout
would let a hung target hang the whole run and an unbounded capture would let a chatty one exhaust
the runner.

Deliberately not asserted: the ``declared_*`` fields are the target's own claim, so what is
pinned is that they carry what the configured path holds and stay ``None`` when nothing is
configured. No threshold from ``qrcheck.toml`` is read, no socket is opened and no directory is
created outside ``tmp_path``."""

from __future__ import annotations

import ast
import base64
import json
import os
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

# A genuine 1x1 PNG, the artefact every delivery mode reproduces byte for byte. A literal keeps
# the module free of an image-library import, and test_the_artefact_fixture_is_a_real_png proves
# it is a PNG, so a corrupted one fails there instead of making every test agree with it.
_PNG_BYTES: Final[bytes] = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00"
    b"\x007n\xf9$\x00\x00\x00\nIDATx\x9cch\x00\x00\x00\x82\x00\x81w\xcdr\xb6\x00\x00\x00"
    b"\x00IEND\xaeB`\x82"
)

#: The same artefact as the base64 text a target may hand back instead.
_PNG_BASE64: Final[str] = base64.b64encode(_PNG_BYTES).decode("ascii")

# A payload with a non-ASCII character, so ``payload_utf8`` and the CLI's bare ``{}`` slot have a
# real encoding to perform: an ASCII payload would let a lossy or latin-1 conversion pass.
_PAYLOAD_TEXT: Final[str] = "héllo wörld — qrcheck"
_PAYLOAD: Final[bytes] = _PAYLOAD_TEXT.encode("utf-8")

# The request's two colours, chosen so the expected ``#rrggbb`` text is not the default pair.
_FOREGROUND: Final[tuple[int, int, int]] = (0x12, 0x34, 0x56)
_BACKGROUND: Final[tuple[int, int, int]] = (0xDD, 0xEE, 0xFF)

# A test-local ``timeout_s``, carried into every transport test that can assert it. It is a value
# of this module's own configuration records, not a threshold read from ``qrcheck.toml``: nothing
# here asserts a duration, only that the configured number is the one handed to the transport.
_TIMEOUT_S: Final[float] = 7.5

# The loopback endpoint an HTTP target record carries. Nothing dials it: ``httpx.MockTransport``
# answers every request in the process, so the port is never bound and no packet is sent.
_HTTP_URL: Final[str] = "http://127.0.0.1:18000/generate"

#: The sibling configuration record the HTTP adapter is asserted to mirror.
_HTTP_MODULE_PATH: Final[Path] = Path(http_module.__file__).resolve()

# The vendor field names the HTTP adapter may never name: a literal equal to one of these anywhere
# in ``qrcheck/targets/http.py`` means the module learned a response shape of its own, which the
# configuration-only contract forbids. The only response-shape vocabulary it may hold is the
# ``_body`` sentinel and the placeholder tokens of :data:`qrcheck.config.models.TEMPLATE_FIELDS`.
# Sorted, so a failure message lists the offenders in one fixed order.
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

# A literal shaped like a dotted path is a hard-coded response key by definition: the adapter's own
# response paths arrive from ``image_field`` and ``metadata_map``, so none may appear in its source.
_DOTTED_PATH_LITERAL: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)

_URL_LITERAL: Final[re.Pattern[str]] = re.compile(r"\bhttps?://", re.IGNORECASE)

#: The one sentinel the HTTP adapter may know, restated rather than imported so the scan does not
#: follow a rename in the module it audits.
_BODY_SENTINEL: Final[str] = "_body"

#: How each ``returns`` kind carries the artefact, as a value builder over the PNG and its path.
#: Iterated in sorted order and asserted complete against ``RETURNS_KINDS``, so a fifth kind cannot
#: be added to the configuration vocabulary without a case here.
_RETURN_VALUES: Final[Mapping[str, Callable[[bytes, Path], object]]] = {
    "base64": lambda png, path: base64.b64encode(png).decode("ascii"),
    "bytes_path": lambda png, path: path,
    "file_path": lambda png, path: str(path),
    "raw_bytes": lambda png, path: png,
}


def _hex_colour(rgb: tuple[int, int, int]) -> str:
    """Return one colour as the ``#rrggbb`` text the HTTP template documents. Derived from the
    request's own tuple rather than imported from the adapter, so a formatter change cannot pass."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **members: object) -> str:
    """Put a throwaway module named `name` into ``sys.modules`` and return its name, so the
    ``library`` adapter's ``importlib`` path runs with no file on disk and no ``sys.path`` edit."""
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
        returns=returns,
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
        returns=returns,
        metadata_map=dict(metadata_map or {}),
    )


def _http_target(**overrides: Any) -> HTTPTarget:
    """Return a resolved ``type = http`` record, overridden per test: the defaults describe the
    simplest loopback target, so a test states only what it changes."""
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
    """Answer every HTTP request in-process and return what the adapter sent: ``httpx.Client`` is
    replaced by a factory injecting an ``httpx.MockTransport``, so no socket is touched."""
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
    """Replace ``subprocess.run`` in the ``cli`` adapter with a recorder: no child starts, the
    vector and keywords are captured, and `stdout` is written into the pipe the adapter handed over
    as the child's stdout -- the adapter reads it back through its own bounded capture, so a fake
    that returned the bytes on a ``CompletedProcess`` would prove nothing."""
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        if write_at is not None:
            Path(write_at(argv)).write_bytes(written)
        os.write(kwargs["stdout"], stdout)
        os.write(kwargs["stderr"], b"")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    return calls


def _assert_argv_only(calls: list[tuple[list[str], dict[str, Any]]]) -> list[str]:
    """Assert the one recorded invocation is argv-only and return its vector -- the properties the
    argv-only contract requires (a list of text words, no shell, the configured timeout) together
    with the bound this module's own capture enforces in place of ``capture_output`` (both streams
    are the adapter's descriptors, never ``subprocess.PIPE``)."""
    assert len(calls) == 1, f"expected exactly one invocation, recorded {len(calls)}"
    argv, kwargs = calls[0]
    assert isinstance(argv, list), f"argv must be a list, not {type(argv).__name__}"
    assert argv, "argv must not be empty"
    assert all(isinstance(word, str) for word in argv), f"non-text argv word in {argv!r}"
    assert kwargs["shell"] is False, "the adapter must never interpose a shell"
    assert kwargs["timeout"] == _TIMEOUT_S, (
        f"the configured timeout_s must reach subprocess.run; got {kwargs['timeout']!r}"
    )
    assert "capture_output" not in kwargs, (
        "the adapter must bound the child's streams through its own pipes rather than capturing "
        "them unbounded"
    )
    for stream in ("stdout", "stderr"):
        descriptor = kwargs[stream]
        assert isinstance(descriptor, int), (
            f"the child's {stream} must be the adapter's own pipe descriptor; got {descriptor!r}"
        )
        assert descriptor not in (subprocess.PIPE, subprocess.DEVNULL, subprocess.STDOUT), (
            f"the child's {stream} must be the adapter's own bounded pipe, not the "
            f"{descriptor!r} sentinel"
        )
    return argv


@pytest.fixture
def private_tempdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point :mod:`tempfile` at a directory inside the test and return it, so nothing lands in the
    host's shared temporary directory that ``CL-ARTIFACTS`` snapshots."""
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
    """Return every string literal in ``qrcheck/targets/http.py`` as ``(lineno, text)``, sorted by
    line so a failure names its offenders in source order."""
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


# The artefact fixture, and the four kinds of `returns`


def test_the_artefact_fixture_is_a_real_png() -> None:
    """The literal every delivery mode reproduces really is a 1x1 PNG: a corrupted fixture would
    otherwise make every adapter test below agree with it."""
    assert mux.mime_type(_PNG_BYTES) == mux.MIME_PNG
    assert mux.decode_base64(_PNG_BASE64) == _PNG_BYTES
    image = mux.decode_image(_PNG_BYTES)
    assert image.size == (1, 1)


def test_returns_kinds_cover_the_configuration_enum() -> None:
    """Every ``returns`` token the configuration accepts has a case here, so a fifth kind cannot
    join the vocabulary while no test covers it."""
    assert sorted(_RETURN_VALUES) == sorted(RETURNS_KINDS)
    assert sorted(RETURNS_KINDS) == ["base64", "bytes_path", "file_path", "raw_bytes"]


@pytest.mark.parametrize("returns", sorted(_RETURN_VALUES))
def test_every_returns_kind_delivers_the_artefact_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, returns: str
) -> None:
    """Each of the four delivery modes yields the artefact unchanged -- ``raw_bytes`` itself,
    ``base64`` its text, the two path kinds a path -- through the port's single decoder."""
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
    """Every way a delivery mode can fail surfaces as ``TargetContractError``: an escaping
    ``OSError`` or ``binascii.Error`` would report the response as an internal fault."""
    (tmp_path / "a-directory").mkdir()

    with pytest.raises(TargetContractError) as raised:
        decode_returned_image(build_value(tmp_path), returns, base_dir=None)

    assert str(raised.value)


def test_an_empty_base64_body_is_a_contract_error() -> None:
    """Base64 that decodes to nothing is an empty artefact, not a zero-length image."""
    with pytest.raises(TargetContractError):
        decode_returned_image("", "base64")


def test_an_unknown_returns_kind_is_a_framework_defect() -> None:
    """A token outside the four raises ``ValueError``: the schema rejects an out-of-enum ``returns``
    at load time, so a kind reaching the port is a framework defect, not a target failure."""
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


# metadata_map: the configured dotted path decides, and nothing is defaulted

# A response shape with one shallow path, one nested path, two different paths to a mask, a null
# and a sequence. Its key names are this fixture's own: the point of the tests below is that the
# adapter knew none of them until configuration named them.
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
    """The value recorded is the one at the configured path, coerced to the field's type: ``mask``
    is reachable two ways here, so a mapper that guessed a key could not return both 3 and 5."""
    assert apply_metadata_map(_RESPONSE, {field: path}) == {field: expected}


def test_an_empty_metadata_map_reads_nothing() -> None:
    """A target that reports no metadata is not asked for any, and gets an empty result."""
    assert apply_metadata_map(_RESPONSE, {}) == {}


def test_metadata_map_reports_a_field_the_response_does_not_carry() -> None:
    """A configured path the response lacks is a contract error naming field and path, so the
    diagnostic says what to fix."""
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
    """``_body`` names a transport's raw body, so the dotted walk does not reach it: special-casing
    it here would give the ``library`` and ``cli`` adapters a notion of raw body they never use."""
    with pytest.raises(KeyError) as raised:
        resolve_dotted_path(_RESPONSE, _BODY_SENTINEL)

    assert _BODY_SENTINEL in str(raised.value)


def test_metadata_reaches_the_symbol_as_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four ``declared_*`` fields carry the configured paths and nothing else: they are the
    target's own claim, and believing them would let a target that lies about its metadata pass."""

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
    """With ``metadata_map`` configured the callable owes an ``(image, metadata)`` pair: the
    artefact alone would leave the adapter reading a PNG as if it were a mapping."""
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


# kwargs_map: the framework's field, the target's keyword argument


def test_kwargs_map_projects_each_configured_field_onto_its_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each mapped request field arrives as the kwarg the configuration names, with the request's
    own value: ``payload`` as the exact bytes, ``module_size_px`` as the pinned integer."""
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
    assert kwargs_seen["text"].decode("utf-8") == _PAYLOAD_TEXT
    assert kwargs_seen["ecl"] == request.ec_level
    assert kwargs_seen["fmt"] == request.output_format
    assert kwargs_seen["scale"] == request.module_size_px
    assert "version" not in kwargs_seen
    assert "quiet_zone_modules" not in kwargs_seen


def test_kwargs_map_names_request_fields_not_template_tokens() -> None:
    """``payload_utf8`` is a template token, not a request field, and is refused: ``kwargs_map``
    keys are :class:`GenerateRequest` field names."""
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


# The CLI adapter: both template shapes, an argv list, no shell, the timeout


def test_a_flag_payload_template_pipes_the_bytes_on_stdin(
    monkeypatch: pytest.MonkeyPatch, private_tempdir: Path
) -> None:
    """A template that starts with ``-`` and carries no placeholder is one argv word, and the
    payload travels as raw bytes on the child's stdin -- the only way to deliver arbitrary bytes."""
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
    """A bare ``{}`` slot renders the payload as UTF-8 text and the path as one word, and the child
    inherits ``DEVNULL`` on standard input rather than a pipe nobody would read."""
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
    """A geometry the request pins nothing for renders empty, never the text ``"None"``: a target
    handed ``"None"`` for a margin would fail for the wrong reason or draw an unasked-for symbol."""
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
    """With no ``output_arg`` only the two stdout kinds can read the answer: ``base64`` arrives
    as text and is decoded by the port's one decoder."""
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
    """The ``timeout_s`` handed to the transport is the resolved record's, not a literal -- the
    property the argv-only contract rests on."""
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
    adapter = CliAdapter(resolved)

    adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert resolved.timeout_s > 0
    assert calls[0][1]["timeout"] == resolved.timeout_s
    assert calls[0][1]["shell"] is False


# The HTTP adapter: the eight placeholders, and a shape that comes from configuration

# A template naming every placeholder the contract defines plus one embedded in literal text. The
# eight-token coverage is asserted before the substitution, so a dropped placeholder fails as a
# fixture fault rather than letting the substitution test pass on a shorter vocabulary.
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
    """Every documented placeholder carries the request's own value into the body, each expectation
    computed from the request rather than written down."""
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
    """A ``GET`` sends the same rendered table as query text: parameters are scalars only, which is
    why a structured template is refused for a ``GET`` instead of stringified."""
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
    """A placeholder whose value the request leaves unset drops its key entirely: rendering
    ``"None"`` would hand a target a literal to parse."""
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
    """A configured dotted path plus ``base64`` reaches the image and its envelope; the metadata
    lands on the ``declared_*`` fields only, so a target's claim is recorded and never believed."""
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
    """The client holds three properties only the transport can: ``timeout_s`` is the only
    ceiling, ``follow_redirects=False`` and ``trust_env=False``."""
    _, client_kwargs = _install_http_client(monkeypatch, _png_response)
    adapter = HttpAdapter(_http_target(), name="example")

    adapter.generate(GenerateRequest(payload=_PAYLOAD))

    assert len(client_kwargs) == 1
    assert client_kwargs[0]["timeout"] == httpx.Timeout(_TIMEOUT_S)
    assert client_kwargs[0]["follow_redirects"] is False
    assert client_kwargs[0]["trust_env"] is False


def test_a_placeholder_the_configuration_does_not_define_is_a_configuration_error() -> None:
    """A placeholder outside the eight raises the schema's own ``ConfigError`` naming
    ``targets.<name>.request_template`` rather than becoming an internal error."""
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


# The HTTP module's own vocabulary: no vendor field, no hard-coded response path


def test_the_scan_reads_the_http_module_and_finds_its_vocabulary() -> None:
    """The scan is not vacuous: it parsed the module and sees the vocabulary it permits, so a moved
    file or an empty walk fails here instead of letting every assertion below pass on nothing."""
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
    """No string literal in the HTTP adapter is a vendor-ish response field name: a literal equal to
    one of the deny-listed names would mean the module had learned a vendor's key."""
    offenders = [
        (lineno, text) for lineno, text in _string_literals() if text in _VENDOR_FIELD_DENYLIST
    ]

    assert not offenders, (
        "qrcheck/targets/http.py must learn no vendor field name; the deny-list is "
        f"{list(_VENDOR_FIELD_DENYLIST)} and the offenders are: {_format_offenders(offenders)}"
    )


def test_the_http_module_hard_codes_no_response_path() -> None:
    """No literal in the adapter is shaped like a dotted path: ``image_field`` and ``metadata_map``
    are paths a user writes down, and one in the source would be a hard-coded response key."""
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
    """A URL literal may only name a loopback host, and the adapter holds none at all: a vendor's
    address here would be a remote endpoint with a loopback check layered over it."""
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


# The artefact read is bounded: a path the target names cannot hang or exhaust the runner


def test_an_artefact_above_the_read_ceiling_is_refused_unread(tmp_path: Path) -> None:
    """A path artefact above ``_MAX_ARTEFACT_BYTES`` is refused, naming the ceiling it passed: a
    target pointing at a multi-gigabyte file would otherwise be read into this process' memory."""
    artefact = tmp_path / "symbol.png"
    with artefact.open("wb") as stream:
        stream.truncate(base_module._MAX_ARTEFACT_BYTES + 1)

    with pytest.raises(TargetContractError) as raised:
        decode_returned_image(artefact, "bytes_path")

    message = str(raised.value)
    assert str(base_module._MAX_ARTEFACT_BYTES) in message
    assert str(artefact) in message


def test_a_path_that_is_not_a_regular_file_is_refused_without_opening_it(tmp_path: Path) -> None:
    """A FIFO, a device or a directory is refused before any open, so a target naming ``/dev/zero``
    or a FIFO it never fills cannot block the read past ``timeout_s``; this test returning at all
    is the proof, because opening the FIFO here would wait for a writer that never comes."""
    fifo = tmp_path / "symbol.png"
    os.mkfifo(fifo)

    with pytest.raises(TargetContractError) as raised:
        decode_returned_image(str(fifo), "bytes_path")

    assert "not a regular file" in str(raised.value)
    assert str(fifo) in str(raised.value)


def test_an_artefact_that_grows_past_the_ceiling_while_read_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file whose size is legal when it is stat'ed but exceeds the ceiling while it is read is
    refused on the bytes actually read: the stat is the cheap gate, the capped read the bound."""
    artefact = tmp_path / "symbol.png"
    with artefact.open("wb") as stream:
        stream.truncate(base_module._MAX_ARTEFACT_BYTES + 1)
    real_stat = Path.stat

    def understated(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        found = real_stat(path, follow_symlinks=follow_symlinks)
        return os.stat_result(
            (
                found.st_mode,
                found.st_ino,
                found.st_dev,
                found.st_nlink,
                found.st_uid,
                found.st_gid,
                len(_PNG_BYTES),
                found.st_atime,
                found.st_mtime,
                found.st_ctime,
            )
        )

    monkeypatch.setattr(Path, "stat", understated)

    with pytest.raises(TargetContractError) as raised:
        decode_returned_image(artefact, "file_path")

    message = str(raised.value)
    assert "grew past" in message
    assert str(base_module._MAX_ARTEFACT_BYTES) in message

