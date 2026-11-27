"""The adapter layer's failure vocabulary is the five declared errors, and no more.

``qrcheck/targets/base.py`` declares exactly five errors, and the port's contract is that an
adapter raises nothing else: every ``except`` inside ``qrcheck/targets/`` converts a library
exception into one of them. Each trigger is exercised through the **real** adapter code path, one
test per trigger: a missing module or attribute, an absent executable, a refused loopback port and
an unsupported protocol are ``TargetUnavailable``; the configured ``timeout_s`` quoted in the
message is ``TargetTimeout``, while a connect timeout is ``TargetUnavailable``; a declared
rejection keeps the target's text byte for byte in ``TargetRejected``; an empty body, undecodable
base64, bytes that are not an image, a missing ``image_field`` and a non-zero exit that declared
no rejection are ``TargetContractError``; and ``TargetUnsupported`` is raised *before any I/O*.

Two distinctions the module exists to hold: ``TargetRejected`` versus ``TargetContractError``
decides whether ``IV-*`` passes, so misclassifying one as the other is a report defect and not
only an exception-type defect; and the five errors are siblings -- every one a ``TargetError``,
none a subclass of another -- so a caller's ``except`` order cannot reclassify a failure. The
HTTP adapter gives a non-2xx status no declared-rejection spelling, which the comment on that
test records.

The vocabulary is also held mechanically: one AST scan reads every module of ``qrcheck/targets/``
and proves that each ``except`` in it either converts what it caught into one of the five or hands
on an error its own module may raise, so a handler that let a library exception escape unwrapped
would fail a test rather than a review."""

from __future__ import annotations

import ast
import base64
import dataclasses
import errno
import json
import math
import os
import socket
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import httpx
import pytest

from qrcheck import targets as targets_package
from qrcheck.config import models
from qrcheck.corpus.capacity import max_payload_bytes
from qrcheck.decode import consensus
from qrcheck.targets import cli as cli_module
from qrcheck.targets import http as http_module
from qrcheck.targets import reference as reference_module
from qrcheck.targets.base import (
    HEX_PREFIX_LIMIT,
    Capability,
    GenerateRequest,
    TargetContractError,
    TargetError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    TargetUnsupported,
    build_request,
    hex_prefix,
    require_capabilities,
)
from qrcheck.targets.cli import CliAdapter
from qrcheck.targets.http import HttpAdapter
from qrcheck.targets.library import LibraryAdapter
from qrcheck.targets.reference import ReferenceAdapter

#: The rejection text every verbatim probe uses. It is deliberately awkward -- markup
#: characters plus non-ASCII letters -- so a re-encoding, an escaping pass or a truncation
#: anywhere between the target and the exception turns the equality assertion red.
_UNUSUAL_TEXT: Final[str] = "payload with <&> and ünïcode"

#: The configured ``timeout_s`` the two timeout probes bind, chosen so it is unlike the
#: schema's default: a message that quotes the configured value cannot be a message that
#: quotes the fallback.
_PROBE_TIMEOUT_S: Final[float] = 2.5

#: The ``contrast_floor`` the below-floor probe binds. 21:1 is the largest ratio the sRGB
#: range can express (black on white, WCAG 2.1 relative luminance), so every other colour
#: pair is below it whatever the schema's own default happens to be.
_PROBE_CONTRAST_FLOOR: Final[float] = 21.0

#: The loopback host every HTTP probe names; the framework's allow-list is
#: ``127.0.0.1``/``localhost``/``::1`` and nothing here dials anything else.
_LOOPBACK_HOST: Final[str] = "127.0.0.1"

#: The pinned version the over-capacity probe encodes at (the ``IV-OVERCAPACITY`` request pins the
#: same one). A version, not a threshold: it is a coordinate in the table the capacity figure is
#: read from.
_PINNED_VERSION: Final[int] = 10

#: The largest Model 2 version (ISO/IEC 18004 §7.1), used for the unpinned over-capacity
#: probe, which is the one request only segno's own ``DataOverflowError`` refuses.
_MAX_VERSION: Final[int] = 40

#: A body no image library can open: two passes over the byte range, so it starts with
#: NUL rather than a PNG or JPEG signature and contains no SVG marker, and it is longer
#: than the prefix limit so the recorded prefix is exercised at its full length.
_NON_IMAGE_BODY: Final[bytes] = bytes(range(256)) * 2

#: A body short enough that its hex form is shorter than the prefix limit's.
_SHORT_BODY: Final[bytes] = b"not an image"

#: The five declared errors, in the order ``qrcheck/targets/base.py`` declares them.
_ERROR_TYPES: Final[tuple[type[TargetError], ...]] = (
    TargetUnavailable,
    TargetTimeout,
    TargetRejected,
    TargetContractError,
    TargetUnsupported,
)


# Request and target builders. Every adapter is built from a real ``qrcheck.config.models`` record,
# never a mock, so the constructor the product uses is the constructor exercised here. Capabilities
# are declared explicitly because they are a gate: a record declaring none would turn every probe
# into a ``TargetUnsupported`` before the trigger under test was reached.


def _legal_request(text: str = _UNUSUAL_TEXT) -> GenerateRequest:
    """Return a request the framework considers valid, built by the single builder."""
    return build_request(text.encode("utf-8"))


def _out_of_contract(**fields: Any) -> GenerateRequest:
    """Return a request constructed straight from the record, bypassing the builder."""
    return GenerateRequest(**fields)


def _library_target(
    *,
    callable_path: str,
    returns: str = "raw_bytes",
    capabilities: tuple[str, ...] = ("png",),
    timeout_s: float = _PROBE_TIMEOUT_S,
) -> models.LibraryTarget:
    """Return a resolved ``type = "library"`` target record for `callable_path`."""
    return models.LibraryTarget(
        name="probe-library",
        type="library",
        timeout_s=timeout_s,
        capabilities=frozenset(capabilities),
        callable=callable_path,
        kwargs_map={},
        returns=returns,
    )


def _cli_target(
    *,
    command: tuple[str, ...],
    capabilities: tuple[str, ...] = ("png",),
    payload_arg: str = "--payload-stdin",
    output_arg: str = "",
    returns: str = "raw_bytes",
    timeout_s: float = _PROBE_TIMEOUT_S,
) -> models.CLITarget:
    """Return a resolved ``type = cli`` record running `command`: the default payload arrives on
    stdin through a bare flag, so nothing exists on disk before the spawn."""
    return models.CLITarget(
        name="probe-cli",
        type="cli",
        timeout_s=timeout_s,
        capabilities=frozenset(capabilities),
        command=command,
        payload_arg=payload_arg,
        output_arg=output_arg,
        returns=returns,
    )


def _http_target(
    *,
    url: str,
    capabilities: tuple[str, ...] = ("png",),
    image_field: str = "image",
    image_encoding: str = "base64",
    timeout_s: float = _PROBE_TIMEOUT_S,
) -> models.HTTPTarget:
    """Return a resolved ``type = http`` record for `url`: the template carries only
    ``{payload_b64}`` and the shape is read through ``image_field``."""
    return models.HTTPTarget(
        name="probe-http",
        type="http",
        timeout_s=timeout_s,
        capabilities=frozenset(capabilities),
        url=url,
        method="POST",
        request_template={"payload": "{payload_b64}"},
        image_field=image_field,
        image_encoding=image_encoding,
    )


def _reference_adapter(
    config: models.Config,
    *,
    contrast_floor: float | None = None,
) -> ReferenceAdapter:
    """Return the segno-backed conforming adapter bound to the test's configuration."""
    record = config.targets["reference"]
    if contrast_floor is not None:
        record = dataclasses.replace(record, contrast_floor=contrast_floor)
    return ReferenceAdapter(record, name="reference")


# Injection helpers, all at the library boundary. The adapter's own mapping logic is under test, so
# the patch is always placed where a library is called and never around the adapter:
# ``httpx.Client`` for the HTTP probes (a real client over a mock transport), ``subprocess.run`` for
# the CLI probes, and one attribute on an importable module for the library probes.


def _patch_http_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Route the adapter's HTTP client through `handler` and record what it sent."""
    real_client = httpx.Client
    seen: list[httpx.Request] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return real_client(
            transport=httpx.MockTransport(dispatch),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(http_module.httpx, "Client", factory)
    return seen


def _patch_spawn(
    monkeypatch: pytest.MonkeyPatch,
    outcome: subprocess.CompletedProcess[bytes] | BaseException,
) -> list[tuple[str, ...]]:
    """Replace ``subprocess.run`` with a recorder that answers `outcome`.

    A completed record's two streams are written into the pipes the adapter handed over, because the
    adapter reads the child's output back through its own bounded captures rather than through
    ``capture_output``; an exception is raised exactly as it stands, so a spawn failure and a
    timeout are still the library's own.
    """
    invocations: list[tuple[str, ...]] = []

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        invocations.append(tuple(argv))
        if isinstance(outcome, BaseException):
            raise outcome
        os.write(kwargs["stdout"], outcome.stdout or b"")
        os.write(kwargs["stderr"], outcome.stderr or b"")
        return subprocess.CompletedProcess(argv, outcome.returncode)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    return invocations


def _completed(
    returncode: int,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    """Return the completed-process record a patched spawn hands back."""
    return subprocess.CompletedProcess(
        args=("qrcheck-fake-target",),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _patched_callable(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    function: Callable[..., object],
) -> str:
    """Install `function` on an importable module and return its ``module:attribute`` path: the
    library adapter resolves with ``importlib`` then ``getattr``, so no file is needed."""
    monkeypatch.setattr(json, attribute, function, raising=False)
    return f"{json.__name__}:{attribute}"


def _closed_loopback_url() -> str:
    """Return a URL on a loopback port that nothing listens on: a socket is bound to port 0 so the
    kernel picks a free port, then closed, so dialling it is refused."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((_LOOPBACK_HOST, 0))
        port = int(probe.getsockname()[1])
    finally:
        probe.close()
    return f"http://{_LOOPBACK_HOST}:{port}/"


def _snapshot(root: Path) -> tuple[str, ...]:
    """Return every entry under `root`, sorted, so "nothing was written" is assertable."""
    return tuple(sorted(entry.relative_to(root).as_posix() for entry in root.rglob("*")))


# TargetUnavailable -- "the process cannot be spawned, the connection is refused,
# the import path is missing"


def test_library_adapter_missing_module_is_unavailable() -> None:
    """A ``callable`` naming a module that does not exist is a missing import path:
    ``ModuleNotFoundError`` becomes ``TargetUnavailable`` with the ``ImportError`` chained."""
    adapter = LibraryAdapter(
        _library_target(callable_path="qrcheck_no_such_module_6f3a:generate"),
        name="probe-library",
    )

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    assert isinstance(raised.value.__cause__, ImportError), (
        "TargetUnavailable for a missing module must chain the ImportError it converted: "
        f"got cause {raised.value.__cause__!r}"
    )
    assert "qrcheck_no_such_module_6f3a" in str(raised.value), (
        "the TargetUnavailable diagnostic must name the missing module: "
        f"{str(raised.value)!r}"
    )


def test_library_adapter_missing_attribute_is_unavailable() -> None:
    """A ``callable`` naming an attribute its module lacks is also a missing import path: ``json``
    imports, so only the attribute walk fails and it must still become ``TargetUnavailable``."""
    adapter = LibraryAdapter(
        _library_target(callable_path="json:no_such_attribute_6f3a"),
        name="probe-library",
    )

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    assert isinstance(raised.value.__cause__, AttributeError), (
        "TargetUnavailable for a missing attribute must chain the AttributeError: "
        f"got cause {raised.value.__cause__!r}"
    )
    assert "no_such_attribute_6f3a" in str(raised.value), (
        "the TargetUnavailable diagnostic must name the missing attribute: "
        f"{str(raised.value)!r}"
    )


def test_library_adapter_non_callable_attribute_is_a_contract_error() -> None:
    """The attribute exists but is not callable, which is a contract fault, not a miss."""
    adapter = LibraryAdapter(
        _library_target(callable_path="json:__doc__"),
        name="probe-library",
    )

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert not isinstance(raised.value, TargetUnavailable), (
        "a non-callable attribute resolved successfully, so it must not be reported as "
        "an unreachable target"
    )
    assert "not callable" in str(raised.value), (
        f"the TargetContractError must say the attribute is not callable: {str(raised.value)!r}"
    )


def test_cli_adapter_missing_executable_is_unavailable(tmp_path: Path) -> None:
    """A ``command`` whose executable does not exist cannot be spawned."""
    missing = tmp_path / "qrcheck-no-such-target-binary"
    adapter = CliAdapter(_cli_target(command=(str(missing),)), name="probe-cli")

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    message = str(raised.value)
    assert "could not be spawned" in message, (
        f"a spawn failure must be reported as TargetUnavailable: {message!r}"
    )
    assert str(missing) in message, (
        f"the TargetUnavailable diagnostic must name the configured executable: {message!r}"
    )


def test_cli_adapter_non_executable_file_is_unavailable(tmp_path: Path) -> None:
    """An existing file without the exec bit cannot be spawned either: ``execve`` fails with
    ``EACCES``, the same ``OSError`` trigger."""
    present = tmp_path / "qrcheck-target-without-exec-bit"
    present.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    present.chmod(0o644)
    adapter = CliAdapter(_cli_target(command=(str(present),)), name="probe-cli")

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    assert str(present) in str(raised.value), (
        "the TargetUnavailable diagnostic must name the file that could not be executed: "
        f"{str(raised.value)!r}"
    )


def test_cli_adapter_scratch_directory_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request whose artefact needs a scratch directory cannot run when ``mkdtemp`` fails: the
    ``OSError`` a full, exhausted or read-only filesystem raises is the same trigger as any other
    spawn failure, so it arrives as ``TargetUnavailable`` naming the configured command, with no
    process spawned and nothing created that the cleanup would have had to remove."""

    def failing_mkdtemp(*args: Any, **kwargs: Any) -> str:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cli_module.tempfile, "mkdtemp", failing_mkdtemp)
    invocations = _patch_spawn(
        monkeypatch,
        AssertionError("the adapter spawned a process it had no scratch directory for"),
    )
    adapter = CliAdapter(
        _cli_target(command=("qrcheck-fake-target",), output_arg="--output={}"),
        name="probe-cli",
    )

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    message = str(raised.value)
    assert "targets.probe-cli.command" in message, (
        f"the diagnostic must name the configured command: {message!r}"
    )
    assert "No space left on device" in message, (
        f"the diagnostic must carry the filesystem's own reason: {message!r}"
    )
    assert not isinstance(raised.value, OSError), (
        "the mkdtemp OSError must be converted, never allowed to escape as itself"
    )
    assert raised.value.__cause__ is None, (
        f"the conversion must not chain the OSError: got cause {raised.value.__cause__!r}"
    )
    assert invocations == [], (
        f"a target cannot run without the scratch directory it writes into; got {invocations!r}"
    )


def test_cli_adapter_capture_pipe_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipe the adapter cannot create is the spawn failure it is: the captured streams are what
    the child is spawned with, so an exhausted descriptor table means the target was not run.

    The conversion has to happen at the capture, not at ``subprocess.run``: an ``OSError`` raised
    while opening the pipe would otherwise escape the closed five-error vocabulary as a framework
    fault, which is the defect this adapter's error taxonomy forbids.
    """

    def failing_pipe() -> tuple[int, int]:
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(os, "pipe", failing_pipe)
    invocations = _patch_spawn(
        monkeypatch,
        AssertionError("the adapter spawned a process it had no capture pipe for"),
    )
    adapter = CliAdapter(
        _cli_target(command=("qrcheck-fake-target",), output_arg="--output={}"),
        name="probe-cli",
    )

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    message = str(raised.value)
    assert "targets.probe-cli.command" in message, (
        f"the diagnostic must name the configured command: {message!r}"
    )
    assert "Too many open files" in message, (
        f"the diagnostic must carry the operating system's own reason: {message!r}"
    )
    assert not isinstance(raised.value, OSError), (
        "the pipe OSError must be converted, never allowed to escape as itself"
    )
    assert raised.value.__cause__ is None, (
        f"the conversion must not chain the OSError: got cause {raised.value.__cause__!r}"
    )
    assert invocations == [], (
        f"a child cannot be spawned without the streams it writes into; got {invocations!r}"
    )


def test_http_adapter_refused_loopback_connection_is_unavailable() -> None:
    """A closed loopback port refuses the connection, which is the HTTP trigger."""
    adapter = HttpAdapter(_http_target(url=_closed_loopback_url()), name="probe-http")

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    assert "could not be reached" in str(raised.value), (
        "a refused connection must be reported as TargetUnavailable: "
        f"{str(raised.value)!r}"
    )
    adapter.close()


def test_http_adapter_unsupported_protocol_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable address arrives as ``httpx.UnsupportedProtocol``, not an ``HTTPError``, so
    the adapter lists it beside the network errors instead of letting it escape unwrapped."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.UnsupportedProtocol("no handler for the request scheme")

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    assert "could not be reached" in str(raised.value), (
        "UnsupportedProtocol must be reported as TargetUnavailable: "
        f"{str(raised.value)!r}"
    )


# TargetTimeout -- "no response within targets.<name>.timeout_s"


def test_cli_adapter_timeout_is_target_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child that outlives ``timeout_s`` is killed and reported as a timeout: the spawn is
    patched, so what is asserted is the conversion and the *configured* timeout in the message."""
    invocations = _patch_spawn(
        monkeypatch,
        subprocess.TimeoutExpired(("qrcheck-fake-target",), _PROBE_TIMEOUT_S),
    )
    adapter = CliAdapter(_cli_target(command=("qrcheck-fake-target",)), name="probe-cli")

    with pytest.raises(TargetTimeout) as raised:
        adapter.generate(_legal_request())

    message = str(raised.value)
    assert f"timeout_s = {_PROBE_TIMEOUT_S:g} s" in message, (
        f"the TargetTimeout message must name the configured timeout_s: {message!r}"
    )
    assert len(invocations) == 1, (
        "exactly one spawn belongs to one generate call; got "
        f"{len(invocations)}: {invocations!r}"
    )


def test_http_adapter_read_timeout_is_target_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response that never arrives inside ``timeout_s`` is a timeout, not an outage."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no response before the ceiling")

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetTimeout) as raised:
        adapter.generate(_legal_request())

    message = str(raised.value)
    assert "gave no response" in message, (
        f"a read timeout must be reported as TargetTimeout: {message!r}"
    )
    assert str(_PROBE_TIMEOUT_S) in message, (
        f"the TargetTimeout message must name the configured timeout_s: {message!r}"
    )


def test_http_adapter_connect_timeout_is_unavailable_not_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect timeout means the endpoint never answered, which is the outage trigger."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("the connection was never accepted")

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetUnavailable) as raised:
        adapter.generate(_legal_request())

    assert not isinstance(raised.value, TargetTimeout), (
        "a connect timeout is an unreachable target, not a slow one"
    )
    assert "accepted no connection" in str(raised.value), (
        "the TargetUnavailable diagnostic must say the connection was never accepted: "
        f"{str(raised.value)!r}"
    )


# A framework fault is not a target verdict -- a ``KeyError`` or ``IndexError`` raised by
# framework code reaches the runner as itself


def test_http_adapter_framework_index_error_surfaces_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``IndexError`` from framework code inside ``generate`` escapes as itself, so a framework
    indexing fault is never reported as the target's contract failure."""

    def faulting_lookup(data: object, path: str) -> object:
        raise IndexError("framework lookup fault")

    monkeypatch.setattr(http_module, "resolve_dotted_path", faulting_lookup)

    def handler(request: httpx.Request) -> httpx.Response:
        encoded = base64.b64encode(_NON_IMAGE_BODY).decode("ascii")
        return httpx.Response(200, json={"image": encoded})

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(IndexError) as raised:
        adapter.generate(_legal_request())

    assert not isinstance(raised.value, TargetContractError), (
        "an IndexError from framework code is a framework defect, not a verdict about the "
        "target"
    )
    assert str(raised.value) == "framework lookup fault", (
        "the framework's own IndexError must arrive unmodified rather than wrapped in an "
        f"adapter error: {str(raised.value)!r}"
    )


def test_http_adapter_framework_key_error_surfaces_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``KeyError`` from framework code outside the localised dotted-path conversion escapes as
    itself, so a framework indexing fault is never reported as the target's contract failure."""

    def faulting_image_gate(data: bytes, output_format: str) -> bytes:
        raise KeyError("framework artefact gate fault")

    monkeypatch.setattr(http_module, "ensure_image_bytes", faulting_image_gate)

    def handler(request: httpx.Request) -> httpx.Response:
        encoded = base64.b64encode(_NON_IMAGE_BODY).decode("ascii")
        return httpx.Response(200, json={"image": encoded})

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(KeyError) as raised:
        adapter.generate(_legal_request())

    assert not isinstance(raised.value, TargetContractError), (
        "a KeyError from framework code is a framework defect, not a verdict about the target"
    )
    assert "framework artefact gate fault" in str(raised.value), (
        "the framework's own KeyError must arrive unmodified rather than wrapped in an adapter "
        f"error: {str(raised.value)!r}"
    )


# TargetRejected -- a declared rejection, with the target's text verbatim


def test_library_adapter_rejected_channel_preserves_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callable that speaks the ``rejected:`` channel is reported verbatim: only the whitespace
    after the prefix is removed, because ``IV-*`` records the text in ``decoded.rejection``."""

    def target(**kwargs: Any) -> bytes:
        raise ValueError(f"rejected: {_UNUSUAL_TEXT}")

    path = _patched_callable(monkeypatch, "_qrcheck_rejected_target", target)
    adapter = LibraryAdapter(_library_target(callable_path=path), name="probe-library")

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(_legal_request())

    assert str(raised.value) == _UNUSUAL_TEXT, (
        "a declared rejection's text must be preserved exactly; got "
        f"{str(raised.value)!r} for the target's {_UNUSUAL_TEXT!r}"
    )


def test_cli_adapter_json_rejection_preserves_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit carrying ``{"error": <text>}`` is a declared rejection whose text is the JSON
    member's value, unchanged: the adapter decodes the line without re-wording or escaping it."""
    stdout = json.dumps({"error": _UNUSUAL_TEXT}, ensure_ascii=False).encode("utf-8")
    _patch_spawn(monkeypatch, _completed(1, stdout=stdout))
    adapter = CliAdapter(_cli_target(command=("qrcheck-fake-target",)), name="probe-cli")

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(_legal_request())

    assert str(raised.value) == _UNUSUAL_TEXT, (
        "a JSON-declared rejection's text must be preserved exactly; got "
        f"{str(raised.value)!r} for the target's {_UNUSUAL_TEXT!r}"
    )


def test_cli_adapter_rejected_prefix_preserves_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second declared spelling, a ``rejected:`` line on stderr, is equally verbatim."""
    stderr = f"rejected: {_UNUSUAL_TEXT}\n".encode("utf-8")
    _patch_spawn(monkeypatch, _completed(3, stderr=stderr))
    adapter = CliAdapter(_cli_target(command=("qrcheck-fake-target",)), name="probe-cli")

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(_legal_request())

    assert str(raised.value) == _UNUSUAL_TEXT, (
        "a 'rejected:'-declared rejection's text must be preserved exactly; got "
        f"{str(raised.value)!r} for the target's {_UNUSUAL_TEXT!r}"
    )


def test_reference_adapter_declared_rejection_is_target_rejected(
    config_factory: Callable[..., models.Config],
) -> None:
    """A real target refuses an empty payload in its own words, through the same error."""
    adapter = _reference_adapter(config_factory())

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(_out_of_contract(payload=b""))

    assert str(raised.value).startswith("empty payload"), (
        "the reference target must declare the empty-payload rejection in its own words: "
        f"{str(raised.value)!r}"
    )



# TargetContractError -- "response present but unusable"


def test_http_adapter_error_status_is_a_contract_error_not_a_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error-shaped HTTP body is unusable data, not a declared rejection: an HTTP target has no
    rejection channel in its configuration, so a 4xx is a contract error carrying the body."""
    body = json.dumps({"error": _UNUSUAL_TEXT}, ensure_ascii=False).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=body)

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert not isinstance(raised.value, TargetRejected), (
        "an HTTP target has no declared-rejection channel, so a 400 must not produce "
        "TargetRejected"
    )
    message = str(raised.value)
    assert "HTTP 400" in message, (
        f"the TargetContractError must name the status that arrived: {message!r}"
    )
    assert "no declared-rejection channel" in message, (
        "the diagnostic must state why a non-2xx status is not a rejection: "
        f"{message!r}"
    )
    assert raised.value.body_prefix == body, (
        "the failed status's body must travel on the error as evidence; got "
        f"{raised.value.body_prefix!r} for {body!r}"
    )


def test_http_adapter_server_error_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5xx with a plain-text body is unusable data whatever its shape."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream unavailable")

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert "HTTP 503" in str(raised.value), (
        f"the TargetContractError must name the status that arrived: {str(raised.value)!r}"
    )


def test_http_adapter_empty_body_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 with nothing in it is "response present but unusable"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert "empty body" in str(raised.value), (
        f"an empty response body must be named as such: {str(raised.value)!r}"
    )


def test_cli_adapter_empty_stdout_artefact_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero exit that delivered no bytes is a contract error, not an empty symbol."""
    _patch_spawn(monkeypatch, _completed(0, stdout=b""))
    adapter = CliAdapter(_cli_target(command=("qrcheck-fake-target",)), name="probe-cli")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert "empty artefact" in str(raised.value), (
        f"an empty artefact must be named as such: {str(raised.value)!r}"
    )


def test_library_adapter_undecodable_base64_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base64 text that does not decode is unusable, and the failure says so."""
    path = _patched_callable(
        monkeypatch, "_qrcheck_bad_base64_target", lambda **kwargs: "!!! not base64 !!!"
    )
    adapter = LibraryAdapter(
        _library_target(callable_path=path, returns="base64"),
        name="probe-library",
    )

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert "not valid base64" in str(raised.value), (
        f"undecodable base64 must be named as such: {str(raised.value)!r}"
    )


def test_http_adapter_undecodable_base64_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same failure through the HTTP transport, where the field carries the text."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"image": "!!! not base64 !!!"})

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert "not valid base64" in str(raised.value), (
        f"undecodable base64 must be named as such: {str(raised.value)!r}"
    )


def test_library_adapter_non_image_bytes_record_the_full_hex_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bytes that are not an image are refused, with the first 256 bytes kept as evidence."""
    path = _patched_callable(
        monkeypatch, "_qrcheck_non_image_target", lambda **kwargs: _NON_IMAGE_BODY
    )
    adapter = LibraryAdapter(_library_target(callable_path=path), name="probe-library")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert len(raised.value.body_prefix) == HEX_PREFIX_LIMIT, (
        "the recorded body prefix must be exactly HEX_PREFIX_LIMIT bytes; got "
        f"{len(raised.value.body_prefix)} for a {len(_NON_IMAGE_BODY)}-byte body"
    )
    recorded = hex_prefix(raised.value.body_prefix)
    assert len(recorded) == 2 * HEX_PREFIX_LIMIT, (
        "the hex_prefix evidence of a non-image body must be 512 hex characters; got "
        f"{len(recorded)}"
    )
    assert recorded == hex_prefix(_NON_IMAGE_BODY), (
        "the recorded prefix must be the body's own leading bytes, not a re-encoding"
    )


def test_http_adapter_non_image_bytes_record_the_full_hex_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same evidence through the HTTP transport, after base64 decoding."""
    encoded = base64.b64encode(_NON_IMAGE_BODY).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"image": encoded})

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(_http_target(url=f"http://{_LOOPBACK_HOST}:1/"), name="probe-http")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert len(hex_prefix(raised.value.body_prefix)) == 2 * HEX_PREFIX_LIMIT, (
        "the hex_prefix evidence of an HTTP non-image body must be 512 hex characters; got "
        f"{len(hex_prefix(raised.value.body_prefix))}"
    )
    assert raised.value.body_prefix == _NON_IMAGE_BODY[:HEX_PREFIX_LIMIT], (
        "the HTTP evidence must be the decoded artefact's leading bytes"
    )


def test_http_adapter_absent_image_field_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured ``image_field`` the response does not carry names both sides -- the dotted path
    and the absent segment -- so a changed shape is distinguishable from a different request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"envelope": {"other": "not the image"}})

    _patch_http_client(monkeypatch, handler)
    adapter = HttpAdapter(
        _http_target(url=f"http://{_LOOPBACK_HOST}:1/", image_field="envelope.image"),
        name="probe-http",
    )

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    message = str(raised.value)
    assert "image_field" in message, (
        f"the diagnostic must name the configured key: {message!r}"
    )
    assert "envelope.image" in message, (
        f"the diagnostic must quote the configured dotted path: {message!r}"
    )


def test_cli_adapter_nonzero_exit_without_declared_rejection_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero exit is a rejection only when the target said so: anything else is a contract
    error, and a fabricated rejection would make the ``IV-*`` checks vacuous."""
    _patch_spawn(monkeypatch, _completed(2, stdout=b"partial output", stderr=b"boom"))
    adapter = CliAdapter(_cli_target(command=("qrcheck-fake-target",)), name="probe-cli")

    with pytest.raises(TargetContractError) as raised:
        adapter.generate(_legal_request())

    assert not isinstance(raised.value, TargetRejected), (
        "a non-zero exit that declared no rejection must not be reported as a rejection"
    )
    assert "declared no rejection" in str(raised.value), (
        f"the diagnostic must say no rejection was declared: {str(raised.value)!r}"
    )
    assert raised.value.body_prefix == b"partial output", (
        "the child's stdout must travel on the error as evidence; got "
        f"{raised.value.body_prefix!r}"
    )


def test_hex_prefix_of_an_empty_body_is_empty() -> None:
    """No bytes means no evidence field, rather than a placeholder of zeroes."""
    assert hex_prefix(b"") == "", "an empty body has no hex prefix"


def test_hex_prefix_of_a_short_body_is_not_padded() -> None:
    """A body shorter than the limit yields exactly its own bytes as hex, unpadded."""
    recorded = hex_prefix(_SHORT_BODY)
    assert recorded == _SHORT_BODY.hex(), (
        f"a short body's hex prefix must be its own bytes; got {recorded!r}"
    )
    assert len(recorded) == 2 * len(_SHORT_BODY) < 2 * HEX_PREFIX_LIMIT, (
        "a body shorter than the prefix limit must not be padded to the limit; got "
        f"{len(recorded)} hex characters for {len(_SHORT_BODY)} bytes"
    )


def test_contract_error_body_prefix_is_bounded_to_the_hex_limit() -> None:
    """The error keeps at most the prefix limit, whatever the body's size."""
    error = TargetContractError("probe", body_prefix=b"\x00" * (HEX_PREFIX_LIMIT * 2))
    assert len(error.body_prefix) == HEX_PREFIX_LIMIT, (
        "TargetContractError.body_prefix must be bounded to HEX_PREFIX_LIMIT bytes; got "
        f"{len(error.body_prefix)}"
    )
    assert str(error) == "probe", "the message must stay the diagnostic, unaltered"


# TargetUnsupported -- "a capability the adapter did not declare", before any I/O


def test_require_capabilities_refuses_a_missing_capability_without_calling_the_target(
    counting_adapter: Any,
    tmp_path: Path,
) -> None:
    """One missing capability is refused and the target is never invoked: the counting stub declares
    all thirteen, so a shortfall probe narrows it to the one it has."""
    counting_adapter.capabilities = frozenset({Capability.png})
    before = _snapshot(tmp_path)

    with pytest.raises(TargetUnsupported) as raised:
        require_capabilities(counting_adapter, {Capability.svg})

    assert counting_adapter.count == 0, (
        "TargetUnsupported must be decided without invoking the target; the stub recorded "
        f"{counting_adapter.count} generate call(s)"
    )
    assert _snapshot(tmp_path) == before, (
        "a capability shortfall must not write anything; tmp_path changed from "
        f"{before!r} to {_snapshot(tmp_path)!r}"
    )
    message = str(raised.value)
    assert str(Capability.svg) in message, (
        f"the diagnostic must name the missing capability: {message!r}"
    )
    assert counting_adapter.name in message, (
        f"the diagnostic must name the adapter: {message!r}"
    )


def test_require_capabilities_names_every_missing_capability(
    counting_adapter: Any,
) -> None:
    """A shortfall of several capabilities reports all of them, in sorted order."""
    counting_adapter.capabilities = frozenset({Capability.png})
    missing = (Capability.svg, Capability.ec_level, Capability.mask_pin)

    with pytest.raises(TargetUnsupported) as raised:
        require_capabilities(counting_adapter, set(missing))

    message = str(raised.value)
    for capability in sorted(missing, key=str):
        assert str(capability) in message, (
            f"the diagnostic must name the missing capability {str(capability)!r}: "
            f"{message!r}"
        )
    assert counting_adapter.count == 0, (
        "the gate must not invoke the target; the stub recorded "
        f"{counting_adapter.count} generate call(s)"
    )


def test_cli_adapter_unsupported_capability_precedes_any_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request needing an undeclared capability never reaches a process: the patched spawn
    raises if called, so a spawn-before-refuse adapter fails here."""
    invocations = _patch_spawn(
        monkeypatch,
        AssertionError("the adapter spawned a process for an undeclared capability"),
    )
    adapter = CliAdapter(
        _cli_target(command=("qrcheck-fake-target",), capabilities=()),
        name="probe-cli",
    )

    with pytest.raises(TargetUnsupported) as raised:
        adapter.generate(_legal_request())

    assert invocations == [], (
        f"no process may be spawned for an unsupported request; got {invocations!r}"
    )
    assert "does not declare capability" in str(raised.value), (
        f"the diagnostic must name the capability gate: {str(raised.value)!r}"
    )


def test_cli_adapter_inexpressible_pin_precedes_any_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version pin no argument template can carry is refused before any process exists: the CLI
    vocabulary has no version placeholder, so running it would let the target choose."""
    invocations = _patch_spawn(
        monkeypatch,
        AssertionError("the adapter spawned a process for a request it cannot convey"),
    )
    adapter = CliAdapter(
        _cli_target(
            command=("qrcheck-fake-target",),
            capabilities=("png", "version_pin"),
        ),
        name="probe-cli",
    )
    request = _out_of_contract(payload=_UNUSUAL_TEXT.encode("utf-8"), version=_PINNED_VERSION)

    with pytest.raises(TargetUnsupported) as raised:
        adapter.generate(request)

    assert invocations == [], (
        f"no process may be spawned for an inexpressible request; got {invocations!r}"
    )
    message = str(raised.value)
    assert "cannot express" in message, (
        f"the diagnostic must say the field cannot be conveyed: {message!r}"
    )
    assert "version" in message, (
        f"the diagnostic must name the inexpressible field: {message!r}"
    )


def test_http_adapter_unsupported_capability_precedes_any_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP adapter refuses an undeclared capability before it builds a client."""

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        raise AssertionError("the adapter created an HTTP client for an unsupported request")

    monkeypatch.setattr(http_module.httpx, "Client", factory)
    adapter = HttpAdapter(
        _http_target(url=f"http://{_LOOPBACK_HOST}:1/", capabilities=()),
        name="probe-http",
    )

    with pytest.raises(TargetUnsupported) as raised:
        adapter.generate(_legal_request())

    assert "does not declare capability" in str(raised.value), (
        f"the diagnostic must name the capability gate: {str(raised.value)!r}"
    )


# Hierarchy, and no library exception escaping the reference adapter


@pytest.mark.parametrize("error_type", _ERROR_TYPES)
def test_every_declared_error_is_a_target_error(error_type: type[TargetError]) -> None:
    """Each of the five is caught by ``except TargetError``, and none of them *is* it."""
    assert issubclass(error_type, TargetError), (
        f"{error_type.__name__} must be a TargetError so a caller can catch the whole "
        "vocabulary"
    )
    assert error_type is not TargetError, (
        f"{error_type.__name__} must be a distinct error, not the base class itself"
    )


def test_no_declared_error_is_a_subclass_of_another() -> None:
    """The five are siblings, so a caller's ``except`` order cannot reclassify a failure: were
    ``TargetRejected`` a ``TargetContractError``, the contract clause would swallow rejections."""
    for outer in _ERROR_TYPES:
        for inner in _ERROR_TYPES:
            if outer is inner:
                continue
            assert not issubclass(inner, outer), (
                f"{inner.__name__} must not be a subclass of {outer.__name__}: the five "
                "errors are siblings"
            )


def test_reference_adapter_over_capacity_at_a_pinned_version_is_rejected(
    config_factory: Callable[..., models.Config],
) -> None:
    """One byte past the pinned version's capacity is a declared rejection: the capacity is read
    from the product's own tables and repeated in the message, so it is auditable."""
    adapter = _reference_adapter(config_factory())
    capacity = max_payload_bytes(_PINNED_VERSION, "M", "byte")
    request = _out_of_contract(payload=b"a" * (capacity + 1), version=_PINNED_VERSION)

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(request)

    message = str(raised.value)
    assert f"version {_PINNED_VERSION}" in message, (
        f"the rejection must name the pinned version: {message!r}"
    )
    assert str(capacity) in message, (
        f"the rejection must state the capacity it exceeded: {message!r}"
    )


def test_reference_adapter_over_capacity_without_a_pin_is_rejected(
    config_factory: Callable[..., models.Config],
) -> None:
    """An unpinned payload past Version 40 is refused by segno, and converted."""
    adapter = _reference_adapter(config_factory())
    capacity = max_payload_bytes(_MAX_VERSION, "M", "byte")
    request = _out_of_contract(payload=b"a" * (capacity + 1))

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(request)

    message = str(raised.value)
    assert f"version {_MAX_VERSION}" in message, (
        f"the unpinned rejection must name the largest Model 2 version: {message!r}"
    )
    assert str(capacity) in message, (
        f"the rejection must state the capacity it exceeded: {message!r}"
    )


def test_reference_adapter_invalid_ec_level_is_rejected(
    config_factory: Callable[..., models.Config],
) -> None:
    """An error level outside L/M/Q/H is refused by the target, never passed to segno: segno
    would raise a bare ``ValueError``, so the target validates first and names the value."""
    adapter = _reference_adapter(config_factory())
    request = _out_of_contract(payload=_UNUSUAL_TEXT.encode("utf-8"), ec_level="X")

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(request)

    assert "'X'" in str(raised.value), (
        f"the rejection must quote the offending error level: {str(raised.value)!r}"
    )


def test_reference_adapter_below_floor_contrast_is_rejected(
    config_factory: Callable[..., models.Config],
) -> None:
    """A colour pair below the target's declared floor is refused, and the floor is quoted."""
    adapter = _reference_adapter(config_factory(), contrast_floor=_PROBE_CONTRAST_FLOOR)
    request = build_request(
        _UNUSUAL_TEXT.encode("utf-8"),
        foreground_rgb=(0x88, 0x88, 0x88),
    )

    with pytest.raises(TargetRejected) as raised:
        adapter.generate(request)

    message = str(raised.value)
    assert "below the declared floor" in message, (
        f"the diagnostic must say the ratio is below the floor: {message!r}"
    )
    assert f"{_PROBE_CONTRAST_FLOOR:.1f}" in message, (
        f"the diagnostic must quote the configured floor: {message!r}"
    )


# The closed vocabulary, held mechanically over every `except` in the adapter layer
# This is the second of the two AST scans that hold the adapter MUSTs: every `except` under
# qrcheck/targets/ converts a library exception into one of the five declared errors rather than
# re-raising it raw. The trigger tests above prove the paths they drive; this scan proves the rule,
# so a handler that no trigger reaches cannot escape it.

#: The fewest handlers the scan must find before its silence means anything. The tree holds 44, so
#: a floor below that tolerates a handler moving between modules while still failing a scan that
#: stopped reading the package -- an empty glob, a renamed directory, a module that no longer
#: parses.
_MINIMUM_HANDLERS: Final[int] = 40

#: The modules that must each hold at least one handler: the port, the five adapter modules and the
#: shared capture module, whose two handlers are the ones the adapters' streams are read through. A
#: closed-vocabulary rule that found no handler in one of them would be a rule about nothing.
_HANDLER_MODULES: Final[tuple[str, ...]] = (
    "base.py",
    "capture.py",
    "cli.py",
    "http.py",
    "library.py",
    "reference.py",
)

#: What the five declared errors are called, read off the imported classes rather than retyped: a
#: rename in ``base.py`` would otherwise leave the scan allowing a name nothing raises.
_DECLARED_ERROR_NAMES: Final[frozenset[str]] = frozenset(error.__name__ for error in _ERROR_TYPES)

#: Every exception type an adapter's handlers may name beyond the five declared errors, per module,
#: each with the reason it is not an adapter error. Each one describes the framework refusing its
#: own wrong input -- a declared capability set, a logo path, a dotted path, an output format, a
#: request template, a url -- and each is raised before any I/O, so none reports a target failure.
#: Both rules read it against the module the handler lives in: it is what a conversion may raise
#: there and what a bare ``raise`` may hand on unchanged. A module absent from this table may name
#: nothing but the five, so ``except ValueError: raise`` inside ``reference.py`` lets a library
#: exception out and is flagged there: an entry here widens one module's vocabulary deliberately,
#: and only this table can widen it.
_MODULE_CONTRACT_ERRORS: Final[Mapping[str, tuple[tuple[str, str], ...]]] = MappingProxyType(
    {
        "base.py": (
            ("ValueError", "declared_capabilities and _checked_logo_path refuse a declaration"),
            ("KeyError", "resolve_dotted_path refuses an index past the sequence it walks"),
        ),
        "cli.py": (
            ("ValueError", "_required_capabilities refuses a format outside the output enum"),
        ),
        "http.py": (
            ("ConfigError", "request_template parsing names the offending configuration key"),
            ("ValueError", "_check_loopback refuses a url urllib cannot parse"),
        ),
    }
)

class _Exemption(NamedTuple):
    """One handler allowed to re-raise exactly what it caught, and the fact that makes it sound.

    A cleanup block re-raises the exception it caught, so the closed vocabulary can only allow one
    where it can also show the exception leaving it is already classified. Two proofs qualify: the
    guarded block can raise only names the module may raise (``reached`` and ``raised`` record the
    walk :func:`_raise_names_reachable_from` performs over it), or the caught type is converted to a
    declared error by a handler of the module ``converted_by`` names, that field recording the
    module, the handler and the error it raises. Every entry carries at least one, and the test
    asserts the proof rather than reading the reason and trusting it.
    """

    reason: str
    caught: tuple[str, ...]
    reached: tuple[str, ...] = ()
    raised: tuple[str, ...] = ()
    converted_by: tuple[str, str, str] | None = None


#: The handlers in the tree that re-raise exactly what they caught, keyed by ``(module file name,
#: enclosing function, the types caught)`` -- the caught types are part of the key because one
#: function may hold several handlers -- each with the reason and the proof that nothing
#: unconverted escapes. ``_build_invocation`` deletes the temporary directory it created moments
#: earlier and re-raises; the two handlers of the ``capture.py`` module both spawn paths share --
#: ``BoundedStreamCapture.__init__`` closes the descriptors a failed thread start would leak, and
#: ``open_capture_pair`` releases the capture it already opened -- re-raise for their caller, which
#: classifies them into its layer's error. The table is checked in both directions: an entry
#: that matches no handler fails, and so does a bare re-raise with no entry.
_CLEANUP_EXEMPTIONS: Final[Mapping[tuple[str, str, tuple[str, ...]], _Exemption]] = (
    MappingProxyType(
        {
            ("cli.py", "CliAdapter._build_invocation", ("Exception",)): _Exemption(
                reason=(
                    "catches Exception only to remove the temporary directory it just created, "
                    "then re-raises unchanged"
                ),
                caught=("Exception",),
                reached=("_render", "_token_value", "_payload_text"),
                raised=("TargetContractError", "ValueError"),
            ),
            ("capture.py", "BoundedStreamCapture.__init__", ("RuntimeError",)): _Exemption(
                reason=(
                    "catches RuntimeError only to close the pipe descriptors a failed thread "
                    "start would leak, then re-raises it for its caller to convert"
                ),
                caught=("RuntimeError",),
                converted_by=("cli.py", "CliAdapter._run", "TargetUnavailable"),
            ),
            ("capture.py", "open_capture_pair", ("OSError", "RuntimeError")): _Exemption(
                reason=(
                    "catches a pipe or drain-thread failure only to release the capture it had "
                    "already opened, then re-raises it to the spawn path that classifies it: the "
                    "cli adapter converts both types into TargetUnavailable, and the rasterizer "
                    "into the RuntimeError the runner records as a framework error"
                ),
                caught=("OSError", "RuntimeError"),
                converted_by=("cli.py", "CliAdapter._run", "TargetUnavailable"),
            ),
        }
    )
)

#: Every handler in the tree that raises nothing at all, keyed the same way with its reason: such a
#: handler either recovers with a documented fallback or hands the exception to a caller that
#: converts it. The table is exact in both directions and in count, so a new silent handler -- the
#: shape a swallowed exception takes -- fails here until it is documented and judged, and so does a
#: second one added to a function whose key the table already holds.
_RECOVERING_HANDLERS: Final[Mapping[tuple[str, str], str]] = MappingProxyType(
    {
        ("base.py", "_utf8_text"): (
            "returns None: a payload with no faithful text form is recorded as base64 instead"
        ),
        ("cli.py", "_json_error_member"): (
            "returns None: a line that is not a JSON object is not a declared rejection, so the "
            "next spelling is tried"
        ),
        ("capture.py", "BoundedStreamCapture._drain"): (
            "returns: the read end was closed or the child closed it, so nothing more can be read "
            "from that stream and the capture is simply finished"
        ),
        ("http.py", "_utf8_text"): (
            "returns None: the key carrying {payload_utf8} is omitted rather than filled lossily"
        ),
        ("http.py", "_body_excerpt"): (
            "returns the hexadecimal excerpt: the refused bytes are still recorded as evidence"
        ),
        ("library.py", "_call_with_watchdog._invoke"): (
            "captures the exception in the holder so the waiting caller re-raises and converts "
            "it in its own frame"
        ),
    }
)

#: A synthetic adapter the self-test drives the same analysis over: three handlers that let a
#: library exception out -- a bare re-raise of an import failure, a re-raise of the caught
#: exception itself, and a conversion to a type outside the vocabulary -- and three the rule must
#: accept, which are a conversion, a documented fallback and a declared error passed through. It
#: is a string rather than a fixture file so the offenders are unambiguously the ones named here.
#: It is scanned under the ``http.py`` label because ``passing_through`` is legitimate only in a
#: module that declares those names itself, which is the rule the parameterised cases below turn
#: around: the same handler is flagged in a module that declares none of them.
_OFFENDER_SOURCE: Final[str] = '''\
class SyntheticAdapter:
    def escaping_import(self) -> object:
        try:
            return self._get()
        except ImportError:
            raise

    def escaping_caught(self) -> object:
        try:
            return self._get()
        except httpx.TimeoutException as exc:
            raise exc

    def escaping_unrelated(self) -> object:
        try:
            return self._get()
        except OSError:
            raise RuntimeError("x")

    def converting(self) -> object:
        try:
            return self._get()
        except ImportError as exc:
            raise TargetUnavailable("the module is not importable") from exc

    def falling_back(self) -> str:
        try:
            return self._text()
        except UnicodeDecodeError:
            return ""

    def passing_through(self) -> object:
        try:
            return self._get()
        except (TargetError, ConfigError, ValueError):
            raise
'''

#: The one handler the bare-``raise`` rule is exercised over, its caught clause substituted per
#: case: a bare ``raise`` hands what it caught straight on, so the only question the rule asks is
#: whether the module the handler stands in may raise that name. The same text is therefore scanned
#: under several module labels below, which is what proves the allowance is a module's own
#: vocabulary rather than a name admitted everywhere.
_BARE_RAISE_SOURCE: Final[str] = """\
class SyntheticAdapter:
    def escaping(self) -> object:
        try:
            return self._get()
        except {caught}:
            raise
"""

#: Two handlers that raise nothing inside one function: they share a single ``(module, function)``
#: key, which is the shape a set of keys cannot tell from one documented recovering handler.
_DUPLICATE_RECOVERY_SOURCE: Final[str] = '''\
class SyntheticAdapter:
    def recovering_twice(self) -> str:
        try:
            first = self._text()
        except UnicodeDecodeError:
            first = ""
        try:
            second = self._text()
        except UnicodeDecodeError:
            second = ""
        return first + second
'''


class _Violation(NamedTuple):
    """One handler that lets something other than a declared adapter error out."""

    module: str
    qualname: str
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f"{self.module}:{self.lineno} {self.qualname}: {self.detail}"


class _Handler(NamedTuple):
    """One ``ast.ExceptHandler`` and what it does with the exception it caught.

    ``qualname`` is the dotted name of the innermost function enclosing the handler, so a defect
    names the function a reader must open (``CliAdapter.generate``, ``_call_with_watchdog._invoke``)
    rather than a bare line number that other edits move.
    """

    module: str
    qualname: str
    lineno: int
    caught: tuple[str, ...]
    caught_as: str | None
    raised: tuple[str, ...]
    bare_raise: bool

    @property
    def key(self) -> tuple[str, str]:
        """Return the ``(module, function)`` pair the tables in this section are keyed by."""
        return (self.module, self.qualname)

    @property
    def signature(self) -> tuple[str, str, tuple[str, ...]]:
        """Return ``(module, function, caught)``, which names one handler even when its function
        holds several: ``CliAdapter._build_invocation`` catches an ``OSError`` from the scratch
        directory and an ``Exception`` around the rendering in two separate handlers."""
        return (self.module, self.qualname, self.caught)

    @property
    def recovers(self) -> bool:
        """Return whether the handler raises nothing at all, neither bare nor explicit."""
        return not self.raised and not self.bare_raise

    def violation(self, detail: str) -> _Violation:
        """Return the violation this handler commits, described by `detail`."""
        return _Violation(self.module, self.qualname, self.lineno, detail)


class _Scan(NamedTuple):
    """One pass over the adapter package: every module parsed, and every handler judged."""

    modules: tuple[str, ...]
    handlers_by_module: Mapping[str, int]
    handlers: tuple[_Handler, ...]
    violations: tuple[_Violation, ...]


class _Closure(NamedTuple):
    """What one handler's guarded statements can reach inside the module that holds them."""

    functions: tuple[str, ...]
    raised: tuple[str, ...]


def _enclosing_scopes(tree: ast.AST) -> dict[int, str]:
    """Map every node's ``id`` to the dotted name of the function or class enclosing it."""
    scopes: dict[int, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                scopes.setdefault(id(child), qualname)
                walk(child, qualname)
            else:
                scopes.setdefault(id(child), prefix)
                walk(child, prefix)

    walk(tree, "")
    return scopes


def _caught_names(handler: ast.ExceptHandler) -> tuple[str, ...]:
    """Return the type names `handler` catches as written; ``except:`` catches nothing named."""
    if handler.type is None:
        return ()
    expressions = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return tuple(ast.unparse(expression) for expression in expressions)


def _raises_within(handler: ast.ExceptHandler) -> list[ast.Raise]:
    """Return every ``raise`` in `handler`'s body, in source order, excluding a nested handler's.

    A nested ``try``'s handlers are their own records, so counting their raises here would judge
    one handler twice and attribute a conversion to the handler around it.
    """
    found: list[ast.Raise] = []
    pending = list(handler.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.ExceptHandler):
            continue
        if isinstance(node, ast.Raise):
            found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda node: node.lineno)


def _raised_name(node: ast.Raise) -> str:
    """Return the type name a non-bare ``raise`` names, without any module path."""
    expression = node.exc
    if isinstance(expression, ast.Call):
        expression = expression.func
    if expression is None:
        raise AssertionError("a bare raise has no exception expression to name")
    return ast.unparse(expression).rpartition(".")[2]


def _call_name(node: ast.Call) -> str:
    """Return the last name component of the function a call invokes."""
    return ast.unparse(node.func).rpartition(".")[2]


def _handler_of(handler: ast.ExceptHandler, module: str, qualname: str) -> _Handler:
    """Return the record of one ``ast.ExceptHandler``."""
    raises = _raises_within(handler)
    return _Handler(
        module=module,
        qualname=qualname,
        lineno=handler.lineno,
        caught=_caught_names(handler),
        caught_as=handler.name,
        raised=tuple(_raised_name(node) for node in raises if node.exc is not None),
        bare_raise=any(node.exc is None for node in raises),
    )


def _contract_error_names(module: str) -> frozenset[str]:
    """Return the non-adapter exception names one module's handlers may raise."""
    return frozenset(name for name, _ in _MODULE_CONTRACT_ERRORS.get(module, ()))


def _pass_through_names(module: str) -> frozenset[str]:
    """Return the names a bare ``raise`` inside `module` may hand on unchanged.

    Only what that module may raise itself: the five declared errors, ``TargetError``, which catches
    all of them, and the module's own contract errors from :data:`_MODULE_CONTRACT_ERRORS`. Nothing
    is allowed everywhere, so a name the module never raises -- segno's ``ValueError`` in
    ``reference.py``, or a ``KeyError`` wherever nothing looks a key up -- leaves the vocabulary
    unwrapped and is flagged.
    """
    return _DECLARED_ERROR_NAMES | {TargetError.__name__} | _contract_error_names(module)


def _violations_of(handler: _Handler) -> tuple[_Violation, ...]:
    """Return every way `handler` breaks the closed vocabulary, and nothing when it keeps it.

    RULE 1: an explicit ``raise`` must name one of the five declared errors or one of the module's
    own contract errors. RULE 2: a bare ``raise`` may only hand on an error that is already inside
    the vocabulary of the module the handler lives in -- exactly the names RULE 1 allows, plus
    ``TargetError`` itself, which catches all five -- so ``http.py`` may pass on the ``ValueError``
    it refuses its own url with while ``reference.py`` may not, and re-raising the caught exception
    itself is never a conversion. A handler listed in :data:`_CLEANUP_EXEMPTIONS` is skipped: its
    reason is the record of why it may re-raise.
    """
    if handler.signature in _CLEANUP_EXEMPTIONS:
        return ()
    allowed = _DECLARED_ERROR_NAMES | _contract_error_names(handler.module)
    found: list[_Violation] = []
    if handler.bare_raise:
        passable = _pass_through_names(handler.module)
        unclassified = tuple(name for name in handler.caught if name not in passable)
        if unclassified:
            found.append(
                handler.violation(
                    f"re-raises {', '.join(unclassified)} raw: a bare raise in {handler.module} "
                    "may only pass on a declared error or one of that module's own contract "
                    "errors"
                )
            )
    for raised in handler.raised:
        if handler.caught_as is not None and raised == handler.caught_as:
            found.append(
                handler.violation(
                    f"re-raises the caught {raised!r} unchanged instead of converting it"
                )
            )
        elif raised not in allowed:
            found.append(
                handler.violation(
                    f"raises {raised}, which is neither one of the five declared errors nor a "
                    f"contract error of {handler.module}"
                )
            )
    return tuple(found)


def _scan(source: str, module: str) -> tuple[tuple[_Handler, ...], tuple[_Violation, ...]]:
    """Return the handlers `source` declares and every violation of the closed vocabulary.

    Args:
        source: The module's text, parsed rather than imported, so a defect is read from the file a
            reviewer opens.
        module: The file name, which selects the module's contract errors and its exemptions.
    """
    tree = ast.parse(source, filename=module)
    scopes = _enclosing_scopes(tree)
    handlers = tuple(
        sorted(
            (
                _handler_of(node, module, scopes.get(id(node)) or "<module>")
                for node in ast.walk(tree)
                if isinstance(node, ast.ExceptHandler)
            ),
            key=lambda handler: handler.lineno,
        )
    )
    violations = tuple(
        violation for handler in handlers for violation in _violations_of(handler)
    )
    return handlers, violations


def _guarded_body(tree: ast.AST, handler: ast.ExceptHandler) -> list[ast.stmt]:
    """Return the statements the `handler` of `tree` guards."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(guard is handler for guard in node.handlers):
            return list(node.body)
    raise AssertionError(f"no try statement guards the handler at line {handler.lineno}")


def _called_names(statements: Iterable[ast.stmt]) -> Iterator[str]:
    """Yield the name of every function called in `statements`, a nested handler's excluded."""
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.ExceptHandler):
            continue
        if isinstance(node, ast.Call):
            yield _call_name(node)
        pending.extend(ast.iter_child_nodes(node))


@lru_cache(maxsize=None)
def _module_source(module: str) -> str:
    """Return one module's text, from the adapter package or the product package above it."""
    directory = Path(targets_package.__file__).parent
    path = directory.parent / module if module == "capture.py" else directory / module
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _package_scan() -> _Scan:
    """Parse every module of the adapter package and judge every handler in it."""
    directory = Path(targets_package.__file__).parent
    modules = tuple(sorted([path.name for path in directory.glob("*.py")] + ["capture.py"]))
    handlers: list[_Handler] = []
    violations: list[_Violation] = []
    handlers_by_module: dict[str, int] = {}
    for module in modules:
        found, bad = _scan(_module_source(module), module)
        handlers_by_module[module] = len(found)
        handlers.extend(found)
        violations.extend(bad)
    return _Scan(modules, handlers_by_module, tuple(handlers), tuple(violations))


def _recovery_counts(
    handlers: Iterable[_Handler],
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    """Return the silent handlers found per ``(module, function)`` and those documented.

    Counts rather than a set of keys: two handlers that raise nothing inside one function share one
    key, so membership alone would call the second one documented. This pair is what
    :data:`_RECOVERING_HANDLERS` is compared against, and the reason a duplicate silent handler
    cannot slip in beside a documented one.
    """
    found = Counter(handler.key for handler in handlers if handler.recovers)
    return found, Counter(_RECOVERING_HANDLERS.keys())


def _raise_names_reachable_from(handler: _Handler) -> _Closure:
    """Return what a ``raise`` inside `handler`'s guarded statements can reach, call by call.

    The exemption in :data:`_CLEANUP_EXEMPTIONS` is only sound if the block it wraps cannot
    propagate anything the adapter has not classified, so that fact is checked rather than asserted
    in prose: the walk starts at the handler's ``try`` body, follows every call that resolves to
    another function of the same module, and collects the exception type of every ``raise`` in what
    it reaches. ``cli.py``'s ``_build_invocation`` renders one argument through ``_render``, which
    substitutes each placeholder through ``_token_value``, which refuses an unknown one with
    ``ValueError`` and a payload it cannot decode through ``_payload_text``'s
    ``TargetContractError``: the two names this returns, and nothing else.
    """
    tree = ast.parse(_module_source(handler.module), filename=handler.module)
    scopes = _enclosing_scopes(tree)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.lineno == handler.lineno
    ]
    if len(nodes) != 1:
        raise AssertionError(
            f"expected one handler at {handler.module}:{handler.lineno}, found {len(nodes)}"
        )
    functions = {
        scopes[id(node)].rpartition(".")[2]: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = list(_called_names(_guarded_body(tree, nodes[0])))
    visited: set[str] = set()
    raised: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited or name not in functions:
            continue
        visited.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Raise) and node.exc is not None:
                raised.add(_raised_name(node))
            elif isinstance(node, ast.Call):
                pending.append(_call_name(node))
    return _Closure(tuple(sorted(visited)), tuple(sorted(raised)))


def test_every_handler_in_the_adapter_layer_converts_to_a_declared_error() -> None:
    """No ``except`` in ``qrcheck/targets/`` lets a library exception out unwrapped: each one
    either names one of the five declared errors in its ``raise`` or passes an already-declared
    error on. A new ``except ImportError: raise`` inside any adapter breaks this, which is what
    makes the conversion rule a test rather than a convention the trigger tests happen to cover."""
    scan = _package_scan()

    assert scan.violations == (), (
        "every except in the adapter layer must convert to a declared error, and these do not: "
        + "; ".join(str(violation) for violation in scan.violations)
    )
    assert len(scan.handlers) >= _MINIMUM_HANDLERS, (
        f"the scan must read the adapter layer: it found {len(scan.handlers)} handlers and the "
        f"floor is {_MINIMUM_HANDLERS}, so it is no longer reading the package it is meant to"
    )


def test_the_scan_discovers_every_module_of_the_adapter_package() -> None:
    """The scan is not vacuous: it discovers the modules from the package itself -- never from a
    hand-written list that a rename would leave stale -- parses all of them, and finds handlers in
    the port and in each of the five adapters. A rule that found nothing in a module would be a
    rule about a file that no longer holds a handler."""
    scan = _package_scan()
    planned = {
        "__init__.py",
        "base.py",
        "capture.py",
        "cli.py",
        "defective.py",
        "http.py",
        "library.py",
        "reference.py",
    }

    # The scan covers every module the package declares and the one shared capture module above
    # it, and the eight that exist today are among them. A superset on the named set, never
    # equality: a new adapter kind is added by creating one module under ``qrcheck/targets/``,
    # and that module must be scanned rather than fail this test for a reason no defect
    # explains.
    directory = Path(targets_package.__file__).parent
    declared = {path.name for path in directory.glob("*.py")} | {"capture.py"}
    assert set(scan.modules) == declared, (
        f"the scan must parse every module the package declares; it read {scan.modules} and the "
        f"package holds {sorted(declared)}"
    )
    assert planned <= set(scan.modules), (
        "the scan must still read the port and the five adapters; it is missing "
        f"{sorted(planned - set(scan.modules))}"
    )
    assert set(scan.handlers_by_module) == set(scan.modules), (
        "every discovered module must have been parsed and its handlers counted; got "
        f"{sorted(scan.handlers_by_module)} for {scan.modules}"
    )
    for module in _HANDLER_MODULES:
        assert scan.handlers_by_module[module] >= 1, (
            f"{module} must hold at least one handler for the closed-vocabulary rule to reach it"
        )
    assert set(_MODULE_CONTRACT_ERRORS) == {"base.py", "cli.py", "http.py"}, (
        "only the three modules whose handlers refuse the framework's own wrong input may name "
        f"anything beyond the five; the table widens {sorted(_MODULE_CONTRACT_ERRORS)}"
    )
    for module, names in _MODULE_CONTRACT_ERRORS.items():
        assert module in scan.modules, (
            f"the contract-error table widens {module!r}, which the package does not declare"
        )
        for name, reason in names:
            assert name not in _DECLARED_ERROR_NAMES, (
                f"{name} is one of the five declared errors and needs no {module} exemption"
            )
            assert reason.strip(), f"the {module} entry for {name} must say why it is not one"


@pytest.mark.parametrize(
    "key",
    sorted(_CLEANUP_EXEMPTIONS),
    ids=[entry[1] for entry in sorted(_CLEANUP_EXEMPTIONS)],
)
def test_an_exempt_cleanup_handler_cannot_leak_an_unconverted_exception(
    key: tuple[str, str, tuple[str, ...]],
) -> None:
    """A handler allowed to re-raise what it caught is the one the table records, it really does
    re-raise, and the exception leaving it is already classified -- proved, not asserted.

    Either every ``raise`` its guarded block can reach is a name its own module may raise, or the
    type it caught is converted to a declared error under a function the entry names. An
    unconverted raise added anywhere on such a path fails here, which is what keeps each exemption a
    fact rather than a hole in the rule."""
    exemption = _CLEANUP_EXEMPTIONS[key]
    scan = _package_scan()
    matching = tuple(handler for handler in scan.handlers if handler.signature == key)

    assert exemption.reason.strip(), f"{key[1]} must say why it may re-raise what it caught"
    assert len(matching) == 1, (
        "every exemption must match exactly one handler that still exists, and there must be no "
        f"stale entry: {key} matches {len(matching)} handlers"
    )
    exempt = matching[0]
    assert exempt.bare_raise, (
        f"{exempt.qualname} is exempt only as a cleanup that re-raises unchanged, so it must "
        "hold a bare raise"
    )
    assert exempt.caught == exemption.caught, (
        f"{exempt.qualname} must catch exactly what its entry records, not a narrowing that would "
        f"leave it exempt for the wrong reason: it catches {exempt.caught}"
    )
    assert not [
        violation
        for violation in scan.violations
        if (violation.module, violation.qualname) == key[:2]
    ], (
        "the exemption must take effect on the handler it names: a violation reported for it "
        "would mean the table's key no longer matches the handler it describes"
    )

    enclosed = _raise_names_reachable_from(exempt)
    allowed = _DECLARED_ERROR_NAMES | _contract_error_names(exempt.module)
    assert set(enclosed.raised) == set(exemption.raised), (
        "the guarded block may only raise what the entry records, or something unconverted "
        f"escapes: the walk found {enclosed.raised} against {exemption.raised}"
    )
    assert set(enclosed.raised) <= allowed, (
        f"the reachable raises must be inside {exempt.module}'s vocabulary: "
        f"{sorted(set(enclosed.raised) - allowed)} are outside it"
    )
    if exemption.reached:
        assert set(exemption.reached) <= set(enclosed.functions), (
            "the walk must reach the helpers the exempt block calls, or the proof above says "
            f"nothing about them: it reached {enclosed.functions}"
        )
    if exemption.converted_by is not None:
        module, converter, error = exemption.converted_by
        assert error in _DECLARED_ERROR_NAMES, (
            f"{converter} must convert {exemption.caught} to a declared error; {error} is not one"
        )
        assert [
            handler
            for handler in scan.handlers
            if handler.key == (module, converter)
            and set(handler.caught) & set(exemption.caught)
            and error in handler.raised
        ], (
            f"{exempt.qualname} re-raises {exemption.caught} to its caller, so {module}'s "
            f"{converter} must convert that type to {error}: no handler there does"
        )
    assert exemption.reached or exemption.converted_by is not None, (
        f"{key[1]} must be sound by the reachable-raise walk or by a converting caller, and its "
        "entry names neither"
    )


def test_every_bare_re_raise_in_the_adapter_layer_is_classified_or_exempt() -> None:
    """The other direction of the table: a handler that re-raises what it caught is either an
    exemption with its proof, or a pass-through whose caught types its own module may raise -- the
    ``except (TargetError, ConfigError, ValueError): raise`` that hands an already-classified error
    on. A bare re-raise of anything else is a violation the scan reports, so one cannot exist
    quietly, and an exemption whose handler has gone fails too."""
    scan = _package_scan()
    bare = tuple(handler for handler in scan.handlers if handler.bare_raise)

    assert bare, "the package holds bare re-raises, or this test asserts nothing"
    assert set(_CLEANUP_EXEMPTIONS) <= {handler.signature for handler in bare}, (
        "every exemption must still match the handler it describes: no handler re-raises under "
        f"{sorted(set(_CLEANUP_EXEMPTIONS) - {handler.signature for handler in bare})}"
    )
    unclassified = [
        handler
        for handler in bare
        if handler.signature not in _CLEANUP_EXEMPTIONS
        and set(handler.caught) - _pass_through_names(handler.module)
    ]
    assert unclassified == [], (
        "a bare re-raise is only legitimate when it hands on an error its own module may raise: "
        + "; ".join(
            f"{handler.module}:{handler.lineno} {handler.qualname}" for handler in unclassified
        )
    )


def test_every_handler_without_a_raise_documents_its_recovery() -> None:
    """A handler that raises nothing is not silently swallowing: it either recovers with the
    documented fallback its reason names or hands the exception to a caller that converts it. The
    table and the tree match in both directions *and in count*, so a new silent handler -- the
    shape a swallowed exception takes -- fails here until it is documented and judged, and so does
    a second one added to a function that already holds a documented silent handler."""
    scan = _package_scan()
    recovering = tuple(handler for handler in scan.handlers if handler.recovers)
    found, documented = _recovery_counts(scan.handlers)

    assert found == documented, (
        "every handler that raises nothing must be documented, once per handler, and every "
        f"documented one must still exist: found {sorted(found.items())} and documented "
        f"{sorted(documented.items())}"
    )
    assert all(reason.strip() for reason in _RECOVERING_HANDLERS.values()), (
        "each documented recovery must say what it does with the exception it caught"
    )
    assert all(handler.caught for handler in recovering), (
        "a bare 'except:' recovers from everything, which no adapter may do"
    )


def test_the_scan_flags_a_raw_re_raise_and_an_undeclared_conversion() -> None:
    """The rule bites: driven over a synthetic adapter it flags exactly the three handlers that let
    a library exception out -- a bare re-raise, a re-raise of the caught exception and a conversion
    to a type outside the vocabulary -- and accepts the three that convert, fall back or pass a
    declared error through. The empty result on the real package above is only worth reading
    because this source does not produce one. It is scanned under the ``http.py`` label, the module
    whose own contract errors are the two ``passing_through`` hands on; the parameterised cases
    below flag that same handler in a module that declares neither of them."""
    handlers, violations = _scan(_OFFENDER_SOURCE, "http.py")

    assert violations, "the offender source must produce violations, or the scan proves nothing"
    assert len(handlers) == 6, (
        f"the synthetic adapter declares six handlers; the scan found {len(handlers)}"
    )
    flagged = tuple(violation.qualname for violation in violations)
    assert flagged == (
        "SyntheticAdapter.escaping_import",
        "SyntheticAdapter.escaping_caught",
        "SyntheticAdapter.escaping_unrelated",
    ), f"the scan must flag the three escapes and nothing else; it flagged {flagged}"
    details = {violation.qualname: violation.detail for violation in violations}
    assert "ImportError" in details["SyntheticAdapter.escaping_import"], (
        "a bare re-raise must be reported with the type it lets out: "
        f"{details['SyntheticAdapter.escaping_import']!r}"
    )
    assert "re-raises the caught 'exc'" in details["SyntheticAdapter.escaping_caught"], (
        "re-raising the caught exception must be named as such: "
        f"{details['SyntheticAdapter.escaping_caught']!r}"
    )
    assert "RuntimeError" in details["SyntheticAdapter.escaping_unrelated"], (
        "a conversion outside the vocabulary must be reported with the type it raises: "
        f"{details['SyntheticAdapter.escaping_unrelated']!r}"
    )
    accepted = {handler.qualname for handler in handlers} - set(flagged)
    assert accepted == {
        "SyntheticAdapter.converting",
        "SyntheticAdapter.falling_back",
        "SyntheticAdapter.passing_through",
    }, f"the three well-behaved handlers must be accepted; the scan accepted {sorted(accepted)}"


@pytest.mark.parametrize(
    ("module", "caught", "flagged"),
    [
        pytest.param("reference.py", "ValueError", True, id="reference-value-error"),
        pytest.param("reference.py", "KeyError", True, id="reference-key-error"),
        pytest.param("reference.py", "ConfigError", True, id="reference-config-error"),
        pytest.param("reference.py", "TargetError", False, id="reference-base-error"),
        pytest.param("reference.py", "TargetRejected", False, id="reference-declared-error"),
        pytest.param("library.py", "ValueError", True, id="library-value-error"),
        pytest.param("defective.py", "KeyError", True, id="defective-key-error"),
        pytest.param(
            "http.py",
            "(TargetError, ConfigError, ValueError)",
            False,
            id="http-declared-and-contract-errors",
        ),
        pytest.param("http.py", "KeyError", True, id="http-key-error"),
        pytest.param("base.py", "KeyError", False, id="base-key-error"),
        pytest.param("cli.py", "ValueError", False, id="cli-value-error"),
        pytest.param("cli.py", "KeyError", True, id="cli-key-error"),
    ],
)
def test_a_bare_raise_may_only_pass_on_what_its_own_module_may_raise(
    module: str, caught: str, flagged: bool
) -> None:
    """A bare ``raise`` is judged against the vocabulary of the module it stands in, never against a
    name admitted everywhere. ``reference.py`` declares no contract errors, so a reference handler
    that caught segno's ``ValueError`` and re-raised it raw lets a library exception out: converting
    it to ``TargetRejected`` is the only thing that makes that name legitimate there, which is why
    refactoring it to a bare ``raise`` must fail. ``http.py``, which refuses
    its own unparseable url with a ``ValueError`` and reports a bad configuration with
    ``ConfigError``, may pass both on unchanged, and the five declared errors plus ``TargetError``
    pass on in every module."""
    handlers, violations = _scan(_BARE_RAISE_SOURCE.format(caught=caught), module)

    assert len(handlers) == 1, (
        f"the source declares one handler; the scan found {len(handlers)} in {module}"
    )
    assert tuple(part.strip() for part in caught.strip("()").split(",")) == handlers[0].caught, (
        f"the handler must catch what this case names; it caught {handlers[0].caught}"
    )
    assert handlers[0].bare_raise, "the case is about a bare raise, and this handler has none"
    if not flagged:
        assert violations == (), (
            f"{module} may raise {caught} itself, so a bare raise of it must be accepted: "
            + "; ".join(str(violation) for violation in violations)
        )
        return
    assert len(violations) == 1, (
        f"{module} may not raise {caught}, so its bare raise must be flagged exactly once: "
        + "; ".join(str(violation) for violation in violations)
    )
    violation = violations[0]
    assert violation.qualname == "SyntheticAdapter.escaping", (
        f"the violation must name the function a reader opens; it names {violation.qualname}"
    )
    for name in handlers[0].caught:
        assert name in violation.detail, (
            f"the violation must name the type that escapes ({name}): {violation.detail!r}"
        )
    assert module in violation.detail, (
        f"the violation must name the module whose vocabulary was exceeded: {violation.detail!r}"
    )


def test_a_second_silent_handler_in_one_function_is_counted_not_absorbed() -> None:
    """A set of ``(module, function)`` keys cannot see a second handler that raises nothing inside a
    function that already holds a documented one, which is the shape a newly swallowed exception
    takes. The comparison is therefore by count, and this source -- two silent handlers in one
    function -- must be counted twice against a table that documents the key not at all."""
    handlers, violations = _scan(_DUPLICATE_RECOVERY_SOURCE, "base.py")

    assert violations == (), (
        "recovering is not a violation, so the source must scan clean: "
        + "; ".join(str(violation) for violation in violations)
    )
    assert len(handlers) == 2, f"the source declares two handlers; the scan found {len(handlers)}"
    assert len({handler.key for handler in handlers}) == 1, (
        "both handlers sit in one function, so the key set holds a single entry -- exactly what "
        "membership cannot look past"
    )
    found, documented = _recovery_counts(handlers)
    assert found == Counter({("base.py", "SyntheticAdapter.recovering_twice"): 2}), (
        f"both silent handlers must be counted under their shared key; counted {sorted(found)}"
    )
    assert found != documented, (
        "one documented key must not satisfy two silent handlers: found "
        f"{sorted(found.items())} against documented {sorted(documented.items())}"
    )


def test_reference_adapter_rejects_at_the_ratio_the_product_measures(
    config_factory: Callable[..., models.Config],
) -> None:
    """The rejection floor is the ratio ``qrcheck.decode.consensus.contrast_ratio`` measures.

    That function is the product's one implementation of the WCAG 2.1 ratio -- ``RN-CONTRAST``
    asserts with it and ``IV-BAD-OPTION`` relies on the rejection built on it -- so a floor bound to
    exactly the ratio it returns must accept the request, and one bound a single ulp above must
    refuse it with the documented message. A second copy of the formula inside the target would put
    this colour pair on the wrong side of one of those two bounds.
    """
    foreground, background = (0x88, 0x88, 0x88), (255, 255, 255)
    measured = consensus.contrast_ratio(foreground, background)
    request = build_request(
        _UNUSUAL_TEXT.encode("utf-8"),
        foreground_rgb=foreground,
        background_rgb=background,
    )

    at_the_floor = _reference_adapter(config_factory(), contrast_floor=measured)
    symbol = at_the_floor.generate(request)
    assert symbol.image_bytes, (
        "a ratio equal to a floor of its own value must encode: the contract refuses a ratio "
        "below the floor, never one at it"
    )

    just_above = math.nextafter(measured, math.inf)
    above_the_floor = _reference_adapter(config_factory(), contrast_floor=just_above)
    with pytest.raises(TargetRejected) as raised:
        above_the_floor.generate(request)

    assert str(raised.value) == (
        f"contrast ratio {measured:.1f} is below the declared floor {just_above:.1f}"
    ), (
        "the rejection message reaches decoded.rejection verbatim, so it is public API: "
        f"{str(raised.value)!r}"
    )


@pytest.mark.parametrize(
    "name",
    (
        "_contrast_ratio",
        "_relative_luminance",
        "_linear_channel",
        "_SRGB_KNEE",
        "_SRGB_SLOPE",
        "_SRGB_OFFSET",
        "_SRGB_SCALE",
        "_SRGB_GAMMA",
        "_LUMINANCE_WEIGHTS",
        "_CONTRAST_OFFSET",
    ),
)
def test_reference_adapter_holds_no_second_copy_of_the_contrast_formula(name: str) -> None:
    """The WCAG ratio has one implementation, :mod:`qrcheck.decode.consensus`.

    The ``_SRGB_*`` constants and the ``_linear_channel``/``_relative_luminance``/
    ``_contrast_ratio`` helpers here would be a second implementation: they would agree with
    :mod:`qrcheck.decode.consensus` only because both copy the same numbers, so retuning either
    side would silently decouple the floor ``IV-BAD-OPTION`` measures from the floor this target
    rejects at. The target reaches the one implementation at the point of use instead, which
    :func:`test_reference_adapter_rejects_at_the_ratio_the_product_measures` pins from the other
    side.
    """
    assert not hasattr(reference_module, name), (
        f"qrcheck.targets.reference must not define {name}: a second copy of the WCAG ratio "
        "decouples the floor IV-BAD-OPTION measures from the floor this target rejects at"
    )

