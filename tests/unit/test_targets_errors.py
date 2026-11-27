"""The adapter layer's failure vocabulary is the five declared errors, and no more.

``qrcheck/targets/base.py`` declares exactly five errors, and the port's contract is that an
adapter raises nothing else: every ``except`` inside ``qrcheck/targets/`` converts a library
exception into one of them (AAP §0.3.4 "Error taxonomy", §0.14.1's conversion list). Each trigger
is exercised through the **real** adapter code path, one test per trigger: a missing module or
attribute, an absent executable, a refused loopback port and an unsupported protocol are
``TargetUnavailable``; the configured ``timeout_s`` quoted in the message is ``TargetTimeout``,
while a connect timeout is ``TargetUnavailable``; a declared rejection keeps the target's text
byte for byte in ``TargetRejected``; an empty body, undecodable base64, bytes that are not an
image, a missing ``image_field`` and a non-zero exit that declared no rejection are
``TargetContractError``; and ``TargetUnsupported`` is raised *before any I/O*.

Two distinctions the module exists to hold: ``TargetRejected`` versus ``TargetContractError``
decides whether ``IV-*`` passes, so misclassifying one as the other is a report defect and not
only an exception-type defect; and the five errors are siblings -- every one a ``TargetError``,
none a subclass of another -- so a caller's ``except`` order cannot reclassify a failure. The
HTTP adapter gives a non-2xx status no declared-rejection spelling, which the comment on that
test records."""

from __future__ import annotations

import base64
import dataclasses
import errno
import json
import math
import os
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

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

#: The pinned version the over-capacity probe encodes at (AAP §0.8.1's ``IV-OVERCAPACITY``
#: request uses the same one). A version, not a threshold: it is a coordinate in the
#: table the capacity figure is read from.
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

#: The five declared errors, in the order AAP §0.3.4 tabulates them.
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
# the import path is missing" (AAP §0.3.4)


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

    monkeypatch.setattr(cli_module.os, "pipe", failing_pipe)
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


# TargetTimeout -- "no response within targets.<name>.timeout_s" (AAP §0.3.4)


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
# framework code reaches the runner as itself (AAP §0.3.4 "Error taxonomy", §0.6.6 row 11)


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



# TargetContractError -- "response present but unusable" (AAP §0.3.4)


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
        "the rejection message is public API (AAP §0.8.1, the IV-BAD-OPTION (d) row): "
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
    """cc121-F02: the WCAG ratio has one owner, so this module must not re-derive it.

    The ``_SRGB_*`` constants and the ``_linear_channel``/``_relative_luminance``/
    ``_contrast_ratio`` helpers this module used to carry were a second implementation: they agreed
    with :mod:`qrcheck.decode.consensus` only because both copied the same numbers, so retuning
    either side would silently decouple the floor ``IV-BAD-OPTION`` measures from the floor this
    target rejects at. The target reaches the one implementation at the point of use instead, which
    :func:`test_reference_adapter_rejects_at_the_ratio_the_product_measures` pins from the other
    side.
    """
    assert not hasattr(reference_module, name), (
        f"qrcheck.targets.reference must not define {name}: a second copy of the WCAG ratio "
        "decouples the floor IV-BAD-OPTION measures from the floor this target rejects at"
    )

