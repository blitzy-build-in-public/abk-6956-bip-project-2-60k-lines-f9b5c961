"""The ``cli`` adapter: an out-of-process QR generator driven as an argv-only subprocess.

A ``type = "cli"`` target is a program run once per request, and this module owns the exchange: the
command line, where the artefact appears, how a non-zero exit is read and how metadata is read. No
shell is interposed: the argument vector is a list handed to ``subprocess.run(argv, shell=False,
timeout=...)``, so a payload carrying ``;`` or ``$(...)`` stays one literal argv word, and the
timeout kills the child.

Both of the child's output streams are captured through the documented byte bound
(:data:`_MAX_CAPTURED_STREAM_BYTES`): a target that writes more than that to its stdout or stderr
is refused with ``TargetContractError``, because the output of an untrusted target may not grow
this process past a number no legitimate artefact approaches.

``payload_arg`` is either a placeholder template (``{}`` as strict UTF-8 text, ``{payload_b64}``,
``{payload_utf8}``, ``{ec_level}``, ``{format}``, ``{size}``, ``{margin}``, ``{fg}``, ``{bg}``) or a
bare ``-``-prefixed flag passed as one argv word with the raw payload on the child's stdin. No
placeholder carries a version pin, a mask pin or a logo overlay, so such a request is refused with
``TargetUnsupported`` before any I/O. A placeholder-bearing ``output_arg`` receives a
``symbol.png``/``symbol.svg`` path inside a ``tempfile.mkdtemp(prefix="qrcheck-")`` directory whose
prefix is a product-wide contract, and the directory is removed in a ``finally`` block.

``returns`` is ``file_path``, ``bytes_path``, ``raw_bytes`` or ``base64``. A non-zero exit is a
declared rejection only for a JSON object with a string ``error`` member or a line beginning
``rejected:``, else it is a ``TargetContractError``, because the ``IV-*`` checks pass on a declared
rejection and a fabricated one would make them vacuous.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from qrcheck.config.models import RETURNS_KINDS, TEMPLATE_FIELDS, CLITarget, ReturnsKind
from qrcheck.targets.base import (
    OUTPUT_FORMATS,
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    HEX_PREFIX_LIMIT,
    TargetContractError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    TargetUnsupported,
    apply_metadata_map,
    declared_capabilities,
    decode_returned_image,
    ensure_image_bytes,
    hex_prefix,
    require_capabilities,
)

__all__ = ["CliAdapter"]

# The prefix of every temporary directory this adapter creates. CL-ARTIFACTS
# scans tempfile.gettempdir() for this prefix before and after a run, so it is a
# contract shared with qrcheck/checks/cleanup.py rather than a private habit.
TEMP_DIR_PREFIX: Final[str] = "qrcheck-"

# The artefact's name inside that directory. The extension carries the requested
# format, so a target which infers its writer from the path is not misled.
_TEMP_ARTEFACT_STEM: Final[str] = "symbol"

# A placeholder is a brace-delimited name: "{payload_b64}", or "{}" for the bare
# payload. One pattern serves both argument templates, so a template carrying an
# unknown placeholder is refused rather than passed through literally.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z0-9_]*)\}")

# The vocabulary that pattern accepts, in diagnostic order: the bare "{}" followed by the eight
# named placeholders of the HTTP adapter's ``request_template``, which both transports share so one
# mental model covers them both. It is derived from ``models.TEMPLATE_FIELDS`` rather than restated,
# so no list can disappear from one transport while it stays in the other. It has two readers: the
# configuration schema refuses an unknown placeholder at load time -- exit 2, naming the key and the
# token -- and this adapter keeps its own ``ValueError`` for the belt-and-braces case of a target
# record built without the loader, exactly as the HTTP adapter keeps its own.
_TEMPLATE_TOKENS: Final[tuple[str, ...]] = (
    "{}",
    *(f"{{{field}}}" for field in TEMPLATE_FIELDS),
)

# Request fields no placeholder can convey, with the capability that governs each. The words of
# ``command`` are spliced literally and the vocabulary above has no version, mask or logo token --
# the schema refuses a placeholder in a ``command`` word outright -- so a request that pins one is
# refused before any I/O rather than run without it, the decision ``qrcheck/targets/http.py``
# already makes for its request template.
_INEXPRESSIBLE_FIELDS: Final[tuple[tuple[str, Capability], ...]] = (
    ("version", Capability.version_pin),
    ("mask", Capability.mask_pin),
    ("logo_overlay_path", Capability.logo_overlay),
)

# The line prefix by which a target declares a rejection (encoding 2).
REJECTION_PREFIX: Final[str] = "rejected:"

# The JSON member by which a target declares a rejection (encoding 1).
REJECTION_JSON_FIELD: Final[str] = "error"

# How much of a failed target's captured output a contract-error message keeps.
DIAGNOSTIC_TAIL_CHARS: Final[int] = 2000

# The most this adapter keeps from either stream of the child it runs. A cli target is untrusted
# (AAP §0.1.3: it "may return garbage, lie about metadata, hang or crash"), so a stream that passes
# this bound is a contract error rather than unbounded growth in the runner's memory. The number is
# the 16 MiB the http adapter reads from one response, because no legitimate artefact -- a PNG or
# an SVG of a Model 2 symbol -- approaches either.
_MAX_CAPTURED_STREAM_BYTES: Final[int] = 16 * 1024 * 1024

# How many bytes one read from a captured stream asks for: one pipe buffer, so a flood costs a
# bounded number of reads rather than one call per byte.
_READ_CHUNK_BYTES: Final[int] = 64 * 1024

# How long a capture waits for its drain thread once the child has exited: the wait is for the
# bytes already in the pipe, never for the child, so a well-behaved child finishes far inside it.
_DRAIN_JOIN_TIMEOUT_S: Final[float] = 5.0

# stdout carries the artefact itself for these return kinds, so a metadata_map
# cannot also read the target's metadata from it.
_STDOUT_RETURNS: Final[frozenset[str]] = frozenset({"base64", "raw_bytes"})

# The colour pair a request carries when it asks for nothing in particular. Only
# a request which changes one of them needs the `color` capability, so a target
# declaring no colour support still serves every ordinary request.
_DEFAULT_FOREGROUND: Final[tuple[int, int, int]] = (0, 0, 0)
_DEFAULT_BACKGROUND: Final[tuple[int, int, int]] = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class _Invocation:
    """What one ``generate`` call runs, and where its artefact will appear.

    Built before any subprocess exists, so a configuration or payload fault is raised without
    spawning anything. ``temp_dir`` is the directory created for this call alone, and it is what
    ``generate`` removes on every path.
    """

    argv: tuple[str, ...]
    stdin_payload: bytes | None
    artefact_path: Path | None
    temp_dir: Path | None


class _BoundedStreamCapture:
    """Capture at most :data:`_MAX_CAPTURED_STREAM_BYTES` of one output stream of a child process.

    One instance owns one ``os.pipe()`` and one daemon thread. ``subprocess.run`` is handed
    :attr:`write_fd` as its ``stdout=`` or ``stderr=``, so the child writes straight into the pipe,
    and the thread reads the other end until the child closes it, keeping at most the documented
    bound. A stream that passes the bound sets :attr:`overflowed` and closes its read end, so the
    child's next write fails instead of this process buffering output it will never use.

    :meth:`finish` releases this process's ends of the pipe once the child has exited, and is safe
    to call more than once, so every path -- success, timeout, spawn failure -- can call it without
    knowing which one it is on.
    """

    def __init__(self, name: str) -> None:
        """Create the pipe and start draining it.

        Args:
        name: ``"stdout"`` or ``"stderr"``, the stream this object holds, for diagnostics.

        Raises:
        OSError: The pipe itself could not be created.
        RuntimeError: The drain thread could not be started.
        """
        self._name: str = name
        self._buffer: bytearray = bytearray()
        self._received: int = 0
        self._overflowed: bool = False
        self._read_fd: int = -1
        self._write_fd: int = -1
        self._read_fd, self._write_fd = os.pipe()
        self._thread: threading.Thread = threading.Thread(
            target=self._drain,
            name=f"qrcheck-capture-{name}",
            daemon=True,
        )
        try:
            self._thread.start()
        except RuntimeError:
            self._close_read_fd()
            self._close_write_fd()
            raise

    @property
    def write_fd(self) -> int:
        """Return the descriptor to hand ``subprocess.run`` as this stream's destination."""
        return self._write_fd

    @property
    def name(self) -> str:
        """Return the stream's name, ``"stdout"`` or ``"stderr"``."""
        return self._name

    @property
    def bytes(self) -> bytes:
        """Return the bytes captured from this stream, at most the documented bound of them."""
        return bytes(self._buffer)

    @property
    def received(self) -> int:
        """Return how many bytes arrived on this stream, the over-bound ones included."""
        return self._received

    @property
    def overflowed(self) -> bool:
        """Return whether this stream delivered more than the documented bound."""
        return self._overflowed

    def finish(self) -> None:
        """Release this process's ends of the pipe once the child has exited.

        The write end is closed first, so a drain thread still reading sees end-of-file and returns;
        the thread is then joined for :data:`_DRAIN_JOIN_TIMEOUT_S`. The read end is closed only
        once the thread has ended, because closing a descriptor another thread may still be reading
        is not safe, and a thread still blocked there is left with its descriptor and its daemon
        status. Idempotent: a second call finds the descriptors already closed.
        """
        self._close_write_fd()
        self._thread.join(_DRAIN_JOIN_TIMEOUT_S)
        if not self._thread.is_alive():
            self._close_read_fd()

    def _drain(self) -> None:
        """Read this stream until the child closes it or the bound is passed."""
        while True:
            try:
                chunk = os.read(self._read_fd, _READ_CHUNK_BYTES)
            except OSError:
                # A read end closed on the over-bound path, or a descriptor the kernel refused:
                # either way nothing more can be read from this stream.
                return
            if not chunk:
                return
            self._received += len(chunk)
            room = _MAX_CAPTURED_STREAM_BYTES - len(self._buffer)
            if room > 0:
                self._buffer += chunk[:room]
            if self._received > _MAX_CAPTURED_STREAM_BYTES:
                self._overflowed = True
                self._close_read_fd()
                return

    def _close_read_fd(self) -> None:
        """Close the read end once, if it is still open."""
        read_fd, self._read_fd = self._read_fd, -1
        if read_fd < 0:
            return
        with contextlib.suppress(OSError):
            os.close(read_fd)

    def _close_write_fd(self) -> None:
        """Close this process's copy of the write end once, if it is still open."""
        write_fd, self._write_fd = self._write_fd, -1
        if write_fd < 0:
            return
        with contextlib.suppress(OSError):
            os.close(write_fd)


class CliAdapter:
    """Drive a configured ``command`` as one argv-only subprocess per request.

    The adapter holds no process and no temporary directory between calls, and raises only the five
    :class:`~qrcheck.targets.base.TargetError` subclasses, because every library exception a
    subprocess can produce is converted where it is caught.
    """

    def __init__(self, config: CLITarget, *, name: str | None = None) -> None:
        """Prepare the adapter for `config` without spawning anything.

        Raises:
        ValueError: An empty ``command``, a NUL in a configured string, a ``returns`` outside its
        set, or a ``metadata_map`` beside a stdout-based ``returns``.
        """
        self.name: str = name if name is not None else config.type
        self.capabilities: frozenset[Capability] = declared_capabilities(config)
        self._command: tuple[str, ...] = tuple(config.command)
        self._payload_arg: str = config.payload_arg
        self._output_arg: str = config.output_arg
        self._returns: ReturnsKind = _checked_returns(config.returns, self.name)
        self._metadata_map: dict[str, str] = dict(config.metadata_map)
        self._timeout_s: float = config.timeout_s

        if not self._command:
            raise ValueError(
                f"targets.{self.name}.command must list at least one argument"
            )
        for index, word in enumerate(self._command):
            _check_no_nul(word, self.name, f"command[{index}]")
        _check_no_nul(self._payload_arg, self.name, "payload_arg")
        _check_no_nul(self._output_arg, self.name, "output_arg")
        if self._metadata_map and self._returns in _STDOUT_RETURNS:
            raise ValueError(
                f"targets.{self.name}.metadata_map reads the target's JSON metadata from "
                f"stdout, but returns = {self._returns!r} delivers the artefact there; "
                "configure a path-based returns or drop metadata_map"
            )

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Run the target once for `request` and return the symbol it produced.

        Raises:
        TargetUnsupported: An undeclared capability, or a pin or overlay no template carries.
        TargetUnavailable: The program could not be spawned, or the scratch directory its
        artefact needs could not be created.
        TargetTimeout: The target outlived ``timeout_s`` and was killed.
        TargetRejected: The target declared a client-side rejection.
        TargetContractError: It exited without a usable artefact or a declared rejection, or wrote
        more than the documented byte bound to one of its streams.
        ValueError: An unknown placeholder, or a payload the template cannot take.
        """
        require_capabilities(self, _required_capabilities(request))
        self._reject_inexpressible(request)
        invocation = self._build_invocation(request)
        try:
            completed, elapsed_ms = self._run(invocation)
            if completed.returncode != 0:
                self._raise_for_failure(completed)
            image_bytes = self._read_artefact(completed, invocation.artefact_path, request)
            metadata = self._read_metadata(completed)
        finally:
            if invocation.temp_dir is not None:
                shutil.rmtree(invocation.temp_dir, ignore_errors=True)

        return GeneratedSymbol(
            image_bytes=image_bytes,
            output_format=request.output_format,
            declared_version=_declared_int(metadata.get("version")),
            declared_ec=_declared_text(metadata.get("ec")),
            declared_mask=_declared_int(metadata.get("mask")),
            declared_mode=_declared_text(metadata.get("mode")),
            elapsed_ms=elapsed_ms,
        )

    def _run(self, invocation: _Invocation) -> tuple[subprocess.CompletedProcess[bytes], float]:
        """Spawn the child once and capture a bounded prefix of each of its two output streams.

        The child is run as it always was -- one argv list, no shell, the configured ``timeout_s``,
        no inherited environment -- but its stdout and stderr are the two bounded pipes of
        :class:`_BoundedStreamCapture` rather than ``capture_output``, so no volume of output can
        grow this process past :data:`_MAX_CAPTURED_STREAM_BYTES` however much of ``timeout_s`` is
        left. Both pipes are released on every path, before any error is converted.

        Args:
        invocation: The command line, stdin bytes and artefact path one ``generate`` call runs.

        Returns:
        The completed process, rebuilt over the bytes captured from both streams, and the duration
        of the spawn in milliseconds measured as soon as the child exited.

        Raises:
        TargetTimeout: The child outlived ``timeout_s`` and was killed; the message names the
        configured timeout.
        TargetUnavailable: The child could not be spawned, or no pipe could be created for its
        output.
        TargetContractError: One of the child's streams passed the documented byte bound, which
        makes its output unusable whatever else happened -- an over-bound stream outranks a
        timeout.
        """
        stdout_capture, stderr_capture = self._open_captures()
        started = time.perf_counter()
        try:
            try:
                completed = subprocess.run(
                    list(invocation.argv),
                    shell=False,
                    stdin=subprocess.DEVNULL if invocation.stdin_payload is None else None,
                    input=invocation.stdin_payload,
                    stdout=stdout_capture.write_fd,
                    stderr=stderr_capture.write_fd,
                    text=False,
                    timeout=self._timeout_s,
                    check=False,
                    env=None,
                )
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                stdout_capture.finish()
                stderr_capture.finish()
        except subprocess.TimeoutExpired:
            self._raise_for_overflowed_stream(stdout_capture, stderr_capture)
            raise TargetTimeout(
                f"target {self.name!r} ({invocation.argv[0]!r}) did not answer within "
                f"timeout_s = {self._timeout_s:g} s and was terminated"
            ) from None
        except OSError as exc:
            # Only a spawn failure reaches here: subprocess.run raises OSError from execvp
            # alone, before any pipe is read.
            self._raise_for_overflowed_stream(stdout_capture, stderr_capture)
            reason = exc.strerror or exc.__class__.__name__
            raise TargetUnavailable(
                f"target {self.name!r} could not be spawned: {invocation.argv[0]!r} "
                f"failed ({reason}); check targets.{self.name}.command"
            ) from None
        self._raise_for_overflowed_stream(stdout_capture, stderr_capture)
        return (
            subprocess.CompletedProcess(
                completed.args,
                completed.returncode,
                stdout_capture.bytes,
                stderr_capture.bytes,
            ),
            elapsed_ms,
        )

    def _open_captures(self) -> tuple[_BoundedStreamCapture, _BoundedStreamCapture]:
        """Return one bounded capture per output stream of the child about to be spawned.

        Opening them is part of spawning: with nowhere to write, the child is not run at all, which
        is the same trigger as any other spawn failure.

        Raises:
        TargetUnavailable: A pipe or a drain thread could not be created, so the target was not run.
        """
        captures: list[_BoundedStreamCapture] = []
        try:
            captures.append(_BoundedStreamCapture("stdout"))
            captures.append(_BoundedStreamCapture("stderr"))
        except (OSError, RuntimeError) as exc:
            for capture in captures:
                capture.finish()
            raise TargetUnavailable(
                f"target {self.name!r} could not be spawned: its output could not be captured "
                f"({exc}); check targets.{self.name}.command"
            ) from None
        return captures[0], captures[1]

    def _raise_for_overflowed_stream(self, *captures: _BoundedStreamCapture) -> None:
        """Refuse a child stream that passed the documented byte bound.

        Args:
        captures: The two captures of the child that just exited, in stream order.

        Raises:
        TargetContractError: One stream delivered more than :data:`_MAX_CAPTURED_STREAM_BYTES`. The
        message names the stream, the bound and how much arrived, and the first
        :data:`~qrcheck.targets.base.HEX_PREFIX_LIMIT` bytes travel on ``body_prefix`` -- the same
        evidence shape a non-image body carries (AAP §0.6.6 row 8).
        """
        for capture in captures:
            if not capture.overflowed:
                continue
            raise TargetContractError(
                f"target {self.name!r} wrote more than the "
                f"{_MAX_CAPTURED_STREAM_BYTES}-byte bound to its {capture.name} "
                f"({capture.received} bytes arrived), so its output is unusable; the stream was "
                "closed at the bound",
                body_prefix=capture.bytes[:HEX_PREFIX_LIMIT],
            )

    def _reject_inexpressible(self, request: GenerateRequest) -> None:
        """Refuse a request neither configured argument template can convey.

        ``version``, ``mask`` and ``logo_overlay_path`` have no placeholder and the words of
        ``command`` are spliced literally, so nothing this adapter runs can carry them.

        Raises:
        TargetUnsupported: A version or mask pin, or a logo overlay. Raised before any I/O.
        """
        for field_name, capability in _INEXPRESSIBLE_FIELDS:
            if getattr(request, field_name) is None:
                continue
            require_capabilities(self, capability)
            raise TargetUnsupported(
                f"adapter {self.name!r} cannot express {field_name} over its argument "
                f"templates: no placeholder in {list(_TEMPLATE_TOKENS)} carries it"
            )

    def _build_invocation(self, request: GenerateRequest) -> _Invocation:
        """Assemble the command line and the artefact path for `request`.

        Nothing is spawned here; the only I/O is the temporary directory a path-bearing
        ``output_arg`` needs.

        Raises:
        ValueError: ``payload_arg`` is neither a template nor a flag, or ``output_arg`` names no
        path.
        TargetUnavailable: No scratch directory for the artefact could be created, so the target
        was not run.
        TargetContractError: The payload cannot be interpolated as UTF-8 text.
        """
        words, stdin_payload = self._payload_argument(request)
        argv = [*self._command, *words]
        template = self._output_arg
        if not template:
            return _Invocation(tuple(argv), stdin_payload, None, None)
        if not _TOKEN_PATTERN.search(template):
            raise ValueError(
                f"targets.{self.name}.output_arg = {template!r} names no artefact path: it "
                "must carry a placeholder such as '{}'"
            )
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))
        except OSError as exc:
            reason = exc.strerror or exc.__class__.__name__
            raise TargetUnavailable(
                f"target {self.name!r} was not run: no scratch directory for its artefact "
                f"could be created ({reason}); check targets.{self.name}.command"
            ) from None
        try:
            artefact_path = temp_dir / f"{_TEMP_ARTEFACT_STEM}.{request.output_format}"
            argv.append(self._render(template, request, "output_arg", path=artefact_path))
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return _Invocation(tuple(argv), stdin_payload, artefact_path, temp_dir)

    def _payload_argument(
        self, request: GenerateRequest
    ) -> tuple[tuple[str, ...], bytes | None]:
        """Return the payload's argv words and, for the stdin form, the bytes to pipe.

        Raises:
        ValueError: ``payload_arg`` is neither a placeholder template nor a flag.
        TargetContractError: A text placeholder was given a payload that is not UTF-8-decodable.
        """
        template = self._payload_arg
        if _TOKEN_PATTERN.search(template):
            return (self._render(template, request, "payload_arg"),), None
        if template.startswith("-"):
            return (template,), request.payload
        raise ValueError(
            f"targets.{self.name}.payload_arg = {template!r} is neither a placeholder "
            "template nor a flag starting with '-'; the configuration schema accepts only "
            "those two shapes"
        )

    def _render(
        self,
        template: str,
        request: GenerateRequest,
        key: str,
        *,
        path: Path | None = None,
    ) -> str:
        """Substitute every placeholder in `template` from `request`.

        Args:
        template: The configured argument template.
        request: The request being rendered.
        key: ``payload_arg`` or ``output_arg``, for the diagnostic.
        path: The artefact path that ``{}`` stands for.

        Raises:
        ValueError: A placeholder is outside the token vocabulary.
        TargetContractError: A text placeholder needs a UTF-8 payload and has none.
        """

        def substitute(match: re.Match[str]) -> str:
            return self._token_value(match.group(1), request, key, path)

        return _TOKEN_PATTERN.sub(substitute, template)

    def _token_value(
        self,
        token: str,
        request: GenerateRequest,
        key: str,
        path: Path | None,
    ) -> str:
        """Return the text one placeholder stands for.

        Raises:
        ValueError: `token` is outside the vocabulary.
        TargetContractError: A text placeholder needs a UTF-8 payload.
        """
        if not token:
            if path is not None:
                return str(path)
            return _payload_text(request.payload, self.name, key)
        if token == "payload_b64":
            return base64.b64encode(request.payload).decode("ascii")
        if token == "payload_utf8":
            return _payload_text(request.payload, self.name, key)
        if token == "ec_level":
            return request.ec_level
        if token == "format":
            return request.output_format
        if token == "size":
            return "" if request.module_size_px is None else str(request.module_size_px)
        if token == "margin":
            return (
                "" if request.quiet_zone_modules is None else str(request.quiet_zone_modules)
            )
        if token == "fg":
            return _hex_colour(request.foreground_rgb)
        if token == "bg":
            return _hex_colour(request.background_rgb)
        raise ValueError(
            f"targets.{self.name}.{key} carries the unknown placeholder '{{{token}}}'; the "
            f"placeholders are {', '.join(_TEMPLATE_TOKENS[:-1])} and {_TEMPLATE_TOKENS[-1]}"
        )

    def _raise_for_failure(self, completed: subprocess.CompletedProcess[bytes]) -> None:
        """Raise what a non-zero exit means: a declared rejection, else a contract error.

        A rejection is raised only for one of the two declared spellings, because a fabricated one
        would make the ``IV-*`` checks vacuous.

        Raises:
        TargetRejected: The target declared a rejection, in its own words.
        TargetContractError: It exited non-zero without declaring one.
        """
        stdout_text = _decoded(completed.stdout)
        stderr_text = _decoded(completed.stderr)
        rejection = _declared_rejection(stderr_text, stdout_text)
        if rejection is not None:
            raise TargetRejected(rejection)
        raise TargetContractError(
            f"target {self.name!r} exited with status {completed.returncode} and declared no "
            f"rejection; stderr tail: {_tail(stderr_text)!r}; "
            f"stdout tail: {_tail(stdout_text)!r}",
            body_prefix=completed.stdout,
        )

    def _read_artefact(
        self,
        completed: subprocess.CompletedProcess[bytes],
        artefact_path: Path | None,
        request: GenerateRequest,
    ) -> bytes:
        """Return the artefact a zero exit delivered, under the configured `returns`.

        Raises:
        TargetContractError: No artefact sits where ``returns`` says to read it, or the bytes are
        not the format the request asked for.
        """
        if self._returns in _STDOUT_RETURNS:
            data = decode_returned_image(completed.stdout, self._returns)
        elif artefact_path is not None:
            data = decode_returned_image(artefact_path, self._returns)
        elif self._returns == "bytes_path":
            # No output_arg was configured, so the artefact sits at the path the
            # target chose; the target reports that path on its last stdout line.
            reported = _last_nonempty_line(_decoded(completed.stdout))
            if reported is None:
                raise TargetContractError(
                    f"returns = 'bytes_path' but target {self.name!r} neither reported an "
                    "artefact path on stdout nor was given an output_arg to write"
                )
            data = decode_returned_image(reported, self._returns)
        else:
            raise TargetContractError(
                f"targets.{self.name}.output_arg is empty, so no artefact path was passed to "
                "the target, but returns = 'file_path' reads the path the framework supplies"
            )
        return ensure_image_bytes(data, request.output_format)

    def _read_metadata(
        self, completed: subprocess.CompletedProcess[bytes]
    ) -> dict[str, object]:
        """Return the metadata the target declared, projected through `metadata_map`.

        Raises:
        TargetContractError: stdout is not the one JSON object a configured ``metadata_map`` reads,
        or a configured field is absent from it.
        """
        if not self._metadata_map:
            return {}
        raw = _json_object(completed.stdout, self.name, sorted(self._metadata_map))
        return apply_metadata_map(raw, self._metadata_map)

    def close(self) -> None:
        """Release the adapter, which holds nothing between calls.

        Each :meth:`generate` owns its subprocess and removes its temporary directory before
        returning or raising, so a second call is the same no-op as the first. The method exists
        because the runner closes every adapter once the run ends.
        """
        return None

    def __repr__(self) -> str:
        """Render the adapter's identity for a test's or a traceback's diagnostics."""
        return (
            f"CliAdapter(name={self.name!r}, command={list(self._command)!r}, "
            f"returns={self._returns!r})"
        )


def _required_capabilities(request: GenerateRequest) -> frozenset[Capability]:
    """Return the capabilities `request` needs the adapter to have declared.

    Only what the request asks for is gated, and ``ec_level`` deliberately is not: a check asserting
    level conformance declares the capability itself and is skipped first.

    Raises:
    ValueError: The request asks for a format outside ``OUTPUT_FORMATS`` -- a framework defect.
    """
    try:
        needed = {Capability(request.output_format)}
    except ValueError:
        raise ValueError(
            "a request may ask for output_format in "
            f"{', '.join(OUTPUT_FORMATS)}; got {request.output_format!r}"
        ) from None
    if request.module_size_px is not None:
        needed.add(Capability.module_size)
    if request.quiet_zone_modules is not None:
        needed.add(Capability.quiet_zone)
    if (request.foreground_rgb, request.background_rgb) != (
        _DEFAULT_FOREGROUND,
        _DEFAULT_BACKGROUND,
    ):
        needed.add(Capability.color)
    return frozenset(needed)


def _checked_returns(returns: str, name: str) -> ReturnsKind:
    """Return `returns` when it is one of the four delivery modes.

    Raises:
    ValueError: It is outside :data:`qrcheck.config.models.RETURNS_KINDS`, which means a
    configuration record was built by hand.
    """
    if isinstance(returns, str) and returns in RETURNS_KINDS:
        return cast(ReturnsKind, returns)
    raise ValueError(
        f"targets.{name}.returns must be one of {', '.join(RETURNS_KINDS)}; got {returns!r}"
    )


def _check_no_nul(value: str, name: str, key: str) -> None:
    """Refuse a configured string carrying a NUL, which no argv word can hold.

    ``execve`` cannot transport one and ``subprocess`` raises a bare ``ValueError`` for it, so
    catching it at construction keeps it from surfacing mid-run as an unexplained framework fault.

    Raises:
    ValueError: `value` contains a NUL character.
    """
    if "\x00" in value:
        raise ValueError(
            f"targets.{name}.{key} carries a NUL character, which cannot be passed as an "
            "argv word"
        )


def _hex_colour(rgb: tuple[int, int, int]) -> str:
    """Return a colour tuple as the lowercase ``#rrggbb`` text of a target option."""
    red, green, blue = rgb
    return f"#{red:02x}{green:02x}{blue:02x}"


def _payload_text(payload: bytes, name: str, key: str) -> str:
    """Return `payload` as UTF-8 text for a text placeholder.

    Raises:
    TargetContractError: The payload is not strictly UTF-8-decodable, or carries a NUL no argv word
    can transport; the message names the binary-safe placeholder that would work.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise TargetContractError(
            f"targets.{name}.{key} interpolates the payload as UTF-8 text, but the "
            f"{len(payload)}-byte payload is not valid UTF-8 (it starts with "
            f"{hex_prefix(payload)}); use {{payload_b64}} for a binary-safe payload"
        ) from None
    if "\x00" in text:
        raise TargetContractError(
            f"targets.{name}.{key} interpolates the payload as UTF-8 text, but the "
            f"{len(payload)}-byte payload carries a NUL that no argv word can hold; use "
            "{payload_b64} for a binary-safe payload"
        )
    return text


def _decoded(stream: bytes | None) -> str:
    """Return captured output as text for diagnostics and rejection detection.

    A target's output is untrusted and need not be UTF-8, so undecodable bytes are replaced rather
    than raised: classifying the exit status must not depend on the encoding of a diagnostic.
    """
    if not stream:
        return ""
    return stream.decode("utf-8", errors="replace")


def _last_nonempty_line(text: str) -> str | None:
    """Return the last line of `text` carrying content, without its line ending."""
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return None


def _json_error_member(line: str) -> str | None:
    """Return the string member ``error`` of a JSON-object line, or ``None``.

    A line that is not a JSON object, or whose ``error`` member is not a string, is not a declared
    rejection, so the caller falls through to the next spelling.
    """
    if not line.lstrip().startswith("{"):
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    message = parsed.get(REJECTION_JSON_FIELD)
    return message if isinstance(message, str) else None


def _declared_rejection(stderr_text: str, stdout_text: str) -> str | None:
    """Return the rejection a target declared, or ``None`` when it declared none.

    Both streams are inspected, stderr before stdout; the JSON spelling is tried on both before the
    ``rejected:`` spelling, so a target writing one to each is classified by the more explicit form.
    """
    candidates = [
        line
        for line in (_last_nonempty_line(stderr_text), _last_nonempty_line(stdout_text))
        if line is not None
    ]
    for line in candidates:
        message = _json_error_member(line)
        if message is not None:
            return message
    for line in candidates:
        without_indent = line.lstrip()
        if without_indent.startswith(REJECTION_PREFIX):
            return without_indent[len(REJECTION_PREFIX) :].lstrip()
    return None


def _tail(text: str) -> str:
    """Return at most the last :data:`DIAGNOSTIC_TAIL_CHARS` characters of `text`."""
    if len(text) <= DIAGNOSTIC_TAIL_CHARS:
        return text
    return "..." + text[-DIAGNOSTIC_TAIL_CHARS:]


def _json_object(stdout: bytes, name: str, keys: list[str]) -> dict[str, object]:
    """Parse the target's stdout as the one JSON object its ``metadata_map`` reads.

    The stdout bytes are the response that arrived, so each refusal carries them on ``body_prefix``.

    Raises:
    TargetContractError: stdout is empty, is not UTF-8, is not JSON, or is not a JSON object --
    naming the configured fields read from it.
    """
    configured = ", ".join(repr(key) for key in keys)
    if not stdout.strip():
        raise TargetContractError(
            f"target {name!r} wrote nothing to stdout, but metadata_map reads {configured} "
            "from it"
        )
    try:
        parsed = json.loads(stdout.decode("utf-8"))
    except UnicodeDecodeError:
        raise TargetContractError(
            f"target {name!r} wrote stdout that is not UTF-8, but metadata_map reads "
            f"{configured} from it; it starts with {hex_prefix(stdout)}",
            body_prefix=stdout,
        ) from None
    except json.JSONDecodeError as exc:
        raise TargetContractError(
            f"target {name!r} wrote stdout that is not JSON ({exc.msg} at character "
            f"{exc.pos}), but metadata_map reads {configured} from it",
            body_prefix=stdout,
        ) from None
    if not isinstance(parsed, dict):
        raise TargetContractError(
            f"target {name!r} wrote a JSON {type(parsed).__name__} to stdout, but "
            f"metadata_map reads {configured} from an object",
            body_prefix=stdout,
        )
    return parsed


def _declared_int(value: object) -> int | None:
    """Return `value` as the integer a target declared, or ``None`` when absent."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _declared_text(value: object) -> str | None:
    """Return `value` as the text a target declared, or ``None`` when absent."""
    return value if isinstance(value, str) else None
