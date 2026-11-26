"""The ``cli`` adapter: an out-of-process QR generator driven as an argv-only subprocess.

A ``type = "cli"`` target is a program the framework runs once per request.  This
module owns everything about that exchange: how the command line is assembled from
configuration, where the artefact is expected, how a non-zero exit is interpreted,
and how the target's own metadata is read.  It is the working counterpart of
``examples/cli_target.py``, which ``tests/integration/test_cli_adapter.py``
executes, so the two files state one contract that ``docs/targets.md`` quotes.

Invocation, and why there is never a shell
------------------------------------------
``command`` is untrusted configuration and the payload is untrusted corpus data,
so the command line is built as a list of strings and handed to
``subprocess.run(argv, shell=False, ...)``.  No shell is ever interposed: the
argument vector is a list rather than a command string, there is no pre-exec
hook, and no shell-interpreting call appears anywhere in this module.  A payload
or a configured argument carrying ``;``, ``$(...)`` or a quote is therefore one
literal argv word.  The child inherits the framework's environment and working
directory and nothing else, and ``timeout_s`` goes straight to
``subprocess.run(timeout=...)``, which kills the child rather than leaving it
running.

``payload_arg`` -- the two permitted shapes
-------------------------------------------
1. **A placeholder template.**  Every ``{...}`` occurrence is substituted from the
   request.  The names are the HTTP adapter's ``request_template`` vocabulary, so
   one mental model covers both transports, plus the bare ``{}`` meaning "the
   payload here":

   ==================  ========================================================
   ``{}``              the payload as UTF-8 text; a ``TargetContractError`` when
                       the payload is not strictly UTF-8-decodable, so a binary
                       payload is refused instead of silently mangled
   ``{payload_b64}``   the payload, base64-encoded: the binary-safe form
   ``{payload_utf8}``  the payload as UTF-8 text, with the same strictness
   ``{ec_level}``      ``L``/``M``/``Q``/``H``
   ``{format}``        ``png``/``svg``
   ``{size}``          ``module_size_px``, empty when the request pins none
   ``{margin}``        ``quiet_zone_modules``, empty when the request pins none
   ``{fg}``/``{bg}``   ``#rrggbb`` from the request's colour tuples
   ==================  ========================================================

   An unknown placeholder is refused with ``ValueError`` naming
   ``targets.<name>.payload_arg``: the configuration schema validates a
   template's shape but not its vocabulary.  ``version`` and ``mask`` pins are
   deliberately absent from the vocabulary, which is shared with the HTTP
   adapter: a target that honours a pin takes it from its configured ``command``.
   The capability gate still requires ``version_pin`` and ``mask_pin`` when a
   request pins one, so such a request reaches only a target that declares it can
   honour it.

2. **A bare flag.**  A template starting with ``-`` and carrying no placeholder is
   passed as a flag, and the **raw payload bytes** are written to the child's
   stdin.  That is how a payload which does not survive an argv word is delivered
   byte for byte; ``examples/cli_target.py`` documents ``--payload-b64 -`` as
   exactly this form.

Anything else is a configuration fault the schema should have caught, and is
refused with ``ValueError`` naming the key.  A command line reads
``command``, then the payload argument, then the output argument.

``output_arg``, the temporary directory, and why its name matters
-----------------------------------------------------------------
A placeholder-bearing ``output_arg`` receives the path of a file the framework
creates for the target to write: ``symbol.png`` or ``symbol.svg`` inside a
directory made by ``tempfile.mkdtemp(prefix="qrcheck-")``.  The extension states
the format the request asked for, which is how a target learns it when its
configuration has no ``{format}`` placeholder of its own.  That prefix is a
**product-wide contract**, not a local convention -- ``CL-ARTIFACTS`` snapshots
``tempfile.gettempdir()`` for names carrying it before and after a run and
asserts the two snapshots are equal, so a leaked directory would fail a check
that has nothing to do with this file.  The directory is removed in a ``finally``
block, so it survives neither a timeout nor a non-zero exit nor a contract error.
An empty ``output_arg`` passes no path and expects the artefact on stdout, which
only a stdout-based ``returns`` can read.

``returns``
-----------
``file_path``   the file the framework named through ``output_arg``
``bytes_path``  the same file, or -- when no ``output_arg`` is configured -- the
                path the target reports on the last non-empty line of its stdout
``raw_bytes``   the artefact itself on stdout
``base64``      base64 text on stdout

Decoding goes through the port's shared
:func:`~qrcheck.targets.base.decode_returned_image` and the result through
:func:`~qrcheck.targets.base.ensure_image_bytes`, so this adapter and the
``library`` adapter accept and refuse exactly the same bodies.  Because a
stdout-based ``returns`` consumes stdout, a ``metadata_map`` configured beside one
is refused at construction.

The declared-rejection encoding
-------------------------------
A non-zero exit is a **declared rejection** only when the target says so; every
other non-zero exit is a contract error, because the ``IV-*`` checks pass on a
declared rejection and a fabricated one would make them vacuous.  Two spellings
are accepted, checked in this order, each on the last non-empty line of stderr
first and of stdout second:

1. a JSON object with a string member ``error`` -- ``{"error": "payload too
   long"}`` -- whose value becomes the rejection text **verbatim**;
2. a line beginning with ``rejected:`` -- ``rejected: contrast below floor`` --
   whose remainder, without the whitespace separating it from the prefix, is the
   text.

``str(TargetRejected(...))`` is therefore exactly what the target said, which is
what the ``IV-*`` checks record in ``decoded.rejection``.

Metadata
--------
When ``metadata_map`` is configured the target reports its claims as one JSON
object on stdout, and :func:`~qrcheck.targets.base.apply_metadata_map` projects
them onto ``version``, ``ec``, ``mask`` and ``mode``.  Those values become
``GeneratedSymbol.declared_*`` and are evidence only: no check asserts them.

Layering
--------
This module imports the standard library, :mod:`qrcheck.config.models` and
:mod:`qrcheck.targets.base` only -- no decoder, no image library, no check, no
report and no registry, which is the property ``tests/unit/test_layering.py``
holds for the whole ``targets`` package.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from qrcheck.config.models import RETURNS_KINDS, CLITarget, ReturnsKind
from qrcheck.targets.base import (
    OUTPUT_FORMATS,
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    apply_metadata_map,
    declared_capabilities,
    decode_returned_image,
    ensure_image_bytes,
    hex_prefix,
    require_capabilities,
)

__all__ = ["CliAdapter"]

# The prefix of every temporary directory this adapter creates.  CL-ARTIFACTS
# scans tempfile.gettempdir() for this prefix before and after a run, so it is a
# contract shared with qrcheck/checks/cleanup.py rather than a private habit.
TEMP_DIR_PREFIX: Final[str] = "qrcheck-"

# The artefact's name inside that directory.  The extension carries the requested
# format, so a target which infers its writer from the path is not misled.
_TEMP_ARTEFACT_STEM: Final[str] = "symbol"

# A placeholder is a brace-delimited name: "{payload_b64}", or "{}" for the bare
# payload.  One pattern serves both argument templates, so a template carrying an
# unknown placeholder is refused rather than passed through literally.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{([A-Za-z0-9_]*)\}")

# The line prefix by which a target declares a rejection (encoding 2).
REJECTION_PREFIX: Final[str] = "rejected:"

# The JSON member by which a target declares a rejection (encoding 1).
REJECTION_JSON_FIELD: Final[str] = "error"

# How much of a failed target's captured output a contract-error message keeps.
DIAGNOSTIC_TAIL_CHARS: Final[int] = 2000

# stdout carries the artefact itself for these return kinds, so a metadata_map
# cannot also read the target's metadata from it.
_STDOUT_RETURNS: Final[frozenset[str]] = frozenset({"base64", "raw_bytes"})

# The colour pair a request carries when it asks for nothing in particular.  Only
# a request which changes one of them needs the `color` capability, so a target
# declaring no colour support still serves every ordinary request.
_DEFAULT_FOREGROUND: Final[tuple[int, int, int]] = (0, 0, 0)
_DEFAULT_BACKGROUND: Final[tuple[int, int, int]] = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class _Invocation:
    """What one ``generate`` call runs, and where its artefact will appear.

    Built before any subprocess exists so a configuration or payload fault is
    raised without spawning anything.  ``temp_dir`` is the directory created for
    this call alone -- ``None`` when no path was passed to the target -- and it is
    what ``generate`` removes on every path, including the failing ones.
    """

    argv: tuple[str, ...]
    stdin_payload: bytes | None
    artefact_path: Path | None
    temp_dir: Path | None


class CliAdapter:
    """Drive a configured ``command`` as one argv-only subprocess per request.

    The adapter holds no process and no temporary directory between calls: each
    :meth:`generate` spawns the target, reads its artefact and removes the
    directory it made before returning or raising.  It raises only the five
    :class:`~qrcheck.targets.base.TargetError` subclasses, because every library
    exception a subprocess can produce is converted at the point it is caught.
    """

    def __init__(self, config: CLITarget, *, name: str | None = None) -> None:
        """Prepare the adapter for `config` without spawning anything.

        Args:
            config: The resolved ``[targets.<name>]`` record of a ``cli`` target.
            name: The target's configured name, which every diagnostic quotes.
                Defaults to ``config.type``.

        Raises:
            ValueError: The configuration cannot describe a subprocess invocation
                it would be safe to run -- an empty ``command``, a NUL in a
                configured string (no argv word may carry one), a ``returns``
                value outside its set, or a ``metadata_map`` beside a
                stdout-based ``returns``, which would consume the same stream
                twice.  The message names the offending dotted key.  These are
                faults ``qrcheck.config.schema`` rejects or cannot see, and none
                of them is a target defect.
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

        Args:
            request: The request to hand the target.

        Returns:
            The artefact byte for byte as the target produced it, with the
            metadata the target declared and the measured target-side duration.

        Raises:
            TargetUnsupported: The request asks for something the target did not
                declare.  Raised before any I/O, so the target is never started.
            TargetUnavailable: The program could not be spawned.
            TargetTimeout: The target outlived ``timeout_s`` and was killed.
            TargetRejected: The target declared a client-side rejection.
            TargetContractError: The target exited zero without a usable artefact,
                or exited non-zero without declaring a rejection.
            ValueError: A template carries an unknown placeholder, or the payload
                cannot be interpolated into the shape the template asks for.  The
                message names the configured key.
        """
        require_capabilities(self, _required_capabilities(request))
        invocation = self._build_invocation(request)
        try:
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    list(invocation.argv),
                    shell=False,
                    stdin=subprocess.DEVNULL if invocation.stdin_payload is None else None,
                    input=invocation.stdin_payload,
                    capture_output=True,
                    text=False,
                    timeout=self._timeout_s,
                    check=False,
                    env=None,
                )
            except subprocess.TimeoutExpired:
                raise TargetTimeout(
                    f"target {self.name!r} ({invocation.argv[0]!r}) did not answer within "
                    f"timeout_s = {self._timeout_s:g} s and was terminated"
                ) from None
            except OSError as exc:
                # Only a spawn failure reaches here: subprocess.run raises OSError
                # from execvp alone, before any pipe is read.
                reason = exc.strerror or exc.__class__.__name__
                raise TargetUnavailable(
                    f"target {self.name!r} could not be spawned: {invocation.argv[0]!r} "
                    f"failed ({reason}); check targets.{self.name}.command"
                ) from None
            elapsed_ms = (time.perf_counter() - started) * 1000.0
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

    def _build_invocation(self, request: GenerateRequest) -> _Invocation:
        """Assemble the command line and the artefact path for `request`.

        Nothing is spawned here.  The only I/O is the temporary directory a
        path-bearing ``output_arg`` needs, and a rendering fault after it is
        created removes that directory before the failure propagates.

        Raises:
            ValueError: ``payload_arg`` is neither a placeholder template nor a
                flag, or ``output_arg`` names no artefact path.
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
        temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))
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
                The configuration schema accepts only those two shapes, so this is a
                framework-visible configuration fault rather than a target defect.
            TargetContractError: A text placeholder was given a payload that is not
                UTF-8-decodable.
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
            path: The artefact path that ``{}`` stands for in ``output_arg``.

        Raises:
            ValueError: A placeholder is outside the token vocabulary.
            TargetContractError: A text placeholder needs a UTF-8 payload and the
                payload is not one.
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
            "placeholders are {}, {payload_b64}, {payload_utf8}, {ec_level}, {format}, "
            "{size}, {margin}, {fg} and {bg}"
        )

    def _raise_for_failure(self, completed: subprocess.CompletedProcess[bytes]) -> None:
        """Raise what a non-zero exit means: a declared rejection, else a contract error.

        A rejection is raised only for output matching one of the two declared
        spellings, because a fabricated one would make the ``IV-*`` checks vacuous.

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
            f"stdout tail: {_tail(stdout_text)!r}"
        )

    def _read_artefact(
        self,
        completed: subprocess.CompletedProcess[bytes],
        artefact_path: Path | None,
        request: GenerateRequest,
    ) -> bytes:
        """Return the artefact a zero exit delivered, under the configured `returns`.

        Raises:
            TargetContractError: The response carries no artefact where ``returns``
                says to read it, or the bytes are not the format the request asked
                for.
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
            TargetContractError: stdout is not the one JSON object a configured
                ``metadata_map`` reads, or a configured field is absent from it.
        """
        if not self._metadata_map:
            return {}
        raw = _json_object(completed.stdout, self.name, sorted(self._metadata_map))
        return apply_metadata_map(raw, self._metadata_map)

    def close(self) -> None:
        """Release the adapter, which holds nothing between calls.

        Each :meth:`generate` owns its subprocess -- ``subprocess.run`` waits for
        it -- and removes its temporary directory before returning or raising, so
        there is no child, pipe or file left to close here and a second call is
        the same no-op as the first.  The method exists because the port requires
        it: the runner closes every adapter once the run ends, including after a
        failure.
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

    Only what the request *asks for* is gated: its output format always, and each
    optional field the caller actually set.  A default request therefore needs one
    capability -- the format -- so a target declaring only ``png`` serves every
    ordinary case.

    ``ec_level`` is deliberately not gated: every request carries a level, so
    requiring the capability would refuse every request to a target that supports
    the format but not level selection.  A check that asserts error-level
    conformance declares ``ec_level`` itself and is resolved to a skip by
    ``qrcheck/registry/resolve.py`` before it ever builds a request.

    Raises:
        ValueError: The request asks for an output format that is not one of
            :data:`~qrcheck.targets.base.OUTPUT_FORMATS`.  ``build_request``
            refuses that upstream, so it is a framework defect here.
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
    if request.logo_overlay_path is not None:
        needed.add(Capability.logo_overlay)
    if request.version is not None:
        needed.add(Capability.version_pin)
    if request.mask is not None:
        needed.add(Capability.mask_pin)
    return frozenset(needed)


def _checked_returns(returns: str, name: str) -> ReturnsKind:
    """Return `returns` when it is one of the four delivery modes.

    Raises:
        ValueError: `returns` is outside :data:`qrcheck.config.models.RETURNS_KINDS`.
            The configuration schema rejects that value at load time, so reaching
            this means a record was built by hand.
    """
    if isinstance(returns, str) and returns in RETURNS_KINDS:
        return cast(ReturnsKind, returns)
    raise ValueError(
        f"targets.{name}.returns must be one of {', '.join(RETURNS_KINDS)}; got {returns!r}"
    )


def _check_no_nul(value: str, name: str, key: str) -> None:
    """Refuse a configured string carrying a NUL, which no argv word can hold.

    ``execve`` cannot transport a NUL inside an argument, and ``subprocess`` raises
    a bare ``ValueError`` for one; catching it at construction keeps that from
    surfacing as an unexplained framework fault mid-run.

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
        TargetContractError: The payload is not strictly UTF-8-decodable, or it
            carries a NUL that no argv word can transport.  Both are refused rather
            than passed on mangled, and the message names the binary-safe
            placeholder that would have worked.
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

    A target's output is untrusted and need not be UTF-8, so undecodable bytes are
    replaced rather than allowed to raise: the classification of the exit status
    must not depend on the encoding of a diagnostic.
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

    A line that is not a JSON object, or whose ``error`` member is not a string, is
    not a declared rejection, so the caller falls through to the next spelling.
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

    Both streams are inspected, stderr before stdout, because a target reasonably
    writes a refusal to either: ``examples/cli_target.py`` writes its JSON line to
    stdout, while a POSIX-minded program writes to stderr.  The JSON spelling is
    tried on both streams before the ``rejected:`` spelling, so a target that
    writes one to each is classified by the more explicit form.
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

    Raises:
        TargetContractError: stdout is empty, is not UTF-8, is not JSON, or is a
            JSON value that is not an object -- naming the configured fields that
            were to be read from it.
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
            f"{configured} from it; it starts with {hex_prefix(stdout)}"
        ) from None
    except json.JSONDecodeError as exc:
        raise TargetContractError(
            f"target {name!r} wrote stdout that is not JSON ({exc.msg} at character "
            f"{exc.pos}), but metadata_map reads {configured} from it"
        ) from None
    if not isinstance(parsed, dict):
        raise TargetContractError(
            f"target {name!r} wrote a JSON {type(parsed).__name__} to stdout, but "
            f"metadata_map reads {configured} from an object"
        )
    return parsed


def _declared_int(value: object) -> int | None:
    """Return `value` as the integer a target declared, or ``None`` when it is absent."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _declared_text(value: object) -> str | None:
    """Return `value` as the text a target declared, or ``None`` when it is absent."""
    return value if isinstance(value, str) else None
