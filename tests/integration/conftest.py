"""The fixtures the integration layer runs on: a real process, a real socket, a real corpus.

Nothing here is a test, and a fixture whose precondition cannot be met calls
:func:`pytest.fail` rather than skip, because a skip in this layer would hide the gate the
consuming module holds. :func:`run_cli` runs the product as a real process so a test asserts
the status a shell would see, and bounds its in-process path on a daemon thread with the same
guard, so a hang fails by name rather than stalling the session; :func:`reference_report`,
:func:`defective_report` and :func:`boundary_runs` are session-scoped, which keeps the five-run
sequence of AAP §0.6.1 #9 inside its 600 s ceiling, and each mints its own report root inside
the run helper, so no run overwrites another's evidence and the defective run's nine expected
failures stay attributable (AAP IM-8); :func:`http_target_server` and :func:`cli_target` drive
both configurable adapters over their real transports; and :func:`write_config` writes a
schema-valid document from caller values and the schema's declared defaults only (AAP §0.12)."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

import pytest

# The schema is the one declaration of every key, type and default, so it is the only source here.
from qrcheck.config import schema


# Locations and fixed values: the root comes from this file, never the working directory.

#: The repository root, derived from this file's location, so no fixture reads the cwd.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Entries whose presence proves ``_REPO_ROOT`` really is the repository root.
_REPO_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "qrcheck.toml", "examples")

#: The directory holding the two examples the adapter integration tests execute.
_EXAMPLES_DIR: Final[str] = "examples"

#: The loopback HTTP generator :func:`http_target_server` starts.
_HTTP_SERVER_SCRIPT: Final[str] = "http_target_server.py"

#: The subprocess generator :func:`cli_target` names.
_CLI_TARGET_SCRIPT: Final[str] = "cli_target.py"

#: The committed configuration file's name, which a reader expects beside a report root.
_CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"

#: The two built-in self-check targets in name order: the conforming one and the defective one.
_BUILTIN_TARGETS: Final[tuple[str, str]] = ("reference", "defective")

#: The module ``python -m`` runs for the process boundary (AAP §0.9.2).
_ENTRY_MODULE: Final[str] = "qrcheck"

#: The error-correction level the whole-suite fixtures run at: the ``run`` command's default.
_DEFAULT_EC_LEVEL: Final[str] = "M"

#: The levels success criterion 2 replays the boundary subset at (AAP §0.6.1 #2).
_BOUNDARY_LEVELS: Final[tuple[str, ...]] = ("L", "Q", "H")

#: The corpus class filter that selects the 40 boundary records.
_BOUNDARY_CLASS_GLOB: Final[str] = "boundary-*"

#: The two ``[checks]`` tables a generated document carries, so a run reads committed thresholds.
_CHECK_GROUPS: Final[tuple[str, str]] = ("performance", "damage")

#: The one ``[targets.*]`` key whose default is a collection, which the committed tables omit.
_OMITTED_TARGET_KEYS: Final[frozenset[str]] = frozenset({"capabilities"})

#: The example's two argument templates; ``--payload-b64`` decodes base64 strictly.
_CLI_PAYLOAD_ARG: Final[str] = "--payload-b64={payload_b64}"
_CLI_OUTPUT_ARG: Final[str] = "--output={}"

#: The capabilities the subprocess example conveys through those two words: PNG, no logo.
_CLI_CAPABILITIES: Final[tuple[str, ...]] = ("png",)

#: The loopback address the example server binds, which no configuration can widen.
_LOOPBACK_HOST: Final[str] = "127.0.0.1"
_PORT_FLOOR: Final[int] = 1
_PORT_CEILING: Final[int] = 65535

# Harness guards, not product thresholds: they bound a hang rather than a measurement.
_PORT_GUARD_S: Final[float] = 30.0
_TERMINATE_GUARD_S: Final[float] = 5.0
_RUN_GUARD_S: Final[float] = 900.0

# How much of a child's stream a failure diagnostic quotes, so the message stays readable.
_DIAGNOSTIC_CHARS: Final[int] = 2000


# The schema's declared defaults


def _default(group: str, key: str) -> Any:
    """Return one key's declared default as :data:`qrcheck.config.schema.SCHEMA` holds it."""
    return schema.SCHEMA[group][key].default


def _group_defaults(group: str) -> dict[str, Any]:
    """Return every declared default of one schema group, keyed by configuration key."""
    return {key: spec.default for key, spec in schema.SCHEMA[group].items()}


def _committed_corpus_path() -> Path:
    """Return the committed corpus's absolute path, from its configured default."""
    return _REPO_ROOT / str(_default("", "corpus_path"))


# The repository root


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root derived from this file rather than from the cwd."""
    for marker in _REPO_MARKERS:
        if not (_REPO_ROOT / marker).exists():
            pytest.fail(
                f"{_REPO_ROOT} is not the repository root: {marker!r} is missing. This "
                f"file's location is the only source of the root, so the checkout's "
                f"layout has changed under the integration tests."
            )
    return _REPO_ROOT


# Generating a configuration document: ``tomllib`` cannot write TOML and no writer dependency may be
# added (AAP §0.7.1), so the serialiser's only promise is determinism.

#: A key TOML may spell bare; anything else -- a target name with a space, say -- is quoted.
_BARE_KEY: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+\Z")


def _toml_key(key: str) -> str:
    """Return ``key`` as a TOML key: bare when it may be, a JSON string literal otherwise."""
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value: object) -> str:
    """Return ``value`` as a TOML scalar or inline array, escaping strings through ``json``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(
        f"a configuration value cannot be written as TOML: {type(value).__name__} "
        f"({value!r}); use a string, bool, int, float, or a list of them"
    )


def _emit_table(lines: list[str], path: tuple[str, ...], table: Mapping[str, Any]) -> None:
    """Append one table -- its scalars, then its sub-tables -- to ``lines``, each sorted."""
    if path:
        if lines:
            lines.append("")
        lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
    for key in sorted(table):
        if not isinstance(table[key], Mapping):
            lines.append(f"{_toml_key(key)} = {_toml_value(table[key])}")
    for key in sorted(table):
        if isinstance(table[key], Mapping):
            _emit_table(lines, (*path, key), table[key])


def _render_document(document: Mapping[str, Any]) -> str:
    """Return ``document`` as deterministic TOML text, LF-separated with one trailing newline."""
    lines: list[str] = []
    _emit_table(lines, (), document)
    return "\n".join(lines) + "\n"


def _target_tables(
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Return the ``[targets.<name>]`` tables a generated document declares, in name order."""
    common = {
        key: spec.default
        for key, spec in schema.SCHEMA["targets.*"].items()
        if not spec.is_required and key not in _OMITTED_TARGET_KEYS
    }
    tables: dict[str, dict[str, Any]] = {
        name: {"type": name} for name in _BUILTIN_TARGETS
    }
    for name, table in (overrides or {}).items():
        tables[name] = dict(table)
    return {name: {**common, **tables[name]} for name in sorted(tables)}


@pytest.fixture(scope="session")
def write_config() -> Callable[..., Path]:
    """Return a writer that puts a complete, schema-valid configuration under a path."""

    def write(
        path: Path,
        *,
        default_adapter: str,
        report_dir: Path,
        corpus_path: Path | None = None,
        seed: int | None = None,
        targets: Mapping[str, Mapping[str, Any]] | None = None,
        **overrides: Any,
    ) -> Path:
        """Write the document to ``path``, creating a missing parent, and return that path."""
        render_overrides = dict(overrides.pop("render", {}) or {})
        checks_overrides = dict(overrides.pop("checks", {}) or {})
        unknown = sorted(set(overrides) - set(schema.SCHEMA[""]))
        if unknown:
            raise TypeError(
                f"write_config got override(s) {unknown} that name no top-level "
                f"configuration key; declared: {sorted(schema.SCHEMA[''])}"
            )

        document: dict[str, Any] = {
            key: spec.default for key, spec in schema.SCHEMA[""].items()
        }
        document["default_adapter"] = default_adapter
        document["report_dir"] = str(Path(report_dir))
        document["corpus_path"] = str(
            _committed_corpus_path() if corpus_path is None else Path(corpus_path)
        )
        document["seed"] = _default("", "seed") if seed is None else seed
        document.update(overrides)
        document["render"] = {**_group_defaults("render"), **render_overrides}
        document["checks"] = {
            group: {
                **_group_defaults(f"checks.{group}"),
                **dict(checks_overrides.get(group, {})),
            }
            for group in _CHECK_GROUPS
        }
        document["targets"] = _target_tables(targets)

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _render_document(document), encoding="utf-8", newline="\n"
        )
        return destination

    return write


# Running the command line: two paths and one result type -- the real process boundary by default,
# which makes an exit status what a shell would see, and an in-process path for repeated calls,
# which the same guard bounds on a daemon thread so a hang fails by name rather than stalling the
# session until the CI job's timeout kills it.


@dataclass(frozen=True, slots=True)
class CliResult:
    """The outcome of one ``qrcheck`` invocation: its status and both streams."""

    exit_code: int
    stdout: str
    stderr: str


def _stream_head(text: str) -> str:
    """Return a stream's leading part, stripped and cut to a bounded length for a diagnostic."""
    stripped = text.strip()
    if len(stripped) <= _DIAGNOSTIC_CHARS:
        return stripped
    return stripped[:_DIAGNOSTIC_CHARS] + "..."


def _as_text(value: str | bytes | None) -> str:
    """Return a captured stream as text, decoding with ``replace`` so a diagnostic cannot fail."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _run_cli_out_of_process(config: Path, args: Sequence[str], timeout: float) -> CliResult:
    """Invoke the command line as a real process and return its status and both streams."""
    argv = [sys.executable, "-m", _ENTRY_MODULE, "--config", str(config), *args]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        pytest.fail(
            f"qrcheck did not finish within {timeout:.0f}s: {' '.join(argv)}; "
            f"stdout head {_stream_head(_as_text(expired.stdout))!r}, "
            f"stderr head {_stream_head(_as_text(expired.stderr))!r}"
        )
    return CliResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _invoke_in_process(config: Path, args: Sequence[str]) -> CliResult:
    """Invoke ``qrcheck.cli.main.main`` under a ``CliRunner`` and return its exit status."""
    import click
    from click.testing import CliRunner

    from qrcheck.cli.main import main as cli_main

    @click.command(
        name="qrcheck",
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("argv", nargs=-1, type=click.UNPROCESSED)
    def _entry(argv: tuple[str, ...]) -> None:
        """Forward the argument vector to ``main`` and exit with the code it returned."""
        raise click.exceptions.Exit(cli_main(list(argv)))

    running = CliRunner(mix_stderr=False).invoke(
        _entry,
        ["--config", str(config), *args],
        catch_exceptions=False,
    )
    return CliResult(
        exit_code=running.exit_code,
        stdout=running.stdout,
        stderr=running.stderr,
    )


def _run_cli_in_process(config: Path, args: Sequence[str], timeout: float) -> CliResult:
    """Invoke the command line in this interpreter, bounded by ``timeout`` seconds."""
    outcome: list[CliResult] = []
    raised: list[BaseException] = []

    def _invoke() -> None:
        """Invoke the unbounded path and park whatever it returned or raised for the caller."""
        try:
            outcome.append(_invoke_in_process(config, args))
        except BaseException as error:
            # Captured, never swallowed: the caller re-raises it unchanged, so a genuine CLI
            # fault reaches the test with its own type, traceback and message.
            raised.append(error)

    argv = ["--config", str(config), *args]
    # A daemon thread, so an abandoned invocation cannot hold the session open after the failure.
    worker = threading.Thread(target=_invoke, name="qrcheck-in-process", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        pytest.fail(
            f"qrcheck did not finish within {timeout:g}s in this interpreter: "
            f"{' '.join(argv)}. A hang in-process must fail this gate by name: the only other "
            f"bound is the e2e job's timeout-minutes, and that kills the job instead of "
            f"reporting the invocation that hung (AAP §0.6.4)."
        )
    if raised:
        raise raised[0]
    return outcome[0]


def _run_cli(
    *, config: Path, args: Sequence[str], in_process: bool, timeout: float
) -> CliResult:
    """Invoke the command line one way or the other and return its status and both streams."""
    if in_process:
        return _run_cli_in_process(config, args, timeout)
    return _run_cli_out_of_process(config, args, timeout)


@pytest.fixture
def run_cli() -> Callable[..., CliResult]:
    """Return a callable that invokes the ``qrcheck`` command line and returns its outcome."""

    def run_cli(
        *args: str,
        config: Path,
        in_process: bool = False,
        timeout: float = _RUN_GUARD_S,
    ) -> CliResult:
        """Invoke one ``qrcheck`` command, for instance ``"run", "--adapter", "reference"``."""
        return _run_cli(
            config=Path(config),
            args=args,
            in_process=in_process,
            timeout=timeout,
        )

    return run_cli


# The whole-suite runs: one per built-in target plus the boundary subset at three levels, each paid
# for once per session; the exit code is handed to the test rather than asserted here.


def _run_suite_and_load(
    factory: pytest.TempPathFactory,
    write: Callable[..., Path],
    *,
    label: str,
    target: str,
    ec_level: str,
    extra_args: Sequence[str] = (),
) -> tuple[Path, dict[str, Any], int]:
    """Run the suite once against ``target``; return ``(report_root, summary, exit_code)``."""
    report_root = factory.mktemp(f"reports-{label}")
    config_path = factory.mktemp(f"cfg-{label}") / _CONFIG_FILE_NAME
    write(
        config_path,
        default_adapter=target,
        report_dir=report_root,
        corpus_path=_committed_corpus_path(),
    )
    result = _run_cli(
        config=config_path,
        args=("run", "--adapter", target, "--ec", ec_level, *extra_args),
        in_process=False,
        timeout=_RUN_GUARD_S,
    )

    summary_path = report_root / "summary.json"
    if not summary_path.is_file():
        pytest.fail(
            f"the {label} run wrote no {summary_path.name} under {report_root} "
            f"(exit code {result.exit_code}); stderr head: {_stream_head(result.stderr)!r}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        pytest.fail(
            f"the {label} run's {summary_path.name} is not a JSON object: "
            f"{_stream_head(repr(summary))}"
        )
    return report_root, summary, result.exit_code


@pytest.fixture(scope="session")
def reference_report(
    tmp_path_factory: pytest.TempPathFactory,
    write_config: Callable[..., Path],
) -> tuple[Path, dict[str, Any], int]:
    """Run the suite once against the conforming target, session-scoped (AAP §0.6.1 #9)."""
    return _run_suite_and_load(
        tmp_path_factory,
        write_config,
        label="reference",
        target="reference",
        ec_level=_DEFAULT_EC_LEVEL,
    )


@pytest.fixture(scope="session")
def defective_report(
    tmp_path_factory: pytest.TempPathFactory,
    write_config: Callable[..., Path],
) -> tuple[Path, dict[str, Any], int]:
    """Run the suite once against the five-injection target, independently of the reference run."""
    return _run_suite_and_load(
        tmp_path_factory,
        write_config,
        label="defective",
        target="defective",
        ec_level=_DEFAULT_EC_LEVEL,
    )


@pytest.fixture(scope="session")
def boundary_runs(
    tmp_path_factory: pytest.TempPathFactory,
    write_config: Callable[..., Path],
) -> dict[str, tuple[Path, dict[str, Any], int]]:
    """Run the boundary subset at L, Q and H, keyed by level in the declared order."""
    return {
        level: _run_suite_and_load(
            tmp_path_factory,
            write_config,
            label=f"boundary-{level}",
            target="reference",
            ec_level=level,
            extra_args=("--class", _BOUNDARY_CLASS_GLOB),
        )
        for level in _BOUNDARY_LEVELS
    }


# The subprocess example target


@dataclass(frozen=True, slots=True)
class CliTarget:
    """The subprocess example target: its script and a ``[targets.<name>]`` table declaring it."""

    script: Path
    target_config: Mapping[str, Any]


@pytest.fixture
def cli_target() -> CliTarget:
    """Return the ``cli`` adapter's example target: its script and its configuration table."""
    script = _REPO_ROOT / _EXAMPLES_DIR / _CLI_TARGET_SCRIPT
    if not script.is_file():
        pytest.fail(f"the subprocess example target is missing: {script}")
    return CliTarget(
        script=script,
        target_config={
            "type": "cli",
            "command": [sys.executable, str(script)],
            "payload_arg": _CLI_PAYLOAD_ARG,
            "output_arg": _CLI_OUTPUT_ARG,
            "returns": "file_path",
            "timeout_s": _default("targets.*", "timeout_s"),
            "capabilities": list(_CLI_CAPABILITIES),
        },
    )


# The loopback example server: one process, one socket, one line of stdout. That line is the whole
# handshake, so an unread or late line is the only way this fixture can hang.


@dataclass(slots=True)
class HttpTargetServer:
    """A running ``examples/http_target_server.py``, addressable and stoppable."""

    port: int
    process: subprocess.Popen[str]
    stopped: bool = False

    @property
    def base_url(self) -> str:
        """Return the server's loopback base URL, ``http://127.0.0.1:<port>``."""
        return f"http://{_LOOPBACK_HOST}:{self.port}"

    def stop(self) -> None:
        """Stop the server, wait for it and close both pipes. Idempotent."""
        if self.stopped:
            return
        self.stopped = True
        returncode, _stderr = _stop_process(self.process)
        if returncode is None:
            pytest.fail(
                f"examples/http_target_server.py (pid {self.process.pid}) survived "
                f"SIGTERM and SIGKILL, {_TERMINATE_GUARD_S:.0f}s each"
            )


def _stop_process(process: subprocess.Popen[str]) -> tuple[int | None, str]:
    """Stop ``process`` and return ``(returncode, stderr_head)``."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=_TERMINATE_GUARD_S)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=_TERMINATE_GUARD_S)
        except subprocess.TimeoutExpired:
            return (None, "")
    stderr_text = ""
    if process.stderr is not None:
        with contextlib.suppress(OSError, ValueError):
            stderr_text = process.stderr.read() or ""
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError, ValueError):
                stream.close()
    return (process.returncode, stderr_text)


def _fail_without_port(process: subprocess.Popen[str], what: str) -> NoReturn:
    """Stop ``process`` and fail the test naming its status and the head of its stderr."""
    returncode, stderr_text = _stop_process(process)
    pytest.fail(
        f"examples/http_target_server.py {what}, so the fixture has no port to give "
        f"(return code {returncode!r}); stderr head: {_stream_head(stderr_text)!r}"
    )


def _read_port_line(process: subprocess.Popen[str], timeout: float) -> str:
    """Read the one stdout line that carries the server's port, guarded by ``timeout``."""
    reader_stream = process.stdout
    if reader_stream is None:  # pragma: no cover - the fixture always asks for a pipe
        raise RuntimeError("the example server was started without a stdout pipe")
    lines: queue.Queue[str] = queue.Queue(maxsize=1)

    def _read() -> None:
        """Read one line, or an empty string when the pipe closes under us."""
        try:
            lines.put(reader_stream.readline())
        except (OSError, ValueError):  # pragma: no cover - pipe closed under a kill
            lines.put("")

    reader = threading.Thread(target=_read, name="qrcheck-port-reader", daemon=True)
    reader.start()
    try:
        line = lines.get(timeout=timeout)
    except queue.Empty:
        _fail_without_port(process, f"printed no port within {timeout:.0f}s")
    reader.join(timeout=_TERMINATE_GUARD_S)
    if not line:
        _fail_without_port(process, "closed its stdout before printing a port")
    return line


@pytest.fixture
def http_target_server(repo_root: Path) -> Iterator[HttpTargetServer]:
    """Start the loopback example server and yield a carrier for it."""
    script = repo_root / _EXAMPLES_DIR / _HTTP_SERVER_SCRIPT
    if not script.is_file():
        pytest.fail(f"the loopback example server is missing: {script}")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root),
    )
    try:
        line = _read_port_line(process, _PORT_GUARD_S)
        try:
            port = int(line.strip())
        except ValueError:
            _fail_without_port(process, f"printed {line.strip()!r}, which is not a port")
        if not _PORT_FLOOR <= port <= _PORT_CEILING:
            _fail_without_port(process, f"printed port {port}, outside 1..65535")
    except BaseException:
        # A failure before the carrier exists cannot use its teardown, so the child is stopped here.
        _stop_process(process)
        raise

    server = HttpTargetServer(port=port, process=process)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def unimportable_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str], Path]:
    """Return a callable that makes one module present yet unimportable for one test.

    The state a ``None`` entry in ``sys.modules`` cannot produce and a broken installation really
    has: ``importlib.util.find_spec`` finds the module and answers "present" while importing it
    raises :exc:`ImportError`. Because the shadowing lives on a child process's own ``sys.path``, it
    reaches this fixture's default out-of-process path, and the distribution metadata still reports
    its version -- what a presence-only check wrongly trusts.
    """

    def shadow(module: str) -> Path:
        """Write the stub for ``module``, prepend its directory to ``PYTHONPATH``, return it."""
        stub_dir = tmp_path / "stub"
        stub_dir.mkdir(exist_ok=True)
        (stub_dir / f"{module}.py").write_text(
            f'raise ImportError("{module} deliberately hidden by the test stub")\n',
            encoding="utf-8",
            newline="\n",
        )
        existing = os.environ.get("PYTHONPATH", "")
        monkeypatch.setenv(
            "PYTHONPATH",
            f"{stub_dir}{os.pathsep}{existing}" if existing else str(stub_dir),
        )
        return stub_dir

    return shadow
