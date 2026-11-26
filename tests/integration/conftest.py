"""The fixtures the integration layer runs on: a real process, a real socket, a real corpus.

Nothing here is a test -- no ``test_*`` function, no ``Test*`` class, no marker -- and a fixture
whose precondition cannot be met calls :func:`pytest.fail` rather than skip, because a skip in
this layer would hide the gate the module consuming it exists to hold.

* :func:`run_cli` runs the product as a **real process** (``python -m qrcheck``), so a test
  asserts the status a shell would see.
* :func:`reference_report`, :func:`defective_report` and :func:`boundary_runs` are
  **session-scoped**, which keeps the five-run sequence of AAP §0.6.1 #9 inside its 600 s
  ceiling; a function-scoped run fixture would multiply it by the tests that asked for it.
* :func:`report_roots` hands out **two distinct** report roots, because ``evidence/`` is laid
  out under one ``report_dir`` and a shared root would let the second run overwrite the first
  run's failures, making the nine expected ones unattributable (AAP IM-8).
* :func:`http_target_server` starts the loopback example server on an ephemeral port and
  :func:`cli_target` names the subprocess example, so both configurable adapters are driven
  over their real transports.
* :func:`write_config` writes a complete, schema-valid document from caller values and the
  schema's declared defaults only -- no threshold is restated here (AAP §0.12) -- with
  **absolute** paths, because the loader resolves a relative path against the document's own
  directory and these documents live under ``tmp_path``.

Every document is rendered in sorted key order, so two calls with the same arguments produce
identical bytes. No clock, ``hash()``, ``id()`` or filesystem order reaches a document, the
only socket opened is the loopback example server, and every artefact lands under ``tmp_path``
or ``tmp_path_factory``; the repository's own ``reports/`` is never touched (AAP §0.11.1).
"""

from __future__ import annotations

import contextlib
import json
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NoReturn

import pytest

# The schema is the one declaration of every configuration key, type and default, so it is the
# only source of a generated document's values (AAP §0.12); importing it loads no CLI, decoder
# or encoder.
from qrcheck.config import schema


# ---------------------------------------------------------------------------
# Locations and fixed values
#
# The repository root comes from this file's own location, never the working directory: the
# example scripts are launched as subprocesses and the corpus is read by absolute path.
# ---------------------------------------------------------------------------

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

#: The two built-in self-check targets, in name order: the conforming one that must come back
#: clean and the defective one whose failing set is asserted by set equality.
_BUILTIN_TARGETS: Final[tuple[str, str]] = ("reference", "defective")

#: The module ``python -m`` runs for the process boundary (AAP §0.9.2).
_ENTRY_MODULE: Final[str] = "qrcheck"

#: The error-correction level the whole-suite fixtures run at: the ``run`` command's default.
_DEFAULT_EC_LEVEL: Final[str] = "M"

#: The levels success criterion 2 replays the boundary subset at (AAP §0.6.1 #2).
_BOUNDARY_LEVELS: Final[tuple[str, ...]] = ("L", "Q", "H")

#: The corpus class filter that selects the 40 boundary records.
_BOUNDARY_CLASS_GLOB: Final[str] = "boundary-*"

#: The two ``[checks]`` tables a generated document carries, so a run reads the committed
#: thresholds rather than a fixture's copy of them.
_CHECK_GROUPS: Final[tuple[str, str]] = ("performance", "damage")

#: The one ``[targets.*]`` key whose declared default is a collection, which the committed
#: tables therefore omit: an empty ``capabilities`` list would say nothing, and the built-in
#: adapters declare their own capabilities in code.
_OMITTED_TARGET_KEYS: Final[frozenset[str]] = frozenset({"capabilities"})

#: The subprocess example's two argument templates, exactly as its docstring documents them;
#: ``--payload-b64`` strictly decodes base64, so a bare ``{}`` beside it would hand the flag raw
#: UTF-8 text and be refused.
_CLI_PAYLOAD_ARG: Final[str] = "--payload-b64={payload_b64}"
_CLI_OUTPUT_ARG: Final[str] = "--output={}"

#: The capabilities the subprocess example conveys through those two words: PNG, no logo.
_CLI_CAPABILITIES: Final[tuple[str, ...]] = ("png",)

#: The loopback address the example server binds, which no configuration can widen.
_LOOPBACK_HOST: Final[str] = "127.0.0.1"
_PORT_FLOOR: Final[int] = 1
_PORT_CEILING: Final[int] = 65535

# Harness guards, not product thresholds: no check reads them, no configuration key declares
# them, and they bound a hang rather than a measurement -- a merely slow value still passes.
_PORT_GUARD_S: Final[float] = 30.0
_TERMINATE_GUARD_S: Final[float] = 5.0
_RUN_GUARD_S: Final[float] = 900.0

# How much of a child's stream a failure diagnostic quotes, so the message stays readable.
_DIAGNOSTIC_CHARS: Final[int] = 2000


# ---------------------------------------------------------------------------
# The schema's declared defaults
# ---------------------------------------------------------------------------


def _default(group: str, key: str) -> Any:
    """Return one key's declared default, exactly as :data:`qrcheck.config.schema.SCHEMA` holds it.

    ``group`` is ``""`` for the top level, otherwise ``render``, ``checks.performance``,
    ``checks.damage`` or ``targets.*``. An undeclared path raises ``KeyError``, which is a
    defect in this module rather than a configuration fault.
    """
    return schema.SCHEMA[group][key].default


def _group_defaults(group: str) -> dict[str, Any]:
    """Return every declared default of one schema group, keyed by configuration key."""
    return {key: spec.default for key, spec in schema.SCHEMA[group].items()}


def _committed_corpus_path() -> Path:
    """Return the committed corpus's absolute path, derived from its configured default.

    The corpus is a repository file rather than package data (AAP §0.9.2), and a generated
    document lives under ``tmp_path`` where a relative path would resolve against the wrong
    directory, so a run fixture names it absolutely.
    """
    return _REPO_ROOT / str(_default("", "corpus_path"))


# ---------------------------------------------------------------------------
# The repository root and the two report roots
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root, derived from this file rather than from the cwd.

    Session-scoped because it cannot change within a session and every subprocess and every
    absolute path this module builds is rooted at it; a root that lacks the repository's own
    entries fails the test, because the checkout's layout has changed under it.
    """
    for marker in _REPO_MARKERS:
        if not (_REPO_ROOT / marker).exists():
            pytest.fail(
                f"{_REPO_ROOT} is not the repository root: {marker!r} is missing. This "
                f"file's location is the only source of the root, so the checkout's "
                f"layout has changed under the integration tests."
            )
    return _REPO_ROOT


@pytest.fixture
def report_roots(tmp_path: Path) -> Mapping[str, Path]:
    """Return one report root per built-in target, distinct by construction.

    ``evidence/`` is laid out under a single ``report_dir`` (AAP §0.3.4), so a reference run and
    a defective run sharing one root would overwrite each other's evidence and the defective
    run's nine expected failures would stop being attributable (AAP IM-8). The roots are created
    here and are otherwise empty: no ``evidence/`` subtree or report file exists until a run
    writes one.
    """
    roots = {target: tmp_path / target for target in _BUILTIN_TARGETS}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return MappingProxyType(roots)


# ---------------------------------------------------------------------------
# Generating a configuration document
#
# ``tomllib`` reads TOML but cannot write it, and no TOML-writing dependency may be added
# (AAP §0.7.1), so the document is rendered here by a serialiser whose only promise is
# determinism: sorted keys, LF endings, one trailing newline.
# ---------------------------------------------------------------------------

#: A key TOML may spell bare; anything else -- a target name with a space in it, say -- is
#: quoted, so a caller cannot generate a document that fails to parse.
_BARE_KEY: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+\Z")


def _toml_key(key: str) -> str:
    """Return ``key`` as a TOML key: bare when it may be, a JSON string literal otherwise."""
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value: object) -> str:
    """Return ``value`` as a TOML scalar or inline array.

    Strings are escaped through :func:`json.dumps`, whose escapes are a subset of TOML's, so a
    path holding a quote or a backslash cannot break the document. A mapping raises
    ``TypeError``: :func:`_emit_table` renders tables, so reaching here with one is a caller
    defect rather than a configuration the product would accept.
    """
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
    """Append one table -- its scalar keys, then its sub-tables -- to ``lines``.

    TOML requires a sub-table to follow every scalar key of its parent, so the two groups are
    walked in sorted key order, separately; ``path`` is empty for the document root, which has
    no header.
    """
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
    """Return the ``[targets.<name>]`` tables a generated document declares, in name order.

    Both built-in targets are always present, and a supplied table *replaces* the default one of
    the same name rather than merging into it; each table carries the common keys' declared
    defaults beside the keys the target supplies.
    """
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
    """Return a writer that puts a complete, schema-valid configuration under a path.

    The document carries the five top-level keys, a ``[targets.*]`` table for each built-in
    target plus any the caller supplies, and the ``[render]``, ``[checks.performance]`` and
    ``[checks.damage]`` tables, so a run started from it sees the committed defaults for
    everything the caller did not choose. ``report_dir`` and ``corpus_path`` are written as
    **absolute** paths, because the loader resolves a relative path against the configuration
    file's own directory and these documents live under ``tmp_path``. Session-scoped and
    stateless: two calls with the same arguments write the same bytes, and an override naming
    no declared top-level key raises ``TypeError`` here instead of being silently dropped.
    """

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
        """Write the document to ``path``, creating a missing parent, and return that path.

        ``corpus_path`` defaults to the committed corpus and ``seed`` to the schema's declared
        default; ``targets`` supplies extra or replacement ``[targets.<name>]`` tables, and
        ``**overrides`` further top-level keys plus the ``render`` and ``checks`` tables, whose
        entries merge over the declared defaults.
        """
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


# ---------------------------------------------------------------------------
# Running the command line
#
# Two paths and one result type: the default is the real process boundary, which makes an
# exit-status assertion an assertion about what a shell would see; the in-process path calls
# the same ``main``, so a test that invokes the command line repeatedly pays for one interpreter.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CliResult:
    """The outcome of one ``qrcheck`` invocation.

    ``stderr`` carries the command line's own diagnostics, which is where several exit-code
    conditions are reported and nowhere else (AAP §0.6.6 rows 1, 2, 10).
    """

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
    """Invoke the command line as a real process and return its exit status and both streams.

    A process that outlives ``timeout`` fails the test rather than being waited out, naming the
    command, the guard and what the child had already written: a hang is a framework defect to
    report, not a slow machine.
    """
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


def _run_cli_in_process(config: Path, args: Sequence[str]) -> CliResult:
    """Invoke ``qrcheck.cli.main.main`` under a ``CliRunner`` and return its exit status.

    ``main`` returns its code instead of exiting (click runs non-standalone), so a ``CliRunner``
    cannot be handed it directly: the thin click command below forwards the argument vector
    unchanged and raises :exc:`click.exceptions.Exit` with the integer ``main`` returned -- how
    click's own standalone machinery carries a subcommand's code to the process status in both
    modes. ``main`` therefore stays the only thing that decides a code, and this path reports
    exactly the status the process path would. Imported at the point of use, so a session that
    never asks for an in-process run loads no command line, check family or decoder wrapper.
    """
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


def _run_cli(
    *, config: Path, args: Sequence[str], in_process: bool, timeout: float
) -> CliResult:
    """Invoke the command line one way or the other and return its exit status and both streams.

    ``timeout`` guards the out-of-process path only; in-process, a call cannot outlive the test
    that made it without failing it directly.
    """
    if in_process:
        return _run_cli_in_process(config, args)
    return _run_cli_out_of_process(config, args, timeout)


@pytest.fixture
def run_cli() -> Callable[..., CliResult]:
    """Return a callable that invokes the ``qrcheck`` command line and returns its outcome.

    The callable asserts nothing: it hands back the exit status, stdout and stderr, and each
    test asserts the code, the diagnostic and the files its own case requires. An exception
    escaping ``main`` would itself be a finding, so the in-process path runs with
    ``catch_exceptions=False`` and lets it reach the test.
    """

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


# ---------------------------------------------------------------------------
# The whole-suite runs
#
# One run per built-in target, plus the boundary subset at each of the three remaining
# error-correction levels, each paid for exactly once per session. The exit code is
# handed to the test rather than asserted here: ``test_reference_run.py`` expects 0 and
# ``test_defective_run.py`` expects 1, and a fixture that asserted either would hide the
# other's condition.
# ---------------------------------------------------------------------------


def _run_suite_and_load(
    factory: pytest.TempPathFactory,
    write: Callable[..., Path],
    *,
    label: str,
    target: str,
    ec_level: str,
    extra_args: Sequence[str] = (),
) -> tuple[Path, dict[str, Any], int]:
    """Run the suite once against ``target`` and return ``(report_root, summary, exit_code)``.

    ``label`` keeps each run's configuration and its report root distinct, so two fixtures never
    share a directory. A run that wrote no ``summary.json``, or wrote one that is not a JSON
    object, fails the test: a fixture that skipped here would hide the gate the consuming module
    exists to hold.
    """
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
    """Run the suite once against the conforming built-in target at the default EC level.

    Session-scoped so the clean run is paid for once however many modules ask for it, and so
    ``pytest -m e2e`` cannot run it twice (AAP §0.6.1 #9).
    """
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
    """Run the suite once against the five-injection built-in target at the default EC level.

    Independent of :func:`reference_report`: each fixture runs its own suite once, so running
    either consuming module alone runs exactly one suite.
    """
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
    """Run the 40-record boundary subset against the conforming target at L, Q and H.

    This is how AAP §0.6.1 #2's four runs are invoked: the corpus class filter selects the
    boundary records once per level, and each run gets its own configuration and its own report
    root so no run can overwrite another's evidence. The mapping is keyed by level, in the
    declared level order.
    """
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


# ---------------------------------------------------------------------------
# The subprocess example target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CliTarget:
    """The subprocess example target: its script and a ``[targets.<name>]`` table declaring it."""

    script: Path
    target_config: Mapping[str, Any]


@pytest.fixture
def cli_target() -> CliTarget:
    """Return the ``cli`` adapter's example target: its script and its configuration table.

    The table mirrors the canonical block in ``examples/cli_target.py``'s own docstring: the two
    argument templates are the spellings that script actually accepts, and the capability list
    names only ``png``, because those two words carry no error level, module size, quiet zone,
    colour, version or mask and the script emits no SVG. A check the declared set cannot satisfy
    therefore reports ``skipped`` with ``unsupported_capability`` instead of measuring a symbol
    built from the script's own defaults. A fresh table per test, so a test that edits it cannot
    reach another test; a missing script fails the test, because the checkout is then not the
    one this suite describes.
    """
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


# ---------------------------------------------------------------------------
# The loopback example server
#
# One process, one socket, one line of stdout. The line is the whole handshake: the server
# binds an ephemeral port and prints it before it serves, so an unread or late line is the
# only way this fixture can hang -- and a hang is a defect, so every read is guarded and
# every failure stops the child and names what it did.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HttpTargetServer:
    """A running ``examples/http_target_server.py``, addressable and stoppable.

    ``stopped`` records that :meth:`stop` has run, because teardown and a test's own explicit
    stop can both call it.
    """

    port: int
    process: subprocess.Popen[str]
    stopped: bool = False

    @property
    def base_url(self) -> str:
        """Return the server's loopback base URL, ``http://127.0.0.1:<port>``."""
        return f"http://{_LOOPBACK_HOST}:{self.port}"

    def stop(self) -> None:
        """Stop the server, wait for it and close both pipes. Idempotent.

        A child that survived ``terminate`` and ``kill`` fails the test: nothing else can reap
        it, so the run is reported rather than left with a stray process behind it.
        """
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
    """Stop ``process`` and return ``(returncode, stderr_head)``.

    ``terminate`` first, because the example server shuts down cleanly on ``SIGTERM`` and exits
    0; ``kill`` only when that guard expires. A process that has already exited is neither
    signalled nor waited on again, which is what makes this callable from the fixture's teardown
    after a test has stopped the server itself. ``returncode`` is ``None`` when the child survived
    both signals, and the stderr head is then empty, because reading a pipe a live process still
    holds would block.
    """
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
    """Stop ``process`` and fail the test, naming its status and the head of its stderr.

    ``what`` is the clause the message completes. A hang or a silent dead child is a defect in
    the example or in this fixture, never an unsupported platform, so nothing here skips.
    """
    returncode, stderr_text = _stop_process(process)
    pytest.fail(
        f"examples/http_target_server.py {what}, so the fixture has no port to give "
        f"(return code {returncode!r}); stderr head: {_stream_head(stderr_text)!r}"
    )


def _read_port_line(process: subprocess.Popen[str], timeout: float) -> str:
    """Read the one stdout line that carries the server's port.

    The read runs on a helper thread and the wait on the caller's, because a pipe read has no
    timeout of its own and a server that never prints would park the whole session. Exactly one
    line is read and the pipe is then left alone, so the server's own output can never be
    mistaken for this fixture's. No line inside ``timeout``, or a closed stdout, fails the test.
    """
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
    """Start the loopback example server and yield a carrier for it.

    The server binds ``127.0.0.1`` on an ephemeral port and prints that port as the only line it
    writes on stdout, so a test never hard-codes one, and nothing here may bind or connect to a
    non-loopback address: the example's own literal is the only host, and a test's client side
    goes through the process-level guard that refuses any other (AAP §0.3.2, NFR-2). Teardown
    stops the child even when the test already did. A missing script, no port inside the guard, a
    line that is not a port in 1..65535, or a child that outlived both signals fails the test.
    """
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
        # A failure before the carrier exists cannot use its teardown, so the child is
        # stopped here; `_stop_process` is idempotent, so one that already stopped is
        # simply reaped.
        _stop_process(process)
        raise

    server = HttpTargetServer(port=port, process=process)
    try:
        yield server
    finally:
        server.stop()
