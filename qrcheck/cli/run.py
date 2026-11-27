"""``qrcheck run``: the only command that writes a report, and the composition point of a run.

This module reads the configuration, resolves the four options a run is defined by -- the
target, the error-correction level, the seed and the selection -- builds the one adapter the
configuration names, hands the run to :func:`qrcheck.runner.run_suite`, and writes the three
report artefacts through :func:`qrcheck.report.write_all_reports`. It owns option plumbing
and deliberately nothing else: no check is walked, no case is built, no request is
constructed, no capability is resolved, no fail-fast decision is taken and no result is
tallied here. All of that belongs to the runner, which is what keeps the execution policy in
one place and lets this command be exercised without a decoder or a target.

When each failure is decided
----------------------------
The order of the steps in :func:`run` is the exit-code guarantee. Every fault that the
configuration, the options, the selection or the target's availability decides ends the
command with exit code 2 **before a single case runs**, so no report is written and the
report directory is left empty:

* the configuration cannot be loaded or fails validation (``ConfigError``);
* the adapter cannot be built from a record the schema accepted -- a ``kwargs_map`` naming
  a field the framework does not have, a ``metadata_map`` beside a stdout-based ``returns``,
  an http template with no payload placeholder, or a target the constructor finds
  unavailable (``ConfigError``, ``TargetError`` and ``ValueError``);
* ``--adapter`` names no declared ``[targets.<name>]`` table, ``--ec`` is outside ``L``,
  ``M``, ``Q``, ``H``, the seed is negative, or ``--report-dir`` is a value the
  configuration's own path rule refuses -- an empty or whitespace-only path, which would
  otherwise resolve to the working directory and fill it with a report set;
* ``--checks`` matches no registered check (``SelectionError``, which echoes the selector);
* a pinned runtime dependency cannot be imported -- the primary decoder, ``segno``,
  ``numpy``, ``Pillow``, ``httpx`` or ``click``. For an ``http`` target it surfaces when the
  adapter is built, where ``httpx`` loads; the runner's pre-flight refuses the same way
  otherwise, before it reads the corpus;
* a run prerequisite fails -- the corpus cannot be read, or the ``--class`` filter leaves no
  record (``RunPrerequisiteError``).

Every fault found during or after the run is reported differently, because a report then
exists or can exist, and exit code 3 outranks 1, which outranks 0. One or more ``failed``
results with no ``error`` result is 1; any ``error`` result is 3; a run of ``passed`` and
``skipped`` results alone is 0, because a skip is neither a failure nor a pass-flavoured
outcome and is tallied in its own integer. A report directory that cannot be written is 3 as
well, and is the one condition where the results the run did produce are streamed to stdout as
one JSON object per line; a directory found unusable before a result existed leaves nothing to
stream, which the diagnostic states, so a CI consumer is never left with an empty stream it
cannot explain.

Two interface gaps the plan leaves open, and how this module settles them
------------------------------------------------------------------------
* **``--config``.** The group owns that option, and this module must not import
  ``qrcheck.cli.main`` -- the group imports the commands, so the reverse import would close a
  cycle. :func:`_config_path` therefore reads the value from the click context exactly as the
  sibling commands do, and :func:`run` takes it as the keyword-only ``config_path`` that the
  group's option supplies. That keyword is why the signature carries more than the seven run
  options AAP §0.8.1 names.
* **``--fail-fast``.** :func:`qrcheck.runner.run_suite` accepts ``fail_fast`` as a
  keyword-only argument, so this module forwards the flag unchanged and implements no part of
  the policy. Filtering the selection here instead would give the flag a second meaning and
  would decide execution policy in the command that exists only to plumb it.

Nothing here prints progress, measures a duration, retries a case or re-runs a check: the
product has no retry path anywhere (AAP §0.5.6). The single line this command writes on
success is deterministic and carries neither a timing nor a path.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import click

from qrcheck.checks import load_families
from qrcheck.cli.exit import EXIT_CONFIG, EXIT_INTERNAL, exit_code_for
from qrcheck.config import ConfigError, load_config
from qrcheck.config.schema import SCHEMA
from qrcheck.dependencies import blocking_problem
from qrcheck.registry import REGISTRY, SelectionError, parse_selection, select
from qrcheck.report import write_all_reports
from qrcheck.report.json_report import dumps_canonical
from qrcheck.runner import RunPrerequisiteError, run_suite
from qrcheck.targets import ADAPTER_TYPES
from qrcheck.targets.base import TargetError

if TYPE_CHECKING:  # Annotation resolution only: this module constructs no record itself
    from qrcheck.report.model import RunResult
    from qrcheck.targets.base import Adapter

__all__ = ("DEFAULT_CONFIG_PATH", "DEFAULT_EC_LEVEL", "EC_LEVELS", "run", "run_command")

#: The four error-correction levels a run may request, in the order the diagnostic and the
#: ``--ec`` help spell them.
EC_LEVELS: Final[tuple[str, ...]] = ("L", "M", "Q", "H")

#: The level a run uses when ``--ec`` is absent, which is the level AAP §0.6.1 criterion 2
#: names for the full-corpus round-trip run.
DEFAULT_EC_LEVEL: Final[str] = "M"

#: Read only when nothing in the group context supplies ``--config``, which happens when
#: :func:`run` is called directly rather than through the command line.
DEFAULT_CONFIG_PATH: Final[str] = "qrcheck.toml"

#: The names a group's context object, or the group's own click parameters, may carry the
#: ``--config`` value under; tried in this order. ``qrcheck.cli.main`` owns that option, so
#: this module reads all three shapes rather than requiring one of them.
_CONFIG_KEYS: Final[tuple[str, ...]] = ("config", "config_path", "config_file")

#: Every diagnostic carries this prefix and goes to stderr, so a line on stdout is always the
#: one-line summary (or a streamed result) and a line on stderr is always something to fix.
_DIAGNOSTIC_PREFIX: Final[str] = "qrcheck run: "

#: The probe file that proves the resolved report directory is writable. It is created inside
#: the report directory itself -- never in the system temporary directory, which the
#: artifact-cleanup check snapshots -- and removed before the run starts, so it leaves no
#: artefact behind.
_PROBE_PREFIX: Final[str] = "qrcheck-probe-"
_PROBE_SUFFIX: Final[str] = ".tmp"
_PROBE_CONTENT: Final[bytes] = b"qrcheck report directory probe\n"


def run(
    adapter: str | None,
    ec: str | None,
    seed: int | None,
    checks: str | None,
    class_globs: tuple[str, ...] | None,
    fail_fast: bool,
    report_dir: str | None,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """Execute one run of the suite and return the process exit code.

    The nine steps below are the exit-code guarantee: no report artefact is written and no
    result is streamed until the configuration, the options, the selection and the run
    prerequisites have all been settled, so an exit-2 path writes no artefact and streams
    nothing. The report directory is created and probed in step 5 -- before the selection is
    resolved in step 6 -- so a failing invocation may leave it behind empty rather than absent,
    the probe file itself removed; an override the schema's own path rule refuses never gets
    that far. An exit-3 path either writes a complete report or streams the results it holds.

    Args:
        adapter: The ``--adapter`` name of a declared ``[targets.<name>]`` table, or ``None``
            for the configuration's ``default_adapter``.
        ec: The ``--ec`` error-correction level, or ``None`` for :data:`DEFAULT_EC_LEVEL`.
            Upper-cased here, so a direct caller may pass ``"m"``.
        seed: The ``--seed`` run seed, or ``None`` for the configuration's ``seed``. Every
            derived seed and every reproducibility claim in the report comes from it.
        checks: The raw ``--checks`` selector -- exact identifiers, identifier globs and
            ``family:<name>`` tokens -- or ``None`` for every registered check.
        class_globs: The ``--class`` corpus class-name globs, in command-line order, or
            ``None`` for the whole corpus. The selection normalises them, so their order
            cannot reach a report.
        fail_fast: ``True`` to stop launching checks once one has produced a ``failed`` or
            ``error`` result. Forwarded unchanged; the policy lives in the runner.
        report_dir: The ``--report-dir`` report root, or ``None`` for the configuration's
            ``report_dir``. An override replaces the configured value for the runner, the
            evidence store and all three writers at once, and is held to the configuration's
            own path rule, so an empty or whitespace-only value is a bad invocation -- exit
            code 2 naming the option -- rather than a report written into the working
            directory.
        config_path: The ``--config`` file, supplied by the group's option. Keyword-only
            because it is not one of the run's seven options.

    Returns:
        :data:`qrcheck.cli.exit.EXIT_CONFIG` for a configuration, adapter, level, seed,
        report-directory value, selection or prerequisite fault;
        :data:`qrcheck.cli.exit.EXIT_INTERNAL` for an unwritable report directory; otherwise
        :func:`qrcheck.cli.exit.exit_code_for`'s reading of the finished run -- 3 when any
        result errored, 1 when any failed, 0 when everything passed or was skipped.

    Raises:
        Exception: anything the framework did not classify -- a constructor fault outside the
            three error classes step 7 maps to exit code 2, a programming error in a check
            that escaped containment, a non-``OSError`` fault of the writer. It propagates
            untouched, because only ``qrcheck.cli.main`` may turn an unclassified exception
            into exit code 3, and swallowing it here would publish a run as if nothing had
            gone wrong.
    """
    # -- 1. the configuration -------------------------------------------------
    try:
        config = load_config(config_path)
    except ConfigError as error:
        # str(error) is "<dotted key>: <message>", so the diagnostic names the key to edit;
        # an unreadable or unparsable file carries no key and reads as one sentence.
        click.echo(f"{_DIAGNOSTIC_PREFIX}configuration error: {error}", err=True)
        return EXIT_CONFIG
    except FileNotFoundError:
        # ``load_config`` folds an unreadable configuration file into ConfigError, so this is
        # the defence for a path that disappeared underneath it: name the file asked for.
        click.echo(
            f"{_DIAGNOSTIC_PREFIX}configuration file not found: {config_path}", err=True
        )
        return EXIT_CONFIG
    except OSError as error:
        # Exactly one filesystem fault escapes the loader unwrapped: the configured
        # report_dir could not be created or written. That is exit 3, and with no run to
        # describe there is nothing to stream -- see :func:`_report_unwritable`.
        return _report_unwritable(error, None, None)

    # -- 2. the target --------------------------------------------------------
    name = adapter or config.default_adapter
    target = config.targets.get(name)
    if target is None:
        click.echo(
            f"{_DIAGNOSTIC_PREFIX}unknown adapter {name!r}; declared: {sorted(config.targets)}",
            err=True,
        )
        return EXIT_CONFIG

    # -- 3. the error-correction level ---------------------------------------
    level = (ec or DEFAULT_EC_LEVEL).strip().upper()
    if level not in EC_LEVELS:
        click.echo(
            f"{_DIAGNOSTIC_PREFIX}invalid error-correction level {level!r};"
            f" expected one of {', '.join(EC_LEVELS)}",
            err=True,
        )
        return EXIT_CONFIG

    # -- 4. the seed ----------------------------------------------------------
    run_seed = config.seed if seed is None else seed
    if run_seed < 0:
        click.echo(
            f"{_DIAGNOSTIC_PREFIX}seed must be a non-negative integer, got {run_seed}",
            err=True,
        )
        return EXIT_CONFIG

    # -- 5. the report directory, resolved once and carried on the config -----
    # An override is held to the configuration's own path rule before anything is created,
    # because the file and the option name the same key and a value the file refuses cannot
    # become a report root merely by arriving on the command line. The rule and its wording
    # have one owner -- the schema entry ``report_dir`` names -- so only the option's name is
    # put in front of the fragment it returns: ``--report-dir ""`` is a bad invocation, exit 2,
    # rather than ``Path("")`` resolving to the working directory and filling it with a report
    # set. The same check refuses a NUL byte, which no filesystem call accepts and which
    # ``Path.resolve`` answers with a ``ValueError`` rather than the ``OSError`` the exit-3
    # contract below covers.
    if report_dir is not None:
        fragment = SCHEMA[""]["report_dir"].check(report_dir)
        if fragment is not None:
            click.echo(f"{_DIAGNOSTIC_PREFIX}--report-dir {fragment}", err=True)
            return EXIT_CONFIG
    chosen_report_dir = (
        Path(report_dir).expanduser().resolve()
        if report_dir is not None
        else config.report_dir
    )
    try:
        _probe_report_dir(chosen_report_dir)
    except OSError as error:
        return _report_unwritable(error, chosen_report_dir, None)
    # One resolved value, carried on the configuration: the runner builds the evidence store
    # from it and the writers create their files under it, so an override that reached only
    # one of them would lay evidence under one root and write the report under another.
    config = dataclasses.replace(config, seed=run_seed, report_dir=chosen_report_dir)

    # -- 6. the selection -----------------------------------------------------
    # Registration happens while a family module imports and ``qrcheck.registry`` never
    # imports ``qrcheck.checks`` (that would close the registration cycle), so the families
    # are loaded here before the registry is read. The call is idempotent, and the runner
    # makes it again for a direct caller.
    load_families()
    selection = parse_selection(checks, tuple(class_globs or ()))
    try:
        select(REGISTRY, selection.checks)
    except SelectionError as error:
        # str(error) echoes the selector verbatim, which is what the empty-selection gate
        # asserts, and appends the detail naming the offending token or family. An empty
        # ``--class`` filter is deliberately not checked here: the runner owns it and raises
        # it as a prerequisite error naming the globs, in step 7.
        click.echo(f"{_DIAGNOSTIC_PREFIX}{error}", err=True)
        return EXIT_CONFIG

    # -- 7. the adapter, and the run -----------------------------------------
    # ADAPTER_TYPES is the only factory table, and a lookup imports the one adapter module
    # for this target kind: no concrete adapter is named in this file. That import is also
    # where the kind's own library loads -- ``httpx`` for an ``http`` target -- so an
    # unusable pinned distribution surfaces here too.
    #
    # Construction sits inside the exit-2 mapping, never above it. A constructor refuses a
    # configuration that is wrong on its own terms but whose shape alone the schema can
    # check: a ``kwargs_map`` naming a field the framework does not have, because that
    # mapping's key vocabulary belongs to the adapter and not to the schema; a
    # ``metadata_map`` beside a stdout-based ``returns``, which would read one stream twice;
    # an http ``request_template`` carrying no payload placeholder. AAP §0.3.4 puts each of
    # these at exit code 2, named by its dotted key and decided before any check runs, and
    # ``str(error)`` already reads "<dotted key>: <message>". A ``TargetError`` raised this
    # early is the pre-flight availability fault, which carries that same code wherever it
    # is found: an adapter that could not be built leaves no run to describe.
    #
    # An ``ImportError`` or ``OSError`` from that same import is the environment fault of AAP
    # §0.7.2 instead -- exit 2 naming the distribution and ``requirements.lock``, through the
    # product's one dependency probe -- and not the exit-3 internal error the re-raise leaves
    # to ``qrcheck.cli.main`` for a fault no dependency explains.
    adapter_obj: Adapter | None = None
    try:
        adapter_obj = ADAPTER_TYPES[str(target.type)](target, name=name)
    except (ConfigError, TargetError, ValueError) as error:
        click.echo(f"{_DIAGNOSTIC_PREFIX}configuration error: {error}", err=True)
        return EXIT_CONFIG
    except (ImportError, OSError):
        problem = blocking_problem()
        if problem is None:
            raise
        click.echo(f"{_DIAGNOSTIC_PREFIX}{problem.message}", err=True)
        return EXIT_CONFIG
    try:
        run_result = run_suite(
            config, selection, adapter_obj, level, run_seed, fail_fast=fail_fast
        )
    except RunPrerequisiteError as error:
        # The message already names the missing dependency, the unreadable path or the globs
        # that left nothing, so it is relayed verbatim: no report is written for it.
        click.echo(f"{_DIAGNOSTIC_PREFIX}{error}", err=True)
        return EXIT_CONFIG
    except OSError as error:
        # The walk's evidence writes cannot arrive here: the runner contains each one as an
        # ``error`` result, so the results it accumulated survive to be written or streamed from
        # ``step 8`` instead of being lost with this exception (AAP §0.6.6 row 11). The one
        # ``OSError`` still raised is ``EvidenceStore.clear`` failing to remove a previous run's
        # evidence tree -- before the walk, before any result exists, so there is nothing to
        # stream and :func:`_report_unwritable` is handed no result.
        return _report_unwritable(error, config.report_dir, None)
    finally:
        # Closing twice is safe -- the port requires ``close`` to tolerate a second call --
        # and necessary here: the runner closes the adapter once its walk begins, but every
        # prerequisite failure raises before that point, and a loopback client or a scratch
        # directory must not outlive the command. A teardown fault is not a check's verdict,
        # so it is swallowed rather than allowed to replace the run's own outcome. There is
        # nothing to close when construction itself failed, because no adapter was made.
        try:
            if adapter_obj is not None:
                adapter_obj.close()
        except Exception:  # noqa: BLE001 - see above: it must not replace the run's outcome
            pass

    # -- 8. the reports -------------------------------------------------------
    try:
        write_all_reports(run_result, config.report_dir)
    except OSError as error:
        return _report_unwritable(error, config.report_dir, run_result)

    # -- 9. the summary line -------------------------------------------------
    # Read from the run's own four tallies and its recorded adapter, level and seed, so the
    # line and ``summary.json`` cannot describe the same run differently. No timing and no
    # path appears in it, which is what keeps two runs of one configuration byte-identical.
    counts = run_result.counts
    click.echo(
        f"adapter={run_result.adapter} ec={run_result.ec_level} seed={run_result.seed}"
        f" passed={counts.passed} failed={counts.failed}"
        f" skipped={counts.skipped} error={counts.error}"
    )
    return exit_code_for(run_result)


def _probe_report_dir(report_directory: Path) -> None:
    """Create ``report_directory`` if absent and prove it is writable, before a run starts.

    Writability is proven the only way it can be: by writing. A permission bit, an access-mode
    guess or a mount flag can each be wrong for the reason the real write would hit, so a
    probe file is created and removed, and a read-only directory, a filesystem with no space
    or a path whose parent is a regular file each surfaces as the ``OSError`` the report would
    have produced -- before the run rather than after it.

    The probe is created inside ``report_directory`` rather than in the system temporary
    directory, which the artifact-cleanup check snapshots, and it is removed in a ``finally``
    so an interrupted probe leaves nothing behind. It runs on every path, including the one
    the loader already proved: the ``--report-dir`` override has been proved by nobody, and a
    single code path is worth one temporary file.

    Args:
        report_directory: The resolved report root this run will write under.

    Raises:
        OSError: ``mkdir`` or the probe write failed. Propagated unwrapped, because a report
            directory the process cannot write is an environment fault -- exit code 3 -- and
            never a configuration fault.
    """
    report_directory.mkdir(parents=True, exist_ok=True)
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=report_directory,
            prefix=_PROBE_PREFIX,
            suffix=_PROBE_SUFFIX,
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(_PROBE_CONTENT)
            probe.flush()
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def _report_unwritable(
    detail: BaseException,
    report_directory: Path | None,
    run_result: RunResult | None,
) -> int:
    """Report an unwritable report directory and stream whatever results the CLI holds.

    The one condition AAP §0.3.4 defines where a report cannot be written but the run's
    findings may already exist. Two things follow, and neither is a retry or a second report
    path: the diagnostic goes to stderr naming the directory and the underlying fault, and
    every result the caller holds is streamed to stdout as one canonical JSON object per line
    -- no wrapper object and no array -- through the product's only JSON formatter, so a CI
    consumer parses results it would otherwise never see.

    When the caller holds no run result the stream stays empty and the diagnostic says so, so an
    empty stdout is never mistaken for a truncated one -- and it says no more than that, because
    a result is either handed to this function or it is not, and what an earlier fault may have
    discarded is not something it can know. The condition it can state is the one that reaches
    it holding nothing: the report directory was found unusable before this command held a
    result. Three faults are of that kind, and the runner is why there are only three -- a
    report directory the configuration's own load could not create, one the writability probe
    refused, and a previous run's evidence tree ``EvidenceStore.clear`` could not remove before
    the walk began; every evidence write *during* the walk is contained by the runner as an
    ``error`` result instead of raising, which is what lets the results of such a run be
    streamed from the other branch (AAP §0.6.6 row 11, §0.3.4). No result is fabricated to fill
    the output. Every fault is decided by the caller, not here: the run may have completed and
    only the report failed, and then the number of results streamed is stated on stderr as well.

    Args:
        detail: The ``OSError`` that prevented the write, quoted in the diagnostic because its
            message names the concrete cause -- a denied permission, a full filesystem, a path
            component that is a file.
        report_directory: The report root that could not be written, or ``None`` when the
            configuration itself failed before one was resolved.
        run_result: The finished run whose results are streamed, or ``None`` when the caller
            holds no result to stream.

    Returns:
        :data:`qrcheck.cli.exit.EXIT_INTERNAL`, always: an unwritable report directory is exit
        code 3 whatever else the run found.
    """
    where = "the report directory" if report_directory is None else str(report_directory)
    click.echo(
        f"{_DIAGNOSTIC_PREFIX}report directory cannot be written: {where}: {detail}",
        err=True,
    )
    if run_result is None:
        # Nothing is streamed and nothing is invented: a consumer reading only stdout has to
        # learn from stderr that the empty stream is this command's whole answer, not a
        # truncation. What follows the colon is only what this function can see -- it was handed
        # no result -- because whether a run produced any is not something it can know.
        click.echo(
            f"{_DIAGNOSTIC_PREFIX}no result was streamed to stdout: the report directory was "
            f"found unusable before this command held a result",
            err=True,
        )
    else:
        streamed = len(run_result.results)
        click.echo(
            f"{_DIAGNOSTIC_PREFIX}{streamed} "
            f"{'result' if streamed == 1 else 'results'} streamed to stdout as JSON lines",
            err=True,
        )
        for result in run_result.results:
            click.echo(_json_line(result.as_dict()))
    return EXIT_INTERNAL


def _json_line(payload: Any) -> str:
    """Return one canonical JSON object flattened onto a single stdout line.

    The JSON-lines contract an unwritable report directory carries is one object per line, and
    the product's only JSON formatter indents by two spaces because its other callers write
    files. Only that layout is therefore flattened and nothing else: the text still comes from
    :func:`qrcheck.report.json_report.dumps_canonical`, so keys stay sorted, non-ASCII stays
    literal and the values are the ones the report would have held, and no value can be
    damaged by the flattening because JSON escapes inside a string the only characters that
    could otherwise introduce a line break.

    Args:
        payload: A JSON-ready value, here one case result's projection.

    Returns:
        The object as a single line, with no trailing newline: ``click.echo`` adds one.
    """
    return " ".join(dumps_canonical(payload).splitlines())


# ---------------------------------------------------------------------------
# The click command
# ---------------------------------------------------------------------------


class _ExitCodeCommand(click.Command):
    """A command whose integer return value becomes the process exit code.

    Click 8.1 discards a callback's return value when the command runs standalone, so a
    callback that merely returned :data:`qrcheck.cli.exit.EXIT_CONFIG` would leave the
    process status at 0 however the run went, and every gate built on this command -- the
    empty-selection gate above all -- would pass silently. The callback still returns the
    integer, so a caller and a test read it directly; this class is where that integer
    becomes the process status, and in non-standalone mode click turns it back into the very
    same return value.
    """

    def invoke(self, ctx: click.Context) -> object:
        """Run the callback and, when it returned an exit code, exit with it."""
        result = super().invoke(ctx)
        if isinstance(result, int):
            ctx.exit(result)
        return result


@click.command(name="run", cls=_ExitCodeCommand)
@click.option(
    "--adapter",
    type=str,
    default=None,
    help="Target to exercise; defaults to default_adapter in the configuration.",
)
@click.option(
    "--ec",
    type=click.Choice(list(EC_LEVELS), case_sensitive=False),
    default=None,
    help=f"Error-correction level; defaults to {DEFAULT_EC_LEVEL}.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Run seed; defaults to seed in the configuration.",
)
@click.option(
    "--checks",
    type=str,
    default=None,
    help="Comma-separated check IDs, identifier globs or family:<name>; default all.",
)
@click.option(
    "--class",
    "class_globs",
    multiple=True,
    type=str,
    help="Corpus class-name glob to include; repeatable, default all records.",
)
@click.option(
    "--fail-fast",
    "fail_fast",
    is_flag=True,
    default=False,
    help="Stop launching checks after the first failed or errored check.",
)
@click.option(
    "--report-dir",
    "report_dir",
    # ``file_okay`` stays at click's default on purpose. AAP §0.8.3 makes an unusable
    # ``--report-dir`` exit 3, and ``file_okay=False`` would make click reject a path that is
    # an existing regular file with its own exit 2 before this command ever sees it. The
    # writability probe in :func:`run` decides that condition instead, so a file standing
    # where the directory belongs is reported with the run's exit-code contract.
    type=click.Path(path_type=str),
    default=None,
    help="Report root; defaults to report_dir in the configuration.",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    adapter: str | None,
    ec: str | None,
    seed: int | None,
    checks: str | None,
    class_globs: tuple[str, ...],
    fail_fast: bool,
    report_dir: str | None,
) -> int:
    """Run the check suite against one target and write the JSON, JUnit and HTML reports."""
    return run(
        adapter,
        ec,
        seed,
        checks,
        class_globs,
        fail_fast,
        report_dir,
        config_path=_config_path(ctx),
    )


def _config_path(ctx: click.Context | None) -> str:
    """Return the configuration file the group's ``--config`` option selected.

    ``qrcheck.cli.main`` owns that option, and how it hands the value down is its contract
    rather than this module's: the group may leave the raw string in the context object, keep
    a mapping or a namespace there, or hold the value only in its own click parameters. All
    three are read, in that order, so registering this command under the group needs no
    adaptation of either module. When none of them carries a value -- including a call with no
    click context at all -- the default configuration file name stands.
    """
    if ctx is not None:
        found = _config_value(ctx.obj)
        if found is not None:
            return found
        root_params = ctx.find_root().params
        for key in _CONFIG_KEYS:
            found = _config_value(root_params.get(key))
            if found is not None:
                return found
    return DEFAULT_CONFIG_PATH


def _config_value(value: object) -> str | None:
    """Return the configuration path ``value`` holds, or ``None`` when it holds none."""
    if isinstance(value, str | Path):
        text = str(value)
        return text or None
    if value is None:
        return None
    for key in _CONFIG_KEYS:
        entry = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
        if isinstance(entry, str | Path) and str(entry):
            return str(entry)
    return None

