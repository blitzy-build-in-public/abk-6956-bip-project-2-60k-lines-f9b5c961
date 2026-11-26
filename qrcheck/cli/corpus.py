"""``qrcheck corpus build``: regenerate the deterministic payload corpus from a seed.

The corpus belongs to two other modules and neither is restated here:
``qrcheck.corpus.builder.build_corpus`` assembles the records, and
``qrcheck.corpus.io.write_corpus`` owns the canonical JSONL layout. This module builds no record,
sorts nothing and serialises nothing; it decides only *whether* the destination has to change, and
reports that decision.

That decision is the whole contract, because ``corpus/default.jsonl`` is a committed repository file
and AAP §0.5.4 promises a regeneration from the same seed is byte-identical to it. The freshly
built records are rendered to a sibling temporary file through the same writer, the bytes are
compared with the destination's, and only a real difference replaces the destination: a matching
regeneration leaves the committed file untouched, and a difference refuses to overwrite unless
``--force`` is given.

The outcome is the exit-code vocabulary of :mod:`qrcheck.cli.exit`: ``0`` when the corpus was
written or already held exactly these bytes; ``2`` for a bad argument or an unusable configuration,
with nothing written; ``3`` when the write failed or the corpus could not be built at all. Exactly
one line reaches standard output: ``seed=<n> records=<count> status=<written|unchanged|forced>``.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import click

from qrcheck.cli.exit import EXIT_CONFIG, EXIT_INTERNAL, EXIT_OK
from qrcheck.config import ConfigError, load_config
from qrcheck.corpus import CorpusRecord, build_corpus, write_corpus

__all__ = ("build", "build_command", "corpus", "corpus_command")

# Read only when nothing in the group context supplies the option, which happens when
# `build` is called directly rather than through the CLI. The group's own --config
# option keeps the value a real command line uses.
DEFAULT_CONFIG_PATH: Final[str] = "qrcheck.toml"

# The names a group's context object, or the group's own click parameters, may carry the
# --config value under; tried in this order.
_CONFIG_KEYS: Final[tuple[str, ...]] = ("config", "config_path", "config_file")

# The temporary rendering that is compared before anything is replaced. The prefix is the
# product-wide one, so a stray file is recognisable as qrcheck's, and it stays in the
# destination's own directory so the comparison and the rename happen on one filesystem.
_TEMP_PREFIX: Final[str] = "qrcheck-corpus-"
_TEMP_SUFFIX: Final[str] = ".jsonl"

# The three statuses the single stdout line can report, one per outcome of the comparison.
_STATUS_WRITTEN: Final[str] = "written"
_STATUS_UNCHANGED: Final[str] = "unchanged"
_STATUS_FORCED: Final[str] = "forced"

# The umask assumed when the process's own cannot be read, which is the macOS path: 0o022 is
# the conventional default, so the mode derived from it is the 0o644 a plain open() creates.
_FALLBACK_UMASK: Final[int] = 0o022


class _DiffersWithoutForce(Exception):
    """The destination holds different bytes and ``--force`` was not given.

    Private to this module: the writing helper raises it so the compare step stays one function, and
    :func:`build` turns it into the exit-2 diagnostic naming the destination and the flag.
    """


def build(
    seed: int | None,
    out: str | None,
    force: bool,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """Regenerate the corpus and return the process exit code.

    ``seed`` defaults to the configured ``seed``, ``out`` to the configured ``corpus_path``, and
    ``force`` allows replacing a destination whose bytes differ. Returns ``EXIT_OK`` when the corpus
    was written or already matched, ``EXIT_CONFIG`` for a bad argument or unusable configuration,
    and ``EXIT_INTERNAL`` when the write or the build itself failed. Nothing is raised: every
    failure is reported on stderr and returned.
    """
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        # str() of a ConfigError is "<dotted key>: <message>", or the bare message for a
        # whole-file fault such as an unreadable file or invalid TOML.
        click.echo(f"qrcheck corpus build: {exc}", err=True)
        return EXIT_CONFIG
    except FileNotFoundError as exc:
        # The loader reports a missing configuration file as a ConfigError, so a
        # FileNotFoundError reaching here is the report_dir probe finding no directory to
        # write in, which is an environment fault rather than a configuration one.
        if not Path(config_path).exists():
            click.echo(
                f"qrcheck corpus build: configuration file not found: {config_path}", err=True
            )
            return EXIT_CONFIG
        click.echo(f"qrcheck corpus build: cannot prepare the report directory: {exc}", err=True)
        return EXIT_INTERNAL
    except OSError as exc:
        # The only OSError the loader lets through is the report_dir mkdir and writability
        # probe (PermissionError included): every configuration-read fault is converted into
        # a ConfigError before it leaves that module, so nothing here is a read failure. A
        # report directory the process cannot write is an environment fault, hence exit 3,
        # and str() of the OSError already names the directory.
        click.echo(f"qrcheck corpus build: cannot prepare the report directory: {exc}", err=True)
        return EXIT_INTERNAL

    run_seed = config.seed if seed is None else seed
    if run_seed < 0:
        click.echo(
            f"qrcheck corpus build: seed must be a non-negative integer: {run_seed}", err=True
        )
        return EXIT_CONFIG

    destination = Path(out) if out is not None else config.corpus_path
    # The path as the caller named it, so a diagnostic repeats what has to change instead of
    # an absolute path derived from it; the configured default is already absolute. An empty
    # --out -- the shape a shell produces for an unset variable -- names no path, so it falls
    # back to the destination it resolves to, which is the working directory: every message
    # built from the label names the entry the caller has to change.
    label = str(destination) if not out else out

    problem = _unusable_destination(destination)
    if problem is not None:
        reason, offender = problem
        named = label if offender == destination else str(offender)
        click.echo(f"qrcheck corpus build: {reason}: {named}", err=True)
        return EXIT_CONFIG

    try:
        records = build_corpus(run_seed)
    except ValueError as exc:
        # build_corpus raises its CorpusError -- a ValueError -- for a seed outside the
        # contract and for a corpus that violates an invariant. The seed was validated above
        # and the CLI rejects a non-integer, so this is a framework defect, reported rather
        # than swallowed: a corpus that silently lost a record must never read as a success.
        click.echo(
            f"qrcheck corpus build: cannot build the corpus for seed {run_seed}: {exc}", err=True
        )
        return EXIT_INTERNAL

    try:
        status = _regenerate(records, destination, force=force)
    except _DiffersWithoutForce:
        click.echo(
            f"qrcheck corpus build: {label} already exists with different content; "
            f"pass --force to overwrite it",
            err=True,
        )
        return EXIT_CONFIG
    except OSError as exc:
        click.echo(f"qrcheck corpus build: cannot write {label}: {exc}", err=True)
        return EXIT_INTERNAL

    # One deterministic line, and the destination only when the caller named it, so the
    # output of the CI drift step carries no path that a machine or a reviewer has to strip.
    named = f" path={out}" if out is not None else ""
    click.echo(f"seed={run_seed} records={len(records)} status={status}{named}")
    return EXIT_OK


def _regenerate(records: Sequence[CorpusRecord], destination: Path, *, force: bool) -> str:
    """Render ``records`` beside ``destination`` and move them into place if they differ.

    The rendering goes through :func:`qrcheck.corpus.io.write_corpus`, so the bytes compared and the
    bytes installed come from the one writer that defines the format. Both replacing branches use
    ``os.replace``, so a reader never observes a partially written corpus, and the temporary file is
    removed in a ``finally``. Returns ``"written"``, ``"unchanged"`` or ``"forced"``.

    Raises:
        _DiffersWithoutForce: The destination holds different bytes and ``force`` is false.
        OSError: The parent directory, temporary file or replace failed, raised unwrapped to name
        the path.
    """
    parent = destination.parent
    # write_corpus creates missing parents, but the temporary file has to be created in a
    # directory that already exists, so the parent is made first -- exactly the directory the
    # writer would have made.
    parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=str(parent))
    os.close(handle)
    temporary = Path(temp_name)
    try:
        write_corpus(records, temporary)
        rendered = temporary.read_bytes()
        if destination.exists():
            if destination.read_bytes() == rendered:
                return _STATUS_UNCHANGED
            if not force:
                raise _DiffersWithoutForce
            _place(temporary, destination)
            return _STATUS_FORCED
        _place(temporary, destination)
        return _STATUS_WRITTEN
    finally:
        temporary.unlink(missing_ok=True)


def _process_umask() -> int:
    """Return the umask this process applies, without changing it.

    ``os.umask`` reports the mask only by *setting* it first, which would leave every file another
    writer created in that window world-writable; the kernel publishes the value in the ``Umask:``
    field of ``/proc/self/status`` instead. Returns the octal mask applied, ``0o022`` when that
    field cannot be read or carries no octal number.
    """
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _FALLBACK_UMASK
    for line in status.splitlines():
        field, _, value = line.partition(":")
        if field == "Umask":
            try:
                return int(value.strip(), 8)
            except ValueError:
                return _FALLBACK_UMASK
    return _FALLBACK_UMASK


def _place(temporary: Path, destination: Path) -> None:
    """Give ``temporary`` the mode ``destination`` should keep, then move it into place.

    ``mkstemp`` creates mode ``0600``, so the file is re-moded before it becomes the corpus: an
    existing destination keeps its mode and a new one gets what a plain ``open(path, "w")`` would
    have created. Without this the committed corpus would silently narrow to owner-only on every
    ``--force`` regeneration.
    """
    if destination.exists():
        mode = stat.S_IMODE(destination.stat().st_mode)
    else:
        # The mask comes from /proc/self/status through _process_umask, never from
        # os.umask: reading it that way would set it to 0000 and restore it, leaving a
        # window in which a concurrently created file is world-writable.
        mode = 0o666 & ~_process_umask()
    os.chmod(temporary, mode)
    os.replace(temporary, destination)


def _unusable_destination(destination: Path) -> tuple[str, Path] | None:
    """Return the reason ``destination`` cannot be built into, or ``None`` when it can.

    Three conditions are decided before anything is written, each exit 2 because each is an argument
    the caller has to change: the destination is a directory, its nearest existing ancestor is not a
    directory, or that ancestor is not writable.

    Returns:
        ``(reason, offending path)`` for the first condition that holds, otherwise ``None``, the
        path naming the entry to change.
    """
    if destination.exists() and destination.is_dir():
        return "destination is a directory", destination
    ancestor = _nearest_existing_ancestor(destination.parent)
    if not ancestor.is_dir():
        return "destination parent is not a directory", ancestor
    if not os.access(ancestor, os.W_OK):
        return "destination parent is not writable", ancestor
    return None


def _nearest_existing_ancestor(start: Path) -> Path:
    """Return ``start`` when it exists, otherwise its nearest existing ancestor.

    A destination several levels deep may not exist yet, and the highest ancestor that does is the
    one whose type and permissions decide whether that tree can be created.
    """
    candidate = start
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            # The filesystem root, which always exists: only reachable for a path that cannot
            # be created at all, and it is the honest answer for which ancestor decides.
            return candidate
        candidate = parent
    return candidate


@click.group(name="corpus")
def corpus_command() -> None:
    """Build the deterministic payload corpus."""


@corpus_command.command(name="build")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Seed to generate from; defaults to seed in the configuration.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Destination file; defaults to corpus_path in the configuration.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing corpus whose content differs from the regenerated one.",
)
@click.pass_context
def build_command(
    ctx: click.Context, seed: int | None, out: str | None, force: bool
) -> int:
    """Regenerate the corpus from a seed and report whether the file changed."""
    return build(seed, out, force, config_path=_config_path(ctx))


# The group is the `corpus` verb the CLI registers, so the name a caller imports it under
# should not have to be looked up: both names are the same click object.
corpus = corpus_command


def _config_path(ctx: click.Context | None) -> str:
    """Return the configuration file the group's ``--config`` option selected.

    The click context object, a mapping and a namespace there are read in that order; a call with no
    click context at all falls back to the default configuration file name.
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
