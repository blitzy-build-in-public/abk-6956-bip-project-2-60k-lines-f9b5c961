"""``qrcheck corpus build``: regenerate the deterministic payload corpus from a seed.

The corpus belongs to two other modules and neither is restated here.
``qrcheck.corpus.builder.build_corpus`` assembles the records -- the per-class counts, the
boundary lengths derived from the capacity tables, the stable ``<class>-<3-digit index>``
identifiers and the invariants the defect-detection contract depends on -- and
``qrcheck.corpus.io.write_corpus`` owns the canonical JSONL layout: one object per line,
sorted keys, literal UTF-8 and ``\\n`` endings. This module therefore builds no record, sorts
nothing and serialises nothing; it decides only *whether* the destination has to change, and
reports that decision.

That decision is the whole contract, because ``corpus/default.jsonl`` is a committed
repository file and AAP §0.5.4 promises a regeneration from the same seed is byte-identical to
it. So the freshly built records are rendered to a sibling temporary file through the same
writer, the rendered bytes are compared with the destination's current bytes, and only a real
difference replaces the destination. A matching regeneration leaves the committed file
untouched -- the path CI takes every run, where the drift check compares rather than rewrites --
and a difference refuses to overwrite unless ``--force`` is given, so a hand-edited corpus is
never silently replaced. The temporary file lives beside the destination (never in the shared
system temporary directory) and is removed in a ``finally``, so an interrupted or refused
regeneration leaves neither a stray ``qrcheck-`` entry nor a half-written corpus.

The outcome is the documented exit-code vocabulary of :mod:`qrcheck.cli.exit`:

* ``0`` -- the corpus was written, or the destination already held exactly these bytes;
* ``2`` -- a bad argument or an unusable configuration: an unreadable or invalid
  ``qrcheck.toml``, a negative seed, a destination that is a directory, a parent that is not a
  writable directory, or an existing destination whose content differs and no ``--force``.
  Nothing has been written when this code is returned;
* ``3`` -- the write itself failed, or the corpus could not be built at all, which is a
  framework defect rather than a user error; the diagnostic names the path or the failing
  invariant.

Exactly one line reaches standard output, free of timings and of any path the caller did not
name: ``seed=<n> records=<count> status=<written|unchanged|forced>``.
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


class _DiffersWithoutForce(Exception):
    """The destination holds different bytes and ``--force`` was not given.

    Private to this module and never leaves it: the writing helper raises it so the compare
    step can stay one function, and :func:`build` turns it into the exit-2 diagnostic that
    names the destination and the flag.
    """


def build(
    seed: int | None,
    out: str | None,
    force: bool,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """Regenerate the corpus and return the process exit code.

    Args:
        seed: Seed to build from, or ``None`` for the configured ``seed``, which is the only
            declaration of the default.
        out: Destination file, or ``None`` for the configured ``corpus_path`` (absolute as
            the loader resolved it). A supplied value is used exactly as given.
        force: Replace an existing destination whose content differs from the regenerated
            corpus. Without it such a destination is left exactly as it is.
        config_path: Configuration file to read, supplied by the group's ``--config``
            option.

    Returns:
        :data:`qrcheck.cli.exit.EXIT_OK` when the corpus was written or already matched,
        :data:`qrcheck.cli.exit.EXIT_CONFIG` for a bad argument or unusable configuration,
        and :data:`qrcheck.cli.exit.EXIT_INTERNAL` when the write or the build itself
        failed.

    Raises:
        Nothing. Every failure is reported on stderr and returned as an exit code, so a
        caller propagates one integer instead of handling this module's faults.
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
        # PermissionError included, and every other remaining read failure: only report_dir
        # faults are environment faults, and those are exit 3 rather than a configuration
        # error.
        click.echo(
            f"qrcheck corpus build: cannot read configuration {config_path}: {exc}", err=True
        )
        return EXIT_INTERNAL

    run_seed = config.seed if seed is None else seed
    if run_seed < 0:
        click.echo(
            f"qrcheck corpus build: seed must be a non-negative integer: {run_seed}", err=True
        )
        return EXIT_CONFIG

    destination = Path(out) if out is not None else config.corpus_path
    # The path as the caller named it, so a diagnostic repeats what has to change instead of
    # an absolute path derived from it; the configured default is already absolute.
    label = out if out is not None else str(destination)

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

    The rendering goes through :func:`qrcheck.corpus.io.write_corpus`, so the bytes compared
    and the bytes installed are produced by the one writer that defines the format; this
    function never formats a record itself. Both branches that replace the destination use
    ``os.replace``, so a reader never observes a partially written corpus, and the temporary
    file is removed in a ``finally`` -- after a replace it is already gone, and on every other
    path -- including the refusal -- it is unlinked.

    Args:
        records: The records ``build_corpus`` returned.
        destination: The corpus file to create or replace.
        force: Whether a differing destination may be replaced.

    Returns:
        ``"written"`` when the destination was absent, ``"unchanged"`` when it already held
        exactly these bytes, ``"forced"`` when a differing destination was replaced.

    Raises:
        _DiffersWithoutForce: the destination holds different bytes and ``force`` is false.
            It is left exactly as it is.
        OSError: the parent directory cannot be created, the temporary file cannot be written
            or read, or the replace failed. Raised unwrapped so the diagnostic can name the
            path; :func:`build` maps it to exit code 3.
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


def _place(temporary: Path, destination: Path) -> None:
    """Give ``temporary`` the mode ``destination`` should keep, then move it into place.

    The temporary file comes from ``mkstemp`` with mode ``0600``, so it is re-moded before it
    becomes the corpus: an existing destination keeps the mode it already had, and a new one
    gets the mode a plain ``open(path, "w")`` would have created, which is ``0666`` less the
    process umask. Without this the committed corpus would silently narrow to owner-only on
    every ``--force`` regeneration.
    """
    if destination.exists():
        mode = stat.S_IMODE(destination.stat().st_mode)
    else:
        # The umask is read by setting it: Python exposes no getter, and this is the
        # documented way to learn the value. The window is two syscalls wide in a
        # single-threaded command, and the mask is restored before anything else happens.
        current = os.umask(0)
        os.umask(current)
        mode = 0o666 & ~current
    os.chmod(temporary, mode)
    os.replace(temporary, destination)


def _unusable_destination(destination: Path) -> tuple[str, Path] | None:
    """Return the reason ``destination`` cannot be built into, or ``None`` when it can.

    Three conditions are decided before anything is written, and each is exit 2 because each
    is an argument the caller has to change rather than an environment that failed mid-flight
    (which is what exit 3 is for): the destination is a directory, the nearest existing
    ancestor is not a directory -- a parent path that names a regular file -- or that ancestor
    is not writable.

    Args:
        destination: The corpus file the caller asked for, as given.

    Returns:
        ``(reason, offending path)`` for the first condition that holds, otherwise ``None``.
        The offending path is the destination for the first condition and the offending
        ancestor for the other two, so the diagnostic names the entry to change.
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

    A destination several levels deep may not exist yet, which is the ordinary case for
    ``--out`` naming a fresh directory; the highest ancestor that does exist is the one whose
    type and permissions decide whether that tree can be created.
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
