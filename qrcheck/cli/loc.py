"""``qrcheck loc``: report per-file line counts and gate on the configured ceiling.

The measurement itself belongs to :func:`qrcheck.loc.count_lines` and is
deliberately not restated here: this module does not walk a directory tree, read a
source file, count a line or carry the exclusion list, so the total it prints, the
``loc_count`` ``summary.json`` records and the number the CI gate compares can never
disagree about the same tree.

The ceiling has one home as well - ``loc_limit`` in the configured ``qrcheck.toml``,
whose default ``qrcheck/config/models.py`` owns. ``--limit`` only overrides that
value for one invocation, and no numeric fallback is defined in this module. The
outcome is the documented exit-code contract:

* ``0`` - the total is *under* the limit;
* ``1`` - the total is *at or above* it, which is what makes
  ``qrcheck loc --limit $((count - 1))`` fail rather than pass;
* ``2`` - the configuration file cannot be used, the counted path is missing or is
  not a directory, or the limit is not a positive integer;
* ``3`` - an I/O error interrupted the read of a path that exists.

The counted root is the directory holding the configuration file, so the committed
``qrcheck.toml`` at the repository root counts the repository itself whichever
directory the command was launched from; ``--path`` names another tree, and a
configuration path with no directory part resolves against the working directory the
command started in. This command writes no file of its own: the only filesystem
change it can cause is the ``report_dir`` creation and writability probe that
loading the configuration performs. Standard output carries the table and nothing
else, which is what keeps the output greppable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import click

from qrcheck.cli.exit import EXIT_CONFIG, EXIT_FAILED, EXIT_INTERNAL, EXIT_OK
from qrcheck.config import ConfigError, load_config
from qrcheck.loc import count_lines

__all__ = ("loc", "loc_command")

# Read only when nothing in the group context supplies the option, which happens
# when this module's function is called directly rather than through the CLI. The
# group's own --config option keeps the value a real command line uses.
DEFAULT_CONFIG_PATH: Final[str] = "qrcheck.toml"

# The names a group's context object, or the group's own click parameters, may
# carry the --config value under; tried in this order.
_CONFIG_KEYS: Final[tuple[str, ...]] = ("config", "config_path", "config_file")


def loc(
    limit: int | None,
    path: str | None,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """Print the line table for one root and return the process exit code.

    Args:
        limit: Ceiling to compare the total against, or ``None`` to use
            ``loc_limit`` from the configuration file, which is the only
            declaration of the product's line budget.
        path: Tree to measure, or ``None`` for the directory holding the
            configuration file.
        config_path: Configuration file to read, supplied by the group's
            ``--config`` option.

    Returns:
        :data:`qrcheck.cli.exit.EXIT_OK` when the total is under the limit,
        :data:`qrcheck.cli.exit.EXIT_FAILED` when it is at or above it,
        :data:`qrcheck.cli.exit.EXIT_CONFIG` for a configuration fault, an unusable
        path or a non-positive limit, and :data:`qrcheck.cli.exit.EXIT_INTERNAL` when
        an I/O error interrupts the read.

    Raises:
        Nothing. Every failure is reported on stderr and returned as an exit code, so
        a caller propagates one integer instead of handling this module's faults.
    """
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        # str() of a ConfigError is "<dotted key>: <message>", or the bare message
        # for a whole-file fault such as an unreadable file or invalid TOML, which
        # belongs to no single key.
        click.echo(f"qrcheck loc: {exc}", err=True)
        return EXIT_CONFIG
    except FileNotFoundError as exc:
        # The loader converts a missing configuration file into a ConfigError, so a
        # FileNotFoundError reaching here is either that boundary widening - still a
        # configuration fault naming the file - or the report_dir probe failing to
        # find a directory to write in, which the loader lets through as an
        # environment fault. The two are told apart by whether the file is there.
        if not Path(config_path).exists():
            click.echo(f"qrcheck loc: configuration file not found: {config_path}", err=True)
            return EXIT_CONFIG
        click.echo(f"qrcheck loc: cannot prepare the report directory: {exc}", err=True)
        return EXIT_INTERNAL
    except OSError as exc:
        # PermissionError included, and every other remaining read failure: only
        # report_dir faults are environment faults, which the loader lets through
        # as OSError, and those are exit 3 rather than a configuration error.
        click.echo(f"qrcheck loc: cannot read configuration {config_path}: {exc}", err=True)
        return EXIT_INTERNAL

    # The configuration file's directory is the repository root for the committed
    # qrcheck.toml, and the working directory when that path carries none, because
    # the loader resolves it against the working directory.
    root = Path(path) if path is not None else config.path.parent
    # The path as the caller named it, so a diagnostic repeats what has to change
    # instead of an absolute path derived from it.
    label = str(root) if path is None else path

    if not root.exists():
        click.echo(f"qrcheck loc: path does not exist: {label}", err=True)
        return EXIT_CONFIG
    if not root.is_dir():
        click.echo(f"qrcheck loc: path is not a directory: {label}", err=True)
        return EXIT_CONFIG

    ceiling = config.loc_limit if limit is None else limit
    if ceiling <= 0:
        click.echo(f"qrcheck loc: limit must be a positive integer: {ceiling}", err=True)
        return EXIT_CONFIG

    try:
        report = count_lines(root)
    except OSError as exc:
        # The existence checks above already answered a missing or non-directory
        # path with exit 2, so a failure here is a read interrupted mid-walk.
        click.echo(f"qrcheck loc: cannot measure {label}: {exc}", err=True)
        return EXIT_INTERNAL

    # One line per file in the order the report already sorted them, then the two
    # figures being compared and the verdict, so a CI log or a grep reads the
    # result without parsing prose.
    for entry in report.files:
        click.echo(f"{entry.lines}\t{entry.path}")
    click.echo(f"total\t{report.total}")
    click.echo(f"limit\t{ceiling}")
    under = report.total < ceiling
    click.echo("result\t" + ("under" if under else "at-or-above"))
    return EXIT_OK if under else EXIT_FAILED


@click.command(name="loc")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Ceiling on tracked physical lines; defaults to loc_limit in the configuration.",
)
@click.option(
    "--path",
    type=click.Path(exists=False, path_type=str),
    default=None,
    help="Tree to count; defaults to the directory holding the configuration file.",
)
@click.pass_context
def loc_command(ctx: click.Context, limit: int | None, path: str | None) -> int:
    """Report per-file physical line counts and gate on the configured ceiling."""
    return loc(limit, path, config_path=_config_path(ctx))


def _config_path(ctx: click.Context | None) -> str:
    """Return the configuration file the group's ``--config`` option selected.

    ``qrcheck.cli.main`` owns that option, and how it hands the value down is its
    contract rather than this module's: the group may leave the raw string in the
    context object, keep a mapping or a namespace there, or hold the value only in
    its own click parameters. All three are read, in that order, so registering this
    command under the group needs no adaptation of either module. When none of them
    carries a value - including a call with no click context at all - the default
    configuration file name stands.
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
