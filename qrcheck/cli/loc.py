"""``qrcheck loc``: report per-file line counts and gate on the configured ceiling.

The measurement belongs to :func:`qrcheck.loc.count_lines` and is deliberately not restated here:
this module walks no tree, reads no source file, counts no line and carries no exclusion list, so
the total it prints, the ``loc_count`` ``summary.json`` records and the number the CI gate compares
cannot disagree about the same tree.

The ceiling has one home as well -- ``loc_limit`` in the configured ``qrcheck.toml`` -- and
``--limit`` only overrides it for one invocation. The outcome is the documented exit-code contract:

* ``0`` - the total is *under* the limit;
* ``1`` - the total is *at or above* it, which is what makes ``qrcheck loc --limit $((count -
  1))`` fail rather than pass;
* ``2`` - the configuration file cannot be used, the counted path is missing or is not a
  directory, or the limit is not a positive integer;
* ``3`` - an I/O error interrupted the read of a path that exists, or the ``report_dir`` probe
  could not create a writable report directory; the diagnostic names the path.

The counted root is the directory holding the configuration file; ``--path`` names another tree.
"""

from __future__ import annotations

from pathlib import Path

import click

from qrcheck.cli.exit import EXIT_CONFIG, EXIT_FAILED, EXIT_INTERNAL, EXIT_OK
from qrcheck.cli.support import DEFAULT_CONFIG_PATH, resolve_config_path
from qrcheck.config import ConfigError, load_config
from qrcheck.loc import count_lines

__all__ = ("loc", "loc_command")


def loc(
    limit: int | None,
    path: str | None,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """Print the line table for one root and return the process exit code.

    ``limit`` defaults to ``loc_limit`` from the configuration file, the only declaration of the
    product's line budget, and ``path`` to the directory holding that file. Returns ``EXIT_OK`` when
    the total is under the limit, ``EXIT_FAILED`` when it is at or above it, ``EXIT_CONFIG`` for a
    configuration fault, an unusable path or a non-positive limit, and ``EXIT_INTERNAL`` when an I/O
    error interrupts the read. Nothing is raised: every failure is reported on stderr and returned
    as an exit code.
    """
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        # str() of a ConfigError is "<dotted key>: <message>", or the bare message for a whole-file
        # fault such as an unreadable file or invalid TOML, which belongs to no single key.
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
        # PermissionError included: every configuration-read fault has already been converted into a
        # ConfigError inside the loader, so the only OSError left is the report_dir mkdir and
        # writability probe, which the loader lets through as an environment fault and which is
        # therefore exit 3. The exception text names the directory that has to change.
        click.echo(f"qrcheck loc: cannot prepare the report directory: {exc}", err=True)
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

    # One line per file in the order the report already sorted them, then the two figures being
    # compared and the verdict, so a CI log or a grep reads the result without parsing prose.
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
    return loc(limit, path, config_path=resolve_config_path(ctx))
