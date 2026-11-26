"""The ``qrcheck`` command line: the console-script target and the five verbs it offers.

``qrcheck`` tests a QR code *generator*.  This module is the entry point that both the
``qrcheck = "qrcheck.cli.main:main"`` console script declared in ``pyproject.toml`` and
``python -m qrcheck`` (through :mod:`qrcheck.__main__`) reach, and it owns exactly three
things: the click group that carries the group-level ``--config`` option, the registration of
the subcommands, and the translation of whatever a command returns or raises into the process
exit code.

The five subcommands, and no others:

    run      exercise one target and write the JSON, JUnit and HTML reports
    doctor   verify the target and the local toolchain before any check runs
    list     enumerate every registered check
    corpus   regenerate the deterministic payload corpus (``corpus build``)
    loc      report tracked physical lines and gate on the configured ceiling

``main`` returns the process exit code and never calls :func:`sys.exit`, because the console
script and ``qrcheck/__main__.py``'s ``SystemExit(main())`` both consume the integer.  click
therefore runs in non-standalone mode (``standalone_mode=False``), which is what lets a
command's return value reach the caller, and it is also why the exceptions click would
otherwise convert into a process status are mapped here:

    ==========================  ==========================================================
    raised                      returned
    ==========================  ==========================================================
    ``click.UsageError``        ``EXIT_CONFIG``, after ``show()`` on stderr
    ``click.ClickException``    ``EXIT_CONFIG``, after ``show()`` on stderr
    ``click.Abort``             ``EXIT_INTERNAL``, after one ``Aborted!`` line on stderr
    ``click.exceptions.Exit``   the integer code the exception carries
    ``SystemExit``              its integer code, else ``EXIT_INTERNAL``
    any other ``Exception``     ``EXIT_INTERNAL``, after the exception's type and message
                                and ``traceback.format_exc()`` on stderr
    ==========================  ==========================================================

A bad invocation is exit 2, never exit 1: exit 1 means a check failed, and conflating the two
would let a mistyped option look like a failing generator.  A broken installation is different
again -- an ``ImportError`` out of a check family or a command module reaches the last arm and
exits 3, which is the code ``qrcheck/cli/list.py`` documents for it.

The interpreter guard is the first statement of ``main`` and so runs before click parses
anything, which is what makes ``qrcheck --help`` refuse on an unsupported interpreter too.
Below CPython 3.11 it writes one diagnostic naming the running interpreter, the required
version and the reason -- the configuration is read with the standard library's ``tomllib``,
which no interpreter below 3.11 provides -- and returns ``EXIT_CONFIG``.  A newer 3.x is
allowed to run and is reported as an unsupported platform by ``qrcheck doctor`` rather than
refused here.

For that reason the subcommands are registered on first use instead of at import: each of them
imports :mod:`qrcheck.config`, whose package ``__init__`` imports the ``tomllib``-based loader,
so registering them while this module loads would raise :exc:`ImportError` on exactly the
interpreter the guard exists to report on.  The group therefore imports nothing from the
package beyond :mod:`qrcheck.cli.exit`, and resolves and registers its commands the first time
click asks which commands it has.

Exit integers come from :mod:`qrcheck.cli.exit` and are never literals here.  Nothing in this
module prints a banner, reads a clock, an environment variable or the filesystem, holds the
configuration in module state, or changes the working directory.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import click

from qrcheck.cli.exit import EXIT_CONFIG, EXIT_INTERNAL, EXIT_OK

__all__ = ("CliContext", "MINIMUM_PYTHON", "cli", "main")

# The oldest interpreter the product runs on.  Below it the guard in `main` refuses; above it
# the run is allowed and `qrcheck doctor` reports the platform as unsupported.
MINIMUM_PYTHON: Final[tuple[int, int]] = (3, 11)

# Every diagnostic this module writes starts with this, so a message from the command line is
# distinguishable on a shared stderr from whatever a command itself printed.
DIAGNOSTIC_PREFIX: Final[str] = "qrcheck: "


@dataclass(frozen=True, slots=True)
class CliContext:
    """The one value the group hands down to every subcommand.

    The configuration path travels in the click context rather than in a module global or an
    environment variable, so two invocations in one process cannot see each other's value and
    a test may call :func:`main` repeatedly with a different ``--config`` each time.  Each
    subcommand reads it through its own context helper, which accepts this record's
    ``config_path`` attribute directly.
    """

    config_path: str


def _write_stderr(message: str, detail: str = "") -> None:
    """Write one prefixed diagnostic line to stderr, followed by ``detail`` verbatim.

    ``sys.stderr`` is written directly rather than through :func:`click.echo`, so the message
    lands on whichever stream is current when the call happens: a ``CliRunner`` invocation and
    a ``capsys``-captured direct call both see it.

    Args:
        message: The diagnostic, without the prefix and without a trailing newline.
        detail: A block written after the diagnostic as-is, newlines included -- a traceback,
            typically.  Empty when there is nothing to add.
    """
    sys.stderr.write(f"{DIAGNOSTIC_PREFIX}{message}\n")
    if detail:
        sys.stderr.write(detail)


def _unsupported_interpreter_message() -> str:
    """Return the diagnostic the interpreter guard writes on a pre-3.11 interpreter.

    It names the three things a reader needs in order to act: the interpreter actually
    running, the ``CPython`` version required, and the reason that version is required at all.

    Returns:
        The message, without the diagnostic prefix and without a trailing newline.
    """
    running = ".".join(str(part) for part in sys.version_info[:3])
    required = f"CPython {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}"
    return (
        f"{required} or newer is required; this interpreter is {running}. qrcheck reads its "
        f"configuration with the standard library's tomllib, which no interpreter below "
        f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} provides."
    )


class _QrcheckGroup(click.Group):
    """A click group that imports and registers its subcommands the first time it needs them.

    Deferring registration is what keeps the interpreter guard meaningful: the five command
    modules import :mod:`qrcheck.config`, whose package ``__init__`` imports the
    ``tomllib``-based loader, so importing them while this module loads would raise
    :exc:`ImportError` on the very interpreter the guard has to report on.  Nothing imports
    them before :func:`main` has compared the interpreter version.

    Both hooks that reach the command set are overridden, because click resolves a subcommand
    through :meth:`get_command` while it formats the help and answers ``list_commands`` from
    :meth:`list_commands`; a ``CliRunner`` invocation and ``qrcheck --help`` therefore each
    find the same five verbs.  Registration itself is re-entrant -- :mod:`importlib` returns
    the modules Python already holds and :meth:`click.Group.add_command` stores a command under
    its own name -- so a second hook call re-binds the same objects and duplicates nothing.
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Resolve ``cmd_name``, registering the subcommands first.

        Args:
            ctx: The context click is resolving the command for.
            cmd_name: The name the caller typed.

        Returns:
            The command, or ``None`` when no registered command carries that name.
        """
        _register_commands()
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return the registered command names, ascending, registering them first.

        Args:
            ctx: The context click is listing the commands for.

        Returns:
            The command names in the sorted order :meth:`click.Group.list_commands` gives.
        """
        _register_commands()
        return super().list_commands(ctx)


def _exit_with_result(result: object, **_group_params: object) -> object:
    """Turn a subcommand's integer return value into the process exit status.

    Every verb of the product returns its exit code instead of exiting, and under
    :func:`main` -- which is what the console script goes through -- that return value is what
    :func:`main` itself returns.  A ``CliRunner`` invocation, however, runs click standalone,
    where :meth:`click.BaseCommand.main` calls ``ctx.exit()`` with no code and would report
    success however the verb's own gate went.  Raising :exc:`click.exceptions.Exit` here makes
    click's own machinery carry the integer to the process status in both modes, so a test
    harness and CI read the same code as the shell does.

    ``run`` and ``doctor`` convert their own code inside the command and so raise before this
    hook is reached; the verbs that return their code plainly -- ``list``, ``corpus build`` and
    ``loc`` -- are what it exists for.  A command that returned something other than an integer
    is passed through untouched, so a verb that returns nothing stays a plain success.
    """
    if isinstance(result, int):
        raise click.exceptions.Exit(result)
    return result


@click.group(name="qrcheck", cls=_QrcheckGroup, result_callback=_exit_with_result)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=str),
    default="qrcheck.toml",
    show_default=True,
    help="Path to the qrcheck TOML configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """Test a QR code generator against two reference decoders and a golden-encoder oracle."""
    ctx.obj = CliContext(config_path=config_path)


def _register_commands() -> None:
    """Import the five subcommands, load the check families, and add the commands to the group.

    Imported here rather than at module scope so that the interpreter guard in :func:`main`
    runs first (see :class:`_QrcheckGroup`), and reached through the group's own resolution
    hooks, so every path to a command -- ``qrcheck <verb>``, ``qrcheck --help`` and a
    ``CliRunner`` invocation of the group alike -- finds the same five.

    :func:`qrcheck.checks.load_families` runs before the commands are registered, because the
    registry it populates is what ``qrcheck list`` and ``qrcheck run`` enumerate: the family
    package registers its checks at import, and the call is idempotent, so a second one
    re-imports nothing and registers nothing twice.

    Raises:
        ImportError: A command module or a check family module cannot be imported.  Deliberately
            not caught here: a broken installation must reach the last arm of :func:`main` and
            exit 3, never degrade to a shorter list of verbs or of checks.
        CheckRegistrationError: A check family contradicts the registry's own declaration of
            check identity.  Also deliberately uncaught, for the same reason.
    """
    from qrcheck.checks import load_families
    from qrcheck.cli.corpus import corpus_command
    from qrcheck.cli.doctor import doctor_command
    from qrcheck.cli.list import list_command
    from qrcheck.cli.loc import loc_command
    from qrcheck.cli.run import run_command

    load_families()
    for command in (run_command, doctor_command, list_command, corpus_command, loc_command):
        cli.add_command(command)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit code.

    Args:
        argv: The arguments to parse, without the program name -- ``["run", "--ec", "M"]``.
            ``None`` reads ``sys.argv[1:]``, which is what the console script and
            ``python -m qrcheck`` rely on.

    Returns:
        One of :data:`~qrcheck.cli.exit.EXIT_OK`, :data:`~qrcheck.cli.exit.EXIT_FAILED`,
        :data:`~qrcheck.cli.exit.EXIT_CONFIG` or :data:`~qrcheck.cli.exit.EXIT_INTERNAL`, per
        the table in the module docstring.  Never ``None``, even for a command that returned
        nothing: the console script passes this value straight on as the process status.

    Raises:
        Nothing.  Every exception is mapped to an exit code, and the type, the message and the
        traceback of an unexpected one reach stderr before the code is returned, so the failure
        is diagnosable rather than swallowed.
    """
    if sys.version_info[:2] < MINIMUM_PYTHON:
        _write_stderr(_unsupported_interpreter_message())
        return EXIT_CONFIG

    try:
        result = cli.main(
            args=None if argv is None else list(argv),
            prog_name="qrcheck",
            standalone_mode=False,
        )
    # `UsageError` is a `ClickException`, so it is caught first and reported the way click
    # reports it: `show()` writes the usage line and the message, plus a hint where click
    # composed one.
    except click.UsageError as exc:
        exc.show(file=sys.stderr)
        return EXIT_CONFIG
    # A missing option value or a failed type conversion is also a bad invocation: exit 1 is
    # reserved for a check that failed.
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return EXIT_CONFIG
    except click.Abort:
        _write_stderr("Aborted!")
        return EXIT_INTERNAL
    # Defensive arms, in the order click itself resolves them.  Non-standalone mode turns both
    # into a return value rather than a process status, so neither normally escapes; should a
    # nested click path raise one, `main` still returns an integer instead of propagating it.
    # `Exit.exit_code` is always an integer by click's own contract, while `SystemExit.code` may
    # be a string or `None` when a caller passed one to the exit builtin, so only the latter
    # needs a type test before it is returned as a status.
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_INTERNAL
    # The last-resort net: an exception no command anticipated is a framework fault, so the
    # process exits 3 and the type, the message and the traceback are printed, not discarded.
    except Exception as exc:
        _write_stderr(f"internal error: {type(exc).__name__}: {exc}", traceback.format_exc())
        return EXIT_INTERNAL
    # A command that returns nothing is a success, and `main` owes its callers an integer.
    return result if isinstance(result, int) else EXIT_OK


if __name__ == "__main__":  # `python -m qrcheck.cli.main`, equivalent to the console script.
    raise SystemExit(main())
