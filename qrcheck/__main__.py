"""Serve ``python -m qrcheck`` as an exact equivalent of the installed ``qrcheck`` script.

``pyproject.toml`` declares the console script as ``qrcheck = "qrcheck.cli.main:main"``, and this
module exists so that ``python -m qrcheck`` reaches the same code path on an interpreter where
the script is not on ``PATH`` -- an uninstalled checkout, a ``-m`` invocation from a test or a
CI step that has not run the editable install.  The two forms therefore agree on everything a
caller can observe: the same five subcommands (``run``, ``doctor``, ``list``, ``corpus``,
``loc``), the same options and defaults, the same stdout and stderr contract, and the same exit
codes with the same precedence.

The exit codes are the ones the command line documents, and this module carries them unchanged:

    ==  ==========================================================================
    0   the run completed with no failed and no errored result, or a read-only
        verb such as ``list`` succeeded; ``skipped`` results are permitted
    1   at least one check failed and none errored
    2   configuration, selection, corpus-filter, corpus-file, primary-decoder or
        target-availability error, decided before any check runs, no report written
    3   a check errored internally, the report directory is not writable, or an
        unhandled framework fault occurred
    ==  ==========================================================================

During and after a run ``3`` outranks ``1``, which outranks ``0``, and code ``2`` is settled
before execution begins.

Nothing here parses, routes or decides anything.  ``main`` is imported from
:mod:`qrcheck.cli.main` and called with no arguments, so it reads the process arguments exactly
as the console script does; its return value is handed to :exc:`SystemExit`, which is what turns
it into the process status.  No argument parsing, option default, interpreter guard, error
mapping or subcommand table lives here, and re-implementing any of them would create a second
entry path that could disagree with the first.  Because the module adds no behaviour of its own,
no adapter error (``TargetUnavailable``, ``TargetTimeout``, ``TargetRejected``,
``TargetContractError``, ``TargetUnsupported``) and no :class:`~qrcheck.report.model.Reason`
originates here.

Importing this module runs nothing: the dispatch is behind the ``__main__`` guard, so
``import qrcheck.__main__`` in a test collects no command and raises no :exc:`SystemExit`.
"""

from __future__ import annotations

from qrcheck.cli.main import main

# The module exports nothing: `main` belongs to `qrcheck.cli.main`, and re-exporting it would
# give callers a second name for the entry point that the console script does not have.
__all__: list[str] = []

if __name__ == "__main__":
    # `main` returns the exit code instead of exiting; `SystemExit` carries that integer to the
    # process status unmodified, so both entry forms report the same code.
    raise SystemExit(main())
