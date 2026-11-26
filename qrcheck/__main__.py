"""Serve ``python -m qrcheck`` as an exact equivalent of the installed ``qrcheck`` script.

``pyproject.toml`` declares the console script as ``qrcheck = "qrcheck.cli.main:main"``, and this
module exists so that ``python -m qrcheck`` reaches the same code path on an interpreter where the
script is not on ``PATH`` -- an uninstalled checkout, a ``-m`` invocation from a test or a CI step
that has not run the editable install. The two forms therefore agree on everything a caller can
observe: the same five subcommands (``run``, ``doctor``, ``list``, ``corpus``, ``loc``), the same
options and defaults, the same stdout and stderr contract, and the same exit codes with the same
precedence -- ``0`` for a run with no failed and no errored result (``skipped`` permitted) or a
read-only verb, ``1`` when a check failed and none errored, ``2`` for a configuration, selection,
corpus-filter, corpus-file, primary-decoder or target-availability error decided before any check
runs, and ``3`` for an internal error, an unwritable report directory or an unhandled framework
fault. During and after a run ``3`` outranks ``1``, which outranks ``0``, and ``2`` is settled
before execution begins.

Nothing here parses, routes or decides anything: ``main`` is imported from :mod:`qrcheck.cli.main`
and called with no arguments, so it reads the process arguments exactly as the console script does,
and its return value is handed to :exc:`SystemExit`. No argument parsing, option default,
interpreter guard, error mapping or subcommand table lives here, so no adapter error
(``TargetUnavailable``, ``TargetTimeout``, ``TargetRejected``, ``TargetContractError``,
``TargetUnsupported``) and no :class:`~qrcheck.report.model.Reason` originates here. Importing this
module runs nothing: the dispatch is behind the ``__main__`` guard, so ``import qrcheck.__main__``
in a test raises no :exc:`SystemExit`.
"""

from __future__ import annotations

from qrcheck.cli.main import main

# The module exports nothing: `main` belongs to `qrcheck.cli.main`, and re-exporting it would give
# callers a second name for the entry point that the console script does not have.
__all__: list[str] = []

if __name__ == "__main__":
    # `main` returns the exit code instead of exiting; `SystemExit` carries that integer to
    # the process status unmodified, so both entry forms report the same code.
    raise SystemExit(main())
