"""Exit-code vocabulary for qrcheck: the only place exit codes are decided.

Every command the product ships -- ``run``, ``doctor``, ``list``, ``corpus build`` and ``loc`` --
returns one of these four integers instead of a literal of its own, so a script gating on the
process status reads a single vocabulary:

    0  the run completed with no ``failed`` and no ``error`` result; a ``skipped`` result is
       permitted and never changes the outcome 1  at least one check ``failed`` and none errored 2 a
       configuration, selection, corpus-filter, corpus-file, primary-decoder or target-availability
       error, decided before any check runs, so no report is written 3  a check errored internally,
       ``report_dir`` is not writable, or an unhandled framework exception occurred

Precedence: 2 is settled before execution and so never competes with a result; during and after a
run, 3 outranks 1, which outranks 0. Nothing here prints, exits, reads a clock, the environment or
the filesystem, and no value is cached between calls, which is what lets the exit-code contract be
tested without a target, a decoder or a report.
"""

from typing import Final

from qrcheck.report.model import RunResult

__all__ = (
    "EXIT_CONFIG",
    "EXIT_FAILED",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "exit_code_for",
)

# 0: the run completed with no failed and no error result; skipped results are allowed.
EXIT_OK: Final[int] = 0
# 1: at least one check failed and none errored.
EXIT_FAILED: Final[int] = 1
# 2: configuration, selection, corpus or primary-decoder error, before any check ran.
EXIT_CONFIG: Final[int] = 2
# 3: a check errored internally, report_dir is unwritable, or an unhandled exception.
EXIT_INTERNAL: Final[int] = 3


def exit_code_for(run_result: RunResult) -> int:
    """Return the process exit code a completed run implies.

    The outcome is read from :attr:`RunResult.counts` alone, never by re-scanning
    ``run_result.results``: the model refuses to build a run whose counts disagree with them. A
    ``skipped`` result is never consulted -- it is neither a failure nor a pass-flavoured outcome --
    so a run of ``passed`` and ``skipped`` results exits :data:`EXIT_OK`. :data:`EXIT_CONFIG` is
    never returned here: exit 2 is decided before execution.

    Returns:
        :data:`EXIT_INTERNAL` when any result carries status ``error``; otherwise
        :data:`EXIT_FAILED` when any carries ``failed``; otherwise :data:`EXIT_OK`.
    """
    if run_result.counts.error > 0:
        return EXIT_INTERNAL
    if run_result.counts.failed > 0:
        return EXIT_FAILED
    return EXIT_OK
