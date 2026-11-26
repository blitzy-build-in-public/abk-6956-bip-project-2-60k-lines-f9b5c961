"""Exit-code vocabulary for qrcheck: the only place exit codes are decided.

Every command the product ships - ``run``, ``doctor``, ``list``, ``corpus build`` and
``loc`` - returns one of these four integers instead of a literal of its own, so a
script gating on the process status reads a single vocabulary and a change to what a
code means happens in this module alone.

The four codes, and what each one means:

    0  the run completed with no ``failed`` and no ``error`` result; a ``skipped``
       result is permitted and never changes the outcome
    1  at least one check ``failed`` and none errored
    2  a configuration, selection, corpus-filter, corpus-file, primary-decoder or
       target-availability error, decided before any check runs, so no report is
       written
    3  a check errored internally, ``report_dir`` is not writable, or an unhandled
       framework exception occurred

Precedence: 2 is settled before execution and so never competes with a result; during
and after a run, 3 outranks 1, which outranks 0.

Nothing here prints, exits, reads a clock, the environment or the filesystem, and no
value is cached between calls: each return value is a pure function of the run's
tallies, which is what lets the exit-code contract be tested without a target, a
decoder or a report.
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
    ``run_result.results``: the model recomputes those tallies from the results and
    refuses to build a run whose counts disagree with them, so the counts are the one
    authoritative reading and this function cannot contradict ``summary.json`` about
    the same run.

    A ``skipped`` result is never consulted. A skip is neither a failure nor a
    pass-flavoured outcome - it is tallied in its own integer and merely permitted -
    so a run of ``passed`` and ``skipped`` results exits :data:`EXIT_OK`. The gate
    that demands a reference run with zero skips asserts that on the report; it is not
    a rule of this function.

    :data:`EXIT_CONFIG` is never a return value here. Exit code 2 is decided *before*
    execution - while the configuration is validated, the selection is parsed, the
    corpus is filtered and loaded, the primary decoder is probed, or the target is
    reached - and a :class:`RunResult` exists only once every check has run. The call
    site that finds no run to report decides 2 for itself.

    Args:
        run_result: The finished run whose tallies are being mapped.

    Returns:
        :data:`EXIT_INTERNAL` when any result carries status ``error``; otherwise
        :data:`EXIT_FAILED` when any result carries status ``failed``; otherwise
        :data:`EXIT_OK`.
    """
    if run_result.counts.error > 0:
        return EXIT_INTERNAL
    if run_result.counts.failed > 0:
        return EXIT_FAILED
    return EXIT_OK
