"""Package façade for :mod:`qrcheck.report`: the one place the product writes a report.

A run produces three artefacts from one :class:`~qrcheck.report.model.RunResult` -- the summary
JSON, the JUnit XML document and the self-contained HTML page, in that order -- and
:func:`write_all_reports` is the only code in the product that writes them. All three are serialised
from the same accumulated model within one call, so they cannot disagree about a status, a reason or
an evidence directory; a second serialisation path would be a second answer to the same question.
``qrcheck.cli.run`` calls it immediately after ``qrcheck.runner.run_suite`` returns, and reads
the returned :class:`ReportPaths` for its one-line summary. Nothing else is decided here -- no
exit code, no duration, no printing, no evidence -- and a writer's exception propagates
untouched, because ``qrcheck.cli.exit`` owns the exit-code contract.

Result types are deliberately not re-exported: ``CaseResult``, ``RunResult``, ``Status``, ``Reason``
and ``failure_id`` have exactly one import path, :mod:`qrcheck.report.model`. The three writers are
imported inside :func:`write_all_reports` rather than at module scope, first because
:mod:`qrcheck.evidence.naming` executes this module's ``__init__`` when it derives a failure ID -- a
module-scope import of the HTML writer would pull Pillow, and of the JUnit writer
:mod:`qrcheck.registry`, into that path -- and second because a function-local import lets a test
substitute one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - the annotations below are the only use
    from qrcheck.report.model import RunResult

__all__ = [
    "HTML_FILENAME",
    "JUNIT_FILENAME",
    "ReportPaths",
    "SUMMARY_FILENAME",
    "write_all_reports",
]

# The three artefact names, spelled out here and nowhere else. A writer carrying its own copy, or
# a caller re-deriving one, would be a second source of truth for a filename that machine
# consumers and the three writer tests locate by name, so the names are a public surface of this
# module.
SUMMARY_FILENAME: str = "summary.json"
JUNIT_FILENAME: str = "junit.xml"
HTML_FILENAME: str = "index.html"


@dataclasses.dataclass(frozen=True)
class ReportPaths:
    """The three artefact paths one run wrote, in the order they were written.

    Returned by :func:`write_all_reports` so that no caller re-derives a filename: the one-line
    stdout summary and the writer tests both read the names from here, which keeps
    :data:`SUMMARY_FILENAME`, :data:`JUNIT_FILENAME` and :data:`HTML_FILENAME` the only place a name
    is spelled out. Frozen because a caller recording what a run wrote must not be able to alter the
    record, and because two runs of the same configuration differ only in the directory the caller
    chose.

    Attributes:
        summary: The summary file, as the JSON writer returned it.
        junit: The JUnit document, as the JUnit writer returned it.
        html: The self-contained page, as the HTML writer returned it.
    """

    summary: Path
    junit: Path
    html: Path


def write_all_reports(
    run_result: RunResult,
    report_dir: Path,
    evidence_root: Path | None = None,
) -> ReportPaths:
    """Write ``run_result``'s three report artefacts under ``report_dir`` and return them.

    The order is fixed -- summary, JUnit document, HTML page -- and the page comes last because it
    embeds the image artefacts the evidence store wrote during the run, so it must not be assembled
    before the tree it reads is final. ``report_dir`` is created here, once, before any artefact is
    attempted, so a destination that cannot be created fails at one point instead of leaving a
    partial report behind. Nothing is retried, cleaned up or logged: the ``OSError`` of a read-only
    or non-directory destination propagates untouched for ``qrcheck.cli.run`` to map to exit 3 with
    the path in its message.

    Args:
        run_result: The completed run; all three artefacts come from this one model.
        report_dir: Destination directory, normally the configured ``report_dir`` or the
            ``--report-dir`` override, used as given and never resolved to an absolute path.
        evidence_root: The directory ``CaseResult.evidence_dir`` is relative to -- the report
            directory itself. Only the HTML writer reads it, to embed the images evidence holds
            already; this function writes no evidence.

    Returns:
        The three paths the writers returned, as :class:`ReportPaths`.

    Raises:
        OSError: ``report_dir`` cannot be created, or a writer cannot create or write its file.
        TypeError: a result field is not JSON-serialisable.
        ValueError: a result field holds a non-finite float, which JSON does not define.
    """
    # Imported here rather than at module scope: see the module docstring. Beyond keeping this
    # package's import stdlib-only, the local import resolves each writer at call time, so a
    # test can substitute one and observe the order they are called in.
    from qrcheck.report.html import write_html
    from qrcheck.report.json_report import write_summary
    from qrcheck.report.junit import write_junit

    directory = Path(report_dir)
    # Created once, before any artefact is attempted, so a destination that cannot take the
    # report raises here rather than after two of the three files are on disk.
    directory.mkdir(parents=True, exist_ok=True)
    # The root the HTML writer joins each ``CaseResult.evidence_dir`` against is the report
    # directory itself, not its evidence child: the evidence store records that value relative
    # to the report directory with the evidence segment already included
    # (``qrcheck.evidence.naming.relative_case_evidence_dir``), so joining it against
    # ``report_dir/evidence`` would resolve one level too deep and leave a failing case's
    # block without its symbol. ``evidence_root`` therefore names the directory the relative
    # evidence path is relative to, and its default is ``report_dir``.
    root = Path(evidence_root) if evidence_root is not None else directory
    summary = write_summary(run_result, directory / SUMMARY_FILENAME)
    junit = write_junit(run_result, directory / JUNIT_FILENAME)
    # Last, so the page embeds the artefacts the run already wrote rather than a tree that is
    # still being filled.
    html = write_html(run_result, directory / HTML_FILENAME, root)
    return ReportPaths(summary=summary, junit=junit, html=html)
