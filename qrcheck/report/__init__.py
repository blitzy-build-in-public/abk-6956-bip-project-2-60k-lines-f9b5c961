"""Package façade for :mod:`qrcheck.report`: the one place the product writes a report.

A run produces three artefacts from one :class:`~qrcheck.report.model.RunResult` - the
summary JSON, the JUnit XML document and the self-contained HTML page, in that order -
and :func:`write_all_reports` is the only code in the product that writes them. Because
all three are serialised from the same accumulated model within one call, they cannot
disagree about a status, a reason or an evidence directory; a second serialisation path
anywhere would be a second answer to the same question, so none exists (AAP §0.8.1,
§0.6.1 criterion 6).

``qrcheck.cli.run`` calls :func:`write_all_reports` immediately after
``qrcheck.runner.run_suite`` returns and reads the returned :class:`ReportPaths` for its
one-line summary. This module decides nothing else: it takes no exit code, measures no
duration, prints nothing, writes no evidence, and creates no directory beyond the report
directory itself. A writer that raises lets its exception propagate untouched, because
``qrcheck.cli.exit`` is the single owner of the exit-code contract - an unwritable
``report_dir`` becomes exit 3 there, with the path in the message (AAP §0.3.4).

Result types are deliberately not re-exported. ``CaseResult``, ``RunResult``, ``Status``,
``Reason`` and ``failure_id`` have exactly one import path,
:mod:`qrcheck.report.model`; re-exporting them here would give every consumer a choice of
two and would make the cost of importing this package depend on the model's imports.

Importing this module stays cheap and stdlib-only. The three writers are imported inside
:func:`write_all_reports` rather than at module scope for two reasons. First,
:mod:`qrcheck.evidence.naming` imports :func:`~qrcheck.report.model.failure_id` in the
functions that derive a failure ID, so it executes this module's ``__init__``; a
module-scope import of the HTML writer would pull :mod:`qrcheck.evidence.mux` and, with
it, Pillow into that path, and a module-scope import of the JUnit writer would pull
:mod:`qrcheck.registry` into it as well. Second, a function-local import resolves each
writer at call time, which is what lets a test substitute one.
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

# The three artefact names, spelled out here and nowhere else. A writer carrying its own
# copy, or a caller re-deriving one, would be a second source of truth for a filename that
# machine consumers and the three writer tests locate by name, so the names are a public
# surface of this module and their values are fixed by AAP §0.3.4.
SUMMARY_FILENAME: str = "summary.json"
JUNIT_FILENAME: str = "junit.xml"
HTML_FILENAME: str = "index.html"


@dataclasses.dataclass(frozen=True)
class ReportPaths:
    """The three artefact paths one run wrote, in the order they were written.

    Returned by :func:`write_all_reports` so that no caller re-derives a filename: the
    one-line stdout summary ``qrcheck.cli.run`` prints and the writer tests both read the
    names from here, which keeps :data:`SUMMARY_FILENAME`, :data:`JUNIT_FILENAME` and
    :data:`HTML_FILENAME` the only place a name is spelled out.

    Frozen because a caller recording what a run wrote must not be able to alter the
    record, and because two runs of the same configuration differ only in the directory
    the caller chose (AAP §0.5.4).

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

    The order is fixed: the summary first, then the JUnit document, then the HTML page.
    The page comes last because it embeds the image artefacts the evidence store wrote
    during the run, so it must not be assembled before the tree it reads is final
    (AAP §0.8.4).

    ``report_dir`` is created here, once, before any artefact is attempted: this is the
    single place that guarantees the destination exists, so a destination that cannot be
    created fails at one point instead of leaving a partial report behind. Nothing is
    retried, cleaned up or logged - the product has no retry path anywhere
    (AAP §0.5.6) - and the ``OSError`` raised by a read-only or non-directory destination
    propagates untouched for ``qrcheck.cli.run`` to map to exit 3 with the path in its
    message (AAP §0.3.4).

    Args:
        run_result: The completed run. All three artefacts are serialised from this one
            model, which is why they cannot disagree about a status.
        report_dir: Destination directory, normally the configured ``report_dir`` or the
            ``--report-dir`` override. Used as given and never resolved to an absolute
            path, because no artefact may record one (AAP §0.5.5).
        evidence_root: The directory :attr:`~qrcheck.report.model.CaseResult.evidence_dir`
            is relative to, which is the report directory itself. Defaults to
            ``report_dir``. Only the HTML writer reads it, to embed the images
            the evidence store already wrote: this function writes no evidence and never
            creates this directory.

    Returns:
        The three paths the writers returned, as :class:`ReportPaths`.

    Raises:
        OSError: ``report_dir`` cannot be created, or a writer cannot create or write its
            file. Propagated untouched for the CLI to turn into exit 3.
        TypeError: a result field is not JSON-serialisable. A check's defect must fail the
            run loudly rather than be published as a plausible-looking report.
        ValueError: a result field holds a non-finite float, which JSON does not define.
    """
    # Imported here rather than at module scope: see the module docstring. Beyond keeping
    # this package's import stdlib-only, the local import resolves each writer at call
    # time, so a test can substitute one and observe the order they are called in.
    from qrcheck.report.html import write_html
    from qrcheck.report.json_report import write_summary
    from qrcheck.report.junit import write_junit

    directory = Path(report_dir)
    # Created once, before any artefact is attempted, so a destination that cannot take
    # the report raises here rather than after two of the three files are on disk.
    directory.mkdir(parents=True, exist_ok=True)
    # The root the HTML writer joins each ``CaseResult.evidence_dir`` against is the report
    # directory itself, not its evidence child: the evidence store records that value
    # relative to the report directory with the evidence segment already included
    # (``qrcheck.evidence.naming.relative_case_evidence_dir``), so joining it against
    # ``report_dir/evidence`` would resolve one level too deep and leave a failing case's
    # block without its symbol. ``evidence_root`` therefore names the directory the
    # relative evidence path is relative to, and its default is ``report_dir``
    # (AAP §0.3.4 evidence layout, §0.6.1 criterion 6).
    root = Path(evidence_root) if evidence_root is not None else directory
    summary = write_summary(run_result, directory / SUMMARY_FILENAME)
    junit = write_junit(run_result, directory / JUNIT_FILENAME)
    # Last, so the page embeds the artefacts the run already wrote rather than a tree that
    # is still being filled.
    html = write_html(run_result, directory / HTML_FILENAME, root)
    return ReportPaths(summary=summary, junit=junit, html=html)
