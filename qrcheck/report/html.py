"""The product's only human interface: one self-contained ``index.html`` per run.

A person who opens this file has three questions - did the run pass, what failed, and
can I see it (AAP §0.8.4) - and the document answers them before they scroll, in the
order: verdict and tallies, then degradation banners, then every case in one table - or
the empty state when the run recorded no case at all - then a detail block per failure,
then the skips, then how to reproduce a case.

The file is self-contained by construction, which is what "opens with no server and no
network" costs:

* **No script of any kind.** There is no script element, no event-handler attribute of
  any kind and no script-scheme URL, so the document has no client-side state and nothing
  to execute.
* **No external origin.** No stylesheet link, no iframe or object element, no ``base``,
  no stylesheet import rule and no external resource reference at all: the single
  ``<style>`` element carries every rule, and every image is an inline ``data:`` URI built by
  :func:`qrcheck.evidence.mux.to_data_uri`. That is also why a target's SVG is embedded
  as a ``data:`` URI inside an image element rather than inlined as markup: an image
  context makes untrusted vector content inert.
* **Escaped text.** Everything the target or the run supplied - a case identifier, a
  payload, a decoder message - is untrusted (AAP §0.3.2) and reaches the document only
  through :data:`html_escape` with ``quote=True``. JSON is rendered as escaped text
  inside ``<pre>``, never interpolated raw.
* **No design system.** There is no component library, no design-token file and no
  external font to conform to, so none is introduced: one inline stylesheet, system font
  stacks only, and a single ``@media (max-width: 720px)`` rule that collapses the layout
  to one column. A later change that introduces a framework would break the
  offline promise this module exists to keep.

The document is a pure function of the :class:`~qrcheck.report.model.RunResult` and the
evidence bytes already on disk, so it is exactly as stable as the result model and no
more (AAP §0.5.4). Nothing here reads a clock - the run's duration comes from
:attr:`~qrcheck.report.model.RunResult.duration_s` - nothing formats through the locale,
nothing prints or logs, every directory listing is sorted, and no timestamp, hostname,
user name or absolute path is emitted (AAP §0.5.5). Two writes of the same run therefore
produce byte-identical files.

The only volatile content it carries is the duration the model already holds, which is
why the report is compared semantically rather than byte for byte across runs. Writing
is direct - no temporary file - because a stray file inside the report directory would be
an artefact outside the closed evidence set that ``CL-ARTIFACTS`` inspects (AAP §0.3.4).
"""

from collections.abc import Mapping, Sequence
from html import escape as html_escape
from pathlib import Path

from qrcheck.evidence.mux import to_data_uri
from qrcheck.evidence.naming import is_evidence_artifact
from qrcheck.report.json_report import dumps_canonical
from qrcheck.report.model import CaseResult, Reason, RunResult, Status

__all__ = ["write_html"]

# The status word -> the CSS class that colour-codes its table row and detail block. Keyed
# by :class:`~qrcheck.report.model.Status`, a ``StrEnum``, so a plain string value finds
# the same entry.
_STATUS_CLASSES: dict[str, str] = {
    Status.PASSED.value: "status-passed",
    Status.FAILED.value: "status-failed",
    Status.SKIPPED.value: "status-skipped",
    Status.ERROR.value: "status-error",
}

# The statuses that get a detail block. A skip is reported in its own section instead: it
# is a real outcome that is never a pass (AAP §0.3.4), so it must not be filed with the
# failures a reader triages.
_DETAIL_STATUSES: tuple[Status, ...] = (Status.FAILED, Status.ERROR)

# One plain sentence per machine-readable skip reason that can occur, in the order the
# reasons are declared in the model. Only the reasons that actually occur are rendered,
# but the vocabulary a reader greps for is always the literal reason value.
_REASON_EXPLANATIONS: tuple[tuple[str, str], ...] = (
    (
        Reason.UNSUPPORTED_CAPABILITY.value,
        "The target did not declare a capability this check requires, so the case was "
        "not attempted and is not counted as a pass.",
    ),
    (
        Reason.MISSING_DEPENDENCY.value,
        "An optional runtime dependency was absent, so the check could not make its "
        "observation and is not counted as a pass.",
    ),
    (
        Reason.UNSUPPORTED_PLATFORM.value,
        "This platform does not expose what the check observes, so the case reports a "
        "skip and is not counted as a pass.",
    ),
)

# The two suffixes an evidence artefact must carry to be worth embedding. The remaining
# members of the closed set are the JSON documents, which are rendered as text from the
# result model rather than re-read from disk.
_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".svg")

# The single stylesheet of the document. Logical properties throughout, system font
# stacks only, colour never the sole carrier of meaning - every status is also spelled out
# in its row - and not one external resource reference, so the file needs no network.
_STYLE: str = """\
*, *::before, *::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 1.5rem;
  background: #ffffff;
  color: #1b1b1f;
  font-family: system-ui, sans-serif;
  font-size: 1rem;
  line-height: 1.5;
}

h1, h2, h3, h4 {
  margin-block: 1.25rem 0.5rem;
  line-height: 1.25;
}

h1 {
  margin-block-start: 0;
  font-size: 1.5rem;
}

h2 { font-size: 1.25rem; }
h3 { font-size: 1.0625rem; }
h4 { font-size: 0.9375rem; }

code, pre, .command {
  font-family: ui-monospace, monospace;
  font-size: 0.875rem;
}

a { color: #1b5fb0; }

a:hover {
  color: #13457f;
  text-decoration-thickness: 2px;
}

a:active {
  color: #0d2f57;
}

a:focus-visible {
  outline: 3px solid #1b5fb0;
  outline-offset: 2px;
}

.skip-link {
  position: absolute;
  inset-block-start: -3rem;
  inset-inline-start: 0.5rem;
  padding: 0.5rem 0.75rem;
  border: 2px solid #1b5fb0;
  border-radius: 0.375rem;
  background: #ffffff;
}

.skip-link:focus {
  inset-block-start: 0.5rem;
}

header {
  border-block-end: 2px solid #d0d0d8;
  padding-block-end: 1rem;
}

.verdict {
  margin-block: 0.5rem;
  font-size: 1.375rem;
  font-weight: 700;
}

.verdict-passed { color: #14602a; }
.verdict-failed { color: #96201a; }
.verdict-errored { color: #5f1d8a; }
.verdict-skipped { color: #6b4b00; }
.verdict-empty { color: #3f3f47; }

.facts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.5rem 1rem;
  margin: 0;
}

.facts div { margin: 0; }

.facts dt {
  color: #55555f;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.facts dd {
  margin: 0;
  font-weight: 600;
  overflow-wrap: break-word;
}

.counts {
  display: grid;
  grid-template-columns: repeat(4, minmax(6rem, 1fr));
  gap: 0.75rem;
  margin: 1rem 0 0;
  padding: 0;
  list-style: none;
}

.count {
  border: 1px solid #d0d0d8;
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
}

.count-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
}

.count-label {
  color: #55555f;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.count-passed { border-inline-start: 4px solid #14602a; }
.count-failed { border-inline-start: 4px solid #96201a; }
.count-skipped { border-inline-start: 4px solid #8a6d00; }
.count-error { border-inline-start: 4px solid #5f1d8a; }

.banners { margin-block: 1rem 0; }

.banner {
  margin-block: 0.5rem;
  padding: 0.5rem 0.75rem;
  border-inline-start: 4px solid #8a6d00;
  background: #fdf8e6;
}

.empty-state {
  margin-block: 1rem 0;
  padding: 0.75rem 1rem;
  border: 1px solid #d0d0d8;
  border-inline-start: 4px solid #55555f;
  border-radius: 0.375rem;
  background: #f7f7fa;
}

table {
  inline-size: 100%;
  border-collapse: collapse;
}

caption {
  padding-block: 0.25rem 0.5rem;
  color: #55555f;
  font-size: 0.875rem;
  text-align: start;
}

th, td {
  padding: 0.375rem 0.5rem;
  border-block-end: 1px solid #e4e4ec;
  text-align: start;
  vertical-align: top;
}

thead th {
  border-block-end: 2px solid #d0d0d8;
  color: #55555f;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

tbody tr.status-passed { background: #f2fbf4; }
tbody tr.status-failed { background: #fdf2f1; }
tbody tr.status-skipped { background: #fbf7ea; }
tbody tr.status-error { background: #f6f1fc; }

tbody tr.status-passed td:nth-child(3) { color: #14602a; font-weight: 600; }
tbody tr.status-failed td:nth-child(3) { color: #96201a; font-weight: 600; }
tbody tr.status-skipped td:nth-child(3) { color: #6b4b00; font-weight: 600; }
tbody tr.status-error td:nth-child(3) { color: #5f1d8a; font-weight: 600; }

pre, .command {
  margin-block: 0.5rem;
  padding: 0.75rem;
  border: 1px solid #d0d0d8;
  border-radius: 0.375rem;
  background: #f7f7fa;
  overflow-x: auto;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

.command { padding: 0.5rem 0.75rem; }

.evidence {
  margin-block: 1rem;
  padding: 0.75rem 1rem;
  border: 1px solid #d0d0d8;
  border-radius: 0.375rem;
}

.evidence-image { margin: 0 0 1rem; }

.evidence-image img {
  max-inline-size: 100%;
  block-size: auto;
  border: 1px solid #d0d0d8;
  background: #ffffff;
}

figcaption {
  color: #55555f;
  font-size: 0.8125rem;
}

.no-image {
  color: #55555f;
  font-style: italic;
}

.evidence-unreadable {
  color: #6b4b00;
}

.skips { padding-inline-start: 1.25rem; }

.skips li { margin-block-end: 0.25rem; }

footer {
  margin-block-start: 2rem;
  border-block-start: 2px solid #d0d0d8;
  padding-block-start: 0.5rem;
}

@media (max-width: 720px) {
  body { padding: 0.75rem; }
  .facts, .counts { grid-template-columns: 1fr; }
  table, thead, tbody, tr, th, td { display: block; }
  caption { display: block; }
  thead tr {
    position: absolute;
    inline-size: 1px;
    block-size: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  tbody tr {
    margin-block-end: 0.5rem;
    padding: 0.25rem 0.5rem;
    border: 1px solid #d0d0d8;
    border-radius: 0.375rem;
  }
  tbody td { border-block-end: 0; padding-block: 0.125rem; }
  tbody td::before {
    content: attr(data-label) ": ";
    color: #55555f;
    font-weight: 600;
  }
}
"""


# The paragraph a detail block carries when the case recorded no image artefact at all: a
# declared rejection has no symbol, and a listed directory may legitimately hold no image
# artefact. An artefact that exists but cannot be read is never reported as this - that is a
# read failure, and the block states it as one (see _render_images), as it does for a
# directory that cannot be listed, which has no listing to be absent from.
_NO_IMAGE: str = '<p class="no-image">no image recorded for this case</p>'


def write_html(
    run_result: RunResult, path: str | Path, evidence_root: str | Path
) -> Path:
    """Write ``run_result`` as a self-contained ``index.html`` and return the path.

    The document is built in memory from the result model and the evidence bytes under
    ``evidence_root``, and written in one direct pass with ``\\n`` line endings and
    exactly one trailing newline, so two writes of the same run are byte-identical.

    Args:
        run_result: The completed run. Its ``results`` are already ordered by
            ``(check_id, case_id)``, and that stored order is the order the table and the
            detail blocks use, so the report and ``summary.json`` agree by construction.
        path: Destination file, normally ``<report_dir>/index.html``. Accepts a ``str``
            as well as a ``Path`` so a caller holding a configured path need not convert.
        evidence_root: The directory ``CaseResult.evidence_dir`` is relative to - the
            run's report directory, whose ``evidence/`` subtree the layout fixes. Each
            case directory is resolved as ``evidence_root / case.evidence_dir``; the path
            is used as given and never made absolute.

    Returns:
        ``path`` as a :class:`~pathlib.Path`, so a caller can record what it wrote.

    Raises:
        TypeError: a ``request``, ``decoded`` or ``expected`` mapping holds a value that
            is not JSON-serialisable, which is a caller defect rather than something to
            hide behind a plausible-looking report.
        ValueError: a ``request``, ``decoded`` or ``expected`` mapping holds a non-finite
            float - ``nan``, ``inf`` or ``-inf`` - which JSON does not define. It is
            refused rather than rendered as a token no JSON parser accepts, and it is
            raised while the document is still in memory, before the file is opened.
        OSError: the destination directory cannot be created or the file cannot be
            written. It propagates untouched: ``qrcheck.cli.run`` is the single place that
            turns an unwritable report directory into exit code 3, and no retry path
            exists anywhere in the product (AAP §0.3.4, §0.5.6).
    """
    target = Path(path)
    document = _render_document(run_result, Path(evidence_root))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    return target


def _render_document(run_result: RunResult, evidence_root: Path) -> str:
    """Assemble the whole document as one string, joined once.

    The order is the one AAP §0.8.4 fixes: the verdict and its tallies first, so a
    reviewer knows the outcome before scrolling; then any degradation; then every result;
    then the failures in full; then the skips; then how to reproduce a case.
    """
    anchors = _detail_anchors(run_result.results)
    fragments: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>qrcheck report — {_esc(run_result.adapter)}</title>",
        "<style>",
        _STYLE.rstrip("\n"),
        "</style>",
        "</head>",
        "<body>",
        '<a class="skip-link" href="#results">Skip to results</a>',
    ]
    fragments += _render_header(run_result)
    fragments += _render_banners(run_result)
    fragments += _render_table(run_result, anchors)
    fragments += _render_details(run_result, evidence_root, anchors)
    fragments += _render_skipped(run_result)
    fragments += _render_footer(run_result)
    fragments += ["</body>", "</html>"]
    return "\n".join(fragments) + "\n"


def _render_header(run_result: RunResult) -> list[str]:
    """Render the verdict line, the run's identity and the four separate tallies.

    The verdict is derived from the tallies and nothing else, and a run with skips is
    never called a plain pass: ``run passed (skips present)`` keeps a skipped check from
    reading as a pass (AAP §0.3.4, §0.4.3). The four tallies are rendered as four
    figures rather than one total, for the same reason. The duration comes from the
    model, so this writer reads no clock of its own.
    """
    verdict, modifier = _verdict(run_result)
    counts = run_result.counts
    fragments: list[str] = [
        "<header>",
        "<h1>qrcheck report</h1>",
        f'<p class="verdict {modifier}">{_esc(verdict)}</p>',
        '<dl class="facts">',
    ]
    facts: tuple[tuple[str, object], ...] = (
        ("adapter", run_result.adapter),
        ("ec_level", run_result.ec_level),
        ("seed", run_result.seed),
        ("loc_count", run_result.loc_count),
        ("duration", f"{run_result.duration_s:.3f} s"),
        ("run_id", run_result.run_id),
    )
    for label, value in facts:
        fragments.append(f"<div><dt>{_esc(label)}</dt><dd>{_esc(value)}</dd></div>")
    fragments.append("</dl>")
    fragments.append('<ul class="counts">')
    tallies: tuple[tuple[str, int], ...] = (
        ("passed", counts.passed),
        ("failed", counts.failed),
        ("skipped", counts.skipped),
        ("error", counts.error),
    )
    for label, value in tallies:
        fragments.append(
            f'<li class="count count-{label}">'
            f'<span class="count-value">{_esc(value)}</span>'
            f'<span class="count-label">{_esc(label)}</span>'
            "</li>"
        )
    fragments.append("</ul>")
    fragments.append("</header>")
    return fragments


def _verdict(run_result: RunResult) -> tuple[str, str]:
    """Return the verdict text and its CSS modifier from the four tallies.

    A run in which no case executed is never called a pass. It reports ``no cases ran``
    in a neutral colour, because a reader must not be able to conclude that the suite
    passed when nothing ran: the four tallies would otherwise read 0/0/0/0 under a green
    ``run passed``, and a table with no rows would say nothing that contradicts them.

    Precedence for every other run is the product's: an internal error outranks a
    failure, because a run that errored cannot be summarised as merely failing, and both
    outrank a skip. There is no verdict that a skip alone can turn into a plain pass.
    """
    counts = run_result.counts
    if counts.total == 0:
        return "no cases ran", "verdict-empty"
    if counts.error:
        return "run errored", "verdict-errored"
    if counts.failed:
        return "run failed", "verdict-failed"
    if counts.skipped:
        return "run passed (skips present)", "verdict-skipped"
    return "run passed", "verdict-passed"


def _render_banners(run_result: RunResult) -> list[str]:
    """Render the degradation banners, derived from the results alone.

    No probe, no configuration read and no decoder import: a banner states what the
    results themselves prove. Single-decoder mode is visible from the recorded
    ``per_decoder`` map of a result (AAP §0.6.6 row 4), and the two dependency skips are
    visible from the reason of a skipped result. Each banner is one sentence naming the
    consequence, so a reader is never left to infer a degradation from an empty column.
    """
    banners: list[str] = []
    if _has_single_decoder(run_result.results):
        banners.append(
            '<p class="banner banner-single-decoder"><strong>Single-decoder mode.</strong> '
            "Only one decoder was available for this run, so each payload was corroborated "
            "by the primary decoder alone and no check was marked passed on the strength of "
            "a decoder that was not installed.</p>"
        )
    skipped_reasons = {
        _reason_text(case.reason)
        for case in run_result.results
        if case.status is Status.SKIPPED
    }
    if Reason.MISSING_DEPENDENCY.value in skipped_reasons:
        banners.append(
            '<p class="banner banner-missing-dependency"><strong>SVG rasterizer '
            "missing.</strong> No SVG rasterizer was discovered, so RN-SVG could not "
            "rasterize its symbols and reported skipped: missing_dependency.</p>"
        )
    if Reason.UNSUPPORTED_PLATFORM.value in skipped_reasons:
        banners.append(
            '<p class="banner banner-unsupported-platform"><strong>Unsupported '
            "platform.</strong> At least one check cannot make its observation on this "
            "platform, so it reported skipped: unsupported_platform and is counted "
            "separately.</p>"
        )
    if not banners:
        return []
    header = '<section class="banners" aria-label="Run degradations">'
    return [header, *banners, "</section>"]


def _has_single_decoder(results: Sequence[CaseResult]) -> bool:
    """Return whether any result recorded a decode with no secondary decoder.

    The test is the model's own evidence: a non-empty ``per_decoder`` map that omits
    ``opencv`` means the run happened in single-decoder mode. Every access is a mapping
    lookup that cannot raise on an absent key, and a result whose ``decoded`` value is not
    a mapping at all is simply skipped, because a report must not fail over evidence it
    knows how to do without.
    """
    for case in results:
        decoded = case.decoded
        if not isinstance(decoded, Mapping):
            continue
        per_decoder = decoded.get("per_decoder")
        if isinstance(per_decoder, Mapping) and per_decoder and "opencv" not in per_decoder:
            return True
    return False


def _render_table(run_result: RunResult, anchors: Mapping[int, str]) -> list[str]:
    """Render one row per result, in the model's stored order.

    The status class separates the four outcomes by colour and the status word spells the
    outcome out, so a skipped row is visibly distinct from a passed one without colour
    being the only signal. The reason column carries the machine-readable value verbatim,
    because that is the vocabulary a reader greps for. A row links to its detail block by
    an in-page anchor, and the anchor is generated here rather than derived from evidence.
    The section carries ``tabindex="-1"`` so the jump from the skip link takes the keyboard
    focus with it instead of leaving the reader's focus on the link.

    A run that recorded no result gets the empty state in place of the table rather than a
    table with only a header: there is no row to read, and a header-only table under four
    zero tallies would leave a reader nothing to contradict a conclusion that the suite
    passed. The paragraph says so in words, so the state does not rest on colour alone.
    """
    if not run_result.results:
        return [
            '<section id="results" tabindex="-1">',
            "<h2>Results</h2>",
            '<p class="empty-state">No cases ran: this run recorded zero results, so no '
            "check was exercised and nothing here can be read as a pass. Select at least "
            "one check and run again.</p>",
            "</section>",
        ]
    fragments: list[str] = [
        '<section id="results" tabindex="-1">',
        "<h2>Results</h2>",
        "<table>",
        "<caption>One row per case, ordered by check ID and then by case ID.</caption>",
        "<thead>",
        '<tr><th scope="col">Check</th><th scope="col">Case</th>'
        '<th scope="col">Status</th><th scope="col">Reason</th>'
        '<th scope="col">Duration</th><th scope="col">Evidence</th></tr>',
        "</thead>",
        "<tbody>",
    ]
    for index, case in enumerate(run_result.results):
        anchor = anchors.get(index)
        if anchor is None:
            evidence_cell = "—"
        else:
            evidence_cell = f'<a href="#{_esc(anchor)}">evidence</a>'
        fragments.append(
            f'<tr class="{_status_class(case.status)}">'
            f'<td data-label="Check"><code>{_esc(case.check_id)}</code></td>'
            f'<td data-label="Case"><code>{_esc(case.case_id)}</code></td>'
            f'<td data-label="Status">{_esc(_status_text(case.status))}</td>'
            f'<td data-label="Reason">{_esc(_reason_text(case.reason))}</td>'
            f'<td data-label="Duration">{_esc(f"{case.duration_ms:.3f} ms")}</td>'
            f'<td data-label="Evidence">{evidence_cell}</td>'
            "</tr>"
        )
    fragments += ["</tbody>", "</table>", "</section>"]
    return fragments


def _render_details(
    run_result: RunResult, evidence_root: Path, anchors: Mapping[int, str]
) -> list[str]:
    """Render one block per failed or errored case that recorded evidence.

    The block answers "can I see it": the recorded images inline as ``data:`` URIs, the
    request that produced them, the decoded-or-error outcome and the seed, plus the one
    command that re-runs the case. The ``request.json`` panel renders the body
    :func:`_request_payload` composes - the artefact's own content, its identity envelope
    merged over the request projection - rather than the request alone, and both JSON
    panels are built from the in-memory model through the canonical JSON formatter instead
    of read back from the files on disk, which is what keeps this writer a pure function of
    the result it is handed. Each block carries ``tabindex="-1"`` so a jump to it moves the
    keyboard focus as well as the viewport.
    """
    blocks: list[str] = []
    for index, case in enumerate(run_result.results):
        anchor = anchors.get(index)
        if anchor is None:
            continue
        blocks.append(
            f'<section id="{_esc(anchor)}" class="evidence {_status_class(case.status)}" '
            'tabindex="-1">'
        )
        blocks.append(f"<h3>{_esc(case.check_id)} — {_esc(case.case_id)}</h3>")
        blocks.append('<dl class="facts">')
        facts: tuple[tuple[str, object], ...] = (
            ("check_id", case.check_id),
            ("case_id", case.case_id),
            ("status", _status_text(case.status)),
            ("reason", _reason_text(case.reason)),
            ("seed", case.seed),
            ("duration", f"{case.duration_ms:.3f} ms"),
            ("evidence_dir", case.evidence_dir or "—"),
        )
        for label, value in facts:
            blocks.append(f"<div><dt>{_esc(label)}</dt><dd>{_esc(value)}</dd></div>")
        blocks.append("</dl>")
        blocks.append(
            f'<p class="command"><code>{_esc(_reproduce_command(case))}</code></p>'
        )
        blocks += _render_images(evidence_root, case)
        blocks.append("<h4>request.json</h4>")
        blocks.append(f"<pre>{_esc(dumps_canonical(_request_payload(case)))}</pre>")
        blocks.append("<h4>decoded.json</h4>")
        blocks.append(f"<pre>{_esc(dumps_canonical(case.decoded))}</pre>")
        if case.expected is not None:
            blocks.append("<h4>expected</h4>")
            blocks.append(f"<pre>{_esc(dumps_canonical(case.expected))}</pre>")
        blocks.append("</section>")
    if not blocks:
        return []
    header = ['<section id="failures">', "<h2>Failures and errors</h2>"]
    return [*header, *blocks, "</section>"]


def _request_payload(case: CaseResult) -> dict[str, object]:
    """Return the request body ``request.json`` carries for ``case``.

    ``request.json`` is not the request projection on its own: the artefact is the case's
    identity envelope - ``check_id``, ``case_id`` and ``seed`` - merged over
    :attr:`~qrcheck.report.model.CaseResult.request`, with the envelope spread last so
    that it wins a key collision, because the envelope identifies the case and a check
    must never be able to override it.
    :meth:`qrcheck.report.evidence.EvidenceStore.record` is the single writer of that
    artefact and composes exactly this mapping, so the panel renders the file's content
    rather than a request without the case's identity. The body is composed from the
    model alone, never re-read from disk, so the block stays a pure function of the result
    it is handed (AAP §0.5.4).
    """
    return {
        **dict(case.request),
        "check_id": case.check_id,
        "case_id": case.case_id,
        "seed": case.seed,
    }


def _render_images(evidence_root: Path, case: CaseResult) -> list[str]:
    """Render the case's recorded images as inline figures, sorted by file name.

    Only the closed evidence set is rendered: a name in the fixed artefact list or a
    ``damage-<kind>-<seed>.png`` carrying a declared kind, which
    :func:`qrcheck.evidence.naming.is_evidence_artifact` decides once for the whole
    product. An unknown file in the directory is ignored rather than published, because
    this writer must not surface something the product never promised to write. An SVG
    goes through a ``data:`` URI inside an image element too: that context is what makes a
    target-supplied vector inert.

    The image area degrades rather than raising, but it says which of three facts it is
    reporting, because a reader triaging a failure reads this block to learn what the run
    actually recorded:

    * nothing - ``case.evidence_dir is None``, or a listing that holds no image artefact -
      renders :data:`_NO_IMAGE`;
    * a listed artefact whose bytes cannot be read renders one
      :func:`_unreadable_paragraph` naming that artefact and the exception class, and every
      artefact that did read is still rendered;
    * a directory that cannot be listed renders one :func:`_unreadable_paragraph` naming the
      exception class.

    Only the exception's class name reaches the document - never its message, its file name
    or a path - so the text stays locale-independent and carries nothing absolute
    (AAP §0.5.5), and the block remains a deterministic function of the evidence tree
    (AAP §0.5.4).
    """
    if case.evidence_dir is None:
        return [_NO_IMAGE]
    directory = evidence_root / case.evidence_dir
    try:
        names = _image_artifact_names(directory)
    except OSError as error:
        return [
            _unreadable_paragraph(
                f"evidence directory could not be listed ({type(error).__name__})"
            )
        ]
    figures: list[str] = []
    unreadable: list[str] = []
    for name in names:
        try:
            data = _read_artifact(directory / name)
        except OSError as error:
            unreadable.append(
                _unreadable_paragraph(
                    f"image present but could not be read ({name}: "
                    f"{type(error).__name__})"
                )
            )
            continue
        figures += [
            '<figure class="evidence-image">',
            f'<img src="{_esc(to_data_uri(data))}" alt="{_esc(name)}">',
            f"<figcaption>{_esc(name)}</figcaption>",
            "</figure>",
        ]
    if not figures and not unreadable:
        return [_NO_IMAGE]
    return figures + unreadable


def _image_artifact_names(directory: Path) -> list[str]:
    """Return the image artefact names in ``directory``, sorted.

    The listing is sorted so the document is identical on every run and on every
    filesystem (AAP §0.5.3). A directory that cannot be listed lets its ``OSError`` reach
    :func:`_render_images`, which states the failure in the block it renders: a directory
    that is absent or unreadable and a directory that holds no image are different facts
    about a run, and only the caller holds the block a reader has to be told in.
    """
    entries = sorted(entry.name for entry in directory.iterdir() if entry.is_file())
    return [name for name in entries if _is_image_artifact(name)]


def _is_image_artifact(name: str) -> bool:
    """Return whether ``name`` is an image member of the closed evidence set."""
    if not name.lower().endswith(_IMAGE_SUFFIXES):
        return False
    return is_evidence_artifact(name)


def _read_artifact(path: Path) -> bytes:
    """Return the bytes of the artefact at ``path``.

    An artefact that cannot be read - permission denied, gone between the listing and this
    read, or an I/O error - lets its ``OSError`` reach :func:`_render_images`, which names
    the artefact and the exception class in the block rather than reporting the run as
    having recorded no image. Nothing else is caught, so a defect of the writer's own still
    propagates instead of becoming report content.
    """
    return path.read_bytes()


def _unreadable_paragraph(text: str) -> str:
    """Return the one-line paragraph that states a degraded image area.

    The caller builds ``text`` from an artefact's file name and an exception class only, and
    it reaches the document through :func:`_esc` like every other dynamic fragment.
    """
    return f'<p class="evidence-unreadable">{_esc(text)}</p>'


def _render_skipped(run_result: RunResult) -> list[str]:
    """Render the skipped cases in their own section, with the reason they carry.

    A skip is never folded into the failures or into the passes: it is listed here with
    its machine-readable reason, and each reason that occurs is explained once in plain
    words so a reader can act on it without consulting the documentation.
    """
    skipped = [case for case in run_result.results if case.status is Status.SKIPPED]
    if not skipped:
        return []
    fragments: list[str] = [
        '<section id="skipped">',
        "<h2>Skipped cases</h2>",
        "<p>A skipped case is a real outcome, never a pass: it is tallied separately and "
        "does not satisfy the check it names.</p>",
        '<ul class="skips">',
    ]
    for case in skipped:
        fragments.append(
            f'<li><code>{_esc(case.check_id)}</code> <code>{_esc(case.case_id)}</code> — '
            f"<code>{_esc(_reason_text(case.reason))}</code></li>"
        )
    fragments.append("</ul>")
    occurring = [
        (value, explanation)
        for value, explanation in _REASON_EXPLANATIONS
        if any(_reason_text(case.reason) == value for case in skipped)
    ]
    if occurring:
        fragments.append('<dl class="reasons">')
        for value, explanation in occurring:
            fragments.append(
                f"<div><dt><code>{_esc(value)}</code></dt>"
                f"<dd>{_esc(explanation)}</dd></div>"
            )
        fragments.append("</dl>")
    fragments.append("</section>")
    return fragments


def _render_footer(run_result: RunResult) -> list[str]:
    """Render the closing note that says how to reproduce a single case.

    The guiding sentence follows the results this run recorded. A run with at least one
    result points the reader at the case's detail block and the rows of the table; a run
    that recorded none says there is no row to re-run, so the page never sends a reader
    looking for a row that does not exist. The command, the seed contract and the note
    about the two machine-readable files are the same on both, because they describe how
    the product works rather than what this run happened to do.
    """
    command = "qrcheck run --checks <CHECK-ID> --seed <seed>"
    if run_result.results:
        guidance = (
            "<p>Every row of the results table can be re-run on its own at the same seed: "
            "copy the command from the case's detail block, or use</p>"
        )
    else:
        guidance = (
            "<p>No case ran, so there is no row to re-run at a seed: a single case is run "
            "with</p>"
        )
    return [
        "<footer>",
        "<h2>Reproducing a case</h2>",
        guidance,
        f'<p class="command"><code>{_esc(command)}</code></p>',
        "<p>The recorded seed regenerates the same payloads and the same damaged images. "
        "The <code>summary.json</code> and <code>junit.xml</code> written beside this file "
        "carry the same results in the forms a machine reads.</p>",
        "</footer>",
    ]


def _detail_anchors(results: Sequence[CaseResult]) -> dict[int, str]:
    """Return the in-page anchor of each result that gets a detail block.

    Keyed by the result's position in the stored order, so the table and the blocks cannot
    disagree about which case an anchor names. Only a failed or errored case with a
    recorded evidence directory has a block; a skip is listed in its own section, and a
    passing case writes no evidence at all.
    """
    anchors: dict[int, str] = {}
    for index, case in enumerate(results):
        if case.evidence_dir is None:
            continue
        if case.status not in _DETAIL_STATUSES:
            continue
        anchors[index] = f"ev-{len(anchors) + 1}"
    return anchors


def _reproduce_command(case: CaseResult) -> str:
    """Return the single command that re-runs ``case`` at its recorded seed."""
    return f"qrcheck run --checks {case.check_id} --seed {case.seed}"


def _status_class(status: Status | str) -> str:
    """Return the CSS class that colour-codes ``status``."""
    return _STATUS_CLASSES.get(str(getattr(status, "value", status)), "status-error")


def _status_text(status: Status | str) -> str:
    """Return the status word a reader sees, which is the model's own vocabulary."""
    return str(getattr(status, "value", status))


def _reason_text(reason: Reason | str | None) -> str:
    """Return the machine-readable reason, or an em dash for a passing case."""
    if reason is None:
        return "—"
    return str(getattr(reason, "value", reason))


def _esc(value: object) -> str:
    """Return ``value`` as text that is safe to interpolate into the document.

    Every dynamic value in this module passes through here, a case identifier a target
    could have influenced included: the escape is the single textual boundary between
    untrusted evidence and the report (AAP §0.3.2), so it is written once and reused
    rather than re-implemented per fragment.
    """
    return html_escape(str(value), quote=True)
