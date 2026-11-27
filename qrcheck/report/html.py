"""The product's only human interface: one self-contained ``index.html`` per run.

A person who opens this file has three questions - did the run pass, what failed, and
can I see it - and the document answers them before they scroll, in the
order: verdict and tallies, then degradation banners, then every case in one table - or
the empty state when the run recorded no case at all - then a detail block per failure,
then the skips, then how to reproduce a case. The three case sections are unconditional:
the failures and the skipped section are always present, and a section whose set is empty
says so in one fixed sentence rather than disappearing, so no reader has to infer an empty
set from a missing heading. A failing or errored case gets its block even when the run could
not write its evidence, because the request, the decoded-or-error outcome, the expected value
and the seed all come from the result model - only the image area is then missing.

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
  payload, a decoder message - is untrusted and reaches the document only
  through :data:`html_escape` with ``quote=True``. JSON is rendered as escaped text
  inside ``<pre>``, never interpolated raw. Markup escaping is not the whole promise,
  though: a code point UTF-8 cannot encode - the lone surrogate a target's JSON can
  carry - is replaced with U+FFFD once the document is complete and before the file is
  opened (:func:`qrcheck.report.json_report.make_utf8_encodable`), so the target's text
  can neither break the page's markup nor cost the run its report.
* **No design system.** There is no component library, no design-token file and no
  external font to conform to, so none is introduced: one inline stylesheet, system font
  stacks only, and a single ``@media (max-width: 720px)`` rule that collapses the layout
  to one column. Every box that can hold an unbreakable identifier - a case id, an
  evidence path, the run id - wraps inside itself at every width, so a long identifier
  never widens the page, and the media query carries only the collapse. A later change
  that introduces a framework would break the offline promise this module exists to keep.

The document is a pure function of the :class:`~qrcheck.report.model.RunResult` and the
evidence bytes already on disk, so it is exactly as stable as the result model and no
more. Nothing here reads a clock - the run's duration comes from
:attr:`~qrcheck.report.model.RunResult.duration_s` - nothing formats through the locale,
nothing prints or logs, every directory listing is sorted, and no timestamp, hostname,
user name or absolute path is emitted. Two writes of the same run therefore produce
byte-identical files.

The only volatile content it carries is the duration the model already holds, which is
why the report is compared semantically rather than byte for byte across runs. Writing
is direct - no temporary file - because a stray file inside the report directory would be
an artefact outside the closed evidence set that ``CL-ARTIFACTS`` inspects.
"""

from collections.abc import Mapping, Sequence
from html import escape as html_escape
from pathlib import Path

from qrcheck.evidence.mux import to_data_uri
from qrcheck.evidence.naming import (
    LOGO_PNG,
    SYMBOL_PNG,
    SYMBOL_REPEAT_PNG,
    SYMBOL_SVG,
    is_evidence_artifact,
)
from qrcheck.report.json_report import dumps_canonical, make_utf8_encodable
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

# The statuses that get a detail block, evidence directory or not: the block's request,
# outcome, expected and seed come from the model, so a case whose evidence could not be
# written loses only its image. A skip is reported in its own section instead: it is a real
# outcome that is never a pass, so it must not be filed with the failures a
# reader triages.
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
        "A prerequisite the check needs was not available in this run, so the check could "
        "not make its observation and is not counted as a pass.",
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

# What each of the four fixed image artefacts shows, in the words a reader who cannot see the
# image needs: a figure's alternative text is built from this and the case's identity, while
# the visible caption stays the artefact's name, so an informative image is described rather
# than announced twice under the same string. A damaged variant is described from the two
# facts its own name carries, by :func:`_image_description`.
_IMAGE_DESCRIPTIONS: dict[str, str] = {
    SYMBOL_PNG: "the symbol the target returned",
    SYMBOL_SVG: "the symbol the target returned",
    SYMBOL_REPEAT_PNG: "the symbol the target returned on the second call of the same request",
    LOGO_PNG: "the overlay bitmap the check composited onto the symbol",
}

# The decoder names a result's ``per_decoder`` map is keyed by. They are spelled here rather
# than imported from :mod:`qrcheck.decode`, because this writer is a pure function of the
# result model: it names the vocabulary it reads instead of depending on the layer that
# produced it.
_PRIMARY_DECODER: str = "zxing"
_SECONDARY_DECODER: str = "opencv"

# The status a decoder records when it returned a payload. Every other shape the map can hold -
# no entry for the decoder at all, ``error`` or ``undetected`` - means the decoder answered
# nothing about the artefact, which is the only distinction the degradation banner draws.
_DECODED_STATUS: str = "decoded"

# The single stylesheet of the document. Logical properties throughout, system font
# stacks only, colour never the sole carrier of meaning - every status is also spelled out
# in its row - and not one external resource reference, so the file needs no network. Every
# box that can hold an unbreakable identifier - the 64-character run_id, an evidence
# directory, a case id - wraps inside itself at every width rather than widening the
# document past the viewport, so its one media query collapses the layout to a single
# column whose track may shrink to nothing and carries no wrap rule of its own.
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

/* A link a reader clicks meets the 24x24 CSS px target size WCAG 2.2 AA asks for. */
.evidence-link {
  display: inline-block;
  min-block-size: 1.5rem;
  padding-block: 0.125rem;
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
  overflow-wrap: anywhere;
}

/* A fact whose value is a case id or an evidence path takes the card's full width, so it
   wraps at the card's measure instead of breaking mid-token over five or six 11 rem lines.
   The block emits those two facts last, so the facts that fit a track share one row rather
   than being split by a full-width one between them. */
.facts .fact-wide { grid-column: 1 / -1; }

.reasons {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.5rem 1rem;
  margin: 0.5rem 0 0;
}

.reasons div { margin: 0; }

.reasons dt {
  color: #55555f;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.reasons dd {
  margin: 0;
  overflow-wrap: anywhere;
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

/* Containment is a property of every width, not of the narrow-viewport collapse: whatever
   a box holds - a case id, an evidence path, a check id in a skip bullet - wraps inside
   its own box, because a page that scrolls sideways hides the Status, Reason, Duration and
   Evidence columns every result needs. `anywhere` rather than `break-word`,
   which leaves a table cell's min-content width intact and so would still widen the table.
   It is declared on the cell that carries a case id and on nothing else, because `anywhere`
   also lowers that cell's minimum width: spread over every cell it lets the fixed-vocabulary
   columns shrink below the words they hold, and `skipped` or `value_mismatch` would then break
   mid-word on a viewport narrow enough that the identifier column has to give up room. */
tbody td[data-label="Case"] { overflow-wrap: anywhere; }

.evidence h3, .skips li { overflow-wrap: anywhere; }

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

.evidence-recorded {
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
  .facts, .counts, .reasons { grid-template-columns: minmax(0, 1fr); }
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
# declared rejection has no symbol, a listed directory may legitimately hold no image
# artefact, and a case whose evidence could not be written has no directory to list. An
# artefact that exists but cannot be read is never reported as this - that is a read
# failure, and the block states it as one (see _render_images), as it does for a directory
# that cannot be listed, which has no listing to be absent from.
_NO_IMAGE: str = '<p class="no-image">no image recorded for this case</p>'

# The paragraph a case section carries instead of its list when its set is empty. Both
# sentences are fixed text - no count, no path and nothing this run measured - because the
# page is a pure function of the results and the tallies and the results table are what
# carry the outcome, so a section that restated a number would be a second source of truth
# for it. A run with no failure and no skip
# therefore says so in words rather than leaving a reader to infer it from two absent
# sections, and neither sentence can be read as a pass.
_NO_FAILURES: str = (
    '<p class="empty-state">No failures or errors in this run: no case recorded a failed or '
    "errored outcome, so there is no failing case to inspect.</p>"
)
_NO_SKIPS: str = (
    '<p class="empty-state">No skipped cases in this run: no case recorded a skipped outcome, '
    "so nothing was left unexercised for a missing capability, dependency or platform.</p>"
)


def write_html(
    run_result: RunResult, path: str | Path, evidence_root: str | Path
) -> Path:
    """Write ``run_result`` as a self-contained ``index.html`` and return the path.

    The document is built in memory from the result model and the evidence bytes under
    ``evidence_root``, and written in one direct pass with ``\\n`` line endings and
    exactly one trailing newline, so two writes of the same run are byte-identical.

    The finished document passes through
    :func:`~qrcheck.report.json_report.make_utf8_encodable` before the destination is created, so a
    code point UTF-8 cannot encode - the lone surrogate a target's JSON can carry, which reaches an
    identifier, a payload or a decoder message as itself - becomes U+FFFD instead of aborting the
    write. Text a target supplied can then make the page say something was unreachable, but never
    cost the run its report or leave a truncated ``index.html`` behind; every character UTF-8 can
    encode is written exactly as escaped, so the page keeps its bytes.

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
            turns an unwritable report directory into exit code 3, and no retry path exists
            anywhere in the product.
    """
    target = Path(path)
    # Made encodable while it is still only a string: the write below cannot then fail on a code
    # point UTF-8 has no bytes for, and the destination is not even created.
    document = make_utf8_encodable(_render_document(run_result, Path(evidence_root)))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    return target


def _render_document(run_result: RunResult, evidence_root: Path) -> str:
    """Assemble the whole document as one string, joined once.

    The order is the reader's: the verdict and its tallies first, so a reviewer knows the
    outcome before scrolling; then any degradation; then every result; then the failures in
    full; then the skips; then how to reproduce a case.

    The header and the footer stay outside the document's single ``main`` region, which holds
    the banners and the three case sections and is exactly what the skip link targets: a jump
    from the top of the page lands on the content rather than on the run's identity, and the
    region a screen reader offers as "main" is the same one.
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
        '<a class="skip-link" href="#main">Skip to main content</a>',
    ]
    fragments += _render_header(run_result)
    fragments += ['<main id="main" tabindex="-1">']
    fragments += _render_banners(run_result)
    fragments += _render_table(run_result, anchors)
    fragments += _render_details(run_result, evidence_root, anchors)
    fragments += _render_skipped(run_result)
    fragments += ["</main>"]
    fragments += _render_footer(run_result)
    fragments += ["</body>", "</html>"]
    return "\n".join(fragments) + "\n"


def _render_header(run_result: RunResult) -> list[str]:
    """Render the verdict line, the run's identity and the four separate tallies.

    The verdict is derived from the tallies and nothing else, and a run with skips is
    never called a plain pass: ``run passed (skips present)`` keeps a skipped check from
    reading as a pass. The four tallies are rendered as four figures rather than one total,
    for the same reason. The duration comes from the model, so this writer reads no clock
    of its own.
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
    results themselves prove. The decoder banner is derived from the ``per_decoder`` map of
    each decode-bearing result and states how many of those results the corroborating
    decoder did not answer for; the run is called single-decoder mode only when that is
    every one of them, and incomplete corroboration otherwise. The two dependency skips are
    visible from the reason of a skipped result. Each banner states the consequence in
    words, so a reader is never left to infer a degradation from an empty column.

    The missing-dependency banner names the skipped check IDs and no cause, because one
    reason token that several different producers record cannot establish which prerequisite
    was absent: a skip of that kind states which check it denied and leaves the missing
    prerequisite to the evidence the skip recorded, which is the only place it is written
    down.
    """
    banners: list[str] = []
    bearing, unanswered, primary_alone = _decoder_counts(run_result.results)
    if primary_alone:
        heading = (
            "Single-decoder mode."
            if unanswered == bearing
            else "Decoder corroboration incomplete."
        )
        banners.append(
            f'<p class="banner banner-single-decoder"><strong>{heading}</strong> The '
            f"corroborating decoder (opencv-python-headless) did not answer for {unanswered} "
            f"of {bearing} decode-bearing cases in this run: it was not installed, or it was "
            "asked and could not read that artefact. Where it did not answer, the payload was "
            "corroborated by the primary decoder alone, and no check was marked passed on the "
            "strength of a decoder that did not answer.</p>"
        )
    skipped_reasons = {
        _reason_text(case.reason)
        for case in run_result.results
        if case.status is Status.SKIPPED
    }
    if Reason.MISSING_DEPENDENCY.value in skipped_reasons:
        missing_dependency = _skipped_check_ids(run_result.results, Reason.MISSING_DEPENDENCY)
        named = ", ".join(f"<code>{_esc(name)}</code>" for name in missing_dependency)
        banners.append(
            '<p class="banner banner-missing-dependency"><strong>Missing dependency.</strong> '
            "Reported skipped: missing_dependency, because a prerequisite the check needs was "
            f"not available in this run: {named}. A check skipped this way is counted "
            "separately and is never counted as a pass; its recorded evidence states which "
            "prerequisite was absent.</p>"
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


def _skipped_check_ids(results: Sequence[CaseResult], reason: Reason) -> list[str]:
    """Return the sorted distinct check IDs of the results skipped for ``reason``.

    The banner names the checks a degradation denied rather than describing a cause, so this
    gathers the one fact the results carry: which check each skip belongs to. The IDs are
    de-duplicated and then sorted, because a whole check can skip once per case and a reader
    needs each ID once, and because no set iteration order may reach the document.
    """
    return sorted(
        {
            case.check_id
            for case in results
            if case.status is Status.SKIPPED and _reason_text(case.reason) == reason.value
        }
    )


def _decoder_counts(results: Sequence[CaseResult]) -> tuple[int, int, int]:
    """Count what the results prove about the corroborating decoder.

    Returns three counts over ``results``, in one pass:

    * how many results are decode-bearing at all - ``decoded`` is a mapping whose
      ``per_decoder`` member is a non-empty mapping, which is the only shape that records
      what a decoder did;
    * of those, how many the secondary decoder did not answer for - no ``opencv`` entry, or
      an entry whose status is not ``decoded``. Two states leave no entry to read: a decoder
      the run never consulted, absent or unimportable, whose reason the evidence records under
      ``unavailable_decoders``, and a decoder that ran and found nothing or raised, which its
      own status records. This count reads the results alone, so it states the one consequence
      both states share - the corroborating decoder did not answer for this artefact - rather
      than choosing between them;
    * of those, how many the primary decoder read alone - the secondary did not answer and
      ``zxing`` recorded ``decoded``. That is the one claim a banner may make, because it is
      the case in which a payload was corroborated by the primary decoder alone.

    The shape of an artefact no decoder could be shown is therefore counted as decode-bearing
    but not as primary-alone: every decoder at ``error``, or a primary that recorded
    ``undetected``, means no payload was read by anyone, so no check can have passed on the
    primary decoder's strength. Every access is a mapping lookup that cannot raise on an
    absent key, and a result whose ``decoded`` value is not a mapping at all is skipped,
    because a report must not fail over evidence it knows how to do without.
    """
    bearing = unanswered = primary_alone = 0
    for case in results:
        decoded = case.decoded
        if not isinstance(decoded, Mapping):
            continue
        per_decoder = decoded.get("per_decoder")
        if not isinstance(per_decoder, Mapping) or not per_decoder:
            continue
        bearing += 1
        if _decoder_status(per_decoder, _SECONDARY_DECODER) == _DECODED_STATUS:
            continue
        unanswered += 1
        if _decoder_status(per_decoder, _PRIMARY_DECODER) == _DECODED_STATUS:
            primary_alone += 1
    return bearing, unanswered, primary_alone


def _decoder_status(per_decoder: Mapping[object, object], decoder: str) -> str | None:
    """Return ``decoder``'s recorded status in ``per_decoder``, or ``None`` when unreadable.

    Each entry of the map is itself a mapping in a well-formed result, but a malformed entry
    must not raise: an absent entry, a non-mapping entry or a status that is not a string
    answers ``None``, which matches no status and therefore proves nothing.
    """
    outcome = per_decoder.get(decoder)
    if not isinstance(outcome, Mapping):
        return None
    status = outcome.get("status")
    return status if isinstance(status, str) else None


def _render_table(run_result: RunResult, anchors: Mapping[int, str]) -> list[str]:
    """Render one row per result, in the model's stored order.

    The status class separates the four outcomes by colour and the status word spells the
    outcome out, so a skipped row is visibly distinct from a passed one without colour
    being the only signal. The reason column carries the machine-readable value verbatim,
    because that is the vocabulary a reader greps for. The evidence column is derived from
    the model - a link to the case's block, an explicit statement that a directory was
    recorded for a case reported elsewhere, or the em dash that means nothing was recorded
    (:func:`_evidence_cell`) - so the page cannot contradict ``summary.json`` about a case
    that persisted evidence. The section keeps ``id="results"`` and ``tabindex="-1"`` so a
    link to the results takes the keyboard focus with it instead of leaving the reader's
    focus on the link.

    A run that recorded no result gets the empty state in place of the table rather than a
    table with only a header: there is no row to read, and a header-only table under four
    zero tallies would leave a reader nothing to contradict a conclusion that the suite
    passed. The paragraph says so in words, so the state does not rest on colour alone.

    The table's structural elements also carry explicit ARIA roles. They are not redundant
    with the native elements: the narrow-viewport rule sets ``display: block`` on the table
    parts, which strips the roles a browser derives from them, so the row and column
    association the caption, ``thead`` and ``th scope`` establish has to survive the collapse
    explicitly.
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
    # The roles below exist because the narrow-viewport rule in _STYLE overrides `display` on
    # the table parts, which strips the roles a browser derives from them; do not delete them
    # as redundant with the native elements.
    fragments: list[str] = [
        '<section id="results" tabindex="-1">',
        "<h2>Results</h2>",
        '<table role="table">',
        "<caption>One row per case, ordered by check ID and then by case ID. In the "
        "Evidence column, a link opens that case's block below, "
        '"recorded, not shown" marks evidence that was persisted for a case reported '
        "elsewhere on this page, and an em dash means nothing was recorded.</caption>",
        '<thead role="rowgroup">',
        '<tr role="row"><th scope="col" role="columnheader">Check</th>'
        '<th scope="col" role="columnheader">Case</th>'
        '<th scope="col" role="columnheader">Status</th>'
        '<th scope="col" role="columnheader">Reason</th>'
        '<th scope="col" role="columnheader">Duration</th>'
        '<th scope="col" role="columnheader">Evidence</th></tr>',
        "</thead>",
        '<tbody role="rowgroup">',
    ]
    for index, case in enumerate(run_result.results):
        fragments.append(
            f'<tr class="{_status_class(case.status)}" role="row">'
            f'<td data-label="Check" role="cell"><code>{_esc(case.check_id)}</code></td>'
            f'<td data-label="Case" role="cell"><code>{_esc(case.case_id)}</code></td>'
            f'<td data-label="Status" role="cell">{_esc(_status_text(case.status))}</td>'
            f'<td data-label="Reason" role="cell">{_esc(_reason_text(case.reason))}</td>'
            f'<td data-label="Duration" role="cell">{_esc(f"{case.duration_ms:.3f} ms")}</td>'
            f'<td data-label="Evidence" role="cell">'
            f"{_evidence_cell(case, anchors.get(index))}</td>"
            "</tr>"
        )
    fragments += ["</tbody>", "</table>", "</section>"]
    return fragments


def _evidence_cell(case: CaseResult, anchor: str | None) -> str:
    """Return the Evidence column of ``case``'s row, derived from the result model alone.

    Three states, and each one says what happened to the case's evidence:

    * ``anchor`` is set - every failed and errored case has a block below, and that block
      exists even where no evidence directory was recorded, so the cell is the link to it and
      the block itself states that no image was recorded;
    * no anchor but ``case.evidence_dir`` is set - a directory was recorded for a case with
      no block of its own, which is a skip that persisted one: the cell says so in words
      instead of showing the mark that means "nothing was recorded", because ``summary.json``
      and the filesystem both record that directory;
    * ``case.evidence_dir is None`` - the em dash, which therefore means only that nothing was
      recorded: a passed case, which writes no evidence at all, or a skipped one whose
      evidence the run did not persist.

    The second state is a fixed phrase, so no path and nothing absolute reaches the cell
, and all three are constants of the model rather than a reading of the
    evidence tree. The table's caption is written to match, so the em dash carries exactly
    this one meaning wherever a reader meets it.
    """
    if anchor is not None:
        return f'<a class="evidence-link" href="#{_esc(anchor)}">evidence</a>'
    if case.evidence_dir is not None:
        return '<span class="evidence-recorded">recorded, not shown</span>'
    return "—"


def _render_details(
    run_result: RunResult, evidence_root: Path, anchors: Mapping[int, str]
) -> list[str]:
    """Render one block per failed or errored case, in the section that holds them.

    The section is rendered whether or not it holds a block: a run with no failed and no
    errored case states that in one fixed sentence (:data:`_NO_FAILURES`) instead of
    omitting the heading, so an empty set is a statement a reader can read rather than an
    absence to infer from the table.

    The block answers "can I see it": the recorded images inline as ``data:`` URIs, the
    request that produced them, the decoded-or-error outcome and the seed, plus the one
    command that re-runs the case at this run's adapter, EC level and seed. The
    ``request.json`` panel renders the body :func:`_request_payload` composes - the case's
    request projection with its identity envelope spread over it, which the artefact the
    store writes carries too, though that artefact can hold more than the panel does (the
    parameters its check resolved) - rather than the request alone, and both JSON panels are
    built from the in-memory model through the canonical JSON formatter instead of read back
    from the files on disk, which is what keeps this writer a pure function of the result it
    is handed. Each block carries ``tabindex="-1"`` so a jump to it moves the keyboard focus
    as well as the viewport.

    A case whose evidence could not be written still gets its block: the facts, the command
    and the panels are derived from the model, so the only part that degrades is the image
    area, which states that no image was recorded (:func:`_render_images`) rather than the
    block being withheld along with everything it carries.

    Each JSON panel is marked ``dir="auto"``: the panels render payload text a target
    supplied, so a payload in a right-to-left script would otherwise reorder the JSON
    punctuation around it and could make the panel read as something other than what the
    evidence says.
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
        # The two facts whose value is a long identifier come last, each taking the grid's whole
        # width (``.facts .fact-wide`` in ``_STYLE``), so they wrap at the block's measure instead
        # of breaking mid-token in a single 11 rem track - and so the five short facts share one
        # row, which a full-width fact in their middle would otherwise split in two.
        facts: tuple[tuple[str, object], ...] = (
            ("check_id", case.check_id),
            ("status", _status_text(case.status)),
            ("reason", _reason_text(case.reason)),
            ("seed", case.seed),
            ("duration", f"{case.duration_ms:.3f} ms"),
            ("case_id", case.case_id),
            ("evidence_dir", case.evidence_dir or "—"),
        )
        wide_facts = ("case_id", "evidence_dir")
        for label, value in facts:
            attribute = ' class="fact-wide"' if label in wide_facts else ""
            blocks.append(
                f"<div{attribute}><dt>{_esc(label)}</dt>"
                f"<dd>{_esc(value)}</dd></div>"
            )
        blocks.append("</dl>")
        blocks.append(
            f'<p class="command"><code>{_esc(_reproduce_command(case, run_result))}</code></p>'
        )
        blocks += _render_images(evidence_root, case)
        blocks.append("<h4>request.json</h4>")
        blocks.append(f'<pre dir="auto">{_esc(dumps_canonical(_request_payload(case)))}</pre>')
        blocks.append("<h4>decoded.json</h4>")
        blocks.append(f'<pre dir="auto">{_esc(dumps_canonical(case.decoded))}</pre>')
        if case.expected is not None:
            blocks.append("<h4>expected</h4>")
            blocks.append(f'<pre dir="auto">{_esc(dumps_canonical(case.expected))}</pre>')
        blocks.append("</section>")
    if not blocks:
        return [
            '<section id="failures">',
            "<h2>Failures and errors</h2>",
            _NO_FAILURES,
            "</section>",
        ]
    header = ['<section id="failures">', "<h2>Failures and errors</h2>"]
    return [*header, *blocks, "</section>"]


def _request_payload(case: CaseResult) -> dict[str, object]:
    """Return the body the ``request.json`` panel renders for ``case``.

    The panel is the case's identity envelope - ``check_id``, ``case_id`` and ``seed`` - spread
    last over :attr:`~qrcheck.report.model.CaseResult.request`, so the envelope wins a key
    collision: it identifies the case, and a check must never be able to override it. That is
    the whole of what the panel shows, and it is composed from the model alone, never re-read
    from disk, so the block stays a pure function of the result it is handed.
    Nothing in the panel is invented: it is the request projection the case's ``summary.json``
    record carries, plus the three identity fields that record holds beside it.

    The ``request.json`` artefact the evidence store writes beside it can carry more than this
    panel. The check support layer's ``record_evidence`` stages the case's request plus the
    parameters the check resolved for it - the ``out_of_contract`` marker of a deliberate probe
    and the ``request_extra`` entries such as ``quiet_zone_measured``,
    ``module_px_measured``, ``capacity``/``requested_len``/``sizing_capacity`` and
    ``expected_module_count``/``oracle_matrix_sha256`` - before the store writes the artefact,
    because those are measurements of the case's artefact that belong with its evidence rather
    than in the result model. Nothing here reads them back: the same measurements reach this
    page inside the case's ``decoded.json`` and ``expected`` panels, and the panel stays the
    model's projection so that a page can never show a request the model does not record.
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

    Each figure's alternative text says what the image shows - the symbol the target returned,
    its second call of the same request, a damaged variant or the composited overlay - and
    which case it belongs to, while the caption underneath names the artefact; the two never
    carry the same string, so an informative image is described once instead of being
    announced twice.

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
    or a path - so the text stays locale-independent and carries nothing absolute, and the
    block remains a deterministic function of the evidence tree.
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
        alternative = f"{_image_description(name)}, recorded for {case.check_id} {case.case_id}"
        figures += [
            '<figure class="evidence-image">',
            f'<img src="{_esc(to_data_uri(data))}" alt="{_esc(alternative)}">',
            f"<figcaption>{_esc(name)}</figcaption>",
            "</figure>",
        ]
    if not figures and not unreadable:
        return [_NO_IMAGE]
    return figures + unreadable


def _image_description(name: str) -> str:
    """Return what the image artefact ``name`` shows, for a reader who cannot see it.

    The four fixed image members are described by :data:`_IMAGE_DESCRIPTIONS`. A
    ``damage-<kind>-<seed>.png`` member is described from the two facts its own name carries -
    the damage kind applied and the derived seed - because those are what make the damaged
    symbol re-derivable, and they are the only description a variant can honestly be given.
    The caller has already admitted ``name`` through
    :func:`qrcheck.evidence.naming.is_evidence_artifact`, so a name outside that closed set
    cannot arrive; the fallback gives a generic description rather than publishing an empty
    alternative, which would turn an informative image into a decorative one.
    """
    description = _IMAGE_DESCRIPTIONS.get(name)
    if description is not None:
        return description
    head, _, seed = name.rpartition(".png")[0].rpartition("-")
    kind = head.removeprefix("damage-")
    if kind != head and kind and seed.isdigit():
        return f"the symbol after {kind} damage with damage seed {seed}"
    return "a recorded evidence image"


def _image_artifact_names(directory: Path) -> list[str]:
    """Return the image artefact names in ``directory``, sorted.

    The listing is sorted so the document is identical on every run and on every
    filesystem. A directory that cannot be listed lets its ``OSError`` reach
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

    A run that skipped nothing gets the section too, with :data:`_NO_SKIPS` in place of the
    list and the reason glossary: the sentence says only that no case recorded a skipped
    outcome, so an empty set is stated rather than inferred, and it claims nothing about
    whether the run passed - the verdict line and the tallies carry that. Neither the
    "real outcome, never a pass" paragraph nor the glossary is rendered for it, because both
    explain skips this run does not have.
    """
    skipped = [case for case in run_result.results if case.status is Status.SKIPPED]
    fragments: list[str] = ['<section id="skipped">', "<h2>Skipped cases</h2>"]
    if not skipped:
        return [*fragments, _NO_SKIPS, "</section>"]
    fragments += [
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
    looking for a row that does not exist. The command names this run's adapter, EC level and
    run seed and leaves ``<CHECK-ID>`` as its only placeholder: an omitted ``--adapter`` or
    ``--ec`` falls back to the configured default target and to EC level ``M``, and
    ``--seed`` is the run seed, so a command naming none of them re-runs the configured
    default target at the configured default seed instead of the run the results came from.
    The command and the note about the two machine-readable files are the same on both
    branches - they state this run's own parameters and the forms a machine reads - while the
    guidance differs, because it describes what this run happened to record.
    """
    command = (
        f"qrcheck run --adapter {run_result.adapter} --ec {run_result.ec_level} "
        f"--checks <CHECK-ID> --seed {run_result.seed}"
    )
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
        "<p>The run seed shown in the header regenerates the same payloads and the same "
        "damaged images. The <code>summary.json</code> and <code>junit.xml</code> written "
        "beside this file carry the same results in the forms a machine reads.</p>",
        "</footer>",
    ]


def _detail_anchors(results: Sequence[CaseResult]) -> dict[int, str]:
    """Return the in-page anchor of each result that gets a detail block.

    Keyed by the result's position in the stored order, so the table and the blocks cannot
    disagree about which case an anchor names, and numbered ``ev-1`` to ``ev-N`` in that
    same order.

    Every failed or errored case gets a block, whether or not it recorded an evidence
    directory: the block's request, decoded-or-error outcome, expected and seed are all
    derived from the model, so a case whose evidence could not be written loses its image and
    nothing else. A skip is listed in its own section instead, however much
    evidence it persisted, and a passing case records no evidence at all - and those are the
    three states :func:`_evidence_cell` reads through this mapping.
    """
    anchors: dict[int, str] = {}
    for index, case in enumerate(results):
        if case.status not in _DETAIL_STATUSES:
            continue
        anchors[index] = f"ev-{len(anchors) + 1}"
    return anchors


def _reproduce_command(case: CaseResult, run_result: RunResult) -> str:
    """Return the single command that re-runs ``case`` as this run recorded it.

    The command carries the run's adapter, EC level and run seed beside the case's check ID,
    because an omitted ``--adapter`` or ``--ec`` falls back to the configured default target
    and to EC level ``M``, and ``--seed`` is the run seed, not the case's recorded one: a case
    seed is a per-case derivation of the run seed, so passing it back re-derives every draw
    from the wrong root and re-runs a different case than the page shows
.
    """
    return (
        f"qrcheck run --adapter {run_result.adapter} --ec {run_result.ec_level} "
        f"--checks {case.check_id} --seed {run_result.seed}"
    )


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
    untrusted evidence and the report, so it is written once and reused rather than
    re-implemented per fragment.
    """
    return html_escape(str(value), quote=True)
