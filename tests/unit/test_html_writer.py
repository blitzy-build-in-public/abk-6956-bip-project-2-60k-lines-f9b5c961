"""The HTML page's substance: its panels, its verdicts, its images, its escapes.

This module is the place the writer's contract is held, and the page's promises to its reader are
fixed here: a failing case's block shows the request that produced it --
the model's projection with the case's identity envelope over it, which the evidence store's
``request.json`` may enrich with the parameters its check resolved -- and no page state may let a
reader read a run that did not pass as a pass.

Pinned: a run that recorded no result states that no case ran, keeps its four zero tallies and
replaces the results table with the empty-state paragraph; the failures and skipped sections are
rendered on every page, and each states in one fixed sentence when its own set is empty rather
than disappearing; the ``request.json`` panel parses to the recorded artefact with its identity
envelope; the two verdicts that already worked stay as they were, so the empty-state branch
cannot reorder the documented precedence; and the page's obligations are read back out of
the document it wrote -- the header's identity, duration and four tallies, one row per result in
``(check_id, case_id)`` order, a skip in the skip block, the failing case's images decoded out of
their ``data:`` URIs, a banner only where the results prove a degradation, the errored case's
traceback, and a second write byte-identical to the first.

Self-containment and escaping are held here too: the page reaches no external origin, and
escaped text keeps its non-ASCII characters literally and still parses back out of the panel.
Markup escaping is not the whole promise either: a target's lone surrogate -- a code point UTF-8
cannot encode -- reaches the document as itself, so the finished document is made encodable before
the file is opened and each such code point becomes U+FFFD. The page is then written whole, with no
raw surrogate in it and no truncated ``index.html`` behind a failed write."""

import dataclasses
import io
import json
import re
import shutil
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import segno

from qrcheck.evidence.mux import decode_base64, decode_image, encode_png
from qrcheck.evidence.naming import (
    DAMAGE_ARTIFACT_TEMPLATE,
    EVIDENCE_DIR_NAME,
    FIXED_ARTIFACT_NAMES,
    LOGO_PNG,
    SYMBOL_PNG,
    SYMBOL_REPEAT_PNG,
)
from qrcheck.report import html as html_module
from qrcheck.report.evidence import EvidenceStore
from qrcheck.report.html import write_html
from qrcheck.report.model import (
    TRACEBACK_KEY,
    CaseResult,
    Counts,
    Reason,
    RunResult,
    Status,
)

# The fragments of the document the assertions read, matched on the markup the writer
# emits: the verdict line with its colour modifier, one named JSON panel with the
# attributes of its ``pre``, and one tally figure.
_VERDICT = re.compile(r'<p class="verdict (?P<modifier>[a-z-]+)">(?P<text>[^<]*)</p>')
_PANEL = re.compile(
    r"<h4>(?P<heading>[^<]*)</h4>\s*<pre(?P<attributes>[^>]*)>(?P<body>.*?)</pre>", re.DOTALL
)
_COUNT_VALUE = re.compile(r'<span class="count-value">(?P<value>[^<]*)</span>')

# The remaining regions the assertions read back: the whole header, one fact inside it, one
# tally label, the degradation banners, the two sections a case can be filed under, one
# embedded figure, and every reference the document carries.
_HEADER = re.compile(r"<header>(?P<body>.*?)</header>", re.DOTALL)
_FACT = re.compile(r"<dt>(?P<label>[^<]*)</dt><dd>(?P<value>[^<]*)</dd>")
_COUNT_LABEL = re.compile(r'<span class="count-label">(?P<label>[^<]*)</span>')
_BANNERS = re.compile(r'<section class="banners"[^>]*>(?P<body>.*?)</section>', re.DOTALL)
_SKIPPED_SECTION = re.compile(r'<section id="skipped">(?P<body>.*?)</section>', re.DOTALL)
_FAILURES_SECTION = re.compile(r'<section id="failures">(?P<body>.*?)</section>', re.DOTALL)
_TABLE_ROW = re.compile(
    r'<tr class="status-(?P<status>[a-z]+)"[^>]*>(?P<cells>.*?)</tr>', re.DOTALL
)
_TABLE_CELL = re.compile(
    r'<td data-label="(?P<label>[A-Za-z]+)"[^>]*>(?P<body>.*?)</td>', re.DOTALL
)
_FIGURE = re.compile(
    r'<figure class="evidence-image">\s*'
    r'<img src="(?P<src>[^"]*)" alt="(?P<alt>[^"]*)">\s*'
    r"<figcaption>(?P<name>[^<]*)</figcaption>",
    re.DOTALL,
)
_IMAGE = re.compile(r"<img\b(?P<attributes>[^>]*)>")
_SRC_ATTRIBUTE = re.compile(r'src="(?P<value>[^"]*)"')
_REFERENCE = re.compile(r'\b(?:src|href)="(?P<value>[^"]*)"')
_TAG = re.compile(r"<[^>]*>")

# The stylesheet the page carries, read as CSS rather than as markup: one fact of a facts grid
# with the attributes of the div that wraps it, the style element's body, one rule with its
# selector list, and a comment to drop before the rules are read.
_FACT_ITEM = re.compile(
    r"<div(?P<attributes>[^>]*)><dt>(?P<label>[^<]*)</dt><dd>(?P<value>[^<]*)</dd></div>"
)
_STYLESHEET = re.compile(r"<style>(?P<body>.*?)</style>", re.DOTALL)
_CSS_RULE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<declarations>[^{}]*)\}")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# Every box in the page that can hold an unbreakable identifier - a case id, an evidence path,
# a check id in a skip bullet - and so has to wrap one inside itself rather than widen the page.
_IDENTIFIER_BOXES: tuple[str, ...] = (
    'tbody td[data-label="Case"]',
    ".facts dd",
    ".reasons dd",
    ".evidence h3",
    ".skips li",
)

# The table's fixed-vocabulary columns. Their words are short and known, so they must NOT wrap
# anywhere: `anywhere` lowers a cell's minimum width, and a wordless break in `skipped` or
# `value_mismatch` is a defect of its own on a viewport where the identifier column is squeezed.
_FIXED_VOCABULARY_CELLS: tuple[str, ...] = (
    "tbody td",
    'tbody td[data-label="Check"]',
    'tbody td[data-label="Status"]',
    'tbody td[data-label="Reason"]',
    'tbody td[data-label="Duration"]',
    'tbody td[data-label="Evidence"]',
)

# The touch target the page's links are held to, and the root font size a rem length resolves at.
_TOUCH_TARGET_PX: int = 44
_ROOT_FONT_PX: int = 16


# The one sentence each case section states when its set is empty, held verbatim: they are
# fixed text by contract - no count, no path, nothing this run measured - so a test that
# matched only a prefix would let a number creep into the page unnoticed.
_EMPTY_FAILURES = (
    '<p class="empty-state">No failures or errors in this run: no case recorded a failed or '
    "errored outcome, so there is no failing case to inspect.</p>"
)
_EMPTY_SKIPS = (
    '<p class="empty-state">No skipped cases in this run: no case recorded a skipped outcome, '
    "so nothing was left unexercised for a missing capability, dependency or platform.</p>"
)

# The accessibility skeleton the writer emits, read back one property at a time: the six column
# labels the table's header row carries in the order the columns are defined, one column header,
# the caption body, and the level of one heading element.
_TABLE_HEADINGS: tuple[str, ...] = (
    "Check",
    "Case",
    "Status",
    "Reason",
    "Duration",
    "Evidence",
)
_TH_HEADING = re.compile(r'<th scope="col"[^>]*>(?P<label>[^<]*)</th>')
_CAPTION = re.compile(r"<caption>(?P<body>.*?)</caption>", re.DOTALL)
_HEADING_LEVEL = re.compile(r"<h(?P<level>[1-4])>")

# A symbol a target could return in vector form, whose own text carries a URL: it reaches
# the page base64-encoded inside a data: URI, so the document holds no reference to it.
_SVG: bytes = b'<svg xmlns="http://www.w3.org/2000/svg" width="21" height="21"></svg>'

# The traceback an errored case carries, already free of any frame path, so this module
# pins the error branch without holding an absolute path of its own.
_TRACEBACK: str = (
    "Traceback (most recent call last):\n"
    '  File "qrcheck/checks/performance.py", line 42, in cases\n'
    '    raise RuntimeError("boom")\n'
    "RuntimeError: boom"
)

# The lone surrogate a target's JSON can carry: ``"\ud800"`` is a legal JSON escape that
# ``json.loads`` materialises as one code point, and UTF-8 has no byte sequence for it. Spelled
# through ``chr`` so this module's own source stays valid UTF-8 and holds no surrogate itself.
_LONE_SURROGATE: str = chr(0xD800)

# What the writer replaces it with: U+FFFD, the replacement character, one code point for one.
_REPLACEMENT_CHARACTER: str = "\ufffd"

# The decoder maps a result can record: both decoders agreeing, which is the healthy shape;
# the primary decoder alone, which is the single-decoder degradation, both when the secondary
# decoder is absent and when it is installed but unusable; and the shapes that prove nothing.
_AGREEING_DECODE: dict[str, object] = {
    "agreement": "agree",
    "per_decoder": {"opencv": {"status": "decoded"}, "zxing": {"status": "decoded"}},
}
_SINGLE_DECODER_DECODE: dict[str, object] = {
    "agreement": "primary_only",
    "per_decoder": {"zxing": {"status": "decoded"}},
}
# The shape a decoder that was consulted and raised leaves: its binding loaded, the call itself
# failed, so it is recorded as an outcome with status error rather than as an unavailable decoder,
# and the primary decoder read the payload alone. The degradation is the same one, so it must
# raise the same banner.
_UNUSABLE_SECONDARY_DECODE: dict[str, object] = {
    "agreement": "primary_only",
    "per_decoder": {"opencv": {"status": "error"}, "zxing": {"status": "decoded"}},
}
# The shape an artefact no decoder could be shown leaves: both decoders errored and nothing
# was read, so no payload was corroborated and no check can have passed on the primary's
# strength. It proves no degradation and must raise no banner.
_UNREADABLE_DECODE: dict[str, object] = {
    "agreement": "undetected",
    "per_decoder": {"opencv": {"status": "error"}, "zxing": {"status": "error"}},
}
# A primary decoder that read no payload beside a secondary that could not run: the banner's
# claim -- a payload corroborated by the primary decoder alone -- would be false here, because
# no payload was read at all.
_NO_PAYLOAD_DECODE: dict[str, object] = {
    "agreement": "undetected",
    "per_decoder": {"opencv": {"status": "error"}, "zxing": {"status": "undetected"}},
}

# A run's identity as the model validates it: a content hash rather than a UUID or a
# date, and two timestamps that carry their timezone.
_RUN_ID = "4f2a9c1b7d3e5061"
_STARTED_AT = "2026-01-01T00:00:00+00:00"
_FINISHED_AT = "2026-01-01T00:01:30+00:00"
_SEED = 20240101

# The request projection a damage check records: the generate request plus the case's
# resolved parameters, which is what the evidence store spreads the identity over.
_REQUEST: dict[str, object] = {
    "ec_level": "M",
    "module_size_px": 4,
    "payload_b64": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
    "payload_len": 32,
    "quiet_zone_modules": 4,
}

# The two runs whose verdicts the empty-state branch must leave alone: one all-passing and
# one whose only result is a skip, which is never a plain pass.
_PASSING_RESULTS: tuple[CaseResult, ...] = (
    CaseResult(
        check_id="RT-ROUNDTRIP",
        case_id="corpus:url-001",
        status=Status.PASSED,
        duration_ms=3.25,
        decoded={"payload_len": 32},
        seed=_SEED,
    ),
    CaseResult(
        check_id="RT-ROUNDTRIP",
        case_id="corpus:url-002",
        status=Status.PASSED,
        duration_ms=2.5,
        decoded={"payload_len": 32},
        seed=_SEED,
    ),
)
_SKIPPED_RESULTS: tuple[CaseResult, ...] = (
    CaseResult(
        check_id="RN-SVG",
        case_id="synthetic:svg-rasterizer",
        status=Status.SKIPPED,
        reason=Reason.MISSING_DEPENDENCY,
        seed=_SEED,
    ),
)


def _failed_case(request: dict[str, object]) -> CaseResult:
    """Return a failed case carrying ``request`` and no evidence directory yet."""
    return CaseResult(
        check_id="DM-RESILIENCE",
        case_id="corpus:random-bytes-000",
        status=Status.FAILED,
        reason=Reason.VALUE_MISMATCH,
        duration_ms=12.5,
        request=request,
        decoded={"payload_len": 31},
        expected={"payload_len": 32},
        seed=_SEED,
    )


def _run(*results: CaseResult, **overrides: object) -> RunResult:
    """Return ``results`` as a run the result model accepts, its tallies derived."""
    fields: dict[str, object] = {
        "run_id": _RUN_ID,
        "started_at": _STARTED_AT,
        "finished_at": _FINISHED_AT,
        "adapter": "reference",
        "seed": _SEED,
        "ec_level": "M",
        "loc_count": 23833,
        "counts": Counts.from_results(results),
        "results": results,
    }
    fields.update(overrides)
    return RunResult(**fields)


def _write_page(root: Path, run: RunResult) -> str:
    """Write ``run``'s report under ``root`` and return the document's text."""
    page = write_html(run, root / "index.html", root)
    return page.read_text(encoding="utf-8")


def _png(payload: str, scale: int = 2) -> bytes:
    """Return a real PNG of ``payload`` from the pinned segno oracle."""
    buffer = io.BytesIO()
    segno.make_qr(payload, error="M", boost_error=False).save(
        buffer, kind="png", scale=scale, border=4
    )
    return buffer.getvalue()


def _derived_png(payload: str, scale: int = 1) -> bytes:
    """Return ``payload``'s symbol re-encoded by the product's own PNG encoder."""
    return encode_png(decode_image(_png(payload, scale=scale)))


def _passed_case(
    check_id: str = "RT-ROUNDTRIP",
    case_id: str = "corpus:url-001",
    decoded: dict[str, object] | None = None,
) -> CaseResult:
    """Return a passing case, decoded by both decoders unless ``decoded`` says otherwise."""
    return CaseResult(
        check_id=check_id,
        case_id=case_id,
        status=Status.PASSED,
        duration_ms=1.0,
        decoded=dict(_AGREEING_DECODE if decoded is None else decoded),
        seed=_SEED,
    )


def _skipped_case(
    check_id: str, case_id: str, reason: Reason = Reason.MISSING_DEPENDENCY
) -> CaseResult:
    """Return a skipped case carrying ``reason``, which is the token a reader greps for."""
    return CaseResult(
        check_id=check_id,
        case_id=case_id,
        status=Status.SKIPPED,
        reason=reason,
        seed=_SEED,
    )


def _error_case(check_id: str, case_id: str) -> CaseResult:
    """Return an errored case carrying the traceback the decoded panel renders."""
    return CaseResult(
        check_id=check_id,
        case_id=case_id,
        status=Status.ERROR,
        reason=Reason.INTERNAL_ERROR,
        decoded={TRACEBACK_KEY: _TRACEBACK},
        seed=_SEED,
    )


def _recorded(case: CaseResult, root: Path, **artifacts: Any) -> CaseResult:
    """Return ``case`` carrying the evidence directory the store wrote for it under ``root``.

    ``artifacts`` are the keyword arguments :meth:`EvidenceStore.record` takes for the images a
    case recorded, so a test names the artefacts it wrote rather than repeating the store's call.
    """
    return dataclasses.replace(case, evidence_dir=EvidenceStore(root).record(case, **artifacts))


def _header(document: str) -> str:
    """Return the body of the document's header region."""
    match = _HEADER.search(document)
    assert match is not None
    return match.group("body")


def _rows(document: str) -> list[dict[str, str]]:
    """Return the results table's rows as their cell text, in document order."""
    rows: list[dict[str, str]] = []
    for row in _TABLE_ROW.finditer(document):
        cells = {
            cell.group("label"): unescape(_TAG.sub("", cell.group("body")))
            for cell in _TABLE_CELL.finditer(row.group("cells"))
        }
        rows.append({"status": row.group("status"), **cells})
    return rows


def _panels(document: str, heading: str) -> list[Any]:
    """Return the JSON every panel headed ``heading`` renders, parsed back out of the page."""
    markup = [match for match in _PANEL.finditer(document) if match.group("heading") == heading]
    return [json.loads(unescape(match.group("body"))) for match in markup]


def _panel(document: str, heading: str) -> Any:
    """Return the JSON the first panel headed ``heading`` renders, parsed out of the page."""
    panels = _panels(document, heading)
    assert panels, heading
    return panels[0]


def _figures(document: str) -> list[tuple[str, bytes]]:
    """Return each embedded figure's artefact name and the bytes its ``data:`` URI holds."""
    figures: list[tuple[str, bytes]] = []
    for figure in _FIGURE.finditer(document):
        source = figure.group("src")
        assert source.startswith("data:"), source
        figures.append((figure.group("name"), decode_base64(source.split(",", 1)[1])))
    return figures


def _stylesheet(document: str) -> str:
    """Return the body of the document's single inline stylesheet."""
    match = _STYLESHEET.search(document)
    assert match is not None
    return match.group("body")


def _declarations(stylesheet: str, selector: str) -> str:
    """Return the declarations of the rule that selects ``selector`` alone, normalised."""
    for rule in _CSS_RULE.finditer(_CSS_COMMENT.sub("", stylesheet)):
        if selector in (part.strip() for part in rule.group("selectors").split(",")):
            return " ".join(rule.group("declarations").split())
    return ""


def _declared_px(declarations: str, name: str) -> float:
    """Return declaration ``name`` in px, converting a rem length at the page root.

    Every box on the page is ``border-box``, so ``min-block-size``/``min-inline-size`` is the
    rendered floor of that axis with its padding inside it.
    """
    match = re.search(rf"(?:^|; ){name}: ([0-9.]+)(rem|px)", declarations)
    assert match is not None, f"{name} is not declared in {declarations!r}"
    return float(match.group(1)) * (_ROOT_FONT_PX if match.group(2) == "rem" else 1)


def _wrapping_selectors(stylesheet: str) -> set[str]:
    """Return the selectors whose rule declares ``overflow-wrap: anywhere`` in ``stylesheet``."""
    selectors: set[str] = set()
    for rule in _CSS_RULE.finditer(_CSS_COMMENT.sub("", stylesheet)):
        if "overflow-wrap: anywhere" not in rule.group("declarations"):
            continue
        selectors.update(
            selector
            for selector in (part.strip() for part in rule.group("selectors").split(","))
            if selector
        )
    return selectors


class _DocumentParser(HTMLParser):
    """Collect what a hostile fragment would change had the writer stopped escaping it:
    ``convert_charrefs`` is on, so an unescaped identifier shows up as a ``script`` start tag."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.text: list[str] = []
        self.rows = 0
        self._in_body = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record ``tag`` and count the rows of the results table's body."""
        self.tags.append(tag)
        if tag == "tbody":
            self._in_body = True
        elif tag == "tr" and self._in_body:
            self.rows += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record a self-closing tag, which no row is."""
        self.tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        """Leave the table body when its closing tag is read."""
        if tag == "tbody":
            self._in_body = False

    def handle_data(self, data: str) -> None:
        """Record the text the document carries as data rather than as markup."""
        self.text.append(data)


def test_zero_result_run_states_that_no_case_ran(tmp_path: Path) -> None:
    """The page says no case ran, so four zero tallies cannot be read as a green run."""
    document = _write_page(tmp_path, _run())

    verdict = _VERDICT.search(document)
    assert verdict is not None
    assert (verdict.group("text"), verdict.group("modifier")) == (
        "no cases ran",
        "verdict-empty",
    )
    assert "run passed" not in document
    assert _COUNT_VALUE.findall(document) == ["0"] * 4
    assert '<section id="results" tabindex="-1">' in document
    assert "<h2>Results</h2>" in document
    assert '<p class="empty-state">No cases ran' in document
    assert "<table" not in document


def test_zero_result_footer_offers_no_row_to_rerun(tmp_path: Path) -> None:
    """The footer keeps the reproduction command but not the promise of table rows."""
    document = _write_page(tmp_path, _run())

    assert "Every row of the results table" not in document
    assert "No case ran, so there is no row to re-run at a seed" in document
    command = (
        "qrcheck run --adapter reference --ec M --checks &lt;CHECK-ID&gt; "
        f"--seed {_SEED}"
    )
    assert command in document
    assert "<code>summary.json</code> and <code>junit.xml</code>" in document


@pytest.mark.parametrize(
    "payload",
    (
        _REQUEST,
        {**_REQUEST, "check_id": "RT-ROUNDTRIP", "case_id": "corpus:url-001", "seed": 1},
    ),
    ids=("request-projection", "envelope-wins-collision"),
)
def test_request_panel_renders_the_recorded_artefact(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """The panel's text parses to the artefact on disk, the case's identity included."""
    case = _failed_case(payload)
    recorded = _recorded(case, tmp_path)
    assert recorded.evidence_dir is not None

    document = _write_page(tmp_path, _run(recorded))
    rendered = _panel(document, "request.json")
    artefact = json.loads(
        (tmp_path / recorded.evidence_dir / "request.json").read_text(encoding="utf-8")
    )

    assert rendered == artefact
    assert rendered["check_id"] == case.check_id
    assert rendered["case_id"] == case.case_id
    assert rendered["seed"] == case.seed


def test_page_embeds_its_image_and_reaches_no_external_origin(tmp_path: Path) -> None:
    """The page stays one self-contained file: inline CSS, a data: image, no origin."""
    case = _failed_case(_REQUEST)
    symbol = SimpleNamespace(image_bytes=_png("qrcheck"), output_format="png")
    recorded = _recorded(case, tmp_path, symbol=symbol)

    document = _write_page(tmp_path, _run(recorded))

    assert document.count("<style>") == 1
    assert "<script" not in document
    assert "<link" not in document
    assert "@import" not in document
    assert "http://" not in document
    assert "https://" not in document
    sources = re.findall(r'src="([^"]*)"', document)
    assert sources
    assert all(source.startswith("data:image/") for source in sources)


def test_hostile_identifier_stays_escaped_and_parses_back_out_of_the_panel(
    tmp_path: Path,
) -> None:
    """A target-supplied case identifier reaches the page as text, never as markup."""
    hostile = "corpus:<script>alert(1)</script>"
    case = dataclasses.replace(_failed_case(_REQUEST), case_id=hostile)
    recorded = _recorded(case, tmp_path)

    document = _write_page(tmp_path, _run(recorded))

    assert "<script" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert _panel(document, "request.json")["case_id"] == hostile


def test_a_lone_surrogate_in_an_identifier_or_a_decoded_value_becomes_u_fffd(
    tmp_path: Path,
) -> None:
    """A target's lone surrogate cannot abort the write or leave a truncated ``index.html``.

    Both the case identifier and a decoder message carry a code point UTF-8 has no bytes for, which
    ``html_escape`` passes through because it makes text inert rather than encodable, so the write
    used to raise half-way through the file. The document is now made encodable before the
    destination is even created: the page is complete, the text a target sent reads as U+FFFD, and
    no raw surrogate reaches it.
    """
    case = dataclasses.replace(
        _failed_case(_REQUEST),
        case_id=f"corpus:bad{_LONE_SURROGATE}",
        decoded={"decoder_message": f"rejected: {_LONE_SURROGATE} is not a code point"},
    )
    # Recorded rather than left unpersisted: a failing case's detail block needs its evidence, and
    # the store now writes a case whose text carries a surrogate instead of refusing the case.
    recorded = _recorded(case, tmp_path)
    assert recorded.evidence_dir is not None

    document = _write_page(tmp_path, _run(recorded))

    page = tmp_path / "index.html"
    assert page.read_bytes().decode("utf-8") == document, "the file must be UTF-8 end to end"
    assert _LONE_SURROGATE not in document
    assert document.endswith("</html>\n"), "the page is written whole, never truncated"
    assert [row["Case"] for row in _rows(document)] == [
        f"corpus:bad{_REPLACEMENT_CHARACTER}"
    ]
    assert f"rejected: {_REPLACEMENT_CHARACTER} is not a code point" in document
    # The evidence that was written for the case reads the same way, so the page and the artefact
    # it renders agree on what the target's text became.
    artefact = (tmp_path / recorded.evidence_dir / "decoded.json").read_text(encoding="utf-8")
    assert f"rejected: {_REPLACEMENT_CHARACTER} is not a code point" in artefact
    assert _LONE_SURROGATE not in artefact


@pytest.mark.parametrize(
    ("results", "text", "modifier", "skips"),
    (
        (_PASSING_RESULTS, "run passed", "verdict-passed", False),
        (_SKIPPED_RESULTS, "run passed (skips present)", "verdict-skipped", True),
    ),
    ids=("all-passing", "skips-only"),
)
def test_a_run_with_results_keeps_its_verdict(
    tmp_path: Path,
    results: tuple[CaseResult, ...],
    text: str,
    modifier: str,
    skips: bool,
) -> None:
    """A run with results keeps its verdict, its table and states the sets it left empty.

    Both case sections are rendered whatever the run recorded, so an empty set is a statement
    rather than a missing heading: neither run here has a failure, so the failures section
    carries its one empty sentence, and the skipped section carries its own exactly when the
    run skipped nothing - a skip is listed instead, under its reason.
    """
    document = _write_page(tmp_path, _run(*results))

    verdict = _VERDICT.search(document)
    assert verdict is not None
    assert (verdict.group("text"), verdict.group("modifier")) == (text, modifier)
    assert '<table role="table">' in document
    assert _FAILURES_SECTION.search(document) is not None
    assert _EMPTY_FAILURES in document
    assert _SKIPPED_SECTION.search(document) is not None
    assert (_EMPTY_SKIPS in document) is (not skips)
    assert document.count('<section id="ev-') == 0
    assert "<img" not in document


def test_header_states_the_run_identity_and_its_four_tallies(tmp_path: Path) -> None:
    """The header carries the run's identity, its duration and four separate tallies."""
    results = (
        _passed_case("RT-ROUNDTRIP", "corpus:url-001"),
        _passed_case("RT-ROUNDTRIP", "corpus:url-002"),
        _failed_case(_REQUEST),
        _skipped_case("RN-SVG", "synthetic:svg-rasterizer", Reason.MISSING_DEPENDENCY),
        _skipped_case("RN-LOGO", "synthetic:logo-overlay", Reason.UNSUPPORTED_CAPABILITY),
        _skipped_case("CL-ARTIFACTS", "synthetic:open-handles", Reason.UNSUPPORTED_PLATFORM),
        _error_case("PF-ENCODE", "synthetic:encode"),
    )
    run = _run(*results)
    header = _header(_write_page(tmp_path, run))

    assert dict(_FACT.findall(header)) == {
        "adapter": "reference",
        "ec_level": "M",
        "seed": str(_SEED),
        "loc_count": "23833",
        "duration": "90.000 s",
        "run_id": _RUN_ID,
    }
    assert _COUNT_LABEL.findall(header) == ["passed", "failed", "skipped", "error"]
    assert _COUNT_VALUE.findall(header) == ["2", "1", "3", "1"]
    assert run.counts == Counts(passed=2, failed=1, skipped=3, error=1)


def test_every_result_gets_one_row_in_check_then_case_order(tmp_path: Path) -> None:
    """One row per result, ordered by check then case, each stating its own outcome."""
    failure = _failed_case(_REQUEST)
    errored = _error_case("PF-ENCODE", "synthetic:encode")
    submitted = (
        _passed_case("RT-ROUNDTRIP", "corpus:url-002"),
        _skipped_case("DM-RESILIENCE", "corpus:random-bytes-001", Reason.UNSUPPORTED_CAPABILITY),
        _recorded(errored, tmp_path),
        _recorded(failure, tmp_path),
        _passed_case("RT-ROUNDTRIP", "corpus:url-001"),
    )
    run = _run(*submitted)
    rows = _rows(_write_page(tmp_path, run))

    assert [(row["Check"], row["Case"]) for row in rows] == [
        (case.check_id, case.case_id) for case in run.results
    ]
    assert [(row["Check"], row["Case"]) for row in rows] == sorted(
        (case.check_id, case.case_id) for case in submitted
    )
    assert [row["status"] for row in rows] == [case.status.value for case in run.results]
    assert [row["Status"] for row in rows] == [case.status.value for case in run.results]
    assert [row["Reason"] for row in rows] == [
        "—" if case.reason is None else case.reason.value for case in run.results
    ]
    assert len(rows) == len(run.results) == len(submitted)
    assert {row["Evidence"] for row in rows} == {"—", "evidence"}


def test_the_skipped_block_lists_each_skip_under_its_own_reason(tmp_path: Path) -> None:
    """A skip is filed in its own block under its reason token, never with the failures.

    The failures section is rendered for this run too and states that it holds none, so a
    skip-only run cannot look like a run whose failure section went missing.
    """
    skips = (
        _skipped_case("RN-SVG", "synthetic:svg-rasterizer", Reason.MISSING_DEPENDENCY),
        _skipped_case("RN-LOGO", "synthetic:logo-overlay", Reason.UNSUPPORTED_CAPABILITY),
        _skipped_case("CL-ARTIFACTS", "synthetic:open-handles", Reason.UNSUPPORTED_PLATFORM),
    )
    recorded = (
        _recorded(skips[0], tmp_path),
        *skips[1:],
    )
    run = _run(*recorded)
    document = _write_page(tmp_path, run)

    section = _SKIPPED_SECTION.search(document)
    assert section is not None
    block = section.group("body")
    for case in run.results:
        assert case.reason is not None
        assert f"<code>{case.reason.value}</code>" in block
    assert [row["Status"] for row in _rows(document)] == ["skipped"] * len(skips)
    assert [row["Reason"] for row in _rows(document)] == [case.reason.value for case in run.results]
    assert block.count("<li><code>") == len(skips)
    failures = _FAILURES_SECTION.search(document)
    assert failures is not None
    assert _EMPTY_FAILURES in failures.group("body")
    assert '<section id="ev-' not in failures.group("body")
    assert _COUNT_VALUE.findall(_header(document)) == ["0", "0", str(len(skips)), "0"]
    assert [row["Evidence"] for row in _rows(document)] == [
        "recorded, not shown" if case.evidence_dir else "—" for case in run.results
    ]


def test_a_missing_evidence_directory_states_the_failure_and_embeds_nothing(
    tmp_path: Path,
) -> None:
    """A failure whose evidence is gone says so, rather than embedding an empty image."""
    failure = _failed_case(_REQUEST)
    recorded = _recorded(
        failure,
        tmp_path,
        symbol=SimpleNamespace(image_bytes=_derived_png("symbol"), output_format="png"),
    )
    assert recorded.evidence_dir is not None
    shutil.rmtree(tmp_path / recorded.evidence_dir)

    document = _write_page(tmp_path, _run(recorded))

    assert _FIGURE.findall(document) == []
    assert "<img" not in document
    assert '<p class="evidence-unreadable">evidence directory could not be listed' in document
    assert [(row["Check"], row["Status"]) for row in _rows(document)] == [
        (recorded.check_id, "failed")
    ]


def test_a_failing_case_embeds_the_images_the_evidence_store_wrote(
    report_dir: Path,
) -> None:
    """Every image a failing case recorded is embedded, and its alternative describes it."""
    failure = _failed_case(_REQUEST)
    damage_name = DAMAGE_ARTIFACT_TEMPLATE.format(kind="blur", seed=failure.seed)
    recorded = _recorded(
        failure,
        report_dir,
        symbol=SimpleNamespace(image_bytes=_derived_png("symbol"), output_format="png"),
        repeat_symbol=SimpleNamespace(image_bytes=_derived_png("repeat"), output_format="png"),
        damage_variants=(("blur", _derived_png("damage")),),
        logo=_derived_png("logo"),
    )
    assert recorded.evidence_dir is not None
    assert recorded.evidence_dir.startswith(f"{EVIDENCE_DIR_NAME}/")
    directory = report_dir / recorded.evidence_dir
    expected = sorted((SYMBOL_PNG, SYMBOL_REPEAT_PNG, LOGO_PNG, damage_name))

    document = _write_page(
        report_dir, _run(recorded, _passed_case("RT-ROUNDTRIP", "corpus:url-1"))
    )
    figures = _figures(document)

    assert [name for name, _ in figures] == expected
    for name, data in figures:
        assert data
        assert data == (directory / name).read_bytes()
    on_disk = sorted(path.name for path in directory.iterdir() if path.suffix in {".png", ".svg"})
    assert on_disk == expected
    assert all(name in FIXED_ARTIFACT_NAMES or name == damage_name for name, _ in figures)
    assert document.count('<section id="ev-') == 1
    alternatives = {
        figure.group("name"): unescape(figure.group("alt"))
        for figure in _FIGURE.finditer(document)
    }
    assert sorted(alternatives) == expected
    for name, alternative in alternatives.items():
        assert alternative, name
        assert alternative != name
        assert recorded.check_id in alternative
        assert recorded.case_id in alternative
    assert len(set(alternatives.values())) == len(expected)
    assert f"blur damage with damage seed {failure.seed}" in alternatives[damage_name]


def test_a_passing_case_embeds_no_image(tmp_path: Path) -> None:
    """A case with no evidence directory contributes a row, no block and no image.

    Both case sections are still rendered, each stating that this run left its set empty,
    so an all-passing page says so in words instead of ending at the results table.
    """
    run = _run(
        _passed_case("RT-ROUNDTRIP", "corpus:url-001"),
        _passed_case("RT-ROUNDTRIP", "corpus:url-002"),
    )
    document = _write_page(tmp_path, run)

    assert "<img" not in document
    assert "data:" not in document
    assert _FIGURE.findall(document) == []
    failures = _FAILURES_SECTION.search(document)
    assert failures is not None
    assert _EMPTY_FAILURES in failures.group("body")
    skipped = _SKIPPED_SECTION.search(document)
    assert skipped is not None
    assert _EMPTY_SKIPS in skipped.group("body")
    assert [(row["Status"], row["Reason"], row["Evidence"]) for row in _rows(document)] == [
        ("passed", "—", "—"),
        ("passed", "—", "—"),
    ]


def test_target_supplied_text_reaches_the_page_as_escaped_text(tmp_path: Path) -> None:
    """Escape the one injection surface, and keep the characters a reader must see."""
    hostile = "<script>alert(1)</script>"
    escaped = "&lt;script&gt;alert(1)&lt;/script&gt;"
    non_ascii = "ünïcode — ✓"
    supplied: dict[str, object] = {
        "hostile": hostile,
        "double": '"double"',
        "single": "'single'",
        "amp": "&",
        "note": non_ascii,
    }
    failure = dataclasses.replace(
        _failed_case(supplied), decoded=supplied, expected=supplied
    )
    recorded = _recorded(failure, tmp_path)

    document = _write_page(tmp_path, _run(recorded, adapter=hostile))

    assert "<script" not in document.lower()
    assert escaped in document
    assert "&amp;" in document
    assert "&quot;double&quot;" in document
    assert "&#x27;single&#x27;" in document
    assert non_ascii in document
    assert dict(_FACT.findall(_header(document)))["adapter"] == escaped
    for heading in ("request.json", "decoded.json", "expected"):
        rendered = _panel(document, heading)
        assert isinstance(rendered, dict)
        for key, value in supplied.items():
            assert rendered[key] == value
    parser = _DocumentParser()
    parser.feed(document)
    parser.close()
    assert "script" not in parser.tags
    assert hostile in "".join(parser.text)
    assert parser.rows == 1


def test_every_reference_is_inline_data_or_a_fragment(tmp_path: Path) -> None:
    """The page holds one inline stylesheet and references nothing outside itself."""
    png_case = _failed_case(_REQUEST)
    svg_case = dataclasses.replace(
        _failed_case(_REQUEST), check_id="RN-SVG", case_id="corpus:url-003"
    )
    recorded = (
        _recorded(
            png_case,
            tmp_path,
            symbol=SimpleNamespace(image_bytes=_derived_png("png-case"), output_format="png"),
        ),
        _recorded(
            svg_case, tmp_path, symbol=SimpleNamespace(image_bytes=_SVG, output_format="svg")
        ),
    )
    document = _write_page(tmp_path, _run(*recorded))
    lowered = document.lower()

    assert document.count("<style>") == 1
    assert "<link" not in lowered
    assert "<script" not in lowered
    assert "@import" not in lowered
    assert "url(http" not in lowered
    references = _REFERENCE.findall(document)
    assert references
    assert all(value.startswith(("data:", "#")) for value in references)
    assert {value.split(":", 1)[0] for value in references if not value.startswith("#")} == {
        "data"
    }
    images = _IMAGE.findall(document)
    assert len(images) == len(_FIGURE.findall(document)) == len(recorded)
    for attributes in images:
        source = _SRC_ATTRIBUTE.search(attributes)
        assert source is not None
        assert source.group("value").startswith("data:image/")
        assert 'alt="' in attributes
    assert '<img src="data:image/svg+xml;base64,' in document


def test_the_page_carries_the_accessibility_skeleton_it_declares(tmp_path: Path) -> None:
    """Every structural property the writer declares is present in the page it wrote.

    One page holds a failing case with evidence, an errored case and a skip, so the results
    table, two detail blocks with their panels and the skip section all exist -- which is what
    makes the heading sequence below the whole document's rather than one region's. The roles
    at the end are the ones the narrow-viewport collapse needs: that rule overrides ``display``
    on the table parts, which strips the roles a browser would otherwise derive from them.
    """
    failure = _failed_case(_REQUEST)
    errored = _error_case("PF-ENCODE", "synthetic:encode")
    results = (
        _recorded(
            failure,
            tmp_path,
            symbol=SimpleNamespace(image_bytes=_derived_png("symbol"), output_format="png"),
        ),
        _recorded(errored, tmp_path),
        _skipped_case("RN-SVG", "synthetic:svg-rasterizer", Reason.MISSING_DEPENDENCY),
    )

    document = _write_page(tmp_path, _run(*results))

    assert '<html lang="en">' in document
    caption = _CAPTION.search(document)
    assert caption is not None
    assert caption.group("body").startswith(
        "One row per case, ordered by check ID and then by case ID."
    )
    assert "em dash means nothing was recorded" in caption.group("body")
    assert _TH_HEADING.findall(document) == list(_TABLE_HEADINGS)
    levels = [int(level) for level in _HEADING_LEVEL.findall(document)]
    assert tuple(levels) == (1, 2, 2, 3, 4, 4, 4, 3, 4, 4, 2, 2)
    assert all(later - earlier <= 1 for earlier, later in zip(levels, levels[1:]))
    assert '<a class="skip-link" href="#main">Skip to main content</a>' in document
    assert '<main id="main" tabindex="-1">' in document
    assert "@media (max-width: 720px)" in document
    for role in ("table", "rowgroup", "row", "columnheader", "cell"):
        assert f'role="{role}"' in document


def test_an_errored_case_renders_its_traceback_in_its_block(tmp_path: Path) -> None:
    """An error gets its own detail block, and its panel holds the traceback it carries."""
    errored = _error_case("PF-ENCODE", "synthetic:encode")
    recorded = _recorded(errored, tmp_path)
    document = _write_page(tmp_path, _run(recorded))

    assert '<section id="ev-1" class="evidence status-error"' in document
    assert document.count('<section id="ev-') == 1
    assert _panel(document, "decoded.json") == dict(recorded.decoded)
    assert _TRACEBACK.splitlines()[-1] in document
    assert "&quot;" in document


def test_a_case_without_an_evidence_directory_still_gets_its_block(tmp_path: Path) -> None:
    """A failed or errored case whose evidence could not be written keeps its block.

    The request, the decoded-or-error outcome, the expected value and the seed are all derived
    from the result model, so only the image area degrades - to the statement that no image was
    recorded - and the table's Evidence cell is the link to the block, never the em dash that
    means nothing was recorded anywhere.
    """
    failed = _failed_case(_REQUEST)
    errored = dataclasses.replace(_error_case("PF-ENCODE", "synthetic:encode"), seed=1)
    assert failed.evidence_dir is None
    assert errored.evidence_dir is None

    document = _write_page(tmp_path, _run(failed, errored))

    assert '<section id="ev-1" class="evidence status-failed"' in document
    assert '<section id="ev-2" class="evidence status-error"' in document
    assert document.count('<section id="ev-') == 2
    assert document.count('<p class="no-image">no image recorded for this case</p>') == 2
    assert "<img" not in document
    assert [row["Evidence"] for row in _rows(document)] == ["evidence", "evidence"]
    assert document.count('<a class="evidence-link" href="#ev-') == 2
    assert "<div><dt>seed</dt><dd>1</dd></div>" in document
    # The evidence directory is a long-identifier fact, so its div carries the class that gives it
    # the whole grid width; the absent directory is still stated as the em dash.
    assert document.count(
        '<div class="fact-wide"><dt>evidence_dir</dt><dd>—</dd></div>'
    ) == 2
    assert _panel(document, "request.json") == {
        **_REQUEST,
        "check_id": failed.check_id,
        "case_id": failed.case_id,
        "seed": failed.seed,
    }
    decoded_panels = _panels(document, "decoded.json")
    assert decoded_panels == [dict(failed.decoded), dict(errored.decoded)]
    assert decoded_panels[1][TRACEBACK_KEY].splitlines()[-1] == "RuntimeError: boom"


@pytest.mark.parametrize(
    ("results", "banner", "phrase"),
    (
        (
            (_passed_case("RT-ROUNDTRIP", "corpus:url-001", _SINGLE_DECODER_DECODE),),
            "banner-single-decoder",
            "Single-decoder mode.",
        ),
        (
            (_passed_case("RT-ROUNDTRIP", "corpus:url-001", _UNUSABLE_SECONDARY_DECODE),),
            "banner-single-decoder",
            "Single-decoder mode.",
        ),
        (
            (
                _passed_case("RT-ROUNDTRIP", "corpus:url-001"),
                _passed_case("RT-ROUNDTRIP", "corpus:url-002", _UNUSABLE_SECONDARY_DECODE),
            ),
            "banner-single-decoder",
            "Decoder corroboration incomplete.",
        ),
        (
            (_skipped_case("RN-SVG", "synthetic:svg-rasterizer", Reason.MISSING_DEPENDENCY),),
            "banner-missing-dependency",
            "Missing dependency.",
        ),
        (
            (_skipped_case("DM-SEEDED", "corpus:random-bytes-000", Reason.MISSING_DEPENDENCY),),
            "banner-missing-dependency",
            "Missing dependency.",
        ),
        (
            (_skipped_case("CL-ARTIFACTS", "synthetic:open-handles", Reason.UNSUPPORTED_PLATFORM),),
            "banner-unsupported-platform",
            "Unsupported platform.",
        ),
    ),
    ids=(
        "single-decoder",
        "single-decoder-unusable-secondary",
        "decoder-corroboration-incomplete",
        "missing-dependency",
        "missing-dependency-dm-seeded",
        "unsupported-platform",
    ),
)
def test_a_degraded_run_states_the_degradation_in_a_banner(
    tmp_path: Path, results: tuple[CaseResult, ...], banner: str, phrase: str
) -> None:
    """Each degradation the results themselves prove reaches the page as a banner."""
    section = _BANNERS.search(_write_page(tmp_path, _run(*results)))

    assert section is not None
    block = section.group("body")
    assert f'class="banner {banner}"' in block
    assert phrase in block


def test_the_missing_dependency_banner_names_the_check_and_no_cause(tmp_path: Path) -> None:
    """The banner names the check a missing prerequisite denied, never a guessed cause.

    ``missing_dependency`` is recorded by more than one producer: an absent SVG rasterizer, a
    check's unmet declared dependency token, and a check whose run-scoped precondition was
    never filled. The token therefore cannot establish which prerequisite was absent, so a
    run of ``DM-SEEDED`` alone -- every case of which skips for a precondition another check
    fills rather than for an install -- must not be reported as a rasterizer problem.
    """
    skip = _skipped_case("DM-SEEDED", "corpus:random-bytes-000", Reason.MISSING_DEPENDENCY)
    section = _BANNERS.search(_write_page(tmp_path, _run(skip)))

    assert section is not None
    block = section.group("body")
    assert 'class="banner banner-missing-dependency"' in block
    assert "<code>DM-SEEDED</code>" in block
    assert "rasteriz" not in block.lower()
    assert "RN-SVG" not in block


def test_the_decoder_banner_states_how_many_cases_it_did_not_answer_for(
    tmp_path: Path,
) -> None:
    """A partly corroborated run reports its counts instead of claiming single-decoder mode.

    One agreeing result beside one whose secondary decoder errored on the artefact proves the
    degradation for one case and not for the run, so the heading says corroboration is
    incomplete and the sentence states how many of how many decode-bearing cases were not
    corroborated -- the only claim the recorded statuses support.
    """
    run = _run(
        _passed_case("RT-ROUNDTRIP", "corpus:url-001"),
        _passed_case("RT-ROUNDTRIP", "corpus:url-002", _UNUSABLE_SECONDARY_DECODE),
    )
    section = _BANNERS.search(_write_page(tmp_path, run))

    assert section is not None
    block = section.group("body")
    assert "Decoder corroboration incomplete." in block
    assert "Single-decoder mode." not in block
    assert "did not answer for 1 of 2" in block


@pytest.mark.parametrize(
    ("decoded", "why"),
    (
        (
            _UNREADABLE_DECODE,
            "no decoder read the artefact, so no payload was corroborated by anyone",
        ),
        (
            _NO_PAYLOAD_DECODE,
            "the primary decoder read no payload, so no check can have passed on its strength",
        ),
        ({"payload_len": 32}, "the result records no decoder map at all"),
    ),
    ids=("unreadable-artefact", "no-payload-read", "no-decoder-map"),
)
def test_a_result_that_proves_no_degradation_carries_no_single_decoder_banner(
    tmp_path: Path, decoded: dict[str, object], why: str
) -> None:
    """The banner is withheld from the shapes that prove nothing, one reason each.

    The decoder shapes here differ from the degraded ones only in their statuses, which is why they
    are held as one parametrised test: an artefact neither decoder could be shown, and a primary
    that read no payload beside a secondary that could not run, both leave nothing for the banner's
    sentence to state, because its claim is about a payload the primary decoder corroborated alone.
    ``why`` names what each shape fails to prove. A result that records no decoder map at all is
    skipped the same way, so the writer never fails over evidence it knows how to do without.
    """
    run = _run(dataclasses.replace(_passed_case(), decoded=decoded))
    document = _write_page(tmp_path, run)

    assert _BANNERS.search(document) is None, why
    assert 'class="banner' not in document, why
    assert "Single-decoder mode." not in document, why


def test_a_healthy_run_carries_no_banner(tmp_path: Path) -> None:
    """A run whose results prove no degradation tells the reader about none."""
    run = _run(_passed_case(), _passed_case(case_id="corpus:url-002"))
    document = _write_page(tmp_path, run)

    assert _BANNERS.search(document) is None
    assert 'class="banner' not in document
    assert "Single-decoder mode." not in document
    skipped = _SKIPPED_SECTION.search(document)
    assert skipped is not None
    assert _EMPTY_SKIPS in skipped.group("body")


def test_two_writes_of_one_run_are_byte_identical(tmp_path: Path) -> None:
    """The page is a pure function of the result model and the evidence bytes on disk."""
    failure = _failed_case(_REQUEST)
    recorded = _recorded(
        failure,
        tmp_path,
        symbol=SimpleNamespace(image_bytes=_derived_png("symbol"), output_format="png"),
    )
    run = _run(
        recorded,
        _passed_case(),
        _skipped_case("RN-SVG", "synthetic:svg-rasterizer"),
        _error_case("PF-ENCODE", "synthetic:encode"),
    )

    first = write_html(run, tmp_path / "first.html", tmp_path)
    second = write_html(run, tmp_path / "second.html", tmp_path)
    again = write_html(run, first, tmp_path)

    assert first.read_bytes() == second.read_bytes() == again.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert not first.read_bytes().endswith(b"\n\n")


def test_each_failing_case_block_carries_its_own_reproduction_command(tmp_path: Path) -> None:
    """The block's command names its own check and the run's target, seed and EC level.

    The two cases differ in their check ID, so a single constant command cannot satisfy both, and
    the count at the end separates the two block commands from the footer's placeholder one. Each
    block's command repeats the run's adapter, EC level and run seed, because those are the values a
    copy has to carry to re-run a case of this run rather than of the configured default target.
    """
    cases = (
        _failed_case(_REQUEST),
        dataclasses.replace(
            _failed_case(_REQUEST), check_id="RT-ROUNDTRIP", case_id="corpus:url-007", seed=7
        ),
    )
    recorded = tuple(_recorded(case, tmp_path) for case in cases)

    document = _write_page(tmp_path, _run(*recorded))

    commands: list[str] = []
    for case in cases:
        start = document.index(f"<h3>{case.check_id} — {case.case_id}</h3>")
        block = document[start : document.index("</section>", start)]
        command = (
            '<p class="command"><code>qrcheck run --adapter reference --ec M '
            f"--checks {case.check_id} --seed {_SEED}</code></p>"
        )
        assert command in block
        assert "&lt;CHECK-ID&gt;" not in block
        commands.append(command)
    assert commands[0] != commands[1]
    assert document.count("qrcheck run --adapter reference --ec M") == len(cases) + 1


def test_the_command_carries_the_run_seed_not_the_case_s_derived_one(tmp_path: Path) -> None:
    """A derived case seed must not reach the command's run-seed slot.

    ``DM-RESILIENCE`` derives each case's seed from the run seed, so passing that derived value back
    into ``--seed`` re-derives every draw from the wrong root and reproduces a different case than
    the page shows. The command carries the run seed, while the block's ``seed`` fact keeps
    recording the case's own derived seed that the damage artefact is named for.
    """
    derived = 1617066119022174236
    failure = dataclasses.replace(_failed_case(_REQUEST), seed=derived)
    recorded = _recorded(failure, tmp_path)

    document = _write_page(tmp_path, _run(recorded))

    assert (
        '<p class="command"><code>qrcheck run --adapter reference --ec M '
        f"--checks DM-RESILIENCE --seed {_SEED}</code></p>" in document
    )
    assert f"--seed {derived}</code>" not in document
    assert f"<div><dt>seed</dt><dd>{derived}</dd></div>" in document


def test_a_command_from_a_non_default_target_names_that_target(tmp_path: Path) -> None:
    """Both commands name the run's adapter and EC level, which the CLI would default away.

    ``--adapter`` defaults to the configured ``default_adapter`` and ``--ec`` to ``M``, so a
    command naming neither re-runs the conforming default target and reports passes where
    the page shows the defective target's failures. A page written for
    a differently targeted run differs in exactly that position.
    """
    failure = _failed_case(_REQUEST)
    recorded = _recorded(failure, tmp_path)

    document = _write_page(tmp_path, _run(recorded, adapter="defective", ec_level="L"))

    block_command = (
        '<p class="command"><code>qrcheck run --adapter defective --ec L '
        f"--checks DM-RESILIENCE --seed {_SEED}</code></p>"
    )
    footer_command = (
        '<p class="command"><code>qrcheck run --adapter defective --ec L '
        f"--checks &lt;CHECK-ID&gt; --seed {_SEED}</code></p>"
    )
    assert block_command in document
    assert footer_command in document
    assert document.count("--adapter defective --ec L") == 2

    other = write_html(_run(recorded), tmp_path / "reference-target.html", tmp_path)
    other_document = other.read_text(encoding="utf-8")
    assert block_command not in other_document
    assert (
        '<p class="command"><code>qrcheck run --adapter reference --ec M '
        f"--checks DM-RESILIENCE --seed {_SEED}</code></p>" in other_document
    )


def test_the_decoded_panel_renders_the_artefact_the_store_wrote(tmp_path: Path) -> None:
    """The decoded.json panel parses to the file beside it, its non-ASCII payload included."""
    decoded: dict[str, object] = {
        "payload_len": 31,
        "payload_text": "ünïcode — ✓",
        "per_decoder": {"opencv": {"status": "decoded"}, "zxing": {"status": "decoded"}},
    }
    failure = dataclasses.replace(_failed_case(_REQUEST), decoded=decoded)
    recorded = _recorded(failure, tmp_path)
    assert recorded.evidence_dir is not None

    document = _write_page(tmp_path, _run(recorded))

    (markup,) = [
        match for match in _PANEL.finditer(document) if match.group("heading") == "decoded.json"
    ]
    assert markup.group("attributes") == ' dir="auto"'
    artefact = json.loads(
        (tmp_path / recorded.evidence_dir / "decoded.json").read_text(encoding="utf-8")
    )
    assert _panel(document, "decoded.json") == artefact == decoded


def test_an_unreadable_artefact_is_stated_and_the_readable_ones_still_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present-but-unreadable image is named, and every artefact that did read is shown.

    File mode cannot express this: the suite may run as root, where a read-only file still
    reads. The module's single read seam is patched instead, which is the one place a read
    failure can be raised from.
    """
    failure = _failed_case(_REQUEST)
    recorded = _recorded(
        failure,
        tmp_path,
        symbol=SimpleNamespace(image_bytes=_derived_png("symbol"), output_format="png"),
        logo=_derived_png("logo"),
    )
    assert recorded.evidence_dir is not None
    real_read = html_module._read_artifact

    def _refuse_the_symbol(path: Path) -> bytes:
        if path.name == SYMBOL_PNG:
            raise OSError("simulated read failure")
        return real_read(path)

    monkeypatch.setattr(html_module, "_read_artifact", _refuse_the_symbol)

    document = _write_page(tmp_path, _run(recorded))

    assert (
        '<p class="evidence-unreadable">image present but could not be read '
        f"({SYMBOL_PNG}: OSError)</p>" in document
    )
    assert [name for name, _ in _figures(document)] == [LOGO_PNG]
    assert (tmp_path / recorded.evidence_dir / SYMBOL_PNG).is_file()


def test_a_file_outside_the_closed_evidence_set_is_never_published(tmp_path: Path) -> None:
    """An unrecognised name in the case directory is ignored rather than surfaced."""
    failure = _failed_case(_REQUEST)
    recorded = _recorded(
        failure,
        tmp_path,
        symbol=SimpleNamespace(image_bytes=_derived_png("symbol"), output_format="png"),
    )
    assert recorded.evidence_dir is not None
    directory = tmp_path / recorded.evidence_dir
    hidden = ("notes.txt", "damage-hologram-1.png")
    for name in hidden:
        (directory / name).write_bytes(b"not a member of the closed evidence set")

    document = _write_page(tmp_path, _run(recorded))

    assert [name for name, _ in _figures(document)] == [SYMBOL_PNG]
    assert document.count("<figure") == 1
    for name in hidden:
        assert name not in document


def test_the_page_never_carries_the_evidence_directory_as_an_absolute_path(
    tmp_path: Path,
) -> None:
    """The evidence directory reaches the page as the relative posix path the model holds."""
    failure = _failed_case(_REQUEST)
    recorded = _recorded(failure, tmp_path)
    assert recorded.evidence_dir is not None

    document = _write_page(tmp_path, _run(recorded))

    assert str(tmp_path) not in document
    facts = dict(_FACT.findall(document))
    assert facts["evidence_dir"] == recorded.evidence_dir
    assert facts["evidence_dir"].startswith(f"{EVIDENCE_DIR_NAME}/")
    assert "\\" not in facts["evidence_dir"]


def test_a_detail_blocks_long_identifier_facts_take_the_whole_facts_grid(
    tmp_path: Path,
) -> None:
    """A case id and an evidence path span the block's facts grid; the short facts do not.

    The block reuses the header's ``.facts`` grid, which gives every fact one 11 rem track, so
    the two facts whose value is a long identifier break mid-token there and push the last fact
    alone into a row of its own. Marking those two lets them take the card's full width, while
    the header's own facts stay unmarked and keep the column count they had.
    """
    failure = _failed_case(_REQUEST)
    recorded = dataclasses.replace(failure, evidence_dir=EvidenceStore(tmp_path).record(failure))
    assert recorded.evidence_dir is not None

    document = _write_page(tmp_path, _run(recorded))

    grids = re.findall(r'<dl class="facts">(?P<body>.*?)</dl>', document, re.DOTALL)
    assert len(grids) == 2, "the header's facts and the failing case's block"
    header, block = (
        [(item.group("label"), item.group("attributes")) for item in _FACT_ITEM.finditer(grid)]
        for grid in grids
    )
    assert [label for label, attributes in block if "fact-wide" in attributes] == [
        "case_id",
        "evidence_dir",
    ]
    assert [label for label, attributes in block if "fact-wide" not in attributes] == [
        "check_id",
        "status",
        "reason",
        "seed",
        "duration",
    ]
    assert all("fact-wide" not in attributes for _, attributes in header)
    assert _declarations(_stylesheet(document), ".facts .fact-wide") == "grid-column: 1 / -1;"


@pytest.mark.parametrize("selector", _IDENTIFIER_BOXES)
def test_every_box_that_can_hold_an_identifier_wraps_it_at_every_width(
    tmp_path: Path, selector: str
) -> None:
    """The wrap rule sits in the base sheet, not only inside the narrow-viewport block.

    A cell computing ``overflow-wrap: normal`` cannot shrink below its longest token, so an
    identifier longer than the product's own vocabulary widens the table and pushes the Status,
    Reason, Duration and Evidence columns off-screen. The declaration therefore has to hold
    above the 720 px breakpoint too, which is what reading the sheet above ``@media`` proves.
    """
    document = _write_page(tmp_path, _run(_failed_case(_REQUEST)))

    base, separator, narrow = _stylesheet(document).partition("@media (max-width: 720px)")

    assert separator and narrow, "the single-column collapse is still declared"
    assert selector in _wrapping_selectors(base)


@pytest.mark.parametrize("selector", _FIXED_VOCABULARY_CELLS)
def test_the_wrap_rule_stays_off_the_fixed_vocabulary_columns(
    tmp_path: Path, selector: str
) -> None:
    """Only the column that carries a case id may break a token; short words stay whole.

    ``overflow-wrap: anywhere`` lowers the minimum width of the box it is declared on, so
    declaring it on every cell lets the table squeeze the Status column below the width of the
    word it holds, and ``skipped`` renders as ``skippe`` / ``d`` at a viewport where the
    identifier column needs the room. The fixed-vocabulary columns therefore keep their default
    wrapping and only the identifier column gives way.
    """
    document = _write_page(tmp_path, _run(_failed_case(_REQUEST)))

    assert selector not in _wrapping_selectors(_stylesheet(document))


def test_the_evidence_link_clears_the_touch_target(tmp_path: Path) -> None:
    """An evidence link is a pointer target, so its rule sizes the box past the touch target.

    The link stays an inline-level box at every width - the narrow-viewport rule turns the
    table's cells into blocks - so neither axis may rest on the width of the label's text, and
    the label is centred in the box instead of pinned to its top.
    """
    document = _write_page(tmp_path, _run(_failed_case(_REQUEST)))
    declarations = _declarations(_stylesheet(document), ".evidence-link")

    assert _declared_px(declarations, "min-block-size") >= _TOUCH_TARGET_PX
    assert _declared_px(declarations, "min-inline-size") >= _TOUCH_TARGET_PX
    assert "align-items: center" in declarations
