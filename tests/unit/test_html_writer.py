"""The HTML writer's request panel, its zero-result state and the verdicts it keeps.

AAP §0.9.1 names this module as the place the writer's contract is held, and AAP §0.8.4
fixes what the page promises its reader: a failing case's block shows the ``request.json``
the evidence store wrote, and no page state may let a reader read a run that did not pass
as a pass. Every case below is rendered through the product's own writers -
:func:`qrcheck.report.html.write_html` for the page and
:meth:`qrcheck.report.evidence.EvidenceStore.record` for the artefact the page is compared
against - so nothing here re-implements the markup it asserts on.

These groups are pinned:

* a run that recorded no result states that no case ran, keeps its four zero tallies,
  replaces the results table with the empty-state paragraph, and stops offering a reader
  a row that does not exist to re-run;
* the ``request.json`` panel parses to the object in the recorded artefact, its identity
  envelope included, and the envelope wins a collision with the request projection;
* the two verdicts that already worked stay exactly as they were - an all-passing run is
  still ``run passed`` and a run whose only results are skips is still
  ``run passed (skips present)`` - so a change to the empty-state branch cannot reorder
  the precedence AAP §0.3.4 fixes.

AAP §0.9.1 also puts self-containment and escaping on this module, and the panel above is
the writer's widest sink of target-supplied text, so both are held here too: the page
reaches no external origin and embeds its image as a ``data:`` URI, and a hostile case
identifier stays escaped text that still parses back out of the panel.

Every run is built in this module and written under ``tmp_path``; nothing reads a clock,
the network, the environment or a directory this module did not create.
"""

import dataclasses
import io
import json
import re
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import pytest
import segno

from qrcheck.report.evidence import EvidenceStore
from qrcheck.report.html import write_html
from qrcheck.report.model import CaseResult, Counts, Reason, RunResult, Status

# The fragments of the document the assertions read, matched on the markup the writer
# emits: the verdict line with its colour modifier, the body of the panel headed
# request.json, and one tally figure.
_VERDICT = re.compile(r'<p class="verdict (?P<modifier>[a-z-]+)">(?P<text>[^<]*)</p>')
_REQUEST_PANEL = re.compile(
    r"<h4>request\.json</h4>\s*<pre>(?P<body>.*?)</pre>", re.DOTALL
)
_COUNT_VALUE = re.compile(r'<span class="count-value">(?P<value>[^<]*)</span>')

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
# one whose only result is a skip, which is never a plain pass (AAP §0.3.4).
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


def _run(*results: CaseResult) -> RunResult:
    """Return ``results`` as a run the result model accepts, its tallies derived."""
    return RunResult(
        run_id=_RUN_ID,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
        adapter="reference",
        seed=_SEED,
        ec_level="M",
        loc_count=23833,
        counts=Counts.from_results(results),
        results=results,
    )


def _write_page(root: Path, run: RunResult) -> str:
    """Write ``run``'s report under ``root`` and return the document's text."""
    page = write_html(run, root / "index.html", root)
    return page.read_text(encoding="utf-8")


def _png(payload: str) -> bytes:
    """Return a real PNG of ``payload`` from the pinned segno oracle."""
    buffer = io.BytesIO()
    segno.make_qr(payload, error="M", boost_error=False).save(
        buffer, kind="png", scale=2, border=4
    )
    return buffer.getvalue()


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
    assert "<table>" not in document


def test_zero_result_footer_offers_no_row_to_rerun(tmp_path: Path) -> None:
    """The footer keeps the reproduction command but not the promise of table rows."""
    document = _write_page(tmp_path, _run())

    assert "Every row of the results table" not in document
    assert "No case ran, so there is no row to re-run at a seed" in document
    assert "qrcheck run --checks &lt;CHECK-ID&gt; --seed &lt;seed&gt;" in document
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
    recorded = dataclasses.replace(case, evidence_dir=EvidenceStore(tmp_path).record(case))
    assert recorded.evidence_dir is not None

    document = _write_page(tmp_path, _run(recorded))
    panel = _REQUEST_PANEL.search(document)
    assert panel is not None
    rendered = json.loads(unescape(panel.group("body")))
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
    recorded = dataclasses.replace(
        case, evidence_dir=EvidenceStore(tmp_path).record(case, symbol=symbol)
    )

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
    recorded = dataclasses.replace(case, evidence_dir=EvidenceStore(tmp_path).record(case))

    document = _write_page(tmp_path, _run(recorded))

    assert "<script" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    panel = _REQUEST_PANEL.search(document)
    assert panel is not None
    assert json.loads(unescape(panel.group("body")))["case_id"] == hostile


@pytest.mark.parametrize(
    ("results", "text", "modifier"),
    (
        (_PASSING_RESULTS, "run passed", "verdict-passed"),
        (_SKIPPED_RESULTS, "run passed (skips present)", "verdict-skipped"),
    ),
    ids=("all-passing", "skips-only"),
)
def test_a_run_with_results_keeps_its_verdict(
    tmp_path: Path, results: tuple[CaseResult, ...], text: str, modifier: str
) -> None:
    """A run with results keeps its verdict, its table and no empty-state paragraph."""
    document = _write_page(tmp_path, _run(*results))

    verdict = _VERDICT.search(document)
    assert verdict is not None
    assert (verdict.group("text"), verdict.group("modifier")) == (text, modifier)
    assert '<p class="empty-state">' not in document
    assert "<table>" in document
