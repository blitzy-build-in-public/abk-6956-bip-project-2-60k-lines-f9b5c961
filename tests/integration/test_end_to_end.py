"""The end-to-end proof: one command, five runs, one ceiling (AAP §0.6.1 #9).

``pytest -m e2e`` is the whole of success criterion 9: in one command this module builds the
corpus and proves it has not drifted from the committed file, consumes the five runs the session
fixtures pay for (``reference`` and ``defective`` at EC M plus the 40-record boundary subset at
L, Q and H), and asserts from the artefacts on disk that the sequence is five runs into five
distinct report roots, that each root carries a parseable ``summary.json``, ``junit.xml`` and
``index.html``, that ``reference`` failed nothing and ``defective`` exactly the nine checks of
AAP §0.4.2 with every boundary run clean, that the five finished inside the 600 000 ms ceiling,
that each failing case's evidence is on disk and its image visible in the HTML report, that each
``run_id`` is a content hash rather than a clock, and that nothing was written into the
repository.

It starts no run of its own: the run fixtures are session-scoped, so the tier costs the five
runs the ceiling is stated against -- about 3,700 encodes and 4,100 decodes -- and a sixth
``run`` here would break the budget this module asserts. Check and decoder behaviour, the HTML
escaping and per-failure-image contract and each command's own exit code belong to
``tests/unit/`` and to the sibling integration modules; the one wall-clock assertion in the
folder lives here, because AAP §0.5.5 permits only the ``PF-*`` checks and this gate to read a
clock.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from xml.etree import ElementTree as ET

import pytest

# The single declaration of what each built-in self-check target must produce. Reading it
# rather than restating the sets is what makes this module's clause of criterion 1 the same
# clause the unit and integration layers hold; the nine identifiers are additionally spelled
# out below as a literal, so a change to that table cannot pass unnoticed here either.
from qrcheck.registry.expectations import SELF_CHECK_TARGETS

# The dominant cost of this module is the run sequence its fixtures pay for, so the marker
# covers the module rather than an individual test. `pyproject.toml` owns its registration
# and the `-m "not e2e"` deselection the fast unit job relies on.
pytestmark = pytest.mark.e2e

# The seed the committed corpus was generated from, which is also the top-level `seed`
# default of AAP §0.3.4. The drift assertion is only meaningful against that value.
_CORPUS_SEED: Final[int] = 20240101

# The committed corpus, relative to the repository root (§0.9.2: a repository file rather
# than package data).
_CORPUS_RELATIVE: Final[tuple[str, ...]] = ("corpus", "default.jsonl")

# The repository directory runtime artefacts must never land in (§0.11.1), and the three
# report files every root must carry.
_REPORTS_DIR: Final[str] = "reports"
_SUMMARY_NAME: Final[str] = "summary.json"
_JUNIT_NAME: Final[str] = "junit.xml"
_HTML_NAME: Final[str] = "index.html"

# The nine top-level keys of `summary.json`, in the order §0.3.4 documents them. The set is a
# stable public surface that CI and tooling key on, so a writer that drops, renames or adds
# one fails here rather than in a consumer's parser.
_SUMMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "adapter",
        "counts",
        "ec_level",
        "finished_at",
        "loc_count",
        "results",
        "run_id",
        "seed",
        "started_at",
    }
)

# The four tallies a run records. They are separate integers by design, which is what keeps a
# `skipped` result from ever being folded into `passed` (§0.3.4).
_COUNT_KEYS: Final[frozenset[str]] = frozenset({"passed", "failed", "skipped", "error"})

# The statuses that make a run's check fail the gate. A `skipped` result is deliberately not
# one of them here: the counts are compared separately, and the reference gate's zero-skip
# requirement is held by `tests/integration/test_reference_run.py`.
_FAILED_OR_ERROR: Final[frozenset[str]] = frozenset({"failed", "error"})

# The nine check identifiers a `--adapter defective` run must fail, listed as a literal so
# this module's expectation does not merely agree with the product's own table by
# construction (§0.4.2). It is asserted equal to `SELF_CHECK_TARGETS["defective"]` below.
_EXPECTED_DEFECTIVE_IDS: Final[frozenset[str]] = frozenset(
    {
        "DT-DETERMINISTIC",
        "IV-OVERCAPACITY",
        "MD-CHARSET",
        "RN-MODULE-SIZE",
        "RN-QUIET-ZONE",
        "RT-ROUNDTRIP",
        "SY-EC-MATCH",
        "SY-MATRIX-EQUIV",
        "SY-VERSION-MIN",
    }
)

# The two built-in self-check targets, the level the whole-suite fixtures run at (the `run`
# command's own default) and the three levels success criterion 2 replays the boundary subset
# at (§0.6.1 #2).
_CONFORMING_TARGET: Final[str] = "reference"
_DEFECTIVE_TARGET: Final[str] = "defective"
_DEFAULT_EC_LEVEL: Final[str] = "M"
_BOUNDARY_LEVELS: Final[tuple[str, ...]] = ("L", "Q", "H")

# Exit codes the sequence must produce (§0.3.4): a clean run is 0 and a run with failures is
# 1. A run that errored internally would be 3, which the expected-set assertion reports as an
# `error` status rather than as a wrong code.
_EXIT_OK: Final[int] = 0
_EXIT_FAILED: Final[int] = 1

# The ceiling on the five-run sequence, in milliseconds (§0.6.1 #9, §0.6.2). The e2e job's
# 30-minute `timeout-minutes` is the hard backstop; this is the measured gate that produces a
# named failure instead of a job kill (§0.6.4).
_RUN_CEILING_MS: Final[float] = 600_000.0

# A `run_id` is a sha256 content hash over the run's parameters, never a UUID or a timestamp
# (§0.3.4), so it is exactly 64 lowercase hex characters.
_RUN_ID: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")

# The two evidence files every failing case carries (§0.3.4), and the data-URI prefix an
# embedded evidence image uses in the HTML report.
_EVIDENCE_FILES: Final[tuple[str, str]] = ("request.json", "decoded.json")
_DATA_URI_PREFIX: Final[str] = "data:image/"

# An `src` or `href` attribute and its value, single- or double-quoted. Only attribute values
# are scanned for a foreign origin: a corpus URL payload puts the literal text "https://"
# into the escaped report table, and a substring search would report that as a remote
# reference when it is nothing of the kind.
_RESOURCE_ATTRIBUTE: Final[re.Pattern[str]] = re.compile(
    r"""\b(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE
)

# The value prefixes that would make the report depend on a remote origin and so stop it
# opening offline (§0.3.4, §0.8.4).
_FOREIGN_ORIGINS: Final[tuple[str, ...]] = ("http:", "https:", "//")


@dataclass(frozen=True, slots=True)
class SuiteRun:
    """One run of the suite, exactly as this folder's session fixtures produced it.

    Attributes:
        label: The run's stable name -- ``reference``, ``defective`` or ``boundary-<level>``
            -- used in every failure message and as the key of :func:`run_sequence`.
        report_root: The run's report directory, under the session temporary base.
        summary: Its parsed ``summary.json``, read by the fixture before the test began.
        exit_code: The status the real ``python -m qrcheck`` process returned.
    """

    label: str
    report_root: Path
    summary: Mapping[str, Any]
    exit_code: int

    @property
    def adapter(self) -> str:
        """Return the adapter the run's own summary names."""
        return str(self.summary["adapter"])

    @property
    def ec_level(self) -> str:
        """Return the error-correction level the run's own summary names."""
        return str(self.summary["ec_level"])

    @property
    def results(self) -> tuple[Mapping[str, Any], ...]:
        """Return the run's results, in the order the summary stores them.

        That order is the result model's own ``(check_id, case_id)`` order, so a test that
        walks this tuple walks the artefact's documented order instead of imposing a second
        ordering rule on the same run (AAP §0.5.3).
        """
        return tuple(self.summary["results"])

    @property
    def failing_check_ids(self) -> frozenset[str]:
        """Return the identifiers of the checks this run failed or errored in."""
        return frozenset(
            str(result["check_id"])
            for result in self.results
            if result["status"] in _FAILED_OR_ERROR
        )


@dataclass(frozen=True, slots=True)
class RepositoryBaseline:
    """What the repository held before the run sequence, so a write into it is detectable.

    Attributes:
        corpus_size: The committed corpus's size in bytes.
        corpus_sha256: Its sha256, in lowercase hex.
        reports_entries: Every path under the repository's ``reports/`` directory, relative
            to it and sorted; empty when that directory does not exist, which is the state of
            a clean checkout.
    """

    corpus_size: int
    corpus_sha256: str
    reports_entries: tuple[str, ...]


def _reports_entries(reports: Path) -> tuple[str, ...]:
    """Return every path under ``reports`` relative to it, sorted, or ``()`` when absent.

    Args:
        reports: The repository's ``reports/`` directory, which may not exist.

    Returns:
        The sorted relative paths of its entries, so two snapshots compare as one tuple and a
        directory listing order can never reach an assertion.
    """
    if not reports.is_dir():
        return ()
    return tuple(sorted(str(path.relative_to(reports)) for path in reports.rglob("*")))


@pytest.fixture(scope="session")
def repository_baseline(repo_root: Path) -> RepositoryBaseline:
    """Snapshot the repository state the proof must leave untouched.

    Session-scoped and requested first by :func:`run_sequence`, so it is taken **before** any
    of the five runs and its comparison in
    :func:`test_nothing_was_written_into_the_repository` covers the whole tier.

    Args:
        repo_root: The repository root the paths are resolved against.

    Returns:
        The corpus's size and digest and the ``reports/`` tree's entries.

    Raises:
        pytest.fail: The committed corpus is missing, so the drift assertion this module also
            holds has nothing to compare against. A skip here would hide the gate.
    """
    corpus = repo_root.joinpath(*_CORPUS_RELATIVE)
    if not corpus.is_file():
        pytest.fail(
            f"the committed corpus is missing: {corpus}; the end-to-end proof compares a "
            f"regeneration against it and cannot run without it"
        )
    data = corpus.read_bytes()
    return RepositoryBaseline(
        corpus_size=len(data),
        corpus_sha256=hashlib.sha256(data).hexdigest(),
        reports_entries=_reports_entries(repo_root / _REPORTS_DIR),
    )


@pytest.fixture(scope="session")
def run_sequence(
    repository_baseline: RepositoryBaseline,
    reference_report: tuple[Path, dict[str, Any], int],
    defective_report: tuple[Path, dict[str, Any], int],
    boundary_runs: Mapping[str, tuple[Path, dict[str, Any], int]],
) -> Mapping[str, SuiteRun]:
    """Return the five runs of this tier, keyed by label and iterated in sorted key order.

    The three run fixtures it consumes are the whole run cost of the tier, so they are
    consumed once and shared: this module starts nothing itself. ``repository_baseline`` is
    declared first because a fixture's dependencies are set up in the order they are
    declared, which is what makes the baseline precede the runs it guards.

    Args:
        repository_baseline: The repository snapshot taken before the runs.
        reference_report: ``(report_root, summary, exit_code)`` for ``--adapter reference``.
        defective_report: The same for ``--adapter defective``.
        boundary_runs: The same, per level, for the boundary subset at L, Q and H.

    Returns:
        A read-only mapping from label to :class:`SuiteRun`.

    Raises:
        pytest.fail: A boundary level is missing from the fixture, or a run's own summary
            names a different adapter or level than the one requested. Either would make the
            expected-set assertion compare a run against another run's expectation.
    """
    del repository_baseline  # consumed for its ordering; the comparison happens in a test
    missing = sorted(set(_BOUNDARY_LEVELS) - set(boundary_runs))
    if missing:
        pytest.fail(
            f"the boundary fixture produced no run for level(s) {missing}; success criterion "
            f"2 replays the boundary subset at {list(_BOUNDARY_LEVELS)}"
        )

    requested: tuple[tuple[str, str, str, tuple[Path, dict[str, Any], int]], ...] = (
        (_CONFORMING_TARGET, _CONFORMING_TARGET, _DEFAULT_EC_LEVEL, reference_report),
        (_DEFECTIVE_TARGET, _DEFECTIVE_TARGET, _DEFAULT_EC_LEVEL, defective_report),
        *(
            (f"boundary-{level}", _CONFORMING_TARGET, level, boundary_runs[level])
            for level in _BOUNDARY_LEVELS
        ),
    )

    runs: dict[str, SuiteRun] = {}
    for label, adapter, level, record in sorted(requested):
        report_root, summary, exit_code = record
        run = SuiteRun(
            label=label, report_root=Path(report_root), summary=summary, exit_code=exit_code
        )
        if run.adapter != adapter or run.ec_level != level:
            pytest.fail(
                f"the {label} run's summary reports adapter={run.adapter!r} "
                f"ec_level={run.ec_level!r}, but it was requested as adapter={adapter!r} "
                f"ec_level={level!r}; the expected-set assertion would then compare one run "
                f"against another run's expectation"
            )
        runs[label] = run
    return MappingProxyType(runs)


def _first_divergence(regenerated: bytes, committed: bytes) -> str:
    """Return a diagnostic naming where two corpus renderings first differ.

    Args:
        regenerated: The bytes ``corpus build`` produced.
        committed: The bytes the repository holds.

    Returns:
        A sentence naming the first differing offset with both sides' context, or the two
        lengths when one rendering is a prefix of the other.
    """
    common = min(len(regenerated), len(committed))
    for offset in range(common):
        if regenerated[offset] != committed[offset]:
            return (
                f"the regenerated corpus first differs from the committed one at byte "
                f"{offset}: regenerated {regenerated[offset : offset + 16]!r}, committed "
                f"{committed[offset : offset + 16]!r}"
            )
    return (
        f"the regenerated corpus is {len(regenerated)} bytes and the committed one is "
        f"{len(committed)}; they agree on all {common} bytes and one is a prefix of the other"
    )


def _load_summary_artefact(report_root: Path) -> Mapping[str, Any]:
    """Read one run's ``summary.json`` from disk and return it as a mapping.

    Args:
        report_root: The run's report directory.

    Returns:
        The parsed document, read from the artefact rather than from a fixture, which is what
        proves the file a consumer receives is the one the schema describes.

    Raises:
        AssertionError: The file is missing, or does not hold a JSON object.
    """
    path = report_root / _SUMMARY_NAME
    assert path.is_file(), f"{path} is missing: the run wrote no {_SUMMARY_NAME}"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), (
        f"{path} holds a {type(document).__name__}, not the JSON object the schema documents"
    )
    return document


def _foreign_resource_values(document: str) -> list[str]:
    """Return every ``src``/``href`` value that would make the report depend on a network.

    Args:
        document: The HTML report's text.

    Returns:
        The offending attribute values in document order; empty when the report is
        self-contained.
    """
    values: list[str] = []
    for match in _RESOURCE_ATTRIBUTE.finditer(document):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        if value.strip().lower().startswith(_FOREIGN_ORIGINS):
            values.append(value)
    return values


def _describe(run: SuiteRun) -> str:
    """Return the identification a failure message needs: label, adapter and level."""
    return f"{run.label} (adapter={run.adapter} ec={run.ec_level})"


def test_corpus_build_and_drift_inside_the_proof(
    tmp_path: Path,
    write_config: Callable[..., Path],
    run_cli: Callable[..., Any],
    repo_root: Path,
) -> None:
    """A regeneration from the committed seed is byte-identical to the committed corpus.

    The CI drift step of AAP §0.6.4 expressed as a test, so the criterion-9 command covers it
    rather than leaving it to a workflow step. Catches a corpus builder whose output drifts
    from the file the runs then read: every boundary record would silently become wrong while
    the suite kept passing against the stale copy.

    The regeneration is written under ``tmp_path`` and the committed file is only ever read,
    so a drifted build fails here instead of overwriting the deliverable it disagrees with.
    """
    committed_path = repo_root.joinpath(*_CORPUS_RELATIVE)
    committed = committed_path.read_bytes()

    build_dir = tmp_path / "corpus-build"
    build_dir.mkdir()
    regenerated_path = build_dir / "corpus.jsonl"
    report_root = tmp_path / "reports-corpus-build"
    report_root.mkdir()
    config_path = write_config(
        tmp_path / "config" / "qrcheck.toml",
        default_adapter=_CONFORMING_TARGET,
        report_dir=report_root,
    )

    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(_CORPUS_SEED),
        "--out",
        str(regenerated_path),
        config=config_path,
    )
    assert result.exit_code == _EXIT_OK, (
        f"corpus build --seed {_CORPUS_SEED} exited {result.exit_code}, expected "
        f"{_EXIT_OK}; stderr head: {result.stderr.strip()[:2000]!r}"
    )
    assert regenerated_path.is_file(), (
        f"corpus build exited {result.exit_code} but wrote no {regenerated_path}; "
        f"stdout head: {result.stdout.strip()[:2000]!r}"
    )

    regenerated = regenerated_path.read_bytes()
    assert regenerated == committed, _first_divergence(regenerated, committed)
    assert committed_path.read_bytes() == committed, (
        f"the drift check wrote the committed corpus {committed_path}; a drift comparison "
        f"must only ever read it"
    )


def test_the_sequence_is_exactly_five_runs_into_five_distinct_roots(
    run_sequence: Mapping[str, SuiteRun],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The tier is five runs, each into its own report root under the temporary base.

    Catches two defects at once. A shared root would let the second run overwrite the first
    run's ``evidence/`` tree, which is laid out under one ``report_dir``, and the defective
    run's nine expected failures would stop being attributable (AAP IM-8). A root inside the
    repository would put a volatile artefact into version control, which §0.11.1 forbids.
    """
    expected_labels = {_CONFORMING_TARGET, _DEFECTIVE_TARGET} | {
        f"boundary-{level}" for level in _BOUNDARY_LEVELS
    }
    assert set(run_sequence) == expected_labels, (
        f"the sequence is not the five runs of AAP §0.6.1 #9: expected "
        f"{sorted(expected_labels)}, got {sorted(run_sequence)}"
    )

    base = Path(tmp_path_factory.getbasetemp()).resolve()
    roots = {label: run_sequence[label].report_root.resolve() for label in sorted(run_sequence)}
    distinct = sorted(set(roots.values()))
    shared = sorted(
        str(path)
        for path in distinct
        if sum(1 for candidate in roots.values() if candidate == path) > 1
    )
    assert shared == [], (
        f"the runs share report root(s) {shared}, so a later run would overwrite an earlier "
        f"run's evidence and its failures would stop being attributable (AAP IM-8)"
    )
    assert len(distinct) == len(roots), (
        f"the sequence produced {len(distinct)} distinct report roots for {len(roots)} runs"
    )
    for label in sorted(roots):
        root = roots[label]
        assert root.is_dir(), f"the {label} run's report root {root} is not a directory"
        assert root.is_relative_to(base), (
            f"the {label} run wrote into {root}, which is outside the session temporary base "
            f"{base}; runtime artefacts are not committed deliverables (AAP §0.11.1)"
        )
    for label in sorted(roots):
        for other in sorted(roots):
            if label == other:
                continue
            assert not roots[other].is_relative_to(roots[label]), (
                f"the {other} run's report root {roots[other]} is nested inside the {label} "
                f"run's {roots[label]}, so one run's evidence tree sits inside another's"
            )


def test_every_root_carries_the_three_artefacts_and_they_parse(
    run_sequence: Mapping[str, SuiteRun],
) -> None:
    """Every run leaves a parseable ``summary.json``, ``junit.xml`` and ``index.html``.

    Catches a writer that emits an artefact the schema does not describe, a JUnit document
    whose tallies disagree with the summary, and a report that stopped being self-contained.
    The HTML half is deliberately narrow: escaping and the per-failure-image contract belong
    to ``tests/unit/test_html_writer.py``, and restating them here would only duplicate it.
    """
    for label in sorted(run_sequence):
        run = run_sequence[label]
        root = run.report_root

        summary = _load_summary_artefact(root)
        assert set(summary) == _SUMMARY_KEYS, (
            f"the {label} run's {_SUMMARY_NAME} does not carry exactly the nine documented "
            f"top-level keys: missing {sorted(_SUMMARY_KEYS - set(summary))}, unexpected "
            f"{sorted(set(summary) - _SUMMARY_KEYS)}"
        )

        counts = summary["counts"]
        assert isinstance(counts, dict), (
            f"the {label} run's counts is a {type(counts).__name__}, not an object"
        )
        assert set(counts) == _COUNT_KEYS, (
            f"the {label} run's counts do not carry the four separate tallies: got "
            f"{sorted(counts)}"
        )

        junit_path = root / _JUNIT_NAME
        assert junit_path.is_file(), f"the {label} run wrote no {_JUNIT_NAME} in {root}"
        document = ET.parse(junit_path).getroot()
        assert document.tag == "testsuites", (
            f"the {label} run's {_JUNIT_NAME} has root element {document.tag!r}, not "
            f"'testsuites'"
        )
        suites = [child for child in document if child.tag == "testsuite"]
        assert len(suites) == 1, (
            f"the {label} run's {_JUNIT_NAME} carries {len(suites)} testsuite elements, and "
            f"the product writes exactly one"
        )
        suite = suites[0]
        assert suite.get("name") == "qrcheck", (
            f"the {label} run's testsuite is named {suite.get('name')!r}, not 'qrcheck'"
        )
        aggregate = {
            "tests": len(summary["results"]),
            "failures": counts["failed"],
            "errors": counts["error"],
            "skipped": counts["skipped"],
        }
        for attribute, expected in sorted(aggregate.items()):
            assert suite.get(attribute) == str(expected), (
                f"the {label} run's JUnit {attribute!r} is {suite.get(attribute)!r} while "
                f"{_SUMMARY_NAME} says {expected}; the two artefacts describe one run and "
                f"cannot disagree"
            )

        html_path = root / _HTML_NAME
        assert html_path.is_file(), f"the {label} run wrote no {_HTML_NAME} in {root}"
        page = html_path.read_text(encoding="utf-8")
        assert "<script" not in page.lower(), (
            f"the {label} run's {_HTML_NAME} carries a script element, so the report is not "
            f"static (AAP §0.8.4)"
        )
        foreign = _foreign_resource_values(page)
        assert foreign == [], (
            f"the {label} run's {_HTML_NAME} references {foreign}, so it would not open "
            f"offline (AAP §0.3.4)"
        )


def test_expected_sets_hold_across_the_sequence(run_sequence: Mapping[str, SuiteRun]) -> None:
    """``reference`` fails nothing and ``defective`` fails exactly the nine declared checks.

    The single-command form of success criterion 1, asserted as set **equality** rather than
    containment, so a tenth failure and a missing one both fail here (AAP §0.4.2). It also
    refuses any ``error`` status and checks the documented exit code of each run, because a
    run that errored internally or returned the wrong status is not the run this gate claims.
    """
    assert SELF_CHECK_TARGETS[_DEFECTIVE_TARGET] == _EXPECTED_DEFECTIVE_IDS, (
        f"the product's declared defective set {sorted(SELF_CHECK_TARGETS[_DEFECTIVE_TARGET])} "
        f"differs from the nine identifiers AAP §0.4.2 fixes: {sorted(_EXPECTED_DEFECTIVE_IDS)}"
    )
    assert SELF_CHECK_TARGETS[_CONFORMING_TARGET] == frozenset(), (
        f"the conforming target is declared to fail "
        f"{sorted(SELF_CHECK_TARGETS[_CONFORMING_TARGET])}, which is not conforming"
    )

    expectations: dict[str, frozenset[str]] = {
        _CONFORMING_TARGET: SELF_CHECK_TARGETS[_CONFORMING_TARGET],
        _DEFECTIVE_TARGET: SELF_CHECK_TARGETS[_DEFECTIVE_TARGET],
    }
    for level in _BOUNDARY_LEVELS:
        expectations[f"boundary-{level}"] = frozenset()

    for label in sorted(run_sequence):
        run = run_sequence[label]
        expected = expectations[label]
        actual = run.failing_check_ids
        assert actual == expected, (
            f"{_describe(run)} failed or errored {sorted(actual)}, expected "
            f"{sorted(expected)}; missing {sorted(expected - actual)}, unexpected "
            f"{sorted(actual - expected)}"
        )
        errored = sorted(
            str(result["check_id"]) for result in run.results if result["status"] == "error"
        )
        assert errored == [], (
            f"{_describe(run)} reported error results for {errored}; an error is a framework "
            f"defect, never a verdict on the target (AAP §0.3.4)"
        )
        expected_exit = _EXIT_FAILED if label == _DEFECTIVE_TARGET else _EXIT_OK
        assert run.exit_code == expected_exit, (
            f"{_describe(run)} exited {run.exit_code}, expected {expected_exit}: exit 0 is a "
            f"run with no failure and no error and exit 1 is a run with at least one failure "
            f"(AAP §0.3.4)"
        )



def test_total_run_duration_is_under_the_six_hundred_second_ceiling(
    run_sequence: Mapping[str, SuiteRun],
) -> None:
    """The five runs together finished inside the 600 000 ms ceiling of AAP §0.6.1 #9.

    The one legitimate wall-clock assertion in this folder: §0.5.5 permits only the ``PF-*``
    checks and this ceiling to read a clock, so nothing here compares runs to each other and
    no minimum duration is asserted. Catches a workload that has grown past its budget and a
    run fixture that started twice. The failure names the total and every run's own duration,
    because the ceiling is only meaningful against the environment §0.6.3 fixes.
    """
    total_ms = 0.0
    breakdown: list[str] = []
    for label in sorted(run_sequence):
        run = run_sequence[label]
        started = datetime.fromisoformat(str(run.summary["started_at"]))
        finished = datetime.fromisoformat(str(run.summary["finished_at"]))
        elapsed_ms = (finished - started).total_seconds() * 1000.0
        assert elapsed_ms >= 0.0, (
            f"{_describe(run)} records finished_at {finished.isoformat()} before started_at "
            f"{started.isoformat()}, so its recorded duration is negative"
        )
        total_ms += elapsed_ms
        breakdown.append(f"  {_describe(run)}: {elapsed_ms:.0f} ms")

    assert total_ms <= _RUN_CEILING_MS, (
        f"the five-run sequence took {total_ms:.0f} ms, over the {_RUN_CEILING_MS:.0f} ms "
        f"ceiling of AAP §0.6.1 #9. Per run:\n" + "\n".join(breakdown) + "\n"
        f"That ceiling is stated on CPython 3.11, Linux x86_64, 2 vCPU with the pinned lock "
        f"(AAP §0.6.3), so a materially slower machine is one explanation and a run fixture "
        f"starting twice is the other."
    )


def test_each_failing_case_is_visible_in_the_html(run_sequence: Mapping[str, SuiteRun]) -> None:
    """Every failing case of the defective run has evidence on disk and an image in the page.

    The substance of criterion 6's "missing an evidence image for a failing case", asserted on
    real output: a failure a reader cannot inspect is a failure that cannot be diagnosed. The
    count is compared rather than each image mapped to its case, because the escaping and
    per-case image contract is held by ``tests/unit/test_html_writer.py`` and parsing the
    document here would only restate it against one run.
    """
    run = run_sequence[_DEFECTIVE_TARGET]
    failing = sorted(
        (result for result in run.results if result["status"] == "failed"),
        key=lambda result: (str(result["check_id"]), str(result["case_id"])),
    )
    assert failing, (
        f"{_describe(run)} failed no case at all, so there is no failure evidence to check; "
        f"the expected-set assertion above names the checks that should have failed"
    )

    for result in failing:
        case = f"{result['check_id']}[{result['case_id']}]"
        evidence = result["evidence_dir"]
        assert evidence is not None, (
            f"the failing case {case} carries no evidence_dir, so its failure cannot be "
            f"re-derived from the report (AAP §0.6.1 #6)"
        )
        directory = run.report_root / str(evidence)
        assert directory.is_dir(), (
            f"the failing case {case} records evidence_dir {evidence!r}, but {directory} does "
            f"not exist"
        )
        for name in _EVIDENCE_FILES:
            assert (directory / name).is_file(), (
                f"{directory} holds no {name}, so the failing case {case} cannot be "
                f"re-derived from its own evidence"
            )

    page = (run.report_root / _HTML_NAME).read_text(encoding="utf-8")
    embedded = page.count(_DATA_URI_PREFIX)
    assert embedded >= len(failing), (
        f"the {_HTML_NAME} of {_describe(run)} embeds {embedded} image(s) for "
        f"{len(failing)} failing case(s); every failing case's symbol must be visible in the "
        f"report (AAP §0.8.4)"
    )


def test_run_ids_identify_the_run_by_content(run_sequence: Mapping[str, SuiteRun]) -> None:
    """Each ``run_id`` is a sha256 content hash, and the five of them are distinct.

    Catches a ``run_id`` that silently became a random value or a timestamp: either would
    still be 64 hex characters, but neither would repeat for the same workload, and the
    stability §0.3.4 promises to tooling would be gone. Two runs of one seed and level that
    differ only in their adapter must still differ here, which is what proves the adapter is
    part of the hashed content rather than a label applied beside it.
    """
    identifiers = {
        label: str(run_sequence[label].summary["run_id"]) for label in sorted(run_sequence)
    }
    for label in sorted(identifiers):
        assert _RUN_ID.fullmatch(identifiers[label]), (
            f"the {label} run's run_id {identifiers[label]!r} is not a 64-character lowercase "
            f"hex digest, so it is not the content hash AAP §0.3.4 specifies"
        )
    assert len(set(identifiers.values())) == len(identifiers), (
        f"two runs of the sequence share one run_id: "
        f"{ {label: identifiers[label] for label in sorted(identifiers)} }"
    )

    conforming = run_sequence[_CONFORMING_TARGET]
    defective = run_sequence[_DEFECTIVE_TARGET]
    assert conforming.summary["seed"] == defective.summary["seed"], (
        f"the two whole-suite runs were asked for different seeds "
        f"({conforming.summary['seed']!r} and {defective.summary['seed']!r}), so a difference "
        f"in their run_id would prove nothing"
    )
    assert conforming.ec_level == defective.ec_level, (
        f"the two whole-suite runs were asked for different levels "
        f"({conforming.ec_level!r} and {defective.ec_level!r})"
    )
    assert identifiers[_CONFORMING_TARGET] != identifiers[_DEFECTIVE_TARGET], (
        f"the reference and defective runs share run_id "
        f"{identifiers[_CONFORMING_TARGET]!r} although they ran against different adapters, "
        f"so the adapter is not part of the hashed content"
    )


def test_nothing_was_written_into_the_repository(
    run_sequence: Mapping[str, SuiteRun],
    repo_root: Path,
    repository_baseline: RepositoryBaseline,
) -> None:
    """The whole tier left the repository exactly as it found it.

    Catches the two ways a run can put a runtime artefact into version control: writing a
    report root under ``reports/`` instead of the temporary base, and regenerating the corpus
    over the committed file. The ``reports/`` comparison is taken against the snapshot
    captured before the runs, so on a clean checkout -- where that directory does not exist
    and the snapshot is empty -- it is exactly the assertion that the tree is empty.
    """
    assert len(run_sequence) == 5, (
        f"this comparison must run after the five runs, but the sequence holds "
        f"{len(run_sequence)}"
    )

    corpus = repo_root.joinpath(*_CORPUS_RELATIVE)
    data = corpus.read_bytes()
    assert len(data) == repository_baseline.corpus_size, (
        f"the committed corpus changed size during the tier: {repository_baseline.corpus_size} "
        f"bytes before, {len(data)} now"
    )
    assert hashlib.sha256(data).hexdigest() == repository_baseline.corpus_sha256, (
        f"the committed corpus's content changed during the tier; its sha256 was "
        f"{repository_baseline.corpus_sha256} and is now {hashlib.sha256(data).hexdigest()}"
    )

    after = _reports_entries(repo_root / _REPORTS_DIR)
    added = sorted(set(after) - set(repository_baseline.reports_entries))
    removed = sorted(set(repository_baseline.reports_entries) - set(after))
    assert added == [] and removed == [], (
        f"the repository's {_REPORTS_DIR}/ tree changed during the tier: added {added}, "
        f"removed {removed}; runtime artefacts stay under the temporary base (AAP §0.11.1)"
    )

