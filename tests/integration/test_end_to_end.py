"""The end-to-end proof: one command, five runs, one ceiling.

``pytest -m e2e`` is the whole of the end-to-end proof. In one command this module builds the corpus
and proves it has not drifted, consumes the five runs the session fixtures pay for -- ``reference``
and ``defective`` at EC M plus the 40-record boundary subset at L, Q and H -- and asserts, from the
artefacts those runs left on disk, that the sequence is five runs into five distinct report roots,
that each root carries a parseable ``summary.json``, ``junit.xml`` and ``index.html``, that
``reference`` failed nothing while ``defective`` failed exactly the nine checks `SELF_CHECK_TARGETS`
declares, that the five finished inside the 600 000 ms ceiling, that every failing case's evidence
and image are in the page, and that nothing was written into the repository.

It starts no run of its own, because a sixth ``run`` would break the budget this module
asserts. Check and decoder behaviour, the HTML contract and each command's exit code are held
elsewhere, and the one wall-clock assertion in this folder -- the ceiling -- lives here
because only the ``PF-*`` checks and this gate may read a clock."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from xml.etree import ElementTree as ET

import pytest

# The single declaration of what each self-check target must produce, so the non-vacuity proof
# has one clause.
from qrcheck.registry.expectations import SELF_CHECK_TARGETS

# The run sequence is the dominant cost, so the marker covers the module (pyproject registers it).
pytestmark = pytest.mark.e2e

# The seed the committed corpus was generated from, also the top-level `seed` default.
_CORPUS_SEED: Final[int] = 20240101

# The committed corpus relative to the root: a repository file, not package data.
_CORPUS_RELATIVE: Final[tuple[str, ...]] = ("corpus", "default.jsonl")

# The directory runtime artefacts must never land in, and the three report files.
_REPORTS_DIR: Final[str] = "reports"
_SUMMARY_NAME: Final[str] = "summary.json"
_JUNIT_NAME: Final[str] = "junit.xml"
_HTML_NAME: Final[str] = "index.html"

# The nine top-level keys of `summary.json`, as the schema documents them.
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

# The four tallies, separate integers so a `skipped` can never fold into `passed`.
_COUNT_KEYS: Final[frozenset[str]] = frozenset({"passed", "failed", "skipped", "error"})

# The statuses that fail the gate; a `skipped` is deliberately not one, being counted separately.
_FAILED_OR_ERROR: Final[frozenset[str]] = frozenset({"failed", "error"})

# The one child element each status writes in `junit.xml`, and none for a pass.
_STATUS_ELEMENT: Final[Mapping[str, str | None]] = MappingProxyType(
    {"error": "error", "failed": "failure", "passed": None, "skipped": "skipped"}
)

# The nine check identifiers a `--adapter defective` run must fail, as a literal.
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

# The two self-check targets, the fixtures' level and the three boundary levels.
_CONFORMING_TARGET: Final[str] = "reference"
_DEFECTIVE_TARGET: Final[str] = "defective"
_DEFAULT_EC_LEVEL: Final[str] = "M"
_BOUNDARY_LEVELS: Final[tuple[str, ...]] = ("L", "Q", "H")

# Exit codes the sequence must produce: clean is 0, failures 1, an internal error 3.
_EXIT_OK: Final[int] = 0
_EXIT_FAILED: Final[int] = 1

# The ceiling on the five-run sequence in milliseconds.
_RUN_CEILING_MS: Final[float] = 600_000.0

# A `run_id` is a sha256 content hash, never a UUID or a timestamp: 64 lowercase hex.
_RUN_ID: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")

# The two evidence files every failing case carries, and the data-URI prefix of an image.
_EVIDENCE_FILES: Final[tuple[str, str]] = ("request.json", "decoded.json")
_DATA_URI_PREFIX: Final[str] = "data:image/"

# An `src` or `href` attribute and its value. Only attribute values are scanned for a remote origin,
# because a corpus URL payload would otherwise match as a remote reference.
_RESOURCE_ATTRIBUTE: Final[re.Pattern[str]] = re.compile(
    r"""\b(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE
)

# The value prefixes that would make the report depend on a remote origin.
_FOREIGN_ORIGINS: Final[tuple[str, ...]] = ("http:", "https:", "//")


@dataclass(frozen=True, slots=True)
class SuiteRun:
    """One run of the suite as this folder's session fixtures produced it: label, root, summary,
    status."""

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
        """Return the run's results in the summary's own ``(check_id, case_id)`` order."""
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
    """What the repository held before the runs: the corpus's size and sha256, and the paths under
    ``reports/``."""

    corpus_size: int
    corpus_sha256: str
    reports_entries: tuple[str, ...]


def _reports_entries(reports: Path) -> tuple[str, ...]:
    """Return every path under ``reports`` relative to it, sorted, or ``()`` when absent."""
    if not reports.is_dir():
        return ()
    return tuple(sorted(str(path.relative_to(reports)) for path in reports.rglob("*")))


@pytest.fixture(scope="session")
def repository_baseline(repo_root: Path) -> RepositoryBaseline:
    """Snapshot the repository state the proof must leave untouched, taken before the runs."""
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
    """Return the five runs of this tier, keyed by label and iterated in sorted key order; a missing
    boundary level or a mismatched adapter fails the test."""
    del repository_baseline  # consumed for its ordering; the comparison happens in a test
    missing = sorted(set(_BOUNDARY_LEVELS) - set(boundary_runs))
    if missing:
        pytest.fail(
            f"the boundary fixture produced no run for level(s) {missing}; the subset must be "
            f"replayed at {list(_BOUNDARY_LEVELS)}"
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
    """Return a diagnostic naming where two corpus renderings first differ."""
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
    """Read one run's ``summary.json`` from disk as a mapping; ``AssertionError`` when it is
    missing or is not a JSON object."""
    path = report_root / _SUMMARY_NAME
    assert path.is_file(), f"{path} is missing: the run wrote no {_SUMMARY_NAME}"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), (
        f"{path} holds a {type(document).__name__}, not the JSON object the schema documents"
    )
    return document


def _foreign_resource_values(document: str) -> list[str]:
    """Return every ``src``/``href`` value that would make the report depend on a network."""
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
    """A regeneration from the committed seed is byte-identical to the committed corpus -- the
    drift check the CI job runs, as a test, written under ``tmp_path``."""
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
    """The tier is five runs, each into its own root under the temporary base, so no run
    overwrites another's evidence."""
    expected_labels = {_CONFORMING_TARGET, _DEFECTIVE_TARGET} | {
        f"boundary-{level}" for level in _BOUNDARY_LEVELS
    }
    assert set(run_sequence) == expected_labels, (
        f"the sequence is not the five runs this tier pays for: expected "
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
        f"run's evidence and its failures would stop being attributable"
    )
    assert len(distinct) == len(roots), (
        f"the sequence produced {len(distinct)} distinct report roots for {len(roots)} runs"
    )
    for label in sorted(roots):
        root = roots[label]
        assert root.is_dir(), f"the {label} run's report root {root} is not a directory"
        assert root.is_relative_to(base), (
            f"the {label} run wrote into {root}, which is outside the session temporary base "
            f"{base}; runtime artefacts are not committed deliverables"
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
    """Every run leaves a parseable ``summary.json``,
    ``junit.xml`` and ``index.html``, with the JUnit tallies agreeing with the summary and one
    ``testcase`` per result carrying exactly that result's status element."""
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

        results = summary["results"]
        testcases = [child for child in suite if child.tag == "testcase"]
        assert len(testcases) == len(results), (
            f"the {label} run's {_JUNIT_NAME} carries {len(testcases)} testcase element(s) "
            f"for the {len(results)} result(s) {_SUMMARY_NAME} records; the aggregate "
            f"attributes are a tally the writer computed itself and cannot stand in for one "
            f"element per result"
        )

        statuses = [str(result["status"]) for result in results]
        unknown = sorted(set(statuses) - set(_STATUS_ELEMENT))
        assert unknown == [], (
            f"the {label} run's {_SUMMARY_NAME} reports the undocumented status(es) {unknown}; "
            f"a case reaches exactly one of {sorted(_STATUS_ELEMENT)}"
        )
        expected_children: Counter[str] = Counter(
            element
            for element in (_STATUS_ELEMENT[status] for status in statuses)
            if element is not None
        )
        observed_children: Counter[str] = Counter(
            child.tag for testcase in testcases for child in testcase
        )
        assert observed_children == expected_children, (
            f"the {label} run's {_JUNIT_NAME} carries the status child elements "
            f"{dict(sorted(observed_children.items()))} while the {len(results)} results of "
            f"{_SUMMARY_NAME} imply {dict(sorted(expected_children.items()))}; a dropped "
            f"element and an invented one both break the documented JUnit shape"
        )
        crowded = sorted(
            testcase.get("name") or "<unnamed>" for testcase in testcases if len(testcase) > 1
        )
        assert crowded == [], (
            f"the {label} run's {_JUNIT_NAME} gives {crowded} more than one status child "
            f"element; a case has exactly one outcome"
        )

        html_path = root / _HTML_NAME
        assert html_path.is_file(), f"the {label} run wrote no {_HTML_NAME} in {root}"
        page = html_path.read_text(encoding="utf-8")
        assert "<script" not in page.lower(), (
            f"the {label} run's {_HTML_NAME} carries a script element, so the report is not "
            f"static"
        )
        foreign = _foreign_resource_values(page)
        assert foreign == [], (
            f"the {label} run's {_HTML_NAME} references {foreign}, so it would not open "
            f"offline"
        )


def test_expected_sets_hold_across_the_sequence(run_sequence: Mapping[str, SuiteRun]) -> None:
    """``reference`` fails nothing and ``defective`` fails exactly the
    nine declared checks, as set **equality**, with no ``error`` status, the documented exit
    codes, and at least one result in every run -- since a run that reported nothing would
    satisfy an empty expectation."""
    assert SELF_CHECK_TARGETS[_DEFECTIVE_TARGET] == _EXPECTED_DEFECTIVE_IDS, (
        f"the product's declared defective set {sorted(SELF_CHECK_TARGETS[_DEFECTIVE_TARGET])} "
        f"differs from the nine identifiers this module expects: {sorted(_EXPECTED_DEFECTIVE_IDS)}"
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
        assert run.results, (
            f"{_describe(run)} reported no results at all, so comparing its failing set "
            f"against the expected {sorted(expected)} would hold on a run that produced "
            f"nothing and this gate would then prove nothing; a run of the sequence must "
            f"report at least one result"
        )
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
            f"defect, never a verdict on the target"
        )
        expected_exit = _EXIT_FAILED if label == _DEFECTIVE_TARGET else _EXIT_OK
        assert run.exit_code == expected_exit, (
            f"{_describe(run)} exited {run.exit_code}, expected {expected_exit}: exit 0 is a "
            f"run with no failure and no error and exit 1 is a run with at least one failure"
        )



def test_total_run_duration_is_under_the_six_hundred_second_ceiling(
    run_sequence: Mapping[str, SuiteRun],
) -> None:
    """The five runs finished inside the 600 000 ms
    ceiling -- the one wall-clock assertion in this folder."""
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
        f"ceiling. Per run:\n" + "\n".join(breakdown) + "\n"
        f"That ceiling is stated on CPython 3.11, Linux x86_64, 2 vCPU with the pinned lock, "
        f"so a materially slower machine is one explanation and a run fixture "
        f"starting twice is the other."
    )


def test_each_failing_case_is_visible_in_the_html(run_sequence: Mapping[str, SuiteRun]) -> None:
    """Every failing case of the defective run has evidence on disk and an
    image in the page, so the report is self-contained on real output."""
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
            f"re-derived from the report"
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
        f"report"
    )


def test_run_ids_identify_the_run_by_content(run_sequence: Mapping[str, SuiteRun]) -> None:
    """Each ``run_id`` is a sha256 content hash and the five are distinct, so
    neither a random value nor a timestamp could pass, and the adapter is hashed content."""
    identifiers = {
        label: str(run_sequence[label].summary["run_id"]) for label in sorted(run_sequence)
    }
    for label in sorted(identifiers):
        assert _RUN_ID.fullmatch(identifiers[label]), (
            f"the {label} run's run_id {identifiers[label]!r} is not a 64-character lowercase "
            f"hex digest, so it is not the content hash a run_id must be"
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
    """The whole tier left the repository as it found it: no report root under
    ``reports/`` and no regenerated corpus over the committed file."""
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
        f"removed {removed}; runtime artefacts stay under the temporary base"
    )
