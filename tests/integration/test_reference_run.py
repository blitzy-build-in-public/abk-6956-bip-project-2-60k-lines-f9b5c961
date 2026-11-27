"""The reference run's gates: a conforming target must come back completely clean.

This module asserts the acceptance evidence a whole run must leave behind, over the committed
240-record corpus at EC M plus the 40-record boundary subset replayed at L, Q and H: a clean
run, all 21 checks executed and passing, the workload equal to the case matrix, round-trip over
every committed and boundary record, the configured performance floors, and the three report
artefacts against their documented schemas.

Each run comes from the session-scoped fixtures in ``tests/integration/conftest.py``, so it is
paid for once in the session's five full runs; no wall-clock quantity is compared, every
collection is walked in sorted order, and no threshold is written in this file."""

from __future__ import annotations

import datetime
import re
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

# Imported for its side effect only: ``qrcheck.checks`` registers the nine families in ``REGISTRY``,
# without which the registry is empty in a fresh interpreter.
import qrcheck.checks
from qrcheck.corpus.io import filter_by_class, load_corpus
from qrcheck.registry.check import REGISTRY

# A full 240-record run is the dominant cost, so the ``e2e`` marker confines it to the CI job.
pytestmark = pytest.mark.e2e


# The contract constants: each value is a fact the report is asserted against, never a threshold --
# the identifiers and counts are the registry's and the documented schema's, the seed and EC level
# the committed configuration's.

#: The 21 check identifiers the registry declares, verbatim.
CHECK_IDS: Final[frozenset[str]] = frozenset(
    {
        "RT-ROUNDTRIP",
        "SY-VERSION-MIN",
        "SY-EC-MATCH",
        "SY-MASK",
        "SY-MATRIX-EQUIV",
        "MD-MODE-OPTIMAL",
        "MD-CHARSET",
        "RN-QUIET-ZONE",
        "RN-MODULE-SIZE",
        "RN-CONTRAST",
        "RN-SVG",
        "RN-LOGO",
        "DM-RESILIENCE",
        "DM-SEEDED",
        "IV-EMPTY",
        "IV-OVERCAPACITY",
        "IV-BAD-OPTION",
        "DT-DETERMINISTIC",
        "PF-ENCODE",
        "PF-DECODE",
        "CL-ARTIFACTS",
    }
)

#: The case matrix: the cases each check runs in one full run at EC M over 240 records.
CASE_MATRIX: Final[Mapping[str, int]] = {
    "RT-ROUNDTRIP": 240,
    "MD-MODE-OPTIMAL": 60,
    "MD-CHARSET": 74,
    "SY-VERSION-MIN": 240,
    "SY-EC-MATCH": 240,
    "SY-MASK": 240,
    "SY-MATRIX-EQUIV": 8,
    "RN-QUIET-ZONE": 12,
    "RN-MODULE-SIZE": 12,
    "RN-CONTRAST": 12,
    "RN-SVG": 12,
    "RN-LOGO": 6,
    "DM-RESILIENCE": 56,
    "DM-SEEDED": 56,
    "IV-EMPTY": 1,
    "IV-OVERCAPACITY": 1,
    "IV-BAD-OPTION": 4,
    "DT-DETERMINISTIC": 6,
    "PF-ENCODE": 1,
    "PF-DECODE": 1,
    "CL-ARTIFACTS": 1,
}

#: The total the matrix comes to, asserted separately from the rows.
DECLARED_CASES: Final[int] = 1_283

#: The nine top-level fields of ``summary.json``, asserted as an exact set.
SUMMARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "started_at",
        "finished_at",
        "adapter",
        "ec_level",
        "seed",
        "loc_count",
        "counts",
        "results",
    }
)

#: The ten fields of every result object, for the same reason.
RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "check_id",
        "case_id",
        "status",
        "reason",
        "duration_ms",
        "request",
        "decoded",
        "expected",
        "evidence_dir",
        "seed",
    }
)

#: The four tallies, four separate integers so a skip can never fold into a pass.
COUNT_KEYS: Final[frozenset[str]] = frozenset({"passed", "failed", "skipped", "error"})

#: The adapter the reference fixture runs, which must be the conforming built-in target.
REFERENCE_ADAPTER: Final[str] = "reference"

#: The EC level the whole-corpus fixture runs at, and the seed the configuration declares.
RUN_EC_LEVEL: Final[str] = "M"
RUN_SEED: Final[int] = 20240101

#: The 40-record boundary subset the boundary fixtures replay at L, Q and H.
BOUNDARY_LEVELS: Final[tuple[str, ...]] = ("L", "Q", "H")
BOUNDARY_CLASS_GLOB: Final[str] = "boundary-*"
BOUNDARY_RECORDS: Final[int] = 40
CORPUS_RECORDS: Final[int] = 240

#: The committed corpus's location relative to the root, and the configuration's name.
CORPUS_PATH: Final[tuple[str, ...]] = ("corpus", "default.jsonl")
CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"

#: The prefix a corpus-driven case identifier carries: ``corpus:<record-id>``.
CORPUS_CASE_PREFIX: Final[str] = "corpus:"

#: The three report artefacts a run writes, in the order the summary names them.
REPORT_ARTEFACTS: Final[tuple[str, ...]] = ("summary.json", "junit.xml", "index.html")

#: The two volatile attributes ``junit.xml`` omits; pytest's own writer emits both.
VOLATILE_XML_ATTRIBUTES: Final[frozenset[str]] = frozenset({"hostname", "timestamp"})

#: How a reference to an external origin begins; a self-contained report may carry none of them.
EXTERNAL_REFERENCE_PREFIXES: Final[tuple[str, ...]] = ("http:", "https:", "//")

#: The platform descriptor directories ``CL-ARTIFACTS`` enumerates: absolute, but a fixed
#: operating-system interface, so they are the documented exemption from the scan below.
PLATFORM_FD_DIRECTORIES: Final[frozenset[str]] = frozenset({"/proc/self/fd", "/dev/fd"})

#: The shape of ``run_id``: the SHA-256 hex digest the runner computes over what the run was.
SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

#: Matches a ``src`` or ``href`` attribute of the HTML report, in either quoting style.
_REFERENCE_ATTRIBUTE: Final[re.Pattern[str]] = re.compile(
    r"""(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)


# Reading the artefacts: every helper returns a value or fails with a diagnostic naming what it saw.


def _results(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return a run's result objects, in the ``(check_id, case_id)`` order the schema fixes."""
    results = summary.get("results")
    assert isinstance(results, list), (
        f"summary.json carries no 'results' list, but a {type(results).__name__}; a run "
        f"always reports its cases"
    )
    return results


def _results_of(summary: Mapping[str, Any], check_id: str) -> list[Mapping[str, Any]]:
    """Return one check's results ordered by case identifier, empty when it did not execute."""
    return sorted(
        (result for result in _results(summary) if result.get("check_id") == check_id),
        key=lambda result: str(result.get("case_id")),
    )


def _single(summary: Mapping[str, Any], check_id: str) -> Mapping[str, Any]:
    """Return the one result ``check_id`` produced in a full run."""
    results = _results_of(summary, check_id)
    assert len(results) == 1, (
        f"{check_id} must report exactly 1 case in a full run per the case matrix, but the run "
        f"reported {len(results)}: {[str(r.get('case_id')) for r in results]}"
    )
    return results[0]


def _assert_all_passed(results: Sequence[Mapping[str, Any]], what: str) -> None:
    """Assert every result is ``passed``, naming any that is not."""
    offenders = sorted(
        f"{result.get('case_id')!r} status={result.get('status')!r} "
        f"reason={result.get('reason')!r}"
        for result in results
        if result.get("status") != "passed"
    )
    assert not offenders, (
        f"{what}: {len(offenders)} of {len(results)} case(s) are not passed against the "
        f"conforming target, which must come back clean; first offenders: {offenders[:10]}"
    )


def _corpus(repo_root: Path) -> list[Any]:
    """Return the committed corpus's records, in the file's own ``id``-ascending order."""
    path = repo_root.joinpath(*CORPUS_PATH)
    assert path.is_file(), (
        f"the committed corpus is missing at {path}; the run's 240 cases are asserted "
        f"against this file rather than against the report's own case list"
    )
    return load_corpus(path)


def _committed_config(repo_root: Path) -> Mapping[str, Any]:
    """Return the committed ``qrcheck.toml``, the single source of every threshold."""
    path = repo_root / CONFIG_FILE_NAME
    assert path.is_file(), (
        f"the committed configuration is missing at {path}; the thresholds a run reports in "
        f"each result's 'expected' are asserted against this document"
    )
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _string_values(node: Any, path: str) -> Iterator[tuple[str, str]]:
    """Yield every string value of a JSON document with the field path it sits at, ordered."""
    if isinstance(node, Mapping):
        for key in sorted(node, key=str):
            yield from _string_values(node[key], f"{path}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _string_values(item, f"{path}[{index}]")
    elif isinstance(node, str):
        yield path, node


def _absolute_path_offenders(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return every field whose string value names an absolute path, sorted; an artefact carries
    none except ``CL-ARTIFACTS``'s platform descriptor directory, the documented exemption."""
    return sorted(
        (path, value)
        for path, value in _string_values(summary, "$")
        if value.startswith("/") and value not in PLATFORM_FD_DIRECTORIES
    )


def _reference_attributes(document: str) -> list[str]:
    """Return every ``src`` and ``href`` value of an HTML document, in document order."""
    return [
        double or single
        for double, single in _REFERENCE_ATTRIBUTE.findall(document)
    ]


# The gates: one test per condition, in the order the report is read -- clean run, executed checks,
# declared case counts, then the case-level proofs and the three report contracts.


def test_reference_run_is_clean(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a suite that cannot come back clean: any failed, errored or skipped case."""
    report_root, summary, exit_code = reference_report
    counts = summary["counts"]
    results = _results(summary)

    assert exit_code == 0, (
        f"the reference run exited {exit_code} instead of 0; a conforming target must produce "
        f"neither a failed nor an errored case (report root {report_root})"
    )
    assert counts["failed"] == 0, (
        f"{counts['failed']} case(s) failed against the conforming target, which must not "
        f"happen; failing checks: "
        f"{sorted({str(result['check_id']) for result in results if result['status'] == 'failed'})}"
    )
    assert counts["error"] == 0, (
        f"{counts['error']} case(s) errored inside the framework, which is a framework defect "
        f"and never a verdict on the target; errored checks: "
        f"{sorted({str(result['check_id']) for result in results if result['status'] == 'error'})}"
    )
    skipped_checks = sorted(
        {str(result["check_id"]) for result in results if result["status"] == "skipped"}
    )
    assert counts["skipped"] == 0, (
        f"{counts['skipped']} case(s) were skipped, but this gate requires zero skips: a skip "
        f"would let a check pass CI without doing its work, as a skipped RN-SVG would rasterize "
        f"nothing (install the SVG rasterizer rather than accept the skip). Skipped checks: "
        f"{skipped_checks}"
    )
    assert sum(counts.values()) == len(results), (
        f"the four tallies sum to {sum(counts.values())} but the run reported {len(results)} "
        f"results, so an outcome was folded into another bucket"
    )
    assert summary["adapter"] == REFERENCE_ADAPTER, (
        f"the run tested adapter {summary['adapter']!r}, not {REFERENCE_ADAPTER!r}; every case "
        f"in this module asserts the conforming target's behaviour"
    )
    assert summary["ec_level"] == RUN_EC_LEVEL, (
        f"the run used EC level {summary['ec_level']!r} rather than {RUN_EC_LEVEL!r}, so the "
        f"240-record whole-corpus round-trip run was not the one measured"
    )
    assert summary["seed"] == RUN_SEED, (
        f"the run used seed {summary['seed']!r} rather than the committed default {RUN_SEED}, "
        f"so the report does not reproduce from the configuration it was started from"
    )
    run_id = summary["run_id"]
    assert isinstance(run_id, str) and SHA256_HEX.fullmatch(run_id) is not None, (
        f"run_id must be the content hash over the run's parameters -- not a per-run UUID and "
        f"not a reading of the wall clock -- but it is {run_id!r}"
    )


def test_every_check_registered_is_executed_and_passes(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a check that never executes, or that executes and never passes."""
    _report_root, summary, _exit_code = reference_report
    results = _results(summary)

    by_check: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_check.setdefault(str(result["check_id"]), []).append(result)

    assert set(by_check) == set(CHECK_IDS), (
        f"the run executed {sorted(set(by_check))}, which is not the 21 registered checks; "
        f"missing: {sorted(set(CHECK_IDS) - set(by_check))}, "
        f"unexpected: {sorted(set(by_check) - set(CHECK_IDS))}"
    )
    assert set(REGISTRY) == set(CHECK_IDS), (
        f"the registry holds {sorted(set(REGISTRY))} rather than the 21 declared identifiers; a "
        f"check reachable only from a test would leave it short"
    )
    without_pass = sorted(
        check_id
        for check_id, check_results in by_check.items()
        if not any(result["status"] == "passed" for result in check_results)
    )
    assert not without_pass, (
        f"{without_pass} executed but produced no passed result; every check must have at least "
        f"one, so presence alone can never count as a pass"
    )
    inconsistent = sorted(
        f"{result['check_id']}[{result['case_id']}] status={result['status']!r} "
        f"reason={result['reason']!r}"
        for result in results
        if (result["status"] == "passed") != (result["reason"] is None)
    )
    assert not inconsistent, (
        f"{len(inconsistent)} result(s) pair a status with the wrong reason: a passing case must "
        f"carry no reason and every other status exactly one; first offenders: {inconsistent[:10]}"
    )


def test_case_matrix_matches_the_declared_workload(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a re-balanced workload: any check running a number of cases the matrix denies.

    The declared 1,287 encode invocations and 1,462 primary decodes -- a decode column that
    counts only the damaged reads, so with ``DM-RESILIENCE``'s eight memoised pristine reads the
    declared workload is 1,470 decodes -- are **not** fields of ``summary.json``, and inventing a
    counter to make them observable would change the product to suit the test."""
    _report_root, summary, _exit_code = reference_report
    observed = Counter(str(result["check_id"]) for result in _results(summary))
    mismatched = {
        check_id: (CASE_MATRIX.get(check_id), observed.get(check_id, 0))
        for check_id in sorted(set(CASE_MATRIX) | set(observed))
        if CASE_MATRIX.get(check_id) != observed.get(check_id)
    }
    assert not mismatched, (
        f"{len(mismatched)} check(s) ran a number of cases the case matrix does not declare, "
        f"as (declared, observed): {mismatched}"
    )
    assert sum(observed.values()) == DECLARED_CASES, (
        f"the full run reported {sum(observed.values())} cases, not the declared "
        f"{DECLARED_CASES}: {dict(sorted(observed.items()))}"
    )


def test_roundtrip_covers_all_240_corpus_records(
    reference_report: tuple[Path, dict[str, Any], int],
    repo_root: Path,
) -> None:
    """Catch a round-trip gate that tests fewer records, or a different corpus, than it claims."""
    _report_root, summary, _exit_code = reference_report
    results = _results_of(summary, "RT-ROUNDTRIP")

    assert len(results) == CORPUS_RECORDS, (
        f"RT-ROUNDTRIP ran {len(results)} cases, but the committed corpus holds "
        f"{CORPUS_RECORDS} records the whole-corpus round-trip gate requires"
    )
    _assert_all_passed(results, "RT-ROUNDTRIP")
    case_ids = [str(result["case_id"]) for result in results]
    assert len(set(case_ids)) == CORPUS_RECORDS, (
        f"RT-ROUNDTRIP's {len(case_ids)} case identifiers are only {len(set(case_ids))} "
        f"distinct, so at least one corpus record was measured twice"
    )
    expected_case_ids = [
        f"{CORPUS_CASE_PREFIX}{record.id}"
        for record in sorted(_corpus(repo_root), key=lambda record: record.id)
    ]
    assert case_ids == expected_case_ids, (
        f"RT-ROUNDTRIP's case identifiers do not match the committed corpus record for record; "
        f"missing: {sorted(set(expected_case_ids) - set(case_ids))[:10]}, "
        f"unexpected: {sorted(set(case_ids) - set(expected_case_ids))[:10]}"
    )


def test_boundary_subset_passes_at_every_ec_level(
    boundary_runs: Mapping[str, tuple[Path, dict[str, Any], int]],
    repo_root: Path,
) -> None:
    """Catch a boundary subset that is not 40 records, or that breaks at L, Q or H."""
    assert set(boundary_runs) == set(BOUNDARY_LEVELS), (
        f"the boundary fixtures cover {sorted(boundary_runs)} rather than "
        f"{list(BOUNDARY_LEVELS)}; the boundary-subset gate runs the subset at every level"
    )
    boundary_records = filter_by_class(_corpus(repo_root), [BOUNDARY_CLASS_GLOB])
    assert len(boundary_records) == BOUNDARY_RECORDS, (
        f"the {BOUNDARY_CLASS_GLOB!r} class filter selects {len(boundary_records)} records, not "
        f"the {BOUNDARY_RECORDS} the boundary-subset gate names"
    )
    expected_case_ids = [
        f"{CORPUS_CASE_PREFIX}{record.id}"
        for record in sorted(boundary_records, key=lambda record: record.id)
    ]

    for level in BOUNDARY_LEVELS:
        report_root, summary, exit_code = boundary_runs[level]
        counts = summary["counts"]
        assert exit_code == 0, (
            f"the boundary run at EC {level} exited {exit_code} instead of 0; the conforming "
            f"target must be clean at every error-correction level (report root {report_root})"
        )
        assert counts["failed"] == 0, (
            f"{counts['failed']} boundary case(s) failed at EC {level}; failing checks: "
            f"{sorted({str(r['check_id']) for r in _results(summary) if r['status'] == 'failed'})}"
        )
        assert counts["error"] == 0, (
            f"{counts['error']} boundary case(s) errored at EC {level}; errored checks: "
            f"{sorted({str(r['check_id']) for r in _results(summary) if r['status'] == 'error'})}"
        )
        assert summary["ec_level"] == level, (
            f"the boundary run for {level} reports EC level {summary['ec_level']!r}, so the "
            f"report describes a different level than the one measured"
        )
        for check_id in ("RT-ROUNDTRIP", "SY-VERSION-MIN", "SY-EC-MATCH"):
            results = _results_of(summary, check_id)
            assert len(results) == BOUNDARY_RECORDS, (
                f"{check_id} ran {len(results)} boundary cases at EC {level}, not "
                f"{BOUNDARY_RECORDS}: the class filter applies to every corpus-driven check"
            )
            _assert_all_passed(results, f"{check_id} at EC {level}")
        roundtrip_case_ids = [
            str(result["case_id"]) for result in _results_of(summary, "RT-ROUNDTRIP")
        ]
        assert roundtrip_case_ids == expected_case_ids, (
            f"the boundary run at EC {level} round-tripped a different record set than "
            f"{BOUNDARY_CLASS_GLOB!r} selects; missing: "
            f"{sorted(set(expected_case_ids) - set(roundtrip_case_ids))[:10]}, unexpected: "
            f"{sorted(set(roundtrip_case_ids) - set(expected_case_ids))[:10]}"
        )
        corpus_requests = [
            result
            for result in _results(summary)
            if str(result["case_id"]).startswith(CORPUS_CASE_PREFIX)
        ]
        assert corpus_requests, (
            f"the boundary run at EC {level} reported no corpus-driven case at all, so the "
            f"--class filter selected nothing and nothing was asserted"
        )
        wrong_level = sorted(
            {
                str(result["case_id"])
                for result in corpus_requests
                if result["request"].get("ec_level") != level
            }
        )
        assert not wrong_level, (
            f"{len(wrong_level)} corpus-driven case(s) at EC {level} sent a request at another "
            f"level, so --ec did not reach the target; first offenders: {wrong_level[:10]}"
        )


def test_performance_floors_are_asserted_by_the_product_from_one_source(
    reference_report: tuple[Path, dict[str, Any], int],
    repo_root: Path,
) -> None:
    """Catch a performance floor the product reported from somewhere other than the config."""
    _report_root, summary, _exit_code = reference_report
    limits = _committed_config(repo_root)["checks"]["performance"]
    encode = _single(summary, "PF-ENCODE")
    decode = _single(summary, "PF-DECODE")

    assert encode["status"] == "passed", (
        f"PF-ENCODE is {encode['status']!r} ({encode['reason']!r}), so an encode floor was "
        f"breached or the measurement never completed: {encode['decoded'].get('message')!r}"
    )
    assert decode["status"] == "passed", (
        f"PF-DECODE is {decode['status']!r} ({decode['reason']!r}), so a decode floor was "
        f"breached or the probe image did not decode: {decode['decoded'].get('message')!r}"
    )
    assert encode["decoded"]["metrics"]["samples"] == limits["encode_samples"], (
        f"PF-ENCODE measured {encode['decoded']['metrics']['samples']} encodes, not the "
        f"configured {limits['encode_samples']}; a retried or dropped sample would show here"
    )
    assert decode["decoded"]["metrics"]["samples"] == limits["decode_samples"], (
        f"PF-DECODE measured {decode['decoded']['metrics']['samples']} decodes, not the "
        f"configured {limits['decode_samples']}; a retried or dropped sample would show here"
    )

    # The three floors ``qrcheck.toml`` declares by key, asserted one by one so an empty mapping
    # cannot pass.
    for key in ("encode_p95_ms", "min_symbols_per_minute"):
        assert key in encode["expected"], (
            f"PF-ENCODE's expected block does not report {key!r}, so the budget it was measured "
            f"against is not in the report: {sorted(encode['expected'])}"
        )
        assert encode["expected"][key] == limits[key], (
            f"PF-ENCODE measured against {key}={encode['expected'][key]!r}, but {CONFIG_FILE_NAME} "
            f"declares {limits[key]!r}: the floor did not come from the single declaration"
        )
    assert "decode_p95_ms" in decode["expected"], (
        f"PF-DECODE's expected block does not report 'decode_p95_ms', so the budget it was "
        f"measured against is not in the report: {sorted(decode['expected'])}"
    )
    assert decode["expected"]["decode_p95_ms"] == limits["decode_p95_ms"], (
        f"PF-DECODE measured against decode_p95_ms={decode['expected']['decode_p95_ms']!r}, but "
        f"{CONFIG_FILE_NAME} declares {limits['decode_p95_ms']!r}: the floor did not come from "
        f"the single declaration"
    )

    # Every other budget the case reported is configured too: none of them may be restated.
    for result in (encode, decode):
        restated = {
            key: (value, limits.get(key))
            for key, value in sorted(result["expected"].items())
            if limits.get(key) != value
        }
        assert not restated, (
            f"{result['check_id']}'s expected block reports values {CONFIG_FILE_NAME} does not "
            f"declare, as key: (reported, configured): {restated}"
        )

    measured = encode["decoded"]["metrics"]
    for key in ("encode_p95_ms", "symbols_per_minute"):
        assert isinstance(measured.get(key), int | float), (
            f"PF-ENCODE's metrics do not report a number for {key!r}: {measured.get(key)!r}; the "
            f"measurement is the point of the case"
        )
    assert isinstance(decode["decoded"]["metrics"].get("decode_p95_ms"), int | float), (
        f"PF-DECODE's metrics do not report a number for 'decode_p95_ms': "
        f"{decode['decoded']['metrics'].get('decode_p95_ms')!r}"
    )


def test_no_evidence_is_written_for_passing_cases(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a run that files evidence for what passed, or records an absolute path."""
    report_root, summary, _exit_code = reference_report
    results = _results(summary)

    with_evidence = sorted(
        f"{result['check_id']}[{result['case_id']}] -> {result['evidence_dir']!r}"
        for result in results
        if result["evidence_dir"] is not None
    )
    assert not with_evidence, (
        f"{len(with_evidence)} passing case(s) recorded an evidence directory, but evidence is "
        f"written only for failing, errored and skipped cases; first offenders: "
        f"{with_evidence[:10]}"
    )
    evidence_root = report_root / "evidence"
    if evidence_root.exists():
        populated = sorted(
            str(path.relative_to(report_root)) for path in evidence_root.rglob("*")
        )
        assert not populated, (
            f"the clean run created {len(populated)} entr(ies) under evidence/, which the "
            f"persistence policy reserves for failures: {populated[:10]}"
        )
    offenders = _absolute_path_offenders(summary)
    assert not offenders, (
        f"{len(offenders)} field(s) of summary.json hold an absolute path, which no artefact may "
        f"carry (evidence paths are relative to report_dir and requests carry paths as "
        f"configured), as field: value: {offenders[:10]}"
    )


def test_the_three_report_artefacts_exist_and_parse(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a missing or malformed ``summary.json``, ``junit.xml`` or ``index.html``."""
    report_root, summary, _exit_code = reference_report
    counts = summary["counts"]
    results = _results(summary)

    for name in REPORT_ARTEFACTS:
        assert (report_root / name).is_file(), (
            f"the run wrote no {name} under {report_root}; a run owes its CI, tooling and human "
            f"consumers all three artefacts"
        )

    root = ET.parse(report_root / "junit.xml").getroot()
    assert root.tag == "testsuites", (
        f"junit.xml's root element is {root.tag!r}, not the 'testsuites' the xunit2 shape fixes"
    )
    suites = list(root)
    assert [suite.tag for suite in suites] == ["testsuite"], (
        f"junit.xml holds {[suite.tag for suite in suites]}, not exactly one 'testsuite'"
    )
    suite = suites[0]
    assert suite.get("name") == "qrcheck", (
        f"the suite is named {suite.get('name')!r}, not 'qrcheck', which is the identity every "
        f"consumer keys on"
    )
    expected_attributes = {
        "tests": len(results),
        "failures": counts["failed"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
    }
    for attribute, count in expected_attributes.items():
        assert suite.get(attribute) == str(count), (
            f"junit.xml's suite reports {attribute}={suite.get(attribute)!r} but summary.json "
            f"yields {count}: the two views disagree about the same run"
        )
    testcases = list(suite)
    assert len(testcases) == len(results), (
        f"junit.xml carries {len(testcases)} testcase(s) for {len(results)} results; every result "
        f"needs one so a CI trend line can key on it"
    )
    volatile = sorted(
        {
            name
            for element in root.iter()
            for name in element.attrib
            if name in VOLATILE_XML_ATTRIBUTES
        }
    )
    assert not volatile, (
        f"junit.xml carries {volatile}, which the writer omits deliberately so the file is "
        f"reproducible; pytest's own writer emits both and this one must not"
    )

    document = (report_root / "index.html").read_text(encoding="utf-8")
    assert "<script" not in document.lower(), (
        "index.html carries a <script> element; the report must open with no script at all"
    )
    external = sorted(
        {
            value
            for value in _reference_attributes(document)
            if value.lower().startswith(EXTERNAL_REFERENCE_PREFIXES)
        }
    )
    assert not external, (
        f"index.html references {external}, so it would need the network to render; every image "
        f"is a data: URI by construction"
    )


def test_summary_schema_is_exact(
    reference_report: tuple[Path, dict[str, Any], int],
    repo_root: Path,
) -> None:
    """Catch a summary that drifted off its schema, lost its ordering, or breached the ceiling."""
    _report_root, summary, _exit_code = reference_report

    assert set(summary) == set(SUMMARY_KEYS), (
        f"summary.json's top level is {sorted(summary)}, not the nine documented fields "
        f"{sorted(SUMMARY_KEYS)}; missing: {sorted(set(SUMMARY_KEYS) - set(summary))}, "
        f"unexpected: {sorted(set(summary) - set(SUMMARY_KEYS))}"
    )
    counts = summary["counts"]
    assert set(counts) == set(COUNT_KEYS), (
        f"summary.json's counts object is {sorted(counts)}, not the four separate tallies "
        f"{sorted(COUNT_KEYS)}; a folded tally would hide a skip"
    )
    for name in sorted(COUNT_KEYS):
        assert type(counts[name]) is int and counts[name] >= 0, (
            f"counts[{name!r}] is {counts[name]!r}, not a non-negative integer; a string or a "
            f"float would make a consumer's arithmetic on the tallies wrong"
        )

    results = _results(summary)
    for result in results:
        assert set(result) == set(RESULT_KEYS), (
            f"the result for {result.get('check_id')}[{result.get('case_id')}] carries "
            f"{sorted(result)} instead of the ten documented fields; missing: "
            f"{sorted(set(RESULT_KEYS) - set(result))}, unexpected: "
            f"{sorted(set(result) - set(RESULT_KEYS))}"
        )
    pairs = [(str(result["check_id"]), str(result["case_id"])) for result in results]
    ordered_pairs = sorted(pairs)
    first_disorder = next(
        (
            index
            for index, (pair, ordered) in enumerate(zip(pairs, ordered_pairs, strict=True))
            if pair != ordered
        ),
        None,
    )
    assert first_disorder is None, (
        f"summary.json's results are not ordered by (check_id, case_id); the first pair out of "
        f"order is at index {first_disorder}: {pairs[first_disorder]!r} where "
        f"{ordered_pairs[first_disorder]!r} belongs"
    )
    assert len(set(pairs)) == len(pairs), (
        f"{len(pairs)} results carry only {len(set(pairs))} distinct (check_id, case_id) pairs, "
        f"so two rows would share one failure identifier and one evidence directory"
    )

    loc_limit = int(_committed_config(repo_root)["loc_limit"])
    loc_count = summary["loc_count"]
    assert type(loc_count) is int and loc_count > 0, (
        f"loc_count is {loc_count!r}, not the positive integer the line budget is measured as"
    )
    assert loc_count <= loc_limit, (
        f"the run measured {loc_count} tracked physical lines against a ceiling of {loc_limit} "
        f"from {CONFIG_FILE_NAME}; the line budget is breached"
    )

    for field in ("started_at", "finished_at"):
        timestamp = summary[field]
        assert isinstance(timestamp, str), (
            f"{field} is {timestamp!r}, not the ISO-8601 text the schema fixes"
        )
        parsed = datetime.datetime.fromisoformat(timestamp)
        assert parsed.utcoffset() == datetime.timedelta(0), (
            f"{field} is {timestamp!r}, which is not a UTC timestamp; the report writes explicit "
            f"UTC so no reader has to guess a zone"
        )

