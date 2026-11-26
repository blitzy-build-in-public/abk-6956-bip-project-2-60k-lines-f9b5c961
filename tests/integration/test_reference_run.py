"""The reference run's gates: a conforming target must come back completely clean.

Holds the acceptance evidence AAP §0.4.1 and §0.6.1 rows #1, #2, #3, #5 and #10 require from a
whole run against the built-in ``reference`` target over the committed 240-record corpus at EC
level M, plus the boundary subset #2 replays at L, Q and H. Each condition below fails on a
vacuous suite: the run is clean, with ``failed``, ``error`` and ``skipped`` all zero; all 21
registered checks executed and passed something, so none is reachable from a test alone (#8);
the observable workload equals the case matrix check for check, 1,283 cases; round-trip holds
for all 240 records with the identifiers tied back to the corpus file (#2); the same round-trip,
version and EC assertions hold for the 40 boundary records at L, Q and H; and the performance
floors are the configured ones, read from ``qrcheck.toml`` by the check and reported in
``expected`` so nothing is restated (#10, §0.6.2). The report contracts are held here too:
``summary.json`` is exactly the documented schema in the documented order, ``junit.xml`` is the
``xunit2`` document carrying no hostname or timestamp, ``index.html`` cites no external origin,
and a clean run writes no evidence (#6).

Both runs come from the session-scoped ``reference_report`` and ``boundary_runs`` fixtures in
``tests/integration/conftest.py``, because #9 allows the tier five runs inside 600 s and a
second reference run "to compare" would break it; nothing here starts a run, calls the command
line or writes a report. No wall-clock quantity is compared -- ``duration_ms`` is recorded and
never asserted, and the run timestamps are checked for shape only (§0.5.5, §0.12) -- and every
collection is walked in sorted order or through the ordering the schema fixes, so a failure
message names the same case on every machine.
"""

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

# Imported for its side effect only: importing ``qrcheck.checks`` imports the nine family
# modules, and each one registers its checks in ``qrcheck.registry.check.REGISTRY``. Without
# this the registry is empty in a fresh interpreter and the reachability half of
# ``test_every_check_registered_is_executed_and_passes`` would compare a run's executed set
# against nothing (AAP §0.4.1, §0.6.1 #8).
import qrcheck.checks  # noqa: F401  (registration side effect)
from qrcheck.corpus.io import filter_by_class, load_corpus
from qrcheck.registry.check import REGISTRY

# The dominant cost of this module is a full 240-record run, which AAP §0.6.1 assigns to the
# ``e2e`` CI job rather than the fast unit job. One module-level marker therefore deselects
# all nine tests together under ``-m "not e2e"``; the marker itself is registered in
# ``pyproject.toml``, which is the single declaration of it.
pytestmark = pytest.mark.e2e


# The contract constants. Each value below is a fact the report is asserted against, never a
# threshold this file chose: the identifiers and the case counts are AAP §0.3.4's and
# §0.8.1's, the seed and the EC level are the committed configuration's, and the corpus size is
# the committed file's.

#: The 21 check identifiers of AAP §0.3.4, verbatim. A check cannot be dropped, renamed or
#: silently duplicated without this set disagreeing with the run.
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

#: AAP §0.8.1's case matrix: the observable number of cases each check runs in one full run
#: at EC M over the default 240-record corpus. A re-balanced matrix fails loudly here rather
#: than quietly changing what "a full run" means for every other gate.
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

#: The total the matrix comes to. AAP §0.8.1 states it beside the row-by-row counts, so it is
#: asserted separately: a matrix that still summed to this while a row moved would be a
#: coincidence, and a sum that moved would be a re-balanced workload.
DECLARED_CASES: Final[int] = 1_283

#: The nine top-level fields of ``summary.json`` (AAP §0.3.4). Unknown keys are additive for
#: consumers, but a *documented* field that disappeared or was renamed breaks every one of
#: them, so the exact set is asserted rather than a subset.
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

#: The ten fields of every result object (AAP §0.3.4), for the same reason.
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

#: The four tallies, kept as four separate integers so a skip can never be folded into a pass
#: (AAP §0.3.4).
COUNT_KEYS: Final[frozenset[str]] = frozenset({"passed", "failed", "skipped", "error"})

#: The adapter the reference fixture runs, which must be the conforming built-in target.
REFERENCE_ADAPTER: Final[str] = "reference"

#: The error-correction level the whole-corpus fixture runs at, and the seed the committed
#: configuration declares. Both are copied from ``qrcheck.toml``'s defaults by the fixture, so
#: they are the values a reader of the report would see there too.
RUN_EC_LEVEL: Final[str] = "M"
RUN_SEED: Final[int] = 20240101

#: The 40-record boundary subset success criterion 2 replays at the three remaining levels
#: (AAP §0.6.1 #2), selected by a corpus class glob rather than by a hand-written id list.
BOUNDARY_LEVELS: Final[tuple[str, ...]] = ("L", "Q", "H")
BOUNDARY_CLASS_GLOB: Final[str] = "boundary-*"
BOUNDARY_RECORDS: Final[int] = 40
CORPUS_RECORDS: Final[int] = 240

#: The committed corpus's location relative to the repository root, as the configuration
#: names it (AAP §0.9.2), and the committed configuration's own file name.
CORPUS_PATH: Final[tuple[str, ...]] = ("corpus", "default.jsonl")
CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"

#: The prefix a corpus-driven case identifier carries (AAP §0.3.4: ``corpus:<record-id>``),
#: which is also what distinguishes a corpus-driven request from a synthetic one.
CORPUS_CASE_PREFIX: Final[str] = "corpus:"

#: The three report artefacts a run writes (AAP §0.3.4), in the order the summary names them.
REPORT_ARTEFACTS: Final[tuple[str, ...]] = ("summary.json", "junit.xml", "index.html")

#: The two attributes AAP §0.3.4 deliberately omits from ``junit.xml`` so the file is
#: reproducible; pytest's own writer emits both, which is why their absence is asserted.
VOLATILE_XML_ATTRIBUTES: Final[frozenset[str]] = frozenset({"hostname", "timestamp"})

#: How a reference to an external origin begins. A self-contained report opens with no
#: network access, so none of these may appear as a ``src`` or ``href`` value.
EXTERNAL_REFERENCE_PREFIXES: Final[tuple[str, ...]] = ("http:", "https:", "//")

#: The platform descriptor directories ``CL-ARTIFACTS`` records as the one it enumerated. They
#: are absolute, but they name a fixed operating-system interface rather than the clone or the
#: machine a run happened in, which is what AAP §0.5.5 forbids in an artefact -- so they are
#: the documented exemption from the absolute-path scan below and nothing else is.
PLATFORM_FD_DIRECTORIES: Final[frozenset[str]] = frozenset({"/proc/self/fd", "/dev/fd"})

#: The shape of ``run_id``: the SHA-256 hex digest the runner computes over what the run was.
#: Asserting the shape is what rejects a per-run UUID or a timestamp as a run's identity, both
#: of which AAP §0.3.4 forbids (and ``RunResult`` refuses).
SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

#: Matches a ``src`` or ``href`` attribute of the HTML report, in either quoting style.
_REFERENCE_ATTRIBUTE: Final[re.Pattern[str]] = re.compile(
    r"""(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)


# Reading the artefacts. Every helper below either returns a value or fails with a diagnostic
# naming what it saw; none skips, retries or repairs. A report that cannot be read is a defect
# in the product, and a test that tolerated it would be the vacuous gate this module exists to
# prevent (AAP §0.4.3).


def _results(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return a run's result objects, in the ``(check_id, case_id)`` order the schema fixes.

    Args:
        summary: The parsed ``summary.json``.

    Returns:
        The result objects, in the order the report stores them.
    """
    results = summary.get("results")
    assert isinstance(results, list), (
        f"summary.json carries no 'results' list, but a {type(results).__name__}; a run "
        f"always reports its cases (AAP §0.3.4)"
    )
    return results


def _results_of(summary: Mapping[str, Any], check_id: str) -> list[Mapping[str, Any]]:
    """Return one check's results, ordered by case identifier.

    Args:
        summary: The parsed ``summary.json``.
        check_id: The check identifier to select, such as ``RT-ROUNDTRIP``.

    Returns:
        That check's result objects; an empty list when the check did not execute, which the
        caller's own count assertion reports.
    """
    return sorted(
        (result for result in _results(summary) if result.get("check_id") == check_id),
        key=lambda result: str(result.get("case_id")),
    )


def _single(summary: Mapping[str, Any], check_id: str) -> Mapping[str, Any]:
    """Return the one result ``check_id`` produced in a full run.

    Args:
        summary: The parsed ``summary.json``.
        check_id: A check whose case matrix is a single constructed case, such as ``PF-ENCODE``.

    Returns:
        That check's only result object.
    """
    results = _results_of(summary, check_id)
    assert len(results) == 1, (
        f"{check_id} must report exactly 1 case in a full run per AAP §0.8.1's case matrix, "
        f"but the run reported {len(results)}: {[str(r.get('case_id')) for r in results]}"
    )
    return results[0]


def _assert_all_passed(results: Sequence[Mapping[str, Any]], what: str) -> None:
    """Assert every result in ``results`` is ``passed``, naming any that is not.

    Args:
        results: The result objects to check.
        what: What they are, for the diagnostic -- a check identifier or a check and level.
    """
    offenders = sorted(
        f"{result.get('case_id')!r} status={result.get('status')!r} "
        f"reason={result.get('reason')!r}"
        for result in results
        if result.get("status") != "passed"
    )
    assert not offenders, (
        f"{what}: {len(offenders)} of {len(results)} case(s) are not passed against the "
        f"conforming target, which AAP §0.4.1 requires to be clean; first offenders: "
        f"{offenders[:10]}"
    )


def _corpus(repo_root: Path) -> list[Any]:
    """Return the committed corpus's records, in the file's own ``id``-ascending order.

    Args:
        repo_root: The repository root the corpus is read from.

    Returns:
        Every record of ``corpus/default.jsonl``.
    """
    path = repo_root.joinpath(*CORPUS_PATH)
    assert path.is_file(), (
        f"the committed corpus is missing at {path}; the run's 240 cases are asserted "
        f"against this file rather than against the report's own case list"
    )
    return load_corpus(path)


def _committed_config(repo_root: Path) -> Mapping[str, Any]:
    """Return the committed ``qrcheck.toml`` parsed with the standard library's TOML reader.

    The configuration is the single source of every threshold (AAP §0.6.2), so this test
    module reads it instead of restating a number: a floor that changed in the document would
    otherwise have to be changed here too, which is exactly the drift the plan forbids.

    Args:
        repo_root: The repository root the document is read from.

    Returns:
        The parsed document.
    """
    path = repo_root / CONFIG_FILE_NAME
    assert path.is_file(), (
        f"the committed configuration is missing at {path}; the thresholds a run reports in "
        f"each result's 'expected' are asserted against this document"
    )
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _string_values(node: Any, path: str) -> Iterator[tuple[str, str]]:
    """Yield every string value of a JSON document with the field path it sits at.

    Args:
        node: The value to walk -- a mapping, a sequence or a scalar.
        path: The field path of ``node`` in the document, such as ``$.results[0].decoded``.

    Yields:
        ``(field_path, value)`` for every string leaf, ordered by key and by index so two runs
        report their offenders in the same order.
    """
    if isinstance(node, Mapping):
        for key in sorted(node, key=str):
            yield from _string_values(node[key], f"{path}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _string_values(item, f"{path}[{index}]")
    elif isinstance(node, str):
        yield path, node


def _absolute_path_offenders(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return every field whose string value names an absolute path.

    AAP §0.5.5 forbids an absolute path in an artefact: evidence paths are relative to
    ``report_dir`` and a request records a path as the configuration gave it. The one
    documented exception is the platform descriptor directory ``CL-ARTIFACTS`` names as the
    one it enumerated, which is a fixed operating-system interface and therefore names neither
    the clone nor the machine a run happened in.

    Args:
        summary: The parsed ``summary.json``.

    Returns:
        ``(field_path, value)`` pairs, sorted; empty when the artefact is clean.
    """
    return sorted(
        (path, value)
        for path, value in _string_values(summary, "$")
        if value.startswith("/") and value not in PLATFORM_FD_DIRECTORIES
    )


def _reference_attributes(document: str) -> list[str]:
    """Return every ``src`` and ``href`` value of an HTML document.

    Args:
        document: The HTML text.

    Returns:
        The attribute values, in document order.
    """
    return [
        double or single
        for double, single in _REFERENCE_ATTRIBUTE.findall(document)
    ]


# The gates: one test per condition, in the order the report is read -- is the run clean, did
# every check execute, did each run the declared number of cases, then the case-level proofs
# (round-trip over the committed corpus, the boundary subset at three levels, the performance
# floors) and finally the three report contracts. No test repeats another's assertion.


def test_reference_run_is_clean(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a suite that cannot come back clean: any failed, errored or skipped case.

    AAP §0.4.1 requires ``counts.failed == 0``, ``counts.error == 0`` **and**
    ``counts.skipped == 0`` from the conforming target. The zero-skip half is never softened:
    a skipped ``RN-SVG`` would let the CI gate pass without rasterizing a single SVG, which is
    exactly why the e2e job installs the rasterizer (``librsvg2-bin``, the package providing
    ``rsvg-convert``) instead of tolerating the skip. A degraded environment therefore fails
    here and names the check that skipped, rather than turning a gate into a formality.

    The identity half -- the adapter, the EC level, the seed and the ``run_id`` shape -- is here
    because every later test compares its case list against those three values: a run of some
    other configuration would make them all assert something other than what they name.
    """
    report_root, summary, exit_code = reference_report
    counts = summary["counts"]
    results = _results(summary)

    assert exit_code == 0, (
        f"the reference run exited {exit_code} instead of 0; AAP §0.4.1 requires a conforming "
        f"target to produce neither a failed nor an errored case (report root {report_root})"
    )
    assert counts["failed"] == 0, (
        f"{counts['failed']} case(s) failed against the conforming target, which AAP §0.4.1 "
        f"forbids; failing checks: "
        f"{sorted({str(result['check_id']) for result in results if result['status'] == 'failed'})}"
    )
    assert counts["error"] == 0, (
        f"{counts['error']} case(s) errored inside the framework, which AAP §0.4.1 forbids; "
        f"errored checks: "
        f"{sorted({str(result['check_id']) for result in results if result['status'] == 'error'})}"
    )
    skipped_checks = sorted(
        {str(result["check_id"]) for result in results if result["status"] == "skipped"}
    )
    assert counts["skipped"] == 0, (
        f"{counts['skipped']} case(s) were skipped, but AAP §0.4.1 requires zero skips: a skip "
        f"would let a check pass CI without doing its work, as a skipped RN-SVG would rasterize "
        f"nothing (install the SVG rasterizer rather than accept the skip). Skipped checks: "
        f"{skipped_checks}"
    )
    assert sum(counts.values()) == len(results), (
        f"the four tallies sum to {sum(counts.values())} but the run reported {len(results)} "
        f"results, so an outcome was folded into another bucket (AAP §0.3.4)"
    )
    assert summary["adapter"] == REFERENCE_ADAPTER, (
        f"the run tested adapter {summary['adapter']!r}, not {REFERENCE_ADAPTER!r}; every case "
        f"in this module asserts the conforming target's behaviour"
    )
    assert summary["ec_level"] == RUN_EC_LEVEL, (
        f"the run used EC level {summary['ec_level']!r} rather than {RUN_EC_LEVEL!r}, so the "
        f"240-record whole-corpus run success criterion 2 names was not the one measured"
    )
    assert summary["seed"] == RUN_SEED, (
        f"the run used seed {summary['seed']!r} rather than the committed default {RUN_SEED}, "
        f"so the report does not reproduce from the configuration it was started from"
    )
    run_id = summary["run_id"]
    assert isinstance(run_id, str) and SHA256_HEX.fullmatch(run_id) is not None, (
        f"run_id must be the content hash over the run's parameters -- not a per-run UUID and "
        f"not a reading of the wall clock (AAP §0.3.4) -- but it is {run_id!r}"
    )


def test_every_check_registered_is_executed_and_passes(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a check that never executes, or that executes and never passes.

    This is AAP §0.4.1's second clause and AAP §0.6.1 #8 together: the set of checks the run
    executed equals the 21 declared identifiers, the registry equals the same set so no check
    exists only for a test, and every executed check has at least one ``passed`` result -- a
    check counted as green merely because it is present would be the vacuous gate the
    self-verification section exists to prevent.

    The status/reason biconditional is asserted in the same pass because it is the model's own
    invariant (``CaseResult`` enforces it): a pass carries no reason and every other status
    carries exactly one, so neither a reason on a pass nor a reasonless failure can hide here.
    """
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
        f"check reachable only from a test would leave it short (AAP §0.6.1 #8)"
    )
    without_pass = sorted(
        check_id
        for check_id, check_results in by_check.items()
        if not any(result["status"] == "passed" for result in check_results)
    )
    assert not without_pass, (
        f"{without_pass} executed but produced no passed result; AAP §0.4.1 requires every "
        f"check to have at least one, so presence alone can never count as a pass"
    )
    inconsistent = sorted(
        f"{result['check_id']}[{result['case_id']}] status={result['status']!r} "
        f"reason={result['reason']!r}"
        for result in results
        if (result["status"] == "passed") != (result["reason"] is None)
    )
    assert not inconsistent, (
        f"{len(inconsistent)} result(s) pair a status with the wrong reason: a passing case must "
        f"carry no reason and every other status exactly one (AAP §0.3.4); first offenders: "
        f"{inconsistent[:10]}"
    )


def test_case_matrix_matches_the_declared_workload(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a re-balanced workload: any check running a number of cases the matrix denies.

    AAP §0.8.1 fixes the case matrix, and the whole comparison is made -- every one of the 21
    rows, not a sample -- so moving cases from one check to another fails loudly instead of
    leaving a workload nobody declared. The total is asserted separately from the rows so a
    matrix that happened to sum to the declared figure while a row moved is still caught by
    the row comparison.

    AAP §0.8.1's 1,287 encode invocations and 1,462 primary decodes are **not** fields of
    ``summary.json``, and they are deliberately not made into fields here: the totals are
    asserted only through what the report does expose -- the per-check case counts below,
    ``PF-ENCODE``'s and ``PF-DECODE``'s measured sample counts, ``DT-DETERMINISTIC``'s six
    paired cases and ``DM-RESILIENCE``'s eight-record by seven-kind shape. Inventing a counter
    to make them directly observable would change the product to suit the test.
    """
    _report_root, summary, _exit_code = reference_report
    observed = Counter(str(result["check_id"]) for result in _results(summary))
    mismatched = {
        check_id: (CASE_MATRIX.get(check_id), observed.get(check_id, 0))
        for check_id in sorted(set(CASE_MATRIX) | set(observed))
        if CASE_MATRIX.get(check_id) != observed.get(check_id)
    }
    assert not mismatched, (
        f"{len(mismatched)} check(s) ran a number of cases AAP §0.8.1's matrix does not "
        f"declare, as (declared, observed): {mismatched}"
    )
    assert sum(observed.values()) == DECLARED_CASES, (
        f"the full run reported {sum(observed.values())} cases, not the declared "
        f"{DECLARED_CASES}: {dict(sorted(observed.items()))}"
    )


def test_roundtrip_covers_all_240_corpus_records(
    reference_report: tuple[Path, dict[str, Any], int],
    repo_root: Path,
) -> None:
    """Catch a round-trip gate that tests fewer records, or a different corpus, than it claims.

    AAP §0.6.1 #2 requires round-trip to hold for all 240 records at EC M. The case list is
    compared against ``corpus/default.jsonl`` read from the repository rather than against the
    report's own list, so a corpus that lost a record, gained one, or was replaced under the run
    fails here: the run is tied to the committed corpus, not to itself.
    """
    _report_root, summary, _exit_code = reference_report
    results = _results_of(summary, "RT-ROUNDTRIP")

    assert len(results) == CORPUS_RECORDS, (
        f"RT-ROUNDTRIP ran {len(results)} cases, but the committed corpus holds "
        f"{CORPUS_RECORDS} records that success criterion 2 requires round-trip for"
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
    """Catch a boundary subset that is not 40 records, or that breaks at L, Q or H.

    This is how AAP §0.6.1 #2's four-run sequence is invoked: 240 records at EC M (the fixture
    above) and the 40-record boundary subset at each of L, Q and H. Round-trip, minimal version
    and EC match are asserted at 40 cases per level, and each corpus-driven request is checked
    to carry the level the run was invoked with, so a run that ignored ``--ec`` cannot pass on a
    report that merely says the right level at the top.

    The synthetic checks are exempt from that per-request assertion because ``--class`` does not
    reach them: they build their own requests, so ``CL-ARTIFACTS`` sends none, the ``IV-*`` probes
    send the level their contract fixes, and ``RN-LOGO`` sends EC H and EC L by design (AAP
    §0.8.1). Their case counts are consequently the same as in the full run, which is why only
    the corpus-driven rows are asserted at 40 here.
    """
    assert set(boundary_runs) == set(BOUNDARY_LEVELS), (
        f"the boundary fixtures cover {sorted(boundary_runs)} rather than "
        f"{list(BOUNDARY_LEVELS)}; success criterion 2 runs the subset at every remaining level"
    )
    boundary_records = filter_by_class(_corpus(repo_root), [BOUNDARY_CLASS_GLOB])
    assert len(boundary_records) == BOUNDARY_RECORDS, (
        f"the {BOUNDARY_CLASS_GLOB!r} class filter selects {len(boundary_records)} records, not "
        f"the {BOUNDARY_RECORDS} success criterion 2 names"
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
    """Catch a performance floor the product reported from somewhere other than the config.

    AAP §0.6.1 #10 and §0.6.2 require the encode p95, the decode p95 and the throughput floor to
    be declared once and every gate to read that one declaration. The claim is made mechanical
    here rather than restated: the thresholds are read from the committed ``qrcheck.toml`` and
    compared with the values the run itself copied into each result's ``expected``, so this file
    carries no floor of its own and a floor edited in the document is followed rather than
    contradicted.

    ``samples`` counts the **measured** invocations: the warm-up is untimed and precedes them
    (AAP §0.8.1), so the count is asserted against the configured sample size rather than
    against the invocation total the case matrix records.
    """
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

    # The three floors the plan names by key, asserted one by one so this test cannot pass by
    # iterating an empty mapping if a check stopped reporting its budget.
    for key in ("encode_p95_ms", "min_symbols_per_minute"):
        assert key in encode["expected"], (
            f"PF-ENCODE's expected block does not report {key!r}, so the budget it was measured "
            f"against is not in the report: {sorted(encode['expected'])}"
        )
        assert encode["expected"][key] == limits[key], (
            f"PF-ENCODE measured against {key}={encode['expected'][key]!r}, but {CONFIG_FILE_NAME} "
            f"declares {limits[key]!r}: the floor did not come from the single declaration "
            f"(AAP §0.6.2)"
        )
    assert "decode_p95_ms" in decode["expected"], (
        f"PF-DECODE's expected block does not report 'decode_p95_ms', so the budget it was "
        f"measured against is not in the report: {sorted(decode['expected'])}"
    )
    assert decode["expected"]["decode_p95_ms"] == limits["decode_p95_ms"], (
        f"PF-DECODE measured against decode_p95_ms={decode['expected']['decode_p95_ms']!r}, but "
        f"{CONFIG_FILE_NAME} declares {limits['decode_p95_ms']!r}: the floor did not come from "
        f"the single declaration (AAP §0.6.2)"
    )

    # Every other budget the case reported is the configured one too: the sample counts, the
    # warm-up and the probe size all live in the same table, so none of them may be restated.
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
    """Catch a run that files evidence for what passed, or records an absolute path.

    AAP §0.3.4 fixes the persistence policy: evidence is written for failing, errored and
    skipped-with-reason cases only, and a passing case records ``evidence_dir = null``, so a
    240-by-21 run does not create tens of thousands of files. The evidence tree's absence is
    therefore the correct outcome of a clean run, and a test that expected files on disk would
    be asserting the opposite of the contract.

    The absolute-path scan enforces AAP §0.5.5 over the whole document: evidence locations are
    relative to ``report_dir``, and a request records a path as the configuration gave it, so no
    field may name the clone or the machine the run happened in. ``CL-ARTIFACTS``'s record of the
    platform descriptor directory it enumerated is the one exempt value, because that is a fixed
    operating-system interface rather than a location on this disk.
    """
    report_root, summary, _exit_code = reference_report
    results = _results(summary)

    with_evidence = sorted(
        f"{result['check_id']}[{result['case_id']}] -> {result['evidence_dir']!r}"
        for result in results
        if result["evidence_dir"] is not None
    )
    assert not with_evidence, (
        f"{len(with_evidence)} passing case(s) recorded an evidence directory, but AAP §0.3.4 "
        f"writes evidence only for failing, errored and skipped cases; first offenders: "
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
        f"{len(offenders)} field(s) of summary.json hold an absolute path, which AAP §0.5.5 "
        f"forbids in an artefact (evidence paths are relative to report_dir and requests carry "
        f"paths as configured), as field: value: {offenders[:10]}"
    )


def test_the_three_report_artefacts_exist_and_parse(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a missing or malformed report: the three artefacts a run owes its consumers.

    AAP §0.6.1 #6 makes JUnit the CI view, JSON the tooling view and HTML the human view, so the
    three must agree with the result model they were written from. ``junit.xml`` is parsed and
    its counts are compared with ``summary.json``'s, and the two attributes AAP §0.3.4 omits on
    purpose -- ``hostname`` and ``timestamp`` -- are asserted absent, because their presence is
    what would make the file irreproducible. For ``index.html`` only the self-containment
    invariants are asserted here: no script and no reference to an external origin. The escaping
    and per-failure-image assertions belong to ``tests/unit/test_html_writer.py``; the overlap
    with that module is deliberate, because §0.6.1 #6 lists both the writer tests and the real
    run's artefacts as its evidence.
    """
    report_root, summary, _exit_code = reference_report
    counts = summary["counts"]
    results = _results(summary)

    for name in REPORT_ARTEFACTS:
        assert (report_root / name).is_file(), (
            f"the run wrote no {name} under {report_root}; a run owes its CI, tooling and human "
            f"consumers all three artefacts (AAP §0.3.4, §0.6.1 #6)"
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
        f"junit.xml carries {volatile}, which AAP §0.3.4 omits deliberately so the file is "
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
        f"is a data: URI by construction (AAP §0.3.4)"
    )


def test_summary_schema_is_exact(
    reference_report: tuple[Path, dict[str, Any], int],
    repo_root: Path,
) -> None:
    """Catch a summary that drifted off its schema, lost its ordering, or breached the ceiling.

    AAP §0.3.4 fixes the nine top-level fields, the four tallies and the ten fields of every
    result as a stable public surface that machine consumers key on, and §0.5.3 fixes
    ``results[]`` order as ``(check_id, case_id)``. Both are asserted exactly -- not as subsets --
    because a renamed or dropped field breaks every consumer while an added one does not.

    ``loc_count`` is checked against the committed ceiling, which is AAP §0.6.1 #11's line
    budget reported by the run itself. What is deliberately *not* asserted here is masked
    equality between two serializations: that needs a second run and belongs to
    ``tests/unit/test_json_writer.py``. The timestamps are checked for shape only, never
    compared -- only the ``PF-*`` checks may assert on a time (AAP §0.5.5).
    """
    _report_root, summary, _exit_code = reference_report

    assert set(summary) == set(SUMMARY_KEYS), (
        f"summary.json's top level is {sorted(summary)}, not the nine documented fields "
        f"{sorted(SUMMARY_KEYS)}; missing: {sorted(set(SUMMARY_KEYS) - set(summary))}, "
        f"unexpected: {sorted(set(summary) - set(SUMMARY_KEYS))}"
    )
    counts = summary["counts"]
    assert set(counts) == set(COUNT_KEYS), (
        f"summary.json's counts object is {sorted(counts)}, not the four separate tallies "
        f"{sorted(COUNT_KEYS)}; a folded tally would hide a skip (AAP §0.3.4)"
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
        f"loc_count is {loc_count!r}, not the positive integer the line budget is measured as "
        f"(AAP §0.6.1 #11)"
    )
    assert loc_count <= loc_limit, (
        f"the run measured {loc_count} tracked physical lines against a ceiling of {loc_limit} "
        f"from {CONFIG_FILE_NAME}; the line budget is breached (AAP §0.6.1 #11)"
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

