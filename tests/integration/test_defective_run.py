"""Set equality between a real ``--adapter defective`` run and its declared expectation.

A suite that cannot fail proves nothing, so the framework ships the defect it is supposed to
catch: ``qrcheck/targets/defective.py`` carries five injections, and this module holds AAP §0.4.2
-- a run against it fails **exactly** the nine checks those injections target, with zero ``error``
results and the other twelve passing in the same run -- and §0.4.3, that nothing was weakened,
skipped, retried or threshold-relaxed to reach that outcome. It is the defective half of success
criterion 1 and the version-and-EC half of criterion 3 (AAP §0.6.1 #1 and #3).

Set equality is the whole point, and it is asserted in both directions: containment would accept
any run whose failures *include* the declared nine, which is exactly what a cascade of collateral
failures looks like. So the observed set is compared with ``SELF_CHECK_TARGETS["defective"]``
**and** with the nine-identifier literal of §0.4.2 as an equality, and every failure message names
what is extra (failed but not declared) beside what is missing (declared but did not fail), sorted.
The consequence columns are part of the expectation, not noise: four of the identifiers fail from a
**declared, unavoidable** consequence of another injection, so a test demanding only the five
primary ones would be wrong. The five injections are declared once in ``DEFECTS`` and once in
``qrcheck/registry/expectations.py``, deliberately independently (the registry must not import an
adapter that imports segno), and the last test here proves the two still agree.

Evidence is written for failing, errored and skipped cases only (AAP §0.3.4), so this is the one
integration module in which on-disk artefacts are guaranteed to exist and the attribution can be
read rather than inferred: every failing case carries ``request.json`` and ``decoded.json`` under
``<report_root>/evidence/<check-id>/<failure-id>/``, recording the case's identity and agreeing
with ``summary.json`` about it. ``RN-QUIET-ZONE`` and ``RN-MODULE-SIZE`` are held to their own
measurement keys instead, because AAP §0.4.2 measures both arithmetically from the image so that
neither needs a decode. Each injection then has one observation from AAP §0.4.5 asserted against a
real failure directory.

It runs nothing itself: both runs come from the session-scoped ``reference_report`` and
``defective_report`` fixtures, so the five-run budget of AAP §0.6.1 #9 is untouched. No retry, no
``xfail``, no skip and no threshold appears here, no wall-clock value is compared (AAP §0.5.5), and
every collection walked is ordered, so two runs assert the same things in the same order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pytest

from qrcheck.registry.check import REGISTRY
from qrcheck.registry.expectations import SELF_CHECK_TARGETS, validate_against_registry
from qrcheck.report.model import Status
from qrcheck.targets.defective import DEFECTS, Defect, expected_failing_ids

# The cost of this module is two full 240-record runs, both session-scoped in
# tests/integration/conftest.py. The marker is the one thing that keeps them out of the fast
# unit job, so it is declared here and nowhere else; pyproject.toml registers the name.
pytestmark = pytest.mark.e2e

# The nine check identifiers a --adapter defective run must fail (AAP §0.4.2), as one literal.
# SELF_CHECK_TARGETS["defective"] is compared against this too: the declaration in the product
# and the acceptance criterion in the plan are two statements of the same set, and a drift
# between them is a defect in one of them rather than something to reconcile silently.
_EXPECTED_FAILING_CHECKS: Final[frozenset[str]] = frozenset(
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

# The twelve checks that must pass in the same run that fails the nine above. Together with
# _EXPECTED_FAILING_CHECKS this partitions the twenty-one declared identifiers, which is what
# makes "the complement of the expected failures" and "these twelve, listed" the same assertion.
_EXPECTED_PASSING_CHECKS: Final[frozenset[str]] = frozenset(
    {
        "CL-ARTIFACTS",
        "DM-RESILIENCE",
        "DM-SEEDED",
        "IV-BAD-OPTION",
        "IV-EMPTY",
        "MD-MODE-OPTIMAL",
        "PF-DECODE",
        "PF-ENCODE",
        "RN-CONTRAST",
        "RN-LOGO",
        "RN-SVG",
        "SY-MASK",
    }
)

# Each injection of AAP §0.4.2 by name, with every identifier it is declared to fail: the check
# designed to catch it plus the consequences the plan declares unavoidable. This is the plan's
# per-defect attribution, and the last test asserts it against the target's own DEFECTS table.
_DECLARED_PER_DEFECT: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "missing_quiet_zone": frozenset({"RN-QUIET-ZONE", "RN-MODULE-SIZE"}),
        "non_deterministic_mask": frozenset({"DT-DETERMINISTIC"}),
        "over_capacity_acceptance": frozenset({"IV-OVERCAPACITY"}),
        "truncated_payload": frozenset({"RT-ROUNDTRIP", "MD-CHARSET"}),
        "wrong_ec_level": frozenset({"SY-EC-MATCH", "SY-VERSION-MIN", "SY-MATRIX-EQUIV"}),
    }
)

# The closed decode-evidence projection of AAP §0.3.4 that makes a failure re-derivable: both
# decoder outcomes with their polarity, the agreement classification, the EC level from the
# decoder and from the format information independently, the version implied by the sampled grid,
# the mask, and the sampled matrix's digest. Sorted, so a missing-key diagnostic lists them in a
# stable order.
_DECODE_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "agreement",
    "ec_level_decoder",
    "ec_level_format",
    "mask",
    "matrix_sha256",
    "per_decoder",
    "polarity_applied",
    "version",
)

# The two checks AAP §0.4.2 measures arithmetically from the image rather than by decoding it, so
# their evidence carries measurements instead of the decode projection above. Holding them to
# _DECODE_EVIDENCE_KEYS would contradict the plan in the one place it explains why they need no
# decode, and would leave every such case failing on a key the check is designed never to produce.
_MEASUREMENT_ONLY_CHECKS: Final[frozenset[str]] = frozenset({"RN-MODULE-SIZE", "RN-QUIET-ZONE"})

# The measurement keys each measurement-only check must persist, so that its failure is as
# re-derivable as a decoded one: the canvas the module geometry was read from, the measurement
# itself, and the value the request asked for.
_MEASUREMENT_EVIDENCE_KEYS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "RN-MODULE-SIZE": ("image_px", "module_px_measured", "module_size_px_requested"),
        "RN-QUIET-ZONE": ("image_px", "quiet_zone_measured", "quiet_zone_requested"),
    }
)

# The four sides RN-QUIET-ZONE measures, in the order the evidence names them.
_QUIET_ZONE_SIDES: Final[tuple[str, ...]] = ("bottom", "left", "right", "top")

# The closed artefact set this module reads, by name. The names are the evidence layout's, and
# reading them here rather than deriving them keeps a renamed artefact visible as a failure of
# this module instead of a KeyError inside a helper.
_REQUEST_JSON: Final[str] = "request.json"
_DECODED_JSON: Final[str] = "decoded.json"
_SYMBOL_PNG: Final[str] = "symbol.png"
_SYMBOL_REPEAT_PNG: Final[str] = "symbol-repeat.png"


def _results(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the run's result objects, proving the report carries the list it must.

    The list is returned as ``summary.json`` stores it -- ordered by ``(check_id, case_id)`` by the
    writer, and not re-sorted here. A missing ``results`` list fails the test rather than letting
    every assertion below rest on an empty iteration.
    """
    results = summary.get("results")
    assert isinstance(results, list), (
        f"summary.json carries no results list: keys present {sorted(summary)}"
    )
    return results


def _statuses_by_check(results: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return each check identifier's statuses, in the order the report lists them."""
    grouped: dict[str, list[str]] = {}
    for result in results:
        grouped.setdefault(result["check_id"], []).append(result["status"])
    return grouped


def _failed_check_ids(results: list[Mapping[str, Any]]) -> frozenset[str]:
    """Return the identifiers of the checks that failed or errored, as one set.

    An error counts beside a failure because both mean the check did not pass: an injection that
    broke the framework rather than the symbol is not an acceptable way to reach the expected set,
    and the tallies are separately asserted to hold zero errors.
    """
    return frozenset(
        result["check_id"]
        for result in results
        if result["status"] in (Status.FAILED, Status.ERROR)
    )


def _case_evidence_dir(report_root: Path, result: Mapping[str, Any]) -> Path:
    """Return the failing case's evidence directory, proving the result records one.

    ``CaseResult`` forbids a null directory on a non-passing result, so its absence means evidence
    was not persisted and the failure cannot be re-derived.
    """
    evidence_dir = result["evidence_dir"]
    assert evidence_dir is not None, (
        f"{result['check_id']}[{result['case_id']}] is {result['status']} with no evidence_dir, "
        f"so its failure cannot be re-derived"
    )
    return report_root / evidence_dir


def _read_evidence(
    report_root: Path, result: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    """Return one JSON artefact of a failing case, parsed.

    An artefact that is absent or is not a JSON object fails the test, naming the case and the
    path that was expected.
    """
    path = _case_evidence_dir(report_root, result) / name
    assert path.is_file(), (
        f"{result['check_id']}[{result['case_id']}] carries no {name} at {path}"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), (
        f"{path} is not a JSON object: {type(payload).__name__}"
    )
    return payload


def _require(container: Mapping[str, Any], key: str, where: str) -> Any:
    """Return ``container[key]``, failing with the key's name when it is absent.

    The diagnostic lists the keys that *are* present, because the reader's next question is always
    what the artefact actually holds.
    """
    assert key in container, (
        f"{where} carries no {key!r}; keys present: {sorted(container)}"
    )
    return container[key]


def _first_failing(
    results: list[Mapping[str, Any]], check_id: str
) -> Mapping[str, Any]:
    """Return ``check_id``'s failing result with the lowest ``case_id``, so the choice is stable.

    A check with no failing result fails the test, because there is then no evidence to read.
    """
    failing = [
        result
        for result in results
        if result["check_id"] == check_id and result["status"] == Status.FAILED
    ]
    assert failing, (
        f"{check_id} has no failed result in this run, so its evidence cannot be inspected"
    )
    return min(failing, key=lambda result: result["case_id"])


def test_defective_run_exits_1(
    defective_report: tuple[Path, dict[str, Any], int],
) -> None:
    """The defective run must exit 1, with a failure and no internal error.

    Catches a run that reaches the expected failing set by *erroring* rather than by judging -- an
    error is exit 3 and means the framework faulted rather than the symbol -- and a run whose
    tallies no longer account for every result it wrote, which would let a result disappear from the
    count a CI gate reads.
    """
    _, summary, exit_code = defective_report
    counts = summary["counts"]
    results = _results(summary)

    assert exit_code == 1, (
        f"a --adapter defective run must exit 1 (at least one check failed and none errored), "
        f"got {exit_code}; counts {dict(sorted(counts.items()))}"
    )
    assert counts["error"] == 0, (
        f"the defective run reported {counts['error']} error result(s): an injection must be "
        f"caught by a check's verdict, never by the framework faulting; errored checks "
        f"{sorted({r['check_id'] for r in results if r['status'] == Status.ERROR})}"
    )
    assert counts["failed"] > 0, (
        "the defective run failed nothing, so the suite is vacuous against the target it ships "
        "to prove itself non-vacuous"
    )
    assert sum(counts.values()) == len(results), (
        f"the tallies {dict(sorted(counts.items()))} account for "
        f"{sum(counts.values())} results but summary.json holds {len(results)}"
    )


def test_failed_check_set_equals_the_declared_expectation(
    defective_report: tuple[Path, dict[str, Any], int],
) -> None:
    """The failing check set must equal the declared nine -- not a superset, not a subset.

    Both directions are broken acceptance criteria: a superset means a failure outside the declared
    blast radius (an injection cascading further than AAP §0.4.2 declares, or a framework defect),
    and a subset means a declared defect went undetected.
    """
    _, summary, _ = defective_report
    observed = _failed_check_ids(_results(summary))
    declared = SELF_CHECK_TARGETS["defective"]

    def _difference_message(name: str, expected: frozenset[str]) -> str:
        return (
            f"the check set failing a --adapter defective run is not the {name}: "
            f"observed {sorted(observed)}; expected {sorted(expected)}; "
            f"failed but not declared {sorted(observed - expected)}; "
            f"declared but did not fail {sorted(expected - observed)}"
        )

    assert observed == declared, _difference_message(
        "declared expectation SELF_CHECK_TARGETS['defective']", declared
    )
    assert observed == _EXPECTED_FAILING_CHECKS, _difference_message(
        "nine identifiers AAP §0.4.2 names", _EXPECTED_FAILING_CHECKS
    )


def test_the_other_twelve_checks_all_pass(
    defective_report: tuple[Path, dict[str, Any], int],
) -> None:
    """The twelve checks outside the declared set must every one of them pass.

    Catches an expected failure that was turned into a skip -- the target declares the same
    capabilities the conforming one does, precisely so that no expected failure can be reached
    that way (AAP §0.4.2) -- and a check that reports nothing at all, which set equality over the
    failing identifiers alone would not see.
    """
    _, summary, _ = defective_report
    results = _results(summary)
    grouped = _statuses_by_check(results)

    problems: list[str] = []
    for check_id in sorted(_EXPECTED_PASSING_CHECKS):
        statuses = grouped.get(check_id)
        if not statuses:
            problems.append(f"{check_id}: no result in the run")
            continue
        not_passed = sorted({status for status in statuses if status != Status.PASSED})
        if not_passed:
            problems.append(
                f"{check_id}: {not_passed} among {len(statuses)} case(s), which must all pass"
            )
    assert not problems, (
        "the checks outside the declared failing set did not all pass: " + "; ".join(problems)
    )

    skipped_ids = sorted({r["check_id"] for r in results if r["status"] == Status.SKIPPED})
    assert summary["counts"]["skipped"] == 0, (
        f"the defective run skipped {summary['counts']['skipped']} case(s) under "
        f"{skipped_ids}: a skip is never a pass, and an expected failure would hide behind one"
    )


def test_every_failed_case_carries_request_and_decoded_evidence(
    defective_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Every failing case must carry the evidence that makes its failure re-derivable.

    Catches a failure recorded without the request that produced it or the decoded-or-error outcome
    that judged it, and a result whose identity disagrees with the evidence filed for it. The decode
    projection of AAP §0.3.4 is required wherever a symbol was decoded; ``RN-QUIET-ZONE`` and
    ``RN-MODULE-SIZE`` are held to their measurement keys instead, because AAP §0.4.2 measures both
    arithmetically from the image and neither performs a decode.
    """
    report_root, summary, _ = defective_report
    results = _results(summary)
    failed = [result for result in results if result["status"] == Status.FAILED]
    assert failed, (
        "the defective run recorded no failed result, so there is no evidence to inspect"
    )

    for result in failed:
        check_id = result["check_id"]
        case_id = result["case_id"]
        where = f"{check_id}[{case_id}]"
        request = _read_evidence(report_root, result, _REQUEST_JSON)
        decoded = _read_evidence(report_root, result, _DECODED_JSON)

        for key in ("check_id", "case_id", "seed"):
            _require(request, key, f"{where} {_REQUEST_JSON}")
        assert request["check_id"] == check_id, (
            f"{where} evidence is filed under a different check: {request['check_id']!r}"
        )
        assert request["case_id"] == case_id, (
            f"{where} evidence is filed under a different case: {request['case_id']!r}"
        )
        assert request["seed"] == result["seed"], (
            f"{where} evidence records seed {request['seed']!r} but the result records "
            f"{result['seed']!r}, so the case could not be reproduced from the report"
        )

        keys = (
            _MEASUREMENT_EVIDENCE_KEYS[check_id]
            if check_id in _MEASUREMENT_ONLY_CHECKS
            else _DECODE_EVIDENCE_KEYS
        )
        for key in keys:
            _require(decoded, key, f"{where} {_DECODED_JSON}")

    # The attribution asserted below reads one case per defect, so every one of the nine must
    # have failed with evidence rather than merely have failed.
    without_evidence = sorted(
        check_id
        for check_id in _EXPECTED_FAILING_CHECKS
        if not any(
            result["check_id"] == check_id and result["evidence_dir"] is not None
            for result in failed
        )
    )
    assert not without_evidence, (
        f"{without_evidence} failed without a failed case carrying evidence, so their defects "
        "are not inspectable from the report"
    )


def test_defect_attribution_is_visible_in_the_evidence(
    defective_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Each injection's own observation must be readable in a real failure directory.

    Catches a defect that fails a check for the wrong reason: the nine identifiers say *which*
    checks must fail, while only the observation named for each injection in AAP §0.4.5 says
    *why*, so a check failing on something other than its injected defect would satisfy the set
    equality above while proving nothing about the defect the target carries.
    """
    report_root, summary, _ = defective_report
    results = _results(summary)

    # D1 wrong_ec_level: the symbol really is one error-correction level down, read from two
    # independent sources -- the primary decoder's own report and the format information -- both
    # of which differ from the level the request asked for and agree with each other.
    ec_case = _first_failing(results, "SY-EC-MATCH")
    ec_request = _read_evidence(report_root, ec_case, _REQUEST_JSON)
    ec_decoded = _read_evidence(report_root, ec_case, _DECODED_JSON)
    requested_ec = _require(ec_request, "ec_level", f"SY-EC-MATCH[{ec_case['case_id']}] request")
    decoder_ec = _require(ec_decoded, "ec_level_decoder", "SY-EC-MATCH decoded")
    format_ec = _require(ec_decoded, "ec_level_format", "SY-EC-MATCH decoded")
    assert decoder_ec != requested_ec, (
        f"SY-EC-MATCH[{ec_case['case_id']}]: the decoder read EC level {decoder_ec!r} for a "
        f"request that asked for {requested_ec!r}, so the downshift is not what failed"
    )
    assert format_ec != requested_ec, (
        f"SY-EC-MATCH[{ec_case['case_id']}]: the format information carries EC level "
        f"{format_ec!r} for a request that asked for {requested_ec!r}"
    )
    assert decoder_ec == format_ec, (
        f"SY-EC-MATCH[{ec_case['case_id']}]: the two independent readings disagree -- decoder "
        f"{decoder_ec!r} against format information {format_ec!r} -- so the evidence does not "
        f"show one downshift but an inconsistency of its own"
    )

    # D3 missing_quiet_zone: one module drawn where four or more were asked, measured on each of
    # the four sides, so the shortfall is read from the image rather than assumed.
    quiet_case = _first_failing(results, "RN-QUIET-ZONE")
    quiet_request = _read_evidence(report_root, quiet_case, _REQUEST_JSON)
    quiet_decoded = _read_evidence(report_root, quiet_case, _DECODED_JSON)
    requested_zone = _require(quiet_request, "quiet_zone_modules", "RN-QUIET-ZONE request")
    measured_zone = _require(quiet_decoded, "quiet_zone_measured", "RN-QUIET-ZONE decoded")
    assert isinstance(measured_zone, Mapping), (
        f"RN-QUIET-ZONE[{quiet_case['case_id']}]: quiet_zone_measured is a "
        f"{type(measured_zone).__name__}, not a per-side mapping"
    )
    for side in _QUIET_ZONE_SIDES:
        measured = _require(measured_zone, side, "RN-QUIET-ZONE quiet_zone_measured")
        assert isinstance(measured, (int, float)) and not isinstance(measured, bool), (
            f"RN-QUIET-ZONE[{quiet_case['case_id']}]: the {side} side measures "
            f"{measured!r}, which is not a number of modules"
        )
        assert measured < requested_zone, (
            f"RN-QUIET-ZONE[{quiet_case['case_id']}]: the {side} side measures {measured} "
            f"module(s) against {requested_zone} requested, so the short zone is not what failed"
        )
    assert max(measured_zone[side] for side in _QUIET_ZONE_SIDES) <= requested_zone - 2, (
        f"RN-QUIET-ZONE[{quiet_case['case_id']}]: the widest measured side "
        f"{max(measured_zone[side] for side in _QUIET_ZONE_SIDES)} is close to the "
        f"{requested_zone} requested, but the injection draws one module where four or more "
        f"were asked -- a marginal shortfall is a different defect"
    )

    # D4 non_deterministic_mask: two calls of the identical request whose sampled matrices
    # differ, with both artefacts on disk so the change is visible and not merely asserted.
    determinism_case = _first_failing(results, "DT-DETERMINISTIC")
    determinism_dir = _case_evidence_dir(report_root, determinism_case)
    for artefact in (_SYMBOL_PNG, _SYMBOL_REPEAT_PNG):
        assert (determinism_dir / artefact).is_file(), (
            f"DT-DETERMINISTIC[{determinism_case['case_id']}] holds no {artefact} in "
            f"{determinism_dir}, so the second call's symbol cannot be compared"
        )
    determinism_decoded = _read_evidence(report_root, determinism_case, _DECODED_JSON)
    first_digest = _require(determinism_decoded, "matrix_sha256", "DT-DETERMINISTIC decoded")
    repeat = _require(determinism_decoded, "repeat", "DT-DETERMINISTIC decoded")
    assert isinstance(repeat, Mapping), (
        f"DT-DETERMINISTIC[{determinism_case['case_id']}]: repeat is a "
        f"{type(repeat).__name__}, not the second call's evidence"
    )
    repeat_digest = _require(repeat, "matrix_sha256", "DT-DETERMINISTIC repeat evidence")
    assert first_digest != repeat_digest, (
        f"DT-DETERMINISTIC[{determinism_case['case_id']}]: both calls sampled matrix "
        f"{first_digest}, so the mask did not change between them"
    )

    # D5 over_capacity_acceptance: a symbol returned for an over-capacity request, decoding to
    # fewer bytes than were asked for, with the first divergence located.
    over_case = _first_failing(results, "IV-OVERCAPACITY")
    over_request = _read_evidence(report_root, over_case, _REQUEST_JSON)
    over_decoded = _read_evidence(report_root, over_case, _DECODED_JSON)
    requested_len = _require(over_request, "payload_len", "IV-OVERCAPACITY request")
    decoded_len = _require(over_decoded, "decoded_len", "IV-OVERCAPACITY decoded")
    divergence = _require(over_decoded, "first_divergence_offset", "IV-OVERCAPACITY decoded")
    assert isinstance(decoded_len, int), (
        f"IV-OVERCAPACITY[{over_case['case_id']}]: decoded_len is {decoded_len!r}, so the "
        f"symbol could not be read and the acceptance is not what failed"
    )
    assert decoded_len < requested_len, (
        f"IV-OVERCAPACITY[{over_case['case_id']}]: the symbol decoded to {decoded_len} of "
        f"{requested_len} requested byte(s), so nothing was silently dropped"
    )
    assert divergence is not None, (
        f"IV-OVERCAPACITY[{over_case['case_id']}]: first_divergence_offset is null, so the "
        f"evidence does not locate where the accepted symbol departs from the request"
    )

    # D2 truncated_payload: the decoded bytes differ from the requested ones, with the digest of
    # both sides and the offset of the first divergence recorded.
    roundtrip_case = _first_failing(results, "RT-ROUNDTRIP")
    roundtrip_request = _read_evidence(report_root, roundtrip_case, _REQUEST_JSON)
    roundtrip_decoded = _read_evidence(report_root, roundtrip_case, _DECODED_JSON)
    decoded_digest = _require(roundtrip_decoded, "payload_sha256", "RT-ROUNDTRIP decoded")
    requested_digest = _require(
        roundtrip_decoded, "requested_payload_sha256", "RT-ROUNDTRIP decoded"
    )
    roundtrip_offset = _require(
        roundtrip_decoded, "first_divergence_offset", "RT-ROUNDTRIP decoded"
    )
    assert decoded_digest != requested_digest, (
        f"RT-ROUNDTRIP[{roundtrip_case['case_id']}]: the decoded payload digests to "
        f"{decoded_digest}, the same as the requested one, so nothing was truncated"
    )
    assert roundtrip_offset is not None, (
        f"RT-ROUNDTRIP[{roundtrip_case['case_id']}]: first_divergence_offset is null, so the "
        f"evidence does not locate where the decoded payload departs from the request"
    )
    assert roundtrip_offset > 0, (
        f"RT-ROUNDTRIP[{roundtrip_case['case_id']}]: the first divergence is at offset "
        f"{roundtrip_offset}, which would mean the payloads differ at their first byte rather "
        f"than at the truncation window"
    )
    roundtrip_requested_len = _require(roundtrip_request, "payload_len", "RT-ROUNDTRIP request")
    roundtrip_decoded_len = _require(roundtrip_decoded, "decoded_len", "RT-ROUNDTRIP decoded")
    truncated = (
        isinstance(roundtrip_decoded_len, int)
        and roundtrip_decoded_len < roundtrip_requested_len
    )
    assert truncated, (
        f"RT-ROUNDTRIP[{roundtrip_case['case_id']}]: the symbol decoded to "
        f"{roundtrip_decoded_len!r} of {roundtrip_requested_len} requested byte(s), so the "
        f"differing digest does not come from a truncated payload"
    )


def test_expectations_validate_against_the_live_registry() -> None:
    """The declared expectation must validate against the twenty-one registered checks.

    Catches a deleted expectation or a deleted check: ``validate_against_registry`` raises rather
    than shrinking what the gate above compares (AAP §0.4.3), so removing an expected failure
    breaks a test instead of silencing one.
    """
    import qrcheck.checks  # noqa: F401 - importing the package registers the nine families

    validate_against_registry()
    validate_against_registry(REGISTRY)

    assert SELF_CHECK_TARGETS["reference"] == frozenset(), (
        f"the conforming target declares expected failures "
        f"{sorted(SELF_CHECK_TARGETS['reference'])}, so its clean-run gate would excuse them"
    )
    registered = frozenset(REGISTRY)
    assert len(registered) == 21, (
        f"the live registry holds {len(registered)} check(s), not the twenty-one AAP §0.3.4 "
        f"declares: {sorted(registered)}"
    )
    unregistered = sorted(SELF_CHECK_TARGETS["defective"] - registered)
    assert not unregistered, (
        f"{unregistered} are expected to fail a defective run but are not registered checks, so "
        f"the gate could never observe them"
    )
    assert _EXPECTED_FAILING_CHECKS.isdisjoint(_EXPECTED_PASSING_CHECKS), (
        "a check cannot be both expected to fail and expected to pass: "
        f"{sorted(_EXPECTED_FAILING_CHECKS & _EXPECTED_PASSING_CHECKS)}"
    )
    partitioned = _EXPECTED_FAILING_CHECKS | _EXPECTED_PASSING_CHECKS
    assert partitioned == registered, (
        "the nine expected to fail and the twelve expected to pass must partition the registry: "
        f"unaccounted {sorted(registered - partitioned)}, "
        f"unknown {sorted(partitioned - registered)}"
    )


def test_reference_and_defective_form_a_contrast_pair(
    reference_report: tuple[Path, dict[str, Any], int],
    defective_report: tuple[Path, dict[str, Any], int],
) -> None:
    """The same suite must come back clean on one target and failing on the other.

    Catches the other form of the vacuity AAP §0.4 opens with: a suite that fails wherever a seeded
    defect sits proves something only if it also passes against a conforming generator given the
    same seed, the same error-correction level and the same corpus.
    """
    _, reference_summary, _ = reference_report
    _, defective_summary, _ = defective_report

    reference_counts = reference_summary["counts"]
    defective_counts = defective_summary["counts"]
    reference_results = _results(reference_summary)
    reference_failed = sorted(
        {r["check_id"] for r in reference_results if r["status"] == Status.FAILED}
    )
    reference_errored = sorted(
        {r["check_id"] for r in reference_results if r["status"] == Status.ERROR}
    )
    assert reference_counts["failed"] == 0, (
        f"the conforming target failed {reference_counts['failed']} case(s) under "
        f"{reference_failed}"
    )
    assert reference_counts["error"] == 0, (
        f"the conforming target errored {reference_counts['error']} case(s) under "
        f"{reference_errored}"
    )
    assert defective_counts["failed"] > 0, (
        "the five-injection target failed nothing, so the contrast pair is two clean runs"
    )

    assert reference_summary["seed"] == defective_summary["seed"], (
        f"the two runs used different seeds ({reference_summary['seed']!r} and "
        f"{defective_summary['seed']!r}), so their outcomes are not comparable"
    )
    assert reference_summary["ec_level"] == defective_summary["ec_level"], (
        f"the two runs used different error-correction levels "
        f"({reference_summary['ec_level']!r} and {defective_summary['ec_level']!r})"
    )
    reference_cases = sorted(r["case_id"] for r in reference_results)
    defective_cases = sorted(r["case_id"] for r in _results(defective_summary))
    assert reference_cases == defective_cases, (
        "the two runs cover different cases, so the contrast is not like-for-like: only in the "
        f"reference run {sorted(set(reference_cases) - set(defective_cases))[:10]}, only in the "
        f"defective run {sorted(set(defective_cases) - set(reference_cases))[:10]}"
    )
    assert reference_summary["adapter"] != defective_summary["adapter"], (
        f"both runs name the adapter {reference_summary['adapter']!r}, so one target was run "
        "twice instead of two targets being contrasted"
    )


def test_defects_table_declares_the_five_injections() -> None:
    """The target's own injection table must declare exactly the five defects and their catches.

    Catches a drift between the two declarations of the expected set: ``DEFECTS`` in the target
    and ``SELF_CHECK_TARGETS`` in the registry are independent on purpose (the registry must not
    import an adapter that imports segno), so a change to either -- a renamed injection, a moved
    consequence, a dropped catch -- has to fail here rather than move the gate above.
    """
    assert len(DEFECTS) == 5, (
        f"the target declares {len(DEFECTS)} injection(s), not the five of AAP §0.4.2: "
        f"{[getattr(defect, 'name', defect) for defect in DEFECTS]}"
    )
    assert all(isinstance(defect, Defect) for defect in DEFECTS), (
        "every DEFECTS entry must be a Defect record, so its catches are data a test can read"
    )

    declared = {defect.name: defect.caught_by | defect.consequential for defect in DEFECTS}
    assert len(declared) == len(DEFECTS), (
        f"two injections share a name: {sorted(defect.name for defect in DEFECTS)}"
    )
    assert set(declared) == set(_DECLARED_PER_DEFECT), (
        "the target's injections are not the five the plan names: missing "
        f"{sorted(set(_DECLARED_PER_DEFECT) - set(declared))}, unexpected "
        f"{sorted(set(declared) - set(_DECLARED_PER_DEFECT))}"
    )
    for name in sorted(_DECLARED_PER_DEFECT):
        expected = _DECLARED_PER_DEFECT[name]
        assert declared[name] == expected, (
            f"injection {name!r} declares {sorted(declared[name])}, but the plan attributes "
            f"{sorted(expected)} to it: extra {sorted(declared[name] - expected)}, missing "
            f"{sorted(expected - declared[name])}"
        )

    primary_catches = [
        check_id for defect in DEFECTS for check_id in sorted(defect.caught_by)
    ]
    assert len(primary_catches) == len(set(primary_catches)), (
        "one check is declared as the primary catch of two injections, so at least one "
        f"injection has no attributable catch: {sorted(primary_catches)}"
    )

    union = frozenset().union(*declared.values())
    assert union == SELF_CHECK_TARGETS["defective"], (
        f"the injections union to {sorted(union)}, but the registry declares "
        f"{sorted(SELF_CHECK_TARGETS['defective'])} as the expected failing set"
    )
    assert union == _EXPECTED_FAILING_CHECKS, (
        f"the injections union to {sorted(union)}, not the nine AAP §0.4.2 names: "
        f"{sorted(_EXPECTED_FAILING_CHECKS)}"
    )
    assert expected_failing_ids() == union, (
        f"the target derives {sorted(expected_failing_ids())} from its table, which is not the "
        f"union of the declared catches and consequences {sorted(union)}"
    )
