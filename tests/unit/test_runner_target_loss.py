"""The run-level target-loss policy: one error result per affected check, never per case.

The run contract fixes what a consumer reports when a target is unreachable or times out --
"``error`` result, ``case_id`` ``synthetic:target``, message naming the transport failure; one
such result per affected check, not per corpus record" -- and this module is the gate that holds it.
The defect it prevents is quadratic noise: ``RT-ROUNDTRIP`` alone has one case per corpus record, so
a policy applied per case would turn one dead target into two hundred and forty identical errors.

Five properties are pinned: **exactly one error per affected check**, counted from the
selection's own capability requirements rather than typed; **the diagnostic names the cause** --
the transport failure the adapter raised, or the ``timeout_s`` the resolved configuration
declares; **the run continues**, with ``CL-ARTIFACTS`` and ``PF-DECODE`` still passing because
they need no target, and ``--fail-fast`` the one policy that stops launching checks while still
returning a complete model; **the exit code becomes 3** through
:func:`qrcheck.cli.exit.exit_code_for`, whose precedence truth table is asserted beside it; and
**``TargetRejected`` keeps per-case granularity**, ``counts.error`` staying at zero and the exit
code 1, which keeps a future refactor from merging the two policies."""

from __future__ import annotations

import dataclasses
import json
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

import qrcheck.decode as decode_module
from qrcheck.checks import load_families
from qrcheck.cli.exit import EXIT_FAILED, EXIT_INTERNAL, EXIT_OK, exit_code_for
from qrcheck.config import models, schema
from qrcheck.registry.check import REGISTRY, CheckContract
from qrcheck.registry.selection import Selection, select
from qrcheck.report import (
    HTML_FILENAME,
    JUNIT_FILENAME,
    SUMMARY_FILENAME,
    write_all_reports,
)
from qrcheck.report.model import CaseResult, Counts, Reason, RunResult, Status
from qrcheck.runner import run_suite
from qrcheck.targets.base import (
    GenerateRequest,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
)

# The case identifier every target-loss result is filed under, written out as the run contract
# states it rather than imported from the runner: it is a public field of ``summary.json`` that a
# CI consumer keys on, so the test pins the contract instead of the constant the implementation
# names today.
TARGET_LOSS_CASE_ID: Final[str] = "synthetic:target"

# The nine top-level keys the run contract fixes for ``summary.json``, and the ten fields of one
# ``results[]`` entry, written out rather than read from the model, so a field added, dropped or
# renamed fails here instead of silently shrinking a machine-readable surface.
SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "started_at",
    "finished_at",
    "adapter",
    "ec_level",
    "seed",
    "loc_count",
    "counts",
    "results",
)
RESULT_FIELDS: Final[tuple[str, ...]] = (
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
)
COUNT_FIELDS: Final[tuple[str, ...]] = ("passed", "failed", "skipped", "error")

# The two checks that invoke no adapter at all: the witnesses that prove a run kept going after
# the target was lost.  ``CL-ARTIFACTS`` sorts before every other identifier, so it also runs
# *before* the loss; ``PF-DECODE`` sorts after it.
TARGET_INDEPENDENT_WITNESSES: Final[tuple[str, ...]] = ("CL-ARTIFACTS", "PF-DECODE")

# The selections this module drives: each names only registered identifiers, so a rename fails at
# :func:`qrcheck.registry.selection.select` rather than silently shrinking the selection.
# ``LOSS_SELECTION`` covers both halves of the policy -- the check that *meets* the loss
# (``RT-ROUNDTRIP``) and the check that receives a loss result without the target being called
# again (``SY-MASK``, which sorts after it).
LOSS_SELECTION: Final[str] = "CL-ARTIFACTS,PF-DECODE,RT-ROUNDTRIP,SY-MASK"
MINIMAL_LOSS_SELECTION: Final[str] = "CL-ARTIFACTS,RT-ROUNDTRIP,SY-MASK"
SINGLE_CHECK_SELECTION: Final[str] = "RT-ROUNDTRIP"
WITNESS_SELECTION: Final[str] = "CL-ARTIFACTS,PF-DECODE"
REJECTION_CLASS: Final[str] = "boundary-numeric"

# The three diagnostics the stub target raises, in this module's own words, so an assertion that
# finds one in the evidence proves the adapter's text reached the result unchanged.
TRANSPORT_FAILURE_TEXT: Final[str] = "cannot spawn target process: connection refused"
TIMEOUT_FAILURE_TEXT: Final[str] = "no response within timeout_s"
REJECTION_TEXT: Final[str] = "declared rejection: payload rejected by the target"

# The name a stub adapter reports when it stands in for a configured target: it must name a
# ``[targets.<name>]`` table, which is what makes the runner record the configured ``timeout_s``.
CONFIGURED_TARGET_NAME: Final[str] = "reference"

# The upper bound of the upward scan that reads a schema minimum -- a search bound, not a
# threshold: no product value is restated by it.
MINIMUM_SCAN_LIMIT: Final[int] = 1_000


# Reading the configuration, never restating it


def schema_minimum(group: str, key: str) -> int:
    """Return the smallest integer value the schema accepts for a key: the bounds live inside
    validator closures with no accessor, so the rule is offered candidates in ascending order."""
    spec = schema.SCHEMA[group][key]
    for candidate in range(MINIMUM_SCAN_LIMIT):
        if spec.check(candidate) is None:
            return candidate
    raise AssertionError(
        f"{group}.{key}: no accepted integer below {MINIMUM_SCAN_LIMIT}; this helper "
        f"reads a lower bound, so the key's rule needs another way in"
    )


def fast_performance_limits(limits: models.PerformanceLimits) -> models.PerformanceLimits:
    """Return the performance table with every count at its schema minimum: only the three *sizes*
    move -- the warm-up, the decode sample count and the probe image side."""
    return dataclasses.replace(
        limits,
        warmup=schema_minimum("checks.performance", "warmup"),
        decode_samples=schema_minimum("checks.performance", "decode_samples"),
        decode_image_px=schema_minimum("checks.performance", "decode_image_px"),
    )


@pytest.fixture
def fast_config(config_factory: Callable[..., models.Config]) -> models.Config:
    """Return a resolved configuration whose performance counts are at their minimum, built through
    the shared ``config_factory`` fixture, so only ``[checks.performance]`` is lowered."""
    return config_factory(performance=fast_performance_limits(config_factory().performance))


# The selection, read through the registry that owns check identity


@pytest.fixture
def registry() -> Mapping[str, CheckContract]:
    """Return the populated check registry, loading the family modules first: this module reads the
    registry *before* a run to derive how many checks the target's loss can affect."""
    load_families()
    return REGISTRY


def selected_contracts(
    registry: Mapping[str, CheckContract], selection: Selection
) -> tuple[CheckContract, ...]:
    """Return the contracts a selection resolves to, in identifier order."""
    return tuple(select(registry, selection.checks))


def target_dependent_ids(contracts: Sequence[CheckContract]) -> tuple[str, ...]:
    """Return the identifiers of the contracts that require a capability of the target -- exactly
    the set that receives one error result each."""
    return tuple(sorted(contract.id for contract in contracts if contract.requires))


def target_independent_ids(contracts: Sequence[CheckContract]) -> tuple[str, ...]:
    """Return the identifiers of the contracts that need nothing from the target."""
    return tuple(sorted(contract.id for contract in contracts if not contract.requires))


# Result inspection helpers


def results_for(run_result: RunResult, check_id: str) -> tuple[CaseResult, ...]:
    """Return every result the run filed under `check_id`, in case-identifier order."""
    return tuple(result for result in run_result.results if result.check_id == check_id)


def error_results(run_result: RunResult) -> tuple[CaseResult, ...]:
    """Return every ``error`` result of a run, in ``(check_id, case_id)`` order."""
    return tuple(result for result in run_result.results if result.status is Status.ERROR)


def assert_one_target_loss_result_per_affected_check(
    run_result: RunResult,
    contracts: Sequence[CheckContract],
) -> None:
    """Assert the run's target-loss results are one per affected check and nothing else: the
    count is derived from the selection's own requirements."""
    affected = target_dependent_ids(contracts)
    losses = error_results(run_result)
    assert tuple(result.check_id for result in losses) == affected
    assert {result.case_id for result in losses} == {TARGET_LOSS_CASE_ID}
    assert len(losses) == len(affected)
    for result in losses:
        assert result.reason in {Reason.TARGET_UNAVAILABLE, Reason.TARGET_TIMEOUT}
        # The evidence names the target, the error type and the adapter's own text, which
        # is what makes a run-level failure diagnosable from the report alone.
        assert result.decoded["adapter"] == run_result.adapter
        assert isinstance(result.decoded["error_text"], str)
        assert result.request == {}
        assert result.expected == {"outcome": "reachable_target"}


def assert_counts(
    run_result: RunResult,
    *,
    passed: int,
    failed: int,
    skipped: int,
    error: int,
) -> None:
    """Assert the run's four tallies exactly, and that they agree with its results."""
    expected = Counts(passed=passed, failed=failed, skipped=skipped, error=error)
    assert run_result.counts == expected
    assert Counts.from_results(run_result.results) == expected


def assert_summary_schema_complete(run_result: RunResult) -> dict[str, Any]:
    """Assert the run projects to a complete ``summary.json`` body and return it: returning a model
    instead of raising is what lets CI receive a consumable report."""
    summary = run_result.as_dict()
    assert sorted(summary) == sorted(SUMMARY_FIELDS)
    counts = summary["counts"]
    assert isinstance(counts, Mapping)
    assert sorted(counts) == sorted(COUNT_FIELDS)
    assert all(isinstance(value, int) for value in counts.values())
    entries = summary["results"]
    assert isinstance(entries, list)
    assert len(entries) == len(run_result.results)
    for entry in entries:
        assert isinstance(entry, Mapping)
        assert sorted(entry) == sorted(RESULT_FIELDS)
    return summary


# The decode seam: PF-DECODE must not need a real decoder either


def echo_decode(
    monkeypatch: pytest.MonkeyPatch,
    patcher: Any,
    make_decode_result: Callable[..., Any],
) -> None:
    """Stub the decode entry so each decode reports the payload it was asked for: ``PF-DECODE``
    asserts the first decode of its own probe image, which a canned result would fail."""

    def stub(
        image_bytes: bytes,
        request: GenerateRequest,
        *,
        secondary: bool | None = None,
    ) -> Any:
        """Log the call on the suite's patcher and answer with the requested payload."""
        patcher.stub(image_bytes, request, secondary=secondary)
        return make_decode_result(payload_bytes=request.payload)

    monkeypatch.setattr(decode_module, "decode_symbol", stub)


# Hand-built runs for the exit-code truth table


def case_result(
    make_case_result: Callable[..., CaseResult],
    index: int,
    status: Status,
    reason: Reason,
) -> CaseResult:
    """Return one result of `status` with a distinct identity, for a hand-built run: ``(check_id,
    case_id)`` must be unique, so the index participates in both halves."""
    return make_case_result(
        check_id="RT-ROUNDTRIP",
        case_id=f"corpus:truth-table-{index:03d}",
        status=status,
        reason=reason,
    )


def run_with_tallies(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    failed: int,
    error: int,
    skipped: int,
) -> RunResult:
    """Return a run whose tallies are exactly the requested counts: ``RunResult`` recomputes them
    from its results, so results are the only way to state a tally."""
    results = [
        case_result(make_case_result, index, Status.FAILED, Reason.VALUE_MISMATCH)
        for index in range(failed)
    ]
    results.extend(
        case_result(make_case_result, failed + index, Status.ERROR, Reason.INTERNAL_ERROR)
        for index in range(error)
    )
    results.extend(
        case_result(
            make_case_result,
            failed + error + index,
            Status.SKIPPED,
            Reason.UNSUPPORTED_CAPABILITY,
        )
        for index in range(skipped)
    )
    return make_run_result(results)


# A lost target: exactly one error per affected check


def test_an_unreachable_target_files_one_error_per_affected_check(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead target yields one ``synthetic:target`` error per check, never one per case: the
    selection carries two target-dependent checks and two that need nothing."""
    echo_decode(monkeypatch, patch_decode, make_decode_result)
    selection = Selection(checks=LOSS_SELECTION)
    contracts = selected_contracts(registry, selection)
    affected = target_dependent_ids(contracts)
    witnesses = target_independent_ids(contracts)
    assert affected and witnesses, "the selection must cover both sides of the policy"
    adapter = error_adapter(
        TargetUnavailable(TRANSPORT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME
    )

    run_result = run_suite(fast_config, selection, adapter, "M", fast_config.seed)

    assert_one_target_loss_result_per_affected_check(run_result, contracts)
    assert_counts(
        run_result,
        passed=len(witnesses),
        failed=0,
        skipped=0,
        error=len(affected),
    )
    assert tuple(sorted({result.check_id for result in run_result.results})) == tuple(
        sorted(contract.id for contract in contracts)
    )
    for check_id in affected:
        (loss,) = results_for(run_result, check_id)
        assert loss.status is Status.ERROR
        assert loss.reason is Reason.TARGET_UNAVAILABLE
        assert loss.decoded["error_type"] == TargetUnavailable.__name__
        assert TRANSPORT_FAILURE_TEXT in loss.decoded["error_text"]
        # The result the runner built is persisted like any other failing case, so the
        # transport failure is inspectable from the report alone.
        assert loss.evidence_dir is not None
        assert (fast_config.report_dir / loss.evidence_dir / "decoded.json").is_file()
    for check_id in witnesses:
        (witness,) = results_for(run_result, check_id)
        assert witness.status is Status.PASSED
    assert exit_code_for(run_result) == EXIT_INTERNAL


def test_a_timed_out_target_files_one_error_per_affected_check_naming_the_timeout(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
) -> None:
    """A timeout is reported the same way, and its evidence names the configured timeout: the value
    is the one the resolved ``Config`` declares, so retuning it cannot leave this behind."""
    selection = Selection(checks=MINIMAL_LOSS_SELECTION)
    contracts = selected_contracts(registry, selection)
    affected = target_dependent_ids(contracts)
    assert affected
    adapter = error_adapter(
        TargetTimeout(TIMEOUT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME
    )
    configured_timeout = fast_config.targets[CONFIGURED_TARGET_NAME].timeout_s

    run_result = run_suite(fast_config, selection, adapter, "M", fast_config.seed)

    assert_one_target_loss_result_per_affected_check(run_result, contracts)
    assert_counts(
        run_result,
        passed=len(target_independent_ids(contracts)),
        failed=0,
        skipped=0,
        error=len(affected),
    )
    for result in error_results(run_result):
        assert result.reason is Reason.TARGET_TIMEOUT
        assert result.decoded["error_type"] == TargetTimeout.__name__
        assert result.decoded["timeout_s"] == pytest.approx(float(configured_timeout))
        assert TIMEOUT_FAILURE_TEXT in result.decoded["error_text"]
        # "The number appears" is asserted on the serialised body, which is what a reader
        # of summary.json or decoded.json actually sees.
        assert repr(float(configured_timeout)) in json.dumps(result.as_dict(), sort_keys=True)
    assert exit_code_for(run_result) == EXIT_INTERNAL


# The run continues: the target-independent checks still run and still pass


def test_the_run_continues_to_the_checks_that_need_no_target(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    counting_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first case loses the target; every later check is still decided and reported. One
    recorded call proves the losing check ran once and the target-independent checks zero times."""
    echo_decode(monkeypatch, patch_decode, make_decode_result)
    selection = Selection(checks=LOSS_SELECTION)
    contracts = selected_contracts(registry, selection)
    affected = target_dependent_ids(contracts)
    witnesses = target_independent_ids(contracts)
    counting_adapter.raise_on_call(1, TargetUnavailable(TRANSPORT_FAILURE_TEXT))

    run_result = run_suite(fast_config, selection, counting_adapter, "M", fast_config.seed)

    assert_one_target_loss_result_per_affected_check(run_result, contracts)
    assert tuple(sorted({result.check_id for result in run_result.results})) == tuple(
        sorted(contract.id for contract in contracts)
    )
    assert_counts(
        run_result,
        passed=len(witnesses),
        failed=0,
        skipped=0,
        error=len(affected),
    )
    for check_id in witnesses:
        (witness,) = results_for(run_result, check_id)
        assert witness.status is Status.PASSED
    # One invocation in total: the check that met the loss was not retried, and neither
    # witness asked the target for anything.
    assert counting_adapter.count == 1
    assert exit_code_for(run_result) == EXIT_INTERNAL


def test_the_target_independent_checks_pass_with_zero_encode_invocations(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    counting_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CL-ARTIFACTS`` and ``PF-DECODE`` need nothing of the target, which makes them usable as
    witnesses: were either to need it, a lost-target run would have no independent check left."""
    echo_decode(monkeypatch, patch_decode, make_decode_result)
    selection = Selection(checks=WITNESS_SELECTION)
    contracts = selected_contracts(registry, selection)
    assert target_dependent_ids(contracts) == ()
    assert tuple(sorted(contract.id for contract in contracts)) == tuple(
        sorted(TARGET_INDEPENDENT_WITNESSES)
    )

    run_result = run_suite(fast_config, selection, counting_adapter, "M", fast_config.seed)

    assert counting_adapter.count == 0
    assert_counts(run_result, passed=len(contracts), failed=0, skipped=0, error=0)
    assert exit_code_for(run_result) == EXIT_OK

    limits = fast_config.performance
    (decode_case,) = results_for(run_result, "PF-DECODE")
    assert decode_case.status is Status.PASSED
    assert decode_case.expected == {
        "decode_image_px": limits.decode_image_px,
        "decode_p95_ms": limits.decode_p95_ms,
        "decode_samples": limits.decode_samples,
        "warmup": limits.warmup,
    }
    metrics = decode_case.decoded["metrics"]
    assert metrics["samples"] == limits.decode_samples
    assert metrics["image_px"] == limits.decode_image_px
    assert metrics["warmup"] == limits.warmup
    assert 0.0 <= metrics["decode_p95_ms"] <= limits.decode_p95_ms
    # The decode seam was stubbed, so this is the count of stubbed decodes the case asked
    # for -- the declared ``warmup + decode_samples`` -- and no real decoder ran.
    assert patch_decode.count == limits.warmup + limits.decode_samples

    (cleanup_case,) = results_for(run_result, "CL-ARTIFACTS")
    assert cleanup_case.status is Status.PASSED
    assert cleanup_case.evidence_dir is None


# The report a target-loss run leaves behind


def test_a_target_loss_run_writes_a_schema_complete_report(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
    report_dir: Path,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run returns a model and the three writers serialise it despite the loss: raising out so
    nothing is written would leave CI without the schema-complete report it receives."""
    echo_decode(monkeypatch, patch_decode, make_decode_result)
    selection = Selection(checks=LOSS_SELECTION)
    contracts = selected_contracts(registry, selection)
    adapter = error_adapter(
        TargetUnavailable(TRANSPORT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME
    )

    run_result = run_suite(fast_config, selection, adapter, "M", fast_config.seed)
    summary = assert_summary_schema_complete(run_result)
    assert_one_target_loss_result_per_affected_check(run_result, contracts)

    paths = write_all_reports(run_result, report_dir)

    assert paths.summary == report_dir / SUMMARY_FILENAME
    assert paths.junit == report_dir / JUNIT_FILENAME
    assert paths.html == report_dir / HTML_FILENAME
    written = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert sorted(written) == sorted(SUMMARY_FIELDS)
    assert written["counts"] == summary["counts"]
    assert [entry["check_id"] for entry in written["results"]] == [
        entry["check_id"] for entry in summary["results"]
    ]
    assert [entry["case_id"] for entry in written["results"]] == [
        entry["case_id"] for entry in summary["results"]
    ]
    document = ElementTree.parse(paths.junit).getroot()
    assert len(document.findall(".//testcase")) == len(run_result.results)
    assert len(document.findall(".//error")) == run_result.counts.error
    assert len(document.findall(".//failure")) == run_result.counts.failed
    page = paths.html.read_text(encoding="utf-8")
    assert page.strip()
    assert "<script" not in page.lower()


# Exit codes: 3 outranks 1, which outranks 0


def test_a_run_whose_check_errored_exits_with_the_internal_code(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
) -> None:
    """Nothing "failed", yet the process must exit 3: an errored check is internal."""
    selection = Selection(checks=MINIMAL_LOSS_SELECTION)
    contracts = selected_contracts(registry, selection)
    adapter = error_adapter(
        TargetUnavailable(TRANSPORT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME
    )

    run_result = run_suite(fast_config, selection, adapter, "M", fast_config.seed)

    assert run_result.counts.error == len(target_dependent_ids(contracts))
    assert run_result.counts.failed == 0
    assert exit_code_for(run_result) == EXIT_INTERNAL == 3


@pytest.mark.parametrize(
    ("failed", "error", "skipped", "expected_code"),
    [
        (0, 0, 0, EXIT_OK),
        (0, 0, 2, EXIT_OK),
        (1, 0, 2, EXIT_FAILED),
        (0, 1, 0, EXIT_INTERNAL),
        (1, 1, 2, EXIT_INTERNAL),
    ],
)
def test_the_exit_code_reads_error_before_failed_before_ok(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    failed: int,
    error: int,
    skipped: int,
    expected_code: int,
) -> None:
    """The documented precedence over the four tallies, with skips never deciding anything: exit
    code 2 is deliberately absent, being decided before a run exists."""
    run_result = run_with_tallies(
        make_case_result, make_run_result, failed=failed, error=error, skipped=skipped
    )
    assert exit_code_for(run_result) == expected_code


# The opposite policy: a declared rejection stays per case


def test_a_declared_rejection_stays_per_case_and_never_becomes_a_lost_target(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
    committed_corpus: Sequence[Any],
) -> None:
    """A rejection answers one request, so every case keeps its own failure and its text: folding it
    into the run-level path would collapse a corpus of failures into one error result."""
    selection = Selection(checks=SINGLE_CHECK_SELECTION, class_globs=(REJECTION_CLASS,))
    contracts = selected_contracts(registry, selection)
    assert target_dependent_ids(contracts) == tuple(
        sorted(contract.id for contract in contracts)
    )
    expected_case_ids = tuple(
        sorted(
            f"corpus:{record.id}"
            for record in committed_corpus
            if record.class_name == REJECTION_CLASS
        )
    )
    assert len(expected_case_ids) > 1, "more than one case is what makes this per-case"
    adapter = error_adapter(TargetRejected(REJECTION_TEXT), name=CONFIGURED_TARGET_NAME)

    run_result = run_suite(fast_config, selection, adapter, "M", fast_config.seed)

    assert tuple(result.case_id for result in run_result.results) == expected_case_ids
    assert TARGET_LOSS_CASE_ID not in {
        result.case_id for result in run_result.results
    }
    for result in run_result.results:
        assert result.status is Status.FAILED
        assert result.reason is Reason.TARGET_REJECTED
        assert result.decoded["rejection"] == REJECTION_TEXT
    assert_counts(
        run_result,
        passed=0,
        failed=len(expected_case_ids),
        skipped=0,
        error=0,
    )
    # One invocation per case: a rejection is a per-request answer, not a reason to retry
    # and not a reason to stop asking.
    assert len(adapter.requests) == len(expected_case_ids)
    assert exit_code_for(run_result) == EXIT_FAILED == 1


# No retry: the case that lost the target is attempted once


def test_the_case_that_loses_the_target_is_invoked_exactly_once(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
) -> None:
    """One loss, one attempt: a retry registers as a second invocation of that case, so a retrying
    runner hammering the same dead target would exceed the number of loss results it produced."""
    selection = Selection(checks=SINGLE_CHECK_SELECTION)
    contracts = selected_contracts(registry, selection)
    assert len(target_dependent_ids(contracts)) == 1
    adapter = error_adapter(
        TargetUnavailable(TRANSPORT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME
    )

    run_result = run_suite(fast_config, selection, adapter, "M", fast_config.seed)

    losses = error_results(run_result)
    assert len(losses) == len(target_dependent_ids(contracts))
    assert tuple(result.case_id for result in losses) == (TARGET_LOSS_CASE_ID,)
    assert len(adapter.requests) == len(losses)
    assert exit_code_for(run_result) == EXIT_INTERNAL


# --fail-fast is the only policy that shortens a run, and it still reports


def test_fail_fast_stops_launching_checks_and_still_returns_a_complete_model(
    fast_config: models.Config,
    registry: Mapping[str, CheckContract],
    error_adapter: Callable[..., Any],
) -> None:
    """The default continues through every selected check; ``--fail-fast`` stops *launching*
    them after a ``failed`` or ``error``, keeps that check whole, and still returns a model."""
    selection = Selection(checks=MINIMAL_LOSS_SELECTION)
    contracts = selected_contracts(registry, selection)
    affected = target_dependent_ids(contracts)
    witnesses = target_independent_ids(contracts)
    assert affected and witnesses
    first_affected = affected[0]
    witnesses_before_the_loss = tuple(
        check_id for check_id in witnesses if check_id < first_affected
    )

    continuing = run_suite(
        fast_config,
        selection,
        error_adapter(TargetUnavailable(TRANSPORT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME),
        "M",
        fast_config.seed,
    )
    stopped = run_suite(
        fast_config,
        selection,
        error_adapter(TargetUnavailable(TRANSPORT_FAILURE_TEXT), name=CONFIGURED_TARGET_NAME),
        "M",
        fast_config.seed,
        fail_fast=True,
    )

    assert tuple(sorted({result.check_id for result in continuing.results})) == tuple(
        sorted(contract.id for contract in contracts)
    )
    assert_one_target_loss_result_per_affected_check(continuing, contracts)
    assert len(error_results(continuing)) == len(affected)

    stopped_pairs = {(result.check_id, result.case_id) for result in stopped.results}
    continuing_pairs = {(result.check_id, result.case_id) for result in continuing.results}
    assert stopped_pairs < continuing_pairs
    assert tuple(result.check_id for result in error_results(stopped)) == (
        first_affected,
    )
    assert len(error_results(stopped)) < len(error_results(continuing))
    assert_counts(
        stopped,
        passed=len(witnesses_before_the_loss),
        failed=0,
        skipped=0,
        error=1,
    )
    assert_summary_schema_complete(stopped)
    assert exit_code_for(stopped) == EXIT_INTERNAL
