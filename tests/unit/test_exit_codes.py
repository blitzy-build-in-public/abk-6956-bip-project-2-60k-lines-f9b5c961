"""The process contract: one test per exit code, and the precedence between them.

:mod:`qrcheck.cli.exit` fixes four statuses and one order: ``0`` when the run completed with no
``failed`` and no ``error`` result, a ``skipped`` result never changing the outcome; ``1`` for at
least one ``failed`` and none errored; ``2`` for a configuration, selection, corpus-filter,
corpus-file, primary-decoder or target-availability fault decided **before any check runs**, so no
report is written and the diagnostic names the key, selector, filter, dependency or path to fix;
``3`` for an ``error`` result, an unwritable ``report_dir`` or an unhandled framework exception.
Precedence is ``2`` before execution and, during and after a run, ``3`` outranking ``1``
outranking ``0``, so exit code 2 never competes with a result -- which is why
:func:`qrcheck.cli.exit.exit_code_for` cannot return it and the tests below assert that property
directly. Every expected status is an ``EXIT_*`` constant rather than a bare integer, because
the vocabulary is decided in :mod:`qrcheck.cli.exit` alone, and the two properties the constants
cannot express -- precedence, and "2 is decided before a run exists" -- are asserted as
inequalities and as the absence of :data:`~qrcheck.cli.exit.EXIT_CONFIG` from the produced set.

The command line is driven through the real click group, so option parsing, the exit-code
plumbing of ``qrcheck/cli/{main,run,doctor,corpus,loc}.py``, the runner, the evidence store and
the three writers are the production objects; only the four boundaries that would otherwise
need a real generator, a real image, a real defect or a full disk are doubled -- the target
factory table ``qrcheck/cli/run.py`` indexes, the decode layer through ``patch_decode``, in the
three internal-error tests one registered check's ``cases`` generator -- the case body it yields
in two of them and the generator itself in the third, because a case, or a case enumeration,
that raises is a state nothing in the product can be asked to raise on demand -- and, in the
evidence-fault test, what the evidence store does with a result, because a write that fails
mid-run is a disk condition no permission bit can stage for root. A target lost *during* a run
belongs to ``tests/unit/test_runner_target_loss.py``. Nothing is skipped, ``xfail``-ed or
retried, no threshold is restated, and every artefact stays under the test's ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

import qrcheck.cli.run as run_module
import qrcheck.report.evidence as evidence_module
from qrcheck.checks import load_families
from qrcheck.cli.exit import (
    EXIT_CONFIG,
    EXIT_FAILED,
    EXIT_INTERNAL,
    EXIT_OK,
    exit_code_for,
)
from qrcheck.cli.main import main
from qrcheck.evidence.naming import EVIDENCE_DIR_NAME
from qrcheck.registry.check import REGISTRY, Case, CheckContract
from qrcheck.registry.resolve import CAPABILITY_CASE_ID
from qrcheck.report import HTML_FILENAME, JUNIT_FILENAME, SUMMARY_FILENAME
from qrcheck.report.model import (
    TRACEBACK_KEY,
    CaseResult,
    Counts,
    Reason,
    RunResult,
    Status,
)
from qrcheck.runner import RunPrerequisiteError
from qrcheck.targets.base import Capability

#: The registry module itself, needed for its private writable store. The package keeps
#: ``qrcheck.registry.check`` bound to this module -- it re-exports no name that would shadow a
#: submodule -- so the dotted form names this module too; it is fetched with
#: :func:`importlib.import_module` because that is this file's explicit dynamic form for
#: resolving a module object rather than binding a name at import time.
_REGISTRY_MODULE: Final[Any] = importlib.import_module("qrcheck.registry.check")

#: The three artefacts a completed run writes, from the one module that names them. A run
#: that exits 0 or 1 writes all three; a run that exits 2 writes none of them.
REPORT_ARTEFACTS: Final[tuple[str, ...]] = (
    SUMMARY_FILENAME,
    JUNIT_FILENAME,
    HTML_FILENAME,
)

#: The corpus class every corpus-driven run here is restricted to. ``nul-byte`` holds more
#: than one record -- so a failing run really is a run of several cases -- and every record
#: in it contains a NUL byte, which no stubbed decode payload can equal by accident.
CLASS_FILTER: Final[str] = "nul-byte"

#: A check whose requirements are ``{png, ec_level}`` and whose only optional dependency is
#: the primary decoder, which the run already requires. Selecting it against a target that
#: declares ``png`` alone therefore reaches exactly one skip -- the capability shortfall --
#: and no environment fact (rasterizer, secondary decoder) can redirect it to another reason.
CAPABILITY_SKIP_CHECK: Final[str] = "SY-VERSION-MIN"

#: The check whose case generator is replaced by a raising one, and the check that must still
#: run afterwards. The raising check must be one the runner walks *in identifier order*: it is
#: an ordinary check, chosen because it sorts before the continuation witness, so "a result
#: with a later check identifier" is a statement about the walk continuing rather than about
#: which check happened to be selected. ``CL-ARTIFACTS`` cannot serve here -- it asserts a
#: property of the whole run, so the runner executes it after every ordinary check and a fault in
#: it would prove nothing about the walk going on. ``IV-BAD-OPTION`` and ``IV-EMPTY`` are both
#: synthetic and read no corpus record, and the two capabilities
#: ``IV-BAD-OPTION`` requires are the ones the stub target declares, so the pair keeps the
#: continuation witness cheap and independent of the corpus.
RAISING_CHECK: Final[str] = "IV-BAD-OPTION"
CONTINUATION_CHECK: Final[str] = "IV-EMPTY"

#: The case identifier the replaced case generator yields, and the message its callable
#: raises. The message is asserted inside the recorded traceback, so a traceback that names
#: some other failure cannot satisfy the test.
RAISING_CASE_ID: Final[str] = "synthetic:internal-error"
RAISING_MESSAGE: Final[str] = "deliberate failure injected by the exit-code suite"

#: The case identifier the runner files a result for a check whose case *generator* raised
#: under, written as a literal so a renamed product constant fails this suite instead of being
#: followed by it. ``qrcheck.runner.CASES_FAILED_CASE_ID`` gives a constructed case a
#: ``synthetic:<name>`` identifier, and this one has to stay distinct from every name a check
#: builds for itself and from the target-loss identifier, or a reader could not tell a broken
#: generator from a broken target.
RAISING_GENERATOR_CASE_ID: Final[str] = "synthetic:cases"

#: The diagnostic a target-availability prerequisite carries. Injected at the runner
#: boundary, the one place the command's own mapping of that class is decided.
UNAVAILABLE_TARGET: Final[str] = "remote"
UNAVAILABLE_DETAIL: Final[str] = "connection refused"

#: The stable half of the diagnostic an unwritable ``report_dir`` produces when no run result
#: exists to stream. The probe decides before any case runs, so the JSON-lines contract for such
#: a directory is unreachable for that trigger and the empty stream has to be stated rather than
#: left to be read as a truncation; this is the phrase a consumer can key on.
NOTHING_STREAMED: Final[str] = "no result was streamed to stdout"

#: The payload every stubbed decode returns. Deliberately not the payload of any corpus
#: record, so a round-trip case fails on the payload comparison and for no other reason.
STUBBED_PAYLOAD: Final[bytes] = b"qrcheck exit-code suite: not the requested payload"

#: The diagnostic the patched evidence store raises once a case has produced its result: a disk
#: that filled up mid-run is the realistic cause of a write that fails after the walk started,
#: and no permission bit produces one for a process running as root.
EVIDENCE_FAILURE: Final[str] = "the evidence device is full"

#: The two count values the truth table ranges over, and the two presence flags it also
#: varies. Two of each is enough to distinguish "any" from "all" from "none" without turning
#: the table into a combinatorial exercise.
_COUNT_VALUES: Final[tuple[int, ...]] = (0, 1, 2)
_PRESENCE_VALUES: Final[tuple[int, ...]] = (0, 1)


# --------------------------------------------------------------------------- #
# Building runs whose tallies are the ones under test
# --------------------------------------------------------------------------- #


def tally_case_result(
    make_case_result: Callable[..., CaseResult],
    index: int,
    status: Status,
    reason: Reason | None,
) -> CaseResult:
    """Return one result of `status`, with an identity unique within its run. A
    :class:`~qrcheck.report.model.RunResult` refuses two results that share a ``(check_id,
    case_id)`` pair, so the index participates in the case identifier and the status
    participates in neither half of the pair.
    """
    return make_case_result(
        case_id=f"corpus:tally-{index:03d}",
        status=status,
        reason=reason,
    )


def run_with_tallies(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    passed: int,
    failed: int,
    skipped: int,
    error: int,
) -> RunResult:
    """Return a run whose four tallies are exactly the requested counts.
    :class:`~qrcheck.report.model.RunResult` recomputes its tallies from its results and
    refuses a record whose counts disagree with them, so results are the only way to state a
    tally.
    """
    plan: tuple[tuple[int, Status, Reason | None], ...] = (
        (passed, Status.PASSED, None),
        (failed, Status.FAILED, Reason.VALUE_MISMATCH),
        (skipped, Status.SKIPPED, Reason.UNSUPPORTED_CAPABILITY),
        (error, Status.ERROR, Reason.INTERNAL_ERROR),
    )
    results: list[CaseResult] = []
    for count, status, reason in plan:
        for _ in range(count):
            results.append(
                tally_case_result(
                    make_case_result,
                    len(results),
                    status=status,
                    reason=reason,
                )
            )
    return make_run_result(tuple(results))


# --------------------------------------------------------------------------- #
# Driving the command line through the three doubled boundaries
# --------------------------------------------------------------------------- #


def install_adapter(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    """Make ``qrcheck run`` build `adapter` for the configured target."""

    def build(target: object, *, name: str) -> Any:
        del target, name
        return adapter

    monkeypatch.setattr(run_module, "ADAPTER_TYPES", {"reference": build})


def replace_case_generator(
    monkeypatch: pytest.MonkeyPatch, check_id: str
) -> CheckContract:
    """Replace one registered check's cases with a single case that raises. The registry is read
    through a read-only view of a module-private store, so a replacement has to be written into
    that store: ``isolated_registry()`` is the documented way to do that, but it empties the
    registry first, and the assertion this serves is precisely that the *other* registered
    checks keep running.
    """

    def execute(context: object) -> CaseResult:
        del context
        raise RuntimeError(RAISING_MESSAGE)

    def cases(context: object) -> Iterator[Case]:
        del context
        return iter([Case(case_id=RAISING_CASE_ID, execute=execute)])

    load_families()
    replacement = dataclasses.replace(REGISTRY[check_id], cases=cases)
    monkeypatch.setitem(_REGISTRY_MODULE._REGISTRY, check_id, replacement)
    return replacement


def replace_cases_with_raising_generator(
    monkeypatch: pytest.MonkeyPatch, check_id: str
) -> CheckContract:
    """Replace one registered check's case generator with one that raises instead of yielding.

    The only way to reach the fault a check that cannot even enumerate its cases produces: the
    replacement raises from ``cases`` itself, before any :class:`Case` exists, so the runner's
    enumeration is what meets the exception. As in :func:`replace_case_generator`, the
    replacement is written into the module-private store because the other registered checks
    have to keep running -- an isolated registry would empty them and the continuation this
    test asserts would be impossible to observe.
    """

    def cases(context: object) -> Iterator[Case]:
        del context
        raise RuntimeError(RAISING_MESSAGE)

    load_families()
    replacement = dataclasses.replace(REGISTRY[check_id], cases=cases)
    monkeypatch.setitem(_REGISTRY_MODULE._REGISTRY, check_id, replacement)
    return replacement


# --------------------------------------------------------------------------- #
# Reading what a run left behind
# --------------------------------------------------------------------------- #


def read_summary(report_dir: Path) -> dict[str, Any]:
    """Return ``summary.json`` of the run under `report_dir`, parsed. The summary is the
    machine-readable statement of a run's tallies and results, so every exit-0/1/3 assertion
    reads the code and the counts from the same artefact: a command that exited 1 without
    reporting a failure, or exited 3 without reporting an error, would fail here even though
    the integer alone looked right.
    """
    return json.loads((report_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))


def assert_no_report(report_dir: Path) -> None:
    """Assert that no run described itself under `report_dir`. Exit code 2 is decided before any
    check runs, so the report root must hold neither an artefact nor an evidence directory: a
    report written on a configuration or prerequisite fault would be read as a completed run.
    """
    assert sorted(report_dir.rglob("*")) == [], (
        f"an exit-2 run wrote {sorted(report_dir.rglob('*'))} under {report_dir}; exit code 2 "
        "is decided before any check runs, so no report may be written for it"
    )


def json_object_lines(stdout: str) -> list[dict[str, Any]]:
    """Return one parsed object per line of `stdout`, asserting each line is an object. The
    contract for an unwritable report directory is JSON lines: one canonical object per result
    and no wrapper, so a CI consumer parses what the command could not file.
    """
    payloads: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parsed = json.loads(line)
        assert isinstance(parsed, dict), f"stdout line is not a JSON object: {line!r}"
        payloads.append(parsed)
    return payloads


# --------------------------------------------------------------------------- #
# Exit code 0, 1 and 3 as a pure function of the four tallies
# --------------------------------------------------------------------------- #


def documented_code(*, failed: int, error: int) -> int:
    """Return the status :func:`qrcheck.cli.exit.exit_code_for` prescribes for a run carrying these
    two tallies. The rule is stated once, here, in the form that function states it: an ``error``
    result outranks a ``failed`` one, either outranks a clean run, and the ``skipped`` tally is
    not consulted at all.
    """
    if error > 0:
        return EXIT_INTERNAL
    if failed > 0:
        return EXIT_FAILED
    return EXIT_OK


#: Every small combination of the four tallies, with the code :func:`exit_code_for` prescribes
#: for it and a readable identifier. ``passed`` and ``skipped`` vary too, because a run whose
#: outcome was read from a pass count or from a "nothing went wrong" flag would still pass a
#: table that held them fixed.
TALLY_TABLE: Final[tuple[Any, ...]] = tuple(
    pytest.param(
        failed,
        error,
        passed,
        skipped,
        documented_code(failed=failed, error=error),
        id=f"failed-{failed}-error-{error}-passed-{passed}-skipped-{skipped}",
    )
    for failed in _COUNT_VALUES
    for error in _COUNT_VALUES
    for passed in _PRESENCE_VALUES
    for skipped in _PRESENCE_VALUES
)


@pytest.mark.parametrize(
    ("failed", "error", "passed", "skipped", "expected"), TALLY_TABLE
)
def test_exit_code_for_reads_the_four_counts(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    failed: int,
    error: int,
    passed: int,
    skipped: int,
    expected: int,
) -> None:
    """Every combination of the tallies maps to the documented status, and never to 2. Exit code 2
    is decided before a run exists, so no run result can imply it: the
    :data:`~qrcheck.cli.exit.EXIT_CONFIG` assertion is as much part of the contract as the
    ``expected`` one, and it is what keeps a future refactor from folding a configuration fault
    into the run's own verdict.
    """
    run_result = run_with_tallies(
        make_case_result,
        make_run_result,
        passed=passed,
        failed=failed,
        skipped=skipped,
        error=error,
    )

    assert run_result.counts == Counts(
        passed=passed, failed=failed, skipped=skipped, error=error
    )
    assert exit_code_for(run_result) == expected
    assert exit_code_for(run_result) != EXIT_CONFIG


@pytest.mark.parametrize(
    ("failed", "error", "skipped", "expected"),
    [
        pytest.param(0, 0, 3, EXIT_OK, id="a-skip-is-not-a-failure"),
        pytest.param(1, 0, 0, EXIT_FAILED, id="one-failure-is-exit-1"),
        pytest.param(1, 1, 0, EXIT_INTERNAL, id="an-error-outranks-a-failure"),
    ],
)
def test_the_documented_corners(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    failed: int,
    error: int,
    skipped: int,
    expected: int,
) -> None:
    """The three corners :mod:`qrcheck.cli.exit` names, each with the code it fixes. A skipped
    result is counted in its own integer, so a run of skips and passes alone is a clean run; a
    single failure is exit 1; and a failure beside an error is exit 3, because the two arms are
    not additive and the internal fault is the one a reader has to see.
    """
    run_result = run_with_tallies(
        make_case_result,
        make_run_result,
        passed=0,
        failed=failed,
        skipped=skipped,
        error=error,
    )

    assert exit_code_for(run_result) == expected


def test_exit_code_for_never_returns_the_configuration_code(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
) -> None:
    """Across the whole table the function produces exactly three of the four codes."""
    produced = {
        exit_code_for(
            run_with_tallies(
                make_case_result,
                make_run_result,
                passed=passed,
                failed=failed,
                skipped=skipped,
                error=error,
            )
        )
        for failed in _COUNT_VALUES
        for error in _COUNT_VALUES
        for passed in _PRESENCE_VALUES
        for skipped in _PRESENCE_VALUES
    }

    assert produced == {EXIT_OK, EXIT_FAILED, EXIT_INTERNAL}
    assert EXIT_CONFIG not in produced


def test_exit_code_for_reads_the_tallies_not_a_pass_verdict(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
) -> None:
    """Three runs of equal size and different composition take three different codes. A function
    that answered "did anything go wrong" would return the same code for the second and the
    third run; a function that read a stored verdict would return the same code for all three.
    """
    clean = run_with_tallies(
        make_case_result, make_run_result, passed=3, failed=0, skipped=0, error=0
    )
    failing = run_with_tallies(
        make_case_result, make_run_result, passed=2, failed=1, skipped=0, error=0
    )
    errored = run_with_tallies(
        make_case_result, make_run_result, passed=2, failed=0, skipped=0, error=1
    )

    assert len(clean.results) == len(failing.results) == len(errored.results)
    assert (exit_code_for(clean), exit_code_for(failing), exit_code_for(errored)) == (
        EXIT_OK,
        EXIT_FAILED,
        EXIT_INTERNAL,
    )


def test_exit_three_outranks_exit_one_which_outranks_exit_zero(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
) -> None:
    """The precedence is the integer order, and it holds on a run with both faults. The numeric
    order is what an operating system reads, so the constants have to be ordered the way the
    precedence is described: a wrapper that combined statuses by taking the maximum -- the
    usual way a shell reports the worst of several steps -- must arrive at the internal fault
    rather than the failed one.
    """
    both = run_with_tallies(
        make_case_result, make_run_result, passed=0, failed=1, skipped=0, error=1
    )

    assert EXIT_INTERNAL > EXIT_FAILED > EXIT_OK >= 0
    assert exit_code_for(both) == EXIT_INTERNAL == max(EXIT_OK, EXIT_FAILED, EXIT_INTERNAL)


# --------------------------------------------------------------------------- #
# The entry point returns the code instead of exiting
# --------------------------------------------------------------------------- #


def test_main_returns_the_exit_code_as_an_integer(capsys: pytest.CaptureFixture[str]) -> None:
    """``main`` hands its caller an ``int``, which is the console script's whole contract. Both
    entry points -- the ``qrcheck`` console script and ``python -m qrcheck`` -- pass this value
    straight to :class:`SystemExit`, so a ``main`` that raised ``SystemExit`` itself would
    double-wrap it and one that returned ``None`` would report every run as a success.
    """
    code = main(["list"])

    assert isinstance(code, int) and not isinstance(code, bool)
    assert code == EXIT_OK
    assert "RT-ROUNDTRIP" in capsys.readouterr().out


def test_main_answers_three_when_the_dependency_probe_itself_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fault inside the import arm's probe cannot escape ``main``, which owes a code for it.

    Nothing encloses ``main``, so a probe raising while a broken import was being diagnosed cost
    the process its exit code: it died at 1 with a bare traceback.
    """

    def raise_probe_fault() -> None:
        raise RuntimeError("the dependency probe itself failed")

    def raise_import_error() -> None:
        raise ImportError("a pinned dependency is hidden")

    monkeypatch.setattr("qrcheck.dependencies.blocking_problem", raise_probe_fault)
    entry = importlib.import_module("qrcheck.cli.main")  # the dotted form names the module too
    monkeypatch.setattr(entry, "_register_commands", raise_import_error)

    code = main(["list"])

    stderr = capsys.readouterr().err
    assert code == EXIT_INTERNAL, "the probe's own fault cost main its exit code"
    assert "internal error: RuntimeError: the dependency probe itself failed" in stderr
    assert "Traceback" in stderr


# --------------------------------------------------------------------------- #
# Exit code 0: a run that skipped a check still succeeded
# --------------------------------------------------------------------------- #


def test_exit_zero_when_the_only_selected_check_is_skipped(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """A capability shortfall is a skip, and a skip is neither a failure nor a pass. The target
    declares ``png`` alone, so the selected check -- which also demands ``ec_level`` -- is
    resolved to ``skipped``/``unsupported_capability`` *before* execution: the run exits 0, the
    skip is tallied in its own integer, and the target is never asked for a symbol.
    """
    stub_adapter.capabilities = frozenset({Capability.png})
    install_adapter(monkeypatch, stub_adapter)

    result = invoke_cli("run", "--checks", CAPABILITY_SKIP_CHECK)

    assert result.exit_code == EXIT_OK
    summary = read_summary(report_dir)
    assert sorted(summary["counts"]) == ["error", "failed", "passed", "skipped"]
    assert summary["counts"] == {"passed": 0, "failed": 0, "skipped": 1, "error": 0}
    assert [
        (entry["check_id"], entry["status"], entry["reason"])
        for entry in summary["results"]
    ] == [
        (
            CAPABILITY_SKIP_CHECK,
            Status.SKIPPED.value,
            Reason.UNSUPPORTED_CAPABILITY.value,
        )
    ]
    (entry,) = summary["results"]
    evidence_dir = entry["evidence_dir"]
    assert isinstance(evidence_dir, str) and evidence_dir, (
        f"a skipped-with-reason case records its evidence directory; got {evidence_dir!r}"
    )
    assert not Path(evidence_dir).is_absolute(), evidence_dir
    case_dir = report_dir / evidence_dir
    assert sorted(path.name for path in case_dir.iterdir()) == ["decoded.json", "request.json"], (
        f"a whole-check skip produced no symbol, so {case_dir} holds the two JSON artefacts"
    )
    request = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
    assert (request["check_id"], request["case_id"], request["seed"]) == (
        CAPABILITY_SKIP_CHECK,
        CAPABILITY_CASE_ID,
        summary["seed"],
    ), f"request.json does not identify the case the skip belongs to: {request!r}"
    decoded = json.loads((case_dir / "decoded.json").read_text(encoding="utf-8"))
    assert isinstance(decoded, dict), f"decoded.json does not hold a JSON object: {decoded!r}"
    # The skip was decided before execution, so no request ever reached the target.
    assert stub_adapter.requests == []
    for artefact in REPORT_ARTEFACTS:
        assert (report_dir / artefact).is_file(), (
            f"a completed run must write {artefact}; {report_dir} holds "
            f"{sorted(path.name for path in report_dir.iterdir())}"
        )
    # The JUnit document carries the skip as a skip, so a CI consumer cannot read it as a pass.
    assert "<skipped" in (report_dir / JUNIT_FILENAME).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Exit code 1: a run that failed and did not error
# --------------------------------------------------------------------------- #


def test_exit_one_when_a_check_fails_without_erroring(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    committed_corpus: Sequence[Any],
    report_dir: Path,
) -> None:
    """A failed check and nothing worse is exit 1, with every artefact still written. The decode
    layer is stubbed to return a payload that is not the one requested, which is the one defect
    ``RT-ROUNDTRIP`` asserts against: every case of the selected class fails with
    ``value_mismatch``, no case errors, the process exits 1 and the full report exists -- the
    difference between "the generator is wrong" and "the framework broke" is the whole reason
    the two codes are distinct.
    """
    patch_decode(make_decode_result(payload_bytes=STUBBED_PAYLOAD))
    install_adapter(monkeypatch, stub_adapter)
    expected_failures = sum(
        1 for record in committed_corpus if record.class_name == CLASS_FILTER
    )
    assert expected_failures > 1, (
        f"the {CLASS_FILTER} class must hold more than one record for this run to be a run "
        "of several cases, not a single one"
    )

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)

    assert result.exit_code == EXIT_FAILED
    summary = read_summary(report_dir)
    assert summary["counts"]["failed"] == expected_failures
    assert summary["counts"]["error"] == 0
    assert summary["counts"]["passed"] == 0
    assert {entry["status"] for entry in summary["results"]} == {Status.FAILED.value}
    assert {entry["reason"] for entry in summary["results"]} == {
        Reason.VALUE_MISMATCH.value
    }
    # One encode per case: a failure is a verdict, never a reason to try again.
    assert len(stub_adapter.requests) == expected_failures
    for artefact in REPORT_ARTEFACTS:
        assert (report_dir / artefact).is_file()
    assert "<failure" in (report_dir / JUNIT_FILENAME).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Exit code 2: six faults decided before any check runs, and no report
# --------------------------------------------------------------------------- #


def test_exit_two_for_an_unknown_configuration_key(
    invoke_cli: Callable[..., Any],
    toml_config_file: Callable[..., Path],
    report_dir: Path,
) -> None:
    """A document with a key the schema does not declare is refused, and the key is named. The
    diagnostic has to name the key, because exit code 2 exists to tell a reader what to fix;
    and the report root has to stay empty, because a report written for a configuration this
    run never accepted would describe a run that did not happen.
    """
    unknown_key = "mystery_probe_key"
    config = toml_config_file(text=f"seed = 1\n{unknown_key} = 2\n")

    result = invoke_cli("run", config=config)

    assert result.exit_code == EXIT_CONFIG
    assert unknown_key in result.stderr
    assert_no_report(report_dir)


def test_exit_two_for_a_selection_that_matches_nothing(
    invoke_cli: Callable[..., Any],
    report_dir: Path,
) -> None:
    """A selector that names no registered check is echoed verbatim, and nothing runs. The
    selector is repeated rather than paraphrased, so a reader of a CI log can paste it back;
    and the sweep happens before the target is built, so this needs no double at all.
    """
    selector = "NOPE-*"

    result = invoke_cli("run", "--checks", selector)

    assert result.exit_code == EXIT_CONFIG
    assert selector in result.stderr
    assert_no_report(report_dir)


def test_exit_two_for_a_corpus_filter_that_matches_nothing(
    invoke_cli: Callable[..., Any],
    report_dir: Path,
) -> None:
    """A ``--class`` filter that leaves no record is a prerequisite fault, not an empty run. A run
    over zero records would satisfy every assertion it has and prove nothing, which is why the
    runner refuses it before the walk and the command exits 2 naming the globs.
    """
    corpus_glob = "no-such-class-*"

    result = invoke_cli(
        "run", "--checks", "RT-ROUNDTRIP", "--class", corpus_glob
    )

    assert result.exit_code == EXIT_CONFIG
    assert corpus_glob in result.stderr
    assert_no_report(report_dir)


def test_exit_two_for_a_missing_primary_decoder(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """Without the primary decoder the run refuses to start rather than skipping round-trip. The
    diagnostic has to name the distribution and the lock file that pins it, because a missing
    required decoder is an environment fault to install, not a target to change; and no report
    may be written, so ``RT-ROUNDTRIP`` is never silently skipped.
    """
    monkeypatch.setitem(sys.modules, "zxingcpp", None)
    install_adapter(monkeypatch, stub_adapter)

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)

    assert result.exit_code == EXIT_CONFIG
    assert "zxing-cpp" in result.stderr
    assert "requirements.lock" in result.stderr
    assert_no_report(report_dir)


def test_exit_two_for_a_missing_corpus_file(
    invoke_cli: Callable[..., Any],
    toml_config_file: Callable[..., Path],
    report_dir: Path,
) -> None:
    """An unreadable corpus is a prerequisite fault, and the path is named."""
    missing_corpus = "corpus/absent.jsonl"
    config = toml_config_file(corpus_path=missing_corpus)

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", config=config)

    assert result.exit_code == EXIT_CONFIG
    assert Path(missing_corpus).name in result.stderr
    assert_no_report(report_dir)


def test_exit_two_for_a_target_unavailable_at_preflight(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """A target found unavailable *before* any case runs is exit 2, not exit 3. The runner's
    prerequisite exception is where a pre-flight availability fault surfaces: the target could
    not be reached, or was never usable, so no case ran and there is no run to report -- which
    is exactly the condition the exit table puts at code 2.
    """

    def refuse(*args: object, **kwargs: object) -> RunResult:
        del args, kwargs
        raise RunPrerequisiteError(
            f"target {UNAVAILABLE_TARGET!r} is unavailable: {UNAVAILABLE_DETAIL}"
        )

    install_adapter(monkeypatch, stub_adapter)
    monkeypatch.setattr(run_module, "run_suite", refuse)

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)

    assert result.exit_code == EXIT_CONFIG
    assert UNAVAILABLE_TARGET in result.stderr
    assert UNAVAILABLE_DETAIL in result.stderr
    assert_no_report(report_dir)
    # The target the command built is still closed on the prerequisite path: a loopback
    # client or a scratch directory must not outlive the command that opened it.
    assert stub_adapter.closed is True


#: A ``library`` target whose ``kwargs_map`` names a field :class:`GenerateRequest` does not
#: have. ``qrcheck.config.schema`` accepts the mapping because it checks that mapping's *shape*
#: -- every key and value non-empty text -- and never its vocabulary, which belongs to the
#: adapter that has to pass the fields on.
UNKNOWN_KWARGS_FIELD_DOCUMENT: Final[str] = (
    'default_adapter = "local"\n'
    "\n"
    "[targets.local]\n"
    'type = "library"\n'
    'callable = "qrcheck_no_such_module:build"\n'
    'returns = "raw_bytes"\n'
    'kwargs_map = { payload = "text", not_a_request_field = "x" }\n'
    'capabilities = ["png"]\n'
)

#: A ``cli`` target that asks ``metadata_map`` to read the target's JSON metadata from stdout
#: while ``returns = "base64"`` puts the artefact there. Both keys are individually valid, so
#: only the adapter can see that they would consume one stream twice.
STDOUT_METADATA_MAP_DOCUMENT: Final[str] = (
    'default_adapter = "local"\n'
    "\n"
    "[targets.local]\n"
    'type = "cli"\n'
    'command = ["qrcheck-no-such-target"]\n'
    'payload_arg = "-p {}"\n'
    'output_arg = "-o {}"\n'
    'returns = "base64"\n'
    'metadata_map = { version = "version" }\n'
    'capabilities = ["png"]\n'
)


@pytest.mark.parametrize(
    ("document", "offending_key"),
    [
        (UNKNOWN_KWARGS_FIELD_DOCUMENT, "targets.local.kwargs_map"),
        (STDOUT_METADATA_MAP_DOCUMENT, "targets.local.metadata_map"),
    ],
    ids=["kwargs_map-names-an-unknown-field", "metadata_map-beside-a-stdout-returns"],
)
def test_exit_two_when_a_configuration_the_schema_accepts_cannot_build_an_adapter(
    invoke_cli: Callable[..., Any],
    toml_config_file: Callable[..., Path],
    report_dir: Path,
    document: str,
    offending_key: str,
) -> None:
    """A configuration fault the adapter finds at construction is exit 2, not exit 3. Both
    documents below are decided from the file alone -- one names a request field the framework
    does not have, the other makes two keys read one stream -- so they belong to the exit-2
    class the moment they are refused, even though the loader cannot see either. The key
    assertion is the one that keeps the test honest: the message has to name the mapping's own
    dotted key, because a diagnostic that named some other key would still exit 2 while telling
    the reader to edit the wrong line. Nothing is imported, spawned or decoded here: the fault
    is raised before the adapter resolves its callable, and no report may be written for a run
    that never started.
    """
    config = toml_config_file(text=document)

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", config=config)

    assert result.exit_code == EXIT_CONFIG
    assert offending_key in result.stderr
    assert_no_report(report_dir)


@pytest.mark.parametrize(
    ("document", "offending_key"),
    [
        (UNKNOWN_KWARGS_FIELD_DOCUMENT, "targets.local.kwargs_map"),
        (STDOUT_METADATA_MAP_DOCUMENT, "targets.local.metadata_map"),
    ],
    ids=["kwargs_map-names-an-unknown-field", "metadata_map-beside-a-stdout-returns"],
)
def test_exit_two_when_doctor_cannot_build_the_configured_adapter(
    invoke_cli: Callable[..., Any],
    toml_config_file: Callable[..., Path],
    report_dir: Path,
    document: str,
    offending_key: str,
) -> None:
    """``doctor`` and ``run`` must answer one configuration fault the same way: exit 2, naming it.

    Both documents are schema-valid -- no cross-field rule links ``metadata_map`` to ``returns``,
    and a ``kwargs_map``'s vocabulary belongs to the adapter, not to the schema -- so the fault
    exists only once the adapter is constructed, which is the moment ``doctor`` used to lose it in
    the generic framework-defect arm and exit 3. The agreement is the assertion: a pre-flight
    command that calls a configuration fault an internal error sends its reader to the framework
    instead of the key to edit, and neither command may write a report for it.
    """
    config = toml_config_file(text=document)

    doctor_result = invoke_cli("doctor", config=config)

    assert doctor_result.exit_code == EXIT_CONFIG
    assert offending_key in doctor_result.stderr

    run_result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", config=config)

    assert run_result.exit_code == EXIT_CONFIG
    assert_no_report(report_dir)


@pytest.mark.parametrize(
    "supplied",
    ["", "   "],
    ids=["empty", "whitespace-only"],
)
def test_exit_two_for_a_report_dir_the_configuration_file_would_refuse(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    report_dir: Path,
    tmp_path: Path,
    supplied: str,
) -> None:
    """A ``--report-dir`` the configuration file would refuse is refused on the command line.

    The option and the configuration file name one key, so the same value cannot be a
    configuration error in the file and a report root here: ``Path("")`` would otherwise
    resolve to the working directory, pass the writability probe and fill it with
    ``summary.json``, ``junit.xml``, ``index.html`` and an evidence tree on a run that exits 0,
    which is what a CI job passing an unset ``$OUT`` would do to its checkout. The refusal is
    exit 2 naming the option, decided before the probe, and the working directory -- pointed at
    this test's own ``tmp_path``, never the checkout -- holds neither an artefact nor an
    evidence directory afterwards. ``IV-EMPTY`` is selected so the refusal cannot depend on
    which check would have run, and no adapter is doubled because none is built.
    """
    monkeypatch.chdir(tmp_path)

    result = invoke_cli("run", "--checks", "IV-EMPTY", "--report-dir", supplied)

    assert result.exit_code == EXIT_CONFIG
    assert "--report-dir" in result.stderr, result.stderr
    # The fragment the schema's own rule returns names the value it refused, so the reader
    # sees which argument was at fault without this test restating the schema's wording.
    assert repr(supplied) in result.stderr, result.stderr
    assert_no_report(report_dir)
    for artefact in REPORT_ARTEFACTS:
        assert not (tmp_path / artefact).exists(), (
            f"--report-dir {supplied!r} wrote {artefact} into the working directory"
        )
    assert not (tmp_path / EVIDENCE_DIR_NAME).exists(), (
        f"--report-dir {supplied!r} laid an evidence tree in the working directory"
    )
#: A ``cli`` target whose ``payload_arg`` carries a one-character typo of ``{payload_b64}``. The
#: placeholder is outside the vocabulary the adapter substitutes, so the document is wrong on its
#: own terms: before the schema refused it, loading succeeded and the adapter raised a bare
#: ``ValueError`` at the first case, which the runner could only report as six ``internal_error``
#: results under exit 3 with a full report written.
UNKNOWN_PAYLOAD_PLACEHOLDER_DOCUMENT: Final[str] = (
    'default_adapter = "local"\n'
    "\n"
    "[targets.local]\n"
    'type = "cli"\n'
    'command = ["qrcheck-no-such-target"]\n'
    'payload_arg = "--payload-b64={payload_base64}"\n'
    'output_arg = "--output={}"\n'
    'returns = "file_path"\n'
    'capabilities = ["png"]\n'
)

#: A ``cli`` target that puts a real placeholder in a ``command`` word. ``command`` is a literal
#: argv, so the word would reach the target as the text ``{ec_level}``: the target rejects it and
#: every case fails ``target_rejected``, for a fault the document alone decides.
PLACEHOLDER_IN_COMMAND_DOCUMENT: Final[str] = (
    'default_adapter = "local"\n'
    "\n"
    "[targets.local]\n"
    'type = "cli"\n'
    'command = ["qrcheck-no-such-target", "--ec", "{ec_level}"]\n'
    'payload_arg = "--payload-b64={payload_b64}"\n'
    'output_arg = "--output={}"\n'
    'returns = "file_path"\n'
    'capabilities = ["png"]\n'
)


@pytest.mark.parametrize(
    ("document", "offending_key", "token"),
    [
        (
            UNKNOWN_PAYLOAD_PLACEHOLDER_DOCUMENT,
            "targets.local.payload_arg",
            "{payload_base64}",
        ),
        (
            PLACEHOLDER_IN_COMMAND_DOCUMENT,
            "targets.local.command",
            "{ec_level}",
        ),
    ],
    ids=["unknown-payload-template-placeholder", "placeholder-in-a-command-word"],
)
def test_exit_two_for_a_cli_placeholder_the_adapter_cannot_fill(
    invoke_cli: Callable[..., Any],
    toml_config_file: Callable[..., Path],
    report_dir: Path,
    document: str,
    offending_key: str,
    token: str,
) -> None:
    """A ``cli`` template the adapter cannot fill is exit 2, decided from the file alone.

    ``payload_arg`` and ``output_arg`` are templates over the eight placeholders a request carries
    plus the bare ``{}``, and ``command`` is a literal argv no adapter renders -- so an unknown
    placeholder in a template, and any placeholder at all in a command word, are faults the
    document states by itself. Both documents below were accepted before the schema held them to
    that vocabulary: the first surfaced as a bare ``ValueError`` from the adapter, which is exit 3
    with ``internal_error`` results and a written report, and the second was spliced verbatim into
    argv, so the target rejected it and every case failed. Exit code 2 is the code the contract
    gives this class, so the diagnostic has to name the dotted key and the token, and no report may
    be written for a run that never started.
    """
    config = toml_config_file(text=document)

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", config=config)

    assert result.exit_code == EXIT_CONFIG
    assert offending_key in result.stderr, result.stderr
    assert token in result.stderr, result.stderr
    assert_no_report(report_dir)


# --------------------------------------------------------------------------- #
# Exit code 3: an internal fault, the report, and the one time results stream
# --------------------------------------------------------------------------- #


def test_exit_three_when_a_check_raises_internally(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    report_dir: Path,
) -> None:
    """A raising case becomes one ``error`` result with its traceback, and the walk goes on. A
    command that stopped at the first exception would satisfy the first two and fail the third,
    which is the point of asserting it separately.
    """
    replace_case_generator(monkeypatch, RAISING_CHECK)
    install_adapter(monkeypatch, stub_adapter)

    result = invoke_cli("run", "--checks", f"{RAISING_CHECK},{CONTINUATION_CHECK}")

    assert result.exit_code == EXIT_INTERNAL
    summary = read_summary(report_dir)
    errored = [
        entry for entry in summary["results"] if entry["status"] == Status.ERROR.value
    ]
    assert len(errored) == 1, (
        "exactly one case raised, so exactly one result may carry 'error'; the run holds "
        f"{[(entry['check_id'], entry['status']) for entry in summary['results']]}"
    )
    failure = errored[0]
    assert (
        failure["check_id"],
        failure["case_id"],
        failure["reason"],
    ) == (
        RAISING_CHECK,
        RAISING_CASE_ID,
        Reason.INTERNAL_ERROR.value,
    )
    assert failure["decoded"]["error_type"] == "RuntimeError"
    traceback_text = failure["decoded"][TRACEBACK_KEY]
    assert "Traceback (most recent call last)" in traceback_text
    assert RAISING_MESSAGE in traceback_text

    later = [entry for entry in summary["results"] if entry["check_id"] > RAISING_CHECK]
    assert [entry["check_id"] for entry in later] == [CONTINUATION_CHECK], (
        "the walk must continue past an errored case; a run that stops at the first "
        "exception would leave no result filed after it"
    )
    assert all(entry["status"] != Status.ERROR.value for entry in later)
    assert summary["counts"]["error"] == 1
    for artefact in REPORT_ARTEFACTS:
        assert (report_dir / artefact).is_file()
    assert "<error" in (report_dir / JUNIT_FILENAME).read_text(encoding="utf-8")


def test_exit_three_when_a_check_cannot_enumerate_its_cases(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """A case *generator* that raises is contained exactly like a case body. The enumeration
    happens before any case can run, so the check contributes one ``synthetic:cases`` error
    result carrying its traceback and the walk moves on; a run that let the exception escape
    would build no :class:`~qrcheck.report.model.RunResult`, write no report and lose every
    result after the broken check.
    """
    replace_cases_with_raising_generator(monkeypatch, RAISING_CHECK)
    install_adapter(monkeypatch, stub_adapter)

    result = invoke_cli("run", "--checks", f"{RAISING_CHECK},{CONTINUATION_CHECK}")

    assert result.exit_code == EXIT_INTERNAL
    summary = read_summary(report_dir)
    errored = [
        entry for entry in summary["results"] if entry["status"] == Status.ERROR.value
    ]
    assert len(errored) == 1, (
        "a check that cannot enumerate its cases produces exactly one error result; the run "
        f"holds {[(entry['check_id'], entry['status']) for entry in summary['results']]}"
    )
    failure = errored[0]
    assert (
        failure["check_id"],
        failure["case_id"],
        failure["reason"],
    ) == (
        RAISING_CHECK,
        RAISING_GENERATOR_CASE_ID,
        Reason.INTERNAL_ERROR.value,
    )
    assert failure["decoded"]["error_type"] == "RuntimeError"
    traceback_text = failure["decoded"][TRACEBACK_KEY]
    assert "Traceback (most recent call last)" in traceback_text
    assert RAISING_MESSAGE in traceback_text

    later = [entry for entry in summary["results"] if entry["check_id"] > RAISING_CHECK]
    assert [entry["check_id"] for entry in later] == [CONTINUATION_CHECK], (
        "the walk must continue to the next check when one check cannot enumerate its cases; "
        "a run that stopped there would leave no result filed after the broken check"
    )
    assert all(entry["status"] != Status.ERROR.value for entry in later)
    assert summary["counts"]["error"] == 1
    for artefact in REPORT_ARTEFACTS:
        assert (report_dir / artefact).is_file(), (
            f"the report must still be written; {report_dir} holds "
            f"{sorted(path.name for path in report_dir.iterdir())}"
        )
    assert "<error" in (report_dir / JUNIT_FILENAME).read_text(encoding="utf-8")


def test_exit_three_when_the_report_directory_is_unwritable(
    invoke_cli: Callable[..., Any],
    unwritable_report_dir: Path,
    report_dir: Path,
) -> None:
    """A ``report_dir`` that cannot be created is exit 3, the path is in the message, and the
    empty stream is stated. A regular file stands where the directory belongs, so ``mkdir``
    fails for every user including root -- a permission bit would prove nothing in a container
    that runs as root. Writability is probed before any case runs, so nothing is streamed and
    the diagnostic has to say that no result exists: an unstated empty stdout is
    indistinguishable from a stream that was lost. The non-vacuous half of the same row is
    ``test_exit_three_streams_json_lines_when_the_report_cannot_be_written``.
    """
    result = invoke_cli("run", "--report-dir", str(unwritable_report_dir))

    assert result.exit_code == EXIT_INTERNAL
    assert str(unwritable_report_dir.resolve()) in result.stderr
    assert NOTHING_STREAMED in result.stderr, (
        "the probe decides before any case runs, so stderr must say that no result was "
        f"streamed; stderr was {result.stderr!r}"
    )
    assert json_object_lines(result.stdout) == []
    assert_no_report(report_dir)


def test_exit_three_streams_json_lines_when_the_report_cannot_be_written(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """A run whose report cannot be written still streams every result, one object per line. The
    writer is made to fail rather than the directory, because a directory the probe accepts and
    the writers then cannot use is the only state that has results *and* no report, and
    ``chmod`` cannot produce it for root.
    """
    stub_adapter.capabilities = frozenset({Capability.png})
    install_adapter(monkeypatch, stub_adapter)

    def fail_to_write(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise OSError("the report device is full")

    monkeypatch.setattr(run_module, "write_all_reports", fail_to_write)

    result = invoke_cli("run", "--checks", CAPABILITY_SKIP_CHECK)

    assert result.exit_code == EXIT_INTERNAL
    assert str(report_dir) in result.stderr
    streamed = json_object_lines(result.stdout)
    assert [entry["check_id"] for entry in streamed] == [CAPABILITY_SKIP_CHECK]
    assert {entry["status"] for entry in streamed} == {Status.SKIPPED.value}
    assert not (report_dir / SUMMARY_FILENAME).exists()


def test_exit_three_when_a_case_evidence_cannot_be_written(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    committed_corpus: Sequence[Any],
    report_dir: Path,
) -> None:
    """A store that cannot write costs each case its evidence, never the run its results.

    Every case of the selected class fails on the stubbed payload and then cannot have that
    failure recorded, which is the full disk a mid-walk write failure stands for: the fault has
    to become one ``error`` result per case, filed in that case's own place so the run never
    holds two results for one ``(check_id, case_id)`` pair, while the walk finishes, all three
    artefacts are written and the command exits 3. A run that let the ``OSError`` escape would
    write no report, stream an empty stdout and state that the run produced no result -- the one
    outcome the exit-code contract rules out for an internal fault, and for a report directory
    that turns unusable.
    """

    def fail_to_record(*args: object, **kwargs: object) -> str | None:
        del args, kwargs
        raise OSError(EVIDENCE_FAILURE)

    monkeypatch.setattr(evidence_module.EvidenceStore, "record", fail_to_record)
    patch_decode(make_decode_result(payload_bytes=STUBBED_PAYLOAD))
    install_adapter(monkeypatch, stub_adapter)
    expected_cases = sum(
        1 for record in committed_corpus if record.class_name == CLASS_FILTER
    )
    assert expected_cases > 1, (
        f"the {CLASS_FILTER} class must hold more than one record, or a discarded result could "
        "not be told from the run's whole output"
    )

    result = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)

    assert result.exit_code == EXIT_INTERNAL
    summary = read_summary(report_dir)
    # The counts are asserted, not the result list alone: the summary re-derives them from the
    # results it holds, so a result swallowed on the way out would leave every count short.
    assert summary["counts"] == {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": expected_cases,
    }
    assert len(summary["results"]) == expected_cases
    identities = [(entry["check_id"], entry["case_id"]) for entry in summary["results"]]
    assert len(set(identities)) == expected_cases, (
        "a case keeps exactly one result when its evidence cannot be written; the run holds "
        f"{identities}"
    )
    assert {entry["status"] for entry in summary["results"]} == {Status.ERROR.value}
    assert {entry["reason"] for entry in summary["results"]} == {
        Reason.INTERNAL_ERROR.value
    }
    for entry in summary["results"]:
        assert entry["evidence_dir"] is None, (
            "nothing was written, so no result may name an evidence directory"
        )
        assert entry["decoded"]["error_type"] == "OSError"
        traceback_text = entry["decoded"][TRACEBACK_KEY]
        assert "Traceback (most recent call last)" in traceback_text
        assert EVIDENCE_FAILURE in traceback_text
        # The check's own write failed first and was contained as an internal error, whose own
        # write then failed too; the displaced outcome is recorded so a reader can still tell a
        # lost failing case from a lost skipped one.
        assert entry["decoded"]["prevented_status"] == Status.ERROR.value
        assert entry["decoded"]["prevented_reason"] == Reason.INTERNAL_ERROR.value
    # One encode per case: containing a write fault re-runs nothing.
    assert len(stub_adapter.requests) == expected_cases
    for artefact in REPORT_ARTEFACTS:
        assert (report_dir / artefact).is_file(), (
            "a contained evidence fault must still leave a complete report; "
            f"{report_dir} holds {sorted(path.name for path in report_dir.iterdir())}"
        )
    assert "<error" in (report_dir / JUNIT_FILENAME).read_text(encoding="utf-8")
    assert NOTHING_STREAMED not in result.stderr, (
        "the command held results and wrote a report, so its diagnostic must not claim it had "
        f"none; stderr was {result.stderr!r}"
    )


def test_exit_three_outranks_exit_one_when_a_run_both_fails_and_errors(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    committed_corpus: Sequence[Any],
    report_dir: Path,
) -> None:
    """A run carrying a failure *and* an error exits 3, and the summary reports both counts. The
    precedence :mod:`qrcheck.cli.exit` states is only meaningful when both faults are present at
    once: the internal fault is the worse of the two, so it decides the status, but the failure is
    still counted and still reported -- a report that collapsed them, or a code that took the
    failure, would each describe the run wrongly.
    """
    expected_failures = sum(
        1 for record in committed_corpus if record.class_name == CLASS_FILTER
    )
    assert expected_failures > 1
    replace_case_generator(monkeypatch, RAISING_CHECK)
    patch_decode(make_decode_result(payload_bytes=STUBBED_PAYLOAD))
    install_adapter(monkeypatch, stub_adapter)

    result = invoke_cli(
        "run", "--checks", f"{RAISING_CHECK},RT-ROUNDTRIP", "--class", CLASS_FILTER
    )

    assert result.exit_code == EXIT_INTERNAL
    summary = read_summary(report_dir)
    assert summary["counts"]["error"] == 1
    assert summary["counts"]["failed"] == expected_failures
    assert {entry["status"] for entry in summary["results"]} == {
        Status.ERROR.value,
        Status.FAILED.value,
    }


# --------------------------------------------------------------------------- #
# A reused report root describes exactly one run
# --------------------------------------------------------------------------- #


def evidence_tree(report_dir: Path) -> list[str]:
    """Return every path under the report root's evidence directory, relative to it and sorted."""
    root = report_dir / EVIDENCE_DIR_NAME
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def test_a_second_run_into_one_report_root_replaces_the_evidence_tree(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    report_dir: Path,
) -> None:
    """A passing run into a root the previous run filled leaves no evidence directory behind."""
    patch_decode(make_decode_result(payload_bytes=STUBBED_PAYLOAD))
    install_adapter(monkeypatch, stub_adapter)
    failing = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)
    assert failing.exit_code == EXIT_FAILED
    assert evidence_tree(report_dir), f"the failing run filed no evidence under {report_dir}"

    def echo(_image_bytes: bytes, request: Any, **_kwargs: Any) -> Any:
        """Return the decoded evidence of a conforming decoder for `request`."""
        return make_decode_result(payload_bytes=request.payload)

    monkeypatch.setattr("qrcheck.decode.decode_symbol", echo)
    passing = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)

    assert passing.exit_code == EXIT_OK
    summary = read_summary(report_dir)
    assert summary["counts"]["failed"] == summary["counts"]["error"] == 0
    assert summary["counts"]["passed"] > 0
    assert {entry["evidence_dir"] for entry in summary["results"]} == {None}
    assert all((report_dir / name).is_file() for name in REPORT_ARTEFACTS)
    assert evidence_tree(report_dir) == [], f"stale evidence survived in {report_dir}"


def test_a_run_that_exits_two_leaves_the_previous_report_and_evidence_untouched(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    report_dir: Path,
) -> None:
    """A rejected run clears nothing, because exit code 2 writes no report at all."""
    patch_decode(make_decode_result(payload_bytes=STUBBED_PAYLOAD))
    install_adapter(monkeypatch, stub_adapter)
    first = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", CLASS_FILTER)
    assert first.exit_code == EXIT_FAILED
    summary_before = (report_dir / SUMMARY_FILENAME).read_bytes()
    tree_before = evidence_tree(report_dir)
    assert tree_before

    rejected = invoke_cli("run", "--checks", "NOPE-*")
    empty = invoke_cli("run", "--checks", "RT-ROUNDTRIP", "--class", "no-such-class-*")

    assert rejected.exit_code == empty.exit_code == EXIT_CONFIG
    assert (report_dir / SUMMARY_FILENAME).read_bytes() == summary_before
    assert evidence_tree(report_dir) == tree_before


# --------------------------------------------------------------------------- #
# The other four verbs answer in the same vocabulary
# --------------------------------------------------------------------------- #


def unreachable_target_document() -> str:
    """Return a configuration document naming a target that can never be reached. The ``library``
    adapter resolves its callable from configuration, so a module that does not exist is an
    unreachable target that needs no port, no process and no double.
    """
    return (
        'default_adapter = "local"\n'
        "\n"
        "[targets.local]\n"
        'type = "library"\n'
        'callable = "qrcheck_no_such_module:build"\n'
        'returns = "raw_bytes"\n'
        "kwargs_map = {}\n"
        'capabilities = ["png"]\n'
    )


def test_the_other_subcommands_answer_in_the_same_vocabulary(
    invoke_cli: Callable[..., Any],
    toml_config_file: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``list``, ``doctor``, ``corpus build`` and ``loc`` use the same four statuses. Four short
    invocations, so that a code chosen in one command cannot drift from the code its sibling
    returns for the same kind of fault: ``list`` always succeeds, ``doctor`` reports an
    unreachable target as a configuration-class failure (that is the product's pre-flight
    target-availability gate, and the reason a *run* against a dying target is the exit-3 state
    instead), and the two argument faults -- a negative seed and a path that does not exist --
    are exit 2 rather than the exit 1 a shell would expect from a failing command.
    """
    list_result = invoke_cli("list")
    assert list_result.exit_code == EXIT_OK

    doctor_config = toml_config_file(text=unreachable_target_document())
    doctor_result = invoke_cli("doctor", config=doctor_config)
    assert doctor_result.exit_code == EXIT_CONFIG
    assert "unreachable" in doctor_result.stderr

    missing_path = tmp_path / "no-such-tree"
    loc_result = invoke_cli("loc", "--path", str(missing_path))
    assert loc_result.exit_code == EXIT_CONFIG
    assert str(missing_path) in loc_result.stderr

    seed_result = invoke_cli("corpus", "build", "--seed", "-1")
    assert seed_result.exit_code == EXIT_CONFIG
    assert "-1" in seed_result.stderr
