"""The process contract: one test per exit code, and the precedence between them.

AAP §0.3.4 fixes four process statuses and one order, and this module is the gate that holds
them at the command line rather than at a model boundary:

    0  the run completed with no ``failed`` and no ``error`` result; a ``skipped`` result is
       permitted and never changes the outcome
    1  at least one check ``failed`` and none errored
    2  a configuration, selection, corpus-filter, corpus-file, primary-decoder or
       target-availability fault, decided **before any check runs**, so no report is written
       and the diagnostic names the key, selector, filter, dependency or path to fix
    3  any check ``error`` result, an unwritable ``report_dir``, or an unhandled framework
       exception: the report is written, except for the unwritable directory, where the
       results the run holds are streamed to stdout as one JSON object per line

Precedence is ``2`` before execution and, during and after a run, ``3`` outranks ``1`` which
outranks ``0``. Exit code 2 therefore never competes with a result, which is why
:func:`qrcheck.cli.exit.exit_code_for` cannot return it and why the tests below assert that
property directly.

Where the numbers come from
---------------------------
Every expected status is an ``EXIT_*`` constant, never a bare integer: the vocabulary is
decided in :mod:`qrcheck.cli.exit` alone, and a test that spelled ``2`` could not tell a
renamed constant from a passing case. The two properties the constants cannot express --
precedence and "2 is decided before a run exists" -- are asserted as inequalities and as
the absence of :data:`~qrcheck.cli.exit.EXIT_CONFIG` from the produced set.

How the suite stays fast, and why these three doubles are the honest ones
------------------------------------------------------------------------
The command line is driven through the real click group (``invoke_cli``), so the option
parsing, the exit-code plumbing of ``qrcheck/cli/{main,run,doctor,corpus,loc}.py``, the
runner, the evidence store and the three writers are all the production objects. Only the
three boundaries that would otherwise need a real generator, a real image or a real defect
are doubled:

* the target, by replacing the one factory table ``qrcheck/cli/run.py`` looks a target up in
  with a mapping that returns the suite's :class:`~qrcheck.targets.base.Adapter` stub -- the
  same table production code indexes, so the command's own resolution still runs;
* the decode layer, through the suite's ``patch_decode`` fixture, because every check under
  test asserts an exit code and not a decoding property;
* one registered check's case generator, in the two internal-error tests, because the state
  under test is "a case raised" and nothing in the product can be asked to raise on demand.

What deliberately is not here: a target lost *during* a run is the one-error-per-check path
that exits 3 and is asserted in ``tests/unit/test_runner_target_loss.py``, and the per-check
adapter-invocation counts that prove no retry exists are asserted in
``tests/unit/test_checks_negative.py``. This module pins the status of the process, not the
policy of the runner or of a check.

No user-specified rules exist for this project (AAP §0.12), so the binding constraints are
the request's own: no check is skipped, ``xfail``-ed or retried, no threshold is restated
here, no run reaches a network beyond the conftest's loopback guard, and every artefact each
run produces stays under the test's ``tmp_path``.
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
from qrcheck.checks import load_families
from qrcheck.cli.exit import (
    EXIT_CONFIG,
    EXIT_FAILED,
    EXIT_INTERNAL,
    EXIT_OK,
    exit_code_for,
)
from qrcheck.cli.main import main
from qrcheck.registry.check import REGISTRY, Case, CheckContract
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

#: The registry module itself, needed for its private writable store. Imported through
#: :func:`importlib.import_module` rather than ``import qrcheck.registry.check as ...``
#: because the package namespace binds ``check`` to the registration decorator, so the
#: dotted form names a function and not this module.
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
#: run afterwards. ``CL-ARTIFACTS`` sorts before every other identifier in the product, so
#: "a result with a later check identifier" is a statement about the walk continuing rather
#: than about which check happened to be selected; ``IV-EMPTY`` requires only ``png`` and one
#: synthetic case, which keeps the continuation witness cheap and independent of the corpus.
RAISING_CHECK: Final[str] = "CL-ARTIFACTS"
CONTINUATION_CHECK: Final[str] = "IV-EMPTY"

#: The case identifier the replaced case generator yields, and the message its callable
#: raises. The message is asserted inside the recorded traceback, so a traceback that names
#: some other failure cannot satisfy the test.
RAISING_CASE_ID: Final[str] = "synthetic:internal-error"
RAISING_MESSAGE: Final[str] = "deliberate failure injected by the exit-code suite"

#: The diagnostic a target-availability prerequisite carries. Injected at the runner
#: boundary, the one place the command's own mapping of that class is decided.
UNAVAILABLE_TARGET: Final[str] = "remote"
UNAVAILABLE_DETAIL: Final[str] = "connection refused"

#: The payload every stubbed decode returns. Deliberately not the payload of any corpus
#: record, so a round-trip case fails on the payload comparison and for no other reason.
STUBBED_PAYLOAD: Final[bytes] = b"qrcheck exit-code suite: not the requested payload"

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
    """Return one result of `status`, with an identity unique within its run.

    A :class:`~qrcheck.report.model.RunResult` refuses two results that share a
    ``(check_id, case_id)`` pair, so the index participates in the case identifier and the
    status participates in neither half of the pair.

    Args:
        make_case_result: The suite's ``CaseResult`` factory.
        index: A zero-based index that keeps the identity unique.
        status: The status the result carries.
        reason: The reason that status requires, or ``None`` for a pass, which is the one
            status the result model refuses a reason on.

    Returns:
        The constructed result.
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
    tally. That is what makes the truth table below a statement about the run the product
    builds rather than about a hand-filled :class:`~qrcheck.report.model.Counts`.

    Args:
        make_case_result: The suite's ``CaseResult`` factory.
        make_run_result: The suite's ``RunResult`` factory.
        passed: How many ``passed`` results the run holds.
        failed: How many ``failed`` results it holds.
        skipped: How many ``skipped`` results it holds.
        error: How many ``error`` results it holds.

    Returns:
        The constructed run.
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
    """Make ``qrcheck run`` build `adapter` for the configured target.

    The command resolves a target through ``qrcheck.cli.run.ADAPTER_TYPES``, so replacing
    that one table is the narrowest injection that leaves the command's own resolution,
    validation and teardown in the path. The replacement accepts the call shape the command
    uses -- the per-type target record positionally and the configured name as a keyword --
    and returns the same stub for every lookup, which no run here makes twice.

    Args:
        monkeypatch: The fixture that owns the replacement's lifetime.
        adapter: The protocol-conforming stub every lookup returns.
    """

    def build(target: object, *, name: str) -> Any:
        del target, name
        return adapter

    monkeypatch.setattr(run_module, "ADAPTER_TYPES", {"reference": build})


def replace_case_generator(
    monkeypatch: pytest.MonkeyPatch, check_id: str
) -> CheckContract:
    """Replace one registered check's cases with a single case that raises.

    The registry is read through a read-only view of a module-private store, so a replacement
    has to be written into that store: ``isolated_registry()`` is the documented way to do
    that, but it empties the registry first, and the assertion this serves is precisely that
    the *other* registered checks keep running. The entry is restored by ``monkeypatch`` at
    the end of the test, whatever the outcome.

    Args:
        monkeypatch: The fixture that owns the replacement's lifetime.
        check_id: The registered check whose case generator is replaced.

    Returns:
        The replacement contract, for a caller that wants to assert its identity.
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


# --------------------------------------------------------------------------- #
# Reading what a run left behind
# --------------------------------------------------------------------------- #


def read_summary(report_dir: Path) -> dict[str, Any]:
    """Return ``summary.json`` of the run under `report_dir`, parsed.

    The summary is the machine-readable statement of a run's tallies and results, so every
    exit-0/1/3 assertion reads the code and the counts from the same artefact: a command that
    exited 1 without reporting a failure, or exited 3 without reporting an error, would fail
    here even though the integer alone looked right.

    Args:
        report_dir: The run's report root.

    Returns:
        The parsed summary.
    """
    return json.loads((report_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))


def assert_no_report(report_dir: Path) -> None:
    """Assert that no run described itself under `report_dir`.

    Exit code 2 is decided before any check runs, so the report root must hold neither an
    artefact nor an evidence directory: a report written on a configuration or prerequisite
    fault would be read as a completed run. The whole tree is checked, not only
    ``summary.json``, because a partially written report would be just as misleading.

    Args:
        report_dir: The run's report root.
    """
    assert sorted(report_dir.rglob("*")) == [], (
        f"an exit-2 run wrote {sorted(report_dir.rglob('*'))} under {report_dir}; exit code 2 "
        "is decided before any check runs, so no report may be written for it"
    )


def json_object_lines(stdout: str) -> list[dict[str, Any]]:
    """Return one parsed object per line of `stdout`, asserting each line is an object.

    The contract for an unwritable report directory is JSON lines: one canonical object per
    result and no wrapper, so a CI consumer parses what the command could not file. A line
    that is not a JSON object -- a banner, a summary sentence, a partial write -- fails here.

    Args:
        stdout: What the command wrote on standard output.

    Returns:
        The parsed objects, in the order the command wrote them.
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
    """Return the status AAP §0.3.4 prescribes for a run carrying these two tallies.

    The rule is stated once, here, in the form the plan states it: an ``error`` result
    outranks a ``failed`` one, either outranks a clean run, and the ``skipped`` tally is not
    consulted at all. The truth table below is generated from this reading rather than from
    the implementation, so a change to ``exit_code_for`` that reorders the two arms fails the
    table instead of moving with it.

    Args:
        failed: How many results carry status ``failed``.
        error: How many results carry status ``error``.

    Returns:
        :data:`~qrcheck.cli.exit.EXIT_INTERNAL`, :data:`~qrcheck.cli.exit.EXIT_FAILED` or
        :data:`~qrcheck.cli.exit.EXIT_OK`.
    """
    if error > 0:
        return EXIT_INTERNAL
    if failed > 0:
        return EXIT_FAILED
    return EXIT_OK


#: Every small combination of the four tallies, with the code the plan prescribes for it and
#: a readable identifier. ``passed`` and ``skipped`` vary too, because a run whose outcome was
#: read from a pass count or from a "nothing went wrong" flag would still pass a table that
#: held them fixed.
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
    """Every combination of the tallies maps to the documented status, and never to 2.

    Exit code 2 is decided before a run exists, so no run result can imply it: the
    :data:`~qrcheck.cli.exit.EXIT_CONFIG` assertion is as much part of the contract as the
    ``expected`` one, and it is what keeps a future refactor from folding a configuration
    fault into the run's own verdict.
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
    """The three corners the plan names, each with the code its sentence fixes.

    A skipped result is counted in its own integer, so a run of skips and passes alone is a
    clean run; a single failure is exit 1; and a failure beside an error is exit 3, because
    the two arms are not additive and the internal fault is the one a reader has to see.
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
    """Across the whole table the function produces exactly three of the four codes.

    The absent one is :data:`~qrcheck.cli.exit.EXIT_CONFIG`: it is decided while the
    configuration, the selection, the corpus and the primary decoder are settled, all of
    which happen before a :class:`~qrcheck.report.model.RunResult` exists.
    """
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
    """Three runs of equal size and different composition take three different codes.

    A function that answered "did anything go wrong" would return the same code for the
    second and the third run; a function that read a stored verdict would return the same
    code for all three. The code tracks the four integers and nothing else.
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
    """The precedence is the integer order, and it holds on a run with both faults.

    The numeric order is what an operating system reads, so the constants have to be ordered
    the way the precedence is described: a wrapper that combined statuses by taking the
    maximum -- the usual way a shell reports the worst of several steps -- must arrive at the
    internal fault rather than the failed one.
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
    """``main`` hands its caller an ``int``, which is the console script's whole contract.

    Both entry points -- the ``qrcheck`` console script and ``python -m qrcheck`` -- pass
    this value straight to :class:`SystemExit`, so a ``main`` that raised ``SystemExit``
    itself would double-wrap it and one that returned ``None`` would report every run as a
    success. ``list`` is the cheapest verb that reaches the registry, and its exit status is
    :data:`~qrcheck.cli.exit.EXIT_OK` by construction.
    """
    code = main(["list"])

    assert isinstance(code, int) and not isinstance(code, bool)
    assert code == EXIT_OK
    assert "RT-ROUNDTRIP" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Exit code 0: a run that skipped a check still succeeded
# --------------------------------------------------------------------------- #


def test_exit_zero_when_the_only_selected_check_is_skipped(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """A capability shortfall is a skip, and a skip is neither a failure nor a pass.

    The target declares ``png`` alone, so the selected check -- which also demands
    ``ec_level`` -- is resolved to ``skipped``/``unsupported_capability`` *before* execution:
    the run exits 0, the skip is tallied in its own integer, and the target is never asked for
    a symbol. Folding the skip into ``passed`` would make a run that covered nothing look
    green, which is the reason the model keeps four separate counters.
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
    """A failed check and nothing worse is exit 1, with every artefact still written.

    The decode layer is stubbed to return a payload that is not the one requested, which is
    the one defect ``RT-ROUNDTRIP`` asserts against: every case of the selected class fails
    with ``value_mismatch``, no case errors, the process exits 1 and the full report exists --
    the difference between "the generator is wrong" and "the framework broke" is the whole
    reason the two codes are distinct.
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
    """A document with a key the schema does not declare is refused, and the key is named.

    The diagnostic has to name the key, because exit code 2 exists to tell a reader what to
    fix; and the report root has to stay empty, because a report written for a configuration
    this run never accepted would describe a run that did not happen.
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
    """A selector that names no registered check is echoed verbatim, and nothing runs.

    The selector is repeated rather than paraphrased, so a reader of a CI log can paste it
    back; and the sweep happens before the target is built, so this needs no double at all.
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
    """A ``--class`` filter that leaves no record is a prerequisite fault, not an empty run.

    A run over zero records would satisfy every assertion it has and prove nothing, which is
    why the runner refuses it before the walk and the command exits 2 naming the globs.
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
    """Without the primary decoder the run refuses to start rather than skipping round-trip.

    ``sys.modules`` is given a ``None`` entry for ``zxingcpp``, which is how an unavailable
    module is modelled in-process: the probe's ``find_spec`` raises and the decoder reads as
    absent. The diagnostic has to name the distribution and the lock file that pins it,
    because a missing required decoder is an environment fault to install, not a target to
    change; and no report may be written, so ``RT-ROUNDTRIP`` is never silently skipped.
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
    """An unreadable corpus is a prerequisite fault, and the path is named.

    This is the run's own pre-flight path end to end: the command is configured with a
    ``corpus_path`` that holds no file, the runner raises its prerequisite error before the
    walk begins, and the process exits 2 having written nothing.
    """
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
    """A target found unavailable *before* any case runs is exit 2, not exit 3.

    The runner's prerequisite exception is where a pre-flight availability fault surfaces:
    the target could not be reached, or was never usable, so no case ran and there is no run
    to report -- which is exactly the condition the exit table puts at code 2. A target lost
    *during* the walk is a different state: it produces one ``error`` result per affected
    check, keeps the report, and exits 3, which is why the two are asserted in different
    modules (``tests/unit/test_runner_target_loss.py`` holds the other half). The error is
    raised here rather than provoked, because a real target that dies on first use is that
    mid-run state, and no configuration can express "unreachable only before the first case".
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
    """A raising case becomes one ``error`` result with its traceback, and the walk goes on.

    Three assertions make this the state the plan describes rather than "the run aborted":
    exactly one result carries ``error``, that result's decoded evidence holds the traceback
    with the raised message inside it, and a result filed under a *later* check identifier is
    still present. A command that stopped at the first exception would satisfy the first two
    and fail the third, which is the point of asserting it separately.
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


def test_exit_three_when_the_report_directory_is_unwritable(
    invoke_cli: Callable[..., Any],
    unwritable_report_dir: Path,
    report_dir: Path,
) -> None:
    """A ``report_dir`` that cannot be created is exit 3, and the path is in the message.

    A regular file stands where the directory belongs, so ``mkdir`` fails for every user
    including root -- a permission bit would prove nothing in a container that runs as root.
    No result exists yet, because the directory is probed before the run starts, so the
    stream is empty; the JSON-lines half of the contract is asserted in the next test, where
    a run result does exist.
    """
    result = invoke_cli("run", "--report-dir", str(unwritable_report_dir))

    assert result.exit_code == EXIT_INTERNAL
    assert str(unwritable_report_dir.resolve()) in result.stderr
    assert json_object_lines(result.stdout) == []
    assert_no_report(report_dir)


def test_exit_three_streams_json_lines_when_the_report_cannot_be_written(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    report_dir: Path,
) -> None:
    """A run whose report cannot be written still streams every result, one object per line.

    The writer is made to fail rather than the directory, because a directory the probe
    accepts and the writers then cannot use is the only state that has results *and* no
    report, and ``chmod`` cannot produce it for root. What is asserted is the contract a CI
    consumer depends on: the process exits 3, the path is named on stderr, and every line of
    stdout parses as a JSON object, so the results the run found are never lost with the file.
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


def test_exit_three_outranks_exit_one_when_a_run_both_fails_and_errors(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    stub_adapter: Any,
    patch_decode: Any,
    make_decode_result: Callable[..., Any],
    committed_corpus: Sequence[Any],
    report_dir: Path,
) -> None:
    """A run carrying a failure *and* an error exits 3, and the summary reports both counts.

    The precedence the plan states is only meaningful when both faults are present at once:
    the internal fault is the worse of the two, so it decides the status, but the failure is
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
# The other four verbs answer in the same vocabulary
# --------------------------------------------------------------------------- #


def unreachable_target_document() -> str:
    """Return a configuration document naming a target that can never be reached.

    The ``library`` adapter resolves its callable from configuration, so a module that does
    not exist is an unreachable target that needs no port, no process and no double. The
    declared ``png`` capability matters: without it the probe would be refused for a missing
    capability, which is a different fault and a different diagnostic.

    Returns:
        A complete TOML document for a configuration with one ``library`` target.
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
    """``list``, ``doctor``, ``corpus build`` and ``loc`` use the same four statuses.

    Four short invocations, so that a code chosen in one command cannot drift from the code
    its sibling returns for the same kind of fault: ``list`` always succeeds, ``doctor``
    reports an unreachable target as a configuration-class failure (that is the product's
    pre-flight target-availability gate, and the reason a *run* against a dying target is the
    exit-3 state instead), and the two argument faults -- a negative seed and a path that
    does not exist -- are exit 2 rather than the exit 1 a shell would expect from a failing
    command.
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
