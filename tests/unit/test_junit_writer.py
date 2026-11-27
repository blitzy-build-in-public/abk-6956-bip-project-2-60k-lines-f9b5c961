"""The JUnit writer's ``xunit2`` shape, its one-child-per-status rule and its hygiene.

This module is the gate that holds ``junit.xml``, the one artefact CI stores as evidence and
gates a merge on. Every document is written by :func:`qrcheck.report.junit.write_junit` from a
:class:`~qrcheck.report.model.RunResult` built by the shared factories.

Pinned: one ``testsuites`` root holding exactly one ``testsuite name="qrcheck"``; ``tests``,
``failures``, ``errors`` and ``skipped`` equal to the run's four integers, so a skipped case can
never count as a pass in the artefact CI reads; one ``testcase`` per result in ``summary.json``
order; ``name`` as ``<CHECK-ID>[<case-id>]``, ``classname`` as ``qrcheck.<family>`` read from
:mod:`qrcheck.registry.check` rather than guessed from the identifier (an unregistered one
degrades to ``qrcheck.unknown``) and ``time`` with exactly three decimals, because a raw float
renders as ``1.25`` or ``1e-05``; exactly one child element per skipped, failed or errored
status; the deliberate absences -- no ``hostname``, no ``timestamp``, no ``file``/``line`` --
because a wall-clock reading makes two identical runs differ; and byte-identical
documents once ``time`` is masked.

XML 1.0 safety is held here too: ``ElementTree`` escapes markup but emits a control character
raw, so only the writer's own filter keeps a hostile identifier, payload or traceback parseable."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from qrcheck.registry.check import (
    CHECK_IDS,
    FAMILY_BY_PREFIX,
    FAMILY_NAMES,
    REGISTRY,
    CheckContract,
    isolated_registry,
    register,
)
from qrcheck.report.json_report import write_summary
from qrcheck.report.junit import write_junit
from qrcheck.report.model import TRACEBACK_KEY, CaseResult, Counts, Reason, RunResult, Status

# The two artefacts written into the per-test report root: the document under test and the
# summary whose ``results[]`` order it has to agree with.
_JUNIT_NAME = "junit.xml"
_SUMMARY_NAME = "summary.json"

# Every ``time`` attribute of a document, matched on the serialised bytes so the comparison
# depends on no parser's attribute order and reaches the root, the suite and every testcase.
_TIME_ATTRIBUTE = re.compile(rb'time="[^"]*"')

# What a ``time`` attribute is replaced by before two documents are compared. Deliberately
# not a duration, so a masked document can never be mistaken for a measured one.
_MASKED_TIME = b'time="<masked>"'

# The two shapes :func:`qrcheck.report.junit.write_junit` fixes for a duration: three decimals
# on a testcase, and any number of them on the suite, whose total is a sum rather than a
# measurement.
_THREE_DECIMALS = re.compile(r"\d+\.\d{3}")
_SUITE_TIME = re.compile(r"\d+\.\d+")

# The display name a testcase carries: ``<CHECK-ID>[<case-id>]``.
_CASE_NAME = re.compile(r"(?P<check_id>[^\[\]]+)\[(?P<case_id>.+)\]")

# The complete set of elements the writer may emit, mapped to the complete set of attribute
# names each may carry. Anything outside this table is a change to the public document shape
# and must be a deliberate change here first.
_ALLOWED_ATTRIBUTES: dict[str, frozenset[str]] = {
    "testsuites": frozenset({"errors", "failures", "skipped", "tests", "time"}),
    "testsuite": frozenset({"errors", "failures", "name", "skipped", "tests", "time"}),
    "testcase": frozenset({"classname", "name", "time"}),
    "skipped": frozenset({"message"}),
    "failure": frozenset({"message"}),
    "error": frozenset({"message"}),
}

# The elements that carry no text of their own: only the status child of a failing case has a
# body. A ``testcase`` with text would mean the decoded detail escaped its element.
_TEXT_FREE_TAGS = frozenset({"testsuites", "testsuite", "testcase", "skipped"})

# Names pytest's own ``--junitxml`` emits, or that a JUnit dialect allows, and that this
# writer deliberately does not: a hostname or a wall-clock reading in the document makes two
# identical runs differ.
_FORBIDDEN_NAMES = frozenset(
    {
        "assertions",
        "file",
        "hostname",
        "id",
        "line",
        "package",
        "properties",
        "system-err",
        "system-out",
        "timestamp",
    }
)

# The tokens a failure body must carry: one value that appears only in the decoded evidence
# and one that appears only in the expected evidence, so an embedded body that lost either
# side - or rendered the same mapping twice - fails here.
_DECODED_TOKEN = "decoded-version-3"
_EXPECTED_TOKEN = "expected-version-7"
_FAILED_DECODED: dict[str, object] = {"decoded_version": _DECODED_TOKEN}
_FAILED_EXPECTED: dict[str, object] = {"required_version": _EXPECTED_TOKEN}

# The traceback an internal framework exception reaches the writer as, under
# :data:`~qrcheck.report.model.TRACEBACK_KEY`. Its frame path is relative on purpose: the
# result model folds an *absolute* one into a portable form, and that rewrite is the model's
# contract rather than this writer's.
_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "qrcheck/checks/roundtrip.py", line 42, in cases\n'
    "    raise RuntimeError('boom')\n"
    "RuntimeError: boom"
)

# The hostile payload injected into a case identifier, a decoded mapping and a traceback: an
# element, an ampersand, both quote characters, non-ASCII text from three scripts, and the
# two control characters XML 1.0 cannot carry (an embedded NUL, which the ``nul-byte``
# corpus class produces on purpose, and a vertical tab).
_HOSTILE_MARKUP = "<script>alert(1)</script>"
_HOSTILE_AMPERSAND = "a & b"
_HOSTILE_QUOTE = 'quote " and \' stay text'
_HOSTILE_NON_ASCII = "héllo 🎉 مرحبا"
_HOSTILE_CONTROL = "nul:\x00 vt:\x0b"
_HOSTILE_TEXT = " ".join(
    (_HOSTILE_MARKUP, _HOSTILE_AMPERSAND, _HOSTILE_QUOTE, _HOSTILE_NON_ASCII, _HOSTILE_CONTROL)
)

# The U+FFFD the writer substitutes for a code point outside the XML 1.0 ``Char`` production.
_HOSTILE_REPLACEMENT = "\ufffd"


def _write_junit(run: RunResult, root: Path, name: str = _JUNIT_NAME) -> Path:
    """Write ``run`` as ``root/name`` through the product's writer and return the path."""
    return write_junit(run, root / name)


def _parse(path: Path) -> ET.Element:
    """Parse ``path`` and return the root element, raising if the document is ill-formed."""
    return ET.parse(path).getroot()


def _suite(path: Path) -> ET.Element:
    """Return the document's single ``testsuite`` child."""
    root = _parse(path)
    assert root.tag == "testsuites"
    suites = root.findall("testsuite")
    assert len(suites) == 1
    return suites[0]


def _testcases(path: Path) -> list[ET.Element]:
    """Return the document's ``testcase`` elements, in document order."""
    return _suite(path).findall("testcase")


def _testcase_identities(path: Path) -> list[tuple[str, str]]:
    """Return each testcase's ``(check_id, case_id)``, parsed back out of its ``name``."""
    identities: list[tuple[str, str]] = []
    for testcase in _testcases(path):
        name = testcase.get("name")
        assert name is not None
        match = _CASE_NAME.fullmatch(name)
        assert match is not None, name
        identities.append((match.group("check_id"), match.group("case_id")))
    return identities


def _mask_time_attributes(data: bytes) -> bytes:
    """Return ``data`` with every ``time`` attribute replaced by a non-duration sentinel."""
    return _TIME_ATTRIBUTE.sub(_MASKED_TIME, data)


def _family_of(check_id: str) -> str:
    """Return the family :mod:`qrcheck.registry.check` declares for ``check_id``: the contract when
    the families have imported, otherwise the constant prefix table the writer falls back to."""
    contract = REGISTRY.get(check_id)
    if contract is not None:
        return contract.family
    return FAMILY_BY_PREFIX[check_id.split("-", 1)[0]]


def _mixed_status_run(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
) -> RunResult:
    """Return a run whose five cases reach all four statuses: two passes, one failure, one skip and
    one error give every tally a distinct value, so a confused document cannot match."""
    results = (
        make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:url-001",
            status=Status.PASSED,
            duration_ms=3.25,
        ),
        make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:url-002",
            status=Status.PASSED,
            duration_ms=2.5,
        ),
        make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:nul-003",
            status=Status.FAILED,
            reason=Reason.VALUE_MISMATCH,
            duration_ms=12.5,
            decoded=dict(_FAILED_DECODED),
            expected=dict(_FAILED_EXPECTED),
        ),
        make_case_result(
            check_id="RN-SVG",
            case_id="synthetic:svg-rasterizer",
            status=Status.SKIPPED,
            reason=Reason.MISSING_DEPENDENCY,
            duration_ms=0.0,
        ),
        make_case_result(
            check_id="PF-DECODE",
            case_id="synthetic:internal",
            status=Status.ERROR,
            reason=Reason.INTERNAL_ERROR,
            duration_ms=7.75,
            decoded={TRACEBACK_KEY: _TRACEBACK},
        ),
    )
    return make_run_result(results)


def _durations_run(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    first_ms: float,
    second_ms: float,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> RunResult:
    """Return a run whose only volatile content is its durations and timestamps: both cases keep
    their identity, status and evidence, so two durations describe the same outcome."""
    results = (
        make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:url-001",
            status=Status.PASSED,
            duration_ms=first_ms,
        ),
        make_case_result(
            check_id="SY-MASK",
            case_id="corpus:url-002",
            status=Status.SKIPPED,
            reason=Reason.UNSUPPORTED_CAPABILITY,
            duration_ms=second_ms,
        ),
    )
    if started_at is None:
        return make_run_result(results)
    assert finished_at is not None
    return make_run_result(results, started_at=started_at, finished_at=finished_at)



# The skeleton and the tallies


def test_document_parses_into_one_suite_named_qrcheck(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The document parses as one ``testsuites`` root holding one suite named ``qrcheck``."""
    run = _mixed_status_run(make_case_result, make_run_result)

    path = _write_junit(run, report_dir)
    root = _parse(path)

    assert root.tag == "testsuites"
    assert [child.tag for child in root] == ["testsuite"]
    assert root[0].get("name") == "qrcheck"
    assert len(_testcases(path)) == len(run.results)


def test_suite_tallies_equal_the_run_counts(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """``tests``/``failures``/``errors``/``skipped`` are the run's four tally integers, recomputed
    from the case results with ``Counts.from_results`` rather than read back from the run."""
    run = _mixed_status_run(make_case_result, make_run_result)
    expected = Counts.from_results(run.results)
    assert expected == run.counts

    path = _write_junit(run, report_dir)
    suite = _suite(path)

    assert suite.get("tests") == str(len(run.results))
    assert suite.get("tests") == str(expected.total)
    assert suite.get("failures") == str(expected.failed)
    assert suite.get("errors") == str(expected.error)
    assert suite.get("skipped") == str(expected.skipped)
    suite_time = suite.get("time")
    assert suite_time is not None
    assert _SUITE_TIME.fullmatch(suite_time) is not None
    assert expected.skipped > 0 and expected.failed > 0 and expected.error > 0


def test_suite_and_root_aggregates_agree(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The five aggregates appear on the root as well, and the two copies cannot disagree."""
    run = _mixed_status_run(make_case_result, make_run_result)

    path = _write_junit(run, report_dir)
    root = _parse(path)
    suite = _suite(path)

    aggregates = sorted(_ALLOWED_ATTRIBUTES["testsuites"])
    assert {name: root.get(name) for name in aggregates} == {
        name: suite.get(name) for name in aggregates
    }
    assert aggregates == ["errors", "failures", "skipped", "tests", "time"]


def test_zero_result_run_writes_a_well_formed_document_with_no_case(
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """A run with no result still yields a parseable document carrying four zero tallies."""
    run = make_run_result(())
    assert run.counts == Counts()

    path = _write_junit(run, report_dir)
    suite = _suite(path)

    assert suite.get("name") == "qrcheck"
    assert suite.get("tests") == "0"
    assert (suite.get("failures"), suite.get("errors"), suite.get("skipped")) == ("0", "0", "0")
    suite_time = suite.get("time")
    assert suite_time is not None
    assert _SUITE_TIME.fullmatch(suite_time) is not None
    assert _testcases(path) == []


# One testcase per result, in the summary's order, with the documented attributes


def test_testcase_order_equals_the_summary_order(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """Both writers present the order the run stores, so neither re-sorts independently."""
    cases = (
        make_case_result(check_id="SY-MASK", case_id="corpus:url-002", status=Status.PASSED),
        make_case_result(
            check_id="CL-ARTIFACTS",
            case_id="synthetic:artifacts",
            status=Status.SKIPPED,
            reason=Reason.UNSUPPORTED_PLATFORM,
        ),
        make_case_result(check_id="RT-ROUNDTRIP", case_id="corpus:url-001", status=Status.PASSED),
        make_case_result(check_id="RT-ROUNDTRIP", case_id="corpus:url-000", status=Status.PASSED),
    )
    run = make_run_result(cases)

    path = _write_junit(run, report_dir)
    write_summary(run, report_dir / _SUMMARY_NAME)
    summary = json.loads((report_dir / _SUMMARY_NAME).read_text(encoding="utf-8"))
    summary_order = [(entry["check_id"], entry["case_id"]) for entry in summary["results"]]

    stored_order = [(result.check_id, result.case_id) for result in run.results]
    assert summary_order == stored_order
    assert stored_order == sorted(stored_order)
    # The input order above is deliberately not the stored one, so the assertions below
    # compare against an order that had to be produced rather than one that was supplied.
    assert [(case.check_id, case.case_id) for case in cases] != stored_order
    assert _testcase_identities(path) == summary_order


@pytest.mark.parametrize("check_id", CHECK_IDS)
def test_testcase_attributes_name_the_check_and_its_family(
    check_id: str,
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """``name`` is ``<CHECK-ID>[<case-id>]``, ``classname`` is ``qrcheck.<family>``, ``time`` the
    duration: the family comes from the registry, and 1250 ms renders as ``1.250``, not ``1.25``."""
    case = make_case_result(
        check_id=check_id,
        case_id="corpus:url-001",
        status=Status.PASSED,
        duration_ms=1250.0,
    )
    run = make_run_result((case,))

    testcases = _testcases(_write_junit(run, report_dir))
    assert len(testcases) == 1
    testcase = testcases[0]

    family = _family_of(check_id)
    assert family in FAMILY_NAMES
    assert FAMILY_BY_PREFIX[check_id.split("-", 1)[0]] == family
    assert testcase.get("name") == f"{check_id}[corpus:url-001]"
    assert testcase.get("classname") == f"qrcheck.{family}"
    testcase_time = testcase.get("time")
    assert testcase_time == "1.250"
    assert _THREE_DECIMALS.fullmatch(testcase_time) is not None


def test_registered_contract_supplies_the_classname_family(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """A registered contract's own ``family`` is what the classname names: registration rejects a
    contract whose family does not own its prefix, so the answer is *read* rather than spelled."""
    contract = CheckContract(
        id="RT-ROUNDTRIP",
        title="Payload round-trip",
        family=FAMILY_BY_PREFIX["RT"],
        requires=frozenset(),
        dependencies=(),
        cases=lambda context: iter(()),
    )
    case = make_case_result(
        check_id=contract.id, case_id="corpus:url-001", status=Status.PASSED
    )
    run = make_run_result((case,))
    before = sorted(REGISTRY)

    with isolated_registry():
        register(contract)
        assert REGISTRY[contract.id] is contract
        testcases = _testcases(_write_junit(run, report_dir))

    assert len(testcases) == 1
    assert testcases[0].get("classname") == f"qrcheck.{contract.family}"
    assert sorted(REGISTRY) == before


def test_stranger_check_id_degrades_its_classname_to_unknown(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """An identifier from neither registry source still yields a document, degraded not lost."""
    check_id = "ZZ-NOT-A-CHECK"
    assert check_id not in REGISTRY
    assert check_id.split("-", 1)[0] not in FAMILY_BY_PREFIX

    case = make_case_result(check_id=check_id, case_id="synthetic:stranger", status=Status.PASSED)
    testcases = _testcases(_write_junit(make_run_result((case,)), report_dir))

    assert len(testcases) == 1
    assert testcases[0].get("classname") == "qrcheck.unknown"
    assert testcases[0].get("name") == f"{check_id}[synthetic:stranger]"


def test_non_string_check_id_degrades_its_classname_to_unknown(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """A non-``str`` identifier has no prefix segment, so it degrades as a stranger does: the
    run carries one case alone, its sort having no order."""
    case = make_case_result(check_id=12345, case_id="synthetic:integer", status=Status.PASSED)
    run = make_run_result((case,))

    testcases = _testcases(_write_junit(run, report_dir))

    assert len(testcases) == 1
    assert testcases[0].get("classname") == "qrcheck.unknown"
    assert testcases[0].get("name") == "12345[synthetic:integer]"


# The one child element each status emits


@pytest.mark.parametrize(
    ("status", "fields", "child_tag"),
    (
        pytest.param(Status.PASSED, {}, None, id="passed"),
        pytest.param(
            Status.SKIPPED,
            {"reason": Reason.MISSING_DEPENDENCY},
            "skipped",
            id="skipped",
        ),
        pytest.param(
            Status.FAILED,
            {
                "reason": Reason.VALUE_MISMATCH,
                "decoded": _FAILED_DECODED,
                "expected": _FAILED_EXPECTED,
            },
            "failure",
            id="failed",
        ),
        pytest.param(
            Status.ERROR,
            {"reason": Reason.INTERNAL_ERROR, "decoded": {TRACEBACK_KEY: _TRACEBACK}},
            "error",
            id="error",
        ),
    ),
)
def test_status_decides_the_single_child_element(
    status: Status,
    fields: dict[str, object],
    child_tag: str | None,
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """A pass carries no child; a skip, a failure and an error carry exactly one, whose ``message``
    is the machine-readable reason token, never a sentence."""
    case = make_case_result(
        check_id="RT-ROUNDTRIP",
        case_id="corpus:url-001",
        status=status,
        **fields,
    )
    testcases = _testcases(_write_junit(make_run_result((case,)), report_dir))
    assert len(testcases) == 1

    children = list(testcases[0])
    if child_tag is None:
        assert children == []
        return

    assert [child.tag for child in children] == [child_tag]
    child = children[0]
    reason = fields["reason"]
    assert isinstance(reason, Reason)
    message = child.get("message")
    assert message == reason.value
    assert message

    body = child.text or ""
    if status is Status.FAILED:
        assert _DECODED_TOKEN in body
        assert _EXPECTED_TOKEN in body
        assert f"case: {case.case_id}" in body
    elif status is Status.ERROR:
        assert "Traceback (most recent call last):" in body
        assert "RuntimeError: boom" in body
    else:
        assert child.text is None


def test_element_counts_across_the_file_equal_the_run_counts(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The status children add up to the tallies: one per non-passing case, none for a pass."""
    run = _mixed_status_run(make_case_result, make_run_result)

    testcases = _testcases(_write_junit(run, report_dir))
    child_tags = [child.tag for testcase in testcases for child in testcase]

    assert len(testcases) == run.counts.total
    assert child_tags.count("skipped") == run.counts.skipped
    assert child_tags.count("failure") == run.counts.failed
    assert child_tags.count("error") == run.counts.error
    assert sum(1 for testcase in testcases if not list(testcase)) == run.counts.passed
    assert len(child_tags) == len(testcases) - run.counts.passed


def test_error_without_a_traceback_renders_its_decoded_mapping(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """An error whose ``decoded`` carries no traceback still gets a body, never an empty one: the
    whole mapping is rendered through the canonical JSON formatter, so it stays inspectable."""
    decoded = {"stage": "assertion", "detail": "no traceback was recorded"}
    case = make_case_result(
        check_id="PF-DECODE",
        case_id="synthetic:internal",
        status=Status.ERROR,
        reason=Reason.INTERNAL_ERROR,
        decoded=decoded,
    )

    testcases = _testcases(_write_junit(make_run_result((case,)), report_dir))
    children = list(testcases[0])

    assert [child.tag for child in children] == ["error"]
    assert children[0].get("message") == Reason.INTERNAL_ERROR.value
    assert json.loads(children[0].text or "") == decoded


# The deliberate absences: nothing volatile, nothing invented


def test_only_documented_elements_and_attributes_are_emitted(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """Every element and attribute name is inside the allow-list; no ``hostname``, no clock."""
    run = _mixed_status_run(make_case_result, make_run_result)

    root = _parse(_write_junit(run, report_dir))
    seen_tags: set[str] = set()
    for element in root.iter():
        seen_tags.add(element.tag)
        assert element.tag in _ALLOWED_ATTRIBUTES, element.tag
        assert set(element.attrib) <= _ALLOWED_ATTRIBUTES[element.tag], element.tag
        for name in (element.tag, *element.attrib):
            assert name not in _FORBIDDEN_NAMES, name
        if element.tag in _TEXT_FREE_TAGS:
            assert element.text is None or not element.text.strip(), element.tag

    assert seen_tags == set(_ALLOWED_ATTRIBUTES)
    assert "timestamp" not in root.attrib
    # ``time`` on the root is the documented duration aggregate, and the only volatile value
    # the document carries - which is why the comparison below masks exactly that attribute.
    assert set(root.attrib) == _ALLOWED_ATTRIBUTES["testsuites"]
    assert "time" in root.attrib


def test_two_runs_that_differ_only_in_durations_agree_once_time_is_masked(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """Duration and timestamps are the whole volatile inventory, and only ``time`` escapes: the
    documents differ before masking and are equal after it."""
    first = _durations_run(make_case_result, make_run_result, first_ms=1.5, second_ms=900.0)
    second = _durations_run(
        make_case_result,
        make_run_result,
        first_ms=250.75,
        second_ms=3.0,
        started_at="2024-06-01T12:00:00+00:00",
        finished_at="2024-06-01T12:00:05+00:00",
    )
    assert first.duration_ms != second.duration_ms

    first_raw = _write_junit(first, report_dir, "first.xml").read_bytes()
    second_raw = _write_junit(second, report_dir, "second.xml").read_bytes()

    assert first_raw != second_raw
    assert _mask_time_attributes(first_raw) == _mask_time_attributes(second_raw)
    # One time attribute on the root, one on the suite, one per testcase - and nothing else.
    assert len(_TIME_ATTRIBUTE.findall(first_raw)) == 2 + len(first.results)
    assert _MASKED_TIME not in first_raw
    # Neither run's timestamps reach the document at all.
    assert b"2024-01-01" not in first_raw
    assert b"2024-06-01" not in second_raw


def test_document_uses_lf_line_endings_and_one_trailing_newline(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The file is UTF-8 with an XML declaration, LF endings and exactly one final newline."""
    run = _mixed_status_run(make_case_result, make_run_result)

    raw = _write_junit(run, report_dir).read_bytes()

    assert raw.startswith(b"<?xml version=")
    declaration, _, _ = raw.partition(b"\n")
    assert declaration == b"<?xml version='1.0' encoding='utf-8'?>"
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert raw.decode("utf-8").startswith("<?xml")


# Hostile text: escaped where XML can escape it, stripped where only stripping helps


@pytest.mark.parametrize("injection", ("case_id", "failure_body", "traceback"))
def test_hostile_text_stays_escaped_and_the_document_still_parses(
    injection: str,
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """Target-supplied text reaches the document as text and the document still parses: the
    identifier lands in an attribute, the payload and traceback in bodies."""
    if injection == "case_id":
        case = make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id=f"corpus:{_HOSTILE_TEXT}",
            status=Status.PASSED,
        )
    elif injection == "failure_body":
        case = make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:url-001",
            status=Status.FAILED,
            reason=Reason.VALUE_MISMATCH,
            decoded={"injected": _HOSTILE_TEXT},
            expected={"injected": _HOSTILE_TEXT},
        )
    else:
        case = make_case_result(
            check_id="PF-DECODE",
            case_id="synthetic:internal",
            status=Status.ERROR,
            reason=Reason.INTERNAL_ERROR,
            decoded={
                TRACEBACK_KEY: (
                    "Traceback (most recent call last):\n"
                    f"  {_HOSTILE_TEXT}\n"
                    "ValueError: boom"
                )
            },
        )
    path = _write_junit(make_run_result((case,)), report_dir)

    # Parsing is the assertion that matters: an XML-illegal character emits raw and this
    # line raises instead.
    root = _parse(path)
    raw = path.read_bytes()

    assert {element.tag for element in root.iter()} <= set(_ALLOWED_ATTRIBUTES)
    assert [element.tag for element in root.iter() if element.tag in {"alert", "script"}] == []
    assert b"&lt;script&gt;" in raw
    assert b"&amp;" in raw
    assert b"\x00" not in raw
    assert b"\x0b" not in raw

    if injection == "case_id":
        rendered = _testcases(path)[0].get("name")
    else:
        bodies = [child.text for testcase in _testcases(path) for child in testcase]
        assert len(bodies) == 1
        rendered = bodies[0]
    assert rendered is not None

    assert _HOSTILE_MARKUP in rendered
    assert _HOSTILE_AMPERSAND in rendered
    assert _HOSTILE_NON_ASCII in rendered
    if injection == "failure_body":
        # The body renders the decoded mapping through the product's canonical JSON
        # formatter, which escapes a control character as ``\u0000`` before the XML writer
        # sees it - so this sink proves the escaping path rather than the stripping path.
        assert "\\u0000" in rendered
        assert "\\u000b" in rendered
        assert '\\"' in rendered
    else:
        assert _HOSTILE_QUOTE in rendered
        assert _HOSTILE_REPLACEMENT in rendered
    assert "\x00" not in rendered
    assert "\x0b" not in rendered

