"""The result model's invariants: what ``CaseResult`` and ``RunResult`` refuse.

AAP §0.3.4 fixes both records exactly - ``CaseResult`` with its ten fields and
``RunResult`` with its nine - and fixes what each per-result field of ``summary.json``
holds: ``request``, ``decoded`` and ``expected`` are objects, and ``evidence_dir`` is a
path relative to ``report_dir``. A record shape is only worth what its enforcement is
worth, so every invariant a writer would otherwise have to rediscover is pinned here, at
the model boundary where the value enters:

* a ``duration_ms`` is finite and non-negative. A ``nan`` or an ``inf`` is not a token
  RFC 8259 defines, so it would make ``summary.json`` unreadable for tooling (§0.1.1
  FR-6) and would render as the text ``nan`` in the HTML table and the JUnit ``time``;
* ``request``, ``decoded`` and ``expected`` are Mappings, so no artefact carries an
  array or a scalar where its schema fixes an object;
* ``decoded[TRACEBACK_KEY]``, when it holds a string, has every absolute path in it
  rewritten into a portable form, so an error case's evidence names the module that failed
  rather than the clone it ran in - which is what keeps ``decoded.json`` byte-identical
  across clones and free of absolute paths (§0.5.4, §0.5.5);
* ``evidence_dir`` is a non-empty relative posix path, so the empty string cannot name
  the report root as if it were a case's own evidence directory;
* a ``RunResult`` finishes at or after it starts, so the duration its HTML header
  renders and its JUnit suite time sums is never negative, while a zero-length run stays
  legal;
* one ``(check_id, case_id)`` pair appears once in a ``RunResult``, because that pair is
  the identity :func:`qrcheck.report.model.failure_id` and the evidence directory are
  derived from (AAP §0.3.4 evidence layout, §0.1.1 FR-5) - a repeat would write two
  cases into one directory while the tallies still balanced.

Every rejected record is paired below with the record the same construction accepts, so
no test here can pass by refusing everything, and each accepted case is one an existing
caller really produces (``qrcheck/checks/support.py``, ``qrcheck/registry/resolve.py``,
``tests/unit/test_failure_ids.py``). Nothing is read from ``qrcheck.toml``, no file is
written and no process is started: these are pure functions of their inputs.
"""

import dataclasses
import json
import re
import traceback
import types
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from qrcheck.checks.support import build
from qrcheck.report.model import TRACEBACK_KEY, CaseResult, Counts, Reason, RunResult, Status

# The field names AAP §0.3.4 states for each record, written out here rather than read
# from the dataclasses under test, so a field that is added, dropped or renamed fails an
# assertion below instead of moving with the implementation.
CASE_RESULT_FIELDS = (
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
RUN_RESULT_FIELDS = (
    "run_id",
    "started_at",
    "finished_at",
    "adapter",
    "seed",
    "ec_level",
    "loc_count",
    "counts",
    "results",
)

# A content hash, not a UUID and not a timestamp, which is the identity shape the model
# accepts.
RUN_ID = "9f2c4a1d8e7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d"
STARTED_AT = "2026-01-01T00:00:05+00:00"
FINISHED_AT = "2026-01-01T00:00:07.500+00:00"
EVIDENCE_DIR = "evidence/RT-ROUNDTRIP/RT-ROUNDTRIP__corpus_url-001__value_mismatch"

NON_FINITE_DURATIONS = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-inf"),
    pytest.param(float("-inf"), id="negative-inf"),
)

FINITE_DURATIONS = (
    pytest.param(0.0, id="zero"),
    pytest.param(0.25, id="sub-millisecond"),
    pytest.param(12.5, id="fractional"),
    pytest.param(1000, id="int"),
    pytest.param(1e-9, id="tiny"),
)

# Everything that is not a mapping, including the two shapes a careless caller would
# most plausibly pass: a JSON array and a JSON scalar.
NON_MAPPING_PAYLOADS = (
    pytest.param(["payload_b64"], id="list"),
    pytest.param(("payload_b64",), id="tuple"),
    pytest.param("payload_b64", id="str"),
    pytest.param(7, id="int"),
    pytest.param(None, id="none"),
)

MAPPING_PAYLOADS = (
    pytest.param({}, id="empty-dict"),
    pytest.param({"payload_b64": "aGk="}, id="dict"),
    pytest.param(types.MappingProxyType({"payload_b64": "aGk="}), id="mapping-proxy"),
)

PAYLOAD_FIELDS = ("request", "decoded", "expected")


def case_result(**overrides: Any) -> CaseResult:
    """Build a failing case result with every field at a legal default."""
    given: dict[str, Any] = {
        "check_id": "RT-ROUNDTRIP",
        "case_id": "corpus:url-001",
        "status": Status.FAILED,
        "reason": Reason.VALUE_MISMATCH,
        "duration_ms": 12.5,
        "request": {"payload_b64": "aGk="},
        "decoded": {"payload_bytes": "aGk="},
        "expected": {"payload_sha256": "0" * 64},
        "evidence_dir": EVIDENCE_DIR,
        "seed": 20240101,
    }
    given.update(overrides)
    return CaseResult(**given)


def run_result(**overrides: Any) -> RunResult:
    """Build a run holding one failing case, with every field at a legal default."""
    given: dict[str, Any] = {
        "run_id": RUN_ID,
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
        "adapter": "reference",
        "seed": 20240101,
        "ec_level": "M",
        "loc_count": 1234,
        "counts": Counts(failed=1),
        "results": (case_result(),),
    }
    given.update(overrides)
    return RunResult(**given)


# --------------------------------------------------------------------------- #
# The record shape AAP §0.3.4 fixes
# --------------------------------------------------------------------------- #


def test_case_result_keeps_the_ten_stated_fields() -> None:
    """The record and its summary projection are exactly the AAP's ten fields."""
    assert tuple(field.name for field in dataclasses.fields(CaseResult)) == CASE_RESULT_FIELDS
    assert sorted(case_result().as_dict()) == sorted(CASE_RESULT_FIELDS)


def test_run_result_keeps_the_nine_stated_public_fields() -> None:
    """The nine public fields, plus the two private parsed-timestamp caches."""
    public = tuple(
        field.name for field in dataclasses.fields(RunResult) if not field.name.startswith("_")
    )
    assert public == RUN_RESULT_FIELDS
    assert sorted(run_result().as_dict()) == sorted(RUN_RESULT_FIELDS)


# --------------------------------------------------------------------------- #
# duration_ms: finite, then non-negative
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("duration_ms", NON_FINITE_DURATIONS)
def test_a_non_finite_duration_is_refused(duration_ms: float) -> None:
    """A nan or an inf would reach summary.json as a token JSON does not define."""
    with pytest.raises(ValueError) as excinfo:
        case_result(duration_ms=duration_ms)
    message = str(excinfo.value)
    assert "CaseResult.duration_ms" in message
    assert "finite" in message


def test_a_negative_duration_is_still_refused() -> None:
    """The finiteness guard sits beside the negative guard, which still bites."""
    with pytest.raises(ValueError, match="must not be negative"):
        case_result(duration_ms=-0.001)


@pytest.mark.parametrize("duration_ms", FINITE_DURATIONS)
def test_a_finite_non_negative_duration_is_accepted(duration_ms: float) -> None:
    """Every duration a real case can produce is still storable."""
    assert case_result(duration_ms=duration_ms).duration_ms == duration_ms


# --------------------------------------------------------------------------- #
# request, decoded and expected are objects
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ("request", "decoded"))
@pytest.mark.parametrize("value", NON_MAPPING_PAYLOADS)
def test_a_non_mapping_request_or_decoded_is_refused(name: str, value: object) -> None:
    """An array or a scalar would stand where summary.json fixes an object."""
    with pytest.raises(ValueError) as excinfo:
        case_result(**{name: value})
    message = str(excinfo.value)
    assert f"CaseResult.{name}" in message
    assert type(value).__name__ in message


# ``expected`` alone may be ``None``, so its table omits the ``none`` case.
@pytest.mark.parametrize("value", NON_MAPPING_PAYLOADS[:-1])
def test_a_non_mapping_expected_is_refused(value: object) -> None:
    """expected is an object when it is set, and None when nothing was asserted."""
    with pytest.raises(ValueError) as excinfo:
        case_result(expected=value)
    message = str(excinfo.value)
    assert "CaseResult.expected" in message
    assert type(value).__name__ in message


@pytest.mark.parametrize("name", PAYLOAD_FIELDS)
@pytest.mark.parametrize("value", MAPPING_PAYLOADS)
def test_a_mapping_payload_is_accepted(name: str, value: Mapping[str, Any]) -> None:
    """A dict, an empty dict and a read-only mapping proxy all satisfy the schema."""
    assert case_result(**{name: value}).as_dict()[name] == value


def test_an_absent_expected_is_accepted() -> None:
    """expected is the one payload a case may leave out."""
    assert case_result(expected=None).expected is None


# --------------------------------------------------------------------------- #
# decoded[TRACEBACK_KEY]: absolute frame paths folded into a portable form
# --------------------------------------------------------------------------- #


def recorded_traceback(call: Callable[[], object]) -> str:
    """Return the traceback text a check records for the exception ``call`` raises."""
    try:
        call()
    except Exception:
        return traceback.format_exc()
    raise AssertionError("call did not raise, so there is no traceback to record")


def stored_traceback(text: str) -> str:
    """Return how the model stores ``text`` as an error result's traceback."""
    result = case_result(
        status=Status.ERROR, reason=Reason.CONTRACT_ERROR, decoded={TRACEBACK_KEY: text}
    )
    return result.decoded[TRACEBACK_KEY]


def test_a_project_frame_is_stored_project_relative() -> None:
    """An internal error's own frames name qrcheck/..., not the clone they ran in."""
    text = recorded_traceback(lambda: build(b"A" * 5000, "M", version=1))
    # The precondition is what lets this test fail: a real traceback spells every frame's
    # file as an absolute path.
    assert 'File "/' in text
    portable = stored_traceback(text)
    assert 'File "/' not in portable
    assert re.search(r'File "qrcheck/checks/support\.py", line \d+, in build', portable)
    assert re.search(r'File "qrcheck/targets/base\.py", line \d+, in build_request', portable)
    assert "ValueError: payload is 5000 bytes" in portable


def test_a_frame_outside_the_project_keeps_its_last_two_segments() -> None:
    """A standard-library frame keeps ``.../<package>/<module>.py`` and nothing above it."""
    text = recorded_traceback(lambda: json.loads("{"))
    assert 'File "/' in text
    portable = stored_traceback(text)
    assert 'File "/' not in portable
    assert re.search(r'File "\.\.\./json/decoder\.py", line \d+, in \w+', portable)


def test_the_rewrite_of_a_real_traceback_is_idempotent() -> None:
    """A second pass over a real error traceback finds nothing left to rewrite."""
    once = stored_traceback(recorded_traceback(lambda: build(b"A" * 5000, "M", version=1)))
    assert stored_traceback(once) == once


PORTABLE_TRACEBACKS = (
    pytest.param('  File ".../json/decoder.py", line 355, in raw_decode', id="marker-form"),
    pytest.param('  File "qrcheck/checks/support.py", line 238, in build', id="project-relative"),
    pytest.param("qrcheck/x.py beside .../y.py", id="relative-and-marker"),
    pytest.param("see http://127.0.0.1:8000/a/b and https://example.com/a/b", id="urls"),
)


@pytest.mark.parametrize("text", PORTABLE_TRACEBACKS)
def test_an_already_portable_traceback_is_stored_untouched(text: str) -> None:
    """A portable text, and a URL that only looks like a path, are left exactly alone."""
    decoded = {TRACEBACK_KEY: text}
    result = case_result(status=Status.ERROR, reason=Reason.CONTRACT_ERROR, decoded=decoded)
    assert result.decoded is decoded
    assert result.decoded[TRACEBACK_KEY] == text


def test_only_the_traceback_text_is_rewritten() -> None:
    """Every other decoded key, and all of request and expected, are stored as given."""
    request = {"cwd": "/var/lib/scratch", "payload_b64": "aGk="}
    expected = {"symbol_path": "/var/lib/scratch/symbol.png"}
    decoded = {
        TRACEBACK_KEY: '  File "/var/lib/scratch/x.py", line 7, in run',
        "per_decoder": {"opencv": {"status": "undetected"}, "zxing": {"status": "ok"}},
        "version": 3,
    }
    result = case_result(
        status=Status.ERROR,
        reason=Reason.CONTRACT_ERROR,
        request=request,
        decoded=decoded,
        expected=expected,
    )
    assert result.decoded[TRACEBACK_KEY] == '  File ".../scratch/x.py", line 7, in run'
    assert result.decoded["per_decoder"] == decoded["per_decoder"]
    assert result.decoded["version"] == 3
    assert sorted(result.decoded) == sorted(decoded)
    assert result.request is request
    assert result.expected is expected


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(None, id="none"),
        pytest.param(7, id="int"),
        pytest.param({"frame": "/var/lib/x/y.py"}, id="mapping"),
    ),
)
def test_a_non_string_traceback_is_stored_as_given(value: object) -> None:
    """Only a string can carry a path, so any other JSON value is left as it was given."""
    decoded = {TRACEBACK_KEY: value}
    result = case_result(status=Status.ERROR, reason=Reason.CONTRACT_ERROR, decoded=decoded)
    assert result.decoded is decoded


# --------------------------------------------------------------------------- #
# evidence_dir: a non-empty relative posix path
# --------------------------------------------------------------------------- #


def test_an_empty_evidence_dir_is_refused() -> None:
    """The empty string would make the evidence root itself the case directory."""
    with pytest.raises(ValueError, match="non-empty relative posix path"):
        case_result(evidence_dir="")


@pytest.mark.parametrize(
    "evidence_dir",
    (
        EVIDENCE_DIR,
        "evidence/SY-EC-MATCH/SY-EC-MATCH__corpus_url-001__value_mismatch",
        "a",
    ),
    ids=("layout", "two-segments", "single-segment"),
)
def test_a_relative_posix_evidence_dir_is_accepted(evidence_dir: str) -> None:
    """The relative form the evidence store writes is the form the model accepts."""
    assert case_result(evidence_dir=evidence_dir).evidence_dir == evidence_dir


# --------------------------------------------------------------------------- #
# RunResult: one instant range, one case identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("started_at", "finished_at"),
    (
        ("2026-01-01T00:00:05+00:00", "2026-01-01T00:00:00+00:00"),
        ("2026-01-01T01:00:06+01:00", "2026-01-01T00:00:05+00:00"),
        ("2026-01-01T00:00:05.500+00:00", "2026-01-01T00:00:05.499+00:00"),
    ),
    ids=("same-zone", "across-zones", "sub-second"),
)
def test_a_run_that_finishes_before_it_starts_is_refused(
    started_at: str, finished_at: str
) -> None:
    """A negative run duration would be rendered as fact in two artefacts."""
    with pytest.raises(ValueError) as excinfo:
        run_result(started_at=started_at, finished_at=finished_at)
    message = str(excinfo.value)
    assert "started_at" in message
    assert "finished_at" in message
    assert started_at in message
    assert finished_at in message


@pytest.mark.parametrize(
    ("started_at", "finished_at"),
    (
        ("2026-01-01T00:00:05+00:00", "2026-01-01T00:00:05+00:00"),
        ("2026-01-01T00:00:05+00:00", "2026-01-01T01:00:05+01:00"),
    ),
    ids=("identical", "identical-across-zones"),
)
def test_a_zero_length_run_is_accepted(started_at: str, finished_at: str) -> None:
    """A run may take no measurable time; equal instants are not a reversal."""
    assert run_result(started_at=started_at, finished_at=finished_at).duration_ms == 0.0


@pytest.mark.parametrize(
    ("started_at", "finished_at", "expected_ms"),
    (
        ("2026-01-01T00:00:05+00:00", "2026-01-01T00:00:05.250+00:00", 250.0),
        ("2026-01-01T00:00:05+00:00", "2026-01-01T01:00:07.500+01:00", 2500.0),
    ),
    ids=("sub-second", "across-zones"),
)
def test_a_forward_run_reports_a_positive_duration(
    started_at: str, finished_at: str, expected_ms: float
) -> None:
    """Elapsed time is the difference between two instants, whatever their zone."""
    result = run_result(started_at=started_at, finished_at=finished_at)
    assert result.duration_ms == pytest.approx(expected_ms)
    assert result.duration_s == pytest.approx(expected_ms / 1000.0)


def test_a_duplicate_case_identity_is_refused() -> None:
    """Two rows for one case would share one failure ID and one evidence directory."""
    with pytest.raises(ValueError) as excinfo:
        run_result(results=(case_result(), case_result()), counts=Counts(failed=2))
    message = str(excinfo.value)
    assert "RT-ROUNDTRIP" in message
    assert "corpus:url-001" in message


def test_a_duplicate_that_differs_only_in_reason_is_refused() -> None:
    """The case identity is (check_id, case_id); the cause is not part of it."""
    results = (case_result(), case_result(reason=Reason.TARGET_TIMEOUT))
    with pytest.raises(ValueError, match="corpus:url-001"):
        run_result(results=results, counts=Counts(failed=2))


def test_every_repeated_pair_is_named_in_one_diagnostic() -> None:
    """Sorting groups each repeat, and one message lists all of them."""
    results = (
        case_result(),
        case_result(check_id="SY-EC-MATCH", case_id="corpus:url-002"),
        case_result(),
        case_result(check_id="SY-EC-MATCH", case_id="corpus:url-002"),
    )
    with pytest.raises(ValueError) as excinfo:
        run_result(results=results, counts=Counts(failed=4))
    message = str(excinfo.value)
    assert "'RT-ROUNDTRIP'/'corpus:url-001'" in message
    assert "'SY-EC-MATCH'/'corpus:url-002'" in message


def test_the_same_case_id_under_two_check_ids_is_accepted() -> None:
    """Two checks may legitimately run the same corpus record in one run."""
    results = (
        case_result(check_id="RT-ROUNDTRIP", case_id="corpus:url-001"),
        case_result(check_id="SY-EC-MATCH", case_id="corpus:url-001"),
    )
    result = run_result(results=results, counts=Counts(failed=2))
    assert [(case.check_id, case.case_id) for case in result.results] == [
        ("RT-ROUNDTRIP", "corpus:url-001"),
        ("SY-EC-MATCH", "corpus:url-001"),
    ]


# --------------------------------------------------------------------------- #
# Reason: the vocabulary names every non-passed cause
# --------------------------------------------------------------------------- #


def test_an_internal_fault_has_a_reason_of_its_own() -> None:
    """An ``error`` row names the framework's fault, not a check's verdict.

    AAP §0.6.6 row 11 contains an internal exception as an ``error`` result carrying
    its traceback instead of aborting the run, so its cause needs a word of its own:
    without one, a real capacity-gate ``ValueError`` raised on the framework side can
    only be filed as ``capacity_violation`` - a check's verdict - and the ``reason``
    ``summary.json`` records and the ``<error message=...>`` ``junit.xml`` emits then
    describe the row as something it is not.
    """
    assert Reason.INTERNAL_ERROR.value == "internal_error"
    assert Reason("internal_error") is Reason.INTERNAL_ERROR
    result = case_result(
        check_id="IV-OVERCAPACITY",
        case_id="synthetic:internal-error",
        status=Status.ERROR,
        reason=Reason.INTERNAL_ERROR,
    )
    assert result.as_dict()["reason"] == "internal_error"
