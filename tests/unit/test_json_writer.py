"""The ``summary.json`` schema, its ordering and its exact volatile-field inventory.

AAP §0.6.1 gate 6 names this module as the gate holding ``summary.json``, the artefact a CI
consumer, a trend line and a triage script all parse. Every document asserted here is written by
:func:`qrcheck.report.json_report.write_summary` from a :class:`~qrcheck.report.model.RunResult`
built by the shared factories, so nothing is transcribed from a run's output and no second
formatter can drift in (AAP §0.3.4, §0.5.3-§0.5.5).

Pinned: the nine top-level keys by **set equality**; ``counts`` as four separate integers,
because a tally folding ``skipped`` into ``passed`` would satisfy every other assertion while
breaking the exit-code rule; the ten per-result fields and their vocabulary, with a pass's
``reason`` ``null``; ``results[]`` ordered by ``(check_id, case_id)`` though the fixture is
deliberately scrambled; ``evidence_dir`` ``null`` for a pass and a relative posix path otherwise,
since an absolute one names the clone; the canonical text form (sorted keys at every depth,
two-space indent, UTF-8, LF, one trailing newline, unserialisable values raising); the volatile
inventory of exactly ``started_at``, ``finished_at`` and each ``results[].duration_ms``; and
``run_id`` as a content hash stable for one input and different for any changed selection, EC
level, seed or corpus.
"""

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.evidence.naming import EVIDENCE_DIR_NAME, relative_case_evidence_dir
from qrcheck.report.json_report import dumps_canonical, write_canonical_json, write_summary
from qrcheck.report.model import CaseResult, Counts, Reason, RunResult, Status
from qrcheck.runner import _run_identifier

# The exact schema, stated once. AAP §0.3.4 fixes both key sets as a stable public surface, so each
# is compared by set equality against the payload a real write produced.
_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
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
_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "case_id",
        "check_id",
        "decoded",
        "duration_ms",
        "evidence_dir",
        "expected",
        "reason",
        "request",
        "seed",
        "status",
    }
)

# The four tallies, kept separate, read from nothing but this declaration.
_COUNT_FIELDS: Final[frozenset[str]] = frozenset({"error", "failed", "passed", "skipped"})

# The complete volatile inventory of AAP §0.5.4: two run timestamps and each result's duration.
_VOLATILE_RUN_FIELDS: Final[frozenset[str]] = frozenset({"started_at", "finished_at"})
_VOLATILE_RESULT_FIELDS: Final[frozenset[str]] = frozenset({"duration_ms"})

# The two closed vocabularies, taken from the enums themselves rather than retyped, so a member
# added to the model is accepted here without this module carrying a stale list.
_STATUS_VALUES: Final[frozenset[str]] = frozenset(member.value for member in Status)
_REASON_VALUES: Final[frozenset[str]] = frozenset(member.value for member in Reason)

# The artefact this module writes into the per-test report root.
_SUMMARY_NAME: Final[str] = "summary.json"

# Payload text carried in the failed case's request mapping, non-ASCII on purpose: with
# ``ensure_ascii=False`` it appears as itself, and a writer that escaped it would turn the
# ``emoji``, ``rtl`` and ``utf8-multibyte`` payloads into unreadable evidence.
_NON_ASCII_PAYLOAD: Final[str] = "héllo 🎉 مرحبا"

# The fixture run's two timestamp pairs: fixed constants, never a clock reading, so two runs built
# by these helpers differ only where the test wants them to.
_BASE_STARTED_AT: Final[str] = "2024-01-01T00:00:00+00:00"
_BASE_FINISHED_AT: Final[str] = "2024-01-01T00:00:01+00:00"
_LATER_STARTED_AT: Final[str] = "2025-06-30T12:34:56+00:00"
_LATER_FINISHED_AT: Final[str] = "2025-06-30T12:35:07+00:00"

# One duration per case, in the role order :func:`_mixed_results` declares; the two tuples differ in
# this field alone.
_BASE_DURATIONS: Final[tuple[float, ...]] = (1.5, 2.25, 12.5, 0.5, 30.0)
_LATER_DURATIONS: Final[tuple[float, ...]] = (9.75, 4.0, 1.25, 7.5, 0.125)

# The order the fixture run's cases are handed to ``RunResult`` in: deliberately not sorted, so the
# ordering assertion compares against an order the model had to produce.
_CONSTRUCTION_ORDER: Final[tuple[str, ...]] = (
    "skip-symbol",
    "pass-second",
    "error-cleanup",
    "pass-roundtrip",
    "fail-roundtrip",
)

# The signature of the shared masking helper, as ``tests/conftest.py`` supplies it.
_MaskVolatile = Callable[[RunResult | Mapping[str, Any]], dict[str, Any]]

# A mutator receives the two result factories and the base run, and returns the run with exactly one
# documented non-volatile field changed.
_Mutation = Callable[[Callable[..., CaseResult], Callable[..., RunResult], RunResult], RunResult]


# --- The fixture run: five cases reaching all four statuses ---


def _mixed_results(
    make_case_result: Callable[..., CaseResult],
    *,
    durations: tuple[float, ...] = _BASE_DURATIONS,
) -> dict[str, CaseResult]:
    """Return the fixture run's five case results keyed by role, reaching all four statuses."""
    roundtrip_pass, second_pass, roundtrip_fail, symbol_skip, cleanup_error = durations
    fail_check, fail_case = "RT-ROUNDTRIP", "corpus:url-003"
    skip_check, skip_case = "SY-MASK", "corpus:url-004"
    error_check, error_case = "CL-ARTIFACTS", "synthetic:artifacts"
    return {
        "pass-roundtrip": make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:url-001",
            status=Status.PASSED,
            duration_ms=roundtrip_pass,
            request={"payload_utf8": "https://example.test/qr"},
            decoded={"agreement": "agree"},
        ),
        "pass-second": make_case_result(
            check_id="RT-ROUNDTRIP",
            case_id="corpus:url-002",
            status=Status.PASSED,
            duration_ms=second_pass,
            request={"payload_utf8": "https://example.test/qr/2"},
            decoded={"agreement": "primary_only"},
        ),
        "fail-roundtrip": make_case_result(
            check_id=fail_check,
            case_id=fail_case,
            status=Status.FAILED,
            reason=Reason.VALUE_MISMATCH,
            duration_ms=roundtrip_fail,
            request={"payload_utf8": _NON_ASCII_PAYLOAD},
            decoded={"payload_utf8": "héllo"},
            expected={"payload_utf8": _NON_ASCII_PAYLOAD},
            evidence_dir=relative_case_evidence_dir(
                fail_check, fail_case, Reason.VALUE_MISMATCH
            ),
        ),
        "skip-symbol": make_case_result(
            check_id=skip_check,
            case_id=skip_case,
            status=Status.SKIPPED,
            reason=Reason.UNSUPPORTED_CAPABILITY,
            duration_ms=symbol_skip,
            request={"capability": "mask_pin"},
            evidence_dir=relative_case_evidence_dir(
                skip_check, skip_case, Reason.UNSUPPORTED_CAPABILITY
            ),
        ),
        "error-cleanup": make_case_result(
            check_id=error_check,
            case_id=error_case,
            status=Status.ERROR,
            reason=Reason.INTERNAL_ERROR,
            duration_ms=cleanup_error,
            request={"observation": "tempdir snapshot"},
            decoded={"message": "RuntimeError: boom"},
            evidence_dir=relative_case_evidence_dir(
                error_check, error_case, Reason.INTERNAL_ERROR
            ),
        ),
    }


def _build_run(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    started_at: str = _BASE_STARTED_AT,
    finished_at: str = _BASE_FINISHED_AT,
    durations: tuple[float, ...] = _BASE_DURATIONS,
    **overrides: Any,
) -> RunResult:
    """Return the fixture run with its cases handed over in a scrambled order."""
    cases = _mixed_results(make_case_result, durations=durations)
    return make_run_result(
        tuple(cases[role] for role in _CONSTRUCTION_ORDER),
        started_at=started_at,
        finished_at=finished_at,
        **overrides,
    )


def _rebuild(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    base: RunResult,
    **overrides: Any,
) -> RunResult:
    """Return the fixture run with `overrides` applied and `base`'s identity held fixed."""
    return _build_run(
        make_case_result, make_run_result, run_id=base.run_id, **overrides
    )


def _run_with_changed_result(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    role: str,
    **changes: Any,
) -> RunResult:
    """Return the fixture run with exactly one case's field(s) altered."""
    unchanged = _build_run(make_case_result, make_run_result)
    cases = _mixed_results(make_case_result)
    cases[role] = dataclasses.replace(cases[role], **changes)
    return make_run_result(
        tuple(cases[name] for name in _CONSTRUCTION_ORDER),
        started_at=_BASE_STARTED_AT,
        finished_at=_BASE_FINISHED_AT,
        run_id=unchanged.run_id,
    )


# The documented non-volatile fields and the change that alters each one, exactly once. The
# identifiers are the field names, so a failing case names the field rather than an index; a
# ``check_id`` change leaves the evidence directory alone so exactly one field moves.
_NONVOLATILE_CHANGES: Final[dict[str, _Mutation]] = {
    "adapter": lambda make_case_result, make_run_result, base: _rebuild(
        make_case_result, make_run_result, base, adapter="defective"
    ),
    "case_id": lambda make_case_result, make_run_result, base: (
        _run_with_changed_result(
            make_case_result,
            make_run_result,
            role="pass-roundtrip",
            case_id="corpus:url-009",
        )
    ),
    "check_id": lambda make_case_result, make_run_result, base: (
        _run_with_changed_result(
            make_case_result,
            make_run_result,
            role="fail-roundtrip",
            check_id="SY-VERSION-MIN",
        )
    ),
    "ec_level": lambda make_case_result, make_run_result, base: _rebuild(
        make_case_result, make_run_result, base, ec_level="H"
    ),
    "evidence_dir": lambda make_case_result, make_run_result, base: (
        _run_with_changed_result(
            make_case_result,
            make_run_result,
            role="fail-roundtrip",
            evidence_dir=relative_case_evidence_dir(
                "RT-ROUNDTRIP", "corpus:url-009", Reason.VALUE_MISMATCH
            ),
        )
    ),
    "loc_count": lambda make_case_result, make_run_result, base: _rebuild(
        make_case_result, make_run_result, base, loc_count=base.loc_count + 1
    ),
    "reason": lambda make_case_result, make_run_result, base: (
        _run_with_changed_result(
            make_case_result,
            make_run_result,
            role="fail-roundtrip",
            reason=Reason.SAMPLING_FAILED,
        )
    ),
    "run_id": lambda make_case_result, make_run_result, base: _build_run(
        make_case_result,
        make_run_result,
        run_id=hashlib.sha256((base.run_id + "!").encode("utf-8")).hexdigest(),
    ),
    "seed": lambda make_case_result, make_run_result, base: _rebuild(
        make_case_result, make_run_result, base, seed=base.seed + 1
    ),
    "status": lambda make_case_result, make_run_result, base: (
        _run_with_changed_result(
            make_case_result,
            make_run_result,
            role="fail-roundtrip",
            status=Status.SKIPPED,
        )
    ),
}


# --- Reading an artefact back ---


def _summary_path(report_dir: Path) -> Path:
    """Return the summary file's path under the per-test report root."""
    return report_dir / _SUMMARY_NAME


def _write_and_load(path: Path, run: RunResult) -> dict[str, Any]:
    """Write ``run`` to ``path`` through the product's writer and return the parsed payload."""
    write_summary(run, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _masked_text(path: Path, mask_volatile: _MaskVolatile) -> str:
    """Return `path`'s payload as canonical text with its volatile fields masked."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return dumps_canonical(mask_volatile(payload))


def _assert_every_object_has_sorted_keys(path: Path) -> None:
    """Fail if any JSON object in `path` holds its keys in a non-sorted order."""
    unsorted: list[tuple[str, ...]] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = tuple(key for key, _ in pairs)
        if keys != tuple(sorted(keys)):
            unsorted.append(keys)
        return dict(pairs)

    json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    assert unsorted == [], f"objects with unsorted keys: {unsorted}"


def _indents_of(text: str, fragment: str) -> frozenset[int]:
    """Return the leading-space counts of every line whose content starts with `fragment`."""
    lines = [line for line in text.splitlines() if line.lstrip(" ").startswith(fragment)]
    assert lines, f"no line starts with {fragment!r}"
    return frozenset(len(line) - len(line.lstrip(" ")) for line in lines)


def _differing_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> frozenset[str]:
    """Return the keys whose value differs between two mappings of the same shape."""
    assert set(before) == set(after), "the two mappings must have the same keys"
    return frozenset(key for key in before if before[key] != after[key])


def _scalars_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a summary payload's top-level fields other than ``results``."""
    return {key: value for key, value in payload.items() if key != "results"}


def _reread(path: Path) -> bytes:
    """Return ``path``'s bytes, so a comparison is made on the artefact a consumer parses."""
    return path.read_bytes()


# --- The schema, exactly as AAP §0.3.4 states it ---


def test_summary_has_exactly_the_documented_top_level_keys(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The payload holds the nine documented fields and no others."""
    run = _build_run(make_case_result, make_run_result)
    payload = _write_and_load(_summary_path(report_dir), run)
    # Set equality, never containment: an unknown key is either an invented field a consumer would
    # not expect or an addition made deliberately -- by changing this assertion. Future keys are
    # only ADDED.
    assert set(payload) == _TOP_LEVEL_FIELDS


def test_a_run_with_no_results_still_carries_the_full_schema(
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """An empty run emits the same nine keys, four zero tallies and an empty ``results``."""
    run = make_run_result(())
    payload = _write_and_load(_summary_path(report_dir), run)
    assert set(payload) == _TOP_LEVEL_FIELDS
    assert payload["counts"] == Counts().as_dict()
    assert payload["results"] == []


def test_counts_are_four_separate_integers(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """``counts`` is four integer tallies that follow the statuses one for one."""
    run = _build_run(make_case_result, make_run_result)
    payload = _write_and_load(_summary_path(report_dir), run)
    counts = payload["counts"]
    assert isinstance(counts, Mapping)
    assert set(counts) == _COUNT_FIELDS
    for name in sorted(_COUNT_FIELDS):
        # ``type(...) is int`` rather than ``isinstance``: ``bool`` is an ``int`` subclass, so a
        # tally that had become a truth value would slip past an ``isinstance`` check.
        assert type(counts[name]) is int, (name, type(counts[name]))
    # The tally is read off the emitted statuses, so a skip folded into a pass is visible here and
    # nowhere else in this module.
    implied = {status.value: 0 for status in Status}
    for result in payload["results"]:
        implied[Status(result["status"]).value] += 1
    assert counts == implied
    assert sum(counts.values()) == len(payload["results"])


def test_each_result_carries_exactly_the_documented_fields(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """Every ``results[]`` entry has the ten documented fields and a valid vocabulary."""
    run = _build_run(make_case_result, make_run_result)
    payload = _write_and_load(_summary_path(report_dir), run)
    seen: set[Status] = set()
    for result in payload["results"]:
        assert set(result) == _RESULT_FIELDS
        assert result["status"] in _STATUS_VALUES
        status = Status(result["status"])
        seen.add(status)
        if status is Status.PASSED:
            assert result["reason"] is None
        else:
            assert result["reason"] in _REASON_VALUES
    # The fixture run reaches all four outcomes, so neither branch above was vacuous.
    assert seen == set(Status)


def test_results_are_emitted_in_check_id_then_case_id_order(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """``results[]`` is ordered by ``(check_id, case_id)`` whatever order it was given in."""
    run = _build_run(make_case_result, make_run_result)
    payload = _write_and_load(_summary_path(report_dir), run)
    emitted = [(result["check_id"], result["case_id"]) for result in payload["results"]]
    cases = _mixed_results(make_case_result)
    constructed = [(cases[role].check_id, cases[role].case_id) for role in _CONSTRUCTION_ORDER]
    # The assertion is only worth making if the input really was out of order.
    assert constructed != sorted(constructed)
    assert emitted != constructed
    assert emitted == sorted(constructed)
    assert emitted == sorted(emitted)


def test_evidence_dir_is_null_for_a_pass_and_relative_for_anything_else(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """A pass records no evidence path; every other status records a relative posix one."""
    run = _build_run(make_case_result, make_run_result)
    payload = _write_and_load(_summary_path(report_dir), run)
    passed = [result for result in payload["results"] if result["status"] == Status.PASSED.value]
    other = [result for result in payload["results"] if result["status"] != Status.PASSED.value]
    # Neither branch may be vacuous, or half this test would assert nothing.
    assert passed
    assert other
    for result in passed:
        assert result["evidence_dir"] is None
    for result in other:
        value = result["evidence_dir"]
        assert isinstance(value, str)
        assert value
        # No absolute path may reach an artefact (AAP §0.5.5): it would name the clone the run
        # happened in, and two clones would disagree about the same case.
        assert not Path(value).is_absolute()
        assert not value.startswith("/")
        assert "\\" not in value
        segments = value.split("/")
        assert ".." not in segments
        assert segments[0] == EVIDENCE_DIR_NAME


def test_summary_text_is_canonical_json(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The file is canonical JSON text: sorted keys, two-space indent, UTF-8, one newline."""
    run = _build_run(make_case_result, make_run_result)
    path = write_summary(run, _summary_path(report_dir))
    data = _reread(path)
    text = data.decode("utf-8")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    # The strongest statement of the canonical form: the file is exactly what the one formatter
    # produces from its own payload, plus the single trailing newline.
    assert text == dumps_canonical(json.loads(text)) + "\n"
    _assert_every_object_has_sorted_keys(path)
    # Two-space indentation, asserted at three depths and once across two depths at a time: a
    # top-level field sits two spaces in, a ``counts`` member four, a result's field six, a nested
    # mapping eight.
    assert _indents_of(text, '"run_id":') == {2}
    assert _indents_of(text, '"counts": {') == {2}
    assert _indents_of(text, '"results": [') == {2}
    assert _indents_of(text, '"error":') == {4}
    assert _indents_of(text, '"case_id":') == {6}
    assert _indents_of(text, '"status":') == {6}
    assert _indents_of(text, '"payload_utf8":') == {8}
    assert _indents_of(text, '"seed":') == {2, 6}
    assert "\t" not in text
    # ``ensure_ascii=False``: the payload appears as itself, not as ``\uXXXX`` escapes, which keeps
    # the ``emoji``, ``rtl`` and ``utf8-multibyte`` evidence readable.
    assert _NON_ASCII_PAYLOAD.encode("utf-8") in data


def test_a_value_json_cannot_carry_is_refused_rather_than_coerced(tmp_path: Path) -> None:
    """A non-serialisable value raises ``TypeError`` and leaves no artefact behind."""
    path = tmp_path / "summary.json"
    payload = {"adapter": object()}
    with pytest.raises(TypeError):
        dumps_canonical(payload)
    with pytest.raises(TypeError):
        write_canonical_json(payload, path)
    # A caller defect must not leave a plausible-looking partial artefact for a consumer.
    assert not path.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_float_is_refused_rather_than_written(value: float, tmp_path: Path) -> None:
    """A non-finite float raises ``ValueError``: JSON defines no token for one."""
    path = tmp_path / "summary.json"
    with pytest.raises(ValueError):
        dumps_canonical({"throughput": value})
    with pytest.raises(ValueError):
        write_canonical_json({"throughput": value}, path)
    assert not path.exists()


def test_write_canonical_json_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    """The writer creates the directories on its path rather than requiring them."""
    path = tmp_path / "missing" / "deeper" / "payload.json"
    payload = {"adapter": "reference", "seed": 1, "unbalanced": ["x", 2, None, True]}
    assert write_canonical_json(payload, path) == path
    assert path.read_text(encoding="utf-8") == dumps_canonical(payload) + "\n"


# --- The volatile-field inventory of AAP §0.5.4 ---


def test_masking_the_volatile_fields_makes_two_runs_byte_equal(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
    mask_volatile: _MaskVolatile,
) -> None:
    """Two runs differing only in timestamps and durations are byte-equal once masked."""
    base = _build_run(make_case_result, make_run_result)
    later = _build_run(
        make_case_result,
        make_run_result,
        started_at=_LATER_STARTED_AT,
        finished_at=_LATER_FINISHED_AT,
        durations=_LATER_DURATIONS,
    )
    base_path = write_summary(base, report_dir / "base.json")
    later_path = write_summary(later, report_dir / "later.json")
    # First prove the difference is real and confined to the volatile fields, so the equality below
    # is the masking at work and not two identical runs agreeing with themselves.
    assert _reread(base_path) != _reread(later_path)
    assert _differing_keys(_scalars_of(base.as_dict()), _scalars_of(later.as_dict())) == (
        _VOLATILE_RUN_FIELDS
    )
    for original, altered in zip(base.as_dict()["results"], later.as_dict()["results"]):
        assert _differing_keys(original, altered) == _VOLATILE_RESULT_FIELDS
    assert _masked_text(base_path, mask_volatile) == _masked_text(later_path, mask_volatile)


def test_mask_volatile_masks_exactly_the_documented_inventory(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    mask_volatile: _MaskVolatile,
) -> None:
    """The shared masking helper touches exactly three fields, and no more may join them."""
    run = _build_run(make_case_result, make_run_result)
    before = run.as_dict()
    masked = mask_volatile(run)
    # Nothing is added or dropped, and the order of the results is preserved.
    assert set(masked) == set(before)
    assert _differing_keys(_scalars_of(before), _scalars_of(masked)) == _VOLATILE_RUN_FIELDS
    assert len(masked["results"]) == len(before["results"])
    for original, altered in zip(before["results"], masked["results"]):
        assert set(altered) == set(original)
        assert _differing_keys(original, altered) == _VOLATILE_RESULT_FIELDS
    # The helper copies: the run it masked still reports what it reported.
    assert run.as_dict() == before


@pytest.mark.parametrize("change", sorted(_NONVOLATILE_CHANGES))
def test_any_non_volatile_change_breaks_the_masked_equality(
    change: str,
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
    mask_volatile: _MaskVolatile,
) -> None:
    """A change to any documented non-volatile field survives the masking and is visible."""
    base = _build_run(make_case_result, make_run_result)
    changed = _NONVOLATILE_CHANGES[change](make_case_result, make_run_result, base)
    base_path = write_summary(base, report_dir / "base.json")
    changed_path = write_summary(changed, report_dir / "changed.json")
    assert changed.as_dict() != base.as_dict()
    assert _reread(changed_path) != _reread(base_path)
    assert _masked_text(changed_path, mask_volatile) != _masked_text(base_path, mask_volatile)


# --- ``run_id``: a content hash, never a UUID and never a clock reading ---


def test_the_run_identifier_is_a_content_hash_stable_in_its_inputs(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    corpus_subset: tuple[Any, ...],
) -> None:
    """The derivation repeats for one input and differs for every changed input."""
    base = _build_run(make_case_result, make_run_result)
    checks = ("RT-ROUNDTRIP", "SY-MASK")
    identifier = _run_identifier("reference", "M", base.seed, checks, corpus_subset)
    # A content hash: the same inputs give the same identity, in any input order.
    assert identifier == _run_identifier("reference", "M", base.seed, checks, corpus_subset)
    assert identifier == _run_identifier(
        "reference", "M", base.seed, ("SY-MASK", "RT-ROUNDTRIP"), corpus_subset
    )
    # A SHA-256 hex digest, so it can never be the UUID or timestamp the model refuses -- and the
    # model accepts it, which is the other half of that contract.
    assert re.fullmatch(r"[0-9a-f]{64}", identifier) is not None
    assert make_run_result((), run_id=identifier).run_id == identifier
    # Every component the identity covers changes the identity.
    assert identifier != _run_identifier("defective", "M", base.seed, checks, corpus_subset)
    assert identifier != _run_identifier("reference", "H", base.seed, checks, corpus_subset)
    assert identifier != _run_identifier("reference", "M", base.seed + 1, checks, corpus_subset)
    assert identifier != _run_identifier(
        "reference", "M", base.seed, ("RT-ROUNDTRIP",), corpus_subset
    )
    assert len(corpus_subset) > 1
    assert identifier != _run_identifier("reference", "M", base.seed, checks, corpus_subset[:1])


@pytest.mark.parametrize(
    "run_id",
    [
        pytest.param("", id="empty"),
        pytest.param("123e4567-e89b-12d3-a456-426614174000", id="uuid"),
        pytest.param("2024-01-01", id="date"),
        pytest.param("2024-01-01T00:00:00+00:00", id="timestamp"),
    ],
)
def test_the_result_model_refuses_a_run_id_that_is_not_a_content_hash(
    run_id: str,
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
) -> None:
    """An empty, UUID-shaped or timestamp-shaped identity is refused by the model."""
    cases = tuple(_mixed_results(make_case_result).values())
    with pytest.raises(ValueError, match="run_id"):
        make_run_result(cases, run_id=run_id)


# --- The writer adds, drops and transforms nothing ---


def test_write_summary_emits_exactly_the_model_projection(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """The parsed file equals ``RunResult.as_dict()``, field for field and value for value."""
    run = _build_run(make_case_result, make_run_result)
    payload = _write_and_load(_summary_path(report_dir), run)
    assert payload == run.as_dict()

