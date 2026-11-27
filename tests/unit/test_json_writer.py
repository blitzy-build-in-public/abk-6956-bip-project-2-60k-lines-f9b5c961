"""The ``summary.json`` schema, its ordering and its exact volatile-field inventory.

This module holds ``summary.json``, the artefact a CI consumer, a trend line and a triage script
all parse. Every document asserted here is written by
:func:`qrcheck.report.json_report.write_summary` from a :class:`~qrcheck.report.model.RunResult`
built by the shared factories, so nothing is transcribed from a run's output and no second
formatter can drift in.

Pinned: the nine top-level keys by **set equality**; ``counts`` as four separate integers,
because a tally folding ``skipped`` into ``passed`` would satisfy every other assertion while
breaking the exit-code rule; the ten per-result fields and their vocabulary, with a pass's
``reason`` ``null``; ``results[]`` ordered by ``(check_id, case_id)`` though the fixture is
deliberately scrambled; ``evidence_dir`` ``null`` for a pass and a relative posix path otherwise,
since an absolute one names the clone; the canonical text form (sorted keys at every depth,
two-space indent, UTF-8, LF, one trailing newline, unserialisable values raising, and a code point
UTF-8 cannot encode replaced by U+FFFD so a target's lone surrogate is written instead of aborting
the run that was reporting it); the volatile
inventory of exactly ``started_at``, ``finished_at``, each ``results[].duration_ms`` and the
measured leaves of each ``PF-*`` probe's ``decoded.metrics``, with the configuration-derived
metrics still compared; and ``run_id``, read off a real run through the public
:func:`qrcheck.runner.run_suite` path and never from the private derivation, as a SHA-256 content
hash that repeats for identical inputs and moves when the adapter name, the selected check set, the
EC level, the seed or the ``--class`` corpus filter changes -- which pins the five arguments the
runner itself supplies to the derivation, not the derivation alone.
"""

import copy
import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.config.models import Config
from qrcheck.evidence.naming import EVIDENCE_DIR_NAME, relative_case_evidence_dir
from qrcheck.registry.selection import Selection
from qrcheck.report.json_report import dumps_canonical, write_canonical_json, write_summary
from qrcheck.report.model import CaseResult, Counts, Reason, RunResult, Status
from qrcheck.runner import run_suite
from qrcheck.targets.base import Adapter

# The exact schema, stated once. The writer emits both key sets as a stable public surface, so each
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

# The complete volatile inventory: two run timestamps, each result's duration, and the measured
# leaves of a ``PF-*`` result's ``decoded.metrics``. The two performance probes must record their
# measurement, so the determinism comparison masks it instead of dropping it, and the masking keeps
# it out of every comparison: a measured latency or throughput is judged once against the configured
# floor and never against another run's measurement.
_VOLATILE_RUN_FIELDS: Final[frozenset[str]] = frozenset({"started_at", "finished_at"})
_VOLATILE_RESULT_FIELDS: Final[frozenset[str]] = frozenset({"duration_ms"})
_VOLATILE_METRIC_FIELDS: Final[frozenset[str]] = frozenset(
    {"encode_p95_ms", "decode_p95_ms", "symbols_per_minute"}
)

# The metrics a probe copies from configuration rather than measuring, so two runs of the same
# configuration agree on them and they must survive every masking.
_STABLE_METRIC_FIELDS: Final[frozenset[str]] = frozenset({"image_px", "samples", "warmup"})

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

# The lone surrogate a target's JSON can carry: ``"\ud800"`` is a legal JSON escape that
# ``json.loads`` materialises as one code point, and UTF-8 has no byte sequence for it. Spelled
# through ``chr`` so this module's own source stays valid UTF-8 and holds no surrogate itself.
_LONE_SURROGATE: Final[str] = chr(0xD800)

# What both writers replace it with: U+FFFD, the replacement character, one code point for one.
_REPLACEMENT_CHARACTER: Final[str] = chr(0xFFFD)

# The fixture run's two timestamp pairs: fixed constants, never a clock reading, so two runs built
# by these helpers differ only where the test wants them to.
_BASE_STARTED_AT: Final[str] = "2024-01-01T00:00:00+00:00"
_BASE_FINISHED_AT: Final[str] = "2024-01-01T00:00:01+00:00"
_LATER_STARTED_AT: Final[str] = "2025-06-30T12:34:56+00:00"
_LATER_FINISHED_AT: Final[str] = "2025-06-30T12:35:07+00:00"

# One duration per case, in the role order :func:`_mixed_results` declares; the two tuples differ in
# this field alone, so the only other difference between the two runs is the probes' measurement.
_BASE_DURATIONS: Final[tuple[float, ...]] = (1.5, 2.25, 12.5, 0.5, 30.0, 4.5, 6.25)
_LATER_DURATIONS: Final[tuple[float, ...]] = (9.75, 4.0, 1.25, 7.5, 0.125, 2.0, 3.5)

# The two performance probes, by the check ID a real run files each under.
_ENCODE_PROBE: Final[str] = "PF-ENCODE"
_DECODE_PROBE: Final[str] = "PF-DECODE"
_PROBE_CHECKS: Final[tuple[str, str]] = (_ENCODE_PROBE, _DECODE_PROBE)

# Each probe's ``decoded.metrics``, shaped like the real result's: the keys that probe publishes.
# The measured leaves differ between the two runs, the way two real runs' measurements do, while the
# metrics configuration fixes -- ``samples``, ``warmup`` and ``image_px`` -- are equal in both,
# because configuration decides them and not the clock.
_BASE_PROBE_METRICS: Final[dict[str, dict[str, float | int]]] = {
    _ENCODE_PROBE: {
        "encode_p95_ms": 0.98,
        "symbols_per_minute": 58630.7,
        "samples": 105,
        "warmup": 5,
    },
    _DECODE_PROBE: {
        "decode_p95_ms": 42.55,
        "image_px": 500,
        "samples": 60,
        "warmup": 5,
    },
}
_LATER_PROBE_METRICS: Final[dict[str, dict[str, float | int]]] = {
    _ENCODE_PROBE: {
        **_BASE_PROBE_METRICS[_ENCODE_PROBE],
        "encode_p95_ms": 1.31,
        "symbols_per_minute": 63260.5,
    },
    _DECODE_PROBE: {
        **_BASE_PROBE_METRICS[_DECODE_PROBE],
        "decode_p95_ms": 39.83,
    },
}

# The measured leaves each probe publishes: its block minus the metrics configuration fixes. The
# sets together are the documented volatile inventory, so the fixture exercises every key a masking
# helper must mask and no key it must leave alone.
_PROBE_MEASURED_FIELDS: Final[dict[str, frozenset[str]]] = {
    probe: frozenset(block) - _STABLE_METRIC_FIELDS
    for probe, block in _BASE_PROBE_METRICS.items()
}

# Each probe's ``expected``: the configured floors its measurement is judged against, equal in both
# runs, so a masked comparison still holds the probe's judgement and not only its numbers.
_PROBE_EXPECTED: Final[dict[str, dict[str, float | int]]] = {
    _ENCODE_PROBE: {"encode_p95_ms": 250.0, "min_symbols_per_minute": 120.0},
    _DECODE_PROBE: {"decode_p95_ms": 400.0, "image_px": 500},
}

# The order the fixture run's cases are handed to ``RunResult`` in: deliberately not sorted, so the
# ordering assertion compares against an order the model had to produce.
_CONSTRUCTION_ORDER: Final[tuple[str, ...]] = (
    "skip-symbol",
    "pass-second",
    "error-cleanup",
    "pass-roundtrip",
    "fail-roundtrip",
    "probe-encode",
    "probe-decode",
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
    probe_metrics: Mapping[str, Mapping[str, float | int]] = _BASE_PROBE_METRICS,
) -> dict[str, CaseResult]:
    """Return the fixture run's seven case results keyed by role, reaching all four statuses."""
    (
        roundtrip_pass,
        second_pass,
        roundtrip_fail,
        symbol_skip,
        cleanup_error,
        encode_probe,
        decode_probe,
    ) = durations
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
        "probe-encode": make_case_result(
            check_id=_ENCODE_PROBE,
            case_id="synthetic:pf-encode",
            status=Status.PASSED,
            duration_ms=encode_probe,
            request={"probe": "encode", "payload_utf8": "https://example.test/qr"},
            decoded={"metrics": dict(probe_metrics[_ENCODE_PROBE])},
            expected=dict(_PROBE_EXPECTED[_ENCODE_PROBE]),
        ),
        "probe-decode": make_case_result(
            check_id=_DECODE_PROBE,
            case_id="synthetic:pf-decode",
            status=Status.PASSED,
            duration_ms=decode_probe,
            request={"probe": "decode", "image_px": 500},
            decoded={"metrics": dict(probe_metrics[_DECODE_PROBE])},
            expected=dict(_PROBE_EXPECTED[_DECODE_PROBE]),
        ),
    }


def _build_run(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    started_at: str = _BASE_STARTED_AT,
    finished_at: str = _BASE_FINISHED_AT,
    durations: tuple[float, ...] = _BASE_DURATIONS,
    probe_metrics: Mapping[str, Mapping[str, float | int]] = _BASE_PROBE_METRICS,
    **overrides: Any,
) -> RunResult:
    """Return the fixture run with its cases handed over in a scrambled order."""
    cases = _mixed_results(make_case_result, durations=durations, probe_metrics=probe_metrics)
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


def _run_with_changed_metrics(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    *,
    role: str,
    probe: str,
    **changes: float | int,
) -> RunResult:
    """Return the fixture run with one probe's ``decoded.metrics`` altered in the named keys."""
    return _run_with_changed_result(
        make_case_result,
        make_run_result,
        role=role,
        decoded={"metrics": {**_BASE_PROBE_METRICS[probe], **changes}},
    )


# The documented non-volatile fields and the change that alters each one, exactly once. The
# identifiers are the field names -- or a dotted path inside one, as ``metrics.samples`` is -- so a
# failing case names the field rather than an index; a ``check_id`` change leaves the evidence
# directory alone so exactly one field moves.
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
    "metrics.samples": lambda make_case_result, make_run_result, base: (
        _run_with_changed_metrics(
            make_case_result,
            make_run_result,
            role="probe-encode",
            probe=_ENCODE_PROBE,
            samples=2 * _BASE_PROBE_METRICS[_ENCODE_PROBE]["samples"],
        )
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


def _probe_metrics(payload: Mapping[str, Any], check_id: str) -> Mapping[str, Any]:
    """Return one performance probe's ``decoded.metrics`` from a summary payload.

    The assertions that read a measurement would pass on a fixture that had lost its probe, so the
    lookup fails loudly rather than defaulting to an empty mapping.
    """
    results = payload["results"]
    assert isinstance(results, list)
    probes = [entry for entry in results if entry.get("check_id") == check_id]
    assert len(probes) == 1, f"expected one {check_id} probe, found {len(probes)}"
    metrics = probes[0].get("decoded", {}).get("metrics")
    assert isinstance(metrics, Mapping), f"{check_id} carries no metrics block: {metrics!r}"
    return metrics


def _probe_metric_differences(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, frozenset[str]]:
    """Return, per probe, the metric keys whose value differs between two summary payloads."""
    return {
        probe: _differing_keys(_probe_metrics(before, probe), _probe_metrics(after, probe))
        for probe in _PROBE_CHECKS
    }


def _reread(path: Path) -> bytes:
    """Return ``path``'s bytes, so a comparison is made on the artefact a consumer parses."""
    return path.read_bytes()


# --- The schema, exactly as the writer emits it ---


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
        # No absolute path may reach an artefact: it would name the clone the run
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


# --- Text UTF-8 cannot encode (QASEC-01) ---


def test_a_lone_surrogate_is_replaced_so_the_text_always_encodes() -> None:
    """A target's lone surrogate becomes U+FFFD: the canonical text always encodes as UTF-8.

    ``json.loads`` turns the JSON escape ``"\\ud800"`` into one code point, ``ensure_ascii=False``
    writes it through, and UTF-8 has no byte sequence for it -- so encoding the canonical text used
    to raise ``UnicodeEncodeError``. The formatter now replaces that one code point, which is the
    guarantee every writer below it relies on.
    """
    payload = {"error_text": f"bad {_LONE_SURROGATE} surrogate"}
    text = dumps_canonical(payload)

    # The bare call that used to raise: the text the canonical formatter returns is always UTF-8.
    assert text.encode("utf-8")
    assert _LONE_SURROGATE not in text
    # One code point replaces one code point, so a payload compared by length is not shortened,
    # and the replacement is the only difference from the same payload carrying U+FFFD already.
    assert text == dumps_canonical({"error_text": f"bad {_REPLACEMENT_CHARACTER} surrogate"})
    assert len(text) == len(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def test_a_payload_carrying_a_lone_surrogate_is_written_as_utf8(tmp_path: Path) -> None:
    """The artefact exists, decodes as UTF-8 and holds U+FFFD where the target's surrogate stood."""
    path = tmp_path / "decoded.json"
    payload = {"error_text": f"bad {_LONE_SURROGATE} surrogate", "payload_len": 3}

    assert write_canonical_json(payload, path) == path

    data = _reread(path)
    # Decoding is the assertion: a raw surrogate reaching the file fails here, and the behaviour
    # this pins replaced it was no file at all.
    text = data.decode("utf-8")
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert b"\xef\xbf\xbd" in data, "the file must carry U+FFFD's own three UTF-8 bytes"
    assert json.loads(text) == {
        "error_text": f"bad {_REPLACEMENT_CHARACTER} surrogate",
        "payload_len": 3,
    }


def test_a_summary_whose_decoded_value_carries_a_lone_surrogate_is_still_written(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
) -> None:
    """A target-influenced surrogate costs the run no report: ``summary.json`` is written."""
    case = make_case_result(
        status=Status.FAILED,
        decoded={"decoder_message": f"could not read {_LONE_SURROGATE}"},
    )
    run = make_run_result((case,))

    path = write_summary(run, _summary_path(report_dir))

    text = _reread(path).decode("utf-8")
    payload = json.loads(text)
    assert payload["results"][0]["decoded"]["decoder_message"] == (
        f"could not read {_REPLACEMENT_CHARACTER}"
    )
    # Every other byte of the document is still the canonical form of what was written.
    assert text == dumps_canonical(payload) + "\n"
    assert _LONE_SURROGATE not in text


# --- The volatile-field inventory ---


def test_masking_the_volatile_fields_makes_two_runs_byte_equal(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    report_dir: Path,
    mask_volatile: _MaskVolatile,
) -> None:
    """Two runs differing only in the volatile fields are byte-equal once masked."""
    base = _build_run(make_case_result, make_run_result)
    later = _build_run(
        make_case_result,
        make_run_result,
        started_at=_LATER_STARTED_AT,
        finished_at=_LATER_FINISHED_AT,
        durations=_LATER_DURATIONS,
        probe_metrics=_LATER_PROBE_METRICS,
    )
    base_path = write_summary(base, report_dir / "base.json")
    later_path = write_summary(later, report_dir / "later.json")
    # First prove the difference is real and confined to the volatile fields, so the equality below
    # is the masking at work and not two identical runs agreeing with themselves. An ordinary result
    # moves in its duration alone; a performance probe moves in its whole ``decoded`` mapping,
    # because that is where its measurement lives.
    assert _reread(base_path) != _reread(later_path)
    assert _differing_keys(_scalars_of(base.as_dict()), _scalars_of(later.as_dict())) == (
        _VOLATILE_RUN_FIELDS
    )
    probe_moves = _VOLATILE_RESULT_FIELDS | {"decoded"}
    for original, altered in zip(base.as_dict()["results"], later.as_dict()["results"]):
        if original["check_id"] in _PROBE_CHECKS:
            assert _differing_keys(original, altered) == probe_moves
        else:
            assert _differing_keys(original, altered) == _VOLATILE_RESULT_FIELDS
    # Inside the probes exactly the measured leaves moved, and the metrics their configuration fixes
    # are the same in both, so the equality below rests on nothing but the documented inventory.
    assert _probe_metric_differences(base.as_dict(), later.as_dict()) == _PROBE_MEASURED_FIELDS
    for probe in _PROBE_CHECKS:
        original = _probe_metrics(base.as_dict(), probe)
        altered = _probe_metrics(later.as_dict(), probe)
        assert set(original) == set(altered) == set(_BASE_PROBE_METRICS[probe])
        for name in sorted(set(original) - _PROBE_MEASURED_FIELDS[probe]):
            assert original[name] == altered[name]
    assert _masked_text(base_path, mask_volatile) == _masked_text(later_path, mask_volatile)


def test_mask_volatile_masks_exactly_the_documented_inventory(
    make_case_result: Callable[..., CaseResult],
    make_run_result: Callable[..., RunResult],
    mask_volatile: _MaskVolatile,
) -> None:
    """The shared masking helper touches the documented fields, and no more may join them."""
    run = _build_run(make_case_result, make_run_result)
    # A deep copy, so a helper that masked a nested mapping in place would move this snapshot too
    # and the non-mutation assertion at the end would pass on nothing.
    before = copy.deepcopy(run.as_dict())
    masked = mask_volatile(run)
    # Nothing is added or dropped, and the order of the results is preserved.
    assert set(masked) == set(before)
    assert _differing_keys(_scalars_of(before), _scalars_of(masked)) == _VOLATILE_RUN_FIELDS
    assert len(masked["results"]) == len(before["results"])
    probe_moves = _VOLATILE_RESULT_FIELDS | {"decoded"}
    for original, altered in zip(before["results"], masked["results"]):
        assert set(altered) == set(original)
        if original["check_id"] in _PROBE_CHECKS:
            assert _differing_keys(original, altered) == probe_moves
        else:
            assert _differing_keys(original, altered) == _VOLATILE_RESULT_FIELDS
    # Inside the probes' metrics blocks exactly the measured leaves moved and every key is still
    # there, so a mask widened to a configuration-derived metric fails here.
    assert _probe_metric_differences(before, masked) == _PROBE_MEASURED_FIELDS
    assert set().union(*_PROBE_MEASURED_FIELDS.values()) == _VOLATILE_METRIC_FIELDS
    for probe in _PROBE_CHECKS:
        original = _probe_metrics(before, probe)
        altered = _probe_metrics(masked, probe)
        assert set(original) == set(altered) == set(_BASE_PROBE_METRICS[probe])
        for name in sorted(set(original) - _PROBE_MEASURED_FIELDS[probe]):
            assert original[name] == altered[name]
    # The helper copies: the run it masked still reports what it reported, nested values included.
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

# The one check that calls no target at all: its contract declares a single run-level case with
# zero encodes and zero decodes, which is what makes a *real* run of it cheap enough for a unit
# test to drive the identity several times.
_CLEANUP_CHECK: Final[str] = "CL-ARTIFACTS"

# The cheapest second check in the same matrix -- ``IV-EMPTY`` sends one request and decodes
# nothing -- so varying the selected-check-set and selector-order inputs costs one encode each.
_EMPTY_INPUT_CHECK: Final[str] = "IV-EMPTY"

# A corpus class the filter narrows to, so a ``--class`` run digests fewer records than the
# whole-corpus run it is compared with (the runner's own corpus digest).
_FILTERED_CLASS: Final[str] = "url"

# The EC level and the adapter name a run is repeated with, one input moved at a time.
_OTHER_EC_LEVEL: Final[str] = "Q"
_OTHER_ADAPTER_NAME: Final[str] = "defective"


def _run(
    config: Config,
    selection: Selection,
    adapter: Adapter,
    *,
    ec_level: str = "M",
    seed: int | None = None,
) -> RunResult:
    """Run `selection` through :func:`qrcheck.runner.run_suite` and return the model it reported.

    ``seed`` defaults to the configured one, so a caller moves exactly the input it means to move
    and leaves the other four alone. This is the public path ``qrcheck run`` takes, not the
    derivation, which is what lets the assertions below pin the runner's own call site.

    Args:
        config: The resolved configuration, which owns the corpus and the report directory.
        selection: The ``--checks`` and ``--class`` selection to run.
        adapter: The target, reached only by the checks that need one; ``CL-ARTIFACTS`` needs none.
        ec_level: The error-correction level to run at.
        seed: The run's seed, or ``None`` for the configured one.

    Returns:
        The finished :class:`~qrcheck.report.model.RunResult`, whose ``run_id`` is the identity
        this module asserts.
    """
    return run_suite(
        config, selection, adapter, ec_level, config.seed if seed is None else seed
    )


def test_the_run_identifier_is_a_content_hash_stable_in_its_inputs(
    config_factory: Callable[..., Config],
    stub_adapter: Adapter,
    make_run_result: Callable[..., RunResult],
) -> None:
    """A real run's identity repeats for one input and moves with each of the five inputs the
    runner hashes, so what fails here is the runner's own call site, not only the derivation."""
    config = config_factory()
    selection = Selection(checks=_CLEANUP_CHECK)
    run_result = _run(config, selection, stub_adapter)
    identifier = run_result.run_id
    # The identity of a run that really happened: one check, one run-level case, and it passed.
    assert len(run_result.results) == 1, (
        f"{_CLEANUP_CHECK} reported {len(run_result.results)} results; an identity has to describe "
        "a run whose case is in it"
    )
    assert run_result.counts.passed == 1, (
        f"the run's single case did not pass: counts are {run_result.counts}"
    )
    # (1) Identical inputs -- the same adapter name, EC level, seed, check set and corpus -- give
    #     the same identity, which is what makes a run reproducible from ``summary.json`` alone.
    repeated = _run(config, selection, stub_adapter).run_id
    assert repeated == identifier, (
        "two runs of identical adapter, EC level, seed, check set and corpus reported different "
        f"identities: {repeated} and {identifier}"
    )
    # (2) The selected check IDs are hashed sorted, not in the order the selector spelled them,
    #     so the same two checks written the other way round are one identity.
    forward = _run(
        config, Selection(checks=f"{_CLEANUP_CHECK},{_EMPTY_INPUT_CHECK}"), stub_adapter
    ).run_id
    backward = _run(
        config, Selection(checks=f"{_EMPTY_INPUT_CHECK},{_CLEANUP_CHECK}"), stub_adapter
    ).run_id
    assert forward == backward, (
        "the selected check IDs were hashed in selector order: the pair and its reverse reported "
        f"{forward} and {backward}"
    )
    # (3) The adapter name. Assigned on the shared stub and put back at once, so no run below
    #     inherits the other name.
    own_name = stub_adapter.name
    stub_adapter.name = _OTHER_ADAPTER_NAME
    renamed = _run(config, selection, stub_adapter).run_id
    stub_adapter.name = own_name
    assert renamed != identifier, (
        f"the run identity ignored the adapter name: {own_name!r} and "
        f"{_OTHER_ADAPTER_NAME!r} reported the same identity"
    )
    # (4) The EC level.
    other_level = _run(config, selection, stub_adapter, ec_level=_OTHER_EC_LEVEL).run_id
    assert other_level != identifier, (
        f"the run identity ignored the EC level: 'M' and {_OTHER_EC_LEVEL!r} reported the same "
        "identity"
    )
    # (5) The seed: the configured one and the next integer are different runs.
    other_seed = _run(config, selection, stub_adapter, seed=config.seed + 1).run_id
    assert other_seed != identifier, (
        f"the run identity ignored the seed: {config.seed} and {config.seed + 1} reported the same "
        "identity"
    )
    # (6) The selected check set: one check and two checks are different runs, though the pair
    #     above proves the *order* they are named in is not part of the identity.
    assert forward != identifier, (
        f"the run identity ignored the selected check set: {_CLEANUP_CHECK} alone and it with "
        f"{_EMPTY_INPUT_CHECK} reported the same identity"
    )
    # (7) The corpus digest: a ``--class`` run digests only the records the filter kept.
    filtered = _run(
        config,
        Selection(checks=_CLEANUP_CHECK, class_globs=(_FILTERED_CLASS,)),
        stub_adapter,
    ).run_id
    assert filtered != identifier, (
        "the run identity ignored the corpus digest: the whole corpus and the "
        f"{_FILTERED_CLASS!r} filter reported the same identity"
    )
    # A SHA-256 hex digest, so it can never be the UUID or timestamp the model refuses -- and the
    # model accepts it, which is the other half of that contract.
    assert re.fullmatch(r"[0-9a-f]{64}", identifier) is not None, (
        f"the run identity is not a SHA-256 hex digest: {identifier!r}"
    )
    assert make_run_result((), run_id=identifier).run_id == identifier


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
