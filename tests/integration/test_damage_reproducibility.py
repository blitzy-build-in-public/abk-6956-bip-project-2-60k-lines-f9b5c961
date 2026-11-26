"""`DM-SEEDED`'s integration proof: the recorded seed regenerates the damaged image.

Held here over the whole 8-record by 7-kind matrix of AAP §0.8.1: ``results[].seed`` and
``decoded.damage[]`` in ``summary.json`` (§0.6.1 #4), the single derivation point §0.5.2
promises, and the byte-identical ``damage-<kind>-<seed>.png`` of §0.5.4 -- observed through
the damage module's public seam, never a second copy of a transform. A clean run writes no
evidence, so the artefact half is observed by a probe whose ``[checks.damage]`` values are
strictly harsher than the committed defaults, which can only add damage and so cannot mask a
defect. Marker ``e2e``: the matrix needs both checks, so ``-m "not e2e"`` deselects it, and
nothing here skips, retries or mocks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pytest

from qrcheck.checks import damage as damage_checks
from qrcheck.config.load import load_config
from qrcheck.config.models import DEFAULT_SEED, Config
from qrcheck.config.seeding import derive_seed
from qrcheck.evidence.naming import DAMAGE_KINDS, damage_artifact_name
from qrcheck.registry.selection import parse_selection
from qrcheck.report.model import CaseResult, RunResult, Status
from qrcheck.runner import run_suite
from qrcheck.targets import ADAPTER_TYPES
from qrcheck.targets.base import Adapter, Capability, GenerateRequest, GeneratedSymbol

# Marker-selected: the module needs DM-RESILIENCE and DM-SEEDED results, which only a run produces.
pytestmark = pytest.mark.e2e


# Fixed values: only shapes, names and seed parts; every threshold comes from the configuration.

#: The conforming built-in target every run in this module drives.
_REFERENCE: Final[str] = "reference"

#: The error-correction level the damage matrix runs at: the ``run`` command's own default.
_EC_LEVEL: Final[str] = "M"

#: The two check identifiers, read from the module that owns them rather than re-spelled.
_RESILIENCE: Final[str] = damage_checks.DM_RESILIENCE
_SEEDED: Final[str] = damage_checks.DM_SEEDED

#: The run seed the matrix runs at, and a second seed proving the pattern follows the seed.
_RUN_SEED: Final[int] = DEFAULT_SEED
_OTHER_RUN_SEED: Final[int] = DEFAULT_SEED + 1

#: The shape of the case matrix (AAP §0.8.1): eight sampled records, each damaged seven ways.
_EXPECTED_RECORDS: Final[int] = 8
_EXPECTED_CASES: Final[int] = _EXPECTED_RECORDS * len(DAMAGE_KINDS)

#: The kinds whose pattern is drawn from the derived seed (AAP §0.5.1): occluder placement and
#: additive noise; the other five are deterministic, so a different seed leaves them identical.
_STOCHASTIC_KINDS: Final[frozenset[str]] = frozenset({"occluders", "noise"})

#: The consensus vocabulary ``decoded["damage"][].agreement`` may hold.
_AGREEMENTS: Final[frozenset[str]] = frozenset(
    {"agree", "primary_only", "secondary_only", "undetected", "disagree"}
)

#: The configuration file name the generated documents are written under.
_CONFIG_NAME: Final[str] = "qrcheck.toml"

#: A guard that bounds a hang, not a product threshold: no check reads it.
_HARNESS_GUARD_S: Final[float] = 600.0

#: ``corpus:<record-id>@<kind>``: the case identifier, split into the record and the kind.
_CASE_ID_PATTERN = re.compile(r"^corpus:(?P<record>[^@]+)@(?P<kind>[^@]+)$")

#: A complete damage artefact name, parsed rather than assumed, so its seed can be compared.
_DAMAGE_NAME_PATTERN = re.compile(r"^damage-(?P<kind>[a-z]+)-(?P<seed>[0-9]+)\.png$")

#: The ``[checks.damage]`` overrides of the stricter-only probe: larger occlusion budget (margin 0),
#: heavier blur, smaller JPEG quality, stronger noise, larger rotation and perspective, smaller
#: smaller downscale; each is harsher, and the probe's first assertion checks that direction.
_HARSHER_DAMAGE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "occlusion_margin": 0.0,
        "blur_radius_per_module": 6.0,
        "jpeg_quality": 1,
        "noise_sigma": 60.0,
        "rotation_deg": 45.0,
        "perspective_deg": 30.0,
        "downscale_px_per_module": 1.0,
    }
)

#: Which ``[checks.damage]`` keys make the probe harsher by shrinking and which by growing.
_HARSHER_WHEN_SMALLER: Final[frozenset[str]] = frozenset(
    {"occlusion_margin", "jpeg_quality", "downscale_px_per_module"}
)

#: The key each kind records its applied value under in ``request.json``'s ``damage.applied``, and
#: the ``[checks.damage]`` attribute it comes from; ``blur`` is the one pair spelled differently.
_APPLIED_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "occluders": "occlusion_margin",
        "blur": "blur_radius_modules",
        "jpeg": "jpeg_quality",
        "noise": "noise_sigma",
        "rotation": "rotation_deg",
        "perspective": "perspective_deg",
        "downscale": "downscale_px_per_module",
    }
)
_LIMIT_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "occluders": "occlusion_margin",
        "blur": "blur_radius_per_module",
        "jpeg": "jpeg_quality",
        "noise": "noise_sigma",
        "rotation": "rotation_deg",
        "perspective": "perspective_deg",
        "downscale": "downscale_px_per_module",
    }
)



# The counting port wrapper: DM-SEEDED claims it never asks the target for a symbol, and the only
# only honest way to assert "zero" here is to count through the port itself, over a real adapter.


class _CountingAdapter:
    """An :class:`~qrcheck.targets.base.Adapter` that counts ``generate`` calls, forwarding the
    port."""

    __slots__ = ("_inner", "calls")

    def __init__(self, inner: Adapter) -> None:
        """Wrap ``inner`` and start the counter at zero."""
        self._inner = inner
        self.calls = 0

    @property
    def name(self) -> str:
        """Return the wrapped adapter's configured target name."""
        return self._inner.name

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Return the wrapped adapter's declared capability set, unchanged."""
        return self._inner.capabilities

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Count this call and delegate it to the wrapped adapter."""
        self.calls += 1
        return self._inner.generate(request)

    def close(self) -> None:
        """Release the wrapped adapter."""
        self._inner.close()


# One in-process run pair, captured as plain data: the memo is run-scoped state a run replaces,
# so reaching back into it would assert on whichever run went last.


@dataclass(frozen=True, slots=True)
class _DamageRun:
    """One ``DM-RESILIENCE`` run, its symbol memo, and the ``DM-SEEDED`` run over it."""

    config: Config
    seed: int
    cases: Mapping[str, tuple[str, str]]
    variants: Mapping[tuple[str, str], bytes]
    resilience: RunResult
    resilience_calls: int
    seeded: RunResult
    seeded_calls: int


def _reference_adapter(config: Config) -> Adapter:
    """Build the conforming target through the product's own factory table, as
    ``qrcheck/cli/run.py`` does."""
    target = config.targets[_REFERENCE]
    return ADAPTER_TYPES[str(target.type)](target, name=_REFERENCE)


def _probe_config(
    factory: pytest.TempPathFactory,
    write: Callable[..., Path],
    *,
    label: str,
    seed: int,
    damage: Mapping[str, Any] | None = None,
) -> Config:
    """Write one generated configuration under its own root, then load and return it."""
    document = factory.mktemp(f"cfg-{label}") / _CONFIG_NAME
    report_root = factory.mktemp(f"reports-{label}")
    write(
        document,
        default_adapter=_REFERENCE,
        report_dir=report_root,
        seed=seed,
        checks={"damage": dict(damage or {})},
    )
    return load_config(document)


def _run_one_check(config: Config, selector: str, seed: int) -> tuple[RunResult, int]:
    """Run one selected check over a counting wrapper; return ``(run_result, generate_calls)``."""
    adapter = _CountingAdapter(_reference_adapter(config))
    assert isinstance(adapter, Adapter), (
        "the counting wrapper is not an Adapter, so its call count says nothing about the "
        "product's adapter usage"
    )
    result = run_suite(config, parse_selection(selector, ()), adapter, _EC_LEVEL, seed)
    return result, adapter.calls


def _split_case(case_id: str) -> tuple[str, str]:
    """Return ``(record id, damage kind)`` from a damage case identifier."""
    match = _CASE_ID_PATTERN.match(case_id)
    if match is None:
        raise AssertionError(
            f"damage case identifier {case_id!r} is not 'corpus:<record-id>@<kind>'"
        )
    return match.group("record"), match.group("kind")


def _case_index(result: RunResult) -> dict[str, tuple[str, str]]:
    """Return every case identifier of `result` mapped to its ``(record id, kind)``."""
    return {case.case_id: _split_case(case.case_id) for case in result.results}


def _by_case(result: RunResult) -> dict[str, CaseResult]:
    """Return the run's results keyed by case identifier."""
    return {case.case_id: case for case in result.results}


def _by_case_id(summary: Mapping[str, Any], check_id: str) -> dict[str, dict[str, Any]]:
    """Return one check's results keyed by case identifier, failing when the check is absent."""
    results = {
        str(entry["case_id"]): entry
        for entry in summary["results"]
        if entry["check_id"] == check_id
    }
    if not results:
        raise AssertionError(
            f"summary.json holds no {check_id} result, so the run it describes never "
            f"executed that check"
        )
    return results


def _capture_variants(
    cases: Mapping[str, tuple[str, str]],
) -> dict[tuple[str, str], bytes]:
    """Copy the current run's damaged variants out of the module's public memo."""
    captured: dict[tuple[str, str], bytes] = {}
    for case_id in sorted(cases):
        record_id, kind = cases[case_id]
        variant = damage_checks.recall_variant(record_id, kind)
        if variant is None:
            raise AssertionError(
                f"case {case_id}: DM-RESILIENCE holds no {kind!r} variant in the module's "
                f"public memo, so the artefact criterion 4 measures cannot be regenerated"
            )
        captured[(record_id, kind)] = variant
    return captured


def _damage_pair(
    factory: pytest.TempPathFactory,
    write: Callable[..., Path],
    *,
    label: str,
    seed: int,
    damage: Mapping[str, Any] | None = None,
) -> _DamageRun:
    """Run ``DM-RESILIENCE`` then ``DM-SEEDED`` in this process and capture both."""
    config = _probe_config(factory, write, label=label, seed=seed, damage=damage)
    resilience, resilience_calls = _run_one_check(config, _RESILIENCE, config.seed)
    cases = _case_index(resilience)
    variants = _capture_variants(cases)
    seeded, seeded_calls = _run_one_check(config, _SEEDED, config.seed)
    return _DamageRun(
        config=config,
        seed=config.seed,
        cases=cases,
        variants=variants,
        resilience=resilience,
        resilience_calls=resilience_calls,
        seeded=seeded,
        seeded_calls=seeded_calls,
    )


@pytest.fixture(scope="module")
def damage_runs(
    tmp_path_factory: pytest.TempPathFactory,
    write_config: Callable[..., Path],
) -> _DamageRun:
    """Run the damage matrix once at the committed defaults, captured as plain data."""
    return _damage_pair(
        tmp_path_factory, write_config, label="damage-pair", seed=_RUN_SEED
    )


# Test 1 - the declared sequence every other assertion keys on


def test_damage_kinds_are_the_declared_seven_in_order() -> None:
    """Catch a reordered, renamed, duplicated or added damage kind (AAP §0.5.3)."""
    assert DAMAGE_KINDS == (
        "occluders",
        "blur",
        "jpeg",
        "noise",
        "rotation",
        "perspective",
        "downscale",
    )
    assert len(set(DAMAGE_KINDS)) == len(DAMAGE_KINDS), (
        f"the declared damage kinds repeat a token: {DAMAGE_KINDS}"
    )
    # The artefact name is built from a kind and a decimal seed and nothing else, so it parses back.
    assert damage_artifact_name(DAMAGE_KINDS[0], 7) == "damage-occluders-7.png"


# Test 2 - every damage case records the seed its damage was derived from


def test_every_damage_case_records_its_derived_seed(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a case whose recorded seed or ``decoded.damage`` is missing or wrong."""
    report_root, summary, exit_code = reference_report
    assert exit_code == 0, (
        f"the reference run exited {exit_code}: {report_root / 'summary.json'} describes a "
        f"run that did not come back clean"
    )
    resilience = _by_case_id(summary, _RESILIENCE)
    seeded = _by_case_id(summary, _SEEDED)
    assert len(resilience) == _EXPECTED_CASES, (
        f"DM-RESILIENCE reported {len(resilience)} cases, expected {_EXPECTED_CASES} "
        f"(§0.8.1's case matrix: {_EXPECTED_RECORDS} records by {len(DAMAGE_KINDS)} kinds)"
    )
    assert len(seeded) == _EXPECTED_CASES, (
        f"DM-SEEDED reported {len(seeded)} cases, expected {_EXPECTED_CASES}"
    )
    # The two checks run the same matrix, so a case missing from either is a matrix defect.
    assert sorted(resilience) == sorted(seeded), (
        "DM-RESILIENCE and DM-SEEDED do not report the same case identifiers"
    )

    run_seed = int(summary["seed"])
    kinds_by_record: dict[str, list[str]] = {}
    for case_id in sorted(resilience):
        record_id, kind = _split_case(case_id)
        result = resilience[case_id]
        derived = derive_seed(run_seed, _RESILIENCE, case_id, kind)
        assert result["seed"] == derived, (
            f"case {case_id}: results[].seed is {result['seed']!r}, but deriving "
            f"(seed={run_seed}, {_RESILIENCE!r}, {case_id!r}, {kind!r}) gives {derived}"
        )

        decoded = result["decoded"]
        assert isinstance(decoded, dict), (
            f"case {case_id}: decoded is a {type(decoded).__name__}, not the projection "
            f"criterion 4 measures"
        )
        assert "damage" in decoded, (
            f"case {case_id}: decoded carries no 'damage' key, which is the field "
            f"AAP §0.6.1 #4 measures; keys present: {sorted(decoded)}"
        )
        entries = decoded["damage"]
        assert isinstance(entries, list) and len(entries) == 1, (
            f"case {case_id}: decoded['damage'] holds {entries!r}; one case applies exactly "
            f"one kind, so it records exactly one entry"
        )
        entry = entries[0]
        assert entry["kind"] == kind, (
            f"case {case_id}: the recorded damage kind is {entry['kind']!r}, not {kind!r}"
        )
        assert entry["seed"] == derived, (
            f"case {case_id}: decoded['damage'][0]['seed'] is {entry['seed']!r}, not the "
            f"derived {derived}"
        )
        assert entry["agreement"] in _AGREEMENTS, (
            f"case {case_id}: agreement {entry['agreement']!r} is outside the consensus "
            f"vocabulary {sorted(_AGREEMENTS)}"
        )
        # The applied values make the damage auditable without consulting the module.
        assert isinstance(entry["applied"], dict) and entry["applied"], (
            f"case {case_id}: the applied threshold values are {entry['applied']!r}, so the "
            f"damage this case applied cannot be audited from its evidence"
        )
        # The occlusion budget comes from the EC level the symbol reports (IM-5), so record it.
        assert decoded["pristine"]["decoded_ec"] in {"L", "M", "Q", "H"}, (
            f"case {case_id}: the pristine reading reports EC level "
            f"{decoded['pristine']['decoded_ec']!r}, which no damage budget can be derived from"
        )
        kinds_by_record.setdefault(record_id, []).append(kind)

    assert len(kinds_by_record) == _EXPECTED_RECORDS, (
        f"the matrix covers {len(kinds_by_record)} records, expected {_EXPECTED_RECORDS}"
    )
    for record_id in sorted(kinds_by_record):
        observed = kinds_by_record[record_id]
        assert len(observed) == len(DAMAGE_KINDS) and frozenset(observed) == frozenset(
            DAMAGE_KINDS
        ), (
            f"record {record_id} was damaged by {sorted(observed)}, not once by every "
            f"declared kind {DAMAGE_KINDS}"
        )

    # A clean run persists nothing (AAP §0.3.4), which is why the stricter-only probe below exists.
    persisted = sorted(
        f"{check_id}:{case_id}"
        for check_id, results in ((_RESILIENCE, resilience), (_SEEDED, seeded))
        for case_id in sorted(results)
        if results[case_id]["evidence_dir"] is not None
    )
    assert not persisted, (
        f"the clean reference run persisted evidence for {persisted}, so a passing case "
        f"wrote files it should not have"
    )
    assert not (report_root / "evidence" / _RESILIENCE).exists(), (
        f"{report_root / 'evidence' / _RESILIENCE} exists after a run with no failures"
    )


# Test 3 - the seeded re-application reproduces the recorded variant


def test_seeded_reapplication_reproduces_the_resilience_variant(
    reference_report: tuple[Path, dict[str, Any], int],
) -> None:
    """Catch a ``DM-SEEDED`` case whose bytes differ from the recorded variant."""
    report_root, summary, _ = reference_report
    resilience = _by_case_id(summary, _RESILIENCE)
    seeded = _by_case_id(summary, _SEEDED)
    assert sorted(resilience) == sorted(seeded), (
        "DM-RESILIENCE and DM-SEEDED do not report the same case identifiers"
    )

    run_seed = int(summary["seed"])
    for case_id in sorted(resilience):
        record_id, kind = _split_case(case_id)
        derived = derive_seed(run_seed, _RESILIENCE, case_id, kind)
        result = seeded[case_id]
        assert result["status"] == "passed", (
            f"case {case_id}: DM-SEEDED is {result['status']!r} "
            f"(reason {result['reason']!r}) against the conforming target, so the seeded "
            f"re-application did not reproduce the recorded variant; evidence: "
            f"{result['evidence_dir']!r}"
        )
        assert result["seed"] == derived, (
            f"case {case_id}: DM-SEEDED recorded seed {result['seed']!r}, not {derived}; the "
            f"re-applied damage would be a different image"
        )
        decoded = result["decoded"]
        assert "variant" in decoded, (
            f"case {case_id}: DM-SEEDED's decoded carries no 'variant' block, which holds "
            f"the byte comparison criterion 4 measures; keys present: {sorted(decoded)}"
        )
        variant = decoded["variant"]
        assert variant["regenerated_sha256"] == variant["recorded_sha256"], (
            f"case {case_id} ({kind}): regenerating from the recorded seed gave "
            f"{variant['regenerated_sha256']} but DM-RESILIENCE recorded "
            f"{variant['recorded_sha256']}; first differing byte at "
            f"{variant['first_difference']!r}"
        )
        assert variant["regenerated_len"] == variant["recorded_len"], (
            f"case {case_id} ({kind}): the regenerated variant is "
            f"{variant['regenerated_len']} bytes, the recorded one "
            f"{variant['recorded_len']}"
        )
        assert variant["first_difference"] is None, (
            f"case {case_id} ({kind}): the first differing byte is at "
            f"{variant['first_difference']!r}, so the two variants are not byte-identical"
        )

        # The damaged image is the same image for both checks, so its sampling is too: same digest.
        assert decoded["matrix_sha256"] == resilience[case_id]["decoded"]["matrix_sha256"], (
            f"case {case_id} ({kind}): the two checks sampled different matrices from the "
            f"same variant"
        )
        assert (
            decoded["damage"][0]["decoded_len"]
            == resilience[case_id]["decoded"]["damage"][0]["decoded_len"]
        ), (
            f"case {case_id} ({kind}): the decode outcome differs between the two checks on "
            f"the same variant, so the regenerated image is not the one that was decoded"
        )



# Test 4 - the variant is a pure function of the seed, and the seed reaches the pattern


def test_damage_application_is_byte_identical_from_the_seed_alone(
    damage_runs: _DamageRun,
    tmp_path_factory: pytest.TempPathFactory,
    write_config: Callable[..., Path],
) -> None:
    """Catch a variant that changes between runs, or a seed that never reaches the pattern."""
    assert len(damage_runs.variants) == _EXPECTED_CASES, (
        f"the captured matrix holds {len(damage_runs.variants)} variants, expected "
        f"{_EXPECTED_CASES}"
    )

    # -- the same run seed, a second independent run -------------------------------
    repeat_config = _probe_config(
        tmp_path_factory, write_config, label="damage-repeat", seed=damage_runs.seed
    )
    repeat, _ = _run_one_check(repeat_config, _RESILIENCE, repeat_config.seed)
    assert _case_index(repeat) == dict(damage_runs.cases), (
        "the second run at the same seed reported a different case matrix, so its variants "
        "are not comparable pair by pair"
    )
    # Read the memo immediately: it is run-scoped, and the next run replaces it.
    repeated = _capture_variants(damage_runs.cases)
    for record_id, kind in sorted(damage_runs.variants):
        assert repeated[(record_id, kind)] == damage_runs.variants[(record_id, kind)], (
            f"record {record_id}, kind {kind!r}: two runs at run seed "
            f"{damage_runs.seed} produced different damaged bytes, so the image is not a "
            f"function of the seed alone"
        )

    # -- a different run seed, for the kinds whose pattern it governs ---------------
    other_config = _probe_config(
        tmp_path_factory, write_config, label="damage-other-seed", seed=_OTHER_RUN_SEED
    )
    other, _ = _run_one_check(other_config, _RESILIENCE, other_config.seed)
    other_cases = _case_index(other)
    assert sorted(other_cases) == sorted(damage_runs.cases), (
        "the run at a different seed reported a different case matrix"
    )
    other_variants = _capture_variants(other_cases)
    records = sorted({record_id for record_id, _ in damage_runs.variants})
    for kind in DAMAGE_KINDS:
        changed = [
            record_id
            for record_id in records
            if other_variants[(record_id, kind)] != damage_runs.variants[(record_id, kind)]
        ]
        if kind == "noise":
            # Additive noise redraws every pixel, so every record's variant must change.
            assert len(changed) == len(records), (
                f"kind 'noise' produced {len(records) - len(changed)} of {len(records)} "
                f"variants unchanged by a different run seed, so the noise pattern is not "
                f"drawn from the derived seed"
            )
        elif kind in _STOCHASTIC_KINDS:
            # Occluder placement is seeded, but a symbol with few codewords has few places to start,
            # so one record's two seeds may coincide; at least one must move.
            assert changed, (
                f"kind {kind!r} produced identical variants for all {len(records)} records "
                f"under two different run seeds, so the placement is not seeded"
            )
        else:
            assert not changed, (
                f"kind {kind!r} changed for {changed} under a different run seed, but it is "
                f"a deterministic image operation with no seeded draw (AAP §0.5.1)"
            )


# Test 5 - DM-SEEDED asks the target for nothing


def test_dm_seeded_issues_zero_adapter_calls(damage_runs: _DamageRun) -> None:
    """Catch a ``DM-SEEDED`` that re-encodes instead of re-applying the damage (AAP §0.8.1)."""
    declared = damage_checks.EXPECTED_ENCODES[_SEEDED]
    assert declared == 0, (
        f"the damage family declares {declared} encodes for DM-SEEDED; this test asserts "
        f"zero, so the two disagree about the contract"
    )
    assert damage_runs.seeded_calls == 0, (
        f"DM-SEEDED called the target {damage_runs.seeded_calls} time(s); it must re-apply "
        f"the recorded damage to the symbol it already holds and never encode again"
    )
    results = damage_runs.seeded.results
    assert len(results) == _EXPECTED_CASES, (
        f"DM-SEEDED reported {len(results)} cases, expected {_EXPECTED_CASES}"
    )
    not_passed = [
        (result.case_id, result.status.value, result.reason)
        for result in results
        if result.status is not Status.PASSED
    ]
    assert not not_passed, (
        f"DM-SEEDED reported {len(not_passed)} non-passing case(s) against the conforming "
        f"target, so the seeded re-application did not reproduce the variant: {not_passed}"
    )


# Test 6 - DM-RESILIENCE pays for one symbol per record, not one per case


def test_dm_resilience_encodes_one_symbol_per_record(damage_runs: _DamageRun) -> None:
    """Catch a damage check that encodes once per kind rather than once per record."""
    declared = damage_checks.EXPECTED_ENCODES[_RESILIENCE]
    assert declared == _EXPECTED_RECORDS, (
        f"the damage family declares {declared} encodes for DM-RESILIENCE; the matrix "
        f"samples {_EXPECTED_RECORDS} records, so the two disagree"
    )
    assert damage_runs.resilience_calls == _EXPECTED_RECORDS, (
        f"DM-RESILIENCE called the target {damage_runs.resilience_calls} time(s) for "
        f"{len(damage_runs.cases)} cases; one symbol per sampled record is "
        f"{_EXPECTED_RECORDS}"
    )
    results = damage_runs.resilience.results
    assert len(results) == _EXPECTED_CASES, (
        f"DM-RESILIENCE reported {len(results)} cases, expected {_EXPECTED_CASES}"
    )
    not_passed = [
        (result.case_id, result.status.value, result.reason)
        for result in results
        if result.status is not Status.PASSED
    ]
    assert not not_passed, (
        f"DM-RESILIENCE reported {len(not_passed)} non-passing case(s) against the "
        f"conforming target at the committed defaults: {not_passed}"
    )
    # Every record's seven cases, and no other case: the matrix is the declared product.
    expected_ids = {
        f"corpus:{record_id}@{kind}"
        for record_id in sorted({record for record, _ in damage_runs.cases.values()})
        for kind in DAMAGE_KINDS
    }
    assert {result.case_id for result in results} == expected_ids, (
        "the reported case identifiers are not the sampled records crossed with the seven "
        "declared kinds"
    )



# Test 7 - the artefacts on disk carry the derived seed and the bytes. A clean run writes no
# evidence, so this test makes its own with a bounded `run --checks DM-RESILIENCE` whose
# [checks.damage] values are strictly harsher: it exists to produce files, not to judge a verdict.


def _read_summary(report_root: Path) -> dict[str, Any]:
    """Read one run's ``summary.json``, failing with its path when the run wrote none."""
    path = report_root / "summary.json"
    assert path.is_file(), (
        f"the probe run wrote no {path}, so the artefacts it was run for do not exist"
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path} does not hold a JSON object"
    return loaded


def test_on_disk_damage_artefacts_match_their_names_and_re_derivation(
    damage_runs: _DamageRun,
    tmp_path_factory: pytest.TempPathFactory,
    write_config: Callable[..., Path],
    run_cli: Callable[..., Any],
) -> None:
    """Catch an artefact named with the wrong seed, holding other bytes, or missing."""
    # -- the probe only ever adds damage ------------------------------------------
    for key in sorted(_HARSHER_DAMAGE):
        probe_value = _HARSHER_DAMAGE[key]
        committed = getattr(damage_runs.config.damage, key)
        if key in _HARSHER_WHEN_SMALLER:
            assert probe_value <= committed, (
                f"probe value {key}={probe_value!r} is milder than the committed "
                f"{committed!r}; the probe must only ever increase the damage"
            )
        else:
            assert probe_value >= committed, (
                f"probe value {key}={probe_value!r} is milder than the committed "
                f"{committed!r}; the probe must only ever increase the damage"
            )

    probe_config = _probe_config(
        tmp_path_factory,
        write_config,
        label="damage-probe",
        seed=_RUN_SEED,
        damage=_HARSHER_DAMAGE,
    )
    probe = run_cli(
        "run",
        "--adapter",
        _REFERENCE,
        "--ec",
        _EC_LEVEL,
        "--checks",
        _RESILIENCE,
        config=probe_config.path,
        in_process=True,
    )
    assert probe.exit_code not in (2, 3), (
        f"the artefact probe exited {probe.exit_code}: exit 2 is a configuration error and "
        f"exit 3 an internal one, neither of which produces the evidence it was run for; "
        f"stderr: {probe.stderr[-2000:]!r}"
    )
    summary = _read_summary(probe_config.report_dir)
    results = _by_case_id(summary, _RESILIENCE)
    assert len(results) == _EXPECTED_CASES, (
        f"the probe reported {len(results)} DM-RESILIENCE cases, expected {_EXPECTED_CASES}"
    )
    failing = [case_id for case_id in sorted(results) if results[case_id]["status"] != "passed"]
    assert failing, (
        f"the stricter-only probe produced no failing case among {len(results)}, so it wrote "
        f"no damage artefact and this test would assert nothing"
    )
    assert probe.exit_code == 1, (
        f"the probe reports {len(failing)} failing case(s) but exited {probe.exit_code}, "
        f"which is not the code a failing run carries"
    )

    # -- every artefact is named by its seed and holds the bytes the check produced --
    on_disk_sha256: dict[str, str] = {}
    observed_kinds: set[str] = set()
    for case_id in failing:
        record_id, kind = _split_case(case_id)
        result = results[case_id]
        derived = derive_seed(int(summary["seed"]), _RESILIENCE, case_id, kind)
        assert result["seed"] == derived, (
            f"case {case_id}: the failing result recorded seed {result['seed']!r}, not the "
            f"derived {derived}"
        )
        evidence_dir = result["evidence_dir"]
        assert evidence_dir is not None, (
            f"case {case_id}: a failing case recorded no evidence directory, so its damaged "
            f"image was never written"
        )
        case_dir = probe_config.report_dir / evidence_dir
        assert case_dir.is_dir(), f"case {case_id}: {case_dir} is not a directory"
        assert case_dir.parent == probe_config.report_dir / "evidence" / _RESILIENCE, (
            f"case {case_id}: the evidence sits under {case_dir.parent}, not under the "
            f"<report_dir>/evidence/{_RESILIENCE}/ layout"
        )

        artefacts = sorted(
            path for path in case_dir.iterdir() if path.name.startswith("damage-")
        )
        assert [path.name for path in artefacts] == [damage_artifact_name(kind, derived)], (
            f"case {case_id}: the damage artefacts are "
            f"{[path.name for path in artefacts]}, expected exactly "
            f"{damage_artifact_name(kind, derived)!r}"
        )
        artefact = artefacts[0]
        parsed = _DAMAGE_NAME_PATTERN.fullmatch(artefact.name)
        assert parsed is not None, (
            f"case {case_id}: {artefact.name!r} is not of the declared form "
            f"damage-<kind>-<seed>.png"
        )
        assert parsed.group("kind") == kind, (
            f"case {case_id}: {artefact.name!r} names kind {parsed.group('kind')!r}, not "
            f"{kind!r}"
        )
        assert int(parsed.group("seed")) == derived, (
            f"case {case_id}: {artefact.name!r} carries seed {parsed.group('seed')}, not the "
            f"derived {derived}"
        )

        # The bytes on disk are the bytes the check produced, read back through the public memo.
        produced = damage_checks.recall_variant(record_id, kind)
        assert produced is not None, (
            f"case {case_id}: the module holds no {kind!r} variant for record {record_id}, "
            f"so the artefact cannot be re-derived from the seed"
        )
        on_disk = artefact.read_bytes()
        assert on_disk == produced, (
            f"case {case_id}: {artefact.name} is not the image the transform produced for "
            f"this symbol and seed"
        )
        on_disk_sha256[case_id] = hashlib.sha256(on_disk).hexdigest()

        # decoded.json carries the same seed, so the evidence pair agrees on the damage.
        decoded = json.loads((case_dir / "decoded.json").read_text(encoding="utf-8"))
        assert decoded["damage"][0]["kind"] == kind
        assert decoded["damage"][0]["seed"] == derived

        # request.json records the thresholds applied, so a later edit cannot rewrite this run.
        request = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
        assert request["damage"]["kind"] == kind
        assert request["damage"]["seed"] == derived
        applied = request["damage"]["applied"]
        assert applied[_APPLIED_KEYS[kind]] == getattr(
            probe_config.damage, _LIMIT_KEYS[kind]
        ), (
            f"case {case_id}: request.json records {_APPLIED_KEYS[kind]}="
            f"{applied[_APPLIED_KEYS[kind]]!r}, but the resolved configuration carries "
            f"{getattr(probe_config.damage, _LIMIT_KEYS[kind])!r}"
        )
        observed_kinds.add(kind)

    # Not all seven kinds are expected to fail: the occlusion budget is bounded by the symbol's own
    # capacity, and more than one kind must be present or the probe proved nothing.
    assert observed_kinds and observed_kinds <= set(DAMAGE_KINDS), (
        f"the probe wrote artefacts for kinds {sorted(observed_kinds)}, which are not all "
        f"declared kinds {DAMAGE_KINDS}"
    )
    assert len(observed_kinds) > 1, (
        f"the probe produced artefacts for only {sorted(observed_kinds)}, so the artefact "
        f"assertions above exercised a single transform"
    )

    # -- DM-SEEDED's own re-application reproduces the file on disk -------------------
    # A second run over the same memo re-applies each variant; its digest must equal the artefact.
    seeded_config = _probe_config(
        tmp_path_factory,
        write_config,
        label="damage-probe-seeded",
        seed=_RUN_SEED,
        damage=_HARSHER_DAMAGE,
    )
    seeded = run_cli(
        "run",
        "--adapter",
        _REFERENCE,
        "--ec",
        _EC_LEVEL,
        "--checks",
        _SEEDED,
        config=seeded_config.path,
        in_process=True,
    )
    # 0 or 1: whether a harsh variant decodes is the transform's business; 2 or 3 is no comparison.
    assert seeded.exit_code in (0, 1), (
        f"the DM-SEEDED probe exited {seeded.exit_code}, which produces no comparable "
        f"evidence; stderr: {seeded.stderr[-2000:]!r}"
    )
    seeded_results = _by_case_id(_read_summary(seeded_config.report_dir), _SEEDED)
    assert len(seeded_results) == _EXPECTED_CASES, (
        f"the DM-SEEDED probe reported {len(seeded_results)} cases, expected "
        f"{_EXPECTED_CASES}"
    )
    # Every case that wrote an artefact must have been re-applied, or it has no re-derivation;
    # DM-SEEDED reports the whole matrix, so only the failing subset has to be present.
    missing = sorted(set(failing) - set(seeded_results))
    assert not missing, (
        f"the DM-SEEDED probe reported no result for {missing}, so those artefacts cannot be "
        f"matched to a re-application"
    )
    for case_id in failing:
        seeded_decoded = seeded_results[case_id]["decoded"]
        assert "variant" in seeded_decoded, (
            f"case {case_id}: the DM-SEEDED probe recorded no 'variant' block, so it never "
            f"re-applied the damage and the artefact on disk has no re-derivation; keys "
            f"present: {sorted(seeded_decoded)}"
        )
        variant = seeded_decoded["variant"]
        assert variant["regenerated_sha256"] == on_disk_sha256[case_id], (
            f"case {case_id}: re-applying the damage from the recorded seed gave "
            f"{variant['regenerated_sha256']}, but the artefact DM-RESILIENCE wrote is "
            f"{on_disk_sha256[case_id]}"
        )
        assert variant["recorded_sha256"] == on_disk_sha256[case_id], (
            f"case {case_id}: the recorded variant digest "
            f"{variant['recorded_sha256']} is not the digest of the file on disk "
            f"{on_disk_sha256[case_id]}"
        )


# Test 8 - the derivation is stable in this process and in a fresh one


#: A one-liner run in a fresh interpreter: it derives one seed from its arguments and prints it.
_SEED_SCRIPT: Final[str] = (
    "import sys\n"
    "from qrcheck.config.seeding import derive_seed\n"
    "print(derive_seed(int(sys.argv[1]), *sys.argv[2:]))\n"
)


def test_rerunning_the_same_seed_rederives_the_same_variant(
    damage_runs: _DamageRun, repo_root: Path
) -> None:
    """Catch a derivation that depends on the process, the hash seed or the call order."""
    recorded = _by_case(damage_runs.resilience)
    assert len(recorded) == _EXPECTED_CASES, (
        f"the run reported {len(recorded)} cases, expected {_EXPECTED_CASES}"
    )
    for case_id in sorted(recorded):
        record_id, kind = _split_case(case_id)
        first = derive_seed(damage_runs.seed, _RESILIENCE, case_id, kind)
        assert first == derive_seed(damage_runs.seed, _RESILIENCE, case_id, kind), (
            f"case {case_id}: deriving the same parts twice gave two different seeds"
        )
        assert recorded[case_id].seed == first, (
            f"case {case_id} (record {record_id}, kind {kind!r}): the result recorded seed "
            f"{recorded[case_id].seed!r}, but the four recorded parts derive {first}"
        )

    # -- the same derivation in a fresh interpreter -------------------------------
    case_id = sorted(recorded)[0]
    _record_id, kind = _split_case(case_id)
    expected = derive_seed(damage_runs.seed, _RESILIENCE, case_id, kind)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _SEED_SCRIPT,
            str(damage_runs.seed),
            _RESILIENCE,
            case_id,
            kind,
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=_HARNESS_GUARD_S,
        check=False,
        # A different hash seed, so a derivation that leaked hash() or set order would disagree.
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )
    assert completed.returncode == 0, (
        f"deriving a seed in a fresh interpreter failed with {completed.returncode}: "
        f"{completed.stderr[-2000:]!r}"
    )
    assert int(completed.stdout.strip()) == expected, (
        f"case {case_id}: a fresh interpreter derived {completed.stdout.strip()}, this "
        f"process derived {expected}, so a damaged image is not reproducible from the "
        f"recorded parts alone"
    )

