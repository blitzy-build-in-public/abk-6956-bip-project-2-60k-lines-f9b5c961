"""The two decoder-absence rows of the state-handling table, held as gates.

AAP §0.6.6 names this module for two rows, and AAP §0.7.2 fixes the distinction between them:

* **row 3 -- the primary decoder is missing.** ``zxing-cpp`` is the authority for every round-trip
  assertion, so its absence is *blocking*: ``qrcheck doctor`` and ``qrcheck run`` both exit 2 naming
  the dependency, and no check is silently skipped in its place. The observable proof of "never
  silently skipped" is that the run writes no report at all: exit code 2 is decided before a single
  case runs (AAP §0.3.4), so there is no result list in which ``RT-ROUNDTRIP`` could appear as a
  skip.
* **row 4 -- the secondary decoder is missing.** ``opencv-python-headless`` corroborates and never
  decides, so its absence is a *supported degradation*: ``qrcheck doctor`` stays green and reports
  the entry as degraded, the run proceeds in single-decoder mode, every decoded result records
  ``agreement == "primary_only"`` with one entry in ``per_decoder``, the evidence a failing case
  writes shows the same single decoder, and ``index.html`` states the degradation. Nothing is
  skipped on its behalf and nothing passes on the strength of the decoder that is not installed.

Absence is forced honestly: nothing is uninstalled and nothing inside the product is patched. A
``None`` entry in ``sys.modules`` makes ``import zxingcpp`` raise :exc:`ImportError` and
``importlib.util.find_spec`` -- the probe :mod:`qrcheck.decode.base` uses -- answer "not present",
and the decoder wrappers resolve their binding lazily on every call, so no cached wrapper can
short-circuit the probe; ``monkeypatch`` restores both entries at teardown, which
:func:`test_module_state_is_restored_after_a_stubbed_call` asserts.

For that reason every stubbed invocation runs **in this interpreter**, through
``run_cli(..., in_process=True)``: the fixture's default path starts a real child process, and a
child does not inherit a ``sys.modules`` entry, so the stub would be invisible there and the
assertions below would prove nothing. The in-process path calls the same
:func:`qrcheck.cli.main.main` the console script calls, so the exit status under test is still the
product's own.

No test here skips for an absent decoder: every supported platform has a wheel (AAP §0.2.3), so a
genuinely missing one means the environment is broken and must surface as a loud failure naming the
distribution and ``requirements.lock``, never as a skip. There is no ``xfail``, no retry and no
``pytest-mock`` (AAP §0.4.3, §0.5.6); no threshold is restated (§0.12); every run is bounded by a
``--checks``/``--class`` selection over the twelve ``otp`` records, so the module finishes in
seconds and carries no ``e2e`` marker; and every artefact lands under the test's ``tmp_path`` with a
fresh report root per test, so a leaked file cannot satisfy another test's absence assertion.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.cli.exit import EXIT_CONFIG, EXIT_FAILED, EXIT_OK
from qrcheck.decode import (
    PRIMARY_DECODER,
    SECONDARY_DECODER,
    decoder_availability,
)
from qrcheck.report import HTML_FILENAME, SUMMARY_FILENAME

# ---------------------------------------------------------------------------
# The names the mechanism turns on
#
# The importable module names key a ``sys.modules`` entry; the distribution names are how
# AAP §0.7.2 writes the dependencies a reader has to install, and they are what the diagnostics
# below assert. Different vocabularies for the same two packages, so both are named here.
# ---------------------------------------------------------------------------

#: The primary decoder's importable module, which a blocked ``sys.modules`` entry hides.
PRIMARY_MODULE: Final[str] = "zxingcpp"

#: The secondary decoder's importable module, blocked the same way.
SECONDARY_MODULE: Final[str] = "cv2"

#: The primary decoder's distribution name, spelled with the hyphen as AAP §0.7.2 spells it.
PRIMARY_DISTRIBUTION: Final[str] = "zxing-cpp"

#: The secondary decoder's distribution name.
SECONDARY_DISTRIBUTION: Final[str] = "opencv-python-headless"

#: The decoder wrapper modules, dropped from ``sys.modules`` beside the blocker so a wrapper that
#: one day caches its binding cannot keep answering for a binding that is gone.
DECODER_WRAPPERS: Final[tuple[str, str]] = (
    "qrcheck.decode.zxing",
    "qrcheck.decode.opencv",
)

#: The dependency statuses ``qrcheck doctor --json`` publishes (AAP §0.8.3): ``missing`` blocks and
#: ``degraded`` does not, which is the whole difference between rows 3 and 4.
STATUS_OK: Final[str] = "ok"
STATUS_MISSING: Final[str] = "missing"
STATUS_DEGRADED: Final[str] = "degraded"

#: The six keys the doctor report object carries, exactly (AAP §0.8.3), so a field that is added,
#: dropped or renamed fails here instead of shrinking a machine-readable surface.
DOCTOR_REPORT_KEYS: Final[frozenset[str]] = frozenset(
    {"adapter", "type", "reachable", "capabilities", "dependencies", "svg_rasterizer"}
)

#: The four keys one doctor dependency entry carries.
DOCTOR_DEPENDENCY_KEYS: Final[frozenset[str]] = frozenset(
    {"name", "version", "status", "blocking"}
)

#: The ``banner`` class and the section label ``qrcheck/report/html.py`` uses for a degradation.
#: The exact sentence is deliberately not pinned: the wording belongs to the writer, so a copy edit
#: must not break this gate.
DEGRADATION_CLASS: Final[str] = "banner-single-decoder"
DEGRADATION_LABEL: Final[str] = "degrad"

# ---------------------------------------------------------------------------
# The selections every run in this module is bounded by
#
# One corpus class and one check per run keeps the module in the fast job while still
# exercising the real encode, the real decoders and the real report writers. ``otp`` holds
# twelve ASCII records at AAP §0.8.1's case matrix, and both decoders read them, which is
# what makes the single-decoder and two-decoder runs comparable.
# ---------------------------------------------------------------------------

#: The configuration file's name, beside the report root under ``tmp_path``.
CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"

#: The report root's directory name, under the same temporary directory.
REPORT_DIR_NAME: Final[str] = "reports"

#: The corpus class every run is filtered to.
CORPUS_CLASS: Final[str] = "otp"

#: The check the degradation runs select: its result carries the whole ``DecodeResult``
#: projection, so ``agreement`` and ``per_decoder`` are readable from ``summary.json``.
ROUNDTRIP_CHECK: Final[str] = "RT-ROUNDTRIP"

#: The check whose failure against the defective target writes evidence: ``wrong_ec_level``
#: encodes one level below the request at EC M (AAP §0.4.2), so every case fails and every
#: case carries a ``decoded.json``.
EC_CHECK: Final[str] = "SY-EC-MATCH"

#: A check that needs no target at all, used to prove the primary-decoder refusal is not
#: specific to the one check whose name mentions a round trip.
TARGET_INDEPENDENT_CHECK: Final[str] = "CL-ARTIFACTS"

#: The two built-in targets by the names the configuration declares.
REFERENCE_TARGET: Final[str] = "reference"
DEFECTIVE_TARGET: Final[str] = "defective"

#: The error-correction level every run here requests, which is ``run``'s own default.
EC_LEVEL: Final[str] = "M"

#: The evidence projections that state single-decoder mode. ``agreement`` is the consensus
#: layer's classification and ``per_decoder`` is keyed by the decoder names, so asserting
#: both is what makes the degradation observable from the report rather than inferred.
SINGLE_DECODER_AGREEMENT: Final[str] = "primary_only"
TWO_DECODER_AGREEMENT: Final[str] = "agree"

#: How much of a captured stream a failure message quotes, so the diagnostic stays readable
#: and still names what the command said.
DIAGNOSTIC_CHARS: Final[int] = 800

#: Distinguishes "absent from ``sys.modules``" from "present and blocked", which is the
#: difference between a module this session never imported and a stub that outlived its test.
PRESENT: Final[object] = object()


@dataclass(frozen=True, slots=True)
class Workspace:
    """One test's configuration document and the report root it points at.

    Both live under ``tmp_path`` and the root is fresh per test, so an absence assertion cannot be
    satisfied by a file another test left behind.
    """

    config: Path
    report_dir: Path


def module_absent(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    """Make ``module`` unimportable for the duration of one test.

    A ``None`` entry in :data:`sys.modules` is Python's own "blocked" marker: ``import module``
    raises :exc:`ImportError` and ``importlib.util.find_spec(module)`` answers ``None``, which is
    precisely the pair a genuinely uninstalled dependency gives. ``module`` is an importable module
    name, never a distribution name such as ``zxing-cpp``.

    The decoder wrapper modules are dropped from ``sys.modules`` as well, so a wrapper that one day
    caches its binding cannot keep answering for a binding that is gone; neither wrapper imports its
    binding at import time today, so the entry for the binding itself is the switch these tests
    observe.
    """
    monkeypatch.setitem(sys.modules, module, None)
    for wrapper in DECODER_WRAPPERS:
        monkeypatch.delitem(sys.modules, wrapper, raising=False)


def workspace(write_config: Callable[..., Path], directory: Path, *, target: str) -> Workspace:
    """Write a complete configuration document for ``target`` under ``directory``.

    The document carries every declared default the fixture emits and points ``report_dir`` at a
    fresh root, so a run started from it sees the committed thresholds for everything this module
    does not choose and writes nothing outside ``directory``. ``run`` still passes ``--adapter``
    explicitly, so the document's ``default_adapter`` and the run can never disagree silently.
    """
    report_dir = directory / REPORT_DIR_NAME
    config = write_config(
        directory / CONFIG_FILE_NAME,
        default_adapter=target,
        report_dir=report_dir,
    )
    return Workspace(config=config, report_dir=report_dir)


def invoke(run_cli: Callable[..., Any], space: Workspace, *arguments: str) -> Any:
    """Run one ``qrcheck`` command in this interpreter and return its outcome.

    ``in_process=True`` is what makes a ``sys.modules`` stub observable: the fixture's default path
    starts a real child process, which would see the real, installed decoders.
    """
    return run_cli(*arguments, config=space.config, in_process=True)


def diagnostic(result: Any) -> str:
    """Return the outcome of one invocation, bounded, for a failure message.

    The exit status and the tail of both streams, so a failing assertion names what the command
    actually did instead of only what it did not do.
    """
    return (
        f"exit code {result.exit_code}, "
        f"stdout tail {result.stdout[-DIAGNOSTIC_CHARS:]!r}, "
        f"stderr tail {result.stderr[-DIAGNOSTIC_CHARS:]!r}"
    )


def load_summary(space: Workspace, result: Any) -> dict[str, Any]:
    """Read and parse the run's ``summary.json``.

    A run that wrote no report, or wrote one that is not a JSON object, fails the test: deliberately
    an assertion and not a skip, because the report is the gate.
    """
    path = space.report_dir / SUMMARY_FILENAME
    assert path.is_file(), (
        f"the run wrote no {SUMMARY_FILENAME} under {space.report_dir}; "
        f"{diagnostic(result)}"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), (
        f"{path} does not hold a JSON object: {document!r}"
    )
    return document


def results_of(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the summary's ``results[]`` array, with the shape it is read as asserted.

    An array that is absent, empty or holds something other than objects fails the test: a run
    selected a check and a corpus class, so zero results means the run did not measure what the
    selection named.
    """
    entries = summary.get("results")
    assert isinstance(entries, list) and entries, (
        f"the summary records no results: results={entries!r}, counts={summary.get('counts')!r}"
    )
    for entry in entries:
        assert isinstance(entry, dict), f"a result entry is not a JSON object: {entry!r}"
    return entries


def dependency(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return the single doctor dependency entry named ``name``, with its four documented keys.

    ``name`` is a distribution name, ``zxing-cpp`` or ``opencv-python-headless``; no entry carrying
    it, or more than one, fails the test.
    """
    entries = document.get("dependencies")
    assert isinstance(entries, list), f"the doctor report carries no dependencies: {document!r}"
    matches = [entry for entry in entries if entry.get("name") == name]
    assert len(matches) == 1, (
        f"the doctor report holds {len(matches)} entries named {name!r}, expected exactly "
        f"one; reported: {sorted(entry.get('name') for entry in entries)}"
    )
    entry = matches[0]
    assert set(entry) == DOCTOR_DEPENDENCY_KEYS, (
        f"the {name!r} entry does not carry exactly {sorted(DOCTOR_DEPENDENCY_KEYS)}: "
        f"{sorted(entry)}"
    )
    return entry


# ---------------------------------------------------------------------------
# Row 3: the primary decoder is missing, and that blocks everything
# ---------------------------------------------------------------------------


def test_doctor_blocks_when_primary_decoder_is_missing(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absent primary decoder is blocking, named, and located on this platform.

    Breaks if ``doctor`` stops treating ``zxing-cpp`` as blocking -- it would print a table and exit
    0 -- or if its diagnostic stops naming the distribution a reader has to install, the interpreter
    and the platform whose wheel tag the pin would have to match (AAP §0.6.6 row 3, §0.7.2).
    """
    module_absent(monkeypatch, PRIMARY_MODULE)
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    table = invoke(run_cli, space, "doctor")
    assert table.exit_code == EXIT_CONFIG, (
        f"doctor exited {table.exit_code} with the primary decoder absent; "
        f"{diagnostic(table)}"
    )
    assert PRIMARY_DISTRIBUTION in table.stderr, (
        f"the diagnostic does not name {PRIMARY_DISTRIBUTION!r}; {diagnostic(table)}"
    )
    assert platform.python_version() in table.stderr, (
        f"the diagnostic does not name the interpreter {platform.python_version()}; "
        f"{diagnostic(table)}"
    )
    assert platform.machine().lower() in table.stderr.lower(), (
        f"the diagnostic does not name the platform {platform.machine()!r}; "
        f"{diagnostic(table)}"
    )

    as_json = invoke(run_cli, space, "doctor", "--json")
    assert as_json.exit_code == EXIT_CONFIG, (
        f"doctor --json exited {as_json.exit_code} with the primary decoder absent; "
        f"{diagnostic(as_json)}"
    )
    document = json.loads(as_json.stdout)
    assert set(document) == DOCTOR_REPORT_KEYS, (
        f"the report object does not carry exactly {sorted(DOCTOR_REPORT_KEYS)}: "
        f"{sorted(document)}"
    )
    primary = dependency(document, PRIMARY_DISTRIBUTION)
    assert primary["status"] == STATUS_MISSING, (
        f"{PRIMARY_DISTRIBUTION!r} is reported as {primary['status']!r}, not "
        f"{STATUS_MISSING!r}; entry: {primary}"
    )
    assert primary["blocking"] is True, (
        f"{PRIMARY_DISTRIBUTION!r} is reported as non-blocking, so a run without it would "
        f"look like a supported degradation; entry: {primary}"
    )
    # The rest of the inventory is still reported: a reader fixing an environment needs the
    # whole list, and the secondary decoder stays non-blocking beside the blocking primary.
    assert document["svg_rasterizer"] is not None, (
        "the report object carries no svg_rasterizer status, so the rasterizer was not "
        f"reported beside the missing decoder: {document!r}"
    )
    secondary = dependency(document, SECONDARY_DISTRIBUTION)
    assert secondary["blocking"] is False, (
        f"{SECONDARY_DISTRIBUTION!r} is reported as blocking; entry: {secondary}"
    )


def test_run_refuses_to_start_without_the_primary_decoder(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run without the primary decoder exits 2 and writes no report at all.

    Breaks if ``run`` degrades instead of refusing -- by letting ``RT-ROUNDTRIP`` become
    ``skipped`` -- or if it starts writing a report before the prerequisite is decided. The gate is
    the *absence* of any report: exit 2 is settled before a single case runs (AAP §0.3.4), so no
    result list exists in which a skip could hide, and the second selection proves the refusal is
    not a property of the one check whose name mentions a round trip.
    """
    module_absent(monkeypatch, PRIMARY_MODULE)
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    for selection in (ROUNDTRIP_CHECK, TARGET_INDEPENDENT_CHECK):
        result = invoke(
            run_cli,
            space,
            "run",
            "--adapter",
            REFERENCE_TARGET,
            "--ec",
            EC_LEVEL,
            "--checks",
            selection,
            "--class",
            CORPUS_CLASS,
        )
        assert result.exit_code == EXIT_CONFIG, (
            f"run --checks {selection} exited {result.exit_code} with the primary decoder "
            f"absent; {diagnostic(result)}"
        )
        assert PRIMARY_DISTRIBUTION in result.stderr, (
            f"the diagnostic for --checks {selection} does not name "
            f"{PRIMARY_DISTRIBUTION!r}; {diagnostic(result)}"
        )
        assert list(space.report_dir.rglob(SUMMARY_FILENAME)) == [], (
            f"run --checks {selection} wrote a {SUMMARY_FILENAME} even though it refused to "
            f"start, so a skipped or passing result exists for a run that must not have "
            f"happened; {diagnostic(result)}"
        )
        assert not (space.report_dir / HTML_FILENAME).exists(), (
            f"run --checks {selection} wrote {HTML_FILENAME} while refusing to start; "
            f"{diagnostic(result)}"
        )


# ---------------------------------------------------------------------------
# Row 4: the secondary decoder is missing, and that only degrades the run
# ---------------------------------------------------------------------------


def test_doctor_degrades_when_only_the_secondary_decoder_is_missing(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absent secondary decoder is reported, never fatal, and never ``missing``.

    Breaks if the corroborating decoder starts blocking a run -- ``doctor`` would exit 2 -- or if
    its absence stops being reported at all, which would leave a reader to infer a single-decoder
    run from the results alone (AAP §0.6.6 row 4, §0.7.2).
    """
    module_absent(monkeypatch, SECONDARY_MODULE)
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    result = invoke(run_cli, space, "doctor", "--json")
    assert result.exit_code == EXIT_OK, (
        f"doctor exited {result.exit_code} with only the secondary decoder absent; a "
        f"degradation must not fail the gate; {diagnostic(result)}"
    )
    document = json.loads(result.stdout)
    secondary = dependency(document, SECONDARY_DISTRIBUTION)
    assert secondary["status"] == STATUS_DEGRADED, (
        f"{SECONDARY_DISTRIBUTION!r} is reported as {secondary['status']!r}, not "
        f"{STATUS_DEGRADED!r}; entry: {secondary}"
    )
    assert secondary["blocking"] is False, (
        f"{SECONDARY_DISTRIBUTION!r} is reported as blocking; entry: {secondary}"
    )
    assert dependency(document, PRIMARY_DISTRIBUTION)["status"] == STATUS_OK, (
        f"the primary decoder is not reported usable beside the absent secondary: "
        f"{document['dependencies']!r}"
    )
    assert SECONDARY_DISTRIBUTION in result.stderr, (
        f"the degradation is not reported on stderr, so it exists only inside the JSON: "
        f"{diagnostic(result)}"
    )


def test_run_degrades_to_single_decoder_mode(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run completes, skips nothing, and records one decoder per decoded result.

    Breaks if a missing optional decoder turns a case into a ``skipped`` result -- which would let a
    check pass CI without measuring anything -- or if the consensus layer stops recording which
    decoders actually ran, which is the evidence row 4 is about.
    """
    module_absent(monkeypatch, SECONDARY_MODULE)
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    result = invoke(
        run_cli,
        space,
        "run",
        "--adapter",
        REFERENCE_TARGET,
        "--ec",
        EC_LEVEL,
        "--checks",
        ROUNDTRIP_CHECK,
        "--class",
        CORPUS_CLASS,
    )
    assert result.exit_code == EXIT_OK, (
        f"the single-decoder run exited {result.exit_code}; a supported degradation must "
        f"not change the outcome of a conforming target; {diagnostic(result)}"
    )
    summary = load_summary(space, result)
    counts = summary["counts"]
    assert counts["skipped"] == 0, (
        f"the run skipped {counts['skipped']} case(s) because the secondary decoder was "
        f"absent; a missing optional decoder must never become a skip; counts: {counts!r}"
    )
    assert counts["failed"] == 0 and counts["error"] == 0, (
        f"the single-decoder run reported failed={counts['failed']} error={counts['error']} "
        f"against a conforming target; counts: {counts!r}"
    )
    entries = results_of(summary)
    assert counts["passed"] == len(entries), (
        f"{counts['passed']} of {len(entries)} results passed with nothing skipped, failed "
        f"or errored; counts: {counts!r}"
    )
    assert len(entries) > 1, (
        f"the {CORPUS_CLASS!r} filter produced {len(entries)} result(s), so 'every result' "
        f"would be satisfied by a single case"
    )
    for entry in entries:
        decoded = entry["decoded"]
        assert decoded["agreement"] == SINGLE_DECODER_AGREEMENT, (
            f"{entry['check_id']} {entry['case_id']} records agreement "
            f"{decoded['agreement']!r}, not {SINGLE_DECODER_AGREEMENT!r}, with the "
            f"secondary decoder absent"
        )
        assert sorted(decoded["per_decoder"]) == [PRIMARY_DECODER], (
            f"{entry['check_id']} {entry['case_id']} records the decoders "
            f"{sorted(decoded['per_decoder'])}, not the primary decoder alone"
        )


def test_evidence_records_a_single_decoder_block(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing case's ``decoded.json`` states the single decoder, in canonical form.

    A passing case records ``evidence_dir`` as ``null`` by design (AAP §0.3.4), so this needs a
    case that really fails: the defective target's ``wrong_ec_level`` injection encodes one level
    below the request at EC M and makes every ``SY-EC-MATCH`` case fail (AAP §0.4.2). Breaks if the
    evidence store writes a decoder block the summary does not, or if the file stops being
    sorted-key JSON with one trailing newline, which keeps two runs of one case byte-identical.
    """
    module_absent(monkeypatch, SECONDARY_MODULE)
    space = workspace(write_config, tmp_path, target=DEFECTIVE_TARGET)

    result = invoke(
        run_cli,
        space,
        "run",
        "--adapter",
        DEFECTIVE_TARGET,
        "--ec",
        EC_LEVEL,
        "--checks",
        EC_CHECK,
        "--class",
        CORPUS_CLASS,
    )
    assert result.exit_code == EXIT_FAILED, (
        f"the defective run exited {result.exit_code}, expected {EXIT_FAILED}: the "
        f"wrong_ec_level injection must fail {EC_CHECK}; {diagnostic(result)}"
    )
    summary = load_summary(space, result)
    entries = results_of(summary)
    failing = [entry for entry in entries if entry["evidence_dir"]]
    assert failing, (
        f"no result of the {EC_CHECK} run over {CORPUS_CLASS!r} at EC {EC_LEVEL} carries an "
        f"evidence directory, so there is nothing to read; counts: {summary['counts']!r}, "
        f"statuses: {sorted({entry['status'] for entry in entries})}"
    )
    evidence_dir = space.report_dir / failing[0]["evidence_dir"]
    decoded_path = evidence_dir / "decoded.json"
    assert decoded_path.is_file(), (
        f"the failure evidence holds no decoded.json: "
        f"{sorted(path.name for path in evidence_dir.iterdir())}"
    )
    raw = decoded_path.read_text(encoding="utf-8")
    document = json.loads(raw)
    assert sorted(document["per_decoder"]) == [PRIMARY_DECODER], (
        f"the evidence records the decoders {sorted(document['per_decoder'])}, not the "
        f"primary decoder alone"
    )
    assert document["agreement"] == SINGLE_DECODER_AGREEMENT, (
        f"the evidence records agreement {document['agreement']!r}, not "
        f"{SINGLE_DECODER_AGREEMENT!r}"
    )
    canonical = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert raw == canonical, (
        f"{decoded_path} is not canonical sorted-key JSON with two-space indent and one "
        f"trailing newline, so two runs of one case would not be byte-identical"
    )


def test_report_states_the_degradation(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-decoder run's ``index.html`` states the reduced decoder set.

    Breaks if a reader can no longer tell from the report that only one decoder ran -- the
    degradation would then be inferable only from an empty column, which is what AAP §0.6.6 row 4
    forbids. The exact sentence is not asserted: the wording belongs to
    :mod:`qrcheck.report.html`, so a copy edit must not break this gate. What is asserted is that
    the notice exists inside the writer's own delimited banner element.
    """
    module_absent(monkeypatch, SECONDARY_MODULE)
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    result = invoke(
        run_cli,
        space,
        "run",
        "--adapter",
        REFERENCE_TARGET,
        "--ec",
        EC_LEVEL,
        "--checks",
        ROUNDTRIP_CHECK,
        "--class",
        CORPUS_CLASS,
    )
    assert result.exit_code == EXIT_OK, (
        f"the single-decoder run exited {result.exit_code}; {diagnostic(result)}"
    )
    html_path = space.report_dir / HTML_FILENAME
    assert html_path.is_file(), (
        f"the run wrote no {HTML_FILENAME} under {space.report_dir}; {diagnostic(result)}"
    )
    lowered = html_path.read_text(encoding="utf-8").lower()
    assert DEGRADATION_CLASS in lowered, (
        f"{HTML_FILENAME} carries no {DEGRADATION_CLASS!r} banner, so the single-decoder "
        f"mode is not stated in a delimited region"
    )
    assert DEGRADATION_LABEL in lowered, (
        f"{HTML_FILENAME} never mentions the degradation it describes; the words "
        f"{DEGRADATION_LABEL!r} and the absent decoder's name are both missing"
    )


# ---------------------------------------------------------------------------
# The control, without which the three tests above would prove nothing
# ---------------------------------------------------------------------------


def test_reference_run_uses_both_decoders_when_present(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Nothing stubbed: the same selection decodes under both decoders and says so.

    This is the control that makes the single-decoder assertions above meaningful -- without it they
    would pass just as well where the secondary decoder never worked. It also proves the degradation
    banner is *derived* from the evidence rather than emitted unconditionally, and it confirms
    ``monkeypatch`` gave the real ``cv2`` back, because it runs after the stubbed tests with nothing
    of its own patched.
    """
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    result = invoke(
        run_cli,
        space,
        "run",
        "--adapter",
        REFERENCE_TARGET,
        "--ec",
        EC_LEVEL,
        "--checks",
        ROUNDTRIP_CHECK,
        "--class",
        CORPUS_CLASS,
    )
    assert result.exit_code == EXIT_OK, (
        f"the two-decoder run exited {result.exit_code}; {diagnostic(result)}"
    )
    summary = load_summary(space, result)
    entries = results_of(summary)
    expected = sorted([PRIMARY_DECODER, SECONDARY_DECODER])
    assert len(entries) > 1, (
        f"the {CORPUS_CLASS!r} filter produced {len(entries)} result(s), so 'every result' "
        f"would be satisfied by a single case"
    )
    for entry in entries:
        decoded = entry["decoded"]
        assert sorted(decoded["per_decoder"]) == expected, (
            f"{entry['check_id']} {entry['case_id']} records the decoders "
            f"{sorted(decoded['per_decoder'])}, not {expected}"
        )
        assert decoded["agreement"] == TWO_DECODER_AGREEMENT, (
            f"{entry['check_id']} {entry['case_id']} records agreement "
            f"{decoded['agreement']!r}, not {TWO_DECODER_AGREEMENT!r}, with both decoders "
            f"available"
        )
    lowered = (space.report_dir / HTML_FILENAME).read_text(encoding="utf-8").lower()
    assert DEGRADATION_CLASS not in lowered, (
        f"{HTML_FILENAME} carries the single-decoder banner although both decoders ran, so "
        f"the notice is emitted unconditionally and states nothing about this run"
    )


# ---------------------------------------------------------------------------
# The mechanism's own postcondition
# ---------------------------------------------------------------------------


def test_module_state_is_restored_after_a_stubbed_call() -> None:
    """No blocked entry outlives a stubbed test, and both decoders are really here.

    ``monkeypatch`` restores every ``sys.modules`` entry it touched, and this test is the assertion
    of that postcondition: a leaked ``None`` entry would make every later module in the session see
    a decoder that is not missing. The availability probes are asserted too, and they *fail* rather
    than skip on a genuinely absent decoder: every supported platform has a wheel (AAP §0.2.3), so
    an absent one means the environment is broken and must be reported by name.
    """
    for module in (PRIMARY_MODULE, SECONDARY_MODULE):
        blocked = sys.modules.get(module, PRESENT) is None
        assert not blocked, (
            f"{module!r} is still blocked in sys.modules, so a stub outlived the test that "
            f"installed it and every later module in this session would treat an installed "
            f"decoder as absent"
        )

    availability = decoder_availability()
    assert availability.primary, (
        f"the {PRIMARY_DISTRIBUTION} distribution is not usable in this environment; it is "
        f"a blocking runtime dependency, so install the pins in requirements.lock"
    )
    assert availability.secondary, (
        f"the {SECONDARY_DISTRIBUTION} distribution is not usable in this environment; it is "
        f"declared in requirements.lock and every supported platform publishes a wheel for "
        f"it, so its absence means a broken install rather than a supported degradation"
    )
