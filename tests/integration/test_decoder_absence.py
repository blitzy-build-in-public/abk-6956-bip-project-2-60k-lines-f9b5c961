"""The absent-dependency rows of the state-handling table, held as gates.

AAP §0.6.6 names this module for two rows and AAP §0.7.2 fixes the distinction between them.
Row 3: ``zxing-cpp`` is the authority for every round-trip assertion, so its absence is
*blocking* -- ``qrcheck doctor`` and ``qrcheck run`` both exit 2 naming the dependency, and no
check is silently skipped in its place, the observable proof being that the run writes no report
at all, since exit 2 is decided before a single case runs (AAP §0.3.4). Row 4:
``opencv-python-headless`` corroborates and never decides, so its absence is a *supported
degradation* -- doctor stays green and reports the entry as degraded, the run proceeds in
single-decoder mode, every decoded result records ``agreement == "primary_only"`` with one
``per_decoder`` entry (or two, the second at status ``error``, when the decoder is installed but
unusable), the evidence a failing case writes shows the same decoder set, and ``index.html``
states the degradation.

A third degradation is held here too (AAP §0.3.5, §0.7.2): with no SVG rasterizer, ``RN-SVG``
skips whole, carrying its discovery order and its evidence directory, and never counts as a pass.

Absence is forced honestly: nothing is uninstalled and nothing inside the product is patched. A
``None`` entry in ``sys.modules`` makes ``import zxingcpp`` raise :exc:`ImportError` and
``importlib.util.find_spec`` -- the probe :mod:`qrcheck.decode.base` uses -- answer "not
present", and the decoder wrappers resolve their binding lazily on every call, so no cached
wrapper can short-circuit the probe; ``monkeypatch`` restores both entries at teardown, which
:func:`test_module_state_is_restored_after_a_stubbed_call` asserts. Every stubbed invocation
therefore runs **in this interpreter** through ``run_cli(..., in_process=True)``, because a
child process does not inherit a ``sys.modules`` entry, while row 3's harder state -- a binding
**present but unimportable**, which ``find_spec`` cannot produce -- runs out of process through
the :func:`unimportable_module` fixture's stub file ahead of ``site-packages`` on
``PYTHONPATH``. The degradation is reached a second way and held here too: an *installed* but
unimportable binding passes the availability probe, so consensus asks it and it answers
``status: "error"``, with only :func:`qrcheck.decode.opencv._load_cv2` replaced.

No test here skips for an absent decoder: every supported platform has a wheel (AAP §0.2.3), so
a genuinely missing one means a broken environment and must surface as a loud failure naming the
distribution and ``requirements.lock``. There is no ``xfail``, no retry and no ``pytest-mock``
(AAP §0.4.3, §0.5.6); no threshold is restated (§0.12); every run is bounded by a
``--checks`` selection, narrowed by ``--class`` where the selected check walks the corpus, so the
module carries no ``e2e`` marker; and every artefact lands under the test's ``tmp_path`` with a
fresh report root per test.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.checks.render import SVG_RASTERIZER_AUTO_ORDER
from qrcheck.cli.exit import EXIT_CONFIG, EXIT_FAILED, EXIT_OK
from qrcheck.decode import (
    PRIMARY_DECODER,
    SECONDARY_DECODER,
    decoder_availability,
)
from qrcheck.report import HTML_FILENAME, SUMMARY_FILENAME

# The names the mechanism turns on. The importable module names key a ``sys.modules`` entry; the
# distribution names are how AAP §0.7.2 writes the dependencies a reader has to install, and
# they are what the diagnostics below assert. Different vocabularies for the same two packages,
# so both are named here.

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

STATUS_SKIPPED: Final[str] = "skipped"
REASON_MISSING_DEPENDENCY: Final[str] = "missing_dependency"

#: The status a decoder records in ``per_decoder`` when it could not be run at all, which is what a
#: present-but-unimportable binding leaves behind: consensus asked it, and it answered with an
#: error rather than a payload.
UNUSABLE_DECODER_STATUS: Final[str] = "error"

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

# The selections every run in this module is bounded by: one corpus class and one check per run
# keeps the module in the fast job while still exercising the real encode, the real decoders and
# the real report writers. ``otp`` holds twelve ASCII records at AAP §0.8.1's case matrix and
# both decoders read them, which is what makes the single-decoder and two-decoder runs
# comparable.

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

#: The check whose only optional dependency is the SVG rasterizer, and the token it names.
SVG_CHECK: Final[str] = "RN-SVG"
RASTERIZER_TOKEN: Final[str] = "svg-rasterizer"

#: The discovery order AAP §0.3.4 fixes for ``[render] svg_rasterizer = "auto"``: the importable
#: ``cairosvg`` first, then the four command-line rasterizers on ``PATH``, which AAP §0.7.2
#: requires ``RN-SVG``'s ``missing_dependency`` skip to record. Spelled as a literal on purpose,
#: so the gate below pins the documented order instead of reading back the constant it exists to
#: check; AAP §0.12's single-declaration rule governs configurable thresholds, and this is a
#: contract value of AAP §0.3.4 rather than one of those.
SVG_RASTERIZER_DISCOVERY_ORDER: Final[tuple[str, ...]] = (
    "cairosvg",
    "rsvg-convert",
    "resvg",
    "magick",
    "convert",
)

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

#: The two fragments that mark a framework fault rather than an environment one. AAP §0.7.2 makes
#: an unusable decoder exit 2 with a named dependency, so neither may appear in its diagnostic.
INTERNAL_FAULT_MARKERS: Final[tuple[str, ...]] = ("internal error", "Traceback")


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


def rasterizer_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every rasterizer candidate unusable, whatever this host has installed.

    Only the per-candidate probe is replaced, so the declared order still runs and is recorded.
    """
    monkeypatch.setattr("qrcheck.checks.render._probe_rasterizer", lambda _name: False)


def workspace(
    write_config: Callable[..., Path],
    directory: Path,
    *,
    target: str,
    render: Mapping[str, Any] | None = None,
) -> Workspace:
    """Write a complete configuration document for ``target`` under ``directory``.

    The document carries every declared default the fixture emits and points ``report_dir`` at a
    fresh root, so a run started from it sees the committed thresholds for everything this module
    does not choose and writes nothing outside ``directory``. ``run`` still passes ``--adapter``
    explicitly, so the document's ``default_adapter`` and the run can never disagree silently.
    ``render`` overrides the ``[render]`` table, so a test states the rasterizer setting.
    """
    report_dir = directory / REPORT_DIR_NAME
    config = write_config(
        directory / CONFIG_FILE_NAME,
        default_adapter=target,
        report_dir=report_dir,
        render=render,
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
# Row 3 again, against the state a broken install really has: present but unimportable
# ---------------------------------------------------------------------------


def test_doctor_and_run_refuse_a_present_but_unimportable_primary_decoder(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    unimportable_module: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """A decoder installed yet impossible to import blocks, exactly as an absent one does.

    ``find_spec`` finds the stub, so a presence-only probe reports the decoder as available
    while the guarded import fails on every call. Breaks if ``run`` reaches its checks in that
    state: the review found it exiting 1 with a complete report of twelve ``value_mismatch``
    failures and ``agreement: secondary_only`` against a conforming target -- one blocking error
    reported as twelve fabricated generator defects (AAP §0.3.5, §0.6.6 row 3, §0.7.2). Both
    verbs are asserted here on purpose: they must agree about whether the decoder is usable.
    """
    unimportable_module(PRIMARY_MODULE)
    space = workspace(write_config, tmp_path, target=REFERENCE_TARGET)

    table = run_cli("doctor", config=space.config)
    assert table.exit_code == EXIT_CONFIG, (
        f"doctor exited {table.exit_code} with {PRIMARY_DISTRIBUTION} present but unimportable; "
        f"{diagnostic(table)}"
    )
    for expected in (PRIMARY_DISTRIBUTION, platform.python_version(), platform.machine().lower()):
        assert expected in table.stderr.lower(), (
            f"the diagnostic does not name {expected!r}; {diagnostic(table)}"
        )

    result = run_cli(
        "run",
        "--adapter",
        REFERENCE_TARGET,
        "--checks",
        ROUNDTRIP_CHECK,
        "--class",
        CORPUS_CLASS,
        config=space.config,
    )
    assert result.exit_code == EXIT_CONFIG, (
        f"run exited {result.exit_code} with {PRIMARY_DISTRIBUTION} present but unimportable; the "
        f"secondary decoder must never become the authority that fails a conforming target; "
        f"{diagnostic(result)}"
    )
    assert PRIMARY_DISTRIBUTION in result.stderr, (
        f"the diagnostic does not name {PRIMARY_DISTRIBUTION!r}; {diagnostic(result)}"
    )
    for marker in INTERNAL_FAULT_MARKERS:
        assert marker not in result.stderr, (
            f"the outcome carries {marker!r}, so a broken install is reported as a framework fault "
            f"rather than an environment one; {diagnostic(result)}"
        )
    assert list(space.report_dir.rglob(SUMMARY_FILENAME)) == [], (
        f"the run wrote a {SUMMARY_FILENAME} while refusing to start, so a fabricated "
        f"value_mismatch result exists for a run that must not have happened; "
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


def test_run_degrades_when_the_secondary_decoder_cannot_be_imported(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decoder that is installed but unusable degrades the run, and the report says so.

    ``module_absent`` above is the easy half of row 4: ``find_spec`` answers ``None``, consensus
    never asks the secondary decoder, and its name never reaches ``per_decoder``. A binding that is
    *installed* but cannot be imported is the half that looks different while meaning the same
    thing -- the availability probe discovers it, so consensus does ask it, and it answers with
    ``status: "error"``. The run is single-decoder in effect while the evidence names both
    decoders, so the degradation has to be derived from what each decoder recorded, not from the
    decoder set. Breaks if ``index.html`` goes silent again for this shape, which is how a degraded
    run came to be indistinguishable from a fully corroborated one (AAP §0.6.6 row 4, §0.7.2).
    """
    assert importlib.util.find_spec(SECONDARY_MODULE) is not None, (
        f"{SECONDARY_MODULE!r} is not installed in this environment, so the binding could not be "
        f"discovered even before it was made unusable and this is not the path under test; it is "
        f"declared in requirements.lock, so install the pins there"
    )
    monkeypatch.setattr("qrcheck.decode.opencv._load_cv2", lambda: None)
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
        f"the present-but-unusable run exited {result.exit_code}; an unusable corroborating "
        f"decoder is a supported degradation, so a conforming target must still pass; "
        f"{diagnostic(result)}"
    )
    summary = load_summary(space, result)
    counts = summary["counts"]
    assert counts["failed"] == 0 and counts["error"] == 0 and counts["skipped"] == 0, (
        f"the present-but-unusable run reported failed={counts['failed']} "
        f"error={counts['error']} skipped={counts['skipped']} against a conforming target; a "
        f"decoder that could not answer must not fail, error or skip a case; counts: {counts!r}"
    )
    entries = results_of(summary)
    for entry in entries:
        decoded = entry["decoded"]
        assert decoded["agreement"] == SINGLE_DECODER_AGREEMENT, (
            f"{entry['check_id']} {entry['case_id']} records agreement "
            f"{decoded['agreement']!r}, not {SINGLE_DECODER_AGREEMENT!r}, although the "
            f"secondary decoder answered with an error"
        )
        assert sorted(decoded["per_decoder"]) == sorted([PRIMARY_DECODER, SECONDARY_DECODER]), (
            f"{entry['check_id']} {entry['case_id']} records the decoders "
            f"{sorted(decoded['per_decoder'])}; an installed-but-unusable decoder is asked, so "
            f"both names are recorded even though only one answered"
        )
        secondary = decoded["per_decoder"][SECONDARY_DECODER]
        assert secondary["status"] == UNUSABLE_DECODER_STATUS, (
            f"{entry['check_id']} {entry['case_id']} records the secondary decoder's status as "
            f"{secondary['status']!r}, not {UNUSABLE_DECODER_STATUS!r}; entry: {secondary}"
        )
    html_path = space.report_dir / HTML_FILENAME
    assert html_path.is_file(), (
        f"the run wrote no {HTML_FILENAME} under {space.report_dir}; {diagnostic(result)}"
    )
    lowered = html_path.read_text(encoding="utf-8").lower()
    assert DEGRADATION_CLASS in lowered, (
        f"{HTML_FILENAME} carries no {DEGRADATION_CLASS!r} banner although the secondary "
        f"decoder could not be imported, so the degradation is inferable only from the "
        f"per-decoder statuses"
    )
    assert DEGRADATION_LABEL in lowered, (
        f"{HTML_FILENAME} never mentions the degradation it describes; the word "
        f"{DEGRADATION_LABEL!r} is missing"
    )


# ---------------------------------------------------------------------------
# The control, without which the four row-4 tests above would prove nothing
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
# The third degradation: no SVG rasterizer, so RN-SVG cannot rasterize
# ---------------------------------------------------------------------------


def test_rn_svg_skips_with_the_discovery_order_it_attempted(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent rasterizer skips ``RN-SVG`` and the skip names the order it walked.

    AAP §0.3.5 and §0.7.2: the rasterizer degrades to ``skipped``/``missing_dependency`` with the
    discovery order it attempted, and AAP §0.3.4 requires evidence for the skip. Breaks if the
    product's ``auto`` order departs from the one AAP §0.3.4 fixes, if the order stops reaching the
    result, or if the degradation becomes a failure, an error or a pass.
    """
    documented_order = list(SVG_RASTERIZER_DISCOVERY_ORDER)
    assert list(SVG_RASTERIZER_AUTO_ORDER) == documented_order, (
        f"the product's ``auto`` discovery order is {list(SVG_RASTERIZER_AUTO_ORDER)}, not the "
        f"{documented_order} AAP §0.3.4 fixes; a reordering or a dropped candidate changes the "
        f"order every report states it attempted"
    )
    rasterizer_absent(monkeypatch)
    space = workspace(
        write_config, tmp_path, target=REFERENCE_TARGET, render={"svg_rasterizer": "auto"}
    )

    result = invoke(run_cli, space, "run", "--adapter", REFERENCE_TARGET, "--checks", SVG_CHECK)
    assert result.exit_code == EXIT_OK, (
        f"the run exited {result.exit_code} with no usable rasterizer; a skip is neither a "
        f"failure nor an error, so the run completes; {diagnostic(result)}"
    )
    summary = load_summary(space, result)
    counts = summary["counts"]
    assert counts == {"passed": 0, "failed": 0, "skipped": 1, "error": 0}, (
        f"the degraded run's tallies are {counts!r}; a whole-check skip counts once, nowhere else"
    )
    entries = results_of(summary)
    assert len(entries) == 1, f"{SVG_CHECK} contributed {len(entries)} results, not one skip"
    (entry,) = entries
    assert entry["check_id"] == SVG_CHECK
    assert entry["case_id"] == "synthetic:capability", entry["case_id"]
    assert entry["status"] == STATUS_SKIPPED, entry["status"]
    assert entry["reason"] == REASON_MISSING_DEPENDENCY, entry["reason"]
    assert entry["expected"] == {"missing": [RASTERIZER_TOKEN]}, entry["expected"]
    assert entry["decoded"] == {"attempted": documented_order}, (
        f"the skip's decoded payload is {entry['decoded']!r}; AAP §0.3.5 and §0.7.2 require the "
        f"discovery order it attempted, which AAP §0.3.4 fixes as {documented_order}"
    )

    evidence_dir = entry["evidence_dir"]
    assert isinstance(evidence_dir, str) and evidence_dir, (
        f"a skipped-with-reason case records its evidence directory (AAP §0.3.4); got "
        f"{evidence_dir!r}"
    )
    assert not Path(evidence_dir).is_absolute(), evidence_dir
    case_dir = space.report_dir / evidence_dir
    names = sorted(path.name for path in case_dir.iterdir())
    assert names == ["decoded.json", "request.json"], (
        f"a whole-check skip produced no symbol, so {case_dir} holds {names}"
    )
    persisted = json.loads((case_dir / "decoded.json").read_text(encoding="utf-8"))
    assert persisted == entry["decoded"], f"{case_dir} decoded.json holds {persisted!r}"


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
