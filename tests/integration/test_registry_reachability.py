"""Registry reachability: the checks that exist are the checks a real run executes.

The product's MUST is that every check implements the same contract and is discoverable from the
registry, and that no check is importable-only or callable solely from a test (AAP §0.1.2,
§0.3.4). ``qrcheck/registry/check.py`` is the single owner of check identity, ``qrcheck list``
publishes it, and a run's ``summary.json`` records what actually executed; this module holds all
three against each other, which is AAP §0.4.1 and §0.6.1 #8 -- a check registered but never run, a
check run but never registered, a listing that omits one, or an identifier renamed, dropped or
doubled each fails here, by name.

The cases: the registry holds exactly the twenty-one declared identifiers, asserted on
non-emptiness and on the count *before* any set comparison, because an empty registry would
satisfy an executed-set equality against an empty result set vacuously; the registered set
equals the ``check_id`` set the reference run recorded, with "registered but never ran" and
"ran but is not registered" reported as two separate lists, and every registered check holding
at least one ``passed`` result rather than only a skip or an error (AAP §0.4.1's fourth
clause); the identifiers are the literal twenty-one, one entry each and an iteration order
ascending by identifier; ``qrcheck list --json`` publishes every one of them with the five
declared keys and the requirement sets the registry declares; the nine families partition the
twenty-one identifiers, compared as a partition so a renamed vocabulary still passes while a
split or merged family fails; every entry carries the six fields of ``CheckContract`` with the
declared types; importing ``qrcheck.checks`` in a fresh interpreter registers all twenty-one,
the property no test-only import path can fake; and the built-in targets' expected failing
sets validate against the registry, so deleting an expectation or a check breaks a test rather
than shrinking what the acceptance gate compares (AAP §0.4.3).

The executed-set half reads the session-scoped ``reference_report`` fixture, so the whole
suite is paid for once per session however many modules ask for it (AAP §0.6.1 #9); the
listing half is a ``qrcheck list --json`` invocation, which registers the checks and prints
them without running one. The marker is ``e2e`` because of that suite, so ``-m "not e2e"``
deselects this module from the fast job (AAP §0.6.4).

Nothing here skips, xfails, retries or mocks (AAP §0.4.3, §0.5.6), no assertion reads a clock -- a
``list`` invocation is never timed -- every collection is compared in sorted order, and every
message names the check identifier it failed on. The only file this module writes is the
configuration the listing is invoked with, under ``tmp_path``; the repository's own ``reports/`` is
never touched, because runtime artefacts are not committed deliverables (AAP §0.11.1). """

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.registry.check import REGISTRY, CheckContract, iter_checks
from qrcheck.registry.expectations import SELF_CHECK_TARGETS, validate_against_registry
from qrcheck.targets.base import Capability

# The end-to-end half of the reachability gate reads the report a full `--adapter reference` run
# left behind, so the module is deselected by `-m "not e2e"` in the fast job and runs once in
# the e2e job (AAP §0.6.4). The `list --json` call it also makes runs no suite.
pytestmark = pytest.mark.e2e

# The twenty-one check identifiers of AAP §0.3.4, written out literally on purpose: comparing
# the registry against qrcheck.registry.check.CHECK_IDS would be the registry agreeing with
# itself, and this is the gate that has to catch a renamed, dropped or doubled identifier. A
# change to this set is a change to the acceptance criteria, not to a test.
_EXPECTED_CHECK_IDS: Final[frozenset[str]] = frozenset(
    {
        "CL-ARTIFACTS",
        "DM-RESILIENCE",
        "DM-SEEDED",
        "DT-DETERMINISTIC",
        "IV-BAD-OPTION",
        "IV-EMPTY",
        "IV-OVERCAPACITY",
        "MD-CHARSET",
        "MD-MODE-OPTIMAL",
        "PF-DECODE",
        "PF-ENCODE",
        "RN-CONTRAST",
        "RN-LOGO",
        "RN-MODULE-SIZE",
        "RN-QUIET-ZONE",
        "RN-SVG",
        "RT-ROUNDTRIP",
        "SY-EC-MATCH",
        "SY-MASK",
        "SY-MATRIX-EQUIV",
        "SY-VERSION-MIN",
    }
)

# Declared separately from the set above rather than derived from it: a mistyped identifier that
# collapsed two entries into one would keep len(_EXPECTED_CHECK_IDS) constant, and the count is
# the acceptance criterion the case matrix is written for.
_EXPECTED_CHECK_COUNT: Final[int] = 21

# The nine families of AAP §0.8.1's component-to-check-family table, as a partition of the
# twenty-one identifiers: one set per family module. Compared as a set of sets rather than by
# family name, because the partition is what the one-module-per-family rule fixes -- a different
# naming vocabulary in the product is cosmetic, a split or merged family is structural.
_EXPECTED_FAMILY_PARTITION: Final[frozenset[frozenset[str]]] = frozenset(
    {
        frozenset({"RT-ROUNDTRIP"}),
        frozenset({"SY-VERSION-MIN", "SY-EC-MATCH", "SY-MASK", "SY-MATRIX-EQUIV"}),
        frozenset({"MD-MODE-OPTIMAL", "MD-CHARSET"}),
        frozenset({"RN-QUIET-ZONE", "RN-MODULE-SIZE", "RN-CONTRAST", "RN-SVG", "RN-LOGO"}),
        frozenset({"DM-RESILIENCE", "DM-SEEDED"}),
        frozenset({"IV-EMPTY", "IV-OVERCAPACITY", "IV-BAD-OPTION"}),
        frozenset({"DT-DETERMINISTIC"}),
        frozenset({"PF-ENCODE", "PF-DECODE"}),
        frozenset({"CL-ARTIFACTS"}),
    }
)

# The five keys AAP §0.8.3 fixes for one `--json` row, and no others: a renamed key, a wrapping
# object or an extra nesting level would make a tooling consumer mis-parse a correct run.
_LIST_JSON_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "title", "family", "requires", "dependencies"}
)

# The closed capability vocabulary, taken from the port's own StrEnum so that a member added,
# renamed or removed there is what the listing is validated against here.
_CAPABILITY_VALUES: Final[frozenset[str]] = frozenset(member.value for member in Capability)

# The child process that proves registration needs no test-only import path: it imports the
# family package exactly as an entry point does, then prints the size of the registry it found.
_REGISTRATION_SCRIPT: Final[str] = (
    "import qrcheck.checks; from qrcheck.registry.check import REGISTRY; "
    "print(len(REGISTRY))"
)

# A harness guard, not a product threshold: no check reads it, no configuration declares it, and
# it bounds a hang rather than measuring anything.
_SUBPROCESS_TIMEOUT_S: Final[float] = 300.0

# What the listing is invoked with. `qrcheck list` reads no configuration at all -- a listing has
# to answer "which checks does this installation have?" before a target is configured -- so these
# values exist only to satisfy the group's `--config` option. The adapter named is one both
# built-in target tables declare, so the document write_config produces is schema-valid.
_CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"
_CONFIG_ADAPTER: Final[str] = "reference"
_CONFIG_REPORT_DIR: Final[str] = "reports-list"


@pytest.fixture(name="registry")
def _registry() -> Mapping[str, CheckContract]:
    """Return the populated registry, imported the way the product's entry points import it.

    Registration is import-time (AAP §0.3.4): ``qrcheck run`` and ``qrcheck list`` both import
    :mod:`qrcheck.checks` before they read the registry, and this fixture does the same. It is
    deliberately *not* autouse and not a module-level import, so the first test below remains the
    only thing between a broken import path and an empty registry: an autouse import would make an
    empty registry impossible to observe and the gate unable to fail. The view is live, so it holds
    the entries the import registered rather than a snapshot taken before it ran.
    """
    import qrcheck.checks  # noqa: F401 -- registration is this import's side effect
    return REGISTRY


def _list_rows(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> list[dict[str, Any]]:
    """Run ``qrcheck list --json`` as a real process and return its parsed rows, in printed
    order.

    A non-zero status, or a stdout that is not an object array, fails the test: a listing
    neither inspects nor writes a report, so that is a defect in the command rather than a
    verdict on a target. """
    config_path = write_config(
        tmp_path / _CONFIG_FILE_NAME,
        default_adapter=_CONFIG_ADAPTER,
        report_dir=tmp_path / _CONFIG_REPORT_DIR,
    )
    result = run_cli("list", "--json", config=config_path)
    assert result.exit_code == 0, (
        f"qrcheck list --json exited {result.exit_code}, so there is no listing to compare: "
        f"{result.stderr.strip()!r}"
    )
    rows = json.loads(result.stdout)
    assert isinstance(rows, list), (
        f"the --json form must be an object array (AAP §0.8.3), got {type(rows).__name__}"
    )
    return rows


def test_registry_is_populated_and_exactly_twenty_one() -> None:
    """Importing the checks registers the twenty-one declared ones, and nothing else happens.

    Catches a family module that stopped registering, a registration that became conditional
    and a check that was dropped or doubled. The non-emptiness assertion comes first, before
    any set comparison, because an empty registry would satisfy the next test's executed-set
    equality vacuously -- the one way this gate could pass while proving nothing. """
    import qrcheck.checks  # noqa: F401 -- registration is this import's side effect

    registered = sorted(REGISTRY)
    assert registered, (
        "the registry is empty after importing qrcheck.checks: no check family registered "
        "itself, so every other assertion in this module would pass vacuously"
    )
    assert len(REGISTRY) == _EXPECTED_CHECK_COUNT, (
        f"the registry holds {len(REGISTRY)} checks {registered}; the acceptance criteria are "
        f"written for {_EXPECTED_CHECK_COUNT}"
    )
    assert set(REGISTRY) == _EXPECTED_CHECK_IDS, (
        f"the registry holds {sorted(set(REGISTRY) - _EXPECTED_CHECK_IDS)} beyond the declared "
        f"checks and lacks {sorted(_EXPECTED_CHECK_IDS - set(REGISTRY))}"
    )


def test_every_registered_check_executed_in_the_reference_run(
    reference_report: tuple[Path, dict[str, Any], int],
    registry: Mapping[str, CheckContract],
) -> None:
    """Every registered check produced results in a real run, and at least one of them passed.

    AAP §0.4.1, verbatim: ``set(REGISTRY) == {r["check_id"] for r in summary["results"]}``. A
    check listed but never executed is one CI cannot trend; a check executed but never listed
    is one a consumer cannot name, so the two directions are reported separately. The
    reference run's own exit code is deliberately not asserted here --
    ``test_reference_run.py`` owns that gate -- so this module reports the same failure
    whichever of the two noticed it first. """
    _report_root, summary, exit_code = reference_report
    registered = set(registry)
    executed = {result["check_id"] for result in summary["results"]}

    never_ran = sorted(registered - executed)
    unregistered = sorted(executed - registered)
    assert registered == executed, (
        f"registered but never ran: {never_ran}; ran but is not registered: {unregistered} "
        f"(the reference run wrote {len(summary['results'])} results and exited {exit_code})"
    )
    assert len(executed) == _EXPECTED_CHECK_COUNT, (
        f"the reference run executed {len(executed)} checks, not {_EXPECTED_CHECK_COUNT}"
    )

    passed = {
        result["check_id"]
        for result in summary["results"]
        if result["status"] == "passed"
    }
    assert passed == registered, (
        f"registered checks with no passed result in the reference run: "
        f"{sorted(registered - passed)}; a check that only skipped or errored proved nothing "
        f"about the target (AAP §0.4.1)"
    )


def test_registry_ids_are_the_literal_set(registry: Mapping[str, CheckContract]) -> None:
    """The registry holds the literal twenty-one identifiers, in an ascending iteration order.

    Catches a renamed, dropped or silently duplicated identifier, which a comparison against
    ``CHECK_IDS`` could not: that tuple is the registry's own declaration. One entry per
    identifier and an iteration order that ignores import order (AAP §0.5.3) are asserted on
    :func:`qrcheck.registry.check.iter_checks`, the surface every consumer iterates; the
    dict's own order is the family modules' import order and is not a contract, so it is not
    asserted here. """
    assert set(registry) == _EXPECTED_CHECK_IDS, (
        f"the registry holds {sorted(set(registry) - _EXPECTED_CHECK_IDS)} beyond the declared "
        f"checks and lacks {sorted(_EXPECTED_CHECK_IDS - set(registry))}"
    )
    assert len(registry) == len(set(registry)) == _EXPECTED_CHECK_COUNT, (
        f"the registry holds {len(registry)} entries for {len(set(registry))} identifiers; a "
        f"duplicate can arrive only through a second registration, which must raise instead"
    )

    first = tuple(contract.id for contract in iter_checks())
    second = tuple(contract.id for contract in iter_checks())
    assert first == second, f"iter_checks() returned two orders in one process: {first}, {second}"
    assert first == tuple(sorted(_EXPECTED_CHECK_IDS)), (
        f"iter_checks() must iterate by identifier ascending, got {first}"
    )


def test_every_registry_entry_appears_in_list_json(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
    registry: Mapping[str, CheckContract],
) -> None:
    """``qrcheck list --json`` publishes every registered check, declaring what it requires.

    The machine-readable half of AAP §0.6.1 #8. A check reachable only from an import would be
    absent from the listing a CI consumer reads, and a listing whose requirement lists disagreed
    with the registry would tell a target author to declare capabilities the product never asks for.
    Sortedness is asserted on the emitted order (AAP §0.5.3), and the row count separately from the
    identifier set, because a duplicated row would satisfy both the set comparison and the order.
    """
    rows = _list_rows(run_cli, write_config, tmp_path)
    listed = [row["id"] for row in rows]
    assert set(listed) == set(registry), (
        f"the listing omits {sorted(set(registry) - set(listed))} and adds "
        f"{sorted(set(listed) - set(registry))}"
    )
    assert listed == sorted(listed), (
        f"the listing must be ascending by identifier (AAP §0.5.3), got {listed}"
    )
    assert len(listed) == _EXPECTED_CHECK_COUNT, (
        f"the listing published {len(listed)} rows for {len(set(listed))} identifiers; the "
        f"registry holds {_EXPECTED_CHECK_COUNT} checks, so a row was repeated or dropped"
    )

    for row in rows:
        check_id = row["id"]
        contract = registry[check_id]
        assert list(row) == sorted(row), (
            f"{check_id}: the --json object's keys must be sorted, got {list(row)}"
        )
        assert set(row) == _LIST_JSON_KEYS, (
            f"{check_id}: the --json object carries {sorted(set(row))}; the contract is exactly "
            f"{sorted(_LIST_JSON_KEYS)}"
        )
        assert isinstance(row["title"], str) and row["title"].strip(), (
            f"{check_id}: the listing carries an empty title"
        )
        assert isinstance(row["family"], str) and row["family"].strip(), (
            f"{check_id}: the listing carries an empty family"
        )
        for field in ("requires", "dependencies"):
            value = row[field]
            assert isinstance(value, list), (
                f"{check_id}: {field} must be a JSON array, got {type(value).__name__}"
            )
            assert value == sorted(value), (
                f"{check_id}: {field} must be emitted sorted (AAP §0.5.3), got {value}"
            )
        for member in row["requires"]:
            assert member in _CAPABILITY_VALUES, (
                f"{check_id}: the listing publishes {member!r}, which is not one of the "
                f"capabilities {sorted(_CAPABILITY_VALUES)}"
            )
        declared_caps = sorted(str(member) for member in contract.requires)
        assert set(row["requires"]) == set(declared_caps), (
            f"{check_id}: the listing publishes the capabilities "
            f"{sorted(row['requires'])} but the registry requires {declared_caps}"
        )
        assert tuple(row["dependencies"]) == contract.dependencies, (
            f"{check_id}: the listing publishes the dependencies {row['dependencies']} but the "
            f"registry declares {list(contract.dependencies)}"
        )


def test_family_partition_matches_the_plan(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The nine families partition the twenty-one checks exactly as the plan's table does.

    AAP §0.8.1's one-module-per-family rule is what the partition encodes, so a family that
    split in two, merged with another or lost a check fails here, and the JUnit ``classname``
    and report section that follow from the family stay meaningful. Compared as a set of sets:
    the family names are the product's vocabulary and may be spelled differently without
    breaking this case, whereas a structural change cannot pass.
    """
    rows = _list_rows(run_cli, write_config, tmp_path)
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(row["family"], set()).add(row["id"])

    partition = frozenset(frozenset(ids) for ids in grouped.values())
    assert partition == _EXPECTED_FAMILY_PARTITION, (
        f"the listing's families partition the checks as "
        f"{sorted(sorted(ids) for ids in partition)}; the plan's mapping partitions them as "
        f"{sorted(sorted(ids) for ids in _EXPECTED_FAMILY_PARTITION)}"
    )
    covered = frozenset().union(*grouped.values())
    assert covered == _EXPECTED_CHECK_IDS, (
        f"the families must cover every declared check exactly once; uncovered: "
        f"{sorted(_EXPECTED_CHECK_IDS - covered)}"
    )


def test_each_entry_satisfies_the_check_contract(
    registry: Mapping[str, CheckContract],
) -> None:
    """Every registry entry carries the six fields of ``CheckContract`` with their types.

    The per-check capability and dependency *tables* belong to
    ``tests/unit/test_registry_contract.py`` and are deliberately not restated here, so
    retuning AAP §0.3.4's requirement table leaves one place to edit. What this case holds is
    the shape every consumer reads -- identity, title, family, requirement types and a
    callable case generator -- so a field that changed type or lost its value fails once,
    naming the check. """
    assert registry, "the registry is empty, so there is no entry to inspect"
    for check_id in sorted(registry):
        entry = registry[check_id]
        # ``CheckContract`` is a frozen dataclass, not a Protocol, so this is a real check of the
        # object the registry holds rather than a structural one.
        assert isinstance(entry, CheckContract), (
            f"{check_id}: the registry holds a {type(entry).__name__}, not a CheckContract"
        )
        assert entry.id == check_id, f"{check_id}: the entry is registered under {entry.id!r}"
        assert isinstance(entry.title, str) and entry.title.strip(), (
            f"{check_id}: the title is empty, and 'qrcheck list' prints it"
        )
        assert isinstance(entry.family, str) and entry.family.strip(), (
            f"{check_id}: the family is empty, and it is the JUnit classname suffix"
        )
        assert isinstance(entry.requires, frozenset), (
            f"{check_id}: requires is a {type(entry.requires).__name__}, not a frozenset"
        )
        not_capabilities = sorted(
            repr(member) for member in entry.requires if not isinstance(member, Capability)
        )
        assert not not_capabilities, (
            f"{check_id}: requires holds {not_capabilities} instead of Capability members"
        )
        assert isinstance(entry.dependencies, tuple), (
            f"{check_id}: dependencies is a {type(entry.dependencies).__name__}, not a tuple"
        )
        not_tokens = sorted(
            repr(token) for token in entry.dependencies if not isinstance(token, str)
        )
        assert not not_tokens, (
            f"{check_id}: dependencies holds {not_tokens} instead of dependency tokens"
        )
        assert callable(entry.cases), (
            f"{check_id}: cases is not callable, so the check contributes no case"
        )


def test_registration_happens_on_package_import_alone(repo_root: Path) -> None:
    """A fresh interpreter that imports ``qrcheck.checks`` finds all twenty-one checks.

    The clause no test-only import path can satisfy: the child inherits nothing from this
    process's ``sys.modules``, so it proves registration belongs to importing the package --
    which is exactly what ``qrcheck run`` and ``qrcheck list`` do -- rather than to a test
    importing one family module by name. ``cwd`` is the repository root so the child resolves
    the same checkout this session runs from, and ``PYTHONHASHSEED`` is fixed so the child's
    registry order cannot vary in a way that would mask a duplicate registration. """
    completed = subprocess.run(
        [sys.executable, "-c", _REGISTRATION_SCRIPT],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    assert completed.returncode == 0, (
        f"a fresh interpreter could not import the check families: {completed.stderr.strip()!r}"
    )
    assert completed.stdout.strip() == str(_EXPECTED_CHECK_COUNT), (
        f"a fresh interpreter that imported only qrcheck.checks registered "
        f"{completed.stdout.strip()!r} checks, not {_EXPECTED_CHECK_COUNT}"
    )


def test_expectations_agree_with_the_registry(registry: Mapping[str, CheckContract]) -> None:
    """The self-check targets' expected failing sets name only registered checks.

    ``validate_against_registry`` raises for an expected identifier that is declared but not
    registered, so deleting a check without deleting its expectation -- or the reverse --
    breaks a test rather than silently shrinking the set the defective run's acceptance
    criterion compares (AAP §0.4.3). ``reference`` must expect nothing to fail; its zero-skip
    requirement is the reference-run gate's, not this one's. """
    validate_against_registry(registry)

    conforming = SELF_CHECK_TARGETS["reference"]
    assert conforming == frozenset(), (
        f"the conforming target is expected to fail {sorted(conforming)}, but it must fail "
        "nothing"
    )
    defective = SELF_CHECK_TARGETS["defective"]
    assert defective <= set(registry), (
        f"the defective target expects unregistered checks: {sorted(defective - set(registry))}"
    )
