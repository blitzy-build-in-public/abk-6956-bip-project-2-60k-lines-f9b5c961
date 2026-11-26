"""Contract test: check identity, the fields each check declares, and resolution to skips.

``qrcheck/registry/check.py`` owns check identity and ordering and
``qrcheck/registry/resolve.py`` is the single place a shortfall becomes a skip (AAP §0.3.4). A
check that never runs, an undeclared identifier or a shortfall counted as a pass lets a report
describe a run that never happened:

* the registry holds **exactly** the twenty-one identifiers, and duplication raises;
* every contract carries the declared fields; iteration is by identifier ascending, also after a
  child process imports the families in reverse :data:`FAMILY_MODULES` order (§0.5.3);
* the nine family modules, their identifier ownership and every check's requirements are the
  plan's;
* an unmet capability is ``unsupported_capability``, a missing dependency
  ``missing_dependency`` carrying the discovery order it was probed in (AAP §0.3.5, §0.7.2),
  with the capability reason outranking and ``passed`` staying at zero (AAP §0.6.6 row 9);
* each built-in target's expected failing set is the declared one and ``list --json`` lists
  them.

The dependency column follows the AAP's table: ``PF-DECODE`` decodes a fixed probe image and
``CL-ARTIFACTS`` observes the run, both carrying the primary decoder as their optional
dependency, which can never become a skip for either (AAP §0.6.6 row 3)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

from qrcheck.checks import FAMILY_MODULES, load_families
from qrcheck.cli.main import cli
from qrcheck.registry import expectations as _expectations_module
from qrcheck.registry.check import (
    CHECK_IDS,
    DEPENDENCY_PRIMARY_DECODER,
    DEPENDENCY_SVG_RASTERIZER,
    DEPENDENCY_TOKENS,
    FAMILY_BY_PREFIX,
    FAMILY_NAMES,
    REGISTRY,
    CheckContract,
    CheckRegistrationError,
    get_check,
    isolated_registry,
    iter_checks,
    register,
    registered_ids,
)
from qrcheck.registry.expectations import (
    DEFECT_EXPECTATIONS,
    SELF_CHECK_TARGETS,
    ExpectationError,
    expected_failures,
    expected_passing,
    validate_against_registry,
)
from qrcheck.registry.resolve import (
    CAPABILITY_CASE_ID,
    Resolution,
    SkipOutcome,
    resolve,
    skip_case_results,
    unmet_capabilities,
    unmet_dependencies,
)
from qrcheck.report.model import CaseResult, Counts, Reason, Status
from qrcheck.targets.base import Capability

# The twenty-one check identifiers, and how many there are. Stated once from AAP §0.3.4's
# checks table, which names each row exactly once and is the acceptance criterion the registry
# is measured against: an identifier added, renamed, split or merged is a change to that table
# and therefore to this literal, never a silent difference between a run and its report.
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
_EXPECTED_CHECK_COUNT: Final[int] = 21

# The nine family modules, in the order AAP §0.8.1's repository tree lists them. The order is
# not alphabetical on purpose: ``FAMILY_NAMES`` in the registry is, and the two must agree as
# sets while the loader keeps the reviewable plan order.
_EXPECTED_FAMILY_MODULES: Final[tuple[str, ...]] = (
    "roundtrip",
    "symbol",
    "mode",
    "render",
    "damage",
    "invalid",
    "determinism",
    "performance",
    "cleanup",
)

# Which identifiers each family module owns, from AAP §0.8.1's component-to-check-family
# mapping. This is the one-module-per-family rule, and it fixes two output surfaces: the JUnit
# ``classname="qrcheck.<family>"`` of every check and the section of the HTML report it lands in.
_EXPECTED_IDS_BY_FAMILY: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "cleanup": frozenset({"CL-ARTIFACTS"}),
        "damage": frozenset({"DM-RESILIENCE", "DM-SEEDED"}),
        "determinism": frozenset({"DT-DETERMINISTIC"}),
        "invalid": frozenset({"IV-BAD-OPTION", "IV-EMPTY", "IV-OVERCAPACITY"}),
        "mode": frozenset({"MD-CHARSET", "MD-MODE-OPTIMAL"}),
        "performance": frozenset({"PF-DECODE", "PF-ENCODE"}),
        "render": frozenset(
            {"RN-CONTRAST", "RN-LOGO", "RN-MODULE-SIZE", "RN-QUIET-ZONE", "RN-SVG"}
        ),
        "roundtrip": frozenset({"RT-ROUNDTRIP"}),
        "symbol": frozenset({"SY-EC-MATCH", "SY-MASK", "SY-MATRIX-EQUIV", "SY-VERSION-MIN"}),
    }
)

# The capability names one group of checks requires, named once so the two requirement sets
# below read as the plan's table rather than as twenty-one restatements.
_PNG_ONLY: Final[frozenset[Capability]] = frozenset({Capability.png})
_PNG_AND_EC_LEVEL: Final[frozenset[Capability]] = frozenset(
    {Capability.png, Capability.ec_level}
)

# What each check requires of the target, from AAP §0.3.4's capability-requirement table. The
# requirements are the whole reason a fourth adapter kind needs no check change, so they are
# asserted per check rather than as a total: a check that lost one requirement would still total
# the same number of capabilities if another gained one, and that pair would go unnoticed.
_EXPECTED_REQUIRES: Final[Mapping[str, frozenset[Capability]]] = MappingProxyType(
    {
        "CL-ARTIFACTS": frozenset(),
        "DM-RESILIENCE": _PNG_ONLY,
        "DM-SEEDED": _PNG_ONLY,
        "DT-DETERMINISTIC": _PNG_ONLY,
        "IV-BAD-OPTION": _PNG_AND_EC_LEVEL,
        "IV-EMPTY": _PNG_ONLY,
        "IV-OVERCAPACITY": frozenset({Capability.png, Capability.version_pin}),
        "MD-CHARSET": _PNG_ONLY,
        "MD-MODE-OPTIMAL": _PNG_ONLY,
        "PF-DECODE": frozenset(),
        "PF-ENCODE": _PNG_ONLY,
        "RN-CONTRAST": frozenset({Capability.png, Capability.color}),
        "RN-LOGO": frozenset(
            {Capability.png, Capability.logo_overlay, Capability.ec_level}
        ),
        "RN-MODULE-SIZE": frozenset({Capability.png, Capability.module_size}),
        "RN-QUIET-ZONE": frozenset({Capability.png, Capability.quiet_zone}),
        "RN-SVG": frozenset({Capability.svg}),
        "RT-ROUNDTRIP": _PNG_ONLY,
        "SY-EC-MATCH": _PNG_AND_EC_LEVEL,
        "SY-MASK": _PNG_ONLY,
        "SY-MATRIX-EQUIV": frozenset(
            {
                Capability.png,
                Capability.ec_level,
                Capability.version_pin,
                Capability.mask_pin,
            }
        ),
        "SY-VERSION-MIN": _PNG_AND_EC_LEVEL,
    }
)

_PRIMARY_DECODER_ONLY: Final[tuple[str, ...]] = (DEPENDENCY_PRIMARY_DECODER,)

# The optional runtime dependencies each check declares, from AAP §0.3.4's requirement table.
# A check declares what it needs, never what provides it -- ``checks/*`` imports no decoder and
# no rasterizer discovery -- so a token is the whole declaration and an unexpected one would be
# a check reaching for a tool through a route the layering test forbids.
_EXPECTED_DEPENDENCIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "CL-ARTIFACTS": _PRIMARY_DECODER_ONLY,
        "DM-RESILIENCE": _PRIMARY_DECODER_ONLY,
        "DM-SEEDED": _PRIMARY_DECODER_ONLY,
        "DT-DETERMINISTIC": _PRIMARY_DECODER_ONLY,
        "IV-BAD-OPTION": _PRIMARY_DECODER_ONLY,
        "IV-EMPTY": _PRIMARY_DECODER_ONLY,
        "IV-OVERCAPACITY": _PRIMARY_DECODER_ONLY,
        "MD-CHARSET": _PRIMARY_DECODER_ONLY,
        "MD-MODE-OPTIMAL": _PRIMARY_DECODER_ONLY,
        "PF-DECODE": _PRIMARY_DECODER_ONLY,
        "PF-ENCODE": _PRIMARY_DECODER_ONLY,
        "RN-CONTRAST": _PRIMARY_DECODER_ONLY,
        "RN-LOGO": _PRIMARY_DECODER_ONLY,
        "RN-MODULE-SIZE": _PRIMARY_DECODER_ONLY,
        "RN-QUIET-ZONE": _PRIMARY_DECODER_ONLY,
        "RN-SVG": (DEPENDENCY_SVG_RASTERIZER,),
        "RT-ROUNDTRIP": _PRIMARY_DECODER_ONLY,
        "SY-EC-MATCH": _PRIMARY_DECODER_ONLY,
        "SY-MASK": _PRIMARY_DECODER_ONLY,
        "SY-MATRIX-EQUIV": _PRIMARY_DECODER_ONLY,
        "SY-VERSION-MIN": _PRIMARY_DECODER_ONLY,
    }
)

# Every capability the port declares, and an availability mapping naming every dependency token
# as present. Together they are the strongest possible declaration a target and a host can make,
# which is what the conforming ``reference`` target makes: a check that still cannot run against
# them has a requirement no target could satisfy (AAP §0.4.1's zero-skip gate).
_EVERY_CAPABILITY: Final[frozenset[Capability]] = frozenset(Capability)
_EVERY_DEPENDENCY_AVAILABLE: Final[Mapping[str, bool]] = MappingProxyType(
    {token: True for token in sorted(DEPENDENCY_TOKENS)}
)

# The nine identifiers a ``--adapter defective`` run must fail, from AAP §0.4.2: the union of the
# five injections' expected sets, which ``qrcheck/registry/expectations.py`` computes rather than
# lists so an injection cannot gain or lose a consequence silently. Written here because
# ``tests/integration/test_defective_run.py`` asserts set equality against it, and a framework
# whose declared expectation drifted with its own implementation would prove nothing.
_EXPECTED_DEFECTIVE_FAILURES: Final[frozenset[str]] = frozenset(
    {
        "DT-DETERMINISTIC",
        "IV-OVERCAPACITY",
        "MD-CHARSET",
        "RN-MODULE-SIZE",
        "RN-QUIET-ZONE",
        "RT-ROUNDTRIP",
        "SY-EC-MATCH",
        "SY-MATRIX-EQUIV",
        "SY-VERSION-MIN",
    }
)

# The five injections of AAP §0.4.2, as the keys ``qrcheck.targets.defective`` names them in its
# own ``DEFECTS`` table. Compared as a set so a renamed injection is a diff here rather than a
# silently empty per-defect set.
_EXPECTED_DEFECT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "wrong_ec_level",
        "truncated_payload",
        "missing_quiet_zone",
        "non_deterministic_mask",
        "over_capacity_acceptance",
    }
)

# The two built-in self-check targets, as ``qrcheck.toml`` declares them under
# ``[targets.<name>]``. The names are the keys of ``SELF_CHECK_TARGETS`` and are never derived
# from a configuration here: this module asks the declaration, not a resolved run.
_CONFORMING_TARGET: Final[str] = "reference"
_DEFECTIVE_TARGET: Final[str] = "defective"

# The run seed these cases record verbatim on a skipped result. Its value is irrelevant to what
# is asserted -- the point is that a skip carries the run's seed rather than a default -- so it
# is the product's own configured default and nothing is read from the clock.
_SEED: Final[int] = 20240101

_ATTEMPTED_ORDER: Final[tuple[str, ...]] = ("rsvg-convert", "resvg")

# A child process that imports the nine family modules in the reverse of ``FAMILY_MODULES`` order
# and prints the registered identifiers. ``importlib`` caches modules, so importing a family
# through either path imports the package first; the reverse loop is what states the order the
# child asks for, and the assertion is that the identifiers it prints are the same tuple a
# forward-importing process holds.
_REVERSE_IMPORT_SCRIPT: Final[str] = (
    "import importlib, json\n"
    "from qrcheck.checks import FAMILY_MODULES\n"
    "for name in reversed(FAMILY_MODULES):\n"
    "    importlib.import_module(f'qrcheck.checks.{name}')\n"
    "from qrcheck.registry.check import registered_ids\n"
    "print(json.dumps(list(registered_ids())))\n"
)

# A bound on the child process, so a hung import fails this module instead of the session. It is
# a guard rather than a performance budget: nothing in AAP §0.6's numeric floors is asserted
# here, and the value is wide enough that a loaded host cannot make it flake.
_SUBPROCESS_TIMEOUT_S: Final[float] = 300.0


# Registration belongs to the entry points, so it is triggered here


def _no_cases(_ctx: object) -> Iterator[object]:
    """Yield nothing: the stub contracts this module registers contribute no case, since
    ``CheckContract`` requires only that ``cases`` be callable."""
    return iter(())


@pytest.fixture(autouse=True)
def _families_loaded() -> tuple[str, ...]:
    """Import the nine check families before every case, as an entry point does:
    ``qrcheck.registry`` stays empty on import, so the executed set cannot drift."""
    return load_families()


# Identity: the twenty-one identifiers, and registration that refuses a duplicate


def test_the_registry_holds_exactly_the_declared_check_ids() -> None:
    """AAP §0.3.4: the registry holds the twenty-one declared identifiers and no others."""
    registered = registered_ids()
    assert set(registered) == _EXPECTED_CHECK_IDS, (
        f"the registry holds {sorted(set(registered) - _EXPECTED_CHECK_IDS)} beyond the "
        f"declared checks and lacks {sorted(_EXPECTED_CHECK_IDS - set(registered))}"
    )
    assert tuple(registered) == CHECK_IDS, "registered_ids() must be the declared tuple"
    assert len(registered) == _EXPECTED_CHECK_COUNT, (
        f"the run executes {len(registered)} checks; the acceptance criteria are written for "
        f"{_EXPECTED_CHECK_COUNT}"
    )
    assert len(REGISTRY) == len(registered), "the published view holds one entry per check"


def test_the_declared_identifier_tuple_is_ascending_and_free_of_duplicates() -> None:
    """A renamed, duplicated or reordered identifier is a change to the acceptance criteria."""
    assert tuple(CHECK_IDS) == tuple(sorted(CHECK_IDS)), "CHECK_IDS must ascend"
    assert len(set(CHECK_IDS)) == len(CHECK_IDS), "CHECK_IDS must hold no duplicate"
    assert len(CHECK_IDS) == _EXPECTED_CHECK_COUNT, "CHECK_IDS must hold every declared check"
    assert set(CHECK_IDS) == _EXPECTED_CHECK_IDS, "CHECK_IDS must be the declared identifiers"


def test_a_duplicate_registration_raises_and_spares_the_real_registry() -> None:
    """A second contract for one identifier must raise rather than shadow the first."""
    real_ids = registered_ids()
    real_count = len(REGISTRY)
    first = CheckContract(
        id="RT-ROUNDTRIP",
        title="stub, first registration",
        family="roundtrip",
        requires=frozenset(),
        dependencies=(),
        cases=_no_cases,
    )
    shadow = CheckContract(
        id="RT-ROUNDTRIP",
        title="stub, shadowing registration",
        family="roundtrip",
        requires=frozenset(),
        dependencies=(),
        cases=_no_cases,
    )

    with isolated_registry() as store:
        assert len(store) == 0, "an isolated registry starts empty"
        assert tuple(REGISTRY) == (), "the published view must show the isolated store"
        register(first)
        assert tuple(REGISTRY) == ("RT-ROUNDTRIP",), "the stub must be the only registration"
        with pytest.raises(CheckRegistrationError) as raised:
            register(shadow)
        assert "RT-ROUNDTRIP" in str(raised.value), (
            "the duplicate diagnostic must name the check it refused"
        )
        assert get_check("RT-ROUNDTRIP") is first, "the registered check must not be replaced"

    assert registered_ids() == real_ids, "the real registry must be restored exactly"
    assert len(REGISTRY) == real_count, "the real registry must hold what it did before"


def test_the_published_registry_cannot_be_written_through() -> None:
    """Only ``register`` may add a check; the view every consumer reads is read-only."""
    with pytest.raises(TypeError):
        REGISTRY["RT-ROUNDTRIP"] = get_check("RT-ROUNDTRIP")


def test_an_unknown_check_id_is_a_named_key_error() -> None:
    """A lookup that misses is a defect to report, never a missing check to skip."""
    with pytest.raises(KeyError) as raised:
        get_check("XX-NOT-A-CHECK")
    assert "XX-NOT-A-CHECK" in str(raised.value), "the diagnostic must name the identifier"


# The contract shape every check carries


@pytest.mark.parametrize("check_id", sorted(_EXPECTED_CHECK_IDS))
def test_a_contract_carries_the_six_fields_the_registry_declares(check_id: str) -> None:
    """``id``, ``title``, ``family``, ``requires``, ``dependencies`` and ``cases``."""
    contract = get_check(check_id)
    assert contract.id == check_id, f"{check_id}: the contract is filed under another id"
    assert isinstance(contract.title, str), f"{check_id}: the title is not a str"
    assert contract.title.strip(), f"{check_id}: the title is empty; qrcheck list prints it"
    assert contract.family in FAMILY_NAMES, f"{check_id}: unknown family {contract.family!r}"
    prefix = check_id.split("-", 1)[0]
    assert FAMILY_BY_PREFIX.get(prefix) == contract.family, (
        f"{check_id}: prefix {prefix!r} belongs to {FAMILY_BY_PREFIX.get(prefix)!r}, not to "
        f"{contract.family!r}"
    )
    assert isinstance(contract.requires, frozenset), (
        f"{check_id}: requires must be a frozenset, got {type(contract.requires).__name__}"
    )
    for member in contract.requires:
        assert isinstance(member, Capability), f"{check_id}: {member!r} is not a Capability"
    assert isinstance(contract.dependencies, tuple), (
        f"{check_id}: dependencies must be a tuple, got {type(contract.dependencies).__name__}"
    )
    unexpected = sorted(token for token in contract.dependencies if token not in DEPENDENCY_TOKENS)
    assert not unexpected, f"{check_id}: unknown dependency tokens {unexpected}"
    assert callable(contract.cases), f"{check_id}: cases is not callable"


# A malformed declaration is refused, never registered as a plausible check

# One field broken per row, everything else valid, so the refusal that fires is the one under
# test: ``id``, ``family``, ``title``, ``cases``, ``requires`` and ``dependencies`` are each
# validated in ``CheckContract`` before a contract can reach the registry, and a check that
# registered with any of them broken would produce a report describing a run it never made.
_MALFORMED_DECLARATIONS: Final[tuple[tuple[Mapping[str, object], str], ...]] = (
    ({"id": "XX-NOT-A-CHECK"}, "XX-NOT-A-CHECK"),
    ({"family": "nonsense"}, "nonsense"),
    ({"family": "cleanup"}, "RT-ROUNDTRIP"),
    ({"title": "   "}, "RT-ROUNDTRIP"),
    ({"cases": "not callable"}, "RT-ROUNDTRIP"),
    ({"requires": 5}, "RT-ROUNDTRIP"),
    ({"requires": {"not-a-capability"}}, "not-a-capability"),
    ({"dependencies": 5}, "RT-ROUNDTRIP"),
    ({"dependencies": ("not-a-token",)}, "not-a-token"),
)


def _valid_declaration() -> dict[str, object]:
    """Return the six fields of a declaration that registers cleanly, before any defect."""
    return {
        "id": "RT-ROUNDTRIP",
        "title": "stub declaration",
        "family": "roundtrip",
        "requires": frozenset(),
        "dependencies": (),
        "cases": _no_cases,
    }


@pytest.mark.parametrize(("defect", "named"), _MALFORMED_DECLARATIONS)
def test_a_malformed_declaration_is_refused_at_construction(
    defect: Mapping[str, object], named: str
) -> None:
    """Each field is validated where the contract is built, and the diagnostic names it."""
    declaration = _valid_declaration()
    declaration.update(defect)
    with pytest.raises(CheckRegistrationError) as raised:
        CheckContract(**declaration)  # type: ignore[arg-type]
    assert named in str(raised.value), (
        f"the refusal must name {named!r} for the defect {defect}, got {raised.value}"
    )


def test_a_contract_normalises_the_two_collection_fields_it_is_given() -> None:
    """A bare name is one requirement, and ``None`` is none of them."""
    declaration = _valid_declaration()
    declaration["requires"] = Capability.png.value
    declaration["dependencies"] = DEPENDENCY_PRIMARY_DECODER
    contract = CheckContract(**declaration)  # type: ignore[arg-type]
    assert contract.requires == frozenset({Capability.png}), (
        "a bare capability name must be one requirement rather than its characters"
    )
    assert contract.dependencies == _PRIMARY_DECODER_ONLY, (
        "a bare dependency token must be one token rather than its characters"
    )

    declaration["requires"] = None
    declaration["dependencies"] = None
    bare = CheckContract(**declaration)  # type: ignore[arg-type]
    assert bare.requires == frozenset(), "None must declare no capability"
    assert bare.dependencies == (), "None must declare no dependency token"


def test_registering_something_that_is_not_a_contract_is_refused() -> None:
    """``register`` is the only writer, and it accepts exactly one type."""
    with pytest.raises(CheckRegistrationError) as raised:
        register("RT-ROUNDTRIP")  # type: ignore[arg-type]
    assert "CheckContract" in str(raised.value), (
        "the refusal must name the type register expects"
    )


# Ordering: import order can never reach an artefact


def test_iteration_is_by_identifier_ascending_everywhere() -> None:
    """AAP §0.3.4, §0.5.3: one order, whichever surface asks."""
    ids = registered_ids()
    assert tuple(ids) == tuple(sorted(ids)), "registered_ids() must ascend"
    assert tuple(contract.id for contract in iter_checks()) == ids, "iter_checks must agree"
    assert tuple(sorted(REGISTRY)) == ids, "the registry view must agree"
    assert ids == tuple(sorted(_EXPECTED_CHECK_IDS)), "the order must be the declared order"


def test_loading_the_families_again_registers_nothing_twice() -> None:
    """The loader is idempotent: a second call re-imports nothing and adds no identifier."""
    before = registered_ids()
    assert load_families() == before, "the first reload changed the registry"
    assert load_families() == before, "the second reload changed the registry"
    assert len(REGISTRY) == len(before), "a reload added a registration"


def test_a_process_importing_the_families_in_reverse_order_holds_the_same_registry() -> None:
    """A child importing the families backwards must print the same identifiers, in order."""
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", _REVERSE_IMPORT_SCRIPT],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    assert completed.returncode == 0, f"the child process failed: {completed.stderr}"
    assert json.loads(completed.stdout) == list(registered_ids()), (
        "a process importing the check families in another order described a different registry"
    )


# The nine family modules and the identifiers each owns


def test_the_nine_family_modules_are_the_ones_the_plan_names() -> None:
    """AAP §0.8.1: nine modules, one per family, and no discovery at run time."""
    assert FAMILY_MODULES == _EXPECTED_FAMILY_MODULES, "the family module order changed"
    assert sorted(FAMILY_MODULES) == sorted(FAMILY_NAMES), (
        "the loader's modules and the registry's family vocabulary must name the same nine"
    )


@pytest.mark.parametrize("family", sorted(_EXPECTED_IDS_BY_FAMILY))
def test_each_family_owns_exactly_the_check_ids_the_plan_assigns_it(family: str) -> None:
    """The one-module-per-family rule, which fixes the JUnit classname and the report section."""
    assert family in FAMILY_MODULES, f"{family}: no family module owns this family"
    owning = frozenset(contract.id for contract in iter_checks() if contract.family == family)
    assert owning == _EXPECTED_IDS_BY_FAMILY[family], (
        f"{family}: owns {sorted(owning)}, the plan assigns "
        f"{sorted(_EXPECTED_IDS_BY_FAMILY[family])}"
    )


def test_the_declared_tables_cover_every_check_identifier() -> None:
    """No check may escape the requirement tables, and no table may name a stranger."""
    assert frozenset(_EXPECTED_REQUIRES) == _EXPECTED_CHECK_IDS, (
        "the capability table must cover exactly the declared checks"
    )
    assert frozenset(_EXPECTED_DEPENDENCIES) == _EXPECTED_CHECK_IDS, (
        "the dependency table must cover exactly the declared checks"
    )
    assert sorted(_EXPECTED_IDS_BY_FAMILY) == sorted(FAMILY_NAMES), (
        "the family mapping must cover exactly the declared families"
    )
    assigned = frozenset().union(*_EXPECTED_IDS_BY_FAMILY.values())
    assert assigned == _EXPECTED_CHECK_IDS, "the family mapping must assign every check"
    assert sum(len(ids) for ids in _EXPECTED_IDS_BY_FAMILY.values()) == _EXPECTED_CHECK_COUNT, (
        "a check is assigned to more than one family"
    )



# What each check requires of a target and of the host


@pytest.mark.parametrize("check_id", sorted(_EXPECTED_CHECK_IDS))
def test_a_check_requires_exactly_the_capabilities_the_plan_declares(check_id: str) -> None:
    """AAP §0.3.4's requirement table: the reason a fourth adapter kind needs no check change."""
    expected = _EXPECTED_REQUIRES[check_id]
    actual = get_check(check_id).requires
    assert actual == expected, (
        f"{check_id}: requires {sorted(str(member) for member in actual)}, the plan declares "
        f"{sorted(str(member) for member in expected)}"
    )


@pytest.mark.parametrize("check_id", sorted(_EXPECTED_CHECK_IDS))
def test_a_check_declares_exactly_the_runtime_dependencies_the_plan_names(check_id: str) -> None:
    """A check declares the tool it needs as a token, never the module that provides it."""
    expected = _EXPECTED_DEPENDENCIES[check_id]
    actual = get_check(check_id).dependencies
    assert actual == expected, (
        f"{check_id}: declares dependencies {actual}, the plan declares {expected}"
    )
    assert tuple(sorted(actual)) == actual, f"{check_id}: the registry must sort the tokens"


def test_the_two_target_independent_checks_require_nothing_of_the_target() -> None:
    """``PF-DECODE`` and ``CL-ARTIFACTS`` run against any target, declaring no capability."""
    for check_id in ("PF-DECODE", "CL-ARTIFACTS"):
        contract = get_check(check_id)
        assert contract.requires == frozenset(), f"{check_id} must require no capability"
        assert unmet_capabilities(contract, capabilities=frozenset()) == (), (
            f"{check_id}: an undeclaring target must leave nothing unmet"
        )
    assert get_check("PF-DECODE").dependencies == _PRIMARY_DECODER_ONLY, (
        "PF-DECODE decodes its own probe image, so it declares the primary decoder "
        "(AAP §0.3.4, 'primary decoder (decode only)')"
    )
    assert get_check("CL-ARTIFACTS").dependencies == _PRIMARY_DECODER_ONLY, (
        "CL-ARTIFACTS declares the primary decoder that AAP §0.3.4's row gives it, and that "
        "token can never skip it: a run without the decoder exits 2 before resolution"
    )


def test_a_target_declaring_everything_satisfies_every_check() -> None:
    """The strongest declaration a conforming target makes leaves no check unrunnable."""
    resolution = resolve(
        iter_checks(),
        capabilities=_EVERY_CAPABILITY,
        availability=_EVERY_DEPENDENCY_AVAILABLE,
    )
    assert resolution.skipped == (), f"unexpected skips: {resolution.skipped}"
    assert resolution.is_complete, "a full declaration must resolve completely"
    assert resolution.runnable_ids == CHECK_IDS, "every check must be runnable"


# Resolution: a shortfall is a skip, and a skip is never a pass


def test_a_capability_shortfall_resolves_to_a_skip_and_never_to_a_pass() -> None:
    """AAP §0.3.4, §0.6.6 row 9: a shortfall is a skip, tallied separately and never a pass."""
    check_id = "RN-QUIET-ZONE"
    contract = get_check(check_id)
    missing = Capability.quiet_zone
    assert missing in contract.requires, "the check under test must require what is withheld"

    # A valid configuration against a reachable target whose declaration is short: this must not
    # raise, because a shortfall is a result to report rather than an exception to escape.
    resolution = resolve(
        [contract],
        capabilities={Capability.png},
        availability=_EVERY_DEPENDENCY_AVAILABLE,
    )
    assert isinstance(resolution, Resolution), "resolve must answer a Resolution"
    assert resolution.skipped_ids == (check_id,), "the short check must be the only skip"
    assert resolution.runnable_ids == (), "a check short of a requirement must not run"
    assert not resolution.is_complete, "a resolution holding a skip is not complete"

    outcome = resolution.skipped[0]
    assert isinstance(outcome, SkipOutcome), "each skip must be a SkipOutcome"
    assert outcome.check_id == check_id
    assert outcome.reason is Reason.UNSUPPORTED_CAPABILITY, (
        f"{check_id}: an undeclared capability must skip with unsupported_capability"
    )
    assert outcome.missing == (missing.value,), "the skip must name the capability it lacks"

    results = skip_case_results(resolution, seed=_SEED)
    assert len(results) == 1, "one skip must become exactly one result"
    result = results[0]
    assert isinstance(result, CaseResult), "each skip must become a CaseResult"
    assert result.check_id == check_id
    assert result.status is Status.SKIPPED, f"{check_id}: a skip must not carry another status"
    assert result.reason is Reason.UNSUPPORTED_CAPABILITY
    assert result.case_id == CAPABILITY_CASE_ID == "synthetic:capability", (
        "a pre-execution skip has the one case identifier the plan fixes for it"
    )
    assert result.evidence_dir is None, (
        "this function is pure; the runner's evidence store attaches the directory (AAP §0.3.4)"
    )
    assert result.request == {} and result.decoded == {}, (
        "a capability shortfall carries no request and no decode evidence"
    )
    assert result.expected == {"missing": [missing.value]}, "the cause must reach the result"
    assert result.seed == _SEED, "a skip must carry the run's seed"

    counts = Counts.from_results(results)
    assert counts.skipped == 1, "the skip must be tallied in its own integer"
    assert counts.passed == 0, "a skipped result must never be counted as a pass"
    assert counts.failed == 0, "a skip is not a failure of the target"
    assert counts.error == 0, "a skip is not an internal error"
    assert counts.total == len(results), "the four tallies must describe exactly the results"


def test_a_missing_runtime_dependency_resolves_to_a_missing_dependency_skip() -> None:
    """AAP §0.3.4: an absent optional tool skips its check with ``missing_dependency``."""
    check_id = "RN-SVG"
    contract = get_check(check_id)
    unavailable = {
        token: token != DEPENDENCY_SVG_RASTERIZER for token in sorted(DEPENDENCY_TOKENS)
    }
    assert unmet_dependencies(contract, unavailable) == (DEPENDENCY_SVG_RASTERIZER,), (
        f"{check_id}: the absent token must be the only one reported missing"
    )
    assert unmet_dependencies(contract, _EVERY_DEPENDENCY_AVAILABLE) == (), (
        f"{check_id}: a fully available host leaves nothing unmet"
    )

    resolution = resolve([contract], capabilities={Capability.svg}, availability=unavailable)
    assert resolution.runnable_ids == (), "a check missing its tool must not run"
    outcome = resolution.skipped[0]
    assert outcome.reason is Reason.MISSING_DEPENDENCY, (
        f"{check_id}: an absent tool must skip with missing_dependency"
    )
    assert outcome.missing == (DEPENDENCY_SVG_RASTERIZER,)

    counts = Counts.from_results(skip_case_results(resolution, seed=_SEED))
    assert counts.skipped == 1
    assert counts.passed == 0, "a skipped result must never be counted as a pass"


def test_a_missing_dependency_skip_records_the_discovery_order_it_attempted() -> None:
    contract = get_check("RN-SVG")
    unavailable = {
        token: token != DEPENDENCY_SVG_RASTERIZER for token in sorted(DEPENDENCY_TOKENS)
    }

    resolution = resolve(
        [contract],
        capabilities={Capability.svg},
        availability=unavailable,
        attempted={DEPENDENCY_SVG_RASTERIZER: _ATTEMPTED_ORDER},
    )
    outcome = resolution.skipped[0]
    assert outcome.reason is Reason.MISSING_DEPENDENCY
    assert outcome.attempted == _ATTEMPTED_ORDER, "the order is carried as it was walked"

    result = skip_case_results(resolution, seed=_SEED)[0]
    assert result.decoded == {"attempted": list(_ATTEMPTED_ORDER)}, result.decoded
    assert result.expected == {"missing": [DEPENDENCY_SVG_RASTERIZER]}, result.expected


def test_a_capability_shortfall_records_no_discovery_order() -> None:
    contract = get_check("RN-SVG")
    nothing_available = {token: False for token in sorted(DEPENDENCY_TOKENS)}

    resolution = resolve(
        [contract],
        capabilities=frozenset(),
        availability=nothing_available,
        attempted={DEPENDENCY_SVG_RASTERIZER: _ATTEMPTED_ORDER},
    )
    outcome = resolution.skipped[0]
    assert outcome.reason is Reason.UNSUPPORTED_CAPABILITY
    assert outcome.attempted == (), "installing the rasterizer would not make the target runnable"
    assert skip_case_results(resolution, seed=_SEED)[0].decoded == {}, (
        "a capability shortfall carries no decode evidence, so its decoded stays empty"
    )


def test_an_attempted_order_must_name_declared_tokens_and_candidate_names() -> None:
    contract = get_check("RN-SVG")
    unavailable = {
        token: token != DEPENDENCY_SVG_RASTERIZER for token in sorted(DEPENDENCY_TOKENS)
    }
    unknown_token = "svg-rasteriser"
    with pytest.raises(ValueError) as unnamed:
        resolve(
            [contract], capabilities={Capability.svg}, availability=unavailable,
            attempted={unknown_token: _ATTEMPTED_ORDER},
        )
    assert unknown_token in str(unnamed.value), f"the refusal must name {unknown_token!r}"

    with pytest.raises(ValueError) as not_a_name:
        resolve(
            [contract], capabilities={Capability.svg}, availability=unavailable,
            attempted={DEPENDENCY_SVG_RASTERIZER: (1,)},  # type: ignore[dict-item]
        )
    assert DEPENDENCY_SVG_RASTERIZER in str(not_a_name.value), not_a_name.value


def test_a_capability_shortfall_outranks_a_missing_runtime_dependency() -> None:
    """One skip per check, carrying the reason that would survive installing the tool."""
    contract = get_check("RN-SVG")
    nothing_available = {token: False for token in sorted(DEPENDENCY_TOKENS)}
    resolution = resolve([contract], capabilities=frozenset(), availability=nothing_available)
    assert len(resolution.skipped) == 1, "a check is skipped once, never once per cause"
    outcome = resolution.skipped[0]
    assert outcome.reason is Reason.UNSUPPORTED_CAPABILITY, (
        "installing the rasterizer would not make an SVG-incapable target runnable"
    )
    assert outcome.missing == (Capability.svg.value,)


def test_a_target_declaring_nothing_leaves_only_the_capability_free_checks_runnable() -> None:
    """No check can be silently passed: without a capability only the capability-free pair runs."""
    resolution = resolve(
        iter_checks(), capabilities=frozenset(), availability=_EVERY_DEPENDENCY_AVAILABLE
    )
    assert resolution.runnable_ids == ("CL-ARTIFACTS", "PF-DECODE"), (
        "the two checks declaring no capability are all a capability-free target can run"
    )
    assert len(resolution.skipped) == _EXPECTED_CHECK_COUNT - 2, (
        "every check that declares a capability must be skipped"
    )

    starved = resolve(iter_checks(), capabilities=frozenset(), availability={})
    assert starved.runnable_ids == (), (
        "with no probe answering a token, even the capability-free pair must skip"
    )
    reasons = {outcome.check_id: outcome.reason for outcome in starved.skipped}
    assert len(reasons) == _EXPECTED_CHECK_COUNT, "every check must be skipped"
    assert reasons["RT-ROUNDTRIP"] is Reason.UNSUPPORTED_CAPABILITY, "RT-ROUNDTRIP needs png"
    assert reasons["PF-DECODE"] is Reason.MISSING_DEPENDENCY, (
        "PF-DECODE requires no capability, so its only possible shortfall is the decoder"
    )
    assert reasons["CL-ARTIFACTS"] is Reason.MISSING_DEPENDENCY, (
        "CL-ARTIFACTS declares the primary decoder as its optional dependency (AAP §0.3.4)"
    )
    for check_id, reason in sorted(reasons.items()):
        assert reason in {Reason.UNSUPPORTED_CAPABILITY, Reason.MISSING_DEPENDENCY}, (
            f"{check_id}: a resolution may only skip with a capability or dependency reason"
        )


# A malformed resolution argument is named, never turned into a skip

# One field broken per row. A skip that names no cause, carries a reason no resolution may
# produce, lists no missing requirement or reports a non-collection order is refused here.
_MALFORMED_SKIPS: Final[tuple[tuple[Mapping[str, object], str], ...]] = (
    ({"check_id": ""}, "check_id"),
    ({"reason": "nonsense"}, "not a Reason"),
    ({"reason": Reason.VALUE_MISMATCH}, Reason.VALUE_MISMATCH.value),
    ({"missing": ()}, "names no missing"),
    ({"missing": DEPENDENCY_SVG_RASTERIZER}, "collection"),
    ({"attempted": DEPENDENCY_SVG_RASTERIZER}, "collection of candidate names"),
)


@pytest.mark.parametrize(("defect", "named"), _MALFORMED_SKIPS)
def test_a_skip_must_name_its_cause_and_the_requirements_it_lacks(
    defect: Mapping[str, object], named: str
) -> None:
    """An unexplained or mis-flavoured skip is refused, naming the field that carries it."""
    fields: dict[str, object] = {
        "check_id": "RN-SVG",
        "reason": Reason.MISSING_DEPENDENCY,
        "missing": (DEPENDENCY_SVG_RASTERIZER,),
    }
    fields.update(defect)
    with pytest.raises(ValueError) as raised:
        SkipOutcome(**fields)  # type: ignore[arg-type]
    assert named in str(raised.value), (
        f"the refusal must name {named!r} for the defect {defect}, got {raised.value}"
    )


def test_a_resolution_may_not_decide_one_check_twice() -> None:
    """A doubled decision would execute a check once while a report counted it twice."""
    contract = get_check("CL-ARTIFACTS")
    with pytest.raises(ValueError) as raised:
        Resolution(runnable=(contract, contract))
    assert "CL-ARTIFACTS" in str(raised.value), "the refusal must name the doubled check"


def test_resolve_reads_a_declaring_target_and_an_availability_probe() -> None:
    """The runner passes an adapter and a probe; both accepted forms must resolve the same."""
    contract = get_check("RN-SVG")

    class _DeclaringTarget:
        """An object shaped like an adapter: it publishes its capabilities."""

        capabilities = frozenset({Capability.svg.value})

    class _ProjectingProbe:
        """An availability source that projects itself to a mapping."""

        def as_mapping(self) -> Mapping[str, bool]:
            """Return every dependency token as available."""
            return _EVERY_DEPENDENCY_AVAILABLE

    class _QueryingProbe:
        """An availability source that answers one token at a time."""

        def available(self, dependency: str) -> bool:
            """Return whether `dependency` is available for this run."""
            return True

    for probe in (_ProjectingProbe(), _QueryingProbe()):
        resolution = resolve([contract], capabilities=_DeclaringTarget(), availability=probe)
        assert resolution.runnable_ids == ("RN-SVG",), (
            f"{type(probe).__name__}: an available rasterizer must leave the check runnable"
        )
        assert resolution.is_complete


def test_resolve_accepts_the_registry_mapping_as_a_selection() -> None:
    """A mapping selection is accepted, so a caller may pass the registry itself."""
    resolution = resolve(
        REGISTRY,
        capabilities=_EVERY_CAPABILITY,
        availability=_EVERY_DEPENDENCY_AVAILABLE,
    )
    assert resolution.runnable_ids == CHECK_IDS, "the registry mapping must resolve in order"


def test_resolve_collapses_a_repeated_selection() -> None:
    """A check named twice is decided once, so a report cannot carry it twice."""
    contract = get_check("RT-ROUNDTRIP")
    resolution = resolve(
        [contract, contract],
        capabilities=_EVERY_CAPABILITY,
        availability=_EVERY_DEPENDENCY_AVAILABLE,
    )
    assert resolution.runnable_ids == ("RT-ROUNDTRIP",)
    assert len(resolution.runnable) == 1, "a repeated selection must resolve one check once"


def test_resolve_names_a_malformed_argument_rather_than_skipping_a_check() -> None:
    """A caller defect is raised, not absorbed: a skip would describe the caller as a target."""
    contract = get_check("CL-ARTIFACTS")
    with pytest.raises(TypeError):
        unmet_capabilities(contract, capabilities=Capability.png.value)
    with pytest.raises(TypeError):
        unmet_capabilities(contract, capabilities=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        unmet_capabilities(contract, capabilities={"not-a-capability"})
    with pytest.raises(TypeError):
        unmet_dependencies(get_check("RN-SVG"), availability=5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve([object()])  # type: ignore[list-item]


def test_an_availability_source_must_project_a_mapping() -> None:
    """A probe answering with something that is not a mapping is a defect, not silence."""
    contract = get_check("RN-SVG")

    class _BrokenProbe:
        """An availability source whose projection is not a mapping."""

        def as_mapping(self) -> object:
            """Return something no consumer can interrogate per token."""
            return 5

    with pytest.raises(TypeError) as raised:
        unmet_dependencies(contract, availability=_BrokenProbe())
    assert "mapping" in str(raised.value), "the refusal must say what it expected"


# Expectations: what each built-in self-check target must produce


def test_the_conforming_target_declares_no_expected_failure() -> None:
    """AAP §0.4.1: every check must pass on ``reference``, and the set says so."""
    assert SELF_CHECK_TARGETS[_CONFORMING_TARGET] == frozenset(), (
        "the conforming target must declare no expected failure"
    )
    assert expected_failures(_CONFORMING_TARGET) == frozenset()
    assert expected_passing(_CONFORMING_TARGET) == _EXPECTED_CHECK_IDS, (
        "every declared check must pass on the conforming target"
    )


def test_the_defective_target_declares_exactly_the_nine_expected_failures() -> None:
    """AAP §0.4.2: the union of the five injections' expected sets, and nothing else."""
    declared = SELF_CHECK_TARGETS[_DEFECTIVE_TARGET]
    assert declared == _EXPECTED_DEFECTIVE_FAILURES, (
        f"declared {sorted(declared)}, the plan declares {sorted(_EXPECTED_DEFECTIVE_FAILURES)}"
    )
    assert len(declared) == 9, "the five injections of AAP §0.4.2 must fail nine checks"
    assert expected_failures(_DEFECTIVE_TARGET) == declared
    assert frozenset().union(*DEFECT_EXPECTATIONS.values()) == declared, (
        "the per-defect sets must derive the declared failing set"
    )
    assert frozenset(DEFECT_EXPECTATIONS) == _EXPECTED_DEFECT_NAMES, (
        f"declared injections {sorted(DEFECT_EXPECTATIONS)}, the plan names five"
    )


def test_the_twelve_checks_outside_the_expected_set_must_pass() -> None:
    """AAP §0.4.2: the complement must pass in the same run that fails the other nine."""
    passing = expected_passing(_DEFECTIVE_TARGET)
    assert passing == _EXPECTED_CHECK_IDS - _EXPECTED_DEFECTIVE_FAILURES, (
        f"expected to pass {sorted(passing)}, the plan declares "
        f"{sorted(_EXPECTED_CHECK_IDS - _EXPECTED_DEFECTIVE_FAILURES)}"
    )
    assert passing & _EXPECTED_DEFECTIVE_FAILURES == frozenset(), (
        "a check cannot be expected to fail and pass in one run"
    )
    assert passing | _EXPECTED_DEFECTIVE_FAILURES == _EXPECTED_CHECK_IDS, (
        "no check may escape the expectation"
    )
    assert len(passing) == 12, "the other twelve checks of AAP §0.4.2 must pass"


def test_the_declared_expectations_hold_against_the_real_registry() -> None:
    """Every expected identifier is a declared check and is registered, so the gate compares."""
    validate_against_registry()
    validate_against_registry(REGISTRY)


def test_an_expectation_for_an_unregistered_check_is_rejected() -> None:
    """Deleting a check must break a test rather than silently shrink what is compared."""
    fabricated = CheckContract(
        id="SY-EC-MATCH",
        title="fabricated expectation",
        family="symbol",
        requires=frozenset(),
        dependencies=(),
        cases=_no_cases,
    )
    partial = {
        check_id: contract
        for check_id, contract in REGISTRY.items()
        if check_id != fabricated.id
    }
    assert fabricated.id not in partial, "the fabricated identifier must be absent"
    assert get_check(fabricated.id) is REGISTRY[fabricated.id], "the real check is registered"

    with pytest.raises(ExpectationError) as raised:
        validate_against_registry(partial)
    assert fabricated.id in str(raised.value), (
        f"{fabricated.id} is expected to fail on the defective target, so a registry without "
        f"it must be rejected by name"
    )
    assert registered_ids() == CHECK_IDS, "the real registry must be left untouched"


def test_a_doctored_expectation_declaration_is_rejected_at_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard names each inconsistency, so a doctored declaration cannot pass unnoticed:
    ``SELF_CHECK_TARGETS`` is doctored on its owning module, the table being read-only."""
    doctored: tuple[tuple[Mapping[str, frozenset[str]], str], ...] = (
        ({" ": frozenset()}, "blank"),
        (
            {_CONFORMING_TARGET: frozenset(), _DEFECTIVE_TARGET: frozenset({"XX-NOT-A-CHECK"})},
            "XX-NOT-A-CHECK",
        ),
        (
            {_CONFORMING_TARGET: frozenset(), _DEFECTIVE_TARGET: frozenset()},
            "per-defect sets derive",
        ),
    )
    for declaration, named in doctored:
        monkeypatch.setattr(_expectations_module, "SELF_CHECK_TARGETS", declaration)
        with pytest.raises(ExpectationError) as raised:
            validate_against_registry(REGISTRY)
        assert named in str(raised.value), (
            f"the refusal must name {named!r} for the declaration {declaration}, "
            f"got {raised.value}"
        )


def test_a_target_with_no_declared_expectation_answers_none() -> None:
    """A user-supplied generator has no declared expectation, and an empty set is not one."""
    assert expected_failures("acme-library") is None, (
        "an empty set would assert that an unknown target must be clean"
    )
    assert expected_passing("acme-library") == _EXPECTED_CHECK_IDS, (
        "nothing is excused for a target with no declared expectation"
    )


# Reachability: the machine-readable surface criterion 8 reads


def test_the_list_command_publishes_every_registered_check_as_json(cli_runner) -> None:
    """``qrcheck list --json`` is the cheap half of the registry-reachability gate: a check the
    listing omits is one no CI consumer can trend."""
    result = cli_runner.invoke(cli, ["list", "--json"])
    assert result.exit_code == 0, f"qrcheck list --json exited {result.exit_code}: {result.stderr}"
    rows = json.loads(result.stdout)
    assert isinstance(rows, list), "the --json form must be an object array"
    assert [row["id"] for row in rows] == list(CHECK_IDS), (
        "the listing must enumerate every registered check, ascending"
    )
    for row in rows:
        check_id = row["id"]
        assert set(row) == {"id", "title", "family", "requires", "dependencies"}, (
            f"{check_id}: the --json object carries exactly the five declared keys"
        )
        assert row["family"] == get_check(check_id).family, f"{check_id}: wrong family"
        assert row["requires"] == sorted(row["requires"]), f"{check_id}: requires must be sorted"
        assert row["dependencies"] == sorted(row["dependencies"]), (
            f"{check_id}: dependencies must be sorted"
        )
        assert set(row["requires"]) == {
            member.value for member in _EXPECTED_REQUIRES[check_id]
        }, f"{check_id}: the listing must publish the plan's requirements"
        assert tuple(row["dependencies"]) == _EXPECTED_DEPENDENCIES[check_id], (
            f"{check_id}: the listing must publish the plan's dependency tokens"
        )

