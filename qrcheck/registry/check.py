"""The single owner of check identity, ordering and the context a check runs in.

This module is where a check *becomes* a check.  Each of the twenty-one checks in
the product registers itself here through :func:`check`, and every consumer that
has to name, order, select or run a check reads the resulting :data:`REGISTRY`
rather than keeping a list of its own.  Four things therefore exist exactly once
in qrcheck, and all four live here:

* the twenty-one check identifiers (:data:`CHECK_IDS`) -- the only literal list of
  them in the product -- and the nine check families with the identifier prefix
  that owns each (:data:`FAMILY_BY_PREFIX`);
* the registry itself (:data:`REGISTRY`), written only while the check family
  modules import, read by ``qrcheck list``, the runner, the JUnit writer and the
  expected-failure table;
* :class:`RunContext`, the immutable bundle of everything a check may know about
  the run it is part of; and
* :class:`Case`, the unit the runner walks: one stable identifier and one
  callable that produces that case's :class:`~qrcheck.report.model.CaseResult`.

Ordering is part of the contract
--------------------------------
Checks are iterated **by identifier ascending** everywhere: in ``qrcheck list``, in
a run's execution order and in every report section.  Import order must never
reach an artefact, so :func:`iter_checks` is the only iteration a consumer may use
and it sorts; nothing here depends on the insertion order of the registry.

Registration is never silent about a defect
-------------------------------------------
An identifier outside :data:`CHECK_IDS`, an unknown family, an identifier whose
prefix belongs to another family, an unknown dependency token, an empty title and
a duplicate identifier all raise :class:`CheckRegistrationError` immediately.  A
duplicate especially: replacing or shadowing a registered check would change the
executed set under a report that already claims to describe it, and
``tests/unit/test_registry_contract.py`` asserts the registry holds exactly the
twenty-one declared identifiers.

Layering
--------
The module imports one name from ``qrcheck.targets.base`` -- :class:`Capability` --
and nothing else from the product at run time.  It must **never** import
``qrcheck.checks``: the check family modules import *this* module in order to
register themselves, so such an import would close the registration cycle and
leave the registry half-populated at whatever point the cycle was cut.  The four
records a :class:`RunContext` carries (:class:`~qrcheck.config.models.Config`,
:class:`~qrcheck.corpus.builder.CorpusRecord`,
:class:`~qrcheck.report.evidence.EvidenceStore`,
:class:`~qrcheck.report.model.CaseResult`) and the
:class:`~qrcheck.targets.base.Adapter` port are imported for type checking only,
so importing this module pulls no image library, decoder, HTTP client, CLI
framework or encoder into a process that only wants to list checks.

A second ``REGISTRY``, ``CheckContract``, ``RunContext`` or ``Case`` anywhere in
the product is a defect: two registries would let ``qrcheck list`` and a run
disagree about which checks exist, and two contexts would let a check read a
threshold from an owner the configuration does not have.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from qrcheck.targets.base import Capability

if TYPE_CHECKING:  # Imported for type checking only; see the module docstring.
    from qrcheck.config.models import Config
    from qrcheck.corpus.builder import CorpusRecord
    from qrcheck.report.evidence import EvidenceStore
    from qrcheck.report.model import CaseResult
    from qrcheck.targets.base import Adapter

__all__ = [
    "CHECK_IDS",
    "Case",
    "CasesFn",
    "CheckContract",
    "CheckDecorator",
    "CheckRegistrationError",
    "DEPENDENCY_PRIMARY_DECODER",
    "DEPENDENCY_SECONDARY_DECODER",
    "DEPENDENCY_SVG_RASTERIZER",
    "DEPENDENCY_TOKENS",
    "FAMILY_BY_PREFIX",
    "FAMILY_NAMES",
    "REGISTRY",
    "RunContext",
    "check",
    "get_check",
    "isolated_registry",
    "iter_checks",
    "register",
    "registered_ids",
]

# The twenty-one registered check identifiers, in ascending ASCII order.
#
# This tuple is the only literal list of check identifiers in the product: the
# registry validates against it, ``qrcheck list`` and the reports print what the
# registry holds, and the reachability gate compares the two sets.  Keeping it
# ascending means a human diff shows an added or removed check where it belongs,
# and the order agreeing with ``sorted(CHECK_IDS)`` is what lets :func:`iter_checks`
# promise a stable iteration without a second sort key.  An identifier is never
# renamed, split or merged: CI trend lines and the expected-failure table key on
# these exact strings.
CHECK_IDS: Final[tuple[str, ...]] = (
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
)

# The nine check families, one per module under ``qrcheck/checks/``, in ascending
# order.  A family name is also the JUnit ``classname`` suffix
# (``classname="qrcheck.<family>"``), so it is a stable output surface rather than
# an internal label.
FAMILY_NAMES: Final[tuple[str, ...]] = (
    "cleanup",
    "damage",
    "determinism",
    "invalid",
    "mode",
    "performance",
    "render",
    "roundtrip",
    "symbol",
)

# Which family owns which identifier prefix.  Registration compares an identifier's
# prefix against its declared family, which is what holds the one-module-per-family
# rule mechanically: a check filed under the wrong family would otherwise land in
# the wrong JUnit ``classname`` and in the wrong report section.
FAMILY_BY_PREFIX: Final[Mapping[str, str]] = MappingProxyType(
    {
        "CL": "cleanup",
        "DM": "damage",
        "DT": "determinism",
        "IV": "invalid",
        "MD": "mode",
        "PF": "performance",
        "RN": "render",
        "RT": "roundtrip",
        "SY": "symbol",
    }
)

# The runtime dependencies a check may declare, as tokens rather than module names.
#
# A check declares *what it needs*, never *what provides it*: ``checks/*`` may not
# import a decoder or the rasterizer's discovery logic, and the availability of each
# token is resolved once per run by the runner and carried in
# :attr:`RunContext.availability`.
#
# A missing primary decoder (zxing-cpp) blocks the run: ``qrcheck doctor`` and
# ``qrcheck run`` exit 2, naming zxing-cpp and the interpreter/platform, before any
# check runs, and ``RT-ROUNDTRIP`` is never silently skipped (AAP §0.3.5, §0.6.6
# row 3, §0.7.2 row 1). The check still declares
# :data:`DEPENDENCY_PRIMARY_DECODER`, because that is the honest record of what it
# needs -- notably ``PF-DECODE`` and ``CL-ARTIFACTS``, which decode but need nothing
# from the target -- and resolution can never turn that record into a skip, because a
# run without the primary decoder never reaches resolution.
#
# Only two tools degrade. The secondary decoder (opencv) leaves the run in
# single-decoder mode, reported as ``primary_only``. The SVG rasterizer makes
# ``RN-SVG`` report ``skipped`` with reason ``missing_dependency`` and the discovery
# order it attempted.
#
# The vocabulary is closed at three, one per runtime tool whose availability a check
# can depend on. There is deliberately no token for the platform file-descriptor
# directory: a declared token can only produce a whole-check ``missing_dependency``
# skip, whereas ``CL-ARTIFACTS`` reports its handle half per case as
# ``unsupported_platform`` while its temporary-file half still asserts
# (AAP §0.8.1).
DEPENDENCY_PRIMARY_DECODER: Final[str] = "primary-decoder"
DEPENDENCY_SECONDARY_DECODER: Final[str] = "secondary-decoder"
DEPENDENCY_SVG_RASTERIZER: Final[str] = "svg-rasterizer"

DEPENDENCY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        DEPENDENCY_PRIMARY_DECODER,
        DEPENDENCY_SECONDARY_DECODER,
        DEPENDENCY_SVG_RASTERIZER,
    }
)


class CheckRegistrationError(ValueError):
    """A check cannot be registered as declared.

    Raised for a duplicate identifier, an identifier outside :data:`CHECK_IDS`, an
    unknown family, an identifier whose prefix belongs to a different family, an
    unknown dependency token, an empty title and a non-callable ``cases``.  The
    message always names the offending value, because these are programming errors
    in a check module rather than user errors in ``qrcheck.toml``, and the message is
    the whole diagnostic a reader of a failed import gets.

    It subclasses :class:`ValueError` so a caller that already catches the builtin
    -- a test that registers a deliberately invalid contract, the CLI turning a
    broken installation into exit code 3 -- needs no import of this module to handle
    it.
    """


@dataclass(frozen=True, slots=True)
class Case:
    """One case a check contributes to a run: an identifier and a callable.

    Exactly two fields, deliberately.  ``case_id`` is the stable name the case is
    filed under -- ``corpus:<record-id>`` for a corpus-driven case,
    ``synthetic:<name>`` for one the framework constructs -- and it is the case
    component of the failure identifier :func:`qrcheck.report.model.failure_id`
    derives, hence part of the leaf of
    ``<report_dir>/evidence/<check-id>/<failure-id>/``, never the leaf itself: two
    checks share a case identifier such as ``corpus:url-000`` and the ``<check-id>``
    segment above the leaf is what separates them.  ``execute`` produces that case's
    result when the runner calls it with the run's :class:`RunContext`.

    The callable is never invoked by this module or by a check's ``cases``
    generator: a check yields every case up front and the runner decides which of
    them run, which is what makes ``--fail-fast`` and the one-error-per-check policy
    on target loss the runner's decisions rather than a check's.

    Raises:
        ValueError: ``case_id`` is not a non-empty string, or ``execute`` is not
            callable.
    """

    case_id: str
    execute: Callable[[RunContext], CaseResult]

    def __post_init__(self) -> None:
        """Reject an unusable case, naming the field that carries the defect.

        Raises:
            ValueError: ``case_id`` is not a ``str`` or holds no visible character,
                or ``execute`` is not callable.
        """
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError(
                f"Case.case_id must be a non-empty str, got {self.case_id!r}"
            )
        if not callable(self.execute):
            raise ValueError(
                f"Case {self.case_id!r}: execute must be callable, got {self.execute!r}"
            )


@dataclass(frozen=True)
class RunContext:
    """Everything a check may know about the run it is part of, and nothing else.

    One instance per run is passed to each check's ``cases(ctx)`` and then to each
    returned :class:`Case`, so a check sees the same view of the run whether it is
    enumerating cases or executing one.  Fields are deliberately raw: the adapter is
    the port itself (a check sees no concrete adapter and cannot import one), the
    configuration is the resolved record that owns every threshold, and the corpus
    is already filtered and sorted by the runner -- the registry filters nothing, so
    ``--class`` cannot behave differently for two checks.

    ``availability`` maps the :data:`DEPENDENCY_*` tokens to whether that dependency
    was found for this run, and an **absent key means unavailable**.  The default is
    the safe direction on purpose: a forgotten probe makes a check that depends on a
    degrading dependency report ``skipped`` with reason ``missing_dependency``, which
    the zero-skip reference gate fails loudly, instead of letting the check run against
    something that was never there and count as a pass.  A missing primary decoder
    never reaches this mapping at all: ``qrcheck doctor`` and ``qrcheck run`` exit 2
    naming zxing-cpp first, so that token is a declaration of what a check needs
    rather than a skip path.

    The instance is frozen: a check cannot change what another check sees.
    """

    adapter: Adapter
    config: Config
    corpus: tuple[CorpusRecord, ...]
    ec_level: str
    seed: int
    availability: Mapping[str, bool]
    evidence: EvidenceStore

    def dependency_available(self, token: str) -> bool:
        """Return whether `token` was resolved as available for this run.

        An absent token is unavailable, never available by omission.

        Args:
            token: One of the :data:`DEPENDENCY_TOKENS` values.

        Returns:
            ``True`` only when the runner recorded the dependency as present.
        """
        return bool(self.availability.get(token, False))


# The shape of a check's ``cases`` generator: given the run's context, yield the
# cases this check contributes. Declared once so the decorator, the contract field
# and every family module's signature agree on it.
CasesFn: TypeAlias = Callable[[RunContext], Iterator[Case]]


def _coerce_requires(value: object, check_id: str) -> frozenset[Capability]:
    """Return `value` as a ``frozenset`` of :class:`Capability` members.

    The declared requirements may be passed as members, as their string values, or
    as a single member rather than a collection of them; every form resolves to the
    same set, so a check module cannot register a requirement that only looks
    satisfied. An unknown name is a registration error naming the name and the
    closed vocabulary.

    Args:
        value: The declared ``requires``, in any of the accepted forms.
        check_id: The identifier being registered, quoted in the diagnostic.

    Returns:
        The requirements as members, empty when nothing was declared.

    Raises:
        CheckRegistrationError: A name is not a :class:`Capability` member, or
            `value` is not a collection of them.
    """
    if value is None:
        return frozenset()
    # ``Capability`` is a ``StrEnum``, so a member is also a ``str``: a bare name and
    # a bare member are both one requirement rather than an iterable of characters.
    declared: Iterable[object]
    if isinstance(value, str):
        declared = (value,)
    elif isinstance(value, Iterable):
        declared = value
    else:
        raise CheckRegistrationError(
            f"check {check_id!r} requires {value!r}, which is not a collection of "
            f"capabilities"
        )
    members: set[Capability] = set()
    for item in declared:
        try:
            members.add(Capability(item))
        except (TypeError, ValueError):
            vocabulary = tuple(member.value for member in Capability)
            raise CheckRegistrationError(
                f"check {check_id!r} requires unknown capability {item!r}; the vocabulary is "
                f"{vocabulary}"
            ) from None
    return frozenset(members)


def _normalise_dependencies(value: object, check_id: str) -> tuple[str, ...]:
    """Return `value` as a sorted, duplicate-free tuple of dependency tokens.

    Sorting and de-duplicating here is what makes two declarations of the same
    requirement produce the same ``summary.json`` and the same ``qrcheck list`` line
    regardless of how the check module wrote them.

    Args:
        value: The declared ``dependencies``, as a token or a collection of tokens.
        check_id: The identifier being registered, quoted in the diagnostic.

    Returns:
        The tokens in ascending order, empty when nothing was declared.

    Raises:
        CheckRegistrationError: A token is outside :data:`DEPENDENCY_TOKENS`, or
            `value` is not a collection of them.
    """
    if value is None:
        return ()
    declared: Iterable[object]
    if isinstance(value, str):
        declared = (value,)
    elif isinstance(value, Iterable):
        declared = value
    else:
        raise CheckRegistrationError(
            f"check {check_id!r} declares dependencies {value!r}, which is not a "
            f"collection of dependency tokens"
        )
    tokens: set[str] = set()
    for item in declared:
        if not isinstance(item, str) or item not in DEPENDENCY_TOKENS:
            raise CheckRegistrationError(
                f"check {check_id!r} declares unknown dependency token {item!r}; the "
                f"vocabulary is {sorted(DEPENDENCY_TOKENS)}"
            )
        tokens.add(item)
    return tuple(sorted(tokens))


@dataclass(frozen=True)
class CheckContract:
    """One registered check: its identity, what it needs, and the cases it contributes.

    A check module declares one of these per check through :func:`check`, and the
    registry keeps them keyed by :attr:`id`. Six fields, each a stable public
    surface:

    * ``id`` -- one of the twenty-one identifiers of :data:`CHECK_IDS`;
    * ``title`` -- the one-line human description ``qrcheck list`` prints;
    * ``family`` -- the module that owns the check, and the JUnit ``classname``
      suffix;
    * ``requires`` -- the capabilities the target must declare for this check to be
      runnable, intersected with the adapter's declaration before execution so an
      unmet requirement becomes a ``skipped`` result with reason
      ``unsupported_capability`` rather than a failure of the target;
    * ``dependencies`` -- the optional runtime dependencies the check needs, as
      :data:`DEPENDENCY_TOKENS` members, resolved per run into
      :attr:`RunContext.availability`;
    * ``cases`` -- the generator producing this check's cases.

    Construction validates and normalises rather than trusting the declaration:
    ``requires`` is coerced to members, ``dependencies`` is sorted and de-duplicated,
    and the identifier/family pair is checked against the prefix table. Everything
    is checked here, at import of the check module, so a malformed check fails the
    process that imports it instead of producing a plausible-looking report.

    ``cases`` stays the raw callable: this module never calls a check, because when
    a case executes (and whether it executes at all, under ``--fail-fast``) is the
    runner's decision.

    Raises:
        CheckRegistrationError: Any field is unusable or inconsistent.
    """

    id: str
    title: str
    family: str
    requires: frozenset[Capability]
    dependencies: tuple[str, ...]
    cases: CasesFn

    def __post_init__(self) -> None:
        """Validate the declaration and normalise the two collection fields.

        Raises:
            CheckRegistrationError: The identifier is outside :data:`CHECK_IDS`, the
                family is unknown or does not own the identifier's prefix, the title
                is empty, a capability or dependency token is unknown, or ``cases``
                is not callable.
        """
        check_id = self.id
        if not isinstance(check_id, str) or check_id not in CHECK_IDS:
            raise CheckRegistrationError(
                f"check id {check_id!r} is not one of the registered check identifiers "
                f"{CHECK_IDS}"
            )
        family = self.family
        if not isinstance(family, str) or family not in FAMILY_NAMES:
            raise CheckRegistrationError(
                f"check {check_id!r} declares family {family!r}, which is not one of "
                f"{FAMILY_NAMES}"
            )
        prefix = check_id.split("-", 1)[0]
        owning_family = FAMILY_BY_PREFIX.get(prefix)
        if owning_family != family:
            raise CheckRegistrationError(
                f"check {check_id!r} has prefix {prefix!r}, which belongs to family "
                f"{owning_family!r}, but the contract declares family {family!r}"
            )
        title = self.title
        if not isinstance(title, str) or not title.strip():
            raise CheckRegistrationError(
                f"check {check_id!r} has an empty title; 'qrcheck list' prints it and a "
                f"blank entry would name nothing"
            )
        if not callable(self.cases):
            raise CheckRegistrationError(
                f"check {check_id!r} declares cases={self.cases!r}, which is not callable"
            )
        object.__setattr__(self, "requires", _coerce_requires(self.requires, check_id))
        object.__setattr__(
            self, "dependencies", _normalise_dependencies(self.dependencies, check_id)
        )


# The decorator :func:`check` returns: it takes a check's ``cases`` generator and
# yields the registered contract, which replaces the function in the module.
CheckDecorator: TypeAlias = Callable[[CasesFn], CheckContract]

# The registry's one writable store. Keyed by check identifier, populated only
# while the check family modules import, and never mutated afterwards: a run reads
# it, it does not add to it. Module-private so nothing outside this file can write
# to it by accident; :func:`register` is the only writer and :data:`REGISTRY` is the
# only reader-facing view.
_REGISTRY: dict[str, CheckContract] = {}

# The read-only view every consumer holds. ``set(REGISTRY)`` is the set of registered
# identifiers and ``len(REGISTRY)`` is how many checks exist, which is exactly what
# the reachability gate compares against a run's executed set.
REGISTRY: Mapping[str, CheckContract] = MappingProxyType(_REGISTRY)


def register(contract: CheckContract) -> CheckContract:
    """Add `contract` to the registry and return it unchanged.

    The only way a check enters the product. A duplicate identifier raises rather
    than replacing the existing entry: a silent replacement would make the executed
    set, the expected-failure table and the list printed by ``qrcheck list``
    disagree about one identifier, and the report would carry the second check's
    results under the first one's evidence directory.

    Args:
        contract: The validated contract to register.

    Returns:
        `contract` itself, so the caller composes without a second lookup.

    Raises:
        CheckRegistrationError: `contract` is not a :class:`CheckContract`, or its
            identifier is already registered.
    """
    if not isinstance(contract, CheckContract):
        raise CheckRegistrationError(
            f"register() expects a CheckContract, got {type(contract).__name__}"
        )
    check_id = contract.id
    existing = _REGISTRY.get(check_id)
    if existing is not None:
        raise CheckRegistrationError(
            f"check {check_id!r} is already registered as {existing.title!r}; a duplicate "
            f"must never replace or shadow a registered check"
        )
    _REGISTRY[check_id] = contract
    return contract


def check(
    *,
    id: str,
    title: str,
    family: str,
    requires: frozenset[Capability] = frozenset(),
    dependencies: tuple[str, ...] = (),
) -> CheckDecorator:
    """Return the decorator that registers one check.

    Applied to a check module's ``cases`` generator, so a family module registers
    every check it owns with one decorator each::

        @check(
            id="RT-ROUNDTRIP",
            title="Payload round-trip",
            family="roundtrip",
            requires=frozenset({Capability.png}),
            dependencies=(DEPENDENCY_PRIMARY_DECODER,),
        )
        def cases(ctx: RunContext) -> Iterator[Case]:
            ...

    The decorated name becomes the :class:`CheckContract`, **not** the function: a
    module that keeps the decorated object can hand it to a test or print its title,
    but it can no longer call it as ``cases(ctx)``. Reach the generator through
    ``contract.cases(ctx)`` instead.

    Args:
        id: One of the twenty-one identifiers of :data:`CHECK_IDS`.
        title: The one-line description ``qrcheck list`` prints.
        family: The owning family name, which must match the identifier's prefix.
        requires: Capabilities the target must declare for this check to run.
        dependencies: Dependency tokens the check needs, as
            :data:`DEPENDENCY_TOKENS` members.

    Returns:
        A decorator mapping a ``cases`` generator to its registered
        :class:`CheckContract`.

    Raises:
        CheckRegistrationError: The declaration is invalid, or the identifier is
            already registered -- raised when the returned decorator is applied,
            since that is when the contract is built.
    """

    def decorator(cases_fn: CasesFn) -> CheckContract:
        """Build the contract around `cases_fn`, register it, and return it."""
        contract = CheckContract(
            id=id,
            title=title,
            family=family,
            requires=requires,
            dependencies=dependencies,
            cases=cases_fn,
        )
        return register(contract)

    return decorator


def iter_checks() -> tuple[CheckContract, ...]:
    """Return every registered check, sorted by identifier ascending.

    The only iteration order any consumer may use: ``qrcheck list``, the runner's
    execution walk, the expected-failure validation and the report sections all read
    it from here, so import order never reaches an artefact and two runs of the same
    configuration describe their checks in the same order.

    Returns:
        The registered contracts in identifier order, empty when the check family
        modules have not been imported yet.
    """
    return tuple(_REGISTRY[check_id] for check_id in sorted(_REGISTRY))


def get_check(check_id: str) -> CheckContract:
    """Return the registered contract for `check_id`.

    Args:
        check_id: The check identifier to look up.

    Returns:
        The registered contract.

    Raises:
        KeyError: No check is registered under `check_id`. This never returns
            ``None``: a caller that cannot find a check has a defect to report, not a
            missing check to skip.
    """
    try:
        return _REGISTRY[check_id]
    except KeyError:
        raise KeyError(
            f"check {check_id!r} is not registered; registered checks: "
            f"{list(registered_ids())}"
        ) from None


def registered_ids() -> tuple[str, ...]:
    """Return the identifiers of every registered check, ascending."""
    return tuple(sorted(_REGISTRY))


@contextlib.contextmanager
def isolated_registry() -> Iterator[dict[str, CheckContract]]:
    """Run a block against an empty registry, then restore what was registered.

    Test-only support, and never used by runtime code. Registration is global and
    import-time, so a test that has to exercise the validation branches or register a
    stub adapter's checks would otherwise mutate the session-wide registry that
    ``len(set(REGISTRY)) == 21`` depends on. Inside the block :data:`REGISTRY` shows
    only what the test registered; leaving it restores exactly the previous contents,
    including when the block raises.

    Yields:
        The registry's writable store, for a test to register stubs into directly or
        through :func:`register`.
    """
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        yield _REGISTRY
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)

