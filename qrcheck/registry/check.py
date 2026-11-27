"""The single owner of check identity, ordering and the context a check runs in.

Each of the twenty-one checks registers itself here through :func:`check`, and every consumer that
names, orders, selects or runs a check reads :data:`REGISTRY` rather than a list of its own. Four
things exist exactly once in qrcheck, all here: the twenty-one identifiers (:data:`CHECK_IDS`, the
only literal list of them) with the nine families that own their prefixes
(:data:`FAMILY_BY_PREFIX`), the registry, :class:`RunContext` (everything a check may know about its
run) and :class:`Case`.

Ordering is part of the contract: checks iterate **by identifier ascending** everywhere -- ``qrcheck
list``, the run's execution order and every report section -- so :func:`iter_checks` is the only
iteration a consumer may use and import order never reaches an artefact. An identifier outside
:data:`CHECK_IDS`, an unknown family, a prefix owned by another family, an unknown dependency token,
an empty title and a duplicate identifier each raise :class:`CheckRegistrationError` at import, a
duplicate especially, since shadowing a registered check changes the executed set under a report
that already claims to describe it.

Layering: :class:`Capability` is the only name imported from ``qrcheck.targets.base`` at run time,
and ``qrcheck.checks`` is **never** imported -- the check family modules import *this* module to
register, so such an import would close the registration cycle. Everything else is a type-checking
import, so a process that only lists checks pulls in no decoder, image library, HTTP client or CLI
framework.
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
# This tuple is the only literal list of check identifiers in the product: the registry validates
# against it, ``qrcheck list`` and the reports print what the registry holds, and the reachability
# gate compares the two sets. Ascending order shows an added or removed check where it belongs in
# a human diff, and it agreeing with ``sorted(CHECK_IDS)`` is what lets :func:`iter_checks`
# promise a stable iteration without a second sort key. An identifier is never renamed, split or
# merged: CI trend lines and the expected-failure table key on these exact strings.
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

# The nine check families, one per module under ``qrcheck/checks/``, in ascending order.  A family
# name is also the JUnit ``classname`` suffix (``classname="qrcheck.<family>"``), so it is a
# stable output surface rather than an internal label.
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

# Which family owns which identifier prefix.  Registration compares an identifier's prefix against
# its declared family, which is what holds the one-module-per-family rule mechanically: a check
# filed under the wrong family would otherwise land in the wrong JUnit ``classname`` and in the
# wrong report section.
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

# The runtime dependencies a check may declare, as tokens rather than module names. A check declares
# *what it needs*, never *what provides it*: ``checks/*`` may not import a decoder or the
# rasterizer's discovery logic, and each token's availability is resolved once per run by the runner
# into :attr:`RunContext.availability`. A missing primary decoder (zxing-cpp) blocks the run:
# ``qrcheck doctor`` and ``qrcheck run`` exit 2, naming zxing-cpp and the interpreter/platform,
# before any check runs, so ``RT-ROUNDTRIP`` is never silently skipped. A check still declares
# :data:`DEPENDENCY_PRIMARY_DECODER`, because that is the honest record of what it needs --
# notably ``PF-DECODE`` and ``CL-ARTIFACTS``, which decode but need nothing from the target -- and
# resolution can never turn that record into a skip, because a run without the primary decoder never
# reaches resolution. Only two tools degrade. The secondary decoder (opencv) leaves the run in
# single-decoder mode, reported as ``primary_only``. The SVG rasterizer makes ``RN-SVG`` report
# ``skipped`` with reason ``missing_dependency`` and the discovery order it attempted. The
# vocabulary is closed at three, one per runtime tool whose availability a check can depend on.
# There is deliberately no token for the platform file-descriptor directory: a declared token can
# only produce a whole-check ``missing_dependency`` skip, whereas ``CL-ARTIFACTS`` reports its
# handle half per case as ``unsupported_platform`` while its temporary-file half still asserts.
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

    Raised for a duplicate identifier, an identifier outside :data:`CHECK_IDS`, an unknown family, a
    prefix that belongs to a different family, an unknown dependency token, an empty title and a
    non-callable ``cases``, always naming the offending value. It subclasses :class:`ValueError`, so
    a caller that already catches the builtin -- a test registering a deliberately invalid contract,
    the CLI turning a broken installation into exit code 3 -- needs no import of this module to
    handle it.
    """


@dataclass(frozen=True, slots=True)
class Case:
    """One case a check contributes to a run: an identifier and a callable.

    ``case_id`` is ``corpus:<record-id>`` or ``synthetic:<name>``; it is the case component of
    :func:`qrcheck.report.model.failure_id`, hence part of the leaf of
    ``<report_dir>/evidence/<check-id>/<failure-id>/`` and never the leaf itself -- two checks share
    a case identifier such as ``corpus:url-000`` and the ``<check-id>`` above it is what separates
    them. ``execute`` produces that case's result when the runner calls it with the run's
    :class:`RunContext`; neither this module nor a check's ``cases`` generator ever calls it, which
    leaves ``--fail-fast`` and the one-error-per-check policy on target loss the runner's decisions.
    Raises :exc:`ValueError` when ``case_id`` is not a non-empty string or ``execute`` is not
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

    One instance per run reaches each check's ``cases(ctx)`` and then each returned :class:`Case`,
    so enumerating cases and executing one see the same view. Fields are raw: the adapter port
    itself, the resolved configuration that owns every threshold, and the corpus already filtered
    and sorted by the runner -- the registry filters nothing, so ``--class`` cannot behave
    differently for two checks. ``availability`` maps the :data:`DEPENDENCY_*` tokens to whether
    that dependency was found, and an **absent key means unavailable**: a forgotten probe then makes
    the check report ``skipped`` with reason ``missing_dependency``, which the zero-skip reference
    gate fails loudly, instead of a pass against something never present. A missing primary decoder
    never reaches this mapping: ``qrcheck doctor`` and ``qrcheck run`` exit 2 naming zxing-cpp
    first, so that token records what a check needs rather than a skip path.

    ``temp_baseline`` is the temporary-file snapshot the run began with: the names in
    :func:`tempfile.gettempdir` that carried the framework's ``qrcheck-`` prefix when the runner
    took it, at the last instant before any check could run. The runner observes it through the
    check layer's own :func:`qrcheck.checks.cleanup.temp_entries`, and only ``CL-ARTIFACTS`` reads
    it. A context built directly -- as a test does -- carries the empty default, under which any
    prefixed entry present is reported as leaked: the safe direction for a snapshot nobody took. The
    instance is frozen, so a check cannot change what another check sees.
    """

    adapter: Adapter
    config: Config
    corpus: tuple[CorpusRecord, ...]
    ec_level: str
    seed: int
    availability: Mapping[str, bool]
    evidence: EvidenceStore
    temp_baseline: frozenset[str] = frozenset()

    def dependency_available(self, token: str) -> bool:
        """Return whether `token` was resolved as available for this run.

        An absent token is unavailable, never available by omission.

        Args:
            token: One of the :data:`DEPENDENCY_TOKENS` values.

        Returns:
            ``True`` only when the runner recorded the dependency as present.
        """
        return bool(self.availability.get(token, False))


# The shape of a check's ``cases`` generator: given the run's context, yield the cases this check
# contributes. Declared once so the decorator, the contract field and every family module's
# signature agree on it.
CasesFn: TypeAlias = Callable[[RunContext], Iterator[Case]]


def _coerce_requires(value: object, check_id: str) -> frozenset[Capability]:
    """Return `value` as a ``frozenset`` of :class:`Capability` members.

    Members, their string values and a single member are all accepted, so a check module cannot
    register a requirement that only looks satisfied; ``check_id`` is quoted in the diagnostic when
    a name is unknown. Returns the requirements as members, empty when nothing was declared, and
    raises :class:`CheckRegistrationError` when a name is not a :class:`Capability` member or
    `value` is not a collection of them.
    """
    if value is None:
        return frozenset()
    # ``Capability`` is a ``StrEnum``, so a member is also a ``str``: a bare name and a bare
    # member are both one requirement rather than an iterable of characters.
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

    Sorting and de-duplicating is what makes two declarations of the same requirement produce the
    same ``summary.json`` and the same ``qrcheck list`` line, however the check module wrote them.
    Returns the tokens in ascending order, empty when nothing was declared, and raises
    :class:`CheckRegistrationError` when a token is outside :data:`DEPENDENCY_TOKENS` or `value` is
    not a collection of them, `check_id` being quoted in the diagnostic.
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

    Eight fields, each a stable public surface: ``id`` (one of :data:`CHECK_IDS`), ``title``
    (the line ``qrcheck list`` prints), ``family`` (owning module and JUnit ``classname``
    suffix), ``requires`` (capabilities the target must declare, else the check is ``skipped``
    with reason ``unsupported_capability``), ``dependencies`` (:data:`DEPENDENCY_TOKENS` members,
    resolved into :attr:`RunContext.availability`), ``cases``, ``corpus_driven`` and
    ``observes_run_end``. Construction validates and normalises at import of the check module -
    ``requires`` coerced to members, ``dependencies`` sorted and de-duplicated, the
    identifier/family pair checked against the prefix table - so a malformed check fails the
    process that imports it, raising :class:`CheckRegistrationError` for any field that is
    unusable or inconsistent. ``cases`` stays the raw callable: when a case executes, and whether
    it does under ``--fail-fast``, is the runner's decision.

    ``corpus_driven`` says whether the cases this check yields are drawn from the corpus the run
    loaded, which is what decides whether the corpus file has to exist: the runner reads
    ``corpus_path`` only when a corpus-driven check is selected, so a run of synthetic-only checks
    can complete on a machine where no corpus has ever been built. It is declared rather than
    inferred, because only the check knows whether it reads :attr:`RunContext.corpus`.

    ``observes_run_end`` says whether the check asserts a property of the whole run rather than of
    one case. It is true for exactly one check, ``CL-ARTIFACTS``, whose subject is the footprint the
    finished run leaves behind: the runner takes that check's "before" snapshot immediately before
    the walk and executes the check after it, so the two snapshots bracket the run.
    Every other check leaves it false and has its cases executed in the ordinary identifier order.
    """

    id: str
    title: str
    family: str
    requires: frozenset[Capability]
    dependencies: tuple[str, ...]
    cases: CasesFn
    corpus_driven: bool = False
    observes_run_end: bool = False

    def __post_init__(self) -> None:
        """Validate the declaration and normalise the two collection fields.

        Raises:
            CheckRegistrationError: The identifier is outside :data:`CHECK_IDS`, the
                family is unknown or does not own the identifier's prefix, the title
                is empty, a capability or dependency token is unknown, ``cases`` is not callable,
                ``corpus_driven`` is not a bool, or ``observes_run_end`` is not a bool.
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
        if not isinstance(self.corpus_driven, bool):
            raise CheckRegistrationError(
                f"check {check_id!r} declares corpus_driven={self.corpus_driven!r}, which is "
                f"not a bool; it decides whether the corpus file must exist for this run"
            )
        if not isinstance(self.observes_run_end, bool):
            raise CheckRegistrationError(
                f"check {check_id!r} declares observes_run_end={self.observes_run_end!r}, which "
                f"is not a bool; it decides whether the runner executes this check after the walk, "
                f"so that it observes the run's end"
            )
        object.__setattr__(self, "requires", _coerce_requires(self.requires, check_id))
        object.__setattr__(
            self, "dependencies", _normalise_dependencies(self.dependencies, check_id)
        )


# The decorator :func:`check` returns: it takes a check's ``cases`` generator and yields the
# registered contract, which replaces the function in the module.
CheckDecorator: TypeAlias = Callable[[CasesFn], CheckContract]

# The registry's one writable store. Keyed by check identifier, populated only while the check
# family modules import, and never mutated afterwards: a run reads it, it does not add to it.
# Module-private so nothing outside this file can write to it by accident; :func:`register` is the
# only writer and :data:`REGISTRY` is the only reader-facing view.
_REGISTRY: dict[str, CheckContract] = {}

# The read-only view every consumer holds. ``set(REGISTRY)`` is the set of registered identifiers
# and ``len(REGISTRY)`` is how many checks exist, which is exactly what the reachability gate
# compares against a run's executed set.
REGISTRY: Mapping[str, CheckContract] = MappingProxyType(_REGISTRY)


def register(contract: CheckContract) -> CheckContract:
    """Add `contract` to the registry and return it unchanged.

    The only way a check enters the product. A duplicate identifier raises rather than replacing the
    existing entry: a silent replacement would make the executed set, the expected-failure table and
    the list ``qrcheck list`` prints disagree about one identifier, and the report would carry the
    second check's results under the first one's evidence directory. Raises
    :class:`CheckRegistrationError` when `contract` is not a :class:`CheckContract` or its
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
    corpus_driven: bool = False,
    observes_run_end: bool = False,
) -> CheckDecorator:
    """Return the decorator that registers one check.

    Applied to a check module's ``cases`` generator. The decorated name becomes the
    :class:`CheckContract`, **not** the function, so reach the generator through
    ``contract.cases(ctx)`` instead of calling it. ``id`` is one of the twenty-one identifiers of
    :data:`CHECK_IDS`, ``title`` the line ``qrcheck list`` prints, ``family`` the owning family
    name, which must match the identifier's prefix, ``requires`` the capabilities the target must
    declare for the check to run, ``dependencies`` the tokens it needs, from
    :data:`DEPENDENCY_TOKENS`, and ``corpus_driven`` whether its cases are drawn from the corpus
    the run loaded -- which is what makes the corpus file a prerequisite of a run that selects it,
    and nothing else one. ``observes_run_end`` says whether the check asserts a property of the
    whole run rather than of one case, which is true for the one check the runner executes after
    the walk so that it observes the run's end, and false for every other check. Returns a
    decorator mapping a ``cases`` generator to its registered contract, and
    raises :class:`CheckRegistrationError` when the declaration is invalid or the identifier is
    already registered - raised when the returned decorator is applied, since that is when the
    contract is built.
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
            corpus_driven=corpus_driven,
            observes_run_end=observes_run_end,
        )
        return register(contract)

    return decorator


def iter_checks() -> tuple[CheckContract, ...]:
    """Return every registered check, sorted by identifier ascending.

    The only iteration order any consumer may use: ``qrcheck list``, the runner's execution walk,
    the expected-failure validation and every report section read it from here, so import order
    never reaches an artefact and two runs of one configuration describe their checks in the same
    order.

    Returns:
        The registered contracts in identifier order, empty when the check family modules have not
        been imported yet.
    """
    return tuple(_REGISTRY[check_id] for check_id in sorted(_REGISTRY))


def get_check(check_id: str) -> CheckContract:
    """Return the registered contract for `check_id`.

    Raises :exc:`KeyError` when no check is registered under `check_id`. This never returns
    ``None``: a caller that cannot find a check has a defect to report, not a check to skip.
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

    Test-only support, never used by runtime code: registration is global and import-time, so a test
    exercising the validation branches or registering stub checks would otherwise mutate the
    session-wide registry that ``len(set(REGISTRY)) == 21`` depends on. Leaving the block restores
    the previous contents exactly, including when the block raises, and it yields the registry's
    writable store.
    """
    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        yield _REGISTRY
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)
