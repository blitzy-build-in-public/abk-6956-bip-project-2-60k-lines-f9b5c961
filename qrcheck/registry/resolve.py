"""Resolve a check selection against what a target and a host can actually do.

Nothing in qrcheck may decide *while running* that a check cannot run: a check that turned out to be
impossible half-way through a case would produce a failure describing the host rather than the
target, and a check that was quietly dropped would let a run look complete when it covered less.
This module makes that decision once, before execution, from the two facts the runner already holds:
the capabilities the adapter declares, intersected with each
:class:`~qrcheck.registry.check.CheckContract`'s own ``requires`` set, and the availability of the
optional runtime dependencies the host provides, keyed by the
:data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` tokens a contract declares. Its output is a
:class:`Resolution`: the checks that will run and the checks that will not, each skip carrying the
reason and the exact requirements it lacks.

A shortfall is never an exception. Every shortfall becomes a :class:`SkipOutcome`, which the runner
turns into ``skipped`` results through :func:`skip_case_results`; nothing here raises because a
target cannot do something, and nothing here decides an exit code. Exactly two reasons exist:
:attr:`~qrcheck.report.model.Reason.UNSUPPORTED_CAPABILITY`, where the target did not declare a
capability the check requires and the configuration is valid, so this target simply cannot serve
this check (an unknown capability *name* is the exit-code-2 configuration error instead, rejected
far earlier by the schema); and :attr:`~qrcheck.report.model.Reason.MISSING_DEPENDENCY`, where the
check needs an optional runtime dependency the host does not provide, such as the SVG rasterizer.
The platform file-descriptor directory is not one of these: that check reports the handle half per
case as ``unsupported_platform`` while its temporary-file half still asserts (AAP §0.8.1), which a
whole-check skip would drop. A capability shortfall outranks a missing dependency, so a check short
of both is skipped once with the capability reason. The discovery order a
missing dependency was probed in travels with the skip as :attr:`SkipOutcome.attempted`, so a
report states what was tried, not only what is absent (AAP §0.3.5, §0.7.2).

A skip is a real outcome, never a flavoured pass: it is counted in its own integer of
:class:`~qrcheck.report.model.Counts` and can never produce a passing exit code, and since the
reference target declares all thirteen capabilities a conforming run reaches zero skips.
:func:`qrcheck.targets.base.require_capabilities` stays as the adapter's own guard for a request
that reaches it by any other route, so the two mechanisms answer different questions. Both tuples of
a :class:`Resolution` are ordered by check identifier ascending, a skip's ``missing`` tuple is
sorted and de-duplicated, and no input is mutated, so two resolutions of one selection describe it
identically and nothing here reads the clock, the environment, the filesystem, the network or a
random source. The only runtime imports are :mod:`qrcheck.registry.check` and
:mod:`qrcheck.report.model`: requirements travel on the contract and availability arrives as data,
so a fourth adapter kind needs no change here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

# ``Capability`` comes from the registry module's own import of the adapter port's enum, not from
# ``qrcheck.targets.base`` directly: this module's runtime dependencies are the registry (which
# owns check identity, the dependency vocabulary and the capability type its contracts already
# carry) and the result model, so it never grows a second edge into the adapter layer to reach a
# type it only annotates.
from qrcheck.registry.check import DEPENDENCY_TOKENS, Capability, CheckContract
from qrcheck.report.model import CaseResult, Reason, Status

__all__ = [
    "CAPABILITY_CASE_ID",
    "Resolution",
    "SkipOutcome",
    "resolve",
    "skip_case_results",
    "unmet_capabilities",
    "unmet_dependencies",
]

# The case identifier a pre-execution skip result is filed under.
#
# One string, defined once, because three artefacts key on it: ``summary.json`` carries it as
# ``case_id``, ``junit.xml`` renders it inside ``name="<CHECK-ID>[<case_id>]"`` and the runner
# accumulates one such result per unrunnable check.  It is deliberately distinct from the runner's
# own ``synthetic:target`` case identifier, which describes a *different* condition -- a target
# that was reachable when the run started and then vanished -- so a reader can tell "this target
# never promised this" from "this target broke".
CAPABILITY_CASE_ID: Final[str] = "synthetic:capability"

# The only two reasons a resolution may skip a check with.  Anything else -- an unsupported
# platform, a decoder disagreement, a target rejection -- describes something that happens *to a
# case while it runs*, and a resolution that claimed one would be inventing an observation it
# never made.
_SKIP_REASONS: Final[frozenset[Reason]] = frozenset(
    {Reason.UNSUPPORTED_CAPABILITY, Reason.MISSING_DEPENDENCY}
)


class _DeclaresCapabilities(Protocol):  # pragma: no cover - declaration, never executed
    """The one attribute resolution reads from an adapter.

    Declared so the accepted input can be annotated precisely without importing the adapter port:
    this module needs to know that *something* publishes a ``capabilities`` collection and nothing
    else about it, so an adapter, a resolved target record or a test double all satisfy this
    structurally.
    """

    @property
    def capabilities(self) -> Iterable[Capability | str]:
        """Return the capabilities this object declares."""
        ...


class _AvailabilityProbe(Protocol):  # pragma: no cover - declaration, never executed
    """An availability source that projects itself to a mapping."""

    def as_mapping(self) -> Mapping[str, bool]:
        """Return the dependency-token to availability mapping."""
        ...


class _AvailabilityQuery(Protocol):  # pragma: no cover - declaration, never executed
    """An availability source that answers one dependency token at a time."""

    def available(self, dependency: str) -> bool:
        """Return whether `dependency` is available for this run."""
        ...


# What the ``capabilities`` argument may be: a collection of members or their names (what the
# configuration resolves to, and what a test passes), or any object that declares a
# ``capabilities`` attribute (an adapter -- what the runner passes).  A member of
# :class:`Capability` is itself a string, so a bare name is read as one capability rather than as
# an iterable of characters.
CapabilitySource: TypeAlias = Iterable[Capability | str] | _DeclaresCapabilities | None

# What the ``availability`` argument may be: a mapping keyed by the
# :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` values (what the runner builds from its
# probes), an object that exposes ``as_mapping()`` or ``available(token)``, or ``None`` for
# "nothing beyond the mandatory dependencies is available".
AvailabilitySource: TypeAlias = (
    Mapping[str, bool] | _AvailabilityProbe | _AvailabilityQuery | None
)

_CAPABILITY_FORMS: Final[str] = (
    "a collection of Capability members or their names, or an object declaring a "
    "`capabilities` attribute"
)
_AVAILABILITY_FORMS: Final[str] = (
    "a mapping of dependency token to bool, or an object exposing as_mapping() or "
    "available(dependency)"
)


def _candidate_order(context: str, value: object) -> tuple[str, ...]:
    """Return the discovery order `value` holds, kept as given rather than sorted.

    Raises:
        ValueError: `value` is a string or is not iterable, or holds an entry that is not
            a non-empty ``str``.
    """
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(
            f"{context}={value!r}, which is not a collection of candidate names; a single "
            f"name would be read as its characters"
        )
    order = tuple(value)
    for candidate in order:
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(
                f"{context} holds the candidate {candidate!r}, which is not a non-empty "
                f"str; a discovery order is a sequence of candidate names"
            )
    return order


@dataclass(frozen=True, slots=True)
class SkipOutcome:
    """One check that will not execute, why, which requirements it lacks, and what was tried.

    Four fields, each a public surface that reaches an artefact: ``check_id`` names the check,
    ``reason`` is the machine-readable cause ``summary.json`` and the JUnit ``<skipped
    message="..."/>`` carry, ``missing`` lists the absent requirements -- capability names for an
    ``unsupported_capability`` skip, dependency tokens for a ``missing_dependency`` one -- and
    ``attempted`` is the discovery order walked for a dependency the host offers through an ordered
    candidate list (AAP §0.3.5, §0.7.2), so a reader learns what to add and what was tried.
    Construction normalises and validates: ``missing`` is sorted and de-duplicated, ``reason`` is
    narrowed to a :class:`~qrcheck.report.model.Reason` member, ``attempted`` is kept as walked, and
    a skip that names no cause raises :exc:`ValueError`, because an unexplained skip is exactly the
    quiet gap the zero-skip reference gate exists to catch.
    """

    check_id: str
    reason: Reason
    missing: tuple[str, ...]
    attempted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the record and normalise ``reason``, ``missing`` and ``attempted``.

        Raises:
            ValueError: ``check_id`` is not a non-empty ``str``, ``reason`` is outside
                :class:`~qrcheck.report.model.Reason` or is not one of the two skip
                reasons, ``missing`` is empty or is not a collection of names, or
                ``attempted`` is not a collection of non-empty names.
        """
        check_id = self.check_id
        if not isinstance(check_id, str) or not check_id.strip():
            raise ValueError(
                f"SkipOutcome.check_id must be a non-empty str, got {check_id!r}"
            )
        try:
            reason = Reason(self.reason)
        except ValueError as exc:
            raise ValueError(f"SkipOutcome.reason is not a Reason: {self.reason!r}") from exc
        if reason not in _SKIP_REASONS:
            vocabulary = sorted(member.value for member in _SKIP_REASONS)
            raise ValueError(
                f"skip of check {check_id!r} carries reason {reason.value!r}; a resolution "
                f"may only skip with one of {vocabulary}"
            )
        declared = self.missing
        if isinstance(declared, str) or not isinstance(declared, Iterable):
            raise ValueError(
                f"skip of check {check_id!r} declares missing={declared!r}, which is not a "
                f"collection of requirement names; a single name would be read as its "
                f"characters"
            )
        missing = tuple(sorted({str(item) for item in declared}))
        if not missing:
            raise ValueError(
                f"skip of check {check_id!r} with reason {reason.value!r} names no missing "
                f"requirement; an unexplained skip is indistinguishable from a check that "
                f"was quietly weakened"
            )
        attempted = _candidate_order(
            f"skip of check {check_id!r} declares attempted", self.attempted
        )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "missing", missing)
        object.__setattr__(self, "attempted", attempted)


@dataclass(frozen=True)
class Resolution:
    """The selection split into what will run and what will not, ordered by check id.

    Two tuples, and a check is in exactly one of them: runnable or skipped, never both and never
    twice, which :meth:`__post_init__` enforces because a double entry would make the runner execute
    a check once while a report counted it twice. Both tuples are sorted by check identifier
    ascending on construction, so import order, set iteration and the order the caller happened to
    pass cannot reach ``summary.json``, ``junit.xml`` or the HTML report through this object. The
    instance is frozen and both fields are tuples, so a caller cannot append to a resolution it has
    been handed: the runner decides execution from this record, and a mutated copy would let the
    executed set and the reported set disagree. A check identifier appearing twice raises
    :exc:`ValueError`.
    """

    runnable: tuple[CheckContract, ...] = ()
    skipped: tuple[SkipOutcome, ...] = ()

    def __post_init__(self) -> None:
        """Sort both tuples by check identifier and reject a duplicated check.

        Raises:
            ValueError: A check identifier appears in both tuples, or twice in one of
                them.
        """
        runnable = tuple(sorted(self.runnable, key=lambda contract: contract.id))
        skipped = tuple(sorted(self.skipped, key=lambda outcome: outcome.check_id))
        decided = [contract.id for contract in runnable]
        decided.extend(outcome.check_id for outcome in skipped)
        duplicated = sorted({check_id for check_id in decided if decided.count(check_id) > 1})
        if duplicated:
            raise ValueError(
                f"a resolution decides each check exactly once; {duplicated} appear more "
                f"than once across runnable and skipped"
            )
        object.__setattr__(self, "runnable", runnable)
        object.__setattr__(self, "skipped", skipped)

    @property
    def is_complete(self) -> bool:
        """Return whether every selected check will run, i.e. nothing was skipped.

        The clean-run gate asks exactly this of a conforming target: a run whose resolution is
        complete cannot report a skip, so a check cannot be made to pass by making it skip.
        """
        return not self.skipped

    @property
    def runnable_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the checks that will run, in execution order."""
        return tuple(contract.id for contract in self.runnable)

    @property
    def skipped_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the checks that will not run, in report order."""
        return tuple(outcome.check_id for outcome in self.skipped)


def _declared_capabilities(source: CapabilitySource) -> frozenset[Capability]:
    """Return the capability set `source` describes, as members.

    Three accepted forms, because the same fact arrives from three places: a collection of members
    or their names (the configuration and the tests), an object publishing ``capabilities`` (the
    adapter the runner passes), and ``None`` for a declaration nobody made, which resolves to the
    empty set -- the safe direction, so a forgotten declaration surfaces as a skip the zero-skip
    reference gate fails loudly rather than as a check that ran and counted as a pass. Raises
    :exc:`TypeError` when `source` is a string or neither a collection nor a declaring object, and
    :exc:`ValueError` when a declared name is not a :class:`Capability` member, which means a
    declaration was built by hand.
    """
    declared = getattr(source, "capabilities", source)
    if declared is None:
        return frozenset()
    if isinstance(declared, (str, bytes)):
        raise TypeError(
            f"capabilities must be {_CAPABILITY_FORMS}, got the single value "
            f"{declared!r}; a member is itself a string, so it would otherwise iterate "
            f"as characters"
        )
    try:
        names = tuple(declared)
    except TypeError:
        raise TypeError(
            f"capabilities must be {_CAPABILITY_FORMS}, got {type(source).__name__}"
        ) from None
    return frozenset(Capability(name) for name in names)


def _available_dependencies(source: AvailabilitySource) -> Mapping[str, bool]:
    """Return the dependency availability `source` describes, as a mapping.

    A mapping keyed by the :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` values (what the runner
    builds from its probes), an object exposing ``as_mapping()``, an object exposing
    ``available(dependency)`` queried once per token, or ``None`` for "nothing available". An
    **absent token means unavailable** -- this module never assumes a dependency nobody probed, so a
    forgotten probe becomes a skip the zero-skip reference gate catches instead of a check that ran
    against something that was never there. Raises :exc:`TypeError` when `source` is none of the
    accepted forms or its ``as_mapping()`` returned something other than a mapping.
    """
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return source
    as_mapping = getattr(source, "as_mapping", None)
    if callable(as_mapping):
        projected = as_mapping()
        if not isinstance(projected, Mapping):
            raise TypeError(
                f"availability.as_mapping() must return {_AVAILABILITY_FORMS}, got "
                f"{type(projected).__name__}"
            )
        return projected
    available = getattr(source, "available", None)
    if callable(available):
        return {
            token: bool(available(token)) for token in sorted(DEPENDENCY_TOKENS)
        }
    raise TypeError(
        f"availability must be {_AVAILABILITY_FORMS}, got {type(source).__name__}"
    )


def _attempted_orders(
    attempted: Mapping[str, Iterable[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Return the discovery orders `attempted` describes, keyed by dependency token.

    A token absent from the mapping has no order to record; a key outside the vocabulary is
    refused.

    Raises:
        TypeError: `attempted` is not a mapping.
        ValueError: A key is not one of the declared dependency tokens, or an order is not a
            collection of non-empty candidate names.
    """
    if attempted is None:
        return {}
    if not isinstance(attempted, Mapping):
        raise TypeError(
            f"attempted must be a mapping of dependency token to the discovery order "
            f"walked for it, got {type(attempted).__name__}"
        )
    orders: dict[str, tuple[str, ...]] = {}
    for token, candidates in attempted.items():
        if token not in DEPENDENCY_TOKENS:
            raise ValueError(
                f"attempted names {token!r}, which is not a declared dependency token; the "
                f"vocabulary is {sorted(DEPENDENCY_TOKENS)}"
            )
        orders[token] = _candidate_order(f"attempted[{token!r}]", candidates)
    return orders


def unmet_capabilities(
    contract: CheckContract,
    capabilities: CapabilitySource = None,
) -> tuple[Capability, ...]:
    """Return the capabilities `contract` requires that `capabilities` does not declare.

    Requirement resolution in one direction: what a check needs, read from the contract's own
    ``requires`` set, against what the target positively declared. Nothing is required that the
    contract did not ask for, so the table of requirements exists only in the nine check family
    modules and cannot drift. Returns the unmet members, sorted by value, empty when the declaration
    satisfies the contract; raises :exc:`TypeError` when `capabilities` is not one of the accepted
    forms and :exc:`ValueError` when a declared name is not a :class:`Capability` member.
    """
    declared = _declared_capabilities(capabilities)
    return tuple(
        sorted((member for member in contract.requires if member not in declared), key=str)
    )


def unmet_dependencies(
    contract: CheckContract,
    availability: AvailabilitySource = None,
) -> tuple[str, ...]:
    """Return the dependency tokens `contract` declares that `availability` lacks.

    A token the mapping does not mention is unavailable: the default direction is the safe one, so
    an unfilled probe skips the check instead of letting it run without the tool it asserts with.
    Returns the absent tokens, sorted, empty when every declared dependency is available, and raises
    :exc:`TypeError` when `availability` is not one of the accepted forms.
    """
    resolved = _available_dependencies(availability)
    return tuple(
        sorted(token for token in contract.dependencies if not resolved.get(token, False))
    )


def resolve(
    selected: Iterable[CheckContract] | Mapping[str, CheckContract],
    capabilities: CapabilitySource = None,
    availability: AvailabilitySource = None,
    *,
    attempted: Mapping[str, Iterable[str]] | None = None,
) -> Resolution:
    """Split `selected` into the checks that can run and the checks that cannot.

    The single pre-execution decision: every selected check is compared against the declared
    capabilities and the resolved dependency availability, and lands in exactly one tuple of the
    returned :class:`Resolution`. A shortfall never raises -- it becomes a :class:`SkipOutcome` with
    ``unsupported_capability`` when the target does not declare something the check requires, or
    ``missing_dependency`` when a declared optional runtime dependency is unavailable, the
    capability reason winning when a check is short of both.

    ``selected`` takes any iterable of contracts or a mapping of identifier to contract, duplicates
    collapsing to the first occurrence; ``capabilities`` a collection of :class:`Capability` members
    or their names, an object declaring ``capabilities`` (an adapter, which is what the runner
    passes), or ``None`` for none declared; ``availability`` a mapping keyed by the
    :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` values, an object exposing ``as_mapping()`` or
    ``available(dependency)``, or ``None`` for nothing available --
    :class:`qrcheck.decode.base.DecoderAvailability` exposes neither method, so the runner
    translates its probes into a mapping before calling here, while ``attempted`` is a mapping
    from a dependency token to the ordered candidate list walked for it, copied into a
    ``missing_dependency`` skip (AAP §0.3.5, §0.7.2). Nothing is mutated, no adapter is touched,
    no probe is run and no result is built, so the executed set is knowable before the first case
    runs; the resolution returned has both tuples sorted by check identifier. Raises
    :exc:`TypeError` when an element of `selected` is not a
    :class:`~qrcheck.registry.check.CheckContract` or `capabilities`/`availability`/`attempted`
    is not in an accepted form, and :exc:`ValueError` when a declared capability name is not a
    :class:`Capability` member, `attempted` names a token no check may declare, or an order holds
    a non-name.
    """
    declared = _declared_capabilities(capabilities)
    available = _available_dependencies(availability)
    orders = _attempted_orders(attempted)
    contracts: Iterable[CheckContract]
    if isinstance(selected, Mapping):
        contracts = selected.values()
    else:
        contracts = selected
    runnable: list[CheckContract] = []
    skipped: list[SkipOutcome] = []
    decided: set[str] = set()
    for contract in contracts:
        if not isinstance(contract, CheckContract):
            raise TypeError(
                f"resolve() expects CheckContract items, got {type(contract).__name__}; "
                f"pass registered checks, such as qrcheck.registry.check.iter_checks()"
            )
        if contract.id in decided:
            continue
        decided.add(contract.id)
        unmet = unmet_capabilities(contract, declared)
        if unmet:
            skipped.append(
                SkipOutcome(
                    check_id=contract.id,
                    reason=Reason.UNSUPPORTED_CAPABILITY,
                    missing=tuple(str(member) for member in unmet),
                )
            )
            continue
        missing = unmet_dependencies(contract, available)
        if missing:
            skipped.append(
                SkipOutcome(
                    check_id=contract.id,
                    reason=Reason.MISSING_DEPENDENCY,
                    missing=missing,
                    # The search walked for each absent dependency (AAP §0.3.5, §0.7.2).
                    attempted=tuple(
                        candidate
                        for token in missing
                        for candidate in orders.get(token, ())
                    ),
                )
            )
            continue
        runnable.append(contract)
    return Resolution(runnable=tuple(runnable), skipped=tuple(skipped))


def skip_case_results(resolution: Resolution, seed: int) -> tuple[CaseResult, ...]:
    """Return the ``skipped`` results a run accumulates for its unrunnable checks.

    One :class:`~qrcheck.report.model.CaseResult` per :class:`SkipOutcome`, in check identifier
    order, and nothing else: the runner adds them beside the executed cases so that a selection
    covering a check the target cannot serve is described rather than silently shortened. Each
    carries :attr:`~qrcheck.report.model.Status.SKIPPED`, one of the two skip reasons,
    ``duration_ms`` of ``0.0`` and the run's seed, recorded verbatim so the report says which run
    the skip belongs to; the runner hands each result to
    :class:`qrcheck.report.evidence.EvidenceStore`, which persists a skipped-with-reason case and
    attaches its directory (AAP §0.3.4). ``decoded`` carries the discovery order walked for a
    missing dependency found through an ordered candidate list, as ``{"attempted": [...]}``
    (AAP §0.3.5, §0.7.2), and is empty for every other skip, where ``expected["missing"]`` states
    the whole cause; ``request`` is an empty mapping rather than null so every consumer reads a
    skip uniformly. A skipped result is never a pass: the four separate tallies of
    :class:`qrcheck.report.model.Counts` keep it out of ``passed``.
    """
    return tuple(
        CaseResult(
            check_id=outcome.check_id,
            case_id=CAPABILITY_CASE_ID,
            status=Status.SKIPPED,
            reason=outcome.reason,
            duration_ms=0.0,
            request={},
            decoded={"attempted": list(outcome.attempted)} if outcome.attempted else {},
            expected={"missing": list(outcome.missing)},
            evidence_dir=None,
            seed=seed,
        )
        for outcome in resolution.skipped
    )

