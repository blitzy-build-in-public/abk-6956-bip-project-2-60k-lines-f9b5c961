"""Resolve a check selection against what a target and a host can actually do.

Nothing in qrcheck may decide *while running* that a check cannot run: a check that
turned out to be impossible half-way through a case would produce a failure that
describes the host rather than the target, and a check that was quietly dropped would
let a run look complete when it covered less.  This module makes that decision once,
before execution, from the two facts the runner already holds:

* the capabilities the adapter declares (its ``capabilities`` set), intersected with
  each :class:`~qrcheck.registry.check.CheckContract`'s own ``requires`` set; and
* the availability of the optional runtime dependencies the host provides, keyed by
  the :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` tokens a contract declares in
  its ``dependencies``.

Its output is a :class:`Resolution`: the checks that will run and the checks that will
not, each skip carrying the reason and the exact requirements it lacks.

A shortfall is never an exception
---------------------------------
Every shortfall becomes a :class:`SkipOutcome`, and the runner turns those into
``skipped`` :class:`~qrcheck.report.model.CaseResult` records through
:func:`skip_case_results`.  Nothing here raises because a target cannot do something,
and nothing here decides an exit code.  Exactly two reasons exist, and this module
produces no other:

* :attr:`~qrcheck.report.model.Reason.UNSUPPORTED_CAPABILITY` -- the target did not
  declare a capability the check requires.  The configuration is valid; this target
  simply cannot serve this check, which is why it is a skip rather than the exit-code-2
  configuration error an unknown capability *name* is (the configuration schema
  rejects that far earlier, where it is wrong in itself).
* :attr:`~qrcheck.report.model.Reason.MISSING_DEPENDENCY` -- the check needs an
  optional runtime dependency the host does not provide (the SVG rasterizer), so its
  assertion could not be made at all.  The platform file-descriptor directory is not
  one of these: that check reports the handle half per case as
  ``unsupported_platform`` while its temporary-file half still asserts (AAP §0.8.1),
  which a whole-check skip here would drop.

A capability shortfall outranks a missing dependency: a check whose target cannot even
produce the artefact would not become runnable by installing the missing tool, so one
skip per check is the honest description and the capability reason is the true one.

A skip is a real outcome, never a flavoured pass: it is counted in its own integer of
:class:`~qrcheck.report.model.Counts` and can never produce a passing exit code.  The
reference target declares all thirteen capabilities, so a conforming run reaches zero
skips, which is what keeps this module from being a hiding place.

The gap this closes against the adapter's own gate
--------------------------------------------------
:func:`qrcheck.targets.base.require_capabilities` refuses an unsupported request before
any I/O, and it stays: it is the adapter's own guard for a request that reaches it by
any other route.  This module is the pre-execution path that turns the same shortfall
into a result a report can show, so the two mechanisms answer different questions and
neither may be removed in favour of the other.

Determinism
-----------
Both tuples of a :class:`Resolution` are ordered by check identifier ascending, the
``missing`` tuple of a skip is sorted and de-duplicated, and no input is mutated, so
two resolutions of the same selection describe it identically.  Nothing here reads the
clock, the environment, the filesystem, the network or a random source.

Layering
--------
The only runtime imports are :mod:`qrcheck.registry.check` -- which owns check identity,
the dependency vocabulary and the capability type -- and :mod:`qrcheck.report.model`,
which owns the status and reason words.  Neither the check modules, nor an adapter
implementation, nor the decode layer is imported here: requirements travel on the
contract and availability arrives as data, so this module has nothing to keep in step
when a fourth adapter kind or a fifth check family appears.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

# ``Capability`` comes from the registry module's own import of the adapter port's
# enum, not from ``qrcheck.targets.base`` directly: this module's runtime dependencies
# are the registry (which owns check identity, the dependency vocabulary and the
# capability type its contracts already carry) and the result model, so it never grows
# a second edge into the adapter layer to reach a type it only annotates.
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
# One string, defined once, because three artefacts key on it: ``summary.json``
# carries it as ``case_id``, ``junit.xml`` renders it inside
# ``name="<CHECK-ID>[<case_id>]"`` and the runner accumulates one such result per
# unrunnable check.  It is deliberately distinct from the runner's own
# ``synthetic:target`` case identifier, which describes a *different* condition -- a
# target that was reachable when the run started and then vanished -- so a reader can
# tell "this target never promised this" from "this target broke".
CAPABILITY_CASE_ID: Final[str] = "synthetic:capability"

# The only two reasons a resolution may skip a check with.  Anything else -- an
# unsupported platform, a decoder disagreement, a target rejection -- describes
# something that happens *to a case while it runs*, and a resolution that claimed one
# would be inventing an observation it never made.
_SKIP_REASONS: Final[frozenset[Reason]] = frozenset(
    {Reason.UNSUPPORTED_CAPABILITY, Reason.MISSING_DEPENDENCY}
)


class _DeclaresCapabilities(Protocol):  # pragma: no cover - declaration, never executed
    """The one attribute resolution reads from an adapter.

    Declared so the accepted input can be annotated precisely without importing the
    adapter port: this module needs to know that *something* publishes a
    ``capabilities`` collection and nothing else about it, so an adapter, a resolved
    target record or a test double all satisfy this structurally.
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


# What the ``capabilities`` argument may be: a collection of members or their names
# (what the configuration resolves to, and what a test passes), or any object that
# declares a ``capabilities`` attribute (an adapter -- what the runner passes).  A
# member of :class:`Capability` is itself a string, so a bare name is read as one
# capability rather than as an iterable of characters.
CapabilitySource: TypeAlias = Iterable[Capability | str] | _DeclaresCapabilities | None

# What the ``availability`` argument may be: a mapping keyed by the
# :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` values (what the runner builds
# from its probes), an object that exposes ``as_mapping()`` or ``available(token)``,
# or ``None`` for "nothing beyond the mandatory dependencies is available".
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


@dataclass(frozen=True, slots=True)
class SkipOutcome:
    """One check that will not execute, why, and which requirements it lacks.

    Three fields, each a public surface that reaches an artefact: ``check_id`` names
    the check, ``reason`` is the machine-readable cause ``summary.json`` and the JUnit
    ``<skipped message="..."/>`` carry, and ``missing`` is the list of absent
    requirements -- capability names for an ``unsupported_capability`` skip, dependency
    tokens for a ``missing_dependency`` one -- so a reader learns what to add rather
    than only that something was skipped.

    Construction normalises and validates rather than trusting the caller: ``missing``
    is sorted and de-duplicated (two resolutions of one selection then describe it
    identically), ``reason`` is narrowed to a :class:`~qrcheck.report.model.Reason`
    member, and a skip that names no cause is refused.  An unexplained skip would be
    exactly the quiet gap the zero-skip reference gate exists to catch, and it would
    also be the shape a weakened check takes when someone tries to silence it.

    Raises:
        ValueError: ``check_id`` is empty, ``reason`` is not one of the two skip
            reasons, or ``missing`` is empty or not a collection of names.
    """

    check_id: str
    reason: Reason
    missing: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the record and normalise ``reason`` and ``missing``.

        Raises:
            ValueError: ``check_id`` is not a non-empty ``str``, ``reason`` is outside
                :class:`~qrcheck.report.model.Reason` or is not one of the two skip
                reasons, or ``missing`` is empty or is not a collection of names.
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
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "missing", missing)


@dataclass(frozen=True)
class Resolution:
    """The selection split into what will run and what will not, ordered by check id.

    Two tuples, and a check is in exactly one of them: a check is runnable or skipped,
    never both and never twice, which :meth:`__post_init__` enforces because a double
    entry would make the runner execute a check once while a report counted it twice.
    Both tuples are sorted by check identifier ascending on construction, so import
    order, set iteration and the order the caller happened to pass cannot reach
    ``summary.json``, ``junit.xml`` or the HTML report through this object.

    The instance is frozen and the two fields are tuples rather than lists so a caller
    cannot append to a resolution it has been handed -- the runner decides execution
    from this record, and a mutated copy would let the executed set and the reported
    set disagree.

    Raises:
        ValueError: One check identifier appears more than once across ``runnable`` and
            ``skipped``.
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

        The clean-run gate asks exactly this of a conforming target: a run whose
        resolution is complete cannot report a skip, so a check cannot be made to pass
        by making it skip.
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

    Three accepted forms, because the same fact arrives from three places: the
    configuration and the tests pass a collection of members or their names, the runner
    passes the adapter itself (which publishes ``capabilities``), and ``None`` stands
    for a declaration nobody made.  ``None`` resolves to the empty set -- the safe
    direction -- so a forgotten declaration surfaces as a skip the zero-skip reference
    gate fails loudly, rather than as a check that ran and counted as a pass.

    Args:
        source: The collection, the declaring object, or ``None``.

    Returns:
        The declared capabilities as members, empty when nothing was declared.

    Raises:
        TypeError: `source` is a string, or is neither a collection nor an object
            declaring ``capabilities``.  This is a defect in the caller, not a target
            condition, so it is named rather than turned into a skip.
        ValueError: A declared name is not a :class:`Capability` member.  The
            configuration schema rejects an unknown member long before this point, so
            reaching it here means a declaration was built by hand.
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

    Three accepted forms plus ``None``: a mapping keyed by the
    :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` values (what the runner builds
    from its probes), an object exposing ``as_mapping()``, an object exposing
    ``available(dependency)`` (queried once per token), and ``None`` for "nothing
    available".  An **absent token means unavailable** -- this module never assumes a
    dependency nobody probed, so a forgotten probe becomes a skip the zero-skip
    reference gate catches instead of a check that ran against something that was never
    there.

    Args:
        source: The mapping, the querying object, or ``None``.

    Returns:
        A mapping that answers ``get(token, False)`` for every token.

    Raises:
        TypeError: `source` is none of the accepted forms, or its ``as_mapping()``
            returned something other than a mapping.
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


def unmet_capabilities(
    contract: CheckContract,
    capabilities: CapabilitySource = None,
) -> tuple[Capability, ...]:
    """Return the capabilities `contract` requires that `capabilities` does not declare.

    Requirement resolution in one direction: what a check needs, read from the
    contract's own ``requires`` set, against what the target positively declared.
    Nothing is required that the contract did not ask for, so the table of requirements
    exists only in the nine check family modules and cannot drift from them.

    Args:
        contract: The check whose requirements are compared.
        capabilities: The declared capabilities -- a collection of members or their
            names, an object declaring ``capabilities``, or ``None`` for none declared.

    Returns:
        The unmet members, sorted by value, empty when the declaration satisfies the
        contract.

    Raises:
        TypeError: `capabilities` is not one of the accepted forms.
        ValueError: A declared name is not a :class:`Capability` member.
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

    A token the mapping does not mention is unavailable: the default direction is the
    safe one, so an unfilled probe skips the check instead of letting it run without the
    tool it asserts with.

    Args:
        contract: The check whose ``dependencies`` are looked up.
        availability: The availability mapping, a querying object, or ``None`` for
            nothing available.

    Returns:
        The absent tokens, sorted, empty when every declared dependency is available.

    Raises:
        TypeError: `availability` is not one of the accepted forms.
    """
    resolved = _available_dependencies(availability)
    return tuple(
        sorted(token for token in contract.dependencies if not resolved.get(token, False))
    )


def resolve(
    selected: Iterable[CheckContract] | Mapping[str, CheckContract],
    capabilities: CapabilitySource = None,
    availability: AvailabilitySource = None,
) -> Resolution:
    """Split `selected` into the checks that can run and the checks that cannot.

    The single pre-execution decision: every selected check is compared against the
    declared capabilities and the resolved dependency availability, and lands in exactly
    one tuple of the returned :class:`Resolution`.  A shortfall never raises -- it
    becomes a :class:`SkipOutcome` with one of two reasons, and no other:

    * ``unsupported_capability`` when the target does not declare something the check
      requires;
    * ``missing_dependency`` when a declared optional runtime dependency is unavailable.

    A capability shortfall outranks a missing dependency, so a check that is short of
    both is skipped once, with the capability reason.

    Accepted forms, so one function serves the runner, a test and a CLI probe:

    * ``capabilities`` -- a collection of :class:`Capability` members or their names, an
      object declaring ``capabilities`` (an adapter, which is what the runner passes),
      or ``None`` for none declared;
    * ``availability`` -- a mapping keyed by the
      :data:`~qrcheck.registry.check.DEPENDENCY_TOKENS` values, an object exposing
      ``as_mapping()`` or ``available(dependency)``, or ``None`` for nothing available.
      :class:`qrcheck.decode.base.DecoderAvailability` exposes neither method, so the
      runner translates its probes into
      ``{DEPENDENCY_PRIMARY_DECODER: ..., DEPENDENCY_SECONDARY_DECODER: ...}`` before
      calling here.

    Nothing is mutated, no adapter is touched, no probe is run and no result is built:
    the outcome is a pure function of the three arguments, which is what makes the
    executed set knowable before the first case runs.

    Args:
        selected: The contracts to resolve, in any order and from any iterable, or a
            mapping of identifier to contract (such as the registry).  Duplicates are
            collapsed, keeping the first occurrence.
        capabilities: The target's declared capabilities, in any accepted form.
        availability: The dependency availability, in any accepted form.

    Returns:
        The resolution, both tuples sorted by check identifier.

    Raises:
        TypeError: `selected` yields something that is not a
            :class:`~qrcheck.registry.check.CheckContract`, or `capabilities` or
            `availability` is not in an accepted form.  Both are defects in the caller.
        ValueError: A declared capability name is not a :class:`Capability` member.
    """
    declared = _declared_capabilities(capabilities)
    available = _available_dependencies(availability)
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
                )
            )
            continue
        runnable.append(contract)
    return Resolution(runnable=tuple(runnable), skipped=tuple(skipped))


def skip_case_results(resolution: Resolution, seed: int) -> tuple[CaseResult, ...]:
    """Return the ``skipped`` results a run accumulates for its unrunnable checks.

    One :class:`~qrcheck.report.model.CaseResult` per :class:`SkipOutcome`, in check
    identifier order, and nothing else: the runner adds them to the run's results beside
    the executed cases so that a selection covering a check the target cannot serve is
    described rather than silently shortened.  Each carries
    :attr:`~qrcheck.report.model.Status.SKIPPED`, one of the two skip reasons,
    ``duration_ms`` of ``0.0`` (nothing ran), no evidence directory and the run's seed.

    There is no evidence to persist: no request was built and no symbol exists, so
    :class:`qrcheck.report.evidence.EvidenceStore` is deliberately not called and no
    directory is invented for a case that never produced an artefact.  The
    machine-readable cause is the record itself -- ``status``, ``reason`` and
    ``expected["missing"]`` -- which reaches ``summary.json``, the JUnit
    ``<skipped message="..."/>`` element and the skipped section of the HTML report.
    ``request`` and ``decoded`` are empty mappings rather than nulls because the record
    declares both as mappings, so a skip serialises as an empty object that every
    consumer can read uniformly instead of a null each of them would special-case.

    A skipped result is never a pass: this function sets no passing status, and the
    four separate tallies of :class:`qrcheck.report.model.Counts` keep it out of
    ``passed`` so a run that skipped a check cannot look green because of it.

    Args:
        resolution: The resolution whose skips become results.
        seed: The run's seed, recorded verbatim so the report says which run this skip
            belongs to.

    Returns:
        The skipped results, sorted by check identifier, empty when nothing was skipped.
    """
    return tuple(
        CaseResult(
            check_id=outcome.check_id,
            case_id=CAPABILITY_CASE_ID,
            status=Status.SKIPPED,
            reason=outcome.reason,
            duration_ms=0.0,
            request={},
            decoded={},
            expected={"missing": list(outcome.missing)},
            evidence_dir=None,
            seed=seed,
        )
        for outcome in resolution.skipped
    )

