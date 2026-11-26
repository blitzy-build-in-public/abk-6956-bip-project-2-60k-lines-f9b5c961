"""Determinism: one identical request, asked twice, must come back as one identical symbol.

Plan item served (AAP §0.8.1, the component-to-check-family table and the case-matrix row
for ``DT-DETERMINISTIC``, and §0.4.5, "defects observable only indirectly"): this module
owns the product's only check of *reproducibility*. A symbol may be valid, in range and
decodable in each of two calls and still be a different symbol, so the defect is invisible
unless a pair of calls is compared -- which is exactly what this check does and the only
thing it does.

No project rules exist for this product: the review-rules source returns no user rules and
AAP §0.12 records the same, so the binding constraints are the AAP's own MUST/MUST NOT
list, honoured here as if they were rules. Two of them shape this file: nothing under
``qrcheck/checks/`` imports a concrete adapter (this module reaches the port's
:class:`~qrcheck.targets.base.Capability` and its error vocabulary, and no adapter), and
an adapter raises only the five declared error types, none of which this module converts
into a different one.

What the check asserts
----------------------
Two invocations of the **identical** request must agree on all three of

* the artefact's image bytes, compared as ``symbol_sha256`` per call;
* the sampled module matrix, compared as :attr:`~qrcheck.decode.base.DecodeResult.matrix_sha256`;
* the mask read out of the symbol's 15-bit format information,
  :attr:`~qrcheck.decode.base.DecodeResult.mask`.

All three, because each catches something the others cannot. A byte comparison alone
would pass a target whose writer is deterministic while the symbol it wrote was not, and
would fail a target whose writer embeds a chunk a symbol need not repeat. The mask is the
one value an encoder chooses freely per call -- it is what
:mod:`qrcheck.targets.defective`'s ``non_deterministic_mask`` injection changes, and a
single symbol can never reveal it. The matrix digest is the structural statement the
format information corroborates semantically.

Reasons emitted, and nothing else
---------------------------------
``Reason.VALUE_MISMATCH``
    At least one component that *could* be measured differs between the two calls.
    ``decoded["divergent"]`` names which; ``decoded["comparison"]`` carries the
    per-component verdict, and both calls' full projections travel beside it, so evidence
    says which of the three diverged rather than only that something did.
``Reason.SAMPLING_FAILED``
    Nothing diverged, but at least one component could not be measured: a call's matrix or
    mask is absent because the module grid could not be recovered, and that call's
    ``sampling`` entry carries the estimate's own reason. The case fails rather than
    passing on the components that were readable, because "identical" cannot be claimed
    over a component nobody read. A proven divergence always wins over this reason, so an
    unmeasurable component never masks a real defect.

``TargetRejected``, ``TargetContractError`` and ``TargetUnsupported`` go through
:func:`qrcheck.checks.support.case_status_for_error`, the product's single translation, so
this check reports for them exactly what every other check reports.
``TargetUnavailable`` and ``TargetTimeout`` propagate unchanged: the
one-error-result-per-check target-loss policy belongs to :mod:`qrcheck.runner`, which can
only apply it if it is the layer that sees the failure. No other reason is constructed
here, and no ``error`` result is ever built -- that status is the runner's alone.

Exactly two invocations per case, and never a third
---------------------------------------------------
Six declared payloads, each asked for twice: **12 encodes and 12 decodes** per full run,
which :data:`EXPECTED_ENCODES` declares and ``tests/unit/test_checks_negative.py``
asserts against a counting stub (AAP §0.4.3, §0.5.6). The multiple per case is **2**, and
it is not negotiable: a third confirming call, a retry loop or a "re-ask until stable"
would register as the declared count plus one, fail that test, and destroy the only
property this check proves -- that the framework asks once and believes the answer. There
is no retry, no re-run, no quarantine and no relaxed threshold anywhere in this module,
and the second call is never skipped because the first one looked wrong: a QR check that
fails intermittently is a framework defect to fix, never one to suppress.

What this module must never own, and therefore never restates
-------------------------------------------------------------
Request construction and validation (:func:`qrcheck.checks.support.build`, itself a
delegation to the port's builder), the generation and decode path
(:func:`~qrcheck.checks.support.generate` and :func:`~qrcheck.checks.support.decode`), the
result vocabulary and the record shape (:mod:`qrcheck.report.model`), the evidence layout
and its closed artefact set (:mod:`qrcheck.report.evidence` through
:func:`qrcheck.checks.support.record_evidence`), and every threshold -- this check asserts
no duration, no size and no limit, so it reads no configuration at all and carries no
numeric tolerance. The one artefact name it is responsible for choosing is the second
symbol, which reaches disk as ``symbol-repeat.png`` through ``repeat_symbol`` and exists
for this check alone.

Determinism and cost
--------------------
Nothing here is random, so nothing here derives a seed: the payloads are module constants,
the case order is ascending by case identifier, and the run seed is recorded on each result
only so a case can be replayed against the same run. No clock value reaches an assertion --
the two calls' duration is recorded, and that is all it is for. No ``hash()``, ``id()``,
``random`` or wall-clock reading decides anything, and every digest is ``sha256``.

Import cost is a contract this module keeps: it imports the standard library, the check
registry, the result model and :mod:`qrcheck.checks.support`, and it does **not** import
:mod:`qrcheck.decode` at run time -- that package loads ``numpy``, and ``qrcheck list``
must not pay for a native image stack to print twenty-one check titles. The decode types
are therefore imported for their annotations only, which is the convention
:mod:`qrcheck.checks.support` already follows.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from qrcheck.checks import support
from qrcheck.registry.check import DEPENDENCY_PRIMARY_DECODER, Case, RunContext, check
from qrcheck.report.model import CaseResult, Reason

# The port, never an adapter. ``Capability`` is what the check's requirement is declared
# with, and the three error types are the closed vocabulary its handler must name to route
# them through the shared translation; ``TargetUnavailable`` and ``TargetTimeout`` are
# deliberately absent from that import, because catching them would stop the runner from
# applying its target-loss policy.
from qrcheck.targets.base import (
    Capability,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Annotations only: see the import-cost note in the module docstring.
    from qrcheck.decode import DecodeResult

__all__ = [
    "CHECK_ID",
    "DETERMINISM_PAYLOADS",
    "EXPECTED_ENCODES",
    "cases",
]

# The check's stable identifier: one of the twenty-one in
# :data:`qrcheck.registry.check.CHECK_IDS`, and part of every failure identifier, evidence
# path and JUnit testcase name this file produces.
CHECK_ID: Final[str] = "DT-DETERMINISTIC"

# The case-identifier prefix AAP §0.3.4 fixes for a case the framework constructs rather
# than draws from the corpus. ``checks/support.py`` owns the rule; this module only states
# the identifier it builds with it.
_SYNTHETIC_PREFIX: Final[str] = "synthetic:"

# The six payloads, one per case, in the order the case identifiers number them. Declared
# constants rather than anything random: this check compares two calls of one request, so a
# payload it had to derive would make the pair prove the derivation instead of the target.
# Every payload is short ASCII that fits every error level comfortably -- a structural
# question such as capacity is another check's business -- and the six differ in length and
# in character class (digits, alphanumerics, a URL, a run, punctuation) so a target that
# mishandles only one mode cannot make all six agree by accident.
DETERMINISM_PAYLOADS: Final[tuple[tuple[str, bytes], ...]] = (
    ("determinism-1", b"qrcheck determinism probe 1"),
    ("determinism-2", b"0123456789012345"),
    ("determinism-3", b"QRCHECK DETERMINISM 3"),
    ("determinism-4", b"https://example.invalid/qrcheck/dt/4"),
    ("determinism-5", b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    ("determinism-6", b"qrcheck/determinism/6: identical twice?"),
)

# How many times one case asks the target for the identical request. Two is the assertion
# itself, not a sample size: a determinism claim needs a pair, and a third call would be a
# retry (see the module docstring).
_INVOCATIONS_PER_CASE: Final[int] = 2

# The declared adapter-invocation count of this check, in the form AAP §0.4.3 fixes for the
# case matrix and ``tests/unit/test_checks_negative.py`` asserts with a counting stub:
# one entry, twelve calls, derived from the payload table above so the two cannot disagree.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {CHECK_ID: _INVOCATIONS_PER_CASE * len(DETERMINISM_PAYLOADS)}
)

# The three components the two calls are compared component by component, in the order
# ``decoded["comparison"]`` states them. The artefact digest is compared as
# ``symbol_sha256`` -- the snapshot key -- because the two calls' bytes are never stored
# twice in the evidence.
_COMPARED_COMPONENTS: Final[tuple[str, ...]] = ("image_bytes", "matrix_sha256", "mask")

# Which snapshot field carries each compared component. A component is named for what is
# compared -- an artefact's bytes, a matrix, a mask -- and the snapshot stores digests, so
# the two vocabularies are related here once instead of in three comparisons.
_COMPARISON_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "image_bytes": "symbol_sha256",
        "matrix_sha256": "matrix_sha256",
        "mask": "mask",
    }
)

# Which fields of the decoded evidence travel per call. Taken from the decode layer's own
# projection (:meth:`qrcheck.decode.base.DecodeResult.as_dict`) rather than rebuilt here, so
# the per-decoder outcome shape has exactly one definition in the product. The documented
# evidence set is covered: ``per_decoder``, ``agreement``, both EC-level readings,
# ``version``, ``mask`` and ``matrix_sha256``.
_SNAPSHOT_KEYS: Final[tuple[str, ...]] = (
    "agreement",
    "ec_level_decoder",
    "ec_level_format",
    "mask",
    "matrix_sha256",
    "per_decoder",
    "polarity_applied",
    "sampling",
    "version",
)


def _case_name(entry: tuple[str, bytes]) -> str:
    """Return one declared payload's case name: the sort key of the case list.

    The single sort key in this module, so "ascending by case identifier" is one order here
    and in every report: a case identifier is ``synthetic:`` plus this name, so sorting the
    pairs by name sorts the cases by identifier.
    """
    return entry[0]


@check(
    id=CHECK_ID,
    title="Identical requests produce identical symbols",
    family="determinism",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def cases(ctx: RunContext) -> Iterator[Case]:
    """Yield this check's six constructed cases, ascending by case identifier.

    One case per declared payload, named ``synthetic:determinism-<n>``: constructed rather
    than drawn from the corpus, because the observation needs a second artefact for one
    request and no corpus record carries one.

    ``requires`` is ``png`` alone. Every request this check builds is a default PNG with no
    module-size, quiet-zone, colour, version or mask pin, so no other capability is
    exercised and ``qrcheck/registry/resolve.py`` cannot turn the check into a skip over a
    capability it never uses. ``dependencies`` declares the primary decoder, which is the
    authority all twenty-one checks read their payload through.

    Args:
        ctx: The run's context. Nothing is read from it here -- the cases are constant, and
            the EC level and seed are read when a case executes -- but the signature is the
            registry's :data:`~qrcheck.registry.check.CasesFn`.

    Yields:
        Six :class:`~qrcheck.registry.check.Case` objects in ascending identifier order.
    """
    for name, payload in sorted(DETERMINISM_PAYLOADS, key=_case_name):
        case_id = f"{_SYNTHETIC_PREFIX}{name}"
        yield support.synthetic_case(CHECK_ID, case_id, _executor(case_id, payload))


def _executor(case_id: str, payload: bytes) -> Callable[[RunContext], CaseResult]:
    """Return the callable that executes one case, closing over its identifier and payload.

    Built per case rather than once per check, so each callable carries its own payload and
    the runner never looks either up.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run this case against the run's target."""
        return _execute(ctx, case_id, payload)

    return execute


def _execute(ctx: RunContext, case_id: str, payload: bytes) -> CaseResult:
    """Ask for `payload` twice, decode each artefact once, and compare the three components.

    The two calls are made with **one** :class:`~qrcheck.targets.base.GenerateRequest`
    instance, so "identical" is a property of the object rather than of two builders that
    happen to agree. Each artefact is decoded once, through the framework's one decode path,
    which keeps the primary decoder the single authority and gives the case its matrix and
    mask without a second opinion anywhere.

    Evidence is written through :func:`qrcheck.checks.support.record_evidence`, which
    persists nothing for a passing case: a failing case's directory holds the first
    artefact as ``symbol.png`` and the second as ``symbol-repeat.png``, which is what makes
    a mask that changed between calls inspectable rather than merely reported.

    Args:
        ctx: The run's context: the adapter, the error level, the seed and the evidence
            store. Configuration is not read, because this check asserts no limit.
        case_id: The case's stable identifier, one of ``synthetic:determinism-<n>``.
        payload: The declared payload this case asks for twice.

    Returns:
        A passed result, or a failed one carrying ``value_mismatch`` or ``sampling_failed``
        with the comparison and both calls' projections as its evidence.

    Raises:
        ValueError: The payload does not fit the symbol at the run's error level, or that
            level is outside ``L``/``M``/``Q``/``H``. Both are framework-side defects in the
            request builder rather than verdicts about the target, so they propagate for the
            runner to record as an ``error`` result with its traceback.
        TargetUnavailable: Propagated unchanged, for the runner's target-loss policy.
        TargetTimeout: Propagated unchanged, for the same reason.
    """
    request = support.build(payload, ctx.ec_level)
    first_symbol = None
    second_symbol = None
    started = time.perf_counter()
    try:
        first_symbol = support.generate(ctx, request)
        first_decoded = support.decode(ctx, first_symbol.image_bytes, request)
        second_symbol = support.generate(ctx, request)
        second_decoded = support.decode(ctx, second_symbol.image_bytes, request)
    except (TargetRejected, TargetContractError, TargetUnsupported) as error:
        # The three errors a per-case verdict can be made of; the shared translation owns
        # which status each becomes, so this check reports them exactly as every other does.
        outcome = support.case_status_for_error(
            CHECK_ID,
            case_id,
            error,
            request=request,
            duration_ms=_elapsed_ms(started),
            seed=ctx.seed,
        )
        return support.record_evidence(
            ctx, outcome, symbol=first_symbol, repeat_symbol=second_symbol
        )

    duration_ms = _elapsed_ms(started)
    first = _snapshot(_sha256(first_symbol.image_bytes), first_decoded)
    second = _snapshot(_sha256(second_symbol.image_bytes), second_decoded)
    comparisons = _comparisons(first, second)
    divergent = tuple(
        component for component, agreed in comparisons.items() if agreed is False
    )
    unmeasured = tuple(
        component for component, agreed in comparisons.items() if agreed is None
    )
    decoded = _evidence(first, second, comparisons, divergent, unmeasured)
    expected: dict[str, Any] = {
        "identical": list(_COMPARED_COMPONENTS),
        "invocations": _INVOCATIONS_PER_CASE,
    }
    request_projection = request.as_dict()

    if divergent:
        # A component that was read and disagreed outranks one that could not be read, so an
        # unsamplable second call can never hide a mask that changed between the two.
        outcome = support.fail(
            CHECK_ID,
            case_id,
            reason=Reason.VALUE_MISMATCH,
            request=request_projection,
            decoded=decoded,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    elif unmeasured:
        outcome = support.fail(
            CHECK_ID,
            case_id,
            reason=Reason.SAMPLING_FAILED,
            request=request_projection,
            decoded=decoded,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    else:
        outcome = support.pass_(
            CHECK_ID,
            case_id,
            request=request_projection,
            decoded=decoded,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
    return support.record_evidence(
        ctx, outcome, symbol=first_symbol, repeat_symbol=second_symbol
    )


def _snapshot(symbol_sha256: str, decoded: DecodeResult) -> dict[str, Any]:
    """Project one call into the fields the comparison and the evidence need.

    The decode layer's own projection is the source -- so the per-decoder outcome shape has
    one definition in the product -- and this function adds the two facts the decode layer
    cannot know: the artefact's digest, which is compared as the image bytes, and the
    authoritative payload's length and digest, which let a reader see whether the two calls
    even decoded to the same bytes.

    Args:
        symbol_sha256: ``sha256`` of the artefact's bytes, exactly as the target returned it.
        decoded: That artefact's decoded evidence.

    Returns:
        A JSON-ready mapping: the snapshot fields of the decode projection plus
        ``symbol_sha256``, ``payload_len`` and ``payload_sha256``, the last two ``None`` when
        the primary decoder produced no payload.
    """
    projection = decoded.as_dict()
    snapshot: dict[str, Any] = {key: projection[key] for key in _SNAPSHOT_KEYS}
    payload = decoded.payload_bytes
    snapshot["payload_len"] = None if payload is None else len(payload)
    snapshot["payload_sha256"] = None if payload is None else _sha256(payload)
    snapshot["symbol_sha256"] = symbol_sha256
    return snapshot


def _comparisons(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, bool | None]:
    """Compare the two calls component by component, in the declared component order.

    Args:
        first: The first call's snapshot.
        second: The second call's snapshot, under the same keys.

    Returns:
        One entry per :data:`_COMPARED_COMPONENTS` name: ``True`` when both readings agree,
        ``False`` when they differ, and ``None`` when either could not be read at all.
    """
    return {
        component: _agrees(
            first[_COMPARISON_KEYS[component]], second[_COMPARISON_KEYS[component]]
        )
        for component in _COMPARED_COMPONENTS
    }


def _agrees(left: object, right: object) -> bool | None:
    """Return whether two readings agree, or ``None`` when either could not be read.

    An absent reading is not an agreement: a matrix nobody sampled and a mask the format
    information did not yield say nothing about determinism, and reporting them as equal
    would let the case pass on the components that happened to be readable.
    """
    if left is None or right is None:
        return None
    return left == right


def _evidence(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    comparisons: Mapping[str, bool | None],
    divergent: tuple[str, ...],
    unmeasured: tuple[str, ...],
) -> dict[str, Any]:
    """Assemble the ``decoded`` mapping the result and ``decoded.json`` carry.

    The first call's projection is the top-level one, so the documented evidence shape --
    ``per_decoder``, ``agreement``, both EC-level readings, ``version``, ``mask`` and
    ``matrix_sha256`` -- is present where a reader of any other check's evidence finds it,
    and the second call travels whole under ``repeat``. That placement is also what keeps
    the HTML report's single-decoder banner working: it looks for a top-level
    ``per_decoder`` map, and a run in which this check was the only one that decoded would
    otherwise report no degradation at all.

    Args:
        first: The first call's snapshot.
        second: The second call's snapshot.
        comparisons: The per-component verdicts.
        divergent: The components that were read and disagreed.
        unmeasured: The components that could not be read.

    Returns:
        The evidence mapping: the first snapshot's fields, ``repeat``, ``comparison``,
        ``divergent`` and ``unmeasured``.
    """
    evidence = dict(first)
    evidence["comparison"] = dict(comparisons)
    evidence["divergent"] = list(divergent)
    evidence["repeat"] = dict(second)
    evidence["unmeasured"] = list(unmeasured)
    return evidence


def _sha256(data: bytes) -> str:
    """Return the lowercase hex ``sha256`` digest of `data`.

    ``sha256`` and not ``hash()``: a report keys on these digests, and ``hash()`` is salted
    per process, so it would make two runs of one case describe it differently.
    """
    return hashlib.sha256(data).hexdigest()


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, as a recorded value only.

    A duration is recorded and never asserted -- no comparison, no threshold and no report
    ordering reads it (AAP §0.5.5) -- so this helper exists to fill the result's field and
    for nothing else.
    """
    return (time.perf_counter() - started) * 1000.0

