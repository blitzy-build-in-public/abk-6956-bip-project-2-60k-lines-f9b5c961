"""Determinism: one identical request, asked twice, must come back as one identical symbol.

``DT-DETERMINISTIC`` is the product's only check of *reproducibility*: a symbol may be valid, in
range and decodable in each of two calls and still be a different symbol, so the defect is
invisible unless a pair of calls is compared.

Two invocations of the **identical** request must agree on all three of the artefact's image bytes
(``symbol_sha256`` per call), the sampled module matrix (``matrix_sha256``) and the mask read out of
the
15-bit format information. All three, because each catches what the others cannot: a byte comparison
alone would pass a target whose writer is deterministic while the symbol it wrote was not, and the
mask is the one value an encoder chooses freely per call.

Reasons emitted, and nothing else: ``Reason.VALUE_MISMATCH`` when a component that could be measured
differs (``decoded["divergent"]`` names which), and ``Reason.SAMPLING_FAILED`` when nothing diverged
but a component could not be measured -- and a proven divergence always wins over that.
``TargetRejected``, ``TargetContractError`` and ``TargetUnsupported`` go through
:func:`qrcheck.checks.support.case_status_for_error`; ``TargetUnavailable`` and ``TargetTimeout``
propagate unchanged, because the target-loss policy belongs to :mod:`qrcheck.runner`.

Exactly two invocations per case, never a third: six declared payloads asked for twice are **12
encodes and 12 decodes** per full run, which :data:`EXPECTED_ENCODES` declares and
``tests/unit/test_checks_negative.py`` asserts against a counting stub.
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

# The port, never an adapter. ``Capability`` declares the check's requirement, and the three
# error types are the closed vocabulary its handler routes through the shared translation;
# ``TargetUnavailable`` and ``TargetTimeout`` are deliberately not imported, because catching them
# would stop the runner applying its target-loss policy.
from qrcheck.targets.base import (
    Capability,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Annotations only: qrcheck.decode loads numpy for its decoders, and importing
    # this module must not start a native image stack in a process that only lists checks.
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

# The case-identifier prefix for a case the framework constructs rather than draws from the
# corpus. ``checks/support.py`` owns the rule; this module only states the identifier it builds
# with it.
_SYNTHETIC_PREFIX: Final[str] = "synthetic:"

# The six payloads, one per case, in the order the case identifiers number them. Declared
# constants rather than anything derived: this check compares two calls of one request, so a derived
# payload would make the pair prove the derivation instead of the target.  Each is short ASCII that
# fits every error level, and the six differ in length and character class (digits, alphanumerics, a
# URL, a run, punctuation) so a target mishandling one mode cannot make all six agree by accident.
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

# The declared adapter-invocation count of this check, published as ``EXPECTED_ENCODES`` so
# ``tests/unit/test_checks_negative.py`` can assert it with a counting stub: one entry, twelve
# calls, derived from the payload table above so the two cannot disagree.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {CHECK_ID: _INVOCATIONS_PER_CASE * len(DETERMINISM_PAYLOADS)}
)

# The three components the two calls are compared by, in the order
# ``decoded["comparison"]`` states them.  The artefact digest is compared as ``symbol_sha256``,
# the snapshot key, because two calls' bytes are never stored twice in the evidence.
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

# Which fields of the decoded evidence travel per call, taken from the decode layer's own
# projection (:meth:`qrcheck.decode.base.DecodeResult.as_dict`) so the per-decoder outcome shape has
# one definition: ``per_decoder``, ``agreement``, both EC-level readings, ``version``, ``mask`` and
# ``matrix_sha256``.
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

    A case identifier is ``synthetic:`` plus this name, so sorting the pairs by name sorts the cases
    by identifier, and this is the one sort key in the module.
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

    One case per declared payload, named ``synthetic:determinism-<n>``: constructed rather than
    drawn from the corpus, because the observation needs a second artefact for one request.
    ``requires`` is ``png`` alone -- every request is a default PNG with no pin -- so no other
    capability is exercised and resolution cannot turn the check into a skip. ``dependencies``
    declares the primary decoder.

    Yields:
        Six :class:`~qrcheck.registry.check.Case` objects in ascending identifier order.
    """
    for name, payload in sorted(DETERMINISM_PAYLOADS, key=_case_name):
        case_id = f"{_SYNTHETIC_PREFIX}{name}"
        yield support.synthetic_case(CHECK_ID, case_id, _executor(case_id, payload))


def _executor(case_id: str, payload: bytes) -> Callable[[RunContext], CaseResult]:
    """Return the callable that executes one case, closing over its identifier and payload.

    Built per case rather than once per check, so each callable carries its own payload and the
    runner never looks either up.
    """

    def execute(ctx: RunContext) -> CaseResult:
        """Run this case against the run's target."""
        return _execute(ctx, case_id, payload)

    return execute


def _execute(ctx: RunContext, case_id: str, payload: bytes) -> CaseResult:
    """Ask for `payload` twice, decode each artefact once, and compare the three components.

    The two calls are made with **one** :class:`~qrcheck.targets.base.GenerateRequest` instance, so
    "identical" is a property of the object rather than of two builders that happen to agree. Each
    artefact is decoded once, through the framework's one decode path. Evidence is persisted only
    for a failing case, whose directory holds ``symbol.png`` and ``symbol-repeat.png``. Returns a
    passed result, or a failed one carrying ``value_mismatch`` or ``sampling_failed`` with the
    comparison and both projections.

    Raises:
        ValueError: The payload does not fit the symbol at the run's error level, or that level is
        outside ``L``/``M``/``Q``/``H`` -- framework-side defects in the request builder.
        TargetUnavailable: Propagated unchanged, for the runner's target-loss policy. TargetTimeout:
        Propagated unchanged, for the same reason.
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

    The decode layer's own projection is the source, so the per-decoder outcome shape has one
    definition in the product; this adds the two facts the decode layer cannot know: the artefact's
    sha256 digest, and the authoritative payload's length and digest. The projection is JSON-ready,
    the last two being ``None`` when the primary decoder produced no payload.
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

    One entry per :data:`_COMPARED_COMPONENTS` name: ``True`` when both readings agree, ``False``
    when they differ, ``None`` when either could not be read at all.
    """
    return {
        component: _agrees(
            first[_COMPARISON_KEYS[component]], second[_COMPARISON_KEYS[component]]
        )
        for component in _COMPARED_COMPONENTS
    }


def _agrees(left: object, right: object) -> bool | None:
    """Return whether two readings agree, or ``None`` when either could not be read.

    An absent reading is not an agreement: a matrix nobody sampled and a mask the format information
    did not yield say nothing about determinism, and reporting them as equal would let the case pass
    on the components that happened to be readable.
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
    ``matrix_sha256`` -- sits where a reader of any other check's evidence finds it, and the second
    call travels whole under ``repeat``. The mapping adds ``comparison``, ``divergent`` and
    ``unmeasured``.
    """
    evidence = dict(first)
    evidence["comparison"] = dict(comparisons)
    evidence["divergent"] = list(divergent)
    evidence["repeat"] = dict(second)
    evidence["unmeasured"] = list(unmeasured)
    return evidence


def _sha256(data: bytes) -> str:
    """Return the lowercase hex ``sha256`` digest of `data`.

    ``sha256`` and not ``hash()``: a report keys on these digests, and ``hash()`` is salted per
    process, so it would make two runs of one case describe it differently.
    """
    return hashlib.sha256(data).hexdigest()


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, as a recorded value only.

    A duration is recorded and never asserted -- no comparison, no threshold and no report ordering
    reads it.
    """
    return (time.perf_counter() - started) * 1000.0

