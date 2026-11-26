"""Round-trip conformance: the decoded payload is byte-identical to the request.

``RT-ROUNDTRIP`` asserts exactly one property -- a symbol a target produced for a payload
decodes, under the framework's primary decoder, back to the very bytes that were requested -- over
all 240 corpus records, so a full run is 240 cases, 240 encodes and 240 decodes (AAP §0.8.1 case
matrix).  That count is declared by :data:`EXPECTED_ENCODES`, which is what lets a counting stub
prove that no case ran twice and that no retry path exists (AAP §0.4.3).  The check is also one of
the nine identifiers a ``--adapter defective`` run must fail: the ``truncated_payload`` injection
encodes only the first 64 bytes of a longer payload, so the symbol still decodes but to the wrong
bytes (AAP §0.4.2).

The assertion is ``DecodeResult.payload_bytes == request.payload`` and nothing else.  Version,
error-correction level, mask, module geometry, quiet zone and encoding mode belong to ``SY-*``,
``MD-*`` and ``RN-*``; repeating one of them here would make a single defect fail several checks
and would stop the ``defective`` target's expected failure set from being exactly enumerable (AAP
§0.4.2).

Three outcomes fail a case, evaluated in this order.  ``agreement == "disagree"`` -- both decoders
returned a payload and the payloads differ -- is a finding in its own right rather than a payload
mismatch (AAP §0.3.2), so it is reported as
:attr:`~qrcheck.report.model.Reason.DECODER_DISAGREEMENT` with both payloads and both metadata
blocks preserved in the evidence.  ``payload_bytes is None`` -- the primary decoder produced no
payload -- fails with :attr:`~qrcheck.report.model.Reason.VALUE_MISMATCH` and the decoder's own
error text, because a decoder that returns nothing is ``undetected``, never a "disagreement", and
the secondary decoder's payload is *not* consulted: the primary decoder is the authority for all
twenty-one checks, so a symbol it could not read never passes on the secondary's strength (AAP
§0.3.2).  A differing ``payload_bytes`` fails with the same reason, with both sides' digests and
the byte offset of the first divergence recorded, which is the evidence AAP §0.4.5 requires for the
``truncated_payload`` defect.

Every case requests a PNG at the run's EC level with no module-size, quiet-zone, colour, version or
mask pin.  Nothing else is pinned deliberately: pinning geometry would add a capability
requirement, and ``requires={png}`` is the complete statement of what this check needs (AAP §0.3.4
capability table).  The three errors a check translates -- ``TargetRejected``,
``TargetContractError`` and ``TargetUnsupported`` -- are turned into a result by
:func:`qrcheck.checks.support.case_status_for_error`, the product's single translation, while
``TargetUnavailable`` and ``TargetTimeout`` propagate to the runner, which owns the
one-error-result-per-check target-loss policy and can only apply it if it sees the error itself.

Only a non-passing case persists evidence: the artefact travels to
:func:`qrcheck.checks.support.record_evidence` byte for byte as the target returned it, beside
``request.json`` and ``decoded.json``, so a failure is re-derivable without consulting this module,
while a passing case writes nothing and records ``evidence_dir`` as null.  ``decoded.json`` holds
the :meth:`~qrcheck.decode.base.DecodeResult.as_dict` projection plus this check's measurements and
nothing else -- no duration, no path -- so it is byte-identical across runs for the same case and
seed.

The only clock this module reads is :func:`time.perf_counter`, around the encode and the decode
together, and the value reaches the result's volatile ``duration_ms`` field alone: never an
assertion, never an artefact.  No ``hash()``, ``id()``, ``random`` or ``time.time()`` appears
anywhere, and cases are yielded in ascending record ``id`` order, so a run's case order cannot
depend on import order, a dict's insertion order or the filesystem.

Layering: only :mod:`qrcheck.targets.base` is imported from ``qrcheck.targets`` -- the capability
vocabulary and three error types, with no concrete adapter reachable from here, which is what lets
a fourth adapter kind be added without touching a check -- and the image stack, the decoders and
the corpus builder are type-only imports.  No user rules exist for this project, so the binding
constraints are the ``MUST``/``MUST NOT`` list of AAP §0.12: ``checks/`` imports no concrete
adapter, and no check is weakened, skipped or retried to make a defective target pass.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Final

from qrcheck.checks import support
from qrcheck.registry.check import DEPENDENCY_PRIMARY_DECODER, Case, RunContext, check
from qrcheck.report.model import CaseResult, Reason
from qrcheck.targets.base import (
    Capability,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Type-only: these names never enter this module's import graph.
    from qrcheck.corpus.builder import CorpusRecord
    from qrcheck.decode.base import DecodeResult
    from qrcheck.targets.base import GenerateRequest

__all__ = ["EXPECTED_ENCODES", "cases"]

# The stable identifier of this check, written once. The registration below, every result
# this module builds and the expected-failure table all name the check through it, so a
# rename cannot leave one of them behind.
CHECK_ID: Final[str] = "RT-ROUNDTRIP"

# One encode and one decode per corpus record, so the expected adapter-invocation count is
# the record count: the 240 records of AAP §0.8.1 (200 non-boundary plus 40 boundary). It
# is a literal rather than a configuration value because it is an acceptance count, not a
# threshold a user may tune; the counting stub of tests/unit/test_checks_negative.py
# compares the invocations it observes against it (AAP §0.4.3), so a retry or a dropped
# case shows up as a mismatch instead of passing quietly.
_EXPECTED_ENCODES_PER_RUN: Final[int] = 240

# The expected adapter-invocation count of every check this module registers, keyed by
# check identifier (AAP §0.4.3).
EXPECTED_ENCODES: dict[str, int] = {CHECK_ID: _EXPECTED_ENCODES_PER_RUN}

# The case-identifier prefix AAP §0.3.4 fixes for a corpus-driven case. It is composed here
# as well as in the support helper that yields the cases because the case's execute
# callable receives only the run context: the identifier its result is filed under has to
# be captured when the case is built, and no public constant exposes the prefix.
_CORPUS_CASE_PREFIX: Final[str] = "corpus:"

# The one ``Agreement`` value that is a finding in its own right (AAP §0.3.2): two payloads
# that differ. ``qrcheck.decode.base`` publishes the vocabulary as a ``Literal`` type rather
# than as constants, so the single value this check tests for is named here instead of
# spelled out at the comparison.
_DISAGREE: Final[str] = "disagree"

# Milliseconds in a second: the unit ``CaseResult.duration_ms`` is recorded in.
_MS_PER_SECOND: Final[float] = 1000.0

# The expected outcome, recorded beside every case's measurements so a reader of
# ``summary.json`` sees what was asserted next to what happened.
_EXPECTED_OUTCOME: Final[str] = "decoded payload bytes equal the requested bytes"


@check(
    id=CHECK_ID,
    title="Payload round-trip: the decoded bytes equal the requested bytes",
    family="roundtrip",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def cases(ctx: RunContext) -> Iterator[Case]:
    """Yield one round-trip case per corpus record, ascending by record ``id``.

    The corpus is already filtered (``--class``) and owned by the runner, so an empty
    passed-in corpus yields no cases and the runner decides what an empty selection means;
    nothing here samples, skips or reorders a record (AAP §0.5.3).  Cases are built
    through :func:`qrcheck.checks.support.corpus_cases`, which owns the identifier form
    and the ordering rule, so this check cannot file a case under a name or a position no
    other corpus-driven check uses.

    The primary decoder is declared as a dependency because the assertion cannot be made
    without it, but its absence never becomes a skip: ``qrcheck doctor`` and
    ``qrcheck run`` exit 2 naming zxing-cpp before any check runs (AAP §0.6.6 row 3).  The
    one skip this check can produce is the belt-and-braces
    :class:`~qrcheck.targets.base.TargetUnsupported` path of a target that does not
    declare ``png``, which the runner normally resolves before execution.

    The decorator replaces this name with the registered
    :class:`~qrcheck.registry.check.CheckContract`, so a consumer reaches the generator
    through the contract -- ``cases.cases(ctx)``, or the registry entry by identifier --
    and the registry stays the single accessor of check identity.

    Args:
        ctx: The run's context; only its corpus, EC level, seed, adapter, evidence store
            and decoder availability are read, and only through the support helpers.

    Yields:
        One case per corpus record, in ascending record ``id`` order, each named
        ``corpus:<record-id>``.

    Raises:
        ValueError: Two records would register the same case identifier, which would let
            two results share one evidence directory and one report entry.
        TypeError: A corpus record or an identifier is not of the declared type.
    """
    yield from support.corpus_cases(
        ctx,
        CHECK_ID,
        ctx.corpus,
        execute_factory=_case_for_record,
    )


def _case_for_record(record: CorpusRecord) -> Callable[[RunContext], CaseResult]:
    """Return the callable that executes `record`'s round-trip case.

    The identifier is captured here, at case construction, because a check's ``execute``
    callable is handed the run context and nothing else; building it per record is what
    lets the closure carry the record, the identifier and the expected bytes without a
    lookup at execution time.

    Args:
        record: The corpus record this case round-trips.

    Returns:
        A callable that produces the case's result when the runner executes it.
    """
    case_id = f"{_CORPUS_CASE_PREFIX}{record.id}"

    def execute(ctx: RunContext) -> CaseResult:
        """Run this record's round-trip case against the run's target."""
        return _execute_case(ctx, record, case_id)

    return execute


def _execute_case(ctx: RunContext, record: CorpusRecord, case_id: str) -> CaseResult:
    """Encode `record`'s payload once, decode the artefact once, and compare the bytes.

    One encode and one decode per case is the whole invocation budget of this check
    (:data:`EXPECTED_ENCODES`), so there is no retry and no second opinion about the
    artefact: a symbol that does not round-trip is reported, not re-requested.

    Args:
        ctx: The run's context.
        record: The corpus record under test.
        case_id: The stable case identifier the result is filed under, composed when the
            case was built.

    Returns:
        A passed result when the authoritative payload equals the requested bytes, and a
        failed result carrying :attr:`~qrcheck.report.model.Reason.VALUE_MISMATCH` or
        :attr:`~qrcheck.report.model.Reason.DECODER_DISAGREEMENT`, its evidence directory,
        the artefact and the measurements otherwise.  A declared rejection or a broken
        response becomes the failed or skipped result
        :func:`qrcheck.checks.support.case_status_for_error` prescribes.

    Raises:
        TargetUnavailable: Re-raised by the support translation; the runner owns the
            target-loss policy.
        TargetTimeout: Re-raised by the support translation, for the same reason.
        ValueError: The request is outside the framework's own contract -- a payload that
            does not fit the requested symbol, for instance -- which is a programming
            defect the runner records as an ``error`` result.
    """
    payload = record.payload_bytes
    request = support.build(payload, ctx.ec_level, output_format="png")
    started = time.perf_counter()
    try:
        symbol, result = support.generate_and_decode(ctx, request)
    except (TargetContractError, TargetRejected, TargetUnsupported) as error:
        translated = support.case_status_for_error(
            CHECK_ID,
            case_id,
            error,
            request=request,
            duration_ms=_elapsed_ms(started),
            seed=ctx.seed,
        )
        # No artefact exists for a rejection or a contract failure, and a skip is a real
        # outcome a reader must be able to inspect, so the request and the decoder-or-error
        # text are persisted either way (AAP §0.3.4 evidence policy).
        return support.record_evidence(ctx, translated)

    duration_ms = _elapsed_ms(started)
    evidence = _evidence(request, result)
    expected = {
        "ec_level": request.ec_level,
        "outcome": _EXPECTED_OUTCOME,
        "payload_len": len(payload),
    }
    reason = _failure_reason(result, payload)
    if reason is not None:
        failed = support.fail(
            CHECK_ID,
            case_id,
            reason=reason,
            request=request.as_dict(),
            decoded=evidence,
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
        return support.record_evidence(ctx, failed, symbol=symbol)
    return support.pass_(
        CHECK_ID,
        case_id,
        request=request.as_dict(),
        decoded=evidence,
        expected=expected,
        duration_ms=duration_ms,
        seed=ctx.seed,
    )


def _failure_reason(result: DecodeResult, requested: bytes) -> Reason | None:
    """Return why `result` does not round-trip `requested`, or ``None`` when it does.

    The three failing outcomes in the order they are decided: two payloads that differ are
    a decoder disagreement before they are a payload mismatch (AAP §0.3.2); no payload at
    all is a mismatch, because the authority produced nothing to compare; and a payload
    that differs from the request is a mismatch.  The comparison is byte-exact -- never
    text, never a decoded string -- so a payload that is not valid UTF-8 is judged on the
    bytes that actually travelled.

    Args:
        result: The decoded evidence for one artefact.
        requested: The payload bytes that were requested.

    Returns:
        The reason the case fails, or ``None`` when the authoritative payload is exactly
        the requested payload.
    """
    if result.agreement == _DISAGREE:
        return Reason.DECODER_DISAGREEMENT
    if result.payload_bytes is None:
        return Reason.VALUE_MISMATCH
    if result.payload_bytes != requested:
        return Reason.VALUE_MISMATCH
    return None


def _evidence(request: GenerateRequest, result: DecodeResult) -> dict[str, Any]:
    """Return the decoded evidence this check records, built on the canonical projection.

    :meth:`~qrcheck.decode.base.DecodeResult.as_dict` already carries the two decoders'
    outcomes with their ``status`` and ``polarity``, the consensus ``agreement``, the
    applied ``polarity_applied``, the symbology identifier, the EC-level pair, the version
    and the mask.  This function adds only what the round-trip assertion needs to be
    re-readable: the authoritative payload's length and digest, the requested payload's
    digest, and the offset of the first byte at which the two differ -- the evidence AAP
    §0.4.5 requires of a ``truncated_payload`` finding, whose bytes differ while their
    length alone does not explain why.

    Args:
        request: The request the artefact was produced for.
        result: The decoded evidence for the artefact.

    Returns:
        The JSON-ready mapping the result and ``decoded.json`` carry.  No duration, no
        timestamp and no path is included, so two runs of the same case produce the same
        bytes.
    """
    decoded = result.payload_bytes
    evidence = result.as_dict()
    evidence["decoded_len"] = None if decoded is None else len(decoded)
    evidence["first_divergence_offset"] = _first_divergence(decoded, request.payload)
    evidence["payload_sha256"] = None if decoded is None else _sha256(decoded)
    evidence["requested_payload_sha256"] = _sha256(request.payload)
    return evidence


def _first_divergence(decoded: bytes | None, requested: bytes) -> int | None:
    """Return the index of the first byte at which the two payloads differ.

    A truncation shows up as the length of the shorter side rather than as a differing
    byte, which is why that index is reported as a divergence instead of ``None``: it is
    where the two payloads stop agreeing.  Equal payloads, and a payload that was never
    read, have no offset.

    Args:
        decoded: The payload the primary decoder returned, or ``None``.
        requested: The payload that was requested.

    Returns:
        The zero-based offset, or ``None`` when the payloads are equal or none was read.
    """
    if decoded is None:
        return None
    shared = min(len(decoded), len(requested))
    for index in range(shared):
        if decoded[index] != requested[index]:
            return index
    return None if len(decoded) == len(requested) else shared


def _sha256(data: bytes) -> str:
    """Return the lower-case hexadecimal SHA-256 digest of `data`.

    ``hashlib`` is used rather than the builtin ``hash()`` because a digest recorded in
    evidence must be stable across processes and machines; ``hash()`` of a bytes object is
    salted per process and would make two evidence files differ for the same payload (AAP
    §0.5.1).

    Args:
        data: The bytes to digest.

    Returns:
        The 64-character hexadecimal digest.
    """
    return hashlib.sha256(data).hexdigest()


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, a ``perf_counter`` reading.

    Args:
        started: The :func:`time.perf_counter` value taken before the case's work.

    Returns:
        The elapsed time in milliseconds, which the result records and no assertion reads.
    """
    return (time.perf_counter() - started) * _MS_PER_SECOND
