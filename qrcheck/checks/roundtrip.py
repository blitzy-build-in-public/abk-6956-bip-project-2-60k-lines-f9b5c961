"""Round-trip conformance: the decoded payload is byte-identical to the request.

``RT-ROUNDTRIP`` asserts exactly one property: a symbol a target produced for a payload decodes,
under the framework's primary decoder, back to the very bytes requested. Each of the corpus's 240
records contributes one case, so a full run is 240 cases, 240 encodes and 240 decodes, which
:data:`EXPECTED_ENCODES` declares. The check is one of the nine identifiers a ``--adapter
defective`` run must fail: the ``truncated_payload`` injection encodes only the first 64 bytes of a
longer payload, so the symbol still decodes but to the wrong bytes.

The assertion is ``payload_bytes == request.payload`` and nothing else. Version, level, mask,
geometry, quiet zone and encoding mode belong to ``SY-*``, ``MD-*`` and ``RN-*``; repeating one here
would make a single defect fail several checks.

Three outcomes fail a case, in this order: ``agreement == "disagree"`` is a finding in its own
right, reported as ``Reason.DECODER_DISAGREEMENT`` with both payloads and metadata blocks;
``payload_bytes is None`` is ``Reason.VALUE_MISMATCH`` with the decoder's own error text, and the
secondary decoder's payload is never consulted; a payload that differs is ``Reason.VALUE_MISMATCH``
with both digests and the offset of the first divergence.

Requests: a PNG at the run's EC level with no pin, so ``requires={png}`` is the complete statement
of what the check needs. ``TargetRejected``, ``TargetContractError`` and ``TargetUnsupported`` are
turned into a result by :func:`qrcheck.checks.support.case_status_for_error`;
``TargetUnavailable`` and ``TargetTimeout`` propagate to the runner. Only a non-passing case
persists evidence, ``decoded.json`` carries no duration and no path, and only
:mod:`qrcheck.targets.base` is imported from ``qrcheck.targets``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
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

# One encode and one decode per corpus record, so the expected count is the record count: the 240
# records the corpus builder emits (200 non-boundary plus 40 boundary).  A literal rather than a
# configuration value because it is an acceptance count, not a tunable threshold; the counting stub
# of tests/unit/test_checks_negative.py compares the invocations it observes against it.
_EXPECTED_ENCODES_PER_RUN: Final[int] = 240

# The expected adapter-invocation count of every check this module registers, keyed by check
# identifier and read-only: no run can mutate the declaration.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType(
    {CHECK_ID: _EXPECTED_ENCODES_PER_RUN}
)

# The one ``Agreement`` value that is a finding in its own right: two payloads that differ.
# ``qrcheck.decode.base`` publishes the vocabulary as a ``Literal``, so the value is named here
# rather than spelled out at the comparison.
_DISAGREE: Final[str] = "disagree"

# The expected outcome, recorded beside every case's measurements so a reader of
# ``summary.json`` sees what was asserted next to what happened.
_EXPECTED_OUTCOME: Final[str] = "decoded payload bytes equal the requested bytes"


@check(
    id=CHECK_ID,
    title="Payload round-trip: the decoded bytes equal the requested bytes",
    family="roundtrip",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    corpus_driven=True,
)
def cases(ctx: RunContext) -> Iterator[Case]:
    """Yield one round-trip case per corpus record, ascending by record ``id``.

    The corpus is already filtered (``--class``) and owned by the runner, so an empty passed-in
    corpus yields no cases and the runner decides what an empty selection means. Cases are built
    through :func:`qrcheck.checks.support.corpus_cases`, which owns the identifier form and the
    ordering rule. The primary decoder is declared as a dependency, but its absence never becomes a
    skip: a run without it exits 2 naming zxing-cpp. The decorator replaces this name with the
    registered contract, so the registry stays the single accessor of check identity.

    Yields:
        One case per corpus record, ascending, each named ``corpus:<record-id>``.
    """
    yield from support.corpus_cases(
        ctx,
        CHECK_ID,
        ctx.corpus,
        execute_factory=_case_for_record,
    )


def _case_for_record(record: CorpusRecord) -> Callable[[RunContext], CaseResult]:
    """Return the case callable for one corpus record's round-trip assertion.

    The callable receives only the run's context, so the record and its identifier are bound here
    rather than looked up when the case runs.
    """
    case_id = support.corpus_case_id(record.id)

    def execute(ctx: RunContext) -> CaseResult:
        """Run this record's round-trip case against the run's target."""
        return _execute_case(ctx, record, case_id)

    return execute


def _execute_case(ctx: RunContext, record: CorpusRecord, case_id: str) -> CaseResult:
    """Encode `record`'s payload once, decode the artefact once, and compare the bytes.

    One encode and one decode per case is the whole invocation budget of this check
    (:data:`EXPECTED_ENCODES`), so there is no retry and no second opinion. A declared rejection or
    a broken response becomes the failed or skipped result
    :func:`qrcheck.checks.support.case_status_for_error` prescribes. Returns a passed result on an
    exact match, and a failing one with ``VALUE_MISMATCH`` or ``DECODER_DISAGREEMENT`` otherwise.

    Raises:
        TargetUnavailable: Re-raised by the support translation; the runner owns the target-loss
        policy. TargetTimeout: Re-raised by the support translation, for the same reason.
        ValueError: The request is outside the framework's own contract, which is a programming
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
            duration_ms=support.elapsed_ms(started),
            seed=ctx.seed,
        )
        # No artefact exists for a rejection or a contract failure, and a skip is a real
        # outcome a reader must be able to inspect, so the request and the decoder-or-error
        # text are persisted either way.
        return support.record_evidence(ctx, translated)

    duration_ms = support.elapsed_ms(started)
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

    Two differing payloads are a decoder disagreement before they are a payload mismatch; no
    payload at all is a mismatch, because the authority produced nothing to compare; and a payload
    that differs is a mismatch. The comparison is byte-exact -- never text, never a decoded string.
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

    :meth:`~qrcheck.decode.base.DecodeResult.as_dict` already carries both decoders' outcomes with
    their ``status`` and ``polarity``, the consensus ``agreement``, the applied
    ``polarity_applied``, the symbology identifier, the EC-level pair, the version and the mask.
    This adds the authoritative payload's length and digest, the requested payload's digest, and the
    first differing offset, so a ``truncated_payload`` failure names the byte where the payloads
    diverge. No duration, timestamp or path is included, so two runs of one case produce the same
    bytes.
    """
    decoded = result.payload_bytes
    evidence = result.as_dict()
    evidence["decoded_len"] = None if decoded is None else len(decoded)
    evidence["first_divergence_offset"] = support.first_divergence(request.payload, decoded)
    evidence["payload_sha256"] = None if decoded is None else support.sha256_hex(decoded)
    evidence["requested_payload_sha256"] = support.sha256_hex(request.payload)
    return evidence
