"""Result identity for qrcheck: status words, failure reasons, tallies and records.

This module owns, for the whole product, the three things a result is described by
that must exist exactly once: the vocabulary of outcomes and causes
(:class:`Status`, :class:`Reason`), the four tallies a run is summarised by
(:class:`Counts`), and the identifier a failure is filed under (:func:`failure_id`).
The runner, the registry, the evidence store and the three report writers import
them from here instead of redefining them, because a second definition of a status
word or of a failure ID would let two artefacts describe the same run differently.

Nothing here reads a clock, the environment, the filesystem or a third-party
package: every value produced is a deterministic function of its inputs, which is
what makes two runs comparable field by field. The one volatile quantity the model
carries - how long a run took - is reachable only through the
:attr:`RunResult.duration_ms` and :attr:`RunResult.duration_s` properties and is
deliberately absent from every ``as_dict`` projection.
"""

import datetime
import enum
import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CaseResult",
    "Counts",
    "Reason",
    "RunResult",
    "Status",
    "failure_id",
]

# The bound on a failure ID in characters. The ID becomes a filesystem path
# segment, so the whole identifier - prefix and digest together - is bounded.
FAILURE_ID_MAX: int = 64

# Anything outside this set becomes a single "_", so a hostile case_id cannot
# traverse out of the evidence directory it names.
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# A drive prefix ("C:") is absolute for the purposes of an artefact path, just as a
# leading "/" or a backslash separator is.
_DRIVE_PREFIX = re.compile(r"[A-Za-z]:")

# ``run_id`` is a content hash. These two shapes are rejected so that a run's
# identity can never become a per-run UUID or a reading of the wall clock.
_UUID_SHAPED = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_DATE_SHAPED = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class Status(enum.StrEnum):
    """The four outcomes a single case can reach.

    A ``StrEnum`` member already *is* its string value, so ``json.dumps``, an
    f-string and an XML attribute all emit ``passed`` without any caller needing a
    ``.value`` lookup, and a comparison against the literal string works.

    ``SKIPPED`` is a real outcome, never a flavoured pass: it is tallied in its own
    integer of :class:`Counts` and can never make an incomplete run look green.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class Reason(enum.StrEnum):
    """The machine-readable cause carried by every non-passing result.

    A passing case carries ``reason is None``. No member describes a selection
    failure or a corpus-filter failure: both are configuration conditions that end
    the command with exit code 2 before any result exists, so they never reach this
    model and need no word here.

    Adding a member is additive and safe for consumers that compare against the
    string value; renaming one changes the public surface that ``summary.json``,
    ``junit.xml`` and CI trend lines key on.
    """

    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    MISSING_DEPENDENCY = "missing_dependency"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    DECODER_DISAGREEMENT = "decoder_disagreement"
    FORMAT_INFO_DISAGREEMENT = "format_info_disagreement"
    TARGET_REJECTED = "target_rejected"
    TARGET_UNAVAILABLE = "target_unavailable"
    TARGET_TIMEOUT = "target_timeout"
    CONTRACT_ERROR = "contract_error"
    VALUE_MISMATCH = "value_mismatch"
    SAMPLING_FAILED = "sampling_failed"
    CAPACITY_VIOLATION = "capacity_violation"


@dataclass(frozen=True)
class Counts:
    """The four tallies of a run, kept as four separate integers.

    The separation is the point: a skipped result can never be folded into
    ``passed``, which is what makes the exit-code rule that reads only ``failed``
    and ``error`` honest. Instances are immutable; derive a variation with
    :func:`dataclasses.replace` rather than mutating one in place.

    Raises:
        ValueError: a tally is negative.
    """

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error: int = 0

    def __post_init__(self) -> None:
        """Reject a negative tally, naming the field that carries it.

        Raises:
            ValueError: one of the four tallies is negative.
        """
        for name in ("passed", "failed", "skipped", "error"):
            count = getattr(self, name)
            if count < 0:
                raise ValueError(f"Counts.{name} must not be negative, got {count}")

    @classmethod
    def from_results(cls, results: Iterable["CaseResult"]) -> "Counts":
        """Tally a sequence of results in a single pass.

        Each result increments exactly one tally: an unknown status raises rather
        than falling into a fifth bucket, so the four integers always sum to the
        number of results they describe.

        Args:
            results: The results to tally, in any order.

        Returns:
            The counts those results imply.

        Raises:
            ValueError: a result carries a status outside :class:`Status`.
        """
        passed = 0
        failed = 0
        skipped = 0
        error = 0
        for result in results:
            status = Status(result.status)
            if status is Status.PASSED:
                passed += 1
            elif status is Status.FAILED:
                failed += 1
            elif status is Status.SKIPPED:
                skipped += 1
            else:
                error += 1
        return cls(passed=passed, failed=failed, skipped=skipped, error=error)

    @property
    def total(self) -> int:
        """Return the number of results the four tallies describe."""
        return self.passed + self.failed + self.skipped + self.error

    def as_dict(self) -> dict[str, int]:
        """Return the summary shape ``{"passed", "failed", "skipped", "error"}``.

        The four keys appear in that fixed order so a writer never re-derives the
        shape of the counts object and two runs' summaries stay comparable.
        """
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "error": self.error,
        }


def failure_id(check_id: str, case_id: str, reason: Reason | str | None) -> str:
    """Return the identifier a failure is filed under, and its evidence directory.

    This is the only definition of a failure ID in the product: the evidence store
    and ``qrcheck.evidence.naming`` derive from it, so one failure has one name no
    matter which writer or check produced it. The result is deterministic (same
    inputs, same output, on every run and every machine), built without a clock,
    ``hash()`` or a random source, and safe to use as a single path segment: every
    character outside ``[A-Za-z0-9._-]`` has been replaced, which is what stops a
    hostile case identifier from writing outside the report directory.

    Args:
        check_id: The stable check identifier, such as ``RT-ROUNDTRIP``.
        case_id: The case identifier, such as ``corpus:url-001``.
        reason: The cause, as a :class:`Reason`, its string value, or ``None``. A
            passing case never writes evidence, so ``None`` renders as ``none`` and
            is a defensive default rather than a provenance.

    Returns:
        The identifier, never longer than :data:`FAILURE_ID_MAX` characters.

    Raises:
        Nothing: every input is accepted.
    """
    reason_text = "none" if reason is None else getattr(reason, "value", reason)
    raw = f"{check_id}__{case_id}__{reason_text}"
    sanitized = _UNSAFE_PATH_CHARS.sub("_", raw)
    if len(sanitized) <= FAILURE_ID_MAX:
        return sanitized
    # 64 - (8 hex digest characters + 1 underscore) = 55: the arithmetic bounds the
    # complete identifier, not just the retained prefix, because this string is a
    # filesystem path segment. Reading the rule as "64-character prefix plus a
    # suffix" would yield 73 characters and defeat the bound.
    digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[: FAILURE_ID_MAX - 9]}_{digest}"


@dataclass(frozen=True)
class CaseResult:
    """One case's outcome, its evidence and the seed that reproduces it.

    The ten fields are the public record shape that ``summary.json``, ``junit.xml``
    and the HTML report all serialise, so their names and order are a stable surface
    that only ever grows. ``request``, ``decoded`` and ``expected`` hold mappings that
    are already JSON-ready - the caller projects a generate request and a decode result
    into them - which is why this module imports neither the target port nor the decode
    layer; that they really are mappings is enforced here, so a list or a scalar cannot
    reach a serialisation whose schema fixes an object.

    The instance is frozen: derive a variation with :func:`dataclasses.replace`
    rather than mutating one in place.

    Raises:
        ValueError: a field breaks one of the invariants enforced by
            :meth:`__post_init__`.
    """

    check_id: str
    case_id: str
    status: Status
    reason: Reason | None = None
    duration_ms: float = 0.0
    request: Mapping[str, Any] = field(default_factory=dict)
    decoded: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Any] | None = None
    evidence_dir: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Coerce the vocabulary and enforce the record's invariants.

        A passing case carries no reason and no evidence directory (a run of a few
        thousand cases writes evidence only for what went wrong), every other status
        carries a reason, the request, decoded and expected payloads are mappings so
        every artefact documents an object where its schema fixes one, a duration is a
        finite non-negative number of milliseconds - a ``nan`` or an ``inf`` would reach
        ``summary.json`` as a token JSON does not define - and an evidence directory is a
        non-empty relative posix path, so no artefact records an absolute location and
        none names the report root itself.

        Raises:
            ValueError: ``status`` or ``reason`` is outside its enum, the status and
                reason disagree, ``request``, ``decoded`` or ``expected`` is not a
                mapping, ``duration_ms`` is negative or not finite, or ``evidence_dir``
                is empty or not a relative posix path.
        """
        try:
            status = Status(self.status)
        except ValueError as exc:
            raise ValueError(f"CaseResult.status is not a Status: {self.status!r}") from exc
        reason: Reason | None = None
        if self.reason is not None:
            try:
                reason = Reason(self.reason)
            except ValueError as exc:
                raise ValueError(f"CaseResult.reason is not a Reason: {self.reason!r}") from exc
        if status is Status.PASSED:
            if reason is not None:
                raise ValueError(f"CaseResult.reason must be None when passed, got {reason!r}")
            if self.evidence_dir is not None:
                raise ValueError(
                    f"CaseResult.evidence_dir must be None when passed, got {self.evidence_dir!r}"
                )
        elif reason is None:
            raise ValueError(f"CaseResult.reason must be set for status {status.value!r}")
        if not isinstance(self.request, Mapping):
            raise ValueError(
                f"CaseResult.request must be a Mapping, got {type(self.request).__name__}"
            )
        if not isinstance(self.decoded, Mapping):
            raise ValueError(
                f"CaseResult.decoded must be a Mapping, got {type(self.decoded).__name__}"
            )
        if self.expected is not None and not isinstance(self.expected, Mapping):
            raise ValueError(
                "CaseResult.expected must be a Mapping or None, got "
                f"{type(self.expected).__name__}"
            )
        if not math.isfinite(self.duration_ms):
            raise ValueError(f"CaseResult.duration_ms must be finite: {self.duration_ms!r}")
        if self.duration_ms < 0:
            raise ValueError(f"CaseResult.duration_ms must not be negative: {self.duration_ms!r}")
        if self.evidence_dir is not None:
            evidence_dir = self.evidence_dir
            if not evidence_dir:
                raise ValueError(
                    "CaseResult.evidence_dir must be a non-empty relative posix path: "
                    f"{evidence_dir!r}"
                )
            absolute = (
                evidence_dir.startswith("/")
                or "\\" in evidence_dir
                or ".." in evidence_dir.split("/")
                or _DRIVE_PREFIX.match(evidence_dir) is not None
            )
            if absolute:
                raise ValueError(
                    f"CaseResult.evidence_dir must be a relative posix path: {evidence_dir!r}"
                )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)

    def as_dict(self) -> dict[str, Any]:
        """Return the ten-key projection every report writer serialises.

        The mappings are returned as given, not copied or coerced: a value that is
        not JSON-serialisable must fail loudly in the writer rather than be silently
        stringified here, where the defect would be invisible.
        """
        return {
            "check_id": self.check_id,
            "case_id": self.case_id,
            "status": self.status.value,
            "reason": None if self.reason is None else self.reason.value,
            "duration_ms": self.duration_ms,
            "request": self.request,
            "decoded": self.decoded,
            "expected": self.expected,
            "evidence_dir": self.evidence_dir,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RunResult:
    """One run of the suite: its identity, its tallies and every case result.

    ``results`` is stored sorted by ``(check_id, case_id)``. That is what makes
    ``summary.json`` and ``junit.xml`` present the same order by construction,
    instead of relying on two writers to agree by hand, and it makes two runs of the
    same selection comparable element by element. That pair is also a case's identity,
    so it appears at most once: a repeated pair would give two rows one failure ID and
    one evidence directory, the second overwriting the first.

    ``run_id`` is a content hash computed by the runner over the run's parameters;
    this model validates it and refuses a UUID or a timestamp, but never invents an
    identity of its own. The two timestamps are parsed once, here, and compared with
    each other, so no consumer re-parses them, the duration costs nothing to read, and
    a run that reports finishing before it started is refused instead of being
    published with a negative duration.

    Raises:
        ValueError: a field breaks one of the invariants enforced by
            :meth:`__post_init__`.
    """

    run_id: str
    started_at: str
    finished_at: str
    adapter: str
    seed: int
    ec_level: str
    loc_count: int
    counts: Counts
    results: tuple[CaseResult, ...] = ()
    _started: datetime.datetime = field(init=False, repr=False, compare=False)
    _finished: datetime.datetime = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Sort the results and enforce the run's invariants.

        The counts are recomputed from the results and compared with the supplied
        tallies, which is the model-level guard that no outcome - a skip above all -
        was folded into another bucket on its way here. Sorting by ``(check_id,
        case_id)`` puts a repeated pair side by side, so the same pass proves that every
        case appears once. The two timestamps are compared as well, because a run that
        finished before it started would be published with a negative duration in the
        HTML header and summed into the JUnit suite time; identical timestamps stay
        legal, since a run may take no measurable time.

        Raises:
            ValueError: ``run_id`` is empty, UUID-shaped or timestamp-shaped; a
                timestamp does not parse or carries no timezone; ``finished_at``
                precedes ``started_at``; two results share one ``(check_id, case_id)``;
                or ``counts`` disagrees with the results it claims to summarise.
        """
        _validate_run_id(self.run_id)
        started = _parse_timestamp("started_at", self.started_at)
        finished = _parse_timestamp("finished_at", self.finished_at)
        if finished < started:
            raise ValueError(
                "RunResult.finished_at must not precede started_at: "
                f"{self.finished_at!r} < {self.started_at!r}"
            )
        ordered = tuple(
            sorted(self.results, key=lambda result: (result.check_id, result.case_id))
        )
        duplicates = sorted(
            {
                (previous.check_id, previous.case_id)
                for previous, result in zip(ordered, ordered[1:])
                if (previous.check_id, previous.case_id) == (result.check_id, result.case_id)
            }
        )
        if duplicates:
            repeated = ", ".join(
                f"{check_id!r}/{case_id!r}" for check_id, case_id in duplicates
            )
            raise ValueError(
                "RunResult.results holds more than one result for a case, whose rows "
                "would share the identity a failure ID and an evidence directory are "
                f"derived from: {repeated}"
            )
        recomputed = Counts.from_results(ordered)
        if recomputed != self.counts:
            raise ValueError(
                f"RunResult.counts {self.counts!r} does not match the results, "
                f"which yield {recomputed!r}"
            )
        object.__setattr__(self, "results", ordered)
        object.__setattr__(self, "_started", started)
        object.__setattr__(self, "_finished", finished)

    @property
    def duration_ms(self) -> float:
        """Return how long the run took, in milliseconds.

        Volatile by definition: it is absent from :meth:`as_dict` and is never
        compared in a determinism assertion.
        """
        return (self._finished - self._started).total_seconds() * 1000.0

    @property
    def duration_s(self) -> float:
        """Return how long the run took, in seconds, volatile as above."""
        return (self._finished - self._started).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        """Return the nine-key projection ``summary.json`` is built from.

        Exactly the top-level fields the summary schema fixes: nothing more may
        appear and none may be renamed. The two duration properties are excluded so
        two runs stay comparable after their timestamps are masked.
        """
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "adapter": self.adapter,
            "ec_level": self.ec_level,
            "seed": self.seed,
            "loc_count": self.loc_count,
            "counts": self.counts.as_dict(),
            "results": [result.as_dict() for result in self.results],
        }


def _parse_timestamp(field_name: str, value: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp that must carry a timezone.

    Args:
        field_name: The :class:`RunResult` field being parsed, for the diagnostic.
        value: The timestamp text.

    Returns:
        The parsed, timezone-aware value.

    Raises:
        ValueError: the text does not parse, or it carries no timezone.
    """
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"RunResult.{field_name} must be an ISO-8601 timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(f"RunResult.{field_name} must carry a timezone: {value!r}")
    return parsed


def _parses_as_isoformat(value: str) -> bool:
    """Return whether ``value`` parses as an ISO-8601 date or timestamp."""
    try:
        datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_run_id(run_id: str) -> None:
    """Reject a run identity that would make the run irreproducible.

    A content hash is any other string, so the check is deliberately narrow: an
    empty identity, a UUID and a timestamp are rejected, and nothing else is.

    Raises:
        ValueError: ``run_id`` is empty, UUID-shaped or timestamp-shaped.
    """
    if not run_id:
        raise ValueError("RunResult.run_id must not be empty")
    if _UUID_SHAPED.fullmatch(run_id) is not None:
        raise ValueError(f"RunResult.run_id must be a content hash, not a UUID: {run_id!r}")
    if _DATE_SHAPED.match(run_id) is not None or _parses_as_isoformat(run_id):
        raise ValueError(f"RunResult.run_id must be a content hash, not a timestamp: {run_id!r}")
