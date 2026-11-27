"""The deliberately broken self-check target: five injections the suite must catch.

The other half of the non-vacuity proof: ``qrcheck run --adapter defective`` must fail **exactly**
the nine checks the five injections target and leave the other twelve passing **in the same run**. A
tenth failure, or a change that makes one of the nine pass, is a defect in the framework; no check
may be weakened, skipped, retried or threshold-relaxed to accommodate this target. Each injection is
the minimum change its own check detects, declared in :data:`DEFECTS`:

D1 ``wrong_ec_level`` -- encodes one error-correction level below the request while its metadata
claims the requested level. Caught by ``SY-EC-MATCH``; ``SY-VERSION-MIN`` and ``SY-MATRIX-EQUIV``
follow, because a lower level fits the same payload in a smaller version at a boundary.

D2 ``truncated_payload`` -- encodes only the first 64 bytes of a longer payload **when the full
length still selects the same version at the level actually encoded**. Caught by ``RT-ROUNDTRIP``;
``MD-CHARSET`` follows. The predicate is :func:`qrcheck.corpus.capacity.truncation_keeps_version`,
the same function the corpus builder's witness check calls.

D3 ``missing_quiet_zone`` -- renders a quiet zone of one module where four or more were requested.
Caught by ``RN-QUIET-ZONE``; ``RN-MODULE-SIZE`` follows from ``(modules + 2 * quiet_zone) *
module_size``. It stops at one module and never zero, so the primary decoder still reads the symbol
and ``RT-ROUNDTRIP`` keeps passing.

D4 ``non_deterministic_mask`` -- cycles the mask per call when the request pins none and honours an
explicit pin. Caught by ``DT-DETERMINISTIC`` and nothing else, because a honoured pin keeps
``SY-MASK`` and ``SY-MATRIX-EQUIV`` clean.

D5 ``over_capacity_acceptance`` -- accepts a payload above the capacity of a pinned version and
level and silently encodes a truncated one. Caught by ``IV-OVERCAPACITY``, which builds its own
payload outside the corpus.

The nine that must fail are ``RT-ROUNDTRIP``, ``SY-VERSION-MIN``, ``SY-EC-MATCH``,
``SY-MATRIX-EQUIV``, ``MD-CHARSET``, ``RN-QUIET-ZONE``, ``RN-MODULE-SIZE``, ``DT-DETERMINISTIC`` and
``IV-OVERCAPACITY``: :func:`expected_failing_ids` derives that set from :data:`DEFECTS` and refuses
to return unless it is exactly those nine. The other twelve must still pass: ``IV-EMPTY`` and the
four ``IV-BAD-OPTION`` probes only because the conforming validation is inherited, and ``RN-LOGO``
and ``DM-RESILIENCE`` only because they derive their budgets from the EC level decoded out of the
symbol.

Segno discipline is inherited: every encode goes through
:meth:`~qrcheck.targets.reference.ReferenceAdapter._encode`, which passes ``boost_error=False`` and
calls ``segno.make_qr``. ``qrcheck/registry/expectations.py`` declares the per-defect expectation
sets independently, and the integration gate keeps the two in agreement by set equality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final

from qrcheck.config.models import TargetConfig
from qrcheck.corpus.capacity import max_payload_bytes, truncation_keeps_version
from qrcheck.targets.base import GenerateRequest
from qrcheck.targets.reference import ReferenceAdapter

__all__ = [
    "DEFECTS",
    "Defect",
    "DefectiveAdapter",
    "expected_failing_ids",
]

# D1: the error-correction level each requested level is really encoded at. ``L`` maps to
# itself because no level lies below it; the declared metadata is the requested level, the lie.
_EC_DOWNSHIFT: Final[Mapping[str, str]] = MappingProxyType(
    {"L": "L", "M": "L", "Q": "M", "H": "Q"}
)

# D2: the truncation window, in bytes. Fixed at 64 so the truncated payload still occupies
# the version the full payload would, which is why the defect is attributable to RT-ROUNDTRIP.
_TRUNCATION_WINDOW_BYTES: Final[int] = 64

# D3: the quiet zone ISO/IEC 18004 requires, and the one module this target draws instead --
# the smallest zone the primary decoder still reads, so RT-ROUNDTRIP keeps passing.
_QUIET_ZONE_MINIMUM_MODULES: Final[int] = 4
_DEFECTIVE_QUIET_ZONE_MODULES: Final[int] = 1

# D4: ISO/IEC 18004 §7.8 defines exactly eight mask patterns.
_MASK_PATTERN_COUNT: Final[int] = 8

# The capacity arithmetic is evaluated in byte mode, the most expensive way to carry a
# character, so a payload clamped to this bound fits in every mode -- and because the level D1
# encodes at (the requested one, or a step below it in the direction of more capacity) holds
# at least as much data as the level this clamp is measured against, it fits that level too.
# What a *request* may lawfully ask for is a different question, settled in the payload's own
# mode by the port's build_request and the conforming target's own capacity rule. D5's clamp
# is what reads this constant, and D2's window makes the same byte-mode choice inside
# ``capacity.truncation_keeps_version``.
_BYTE_MODE: Final[str] = "byte"

# The nine check IDs this target must fail: the comparison target
# :func:`expected_failing_ids` validates its derivation against -- the set itself is built from
# DEFECTS, and qrcheck/registry/expectations.py owns the declaration consumers read.
_EXPECTED_FAILING_IDS: Final[frozenset[str]] = frozenset(
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


@dataclass(frozen=True)
class Defect:
    """One injection: what the code does, what catches it, and why the fallout is bounded.

    The record is data rather than documentation, so :func:`expected_failing_ids` and the tests read
    it instead of restating each injection's catches.

    Attributes:
        name: The injection's stable identifier.
        injection: Exactly what the code does, checkable against the hook implementing it.
        caught_by: The check or checks designed to catch it; no check appears here twice.
        consequential: Failures that follow unavoidably, declared rather than hidden.
        rationale: Why the consequence -- or its absence -- is what it is.
    """

    name: str
    injection: str
    caught_by: frozenset[str]
    consequential: frozenset[str]
    rationale: str


DEFECTS: Final[tuple[Defect, ...]] = (
    Defect(
        name="wrong_ec_level",
        injection=(
            "encode at one error-correction level below the requested one "
            "(L->L, M->L, Q->M, H->Q) while declaring the requested level in the "
            "returned metadata"
        ),
        caught_by=frozenset({"SY-EC-MATCH"}),
        consequential=frozenset({"SY-VERSION-MIN", "SY-MATRIX-EQUIV"}),
        rationale=(
            "a lower error-correction level fits the same payload in a smaller version "
            "at a capacity boundary, so the minimal-version assertion must fail too; "
            "and the module matrix at a pinned error-correction level necessarily "
            "differs from the oracle's"
        ),
    ),
    Defect(
        name="truncated_payload",
        injection=(
            "encode only the first 64 bytes of a payload longer than 64 bytes when "
            "the full length selects the same version at the level actually encoded, "
            "leaving a payload whose truncation would cross a version boundary intact"
        ),
        caught_by=frozenset({"RT-ROUNDTRIP"}),
        consequential=frozenset({"MD-CHARSET"}),
        rationale=(
            "with the window fixed this way the truncated payload is shorter but still "
            "occupies the same version, so SY-VERSION-MIN keeps the error-correction "
            "shift as its single cause and no other check sees a difference; "
            "MD-CHARSET fails because its multi-byte, binary and alphanumeric payloads "
            "are longer than the window"
        ),
    ),
    Defect(
        name="missing_quiet_zone",
        injection=(
            "render a quiet zone of one module when four or more are requested, never "
            "zero, and size the image from the zone actually drawn"
        ),
        caught_by=frozenset({"RN-QUIET-ZONE"}),
        consequential=frozenset({"RN-MODULE-SIZE"}),
        rationale=(
            "the dimension contract is (modules + 2 * quiet_zone) * module_size, so a "
            "short quiet zone necessarily breaks it; stopping at one module rather "
            "than zero keeps the primary decoder able to read the symbol, and both "
            "assertions are measured arithmetically from the image so neither needs a "
            "decode"
        ),
    ),
    Defect(
        name="non_deterministic_mask",
        injection=(
            "cycle the mask per call (0, 1, 2, ... modulo 8) when the request pins "
            "none, and honour an explicit mask pin"
        ),
        caught_by=frozenset({"DT-DETERMINISTIC"}),
        consequential=frozenset(),
        rationale=(
            "because the pin is honoured, SY-MASK and SY-MATRIX-EQUIV stay clean: the "
            "defect is visible only across two calls of the identical request, which "
            "is exactly what the determinism check compares"
        ),
    ),
    Defect(
        name="over_capacity_acceptance",
        injection=(
            "accept a payload above the capacity of the pinned version and level and "
            "silently encode a truncated payload instead of rejecting the request"
        ),
        caught_by=frozenset({"IV-OVERCAPACITY"}),
        consequential=frozenset(),
        rationale=(
            "IV-OVERCAPACITY builds its own over-capacity payload outside the corpus, "
            "so no corpus-driven check sees the acceptance"
        ),
    ),
)


def expected_failing_ids() -> frozenset[str]:
    """Return the check IDs a ``--adapter defective`` run must fail -- exactly nine.

    Derived from :data:`DEFECTS` by unioning every defect's ``caught_by`` with its
    ``consequential``, and validated by an explicit raise rather than an ``assert``, because
    ``python -O`` strips those.

    Returns: The nine IDs: ``DT-DETERMINISTIC``, ``IV-OVERCAPACITY``, ``MD-CHARSET``,
    ``RN-MODULE-SIZE``, ``RN-QUIET-ZONE``, ``RT-ROUNDTRIP``, ``SY-EC-MATCH``, ``SY-MATRIX-EQUIV``,
    ``SY-VERSION-MIN``.

    Raises:
        RuntimeError: The union is not those nine, or one check is the primary catch of two defects.
    """
    failing: set[str] = set()
    primary: set[str] = set()
    for defect in DEFECTS:
        already_claimed = primary & defect.caught_by
        if already_claimed:
            raise RuntimeError(
                f"defect {defect.name!r} claims a primary catch already declared by "
                f"another defect: {', '.join(sorted(already_claimed))}"
            )
        primary |= defect.caught_by
        failing |= defect.caught_by
        failing |= defect.consequential

    derived = frozenset(failing)
    if derived != _EXPECTED_FAILING_IDS:
        missing = ", ".join(sorted(_EXPECTED_FAILING_IDS - derived)) or "none"
        unexpected = ", ".join(sorted(derived - _EXPECTED_FAILING_IDS)) or "none"
        raise RuntimeError(
            "the declared defects derive a failing set that is not the nine checks a "
            f"--adapter defective run must fail; missing: {missing}; "
            f"unexpected: {unexpected}"
        )
    return derived


class DefectiveAdapter(ReferenceAdapter):
    """The conforming generator with the five injections switched on.

    Everything that is not an injection is inherited unchanged: the capability set (all thirteen, so
    an expected failure can never become a skip), the shape, payload and contrast rules, the segno
    encode and the exception conversion. The five deltas override six hooks only, so the
    ``IV-EMPTY`` and ``IV-BAD-OPTION`` contracts pass on the inherited messages.
    """

    def __init__(self, config: TargetConfig, *, name: str | None = None) -> None:
        """Bind the adapter to its resolved ``[targets.<name>]`` record.

        Starts D4's mask cycle at zero.

        Raises:
            TypeError: ``config.contrast_floor`` is not a real number -- the record was built by
                hand.
        """
        super().__init__(config, name=name)
        self._mask_counter: int = 0

    # D1: encode one error-correction level down, claim the requested one

    def _effective_ec(self, request: GenerateRequest) -> str:
        """Return the level to encode at: one below the requested one (D1).

        Both the capacity rule and the segno call read this method, so the whole render path carries
        the lower level while :meth:`_declared_ec` still claims the requested one -- that split is
        the entire injection.

        Returns: The mapped level: ``L`` for a request for L or M, ``M`` for Q, ``H`` for H.
        """
        return _EC_DOWNSHIFT.get(request.ec_level, request.ec_level)

    def _declared_ec(self, request: GenerateRequest) -> str:
        """Return the level this target claims it encoded at: the requested one (D1).

        The symbol carries the lower level and the metadata says the requested one. ``SY-EC-MATCH``
        reads the primary decoder's level and the format information and never this value, which is
        what makes the lie evidence rather than a pass.
        """
        return request.ec_level

    # D2 and D5: the payload window, and the capacity clamp

    def _effective_payload(self, request: GenerateRequest) -> bytes:
        """Return the bytes to encode: the payload after D2's window then D5's clamp.

        D2 cuts a payload longer than the window when the full length still selects the same version
        at the level actually encoded -- measured at the *effective* level, so the truncated symbol
        occupies the version the whole one would have. D5 then clamps to the pinned version's
        capacity at the level the request names, where the conforming target refuses.

        Returns: The payload to encode: at most the window, and at most the pinned version's
        capacity.
        """
        payload = request.payload
        ec_level = self._effective_ec(request)
        if len(payload) > _TRUNCATION_WINDOW_BYTES and truncation_keeps_version(
            len(payload), _TRUNCATION_WINDOW_BYTES, ec_level
        ):
            payload = payload[:_TRUNCATION_WINDOW_BYTES]

        version = request.version
        if version is not None:
            # The requested level, not the effective one: clamping to the requested capacity is
            # always
            # encodable, while clamping to the effective one would let D1 swallow an over-capacity
            # request.
            # the level
            # above it, so clamping to the requested capacity is always encodable while clamping to
            # the
            # effective capacity would let D1's downshift swallow an over-capacity request.
            capacity = max_payload_bytes(version, request.ec_level, _BYTE_MODE)
            payload = payload[:capacity]
        return payload

    def _check_payload(self, request: GenerateRequest) -> None:
        """Refuse an empty payload, and deliberately not an oversized one (D5).

        Only the empty-payload rule survives, asked with the pin removed so its message keeps one
        definition.

        Raises:
            TargetRejected: The payload is empty, carrying the conforming target's own message.
        """
        if request.version is None:
            super()._check_payload(request)
            return
        super()._check_payload(replace(request, version=None))

    # D3: the quiet zone

    def _effective_quiet_zone(self, request: GenerateRequest) -> int:
        """Return the quiet zone to render, in modules: one where four were asked (D3).

        The substitution stops at one module and never goes to zero: a symbol with no quiet zone is
        undetectable, which would fail ``RT-ROUNDTRIP`` too. A request below the ISO/IEC 18004
        minimum is rendered as asked, because the injection targets the four-module minimum a
        conforming request asks for, and the renderer sizes the image from this value.

        Returns: One module when the requested zone is four or more, otherwise the requested zone.
        """
        requested = super()._effective_quiet_zone(request)
        if requested >= _QUIET_ZONE_MINIMUM_MODULES:
            return _DEFECTIVE_QUIET_ZONE_MODULES
        return requested

    # D4: the mask cycle

    def _effective_mask(self, request: GenerateRequest) -> int:
        """Return the mask to encode with: the pin, or the next value of the cycle (D4).

        A pin is honoured exactly, which keeps ``SY-MASK`` and ``SY-MATRIX-EQUIV`` clean, and the
        cycle is what ``DT-DETERMINISTIC`` exposes across two identical requests.

        Returns: The pinned mask, otherwise the counter modulo the eight mask patterns of clause
        7.8.
        """
        if request.mask is not None:
            return request.mask
        mask = self._mask_counter % _MASK_PATTERN_COUNT
        self._mask_counter += 1
        return mask
