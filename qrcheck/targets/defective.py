"""The deliberately broken self-check target: five injections the suite must catch.

This target is the other half of the framework's non-vacuity proof.  ``qrcheck run
--adapter reference`` must come back with zero failed, zero errored and zero skipped
results; ``qrcheck run --adapter defective`` must fail **exactly** the nine checks the
five injections target and leave the other twelve passing **in the same run**.  A
"fix" that makes a tenth check fail is a defect in the framework, not in this target,
and so is a change that makes one of the nine pass.  No check may be weakened,
skipped, retried or threshold-relaxed to accommodate this target: the injections here
and their expected failing set are acceptance criteria in their own right.

The five injections, each the minimum change its own check detects
------------------------------------------------------------------
D1 ``wrong_ec_level``
    Encodes one error-correction level below the request (L->L, M->L, Q->M, H->Q)
    while the metadata it declares claims the requested level.  Caught by
    ``SY-EC-MATCH``; ``SY-VERSION-MIN`` and ``SY-MATRIX-EQUIV`` follow necessarily,
    because a lower level fits the same payload in a smaller version at a boundary
    and a pinned-EC matrix cannot match.
D2 ``truncated_payload``
    Encodes only the first 64 bytes of a longer payload **when the full length still
    selects the same version at the level actually encoded**, and leaves a payload
    alone when truncating it would cross a version boundary.  Caught by
    ``RT-ROUNDTRIP``; ``MD-CHARSET`` follows, because its multi-byte and binary
    payloads are longer than the window.  With the window fixed this way the
    truncated payload still occupies the same version, so ``SY-VERSION-MIN`` keeps
    D1 as its single cause and no other check sees a difference.  The eight synthetic
    ``SY-MATRIX-EQUIV`` payloads are 64 bytes or shorter by construction, so the
    injection never reaches them.  The predicate itself is
    :func:`qrcheck.corpus.capacity.truncation_keeps_version`, and the corpus builder's
    witness check calls the same function, so the injection and the proof that it is
    observable are one rule rather than two that could disagree.  This target evaluates
    it at the level the symbol is encoded at, which is the reading the cascade
    containment requires: only that level keeps the shortened symbol on the version the
    whole one occupies, while at the requested level the cut would move a payload across
    a version boundary and put a check that must pass at risk.
D3 ``missing_quiet_zone``
    Renders a quiet zone of one module where four or more were requested, and sizes
    the image from the zone actually drawn.  Caught by ``RN-QUIET-ZONE``;
    ``RN-MODULE-SIZE`` follows, because the dimension contract is
    ``(modules + 2 * quiet_zone) * module_size``.  The stop is at one module and
    never zero, so the primary decoder still reads the symbol and ``RT-ROUNDTRIP``
    keeps passing for these cases.
D4 ``non_deterministic_mask``
    Cycles the mask per call (0, 1, 2, ... modulo 8) when the request pins none, and
    honours an explicit pin.  Caught by ``DT-DETERMINISTIC`` and by nothing else:
    because a pin is honoured, ``SY-MASK`` and ``SY-MATRIX-EQUIV`` stay clean, and an
    unpinned mask is an encoder choice rather than a framework expectation.
D5 ``over_capacity_acceptance``
    Accepts a payload above the capacity of a pinned version and level and silently
    encodes a truncated payload instead of refusing it.  Caught by
    ``IV-OVERCAPACITY``, which builds its own over-capacity payload outside the
    corpus, so no corpus-driven check sees it.

The nine checks that must fail
------------------------------
``RT-ROUNDTRIP``, ``SY-VERSION-MIN``, ``SY-EC-MATCH``, ``SY-MATRIX-EQUIV``,
``MD-CHARSET``, ``RN-QUIET-ZONE``, ``RN-MODULE-SIZE``, ``DT-DETERMINISTIC``,
``IV-OVERCAPACITY``.  :func:`expected_failing_ids` derives that set from
:data:`DEFECTS` and refuses to return unless it is exactly those nine.

The twelve checks that must still pass
--------------------------------------
``SY-MASK``, ``MD-MODE-OPTIMAL``, ``RN-CONTRAST``, ``RN-SVG``, ``RN-LOGO``,
``DM-RESILIENCE``, ``DM-SEEDED``, ``IV-EMPTY``, ``IV-BAD-OPTION``, ``PF-ENCODE``,
``PF-DECODE``, ``CL-ARTIFACTS``.  This half is the easy one to break by accident:
``IV-EMPTY`` and the four ``IV-BAD-OPTION`` probes pass here only because the
conforming validation is inherited rather than re-implemented, and ``RN-LOGO`` and
``DM-RESILIENCE`` pass only because they derive their budgets from the EC level
decoded out of the symbol rather than from the requested one.

Segno discipline, inherited and never re-implemented
----------------------------------------------------
Every encode goes through :meth:`~qrcheck.targets.reference.ReferenceAdapter._encode`,
which passes ``boost_error=False`` and calls ``segno.make_qr`` -- Model 2 only, never
Micro QR.  A boosted level would change the EC level the injection is about, and a
Micro symbol has no ``4V+17`` module count, no Version 1-40 number and a different
format-information layout, so every symbol-geometry assertion would be wrong.  This
module therefore contains no encoding path of its own; it changes only the six
protected hooks named below, two of which carry D1 and two of which carry D2 and D5.

What this table declares, and what is declared beside it
--------------------------------------------------------
:data:`DEFECTS` is the declaration of what each injection does and what it is expected
to break, and :func:`expected_failing_ids` derives the failing set from it, refusing to
return unless that set is exactly the nine IDs the acceptance criteria name.  It is not
the declaration a consumer reads.  ``qrcheck/registry/expectations.py`` declares the
per-defect expectation sets itself, deliberately: it must not import a concrete adapter,
because ``qrcheck list`` imports ``qrcheck.registry`` on every invocation and this
target imports segno.  The two declarations are therefore independent, and they are kept
in agreement by the integration gate -- ``tests/integration/test_defective_run.py``
asserts **set equality** between the check IDs that failed or errored and
``SELF_CHECK_TARGETS["defective"]`` -- so an edit on one side never moves the other.

State, and why it lives on the instance
---------------------------------------
The mask counter D4 needs is the only mutable state in this module, and it is
per-instance: a module-level or class-level counter would make two adapters share one
sequence and would make ``DT-DETERMINISTIC``'s observation depend on run order.
Nothing here reads a clock, ``random``, ``hash()``, ``id()``, the environment or a
file, so two runs of the same requests produce the same symbols.

Layering, and which errors can escape
-------------------------------------
This module imports the standard library, :mod:`qrcheck.targets.reference` (which
supplies segno and the whole render path), :mod:`qrcheck.targets.base` and
:mod:`qrcheck.corpus.capacity`.  It imports no decoder, no image library, no
transport client and none of ``qrcheck.checks``, ``qrcheck.report``,
``qrcheck.decode`` or ``qrcheck.registry``.

The errors that can escape it are the same three the conforming target raises:
:class:`~qrcheck.targets.base.TargetRejected` for a request the target's own contract
refuses -- an empty payload, an error level outside L/M/Q/H, a version or mask
outside its range, a non-positive module size, a negative quiet zone, a malformed
colour, a below-floor contrast ratio, or a payload that does not fit even Version 40
-- :class:`~qrcheck.targets.base.TargetContractError` when segno or the stream writer
fails for a reason the request does not explain, and
:class:`~qrcheck.targets.base.TargetUnsupported` from the capability gate.
:class:`~qrcheck.targets.base.TargetUnavailable` and
:class:`~qrcheck.targets.base.TargetTimeout` are never raised: this adapter is an
in-process call.  Note the one deliberate hole D5 opens: at a pinned version an
over-capacity payload is **not** rejected, which is the defect, not an oversight.
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

# D1: the error-correction level each requested level is really encoded at.  ``L``
# maps to itself because no level lies below it, so a request for L is encoded and
# declared truthfully -- the documented edge of the injection rather than a special
# case.  The declared metadata is the requested level in every row, which is the lie.
_EC_DOWNSHIFT: Final[Mapping[str, str]] = MappingProxyType(
    {"L": "L", "M": "L", "Q": "M", "H": "Q"}
)

# D2: the truncation window, in bytes.  Fixed at 64 so the truncated payload still
# occupies the version the full payload would, which is the whole reason the defect
# is attributable to RT-ROUNDTRIP alone.
_TRUNCATION_WINDOW_BYTES: Final[int] = 64

# D3: the quiet zone ISO/IEC 18004 requires, and the one module this target draws
# instead.  One module is the smallest zone the primary decoder still reads, which is
# what keeps RT-ROUNDTRIP passing on these cases; zero would make the symbol
# undetectable and would turn one defect into two.
_QUIET_ZONE_MINIMUM_MODULES: Final[int] = 4
_DEFECTIVE_QUIET_ZONE_MODULES: Final[int] = 1

# D4: ISO/IEC 18004 §7.8 defines exactly eight mask patterns.
_MASK_PATTERN_COUNT: Final[int] = 8

# The capacity arithmetic is evaluated in byte mode, the most expensive way to carry a
# character, so a payload clamped to this bound fits in every mode -- and because the level D1
# encodes at (the requested one, or a step below it in the direction of more capacity) holds
# at least as much data as the level this clamp is measured against, it fits that level too.
# What a *request* may lawfully ask for is a different question, settled in the payload's own
# mode by the port's build_request and the conforming target's own capacity rule.  D5's clamp
# is what reads this constant, and D2's window makes the same byte-mode choice inside
# ``capacity.truncation_keeps_version``.
_BYTE_MODE: Final[str] = "byte"

# The nine check IDs §0.4.2 declares this target must fail.  It is only the comparison
# target :func:`expected_failing_ids` validates its derivation against -- the set that
# function returns is built from DEFECTS, never read out of here -- and
# qrcheck/registry/expectations.py owns the declaration consumers use.
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

    The record is data rather than documentation, so the tests and
    :func:`expected_failing_ids` read it rather than restating each injection's catches.
    The consumer table is a separate declaration: ``qrcheck/registry/expectations.py``
    declares its own per-defect expectation sets and never imports this one, and the
    integration gate proves the two equal, so an edit on one side does not move the
    other.  A test can assert each entry against the evidence a failing check records.

    Attributes:
        name: The injection's stable identifier, as §0.4.2 writes it.
        injection: Exactly what the code does, stated in the terms a reader can check
            against the hook that implements it.
        caught_by: The check or checks designed to catch the injection.  No check
            appears here for two defects: one injection, one attributable catch.
        consequential: Failures that follow unavoidably from the injection.  Declared
            rather than hidden, because a consequence kept out of the expected set
            would look like a framework defect.
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

    The set is *derived* from :data:`DEFECTS` by unioning every defect's
    :attr:`~Defect.caught_by` with its :attr:`~Defect.consequential`, so the
    declaration above and the failures it claims cannot drift apart: adding, removing
    or moving an ID changes this answer and therefore the acceptance gate that compares
    a real run against it.  It is not a stored constant, and no consumer reads it as
    one -- ``qrcheck/registry/expectations.py`` owns the declaration
    ``SELF_CHECK_TARGETS`` that the end-to-end gate compares against.

    The derivation is validated before it is returned, and the validation is an
    explicit raise rather than an ``assert``: ``python -O`` strips assertions, and a
    stripped check here would let the acceptance gate compare a run against a silently
    wrong set.

    Returns:
        ``DT-DETERMINISTIC``, ``IV-OVERCAPACITY``, ``MD-CHARSET``, ``RN-MODULE-SIZE``,
        ``RN-QUIET-ZONE``, ``RT-ROUNDTRIP``, ``SY-EC-MATCH``, ``SY-MATRIX-EQUIV`` and
        ``SY-VERSION-MIN`` -- the nine IDs of §0.4.2, and nothing else.

    Raises:
        RuntimeError: The derived union is not those nine IDs, or one check ID is
            declared as the primary catch of two defects.  Either one means
            :data:`DEFECTS` was edited without the acceptance criteria moving with it.
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
    """The conforming generator with the five injections of §0.4.2 switched on.

    Everything that is not an injection is inherited unchanged: the capability set
    (all thirteen members, so an expected failure can never be turned into a skip), the
    request-shape rule, the contrast rule, the module size, the colours, the version
    pin, the segno encode with ``boost_error=False`` and Model 2 only, the artefact
    renderer and the conversion of library exceptions into the five adapter errors.
    The five deltas are expressed by overriding only six hooks -- :meth:`_effective_ec`
    and :meth:`_declared_ec` for D1, :meth:`_effective_payload` and
    :meth:`_check_payload` for D2 and D5, :meth:`_effective_quiet_zone` for D3 and
    :meth:`_effective_mask` for D4 -- which is why the ``IV-EMPTY`` and
    ``IV-BAD-OPTION`` contracts keep passing against this target: the conforming
    validation runs here verbatim, including its message text.

    The adapter holds one piece of mutable state, the mask counter D4 needs, and it is
    per-instance and reset in the constructor, so two adapters never share a sequence
    and ``DT-DETERMINISTIC``'s observation does not depend on run order.  Everything
    else is a pure function of the request and the capacity tables, so the same request
    replayed against a fresh adapter produces the same symbol as the injection allows.

    Which errors can escape: the same three the conforming target raises.
    :class:`~qrcheck.targets.base.TargetRejected` for a request the target's own
    contract refuses -- an empty payload, an error level outside L/M/Q/H, a version or
    mask outside its range, a non-positive module size, a negative quiet zone, a
    malformed colour, a below-floor contrast ratio, or a payload beyond Version 40 --
    :class:`~qrcheck.targets.base.TargetContractError` when segno or the stream writer
    fails for a reason the request does not explain, and
    :class:`~qrcheck.targets.base.TargetUnsupported` from the capability gate.
    :class:`~qrcheck.targets.base.TargetUnavailable` and
    :class:`~qrcheck.targets.base.TargetTimeout` are never raised, because this target
    spawns no process and runs no watchdog, so no ceiling is applied to its encode.  The
    one deliberate hole is D5: at a pinned version an over-capacity payload is accepted
    and truncated rather than refused.
    """

    def __init__(self, config: TargetConfig, *, name: str | None = None) -> None:
        """Bind the adapter to its resolved ``[targets.<name>]`` record.

        The inherited constructor sets the declared name, the thirteen capabilities and
        the configured contrast floor; this one starts D4's mask cycle at zero, so the
        first unpinned request of a run is encoded with mask 0 and the second with mask
        1.

        Args:
            config: The resolved target record.  Only ``contrast_floor`` is read: the
                capability set is declared in full by the target itself, and
                ``timeout_s`` is not read, because the encode here is a local segno call
                with no boundary to bound.
            name: The configured target name, used by every diagnostic.  Defaults to
                ``config.type``, which is ``"defective"`` for this record.

        Raises:
            TypeError: ``config.contrast_floor`` is not a real number.  The
                configuration schema rejects that at load time, so it means the record
                was hand-built.
        """
        super().__init__(config, name=name)
        self._mask_counter: int = 0

    # ------------------------------------------------------------------
    # D1: encode one error-correction level down, claim the requested one
    # ------------------------------------------------------------------

    def _effective_ec(self, request: GenerateRequest) -> str:
        """Return the level to encode at: one below the requested one (D1).

        Both the capacity rule and the segno call read this method, so the whole render
        path -- codewords, format information, the module matrix the oracle is compared
        against -- carries the lower level, while :meth:`_declared_ec` still claims the
        requested one.  That split is the entire injection: nothing here falsifies the
        format information, the matrix or the decoder's metadata, because a target that
        faked those would make ``SY-EC-MATCH`` pass and leave the suite vacuous.

        Returns:
            The mapped level: ``L`` for a request for L or M, ``M`` for Q, ``H`` for H.

        Raises:
            Nothing: the inherited shape rule refuses a level outside L/M/Q/H before
                this is reached, so the mapping is total over the values that arrive.
                The fallback exists so a future path that skipped that rule reports a
                segno complaint as a rejection instead of a ``KeyError``.
        """
        return _EC_DOWNSHIFT.get(request.ec_level, request.ec_level)

    def _declared_ec(self, request: GenerateRequest) -> str:
        """Return the level this target claims it encoded at: the requested one (D1).

        The symbol carries the lower level; the metadata says the requested one.  The
        inherited implementation returns the same value today, so this override changes
        no behaviour -- it states the lie in this file, so the injection stays whole
        here even if the conforming claim ever stops being the requested level.
        ``SY-EC-MATCH`` reads the primary decoder's level and the format information and
        never this value, which is what makes the lie evidence rather than a pass.

        Returns:
            ``request.ec_level``, unchanged.
        """
        return request.ec_level

    # ------------------------------------------------------------------
    # D2 and D5: the payload window, and the capacity clamp
    # ------------------------------------------------------------------

    def _effective_payload(self, request: GenerateRequest) -> bytes:
        """Return the bytes to encode: the payload after D2's window then D5's clamp.

        The two injections act in this fixed order, and the order is what makes each
        one attributable:

        1. **D2, the 64-byte window.**  A payload longer than the window is cut to it
           when the full length still selects the same version at the level actually
           encoded.  The rule is
           :func:`qrcheck.corpus.capacity.truncation_keeps_version`, the same function
           the corpus builder's witness check calls, so the injection and the assertion
           that it stays observable cannot drift apart.  It is evaluated at request time,
           so the affected record set is derived rather than listed, and a payload whose
           truncation would cross a version boundary is left intact.  Measuring at the
           *effective* level is what makes the injection cost nothing but the payload:
           the truncated symbol then occupies exactly the version the untruncated one
           would have, so no version-asserting check that must keep passing can detect
           the cut.
        2. **D5, the capacity clamp.**  With a version pinned, a payload above that
           version's capacity at the level the request names is cut to the capacity and
           encoded, where the conforming target refuses it with ``data too large for
           version ...``.  The clamp measures the *requested* level, not the encoded
           one, so the fault -- an over-capacity request being accepted -- is visible
           whatever the request's size and cannot be absorbed by D1's lower level: a
           payload over the requested capacity is truncated even when it would have fit
           the level actually used.  No case other than ``IV-OVERCAPACITY`` pins a
           version with a payload over its capacity, so nothing else sees the clamp.
           For the ``IV-OVERCAPACITY`` probe the D2 predicate is false -- a 64-byte
           payload selects a far smaller version -- so D5 is what fires, and the symbol
           decodes to fewer bytes than were requested, which is exactly the observation
           that check records.  An unpinned request is untouched.

        Both produce a valid QR symbol, never an unopenable image or an exception: the
        injected behaviour is "accepts and truncates", and a crash would erase the
        distinction ``IV-OVERCAPACITY`` asserts on.

        Returns:
            The payload to encode: at most the window, and at most the pinned version's
            capacity at the level the request names.

        Raises:
            Nothing: the version and level passed to the capacity tables have already
                been validated by the inherited shape rule.
        """
        payload = request.payload
        ec_level = self._effective_ec(request)
        if len(payload) > _TRUNCATION_WINDOW_BYTES and truncation_keeps_version(
            len(payload), _TRUNCATION_WINDOW_BYTES, ec_level
        ):
            payload = payload[:_TRUNCATION_WINDOW_BYTES]

        version = request.version
        if version is not None:
            # The requested level, not the effective one: a level's capacity is never
            # larger than the level above it, so clamping to the requested capacity is
            # always encodable at the effective level, while clamping to the effective
            # capacity would let D1's downshift swallow an over-capacity request whole.
            capacity = max_payload_bytes(version, request.ec_level, _BYTE_MODE)
            payload = payload[:capacity]
        return payload

    def _check_payload(self, request: GenerateRequest) -> None:
        """Refuse an empty payload, and deliberately not an oversized one (D5).

        The inherited rule refuses two things: an empty payload, and a payload that
        does not fit a pinned version.  Only the first survives here, because accepting
        the second is D5.  The inherited rule is asked with the pin removed, so the
        empty-payload message -- public API, quoted verbatim by ``IV-EMPTY`` -- keeps
        one definition in :mod:`qrcheck.targets.reference` instead of a copy that could
        drift, and the version-independent half it contributes is exactly the empty
        check.  An unpinned request is passed through the inherited rule unchanged, so
        D5 does not widen to requests it was never declared to touch.

        Raises:
            TargetRejected: The payload is empty, carrying the conforming target's own
                message.  An over-capacity payload at a pinned version is accepted and
                truncated by :meth:`_effective_payload` instead.
        """
        if request.version is None:
            super()._check_payload(request)
            return
        super()._check_payload(replace(request, version=None))

    # ------------------------------------------------------------------
    # D3: the quiet zone
    # ------------------------------------------------------------------

    def _effective_quiet_zone(self, request: GenerateRequest) -> int:
        """Return the quiet zone to render, in modules: one where four were asked (D3).

        The substitution stops at one module and never goes to zero: a symbol with no
        quiet zone is undetectable, which would fail ``RT-ROUNDTRIP`` as well and turn
        one declared defect into two.  A request below the ISO/IEC 18004 minimum is
        rendered exactly as asked, because the injection targets the four-module
        minimum a conforming request asks for.

        The renderer sizes the image from this value, so "sizes the image from the zone
        it actually drew" follows without any compensating arithmetic -- which is what
        keeps the failure attributable to the dimension contract alone.

        Returns:
            One module when the requested zone is four or more, otherwise the requested
            zone unchanged.
        """
        requested = super()._effective_quiet_zone(request)
        if requested >= _QUIET_ZONE_MINIMUM_MODULES:
            return _DEFECTIVE_QUIET_ZONE_MODULES
        return requested

    # ------------------------------------------------------------------
    # D4: the mask cycle
    # ------------------------------------------------------------------

    def _effective_mask(self, request: GenerateRequest) -> int:
        """Return the mask to encode with: the pin, or the next value of the cycle (D4).

        A pin is honoured exactly, which is what keeps ``SY-MASK`` and
        ``SY-MATRIX-EQUIV`` clean: an unpinned mask is an encoder choice, so cycling it
        breaks nothing a single symbol can reveal.  The cycle is what
        ``DT-DETERMINISTIC`` exposes, by asking for the identical request twice and
        comparing the two samplings.

        The counter advances exactly once per generated symbol -- the segno call is the
        only caller -- so the sequence a run sees is 0, 1, 2, ... and, being
        per-instance, it is not shared with any other adapter.

        Returns:
            The pinned mask when the request carries one, otherwise the counter's value
            modulo the eight mask patterns of ISO/IEC 18004 §7.8.
        """
        if request.mask is not None:
            return request.mask
        mask = self._mask_counter % _MASK_PATTERN_COUNT
        self._mask_counter += 1
        return mask
