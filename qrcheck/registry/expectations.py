"""The single declaration of what each built-in self-check target must produce.

The framework proves it is not vacuous with the two targets that ship with it. ``reference`` must
come back with zero failed, zero errored and zero skipped results; ``defective`` must fail
**exactly** the nine checks its five injections target and leave the other twelve passing in the
same run. This module is the one place that second half is stated: :data:`SELF_CHECK_TARGETS` maps a
target name to the set of identifiers a run against it is expected to fail, and
``tests/integration/test_defective_run.py`` asserts set **equality** -- not containment -- against
the identifiers that actually failed or errored in a real ``--adapter defective`` run, so a tenth
failure and a missing one are both framework defects.

``defective`` carries five injections, each documented in :mod:`qrcheck.targets.defective`. One
immutable set is declared per injection -- :data:`WRONG_EC_LEVEL_EXPECTED` and its four siblings,
keyed by injection name in :data:`DEFECT_EXPECTATIONS` so a reviewer can read the two declarations
side by side -- and :data:`SELF_CHECK_TARGETS` unions them rather than restating the nine
identifiers, so an injection can neither gain nor lose a consequence without this module saying so;
the union is checked against a nine-identifier literal as this module imports.

Nothing here imports :mod:`qrcheck.targets.defective` or any other concrete adapter, deliberately:
``qrcheck list`` imports ``qrcheck.registry`` on every invocation and the defective target imports
segno, so an import of it here would pull an encoder into a process that only wants to print
twenty-one check titles. The two declarations are independent, and their agreement is proven by the
integration test above rather than by an import of the target's ``DEFECTS`` table, which would trade
a proven equality for an import-time coupling.

What a caller may rely on: :func:`expected_failures` answers ``None`` for any name that is not a
built-in self-check target, because a user-supplied ``library``, ``cli`` or ``http`` generator has
no declared expectation and an empty set would claim the opposite; :func:`expected_passing` is the
complement over the twenty-one declared identifiers, so the two answers partition them; and
:func:`validate_against_registry` re-checks the declaration against the registry, so a deleted check
or a deleted expectation raises instead of silently shrinking what the acceptance gate compares. The
module reads no file, decides no exit code, prints nothing, mutates no registry and imports nothing
outside :mod:`qrcheck.registry.check`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from qrcheck.registry.check import CHECK_IDS, REGISTRY, CheckContract

__all__ = [
    "DEFECT_EXPECTATIONS",
    "ExpectationError",
    "MISSING_QUIET_ZONE_EXPECTED",
    "NON_DETERMINISTIC_MASK_EXPECTED",
    "OVER_CAPACITY_ACCEPTANCE_EXPECTED",
    "SELF_CHECK_TARGETS",
    "TRUNCATED_PAYLOAD_EXPECTED",
    "WRONG_EC_LEVEL_EXPECTED",
    "expected_failures",
    "expected_passing",
    "validate_against_registry",
]

# The adapter names the two built-in self-check targets are configured under, exactly as
# qrcheck.toml declares them in [targets.reference] and [targets.defective]. Private because a
# caller selects a target through the configuration, never through this table, and the names are
# quoted in diagnostics rather than compared by consumers.
_CONFORMING_TARGET: Final[str] = "reference"
_DEFECTIVE_TARGET: Final[str] = "defective"

# The twenty-one declared identifiers as a set, derived from the registry's own tuple so the list
# exists once in the product. Used by the complement query and by the declarative check that no
# expected identifier is a stranger.
_CHECK_ID_SET: Final[frozenset[str]] = frozenset(CHECK_IDS)

# The nine identifiers a --adapter defective run must fail (§0.4.2), as the one literal the
# per-defect sets are validated against at import. It is written here and nowhere else in this
# module: SELF_CHECK_TARGETS is built by unioning the five sets below, so this literal is the
# acceptance criterion those sets are checked *against* rather than a second declaration of them.
_EXPECTED_DEFECTIVE_IDS: Final[frozenset[str]] = frozenset(
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


class ExpectationError(ValueError):
    """The declared expectations are inconsistent with the checks that exist.

    Raised by :func:`validate_against_registry` and by this module's own import-time consistency
    check. Every trigger is a defect in the product rather than a user error in ``qrcheck.toml``: an
    expected identifier that is not one of the twenty-one declared checks, a per-defect set that no
    longer unions to the nine identifiers a defective run must fail, a blank target name, or an
    expected identifier that is declared but never registered. It subclasses :class:`ValueError` for
    the same reason :class:`~qrcheck.registry.check.CheckRegistrationError` does, and the message
    names the target and the offending identifiers in sorted order. Deleting an expectation or
    deleting a check therefore breaks a test instead of silencing one (§0.4.3).
    """


# One immutable set per injection of qrcheck/targets/defective.py, holding every check identifier
# that injection must make fail: the check designed to catch it plus any consequence the plan
# declares unavoidable. Named after the injection so the two declarations can be read side by
# side.
#
# D1 secures the payload at one error-correction level below the request while declaring the
# requested level. SY-EC-MATCH is the catch; SY-VERSION-MIN follows because a lower level fits a
# boundary payload in a smaller version, and SY-MATRIX-EQUIV because a pinned-EC matrix cannot
# match the oracle's.
WRONG_EC_LEVEL_EXPECTED: Final[frozenset[str]] = frozenset(
    {"SY-EC-MATCH", "SY-VERSION-MIN", "SY-MATRIX-EQUIV"}
)

# D2 encodes only the first 64 bytes of a longer payload when the full length selects the same
# version, so the symbol still decodes -- to the wrong bytes. Caught by RT-ROUNDTRIP; MD-CHARSET
# follows because its multi-byte and binary payloads are longer than that window.
TRUNCATED_PAYLOAD_EXPECTED: Final[frozenset[str]] = frozenset(
    {"RT-ROUNDTRIP", "MD-CHARSET"}
)

# D3 draws a one-module quiet zone where four or more were requested. Caught by RN-QUIET-ZONE;
# RN-MODULE-SIZE follows because the dimension contract (modules + 2 * quiet_zone) * module_size
# breaks with the zone actually drawn.
MISSING_QUIET_ZONE_EXPECTED: Final[frozenset[str]] = frozenset(
    {"RN-QUIET-ZONE", "RN-MODULE-SIZE"}
)

# D4 cycles the mask per call whenever the request pins none, and honours an explicit pin.
# DT-DETERMINISTIC is the only observation that sees the change -- it compares two calls of the
# identical request -- and the honoured pin is why SY-MASK and SY-MATRIX-EQUIV stay clean rather
# than gaining a second cause.
NON_DETERMINISTIC_MASK_EXPECTED: Final[frozenset[str]] = frozenset({"DT-DETERMINISTIC"})

# D5 accepts a payload above the capacity of the pinned version and level instead of rejecting it.
# IV-OVERCAPACITY builds that payload itself, outside the corpus, so no corpus-driven check
# observes the acceptance and this set stands alone.
OVER_CAPACITY_ACCEPTANCE_EXPECTED: Final[frozenset[str]] = frozenset(
    {"IV-OVERCAPACITY"}
)

# The five sets keyed by the injection names qrcheck/targets/defective.py uses in its DEFECTS
# table. This mapping is what makes the two declarations comparable entry by entry: a renamed
# injection, a moved consequence or a dropped catch shows up as a diff in one place. Read-only,
# like every table in this module.
DEFECT_EXPECTATIONS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "wrong_ec_level": WRONG_EC_LEVEL_EXPECTED,
        "truncated_payload": TRUNCATED_PAYLOAD_EXPECTED,
        "missing_quiet_zone": MISSING_QUIET_ZONE_EXPECTED,
        "non_deterministic_mask": NON_DETERMINISTIC_MASK_EXPECTED,
        "over_capacity_acceptance": OVER_CAPACITY_ACCEPTANCE_EXPECTED,
    }
)

# What each built-in self-check target must produce, and the only declaration of it.
#
# ``reference`` conforms, so its expected-failing set is empty -- and the reference gate
# additionally demands zero skipped results, so "make it pass by skipping" is not available
# either. ``defective`` carries every injection at once, so its set is the union of the five
# per-defect sets, computed rather than typed: the nine identifiers are never listed a second time
# here, which is what keeps a per-defect edit from silently disagreeing with the acceptance gate.
SELF_CHECK_TARGETS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        _CONFORMING_TARGET: frozenset(),
        _DEFECTIVE_TARGET: frozenset().union(*DEFECT_EXPECTATIONS.values()),
    }
)


def _declared_failing_union() -> frozenset[str]:
    """Return every check identifier the five per-defect sets expect to fail.

    The union is derived from :data:`DEFECT_EXPECTATIONS` alone, so it answers "what do the
    injections claim" independently of what :data:`SELF_CHECK_TARGETS` currently says; comparing the
    two is how a disagreement between them is detected.

    Returns:
        The union of the five per-defect sets, as an immutable set.
    """
    return frozenset().union(*DEFECT_EXPECTATIONS.values())


def _verify_union_is_expected() -> None:
    """Fail the import unless the per-defect sets unite to the nine declared identifiers.

    Run at module definition time, because this is a declaration invariant rather than validation of
    user input: editing a per-defect set so that the accepted set changes must break the process
    that imports it, never a run that silently compares its results against a different set. An
    explicit raise rather than an ``assert``, since ``python -O`` strips assertions and this guard
    is the defective run's acceptance criterion itself: the union is not exactly
    :data:`_EXPECTED_DEFECTIVE_IDS` and the message names what is missing and what is extra.
    """
    declared = SELF_CHECK_TARGETS[_DEFECTIVE_TARGET]
    if declared == _EXPECTED_DEFECTIVE_IDS:
        return
    raise ExpectationError(
        f"the per-defect sets no longer union to the nine checks a "
        f"--adapter {_DEFECTIVE_TARGET} run must fail: missing "
        f"{sorted(_EXPECTED_DEFECTIVE_IDS - declared)}, unexpected "
        f"{sorted(declared - _EXPECTED_DEFECTIVE_IDS)}"
    )


_verify_union_is_expected()


def expected_failures(adapter: str) -> frozenset[str] | None:
    """Return the check identifiers a run against `adapter` must fail, if declared.

    The declaration covers the two built-in self-check targets only. ``None`` -- rather than an
    empty set -- is the answer for every other name, because the framework holds no expectation
    about a user-supplied ``library``, ``cli`` or ``http`` generator whose defects are unknown: an
    empty set would assert that such a target must be clean, a claim nothing in the product makes. A
    caller that needs "these must pass" asks :func:`expected_passing`, which answers the complement
    over all twenty-one checks. ``adapter`` is a target name as declared in ``qrcheck.toml`` or
    given to ``--adapter``.
    """
    return SELF_CHECK_TARGETS.get(adapter)


def expected_passing(adapter: str) -> frozenset[str]:
    """Return the check identifiers a run against `adapter` must pass.

    The complement of :func:`expected_failures` over the twenty-one declared identifiers, so the two
    answers partition them for every adapter name: a target cannot be expected to fail and pass the
    same check, and no check escapes both. For ``defective`` this is the twelve checks §0.4.2
    requires to pass in the same run that fails the other nine; for ``reference`` it is all
    twenty-one. An adapter with no declared expectation yields every identifier, because nothing is
    excused for it: the caller decides what a failure of one of them means, and this module never
    claims such a target must be clean, it only declines to rule any of its checks out.
    """
    failing = expected_failures(adapter)
    if failing is None:
        return _CHECK_ID_SET
    return _CHECK_ID_SET - failing


def validate_against_registry(
    registry: Mapping[str, CheckContract] | None = None,
) -> None:
    """Check the declared expectations against the checks that actually exist.

    Four defects are reported together, so one call names everything that is wrong rather than the
    first thing a reader has to fix: a target name that is blank, an expected identifier outside the
    twenty-one declared checks, a ``defective`` entry that is not the union of the per-defect sets,
    and an expected identifier that is declared but not registered in `registry`. Deleting a check
    or an expectation therefore breaks a test instead of silently shrinking what the acceptance gate
    compares (§0.4.3). It reads only what it is given and the module's own tables -- no I/O, no
    mutation of the registry, no import of a check module -- so the caller decides when the check
    families have been imported, and ``None`` reads the product's
    :data:`~qrcheck.registry.check.REGISTRY` as it stands at call time. It either passes or raises
    :class:`ExpectationError`, never returning a problem list a caller could ignore.
    """
    registered: Mapping[str, CheckContract] = REGISTRY if registry is None else registry
    registered_ids = frozenset(registered)
    problems: list[str] = []

    for target, expected in SELF_CHECK_TARGETS.items():
        if not isinstance(target, str) or not target.strip():
            problems.append(f"target name {target!r} is blank")
            continue
        undeclared = frozenset(expected) - _CHECK_ID_SET
        if undeclared:
            problems.append(
                f"target {target!r} expects identifiers that are not check "
                f"identifiers: {sorted(undeclared)}"
            )
        unregistered = frozenset(expected) - registered_ids
        if unregistered:
            problems.append(
                f"target {target!r} expects unregistered checks: {sorted(unregistered)}"
            )

    aggregated = frozenset(SELF_CHECK_TARGETS.get(_DEFECTIVE_TARGET, frozenset()))
    declared_union = _declared_failing_union()
    if aggregated != declared_union:
        problems.append(
            f"target {_DEFECTIVE_TARGET!r} declares {sorted(aggregated)}, but the "
            f"per-defect sets derive {sorted(declared_union)}"
        )

    if problems:
        raise ExpectationError(
            "the declared expectations do not match the check registry: "
            + "; ".join(problems)
        )
