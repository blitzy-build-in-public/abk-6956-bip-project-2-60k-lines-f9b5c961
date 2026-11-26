"""The package surface of the check registry: what a check is, and which ones run.

Four modules own the registry -- what a check is (:mod:`qrcheck.registry.check`, which holds the
twenty-one identifiers, the nine families, :class:`~qrcheck.registry.check.CheckContract`, the
registration decorator and :data:`~qrcheck.registry.check.REGISTRY`), which checks a run will
attempt (:mod:`qrcheck.registry.selection`'s ``--checks``/``--class`` grammar and
:mod:`qrcheck.registry.resolve`'s resolution, which turns an unrunnable check into a ``skipped``
result before execution rather than a pass), and what the built-in ``reference`` and ``defective``
targets must produce (:mod:`qrcheck.registry.expectations`, the product's proof that its own suite
is not vacuous) -- and this module re-exports them so a consumer names one package,
``qrcheck.registry``, instead of the module each concern lives in.

Nothing is defined here: every name in ``__all__`` is the exact object its home module defines, and
no second definition may exist, because two registries would let ``qrcheck list`` and a run disagree
about which checks exist and two expected-failure sets would let the non-vacuity gate compare a run
against a stale criterion.

Two load-bearing invariants. **This package never imports** :mod:`qrcheck.checks`: the nine check
family modules import *this* package to register themselves, so an import in this direction would
close the registration cycle, and registration belongs to the entry points -- ``qrcheck run``
imports ``qrcheck.checks`` through :mod:`qrcheck.runner`, and ``qrcheck list`` imports it itself.
And **importing it leaves the registry empty**, because a package that populated the registry on
import would make the executed set depend on whether some unrelated module happened to import this
one (AAP §0.5).
"""

from __future__ import annotations

# What a check is: the identifiers, the families, the contract, the decorator and the registry the
# family modules write to -- owned by qrcheck/registry/check.py.
from qrcheck.registry.check import (
    CHECK_IDS,
    Case,
    CheckContract,
    CheckRegistrationError,
    DEPENDENCY_PRIMARY_DECODER,
    DEPENDENCY_SECONDARY_DECODER,
    DEPENDENCY_SVG_RASTERIZER,
    DEPENDENCY_TOKENS,
    FAMILY_BY_PREFIX,
    FAMILY_NAMES,
    REGISTRY,
    RunContext,
    check,
    get_check,
    isolated_registry,
    iter_checks,
    register,
    registered_ids,
)

# Which checks a run will attempt, from a --checks/--class selection -- owned by
# qrcheck/registry/selection.py.
from qrcheck.registry.selection import Selection, SelectionError, matches, parse_selection, select

# Which selected checks are runnable against a target's declared capabilities and the runtime
# dependencies present -- owned by qrcheck/registry/resolve.py.
from qrcheck.registry.resolve import (
    CAPABILITY_CASE_ID,
    Resolution,
    SkipOutcome,
    resolve,
    skip_case_results,
    unmet_capabilities,
    unmet_dependencies,
)

# What the built-in self-check targets must fail, which is the acceptance criterion a defective
# run is compared against -- owned by qrcheck/registry/expectations.py.
from qrcheck.registry.expectations import (
    DEFECT_EXPECTATIONS,
    ExpectationError,
    MISSING_QUIET_ZONE_EXPECTED,
    NON_DETERMINISTIC_MASK_EXPECTED,
    OVER_CAPACITY_ACCEPTANCE_EXPECTED,
    SELF_CHECK_TARGETS,
    TRUNCATED_PAYLOAD_EXPECTED,
    WRONG_EC_LEVEL_EXPECTED,
    expected_failures,
    expected_passing,
    validate_against_registry,
)

__all__: list[str] = [
    "CAPABILITY_CASE_ID",
    "CHECK_IDS",
    "Case",
    "CheckContract",
    "CheckRegistrationError",
    "DEFECT_EXPECTATIONS",
    "DEPENDENCY_PRIMARY_DECODER",
    "DEPENDENCY_SECONDARY_DECODER",
    "DEPENDENCY_SVG_RASTERIZER",
    "DEPENDENCY_TOKENS",
    "ExpectationError",
    "FAMILY_BY_PREFIX",
    "FAMILY_NAMES",
    "MISSING_QUIET_ZONE_EXPECTED",
    "NON_DETERMINISTIC_MASK_EXPECTED",
    "OVER_CAPACITY_ACCEPTANCE_EXPECTED",
    "REGISTRY",
    "Resolution",
    "RunContext",
    "SELF_CHECK_TARGETS",
    "Selection",
    "SelectionError",
    "SkipOutcome",
    "TRUNCATED_PAYLOAD_EXPECTED",
    "WRONG_EC_LEVEL_EXPECTED",
    "check",
    "expected_failures",
    "expected_passing",
    "get_check",
    "isolated_registry",
    "iter_checks",
    "matches",
    "parse_selection",
    "register",
    "registered_ids",
    "resolve",
    "select",
    "skip_case_results",
    "unmet_capabilities",
    "unmet_dependencies",
    "validate_against_registry",
]
