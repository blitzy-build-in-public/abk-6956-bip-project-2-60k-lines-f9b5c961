"""Registration of the twenty-one checks: the nine family modules, and the one loader.

Nothing is checked in this module.  A check lives in the family module that owns its
identifier prefix -- ``RT-ROUNDTRIP`` in :mod:`qrcheck.checks.roundtrip`, the four
``SY-*`` checks in :mod:`qrcheck.checks.symbol`, and so on -- and registers itself there
through :func:`qrcheck.registry.check.check` while that module imports.  What this module
owns is the only list of those nine modules (:data:`FAMILY_MODULES`) and the single entry
point that imports them (:func:`load_families`), so *which* checks exist never depends on
whether some unrelated module happened to import a family.

Why an explicit tuple rather than discovery
-------------------------------------------
``pkgutil.iter_modules``, a directory walk and an entry-point group would each make the
registered set depend on the contents of a directory at run time: a stray ``.py`` would
join the run, and a missing module would shrink the registry instead of failing the import.
The tuple is frozen, is reviewable in a diff, and a missing family module raises
:class:`ImportError` at import of this package -- loudly.  The product's own gate that all
twenty-one checks exist cannot be met by quietly having fewer of them.

Import order never reaches an artefact
--------------------------------------
The nine names below are in the plan's tree order, which is deliberately not the
alphabetical order of :data:`qrcheck.registry.check.FAMILY_NAMES`.  Import order is
therefore not significant: :func:`qrcheck.registry.check.iter_checks` sorts by check
identifier, and that sorted iteration is the only one ``qrcheck list``, the runner's
execution walk and the three report writers use (AAP §0.3.4, §0.5.3).  Two runs of the
same configuration describe their checks in the same order whatever order they were
imported in.

What a full run costs
---------------------
The twenty-one checks make one full run at EC M over the default 240-record corpus exactly
1,283 cases, which invoke the target 1,287 times and the primary decoder 1,462 times
(AAP §0.8.1's case matrix).  Each family module publishes its own share of the encode
count as ``EXPECTED_ENCODES``, and the sum over the nine modules is that 1,287; a case
re-run would show up as a count above the declaration rather than as a quiet success.

Layering
--------
The entry points import this package -- :mod:`qrcheck.runner` before a run, and
``qrcheck list`` itself -- and this package imports nothing back: no runner, no CLI, no
concrete adapter, no decoder or image library, and no configuration.  Importing
``qrcheck.checks`` costs the standard library, a few pure-Python product modules and the
registration itself.  :mod:`qrcheck.registry` must never import this package in turn: the
family modules import the registry in order to register, so the reverse import would close
the registration cycle and run it at whatever point the cycle was cut.

Binding constraints
-------------------
No project rules exist for this repository; the rules document is empty.  The constraints
on this module are therefore the plan's own MUST/MUST NOT list: nine named modules, one
table, no discovery, no conditional or duplicate registration, no ``print``, no file or
directory creation, and no silent degradation to a smaller registry.

Public surface
--------------
:data:`FAMILY_MODULES`, :data:`FAMILY_MODULE_OBJECTS` and :func:`load_families`, and
nothing else.  The checks themselves are reached through :mod:`qrcheck.registry.check` and
are never re-exported from here: a second name for a check would be a second thing to keep
in step with the registry.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from qrcheck.registry.check import registered_ids

__all__: list[str] = [
    "FAMILY_MODULES",
    "FAMILY_MODULE_OBJECTS",
    "load_families",
]

# The nine check family modules of ``qrcheck/checks/``, each the sole owner of the checks
# whose identifier prefix maps to it in ``qrcheck.registry.check.FAMILY_BY_PREFIX``:
# ``RT-ROUNDTRIP``; the four ``SY-*``; ``MD-MODE-OPTIMAL`` and ``MD-CHARSET``; the five
# ``RN-*``; ``DM-RESILIENCE`` and ``DM-SEEDED``; the three ``IV-*``; ``DT-DETERMINISTIC``;
# ``PF-ENCODE`` and ``PF-DECODE``; ``CL-ARTIFACTS``.  Twenty-one checks, nine modules, and
# ``support`` is excluded on purpose: it is the shared helper every family imports, not a
# family, and it registers nothing.
#
# The order is the plan's tree order.  It is not the sorted order and does not need to be:
# see the module docstring on why registration order cannot reach an artefact.
FAMILY_MODULES: tuple[str, ...] = (
    "roundtrip",
    "symbol",
    "mode",
    "render",
    "damage",
    "invalid",
    "determinism",
    "performance",
    "cleanup",
)

# The imported module objects, keyed by family name in ``FAMILY_MODULES`` order.
#
# Populated by importing the package, which is what runs each module's ``@check``
# decorators, and refreshed by :func:`load_families`.  Held for the tests and diagnostics
# that need to reach a family module's own declarations -- its ``EXPECTED_ENCODES``, for
# instance -- without importing a second path to the same file, which would register its
# checks twice and raise ``CheckRegistrationError``.  Nothing here is a second definition
# of a check: the objects in it are the modules Python already holds in ``sys.modules``.
FAMILY_MODULE_OBJECTS: dict[str, ModuleType] = {
    name: importlib.import_module(f"{__name__}.{name}") for name in FAMILY_MODULES
}


def load_families() -> tuple[str, ...]:
    """Import the nine check family modules and return the check identifiers registered.

    The idempotent entry point for the runner and for any test that wants the registry
    populated without relying on this package's import side effects alone.  ``importlib``
    caches every module, so calling it again re-binds the same module objects, re-runs no
    decorator and cannot duplicate a registration; the identifiers it returns are therefore
    the same tuple every time, and the registry stays at twenty-one entries.

    Returns:
        Every registered check identifier, ascending -- the twenty-one ``CHECK_IDS`` when
        all nine families imported.

    Raises:
        ImportError: A family module of :data:`FAMILY_MODULES` does not exist or fails to
            import.  Deliberately not caught: a broken or missing check must fail the
            process that imports the checks, never degrade to a smaller registry that a
            report would then describe as complete.
        CheckRegistrationError: A family module registers an identifier twice, or registers
            one that contradicts its declaration (see
            :class:`qrcheck.registry.check.CheckContract`).  Also deliberately uncaught: the
            executed set, the expected-failure table and ``qrcheck list`` must never
            disagree about which checks exist.
    """
    for name in FAMILY_MODULES:
        FAMILY_MODULE_OBJECTS[name] = importlib.import_module(f"{__name__}.{name}")
    return registered_ids()
