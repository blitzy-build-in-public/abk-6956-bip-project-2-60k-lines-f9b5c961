"""Registration of the twenty-one checks: the nine family modules, and the one loader.

Nothing is checked here. A check lives in the family module that owns its identifier prefix and
registers itself there through :func:`qrcheck.registry.check.check` at import. This module owns
the only list of those nine modules (:data:`FAMILY_MODULES`) and the single entry point that
imports them (:func:`load_families`). The tuple is explicit rather than discovered, so a stray
``.py`` cannot join the run and a missing module raises :class:`ImportError` at import of this
package instead of shrinking the registry. Import order is not significant:
:func:`qrcheck.registry.check.iter_checks` sorts by check identifier.

What a full run costs: 1,283 cases at EC M over the default 240-record corpus, 1,287 target
invocations and 1,470 primary decodes. Each family publishes its share of the encode count as
``EXPECTED_ENCODES`` and the sum is that 1,287. The decode count is the 1,462 of the nine families'
case matrix plus eight, because ``DM-RESILIENCE`` also decodes each of its eight sampled
records' pristine symbol once -- a memoised read the damage budget needs -- so the matrix's
decode column counts only the damaged reads. A conforming target performs 1,469 of them: the
over-capacity case it rejects leaves no image to decode.

Layering: the entry points import this package and this package imports nothing back -- no
runner, no CLI, no concrete adapter, no decoder or image library, no configuration -- since the
reverse import would close the registration cycle. Public surface: :data:`FAMILY_MODULES`,
:data:`FAMILY_MODULE_OBJECTS` and :func:`load_families`, and nothing else.
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

# The nine check family modules of ``qrcheck/checks/``, each the sole owner of the checks whose
# identifier prefix maps to it in ``qrcheck.registry.check.FAMILY_BY_PREFIX``: ``RT-ROUNDTRIP``;
# the four ``SY-*``; ``MD-MODE-OPTIMAL`` and ``MD-CHARSET``; the five ``RN-*``; ``DM-RESILIENCE``
# and ``DM-SEEDED``; the three ``IV-*``; ``DT-DETERMINISTIC``; ``PF-ENCODE`` and ``PF-DECODE``;
# ``CL-ARTIFACTS``.  Twenty-one checks, nine modules; ``support`` is excluded on purpose, being
# the shared helper every family imports that registers nothing.  The order is the package's own
# module order, not the sorted order, and does not need to be: see the module docstring.
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
# Populated by importing the package, which runs each module's ``@check`` decorators, and refreshed
# by :func:`load_families`.  Held for the tests that need a family's own declarations -- its
# ``EXPECTED_ENCODES`` -- without a second import path, which would register its checks twice and
# raise ``CheckRegistrationError``.  These are the modules Python already holds: no second
# definition of a check lives here.
FAMILY_MODULE_OBJECTS: dict[str, ModuleType] = {
    name: importlib.import_module(f"{__name__}.{name}") for name in FAMILY_MODULES
}


def load_families() -> tuple[str, ...]:
    """Import the nine check family modules and return the check identifiers registered.

    The idempotent entry point for the runner and for any test that wants the registry populated
    without relying on this package's import side effects: ``importlib`` caches every module, so
    calling it again re-runs no decorator and cannot duplicate a registration.

    Returns:
        Every registered check identifier, ascending -- the twenty-one ``CHECK_IDS``.

    Raises:
            ImportError: A family module of :data:`FAMILY_MODULES` does not exist or fails to
            import. Deliberately not caught: a broken or missing check must fail the process that
            imports the checks, never degrade to a smaller registry a report would describe as
            complete. CheckRegistrationError: A family module registers an identifier twice, or
            registers one that contradicts its declaration (see
            :class:`qrcheck.registry.check.CheckContract`).
    """
    for name in FAMILY_MODULES:
        FAMILY_MODULE_OBJECTS[name] = importlib.import_module(f"{__name__}.{name}")
    return registered_ids()
