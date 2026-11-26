"""The :mod:`qrcheck` package surface: one version literal and five lazily resolved names.

``qrcheck`` tests a QR code *generator* rather than a scanner: it drives a declared target -- one of
the three configurable adapters (``library``, ``cli``, ``http``) or one of the two built-in
self-check targets (``reference``, ``defective``) -- decodes every symbol the target produces with
two independent reference decoders, and compares the outcome against a golden-encoder oracle used
only as an expectation source. This module holds only the distribution version and the five names a
library consumer or the documentation may import from the top level: the run orchestrator, the
line-count metric, the configuration loader and the two corpus functions. Each resolves on first
access through a :pep:`562` module ``__getattr__``, so ``import qrcheck`` imports no submodule and
no third-party package, and the console script's start-up cost stays independent of a run's decode
workload.

Nothing happens at import time -- no logging set-up, no argument parsing, no directory creation, no
environment read, no file IO, no network client -- and no second version literal, capability enum or
report schema lives here. With no target, no check and no result, this module raises none of the
five adapter errors (``TargetUnavailable``, ``TargetTimeout``, ``TargetRejected``,
``TargetContractError``, ``TargetUnsupported``) and reports no
:class:`~qrcheck.report.model.Reason`; its only failure mode is :exc:`AttributeError` for an
attribute it does not export.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Documented for a static reader; __getattr__ resolves these at first use.
    from qrcheck.config.load import load_config
    from qrcheck.corpus import build_corpus, load_corpus
    from qrcheck.loc import count_lines
    from qrcheck.runner import run_suite

# The distribution version, and the product's only version literal: `pyproject.toml` declares the
# same 0.1.0 statically (never dynamically, because a wheel build would have to import the package
# to resolve it), and `qrcheck doctor` reports this value rather than a copy of it.
__version__: str = "0.1.0"

# Which module owns each public name, declared in one place so the surface can be read at a glance
# and `__getattr__` needs no branch per name. Both corpus names share an owner, which is itself a
# lazily resolved façade.
_LAZY_EXPORTS: dict[str, str] = {
    "build_corpus": "qrcheck.corpus",
    "count_lines": "qrcheck.loc",
    "load_config": "qrcheck.config.load",
    "load_corpus": "qrcheck.corpus",
    "run_suite": "qrcheck.runner",
}

# The declared surface, sorted so that `from qrcheck import *`, `dir()` and any surface assertion
# are stable across runs and machines (AAP §0.5.3). Kept in lock-step with `_LAZY_EXPORTS` above
# by construction: the one list is derived from the other.
__all__: list[str] = sorted(["__version__", *_LAZY_EXPORTS])


def __getattr__(name: str) -> object:
    """Resolve one public name, importing the module that defines it on first access.

    The first access imports its owning module, fetches the object and caches it in this module's
    globals; later accesses take the ordinary attribute path and cost no import machinery, so a
    consumer that needs the line-count metric never pays for numpy and one that needs the corpus
    never pays for a decoder.

    Raises :exc:`AttributeError` for a name that is not lazily exported -- the only failure mode,
    and it matches what an ordinary module raises for an unknown attribute, so ``hasattr`` and
    ``getattr(..., default)`` keep behaving as their callers expect. A module that cannot be
    imported is not translated: its own ``ModuleNotFoundError`` reaches the caller, because a broken
    installation is an environment fault to report, not a missing attribute.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the sorted names of this module, with ``__all__`` always included.

    Sorted for the same reason ``__all__`` is: introspection and any listing built from it are then
    byte-stable across runs and machines, whichever names a caller happened to touch first.
    """
    return sorted(set(globals()) | set(__all__))
