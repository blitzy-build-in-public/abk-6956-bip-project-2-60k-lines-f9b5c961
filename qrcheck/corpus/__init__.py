"""Deterministic QR payload corpus: capacity tables, payload classes, the 240-record builder, and
the canonical JSONL format.

Importing this package performs no work: it is a :pep:`562` façade, so the five public names below
and the four submodules beside them are imported on first access and never before. Python executes
this module before any submodule, which is what makes the corpus's two halves
differ: :mod:`qrcheck.corpus.capacity` imports nothing at all, so a consumer that needs only the
capacity tables -- :mod:`qrcheck.targets.base` enforces a request's size through them -- loads no
third-party module. numpy, and with it :func:`qrcheck.corpus.payloads._verify_boundary_predicates`
re-checking the boundary arithmetic, arrives only when payload generation is actually reached:
building a corpus, or importing :mod:`qrcheck.corpus.payloads` itself.

Nothing else happens here. This module reads no file or configuration, generates no payload, seeds
no random source, registers nothing and defines no second corpus entry point: a re-export is a name
binding, not a conversion.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Documented for a static reader; __getattr__ imports these names at use.
    from qrcheck.corpus.builder import CorpusRecord, build_corpus
    from qrcheck.corpus.io import filter_by_class, load_corpus, write_corpus

# The public surface, sorted so ``import *``, ``dir()`` and surface assertions are stable.
__all__ = ("CorpusRecord", "build_corpus", "filter_by_class", "load_corpus", "write_corpus")

# Which module owns each public name. Both owners reach ``qrcheck.corpus.payloads``, so
# resolving any of the five imports numpy; ``capacity`` below costs nothing.
_EXPORT_MODULES: dict[str, str] = {
    "CorpusRecord": "qrcheck.corpus.builder",
    "build_corpus": "qrcheck.corpus.builder",
    "filter_by_class": "qrcheck.corpus.io",
    "load_corpus": "qrcheck.corpus.io",
    "write_corpus": "qrcheck.corpus.io",
}

# The package's own submodules, reachable from ``import qrcheck.corpus`` alone.
_SUBMODULES: frozenset[str] = frozenset({"builder", "capacity", "io", "payloads"})


def __getattr__(name: str) -> Any:
    """Return a public name or a submodule of the package, importing it on first access.

    Accessing a public name is what imports its owning module, so merely importing
    ``qrcheck.corpus`` stays free of every third-party module.

    Raises:
        AttributeError: ``name`` is neither a public name nor one of the package's submodules.
            Nothing else is raised for an unknown attribute, because callers probe surface with
            ``hasattr``.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        return globals().setdefault(
            name, getattr(importlib.import_module(module_name), name)
        )
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sorted public names and submodules of the package."""
    return sorted(set(__all__) | set(_SUBMODULES))
