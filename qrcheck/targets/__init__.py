"""The adapter factory map: the single extension point for a new target kind.

``ADAPTER_TYPES`` maps each ``type`` a ``[targets.<name>]`` table may declare to the class
implementing the :class:`~qrcheck.targets.base.Adapter` port, and
``ADAPTER_TYPES[target.type](target, name=target_name)`` is the only way an adapter is built. Adding
a kind takes four steps: the module, one entry here, the ``AdapterType`` value in
:mod:`qrcheck.config.models` and the per-type keys in :mod:`qrcheck.config.schema`. **No check
module changes**, which ``tests/unit/test_layering.py`` holds in both directions.

The table resolves lazily: importing any ``qrcheck.targets`` submodule executes this file first, so
a table holding the five classes would pull ``segno`` and ``httpx`` into processes that never build
an adapter. Iterating, the length, membership and the ``repr`` import nothing; only a value lookup
imports, and an undeclared key raises ``KeyError`` having imported nothing.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final

from .base import Adapter

__all__ = ["ADAPTER_TYPES"]

# Keyed by the ``type`` value of a ``[targets.<name>]`` table, in the order of
# ``qrcheck.config.models.ADAPTER_TYPE_NAMES``. Each entry is ``(module name, class name)``, so
# importing this package imports no adapter; the class is imported on lookup.
# ``qrcheck.config.models.ADAPTER_TYPE_NAMES``. Each entry is ``(module name, class name)`` and
# nothing else, so importing this package imports no adapter; the class is imported when its
# ``type`` is resolved and called as ``ADAPTER_TYPES[type](record, name=target_name)``. The
# ``MappingProxyType`` matters: configuration vocabulary must not be mutable run state.
_ADAPTER_ENTRIES: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "library": ("qrcheck.targets.library", "LibraryAdapter"),
        "cli": ("qrcheck.targets.cli", "CliAdapter"),
        "http": ("qrcheck.targets.http", "HttpAdapter"),
        "reference": ("qrcheck.targets.reference", "ReferenceAdapter"),
        "defective": ("qrcheck.targets.defective", "DefectiveAdapter"),
    }
)


class _AdapterTypes(Mapping[str, Callable[..., Adapter]]):
    """The factory map, resolving each adapter class the first time it is asked for.

    A row holds a module name and a class name, so importing ``qrcheck.targets`` imports no adapter.
    Enumeration, length, membership and ``repr`` never resolve a row; ``__getitem__`` is the one
    operation that does. The view is read-only by construction: it declares no mutator, and the rows
    it reads are a ``MappingProxyType`` over a private literal.
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> Callable[..., Adapter]:
        """Return the adapter class registered for ``key``, importing its module.

        Returns: The class implementing the port for that kind, ready to be called with the target's
        resolved per-type record and the configured target name.

        Raises:
        KeyError: ``key`` is not a declared target type; raised from the row lookup, so an unknown
        ``type`` imports no adapter.
        """
        module_name, class_name = _ADAPTER_ENTRIES[key]
        adapter: Callable[..., Adapter] = getattr(
            importlib.import_module(module_name), class_name
        )
        return adapter

    def __iter__(self) -> Iterator[str]:
        """Iterate the declared ``type`` values in declaration order, importing nothing."""
        return iter(_ADAPTER_ENTRIES)

    def __len__(self) -> int:
        """Return the number of declared target types, importing nothing."""
        return len(_ADAPTER_ENTRIES)

    def __contains__(self, key: object) -> bool:
        """Report whether ``key`` is a declared target type, importing nothing."""
        return key in _ADAPTER_ENTRIES

    def __repr__(self) -> str:
        """Render the declared keys, each shown as ``<lazy>`` so nothing is imported."""
        entries = ", ".join(f"{key!r}: <lazy>" for key in _ADAPTER_ENTRIES)
        return f"{type(self).__name__}({{{entries}}})"


ADAPTER_TYPES: Mapping[str, Callable[..., Adapter]] = _AdapterTypes()
