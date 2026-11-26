"""The adapter factory map: the single extension point for a new target kind.

``ADAPTER_TYPES`` maps each ``type`` a ``[targets.<name>]`` table may declare to
the class that implements the :class:`~qrcheck.targets.base.Adapter` port for it.
Every consumer that resolves a target -- ``qrcheck/cli/run.py``,
``qrcheck/cli/doctor.py`` and the tests -- indexes this table and calls the class
with the target's resolved per-type record, so
``ADAPTER_TYPES[target.type](target, name=target_name)`` is the only way an
adapter is ever built.  Nothing here instantiates one.

Adding a target kind
--------------------
Four steps, and no fifth:

1. Add ``qrcheck/targets/<kind>.py`` implementing the ``Adapter`` protocol from
   :mod:`qrcheck.targets.base` -- ``name``, ``capabilities``, ``generate`` and
   ``close``, raising only the five declared error types.
2. Add one entry to ``ADAPTER_TYPES`` below.
3. Add the ``type`` value to ``AdapterType`` in :mod:`qrcheck.config.models`,
   whose ``ADAPTER_TYPE_NAMES`` this table's keys mirror exactly.
4. Add the kind's per-type keys to :mod:`qrcheck.config.schema`.

**No check module changes.**  That is the property this extension point exists
to preserve: a check sees only the port, so a fourth target kind needs no check
edit.  ``tests/unit/test_layering.py`` walks the AST of every module in the
check layer to hold it, and the same test holds the opposite direction -- no
module under this package imports the check layer, which is why the entries
below are the whole of what this package may reach for.

Layering, and why the table resolves lazily
-------------------------------------------
This module and ``qrcheck/cli/doctor.py`` are the only places allowed to name a
concrete adapter.  Importing any ``qrcheck.targets`` submodule executes this
file first, so a table holding the five classes would run on every
``import qrcheck.targets`` and pull ``segno`` (through the two built-in targets)
and ``httpx`` -- and, with ``httpx``, ``click`` -- into processes that never
build an adapter: ``import qrcheck.registry.check`` needs only the port's
``Capability``, and ``qrcheck list`` needs only the check registry.  Each entry
therefore records a module name and a class name, and the class itself is
imported when its ``type`` is resolved.

What the extension point promises does not change.  The table is still one table
readable in one place, adding a kind is still one entry, and
``from qrcheck.targets import ADAPTER_TYPES`` still yields a
:class:`~collections.abc.Mapping` whose values are the classes themselves, called
as ``ADAPTER_TYPES[target.type](target, name=target_name)``.  Iterating it,
taking its length, testing membership and reading its ``repr`` import nothing;
only a value lookup imports, and a lookup of an undeclared key raises
``KeyError`` having imported nothing.  This module performs no I/O, reads no
file, no environment variable and no clock, and instantiates no adapter.

Conformance
-----------
``tests/unit/test_adapter_types.py`` asserts that the keys here equal
``ADAPTER_TYPE_NAMES`` and that each entry, given a record of its own dataclass
type, returns an object for which ``isinstance(obj, Adapter)`` holds --
``isinstance`` and never ``issubclass``, because a runtime-checkable protocol
with non-method members (``name`` and ``capabilities``) rejects ``issubclass``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final

from .base import Adapter

__all__ = ["ADAPTER_TYPES"]

# Keyed by the ``type`` value of a ``[targets.<name>]`` table, in the declaration
# order of ``qrcheck.config.models.ADAPTER_TYPE_NAMES``: the schema validates a
# ``type`` against that tuple and this table resolves the validated value, so the
# two must not drift.  Each entry is ``(module name, class name)`` and nothing
# else: the class is imported when its ``type`` is resolved, so importing this
# package imports no adapter at all.  Every implementation takes its resolved
# per-type record positionally and the configured target name as a keyword-only
# argument that defaults to ``config.type``, which is why the resolved class is
# still called as ``ADAPTER_TYPES[target.type](target, name=target_name)``.  The
# ``MappingProxyType`` is not decoration: the table is configuration vocabulary
# rather than run state, and a caller able to mutate it could redirect a whole run.
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

    A row holds a module name and a class name rather than the class, so importing
    ``qrcheck.targets`` imports no adapter and ``segno``, ``httpx`` and ``click`` reach a
    process only when a target of that kind is actually built.  Enumeration, length,
    membership and ``repr`` never resolve a row and so import nothing; ``__getitem__`` is
    the one operation that does, and it returns the class itself, leaving
    ``ADAPTER_TYPES[target.type](target, name=target_name)`` the only way an adapter is
    ever built.  The view is read-only by construction: it declares no mutator, and the
    rows it reads are a ``MappingProxyType`` over a private literal.
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> Callable[..., Adapter]:
        """Return the adapter class registered for ``key``, importing its module.

        Args:
            key: One of the ``type`` values ``qrcheck.config.models.AdapterType`` declares.

        Returns:
            The class implementing the port for that kind, ready to be called with the
            target's resolved per-type record and the configured target name.

        Raises:
            KeyError: ``key`` is not a declared target type.  It is raised from the row
                lookup, so an unknown ``type`` imports no adapter.
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
