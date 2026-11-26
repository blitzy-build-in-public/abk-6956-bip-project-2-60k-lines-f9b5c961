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
module under this package imports the check layer, which is why the five imports
below are the whole of what this package may reach for.

Layering, and why these imports are eager
-----------------------------------------
This module and ``qrcheck/cli/doctor.py`` are the only places allowed to import a
concrete adapter.  Importing any ``qrcheck.targets`` submodule executes this file
first, so the imports below run on every ``import qrcheck.targets`` and pull in
``segno`` (through the two built-in targets) and ``httpx`` (through the HTTP
adapter).  That cost is accepted: both are mandatory, exactly pinned runtime
dependencies of the distribution, and one table readable in one place is what
makes the extension point auditable.  It is recorded here so a later reader does
not "optimise" the table into a module-level ``__getattr__`` -- which would break
``from qrcheck.targets import ADAPTER_TYPES`` for a static checker -- or, worse,
add a further heavy import.  This module performs no I/O, reads no file, no
environment variable and no clock, and instantiates no adapter.

Conformance
-----------
``tests/unit/test_adapter_types.py`` asserts that the keys here equal
``ADAPTER_TYPE_NAMES`` and that each entry, given a record of its own dataclass
type, returns an object for which ``isinstance(obj, Adapter)`` holds --
``isinstance`` and never ``issubclass``, because a runtime-checkable protocol
with non-method members (``name`` and ``capabilities``) rejects ``issubclass``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from .base import Adapter
from .cli import CliAdapter
from .defective import DefectiveAdapter
from .http import HttpAdapter
from .library import LibraryAdapter
from .reference import ReferenceAdapter

__all__ = ["ADAPTER_TYPES"]

# Keyed by the ``type`` value of a ``[targets.<name>]`` table, in the declaration
# order of ``qrcheck.config.models.ADAPTER_TYPE_NAMES``: the schema validates a
# ``type`` against that tuple and this table resolves the validated value, so the
# two must not drift.  Each value is the class itself, called as
# ``ADAPTER_TYPES[target.type](target, name=target_name)`` -- every implementation
# takes its resolved per-type record positionally and the configured target name as
# a keyword-only argument that defaults to ``config.type``.  The
# ``MappingProxyType`` is not decoration: the table is configuration vocabulary
# rather than run state, and a caller able to mutate it could redirect a whole run.
ADAPTER_TYPES: Mapping[str, Callable[..., Adapter]] = MappingProxyType(
    {
        "library": LibraryAdapter,
        "cli": CliAdapter,
        "http": HttpAdapter,
        "reference": ReferenceAdapter,
        "defective": DefectiveAdapter,
    }
)
