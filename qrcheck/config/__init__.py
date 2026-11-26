"""Public façade of :mod:`qrcheck.config`: resolution, diagnostics, records and seeding.

Four things are reachable from the package root, and nothing else:

* :func:`load_config` - read a ``qrcheck.toml``, validate it and resolve it into records;
* :class:`ConfigError` - the one diagnostic type, carrying the dotted key that has to change;
* the resolved records - :class:`Config`, :class:`TargetConfig`, :class:`LibraryTarget`,
  :class:`CLITarget`, :class:`HTTPTarget`, :class:`PerformanceLimits`, :class:`DamageLimits`
  and :class:`RenderConfig`;
* :func:`derive_seed` - the product's only seed-derivation point (AAP §0.5.2).

Importing this module reads no file, creates no directory, validates nothing and performs no
I/O: the names below are bindings to objects defined elsewhere, not wrappers, defaults or
copies. A second value of any threshold here would be a defect, so none is defined (AAP
§0.12: each declared value has exactly one owner).

The submodules stay importable for the interfaces that need the parts this façade does not
publish: ``qrcheck.config.schema.validate``, ``qrcheck.config.schema.apply_defaults`` and
``qrcheck.config.schema.SCHEMA`` for the raw-document entry points, and the
``qrcheck.config.models`` vocabularies (``AdapterType``, ``CapabilityName``, ``ReturnsKind``,
``ImageEncoding``, ``HttpMethod``, ``SVGRasterizer`` and their ``*_NAMES`` tuples) together
with its ``DEFAULT_*`` constants for the callers that build records directly.
"""

from qrcheck.config.load import load_config
from qrcheck.config.models import (
    CLITarget,
    Config,
    DamageLimits,
    HTTPTarget,
    LibraryTarget,
    PerformanceLimits,
    RenderConfig,
    TargetConfig,
)
from qrcheck.config.schema import ConfigError
from qrcheck.config.seeding import derive_seed

# The package's public surface, written out sorted (uppercase names before lowercase, as
# ``sorted()`` orders them) so ``import *``, ``dir()`` and any surface assertion are stable
# across runs and across machines (AAP §0.5.3).
__all__: tuple[str, ...] = (
    "CLITarget",
    "Config",
    "ConfigError",
    "DamageLimits",
    "HTTPTarget",
    "LibraryTarget",
    "PerformanceLimits",
    "RenderConfig",
    "TargetConfig",
    "derive_seed",
    "load_config",
)


def __dir__() -> list[str]:
    """Return the sorted public surface, so ``dir()`` reports exactly :data:`__all__`.

    Importing a submodule binds it as an attribute of its parent package, so the attributes
    ``load``, ``models``, ``schema`` and ``seeding`` exist here once this module has run.
    They are the machinery of those imports rather than part of the façade, and publishing
    them would make the documented surface disagree with what introspection reports.
    """
    return sorted(__all__)
