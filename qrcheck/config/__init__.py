"""Public façade of :mod:`qrcheck.config`: resolution, diagnostics, records and seeding.

Four things are reachable from the package root and nothing else: :func:`load_config`,
:class:`ConfigError` carrying the dotted key that has to change, the resolved records (``Config``,
``TargetConfig``, ``LibraryTarget``, ``CLITarget``, ``HTTPTarget``, ``PerformanceLimits``,
``DamageLimits``, ``RenderConfig``) and :func:`derive_seed`, the product's only seed-derivation
point. Importing it reads no file, creates no directory and performs no I/O: the names below are
bindings to objects defined elsewhere, and no threshold has a second value here.

The submodules stay importable for the interfaces that need what this façade does not publish:
``schema.validate``, ``schema.apply_defaults`` and ``schema.SCHEMA``, plus the ``models``
vocabularies, ``DEFAULT_*`` constants and ``LOOPBACK_HOSTS`` allow-list for callers that build
records directly.
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
# across runs and across machines.
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

    Importing a submodule binds it as an attribute of its parent package, and publishing those would
    make the documented surface disagree with what introspection reports.
    """
    return sorted(__all__)
