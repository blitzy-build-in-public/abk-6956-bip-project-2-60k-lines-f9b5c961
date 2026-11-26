"""Package façade for :mod:`qrcheck.evidence`: the evidence layout and the image boundary.

The package owns two things and nothing else. :mod:`qrcheck.evidence.naming` names every
evidence location and sanitizes every path segment that becomes one; it imports only the
standard library, so a process that merely writes a summary can import it without an image
stack. :mod:`qrcheck.evidence.mux` is the framework's single image boundary - the one place
Pillow is imported, and the one route an image takes into a report.

Why the two halves are re-exported differently
----------------------------------------------
:mod:`qrcheck.report` imports :data:`EVIDENCE_DIR_NAME` from ``qrcheck.evidence.naming`` to
default the evidence root, and that import must not drag an image stack into a run that
never renders a picture (AAP §0.8.1, §0.9.1). Python executes this module before any
submodule, so an eager ``from qrcheck.evidence.mux import to_data_uri`` here would pull
Pillow into every ``import qrcheck.evidence.naming`` and defeat that guarantee.

The same asymmetry is what keeps the import cycle open: ``qrcheck.report`` imports
``qrcheck.evidence.naming``, while ``naming`` imports ``qrcheck.report.model`` inside the
two functions that derive a failure ID. The two packages must therefore be able to
half-initialise in either order, and importing ``mux`` - which reaches ``qrcheck.report``
transitively - at this module's scope would close the cycle at the wrong moment.

So every public name of ``naming`` is re-exported eagerly (cheap, stdlib only, no side
effects), and every public name of ``mux`` is re-exported lazily through :pep:`562`
``__getattr__`` / ``__dir__``, where the first attribute access is what imports
``qrcheck.evidence.mux``.

Nothing else happens here. This module creates no directory, reads no file or
configuration, imports no image library, registers nothing, defines no version literal and
adds no second report entry point: a re-export is a name binding, not a conversion.
"""

import importlib

from qrcheck.evidence.naming import (
    DAMAGE_ARTIFACT_TEMPLATE,
    DAMAGE_KINDS,
    DECODED_JSON,
    EVIDENCE_DIR_NAME,
    FIXED_ARTIFACT_NAMES,
    LOGO_PNG,
    MAX_COMPONENT_LENGTH,
    REQUEST_JSON,
    SYMBOL_PNG,
    SYMBOL_REPEAT_PNG,
    SYMBOL_SVG,
    case_evidence_dir,
    damage_artifact_name,
    evidence_dir,
    is_evidence_artifact,
    relative_case_evidence_dir,
    relative_evidence_dir,
    sanitize,
    symbol_artifact_name,
)

# The lazy half, declared in one place. A name in this set is resolved from
# ``qrcheck.evidence.mux`` on first access and cached in this module's globals; the set is
# disjoint from the eager names above, so a cache entry can never shadow one of them.
# mux's magic-byte and MIME constants are deliberately not re-exported: the adapters read
# them from the module that owns them, and this façade publishes only the nine names its
# consumers import from the package.
_MUX_EXPORTS: frozenset[str] = frozenset(
    {
        "OpenImage",
        "decode_base64",
        "decode_image",
        "encode_base64",
        "encode_png",
        "mime_type",
        "open_image",
        "open_images",
        "to_data_uri",
    }
)

# The package's public surface, sorted so ``dir()`` and ``import *`` are stable across runs
# (AAP §0.5.3). ``import *`` resolves the lazy half through ``__getattr__``.
__all__: list[str] = sorted(
    [
        "DAMAGE_ARTIFACT_TEMPLATE",
        "DAMAGE_KINDS",
        "DECODED_JSON",
        "EVIDENCE_DIR_NAME",
        "FIXED_ARTIFACT_NAMES",
        "LOGO_PNG",
        "MAX_COMPONENT_LENGTH",
        "REQUEST_JSON",
        "SYMBOL_PNG",
        "SYMBOL_REPEAT_PNG",
        "SYMBOL_SVG",
        "case_evidence_dir",
        "damage_artifact_name",
        "evidence_dir",
        "is_evidence_artifact",
        "relative_case_evidence_dir",
        "relative_evidence_dir",
        "sanitize",
        "symbol_artifact_name",
        *_MUX_EXPORTS,
    ]
)


def __getattr__(name: str) -> object:
    """Return one of the lazily re-exported ``qrcheck.evidence.mux`` names.

    Accessing a mux name is what imports ``qrcheck.evidence.mux`` - and, with it, Pillow -
    so merely importing ``qrcheck.evidence`` or ``qrcheck.evidence.naming`` stays free of
    an image stack. The resolved object is cached in this module's globals, so later
    lookups take the ordinary attribute path.

    Args:
        name: The attribute being resolved on the package.

    Returns:
        The attribute of the same name from ``qrcheck.evidence.mux``.

    Raises:
        AttributeError: if ``name`` is not an exported mux name. Nothing else is raised for
            an unknown attribute: callers use ``hasattr`` to probe optional surface, and a
            cache miss has to answer ``False`` rather than raise a different exception.
    """
    if name in _MUX_EXPORTS:
        cache = globals().setdefault(
            name, getattr(importlib.import_module("qrcheck.evidence.mux"), name)
        )
        return cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sorted public surface of the package, matching :data:`__all__`."""
    return sorted(__all__)
