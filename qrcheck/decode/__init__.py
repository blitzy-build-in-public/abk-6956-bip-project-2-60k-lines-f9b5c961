"""Decoded evidence: the façade every check reads decoded evidence through.

All twenty-one checks consume their decoded evidence through this package, and
nothing here knows what any check asserts. Every piece of that evidence has one
owner elsewhere and is only re-exported here: the types and the availability
probes in :mod:`qrcheck.decode.base`, the two-decoder pipeline and the one-
authority rule in :mod:`qrcheck.decode.consensus`, the decoders in
:mod:`qrcheck.decode.zxing` and :mod:`qrcheck.decode.opencv`, the module-grid
sampler in :mod:`qrcheck.decode.matrix` and the format-information reader in
:mod:`qrcheck.decode.format_info`. This module defines no type, helper, constant
or decoding step of its own, so there is no second implementation of any of them.

Import cost is a contract. ``import qrcheck.decode`` must load the standard
library, ``numpy`` and :mod:`qrcheck.decode.base` and nothing more: ``cv2``,
``zxingcpp`` and ``PIL`` are never imported here (availability is probed with
``importlib.util.find_spec``, which never imports what it finds), and
:func:`decode_symbol` imports the consensus pipeline inside its own body. That
keeps ``qrcheck list`` and ``qrcheck doctor`` from paying for a native image
stack, and it keeps the degradation path below reachable on a host where the
secondary decoder is not installed.

The blocking-versus-degrading split belongs to this package's consumers, never to
this module. The primary decoder's absence is **blocking**: it is the authority
for every round-trip assertion, so no check may be silently skipped in its place
and ``qrcheck doctor`` and ``qrcheck run`` exit 2 naming ``zxing-cpp``. The
secondary decoder's absence is a **supported degradation**: the run proceeds in
single-decoder mode, ``agreement`` records ``primary_only``, the report carries a
banner stating it, and no check passes on the strength of the decoder that is
missing. :func:`decoder_availability`, :func:`primary_available` and
:func:`secondary_available` report those facts; nothing here raises, prints,
writes or decides an exit code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qrcheck.decode.base import (
    DECODER_ORDER,
    DecodeResult,
    DecoderAvailability,
    DecoderOutcome,
    Matrix,
    ModuleCountEstimate,
    PRIMARY_DECODER,
    SECONDARY_DECODER,
    SampledMatrix,
    decoder_availability,
    matrix_sha256,
    primary_available,
    secondary_available,
)

if TYPE_CHECKING:  # pragma: no cover - annotation resolution only, never executed
    from qrcheck.targets.base import GenerateRequest

__all__ = [
    "DECODER_ORDER",
    "DecodeResult",
    "DecoderAvailability",
    "DecoderOutcome",
    "Matrix",
    "ModuleCountEstimate",
    "PRIMARY_DECODER",
    "SECONDARY_DECODER",
    "SampledMatrix",
    "decoder_availability",
    "matrix_sha256",
    "primary_available",
    "secondary_available",
]


def decode_symbol(
    image_bytes: bytes,
    request: GenerateRequest,
    *,
    secondary: bool | None = None,
) -> DecodeResult:
    """Decode one artefact and return the single evidence object every check reads.

    This is a delegation, not a second pipeline: the consensus layer's
    :func:`qrcheck.decode.consensus.decode_symbol` is imported on the first call
    and is the only implementation of the decoding steps. The authority rule it
    applies is the one every check relies on: the primary decoder's payload is
    authoritative for all twenty-one checks, the secondary decoder only
    corroborates, and a symbol the primary could not read never passes on the
    secondary's strength.

    Args:
        image_bytes: The artefact exactly as the target returned it. Nothing is
            written and no path is accepted.
        request: The request the artefact was produced for; the consensus layer
            reads its colours to decide the polarity.
        secondary: ``True`` to run the secondary decoder, ``False`` to record
            single-decoder mode, ``None`` (the default) to probe whether it is
            installed.

    Returns:
        The assembled :class:`DecodeResult`. ``payload_bytes`` is the primary
        decoder's payload whenever it produced one and ``None`` otherwise, and
        ``ec_level_decoder`` and ``ec_level_format`` are recorded independently
        and never reconciled.

    Raises:
        ValueError: ``image_bytes`` is not bytes, or ``request`` carries no
            colours - both caller defects. Unusable bytes, an SVG artefact, an
            unreadable symbol and an unsamplable grid all return a result
            instead of raising.
    """
    from qrcheck.decode.consensus import decode_symbol as _decode_symbol

    return _decode_symbol(image_bytes, request, secondary=secondary)
