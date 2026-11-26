"""Decoded evidence: the façade every check reads decoded evidence through.

Re-exports only: the types and availability probes of :mod:`qrcheck.decode.base`, the two-decoder
pipeline and one-authority rule of :mod:`qrcheck.decode.consensus`, the two decoders, the
module-grid sampler and the format-information reader. This module defines no type, helper or
decoding step of its own, so there is no second implementation of any of them.

Import cost is a contract: ``import qrcheck.decode`` loads the standard library, numpy and ``base``
only -- never ``cv2``, ``zxingcpp`` or ``PIL`` -- and :func:`decode_symbol` imports the consensus
pipeline inside its own body, so ``qrcheck list`` and ``qrcheck doctor`` pay for no native image
stack and the single-decoder degradation path stays reachable. Nothing here raises, prints, writes
or decides an exit code: the blocking-versus-degrading split belongs to this package's consumers.
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

    A delegation, not a second pipeline: :func:`qrcheck.decode.consensus.decode_symbol` is imported
    on the first call and is the only implementation of the decoding steps. The primary decoder's
    payload stays authoritative for all twenty-one checks, so a symbol it could not read never
    passes on the secondary's strength.

    Raises:
        ValueError: ``image_bytes`` is not bytes, or ``request`` carries no colours -- both
            caller defects. Unusable bytes, an SVG artefact, an unreadable symbol and an unsamplable
            grid return a result.
    """
    from qrcheck.decode.consensus import decode_symbol as _decode_symbol

    return _decode_symbol(image_bytes, request, secondary=secondary)
