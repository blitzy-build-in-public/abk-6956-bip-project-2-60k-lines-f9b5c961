"""Decoded-evidence types and decoder availability probes.

This module is the single owner of the *shape* of decoded evidence. Two
reference decoders, the consensus layer, the module-grid sampler and all
twenty-one checks speak the vocabulary defined here, and nothing else in the
product may define a second copy of it:

* :class:`DecoderOutcome` -- one decoder's answer, with the polarity that
  produced it.
* :class:`ModuleCountEstimate` and :class:`SampledMatrix` -- what grid
  sampling measured, including the path where no grid could be recovered.
* :class:`DecodeResult` -- the single evidence object every check consumes,
  and the exact projection persisted as ``decoded.json``.
* :class:`DecoderAvailability`, :func:`primary_available`,
  :func:`secondary_available`, :func:`decoder_availability` -- the probes
  ``registry/resolve.py``, ``runner.py`` and ``cli/doctor.py`` consume to
  decide that a missing primary decoder blocks a run while a missing
  secondary decoder only degrades it.

Constraints that hold for every symbol in this module:

* It imports only the standard library and ``numpy``. It imports no other
  ``qrcheck`` package, and neither ``cv2`` nor ``zxingcpp`` is imported at
  module import time; availability is resolved with
  :func:`importlib.util.find_spec`, which never imports the package it finds.
* It writes nothing, reads no configuration, logs nothing, decides no exit
  code and knows no check, status or reason vocabulary owned elsewhere. No
  adapter error originates here: the five adapter errors belong to
  ``qrcheck/targets/base.py``, statuses and reasons to
  ``qrcheck/report/model.py``.
* Every ``as_dict()`` returns plain ``dict``/``list``/``str``/``int``/
  ``float``/``bool``/``None`` -- never ``bytes``, never a numpy array, never a
  dataclass instance -- so the projection can be JSON-serialised verbatim.
* Determinism: decoder-keyed maps are stored sorted, digests come from
  :func:`hashlib.sha256`, and no wall clock, ``hash()``, ``id()`` or set
  iteration order reaches a key, a value or a digest.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import types
from collections.abc import Mapping
from typing import Any, Final, Literal

import numpy

__all__ = [
    "Agreement",
    "DECODER_ORDER",
    "DecodeResult",
    "DecoderAvailability",
    "DecoderName",
    "DecoderOutcome",
    "DecoderStatus",
    "Matrix",
    "ModuleCountEstimate",
    "PRIMARY_DECODER",
    "Polarity",
    "PolarityOutcome",
    "SECONDARY_DECODER",
    "SampledMatrix",
    "decoder_availability",
    "matrix_sha256",
    "primary_available",
    "secondary_available",
]

#: Which decoder produced an outcome. The vocabulary is closed at two.
DecoderName = Literal["opencv", "zxing"]

#: A decoder that found no symbol is ``undetected``; one that raised is
#: ``error``. Neither is a disagreement.
DecoderStatus = Literal["decoded", "undetected", "error"]

#: The polarity at which the authoritative payload was read -- the
#: ``polarity_applied`` vocabulary of :class:`DecodeResult`.
Polarity = Literal["original", "inverted"]

#: The polarity a single decoder's outcome came from. ``both`` only arises for
#: an inverted request whose decoder returned the same payload at each
#: polarity.
PolarityOutcome = Literal["original", "inverted", "both"]

#: The consensus classification of the two decoders' answers. ``undetected``
#: covers a symbol neither decoder could read; ``disagree`` requires two
#: differing payloads.
Agreement = Literal["agree", "primary_only", "secondary_only", "undetected", "disagree"]

#: The decoder whose payload is authoritative for every check -- one
#: authority, never the secondary decoder.
PRIMARY_DECODER: Final = "zxing"

#: The corroborating decoder. Its silence is recorded, never treated as a
#: defect report, and never rescues a symbol the primary decoder could not
#: read.
SECONDARY_DECODER: Final = "opencv"

#: Lexicographic decoder order, which is the splice order of every
#: ``per_decoder`` map so the evidence bytes do not depend on import or call
#: order.
DECODER_ORDER: Final[tuple[str, str]] = ("opencv", "zxing")

_AGREEMENT_VALUES: Final[frozenset[str]] = frozenset(
    {"agree", "primary_only", "secondary_only", "undetected", "disagree"}
)
_DECODER_STATUS_VALUES: Final[frozenset[str]] = frozenset({"decoded", "undetected", "error"})
_POLARITY_VALUES: Final[frozenset[str]] = frozenset({"original", "inverted"})
_POLARITY_OUTCOME_VALUES: Final[frozenset[str]] = frozenset({"original", "inverted", "both"})
_SAMPLING_SOURCES: Final[frozenset[str]] = frozenset({"primary", "straight_qrcode"})

#: The sampled module grid: a ``numpy.ndarray`` of dtype ``numpy.bool_``, shape
#: ``(N, N)``, C-order, ``True`` for a dark module, row index ``y`` and column
#: index ``x``. ``N = 4V + 17`` for a Model 2 symbol of version ``V``, which is
#: how the version is recovered from the sampled dimension alone. This is the
#: type ``SY-MASK``, ``SY-EC-MATCH``, ``SY-MATRIX-EQUIV``, ``SY-VERSION-MIN``
#: and the measured quiet zone read through the facts recorded beside it.
Matrix = numpy.ndarray


def matrix_sha256(matrix: Matrix) -> str:
    """Hash a sampled module grid canonically.

    This is the single definition of the digest ``DT-DETERMINISTIC`` compares
    across two calls of an identical request, and of the ``matrix_sha256``
    recorded in evidence. The grid is normalised to ``numpy.bool_`` in C order
    and packed to bits, and the module count is mixed into the digest, so two
    grids that are equal always hash equally and a grid of a different
    dimension can never collide with a prefix of another.

    Raises:
        ValueError: ``matrix`` is empty or not square -- a sampled grid is
            always ``(N, N)``, so anything else is a producer defect rather
            than evidence.
    """
    grid = numpy.asarray(matrix)
    if grid.ndim != 2:
        raise ValueError(f"matrix must be 2-dimensional; got {grid.ndim} dimension(s)")
    rows, columns = int(grid.shape[0]), int(grid.shape[1])
    if rows == 0 or columns == 0:
        raise ValueError("matrix must not be empty")
    if rows != columns:
        raise ValueError(f"matrix must be square; got {rows}x{columns}")
    flags = numpy.ascontiguousarray(grid.astype(numpy.bool_, copy=False)).astype(numpy.uint8)
    payload = f"{rows}x{rows}|".encode("ascii") + numpy.packbits(flags).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _payload_text(payload: bytes | None) -> str | None:
    """Return ``payload`` as text when it round-trips through UTF-8, else ``None``.

    A payload that is not valid UTF-8 has no faithful text form, so evidence
    records only its base64 form rather than a lossy replacement string.
    """
    if payload is None:
        return None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text.encode("utf-8") == payload else None


def _sorted_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``values`` as a plain dict with lexicographically sorted keys."""
    return dict(sorted(values.items()))


@dataclasses.dataclass(frozen=True, slots=True)
class DecoderOutcome:
    """One decoder's answer about one symbol, as recorded in ``decoded.json``.

    ``status`` is ``decoded`` when the decoder returned a payload,
    ``undetected`` when it found no symbol at all and ``error`` when it raised;
    only two differing ``decoded`` payloads constitute a disagreement.
    ``polarity`` records which polarity produced this answer, because the
    framework runs both decoders on the 255-inverted copy for an inverted
    request rather than relying on a decoder's inversion support.

    ``reported_ec`` and ``reported_version`` are the decoder's own metadata,
    recorded as independent evidence and never as the assertion. ``zxing-cpp``
    2.2.0 reports an EC level but neither version nor mask, and the secondary
    decoder reports no metadata at all, so ``None`` there is the answer rather
    than a gap; a future binding that exposes a version populates the field
    without a change to any consumer.

    Two fields extend the documented seven, each because the consensus rules and
    the grid sampler need evidence no other field carries:

    * ``payload`` -- the verbatim bytes this decoder returned. The consensus
      rules record both payloads verbatim when the decoders disagree, and the
      first-divergence offset of a truncated payload is only computable from
      them.
    * ``position`` -- the corner quadrangle in source-image pixel coordinates
      as the primary decoder reports it, which is the independent geometry
      input grid sampling needs because no pinned decoder exposes a version.
    """

    decoder: DecoderName
    status: DecoderStatus
    polarity: PolarityOutcome
    payload_sha256: str | None
    reported_ec: str | None
    reported_version: int | None
    error_text: str | None
    payload: bytes | None = None
    position: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Fill or verify the payload digest and reject off-vocabulary values.

        Raises:
            ValueError: ``decoder``, ``status`` or ``polarity`` is outside its
                closed vocabulary, or ``payload_sha256`` was supplied with a
                payload and disagrees with it -- the digest and the payload it
                names can never describe different bytes.
        """
        if self.decoder not in DECODER_ORDER:
            raise ValueError(f"unknown decoder {self.decoder!r}; expected one of {DECODER_ORDER}")
        if self.status not in _DECODER_STATUS_VALUES:
            raise ValueError(f"unknown decoder status {self.status!r}")
        if self.polarity not in _POLARITY_OUTCOME_VALUES:
            raise ValueError(f"unknown decoder polarity {self.polarity!r}")
        if self.payload is not None:
            digest = hashlib.sha256(self.payload).hexdigest()
            if self.payload_sha256 is None:
                object.__setattr__(self, "payload_sha256", digest)
            elif self.payload_sha256 != digest:
                raise ValueError(
                    f"{self.decoder} payload_sha256 {self.payload_sha256} does not match "
                    f"its payload ({digest})"
                )

    def as_dict(self) -> dict[str, Any]:
        """Project this outcome to a JSON-safe mapping with sorted keys.

        The seven documented fields are always present. ``payload_b64`` is
        added whenever a payload exists so the bytes survive byte-exactly, and
        ``payload_text`` only when the payload round-trips through UTF-8. The
        corner quadrangle is emitted as a list of ``[x, y]`` pairs.
        """
        projected: dict[str, Any] = {
            "decoder": self.decoder,
            "error_text": self.error_text,
            "payload_sha256": self.payload_sha256,
            "polarity": self.polarity,
            "position": None
            if self.position is None
            else [[float(point[0]), float(point[1])] for point in self.position],
            "reported_ec": self.reported_ec,
            "reported_version": self.reported_version,
            "status": self.status,
        }
        if self.payload is not None:
            projected["payload_b64"] = base64.b64encode(self.payload).decode("ascii")
            text = _payload_text(self.payload)
            if text is not None:
                projected["payload_text"] = text
        return _sorted_dict(projected)


@dataclasses.dataclass(frozen=True, slots=True)
class ModuleCountEstimate:
    """What grid sampling measured while choosing the module count ``N``.

    Recorded so a sampling failure is diagnosable from evidence alone: the
    estimated module count, the module pitch it came from, the snap to the
    nearest legal ``4V + 17``, the finder pattern's dark-run width in pixels
    and the timing-pattern transition counts on both axes.

    ``accepted`` is true only when the snap was confirmed by the timing pattern
    -- an independent check that the pitch was right -- and ``reason`` states
    what refused it otherwise.
    """

    n_est: int | None
    pitch: float | None
    snapped_n: int | None
    finder_run_px: int | None
    timing_transitions: tuple[int, int] | None
    accepted: bool
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        """Project this estimate to a JSON-safe mapping with sorted keys."""
        return _sorted_dict(
            {
                "accepted": self.accepted,
                "finder_run_px": self.finder_run_px,
                "n_est": self.n_est,
                "pitch": None if self.pitch is None else float(self.pitch),
                "reason": self.reason,
                "snapped_n": self.snapped_n,
                "timing_transitions": None
                if self.timing_transitions is None
                else [int(self.timing_transitions[0]), int(self.timing_transitions[1])],
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SampledMatrix:
    """The sampled module grid, or the reason no grid could be recovered.

    On success ``matrix`` holds the grid, ``module_count`` is ``N``, ``version``
    is the version implied by ``N = 4V + 17`` and ``reason`` is ``None``. When
    neither the corner-quad path nor the rectified-image fallback could
    establish a pitch, ``matrix``, ``module_count`` and ``version`` are all
    ``None`` and ``reason`` is ``"sampling_failed"``, which lets a check report
    that reason instead of a wrong verdict.

    ``source`` names which input established the grid: ``"primary"`` for the
    primary decoder's corner quadrangle, ``"straight_qrcode"`` for the
    secondary decoder's rectified image. ``estimate`` carries the measurements
    behind that choice. ``rectified_luma`` is the rectified square's full-size
    ``float64`` luminance, kept only for a consumer that renders it: it is
    ``None`` unless the sampler was asked for it with
    ``sample_matrix(..., keep_rectified=True)``, because the array costs 8 bytes
    per pixel and lives as long as this result does. It is never serialised,
    because the evidence artefact set is closed.
    """

    matrix: Matrix | None
    version: int | None
    module_count: int | None
    reason: str | None
    estimate: ModuleCountEstimate
    source: str
    rectified_luma: numpy.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject an unknown sampling source.

        Raises:
            ValueError: ``source`` is neither ``primary`` nor
                ``straight_qrcode``.
        """
        if self.source not in _SAMPLING_SOURCES:
            raise ValueError(
                f"unknown sampling source {self.source!r}; expected one of "
                f"{sorted(_SAMPLING_SOURCES)}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Project this sampling result to a JSON-safe mapping with sorted keys.

        The numbers, the estimate and the reason reach the evidence; the
        rectified luminance array never does.
        """
        return _sorted_dict(
            {
                "estimate": self.estimate.as_dict(),
                "module_count": self.module_count,
                "reason": self.reason,
                "source": self.source,
                "version": self.version,
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DecodeResult:
    """The single decoded-evidence object every check consumes.

    ``payload_bytes`` is the authoritative payload -- the primary decoder's
    whenever it produced one -- and is ``None`` only when the primary decoder
    could not read the symbol. ``text`` is the rendering the decoder reported,
    while ``payload_bytes`` stays the byte-exact answer that round-trip
    assertions compare. ``symbology_identifier`` is ``]Q1`` without an ECI
    designator and ``]Q2`` with one.

    ``ec_level_decoder`` and ``ec_level_format`` are the two independent
    EC-level readings: the decoder's metadata and the symbol's own 15-bit
    format information. ``version`` and ``mask`` come from the sampled grid and
    the format information respectively, because no pinned decoder exposes
    either. ``matrix`` is the sampled grid and ``matrix_sha256`` its canonical
    digest.

    ``agreement`` classifies the two decoders' answers, ``polarity_applied``
    records the polarity at which the authoritative payload was read, and
    ``rejection`` carries verbatim the text of a declared client-side rejection
    -- the expected outcome of the invalid-input checks. ``metrics`` carries
    the performance measurements and is empty for every other check.

    ``sampling`` extends the documented fields so a check can tell "the grid
    could not be recovered" (``reason == "sampling_failed"``) from "the symbol
    decoded but the assertion failed". ``format_info`` extends them again with
    the format-information reading itself -- the projection
    ``qrcheck.decode.format_info.FormatInfo.as_dict()`` returns, both copies'
    distances included -- so a check can tell "the grid could not be recovered"
    from "the grid was recovered and its format information could not be read",
    and a damaged-but-readable copy stays diagnosable from evidence alone.

    ``secondary_comparable`` extends them once more, and exists so that a reader
    of the evidence can tell the two shapes of ``primary_only`` apart, because
    the classification alone cannot:

    * ``True`` -- both decoders answered and their payloads could be compared:
      they are equal, or the primary's payload round-trips through UTF-8, so the
      secondary's text could have contradicted it.
    * ``False`` -- both decoders answered with differing payloads that cannot be
      compared, which is the documented refinement: the primary's payload does
      not round-trip through UTF-8 and the text-only secondary decoder therefore
      cannot speak about it, so a disagreement is declined rather than reported.
    * ``None`` -- no comparison was attempted: the secondary decoder did not run,
      found nothing or errored, or the primary decoder produced no payload.

    The secondary decoder's text-only API (``opencv-python-headless`` 4.10) is
    why the ``False`` case exists at all; this field is the only place the
    narrowing of the disagreement rule is visible, so no consumer has to infer
    it from a decoder's status.

    ``unavailable_decoders`` extends them for the one case in which a decoder is
    not an answer at all: a decoder the run did not consult, mapped to the reason
    it could not be. It is empty whenever every decoder was consulted, which is
    the ordinary case, and it is why single-decoder mode's evidence lists one
    decoder in ``per_decoder``: a binding that cannot be
    imported raised nothing about the artefact, so it is recorded here instead of
    as an outcome with ``status == "error"``, which means a decoder that ran and
    raised. A decoder appears in ``per_decoder`` or in
    ``unavailable_decoders``, never in both, and the degradation therefore stays
    recorded in the evidence a reader has even though the noise of a decoder that
    never ran is gone.
    """

    payload_bytes: bytes | None
    text: str | None
    symbology_identifier: str | None
    content_type: str | None
    ec_level_decoder: str | None
    ec_level_format: str | None
    version: int | None
    mask: int | None
    matrix: Matrix | None
    matrix_sha256: str | None
    per_decoder: Mapping[str, DecoderOutcome]
    agreement: Agreement
    polarity_applied: Polarity
    rejection: str | None
    metrics: Mapping[str, float]
    sampling: SampledMatrix | None = None
    format_info: Mapping[str, Any] | None = None
    secondary_comparable: bool | None = None
    unavailable_decoders: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the mappings and make the digest unable to go stale.

        ``per_decoder``, ``metrics``, ``format_info`` and ``unavailable_decoders``
        become read-only mappings with sorted keys, so iteration order cannot
        depend on how a producer built them and a result cannot be mutated after
        construction. When a grid is present the digest is computed here if the
        caller passed ``None``.

        Raises:
            ValueError: ``agreement`` or ``polarity_applied`` is outside its
                closed vocabulary, ``secondary_comparable`` is neither a
                ``bool`` nor ``None``, a ``per_decoder`` or an
                ``unavailable_decoders`` key is not a known decoder, a decoder is
                recorded in both places -- it either answered or did not run --
                an unavailable decoder carries no reason, or ``matrix_sha256`` was
                supplied with a grid and disagrees with it.
        """
        if self.agreement not in _AGREEMENT_VALUES:
            raise ValueError(
                f"unknown agreement {self.agreement!r}; expected one of {sorted(_AGREEMENT_VALUES)}"
            )
        if self.polarity_applied not in _POLARITY_VALUES:
            raise ValueError(f"unknown polarity {self.polarity_applied!r}")
        comparable = self.secondary_comparable
        if comparable is not None and not isinstance(comparable, bool):
            raise ValueError(f"secondary_comparable must be a bool or None; got {comparable!r}")
        unknown = sorted(name for name in self.per_decoder if name not in DECODER_ORDER)
        if unknown:
            raise ValueError(f"unknown decoder name(s) in per_decoder: {unknown}")
        unconsulted = sorted(self.unavailable_decoders.items())
        unknown_unavailable = sorted(
            name for name, _ in unconsulted if name not in DECODER_ORDER
        )
        if unknown_unavailable:
            raise ValueError(
                f"unknown decoder name(s) in unavailable_decoders: {unknown_unavailable}"
            )
        unreasoned = sorted(
            name
            for name, reason in unconsulted
            if not isinstance(reason, str) or not reason.strip()
        )
        if unreasoned:
            raise ValueError(
                f"decoder(s) recorded as unavailable without a reason: {unreasoned}"
            )
        both = sorted(name for name, _ in unconsulted if name in self.per_decoder)
        if both:
            raise ValueError(
                f"decoder(s) recorded both as consulted and as unavailable: {both}"
            )
        object.__setattr__(
            self, "per_decoder", types.MappingProxyType(dict(sorted(self.per_decoder.items())))
        )
        object.__setattr__(
            self,
            "unavailable_decoders",
            types.MappingProxyType(dict(unconsulted)),
        )
        object.__setattr__(
            self, "metrics", types.MappingProxyType(dict(sorted(self.metrics.items())))
        )
        if self.format_info is not None:
            object.__setattr__(
                self,
                "format_info",
                types.MappingProxyType(dict(sorted(self.format_info.items()))),
            )
        if self.matrix is not None:
            digest = matrix_sha256(self.matrix)
            if self.matrix_sha256 is None:
                object.__setattr__(self, "matrix_sha256", digest)
            elif self.matrix_sha256 != digest:
                raise ValueError(
                    f"matrix_sha256 {self.matrix_sha256} does not match the sampled matrix "
                    f"({digest})"
                )

    @property
    def primary(self) -> DecoderOutcome | None:
        """The primary decoder's outcome, or ``None`` when it did not run."""
        return self.per_decoder.get(PRIMARY_DECODER)

    @property
    def secondary(self) -> DecoderOutcome | None:
        """The secondary decoder's outcome, or ``None`` in single-decoder mode."""
        return self.per_decoder.get(SECONDARY_DECODER)

    def as_dict(self) -> dict[str, Any]:
        """Project this result to the canonical JSON-safe evidence mapping.

        Every documented field reaches the projection; ``payload_bytes``
        becomes ``payload_b64`` and ``payload_len`` (plus ``payload_text`` when
        the bytes round-trip through UTF-8) so no raw ``bytes`` object survives,
        and ``matrix`` is represented by ``matrix_sha256`` so the grid itself
        never reaches the evidence. ``unavailable_decoders`` is always present
        and is ``{}`` when every decoder was consulted, so a consumer never has
        to tell an absent key from an empty one. No timestamp, hostname,
        absolute path or duration is emitted: a duration belongs to the result
        model, not to the decoded evidence.
        """
        projected: dict[str, Any] = {
            "agreement": self.agreement,
            "content_type": self.content_type,
            "ec_level_decoder": self.ec_level_decoder,
            "ec_level_format": self.ec_level_format,
            "format_info": None
            if self.format_info is None
            else _sorted_dict(dict(self.format_info)),
            "mask": self.mask,
            "matrix_sha256": self.matrix_sha256,
            "metrics": _sorted_dict(
                {name: float(value) for name, value in self.metrics.items()}
            ),
            "per_decoder": _sorted_dict(
                {name: outcome.as_dict() for name, outcome in self.per_decoder.items()}
            ),
            "polarity_applied": self.polarity_applied,
            "rejection": self.rejection,
            "sampling": None if self.sampling is None else self.sampling.as_dict(),
            "secondary_comparable": self.secondary_comparable,
            "symbology_identifier": self.symbology_identifier,
            "text": self.text,
            "unavailable_decoders": _sorted_dict(dict(self.unavailable_decoders)),
            "version": self.version,
        }
        if self.payload_bytes is not None:
            projected["payload_b64"] = base64.b64encode(self.payload_bytes).decode("ascii")
            projected["payload_len"] = len(self.payload_bytes)
            text = _payload_text(self.payload_bytes)
            if text is not None:
                projected["payload_text"] = text
        return _sorted_dict(projected)


@dataclasses.dataclass(frozen=True, slots=True)
class DecoderAvailability:
    """Which reference decoders are installed, and at which versions.

    The primary decoder's absence blocks a run: no check may be silently
    skipped in place of the authority for every round-trip assertion. The
    secondary decoder's absence is a supported degradation -- the run proceeds
    in single-decoder mode, ``agreement`` records ``primary_only`` and no check
    passes on the strength of the missing decoder.

    ``primary_version`` and ``secondary_version`` are the installed
    distribution versions, or ``None`` when the distribution is absent.
    ``qrcheck doctor --json`` consumes these four values through the two
    dependency rows it builds, and ``qrcheck run`` reads ``primary`` and
    ``secondary`` for its dependency probes; ``qrcheck list --json`` publishes
    the check registry and reads none of them. :meth:`as_dict` is the record's
    JSON-safe projection for a caller that needs the four values as one mapping.
    """

    primary: bool
    secondary: bool
    primary_version: str | None
    secondary_version: str | None

    def as_dict(self) -> dict[str, Any]:
        """Project this availability record to a JSON-safe mapping."""
        return _sorted_dict(
            {
                "primary": self.primary,
                "primary_version": self.primary_version,
                "secondary": self.secondary,
                "secondary_version": self.secondary_version,
            }
        )


def _module_present(module: str) -> bool:
    """Report whether ``module`` is importable without importing it.

    ``find_spec`` locates the module's loader and never executes the module, so
    a probe cannot pay the cost of loading a decoder or depend on its import
    side effects. A missing parent package raises ``ImportError`` and a module
    whose ``__spec__`` is unset or ``None`` raises ``ValueError``; both mean
    "not usable" and neither may escape a probe.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _distribution_version(distribution: str) -> str | None:
    """Return the installed version of ``distribution``, or ``None`` if absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def primary_available() -> bool:
    """Report whether the primary reference decoder (``zxing-cpp``) is installed.

    Never raises and never imports ``zxingcpp``.
    """
    return _module_present("zxingcpp")


def secondary_available() -> bool:
    """Report whether the secondary reference decoder (``opencv``) is installed.

    Never raises and never imports ``cv2``.
    """
    return _module_present("cv2")


def decoder_availability() -> DecoderAvailability:
    """Probe both reference decoders and their installed versions.

    Never raises: an absent distribution reports ``False`` with a ``None``
    version, which ``qrcheck run`` turns into an exit-2 prerequisite failure for
    the primary decoder or a single-decoder degradation for the secondary, and
    which ``qrcheck doctor`` reports as missing or degraded.
    """
    return DecoderAvailability(
        primary=primary_available(),
        secondary=secondary_available(),
        primary_version=_distribution_version("zxing-cpp"),
        secondary_version=_distribution_version("opencv-python-headless"),
    )
