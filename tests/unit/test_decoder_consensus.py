"""The consensus contract: two decoders, one authority, and silence that is not a defect.

``qrcheck.decode.consensus`` is the only place that decides what a symbol decoded to, so its rules
are the product's rather than its own (AAP §0.3.2, "consensus rules (these are contract, not
implementation detail)"). Each rule is pinned here with stubbed decoders -- no real decoder runs,
no network is touched, and the only image is a three-pixel PNG built here so the pipeline has
genuine bytes to open:

* ``disagree`` requires two payloads that differ, and both payloads and both metadata blocks stay
  in ``per_decoder`` so evidence can show them side by side;
* ``payload_bytes`` is the primary decoder's payload and a differing secondary payload never
  replaces it, so a symbol only the secondary could read is ``secondary_only`` with
  ``payload_bytes`` left ``None`` and ``RT-ROUNDTRIP`` still fails;
* silence is never a disagreement (IM-3): a decoder that returns no payload is ``undetected``, and
  each silence case asserts ``!= "disagree"`` explicitly rather than inferring it;
* the decoder's EC level and the symbol's own format information are never reconciled -- a
  disagreement between them is ``format_info_disagreement``, and collapsing one into the other here
  would delete that finding;
* inversion is the framework's job (IM-4): an inverted request is decoded twice with both decoders,
  and the call counters below are what make "no second pass for an ordinary request" testable,
  because an implementation that always inverted would satisfy every payload assertion while
  doubling the decode cost of a whole run;
* ``per_decoder`` order is lexicographic, in ``as_dict()`` too, so the evidence bytes never depend
  on the order the decoders happened to be called;
* a payload that does not round-trip through UTF-8 is ``primary_only`` with ``secondary_comparable``
  ``false``, which is what keeps the forty ``random-bytes`` records from failing the conforming
  reference gate, while a plain ASCII disagreement stays a disagreement with it ``true`` -- and a
  secondary decoder that said nothing records ``null``, so the evidence separates a secondary answer
  that could not be compared from one that never existed.

:func:`_install_decoders` monkeypatches the consensus module's ``read_primary`` and
``read_secondary`` with stubs that answer from a polarity-keyed table and build their outcomes
through the ``make_decoder_outcome`` fixture, so every field name comes from
:mod:`qrcheck.decode.base` rather than being re-typed here. The check-level cases drive the two
real checks, ``RT-ROUNDTRIP`` and ``SY-EC-MATCH``, through the shared ``run_context`` fixture with
only the decode entry stubbed. Real-symbol decoding belongs to
``tests/unit/test_matrix_sampling.py`` and ``tests/unit/test_format_info.py``; nothing here reads a
configuration value.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from collections.abc import Callable, Mapping
from typing import Any, Final, NamedTuple

import numpy
import pytest

from qrcheck.checks import roundtrip, symbol
from qrcheck.decode import consensus
from qrcheck.decode.base import (
    PRIMARY_DECODER,
    SECONDARY_DECODER,
    DecodeResult,
    DecoderOutcome,
    ModuleCountEstimate,
    SampledMatrix,
    matrix_sha256,
)
from qrcheck.evidence.mux import decode_image, encode_png
from qrcheck.report.model import CaseResult, Reason, Status
from qrcheck.targets.base import GenerateRequest, build_request

__all__ = [
    "test_agreeing_decoders_are_agreement_and_a_conforming_case_passes",
    "test_agreeing_ec_readings_produce_no_format_info_disagreement",
    "test_ascii_payload_disagreement_is_still_a_disagreement",
    "test_binary_payload_is_recorded_as_primary_only_not_a_disagreement",
    "test_consensus_records_both_ec_readings_without_reconciling_them",
    "test_contrast_ratio_matches_the_wcag_reference_values",
    "test_decoder_ordering_is_lexicographic_for_every_classification",
    "test_inverted_request_is_decoded_twice_by_both_decoders",
    "test_inverted_request_whose_original_copy_reads_reports_original_polarity",
    "test_inverting_twice_restores_the_image",
    "test_non_inverted_request_is_decoded_exactly_once_by_each_decoder",
    "test_payload_disagreement_fails_roundtrip_with_decoder_disagreement",
    "test_payload_disagreement_keeps_the_primary_payload_authoritative",
    "test_relative_luminance_matches_the_wcag_reference_values",
    "test_secondary_cannot_rescue_a_symbol_the_primary_could_not_read",
    "test_secondary_comparable_refuses_a_value_outside_its_three",
    "test_secondary_decoder_answering_at_both_polarities_records_both",
    "test_secondary_silence_is_primary_only_and_leaves_the_primary_authoritative",
    "test_secondary_disabled_degrades_the_run_to_primary_only",
    "test_silence_in_both_decoders_is_undetected_and_never_a_disagreement",
    "test_sym_ec_match_fails_with_format_info_disagreement_and_keeps_both_values",
]

# ---------------------------------------------------------------------------
# Fixed inputs
#
# A request parameter and not a configuration threshold: no value in this module is read from
# ``qrcheck.toml``, and none of the numeric budgets the plan declares -- the p95 ceilings, the
# throughput floor, the line limit or the damage budgets -- is restated here.
# ---------------------------------------------------------------------------

#: The error-correction level every request in this module asks for. ``M`` is the level
#: ``run_context`` defaults the checks to, so a check-level case and a consensus-level case
#: describe the same run.
_EC_LEVEL: Final[str] = "M"

#: The payload the primary decoder returns when a case needs a payload that is unambiguously
#: text: plain ASCII, so it round-trips through UTF-8 and the secondary decoder's rendering
#: can genuinely corroborate it.
_PAYLOAD_A: Final[bytes] = b"qrcheck-consensus-payload-A"

#: A second plain-ASCII payload, one byte removed from :data:`_PAYLOAD_A` at the tail so the
#: two are different payloads rather than different lengths of one.
_PAYLOAD_B: Final[bytes] = b"qrcheck-consensus-payload-B"

#: Bytes that are not valid UTF-8, standing for the ``random-bytes`` corpus class and for
#: every ``payload_b64`` record: no decoder can render them as text, which is exactly the
#: case the binary refinement exists for. ``0x81`` is an invalid UTF-8 start byte, so a strict
#: decode fails and a replacement decode cannot round-trip.
_BINARY_PAYLOAD: Final[bytes] = bytes((0x00, 0x81, 0xFE, 0xFF, 0x7F, 0xC3))

#: What a text-only decoder necessarily returns for :data:`_BINARY_PAYLOAD`: the UTF-8
#: encoding of the replacement-character rendering. Asserted to differ from the payload, so the
#: refinement case proves something rather than comparing a payload with itself.
_SECONDARY_TEXT_VIEW: Final[bytes] = _BINARY_PAYLOAD.decode("utf-8", errors="replace").encode(
    "utf-8"
)

#: The error text a stub records for a symbol it found nothing in, mirroring the two real
#: wrappers' "no symbol" messages closely enough that a reader of ``decoded.json`` sees the
#: same shape.
_NO_SYMBOL_TEXT: Final[str] = "no QR symbol found"

#: The colours of an inverted (light-on-dark) request: light modules on a dark background, so
#: ``is_inverted_request`` is true because the foreground is the lighter of the two.
_INVERTED_FOREGROUND: Final[tuple[int, int, int]] = (255, 255, 255)
_INVERTED_BACKGROUND: Final[tuple[int, int, int]] = (0, 0, 0)

#: The two polarities a decoder can be asked about, in the order the pipeline visits them.
_ORIGINAL: Final[str] = "original"
_INVERTED: Final[str] = "inverted"

#: The module grid the sampling stub hands back, and the format-information reading that goes
#: with it. A Version-2 grid is ``4 * 2 + 17 = 25`` modules on a side; the grid is synthetic --
#: only its dimension, its digest and the fact that a format information reading was possible
#: matter to the case that uses it.
_GRID_VERSION: Final[int] = 2
_GRID_MODULE_COUNT: Final[int] = 4 * _GRID_VERSION + 17
_GRID_MASK: Final[int] = 3
_GRID_EC_LEVEL: Final[str] = "Q"


def _png_bytes(side: int = 3) -> bytes:
    """Build a valid, deterministic grayscale PNG of `side` by `side` white pixels. Written with
    :mod:`struct` and :mod:`zlib` rather than with Pillow, because the whole point of the image
    here is to prove the framework's own byte path works: the bytes are a real PNG
    (``IHDR``/``IDAT``/``IEND`` with correct CRCs) that :func:`decode_image` can open and
    :func:`encode_png` can re-encode, and no third-party image API is imported directly.
    """

    def chunk(kind: bytes, data: bytes) -> bytes:
        """Return one PNG chunk: length, type, payload and the CRC of type plus payload."""
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    # Eight-bit greyscale, no interlace: filter type 0 on every row, every pixel fully white.
    header = struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes([0xFF]) * side for _ in range(side))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _symbol_png() -> bytes:
    """Return the module's one artefact: a tiny valid PNG, canonicalised through the mux."""
    return encode_png(decode_image(_png_bytes()))


#: The one PNG every consensus-level case decodes. Built once per module: encoding it per test
#: would add nothing and the stubs ignore its contents anyway.
_SYMBOL_PNG: Final[bytes] = _symbol_png()


def _request(payload: bytes, **overrides: Any) -> GenerateRequest:
    """Return the legal request the pipeline is decoded for. Built through
    :func:`qrcheck.targets.base.build_request` -- the product's single request constructor --
    so the colours the polarity decision reads are the validated ones, and so a case cannot
    describe a request the framework would refuse to send.
    """
    return build_request(payload, _EC_LEVEL, **overrides)


def _inverted_request(payload: bytes) -> GenerateRequest:
    """Return the light-on-dark request :func:`consensus.is_inverted_request` reports true for."""
    return _request(
        payload,
        foreground_rgb=_INVERTED_FOREGROUND,
        background_rgb=_INVERTED_BACKGROUND,
    )


class _DecoderStub:
    """One decoder entry replaced by a recording stub with a polarity-keyed answer table. An
    absent key -- and a key mapped to ``None`` -- means it found no symbol at that polarity,
    which is the ``undetected`` status and never a disagreement.
    """

    def __init__(
        self,
        decoder: str,
        answers: Mapping[str, bytes | None],
        make_outcome: Callable[..., DecoderOutcome],
        reading_factory: Callable[[DecoderOutcome, bytes | None], object],
    ) -> None:
        """Configure the stub's identity, its answers and how it wraps an outcome as a reading."""
        self.decoder: str = decoder
        self.calls: list[tuple[str, numpy.ndarray]] = []
        self._answers: dict[str, bytes | None] = dict(answers)
        self._make_outcome = make_outcome
        self._reading_factory = reading_factory
        self._reports_ec: bool = decoder == PRIMARY_DECODER

    @property
    def polarities(self) -> tuple[str, ...]:
        """Return the polarity of every call, in call order."""
        return tuple(polarity for polarity, _ in self.calls)

    @property
    def pass_count(self) -> int:
        """Return how many times the decoder was called."""
        return len(self.calls)

    def __call__(self, image: numpy.ndarray, *, polarity: str = _ORIGINAL) -> object:
        """Record one call and answer it from the table."""
        self.calls.append((polarity, numpy.array(image, copy=True)))
        payload = self._answers.get(polarity)
        decoded = payload is not None
        outcome = self._make_outcome(
            decoder=self.decoder,
            status="decoded" if decoded else "undetected",
            polarity=polarity,
            payload=payload,
            reported_ec=_EC_LEVEL if decoded and self._reports_ec else None,
            error_text=None if decoded else _NO_SYMBOL_TEXT,
        )
        return self._reading_factory(outcome, payload)


def _primary_reading(outcome: DecoderOutcome, payload: bytes | None) -> object:
    """Wrap the primary decoder's outcome in the reading record the pipeline expects. The rendered
    text is the payload decoded with replacement characters, which is what a decoder that
    reports text produces for bytes it cannot render -- so the pipeline's "does this payload
    round-trip through UTF-8?" question is answered by the payload itself rather than by a flag
    this stub invented.
    """
    decoded = payload is not None
    return consensus.zxing.PrimaryReading(
        outcome=outcome,
        text=None if payload is None else payload.decode("utf-8", errors="replace"),
        symbology_identifier="]Q1" if decoded else None,
        content_type="Text" if decoded else None,
    )


def _secondary_reading(outcome: DecoderOutcome, payload: bytes | None) -> object:
    """Wrap the secondary decoder's outcome in its reading record, with no rectified grid.
    ``straight_qrcode`` stays ``None``: the rectified image is the module-count fallback, which
    is grid sampling's business (``tests/unit/test_matrix_sampling.py``), and handing one over
    here would let a case accidentally assert against a sampled grid it never intended.
    """
    del payload  # The secondary decoder carries no text: it answers in bytes only.
    return consensus.opencv.SecondaryReading(outcome=outcome, straight_qrcode=None)


def _install_decoders(
    monkeypatch: pytest.MonkeyPatch,
    make_outcome: Callable[..., DecoderOutcome],
    *,
    primary: Mapping[str, bytes | None],
    secondary: Mapping[str, bytes | None] | None = None,
) -> tuple[_DecoderStub, _DecoderStub | None]:
    """Replace both decode entries in the consensus module's namespace with recording stubs. The
    patch target is the name the pipeline itself resolves -- ``consensus.zxing.read_primary``
    and ``consensus.opencv.read_secondary`` -- so the stub stands exactly where the real call
    would, and ``monkeypatch`` restores the real entry when the test ends.
    """
    primary_stub = _DecoderStub(
        PRIMARY_DECODER, primary, make_outcome, _primary_reading
    )
    monkeypatch.setattr(consensus.zxing, "read_primary", primary_stub, raising=True)
    secondary_stub: _DecoderStub | None = None
    if secondary is not None:
        secondary_stub = _DecoderStub(
            SECONDARY_DECODER, secondary, make_outcome, _secondary_reading
        )
        monkeypatch.setattr(consensus.opencv, "read_secondary", secondary_stub, raising=True)
    return primary_stub, secondary_stub


def _execute_first_case(contract: Any, ctx: Any) -> CaseResult:
    """Execute the first case a registered check yields for `ctx` and return its result. One case,
    not the whole corpus: the corpus-driven checks yield 240 cases each and only the verdict of
    one is under test.
    """
    case = next(iter(contract.cases(ctx)))
    return case.execute(ctx)


class _GridCase(NamedTuple):
    """The two inputs the EC-pair case stubs: the sampled grid and the format reading."""

    grid: numpy.ndarray
    format_reading: Any


def _grid_fixture() -> _GridCase:
    """Return a synthetic sampled grid and the format-information reading that goes with it."""
    grid = numpy.zeros((_GRID_MODULE_COUNT, _GRID_MODULE_COUNT), dtype=numpy.bool_)
    grid[0, :] = True
    grid[:, 0] = True
    reading = consensus.format_info.FormatInfo(
        ec_level=_GRID_EC_LEVEL, mask=_GRID_MASK, bit_errors=0
    )
    return _GridCase(grid=grid, format_reading=reading)


_GRID: Final[_GridCase] = _grid_fixture()

# ---------------------------------------------------------------------------
# Payload disagreement: two payloads that differ, one authority unchanged
# ---------------------------------------------------------------------------


def test_payload_disagreement_keeps_the_primary_payload_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """Two differing payloads are ``disagree``, and the primary's bytes stay authoritative. The
    evidence contract is asserted as well as the classification: both payload digests and both
    metadata blocks must be in ``per_decoder``, because a reader of ``decoded.json`` has to be
    able to see the two answers side by side to judge the finding.
    """
    primary, secondary = _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A},
        secondary={_ORIGINAL: _PAYLOAD_B},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.agreement == "disagree", (
        "payload disagreement: two differing payloads must classify as 'disagree', got "
        f"{result.agreement!r}"
    )
    assert result.payload_bytes == _PAYLOAD_A, (
        "payload disagreement: the primary decoder's payload must stay authoritative, got "
        f"{result.payload_bytes!r}"
    )
    assert primary.pass_count == 1 and secondary.pass_count == 1, (
        "payload disagreement: a non-inverted request must decode once per decoder, got "
        f"{primary.pass_count} primary and {secondary.pass_count} secondary passes"
    )

    projection = result.as_dict()
    blocks = projection["per_decoder"]
    assert blocks[PRIMARY_DECODER]["payload_sha256"] == hashlib.sha256(_PAYLOAD_A).hexdigest(), (
        "payload disagreement: the primary payload's digest must be recorded in per_decoder"
    )
    assert blocks[SECONDARY_DECODER]["payload_sha256"] == hashlib.sha256(_PAYLOAD_B).hexdigest(), (
        "payload disagreement: the secondary payload's digest must be recorded in per_decoder"
    )
    assert blocks[PRIMARY_DECODER]["payload_b64"] != blocks[SECONDARY_DECODER]["payload_b64"], (
        "payload disagreement: both payloads must be present verbatim so evidence can show them"
    )
    assert blocks[PRIMARY_DECODER]["reported_ec"] == _EC_LEVEL, (
        "payload disagreement: the primary decoder's EC metadata must be recorded"
    )
    assert blocks[SECONDARY_DECODER]["reported_ec"] is None, (
        "payload disagreement: the secondary decoder reports no EC metadata, which is evidence "
        "and not a gap"
    )
    assert blocks[SECONDARY_DECODER]["status"] == "decoded", (
        "payload disagreement: the secondary decoder did decode, it just disagreed"
    )


def test_payload_disagreement_fails_roundtrip_with_decoder_disagreement(
    patch_decode: Any,
    make_decode_result: Callable[..., DecodeResult],
    make_decoder_outcome: Callable[..., DecoderOutcome],
    run_context: Any,
) -> None:
    """``RT-ROUNDTRIP`` reports ``decoder_disagreement`` before it compares any payload. The
    primary's payload here *equals* the requested payload, so the case can only fail for the
    disagreement itself -- which is what makes this the assertion that the disagreement
    outranks a matching payload rather than being folded into a value mismatch.
    """
    requested = run_context.corpus[0].payload_bytes
    patch_decode.result = make_decode_result(
        payload_bytes=requested,
        per_decoder={
            PRIMARY_DECODER: make_decoder_outcome(decoder=PRIMARY_DECODER, payload=requested),
            SECONDARY_DECODER: make_decoder_outcome(
                decoder=SECONDARY_DECODER, payload=_PAYLOAD_B
            ),
        },
        agreement="disagree",
    )
    result = _execute_first_case(roundtrip.cases, run_context)
    assert result.status is Status.FAILED, (
        f"payload disagreement: RT-ROUNDTRIP must fail, got {result.status!r}"
    )
    assert result.reason is Reason.DECODER_DISAGREEMENT, (
        "payload disagreement: the reason must be decoder_disagreement even though the primary "
        f"payload matched the request, got {result.reason!r}"
    )
    blocks = result.decoded["per_decoder"]
    assert blocks[PRIMARY_DECODER]["payload_b64"] != blocks[SECONDARY_DECODER]["payload_b64"], (
        "payload disagreement: the failed case's evidence must carry both payloads"
    )


# ---------------------------------------------------------------------------
# The two EC-level readings are independent, and never reconciled here
# ---------------------------------------------------------------------------


def _sampled_grid(*_args: object, **_kwargs: object) -> SampledMatrix:
    """Stand in for ``matrix.sample_matrix``: report the fixed synthetic grid as sampled."""
    return SampledMatrix(
        matrix=_GRID.grid,
        version=_GRID_VERSION,
        module_count=_GRID_MODULE_COUNT,
        reason=None,
        estimate=ModuleCountEstimate(
            n_est=_GRID_MODULE_COUNT,
            pitch=4.0,
            snapped_n=_GRID_MODULE_COUNT,
            finder_run_px=28,
            timing_transitions=(9, 9),
            accepted=True,
            reason=None,
        ),
        source="primary",
    )


def _format_reading(_grid: object) -> Any:
    """Stand in for ``format_info.decode_format_info``: report the fixed format information."""
    return _GRID.format_reading


def test_consensus_records_both_ec_readings_without_reconciling_them(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """The decoder's EC level and the format information's are both kept, side by side. A layer
    that reconciled them -- adopting one reading as "the" level -- would delete the
    ``format_info_disagreement`` finding before any check could report it, so the two values
    are asserted to survive into the result and into its evidence projection unchanged.
    """
    _install_decoders(monkeypatch, make_decoder_outcome, primary={_ORIGINAL: _PAYLOAD_A})
    monkeypatch.setattr(consensus.matrix, "sample_matrix", _sampled_grid, raising=True)
    monkeypatch.setattr(
        consensus.format_info, "decode_format_info", _format_reading, raising=True
    )

    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=False)
    assert result.ec_level_decoder == _EC_LEVEL, (
        "format-information disagreement: the decoder's own reading must be recorded verbatim, "
        f"got {result.ec_level_decoder!r}"
    )
    assert result.ec_level_format == _GRID_EC_LEVEL, (
        "format-information disagreement: the format information's reading must be recorded "
        f"verbatim, got {result.ec_level_format!r}"
    )
    assert result.ec_level_decoder != result.ec_level_format, (
        "format-information disagreement: a reconciling layer would have collapsed these two "
        "into one value"
    )
    assert result.version == _GRID_VERSION, (
        "format-information disagreement: the version must come from the grid, got "
        f"{result.version!r}"
    )
    assert result.mask == _GRID_MASK, (
        f"format-information disagreement: the mask must come from the format information, got "
        f"{result.mask!r}"
    )

    projection = result.as_dict()
    assert projection["ec_level_decoder"] == _EC_LEVEL, (
        "format-information disagreement: ec_level_decoder must reach the evidence projection"
    )
    assert projection["ec_level_format"] == _GRID_EC_LEVEL, (
        "format-information disagreement: ec_level_format must reach the evidence projection"
    )
    assert projection["matrix_sha256"] == matrix_sha256(_GRID.grid), (
        "format-information disagreement: the evidence must carry the sampled grid's digest"
    )
    assert projection["format_info"] is not None, (
        "format-information disagreement: the reading itself must be recorded, not just its "
        "two fields"
    )


def test_sym_ec_match_fails_with_format_info_disagreement_and_keeps_both_values(
    patch_decode: Any,
    make_decode_result: Callable[..., DecodeResult],
    run_context: Any,
) -> None:
    """``SY-EC-MATCH`` reports the disagreement between its two sources, with both values. The two
    readings disagree with *each other* here while both are present, which is the one cause
    ``format_info_disagreement`` names: the check must not report it as a mismatch against the
    request, and the evidence must carry the pair.
    """
    patch_decode.result = make_decode_result(ec_level_decoder="M", ec_level_format="Q")
    result = _execute_first_case(symbol.SY_EC_MATCH, run_context)
    assert result.status is Status.FAILED, (
        f"format-information disagreement: SY-EC-MATCH must fail, got {result.status!r}"
    )
    assert result.reason is Reason.FORMAT_INFO_DISAGREEMENT, (
        "format-information disagreement: the reason must be format_info_disagreement, got "
        f"{result.reason!r}"
    )
    assert result.decoded["ec_level_decoder"] == "M", (
        "format-information disagreement: the decoder's reading must be in the evidence"
    )
    assert result.decoded["ec_level_format"] == "Q", (
        "format-information disagreement: the format information's reading must be in the evidence"
    )


def test_agreeing_ec_readings_produce_no_format_info_disagreement(
    patch_decode: Any,
    make_decode_result: Callable[..., DecodeResult],
    run_context: Any,
) -> None:
    """Two readings that agree, and match the request, produce no reason at all. The contrast case
    for the test above: without it, a check that reported ``format_info_disagreement``
    unconditionally would pass the failing case.
    """
    patch_decode.result = make_decode_result(ec_level_decoder="M", ec_level_format="M")
    result = _execute_first_case(symbol.SY_EC_MATCH, run_context)
    assert result.status is Status.PASSED, (
        f"agreeing EC readings: SY-EC-MATCH must pass, got {result.status!r} ({result.reason!r})"
    )
    assert result.reason is None, (
        f"agreeing EC readings: a passing case carries no reason, got {result.reason!r}"
    )
    assert result.decoded["ec_level_decoder"] == result.decoded["ec_level_format"] == "M", (
        "agreeing EC readings: both readings must still be recorded in the evidence"
    )


# ---------------------------------------------------------------------------
# Silence is not disagreement, and the secondary decoder never rescues a symbol
# ---------------------------------------------------------------------------


def test_silence_in_both_decoders_is_undetected_and_never_a_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """Neither decoder returning a payload is ``undetected``, explicitly not ``disagree``."""
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: None},
        secondary={_ORIGINAL: None},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.agreement == "undetected", (
        f"silence in both decoders: agreement must be 'undetected', got {result.agreement!r}"
    )
    assert result.agreement != "disagree", (
        "silence in both decoders: two decoders finding nothing is not a disagreement (IM-3)"
    )
    assert result.payload_bytes is None, (
        f"silence in both decoders: no payload may be reported, got {result.payload_bytes!r}"
    )
    for name, block in result.as_dict()["per_decoder"].items():
        assert block["status"] == "undetected", (
            f"silence in both decoders: {name} must record 'undetected', got {block['status']!r}"
        )
        assert block["error_text"], (
            f"silence in both decoders: {name} must keep the decoder's own message"
        )


def test_secondary_silence_is_primary_only_and_leaves_the_primary_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """A secondary decoder that finds nothing is ``primary_only``, not a disagreement. This is the
    state a conforming symbol produces under damage, and for a one-module quiet zone the
    secondary decoder cannot detect at all -- the ``primary_only`` case consensus exists to
    record rather than to judge.
    """
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A},
        secondary={_ORIGINAL: None},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.agreement == "primary_only", (
        f"secondary silence: agreement must be 'primary_only', got {result.agreement!r}"
    )
    assert result.agreement != "disagree", (
        "secondary silence: one decoder's silence is never a disagreement (IM-3)"
    )
    assert result.secondary_comparable is None, (
        "secondary silence: no comparison was attempted, so secondary_comparable must be None, "
        f"got {result.secondary_comparable!r}"
    )
    assert result.as_dict()["secondary_comparable"] is None, (
        "secondary silence: the evidence projection must record that nothing was compared, got "
        f"{result.as_dict()['secondary_comparable']!r}"
    )
    assert result.payload_bytes == _PAYLOAD_A, (
        f"secondary silence: the primary payload stays authoritative, got {result.payload_bytes!r}"
    )
    assert result.text == _PAYLOAD_A.decode("utf-8"), (
        "secondary silence: the primary decoder's rendering must be recorded"
    )


def test_secondary_cannot_rescue_a_symbol_the_primary_could_not_read(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
    patch_decode: Any,
    make_decode_result: Callable[..., DecodeResult],
    run_context: Any,
) -> None:
    """``secondary_only`` leaves the payload empty, and the check fails anyway. The secondary
    decoder returns exactly the requested bytes, so nothing but the authority rule can make
    this case fail -- which is precisely the property being pinned.
    """
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: None},
        secondary={_ORIGINAL: _PAYLOAD_A},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.agreement == "secondary_only", (
        f"secondary rescue: agreement must be 'secondary_only', got {result.agreement!r}"
    )
    assert result.payload_bytes is None, (
        "secondary rescue: the authoritative payload must stay None when the primary decoder "
        f"read nothing, got {result.payload_bytes!r}"
    )
    assert result.as_dict()["per_decoder"][SECONDARY_DECODER]["payload_b64"] == (
        base64.b64encode(_PAYLOAD_A).decode("ascii")
    ), "secondary rescue: the secondary payload must be recorded for corroboration only"

    requested = run_context.corpus[0].payload_bytes
    patch_decode.result = make_decode_result(
        payload_bytes=None,
        per_decoder={
            PRIMARY_DECODER: make_decoder_outcome(
                decoder=PRIMARY_DECODER, status="undetected", payload=None
            ),
            SECONDARY_DECODER: make_decoder_outcome(
                decoder=SECONDARY_DECODER, payload=requested
            ),
        },
        agreement="secondary_only",
    )
    case_result = _execute_first_case(roundtrip.cases, run_context)
    assert case_result.status is Status.FAILED, (
        "secondary rescue: RT-ROUNDTRIP must fail even though the secondary decoded the payload, "
        f"got {case_result.status!r}"
    )
    assert case_result.reason is Reason.VALUE_MISMATCH, (
        "secondary rescue: an absent authoritative payload is a value mismatch, never a decoder "
        f"disagreement, got {case_result.reason!r}"
    )
    assert case_result.decoded["per_decoder"][SECONDARY_DECODER]["payload_b64"] == (
        base64.b64encode(requested).decode("ascii")
    ), "secondary rescue: the check's evidence must keep the secondary's corroborating payload"


def test_secondary_disabled_degrades_the_run_to_primary_only(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """Single-decoder mode records the primary alone and reports ``primary_only``. The degradation
    a report shows a banner for (AAP §0.6.6 row 4): the secondary decoder is absent, so it is
    not called at all and its silence is never evidence about the symbol.
    """
    primary, secondary = _install_decoders(
        monkeypatch, make_decoder_outcome, primary={_ORIGINAL: _PAYLOAD_A}
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=False)
    assert secondary is None, "single-decoder mode: no secondary stub should be installed"
    assert tuple(result.per_decoder) == (PRIMARY_DECODER,), (
        "single-decoder mode: only the primary decoder may appear, got "
        f"{tuple(result.per_decoder)!r}"
    )
    assert result.agreement == "primary_only", (
        f"single-decoder mode: agreement must be 'primary_only', got {result.agreement!r}"
    )
    assert result.secondary_comparable is None, (
        "single-decoder mode: the secondary decoder never ran, so no comparison was attempted and "
        f"secondary_comparable must be None, got {result.secondary_comparable!r}"
    )
    assert result.payload_bytes == _PAYLOAD_A, (
        f"single-decoder mode: the primary payload is still authoritative, got "
        f"{result.payload_bytes!r}"
    )
    assert primary.pass_count == 1, (
        f"single-decoder mode: the primary decoder must still be called once, got "
        f"{primary.pass_count}"
    )


# ---------------------------------------------------------------------------
# Agreement, ordering and the documented binary refinement
# ---------------------------------------------------------------------------


def test_agreeing_decoders_are_agreement_and_a_conforming_case_passes(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
    patch_decode: Any,
    make_decode_result: Callable[..., DecodeResult],
    run_context: Any,
) -> None:
    """Two identical payloads are ``agree``, and the round-trip case passes with no reason."""
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A},
        secondary={_ORIGINAL: _PAYLOAD_A},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.agreement == "agree", (
        f"agreeing decoders: agreement must be 'agree', got {result.agreement!r}"
    )
    assert result.secondary_comparable is True, (
        "agreeing decoders: two equal payloads were compared, so secondary_comparable must be "
        f"True, got {result.secondary_comparable!r}"
    )
    assert result.payload_bytes == _PAYLOAD_A, (
        f"agreeing decoders: the shared payload must be authoritative, got {result.payload_bytes!r}"
    )

    requested = run_context.corpus[0].payload_bytes
    patch_decode.result = make_decode_result(payload_bytes=requested)
    case = next(iter(roundtrip.cases.cases(run_context)))
    assert case.case_id == f"corpus:{run_context.corpus[0].id}", (
        f"agreeing decoders: the driven case must be the first corpus record, got {case.case_id!r}"
    )
    case_result = case.execute(run_context)
    assert case_result.status is Status.PASSED, (
        f"agreeing decoders: RT-ROUNDTRIP must pass on an agreeing decode, got "
        f"{case_result.status!r} ({case_result.reason!r})"
    )
    assert case_result.reason is None, (
        f"agreeing decoders: a conforming case carries no reason, got {case_result.reason!r}"
    )


@pytest.mark.parametrize(
    ("primary_answers", "secondary_answers", "expected"),
    [
        pytest.param({_ORIGINAL: _PAYLOAD_A}, {_ORIGINAL: _PAYLOAD_A}, "agree", id="agree"),
        pytest.param(
            {_ORIGINAL: _PAYLOAD_A}, {_ORIGINAL: None}, "primary_only", id="primary-only"
        ),
        pytest.param(
            {_ORIGINAL: None}, {_ORIGINAL: _PAYLOAD_A}, "secondary_only", id="secondary-only"
        ),
        pytest.param({_ORIGINAL: None}, {_ORIGINAL: None}, "undetected", id="undetected"),
        pytest.param({_ORIGINAL: _PAYLOAD_A}, {_ORIGINAL: _PAYLOAD_B}, "disagree", id="disagree"),
        pytest.param(
            {_ORIGINAL: _BINARY_PAYLOAD},
            {_ORIGINAL: _SECONDARY_TEXT_VIEW},
            "primary_only",
            id="primary-only-incomparable",
        ),
    ],
)
def test_decoder_ordering_is_lexicographic_for_every_classification(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
    primary_answers: Mapping[str, bytes | None],
    secondary_answers: Mapping[str, bytes | None],
    expected: str,
) -> None:
    """Every classification keeps the lexicographic decoder order, in the record and in evidence.
    Sorted keys are what make two runs' evidence byte-identical: an insertion-ordered map would
    put the decoders in whatever order the pipeline happened to call them. The
    ``primary-only-incomparable`` row is the documented refinement's shape -- two differing
    payloads the text-only secondary decoder cannot speak about -- which must keep the same order
    as every other classification.
    """
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary=primary_answers,
        secondary=secondary_answers,
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.agreement == expected, (
        f"decoder ordering ({expected}): the classification of this answer pair must be "
        f"{expected!r}, got {result.agreement!r}"
    )
    assert tuple(result.per_decoder) == (SECONDARY_DECODER, PRIMARY_DECODER), (
        f"decoder ordering ({expected}): per_decoder must iterate in lexicographic order "
        f"('opencv', 'zxing'), got {tuple(result.per_decoder)!r}"
    )
    assert tuple(result.as_dict()["per_decoder"]) == (SECONDARY_DECODER, PRIMARY_DECODER), (
        f"decoder ordering ({expected}): the evidence projection must keep the same order, got "
        f"{tuple(result.as_dict()['per_decoder'])!r}"
    )


def test_binary_payload_is_recorded_as_primary_only_not_a_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """A payload the secondary decoder cannot speak about is ``primary_only``, as documented. The
    secondary decoder answers in text, so for bytes that are not losslessly text its re-encoded
    view differs from a binary primary payload even for a conforming symbol.
    """
    assert _SECONDARY_TEXT_VIEW != _BINARY_PAYLOAD, (
        "binary refinement: the stub's text view must differ from the payload, or this case "
        "would prove nothing"
    )
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _BINARY_PAYLOAD},
        secondary={_ORIGINAL: _SECONDARY_TEXT_VIEW},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_BINARY_PAYLOAD), secondary=True)
    assert result.agreement == "primary_only", (
        "binary refinement: a text-only contradictory view of a binary payload must classify as "
        f"'primary_only', got {result.agreement!r}"
    )
    assert result.secondary_comparable is False, (
        "binary refinement: both decoders answered with differing payloads the text-only secondary "
        f"cannot be held to, so secondary_comparable must be False, got "
        f"{result.secondary_comparable!r}"
    )
    assert result.as_dict()["secondary_comparable"] is False, (
        "binary refinement: the recorded evidence must carry secondary_comparable False rather "
        f"than look like a secondary decoder that said nothing, got "
        f"{result.as_dict()['secondary_comparable']!r}"
    )
    assert result.payload_bytes == _BINARY_PAYLOAD, (
        f"binary refinement: the primary payload stays authoritative, got {result.payload_bytes!r}"
    )
    rendering = result.text
    assert rendering is not None and rendering.encode("utf-8") != _BINARY_PAYLOAD, (
        "binary refinement: the decoder's rendering must not round-trip back to the payload, "
        f"which is the predicate the refinement turns on, got {rendering!r}"
    )
    assert "payload_text" not in result.as_dict(), (
        "binary refinement: a payload with no faithful rendering must not be projected as text"
    )
    assert result.as_dict()["per_decoder"][SECONDARY_DECODER]["payload_b64"] == (
        base64.b64encode(_SECONDARY_TEXT_VIEW).decode("ascii")
    ), "binary refinement: the secondary decoder's own bytes must still be recorded"


def test_ascii_payload_disagreement_is_still_a_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """The refinement must not widen: two differing text payloads are still a disagreement."""
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A},
        secondary={_ORIGINAL: _PAYLOAD_B},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _request(_PAYLOAD_A), secondary=True)
    assert result.payload_bytes is not None and result.text is not None, (
        "ascii disagreement: this payload must round-trip through UTF-8, or it is the wrong "
        "payload for this case"
    )
    assert result.agreement == "disagree", (
        "ascii disagreement: two differing ASCII payloads must still classify as 'disagree', got "
        f"{result.agreement!r}"
    )
    assert result.secondary_comparable is True, (
        "ascii disagreement: two text payloads could be compared, so secondary_comparable must be "
        f"True, got {result.secondary_comparable!r}"
    )


@pytest.mark.parametrize("value", ["decoded", 1], ids=["status-string", "integer"])
def test_secondary_comparable_refuses_a_value_outside_its_three(
    make_decode_result: Callable[..., DecodeResult], value: Any
) -> None:
    """The recorded field has exactly three values, and anything else is a producer defect. A
    decoder's status or an integer would reach ``decoded.json`` as a value no reader can
    interpret, so the record refuses it at construction rather than recording it.
    """
    with pytest.raises(ValueError, match="secondary_comparable"):
        make_decode_result(secondary_comparable=value)


# ---------------------------------------------------------------------------
# Polarity: the framework supplies the second pass, and only when it is needed
# ---------------------------------------------------------------------------


def test_inverted_request_is_decoded_twice_by_both_decoders(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """A light-on-dark request is decoded twice by each decoder, and the normalised copy wins.
    Neither pinned decoder can be relied on to read an inverted palette -- ``zxing-cpp`` 2.2.0
    has no inversion switch and OpenCV 4.10 cannot detect such a symbol at all -- so the
    framework builds the 255-inverted copy itself, hands it to both decoders, and records the
    polarity the authoritative payload came from.
    """
    primary, secondary = _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: None, _INVERTED: _PAYLOAD_A},
        secondary={_ORIGINAL: None, _INVERTED: _PAYLOAD_A},
    )
    request = _inverted_request(_PAYLOAD_A)
    assert consensus.is_inverted_request(request) is True, (
        "polarity: the request's foreground must be lighter than its background for this case"
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, request, secondary=True)
    assert primary.polarities == (_ORIGINAL, _INVERTED), (
        "polarity: the primary decoder must be called on the original artefact and then on the "
        f"inverted copy, got {primary.polarities!r}"
    )
    assert secondary is not None and secondary.polarities == (_ORIGINAL, _INVERTED), (
        "polarity: the secondary decoder must receive both copies too, got "
        f"{None if secondary is None else secondary.polarities!r}"
    )
    assert numpy.array_equal(
        primary.calls[1][1], consensus.invert_image(primary.calls[0][1])
    ), "polarity: the second pass must be the 255-inverted copy of the first, pixel for pixel"
    assert numpy.array_equal(secondary.calls[1][1], primary.calls[1][1]), (
        "polarity: both decoders must read the same normalised copy, not two separate inversions"
    )
    assert result.polarity_applied == _INVERTED, (
        f"polarity: polarity_applied must record 'inverted', got {result.polarity_applied!r}"
    )
    assert result.per_decoder[PRIMARY_DECODER].polarity == _INVERTED, (
        "polarity: each decoder's outcome must record the polarity that produced its answer, got "
        f"{result.per_decoder[PRIMARY_DECODER].polarity!r} for the primary decoder"
    )
    assert result.payload_bytes == _PAYLOAD_A, (
        "polarity: the normalised copy's payload must be authoritative, got "
        f"{result.payload_bytes!r}"
    )
    assert result.agreement == "agree", (
        f"polarity: both decoders read the same payload, so agreement must be 'agree', got "
        f"{result.agreement!r}"
    )


def test_inverted_request_whose_original_copy_reads_reports_original_polarity(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """When the original copy reads, ``polarity_applied`` is ``original`` and the outcome
    ``both``. A decoder that reads either palette answers at both polarities; the recorded
    outcome merges the two into ``both``, and the first polarity at which the primary produced
    a payload stays authoritative, so nothing claims the inversion was needed when it was not.
    """
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A, _INVERTED: _PAYLOAD_A},
        secondary={_ORIGINAL: _PAYLOAD_A, _INVERTED: _PAYLOAD_A},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _inverted_request(_PAYLOAD_A), secondary=True)
    assert result.polarity_applied == _ORIGINAL, (
        "polarity: the original copy read, so polarity_applied must be 'original', got "
        f"{result.polarity_applied!r}"
    )
    assert result.per_decoder[PRIMARY_DECODER].polarity == "both", (
        "polarity: a decoder that answered at each polarity must record 'both', got "
        f"{result.per_decoder[PRIMARY_DECODER].polarity!r}"
    )
    assert result.per_decoder[SECONDARY_DECODER].polarity == "both", (
        "polarity: the secondary decoder answered at each polarity too, got "
        f"{result.per_decoder[SECONDARY_DECODER].polarity!r}"
    )
    assert result.payload_bytes == _PAYLOAD_A, (
        f"polarity: the payload must be the agreed one, got {result.payload_bytes!r}"
    )


def test_secondary_decoder_answering_at_both_polarities_records_both(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """Each decoder's recorded polarity is its own, not the pipeline's global answer. The primary
    decoder reads only the original copy while the secondary reads both: the applied polarity
    is ``original``, the primary records ``original`` and the secondary records ``both``.
    """
    _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A},
        secondary={_ORIGINAL: _PAYLOAD_A, _INVERTED: _PAYLOAD_A},
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, _inverted_request(_PAYLOAD_A), secondary=True)
    assert result.polarity_applied == _ORIGINAL, (
        f"polarity: polarity_applied must follow the primary decoder, got "
        f"{result.polarity_applied!r}"
    )
    assert result.per_decoder[PRIMARY_DECODER].polarity == _ORIGINAL, (
        "polarity: the primary decoder answered at the original polarity only, got "
        f"{result.per_decoder[PRIMARY_DECODER].polarity!r}"
    )
    assert result.per_decoder[SECONDARY_DECODER].polarity == "both", (
        "polarity: the secondary decoder answered at both polarities, got "
        f"{result.per_decoder[SECONDARY_DECODER].polarity!r}"
    )


def test_non_inverted_request_is_decoded_exactly_once_by_each_decoder(
    monkeypatch: pytest.MonkeyPatch,
    make_decoder_outcome: Callable[..., DecoderOutcome],
) -> None:
    """An ordinary symbol is decoded once per decoder: no second pass, no doubled cost. Without
    this case an implementation that inverted unconditionally would satisfy every payload
    assertion in this module while doubling the decode cost of a whole run, so the pass count
    is the assertion, not a side note.
    """
    primary, secondary = _install_decoders(
        monkeypatch,
        make_decoder_outcome,
        primary={_ORIGINAL: _PAYLOAD_A},
        secondary={_ORIGINAL: _PAYLOAD_A},
    )
    request = _request(_PAYLOAD_A)
    assert consensus.is_inverted_request(request) is False, (
        "one pass per decoder: the default palette is not inverted, or this case is wrong"
    )
    result = consensus.decode_symbol(_SYMBOL_PNG, request, secondary=True)
    assert primary.pass_count == 1, (
        f"one pass per decoder: the primary decoder must be called once, got {primary.pass_count}"
    )
    assert secondary is not None and secondary.pass_count == 1, (
        "one pass per decoder: the secondary decoder must be called once, got "
        f"{None if secondary is None else secondary.pass_count}"
    )
    assert result.polarity_applied == _ORIGINAL, (
        "one pass per decoder: polarity_applied must be 'original', got "
        f"{result.polarity_applied!r}"
    )
    assert result.per_decoder[PRIMARY_DECODER].polarity == _ORIGINAL, (
        "one pass per decoder: a single original pass records 'original', got "
        f"{result.per_decoder[PRIMARY_DECODER].polarity!r}"
    )


def test_inverting_twice_restores_the_image() -> None:
    """``invert_image`` is the per-pixel complement, and it is its own inverse."""
    image = numpy.array([[0, 1, 127], [128, 254, 255]], dtype=numpy.uint8)
    inverted = consensus.invert_image(image)
    assert inverted.dtype == numpy.uint8, (
        f"invert_image: the dtype must be preserved, got {inverted.dtype}"
    )
    assert inverted.shape == image.shape, (
        f"invert_image: the shape must be preserved, got {inverted.shape} for {image.shape}"
    )
    assert numpy.array_equal(inverted, 255 - image), (
        "invert_image: every pixel must be replaced by 255 minus itself"
    )
    assert inverted[0, 0] == 255 and inverted[1, 2] == 0, (
        "invert_image: the extremes must map to each other at the same coordinates"
    )
    assert numpy.array_equal(consensus.invert_image(inverted), image), (
        "invert_image: inverting twice must restore the image exactly"
    )
    with pytest.raises(ValueError):
        consensus.invert_image(image.astype(numpy.float64))


def test_relative_luminance_matches_the_wcag_reference_values() -> None:
    """The luminance helper reproduces WCAG 2.x's own reference values. The helper is the
    definition behind ``contrast_floor``, so a drift here would silently change which requests
    a target is expected to refuse.
    """
    assert consensus.relative_luminance((0, 0, 0)) == 0.0, (
        "relative luminance: black must be exactly 0.0"
    )
    assert consensus.relative_luminance((255, 255, 255)) == 1.0, (
        "relative luminance: white must be exactly 1.0"
    )
    grey = consensus.relative_luminance((128, 128, 128))
    assert grey == pytest.approx(0.2159, abs=1e-4), (
        f"relative luminance: mid-grey must be WCAG's 0.2159, got {grey}"
    )
    assert 0.0 < grey < 1.0, (
        "relative luminance: a mid-grey reading must lie strictly between black and white, got "
        f"{grey!r}"
    )


def test_contrast_ratio_matches_the_wcag_reference_values() -> None:
    """The contrast ratio is WCAG's ``(lighter + 0.05) / (darker + 0.05)``, symmetric in its
    arguments.
    """
    black, white, grey = (0, 0, 0), (255, 255, 255), (128, 128, 128)
    assert consensus.contrast_ratio(black, white) == pytest.approx(21.0), (
        "contrast ratio: black on white must be 21:1, the definition's maximum"
    )
    assert consensus.contrast_ratio(white, black) == consensus.contrast_ratio(black, white), (
        "contrast ratio: the ratio must be symmetric, so an inverted palette measures the same"
    )
    assert consensus.contrast_ratio(grey, grey) == pytest.approx(1.0), (
        "contrast ratio: a colour against itself must be exactly 1:1"
    )
    expected_grey = (1.0 + 0.05) / (consensus.relative_luminance(grey) + 0.05)
    assert consensus.contrast_ratio(grey, white) == pytest.approx(expected_grey), (
        f"contrast ratio: mid-grey on white must follow the formula, got "
        f"{consensus.contrast_ratio(grey, white)} for an expected {expected_grey}"
    )
    # ``#888888`` on white measures ~3.5:1, the below-floor pair the invalid-input probe builds,
    # so the helper and that probe cannot drift apart.
    assert consensus.contrast_ratio((0x88, 0x88, 0x88), white) == pytest.approx(3.54, abs=0.01), (
        "contrast ratio: #888888 on white must measure about 3.5:1"
    )

