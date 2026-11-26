"""Contract test: the corpus capacity tables are provably segno's own tables.

``qrcheck/corpus/capacity.py`` derives every boundary payload length in the corpus, so one
transcription slip there would silently mis-size every boundary record and every capacity assertion
built on it (IM-6). This module compares the embedded tables with ``segno.consts`` row for row and
ties the arithmetic to segno's real acceptance boundary by encoding the largest legal payload of
each mode at sampled versions, asserting segno accepts it and rejects one character more.

:func:`_encode` is the single call site that neutralises the two segno defaults which would make
the comparison meaningless: ``boost_error=False``, because the default spends spare capacity on a
higher error level without changing the version, so an oracle built with it reports a different
level than the one requested and every boundary comparison is wrong; and ``segno.make_qr`` rather
than bare ``segno.make``, because a short payload becomes a **Micro QR** symbol with no ``4V+17``
module count and a different format-information layout -- ``make_qr`` is Model 2 by construction
and takes no ``micro`` argument at all, which ``test_the_model2_entry_point_avoids_micro_symbols``
asserts instead of trusting.

Every expectation is read from ``segno.consts`` at test time; the only literals are ISO/IEC 18004
facts, named once below with the clause they come from. Nothing here reads a product threshold, and
no case is skipped, marked or retried.
"""

from __future__ import annotations

import itertools

import pytest
import segno
from segno import consts as segno_consts

from qrcheck.corpus import builder, capacity

# ISO/IEC 18004 §7.5: Model 2 QR symbols carry Versions 1-40. Micro QR (M1-M4) and
# Structured Append are out of scope for the product, so 1 and 40 are the only ends of the
# range any assertion refers to.
FIRST_MODEL2_VERSION: int = 1
LAST_MODEL2_VERSION: int = 40

# segno keys SYMBOL_CAPACITY and ECC by integer error level, not by "L"/"M"/"Q"/"H": keying
# them with a string raises KeyError, which reads like a capacity disagreement rather than a
# lookup mistake, so every lookup goes through this one mapping.
SEGNO_ERROR_LEVELS: dict[str, int] = {
    "L": segno_consts.ERROR_LEVEL_L,
    "M": segno_consts.ERROR_LEVEL_M,
    "Q": segno_consts.ERROR_LEVEL_Q,
    "H": segno_consts.ERROR_LEVEL_H,
}

# The Micro QR version names segno reports on a QRCode: a Micro symbol answers "M1"-"M4"
# where a Model 2 symbol answers an integer version. Used to assert that a Micro symbol is
# recognisably different from the Model 2 symbol :func:`_encode` returns.
MICRO_VERSION_NAMES: tuple[str, ...] = ("M1", "M2", "M3", "M4")

# The whole of every table: 40 versions x 4 error-correction levels, sorted by version then
# level, as the parametrization each row-for-row comparison runs over.
VERSIONS: tuple[int, ...] = tuple(range(FIRST_MODEL2_VERSION, LAST_MODEL2_VERSION + 1))
VERSION_EC_PAIRS: tuple[tuple[int, str], ...] = tuple(
    sorted(itertools.product(VERSIONS, capacity.EC_LEVELS))
)
VERSION_EC_IDS: list[str] = [f"v{version}-{ec}" for version, ec in VERSION_EC_PAIRS]
EC_MODE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted(itertools.product(capacity.EC_LEVELS, capacity.MODES))
)
EC_MODE_IDS: list[str] = [f"{ec}-{mode}" for ec, mode in EC_MODE_PAIRS]

# Versions at which the payload-length arithmetic is checked against segno's own encoder:
# both ends of the range, Version 7 (the first that carries version-information blocks), and
# the ISO/IEC 18004 §7.4 version-group boundaries 9|10 and 26|27 with a neighbour either
# side, because the character-count indicator changes width across each boundary.
SAMPLE_VERSIONS: tuple[int, ...] = (1, 2, 7, 10, 26, 27, 40)
SAMPLE_POINTS: tuple[tuple[int, str, str], ...] = tuple(
    sorted(itertools.product(SAMPLE_VERSIONS, capacity.EC_LEVELS, capacity.MODES))
)
SAMPLE_POINT_IDS: list[str] = [
    f"v{version}-{ec}-{mode}" for version, ec, mode in SAMPLE_POINTS
]

# Payload lengths whose minimal version is cross-checked against segno's own version search:
# an empty payload, the smallest lengths of each mode, and lengths that land in different
# version groups for different levels, so a character-count-indicator slip cannot hide.
MINIMAL_VERSION_LENGTHS: tuple[int, ...] = (0, 1, 7, 17, 53, 100, 200, 500)
MINIMAL_VERSION_POINTS: tuple[tuple[int, str, str], ...] = tuple(
    sorted(itertools.product(MINIMAL_VERSION_LENGTHS, capacity.EC_LEVELS, capacity.MODES))
)
MINIMAL_VERSION_IDS: list[str] = [
    f"len{length}-{ec}-{mode}" for length, ec, mode in MINIMAL_VERSION_POINTS
]

# The seed the committed corpus is built from (``qrcheck.toml`` ``seed``). Boundary lengths
# are arithmetic rather than seeded, so any seed yields the same lengths; seeding fixes only
# the fillers, and using the committed seed keeps this test aligned with the shipped corpus.
CORPUS_SEED: int = 20240101

# The three levels a boundary record's note records: EC L is the boundary itself — the
# payload is exactly that version's L capacity — and M and H are the levels at which the same
# payload needs a larger version, which is the step the wrong-EC-level defect is caught on.
BOUNDARY_ECS: tuple[str, ...] = ("L", "M", "H")

# One filler per mode, chosen so segno's mode selection agrees with the mode under test: the
# numeric filler is digits only; the alphanumeric filler draws on ISO/IEC 18004 §7.3.4's
# 45-character set and begins with a letter, so no prefix of it is digits-only; the byte
# filler is lowercase, which is outside both smaller charsets and so forces byte mode.
MODE_FILLERS: dict[str, str] = {
    "numeric": "0123456789",
    "alphanumeric": "A1B2C3D4E5F6G7H8I9J0KLMNOPQRSTUVWXYZ $%*+-./:",
    "byte": "qrcheck",
}


def _payload(mode: str, length: int) -> str:
    """Return a `length`-character payload whose encoding mode is exactly `mode`."""
    filler = MODE_FILLERS[mode]
    return (filler * (length // len(filler) + 1))[:length]


def _encode(
    content: str | bytes,
    *,
    error: str | None = None,
    version: int | None = None,
) -> segno.QRCode:
    """Encode `content` with segno as a Model 2 symbol, never boosting the error level. The only
    segno call site in this module, so ``boost_error=False`` cannot be forgotten, and Model 2
    is guaranteed structurally: ``segno.make_qr`` takes no ``micro`` argument, which is exactly
    what makes it the Model 2 entry point, while bare ``segno.make`` would return a Micro
    symbol for a short payload.
    """
    return segno.make_qr(content, error=error, version=version, boost_error=False)


def _payload_length(record: builder.CorpusRecord) -> int:
    """Return `record`'s length in its mode's own unit: characters, or bytes for "byte"."""
    if record.expected_mode == "byte":
        return len(record.payload_bytes)
    if record.payload_utf8 is None:
        raise AssertionError(f"{record.id}: a non-byte record must carry payload_utf8")
    return len(record.payload_utf8)


@pytest.mark.parametrize(("version", "ec"), VERSION_EC_PAIRS, ids=VERSION_EC_IDS)
def test_max_bits_equals_segno_symbol_capacity(version: int, ec: str) -> None:
    """MAX_BITS carries segno's Table 7 data capacity in bits, on every row."""
    oracle_bits = segno_consts.SYMBOL_CAPACITY[version][SEGNO_ERROR_LEVELS[ec]]
    assert capacity.MAX_BITS[version][ec] == oracle_bits
    assert capacity.capacity_bits(version, ec) == oracle_bits


@pytest.mark.parametrize(("version", "ec"), VERSION_EC_PAIRS, ids=VERSION_EC_IDS)
def test_ecc_blocks_equals_segno_ecc(version: int, ec: str) -> None:
    """ECC_BLOCKS, TOTAL_CODEWORDS and both accessors carry segno's Tables 1 and 9."""
    groups = segno_consts.ECC[version][SEGNO_ERROR_LEVELS[ec]]
    blocks, ecc_per_block = capacity.ECC_BLOCKS[version][ec]

    # Table 9 splits some levels into groups that differ in block count and in codewords per
    # block but not in error-correction codewords per block, which is what makes the
    # aggregated pair lossless; asserting that this set has exactly one member is the check.
    assert {group.num_total - group.num_data for group in groups} == {ecc_per_block}
    assert blocks == sum(group.num_blocks for group in groups)
    assert capacity.TOTAL_CODEWORDS[version] == sum(
        group.num_blocks * group.num_total for group in groups
    )

    data_codewords = sum(group.num_blocks * group.num_data for group in groups)
    assert capacity.data_codewords(version, ec) == data_codewords
    assert capacity.MAX_BITS[version][ec] == data_codewords * 8
    assert data_codewords + blocks * ecc_per_block == capacity.TOTAL_CODEWORDS[version]


def test_capacity_tables_hold_model2_versions_only() -> None:
    """The three tables are keyed by Versions 1-40 and by nothing else."""
    expected_keys = set(VERSIONS)
    for table in (capacity.TOTAL_CODEWORDS, capacity.ECC_BLOCKS, capacity.MAX_BITS):
        assert set(table) == expected_keys
        for key in table:
            assert isinstance(key, int) and not isinstance(key, bool), key

    # segno's own tables do carry the Micro rows — integer sentinels at and below 0 — and
    # string-keyed rectangular Micro layouts inside ECC. "No Micro row here" is therefore a
    # real exclusion: the sentinels exist in the oracle and are asserted absent from the
    # product's tables, rather than being absent from both.
    micro_sentinels = {
        segno_consts.VERSION_M1,
        segno_consts.VERSION_M2,
        segno_consts.VERSION_M3,
        segno_consts.VERSION_M4,
    }
    assert micro_sentinels == set(segno_consts.MICRO_VERSIONS)
    assert micro_sentinels <= set(segno_consts.SYMBOL_CAPACITY)
    assert micro_sentinels <= set(segno_consts.ECC)
    for table in (capacity.TOTAL_CODEWORDS, capacity.ECC_BLOCKS, capacity.MAX_BITS):
        assert micro_sentinels.isdisjoint(table)

    for version in VERSIONS:
        assert set(capacity.MAX_BITS[version]) == set(capacity.EC_LEVELS)
        assert set(capacity.ECC_BLOCKS[version]) == set(capacity.EC_LEVELS)

    assert capacity.EC_LEVELS == ("L", "M", "Q", "H")
    assert capacity.MODES == ("numeric", "alphanumeric", "byte")


@pytest.mark.parametrize(("ec", "mode"), EC_MODE_PAIRS, ids=EC_MODE_IDS)
def test_no_capacity_answer_ever_names_a_micro_version(ec: str, mode: str) -> None:
    """Neither accessor answers with a Micro version or leaves the Model 2 range."""
    largest = capacity.max_payload_bytes(LAST_MODEL2_VERSION, ec, mode)
    assert capacity.max_payload_bytes(FIRST_MODEL2_VERSION, ec, mode) > 0

    for length in (0, 1, 7, 17, largest - 1, largest):
        version = capacity.minimal_version(length, ec, mode)
        assert version is not None
        assert FIRST_MODEL2_VERSION <= version <= LAST_MODEL2_VERSION
        assert capacity.max_payload_bytes(version, ec, mode) >= length
        if version > FIRST_MODEL2_VERSION:
            assert capacity.max_payload_bytes(version - 1, ec, mode) < length

    assert capacity.minimal_version(largest, ec, mode) == LAST_MODEL2_VERSION
    assert capacity.minimal_version(largest + 1, ec, mode) is None


@pytest.mark.parametrize(("version", "ec", "mode"), SAMPLE_POINTS, ids=SAMPLE_POINT_IDS)
def test_max_payload_bytes_matches_segno_acceptance(
    version: int, ec: str, mode: str
) -> None:
    """segno accepts exactly `max_payload_bytes` characters in `mode` and rejects one more. This
    is the assertion that ties the arithmetic to the real encoder rather than to a second
    transcription: the payload is built from the mode's own charset, and segno is asked to
    encode it at a pinned version, so a length that is one character too long is precisely a
    `DataOverflowError`.
    """
    length = capacity.max_payload_bytes(version, ec, mode)
    symbol = _encode(_payload(mode, length), error=ec, version=version)
    assert symbol.version == version
    assert symbol.error == ec
    assert symbol.mode == mode

    with pytest.raises(segno.DataOverflowError):
        _encode(_payload(mode, length + 1), error=ec, version=version)


@pytest.mark.parametrize(
    ("length", "ec", "mode"), MINIMAL_VERSION_POINTS, ids=MINIMAL_VERSION_IDS
)
def test_minimal_version_equals_the_version_segno_chooses(
    length: int, ec: str, mode: str
) -> None:
    """`minimal_version` is the version segno itself chooses, and one lower overflows."""
    payload = _payload(mode, length)
    expected = capacity.minimal_version(length, ec, mode)
    assert expected is not None

    symbol = _encode(payload, error=ec)
    assert symbol.version == expected
    assert symbol.error == ec
    if length:
        # An empty payload has no mode of its own; every other filler encodes in `mode`.
        assert symbol.mode == mode

    if expected > FIRST_MODEL2_VERSION:
        # Version 1 has no Model 2 predecessor, so the shortfall is checked above it only.
        with pytest.raises(segno.DataOverflowError):
            _encode(payload, error=ec, version=expected - 1)


def test_the_model2_entry_point_avoids_micro_symbols() -> None:
    """`_encode` cannot produce a Micro symbol, while bare ``segno.make`` would."""
    bare = segno.make(_payload("numeric", 1))
    assert bare.version in MICRO_VERSION_NAMES

    model2 = _encode(_payload("numeric", 1))
    assert model2.version == FIRST_MODEL2_VERSION
    assert isinstance(model2.version, int)


def test_boundary_corpus_payloads_sit_on_capacity_boundaries() -> None:
    """Every boundary record's length is a table capacity, and its note recomputes. The corpus
    builder must derive boundary lengths from these tables rather than transcribe them, and
    each boundary payload must be exactly the EC-L capacity of its version so the same payload
    needs the next version at EC M — the one-step difference the wrong-EC-level defect is
    observed through.
    """
    records = [
        record
        for record in builder.build_corpus(CORPUS_SEED)
        if record.class_name in builder.BOUNDARY_CLASSES
    ]
    assert len(records) == builder.BOUNDARY_RECORDS

    verified: set[tuple[str, int]] = set()
    for record in records:
        mode = record.expected_mode
        length = _payload_length(record)
        version = capacity.minimal_version(length, "L", mode)
        assert version is not None
        capacity_l = capacity.max_payload_bytes(version, "L", mode)
        assert length == capacity_l, (
            f"{record.id}: length {length} is not the EC-L capacity {capacity_l} of "
            f"version {version}"
        )

        versions = {
            ec: capacity.minimal_version(length, ec, mode) for ec in BOUNDARY_ECS
        }
        assert record.notes == (
            f"max_payload_bytes({version}, 'L', {mode!r}) = {capacity_l}; "
            f"minimal version L={versions['L']}, M={versions['M']}, H={versions['H']}"
        ), record.id
        assert versions["M"] is not None and versions["M"] > version, record.id

        if (mode, length) in verified:
            continue
        verified.add((mode, length))
        content = record.payload_bytes if mode == "byte" else record.payload_utf8
        assert content is not None, record.id
        symbol_l = _encode(content, error="L")
        assert symbol_l.version == version, record.id
        assert symbol_l.mode == mode, record.id
        assert _encode(content, error="M").version == versions["M"], record.id

    # Every mode the corpus declares has a boundary witness, so the arithmetic above is
    # exercised across all three charsets rather than in one of them.
    assert {mode for mode, _ in verified} == set(capacity.MODES)

    # Boundary lengths are arithmetic, not seeded: a different seed refills the payloads and
    # must leave every boundary length exactly where the tables put it.
    other_seed = [
        (record.expected_mode, _payload_length(record))
        for record in builder.build_corpus(CORPUS_SEED + 1)
        if record.class_name in builder.BOUNDARY_CLASSES
    ]
    assert other_seed == [(record.expected_mode, _payload_length(record)) for record in records]

