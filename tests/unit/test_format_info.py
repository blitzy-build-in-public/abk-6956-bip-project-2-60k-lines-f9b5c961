"""The 15-bit format-information contract of ``qrcheck.decode.format_info``.

The fifteen format bits beside the top-left finder pattern, repeated along the top-right and
bottom-left edges, carry the EC level and the mask reference, which makes them the product's
only source of the *mask* and its independent second source of the *EC level*: ``zxing-cpp``
2.2.0 reports an EC level but neither version nor mask and ``opencv`` reports none of the three,
yet ``SY-MASK``, ``SY-MATRIX-EQUIV`` and ``SY-EC-MATCH`` must assert all three (IM-1), so a
wrong reading here is wrong in three checks at once and the contract is pinned exhaustively
rather than by example.

ISO/IEC 18004:2015(E) §7.9 defines a BCH(15, 5) code with generator ``0x537``, codewords
XOR-masked with ``0x5412`` and minimum Hamming distance 7; the 32 codewords are generated
through ``encode_format_info`` rather than transcribed, and ``2 * 3 < 7`` is why radius-three
correction is unambiguous, so all 32 x C(15, 3) = 14560 three-bit corruptions must return the
original pair with ``bit_errors == 3`` while a four-bit corruption must never be reported as
that pair. A majority vote across the two copies would invent a third value present in neither,
so the reader prefers the copy that spent less redundancy and reports both; an unreadable grid,
an unusable module count or a copy damaged beyond its redundancy all come back ``None``, because
a symbol defect must become a failed check rather than an ``error`` result; and since every case
writes its bits and reads them back through the same ``format_info_positions`` table, a
mirrored, rotated or bit-reversed convention would satisfy all of them at once, so a
``segno``-placed symbol proves the convention against the ``error`` and ``mask`` it was built
with.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
from collections import Counter
from typing import Final

import numpy
import pytest
import segno

from qrcheck.decode import base as decode_base
from qrcheck.decode import format_info as format_info_module
from qrcheck.decode.format_info import (
    BCH_GENERATOR,
    EC_INDICATORS,
    FORMAT_MASK,
    INDICATOR_EC_LEVELS,
    MAX_CORRECTABLE_BIT_ERRORS,
    FormatInfo,
    decode_bits,
    decode_format_info,
    encode_format_info,
    format_info_positions,
    read_format_copies,
)

# EC_INDICATORS is stored in sorted key order, so this tuple is ("H", "L", "M",
# "Q"): ISO/IEC 18004:2015(E) Table 12 numbers the levels L=01, M=00, Q=11,
# H=10, which is not the order of increasing recovery capacity. Iterating the
# sorted keys rather than a hand-written sequence keeps the tables and the
# parametrisation ids identical on every interpreter and every import.
EC_LEVELS: Final[tuple[str, ...]] = tuple(sorted(EC_INDICATORS))

#: The three-bit mask reference has exactly eight legal values, 0-7.
MASKS: Final[tuple[int, ...]] = tuple(range(8))

#: ISO/IEC 18004:2015(E) Table 12 exactly as the standard numbers it: the two-bit
#: error correction level indicator that leads the fifteen format bits. This is the
#: one table typed rather than derived, because it is data from the standard and no
#: part of the implementation can produce it -- and it is typed here once, so both
#: the constant assertion and the codeword-structure assertion measure the module
#: against the standard rather than against itself.
TABLE_12_INDICATORS: Final[dict[str, int]] = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}

#: The 32 (EC level, mask) pairs -- one per legal format codeword -- in sorted order.
CODEWORD_CASES: Final[tuple[tuple[str, int], ...]] = tuple(
    (ec_level, mask) for ec_level in EC_LEVELS for mask in MASKS
)

#: Model 2 versions sampled by the placement assertions: the smallest, the first
#: with an alignment pattern, the first with version information, two mid-range
#: ones and the largest. ``N = 4V + 17`` is the only dimension input they need.
VERSIONS: Final[tuple[int, ...]] = (1, 2, 7, 10, 20, 40)

#: The module count the two-copy assertions build grids at -- Version 2, whose
#: 25 x 25 grid is big enough for every format position and small enough to build
#: once per parametrised case.
GRID_MODULE_COUNT: Final = 25

#: The versions the real-symbol proof samples: the smallest symbol, the first with
#: an alignment pattern, and the first whose format information shares a grid with
#: the version information, so the same fixed bit layout is exercised in three
#: different surroundings.
REAL_SYMBOL_VERSIONS: Final[tuple[int, ...]] = (1, 2, 7)

#: ``QRCHECK`` is seven characters, which Version 1 carries at EC level H, so one
#: constant payload keeps every sampled (version, EC level, mask) case encodable.
REAL_SYMBOL_PAYLOAD: Final = "QRCHECK"

#: The 3 x 4 x 8 real symbols the placement proof decodes, in declaration order.
REAL_SYMBOL_CASES: Final[tuple[tuple[int, str, int], ...]] = tuple(
    (version, ec_level, mask)
    for version in REAL_SYMBOL_VERSIONS
    for ec_level in EC_LEVELS
    for mask in MASKS
)

#: The 15-bit space: every value a single format copy can present.
_MAX_RAW: Final = 0x7FFF


def _codeword_id(case: tuple[str, int]) -> str:
    """Render one (EC level, mask) pair as a stable, readable parametrisation id."""
    ec_level, mask = case
    return f"{ec_level}-mask{mask}"


def _version_id(version: int) -> str:
    """Render one Model 2 version as a stable parametrisation id."""
    return f"version-{version}"


def _symbol_id(case: tuple[int, str, int]) -> str:
    """Render one (version, EC level, mask) triple as a stable parametrisation id."""
    version, ec_level, mask = case
    return f"{_version_id(version)}-{_codeword_id((ec_level, mask))}"


def _bits(*positions: int) -> int:
    """The 15-bit error pattern that flips exactly the modules at ``positions``."""
    pattern = 0
    for position in positions:
        pattern |= 1 << position
    return pattern


def _distance(left: int, right: int) -> int:
    """Hamming distance between two 15-bit values."""
    return (left ^ right).bit_count()


def _bch_remainder(codeword: int) -> int:
    """Remainder of ``codeword`` divided by :data:`BCH_GENERATOR`. Deliberately a different
    formulation from the module's own ``_bch_codeword``: bit-at-a-time elimination from the
    most significant bit downwards, instead of a loop over ``bit_length``.
    """
    remainder = codeword
    for shift in range(remainder.bit_length() - 1, 9, -1):
        if (remainder >> shift) & 1:
            remainder ^= BCH_GENERATOR << (shift - 10)
    return remainder


def _sphere_size(radius: int) -> int:
    """How many 15-bit values lie within ``radius`` bit flips of one codeword."""
    return sum(math.comb(15, errors) for errors in range(radius + 1))


def _grid(module_count: int, copy_zero: int, copy_one: int) -> decode_base.Matrix:
    """Build a module grid carrying ``copy_zero`` and ``copy_one`` in the two copies. The result
    is the shape every consumer of the sampler sees -- a ``numpy.bool_`` array of
    ``module_count`` squared modules, dark = ``True``, indexed ``grid[row][column]`` -- with
    every module that carries no format bit left light, so only the fifteen bits of each copy
    are ever set.
    """
    grid = numpy.zeros((module_count, module_count), dtype=bool)
    for raw, positions in zip((copy_zero, copy_one), format_info_positions(module_count)):
        for bit, (row, column) in enumerate(positions):
            grid[row][column] = bool((raw >> bit) & 1)
    return grid


def _flipped(
    grid: decode_base.Matrix,
    module_count: int,
    copy_index: int,
    flips: tuple[int, ...],
) -> decode_base.Matrix:
    """Copy of ``grid`` with the listed bit positions of one format copy inverted."""
    positions = format_info_positions(module_count)[copy_index]
    flipped = grid.copy()
    for bit in flips:
        row, column = positions[bit]
        flipped[row][column] = not flipped[row][column]
    return flipped


@functools.lru_cache(maxsize=None)
def _space_projection() -> tuple[tuple[tuple[int, str, int, int], ...], tuple[int, ...]]:
    """Project the whole 15-bit space once: ``(decodable, undecodable)``. The first element holds
    ``(raw, ec_level, mask, bit_errors)`` for each copy the decoder accepts and the second
    holds every copy it rejects, so the three exhaustive assertions share one pass instead of
    re-deriving it.
    """
    decodable: list[tuple[int, str, int, int]] = []
    undecodable: list[int] = []
    for raw in range(_MAX_RAW + 1):
        decoded = decode_bits(raw)
        if decoded is None:
            undecodable.append(raw)
        else:
            decodable.append((raw, *decoded))
    return (tuple(decodable), tuple(undecodable))


def test_the_module_publishes_the_names_the_checks_import() -> None:
    """The public surface is the eleven names its consumers were written against."""
    assert set(format_info_module.__all__) == {
        "BCH_GENERATOR",
        "EC_INDICATORS",
        "FORMAT_MASK",
        "FormatInfo",
        "INDICATOR_EC_LEVELS",
        "MAX_CORRECTABLE_BIT_ERRORS",
        "decode_bits",
        "decode_format_info",
        "encode_format_info",
        "format_info_positions",
        "read_format_copies",
    }
    for name in format_info_module.__all__:
        assert hasattr(format_info_module, name), f"{name} is exported but not defined"


def test_the_constants_are_the_ones_iso_18004_states() -> None:
    """``0x5412``, ``0x537``, a radius of 3, and the four EC indicators of Table 12."""
    # ISO/IEC 18004:2015(E) §7.9.1: the 15 format bits are XOR-masked with
    # 101010000010010 = 0x5412 before placement.
    assert FORMAT_MASK == 0x5412
    # Annex C: the BCH(15, 5) generator x^10 + x^8 + x^5 + x^4 + x^2 + x + 1 =
    # 0b10100110111 = 0x537, a code of minimum distance 7.
    assert BCH_GENERATOR == 0x537
    assert BCH_GENERATOR.bit_length() == 11
    # floor((7 - 1) / 2) = 3 correctable bit errors per copy.
    assert MAX_CORRECTABLE_BIT_ERRORS == 3
    # Table 12: the two-bit EC indicator is L=01, M=00, Q=11, H=10.
    assert dict(EC_INDICATORS) == TABLE_12_INDICATORS
    assert dict(INDICATOR_EC_LEVELS) == {
        indicator: level for level, indicator in TABLE_12_INDICATORS.items()
    }
    # Four levels, four distinct indicators, each used exactly once.
    assert sorted(EC_INDICATORS.values()) == [0, 1, 2, 3]


@pytest.mark.parametrize("case", CODEWORD_CASES, ids=_codeword_id)
def test_every_legal_codeword_decodes_to_the_pair_that_encoded_it(
    case: tuple[str, int],
) -> None:
    """An undamaged copy of any legal codeword costs zero bit errors."""
    ec_level, mask = case
    raw = encode_format_info(ec_level, mask)
    assert 0 <= raw <= _MAX_RAW, f"{ec_level}/{mask} encoded to {raw!r}, not fifteen bits"
    assert decode_bits(raw) == (ec_level, mask, 0)


@pytest.mark.parametrize("case", CODEWORD_CASES, ids=_codeword_id)
def test_every_codeword_carries_its_data_bits_and_a_zero_bch_remainder(
    case: tuple[str, int],
) -> None:
    """Unmasked, each codeword is ``indicator << 13 | mask << 10 | parity``. This is the
    structural half of the contract: the ``0x5412`` mask is applied, the ten parity bits are
    the remainder of the polynomial division by ``0x537`` (so the value is a codeword at all),
    and the five data bits are exactly the two-bit EC indicator of Table 12 followed by the
    three-bit mask reference.
    """
    ec_level, mask = case
    unmasked = encode_format_info(ec_level, mask) ^ FORMAT_MASK
    assert _bch_remainder(unmasked) == 0, (
        f"{ec_level}/{mask} is not a codeword of the 0x537 generator: "
        f"remainder {_bch_remainder(unmasked):#x}"
    )
    # Bits 14-13: the EC indicator, L=01, M=00, Q=11, H=10 (Table 12).
    assert unmasked >> 13 == TABLE_12_INDICATORS[ec_level]
    # Bits 12-10: the three-bit mask reference.
    assert (unmasked >> 10) & 0b111 == mask


def test_the_masked_and_unmasked_views_are_disjoint_so_the_mask_is_applied() -> None:
    """The mask is not a no-op: no placed value is also a bare codeword. ``0x5412`` has Hamming
    weight 5 while the code's minimum distance is 7, so masking every codeword moves it out of
    the code entirely; the two sets of 32 values therefore cannot share a member.
    """
    placed = {encode_format_info(ec_level, mask) for ec_level, mask in CODEWORD_CASES}
    bare = {value ^ FORMAT_MASK for value in placed}
    assert len(placed) == len(CODEWORD_CASES)
    assert len(bare) == len(CODEWORD_CASES)
    assert placed.isdisjoint(bare)


@pytest.mark.parametrize("case", CODEWORD_CASES, ids=_codeword_id)
def test_decoding_the_wrong_side_of_the_mask_never_recovers_the_pair(
    case: tuple[str, int],
) -> None:
    """``decode_bits`` unmasks its own input, so it must not be fed an unmasked value. It may then
    return nothing or another codeword, but never the pair that encoded it, which is what
    proves the ``0x5412`` step happens inside the decoder rather than in its caller.
    """
    ec_level, mask = case
    decoded = decode_bits(encode_format_info(ec_level, mask) ^ FORMAT_MASK)
    assert decoded is None or (decoded[0], decoded[1]) != (ec_level, mask)


def test_the_minimum_hamming_distance_is_exactly_seven() -> None:
    """The property that makes three-bit correction safe and four-bit correction unsafe. The
    masked and unmasked views must show the same distances, because ``0x5412`` is a constant
    XOR.
    """
    placed = tuple(encode_format_info(ec_level, mask) for ec_level, mask in CODEWORD_CASES)
    bare = tuple(value ^ FORMAT_MASK for value in placed)
    assert len(placed) == 32
    assert len(set(placed)) == 32, "two (EC level, mask) pairs share a codeword"

    distances = Counter(
        _distance(left, right) for left, right in itertools.combinations(placed, 2)
    )
    assert sum(distances.values()) == math.comb(32, 2) == 496
    assert min(distances) == 7
    assert sorted(distances) == [7, 8, 15]
    assert distances == Counter({7: 240, 8: 240, 15: 16})
    bare_distances = Counter(
        _distance(left, right) for left, right in itertools.combinations(bare, 2)
    )
    assert bare_distances == distances


#: One arbitrary legal (EC level, mask) pair for the damage and placement cases.
#: It is an input rather than an expectation: every other expected pair in this
#: module is produced by ``encode_format_info`` or read back out of that encoding,
#: and the real-symbol proof takes its expectation from the symbol instead.
SAMPLE_CASE: Final[tuple[str, int]] = ("Q", 5)

#: Grids whose dimension is not ``4V + 17``, or which cannot be indexed at all. The
#: reader must answer "no copy" for every one of them instead of raising.
_UNUSABLE_GRIDS: Final[tuple[tuple[str, object], ...]] = (
    ("twenty-modules", numpy.zeros((20, 20), dtype=bool)),
    ("twenty-two-modules", numpy.zeros((22, 22), dtype=bool)),
    ("one-hundred-seventy-eight-modules", numpy.zeros((178, 178), dtype=bool)),
    ("zero-modules", numpy.zeros((0, 0), dtype=bool)),
    ("short-rows", numpy.zeros((21, 3), dtype=bool)),
    ("ragged-list", [[False] * width for width in range(21)]),
    ("empty-list", []),
    ("none", None),
    ("integer", 5),
    ("text", "not a module grid"),
)


@pytest.mark.parametrize("case", CODEWORD_CASES, ids=_codeword_id)
def test_every_three_bit_error_is_corrected_back_to_its_own_codeword(
    case: tuple[str, int],
) -> None:
    """All 32 x C(15, 3) = 14560 three-bit corruptions are corrected, and reported. ``bit_errors
    == 3`` is part of the contract rather than a detail: the evidence records how much
    redundancy a copy spent, and a decoder that snapped to the right answer while reporting
    zero would hide that the symbol was damaged.
    """
    ec_level, mask = case
    raw = encode_format_info(ec_level, mask)
    patterns = tuple(itertools.combinations(range(15), 3))
    assert len(patterns) == math.comb(15, 3) == 455
    for pattern in patterns:
        corrupted = raw ^ _bits(*pattern)
        assert decode_bits(corrupted) == (ec_level, mask, 3), (
            f"{ec_level}/mask{mask} corrupted at {pattern} decoded to "
            f"{decode_bits(corrupted)!r}"
        )


def test_a_four_bit_error_is_never_reported_as_the_original_pair() -> None:
    """All 32 x C(15, 4) = 43680 four-bit corruptions, and what a decoder may say about them. Four
    bit errors lie outside the radius-three sphere, so the original pair is unreachable and
    must never be returned: a decoder that answered with it would be claiming a correction the
    code cannot support.
    """
    patterns = tuple(itertools.combinations(range(15), 4))
    assert len(patterns) == math.comb(15, 4) == 1365
    undetected = 0
    misdecoded = 0
    for ec_level, mask in CODEWORD_CASES:
        raw = encode_format_info(ec_level, mask)
        for pattern in patterns:
            corrupted = raw ^ _bits(*pattern)
            decoded = decode_bits(corrupted)
            if decoded is None:
                undetected += 1
                continue
            decoded_level, decoded_mask, bit_errors = decoded
            assert (decoded_level, decoded_mask) != (ec_level, mask), (
                f"{ec_level}/mask{mask} corrupted at {pattern} was reported as itself"
            )
            assert bit_errors == MAX_CORRECTABLE_BIT_ERRORS
            assert _distance(corrupted, encode_format_info(decoded_level, decoded_mask)) == (
                bit_errors
            ), f"{decoded_level}/mask{decoded_mask} is not inside the radius it claims"
            misdecoded += 1
    assert undetected + misdecoded == len(CODEWORD_CASES) * len(patterns)
    # Some patterns are detected as undecodable, so detection works at distance four.
    assert undetected > 0
    # And some are not: distance 7 cannot distinguish a four-bit error from a
    # three-bit error on another codeword. Recorded here so the split is visible.
    assert misdecoded > 0


def test_the_whole_fifteen_bit_space_decodes_within_the_correction_radius() -> None:
    """An exhaustive sweep of all 32768 possible copies, not a sample of them. The count of
    decodable values is ``32 x (1 + 15 + 105 + 455) = 18432`` only because those radius-three
    spheres do not overlap, which restates the minimum distance of 7 as arithmetic over the
    whole space; an overlapping pair would make the union smaller.
    """
    decodable, undecodable = _space_projection()
    assert len(decodable) + len(undecodable) == _MAX_RAW + 1 == 32768
    for raw, ec_level, mask, bit_errors in decodable:
        assert 0 <= bit_errors <= MAX_CORRECTABLE_BIT_ERRORS
        assert _distance(raw, encode_format_info(ec_level, mask)) == bit_errors
    assert len(decodable) == len(CODEWORD_CASES) * _sphere_size(MAX_CORRECTABLE_BIT_ERRORS)
    assert len(undecodable) == (_MAX_RAW + 1) - len(decodable)
    assert set(undecodable).isdisjoint({raw for raw, *_ in decodable})


@pytest.mark.parametrize("case", CODEWORD_CASES, ids=_codeword_id)
def test_both_clean_copies_are_read_and_agree(case: tuple[str, int]) -> None:
    """An undamaged symbol reports the pair with both copies at zero distance."""
    ec_level, mask = case
    raw = encode_format_info(ec_level, mask)
    grid = _grid(GRID_MODULE_COUNT, raw, raw)
    assert isinstance(grid, decode_base.Matrix)
    assert read_format_copies(grid) == (raw, raw)
    info = decode_format_info(grid)
    assert isinstance(info, FormatInfo)
    assert (info.ec_level, info.mask) == (ec_level, mask)
    assert info.bit_errors == 0
    assert info.copy_index == 0
    assert info.copy_bit_errors == (0, 0)


@pytest.mark.parametrize(
    ("corrupted_copy", "error_count"),
    # The sorted product of copy then error count, so the parametrisation order is fixed
    # by this declaration rather than by decorator order.
    [
        pytest.param(copy_index, error_count, id=f"copy{copy_index}-{error_count}-bit")
        for copy_index in (0, 1)
        for error_count in range(1, MAX_CORRECTABLE_BIT_ERRORS + 1)
    ],
)
def test_a_corrupted_copy_is_visible_per_copy_not_averaged_away(
    corrupted_copy: int,
    error_count: int,
) -> None:
    """Damaging one copy leaves the pair intact and names the damaged copy. A reader that merged
    the copies -- by majority vote, or by taking whichever bit "looked" right -- would invent a
    third value present in neither copy and would lose the fact that the symbol is damaged.
    """
    raw = encode_format_info(*SAMPLE_CASE)
    grid = _grid(GRID_MODULE_COUNT, raw, raw)
    corrupted = _flipped(grid, GRID_MODULE_COUNT, corrupted_copy, tuple(range(error_count)))
    assert read_format_copies(corrupted) != (raw, raw), "the corruption did not land in the grid"
    info = decode_format_info(corrupted)
    assert info is not None
    assert (info.ec_level, info.mask) == SAMPLE_CASE
    assert info.copy_index == 1 - corrupted_copy, "the undamaged copy is the one to read"
    assert info.bit_errors == 0
    distances: list[int | None] = [0, 0]
    distances[corrupted_copy] = error_count
    assert info.copy_bit_errors == tuple(distances)
    assert info.as_dict()["copy_bit_errors"] == distances


@pytest.mark.parametrize(
    ("copy_zero_errors", "copy_one_errors"),
    ((2, 1), (1, 2)),
    ids=("copy-one-spent-less", "copy-zero-spent-less"),
)
def test_the_copy_that_spent_less_redundancy_is_the_one_that_is_read(
    copy_zero_errors: int,
    copy_one_errors: int,
) -> None:
    """Both copies damaged: the answer comes from the one at the smaller distance."""
    raw = encode_format_info(*SAMPLE_CASE)
    grid = _grid(GRID_MODULE_COUNT, raw, raw)
    grid = _flipped(grid, GRID_MODULE_COUNT, 0, tuple(range(copy_zero_errors)))
    grid = _flipped(grid, GRID_MODULE_COUNT, 1, tuple(range(copy_one_errors)))
    info = decode_format_info(grid)
    assert info is not None
    assert (info.ec_level, info.mask) == SAMPLE_CASE
    assert info.copy_index == (0 if copy_zero_errors < copy_one_errors else 1)
    assert info.bit_errors == min(copy_zero_errors, copy_one_errors)
    assert info.copy_bit_errors == (copy_zero_errors, copy_one_errors)


def test_a_copy_that_cannot_be_recovered_is_recorded_as_none() -> None:
    """An unreadable copy is ``None`` beside the distance of the copy that was read. ``None`` is
    not interchangeable with a distance of 0 or of 3: it says the copy carries no recoverable
    codeword at all, which is evidence about the symbol that a numeric distance would have
    silently discarded.
    """
    raw = encode_format_info(*SAMPLE_CASE)
    unrecoverable = _space_projection()[1][0]
    assert decode_bits(unrecoverable) is None
    grid = _grid(GRID_MODULE_COUNT, unrecoverable, raw)
    info = decode_format_info(grid)
    assert info is not None
    assert (info.ec_level, info.mask) == SAMPLE_CASE
    assert info.copy_index == 1
    assert info.bit_errors == 0
    assert info.copy_bit_errors == (None, 0)
    assert info.as_dict()["copy_bit_errors"] == [None, 0]


@pytest.mark.parametrize("version", VERSIONS, ids=_version_id)
def test_each_copy_holds_fifteen_disjoint_positions(version: int) -> None:
    """Thirty module positions, in range, disjoint, off the timing patterns. The fixed dark module
    at ``(4V + 9, 8)`` -- ``(N - 8, 8)`` -- belongs to neither copy, and row 6 and column 6
    carry the timing patterns; shifting any of these by one would still produce a
    plausible-looking answer, which is a worse failure than ``None``.
    """
    module_count = 4 * version + 17
    copy_zero, copy_one = format_info_positions(module_count)
    assert len(copy_zero) == 15
    assert len(copy_one) == 15
    assert len(set(copy_zero) | set(copy_one)) == 30
    assert set(copy_zero).isdisjoint(copy_one)
    for row, column in copy_zero + copy_one:
        assert 0 <= row < module_count
        assert 0 <= column < module_count
        assert row != 6 and column != 6, "a timing-pattern module carries no format bit"
        assert (row, column) != (module_count - 8, 8), "the dark module is not a format bit"


def test_copy_zero_never_moves() -> None:
    """Copy 0 wraps the top-left finder pattern and is identical for every version."""
    expected = (
        *((row, 8) for row in range(6)),
        (7, 8),
        (8, 8),
        (8, 7),
        *((8, column) for column in range(5, -1, -1)),
    )
    for version in VERSIONS:
        copy_zero, _ = format_info_positions(4 * version + 17)
        assert copy_zero == expected, f"copy 0 moved at version {version}"


@pytest.mark.parametrize("version", VERSIONS, ids=_version_id)
def test_copy_one_follows_the_module_count(version: int) -> None:
    """Copy 1 is split: bits 0-7 along row 8, bits 8-14 up the last seven of column 8."""
    module_count = 4 * version + 17
    _, copy_one = format_info_positions(module_count)
    assert copy_one[:8] == tuple((8, module_count - 1 - offset) for offset in range(8))
    assert copy_one[8:] == tuple((module_count - 7 + offset, 8) for offset in range(7))


# ISO/IEC 18004:2015(E) §7.9.1 is the clause that places these fifteen bits twice, and segno
# places them by that clause rather than by this module's table. A grid written and read back
# through ``format_info_positions`` cannot prove the convention, because a mirrored, rotated or
# bit-reversed table satisfies both halves of that round trip; the expectation here is instead
# the pair the symbol was built with -- segno's own ``error`` and ``mask`` -- which no reading
# of this module's table can derive.
@pytest.mark.parametrize("case", REAL_SYMBOL_CASES, ids=_symbol_id)
def test_a_real_symbol_reports_the_level_and_mask_it_was_built_with(
    case: tuple[int, str, int],
) -> None:
    """A real symbol's own format modules decode to its own EC level and mask. ``segno.make_qr``
    builds Model 2 only and ``boost_error=False`` keeps the EC level the case asked for (AAP
    §0.3.5), so ``symbol.error`` and ``symbol.mask`` are the expectation this module cannot
    supply for itself: they are what an ISO/IEC 18004:2015(E) §7.9.1 placement put in the grid.
    """
    version, ec_level, mask = case
    symbol = segno.make_qr(
        REAL_SYMBOL_PAYLOAD,
        error=ec_level,
        version=version,
        mask=mask,
        boost_error=False,
    )
    # ``matrix`` is rows of module values -- ``1`` dark, no quiet zone -- which is
    # exactly the grid a sampler hands the reader of a sampled matrix.
    grid = numpy.array(symbol.matrix, dtype=bool)
    module_count = 4 * version + 17
    assert grid.shape == (module_count, module_count)
    copy_zero, copy_one = read_format_copies(grid)
    assert copy_zero is not None and copy_one is not None, "a real symbol lost a format copy"
    assert copy_zero == copy_one, "the standard writes the same fifteen bits twice"
    info = decode_format_info(grid)
    assert info is not None
    assert info.bit_errors == 0
    assert (info.ec_level, info.mask) == (symbol.error, symbol.mask)


@pytest.mark.parametrize("module_count", (20, 22, 178, 0, -3), ids=str)
def test_an_illegal_module_count_is_a_programmer_error(module_count: int) -> None:
    """A dimension that is not ``4V + 17`` is a caller bug, so it raises. Symbol damage never
    reaches this: ``read_format_copies`` screens the grid first and answers ``None``, which
    keeps a bad symbol out of the exception path.
    """
    with pytest.raises(ValueError, match="Model 2"):
        format_info_positions(module_count)


@pytest.mark.parametrize(
    ("label", "grid"),
    _UNUSABLE_GRIDS,
    ids=[label for label, _ in _UNUSABLE_GRIDS],
)
def test_an_unusable_grid_reads_no_copy_and_returns_none(label: str, grid: object) -> None:
    """No dimension, no exception: an unreadable grid yields ``None`` and nothing else."""
    assert read_format_copies(grid) == (None, None), f"{label} returned a copy it could not read"
    assert decode_format_info(grid) is None, f"{label} returned format information"


@pytest.mark.parametrize("index", range(4), ids=("first", "second", "third", "fourth"))
def test_a_grid_carrying_no_legal_format_information_decodes_to_none(index: int) -> None:
    """A copy more than three bits from every codeword is undecodable, not guessed. The values are
    the smallest few that this module's own distance arithmetic puts outside every radius-three
    sphere, so they are not lifted out of the decoder.
    """
    unrecoverable = _space_projection()[1][index]
    assert decode_bits(unrecoverable) is None
    grid = _grid(GRID_MODULE_COUNT, unrecoverable, unrecoverable)
    assert read_format_copies(grid) == (unrecoverable, unrecoverable)
    assert decode_format_info(grid) is None


@pytest.mark.parametrize(
    ("label", "dark"),
    (("all-light", False), ("all-dark", True)),
    ids=("all-light", "all-dark"),
)
def test_a_uniform_grid_decodes_to_none(label: str, dark: bool) -> None:
    """A blank or a fully dark symbol carries no format information at all. The ``0x5412`` mask is
    what makes the all-light case honest: without it, fifteen light modules would present the
    all-zero word, which is a valid-looking codeword.
    """
    grid = numpy.full((GRID_MODULE_COUNT, GRID_MODULE_COUNT), dark, dtype=bool)
    assert decode_format_info(grid) is None, f"{label} decoded to format information"


def test_decode_bits_accepts_any_index_and_rejects_everything_else() -> None:
    """A copy read out of a numpy grid is a numpy integer; anything unusable is ``None``."""
    raw = encode_format_info(*SAMPLE_CASE)
    assert decode_bits(numpy.uint16(raw)) == (*SAMPLE_CASE, 0)
    assert decode_bits(numpy.int64(raw)) == (*SAMPLE_CASE, 0)
    for unusable in (-1, _MAX_RAW + 1, 0xFFFF, 3.0, "0x5412", None, (1, 2), b"\x00\x01"):
        assert decode_bits(unusable) is None, f"{unusable!r} was decoded instead of rejected"


@pytest.mark.parametrize("ec_level", ("", "X", "l", "MM", None, 1), ids=repr)
def test_encode_format_info_rejects_an_unknown_ec_level(ec_level: object) -> None:
    """Only the four names of Table 12 are encodable."""
    with pytest.raises(ValueError, match="EC level"):
        encode_format_info(ec_level, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("mask", (-1, 8, 1000, "3", None, 3.0), ids=repr)
def test_encode_format_info_rejects_an_out_of_range_mask(mask: object) -> None:
    """The mask reference is three bits: 0-7, and nothing that merely looks like one."""
    with pytest.raises(ValueError, match="mask"):
        encode_format_info("M", mask)  # type: ignore[arg-type]


def test_format_info_cannot_claim_more_damage_than_the_code_corrects() -> None:
    """The record refuses a distance, a copy index or a level outside its contract."""
    with pytest.raises(ValueError, match="bit_errors"):
        FormatInfo(ec_level="M", mask=0, bit_errors=MAX_CORRECTABLE_BIT_ERRORS + 1)
    with pytest.raises(ValueError, match="copy_index"):
        FormatInfo(ec_level="M", mask=0, bit_errors=0, copy_index=2)
    with pytest.raises(ValueError, match="unknown EC level"):
        FormatInfo(ec_level="X", mask=0, bit_errors=0)
    with pytest.raises(ValueError, match="mask"):
        FormatInfo(ec_level="M", mask=8, bit_errors=0)


def test_format_info_records_the_distance_of_the_copy_it_used() -> None:
    """The used copy's distance is the reported one; the other is kept beside it."""
    defaulted = FormatInfo(ec_level="L", mask=3, bit_errors=2, copy_index=1)
    assert defaulted.copy_bit_errors == (None, 2)
    with pytest.raises(ValueError, match="copy_bit_errors"):
        FormatInfo(ec_level="L", mask=3, bit_errors=2, copy_index=1, copy_bit_errors=(2, 1))


def test_format_info_rejects_malformed_copy_distances() -> None:
    """Both copies' distances are recorded, and each is ``None`` or a 0-3 integer."""
    with pytest.raises(ValueError, match="one distance per copy"):
        FormatInfo(ec_level="L", mask=3, bit_errors=0, copy_bit_errors=(0,))
    with pytest.raises(ValueError, match="each copy distance"):
        FormatInfo(ec_level="L", mask=3, bit_errors=3, copy_bit_errors=(3, 4))


def test_as_dict_projects_the_evidence_keys_exactly_once() -> None:
    """``decoded.json`` keys cannot drift: the projection is asserted in one place."""
    info = FormatInfo(ec_level="Q", mask=5, bit_errors=2, copy_index=1, copy_bit_errors=(3, 2))
    projected = info.as_dict()
    assert list(projected) == [
        "bit_errors",
        "copy_bit_errors",
        "copy_index",
        "ec_level",
        "mask",
    ]
    assert projected == {
        "bit_errors": 2,
        "copy_bit_errors": [3, 2],
        "copy_index": 1,
        "ec_level": "Q",
        "mask": 5,
    }
    assert json.loads(json.dumps(projected)) == projected
    unreadable = FormatInfo(
        ec_level="L",
        mask=0,
        bit_errors=0,
        copy_index=1,
        copy_bit_errors=(None, 0),
    )
    assert unreadable.as_dict()["copy_bit_errors"] == [None, 0]
    assert json.loads(json.dumps(unreadable.as_dict()))["copy_bit_errors"] == [None, 0]

