"""The symbol's own 15-bit format information: error correction level, mask, BCH correction.

This module is the single source of the *mask* and the independent second source
of the *error correction level* for the whole framework, which is why no pinned
decoder has to expose either: ``zxing-cpp`` 2.2.0 reports an EC level but neither
version nor mask, and ``opencv`` reports none of the three. ``SY-MASK`` has no
other input, and ``SY-EC-MATCH`` compares this answer against the decoder's, so
a ``format_info_disagreement`` is a finding about the symbol rather than a
restatement of one library's opinion.

The bits are read out of the sampled module grid, so nothing here consults a
decoder. What is implemented, clause by clause:

* ISO/IEC 18004:2015(E) §7.9 -- format information: five data bits (a two-bit EC
  level indicator followed by a three-bit mask reference) and ten BCH(15, 5)
  parity bits.
* ISO/IEC 18004:2015(E) §7.9.1 -- placement of both copies in a Model 2 symbol,
  including the two module positions the timing patterns occupy and the fixed
  dark module that belongs to neither copy.
* ISO/IEC 18004:2015(E) Table 12 -- the EC level indicators ``L=01``, ``M=00``,
  ``Q=11``, ``H=10``.
* ISO/IEC 18004:2015(E) Annex C / Table C.1 -- the 32 valid bit sequences, which
  this module generates from the BCH routine rather than transcribing, so the
  only hand-written values are the generator polynomial and the XOR mask.

Model 2 QR only: a Micro QR symbol has a different format layout and is out of
scope.

The module writes no file, reads no configuration, prints and logs nothing, and
imports no image or decoder library -- ``numpy`` does not even reach
``sys.modules``, because :data:`~qrcheck.decode.base.Matrix` resolves for type
checkers only. It decides no :class:`~qrcheck.report.model.Reason` and never
raises for bad symbol data: an unreadable grid, an unreadable copy, and a copy
corrupted beyond three bits all come back as ``None``, which the consensus layer
records as absent format information and the checks report as
``sampling_failed``. An exception escaping this module would turn a defective
symbol into an ``error`` result and destroy the run-level attribution the
self-check targets depend on, so :class:`ValueError` is raised only for a
programmer error in a caller's argument.
"""

from __future__ import annotations

import dataclasses
import operator
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - annotations only; numpy must stay out of sys.modules
    from qrcheck.decode.base import Matrix

__all__ = [
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
]

# ISO/IEC 18004:2015(E) §7.9.1, Table C.1: the 15 format bits are XOR-masked with
# 101010000010010 before anything is placed, so a symbol whose format modules are
# all light can never present itself as the all-zero (i.e. valid-looking)
# codeword. Literal value 0x5412.
FORMAT_MASK: Final = 0x5412

# ISO/IEC 18004:2015(E) §7.9.1 / Annex C: the BCH(15, 5) generator polynomial
# x^10 + x^8 + x^5 + x^4 + x^2 + x + 1, whose coefficients give 0b10100110111 =
# 0x537. Its code has minimum Hamming distance 7, so any pattern with at most
# three bit errors lies within the correction radius of exactly one codeword.
BCH_GENERATOR: Final = 0x537

#: Derived from that minimum distance: floor((7 - 1) / 2) = 3 bit errors can be
#: corrected per copy, and 2 * 3 < 7 is why no two candidates can ever tie.
MAX_CORRECTABLE_BIT_ERRORS: Final = 3

# ISO/IEC 18004:2015(E) Table 12: the two-bit error correction level indicator
# that leads the format information -- L=01, M=00, Q=11, H=10. The values are not
# ordered by recovery capacity (M is 00 and Q is 11), which is why the mapping is
# declared and never derived. Stored in sorted key order so iterating it is
# deterministic.
EC_INDICATORS: Final[Mapping[str, int]] = types.MappingProxyType(
    {"H": 0b10, "L": 0b01, "M": 0b00, "Q": 0b11}
)

#: The inverse of :data:`EC_INDICATORS`, so a decoded indicator maps back to the
#: level name a check compares against.
INDICATOR_EC_LEVELS: Final[Mapping[int, str]] = types.MappingProxyType(
    {indicator: level for level, indicator in sorted(EC_INDICATORS.items())}
)

#: Canonical iteration order for the 32 codewords: EC level then mask, both
#: ascending, so the candidate set is identical on every import and no set or
#: dict iteration order can reach a decoding result.
_EC_LEVEL_ORDER: Final[tuple[str, ...]] = tuple(sorted(EC_INDICATORS))

# The legal Model 2 module counts are 4V + 17 for versions 1..40, which span
# Version 1: 4 * 1 + 17 = 21 through Version 40: 4 * 40 + 17 = 177 in steps of 4.
_MIN_MODULE_COUNT: Final = 21
_MAX_MODULE_COUNT: Final = 177
_MODULE_COUNT_STEP: Final = 4


@dataclasses.dataclass(frozen=True, slots=True)
class FormatInfo:
    """The format information recovered from a symbol, with the evidence behind it.

    ``ec_level`` and ``mask`` are the recovered values; ``mask`` in 0-7 is the
    only mask evidence the product has, and ``ec_level`` is the reading that is
    independent of both decoders. ``bit_errors`` is the Hamming distance between
    the copy that was used and the codeword it was corrected to, so a perfectly
    printed symbol reports 0 and a damaged one reports how much of its redundancy
    was spent.

    ``copy_index`` names which of the two copies was used (0 or 1), and
    ``copy_bit_errors`` records the distance measured on each copy -- ``None``
    where that copy could not be recovered at all. Because both copies carry the
    same 15 bits, the pair is the redundancy evidence: one copy at distance 0 and
    the other at distance 2 says the second copy was damaged, not that the symbol
    is ambiguous.

    Pass ``copy_bit_errors`` as ``None`` to derive it from ``bit_errors`` at
    ``copy_index``; the invariant that ``copy_bit_errors[copy_index]`` equals
    ``bit_errors`` is enforced either way.
    """

    ec_level: str
    mask: int
    bit_errors: int
    copy_index: int = 0
    copy_bit_errors: tuple[int | None, int | None] | None = None

    def __post_init__(self) -> None:
        """Normalise ``copy_bit_errors`` and reject off-contract values.

        Raises:
            ValueError: ``ec_level`` is not one of ``L``/``M``/``Q``/``H``,
                ``mask`` is outside 0-7, ``bit_errors`` is outside 0-3,
                ``copy_index`` is not 0 or 1, ``copy_bit_errors`` is not a
                two-element sequence, a per-copy distance is outside 0-3, or the
                distance recorded for the used copy disagrees with
                ``bit_errors``.
        """
        if self.ec_level not in EC_INDICATORS:
            raise ValueError(
                f"unknown EC level {self.ec_level!r}; expected one of {_EC_LEVEL_ORDER}"
            )
        if not isinstance(self.mask, int) or not 0 <= self.mask <= 7:
            raise ValueError(f"mask must be an integer in 0-7; got {self.mask!r}")
        if not isinstance(self.bit_errors, int) or not (
            0 <= self.bit_errors <= MAX_CORRECTABLE_BIT_ERRORS
        ):
            raise ValueError(
                f"bit_errors must be an integer in 0-{MAX_CORRECTABLE_BIT_ERRORS}; "
                f"got {self.bit_errors!r}"
            )
        if self.copy_index not in (0, 1):
            raise ValueError(f"copy_index must be 0 or 1; got {self.copy_index!r}")
        if self.copy_bit_errors is None:
            distances = tuple(
                self.bit_errors if index == self.copy_index else None for index in (0, 1)
            )
            object.__setattr__(self, "copy_bit_errors", distances)
        else:
            if len(self.copy_bit_errors) != 2:
                raise ValueError(
                    "copy_bit_errors must hold one distance per copy; "
                    f"got {self.copy_bit_errors!r}"
                )
            for distance in self.copy_bit_errors:
                if distance is None:
                    continue
                if not isinstance(distance, int) or not (
                    0 <= distance <= MAX_CORRECTABLE_BIT_ERRORS
                ):
                    raise ValueError(
                        "each copy distance must be None or an integer in "
                        f"0-{MAX_CORRECTABLE_BIT_ERRORS}; got {distance!r}"
                    )
            used = self.copy_bit_errors[self.copy_index]
            if used != self.bit_errors:
                raise ValueError(
                    f"copy_bit_errors[{self.copy_index}] is {used!r} but bit_errors is "
                    f"{self.bit_errors!r}; the used copy's distance is the reported one"
                )

    def as_dict(self) -> dict[str, Any]:
        """Project this record to a JSON-safe mapping with sorted keys.

        ``copy_bit_errors`` becomes a list because a JSON object has no tuple, and
        ``None`` entries survive as ``null`` so a copy that could not be read stays
        distinguishable from a clean one.
        """
        distances = self.copy_bit_errors if self.copy_bit_errors is not None else (None, None)
        projected: dict[str, Any] = {
            "bit_errors": self.bit_errors,
            "copy_bit_errors": list(distances),
            "copy_index": self.copy_index,
            "ec_level": self.ec_level,
            "mask": self.mask,
        }
        return dict(sorted(projected.items()))


def _bch_codeword(data: int) -> int:
    """Return the unmasked 15-bit codeword holding the five data bits ``data``.

    ISO/IEC 18004:2015(E) §7.9.1: the data bits are shifted left over the ten
    parity positions and the remainder of that polynomial division by
    :data:`BCH_GENERATOR` supplies the parity, so the result is
    ``data << 10 | parity``. This is the module's only BCH implementation:
    :func:`encode_format_info` masks its result for placement and the candidate
    set :func:`decode_bits` searches is generated from it, so no codeword is ever
    transcribed by hand.
    """
    remainder = data << 10
    while remainder.bit_length() > 10:
        remainder ^= BCH_GENERATOR << (remainder.bit_length() - 11)
    return (data << 10) | remainder


def encode_format_info(ec_level: str, mask: int) -> int:
    """Encode an EC level and mask reference into its 15-bit format information.

    The inverse of the codeword search in :func:`decode_bits`, and the reason this
    module needs no format table: all 32 legal values -- at exactly the
    ISO/IEC 18004:2015(E) Annex C / Table C.1 sequences -- and any corrupted
    variant of them can be constructed on demand by a test.

    Args:
        ec_level: ``L``, ``M``, ``Q`` or ``H``.
        mask: mask reference in 0-7.

    Returns:
        The 15-bit value as it appears in the symbol: the two-bit EC indicator
        (:data:`EC_INDICATORS`), then the three-bit mask reference, then the ten
        BCH parity bits, XOR :data:`FORMAT_MASK`.

    Raises:
        ValueError: ``ec_level`` is not one of the four levels, or ``mask`` is not
            an integer in 0-7. Both are programmer errors in a caller's argument,
            never symbol damage.
    """
    if ec_level not in EC_INDICATORS:
        raise ValueError(f"unknown EC level {ec_level!r}; expected one of {_EC_LEVEL_ORDER}")
    if not isinstance(mask, int) or not 0 <= mask <= 7:
        raise ValueError(f"mask must be an integer in 0-7; got {mask!r}")
    data = (EC_INDICATORS[ec_level] << 3) | mask
    return _bch_codeword(data) ^ FORMAT_MASK


def _legal_codewords() -> tuple[tuple[int, str, int], ...]:
    """Build the 32 unmasked codewords as ``(codeword, ec_level, mask)`` triples.

    Generated in :data:`_EC_LEVEL_ORDER` then ascending mask, so the tuple is
    identical on every import and on every interpreter.
    """
    return tuple(
        (_bch_codeword((EC_INDICATORS[level] << 3) | mask), level, mask)
        for level in _EC_LEVEL_ORDER
        for mask in range(8)
    )


#: The unmasked half of ISO/IEC 18004:2015(E) Table C.1, generated at import time.
_LEGAL_CODEWORDS: Final[tuple[tuple[int, str, int], ...]] = _legal_codewords()


def _is_legal_module_count(module_count: int) -> bool:
    """Whether ``module_count`` is a Model 2 dimension, i.e. ``4V + 17`` for version 1-40."""
    return _MIN_MODULE_COUNT <= module_count <= _MAX_MODULE_COUNT and (
        module_count - _MIN_MODULE_COUNT
    ) % _MODULE_COUNT_STEP == 0


def _version_for_module_count(module_count: int) -> int:
    """Return the version implied by a Model 2 module count.

    Used only to validate :func:`format_info_positions`' argument: the version a
    sampled grid implies is owned by ``qrcheck/decode/matrix.py``, so this module
    publishes no version helper of its own.

    Raises:
        ValueError: ``module_count`` is not ``4V + 17`` for a version in 1-40.
    """
    if not isinstance(module_count, int) or not _is_legal_module_count(module_count):
        raise ValueError(
            f"module count {module_count!r} is not a Model 2 QR dimension; expected "
            f"4V + 17 for version 1-40, i.e. {_MIN_MODULE_COUNT} to {_MAX_MODULE_COUNT} "
            f"in steps of {_MODULE_COUNT_STEP}"
        )
    return (module_count - _MIN_MODULE_COUNT) // _MODULE_COUNT_STEP + 1


# ISO/IEC 18004:2015(E) §7.9.1, copy 0 of 2 -- the copy wrapped around the
# top-left finder pattern, indexed by bit number with bit 0 the least significant
# bit of the 15-bit codeword:
#
#   down column 8        left along row 8
#        0  (0, 8)        14  (8, 0)
#        1  (1, 8)        13  (8, 1)
#        2  (2, 8)        12  (8, 2)
#        3  (3, 8)        11  (8, 3)
#        4  (4, 8)        10  (8, 4)
#        5  (5, 8)         9  (8, 5)
#   (6, 8) and (8, 6) are timing patterns and hold no format bit:
#        6  (7, 8)         8  (8, 7)
#        7  (8, 8)  -- the corner module itself
#
# The standard's figure numbers the modules from the most significant bit down, so
# a reader accumulates MSB-first by walking this tuple in reverse.
_COPY0_POSITIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 8),
    (1, 8),
    (2, 8),
    (3, 8),
    (4, 8),
    (5, 8),
    (7, 8),
    (8, 8),
    (8, 7),
    (8, 5),
    (8, 4),
    (8, 3),
    (8, 2),
    (8, 1),
    (8, 0),
)


def format_info_positions(
    module_count: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return the ``(row, column)`` coordinates of format bits 0-14 in both copies.

    ``positions[bit]`` is the module carrying bit ``bit``, with bit 0 the least
    significant bit of the 15-bit codeword, so the standard's MSB-first reading
    order is ``reversed(positions)``. Coordinates are indexed as
    ``matrix[row][column]``, matching :data:`~qrcheck.decode.base.Matrix`.

    Copy 0 sits around the top-left finder pattern and never moves. Copy 1 is
    split between the top-right and bottom-left corners, so it depends on the
    module count: bits 0-7 run along row 8 from column ``N - 1`` down to
    ``N - 8``, and bits 8-14 run up column 8 from row ``N - 7`` to ``N - 1``.
    The fixed dark module at ``(N - 8, 8)`` carries no format bit and is never
    returned, and neither is any timing-pattern module (row 6 or column 6).

    Args:
        module_count: ``N = 4V + 17`` for a Model 2 symbol of version ``V``.

    Returns:
        ``(copy 0, copy 1)``, each a 15-position tuple.

    Raises:
        ValueError: ``module_count`` is not a legal Model 2 dimension. That is a
            programmer error in the caller's argument; a grid whose dimension is
            unusable yields ``None`` from :func:`read_format_copies` instead of
            reaching this function's error.
    """
    _version_for_module_count(module_count)
    copy_one: tuple[tuple[int, int], ...] = (
        # bits 0-7: row 8, columns N-1 down to N-8
        *((8, module_count - 1 - offset) for offset in range(8)),
        # bits 8-14: column 8, rows N-7 up to N-1 (the dark module at N-8 is not here)
        *((module_count - 7 + offset, 8) for offset in range(7)),
    )
    return (_COPY0_POSITIONS, copy_one)


def _module_count(matrix: Matrix) -> int | None:
    """Return the module count of a grid, or ``None`` when it is not a legal one.

    Only the outer length is consulted: a grid whose rows or columns are shorter
    than the module count is caught when a position is read, so a ragged grid
    degrades copy by copy instead of failing as a whole.
    """
    try:
        module_count = len(matrix)
    except TypeError:
        return None
    if not isinstance(module_count, int) or not _is_legal_module_count(module_count):
        return None
    return module_count


def _read_copy(matrix: Matrix, positions: tuple[tuple[int, int], ...]) -> int | None:
    """Accumulate one copy's 15 bits MSB-first, or ``None`` if it cannot be read.

    Bit 14 is read first so the returned integer is the codeword in its natural
    orientation. A position outside the grid (short row, ragged grid, non-indexable
    object) makes the whole copy unreadable, because a copy missing a module is not
    a copy with an unknown bit.
    """
    raw = 0
    try:
        for bit in range(len(positions) - 1, -1, -1):
            row, column = positions[bit]
            raw = (raw << 1) | (1 if matrix[row][column] else 0)
    except (IndexError, KeyError, TypeError):
        return None
    return raw


def read_format_copies(matrix: Matrix) -> tuple[int | None, int | None]:
    """Read both copies of the 15-bit format information out of a module grid.

    Any object indexable as ``matrix[row][column]`` works: the sampled
    ``numpy.bool_`` grid of :data:`~qrcheck.decode.base.Matrix` and a plain
    list-of-lists are both valid, and a truthy cell means a **dark** module, so a
    set bit is a ``1``. No polarity normalisation happens here -- an inverted
    symbol is normalised by the consensus layer before it samples a grid.

    Args:
        matrix: the sampled module grid, ``N x N`` with dark modules truthy.

    Returns:
        ``(copy 0, copy 1)`` as raw 15-bit integers, each ``None`` when that copy
        could not be read -- unreadable object, unusable dimension, or a grid too
        short or too ragged to hold every one of its 15 positions.
    """
    module_count = _module_count(matrix)
    if module_count is None:
        return (None, None)
    copy_zero, copy_one = format_info_positions(module_count)
    return (_read_copy(matrix, copy_zero), _read_copy(matrix, copy_one))


def decode_bits(raw: int) -> tuple[str, int, int] | None:
    """Decode one copy's raw 15 bits into ``(ec_level, mask, bit_errors)``.

    The copy is unmasked with :data:`FORMAT_MASK` and then matched against the 32
    generated codewords. Because the code's minimum Hamming distance is 7 and
    ``2 * 3 < 7``, the correction spheres of radius three cannot overlap, so at
    most one candidate is ever within range and no tie can occur; the search
    still keeps the nearest candidate and breaks an impossible tie by codeword
    order rather than by iteration order, so the answer never depends on how the
    candidate set was built.

    Args:
        raw: the 15 bits as read from the grid, mask still applied. Anything
            supporting ``__index__`` is accepted -- a Python ``int`` and a
            ``numpy`` integer alike -- and anything else yields ``None``, so a
            caller cannot turn an unusable value into an exception.

    Returns:
        The recovered level and mask together with the number of bit errors
        corrected (0-3), or ``None`` when no codeword lies within three bits --
        which is the honest answer for a copy damaged beyond its redundancy.
    """
    try:
        value = operator.index(raw)
    except TypeError:
        return None
    if not 0 <= value <= 0x7FFF:
        return None
    unmasked = value ^ FORMAT_MASK
    best: tuple[str, int, int] | None = None
    for codeword, ec_level, mask in _LEGAL_CODEWORDS:
        bit_errors = (unmasked ^ codeword).bit_count()
        if bit_errors > MAX_CORRECTABLE_BIT_ERRORS:
            continue
        if best is None or bit_errors < best[2]:
            best = (ec_level, mask, bit_errors)
    return best


def decode_format_info(matrix: Matrix) -> FormatInfo | None:
    """Recover the symbol's format information from both of its copies.

    Each copy is decoded independently and the one with the smaller
    ``bit_errors`` wins, because a symbol that spent redundancy on one copy is
    read from the copy that spent less. A tie keeps copy 0, which is a fixed
    choice and not whichever copy happened to decode first.

    Args:
        matrix: the sampled module grid, ``N x N`` with dark modules truthy.

    Returns:
        The recovered :class:`FormatInfo` carrying both copies' distances, or
        ``None`` when neither copy could be recovered from the grid. Never raises
        for unreadable symbol data.
    """
    raw_zero, raw_one = read_format_copies(matrix)
    decoded_zero = None if raw_zero is None else decode_bits(raw_zero)
    decoded_one = None if raw_one is None else decode_bits(raw_one)
    copy_bit_errors: tuple[int | None, int | None] = (
        None if decoded_zero is None else decoded_zero[2],
        None if decoded_one is None else decoded_one[2],
    )
    copy_index = 0
    chosen: tuple[str, int, int] | None = None
    for index, decoded in enumerate((decoded_zero, decoded_one)):
        if decoded is None or (chosen is not None and decoded[2] >= chosen[2]):
            # A copy that could not be decoded is skipped; an equal distance keeps
            # the earlier copy, which is the documented copy-0 tie-break.
            continue
        copy_index, chosen = index, decoded
    if chosen is None:
        return None
    ec_level, mask, bit_errors = chosen
    return FormatInfo(
        ec_level=ec_level,
        mask=mask,
        bit_errors=bit_errors,
        copy_index=copy_index,
        copy_bit_errors=copy_bit_errors,
    )
