"""Deterministic payload generation for the corpus: one generator per class.

The corpus is the fixed input every check case is built from, so a record's payload must
be reproducible from the run seed alone: it is a pure function of
``(run_seed, class_name, index)``. Every draw goes through
``qrcheck.config.seeding.derive_seed`` -- the product's only seed-derivation point -- and
is consumed by a PCG64-backed ``numpy`` generator, whose bit stream is stable across numpy
releases. Nothing here reads the clock, the locale, the environment, the filesystem or the
network, and nothing iterates a set or a mapping whose order it did not write itself.

This module owns payload generation only: how many records a class has is
``qrcheck/corpus/builder.py``'s ``CLASS_COUNTS``, and the record id, the ``class`` key and
the ``payload_utf8``/``payload_b64`` choice are the builder's too. Scope is Model 2 QR
only, so no Micro QR or Structured Append payload shape appears here.
"""

__all__ = [
    "ALPHANUMERIC_CHARSET",
    "BOUNDARY_ALPHANUMERIC_VERSIONS",
    "BOUNDARY_BYTE_VERSIONS",
    "BOUNDARY_NUMERIC_VERSIONS",
    "CLASS_GENERATORS",
    "Generator",
    "PayloadDraft",
    "mode_of",
]

from collections.abc import Callable
from dataclasses import dataclass

import numpy

from qrcheck.config.seeding import derive_seed
from qrcheck.corpus import capacity

# ISO/IEC 18004 Table 5: the alphanumeric character set in the standard's own order --
# ten digits, twenty-six uppercase letters, then the nine symbols -- 45 characters in
# all. A lowercase letter, a comma or an underscore lies outside it and forces byte
# mode, which is why no alphanumeric payload may contain one.
ALPHANUMERIC_CHARSET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

# The 35 alphanumeric characters that are not digits. A boundary-alphanumeric payload
# draws its first character from here, so the payload can never collapse into numeric
# mode whatever the seeded draw produces.
ALPHANUMERIC_NON_DIGIT: str = ALPHANUMERIC_CHARSET[10:]

# Boundary versions. A boundary payload of mode ``m`` at version ``v`` is exactly
# ``capacity.max_payload_bytes(v, "L", m)`` characters long, which makes the same payload
# need Version ``v`` at EC L and Version ``v + 1`` at EC M: L is one step down from the
# run's default level M, so the payload's minimal version differs by exactly one between
# the requested level and the level below it -- the predicate the wrong-EC-level defect
# relies on, satisfied by construction rather than by luck. Only Versions 1-5 have that
# property with the ISO capacity tables; from Version 6 on one EC step costs more than
# one version. Records therefore cycle through these five versions.
BOUNDARY_NUMERIC_VERSIONS: tuple[int, ...] = (1, 2, 3, 4, 5)
BOUNDARY_ALPHANUMERIC_VERSIONS: tuple[int, ...] = (1, 2, 3, 4, 5)
BOUNDARY_BYTE_VERSIONS: tuple[int, ...] = (1, 2, 3, 4, 5)

_LOWERCASE: str = "abcdefghijklmnopqrstuvwxyz"
_UPPERCASE: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS: str = "0123456789"

# Printable ASCII for the embedded-NUL class: no control character other than the NUL the
# generator inserts, so the only unusual byte in the payload is the one under test.
_PRINTABLE: str = _LOWERCASE + _UPPERCASE + _DIGITS + " .:-_/"

# RFC 4648 section 6: the base32 alphabet. An otpauth secret is drawn from these 32
# characters and from nothing else, so the URI stays a valid provisioning string.
_BASE32: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

# Code-point ranges, as half-open intervals ready for ``Generator.integers``. Every code
# point in the emoji range encodes to a four-byte UTF-8 sequence, which is what gives the
# emoji class its four-byte coverage. The two-byte range starts at U+00A0 to skip the C1
# controls, and the three-byte range stops below U+D800 so no lone surrogate can be
# produced.
_UTF8_RANGES: dict[int, tuple[int, int]] = {
    2: (0x00A0, 0x0800),
    3: (0x0800, 0xD800),
    4: (0x10000, 0x110000),
}
_UTF8_WIDTHS: tuple[int, ...] = (2, 3, 4)
_EMOJI_RANGE: tuple[int, int] = (0x1F300, 0x1FB00)
_HEBREW_RANGE: tuple[int, int] = (0x05D0, 0x05EB)
_ARABIC_RANGE: tuple[int, int] = (0x0621, 0x064B)

# utf8-multibyte payload byte lengths. The first band is 65-84 bytes on purpose: at EC M a
# 64-byte payload fits Version 5, whose byte capacity is 84, so truncating any payload in
# that band to 64 bytes keeps the SAME EC-M version. That is the predicate the truncating
# defect is detected through; the longer bands exercise larger symbols and version
# information blocks. A fixed ASCII prefix of ``_UTF8_PREFIX_BYTES`` bytes starts every
# payload, so mode is byte from the first byte onward.
_UTF8_MULTIBYTE_BANDS: tuple[tuple[int, int], ...] = ((65, 84), (85, 140), (141, 200))
_UTF8_PREFIX_BYTES: int = 8

# The three pathological repeat shapes, selected by index so that consecutive records
# differ in shape as well as in content.
_REPEAT_SHAPES: tuple[str, ...] = ("single", "alternating", "block")
_REPEAT_CHARS: tuple[str, ...] = ("A", "7", "z", "=", "-", "+")
_REPEAT_BLOCK_ALPHABET: str = _UPPERCASE + _DIGITS

# Characters the WiFi payload format requires escaped with a backslash, and the printable
# alphabet the SSID is drawn from -- it deliberately includes ``;`` and ``:`` so the
# escaping is exercised rather than assumed.
_WIFI_SPECIALS: tuple[str, ...] = ("\\", ";", ",", ":", '"')
_WIFI_ESCAPES: dict[str, str] = {char: "\\" + char for char in _WIFI_SPECIALS}
_WIFI_SSID_ALPHABET: str = _LOWERCASE + _UPPERCASE + _DIGITS + " -_;:"
_WIFI_PASSWORD_ALPHABET: str = _LOWERCASE + _UPPERCASE + _DIGITS

# Seeded name pools for the structured-data classes, so a vCard, a MECARD and an otpauth
# URI read as real records rather than as filler. Every entry is ASCII.
_GIVEN_NAMES: tuple[str, ...] = (
    "Ada", "Grace", "Alan", "Edsger", "Barbara", "Donald", "Frances", "Ken",
    "Margaret", "Niklaus", "Radia", "Tony",
)
_FAMILY_NAMES: tuple[str, ...] = (
    "Lovelace", "Hopper", "Turing", "Dijkstra", "Liskov", "Knuth", "Allen", "Thompson",
    "Hamilton", "Wirth", "Perlman", "Hoare",
)
_ISSUERS: tuple[str, ...] = (
    "Example", "Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Soylent",
)

_DIGIT_SET: frozenset[str] = frozenset(_DIGITS)
_ALPHANUMERIC_SET: frozenset[str] = frozenset(ALPHANUMERIC_CHARSET)


@dataclass(frozen=True, slots=True)
class PayloadDraft:
    """One generated payload with the facts the corpus record needs about it.

    ``payload`` is the payload exactly as it will be encoded: ``str`` for the classes
    whose content is text and ``bytes`` for the two classes that carry arbitrary bytes.
    ``expected_mode`` is the encoding mode derived from the payload's own bytes, never a
    claim made by the generator, and ``notes`` records how the payload was derived --
    for a boundary class, the capacity arithmetic that fixed its length.
    """

    payload: str | bytes
    expected_mode: str
    notes: str

    @property
    def payload_bytes(self) -> bytes:
        """Return the payload as the bytes a symbol carries.

        A ``str`` payload is UTF-8 encoded, which is precisely what the encoder receives;
        a ``bytes`` payload is returned unchanged. Guarantees the result is ``bytes`` for
        every class, and raises nothing: UTF-8 encoding of a ``str`` is total.
        """
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
        return self.payload


# A class generator: the run seed and the record's index in, one draft out. CLASS_GENERATORS
# is the mapping the corpus builder walks, one entry per payload class.
Generator = Callable[[int, int], PayloadDraft]


def mode_of(payload: bytes) -> str:
    """Return the QR encoding mode that ``payload`` is limited to.

    The answer is a fact about the bytes: ``"numeric"`` when every byte is an ASCII digit,
    ``"alphanumeric"`` when every byte is a member of ``ALPHANUMERIC_CHARSET``, and
    ``"byte"`` otherwise -- which includes every payload holding a byte outside ASCII, so
    any UTF-8 multibyte text is byte mode by construction. The empty payload is reported
    as ``"numeric"``: it is contained in every charset, numeric is the smallest of them,
    and the corpus never emits one. Both readings select Version 1 at every EC level, so
    the choice is unobservable in a version assertion.

    Raises ValueError when ``payload`` is not ``bytes`` -- a ``str`` must be encoded by the
    caller, so that one payload is never classified by two different encodings.
    """
    if not isinstance(payload, bytes):
        raise ValueError(f"mode_of expects encoded bytes, got {type(payload).__name__}")
    if not payload.isascii():
        return "byte"
    text = payload.decode("ascii")
    if all(char in _DIGIT_SET for char in text):
        return "numeric"
    if all(char in _ALPHANUMERIC_SET for char in text):
        return "alphanumeric"
    return "byte"


def _rng(run_seed: int, class_name: str, index: int) -> numpy.random.Generator:
    """Return the one generator a payload draw uses, keyed on its class and index."""
    return numpy.random.Generator(numpy.random.PCG64(derive_seed(run_seed, class_name, index)))


def _length(rng: numpy.random.Generator, low: int, high: int) -> int:
    """Return a seeded integer in ``low..high`` inclusive."""
    return int(rng.integers(low, high + 1))


def _token(rng: numpy.random.Generator, alphabet: str, low: int, high: int) -> str:
    """Return a seeded token of ``low..high`` characters drawn from ``alphabet``."""
    length = _length(rng, low, high)
    return "".join(alphabet[int(code)] for code in rng.integers(0, len(alphabet), length))


def _content_draft(class_name: str, index: int, payload: str | bytes, shape: str) -> PayloadDraft:
    """Wrap a content-class payload, deriving its mode from the encoded bytes."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return PayloadDraft(
        payload=payload,
        expected_mode=mode_of(data),
        notes=f"{class_name}[{index}]: {shape}, {len(data)} payload bytes",
    )


def _boundary_draft(
    class_name: str,
    index: int,
    version: int,
    expected_mode: str,
    payload: str,
) -> PayloadDraft:
    """Verify a boundary payload's mode and capacity arithmetic, then describe it.

    Raises RuntimeError, naming the class, the index and the values observed, when the
    payload is not in ``expected_mode`` -- a wrong ``expected_mode`` would silently
    invalidate the mode-optimality check -- or when the payload does not need Version
    ``version`` at EC L and ``version + 1`` at EC M, or does not fit at EC H.
    """
    observed = mode_of(payload.encode("utf-8"))
    if observed != expected_mode:
        raise RuntimeError(
            f"{class_name}[{index}]: payload was built for {expected_mode} mode but "
            f"mode_of reported {observed}"
        )
    length = len(payload)
    l_version = capacity.minimal_version(length, "L", expected_mode)
    m_version = capacity.minimal_version(length, "M", expected_mode)
    h_version = capacity.minimal_version(length, "H", expected_mode)
    if l_version != version or m_version != version + 1:
        raise RuntimeError(
            f"{class_name}[{index}]: a length of {length} needs Version {l_version} at EC L "
            f"and {m_version} at EC M, not the expected {version} and {version + 1}"
        )
    if h_version is None or h_version > 40:
        raise RuntimeError(
            f"{class_name}[{index}]: a length of {length} does not fit at EC H "
            f"(Version {h_version})"
        )
    return PayloadDraft(
        payload=payload,
        expected_mode=observed,
        notes=(
            f"max_payload_bytes({version}, 'L', '{expected_mode}') = {length}; "
            f"minimal version L={l_version}, M={m_version}, H={h_version}"
        ),
    )


def _boundary_numeric(run_seed: int, index: int) -> PayloadDraft:
    """Return a digits-only payload that fills Version ``v`` at EC L exactly.

    The version is ``BOUNDARY_NUMERIC_VERSIONS[index % 5]`` and the length is the numeric
    capacity of that version at EC L, so the payload needs Version ``v`` at L and
    ``v + 1`` at M. Its digits sit wholly inside the numeric charset, which makes the
    numeric minimal version strictly below the byte-mode minimum for the same payload --
    the property the mode-optimality check asserts.
    """
    version = BOUNDARY_NUMERIC_VERSIONS[index % len(BOUNDARY_NUMERIC_VERSIONS)]
    length = capacity.max_payload_bytes(version, "L", "numeric")
    rng = _rng(run_seed, "boundary-numeric", index)
    payload = "".join(_DIGITS[int(digit)] for digit in rng.integers(0, 10, length))
    return _boundary_draft("boundary-numeric", index, version, "numeric", payload)


def _boundary_alphanumeric(run_seed: int, index: int) -> PayloadDraft:
    """Return an alphanumeric-charset payload filling Version ``v`` at EC L exactly.

    The version is ``BOUNDARY_ALPHANUMERIC_VERSIONS[index % 5]`` and the length is the
    alphanumeric capacity of that version at EC L. Only the 45 characters of
    ``ALPHANUMERIC_CHARSET`` are drawn -- never a lowercase letter, which would force byte
    mode -- and the first character comes from the non-digit subset so the payload cannot
    collapse into numeric mode. Every character is ASCII, which is this class's share of
    the charset coverage.
    """
    version = BOUNDARY_ALPHANUMERIC_VERSIONS[index % len(BOUNDARY_ALPHANUMERIC_VERSIONS)]
    length = capacity.max_payload_bytes(version, "L", "alphanumeric")
    rng = _rng(run_seed, "boundary-alphanumeric", index)
    characters = [
        ALPHANUMERIC_CHARSET[int(code)]
        for code in rng.integers(0, len(ALPHANUMERIC_CHARSET), length)
    ]
    characters[0] = ALPHANUMERIC_NON_DIGIT[
        int(rng.integers(0, len(ALPHANUMERIC_NON_DIGIT)))
    ]
    return _boundary_draft(
        "boundary-alphanumeric", index, version, "alphanumeric", "".join(characters)
    )


def _boundary_byte(run_seed: int, index: int) -> PayloadDraft:
    """Return a byte-mode payload filling Version ``v`` at EC L exactly.

    The version is ``BOUNDARY_BYTE_VERSIONS[index % 5]`` and the length is the byte
    capacity of that version at EC L. Lowercase ASCII letters only: they are outside both
    smaller charsets, so the payload is unambiguously byte mode and can never be re-read
    as numeric or alphanumeric.
    """
    version = BOUNDARY_BYTE_VERSIONS[index % len(BOUNDARY_BYTE_VERSIONS)]
    length = capacity.max_payload_bytes(version, "L", "byte")
    rng = _rng(run_seed, "boundary-byte", index)
    payload = "".join(chr(int(code)) for code in rng.integers(0x61, 0x7B, length))
    return _boundary_draft("boundary-byte", index, version, "byte", payload)


def _utf8_multibyte(run_seed: int, index: int) -> PayloadDraft:
    """Return UTF-8 text mixing two-, three- and four-byte code points.

    Every payload opens with a seeded eight-letter ASCII word, then cycles through a
    two-, a three- and a four-byte code point before drawing further widths from the same
    three ranges, so all three UTF-8 widths appear and the four-byte coverage the charset
    check asserts is guaranteed rather than probable. The byte length lands inside one of
    ``_UTF8_MULTIBYTE_BANDS``; the first band exists so that truncating the payload to 64
    bytes keeps the same EC-M version, which is what makes the truncating defect observable
    instead of version-shifting.
    """
    rng = _rng(run_seed, "utf8-multibyte", index)
    low, high = _UTF8_MULTIBYTE_BANDS[index % len(_UTF8_MULTIBYTE_BANDS)]
    target = _length(rng, low, high)
    characters = [
        _LOWERCASE[int(letter)]
        for letter in rng.integers(0, len(_LOWERCASE), _UTF8_PREFIX_BYTES)
    ]
    size = _UTF8_PREFIX_BYTES
    rotation = index % len(_UTF8_WIDTHS)
    widths = _UTF8_WIDTHS[rotation:] + _UTF8_WIDTHS[:rotation]
    drawn = 0
    widest = max(_UTF8_WIDTHS)
    while size + widest <= target:
        width = widths[drawn % len(widths)]
        range_low, range_high = _UTF8_RANGES[width]
        characters.append(chr(int(rng.integers(range_low, range_high))))
        size += width
        drawn += 1
    while size < target:
        characters.append(_LOWERCASE[int(rng.integers(0, len(_LOWERCASE)))])
        size += 1
    return _content_draft(
        "utf8-multibyte",
        index,
        "".join(characters),
        f"ASCII prefix and {drawn} multibyte code points (2/3/4-byte cycle)",
    )


def _emoji(run_seed: int, index: int) -> PayloadDraft:
    """Return emoji code points, every one of them a four-byte UTF-8 sequence.

    The count is 4..24 code points drawn from U+1F300-U+1FAFF, the pictorial range whose
    every member encodes to four bytes, so this class carries the four-byte coverage of the
    charset check. There is no ASCII prefix and no separator: the payload is emoji from its
    first byte.
    """
    rng = _rng(run_seed, "emoji", index)
    count = _length(rng, 4, 24)
    payload = "".join(
        chr(int(codepoint))
        for codepoint in rng.integers(_EMOJI_RANGE[0], _EMOJI_RANGE[1], count)
    )
    return _content_draft("emoji", index, payload, f"{count} four-byte emoji code points")


def _rtl(run_seed: int, index: int) -> PayloadDraft:
    """Return right-to-left text: Hebrew and Arabic letters mixed with digits and spaces.

    The first character is always a Hebrew letter and the next two are always an ASCII
    digit and a space, so the right-to-left coverage and the digits-and-spaces mixture hold
    whatever the seed draws; each remaining character is Hebrew four times in ten, Arabic
    four times, a digit once and a space once.
    """
    rng = _rng(run_seed, "rtl", index)
    length = _length(rng, 12, 48)
    characters = [chr(int(rng.integers(_HEBREW_RANGE[0], _HEBREW_RANGE[1])))]
    characters.append(_DIGITS[int(rng.integers(0, len(_DIGITS)))])
    characters.append(" ")
    for _ in range(length - len(characters)):
        draw = int(rng.integers(0, 10))
        if draw < 4:
            characters.append(chr(int(rng.integers(_HEBREW_RANGE[0], _HEBREW_RANGE[1]))))
        elif draw < 8:
            characters.append(chr(int(rng.integers(_ARABIC_RANGE[0], _ARABIC_RANGE[1]))))
        elif draw == 8:
            characters.append(_DIGITS[int(rng.integers(0, len(_DIGITS)))])
        else:
            characters.append(" ")
    return _content_draft(
        "rtl", index, "".join(characters), f"{length} right-to-left characters"
    )


def _nul_byte(run_seed: int, index: int) -> PayloadDraft:
    """Return printable ASCII with one embedded NUL, as ``bytes``.

    The payload is 8..40 bytes of printable ASCII with a single ``0x00`` written over one of
    those positions, so the NUL is embedded rather than trailing. It is returned as
    ``bytes`` because an embedded NUL does not survive a round trip through a JSON string,
    which is why the record is stored as base64.
    """
    rng = _rng(run_seed, "nul-byte", index)
    length = _length(rng, 8, 40)
    printable = bytes(
        ord(_PRINTABLE[int(code)]) for code in rng.integers(0, len(_PRINTABLE), length)
    )
    position = int(rng.integers(0, length))
    payload = printable[:position] + b"\x00" + printable[position + 1 :]
    return _content_draft(
        "nul-byte", index, payload, f"printable ASCII with an embedded NUL at offset {position}"
    )


def _url(run_seed: int, index: int) -> PayloadDraft:
    """Return an ``https`` URL with mixed-case path segments, a query and a numeric id.

    Scheme and host are lowercase, so the payload is byte mode -- correct and intended, a
    URL being data rather than an alphanumeric-only string. Every component is seeded, so
    the URL differs between seeds and repeats exactly for one seed and index.
    """
    rng = _rng(run_seed, "url", index)
    host = _token(rng, _LOWERCASE, 6, 12)
    segments = _length(rng, 1, 3)
    path = "/".join(
        _token(rng, _UPPERCASE, 3, 8) + _token(rng, _LOWERCASE, 3, 6)
        for _ in range(segments)
    )
    identifier = int(rng.integers(1, 1_000_000))
    reference = _token(rng, _LOWERCASE + _DIGITS, 4, 10)
    payload = f"https://{host}.example.com/{path}?id={identifier}&ref={reference}"
    return _content_draft("url", index, payload, f"{segments} mixed-case path segment(s)")


def _vcard(run_seed: int, index: int) -> PayloadDraft:
    """Return a vCard 3.0 record with a seeded name, telephone number and address.

    Line endings are a literal LF, fixed, so regenerating the record reproduces it byte for
    byte. The fields are the version 3.0 minimum in the order a reader expects: ``BEGIN``,
    ``VERSION``, ``N``, ``FN``, ``TEL``, ``EMAIL`` and ``END``.
    """
    rng = _rng(run_seed, "vcard", index)
    given = _GIVEN_NAMES[int(rng.integers(0, len(_GIVEN_NAMES)))]
    family = _FAMILY_NAMES[int(rng.integers(0, len(_FAMILY_NAMES)))]
    number = "".join(_DIGITS[int(digit)] for digit in rng.integers(0, 10, 10))
    lines = (
        "BEGIN:VCARD",
        "VERSION:3.0",
        f"N:{family};{given};;;",
        f"FN:{given} {family}",
        f"TEL:+1-{number[:3]}-{number[3:6]}-{number[6:]}",
        f"EMAIL:{given.lower()}.{family.lower()}@example.com",
        "END:VCARD",
    )
    return _content_draft("vcard", index, "\n".join(lines), "vCard 3.0, LF line endings")


def _wifi(run_seed: int, index: int) -> PayloadDraft:
    """Return a ``WIFI:`` configuration payload with a seeded SSID and passphrase.

    The SSID alphabet includes ``;`` and ``:``, both reserved by the format, so the escaping
    is exercised: each reserved character is prefixed with a backslash. The payload ends
    with the format's double terminator.
    """
    rng = _rng(run_seed, "wifi", index)
    ssid = _wifi_escape(_token(rng, _WIFI_SSID_ALPHABET, 6, 14))
    passphrase = _wifi_escape(_token(rng, _WIFI_PASSWORD_ALPHABET, 12, 20))
    payload = f"WIFI:T:WPA;S:{ssid};P:{passphrase};;"
    return _content_draft("wifi", index, payload, "WIFI: WPA configuration payload")


def _mecard(run_seed: int, index: int) -> PayloadDraft:
    """Return a ``MECARD:`` payload with a seeded name, telephone number and address."""
    rng = _rng(run_seed, "mecard", index)
    given = _GIVEN_NAMES[int(rng.integers(0, len(_GIVEN_NAMES)))]
    family = _FAMILY_NAMES[int(rng.integers(0, len(_FAMILY_NAMES)))]
    number = "".join(_DIGITS[int(digit)] for digit in rng.integers(0, 10, 10))
    mailbox = f"{given.lower()}.{family.lower()}@example.com"
    payload = f"MECARD:N:{family},{given};TEL:{number};EMAIL:{mailbox};;"
    return _content_draft("mecard", index, payload, "MECARD contact payload")


def _otp(run_seed: int, index: int) -> PayloadDraft:
    """Return an ``otpauth://totp`` URI with a seeded base32 secret.

    The secret is 32 characters of the RFC 4648 base32 alphabet, so it stays a valid secret,
    and the whole payload is ASCII -- this class's share of the charset coverage, together
    with the boundary-alphanumeric class.
    """
    rng = _rng(run_seed, "otp", index)
    issuer = _ISSUERS[int(rng.integers(0, len(_ISSUERS)))]
    account = _token(rng, _LOWERCASE + _DIGITS, 6, 12)
    secret = "".join(_BASE32[int(code)] for code in rng.integers(0, len(_BASE32), 32))
    payload = f"otpauth://totp/{issuer}:{account}?secret={secret}&issuer={issuer}"
    return _content_draft("otp", index, payload, "otpauth totp URI, ASCII only")


def _repeat_run(run_seed: int, index: int) -> PayloadDraft:
    """Return a payload of pathological repeats for the encoder's mode segmentation.

    The shape follows ``_REPEAT_SHAPES[index % 3]``: one character repeated, a two-character
    alternation, or a three-to-five character block repeated. Length is 40..200 characters,
    long enough that a segmented encoder has to decide how to spend its runs.
    """
    rng = _rng(run_seed, "repeat-run", index)
    length = _length(rng, 40, 200)
    shape = _REPEAT_SHAPES[index % len(_REPEAT_SHAPES)]
    if shape == "single":
        unit = _REPEAT_CHARS[int(rng.integers(0, len(_REPEAT_CHARS)))]
    elif shape == "alternating":
        first = int(rng.integers(0, len(_REPEAT_CHARS)))
        offset = 1 + int(rng.integers(0, len(_REPEAT_CHARS) - 1))
        unit = _REPEAT_CHARS[first] + _REPEAT_CHARS[(first + offset) % len(_REPEAT_CHARS)]
    else:
        unit = _token(rng, _REPEAT_BLOCK_ALPHABET, 3, 5)
    payload = (unit * (length // len(unit) + 1))[:length]
    return _content_draft(
        "repeat-run", index, payload, f"{shape} repeat over {length} characters"
    )


def _random_bytes(run_seed: int, index: int) -> PayloadDraft:
    """Return 8..64 arbitrary bytes, as ``bytes``.

    Every byte value is possible, so the payload is usually invalid UTF-8 -- which is the
    point: it proves the corpus carries payloads no text-based path can silently normalise,
    and it is why the record is stored as base64.
    """
    rng = _rng(run_seed, "random-bytes", index)
    length = _length(rng, 8, 64)
    payload = rng.integers(0, 256, size=length, dtype=numpy.uint8).tobytes()
    return _content_draft("random-bytes", index, payload, f"{length} arbitrary byte values")


def _wifi_escape(value: str) -> str:
    """Return ``value`` with each character the WiFi payload format reserves escaped."""
    return "".join(_WIFI_ESCAPES.get(character, character) for character in value)


# The fourteen payload classes of the corpus, one generator each. Consumers iterate
# ``sorted(CLASS_GENERATORS)``; the declaration order here follows the class inventory and
# carries no meaning of its own.
CLASS_GENERATORS: dict[str, Generator] = {
    "boundary-numeric": _boundary_numeric,
    "boundary-alphanumeric": _boundary_alphanumeric,
    "boundary-byte": _boundary_byte,
    "utf8-multibyte": _utf8_multibyte,
    "emoji": _emoji,
    "rtl": _rtl,
    "nul-byte": _nul_byte,
    "url": _url,
    "vcard": _vcard,
    "wifi": _wifi,
    "mecard": _mecard,
    "otp": _otp,
    "repeat-run": _repeat_run,
    "random-bytes": _random_bytes,
}

# Each boundary class with the version tuple it cycles and the mode its payloads must be in.
_BOUNDARY_SPECS: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("boundary-numeric", BOUNDARY_NUMERIC_VERSIONS, "numeric"),
    ("boundary-alphanumeric", BOUNDARY_ALPHANUMERIC_VERSIONS, "alphanumeric"),
    ("boundary-byte", BOUNDARY_BYTE_VERSIONS, "byte"),
)


def _verify_boundary_predicates() -> None:
    """Raise RuntimeError unless every boundary version meets the length arithmetic.

    For each version of each boundary tuple it recomputes the length the generator uses and
    checks the three capacity claims that make a boundary record useful: Version ``v`` at
    EC L, ``v + 1`` at EC M, and a fit at EC H, where the boundary subset is also run. A
    tuple edited without the arithmetic being re-checked would make every record of its
    class wrong while still looking plausible, so the check runs once, when the module is
    imported.
    """
    for class_name, versions, mode in _BOUNDARY_SPECS:
        for version in versions:
            length = capacity.max_payload_bytes(version, "L", mode)
            l_version = capacity.minimal_version(length, "L", mode)
            m_version = capacity.minimal_version(length, "M", mode)
            h_version = capacity.minimal_version(length, "H", mode)
            if l_version != version or m_version != version + 1:
                raise RuntimeError(
                    f"{class_name}: Version {version} at EC L needs {length} characters, "
                    f"which takes Version {l_version} at EC L and {m_version} at EC M, not "
                    f"the expected {version} and {version + 1}"
                )
            if h_version is None or h_version > 40:
                raise RuntimeError(
                    f"{class_name}: Version {version} at EC L needs {length} characters, "
                    f"which does not fit at EC H (Version {h_version})"
                )


_verify_boundary_predicates()
