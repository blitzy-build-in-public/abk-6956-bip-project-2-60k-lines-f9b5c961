"""Contract test: the corpus's record schema, canonical wire form, inventory and predicates.

``corpus/default.jsonl`` is a committed repository file and the fixed input every corpus-driven
case is built from, so three promises rest on it: the wire form is canonical -- one JSON object
per line, keys sorted at every depth, non-ASCII payloads written literally rather than as escape
sequences, LF endings and exactly one trailing newline -- which makes a regenerated corpus
byte-comparable with the committed one (``tests/integration/test_corpus_determinism.py``); the
inventory is the builder's, with ``builder.CLASS_COUNTS`` matching the table below record for
record, together with the 240-record total and the 40-record boundary subset; and the
discriminating predicates hold, since the ``truncated_payload`` and ``wrong_ec_level``
injections are observable only through a payload longer than the 64-byte truncation window and a
boundary payload whose minimal version differs between EC L and M, without which witnesses they
would be silently inert while every count assertion still passed.

Two properties no single-record assertion would catch are asserted too: ``build_corpus`` run
twice with one seed compares equal, because a builder iterating a ``set`` or seeding from
``hash()`` would differ across machines; and the filter is asked for a mis-cased class name and
for a glob that matches nothing, because "no records selected" is the runner's exit-2 condition
that this layer must report rather than raise. Nothing here reads a product threshold, opens a
socket, spawns a process or reads the clock.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from qrcheck.corpus import builder, capacity, payloads
from qrcheck.corpus.builder import CorpusRecord, build_corpus
from qrcheck.corpus.io import filter_by_class, load_corpus, write_corpus

# The builder's inventory, written out once: 240 records over 14 classes, of which the three
# ``boundary-*`` classes are the 40-record subset the L, Q and H runs are driven over. This is the
# one place a count literal is legitimate -- ``builder.CLASS_COUNTS`` is the single home of those
# numbers and must match this table -- so every other assertion reads this table or the builder.

EXPECTED_CLASS_COUNTS: Final[dict[str, int]] = {
    "boundary-alphanumeric": 12,
    "boundary-byte": 16,
    "boundary-numeric": 12,
    "emoji": 12,
    "mecard": 10,
    "nul-byte": 6,
    "otp": 12,
    "random-bytes": 40,
    "repeat-run": 30,
    "rtl": 8,
    "url": 30,
    "utf8-multibyte": 24,
    "vcard": 16,
    "wifi": 12,
}

# The builder's own totals, ``builder.TOTAL_RECORDS`` and ``builder.BOUNDARY_RECORDS``, and the
# three boundary classes that give the subset its definition -- the `boundary-*` glob selects
# exactly these.
EXPECTED_TOTAL_RECORDS: Final[int] = 240
EXPECTED_BOUNDARY_RECORDS: Final[int] = 40
EXPECTED_BOUNDARY_CLASSES: Final[tuple[str, ...]] = (
    "boundary-alphanumeric",
    "boundary-byte",
    "boundary-numeric",
)

# ``builder.MD_CHARSET_CLASSES``: the classes the six ``MD-CHARSET`` coverages are drawn from --
# ASCII from the alphanumeric boundary class and ``otp``, UTF-8 multibyte from
# ``utf8-multibyte``, emoji and four-byte code points from ``emoji`` plus the widest
# ``utf8-multibyte`` records, right-to-left from ``rtl`` and an embedded NUL from ``nul-byte``.
EXPECTED_MD_CHARSET_CLASSES: Final[tuple[str, ...]] = (
    "boundary-alphanumeric",
    "emoji",
    "nul-byte",
    "otp",
    "rtl",
    "utf8-multibyte",
)

# The ``truncated_payload`` defect cuts a longer payload back to 64 bytes. The window is a
# property of the defect the corpus must make observable, not a tunable, so it is stated here
# once rather than read from the builder's private constant.
TRUNCATION_BYTES: Final[int] = 64

# The seed this module builds its own corpora from. Any non-negative seed is equally valid -- the
# builder must be a pure function of whatever seed it is handed -- and the configured default seed
# is already exercised by the ``built_corpus`` fixture, so a different one here is deliberate
# rather than an oversight.
TEST_SEED: Final[int] = 7

# A second seed, used only to show the seed actually reaches the payloads: two seeds that
# produced the same corpus would pass the build-twice determinism assertion while proving
# nothing about reproducibility.
OTHER_SEED: Final[int] = 8

# The boundary record's ``notes`` text, exactly as
# ``qrcheck/corpus/payloads.py::_boundary_draft`` writes it:
# "max_payload_bytes(2, 'L', 'byte') = 32; minimal version L=2, M=3, H=4".
BOUNDARY_NOTES: Final[re.Pattern[str]] = re.compile(
    r"max_payload_bytes\((?P<version>\d{1,2}), 'L', '(?P<mode>numeric|alphanumeric|byte)'\) = "
    r"(?P<length>\d+); minimal version L=(?P<l_version>\d{1,2}), "
    r"M=(?P<m_version>\d{1,2}), H=(?P<h_version>\d{1,2})"
)

# A content record's ``notes`` text, exactly as
# ``qrcheck/corpus/payloads.py::_content_draft`` writes it:
# "emoji[0]: 23 four-byte emoji code points, 92 payload bytes".
CONTENT_NOTES: Final[re.Pattern[str]] = re.compile(
    r"(?P<class_name>[a-z0-9][a-z0-9-]*)\[(?P<index>\d{1,3})\]: (?P<shape>.+), "
    r"(?P<length>\d+) payload bytes"
)

# The payload field each class is expected to use. ``builder._payload_fields`` sends text a
# UTF-8 codec round-trips and that holds no NUL to ``payload_utf8`` and everything else to
# ``payload_b64``, so exactly the two classes whose generators return ``bytes`` are base64 --
# a 4-byte code point is valid UTF-8 text and therefore rides in ``payload_utf8``.
BASE64_CLASSES: Final[tuple[str, ...]] = ("nul-byte", "random-bytes")


# ---------------------------------------------------------------------------
# Fixtures: one build and one written file for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def records() -> list[CorpusRecord]:
    """Return the corpus the builder produces from :data:`TEST_SEED`, once for the module.
    Module-scoped because ``build_corpus`` walks 240 payload generators and every test below
    reads the same records; a per-test build would repeat that work dozens of times without
    asserting anything extra.
    """
    return build_corpus(TEST_SEED)


@pytest.fixture(scope="module")
def corpus_path(records: list[CorpusRecord], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write ``records`` through :func:`~qrcheck.corpus.io.write_corpus` and return the path. The
    canonical writer is the only formatter this module uses, so the malformed-input cases are
    built by editing these exact lines rather than by rendering records a second way.
    """
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    write_corpus(records, path)
    return path


# ---------------------------------------------------------------------------
# Canonical wire form and round trip
# ---------------------------------------------------------------------------


def test_write_then_load_returns_the_same_records(
    records: list[CorpusRecord], corpus_path: Path
) -> None:
    """Loading the written file returns the records it was written from, in the same order."""
    loaded = load_corpus(corpus_path)
    assert loaded == records
    assert [record.id for record in loaded] == [record.id for record in records]


def test_the_file_is_utf8_with_lf_endings_and_one_trailing_newline(corpus_path: Path) -> None:
    """The corpus is UTF-8 text with LF endings, one final newline and no trailing space."""
    data = corpus_path.read_bytes()
    text = data.decode("utf-8")
    assert b"\r" not in data
    assert text.endswith("\n")
    assert not text.endswith("\n\n"), "the file must end with exactly one newline"
    assert "\n\n" not in text, "a blank line would read back as a malformed record"
    lines = text.split("\n")[:-1]
    assert len(lines) == EXPECTED_TOTAL_RECORDS
    for line in lines:
        assert line == line.rstrip(), f"line ends with whitespace: {line[-20:]!r}"


def test_every_line_is_canonical_sorted_key_json(
    records: list[CorpusRecord], corpus_path: Path
) -> None:
    """Each line is one JSON object in exactly the canonical rendering of its record."""
    lines = corpus_path.read_text(encoding="utf-8").split("\n")[:-1]
    for line, record in zip(lines, records):
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert list(parsed) == sorted(parsed), f"{record.id}: keys are not sorted"
        assert line == json.dumps(parsed, sort_keys=True, ensure_ascii=False), (
            f"{record.id}: line is not the canonical rendering (separators, escaping or key "
            f"order differ)"
        )
        assert parsed == record.to_dict()


def test_non_ascii_payloads_are_written_as_literal_utf8(
    records: list[CorpusRecord], corpus_path: Path
) -> None:
    """``ensure_ascii=False`` is proven on a payload that would otherwise be escaped. Asserting
    "no ``\\u`` appears in the file" would also hold for an empty file, so the assertion is
    made twice over a witness the ``emoji``, ``rtl`` and ``utf8-multibyte`` classes guarantee:
    the escaping of its payload differs from the payload itself, and the payload's UTF-8 bytes
    are present in the file verbatim.
    """
    witnesses = [
        record
        for record in records
        if record.payload_utf8 is not None and not record.payload_utf8.isascii()
    ]
    assert witnesses, "no non-ASCII payload exists, so ensure_ascii=False cannot be observed"
    data = corpus_path.read_bytes()
    assert any(byte > 0x7F for byte in data)
    for witness in witnesses:
        payload = witness.payload_utf8
        assert payload is not None
        assert json.dumps(payload) != json.dumps(payload, ensure_ascii=False)
        assert payload.encode("utf-8") in data, f"{witness.id}: payload is not literal UTF-8"


def test_an_empty_corpus_writes_an_empty_file(tmp_path: Path) -> None:
    """No records means a zero-byte file, because there is no last line to terminate."""
    path = tmp_path / "empty.jsonl"
    write_corpus([], path)
    assert path.read_bytes() == b""
    assert load_corpus(path) == []


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def test_the_builder_inventory_is_the_declared_inventory() -> None:
    """The builder's counts, total and boundary subset are the figures declared in this module."""
    assert builder.CLASS_COUNTS == EXPECTED_CLASS_COUNTS
    assert builder.CLASS_NAMES == tuple(sorted(EXPECTED_CLASS_COUNTS))
    assert builder.TOTAL_RECORDS == EXPECTED_TOTAL_RECORDS == sum(EXPECTED_CLASS_COUNTS.values())
    assert builder.BOUNDARY_RECORDS == EXPECTED_BOUNDARY_RECORDS
    assert builder.BOUNDARY_CLASSES == EXPECTED_BOUNDARY_CLASSES
    assert (
        sum(EXPECTED_CLASS_COUNTS[name] for name in EXPECTED_BOUNDARY_CLASSES)
        == EXPECTED_BOUNDARY_RECORDS
    )
    assert builder.MD_CHARSET_CLASSES == EXPECTED_MD_CHARSET_CLASSES
    assert len(EXPECTED_CLASS_COUNTS) == 14


def test_the_built_corpus_holds_the_inventory(records: list[CorpusRecord]) -> None:
    """240 records over the 14 classes, 40 of them boundary records and 200 of them not."""
    counts = Counter(record.class_name for record in records)
    assert len(records) == builder.TOTAL_RECORDS == EXPECTED_TOTAL_RECORDS
    assert counts == builder.CLASS_COUNTS
    assert sorted(counts) == sorted(EXPECTED_CLASS_COUNTS)
    boundary = [record for record in records if record.class_name in builder.BOUNDARY_CLASSES]
    assert len(boundary) == builder.BOUNDARY_RECORDS == EXPECTED_BOUNDARY_RECORDS
    assert len(records) == len(boundary) + (EXPECTED_TOTAL_RECORDS - EXPECTED_BOUNDARY_RECORDS)
    for class_name in builder.BOUNDARY_CLASSES:
        assert counts[class_name] == EXPECTED_CLASS_COUNTS[class_name]


def test_the_inventory_does_not_depend_on_the_seed(
    records: list[CorpusRecord], built_corpus: tuple[CorpusRecord, ...]
) -> None:
    """The configured default seed builds the same ids, classes and counts as this module's."""
    assert [record.id for record in built_corpus] == [record.id for record in records]
    assert Counter(record.class_name for record in built_corpus) == builder.CLASS_COUNTS


def test_the_committed_corpus_carries_the_same_inventory(
    committed_corpus: tuple[CorpusRecord, ...]
) -> None:
    """A read-only count check on ``corpus/default.jsonl`` itself. Regeneration against the
    committed file is ``tests/integration/test_corpus_determinism.py``'s assertion; this module
    only confirms the committed file holds the inventory ``builder`` declares.
    """
    assert len(committed_corpus) == EXPECTED_TOTAL_RECORDS
    assert Counter(record.class_name for record in committed_corpus) == builder.CLASS_COUNTS
    boundary = [
        record for record in committed_corpus if record.class_name in builder.BOUNDARY_CLASSES
    ]
    assert len(boundary) == EXPECTED_BOUNDARY_RECORDS


# ---------------------------------------------------------------------------
# Record schema (``qrcheck.corpus.builder.CorpusRecord``)
# ---------------------------------------------------------------------------


def test_ids_are_unique_well_formed_and_ascending(records: list[CorpusRecord]) -> None:
    """Every id is ``<class>-<3-digit index>`` for its own class, unique and ascending."""
    ids = [record.id for record in records]
    assert len(set(ids)) == len(ids), "record ids must be unique"
    assert ids == sorted(ids), "the built corpus must be ordered by id ascending"
    for record in records:
        pattern = rf"{re.escape(record.class_name)}-\d{{3}}"
        assert re.fullmatch(pattern, record.id), (
            f"{record.id} is not '<class>-<3-digit index>' for class {record.class_name}"
        )
        assert record.id.startswith(f"{record.class_name}-")


def test_each_record_states_exactly_one_payload_field(records: list[CorpusRecord]) -> None:
    """``payload_utf8`` and ``payload_b64`` are mutually exclusive and one is always set."""
    for record in records:
        assert (record.payload_utf8 is None) != (record.payload_b64 is None), (
            f"{record.id}: exactly one of payload_utf8 and payload_b64 must be set"
        )
        assert record.payload_bytes, f"{record.id}: the payload must not be empty"
    assert sorted({record.class_name for record in records if record.payload_b64 is not None}) == (
        list(BASE64_CLASSES)
    )
    assert sorted({record.class_name for record in records if record.payload_utf8 is not None}) == [
        name for name in builder.CLASS_NAMES if name not in BASE64_CLASSES
    ]


def test_expected_mode_is_in_the_mode_domain_and_derived_from_the_bytes(
    records: list[CorpusRecord]
) -> None:
    """``expected_mode`` is one of the three modes and agrees with the payload's own bytes.
    ``MD-MODE-OPTIMAL`` asserts a target against this field, so a record whose claim disagreed
    with its payload would make that check assert the wrong thing.
    """
    for record in records:
        assert record.expected_mode in capacity.MODES, (
            f"{record.id}: expected_mode {record.expected_mode!r} is not one of {capacity.MODES}"
        )
        assert record.expected_mode == payloads.mode_of(record.payload_bytes), (
            f"{record.id}: expected_mode {record.expected_mode!r} disagrees with the mode of "
            f"its payload bytes"
        )


def test_every_record_carries_a_notes_description(records: list[CorpusRecord]) -> None:
    """``notes`` is always a non-empty str, and a content record names its class and length."""
    for record in records:
        assert isinstance(record.notes, str)
        assert record.notes.strip(), f"{record.id}: notes must not be empty"
        if record.class_name in builder.BOUNDARY_CLASSES:
            assert record.notes.startswith("max_payload_bytes("), (
                f"{record.id}: a boundary record's notes must state its capacity arithmetic"
            )
            continue
        match = CONTENT_NOTES.fullmatch(record.notes)
        assert match, f"{record.id}: notes are not the content form: {record.notes!r}"
        assert match["class_name"] == record.class_name
        assert int(match["index"]) == int(record.id.rsplit("-", 1)[1])
        assert match["shape"].strip(), f"{record.id}: notes do not describe the payload shape"
        assert int(match["length"]) == len(record.payload_bytes)


def test_boundary_notes_recompute_through_the_capacity_tables(
    records: list[CorpusRecord]
) -> None:
    """Each boundary record's stated arithmetic reproduces through ``qrcheck.corpus.capacity``.
    The notes are not prose to be read: they are the length, version and level claims the
    boundary subset exists for, so every number in them is recomputed here.
    """
    boundary = [
        record for record in records if record.class_name in builder.BOUNDARY_CLASSES
    ]
    assert len(boundary) == EXPECTED_BOUNDARY_RECORDS
    for record in boundary:
        match = BOUNDARY_NOTES.fullmatch(record.notes)
        assert match, f"{record.id}: notes are not the boundary form: {record.notes!r}"
        version = int(match["version"])
        mode = match["mode"]
        length = int(match["length"])
        assert mode == record.expected_mode
        assert capacity.max_payload_bytes(version, "L", mode) == length, (
            f"{record.id}: notes claim capacity {length} for Version {version} at EC L"
        )
        assert len(record.payload_bytes) == length, (
            f"{record.id}: payload holds {len(record.payload_bytes)} bytes, notes claim {length}"
        )
        assert capacity.minimal_version(length, "L", mode) == int(match["l_version"])
        assert capacity.minimal_version(length, "M", mode) == int(match["m_version"])
        assert capacity.minimal_version(length, "H", mode) == int(match["h_version"])


# ---------------------------------------------------------------------------
# Determinism: the corpus is a pure function of the seed
# ---------------------------------------------------------------------------


def test_building_twice_with_one_seed_returns_equal_records(
    records: list[CorpusRecord]
) -> None:
    """Two builds from the same seed agree record for record, including the payload bytes. A
    builder that iterated a set or drew from ``hash()`` would differ here and nowhere else.
    """
    first = build_corpus(TEST_SEED)
    second = build_corpus(TEST_SEED)
    assert first == second
    assert first == records
    assert [record.id for record in first] == [record.id for record in second]
    assert [record.payload_bytes for record in first] == [
        record.payload_bytes for record in second
    ]


def test_a_different_seed_selects_different_payloads(records: list[CorpusRecord]) -> None:
    """The seed reaches the payloads, so the build-twice comparison above is not vacuous."""
    other = build_corpus(OTHER_SEED)
    assert [record.id for record in other] == [record.id for record in records]
    assert Counter(record.class_name for record in other) == builder.CLASS_COUNTS
    differing = [
        left.id
        for left, right in zip(records, other)
        if left.payload_bytes != right.payload_bytes
    ]
    assert differing, "two seeds produced identical payloads for every record"


# ---------------------------------------------------------------------------
# Class filtering (``filter_by_class``; the empty result is the runner's exit-2 condition, so
# this module asserts the fact and not an exception)
# ---------------------------------------------------------------------------


def test_filter_selects_the_boundary_subset(records: list[CorpusRecord]) -> None:
    """``boundary-*`` selects exactly the 40 boundary records, in ``id`` order. It is the same view
    the L, Q and H runs are driven over, so the glob and the count are asserted together: the count
    alone would not show which classes the glob reaches.
    """
    selected = filter_by_class(records, ("boundary-*",))
    assert isinstance(selected, tuple)
    assert len(selected) == builder.BOUNDARY_RECORDS == EXPECTED_BOUNDARY_RECORDS
    expected = [record for record in records if record.class_name in builder.BOUNDARY_CLASSES]
    assert list(selected) == expected
    assert {record.class_name for record in selected} == set(builder.BOUNDARY_CLASSES)
    assert [record.id for record in selected] == sorted(record.id for record in selected)


def test_filter_orders_its_result_independently_of_the_input_order(
    records: list[CorpusRecord]
) -> None:
    """Reversing the input cannot change the result, because the filter sorts by ``id``."""
    assert filter_by_class(list(reversed(records)), ("boundary-*",)) == filter_by_class(
        records, ("boundary-*",)
    )


def test_filter_globs_are_or_combined(records: list[CorpusRecord]) -> None:
    """Two class globs select the union: ``url`` alone yields 30 records, ``otp`` 12, both 42."""
    selected = filter_by_class(records, ("url", "otp"))
    assert len(selected) == EXPECTED_CLASS_COUNTS["url"] + EXPECTED_CLASS_COUNTS["otp"] == 42
    assert {record.class_name for record in selected} == {"url", "otp"}
    assert set(selected) == set(filter_by_class(records, ("url",))) | set(
        filter_by_class(records, ("otp",))
    )
    assert len(selected) == len(set(selected))


def test_filter_with_no_globs_keeps_every_record(records: list[CorpusRecord]) -> None:
    """No filter, an empty sequence and a bare string behave as documented. A bare string is one
    pattern rather than one pattern per character, which is why it must agree with the
    one-element tuple.
    """
    assert filter_by_class(records) == tuple(records)
    assert filter_by_class(records, ()) == tuple(records)
    assert filter_by_class(records, None) == tuple(records)
    assert filter_by_class(records, "boundary-*") == filter_by_class(records, ("boundary-*",))


def test_filter_returns_an_empty_tuple_when_nothing_matches(
    records: list[CorpusRecord]
) -> None:
    """A glob that matches nothing yields ``()``; raising belongs to the runner, not here."""
    selected = filter_by_class(records, ("no-such-class-*",))
    assert selected == ()
    assert isinstance(selected, tuple)


def test_filter_matching_is_case_sensitive(records: list[CorpusRecord]) -> None:
    """Class matching is ``fnmatchcase``, so a differently-cased glob matches nothing."""
    assert filter_by_class(records, ("URL",)) == ()
    assert filter_by_class(records, ("Boundary-*",)) == ()
    assert filter_by_class(records, ("ur*",)) == filter_by_class(records, ("url",))


# The six MD-CHARSET coverages, mapped onto ``builder.MD_CHARSET_CLASSES`` below. Each coverage is
# a predicate over the payload's own decoded bytes rather than a class-name lookup: the point is
# that the corpus carries a payload that really holds the thing the check exercises, so a class
# that stopped emitting it would fail here.

# The code-point boundaries the coverage predicates read, as half-open intervals. The emoji
# interval is the range ``qrcheck/corpus/payloads.py`` draws from, whose every member encodes
# to four UTF-8 bytes; the right-to-left interval covers the Hebrew and Arabic blocks; and a
# code point at or above U+10000 is the definition of a four-byte UTF-8 sequence.
EMOJI_RANGE: Final[tuple[int, int]] = (0x1F300, 0x1FB00)
RIGHT_TO_LEFT_RANGE: Final[tuple[int, int]] = (0x0590, 0x0700)
FOUR_BYTE_FIRST: Final[int] = 0x10000


def _code_points(payload: bytes) -> list[int]:
    """Return the Unicode code points ``payload`` holds, decoded as strict UTF-8. Called only on
    records of the text classes, whose payloads are valid UTF-8 by construction, so a decode
    error here is a corpus defect rather than a case to tolerate.
    """
    return [ord(character) for character in payload.decode("utf-8")]


def _is_ascii(payload: bytes) -> bool:
    """Return whether every byte of the payload is ASCII."""
    return payload.isascii()


def _has_multibyte_sequence(payload: bytes) -> bool:
    """Return whether any byte of the payload is outside ASCII."""
    return any(byte >= 0x80 for byte in payload)


def _has_emoji(payload: bytes) -> bool:
    """Return whether the payload holds a code point from the emoji range."""
    low, high = EMOJI_RANGE
    return any(low <= code_point < high for code_point in _code_points(payload))


def _has_right_to_left(payload: bytes) -> bool:
    """Return whether the payload holds a Hebrew or Arabic code point."""
    low, high = RIGHT_TO_LEFT_RANGE
    return any(low <= code_point < high for code_point in _code_points(payload))


def _has_embedded_nul(payload: bytes) -> bool:
    """Return whether the payload holds a NUL byte."""
    return b"\x00" in payload


def _has_four_byte_code_point(payload: bytes) -> bool:
    """Return whether the payload holds a code point that encodes to four UTF-8 bytes."""
    return any(code_point >= FOUR_BYTE_FIRST for code_point in _code_points(payload))


# The coverage-to-class mapping, one row per (coverage, class) pair ``CHARSET_COVERAGES`` holds:
# ASCII from ``otp`` and ``boundary-alphanumeric``; four-byte code points from ``emoji`` and
# from the widest ``utf8-multibyte`` records.
CHARSET_COVERAGES: Final[tuple[tuple[str, str, Callable[[bytes], bool]], ...]] = (
    ("ASCII", "boundary-alphanumeric", _is_ascii),
    ("ASCII", "otp", _is_ascii),
    ("embedded NUL", "nul-byte", _has_embedded_nul),
    ("emoji", "emoji", _has_emoji),
    ("four-byte code point", "emoji", _has_four_byte_code_point),
    ("four-byte code point", "utf8-multibyte", _has_four_byte_code_point),
    ("right-to-left", "rtl", _has_right_to_left),
    ("UTF-8 multibyte", "utf8-multibyte", _has_multibyte_sequence),
)


def test_the_six_documented_coverages_are_all_represented() -> None:
    """The mapping covers exactly the six coverages the charset check asserts."""
    assert {coverage for coverage, _, _ in CHARSET_COVERAGES} == {
        "ASCII",
        "UTF-8 multibyte",
        "emoji",
        "right-to-left",
        "embedded NUL",
        "four-byte code point",
    }
    assert {class_name for _, class_name, _ in CHARSET_COVERAGES} == set(
        EXPECTED_MD_CHARSET_CLASSES
    )


@pytest.mark.parametrize(("coverage", "class_name", "predicate"), CHARSET_COVERAGES)
def test_each_charset_coverage_resolves_to_a_record(
    records: list[CorpusRecord],
    coverage: str,
    class_name: str,
    predicate: Callable[[bytes], bool],
) -> None:
    """At least one record of the mapped class carries the coverage in its payload bytes."""
    matching = [
        record.id
        for record in records
        if record.class_name == class_name and predicate(record.payload_bytes)
    ]
    assert matching, f"no {class_name} record covers the {coverage} coverage"


# ---------------------------------------------------------------------------
# The discriminating predicates: the defective target's injections must be observable
# ---------------------------------------------------------------------------


def _truncation_witnesses(records: list[CorpusRecord], ec: str) -> list[str]:
    """Return the ids of records whose cut to the window keeps their Version at ``ec``."""
    window_version = capacity.minimal_version(TRUNCATION_BYTES, ec, "byte")
    return [
        record.id
        for record in records
        if record.class_name in builder.MD_CHARSET_CLASSES
        and len(record.payload_bytes) > TRUNCATION_BYTES
        and capacity.minimal_version(len(record.payload_bytes), ec, "byte") == window_version
    ]


def test_the_truncation_witness_exists_at_every_possible_level(
    records: list[CorpusRecord]
) -> None:
    """A cut to 64 bytes is observable at every level where the arithmetic allows it. A level is
    impossible when the window is itself the byte capacity of a Version there -- at EC H the
    64-byte window fills Version 7, so the next byte already needs Version 8 and no payload
    longer than the window can keep its version.
    """
    possible = [
        ec
        for ec in capacity.EC_LEVELS
        if capacity.truncation_keeps_version(TRUNCATION_BYTES + 1, TRUNCATION_BYTES, ec)
    ]
    assert possible, "no error-correction level can witness the truncation at all"
    for ec in possible:
        witnesses = _truncation_witnesses(records, ec)
        assert witnesses, (
            f"no MD-CHARSET record longer than {TRUNCATION_BYTES} bytes keeps its Version when "
            f"cut to {TRUNCATION_BYTES} bytes at EC {ec}, so the truncated_payload defect would "
            f"be inert there"
        )
    assert _truncation_witnesses(records, "M"), (
        "EC M is the run's default level, so the truncating defect must be observable there"
    )


def test_the_truncation_witnesses_are_longer_than_the_window(
    records: list[CorpusRecord]
) -> None:
    """Each witness is an ``MD-CHARSET`` record the injection can actually reach."""
    by_id = {record.id: record for record in records}
    witnesses = _truncation_witnesses(records, "M")
    assert witnesses
    for record_id in witnesses:
        record = by_id[record_id]
        assert record.class_name in builder.MD_CHARSET_CLASSES
        assert len(record.payload_bytes) > TRUNCATION_BYTES
        assert capacity.truncation_keeps_version(
            len(record.payload_bytes), TRUNCATION_BYTES, "M"
        )


def test_the_truncation_predicate_is_defined_for_every_long_record(
    records: list[CorpusRecord]
) -> None:
    """Every payload over the window fits a Version at EC M, so the predicate is total. A record
    too large for Version 40 would make ``minimal_version`` answer ``None``, and two overflows
    are never a version match.
    """
    longer = [record for record in records if len(record.payload_bytes) > TRUNCATION_BYTES]
    assert longer, f"no corpus record is longer than the {TRUNCATION_BYTES}-byte window"
    for record in longer:
        assert capacity.minimal_version(len(record.payload_bytes), "M", "byte") is not None


def test_each_boundary_class_separates_ec_l_from_ec_m_by_version(
    records: list[CorpusRecord]
) -> None:
    """A boundary payload needs a different minimal version at EC L and at EC M. That one-step
    difference is what makes the ``wrong_ec_level`` defect fail ``SY-VERSION-MIN`` as well as
    ``SY-EC-MATCH``, so the witness is asserted per boundary class: a class whose payloads stopped
    being boundary-length would leave that half of the expected failure set unproducible.
    """
    by_id = {record.id: record for record in records}
    for class_name in builder.BOUNDARY_CLASSES:
        members = [record for record in records if record.class_name == class_name]
        assert len(members) == builder.CLASS_COUNTS[class_name]
        witnesses = [record.id for record in members if _versions_differ(record)]
        assert witnesses, (
            f"no {class_name} record needs a different minimal version at EC L and EC M, so the "
            f"wrong_ec_level defect could not fail SY-VERSION-MIN"
        )
        for record_id in witnesses:
            record = by_id[record_id]
            mode = payloads.mode_of(record.payload_bytes)
            l_version = capacity.minimal_version(len(record.payload_bytes), "L", mode)
            m_version = capacity.minimal_version(len(record.payload_bytes), "M", mode)
            assert l_version is not None and m_version is not None
            assert l_version < m_version, (
                f"{record_id}: EC L needs Version {l_version}, EC M needs {m_version}"
            )


def _versions_differ(record: CorpusRecord) -> bool:
    """Return whether the record's minimal version differs between EC L and EC M."""
    mode = payloads.mode_of(record.payload_bytes)
    length = len(record.payload_bytes)
    return capacity.minimal_version(length, "L", mode) != capacity.minimal_version(
        length, "M", mode
    )


# ---------------------------------------------------------------------------
# Malformed input (``load_corpus``: a diagnostic naming the 1-based line number)
# ---------------------------------------------------------------------------


def _canonical_lines(corpus_path: Path) -> list[str]:
    """Return the written corpus's lines without their terminators. The malformed files below are
    edits of these exact lines, so no second renderer enters the module: the only writer under
    test is ``qrcheck.corpus.io.write_corpus``.
    """
    return corpus_path.read_text(encoding="utf-8").split("\n")[:-1]


def _write_lines(tmp_path: Path, lines: list[str]) -> Path:
    """Join ``lines`` with LF terminators into a file under ``tmp_path`` and return its path."""
    path = tmp_path / "malformed.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_corpus_error_is_a_value_error() -> None:
    """``builder.CorpusError`` is a ``ValueError``, so a caller needs no import to handle it."""
    assert issubclass(builder.CorpusError, ValueError)


def test_load_rejects_a_blank_line_and_names_its_number(corpus_path: Path, tmp_path: Path) -> None:
    """A blank line is rejected by its 1-based line number, not skipped."""
    lines = _canonical_lines(corpus_path)
    path = _write_lines(tmp_path, [lines[0], lines[1], "", lines[2]])
    with pytest.raises(ValueError) as excinfo:
        load_corpus(path)
    assert "line 3" in str(excinfo.value), str(excinfo.value)


def test_load_rejects_a_line_that_is_not_json(corpus_path: Path, tmp_path: Path) -> None:
    """A line that does not parse as JSON is rejected by its line number."""
    lines = _canonical_lines(corpus_path)
    path = _write_lines(tmp_path, [lines[0], "not json at all", lines[1]])
    with pytest.raises(ValueError) as excinfo:
        load_corpus(path)
    message = str(excinfo.value)
    assert "line 2" in message, message
    assert "not valid JSON" in message, message


def test_load_rejects_a_record_missing_a_required_key(
    corpus_path: Path, tmp_path: Path
) -> None:
    """A record without ``expected_mode`` is rejected, naming the line and the missing key."""
    lines = _canonical_lines(corpus_path)
    parsed = json.loads(lines[0])
    del parsed["expected_mode"]
    path = _write_lines(
        tmp_path, [lines[1], json.dumps(parsed, sort_keys=True, ensure_ascii=False)]
    )
    with pytest.raises(ValueError) as excinfo:
        load_corpus(path)
    message = str(excinfo.value)
    assert "line 2" in message, message
    assert "expected_mode" in message, message
