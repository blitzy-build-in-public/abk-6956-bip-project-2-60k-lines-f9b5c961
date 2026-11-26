"""The corpus drift gate: rebuild from the committed seed and compare bytes, never rewrite.

``corpus/default.jsonl`` is a committed repository file and the fixed input every corpus-driven
case is built from, so AAP §0.5.4 promises it is **byte-identical** when regenerated from
``--seed``, and the §0.6.4 CI drift step re-runs exactly that regeneration, compares the fresh
rendering with the committed bytes and fails on any difference *before* anything is replaced.
This module is the executable form of that step: it drives the real command line, builds into
``tmp_path`` and compares **bytes**, so a change to a payload generator, to the class inventory
or to the writer's canonical form fails here instead of silently redefining the baseline every
check reads. It asserts the drift gate itself; the committed file held to the canonical form of
§0.5.3 and §0.5.4 -- UTF-8, LF endings, one trailing newline, sorted keys on every line, the
240-record 14-class inventory of §0.8.2 with its 40-record boundary subset, one payload field
per record and capacity arithmetic in every boundary record's ``notes``; the compare-then-fail
ordering of ``corpus build``, which refuses a differing destination with exit 2 and leaves it
untouched unless ``--force`` is given; the writer's canonical form in isolation; the seed
reaching generation in both directions, since a builder that ignored its seed could not produce
a different file; and the guard that nothing here writes the committed corpus, because a
rewritten baseline would make the drift check compare a file against itself.

It registers no marker -- it belongs to the fast job, generating and comparing text only -- and
skips nothing, retries nothing, monkeypatches nothing and carries no threshold. The seed and the
class inventory are the plan's own numbers, cited once below; every other value is read from the
committed file.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.corpus.builder import CorpusRecord
from qrcheck.corpus.io import load_corpus, write_corpus

# The plan's numbers, written out once. AAP §0.3.4 declares the configured default seed and
# §0.8.2 the corpus inventory, and the seed is also the value the §0.6.4 CI drift step
# rebuilds with, so the two must be the same number: a test using a different seed would
# disagree with CI about what drift means.

#: The configured default seed (AAP §0.3.4): the seed the committed corpus was built from and the
#: one the §0.6.4 CI drift step regenerates it with, so a test using another would disagree with CI
#: about what drift means.
PLAN_SEED: Final[int] = 20240101

#: A seed the committed bytes must **not** be reproducible from: two seeds producing the same file
#: would let a builder that ignored its seed pass every byte-identity assertion.
OTHER_SEED: Final[int] = 7

#: The committed corpus, relative to the repository root (AAP §0.9.2).
COMMITTED_CORPUS_RELPATH: Final[Path] = Path("corpus") / "default.jsonl"

#: The class-name prefix of the three classes that make the 40-record boundary subset (AAP §0.8.2),
#: which is the record set success criterion 2 runs at EC L, Q and H.
BOUNDARY_CLASS_PREFIX: Final[str] = "boundary-"

#: AAP §0.8.2: 240 records in total, of which the boundary subset is 40.
PLAN_TOTAL_RECORDS: Final[int] = 240
PLAN_BOUNDARY_RECORDS: Final[int] = 40

#: AAP §0.8.2's full 14-class inventory, compared as a whole mapping so a re-balance that moves
#: records between classes fails loudly instead of passing on a total that happens to be unchanged.
PLAN_CLASS_COUNTS: Final[dict[str, int]] = {
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

#: AAP §0.8.2: the three values a record's ``expected_mode`` may hold.
EXPECTED_MODES: Final[frozenset[str]] = frozenset({"numeric", "alphanumeric", "byte"})

#: AAP §0.8.2: a record id is ``<class>-<3-digit index>``, so the captured class group must equal
#: the record's own ``class`` field.
RECORD_ID: Final[re.Pattern[str]] = re.compile(
    r"(?P<class_name>[a-z0-9][a-z0-9-]*)-(?P<index>\d{3})"
)

#: AAP §0.8.2: a boundary record's ``notes`` states the capacity arithmetic that produced its
#: length, exactly as ``qrcheck/corpus/builder.py`` writes it, for instance
#: ``max_payload_bytes(2, 'L', 'byte') = 32; minimal version L=2, M=3, H=4``. The same pattern is
#: how a non-boundary record is shown **not** to claim a boundary note.
CAPACITY_ARITHMETIC: Final[re.Pattern[str]] = re.compile(
    r"max_payload_bytes\(\d{1,2}, '[LMQH]', '(?:numeric|alphanumeric|byte)'\) = \d+; "
    r"minimal version L=\d{1,2}, M=\d{1,2}, H=\d{1,2}"
)

#: How many bytes of context a byte-difference diagnostic quotes around the first difference, which
#: keeps the message readable while still showing what actually changed.
_DIAGNOSTIC_BYTES: Final[int] = 40


# Byte comparison and canonical-line helpers. A byte-identity assertion whose message says only
# "not equal" costs the next reader an hour, so every comparison here reports the first
# differing offset, the line it falls on in each rendering, both lengths and a window of context
# from each side.


def _line_number(data: bytes, offset: int) -> int:
    """Return the 1-based line number of ``offset`` within ``data``.

    ``offset`` may be one past the end, when one rendering is a prefix of the other.
    """
    return data.count(b"\n", 0, min(offset, len(data))) + 1


def _difference(expected: bytes, actual: bytes, what: str) -> str:
    """Describe where two renderings of the same corpus first diverge.

    Names the first differing byte offset, the line it falls on in each rendering, both lengths and
    a window of context from each side, so a byte-identity failure costs a reader no investigation.
    """
    shared = min(len(expected), len(actual))
    offset = next(
        (index for index in range(shared) if expected[index] != actual[index]), shared
    )
    window_start = max(0, offset - _DIAGNOSTIC_BYTES)
    window_end = offset + _DIAGNOSTIC_BYTES
    return (
        f"{what}: the renderings differ at byte offset {offset} (expected file line "
        f"{_line_number(expected, offset)}, actual file line {_line_number(actual, offset)}); "
        f"expected {len(expected)} bytes, saw {len(actual)}. From offset {window_start}: "
        f"expected {expected[window_start:window_end]!r}, "
        f"actual {actual[window_start:window_end]!r}"
    )


def _assert_same_bytes(actual: bytes, expected: bytes, what: str) -> None:
    """Assert two renderings are byte-identical, naming the first difference when they are not."""
    assert actual == expected, _difference(expected, actual, what)


def _canonical_record(line: str, number: int) -> dict[str, Any]:
    """Return the record on corpus line ``number``, asserting that line's canonical form.

    A line carrying trailing whitespace, holding no JSON object, or stating its keys out of order
    fails the test.
    """
    assert line == line.rstrip(), (
        f"corpus line {number} ends with whitespace: {line[-20:]!r}; trailing space is not "
        f"canonical and would make a regenerated corpus differ from the committed one"
    )
    # object_pairs_hook=list preserves the order the line states, which json.loads would
    # otherwise discard: sorted keys are a property of the text, not of the parsed mapping.
    pairs = json.loads(line, object_pairs_hook=list)
    keys = [key for key, _ in pairs]
    assert keys == sorted(keys), (
        f"corpus line {number} does not hold its keys in sorted order: {keys}; AAP §0.5.3 "
        f"requires sorted keys in every JSON artefact, and a reordering would make the file "
        f"fail the byte comparison even with identical records"
    )
    return dict(pairs)


# Fixtures. Every path a test writes to is under ``tmp_path``, and the configuration is
# generated by the integration layer's writer, so it carries the declared default for everything
# this module does not choose and no value is restated here.


@pytest.fixture(scope="module")
def committed_corpus_path(repo_root: Path) -> Path:
    """Return the committed corpus's path, derived from the repository root.

    Module-scoped because it cannot change within a module and every test compares against the same
    file, and derived from :func:`repo_root` so a session started anywhere compares the right
    baseline. A missing file fails the test: it is a defective checkout, not a reason to skip,
    because every assertion here compares against it.
    """
    path = repo_root / COMMITTED_CORPUS_RELPATH
    assert path.is_file(), (
        f"the committed corpus is missing: {path}. This module compares every regeneration "
        f"against it, so a checkout without it cannot hold the drift gate."
    )
    return path


@pytest.fixture
def corpus_config(
    tmp_path: Path,
    write_config: Callable[..., Path],
    committed_corpus_path: Path,
) -> Path:
    """Return a schema-valid configuration whose every path lives under ``tmp_path``.

    The report root is absolute and private to the test, so a build never writes into the
    repository's own ``reports/``.
    """
    return write_config(
        tmp_path / "qrcheck.toml",
        default_adapter="reference",
        report_dir=tmp_path / "reports",
        corpus_path=committed_corpus_path,
        seed=PLAN_SEED,
    )


@pytest.fixture
def differing_corpus(tmp_path: Path, committed_corpus_path: Path) -> Path:
    """Return a destination holding different but valid-looking content.

    The content is the committed corpus's first record alone: a well-formed line that is
    nevertheless not the corpus, so the comparison inside ``corpus build`` has a real difference to
    refuse or to replace.
    """
    first_record = committed_corpus_path.read_bytes().split(b"\n", 1)[0] + b"\n"
    destination = tmp_path / "corpus.jsonl"
    destination.write_bytes(first_record)
    return destination


def _fingerprint(path: Path) -> tuple[int, str]:
    """Return the byte length and SHA-256 digest of a file's contents, as an orderable pair."""
    data = path.read_bytes()
    return (len(data), hashlib.sha256(data).hexdigest())


@pytest.fixture(scope="module", autouse=True)
def corpus_fingerprint(committed_corpus_path: Path) -> Iterator[tuple[int, str]]:
    """Guard the committed corpus for the whole module and expose its opening fingerprint.

    Autouse and module-scoped, so the fingerprint is taken before this module's first test and
    re-checked after its last: a test that regenerated into the repository is then reported as a
    failure instead of being left to corrupt the baseline the drift step compares against on the
    next run. The opening value is yielded to
    :func:`test_committed_corpus_is_never_modified_by_the_suite`, which asserts the same thing from
    inside the run.
    """
    before = _fingerprint(committed_corpus_path)
    yield before
    after = _fingerprint(committed_corpus_path)
    assert after == before, (
        f"the committed corpus changed while this module ran: (size, sha256) was {before} and "
        f"is now {after}. Every test here writes under tmp_path; a regeneration into "
        f"{committed_corpus_path.parent} would corrupt the drift baseline."
    )


# ---------------------------------------------------------------------------
# The drift gate (AAP §0.5.4, §0.6.4)
# ---------------------------------------------------------------------------


def test_build_reproduces_committed_corpus_byte_for_byte(
    tmp_path: Path,
    committed_corpus_path: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """``corpus build --seed 20240101`` must equal the committed file byte for byte.

    Catches exactly the drift the §0.6.4 CI step exists to catch: a payload generator, a class
    count or the canonical writer changed while the committed corpus was not regenerated, so
    every comparison and every downstream check would read a baseline that nothing produces.
    """
    built = tmp_path / "corpus.jsonl"
    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(PLAN_SEED),
        "--out",
        str(built),
        config=corpus_config,
    )
    assert result.exit_code == 0, (
        f"corpus build --seed {PLAN_SEED} exited {result.exit_code}, not 0; "
        f"stdout {result.stdout!r}, stderr {result.stderr!r}"
    )
    assert built.is_file(), (
        f"corpus build exited 0 without writing {built}; stdout {result.stdout!r}, "
        f"stderr {result.stderr!r}"
    )
    _assert_same_bytes(
        built.read_bytes(),
        committed_corpus_path.read_bytes(),
        f"corpus build --seed {PLAN_SEED} against {COMMITTED_CORPUS_RELPATH.as_posix()}",
    )


def _assert_ids(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert the ids are unique, ascending in file order and consistent with their class."""
    ids = [str(record["id"]) for record in records]
    duplicated = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
    assert not duplicated, f"record id(s) appear more than once: {duplicated}"
    for previous, current in zip(ids, ids[1:]):
        assert previous < current, (
            f"record {current!r} follows {previous!r}, so the file is not in ascending id order; "
            f"AAP §0.5.3 makes file order a contract, and a reordering fails the byte comparison "
            f"while every record stays the same"
        )
    for record in records:
        record_id = str(record["id"])
        match = RECORD_ID.fullmatch(record_id)
        assert match is not None, (
            f"record id {record_id!r} is not '<class>-<3-digit index>' (AAP §0.8.2)"
        )
        assert match["class_name"] == record["class"], (
            f"record {record_id!r} states class {record['class']!r}, but its id names "
            f"{match['class_name']!r}"
        )


def _assert_inventory(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert the per-class counts and the boundary subset are the plan's."""
    counts = Counter(str(record["class"]) for record in records)
    observed = dict(sorted(counts.items()))
    expected = dict(sorted(PLAN_CLASS_COUNTS.items()))
    assert observed == expected, (
        f"the corpus inventory drifted: expected {expected}, saw {observed}"
    )
    boundary = [
        record
        for record in records
        if str(record["class"]).startswith(BOUNDARY_CLASS_PREFIX)
    ]
    assert len(boundary) == PLAN_BOUNDARY_RECORDS, (
        f"the boundary subset holds {len(boundary)} record(s); AAP §0.8.2 makes it "
        f"{PLAN_BOUNDARY_RECORDS}, which is the set success criterion 2 runs at EC L, Q and H"
    )


def _assert_payload_fields(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert exactly one payload field per record, strictly decodable, with a declared mode."""
    for record in records:
        record_id = str(record["id"])
        has_text = "payload_utf8" in record
        has_base64 = "payload_b64" in record
        assert has_text != has_base64, (
            f"record {record_id!r} carries {'both' if has_text else 'neither'} of payload_utf8 "
            f"and payload_b64; AAP §0.8.2 makes the two mutually exclusive"
        )
        if has_base64:
            encoded = str(record["payload_b64"])
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise AssertionError(
                    f"record {record_id!r} holds payload_b64 that is not strict base64: {exc}"
                ) from exc
            assert payload, f"record {record_id!r} holds a payload_b64 that decodes to nothing"
        else:
            assert str(record["payload_utf8"]), (
                f"record {record_id!r} holds an empty payload_utf8"
            )
        assert record["expected_mode"] in EXPECTED_MODES, (
            f"record {record_id!r} states expected_mode {record['expected_mode']!r}; the "
            f"allowed values are {sorted(EXPECTED_MODES)}"
        )


def _assert_notes(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert boundary records state the capacity arithmetic and no other record claims one."""
    for record in records:
        record_id = str(record["id"])
        notes = str(record["notes"])
        arithmetic = CAPACITY_ARITHMETIC.search(notes)
        if str(record["class"]).startswith(BOUNDARY_CLASS_PREFIX):
            assert notes.strip(), (
                f"boundary record {record_id!r} carries no notes; AAP §0.8.2 requires the "
                f"capacity arithmetic that produced its length"
            )
            assert arithmetic is not None, (
                f"boundary record {record_id!r} does not state the capacity arithmetic that "
                f"produced its length: {notes!r}"
            )
        else:
            assert arithmetic is None, (
                f"non-boundary record {record_id!r} states a boundary note: {notes!r}"
            )


def test_committed_corpus_is_canonically_formed(committed_corpus_path: Path) -> None:
    """The committed file must itself be the canonical corpus, read without the CLI.

    Catches a committed file that would pass the byte comparison only because both sides are wrong:
    CRLF endings, a lost final newline, unsorted keys, a record claiming two payload fields, a class
    inventory that no longer matches AAP §0.8.2, or a boundary record whose ``notes`` no longer
    states the capacity arithmetic that produced its length.
    """
    data = committed_corpus_path.read_bytes()
    assert b"\r" not in data, (
        f"{committed_corpus_path} contains a carriage return; the corpus is written with \\n "
        f"line endings on every platform"
    )
    text = data.decode("utf-8")
    assert text.endswith("\n"), (
        f"{committed_corpus_path} does not end with a newline; every corpus line, the last one "
        f"included, is terminated"
    )
    assert not text.endswith("\n\n"), (
        f"{committed_corpus_path} ends with a blank line rather than exactly one newline"
    )
    assert "\n\n" not in text, (
        f"{committed_corpus_path} holds a blank line; a blank line reads back as a malformed "
        f"record"
    )
    lines = text.split("\n")[:-1]
    assert len(lines) == PLAN_TOTAL_RECORDS, (
        f"{committed_corpus_path} holds {len(lines)} record(s); the plan's inventory is "
        f"{PLAN_TOTAL_RECORDS} (AAP §0.8.2)"
    )
    records = [_canonical_record(line, number) for number, line in enumerate(lines, 1)]
    _assert_ids(records)
    _assert_inventory(records)
    _assert_payload_fields(records)
    _assert_notes(records)


# ---------------------------------------------------------------------------
# Compare then fail, never overwrite first (AAP §0.6.4, §0.8.3)
# ---------------------------------------------------------------------------


def test_build_refuses_to_overwrite_without_force(
    differing_corpus: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """A differing destination without ``--force`` must exit 2 and stay byte-identical.

    Catches the drift step's ordering being inverted: a build that replaced the destination
    before comparing would destroy the baseline it is meant to report a difference against,
    turning every drift into a silent acceptance.
    """
    before = differing_corpus.read_bytes()
    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(PLAN_SEED),
        "--out",
        str(differing_corpus),
        config=corpus_config,
    )
    assert result.exit_code == 2, (
        f"corpus build over a differing destination without --force exited "
        f"{result.exit_code}, not 2 (AAP §0.3.4); stdout {result.stdout!r}, "
        f"stderr {result.stderr!r}"
    )
    assert "--force" in result.stderr, (
        f"the refusal does not name the option that would allow the overwrite: "
        f"stderr {result.stderr!r}"
    )
    assert differing_corpus.name in result.stderr, (
        f"the refusal does not name the destination it left alone ({differing_corpus.name}): "
        f"stderr {result.stderr!r}"
    )
    after = differing_corpus.read_bytes()
    assert after == before, (
        f"corpus build changed {differing_corpus} while refusing to overwrite it: the file "
        f"held {len(before)} bytes and now holds {len(after)}"
    )


def test_force_flag_overwrites_with_the_canonical_bytes(
    differing_corpus: Path,
    committed_corpus_path: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """``--force`` must replace a differing destination with the committed bytes exactly.

    Catches an overwrite path that installs a different rendering than the one compared -- a
    stale temporary file, a partial write, or a second formatter bypassing
    ``qrcheck.corpus.io`` -- which would leave the corpus unreproducible.
    """
    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(PLAN_SEED),
        "--out",
        str(differing_corpus),
        "--force",
        config=corpus_config,
    )
    assert result.exit_code == 0, (
        f"corpus build --force exited {result.exit_code}, not 0; stdout {result.stdout!r}, "
        f"stderr {result.stderr!r}"
    )
    _assert_same_bytes(
        differing_corpus.read_bytes(),
        committed_corpus_path.read_bytes(),
        f"corpus build --seed {PLAN_SEED} --force",
    )


# ---------------------------------------------------------------------------
# The writer's canonical form on its own (AAP §0.5.4)
# ---------------------------------------------------------------------------


def test_load_write_round_trip_is_byte_stable(
    tmp_path: Path, committed_corpus_path: Path
) -> None:
    """Reading and re-writing the committed corpus must reproduce it byte for byte.

    Catches the writer drifting from the text it wrote: a missing final newline, an unsorted
    key or an escaped non-ASCII character shows up here on its own, with no payload generator
    and no command line in the way, which is what makes the failing half identifiable.
    """
    records: list[CorpusRecord] = load_corpus(committed_corpus_path)
    assert len(records) == PLAN_TOTAL_RECORDS, (
        f"load_corpus read {len(records)} record(s) from {committed_corpus_path}; the plan's "
        f"inventory is {PLAN_TOTAL_RECORDS}"
    )
    round_tripped = tmp_path / "roundtrip.jsonl"
    write_corpus(records, round_tripped)
    _assert_same_bytes(
        round_tripped.read_bytes(),
        committed_corpus_path.read_bytes(),
        "load_corpus then write_corpus",
    )


# ---------------------------------------------------------------------------
# The seed reaches generation (AAP §0.5.1, §0.5.4)
# ---------------------------------------------------------------------------


def test_seed_is_actually_wired_into_the_generated_content(
    tmp_path: Path,
    committed_corpus_path: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """A different seed must produce different bytes; the committed seed the committed file.

    Catches a builder that accepts a seed and ignores it: such a builder satisfies every
    byte-identity assertion because it always produces the same corpus, while making the
    reproducibility promise of AAP §0.5.1 vacuous.
    """
    committed = committed_corpus_path.read_bytes()
    other = tmp_path / "seed-other.jsonl"
    other_result = run_cli(
        "corpus", "build", "--seed", str(OTHER_SEED), "--out", str(other), config=corpus_config
    )
    assert other_result.exit_code == 0, (
        f"corpus build --seed {OTHER_SEED} exited {other_result.exit_code}, not 0; "
        f"stdout {other_result.stdout!r}, stderr {other_result.stderr!r}"
    )
    assert other.read_bytes() != committed, (
        f"corpus build --seed {OTHER_SEED} reproduced {COMMITTED_CORPUS_RELPATH.as_posix()} "
        f"byte for byte, so the seed does not reach the generated payloads"
    )
    rebuilt = tmp_path / "seed-committed.jsonl"
    rebuilt_result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(PLAN_SEED),
        "--out",
        str(rebuilt),
        config=corpus_config,
    )
    assert rebuilt_result.exit_code == 0, (
        f"corpus build --seed {PLAN_SEED} exited {rebuilt_result.exit_code}, not 0; "
        f"stdout {rebuilt_result.stdout!r}, stderr {rebuilt_result.stderr!r}"
    )
    _assert_same_bytes(
        rebuilt.read_bytes(),
        committed,
        f"corpus build --seed {PLAN_SEED} after the {OTHER_SEED} build",
    )


# ---------------------------------------------------------------------------
# The guard: the suite never writes the baseline (AAP §0.11.1)
# ---------------------------------------------------------------------------


def test_committed_corpus_is_never_modified_by_the_suite(
    committed_corpus_path: Path, corpus_fingerprint: tuple[int, str]
) -> None:
    """The committed corpus's size and digest must be what they were before the module ran.

    Catches a test that regenerated into the repository -- by naming the committed path
    instead of one under ``tmp_path``, for instance -- which would rewrite the drift baseline
    and make the next run's byte comparison compare the file with itself. The autouse guard
    makes the same assertion again after the module's last test, so a late write cannot slip
    past this one.
    """
    current = _fingerprint(committed_corpus_path)
    assert current == corpus_fingerprint, (
        f"{committed_corpus_path} changed while this module ran: (size, sha256) was "
        f"{corpus_fingerprint} before the module and is {current} now. Every test here must "
        f"write under tmp_path."
    )
