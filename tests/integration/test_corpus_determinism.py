"""The corpus drift gate: rebuild from the committed seed and compare bytes, never rewrite.

``corpus/default.jsonl`` is a committed repository file and the fixed input every
corpus-driven case is built from, so it is **byte-identical** when regenerated from ``--seed``,
and the CI drift step re-runs exactly that. This module is the executable form of that step: it
drives the real command line, builds into ``tmp_path`` and compares **bytes**.

It also holds the committed file to its canonical form, the compare-then-fail ordering of
``corpus build`` (exit 2 unless ``--force``), the writer's canonical form in isolation, the seed
reaching generation in both directions, and the guard that nothing here writes the committed
corpus. No marker, skip, retry, monkeypatch or threshold appears."""

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

# The numbers, written out once: ``qrcheck.toml`` declares the default seed and the committed
# corpus holds the inventory, and the seed is also what the CI drift step rebuilds with, so a test
# using a different seed would disagree with CI about what drift means.

#: The configured default seed, ``qrcheck.toml``'s ``seed``: the corpus was built from it and CI
#: regenerates it with it, so a test using another would disagree with CI about what drift means.
DEFAULT_SEED: Final[int] = 20240101

#: A seed the committed bytes must **not** be reproducible from, or a seed-ignoring builder passes.
OTHER_SEED: Final[int] = 7

#: The committed corpus, relative to the repository root.
COMMITTED_CORPUS_RELPATH: Final[Path] = Path("corpus") / "default.jsonl"

#: The class-name prefix of the three classes making the 40-record boundary subset.
BOUNDARY_CLASS_PREFIX: Final[str] = "boundary-"

#: 240 records in total, of which the boundary subset is 40.
EXPECTED_TOTAL_RECORDS: Final[int] = 240
EXPECTED_BOUNDARY_RECORDS: Final[int] = 40

#: The committed corpus's 14-class inventory, compared as a mapping so moving records between
#: classes fails instead of passing on an unchanged total.
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

#: The three values a record's ``expected_mode`` may hold.
EXPECTED_MODES: Final[frozenset[str]] = frozenset({"numeric", "alphanumeric", "byte"})

#: An id is ``<class>-<3-digit index>``, so the captured group must equal the class.
RECORD_ID: Final[re.Pattern[str]] = re.compile(
    r"(?P<class_name>[a-z0-9][a-z0-9-]*)-(?P<index>\d{3})"
)

#: A boundary record's ``notes`` states the capacity arithmetic that produced its length, exactly
#: as ``qrcheck/corpus/builder.py`` writes it, e.g.
#: ``max_payload_bytes(2, 'L', 'byte') = 32; minimal version L=2, M=3, H=4``; the same pattern
#: shows a non-boundary record claims no such note.
CAPACITY_ARITHMETIC: Final[re.Pattern[str]] = re.compile(
    r"max_payload_bytes\(\d{1,2}, '[LMQH]', '(?:numeric|alphanumeric|byte)'\) = \d+; "
    r"minimal version L=\d{1,2}, M=\d{1,2}, H=\d{1,2}"
)

#: How many bytes of context a byte-difference diagnostic quotes, keeping the message readable.
_DIAGNOSTIC_BYTES: Final[int] = 40


# Byte comparison and canonical-line helpers: a byte-identity assertion whose message says only
# "not equal" costs the next reader an hour, so every comparison reports the first differing offset,
# the line it falls on in each rendering, both lengths and a window of context from each side.


def _line_number(data: bytes, offset: int) -> int:
    """Return the 1-based line number of ``offset``
    within ``data``, which may be one past the end."""
    return data.count(b"\n", 0, min(offset, len(data))) + 1


def _difference(expected: bytes, actual: bytes, what: str) -> str:
    """Describe where two renderings first diverge: the offset, both lengths and a window of
    context."""
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
    """Return the record on corpus line ``number``, asserting that line's canonical form."""
    assert line == line.rstrip(), (
        f"corpus line {number} ends with whitespace: {line[-20:]!r}; trailing space is not "
        f"canonical and would make a regenerated corpus differ from the committed one"
    )
    # object_pairs_hook=list preserves the order the line states, which json.loads would
    # otherwise discard: sorted keys are a property of the text, not of the parsed mapping.
    pairs = json.loads(line, object_pairs_hook=list)
    keys = [key for key, _ in pairs]
    assert keys == sorted(keys), (
        f"corpus line {number} does not hold its keys in sorted order: {keys}; the canonical "
        f"form requires sorted keys in every JSON artefact, and a reordering would make the file "
        f"fail the byte comparison even with identical records"
    )
    return dict(pairs)


# Fixtures: every path a test writes to is under ``tmp_path``, and the configuration comes from the
# integration layer's writer, so it carries the declared default for everything not chosen here.


@pytest.fixture(scope="module")
def committed_corpus_path(repo_root: Path) -> Path:
    """Return the committed corpus's path, derived from :func:`repo_root`."""
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
    """Return a schema-valid configuration whose every path lives under ``tmp_path``."""
    return write_config(
        tmp_path / "qrcheck.toml",
        default_adapter="reference",
        report_dir=tmp_path / "reports",
        corpus_path=committed_corpus_path,
        seed=DEFAULT_SEED,
    )


@pytest.fixture
def differing_corpus(tmp_path: Path, committed_corpus_path: Path) -> Path:
    """Return a destination holding valid-looking content that is nevertheless not the corpus."""
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
    """Guard the committed corpus for the whole module, exposing its opening fingerprint."""
    before = _fingerprint(committed_corpus_path)
    yield before
    after = _fingerprint(committed_corpus_path)
    assert after == before, (
        f"the committed corpus changed while this module ran: (size, sha256) was {before} and "
        f"is now {after}. Every test here writes under tmp_path; a regeneration into "
        f"{committed_corpus_path.parent} would corrupt the drift baseline."
    )


# The drift gate


def test_build_reproduces_committed_corpus_byte_for_byte(
    tmp_path: Path,
    committed_corpus_path: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """``corpus build --seed 20240101`` must equal the committed file byte for byte."""
    built = tmp_path / "corpus.jsonl"
    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(DEFAULT_SEED),
        "--out",
        str(built),
        config=corpus_config,
    )
    assert result.exit_code == 0, (
        f"corpus build --seed {DEFAULT_SEED} exited {result.exit_code}, not 0; "
        f"stdout {result.stdout!r}, stderr {result.stderr!r}"
    )
    assert built.is_file(), (
        f"corpus build exited 0 without writing {built}; stdout {result.stdout!r}, "
        f"stderr {result.stderr!r}"
    )
    _assert_same_bytes(
        built.read_bytes(),
        committed_corpus_path.read_bytes(),
        f"corpus build --seed {DEFAULT_SEED} against {COMMITTED_CORPUS_RELPATH.as_posix()}",
    )


def _assert_ids(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert the ids are unique, ascending in file order and consistent with their class."""
    ids = [str(record["id"]) for record in records]
    duplicated = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
    assert not duplicated, f"record id(s) appear more than once: {duplicated}"
    for previous, current in zip(ids, ids[1:]):
        assert previous < current, (
            f"record {current!r} follows {previous!r}, so the file is not in ascending id order; "
            f"file order is part of the canonical form, and a reordering fails the byte comparison "
            f"while every record stays the same"
        )
    for record in records:
        record_id = str(record["id"])
        match = RECORD_ID.fullmatch(record_id)
        assert match is not None, (
            f"record id {record_id!r} is not '<class>-<3-digit index>'"
        )
        assert match["class_name"] == record["class"], (
            f"record {record_id!r} states class {record['class']!r}, but its id names "
            f"{match['class_name']!r}"
        )


def _assert_inventory(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert the per-class counts and the boundary subset are the declared ones."""
    counts = Counter(str(record["class"]) for record in records)
    observed = dict(sorted(counts.items()))
    expected = dict(sorted(EXPECTED_CLASS_COUNTS.items()))
    assert observed == expected, (
        f"the corpus inventory drifted: expected {expected}, saw {observed}"
    )
    boundary = [
        record
        for record in records
        if str(record["class"]).startswith(BOUNDARY_CLASS_PREFIX)
    ]
    assert len(boundary) == EXPECTED_BOUNDARY_RECORDS, (
        f"the boundary subset holds {len(boundary)} record(s); it must hold "
        f"{EXPECTED_BOUNDARY_RECORDS}, which is the set the boundary runs replay at EC L, Q and H"
    )


def _assert_payload_fields(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert exactly one payload field per record, strictly decodable, with a declared mode."""
    for record in records:
        record_id = str(record["id"])
        has_text = "payload_utf8" in record
        has_base64 = "payload_b64" in record
        assert has_text != has_base64, (
            f"record {record_id!r} carries {'both' if has_text else 'neither'} of payload_utf8 "
            f"and payload_b64; the two are mutually exclusive"
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
                f"boundary record {record_id!r} carries no notes; it must state the "
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
    """The committed file must itself be in canonical form."""
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
    assert len(lines) == EXPECTED_TOTAL_RECORDS, (
        f"{committed_corpus_path} holds {len(lines)} record(s); the inventory is "
        f"{EXPECTED_TOTAL_RECORDS}"
    )
    records = [_canonical_record(line, number) for number, line in enumerate(lines, 1)]
    _assert_ids(records)
    _assert_inventory(records)
    _assert_payload_fields(records)
    _assert_notes(records)


# Compare then fail, never overwrite first


def test_build_refuses_to_overwrite_without_force(
    differing_corpus: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """A differing destination without ``--force`` must exit 2 and stay byte-identical."""
    before = differing_corpus.read_bytes()
    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(DEFAULT_SEED),
        "--out",
        str(differing_corpus),
        config=corpus_config,
    )
    assert result.exit_code == 2, (
        f"corpus build over a differing destination without --force exited "
        f"{result.exit_code}, not 2; stdout {result.stdout!r}, "
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
    """``--force`` must replace a differing destination with the committed bytes exactly."""
    result = run_cli(
        "corpus",
        "build",
        "--seed",
        str(DEFAULT_SEED),
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
        f"corpus build --seed {DEFAULT_SEED} --force",
    )


# The writer's canonical form on its own


def test_load_write_round_trip_is_byte_stable(
    tmp_path: Path, committed_corpus_path: Path
) -> None:
    """Reading and re-writing the committed corpus must reproduce it byte for byte."""
    records: list[CorpusRecord] = load_corpus(committed_corpus_path)
    assert len(records) == EXPECTED_TOTAL_RECORDS, (
        f"load_corpus read {len(records)} record(s) from {committed_corpus_path}; the inventory "
        f"is {EXPECTED_TOTAL_RECORDS}"
    )
    round_tripped = tmp_path / "roundtrip.jsonl"
    write_corpus(records, round_tripped)
    _assert_same_bytes(
        round_tripped.read_bytes(),
        committed_corpus_path.read_bytes(),
        "load_corpus then write_corpus",
    )


# The seed reaches generation


def test_seed_is_actually_wired_into_the_generated_content(
    tmp_path: Path,
    committed_corpus_path: Path,
    corpus_config: Path,
    run_cli: Callable[..., Any],
) -> None:
    """A different seed must produce different bytes; the committed seed the committed file."""
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
        str(DEFAULT_SEED),
        "--out",
        str(rebuilt),
        config=corpus_config,
    )
    assert rebuilt_result.exit_code == 0, (
        f"corpus build --seed {DEFAULT_SEED} exited {rebuilt_result.exit_code}, not 0; "
        f"stdout {rebuilt_result.stdout!r}, stderr {rebuilt_result.stderr!r}"
    )
    _assert_same_bytes(
        rebuilt.read_bytes(),
        committed,
        f"corpus build --seed {DEFAULT_SEED} after the {OTHER_SEED} build",
    )


# The guard: the suite never writes the baseline


def test_committed_corpus_is_never_modified_by_the_suite(
    committed_corpus_path: Path, corpus_fingerprint: tuple[int, str]
) -> None:
    """The committed corpus's size and digest must be what they were before the module ran."""
    current = _fingerprint(committed_corpus_path)
    assert current == corpus_fingerprint, (
        f"{committed_corpus_path} changed while this module ran: (size, sha256) was "
        f"{corpus_fingerprint} before the module and is {current} now. Every test here must "
        f"write under tmp_path."
    )
