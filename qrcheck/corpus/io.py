"""The corpus file format: canonical JSONL read, write and class filtering.

``corpus/default.jsonl`` is a committed repository file, and AAP §0.5.4 promises it is
**byte-identical** when regenerated from ``--seed`` so the CI drift check can compare a fresh
build against the committed one instead of overwriting it. That promise is kept here and
nowhere else: one record per line, keys sorted at every depth, non-ASCII characters written
literally as UTF-8, ``\\n`` line endings and exactly one trailing newline. A second formatter
anywhere in the package would let two corpus files disagree about the same records.

:func:`write_corpus` renders records to that text in the order the caller supplies --
``qrcheck.corpus.builder.build_corpus`` is what guarantees ``id`` ascending, so nothing here
re-sorts a corpus behind a caller's back; :func:`load_corpus` reads it back so that
``load_corpus(p)`` after ``write_corpus(records, p)`` returns a list equal to ``records``;
:func:`filter_by_class` applies the ``--class`` globs that select a subset, such as the
40-record boundary subset success criterion 2 runs at EC L, Q and H. An **empty** record
sequence writes an **empty** (zero-byte) file, because a corpus with no records has no last
line to terminate, and an empty filter result is an empty tuple rather than an error,
because whether an empty selection is an error belongs to ``qrcheck.runner``.

No clock, no environment, no locale-dependent behaviour, no network and no module state: the
only effect of this module is the file it is handed. Standard library only, so the corpus can
be read and checked without an image stack; the single repository import is
``qrcheck.corpus.builder``, and the record type is never re-declared here.
"""

__all__ = ["filter_by_class", "load_corpus", "write_corpus"]

import json
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from qrcheck.corpus.builder import CorpusError, CorpusRecord


def write_corpus(records: Sequence[CorpusRecord], path: str | Path) -> None:
    """Write ``records`` to ``path`` as canonical JSONL.

    One JSON object per line, in the order given: the caller supplies ``build_corpus``'s
    ``id``-sorted list and this function preserves it, so file order is the caller's decision.
    Each line is ``json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False)`` with the
    default ``", "``/``": "`` separators and no indent -- sorted keys so a diff shows a value
    change rather than a reordering, literal UTF-8 so the ``emoji``, ``rtl`` and
    ``utf8-multibyte`` payloads stay readable, one object per line so the format stays JSONL.

    The file is UTF-8 with ``\\n`` line endings regardless of platform, every line ends with
    exactly one ``\\n``, and there is no other trailing whitespace. An **empty** ``records``
    sequence writes an **empty** (zero-byte) file: there is no last record, so there is no
    trailing newline to write.

    The text is rendered before the file is opened, so a record that cannot be serialised
    leaves no file at all, and the file is then written directly rather than through a
    temporary file and a rename -- ``CL-ARTIFACTS`` asserts the run leaves no temporary entry
    behind, and regeneration must be comparable with the committed corpus rather than swapped
    in atomically. The parent directory is created when it is missing.

    Args:
        records: The records to write, normally ``build_corpus()``'s return value; an empty
            sequence is legal and produces an empty file.
        path: Destination file, used exactly as given and never resolved to an absolute path.

    Returns:
        ``None``; the CLI is the single layer that reports what a regeneration did.

    Raises:
        TypeError: a record holds a value that is not JSON-serialisable, which is a caller
            defect rather than something to paper over with a ``default=`` hook.
        OSError: the parent directory cannot be created or the file cannot be written. It
            propagates untouched -- ``qrcheck.cli.corpus`` is the single place that turns an
            unwritable destination into an exit code, and this module decides none.
    """
    destination = Path(path)
    # A record that cannot be serialised must leave no artefact behind, so the whole text is
    # rendered before the filesystem is touched.
    text = "".join(_render(record) + "\n" for record in records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load_corpus(path: str | Path) -> list[CorpusRecord]:
    """Read the canonical JSONL file at ``path`` into records, in file order.

    The inverse of :func:`write_corpus`: for any record list ``records``, ``load_corpus(p)``
    after ``write_corpus(records, p)`` returns a ``list`` equal to ``records``. The result is
    a ``list`` in **file order**, not re-sorted, so the order a corpus file states is the
    order it is read back in and the comparison stays type-identical.

    Line endings are tolerated rather than trusted: the text is read as UTF-8 and each line's
    terminator is stripped, so a hand-edited ``\\r\\n`` file reads the same as a written one
    while the writer still emits only ``\\n``. The single empty segment produced by the final
    newline is ignored; an empty file is an empty corpus.

    Nothing is dropped silently: a blank line anywhere else, a line that is not JSON, a line
    that is not a JSON object and a line ``CorpusRecord.from_dict`` rejects all raise, because
    a corpus that quietly lost a malformed record would make the drift check and ``--class``
    filtering report on a different corpus than the one on disk.

    Args:
        path: The corpus file to read, used exactly as given.

    Returns:
        The records the file holds, in file order.

    Raises:
        CorpusError: a line is malformed. The message names the 1-based line number, the
            record ``id`` when the line carries a usable one, and the underlying reason.
            ``CorpusError`` is a ``ValueError`` subclass, the type callers already handle.
        OSError: the file is missing or unreadable. ``FileNotFoundError`` in particular
            propagates untouched: "the corpus file does not exist" is a fact for
            ``qrcheck.runner`` to turn into exit 2, not a decision taken here.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        # The one empty segment the final newline produces; a second one is a blank line.
        lines.pop()
    return [
        _parse_line(line.rstrip("\r\n"), number, source)
        for number, line in enumerate(lines, start=1)
    ]


def filter_by_class(
    records: Sequence[CorpusRecord], globs: Sequence[str] | None = None
) -> tuple[CorpusRecord, ...]:
    """Return the records whose class matches any of ``globs``, ordered by ``id``.

    ``globs`` are shell-style patterns matched against each record's class name with
    :func:`fnmatch.fnmatchcase`, so matching is case-sensitive and locale-independent: an
    ``"URL"`` glob never selects the ``url`` class, whatever the process locale is. Multiple
    globs are a union -- a record matching any pattern is kept once -- and ``"boundary-*"``
    selects exactly the three boundary classes, which is how the 40-record subset success
    criterion 2 runs at EC L, Q and H is invoked. An empty ``globs`` sequence, or ``None``,
    means "no filter" and returns every record; a bare string is one pattern, not one pattern
    per character.

    The result is a ``tuple`` sorted by ``id`` ascending, so the output is stable regardless
    of the order ``records`` arrived in -- which is what lets a report iterate it. A filter
    that matches nothing returns an empty tuple rather than raising: reporting an empty
    selection as a configuration error naming the globs is ``qrcheck.runner``'s job, and this
    module neither raises that condition nor prints anything.

    Args:
        records: The records to filter, normally a loaded or freshly built corpus.
        globs: Class-name patterns to keep; empty or ``None`` keeps every record.

    Returns:
        The matching records, sorted by ``id`` ascending; ``()`` when nothing matched.
    """
    patterns = (globs,) if isinstance(globs, str) else tuple(globs or ())
    if patterns:
        selected = [
            record
            for record in records
            if any(fnmatchcase(record.class_name, pattern) for pattern in patterns)
        ]
    else:
        selected = list(records)
    # Ordered by id, so the result does not depend on the order the caller's records arrived
    # in; the id is the only ordering key the corpus defines (AAP §0.5.3).
    return tuple(sorted(selected, key=lambda record: record.id))


def _render(record: CorpusRecord) -> str:
    """Return one record's canonical line, without its terminating newline.

    ``sort_keys`` states the key order, ``ensure_ascii=False`` keeps non-ASCII payloads
    literal, and no ``indent`` keeps the record on one line -- the three clauses that make a
    regenerated corpus byte-identical to the committed one.
    """
    return json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False)


def _parse_line(line: str, number: int, source: Path) -> CorpusRecord:
    """Return the record ``line`` holds, or raise CorpusError naming line ``number``.

    ``number`` is the 1-based line number in ``source``. The three failure modes are kept
    apart in the message -- blank line, not JSON, not a JSON object -- and a record rejected
    by ``CorpusRecord.from_dict`` additionally names the record ``id`` when the line carries
    one, because that is what a reader needs to find the offending record in a 240-line file.
    """
    where = f"{source}: line {number}"
    if not line.strip():
        raise CorpusError(f"{where}: blank line; every corpus line holds exactly one JSON object")
    try:
        raw = json.loads(line)
    except ValueError as exc:
        raise CorpusError(f"{where}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise CorpusError(
            f"{where}: expected a JSON object holding one record, got {type(raw).__name__}"
        )
    try:
        return CorpusRecord.from_dict(raw)
    except ValueError as exc:
        record_id = raw.get("id")
        subject = "" if not isinstance(record_id, str) else f" {record_id!r}"
        raise CorpusError(f"{where}: invalid record{subject}: {exc}") from exc
