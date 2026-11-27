"""The corpus file format: canonical JSONL read, write and class filtering.

``corpus/default.jsonl`` is a committed repository file, promised to be **byte-identical** when
regenerated from ``--seed`` so the CI drift check can compare a fresh build against the committed
one. That promise is kept here and nowhere else: one record per line, keys sorted at every depth,
non-ASCII characters written literally as UTF-8, `` `` line endings and exactly one trailing
newline. A second formatter anywhere in the package would let two corpus files disagree about the
same records.

:func:`write_corpus` renders records in the order the caller supplies --
``qrcheck.corpus.builder.build_corpus`` guarantees ``id`` ascending, so nothing here re-sorts;
:func:`load_corpus` reads it back, so ``load_corpus(p)`` after ``write_corpus(records, p)`` returns
a list equal to ``records``; :func:`filter_by_class` applies the ``--class`` globs, such as the
40-record boundary subset. An **empty** record sequence writes a zero-byte file, and an empty filter
result is an empty tuple rather than an error, because whether an empty selection is an error
belongs to ``qrcheck.runner``.

No clock, no environment, no locale-dependent behaviour and no module state: the only effect of this
module is the file it is handed. Standard library only.
"""

__all__ = ["filter_by_class", "load_corpus", "write_corpus"]

import json
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path

from qrcheck.corpus.builder import CorpusError, CorpusRecord


def write_corpus(records: Sequence[CorpusRecord], path: str | Path) -> None:
    """Write ``records`` to ``path`` as canonical JSONL.

    One JSON object per line, in the order given, each line ``json.dumps(record.to_dict(),
    sort_keys=True, ensure_ascii=False)`` with no indent -- sorted keys so a diff shows a value
    change rather than a reordering, literal UTF-8 so non-ASCII payloads stay readable. Line endings
    are `` ``, and an empty ``records`` sequence writes a zero-byte file. The text is rendered
    before the file is opened, so a record that cannot be serialised leaves no file.

    Raises:
        TypeError: A record holds a value that is not JSON-serialisable -- a caller defect.
        OSError: The parent directory cannot be created or the file cannot be written.
    """
    destination = Path(path)
    # A record that cannot be serialised must leave no artefact, so the text is rendered first.
    text = "".join(_render(record) + "\n" for record in records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load_corpus(path: str | Path) -> list[CorpusRecord]:
    """Read the canonical JSONL file at ``path`` into records, in file order.

    The inverse of :func:`write_corpus`: ``load_corpus(p)`` after ``write_corpus(records, p)``
    returns a ``list`` equal to ``records``, in file order and not re-sorted. Nothing is dropped
    silently -- a blank line, a line that is not JSON or not a JSON object, and a line
    ``CorpusRecord.from_dict`` rejects all raise, because a corpus that quietly lost a malformed
    record would make the drift check report on a different corpus than the one on disk.

    Raises:
        CorpusError: A line is malformed; the message names the 1-based line number and the record
            ``id`` when the line carries one.
        OSError: The file is missing or unreadable; ``FileNotFoundError`` propagates untouched.
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

    The globs are shell-style patterns matched with :func:`fnmatch.fnmatchcase`, so matching is
    case-sensitive and locale-independent; an empty sequence or ``None`` means "no filter". The
    ``tuple`` is sorted by ``id`` ascending, so the output is stable whatever order ``records``
    arrived in, and a filter matching nothing returns an empty tuple.

    Returns: The matching records, sorted by ``id`` ascending; ``()`` when nothing matched.
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
    # Ordered by id, so the result does not depend on the caller's arrival order.
    return tuple(sorted(selected, key=lambda record: record.id))


def _render(record: CorpusRecord) -> str:
    """Return one record's canonical line, without its terminating newline.

    ``sort_keys`` states the key order, ``ensure_ascii=False`` keeps non-ASCII payloads literal and
    no ``indent`` keeps the record on one line -- the three clauses that make a regenerated corpus
    byte-identical to the committed one.
    """
    return json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False)


def _parse_line(line: str, number: int, source: Path) -> CorpusRecord:
    """Return the record ``line`` holds, or raise CorpusError naming line ``number``.

    ``number`` is the 1-based line number in ``source``. The three failure modes are kept apart in
    the message -- blank line, not JSON, not a JSON object -- and a record rejected by
    ``CorpusRecord.from_dict`` additionally names the ``id`` when the line carries one.
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
