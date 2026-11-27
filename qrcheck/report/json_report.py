"""The product's only JSON writer: one canonical text form, and ``summary.json``.

Every JSON artefact is produced here in three narrowing steps: :func:`dumps_canonical` returns the
canonical text, :func:`write_canonical_text` encodes such a text and writes it, and
:func:`write_canonical_json` composes the two. ``summary.json`` comes from :func:`write_summary`;
the evidence store (``qrcheck.report.evidence``) serialises ``request.json`` and ``decoded.json``
itself and writes both through :func:`write_canonical_text`, because it must hold both texts before
it creates a directory; ``qrcheck.report.junit`` renders a failure body with
:func:`dumps_canonical`. One formatter is what makes the same model value yield the same bytes, so
an artefact is byte-identical for the same case and seed; a second formatter could order keys,
escape characters or terminate the file differently.

The form is fixed rather than configurable: keys sorted lexicographically at every depth, two-space
indent, UTF-8 with non-ASCII characters written literally, exactly one trailing newline per file.
The literal characters matter because the ``emoji``, ``rtl`` and ``utf8-multibyte`` corpus classes
appear verbatim in evidence, and sorted keys are what makes two runs comparable field by field once
the documented volatile fields are masked: ``started_at``, ``finished_at``, every
``results[].duration_ms``, and the measured leaves of the two ``PF-*`` results'
``decoded.metrics`` -- ``encode_p95_ms``, ``decode_p95_ms`` and ``symbols_per_minute``. Those three
are a measurement judged once against the configured floor in ``expected`` and never against
another run's measurement, while the metrics a probe copies from configuration -- ``samples``,
``warmup`` and ``image_px`` -- stay in the comparison. ``docs/performance.md`` §7 owns the recipe.

Nothing here reads a clock, the environment or configuration, and nothing prints or logs, so an
artefact is a pure function of the value it serialises. No value is escaped or coerced to make it
serialisable: an unserialisable payload raises :exc:`TypeError` and a non-finite float raises
:exc:`ValueError` because JSON defines no token for one, each before the destination file exists.
The single substitution this module makes is about bytes rather than JSON: a code point UTF-8
cannot encode -- the unpaired surrogate a target's own JSON can carry, which ``json.loads``
materialises and ``ensure_ascii=False`` would otherwise write straight through -- is replaced with
U+FFFD by :func:`make_utf8_encodable`, so text a target supplied becomes a written artefact instead
of a :exc:`UnicodeEncodeError` that would abort the run reporting it.
"""

import json
from pathlib import Path
from typing import Any, Final

from qrcheck.report.model import RunResult

__all__ = [
    "dumps_canonical",
    "make_utf8_encodable",
    "write_canonical_json",
    "write_canonical_text",
    "write_summary",
]

# The code points a Python ``str`` can hold that UTF-8 cannot encode: the UTF-16 surrogate range.
# A surrogate names half of a UTF-16 code unit, so it exists only inside a pair, and UTF-8 has no
# byte sequence for it -- ``str.encode("utf-8")`` refuses the whole text rather than the character.
# ``json.loads`` materialises the escape ``"\ud800"``, which is legal JSON and a form a target's
# response really can carry, as exactly such a lone surrogate.
_UNENCODABLE_CODEPOINTS: Final[tuple[int, ...]] = tuple(range(0xD800, 0xE000))

# Built once at import and reused for every document, so making a text writable costs one
# ``str.translate`` call and no regular expression. U+FFFD is the replacement character, one code
# point for one code point, so the text keeps its length and a reader sees that something stood
# there. ``qrcheck.report.junit`` builds its XML-illegal-character filter the same way.
_UNENCODABLE_CHARS: Final[dict[int, str]] = str.maketrans(
    {codepoint: "\ufffd" for codepoint in _UNENCODABLE_CODEPOINTS}
)


def make_utf8_encodable(text: str) -> str:
    """Return ``text`` with every code point UTF-8 cannot encode replaced by U+FFFD.

    The guarantee every writer here relies on: the returned text always encodes as UTF-8, so
    ``make_utf8_encodable(text).encode("utf-8")`` cannot raise :exc:`UnicodeEncodeError`. Only the
    surrogate range U+D800-U+DFFF is touched -- the sole code points a ``str`` can hold that UTF-8
    has no byte sequence for -- and each is replaced one for one, so the result has exactly the
    length of the text passed in while every other character, the non-ASCII characters
    ``ensure_ascii=False`` writes literally included, comes back unchanged and identical on every
    run.

    U+FFFD is the code point Unicode defines for exactly this purpose, so a reader of an artefact
    sees that something stood there and could not be represented: dropping the surrogate would
    silently shorten a payload a length assertion is compared against, and escaping it as
    ``\\ud800`` would put back the very text that cannot be written.

    Args:
        text: Any text, typically the canonical JSON of a payload a target influenced.

    Returns:
        ``text`` unchanged when it holds no surrogate, otherwise a copy of the same length in which
        each unencodable code point is U+FFFD.
    """
    return text.translate(_UNENCODABLE_CHARS)


def dumps_canonical(payload: Any) -> str:
    """Return the canonical JSON text for ``payload``.

    ``sort_keys=True`` orders keys at every depth, ``indent=2`` fixes the layout (and thereby the
    ``", "`` and ``": "`` separators, so no ``separators`` argument is passed and no other layout
    can drift in), and ``ensure_ascii=False`` writes non-ASCII characters as themselves. The text
    carries no trailing newline: :func:`write_canonical_text` adds exactly one, and an in-body
    snippet such as a JUnit failure detail must not end in whitespace. There is deliberately no
    ``default=`` hook and no coercion, so a stray ``bytes``, ``Path`` or adapter object has to raise
    here instead of being stringified into a plausible-looking report; ``allow_nan`` is ``False``
    for the same reason, because a metric computed from a zero elapsed time is ``inf`` and JSON
    defines no ``NaN``/``Infinity`` token that a strict RFC 8259 parser accepts.

    Raises :exc:`TypeError` for a value that is not JSON-serialisable, a caller defect rather than
    something to paper over, and :exc:`ValueError` for a non-finite float (``nan``, ``inf``,
    ``-inf``) or a self-referential container, the message naming which of the two was found.

    The text returned is always encodable as UTF-8: :func:`make_utf8_encodable` replaces every
    code point UTF-8 cannot encode -- the lone surrogate a target's ``"\\ud800"`` escape becomes --
    with U+FFFD, in a text of the same length, while every character UTF-8 can encode is still
    written exactly as it arrived. A payload a target influenced therefore yields a document
    instead of the :exc:`UnicodeEncodeError` that would abort the run writing it.
    """
    try:
        text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        # ``allow_nan=False`` refuses a non-finite float, which JSON defines no token for.
        # The only other ValueError json raises here is a self-referential payload, so the
        # diagnostic names the one that actually happened instead of assuming the first,
        # and both are refused here so that no caller opens a destination first.
        reason = (
            "a self-referential container"
            if "Circular reference" in str(exc)
            else "a non-finite float (nan, inf or -inf), which JSON does not define"
        )
        raise ValueError(f"payload holds {reason}: {exc}") from exc
    # Last, after the serialisation the form above fixes: the replacement cannot reorder a key,
    # add whitespace or change a character UTF-8 can encode, so the canonical bytes of an
    # encodable payload are the same as they were before it existed.
    return make_utf8_encodable(text)


def write_canonical_text(text: str, path: Path) -> Path:
    """Write canonical JSON ``text`` to ``path`` and return ``path``.

    Exactly one ``\\n`` is appended, so the file is UTF-8 with ``\\n`` line endings, no byte-order
    mark and a single trailing newline: byte-identical across runs and platforms. The
    parent directory is created when it is missing, and the text is encoded to UTF-8 before any
    filesystem I/O, so a text UTF-8 cannot represent leaves no file at all rather than an empty or
    truncated one. Reaching that refusal takes a caller assembling a text of its own: the canonical
    formatter replaces every unencodable code point with U+FFFD before its text gets here
    (:func:`make_utf8_encodable`), which is what a lone surrogate from a target needs. The bytes
    are written directly rather than through a temporary file and a rename, because a temporary
    file inside the report directory would be an artefact outside the closed evidence layout that
    ``CL-ARTIFACTS`` inspects for. Filesystem errors propagate untouched:
    ``qrcheck.cli.run`` is the single place that turns them into exit code 3, and the product has
    no retry path.

    ``text`` is written exactly as given -- nothing escaped, replaced, dropped or re-encoded -- and
    ``path`` is used as given, never resolved to an absolute path. Raises :exc:`UnicodeEncodeError`
    before anything is created or opened, and :exc:`OSError` when the directory or the file cannot
    be written.
    """
    # Encode before any I/O: an unencodable text must not leave an empty artefact.
    data = (text + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
    return path


def write_canonical_json(payload: Any, path: Path) -> Path:
    """Write ``payload`` to ``path`` in the canonical form and return ``path``.

    A one-line composition of :func:`dumps_canonical` with :func:`write_canonical_text`: the text is
    serialised and encoded before the file is opened, and the file is written directly rather than
    through a temporary file, so an unserialisable payload leaves no artefact to parse and nothing
    appears outside the closed evidence layout ``CL-ARTIFACTS`` inspects for. The file
    is UTF-8 with ``\\n`` line endings, one trailing newline and no byte-order mark,
    and filesystem errors propagate untouched for ``qrcheck.cli.run`` to turn into exit code 3.

    Raises :exc:`TypeError` for a value that is not JSON-serialisable, :exc:`ValueError` for a
    non-finite float (``nan``, ``inf``, ``-inf``), and :exc:`OSError` when the directory or the file
    cannot be written. No :exc:`UnicodeEncodeError` arises from the payload: the canonical
    formatter replaces every code point UTF-8 cannot encode with U+FFFD
    (:func:`make_utf8_encodable`), so the text this composes always encodes.
    """
    # Serialise and encode first: TypeError and ValueError must leave no artefact at all, and
    # dumps_canonical has already made the text UTF-8-encodable.
    return write_canonical_text(dumps_canonical(payload), path)


def write_summary(run_result: RunResult, path: Path) -> Path:
    """Write ``run_result`` as ``summary.json`` and return ``path``.

    The schema lives in :meth:`qrcheck.report.model.RunResult.as_dict`, and this writer adds no key,
    no wrapper and no metadata of its own. The nine top-level keys are ``run_id``, ``started_at``,
    ``finished_at``, ``adapter``, ``ec_level``, ``seed``, ``loc_count``, ``counts`` (``{"passed",
    "failed", "skipped", "error"}``) and ``results``; each result object carries the ten fields
    ``check_id``, ``case_id``, ``status``, ``reason``, ``duration_ms``, ``request``, ``decoded``,
    ``expected``, ``evidence_dir`` and ``seed``. That set is a stable public surface that CI and
    tooling key on, so unknown keys may only ever be added: dropping or renaming a documented field
    would break every machine consumer, and a field that has to appear belongs in ``as_dict`` rather
    than assembled here. ``results`` is not re-sorted, because :class:`RunResult` already stores it
    ordered by ``(check_id, case_id)`` and a second sort would be a second ordering rule for the
    same artefact.

    Raises :exc:`TypeError` for a field value that is not JSON-serialisable and :exc:`ValueError`
    for a non-finite float such as a metric computed from a zero elapsed time -- both raised before
    the destination is opened, so no ``summary.json`` is left behind. Unencodable text is not one
    of them: a code point UTF-8 cannot encode, which a target-influenced field can carry, is
    replaced with U+FFFD by the canonical formatter and the report is still written. Raises
    :exc:`OSError` when the destination cannot be created or written.
    """
    return write_canonical_json(run_result.as_dict(), path)
