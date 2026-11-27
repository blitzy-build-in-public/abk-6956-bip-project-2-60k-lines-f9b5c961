"""The product's only JSON writer: one canonical text form, and ``summary.json``.

Every JSON artefact is produced here in three narrowing steps: :func:`dumps_canonical` returns the
canonical text, :func:`write_canonical_text` encodes such a text and writes it, and
:func:`write_canonical_json` composes the two. ``summary.json`` comes from :func:`write_summary`;
the evidence store (``qrcheck.report.evidence``) serialises ``request.json`` and ``decoded.json``
itself and writes both through :func:`write_canonical_text`, because it must hold both texts before
it creates a directory; ``qrcheck.report.junit`` renders a failure body with
:func:`dumps_canonical`. One formatter is what makes the same model value yield the same bytes,
which is what makes AAP §0.5.4's "byte-identical for the same case and seed" true; a second
formatter could order keys, escape characters or terminate the file differently.

The form is fixed rather than configurable: keys sorted lexicographically at every depth, two-space
indent, UTF-8 with non-ASCII characters written literally, exactly one trailing newline per file.
The literal characters matter because the ``emoji``, ``rtl`` and ``utf8-multibyte`` corpus classes
appear verbatim in evidence, and sorted keys are what makes two runs comparable field by field once
the documented volatile fields are masked (AAP §0.5.3-§0.5.5): ``started_at``, ``finished_at``,
every ``results[].duration_ms``, and the measured leaves of the two ``PF-*`` results'
``decoded.metrics`` -- ``encode_p95_ms``, ``decode_p95_ms`` and ``symbols_per_minute``. Those three
are a measurement judged once against the configured floor in ``expected`` and never against
another run's measurement, while the metrics a probe copies from configuration -- ``samples``,
``warmup`` and ``image_px`` -- stay in the comparison. ``docs/performance.md`` §7 owns the recipe.

Nothing here reads a clock, the environment or configuration, and nothing prints or logs, so an
artefact is a pure function of the value it serialises. No value is sanitised, replaced or escaped
to make it writable: an unserialisable payload raises :exc:`TypeError`, a non-finite float raises
:exc:`ValueError` because JSON defines no token for one, and an unencodable text raises
:exc:`UnicodeEncodeError` -- each before the destination file exists.
"""

import json
from pathlib import Path
from typing import Any

from qrcheck.report.model import RunResult

__all__ = ["dumps_canonical", "write_canonical_json", "write_canonical_text", "write_summary"]


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
    """
    try:
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
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


def write_canonical_text(text: str, path: Path) -> Path:
    """Write canonical JSON ``text`` to ``path`` and return ``path``.

    Exactly one ``\\n`` is appended, so the file is UTF-8 with ``\\n`` line endings, no byte-order
    mark and a single trailing newline: byte-identical across runs and platforms (AAP §0.5.4). The
    parent directory is created when it is missing, and the text is encoded to UTF-8 before any
    filesystem I/O, so a text UTF-8 cannot represent -- a lone surrogate an HTTP target's JSON
    metadata can carry, which ``json.loads`` materialises and ``ensure_ascii=False`` writes straight
    through -- leaves no file at all rather than an empty one. The bytes are written directly rather
    than through a temporary file and a rename, because a temporary file inside the report directory
    would be an artefact outside the closed evidence layout that ``CL-ARTIFACTS`` inspects for (AAP
    §0.3.4). Filesystem errors propagate untouched: ``qrcheck.cli.run`` is the single place that
    turns them into exit code 3, and the product has no retry path (AAP §0.3.4, §0.5.6).

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
    through a temporary file, so an unserialisable or unencodable payload leaves no artefact to
    parse and nothing appears outside the closed evidence layout ``CL-ARTIFACTS`` inspects for (AAP
    §0.3.4). The file is UTF-8 with ``\\n`` line endings, one trailing newline and no byte-order
    mark (AAP §0.5.4), and filesystem errors propagate untouched for ``qrcheck.cli.run`` to turn
    into exit code 3 (AAP §0.3.4, §0.5.6).

    Raises :exc:`TypeError` for a value that is not JSON-serialisable, :exc:`ValueError` for a
    non-finite float (``nan``, ``inf``, ``-inf``), :exc:`UnicodeEncodeError` for a character UTF-8
    cannot encode, and :exc:`OSError` when the directory or the file cannot be written.
    """
    # Serialise and encode first: TypeError, ValueError and UnicodeEncodeError must leave no
    # artefact at all.
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
    same artefact (AAP §0.5.3).

    Raises :exc:`TypeError` for a field value that is not JSON-serialisable, :exc:`ValueError` for a
    non-finite float such as a metric computed from a zero elapsed time, :exc:`UnicodeEncodeError`
    for a character UTF-8 cannot encode, and :exc:`OSError` when the destination cannot be created
    or written -- each raised before the file is opened, so no ``summary.json`` is left behind.
    """
    return write_canonical_json(run_result.as_dict(), path)
