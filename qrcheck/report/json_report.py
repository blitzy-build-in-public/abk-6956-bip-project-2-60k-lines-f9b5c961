"""The product's only JSON writer: one canonical text form, and ``summary.json``.

Every JSON artefact ``qrcheck`` emits is produced here, through three steps that only
ever narrow: :func:`dumps_canonical` returns the canonical text, :func:`write_canonical_text`
encodes such a text and writes it, and :func:`write_canonical_json` composes the two for a
caller that holds a payload. ``summary.json`` comes from :func:`write_summary`; the
evidence store (``qrcheck.report.evidence``) serialises ``request.json`` and
``decoded.json`` itself and writes both through :func:`write_canonical_text`, because it
must hold both texts before it creates a directory; and ``qrcheck.report.junit`` renders a
failure body with :func:`dumps_canonical`. Because there is exactly one formatter, the same
model value always yields the same bytes, which is what makes AAP §0.5.4's "byte-identical
for the same case and seed" true. A second JSON formatter anywhere in the package would be
a defect: it could order keys, escape characters or terminate the file differently, and two
artefacts would then disagree about the same run.

The form is fixed rather than configurable: keys sorted lexicographically at every depth,
two-space indent, UTF-8 with non-ASCII characters written literally, and exactly one
trailing newline per file. The literal characters matter because the ``emoji``, ``rtl``
and ``utf8-multibyte`` corpus classes appear verbatim in evidence, and the sorted keys are
what makes two runs comparable field by field once the documented volatile fields -
``started_at``, ``finished_at`` and ``results[].duration_ms`` - are masked (AAP §0.5.3,
§0.5.4).

Nothing here reads a clock, the environment or configuration, and nothing prints or logs:
the only effect this module has is the file :func:`write_canonical_text` is handed, so an
artefact is a pure function of the value it serialises. None of the three writers sanitises,
replaces or escapes a value to make it writable: a payload that cannot be serialised raises
:exc:`TypeError` and a text that UTF-8 cannot encode raises :exc:`UnicodeEncodeError`, both
before the destination file exists, because a caller defect must be reported rather than
stringified or mangled into a plausible-looking artefact.
"""

import json
from pathlib import Path
from typing import Any

from qrcheck.report.model import RunResult

__all__ = ["dumps_canonical", "write_canonical_json", "write_canonical_text", "write_summary"]


def dumps_canonical(payload: Any) -> str:
    """Return the canonical JSON text for ``payload``.

    ``sort_keys=True`` orders keys at every depth, ``indent=2`` fixes the layout and
    thereby the ``", "``/``": "`` separators, so no ``separators`` argument is passed and
    no other layout can drift in, and ``ensure_ascii=False`` writes non-ASCII characters
    as themselves rather than as ``\\uXXXX`` escapes.

    The returned text carries no trailing newline: :func:`write_canonical_text` adds
    exactly one, and an in-body snippet such as a JUnit failure detail must not end in
    whitespace.

    There is deliberately no ``default=`` hook and no coercion. ``request`` and
    ``decoded`` mappings are built JSON-ready by the checks, so a stray ``bytes``,
    ``Path`` or adapter object has to raise here instead of being stringified into a
    plausible-looking report that hides the defect behind a green artefact. The
    canonical form also carries no ``allow_nan=False``: the model's numeric fields are
    finite by construction, and adding the guard would make this one formatter reject
    payloads for a reason the next formatter could disagree about.

    Args:
        payload: Any JSON-ready value: a mapping, sequence or scalar as the model and
            the evidence store produce them.

    Returns:
        The text, with keys sorted, indented by two spaces, and no trailing newline.

    Raises:
        TypeError: ``payload`` holds a value that is not JSON-serialisable, which is a
            caller defect rather than something to paper over here.
    """
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def write_canonical_text(text: str, path: Path) -> Path:
    """Write canonical JSON ``text`` to ``path`` and return ``path``.

    ``text`` is one artefact's canonical JSON as :func:`dumps_canonical` returns it, and
    exactly one ``\\n`` is appended to it, so the file ends in a single trailing newline.
    The result is UTF-8 with ``\\n`` line endings and no byte-order mark, so the artefact
    is byte-identical across runs and across platforms (AAP §0.5.4). The parent directory
    is created when it is missing.

    The text is encoded to UTF-8 before any filesystem I/O: the bytes are complete
    before ``mkdir`` and before the file is opened, so a text UTF-8 cannot represent
    leaves no file at all rather than an empty one. That case is reachable without any
    caller inventing it - an HTTP target's JSON metadata can carry a lone surrogate,
    ``json.loads`` materialises it, and ``ensure_ascii=False`` writes it straight through
    to the text - so it is a documented ``Raises`` entry rather than an accident.

    The bytes are then written directly rather than through a temporary file and a
    rename, because a temporary file inside the report directory would be an artefact
    outside the closed evidence layout, which is exactly what ``CL-ARTIFACTS`` inspects
    for (AAP §0.3.4).

    Filesystem errors are the caller's to interpret. An unwritable or uncreatable report
    directory propagates ``OSError``/``PermissionError`` untouched, because
    ``qrcheck.cli.run`` is the single place that turns that condition into exit code 3
    with the path in the message, and the product contains no retry path anywhere
    (AAP §0.3.4, §0.5.6).

    Args:
        text: One artefact's canonical JSON, as :func:`dumps_canonical` returns it. It is
            written as given: nothing is escaped, replaced, dropped or re-encoded to make
            an unencodable text fit.
        path: Destination file, normally resolved against the caller's ``report_dir``.
            It is used exactly as given and never resolved to an absolute path here.

    Returns:
        ``path``, unchanged, so a caller can record what it wrote.

    Raises:
        UnicodeEncodeError: ``text`` holds a character UTF-8 cannot encode, such as a
            lone surrogate. It is raised before anything is created or opened, so no file
            and no half-written artefact is left behind.
        OSError: the parent directory cannot be created or the file cannot be written.
    """
    # Encode before any I/O: an unencodable text must not leave an empty artefact.
    data = (text + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
    return path


def write_canonical_json(payload: Any, path: Path) -> Path:
    """Write ``payload`` to ``path`` in the canonical form and return ``path``.

    The parent directory is created when it is missing, and the file is UTF-8 with
    ``\\n`` line endings, one trailing newline and no byte-order mark, so the artefact is
    byte-identical across runs and across platforms.

    The text is serialised, and then encoded, before the file is opened, and the file is
    written directly rather than through a temporary file and a rename. Both choices are
    deliberate: serialising and encoding first means a payload that cannot be serialised,
    or that UTF-8 cannot encode, leaves no file at all for a consumer to parse, and a
    temporary file inside the report directory would be an artefact outside the closed
    evidence layout, which is exactly what ``CL-ARTIFACTS`` inspects for (AAP §0.3.4).
    :func:`write_canonical_text` performs both steps, so this writer is a one-line
    composition of it with :func:`dumps_canonical`.

    Filesystem errors are the caller's to interpret. An unwritable or uncreatable report
    directory propagates ``OSError``/``PermissionError`` untouched, because
    ``qrcheck.cli.run`` is the single place that turns that condition into exit code 3
    with the path in the message, and the product contains no retry path anywhere
    (AAP §0.3.4, §0.5.6).

    Args:
        payload: Any JSON-ready value; see :func:`dumps_canonical`.
        path: Destination file, normally resolved against the caller's ``report_dir``.
            It is used exactly as given and never resolved to an absolute path here.

    Returns:
        ``path``, unchanged, so a caller can record what it wrote.

    Raises:
        TypeError: ``payload`` holds a value that is not JSON-serialisable.
        UnicodeEncodeError: the serialised payload holds a character UTF-8 cannot encode,
            such as a lone surrogate carried in by a target's JSON metadata. Raised before
            the file is opened, so no artefact is left behind.
        OSError: the parent directory cannot be created or the file cannot be written.
    """
    # Serialise and encode first: TypeError and UnicodeEncodeError must leave no artefact.
    return write_canonical_text(dumps_canonical(payload), path)


def write_summary(run_result: RunResult, path: Path) -> Path:
    """Write ``run_result`` as ``summary.json`` and return ``path``.

    The schema lives in :meth:`qrcheck.report.model.RunResult.as_dict`, and this writer
    adds no key, no wrapper and no metadata of its own. The nine top-level keys are
    ``run_id``, ``started_at``, ``finished_at``, ``adapter``, ``ec_level``, ``seed``,
    ``loc_count``, ``counts`` (``{"passed", "failed", "skipped", "error"}``) and
    ``results``; each result object carries the ten fields ``check_id``, ``case_id``,
    ``status``, ``reason``, ``duration_ms``, ``request``, ``decoded``, ``expected``,
    ``evidence_dir`` and ``seed``. That set is a stable public surface that CI and
    tooling key on, so unknown keys may only ever be added: dropping or renaming a
    documented field here would break every machine consumer, and a field that has to
    appear belongs in ``as_dict`` rather than assembled in this writer.

    ``results`` is not re-sorted here. :class:`RunResult` already stores it ordered by
    ``(check_id, case_id)``, and a second sort would be a second ordering rule for the
    same artefact (AAP §0.5.3).

    Args:
        run_result: The completed run; its ``as_dict`` projection is written verbatim.
        path: Destination file, normally ``<report_dir>/summary.json``.

    Returns:
        ``path``, unchanged.

    Raises:
        TypeError: a result field holds a value that is not JSON-serialisable.
        UnicodeEncodeError: a result field holds a character UTF-8 cannot encode, such as
            a lone surrogate carried in by a target's JSON metadata. Raised before the
            file is opened, so no ``summary.json`` is left behind.
        OSError: the destination cannot be created or written.
    """
    return write_canonical_json(run_result.as_dict(), path)
