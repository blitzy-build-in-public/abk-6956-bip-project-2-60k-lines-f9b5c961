"""The product's only JSON writer: one canonical text form, and ``summary.json``.

Every JSON artefact ``qrcheck`` emits is produced here. ``summary.json`` comes from
:func:`write_summary`; the evidence store (``qrcheck.report.evidence``) writes
``request.json`` and ``decoded.json`` through :func:`write_canonical_json`; and
``qrcheck.report.junit`` renders a failure body with :func:`dumps_canonical`. Because
there is exactly one formatter, the same model value always yields the same bytes, which
is what makes AAP §0.5.4's "byte-identical for the same case and seed" true. A second
JSON formatter anywhere in the package would be a defect: it could order keys, escape
characters or terminate the file differently, and two artefacts would then disagree
about the same run.

The form is fixed rather than configurable: keys sorted lexicographically at every depth,
two-space indent, UTF-8 with non-ASCII characters written literally, and exactly one
trailing newline per file. The literal characters matter because the ``emoji``, ``rtl``
and ``utf8-multibyte`` corpus classes appear verbatim in evidence, and the sorted keys are
what makes two runs comparable field by field once the documented volatile fields -
``started_at``, ``finished_at`` and ``results[].duration_ms`` - are masked (AAP §0.5.3,
§0.5.4).

Nothing here reads a clock, the environment or configuration, and nothing prints or logs:
the only effect this module has is the file :func:`write_canonical_json` is handed, so an
artefact is a pure function of the value it serialises.
"""

import json
from pathlib import Path
from typing import Any

from qrcheck.report.model import RunResult

__all__ = ["dumps_canonical", "write_canonical_json", "write_summary"]


def dumps_canonical(payload: Any) -> str:
    """Return the canonical JSON text for ``payload``.

    ``sort_keys=True`` orders keys at every depth, ``indent=2`` fixes the layout and
    thereby the ``", "``/``": "`` separators, so no ``separators`` argument is passed and
    no other layout can drift in, and ``ensure_ascii=False`` writes non-ASCII characters
    as themselves rather than as ``\\uXXXX`` escapes.

    The returned text carries no trailing newline: :func:`write_canonical_json` adds
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


def write_canonical_json(payload: Any, path: Path) -> Path:
    """Write ``payload`` to ``path`` in the canonical form and return ``path``.

    The parent directory is created when it is missing, and the file is UTF-8 with
    ``\\n`` line endings, one trailing newline and no byte-order mark, so the artefact is
    byte-identical across runs and across platforms.

    The text is serialised before the file is opened, and the file is then written
    directly rather than through a temporary file and a rename. Both choices are
    deliberate: serialising first means a non-serialisable payload leaves no file at all
    for a consumer to parse, and a temporary file inside the report directory would be an
    artefact outside the closed evidence layout, which is exactly what ``CL-ARTIFACTS``
    inspects for (AAP §0.3.4).

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
        OSError: the parent directory cannot be created or the file cannot be written.
    """
    # Serialise first: a TypeError must not leave a half-written or empty artefact.
    text = dumps_canonical(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


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
        OSError: the destination cannot be created or written.
    """
    return write_canonical_json(run_result.as_dict(), path)
