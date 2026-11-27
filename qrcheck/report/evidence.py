"""The persistence policy for per-case evidence: who gets a directory, and what goes in it.

``qrcheck.runner`` hands every case's outcome to :class:`EvidenceStore`, and this module is the only
place that decides *whether* evidence reaches the filesystem. That decision is a policy, not an
optimisation: a full run is a few thousand cases, so a directory per case would write tens of
thousands of files, while a passing case whose bytes are reproducible from
``(adapter, seed, case_id)`` has nothing to inspect. Only ``failed``, ``error`` and ``skipped``
results are persisted; a skip is persisted because its machine-readable reason is the only record
of why a check never ran, and a skip must never be mistaken for a pass.

The layout and the artefact names are **not** defined here: :mod:`qrcheck.evidence.naming` owns
``<report_dir>/evidence/<check-id>/<failure-id>/``, the closed artefact set and path sanitization,
and the failure ID is derived inside :func:`qrcheck.evidence.naming.case_evidence_dir` from
:func:`qrcheck.report.model.failure_id`, its only definition, so the directory a case is filed under
cannot disagree with the identifier ``summary.json`` records for it. Nothing else may ever be
written -- no manifest, no index, no README, no per-directory metadata -- because a second kind of
name inside an evidence directory would break the closed set the HTML writer and ``CL-ARTIFACTS``
inspect against.

The closed artefact set, in the order this module writes it:

=============================  ====================================================
``request.json``               the case's identity envelope plus its request projection
``decoded.json``               the decoded-or-error evidence, persisted verbatim
``symbol.png`` / ``symbol.svg``  the target's artefact, byte for byte as received
``symbol-repeat.png``          ``DT-DETERMINISTIC``'s second call of the same request
``damage-<kind>-<seed>.png``   one damaged variant, named so that it is re-derivable
``logo.png``                   the overlay bitmap the target composited onto the symbol
=============================  ====================================================

What is persisted is the caller's projection, never a re-derivation of it: ``request`` already
carries the ``GenerateRequest`` fields plus the case's resolved parameters (damage kind and
thresholds, quiet-zone request, contrast request) and the ``out_of_contract`` marker
``qrcheck/checks/support.py`` sets for the ``IV-*`` probes, and ``decoded`` already carries
``per_decoder`` (each outcome's ``status`` and ``polarity``), ``unavailable_decoders`` (the reason
each decoder the run did not consult could not be), ``polarity_applied``, ``agreement``,
the EC-level pair, ``version``, ``mask``, ``matrix_sha256`` and the check-specific measurements, or
an error result's traceback. Both mappings are written verbatim: validating their shape here would
turn a legitimate error result into a framework failure.

Guarantees. Both JSON artefacts are serialised by :func:`qrcheck.report.json_report.dumps_canonical`
and written by :func:`qrcheck.report.json_report.write_canonical_text` -- sorted keys, two-space
indent, UTF-8 with one trailing newline -- and neither carries a timestamp, duration, hostname or
absolute path, which is what makes ``request.json`` and ``decoded.json`` byte-identical for the same
case and seed; ``duration_ms`` stays on the result, never in evidence. Writing the same case twice
leaves the same files with the same bytes in the same place: the directory is created with
``exist_ok=True``, the files are written directly, and no temporary file exists at any moment for
``CL-ARTIFACTS`` to find. Every artefact name, every artefact's bytes and both JSON
bodies are resolved before the case directory is created, so a malformed case raises with no
evidence directory at all -- not even one holding ``request.json`` alone -- and the only exception
that can follow the directory's creation is an :exc:`OSError`. Nothing below ``report_dir`` is
written through a link: the components this module composes -- the ``evidence`` root, the sanitized
check-id directory and the failure-id directory -- and every artefact destination in it are
verified not to be symlinks, and the case directory's real path is verified to lie inside the
report root's, all before the case directory is created, so a pre-planted symlink is refused by
name instead of followed (CWE-59) and no evidence byte can reach a path outside ``report_dir``.
``report_dir`` itself is the operator's own path and is never inspected: a report root that is a
symlink keeps working, and only what this store builds below it is checked. ``symbol`` and
``repeat_symbol`` are duck-typed on ``image_bytes: bytes`` and ``output_format: str`` (the
:class:`qrcheck.targets.base.GeneratedSymbol` shape, imported for type checking only), so the report
package never depends on the adapter layer and a configuration-only process pulls in no Pillow,
numpy or zxing.

Failure modes, three, none of them retried and none logged. Two are caller defects
raised before anything is written: :exc:`ValueError` from this module or from
:mod:`qrcheck.evidence.naming` for a malformed artefact request (an unknown symbol format, a non-PNG
repetition, an undeclared damage kind) or from :func:`qrcheck.report.json_report.dumps_canonical`
for a non-finite float (``nan``, ``inf``, ``-inf``), which JSON defines no token for; and
:exc:`TypeError` for a value the canonical writer cannot serialise or a non-bytes-like
``image_bytes``. Each message names the offending value or the artefact at fault. Text UTF-8 cannot
encode is not one of them: the lone surrogate a target's JSON metadata can carry in is replaced with
U+FFFD by the canonical formatter (:func:`qrcheck.report.json_report.make_utf8_encodable`), so it
becomes a written artefact instead of costing the run the report that describes it. The third is
:exc:`OSError`, raised two ways: before anything is created, when a component below
``report_dir`` or an artefact destination is a symlink or the case directory's real path leaves
the report root -- the refusal naming the path it declined to write -- and when the directory or
an artefact cannot be written (:exc:`PermissionError` included). It is not converted, retried or
logged: ``qrcheck.runner`` contains it as one ``error`` result for the case
and ``qrcheck.cli.run`` turns an unwritable ``report_dir`` into exit 3.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qrcheck.evidence.naming import (
    DECODED_JSON,
    EVIDENCE_DIR_NAME,
    LOGO_PNG,
    REQUEST_JSON,
    SYMBOL_REPEAT_PNG,
    case_evidence_dir,
    damage_artifact_name,
    symbol_artifact_name,
)
from qrcheck.report.json_report import dumps_canonical, write_canonical_text
from qrcheck.report.model import CaseResult, Status

if TYPE_CHECKING:  # Imported for type checking only; see the module docstring.
    from qrcheck.targets.base import GeneratedSymbol

__all__ = ["EvidenceStore"]


def _artifact_bytes(name: str, value: Any) -> bytes:
    """Return the bytes to write for artefact ``name``, or raise :exc:`TypeError`.

    ``Path.write_bytes`` accepted any buffer-protocol object -- ``bytes``, ``bytearray``,
    ``memoryview`` and the array-likes a check may hand over -- and exactly that set is kept. The
    conversion is deliberately ``bytes(memoryview(value))`` and never ``bytes(value)``, which would
    turn an ``int`` into that many NUL bytes and write a plausible-looking artefact instead of
    reporting the defect. The copy returned is the snapshot written later, taken before the case
    directory exists, with no re-encoding and no image round-trip; the message for a non-bytes-like
    ``value`` names ``name`` and the type that arrived.
    """
    try:
        return bytes(memoryview(value))
    except (TypeError, ValueError):
        # A non-buffer object and a released ``memoryview`` both fail here; neither is
        # bytes this store can persist, and either is a caller defect.
        raise TypeError(
            f"artefact {name} holds image bytes that are not bytes-like, got "
            f"{type(value).__name__}"
        ) from None


def _canonical_text(name: str, payload: Any) -> str:
    """Return the canonical JSON text for artefact ``name``, or raise the writer's refusal.

    The serialisation belongs to :func:`qrcheck.report.json_report.dumps_canonical`, the product's
    only JSON formatter; this function adds the artefact name to the diagnostic, so a caller who
    persisted an unserialisable or non-finite value is told which file it was for. Raises
    :exc:`TypeError` or :exc:`ValueError` from that writer, and nothing else: the text returned
    always encodes as UTF-8, because that formatter replaces every code point UTF-8 cannot encode --
    the lone surrogate a target's JSON can carry -- with U+FFFD before it returns, which is what
    lets record() create the case directory knowing neither write that follows can fail on encoding.
    """
    try:
        return dumps_canonical(payload)
    except TypeError as exc:
        raise TypeError(f"artefact {name}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"artefact {name}: {exc}") from exc


def _refuse_unsafe_case_dir(
    report_dir: Path, case_dir: Path, artifact_names: Sequence[str]
) -> None:
    """Raise :exc:`OSError` unless ``case_dir`` and its artefacts are safe to write.

    Two independent checks, both made before anything is created: a symlink planted in the report
    tree would otherwise be followed and the artefact files would land outside it (CWE-59), which
    sanitizing the path segments cannot prevent because the hazard is in the directory layout rather
    than in an identifier. First, every component composed below ``report_dir`` -- the ``evidence``
    root, the sanitized check-id directory, the failure-id directory -- and then every artefact
    destination in ``case_dir`` is refused when it is a symlink, the message naming the full path of
    the entry at fault so an operator can see what is in the way. Second, the case directory's real
    path is compared with the report root's, which catches any resolution this module did not
    compose itself: the case directory must be the report root or lie inside it.

    ``report_dir`` itself is never inspected. It is the operator's own path, so a report root that
    is a symlink keeps working, as does one reached through a symlinked ancestor; only what this
    store builds below it is verified. Nothing here creates, removes or reads a file: the checks are
    ``is_symlink`` and ``resolve``, so a refusal leaves the filesystem exactly as it was.

    Raises:
        OSError: A component below ``report_dir`` or an artefact destination is a symlink, or the
            case directory's real path leaves the report root's. The message names the path that was
            refused; it is an :exc:`OSError` because that is the store's documented write fault,
            which ``qrcheck.runner`` contains as one ``error`` result per case.
    """
    current = report_dir
    for part in case_dir.relative_to(report_dir).parts:
        current = current / part
        if current.is_symlink():
            raise OSError(
                f"refusing to write evidence into {current}: it is a symlink, so the write would "
                f"follow it out of the report directory {report_dir}"
            )
    for name in artifact_names:
        destination = case_dir / name
        if destination.is_symlink():
            raise OSError(
                f"refusing to write evidence artefact {destination}: it is a symlink, so the write "
                f"would replace the file it names instead of an artefact under {report_dir}"
            )
    resolved_root = report_dir.resolve()
    resolved_case = case_dir.resolve()
    if resolved_case != resolved_root and resolved_root not in resolved_case.parents:
        raise OSError(
            f"refusing to write evidence into {case_dir}: its real path leaves the report "
            f"directory {report_dir}"
        )


class EvidenceStore:
    """Writes, and declines to write, the evidence of one run's cases.

    One instance per run, constructed with the run's ``report_dir`` exactly as configuration gave it
    -- relative or absolute, never rewritten to an absolute path, resolved only inside
    :meth:`record` to compare it with the case directory's real path -- and the constructor performs
    no I/O, so a run that persists nothing leaves nothing behind on disk. The store is not a cache
    and keeps no state beyond that directory: every decision is a pure function of the
    :class:`qrcheck.report.model.CaseResult` and the artefacts supplied with it, which is what makes
    two runs of the same case produce the same directory and the same bytes.
    """

    def __init__(self, report_dir: str | Path) -> None:
        """Record where the run's reports live; create nothing.

        Args:
            report_dir: The run's report directory as configured. It is stored as given and
                never rewritten: this constructor creates, probes and resolves nothing, the
                first write creates what it needs, and an unwritable directory is reported
                by that write.
        """
        self.report_dir: Path = Path(report_dir)

    @property
    def base_dir(self) -> Path:
        """Return the evidence root, ``<report_dir>/evidence``.

        The single derivation of the evidence root in the product: the report dispatcher and the
        writers read it from here rather than joining the name themselves, so the root of everything
        this store writes has one definition.
        """
        return self.report_dir / EVIDENCE_DIR_NAME

    def clear(self) -> None:
        """Remove the evidence tree a previous run left under this report root, if any.

        The report-root contract this method keeps: a reused root describes this run alone, since
        the report files and the evidence tree are both replaced.

        Raises:
            OSError: the tree could not be removed, for ``qrcheck.cli.run`` to map to exit 3.
        """
        base = self.base_dir
        if base.is_dir() and not base.is_symlink():
            shutil.rmtree(base)
        elif base.exists() or base.is_symlink():
            base.unlink()

    def record(
        self,
        case_result: CaseResult,
        *,
        symbol: GeneratedSymbol | None = None,
        repeat_symbol: GeneratedSymbol | None = None,
        damage_variants: Sequence[tuple[str, bytes]] = (),
        logo: bytes | None = None,
    ) -> str | None:
        """Persist one case's evidence according to the persistence policy.

        A **passed** case is not persisted at all: no directory, no file, ``None`` returned. That is
        the policy this class owns rather than an optimisation -- a passing case has nothing to
        inspect, and one directory per case would bury the failing ones under tens of thousands of
        files -- and it keeps the invariant :class:`qrcheck.report.model.CaseResult` enforces, that
        a passed result carries ``evidence_dir is None``. A **failed**, **errored**
        or **skipped** case gets ``<report_dir>/evidence/<check-id>/<failure-id>/`` holding
        ``request.json`` (the identity envelope ``check_id``, ``case_id`` and ``seed`` merged over
        ``case_result.request``, the envelope winning a collision because it identifies the case and
        must never be overridable by a check), ``decoded.json`` verbatim, ``symbol.png`` or
        ``symbol.svg`` untouched, ``symbol-repeat.png`` when ``repeat_symbol`` is given,
        ``damage-<kind>-<seed>.png`` per ``damage_variants`` entry with ``case_result.seed`` in the
        name, and ``logo.png`` when ``logo`` is given. Nothing is created until every artefact is
        ready, so a refused case leaves no directory behind, and nothing below ``report_dir`` is
        written through a link: every component and every artefact destination is verified before
        the case directory is created. The caller applies the return value with
        ``dataclasses.replace``: the frozen result is never mutated.

        ``case_result``'s status decides whether anything is written, and its ``check_id``,
        ``case_id``, ``seed``, ``request`` and ``decoded`` decide what is. ``symbol`` and
        ``repeat_symbol`` are duck-typed on ``image_bytes`` and ``output_format`` and written
        exactly as received, the repetition being PNG because its name is fixed and being what makes
        ``DT-DETERMINISTIC``'s defect observable; each ``(kind, png_bytes)`` pair is written under
        the name the naming module derives, ``DM-SEEDED`` passing the variant it re-derived from the
        recorded seed so it lands beside ``DM-RESILIENCE``'s; and ``logo`` is the overlay bitmap,
        persisted so the covered area can be re-measured.

        Returns the case directory relative to ``report_dir`` as a posix path, such as
        ``evidence/RT-ROUNDTRIP/RT-ROUNDTRIP__corpus_url-001__value_mismatch`` -- the form
        ``summary.json`` records -- or ``None`` for a passed case, never an absolute path. Raises
        :exc:`ValueError` for an unknown symbol format or repetition, an undeclared damage kind, or
        a non-finite float in a persisted mapping; :exc:`TypeError` for a non-bytes-like
        ``image_bytes`` or an unserialisable value; and :exc:`OSError` either before anything is
        created, when a component below ``report_dir`` or an artefact destination is a symlink or
        the case directory's real path leaves the report root, or when the directory or an artefact
        cannot be written -- the latter being the only failure that follows the directory's
        creation. A character UTF-8 cannot encode is not a refusal: both persisted bodies are
        written with it replaced by U+FFFD, so a target's lone surrogate yields evidence rather than
        an exception.
        """
        if case_result.status is Status.PASSED:
            # The policy branch: a green case leaves the filesystem alone.
            return None

        case_dir = case_evidence_dir(
            self.report_dir, case_result.check_id, case_result.case_id, case_result.reason
        )
        # Resolve every artefact name, and capture its bytes, before the directory exists:
        # a malformed request must not leave a partial case behind. The order of this plan
        # is the order the files are written, and the legal damage kinds are validated by
        # the naming module, which owns that set.
        binaries: list[tuple[str, bytes]] = []
        if symbol is not None:
            artifact_name = symbol_artifact_name(symbol.output_format)
            binaries.append((artifact_name, _artifact_bytes(artifact_name, symbol.image_bytes)))
        if repeat_symbol is not None:
            if repeat_symbol.output_format != "png":
                raise ValueError(
                    f"artefact {SYMBOL_REPEAT_PNG} holds a PNG, got "
                    f"repeat_symbol.output_format {repeat_symbol.output_format!r}"
                )
            binaries.append(
                (SYMBOL_REPEAT_PNG, _artifact_bytes(SYMBOL_REPEAT_PNG, repeat_symbol.image_bytes))
            )
        for kind, png_bytes in damage_variants:
            artifact_name = damage_artifact_name(kind, case_result.seed)
            binaries.append((artifact_name, _artifact_bytes(artifact_name, png_bytes)))
        if logo is not None:
            binaries.append((LOGO_PNG, _artifact_bytes(LOGO_PNG, logo)))

        # The envelope is spread last so that it wins a collision with the request
        # projection: check_id, case_id and seed are the identity of the case.
        request_payload: dict[str, Any] = {
            **dict(case_result.request),
            "check_id": case_result.check_id,
            "case_id": case_result.case_id,
            "seed": case_result.seed,
        }

        # Both bodies are serialised before the directory exists: a value the canonical writer
        # cannot serialise must not leave a case directory holding request.json alone. Text UTF-8
        # cannot encode is not such a value -- the formatter has already replaced it with U+FFFD.
        request_text = _canonical_text(REQUEST_JSON, request_payload)
        decoded_text = _canonical_text(DECODED_JSON, dict(case_result.decoded))

        # The paths are inspected last, every artefact and both bodies already in hand: a
        # refusal must leave no case directory behind, and the names verified here are exactly
        # the closed set written below.
        artifact_names = (REQUEST_JSON, DECODED_JSON, *(name for name, _ in binaries))
        _refuse_unsafe_case_dir(self.report_dir, case_dir, artifact_names)

        # Deterministic overwrite: exist_ok keeps a second write of the same case from
        # failing, and every file is written directly rather than through a temporary file
        # and a rename (a temporary file under report_dir would be an artefact outside the
        # closed set, which is what CL-ARTIFACTS inspects for).
        case_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_text(request_text, case_dir / REQUEST_JSON)
        write_canonical_text(decoded_text, case_dir / DECODED_JSON)
        for artifact_name, artifact_bytes in binaries:
            (case_dir / artifact_name).write_bytes(artifact_bytes)

        return case_dir.relative_to(self.report_dir).as_posix()
