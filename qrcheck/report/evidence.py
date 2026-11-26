"""The persistence policy for per-case evidence: who gets a directory, and what goes in it.

``qrcheck.runner`` hands every case's outcome to :class:`EvidenceStore`, and this module is the only
place that decides *whether* evidence reaches the filesystem. That decision is a policy, not an
optimisation: a full run is a few thousand cases (AAP §0.8.1 case matrix), so a directory per case
would write tens of thousands of files, while a passing case whose bytes are reproducible from
``(adapter, seed, case_id)`` has nothing to inspect. Only ``failed``, ``error`` and ``skipped``
results are persisted (AAP §0.3.4); a skip is persisted because its machine-readable reason is the
only record of why a check never ran, and a skip must never be mistaken for a pass.

The layout and the artefact names are **not** defined here: :mod:`qrcheck.evidence.naming` owns
``<report_dir>/evidence/<check-id>/<failure-id>/``, the closed artefact set and path sanitization,
and the failure ID is derived inside :func:`qrcheck.evidence.naming.case_evidence_dir` from
:func:`qrcheck.report.model.failure_id`, its only definition, so the directory a case is filed under
cannot disagree with the identifier ``summary.json`` records for it. Nothing else may ever be
written -- no manifest, no index, no README, no per-directory metadata -- because a second kind of
name inside an evidence directory would break the closed set the HTML writer and ``CL-ARTIFACTS``
inspect against.

The closed artefact set, in the order this module writes it (AAP §0.3.4):

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
``per_decoder`` (each outcome's ``status`` and ``polarity``), ``polarity_applied``, ``agreement``,
the EC-level pair, ``version``, ``mask``, ``matrix_sha256`` and the check-specific measurements, or
an error result's traceback. Both mappings are written verbatim: validating their shape here would
turn a legitimate error result into a framework failure.

Guarantees. Both JSON artefacts are serialised by :func:`qrcheck.report.json_report.dumps_canonical`
and written by :func:`qrcheck.report.json_report.write_canonical_text` -- sorted keys, two-space
indent, UTF-8 with one trailing newline -- and neither carries a timestamp, duration, hostname or
absolute path, which is what makes ``request.json`` and ``decoded.json`` byte-identical for the same
case and seed (AAP §0.5.4); ``duration_ms`` stays on the result, never in evidence. Writing the
same case twice leaves the same files with the same bytes in the same place: the directory is
created with ``exist_ok=True``, the files are written directly, and no temporary file exists at any
moment for ``CL-ARTIFACTS`` to find. Every artefact name, every artefact's bytes and both JSON
bodies are resolved before the case directory is created, so a malformed case raises with no
evidence directory at all -- not even one holding ``request.json`` alone -- and the only exception
that can follow the directory's creation is an :exc:`OSError`. ``symbol`` and ``repeat_symbol`` are
duck-typed on ``image_bytes: bytes`` and ``output_format: str`` (the
:class:`qrcheck.targets.base.GeneratedSymbol` shape, imported for type checking only), so the report
package never depends on the adapter layer and a configuration-only process pulls in no Pillow,
numpy or zxing.

Failure modes, four, none of them retried and none logged (AAP §0.5.6). The first three are caller
defects raised before anything is written: :exc:`ValueError` from this module or from
:mod:`qrcheck.evidence.naming` for a malformed artefact request (an unknown symbol format, a non-PNG
repetition, an undeclared damage kind) or from :func:`qrcheck.report.json_report.dumps_canonical`
for a non-finite float (``nan``, ``inf``, ``-inf``), which JSON defines no token for;
:exc:`TypeError` for a value the canonical writer cannot serialise or a non-bytes-like
``image_bytes``; and :exc:`UnicodeEncodeError` for a character UTF-8 cannot encode, such as the lone
surrogate an HTTP target's JSON metadata can carry in. Each message names the offending value or the
artefact at fault. The fourth is :exc:`OSError` (including :exc:`PermissionError`) propagated
untouched, because ``qrcheck.cli.run`` turns an unwritable ``report_dir`` into exit 3.
"""

from __future__ import annotations

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
    :exc:`TypeError` or :exc:`ValueError` from that writer, and :exc:`UnicodeEncodeError` for a
    character UTF-8 cannot encode -- never converted into the ``ValueError``, because a lone
    surrogate is not a serialisation defect.
    """
    try:
        text = dumps_canonical(payload)
    except TypeError as exc:
        raise TypeError(f"artefact {name}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"artefact {name}: {exc}") from exc
    # Encoded here as well as in write_canonical_text, although the text is what is returned:
    # proving it encodes is what lets record() create the case directory knowing neither write
    # that follows can fail on encoding. It is encoded outside the handlers above because
    # UnicodeEncodeError is a ValueError: catching it there would report an unencodable
    # character as an unserialisable value.
    text.encode("utf-8")
    return text


class EvidenceStore:
    """Writes, and declines to write, the evidence of one run's cases.

    One instance per run, constructed with the run's ``report_dir`` exactly as configuration gave it
    -- relative or absolute, never resolved here (AAP §0.5.5) -- and the constructor performs no
    I/O, so a run that persists nothing leaves nothing behind on disk. The store is not a cache and
    keeps no state beyond that directory: every decision is a pure function of the
    :class:`qrcheck.report.model.CaseResult` and the artefacts supplied with it, which is what makes
    two runs of the same case produce the same directory and the same bytes.
    """

    def __init__(self, report_dir: str | Path) -> None:
        """Record where the run's reports live; create nothing.

        Args:
            report_dir: The run's report directory as configured. It is stored as given
                and is never resolved, created or probed: the first write creates what it
                needs, and an unwritable directory is reported by that write.
        """
        self.report_dir: Path = Path(report_dir)

    @property
    def base_dir(self) -> Path:
        """Return the evidence root, ``<report_dir>/evidence``.

        The single derivation of the evidence root in the product: the report dispatcher and the
        writers read it from here rather than joining the name themselves, so the root of everything
        this store writes has one definition (AAP §0.8.1).
        """
        return self.report_dir / EVIDENCE_DIR_NAME

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
        files (AAP §0.3.4) -- and it keeps the invariant :class:`qrcheck.report.model.CaseResult`
        enforces, that a passed result carries ``evidence_dir is None``. A **failed**, **errored**
        or **skipped** case gets ``<report_dir>/evidence/<check-id>/<failure-id>/`` holding
        ``request.json`` (the identity envelope ``check_id``, ``case_id`` and ``seed`` merged over
        ``case_result.request``, the envelope winning a collision because it identifies the case and
        must never be overridable by a check), ``decoded.json`` verbatim, ``symbol.png`` or
        ``symbol.svg`` untouched, ``symbol-repeat.png`` when ``repeat_symbol`` is given,
        ``damage-<kind>-<seed>.png`` per ``damage_variants`` entry with ``case_result.seed`` in the
        name, and ``logo.png`` when ``logo`` is given. Nothing is created until every artefact is
        ready, so a refused case leaves no directory behind, and the caller applies the return value
        with ``dataclasses.replace``: the frozen result is never mutated.

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
        ``image_bytes`` or an unserialisable value; :exc:`UnicodeEncodeError` for a character UTF-8
        cannot encode; and :exc:`OSError` when the directory or an artefact cannot be written, the
        only failure that follows the directory's creation.
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

        # Both bodies are serialised, and encoded, before the directory exists: a value
        # the canonical writer cannot serialise, or a lone surrogate it cannot encode,
        # must not leave a case directory holding request.json alone.
        request_text = _canonical_text(REQUEST_JSON, request_payload)
        decoded_text = _canonical_text(DECODED_JSON, dict(case_result.decoded))

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
