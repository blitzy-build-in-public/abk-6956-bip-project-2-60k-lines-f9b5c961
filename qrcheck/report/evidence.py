"""The persistence policy for per-case evidence: who gets a directory, and what goes in it.

``qrcheck.runner`` hands every case's outcome to :class:`EvidenceStore`, and this module alone
decides *whether* evidence reaches the filesystem.  That decision is a policy, not an
optimisation: a full run is a few thousand cases (AAP §0.8.1 case matrix), so a directory per case
would write tens of thousands of files, while a passing case whose bytes are reproducible from
``(adapter, seed, case_id)`` has nothing to inspect.  Only ``failed``, ``error`` and ``skipped``
results are persisted (AAP §0.3.4) -- a skip because its machine-readable reason is the only
record of why a check never ran, and a skip must never be mistaken for a pass.

The layout and the artefact names are **not** defined here.  :mod:`qrcheck.evidence.naming` owns
``<report_dir>/evidence/<check-id>/<failure-id>/``, the closed artefact set and path sanitization,
and this module names a file only by importing those constants and builders (AAP §0.8.1, "Single
owner of each cross-cutting concern").  The failure ID is derived inside
:func:`qrcheck.evidence.naming.case_evidence_dir` from
:func:`qrcheck.report.model.failure_id`, its only definition, so the directory a case is filed
under cannot disagree with the identifier ``summary.json`` records for it.

The closed artefact set, in the order this module writes it (AAP §0.3.4): ``request.json`` (the
case's identity envelope plus its request projection), ``decoded.json``, ``symbol.png`` or
``symbol.svg`` byte for byte as received, ``symbol-repeat.png`` for ``DT-DETERMINISTIC``'s second
call of the same request, ``damage-<kind>-<seed>.png`` named so it is re-derivable, and
``logo.png``.  Nothing else may ever be written -- no manifest, index, README or per-directory
metadata -- because a second kind of name inside an evidence directory would break the closed set
the HTML writer and ``CL-ARTIFACTS`` inspect against.

What is persisted is the caller's projection, never a re-derivation of it: ``request`` carries the
``GenerateRequest`` fields plus the case's resolved parameters (damage kind and thresholds, the
quiet-zone and contrast requests) and the ``out_of_contract`` marker
``qrcheck/checks/support.py`` sets for the ``IV-*`` probes, and ``decoded`` carries
``per_decoder``, ``polarity_applied``, ``agreement``, the EC-level pair, ``version``, ``mask``,
``matrix_sha256`` and the check-specific measurements, or the traceback of an error result.  Both
mappings are written verbatim; validating their shape here would turn a legitimate error result,
whose ``decoded`` holds only a traceback, into a framework failure.

Guarantees:

* **Byte form.** Both JSON artefacts are serialised by
  :func:`qrcheck.report.json_report.dumps_canonical` and written by
  :func:`qrcheck.report.json_report.write_canonical_text` -- sorted keys, two-space indent, UTF-8
  with one trailing newline -- and neither carries a timestamp, a duration, a hostname or an
  absolute path, which is what makes ``request.json`` and ``decoded.json`` byte-identical for the
  same case and seed (AAP §0.5.4).  Nothing here adds a field of its own, and the result's
  ``duration_ms`` stays on the result, never in evidence.
* **Deterministic overwrite.** Writing the same case twice leaves the same files with the same
  bytes in the same place: the directory is created with ``exist_ok=True`` and the files are
  written directly.  Nothing is appended, deleted or renamed, and no temporary file exists at any
  moment -- a temporary file inside ``report_dir`` is exactly what ``CL-ARTIFACTS`` looks for.
* **Validation before I/O.** Every artefact name, every artefact's bytes and both JSON bodies are
  resolved before the case directory is created, so a malformed case raises with no evidence
  directory at all -- not even one holding ``request.json`` alone -- and the only exception that
  can follow the directory's creation is an :exc:`OSError` from the filesystem.
* **No adapter or decoder dependency.** ``symbol`` and ``repeat_symbol`` are duck-typed on
  ``image_bytes: bytes`` and ``output_format: str`` (the
  :class:`qrcheck.targets.base.GeneratedSymbol` shape), imported for type checking only: a runtime
  import would make the report package depend on the adapter layer, which the layering rule
  forbids, and pull Pillow, numpy and zxing into a configuration-only process.

Four failure modes, none retried and none logged (AAP §0.5.6); the first three are caller defects
raised before anything is written.  :exc:`ValueError` from this module or
:mod:`qrcheck.evidence.naming` when the artefact request itself is malformed -- an unknown symbol
format, a non-PNG repetition, an undeclared damage kind -- or from
:func:`qrcheck.report.json_report.dumps_canonical` when a persisted mapping holds a non-finite
float (``nan``, ``inf`` or ``-inf``), which JSON defines no token for; each message names the
offending value or artefact.  :exc:`TypeError` when a persisted mapping holds a value the
canonical writer cannot serialise, or a supplied artefact's ``image_bytes`` is not bytes-like:
both fail here rather than being stringified into a plausible-looking artefact.
:exc:`UnicodeEncodeError` when a persisted mapping holds a character UTF-8 cannot encode, such as
the lone surrogate an HTTP target's JSON metadata can carry in.  :exc:`OSError` (including
:exc:`PermissionError`) from the filesystem is propagated untouched, and ``qrcheck.cli.run`` is
the single place that turns an unwritable ``report_dir`` into exit code 3.
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

    ``Path.write_bytes`` accepted any object the buffer protocol exposes - ``bytes``,
    ``bytearray``, ``memoryview`` and the array-likes a check may hand over - and this
    function keeps exactly that set, so no caller that worked before stops working. The
    conversion is deliberately ``bytes(memoryview(value))`` and never ``bytes(value)``:
    the latter would turn an ``int`` into that many NUL bytes and write a
    plausible-looking artefact instead of reporting the defect.

    The copy it returns is the snapshot that is later written, so the file receives the
    bytes the caller supplied at the moment of the call, taken before the case directory
    exists, with no re-encoding and no image round-trip.

    Args:
        name: The artefact name the bytes will be filed under, so that the message names
            what the caller has to fix.
        value: The artefact's bytes exactly as the caller supplied them.

    Returns:
        Those bytes.

    Raises:
        TypeError: ``value`` is not bytes-like. The message names ``name`` and the type
            that arrived.
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

    The serialisation itself belongs to
    :func:`qrcheck.report.json_report.dumps_canonical`, the product's only JSON formatter;
    this function adds the artefact name to the diagnostic so that a caller who persisted
    an unserialisable value - or a non-finite one, which the writer refuses for the same
    reason - is told which file it was for, not merely which type was refused.

    Args:
        name: The artefact name the text is destined for, so that the message names it.
        payload: The projection to serialise.

    Returns:
        The canonical text, without a trailing newline.

    Raises:
        TypeError: ``payload`` holds a value that is not JSON-serialisable. The message
            names ``name`` and then the value the writer refused.
        ValueError: ``payload`` holds a non-finite float, which JSON does not define, so
            the artefact could not be canonical JSON. The message names ``name`` and then
            the value the writer refused.
        UnicodeEncodeError: the text holds a character UTF-8 cannot encode, such as a lone
            surrogate. Raised before anything is written, and never converted into the
            ``ValueError`` above - a lone surrogate is not a serialisation defect.
    """
    try:
        text = dumps_canonical(payload)
    except TypeError as exc:
        raise TypeError(f"artefact {name}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"artefact {name}: {exc}") from exc
    # Encoded here as well as in write_canonical_text, although the text is what is
    # returned: proving it encodes is what lets record() create the case directory
    # knowing neither write that follows can fail on encoding. It is encoded outside the
    # handlers above because UnicodeEncodeError is a ValueError: catching it there would
    # report an unencodable character as an unserialisable value.
    text.encode("utf-8")
    return text


class EvidenceStore:
    """Writes, and declines to write, the evidence of one run's cases.

    One instance per run, constructed with the run's ``report_dir`` exactly as
    configuration gave it - relative or absolute, never resolved here (AAP §0.5.5).
    The constructor performs no I/O and creates no directory, so a run that persists
    nothing leaves nothing behind on disk.

    The store is not a cache and keeps no state beyond the report directory: every
    decision is a pure function of the :class:`qrcheck.report.model.CaseResult` and the
    artefacts supplied with it, which is what makes two runs of the same case produce
    the same directory and the same bytes.
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

        The single derivation of the evidence root in the product: the report dispatcher
        and the writers read it from here rather than joining the name themselves, so the
        root of everything this store writes has one definition (AAP §0.8.1).
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

        A **passed** case is not persisted at all: no directory is created, no file is
        written and ``None`` is returned. That is the policy this class exists to own,
        not an optimisation - a passing case has nothing to inspect, and one directory
        per passing case would bury the failing ones under tens of thousands of files
        (AAP §0.3.4). It also keeps the invariant
        :class:`qrcheck.report.model.CaseResult` enforces, that a passed result carries
        ``evidence_dir is None``.

        A **failed**, **errored** or **skipped** case gets its directory under
        ``<report_dir>/evidence/<check-id>/<failure-id>/``, with the artefacts below:

        * ``request.json`` - the identity envelope ``check_id``, ``case_id`` and ``seed``
          merged with ``case_result.request``. The envelope wins a key collision: it
          identifies the case and must never be overridable by a check.
        * ``decoded.json`` - ``case_result.decoded`` verbatim.
        * ``symbol.png`` or ``symbol.svg`` - the artefact's bytes untouched.
        * ``symbol-repeat.png`` - only when ``repeat_symbol`` is given.
        * ``damage-<kind>-<seed>.png`` - one per entry of ``damage_variants``, in the
          order given, with ``case_result.seed`` in the name.
        * ``logo.png`` - only when ``logo`` is given.

        Nothing is created until every artefact above is ready: the names are resolved,
        both JSON bodies are serialised and encoded, and each image byte string is
        captured first, so a case this method refuses leaves no directory behind.

        The caller applies the returned value with
        ``dataclasses.replace(case_result, evidence_dir=returned)``; this method never
        mutates the (frozen) result it is given.

        Args:
            case_result: The case's outcome. Only its status decides whether anything is
                written; its ``check_id``, ``case_id``, ``seed``, ``request`` and
                ``decoded`` decide what is.
            symbol: The artefact the target produced for this case, duck-typed on
                ``image_bytes: bytes`` and ``output_format: str``. Its bytes are written
                exactly as received, so a reader sees what the target actually emitted.
            repeat_symbol: The artefact of a second, identical invocation, which is what
                makes ``DT-DETERMINISTIC``'s defect observable across two calls. It must
                be PNG, because its artefact name is fixed.
            damage_variants: ``(kind, png_bytes)`` pairs, each written under the name the
                naming module derives from ``kind`` and ``case_result.seed``. Normally one
                pair per case; ``DM-SEEDED`` passes the variant it re-derived from the
                recorded seed so that it lands under the same name as ``DM-RESILIENCE``'s.
            logo: The generated overlay bitmap the target composited onto the symbol,
                persisted so the covered area can be re-measured.

        Returns:
            The case directory relative to ``report_dir`` as a posix path, such as
            ``evidence/RT-ROUNDTRIP/RT-ROUNDTRIP__corpus_url-001__value_mismatch``, which
            is the form ``summary.json`` records; or ``None`` for a passed case. No
            absolute path is ever returned, and none is written into an artefact.

        Raises:
            ValueError: ``symbol.output_format`` or ``repeat_symbol.output_format`` is
                neither ``"png"`` nor ``"svg"`` (the repetition must be PNG), or a damage
                kind is not one of the declared kinds. Nothing has been written when this
                is raised: every name is resolved first. The same exception is raised when
                a persisted mapping holds a non-finite float - ``nan``, ``inf`` or
                ``-inf`` - which JSON does not define, so ``request.json`` or
                ``decoded.json`` could not be canonical JSON; its message names the
                artefact, and neither body has been written when it is raised.
            TypeError: a supplied artefact's ``image_bytes`` is not bytes-like, or a
                mapping persisted here holds a value that is not JSON-serialisable. The
                message names the artefact at fault, and nothing has been written when it
                is raised.
            UnicodeEncodeError: a mapping persisted here holds a character UTF-8 cannot
                encode, such as a lone surrogate carried in by a target's JSON metadata.
                Raised before the case directory exists, so no evidence is left behind.
            OSError: the case directory cannot be created or an artefact cannot be
                written - the only failure that follows the directory's creation.
                Propagated untouched for ``qrcheck.cli.run`` to map to exit 3.
        """
        if case_result.status is Status.PASSED:
            # The policy branch: a green case leaves the filesystem alone.
            return None

        case_dir = case_evidence_dir(
            self.report_dir, case_result.check_id, case_result.case_id, case_result.reason
        )
        # Resolve every artefact name, and capture its bytes, before the directory
        # exists: a malformed request must not leave a partial case behind. The order of
        # this plan is the order the files are written, and the legal damage kinds are
        # validated by the naming module, which owns that set.
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
        # failing, and every file is written directly rather than through a temporary
        # file and a rename (a temporary file under report_dir would be an artefact
        # outside the closed set, which is what CL-ARTIFACTS inspects for).
        case_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_text(request_text, case_dir / REQUEST_JSON)
        write_canonical_text(decoded_text, case_dir / DECODED_JSON)
        for artifact_name, artifact_bytes in binaries:
            (case_dir / artifact_name).write_bytes(artifact_bytes)

        return case_dir.relative_to(self.report_dir).as_posix()
