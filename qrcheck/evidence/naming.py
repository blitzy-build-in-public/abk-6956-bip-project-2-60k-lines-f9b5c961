"""The single owner of the evidence layout, the artefact names and path sanitization.

One failure produces one evidence directory, and every module that names that
directory or anything inside it reads the name from here (AAP §0.3.4 "Evidence layout
and naming", §0.8.1 "Single owner of each cross-cutting concern"). The consumers are
fixed, and no second definition of any name below may exist:

* ``qrcheck/report/evidence.py`` reads :data:`EVIDENCE_DIR_NAME` to place the
  evidence root at ``<report_dir>/evidence``, and builds every case directory with
  :func:`evidence_dir` or :func:`case_evidence_dir`, writing only the closed artefact
  set (members of :data:`FIXED_ARTIFACT_NAMES`, :func:`symbol_artifact_name`,
  :func:`damage_artifact_name`), whose completeness :func:`is_evidence_artifact`
  decides.
* ``qrcheck/report/html.py`` renders a case's images from that same closed set,
  admitted by :func:`is_evidence_artifact` and listed in sorted-name order.

``qrcheck/checks/damage.py`` must import :data:`DAMAGE_KINDS` from here rather than
restating the order. Two spellings of one path would let a report link to evidence
that was written somewhere else, which is exactly what this module exists to prevent.

The layout is literal (AAP §0.3.4)::

    <report_dir> / evidence / <check-id> / <failure-id>
                   |          |            |
                   |          |            +-- sanitize(failure_id)
                   |          +--------------- sanitize(check_id)
                   +-------------------------- EVIDENCE_DIR_NAME

The failure ID is never re-derived here. :func:`qrcheck.report.model.failure_id` is
its only definition (AAP §0.8.1), and :func:`case_evidence_dir` calls it, so the
identifier inside a path cannot disagree with the identifier ``summary.json`` records
for the same case.

What this module never does:

* **No I/O.** It creates no directory, writes no file and reads none: *whether* a
  case's evidence is persisted is ``qrcheck.report.evidence.EvidenceStore``'s policy,
  and this module decides only where and how.
* **No volatility.** It emits no absolute path, no hostname, no user name, no process
  id, no timestamp, and nothing derived from ``hash()`` or ``id()`` (AAP §0.5.5), so
  two runs of the same case produce byte-identical path names.
* **No heavy imports.** Only the standard library is imported at module scope, so
  ``qrcheck.report`` can import this module without pulling Pillow or numpy into a
  configuration-only process (AAP §0.9.1). ``qrcheck.report.model`` is imported
  inside the two functions that need it, for the cycle-breaking reason documented
  there.
"""

import hashlib
import re
from pathlib import Path

# The middle path segment of every evidence location. ``qrcheck/report/evidence.py``
# reads this constant for the evidence root, so its name and value are a fixed public
# surface.
EVIDENCE_DIR_NAME: str = "evidence"

# The closed artefact set of AAP §0.3.4: "the set is closed; a check may write only the
# names below". A name is defined once, as a constant, so no writer spells it inline.
REQUEST_JSON: str = "request.json"
DECODED_JSON: str = "decoded.json"
SYMBOL_PNG: str = "symbol.png"
SYMBOL_SVG: str = "symbol.svg"
SYMBOL_REPEAT_PNG: str = "symbol-repeat.png"
LOGO_PNG: str = "logo.png"

# Those six names in lexicographic order, kept as a tuple rather than a set so a caller
# iterating it - the evidence store, the HTML writer, ``CL-ARTIFACTS`` - gets a stable
# order without sorting at every call site (AAP §0.5.3).
FIXED_ARTIFACT_NAMES: tuple[str, ...] = (
    DECODED_JSON,
    LOGO_PNG,
    REQUEST_JSON,
    SYMBOL_REPEAT_PNG,
    SYMBOL_PNG,
    SYMBOL_SVG,
)

# The documentable form of AAP §0.3.4's ``damage-<kind>-<seed>.png``.
DAMAGE_ARTIFACT_TEMPLATE: str = "damage-{kind}-{seed}.png"

# The order damage variants are emitted in, which AAP §0.5.3 fixes as a contract: the
# permutation must not be sorted, and it is written out rather than derived from a set,
# because set iteration order is exactly the ambient state the determinism rule forbids.
# Every damage case is reproducible from ``(run seed, check id, case id, kind)``; the
# kind is the token used in the artefact filename.
DAMAGE_KINDS: tuple[str, ...] = (
    "occluders",
    "blur",
    "jpeg",
    "noise",
    "rotation",
    "perspective",
    "downscale",
)

# The bound :func:`sanitize` enforces on one path segment, in characters.
MAX_COMPONENT_LENGTH: int = 128

# Everything outside this set becomes a single "_", matching
# ``qrcheck.report.model``'s rule for a failure ID so that a component sanitized here
# and an identifier sanitized there can never disagree (IM-7).
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# A complete damage artefact name: a known kind and a decimal seed, and nothing else -
# no separator, no suffix, no leading path, which is what makes the fullmatch safe to
# use as a membership test for the closed artefact set.
_DAMAGE_ARTIFACT_PATTERN = re.compile(
    r"damage-(?:" + "|".join(DAMAGE_KINDS) + r")-[0-9]+\.png"
)

# The retained prefix of a truncated component: the bound minus the 1-character
# separator and the 8 hexadecimal digest characters the suffix adds.
_TRUNCATED_PREFIX_LENGTH: int = MAX_COMPONENT_LENGTH - 9


def sanitize(component: str) -> str:
    """Reduce ``component`` to a single safe path segment.

    Every character outside ``[A-Za-z0-9._-]`` is replaced by ``_``, which removes
    path separators, parent-directory references and shell metacharacters in one
    stroke: a hostile case identifier cannot write outside ``report_dir`` because
    after this call it carries nothing a filesystem reads as a path (IM-7, AAP §0.3.2
    security architecture, NFR-8). ``"../etc"`` therefore becomes ``".._etc"``, which
    is an ordinary file name and cannot traverse; a result that still looks like
    ``..`` is rejected below rather than relied on to be harmless.

    The function is idempotent - ``sanitize(sanitize(x)) == sanitize(x)`` for every
    ``x`` that does not raise - because a component that already came from
    :func:`qrcheck.report.model.failure_id` is sanitized again here and the two must
    not drift.

    Args:
        component: The raw path segment, typically a check ID, a failure ID or a
            caller-supplied case identifier.

    Returns:
        The sanitized segment, never longer than :data:`MAX_COMPONENT_LENGTH`
        characters. A result longer than that is truncated to the first
        :data:`_TRUNCATED_PREFIX_LENGTH` characters, then ``_``, then the first 8
        hexadecimal characters of the SHA-256 digest of the *fully sanitized* segment,
        so the truncation stays deterministic and two long inputs that differ only
        beyond the cut point do not collide.

    Raises:
        TypeError: if ``component`` is not a ``str``.
        ValueError: if the sanitized result is empty, or is ``.`` or ``..`` - the two
            components that would collapse or escape the directory they name. The
            message names the offending component.
    """
    if not isinstance(component, str):
        raise TypeError(
            "evidence path component must be str, got " + type(component).__name__
        )
    sanitized = _UNSAFE_PATH_CHARS.sub("_", component)
    if not sanitized:
        raise ValueError(
            f"evidence path component {component!r} is empty after sanitization, "
            "so it cannot name a directory"
        )
    if sanitized in (".", ".."):
        raise ValueError(
            f"evidence path component {component!r} reduces to {sanitized!r}, "
            "which does not name a directory"
        )
    if len(sanitized) <= MAX_COMPONENT_LENGTH:
        return sanitized
    # A content digest, never hash(): the value becomes part of a path, and it has to
    # be the same on every machine and every run (AAP §0.5.5).
    digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[:_TRUNCATED_PREFIX_LENGTH]}_{digest}"


def symbol_artifact_name(output_format: str) -> str:
    """Return the artefact name a symbol of ``output_format`` is stored under.

    This is the only place the two literals are chosen, so the evidence store never
    hard-codes them and the format the target was asked for is the single reason a
    name is picked (AAP §0.3.4).

    Args:
        output_format: The requested artefact format, ``"png"`` or ``"svg"``.

    Returns:
        :data:`SYMBOL_PNG` for ``"png"``, :data:`SYMBOL_SVG` for ``"svg"``.

    Raises:
        ValueError: if ``output_format`` is neither, naming the value and the two
            legal ones.
    """
    if output_format == "png":
        return SYMBOL_PNG
    if output_format == "svg":
        return SYMBOL_SVG
    raise ValueError(f"symbol output_format must be 'png' or 'svg', got {output_format!r}")


def damage_artifact_name(kind: str, seed: int) -> str:
    """Return the artefact name of one damaged variant of a symbol.

    The seed in the name is what makes a damaged image re-derivable: a reader holding
    the run seed, the check ID, the case ID and the kind can regenerate this exact
    file, which is criterion 4's proof (AAP §0.6.1) and the rule ``DM-SEEDED``
    re-applies to the symbol it already holds (AAP §0.4.2).

    Args:
        kind: One of :data:`DAMAGE_KINDS`.
        seed: The non-negative derived seed that governs the damage, as
            :func:`qrcheck.config.seeding.derive_seed` produced it.

    Returns:
        ``damage-<kind>-<seed>.png``, the form AAP §0.3.4 fixes.

    Raises:
        ValueError: if ``kind`` is not a declared damage kind - the message lists the
            legal kinds in their declared order (AAP §0.5.3) - or if ``seed`` is not
            a non-negative ``int``.
    """
    if kind not in DAMAGE_KINDS:
        legal = ", ".join(DAMAGE_KINDS)
        raise ValueError(f"unknown damage kind {kind!r}; declared kinds, in order: {legal}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"damage seed must be a non-negative int, got {seed!r}")
    return DAMAGE_ARTIFACT_TEMPLATE.format(kind=kind, seed=seed)


def is_evidence_artifact(name: str) -> bool:
    """Return whether ``name`` is a name the closed evidence set allows.

    AAP §0.3.4 closes the set, so completeness is decided here once instead of by each
    writer keeping its own list: ``qrcheck/report/html.py`` uses this to ignore an
    unknown directory entry rather than publish it, and the evidence store names every
    artefact it writes through the closed-set names and helpers this module owns.

    Args:
        name: A bare file name, never a path.

    Returns:
        ``True`` for a member of :data:`FIXED_ARTIFACT_NAMES`, and for a
        ``damage-<kind>-<seed>.png`` name carrying a declared kind and a decimal
        seed; ``False`` for anything else, a name bearing a path separator included.

    Raises:
        TypeError: if ``name`` is not a ``str``.
    """
    if not isinstance(name, str):
        raise TypeError(f"evidence artifact name must be str, got {type(name).__name__}")
    if name in FIXED_ARTIFACT_NAMES:
        return True
    return _DAMAGE_ARTIFACT_PATTERN.fullmatch(name) is not None


def evidence_dir(report_dir: str | Path, check_id: str, failure_id: str) -> Path:
    """Return the evidence directory of one failure, under ``report_dir``.

    The layout is AAP §0.3.4's, literally::

        <report_dir>/evidence/<check-id>/<failure-id>/

    Both variable segments pass through :func:`sanitize`, so neither can leave the
    root the caller supplied. The call is pure - it creates nothing, because whether a
    case's evidence is written is
    ``qrcheck.report.evidence.EvidenceStore``'s decision - and the returned path
    carries the caller's root exactly as given: this module never resolves it, never
    makes it absolute and never invents one (AAP §0.5.5).

    Args:
        report_dir: The run's report directory as configured, relative or absolute.
        check_id: The stable check identifier, such as ``RT-ROUNDTRIP``.
        failure_id: The identifier :func:`qrcheck.report.model.failure_id` derived.

    Returns:
        The directory path. It is not created.

    Raises:
        TypeError: if a segment, or ``report_dir``, is not a ``str`` or a path.
        ValueError: if either segment sanitizes to nothing, ``.`` or ``..``.
    """
    return Path(report_dir) / EVIDENCE_DIR_NAME / sanitize(check_id) / sanitize(failure_id)


def relative_evidence_dir(check_id: str, failure_id: str) -> str:
    """Return the same layout as a relative posix path, for ``summary.json``.

    ``CaseResult.evidence_dir`` must be a relative posix path - an invariant
    ``qrcheck.report.model`` enforces - so the segments are joined with explicit
    forward slashes: neither ``os.path.join`` nor ``Path`` stringification is used,
    either of which would put a backslash into the summary on Windows (AAP §0.5.5).

    Args:
        check_id: The stable check identifier.
        failure_id: The identifier :func:`qrcheck.report.model.failure_id` derived.

    Returns:
        ``evidence/<check-id>/<failure-id>``, always with forward slashes.

    Raises:
        TypeError: if either argument is not a ``str``.
        ValueError: if either segment sanitizes to nothing, ``.`` or ``..``.
    """
    return "/".join((EVIDENCE_DIR_NAME, sanitize(check_id), sanitize(failure_id)))


def case_evidence_dir(
    report_dir: str | Path, check_id: str, case_id: str, reason: str
) -> Path:
    """Return the evidence directory of a case, deriving its failure ID once.

    The identifier inside the directory name comes from
    :func:`qrcheck.report.model.failure_id` and from nothing else, so it cannot
    disagree with the identifier ``summary.json`` records for the same result
    (AAP §0.8.1): one derivation, one path.

    Args:
        report_dir: The run's report directory as configured.
        check_id: The stable check identifier.
        case_id: The case identifier, such as ``corpus:url-001``.
        reason: The failure cause, as a :class:`qrcheck.report.model.Reason` member or
            its string value.

    Returns:
        The directory path. It is not created.

    Raises:
        TypeError: if a segment, or ``report_dir``, is not a ``str`` or a path.
        ValueError: if either segment sanitizes to nothing, ``.`` or ``..``.
    """
    # Imported here rather than at module scope on purpose:
    # ``qrcheck/report/__init__.py`` imports this module, so a module-scope
    # ``qrcheck.report.model`` import would close the cycle during partial
    # initialisation and raise ImportError.
    from qrcheck.report.model import failure_id

    return evidence_dir(report_dir, check_id, failure_id(check_id, case_id, reason))


def relative_case_evidence_dir(check_id: str, case_id: str, reason: str) -> str:
    """Return the case layout as a relative posix path, deriving its failure ID once.

    The ``CaseResult`` counterpart of :func:`case_evidence_dir`: same derivation, same
    segments, always forward-slash separated so the value satisfies
    ``CaseResult.evidence_dir`` on every platform (AAP §0.5.5).

    Args:
        check_id: The stable check identifier.
        case_id: The case identifier, such as ``corpus:url-001``.
        reason: The failure cause, as a :class:`qrcheck.report.model.Reason` member or
            its string value.

    Returns:
        ``evidence/<check-id>/<failure-id>``, always with forward slashes.

    Raises:
        TypeError: if either argument is not a ``str``.
        ValueError: if either segment sanitizes to nothing, ``.`` or ``..``.
    """
    # Function-local for the same cycle-breaking reason as ``case_evidence_dir``.
    from qrcheck.report.model import failure_id

    return relative_evidence_dir(check_id, failure_id(check_id, case_id, reason))


__all__ = [
    "DAMAGE_ARTIFACT_TEMPLATE",
    "DAMAGE_KINDS",
    "DECODED_JSON",
    "EVIDENCE_DIR_NAME",
    "FIXED_ARTIFACT_NAMES",
    "LOGO_PNG",
    "MAX_COMPONENT_LENGTH",
    "REQUEST_JSON",
    "SYMBOL_PNG",
    "SYMBOL_REPEAT_PNG",
    "SYMBOL_SVG",
    "case_evidence_dir",
    "damage_artifact_name",
    "evidence_dir",
    "is_evidence_artifact",
    "relative_case_evidence_dir",
    "relative_evidence_dir",
    "sanitize",
    "symbol_artifact_name",
]
