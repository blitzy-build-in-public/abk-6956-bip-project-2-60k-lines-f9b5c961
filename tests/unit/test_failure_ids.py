"""One failure ID, one definition: sanitized, bounded and safe as a path segment.

``qrcheck.report.model.failure_id`` is the product's only derivation of the name a
failure is filed under, and ``qrcheck.evidence.naming`` calls it instead of
re-deriving it, so the identifier ``summary.json`` records and the last segment of an
evidence directory cannot disagree (AAP §0.8.1 "single owner of each cross-cutting
concern", §0.3.4 "Stable identifiers"). Four properties of that identifier are pinned
here, and each one is load-bearing:

* **it is a stated formula.** ``<CHECK-ID>__<case_id>__<reason>``, every character
  outside ``[A-Za-z0-9._-]`` replaced by ``_``, and - when that text exceeds the bound
  - a retained prefix followed by an 8-hex SHA-256 digest. The formula is recomputed in
  :func:`reference_failure_id` from the plan's own words rather than transcribed from a
  run, so a change to it fails here instead of silently renaming every evidence
  directory and every CI trend line;
* **it is one path segment.** The sanitization is a security control, not a formatting
  nicety (AAP §0.3.2 security architecture, IM-7): a hostile ``case_id`` such as
  ``"../../etc/passwd"`` must still land inside the report directory, which
  :func:`test_evidence_dir_never_leaves_the_report_directory` asserts by resolution
  rather than by inspecting characters;
* **it is bounded.** The identifier becomes a directory name, so its whole length -
  prefix and digest together, not the prefix alone - stays within
  ``qrcheck.report.model.FAILURE_ID_MAX``;
* **it is a pure function of its three inputs**, stable across processes and
  ``PYTHONHASHSEED`` values, because the directories it names are promised to be
  byte-stable (AAP §0.5.4).

Two behaviours of the neighbouring :func:`qrcheck.evidence.naming.sanitize` are
asserted rather than assumed, because they decide whether a component is usable at
all: it refuses an empty component and the two components that are still dangerous
after the character rule (``.`` and ``..``), and it replaces one ``_`` per *code
point*, so a two-byte ``ü`` and a four-byte emoji each cost one character and
``MAX_COMPONENT_LENGTH`` counts characters rather than bytes.

No threshold from ``qrcheck.toml`` is read here, nothing is skipped, retried or
marked, every collection is iterated in sorted order, nothing is written outside
``tmp_path``, and the only process this module starts is one short child interpreter
printing identifiers under a chosen ``PYTHONHASHSEED``.
"""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qrcheck.evidence import naming as naming_module
from qrcheck.evidence.naming import (
    EVIDENCE_DIR_NAME,
    MAX_COMPONENT_LENGTH,
    case_evidence_dir,
    evidence_dir,
    relative_case_evidence_dir,
    relative_evidence_dir,
    sanitize,
)
from qrcheck.report import model as model_module
from qrcheck.report.model import CaseResult, Reason, Status, failure_id

# Resolved from the imported module, never from the working directory: before
# qrcheck/__init__.py exists the package imports as a namespace package, whose
# qrcheck.__file__ is None, while a submodule's __file__ is always real.
_MODEL_PATH = Path(model_module.__file__).resolve()
_NAMING_PATH = Path(naming_module.__file__).resolve()
_PACKAGE_DIR = _MODEL_PATH.parents[1]  # .../qrcheck
_REPOSITORY_ROOT = _MODEL_PATH.parents[2]  # the directory that holds qrcheck/

# The bound the identifier must respect, read from the module that owns it rather than
# restated, so this module cannot agree with a stale copy of the number.
FAILURE_ID_LIMIT: int = model_module.FAILURE_ID_MAX

# The sanitization rule, stated once (AAP §0.3.4: "every character outside
# [A-Za-z0-9._-] replaced by _"). It is written out here instead of imported from the
# module under test, so widening the replacement set - dropping "/" above all - fails
# the assertions below instead of moving with the implementation.
UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# AAP §0.3.4: the truncation suffix is "an 8-hex sha256 suffix". Its length is the only
# reason the retained prefix is shorter than the bound, and it is the one arithmetic
# constant the formula needs.
DIGEST_CHARACTERS: int = 8


def reference_sanitize(text: str) -> str:
    """Apply the documented replacement rule without imposing any bound.

    One ``_`` replaces each code point outside ``[A-Za-z0-9._-]``; nothing else
    changes, so the result has the input's length in code points.
    """
    return UNSAFE_SEGMENT_CHARS.sub("_", text)


def reference_failure_id(
    check_id: str, case_id: str, reason: Reason | str | None
) -> str:
    """Recompute the documented failure-ID formula independently of the module.

    The expectation is the plan's rule written out again - never a value copied from a
    run - so a change to the formula has to be a deliberate change in two places. The
    retained prefix absorbs the separator and the digest, so the *whole* identifier
    stays within :data:`FAILURE_ID_LIMIT`: reading the rule as "a full-length prefix
    plus a suffix" would exceed the bound the directory name has to respect.
    """
    reason_text = "none" if reason is None else getattr(reason, "value", reason)
    sanitized = reference_sanitize(f"{check_id}__{case_id}__{reason_text}")
    if len(sanitized) <= FAILURE_ID_LIMIT:
        return sanitized
    digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:DIGEST_CHARACTERS]
    retained = FAILURE_ID_LIMIT - DIGEST_CHARACTERS - 1
    return f"{sanitized[:retained]}_{digest}"


def _raw(check_id: str, case_id: str, reason: Reason | str | None) -> str:
    """The identifier text before any replacement, exactly as the formula joins it."""
    reason_text = "none" if reason is None else getattr(reason, "value", reason)
    return f"{check_id}__{case_id}__{reason_text}"


def _label(value: object) -> str:
    """Render a table entry as a bounded, ASCII, collision-free test id."""
    rendered = repr(value).encode("ascii", "backslashreplace").decode("ascii")
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:DIGEST_CHARACTERS]
    return f"{rendered[:60]}-{digest}"


# --------------------------------------------------------------------------- #
# Case tables
# --------------------------------------------------------------------------- #

# Ordinary identifiers the product actually produces: a check ID, a case ID and a
# reason from the three families that file evidence. Each triple is short enough to stay
# inside the bound, which is what lets the shape assertion expect no truncation suffix.
_ORDINARY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("CL-ARTIFACTS", "synthetic:run", "contract_error"),
    ("DM-SEEDED", "synthetic:occluders-h", "sampling_failed"),
    ("IV-OVERCAPACITY", "synthetic:over-capacity", "capacity_violation"),
    ("RT-ROUNDTRIP", "corpus:url-001", "value_mismatch"),
    ("SY-EC-MATCH", "corpus:boundary-byte-017", "format_info_disagreement"),
)

ORDINARY_CASES: tuple[tuple[str, str, str], ...] = tuple(sorted(_ORDINARY_SPECS))

# Hostile case identifiers: every shape a case ID could take if a check ever fed a
# target-supplied or corpus-supplied string here. Sorted and de-duplicated, so the
# parametrisation order does not depend on the order the entry was edited in.
_HOSTILE_SPECS: tuple[str, ...] = (
    "",
    "\x00",
    "\n",
    "\t",
    ".",
    "..",
    "...",
    "../",
    "../../etc",
    "/etc/passwd",
    "C:\\temp\\x",
    "a b",
    "a/../../b",
    "corpus/../../x",
    "corpus:url-001",
    "..\\..\\windows",
    "*?<>|\"'",
    "\u00fcn\u00efcode",
    "\U0001f600",
)

HOSTILE_CASE_IDS: tuple[str, ...] = tuple(sorted(set(_HOSTILE_SPECS)))

# The components ``sanitize`` refuses outright: an empty component names nothing, and
# "." and ".." survive the character rule while still escaping or collapsing the
# directory they are joined to. They are unusable as a check ID and are asserted
# separately from the components that are merely sanitized (AAP §0.3.2, IM-7).
REFUSED_COMPONENTS: tuple[str, ...] = ("", ".", "..")

# The hostile components that survive :func:`sanitize` - the traversal attempts minus
# the three it refuses - usable as a check ID and as an idempotence sample alike.
USABLE_HOSTILE_COMPONENTS: tuple[str, ...] = tuple(
    component for component in HOSTILE_CASE_IDS if component not in REFUSED_COMPONENTS
)

# Every (check id, case id) pair that tries to leave the report directory, sorted so the
# parametrised order is fixed. Each pair is asserted by resolution, not by inspection.
TRAVERSAL_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (check_id, case_id)
        for check_id in USABLE_HOSTILE_COMPONENTS
        for case_id in HOSTILE_CASE_IDS
    )
)

# The characters §0.3.4's rule must replace, one sample each: path separators, quoting
# and shell metacharacters, a control character, a two-byte code point and a four-byte
# one. Sorted, so the ids are stable.
UNSAFE_CHARACTERS: tuple[str, ...] = tuple(
    sorted({":", "/", "\\", " ", "*", "?", '"', "'", "<", ">", "|", "\x00", "\n", "\t"})
) + ("\u00fc", "\U0001f600")

# Input lengths either side of the component bound, plus one far beyond it, so both the
# pass-through and the truncating branch are exercised.
COMPONENT_LENGTHS: tuple[int, ...] = (
    1,
    MAX_COMPONENT_LENGTH - 1,
    MAX_COMPONENT_LENGTH,
    MAX_COMPONENT_LENGTH + 1,
    5000,
)

# Case-ID lengths that far exceed the identifier bound, so the truncation branch is
# reached for the smallest and the largest shape worth pinning.
LONG_CASE_ID_LENGTHS: tuple[int, ...] = (500, 5000)


def _long_case_ids(length: int) -> tuple[str, str]:
    """Two case IDs of ``length`` characters that differ only in their final one.

    They share far more than the retained prefix, which is what turns the suffix into
    the only thing that keeps their evidence directories apart.
    """
    stem = "corpus:" + "e" * (length - len("corpus:") - 1)
    return f"{stem}a", f"{stem}b"


# Every shape the formula has to serve in one table: a member and a string reason, a
# missing reason, an empty triple, a hostile character, and lengths either side of the
# bound. Sorted by ``repr`` so the parametrisation order is fixed.
_FORMULA_SPECS: tuple[tuple[str, str, Reason | str | None], ...] = (
    ("", "", ""),
    ("CL-ARTIFACTS", "synthetic:run", Reason.CONTRACT_ERROR),
    ("DM-SEEDED", "synthetic:occluders-h", None),
    ("DT-DETERMINISTIC", "corpus:repeat-run-000", "value_mismatch"),
    ("IV-OVERCAPACITY", "synthetic:capacity-plus-one", Reason.CAPACITY_VIOLATION),
    ("MD-CHARSET", "corpus:utf8-multibyte-000", Reason.DECODER_DISAGREEMENT),
    ("RT-ROUNDTRIP", "\u00fcn\u00efcode", Reason.VALUE_MISMATCH),
    ("RT-ROUNDTRIP", "corpus:url-001", None),
    ("RT-ROUNDTRIP", "corpus:url-001", Reason.VALUE_MISMATCH),
    ("RT-ROUNDTRIP", "e" * LONG_CASE_ID_LENGTHS[0], "value_mismatch"),
    ("RT-ROUNDTRIP", "e" * LONG_CASE_ID_LENGTHS[1], "value_mismatch"),
    ("RN-QUIET-ZONE", "corpus:wifi-000", Reason.SAMPLING_FAILED),
    ("SY-EC-MATCH", "corpus:boundary-numeric-000", Reason.FORMAT_INFO_DISAGREEMENT),
    ("SY-MATRIX-EQUIV", "synthetic:v10-mask3-a", Reason.VALUE_MISMATCH),
)

FORMULA_CASES: tuple[tuple[str, str, Reason | str | None], ...] = tuple(
    sorted(_FORMULA_SPECS, key=repr)
)

# The triples whose three parts each survive :func:`sanitize` on their own, so the
# "already-sanitized input" property can be stated for them.
_RESANITIZABLE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("DM-SEEDED", "synthetic:occluders-h", "sampling_failed"),
    ("RN-QUIET-ZONE", "corpus:wifi-000", "value_mismatch"),
    ("RT-ROUNDTRIP", "corpus/url 001", "value_mismatch"),
    ("SY-EC-MATCH", "corpus:url-001", "format_info_disagreement"),
)

RESANITIZABLE_CASES: tuple[tuple[str, str, str], ...] = tuple(sorted(_RESANITIZABLE_SPECS))

# The two hash seeds the child interpreter runs under. PYTHONHASHSEED perturbs only
# CPython's own string hashing, so an identifier that agrees under both cannot be
# leaning on ``hash()`` or on dictionary order.
HASH_SEEDS: tuple[str, ...] = ("0", "12345")

# The triples the child interpreter computes: ordinary, hostile and long, so the
# comparison covers the truncating branch and the replacement rule as well as the
# plain join. Reason values are plain strings here because the table is embedded into
# the child's source with ``!a`` and has to be a literal.
_SUBPROCESS_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("CL-ARTIFACTS", "synthetic:run", ""),
    ("DM-SEEDED", "synthetic:occluders-h", None),
    ("DT-DETERMINISTIC", "../../etc/passwd", "value_mismatch"),
    ("IV-OVERCAPACITY", "synthetic:capacity-plus-one", "capacity_violation"),
    ("MD-CHARSET", "corpus:" + "e" * LONG_CASE_ID_LENGTHS[0], "value_mismatch"),
    ("RT-ROUNDTRIP", "corpus:url-001", "value_mismatch"),
    ("RN-QUIET-ZONE", "\u00fcn\u00efcode \U0001f600", "sampling_failed"),
    ("SY-EC-MATCH", "corpus:boundary-numeric-000", "format_info_disagreement"),
)

SUBPROCESS_CASES: tuple[tuple[str, str, str | None], ...] = tuple(
    sorted(_SUBPROCESS_SPECS, key=repr)
)

# The one module allowed to define a failure ID, repository-relative (AAP §0.8.1).
FAILURE_ID_OWNER: str = "qrcheck/report/model.py"

# The module that must consume it rather than re-derive it.
EVIDENCE_NAMING_MODULE: str = "qrcheck/evidence/naming.py"


# --------------------------------------------------------------------------- #
# The documented formula
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ORDINARY_CASES, ids=_label)
def test_the_identifier_joins_the_three_sanitized_segments(
    case: tuple[str, str, str],
) -> None:
    """The ID is the three inputs joined by ``__``, each segment sanitized on its own.

    The expectation is built from :func:`sanitize` rather than transcribed, so this
    states the rule and not one example: an ordinary triple has to equal exactly
    ``"__".join(sanitize(part) ...)``, and because none of these triples reaches the
    bound, it also has to equal the untruncated text with no suffix at all.
    """
    check_id, case_id, reason = case
    expected = "__".join(sanitize(part) for part in (check_id, case_id, reason))
    assert failure_id(check_id, case_id, reason) == expected
    assert len(expected) < FAILURE_ID_LIMIT, "the ordinary table must stay inside the bound"
    assert failure_id(check_id, case_id, reason) == reference_sanitize(
        _raw(check_id, case_id, reason)
    )


@pytest.mark.parametrize("case", ORDINARY_CASES, ids=_label)
def test_a_reason_member_and_its_string_value_produce_the_same_identifier(
    case: tuple[str, str, str],
) -> None:
    """A :class:`Reason` member and its string value are interchangeable as an input.

    Every caller may pass either - the registry resolves reasons from enums while a
    deserialised result carries plain strings - and the two must not file the same
    failure under two names.
    """
    check_id, case_id, reason = case
    assert failure_id(check_id, case_id, Reason(reason)) == failure_id(check_id, case_id, reason)


@pytest.mark.parametrize(
    "reason", sorted(Reason, key=lambda member: member.value), ids=lambda member: member.value
)
def test_every_reason_member_renders_its_declared_value(reason: Reason) -> None:
    """Every reason in the vocabulary renders as its own ``value``, with no prefix."""
    fid = failure_id("RT-ROUNDTRIP", "corpus:url-001", reason)
    assert fid.endswith("__" + reason.value)


def test_a_missing_reason_still_yields_a_usable_identifier() -> None:
    """A ``None`` reason renders ``none``, so the identifier is never malformed.

    A passing case writes no evidence, so ``None`` is a defensive default rather than
    a provenance; it still has to produce a string bounded and safe like any other.
    """
    fid = failure_id("RT-ROUNDTRIP", "corpus:url-001", None)
    assert fid == reference_failure_id("RT-ROUNDTRIP", "corpus:url-001", None)
    assert fid.endswith("_none")
    assert len(fid) <= FAILURE_ID_LIMIT


@pytest.mark.parametrize("case", FORMULA_CASES, ids=_label)
def test_the_documented_formula_is_reproduced(
    case: tuple[str, str, Reason | str | None],
) -> None:
    """``failure_id`` returns exactly the identifier the documented formula produces.

    Both branches are covered: the untruncated text for a short triple, and the retained
    prefix plus digest once the text exceeds the bound - for a 500- and a 5000-character
    case ID alike, which is what makes the bound independent of the input size.
    """
    check_id, case_id, reason = case
    assert failure_id(check_id, case_id, reason) == reference_failure_id(
        check_id, case_id, reason
    )


@pytest.mark.parametrize("case", FORMULA_CASES, ids=_label)
def test_a_failure_id_is_already_a_safe_path_component(
    case: tuple[str, str, Reason | str | None],
) -> None:
    """Re-sanitizing an identifier changes nothing, which is what the layout relies on.

    ``evidence_dir`` runs the identifier through ``sanitize`` again before joining it; if
    that second pass could alter the string, the name recorded in ``summary.json`` and
    the directory on disk would diverge.
    """
    fid = failure_id(*case)
    assert sanitize(fid) == fid
    assert re.fullmatch(r"[A-Za-z0-9._-]+", fid), fid


# --------------------------------------------------------------------------- #
# The character rule and its idempotence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("character", UNSAFE_CHARACTERS, ids=_label)
def test_each_unsafe_character_becomes_exactly_one_underscore(character: str) -> None:
    """One ``_`` replaces the character in place, leaving its neighbours alone."""
    assert sanitize(character) == "_"
    assert sanitize(f"a{character}b") == "a_b"
    assert len(sanitize(character)) == len(character)


@pytest.mark.parametrize("case_id", HOSTILE_CASE_IDS, ids=_label)
def test_a_hostile_case_id_leaves_nothing_a_filesystem_reads_as_a_path(
    case_id: str,
) -> None:
    """The identifier matches the safe set exactly and keeps the input's length.

    The length assertion states that the replacement is one-for-one per code point: a
    separator, a quote, a control character and a multi-byte character each cost the
    identifier a single character, so the bound counts characters rather than bytes
    (AAP §0.3.4, §0.3.2, IM-7).
    """
    raw = _raw("RT-ROUNDTRIP", case_id, "value_mismatch")
    fid = failure_id("RT-ROUNDTRIP", case_id, "value_mismatch")
    assert len(raw) <= FAILURE_ID_LIMIT, "the hostile table must stay inside the bound"
    assert fid == reference_sanitize(raw)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", fid), fid
    assert len(fid) == len(raw)


def test_a_multibyte_character_costs_one_character_not_one_byte() -> None:
    """The rule counts code points: a four-byte emoji becomes one ``_``, not four.

    This is what the implementation does - ``re`` matches the code points of a ``str`` -
    and it is stated and asserted rather than assumed, because a byte-length rule would
    silently shorten a bound expressed in characters.
    """
    for text in ("\u00fc", "\U0001f600", "a\u00fc\U0001f600b", "\u00fcn\u00efcode"):
        assert len(text.encode("utf-8")) >= len(text)
        assert sanitize(text) == reference_sanitize(text)
        assert len(sanitize(text)) == len(text)
    emoji = "\U0001f600"
    assert len(emoji.encode("utf-8")) > len(emoji), "the sample must be a multi-byte code point"
    assert sanitize(emoji) == "_"
    raw = _raw("RT-ROUNDTRIP", emoji, "value_mismatch")
    assert failure_id("RT-ROUNDTRIP", emoji, "value_mismatch") == reference_sanitize(raw)
    assert len(failure_id("RT-ROUNDTRIP", emoji, "value_mismatch")) == len(raw)


@pytest.mark.parametrize("component", USABLE_HOSTILE_COMPONENTS, ids=_label)
def test_sanitizing_an_already_sanitized_component_changes_nothing(component: str) -> None:
    """``sanitize`` is idempotent, so the second pass the layout makes cannot drift."""
    once = sanitize(component)
    assert sanitize(once) == once


@pytest.mark.parametrize("case", RESANITIZABLE_CASES, ids=_label)
def test_a_failure_id_of_already_sanitized_parts_is_unchanged(
    case: tuple[str, str, str],
) -> None:
    """Pre-sanitizing the three parts does not rename the failure.

    The evidence store and ``build_request`` may hand over a component that has already
    been reduced once; the identifier has to be the same either way, otherwise one
    failure would be filed under two names.
    """
    check_id, case_id, reason = case
    assert len(_raw(check_id, case_id, reason)) <= FAILURE_ID_LIMIT
    sanitized = tuple(sanitize(part) for part in (check_id, case_id, reason))
    assert failure_id(*sanitized) == failure_id(check_id, case_id, reason)


# --------------------------------------------------------------------------- #
# The bounds and the truncation suffix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("length", COMPONENT_LENGTHS)
def test_a_component_is_bounded_by_the_component_limit(length: int) -> None:
    """A component within the limit passes through; a longer one is truncated with a suffix.

    ``MAX_COMPONENT_LENGTH`` is the bound :func:`sanitize` enforces on one path segment,
    and the retained prefix absorbs the separator and the digest so the *whole* segment
    stays inside it.
    """
    component = "c" * length
    segment = sanitize(component)
    assert len(segment) <= MAX_COMPONENT_LENGTH
    if length <= MAX_COMPONENT_LENGTH:
        assert segment == component
        return
    assert len(segment) == MAX_COMPONENT_LENGTH
    retained, _, suffix = segment.rpartition("_")
    assert component.startswith(retained)
    assert len(retained) == MAX_COMPONENT_LENGTH - DIGEST_CHARACTERS - 1
    assert suffix == hashlib.sha256(component.encode("utf-8")).hexdigest()[:DIGEST_CHARACTERS]
    assert re.fullmatch(rf"[0-9a-f]{{{DIGEST_CHARACTERS}}}", suffix), suffix
    assert sanitize(segment) == segment


@pytest.mark.parametrize("length", LONG_CASE_ID_LENGTHS)
def test_a_long_case_id_stays_within_the_identifier_bound(length: int) -> None:
    """The whole identifier is bounded, not merely its retained prefix."""
    case_id = "corpus:" + "e" * (length - len("corpus:"))
    assert len(case_id) == length
    fid = failure_id("RT-ROUNDTRIP", case_id, "value_mismatch")
    assert len(fid) == FAILURE_ID_LIMIT
    assert re.fullmatch(r"[A-Za-z0-9._-]+", fid), fid
    pre_truncation = reference_sanitize(_raw("RT-ROUNDTRIP", case_id, "value_mismatch"))
    assert len(pre_truncation) > FAILURE_ID_LIMIT, "the sample must need truncating"
    retained, _, suffix = fid.rpartition("_")
    assert pre_truncation.startswith(retained)
    assert len(retained) == FAILURE_ID_LIMIT - DIGEST_CHARACTERS - 1
    assert suffix == hashlib.sha256(pre_truncation.encode("utf-8")).hexdigest()[
        :DIGEST_CHARACTERS
    ]
    assert re.fullmatch(rf"[0-9a-f]{{{DIGEST_CHARACTERS}}}", suffix), suffix


@pytest.mark.parametrize("length", LONG_CASE_ID_LENGTHS)
def test_the_truncation_suffix_is_deterministic(length: int) -> None:
    """The same input yields the same suffix on every call in the same process."""
    case_id = "corpus:" + "e" * (length - len("corpus:"))
    identifiers = {failure_id("RT-ROUNDTRIP", case_id, "value_mismatch") for _ in range(4)}
    assert len(identifiers) == 1, identifiers


@pytest.mark.parametrize("length", LONG_CASE_ID_LENGTHS)
def test_the_suffix_separates_inputs_that_differ_beyond_the_cut(length: int) -> None:
    """The digest covers the full pre-truncation text, so two long IDs cannot collide.

    Two case IDs sharing everything but their final character retain an identical prefix.
    If the suffix were taken from the retained text alone their evidence directories would
    be the same directory and one failure would overwrite the other.
    """
    first, second = (
        failure_id("RT-ROUNDTRIP", case_id, "value_mismatch")
        for case_id in _long_case_ids(length)
    )
    retained = FAILURE_ID_LIMIT - DIGEST_CHARACTERS - 1
    assert first != second
    assert first[:retained] == second[:retained]
    assert first[-DIGEST_CHARACTERS:] != second[-DIGEST_CHARACTERS:]


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case_id", HOSTILE_CASE_IDS, ids=_label)
def test_a_failure_id_is_never_empty_and_never_a_dot_segment(case_id: str) -> None:
    """No input produces an empty name, ``.`` or ``..`` - the three unusable segments.

    The identifier always carries the two ``__`` separators, so even an empty triple
    yields a non-empty string; the assertions here are what the layout may rely on
    without repeating the rule.
    """
    for reason in (Reason.VALUE_MISMATCH, None):
        fid = failure_id("SY-EC-MATCH", case_id, reason)
        assert fid, f"{case_id!r} produced an empty identifier"
        assert "/" not in fid, fid
        assert "\\" not in fid, fid
        assert fid not in (".", ".."), fid
        assert not fid.startswith("/"), fid


@pytest.mark.parametrize("component", REFUSED_COMPONENTS, ids=_label)
def test_a_degenerate_component_is_refused_by_the_layout(component: str) -> None:
    """An empty, ``.`` or ``..`` component raises instead of naming a directory.

    The character rule alone cannot reject these: ``.`` and ``..`` are inside the safe
    set and are exactly the two names that collapse or escape the directory they are
    joined to, so the layout refuses them by name (AAP §0.3.2, IM-7).
    """
    with pytest.raises(ValueError) as raised:
        sanitize(component)
    assert repr(component) in str(raised.value), str(raised.value)
    with pytest.raises(ValueError):
        evidence_dir("reports", component, failure_id("SY-EC-MATCH", "corpus:url-001", None))
    with pytest.raises(ValueError):
        relative_evidence_dir("SY-EC-MATCH", component)


def test_a_non_string_component_is_a_type_error() -> None:
    """A path component must be text; a number and a path object are not coerced."""
    for component in (None, 1, Path("x")):
        with pytest.raises(TypeError):
            sanitize(component)


@pytest.mark.parametrize("pair", TRAVERSAL_PAIRS, ids=_label)
def test_evidence_dir_never_leaves_the_report_directory(
    tmp_path: Path, pair: tuple[str, str]
) -> None:
    """Resolving the path is the assertion; inspecting the characters is not.

    A hostile check ID and a hostile case ID together must still name a directory under
    the configured report root, exactly three levels below it, because the layout is
    ``<report_dir>/evidence/<check-id>/<failure-id>`` (AAP §0.3.4).
    """
    check_id, case_id = pair
    root = tmp_path.resolve()
    fid = failure_id(check_id, case_id, Reason.VALUE_MISMATCH)
    absolute = evidence_dir(tmp_path, check_id, fid)
    relative = relative_evidence_dir(check_id, fid)
    assert absolute.resolve().is_relative_to(root), f"{check_id!r}/{case_id!r} -> {absolute}"
    assert (tmp_path / relative).resolve().is_relative_to(root), relative
    assert (tmp_path / relative).resolve() == absolute.resolve(), relative
    assert absolute.name == fid, absolute.name
    assert len(absolute.relative_to(tmp_path).parts) == 3, absolute


@pytest.mark.parametrize("pair", TRAVERSAL_PAIRS, ids=_label)
def test_case_evidence_dir_never_leaves_the_report_directory(
    tmp_path: Path, pair: tuple[str, str]
) -> None:
    """The entry point that derives the identifier is as safe as the one that takes it."""
    check_id, case_id = pair
    root = tmp_path.resolve()
    absolute = case_evidence_dir(tmp_path, check_id, case_id, Reason.VALUE_MISMATCH)
    relative = relative_case_evidence_dir(check_id, case_id, Reason.VALUE_MISMATCH)
    assert absolute.resolve().is_relative_to(root), f"{check_id!r}/{case_id!r} -> {absolute}"
    assert (tmp_path / relative).resolve().is_relative_to(root), relative
    assert (tmp_path / relative).resolve() == absolute.resolve(), relative
    assert absolute.name == failure_id(check_id, case_id, Reason.VALUE_MISMATCH)


def test_the_evidence_layout_is_evidence_then_check_then_failure(tmp_path: Path) -> None:
    """The layout is literally ``<report_dir>/evidence/<check-id>/<failure-id>``."""
    fid = failure_id("SY-EC-MATCH", "corpus:url-001", "value_mismatch")
    absolute = evidence_dir(tmp_path, "SY-EC-MATCH", fid)
    assert absolute.parts[-3:] == (EVIDENCE_DIR_NAME, "SY-EC-MATCH", fid)
    assert absolute.as_posix().endswith(f"{EVIDENCE_DIR_NAME}/SY-EC-MATCH/{fid}")
    assert absolute == tmp_path / EVIDENCE_DIR_NAME / "SY-EC-MATCH" / fid


def test_the_relative_form_reconstructs_the_absolute_path(tmp_path: Path) -> None:
    """``summary.json``'s relative form plus ``report_dir`` is the directory on disk."""
    fid = failure_id("SY-EC-MATCH", "corpus:url-001", "value_mismatch")
    relative = relative_evidence_dir("SY-EC-MATCH", fid)
    assert relative == "/".join((EVIDENCE_DIR_NAME, "SY-EC-MATCH", fid))
    assert "\\" not in relative, relative
    assert not relative.startswith("/"), relative
    assert tmp_path / relative == evidence_dir(tmp_path, "SY-EC-MATCH", fid)


def test_a_case_result_records_the_identifier_its_directory_carries() -> None:
    """The identifier in the result and the identifier in the path are one derivation."""
    relative = relative_case_evidence_dir(
        "SY-EC-MATCH", "corpus:url-001", Reason.VALUE_MISMATCH
    )
    result = CaseResult(
        check_id="SY-EC-MATCH",
        case_id="corpus:url-001",
        status=Status.FAILED,
        reason=Reason.VALUE_MISMATCH,
        evidence_dir=relative,
    )
    assert result.evidence_dir == relative
    assert relative.split("/")[-1] == failure_id(
        "SY-EC-MATCH", "corpus:url-001", Reason.VALUE_MISMATCH
    )


@pytest.mark.parametrize("case_id", HOSTILE_CASE_IDS, ids=_label)
def test_a_hostile_case_id_still_yields_a_record_the_model_accepts(case_id: str) -> None:
    """The relative form satisfies ``CaseResult``'s relative-posix-path invariant.

    ``CaseResult`` rejects an absolute path, a backslash and a ``..`` component. A hostile
    case ID has to come through that check, which is the last place the rendering of a
    ``case_id`` could still reach a report or name a directory.
    """
    relative = relative_case_evidence_dir("SY-EC-MATCH", case_id, Reason.VALUE_MISMATCH)
    result = CaseResult(
        check_id="SY-EC-MATCH",
        case_id=case_id,
        status=Status.FAILED,
        reason=Reason.VALUE_MISMATCH,
        evidence_dir=relative,
    )
    assert result.evidence_dir == relative


# --------------------------------------------------------------------------- #
# Determinism across processes
# --------------------------------------------------------------------------- #


def _subprocess_source() -> str:
    """Build the probe payload: the identifier table in, identifiers out.

    The table is embedded with ``!a`` - an ASCII-only ``repr`` - so the payload survives
    an ``argv`` entry whatever characters a case ID holds. It imports ``json`` and
    ``qrcheck.report.model`` and nothing else, so the comparison cannot be satisfied by
    an accident of import order in a larger program.
    """
    return (
        "import json\n"
        "\n"
        "from qrcheck.report.model import failure_id\n"
        f"\nCASES = {SUBPROCESS_CASES!a}\n"
        "\nprint(json.dumps([failure_id(*case) for case in CASES]))\n"
    )


def _identifiers_in_subprocess(hash_seed: str) -> list[str]:
    """Compute every table identifier in a fresh interpreter under one hash seed."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    # qrcheck is not installed into the interpreter's site-packages here, so the checkout
    # root must be reachable both as the working directory and on the path; neither
    # depends on who launched pytest or from where.
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(_REPOSITORY_ROOT), environment.get("PYTHONPATH", "")) if entry
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _subprocess_source()],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_REPOSITORY_ROOT),
            env=environment,
            timeout=120,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"computing identifiers under PYTHONHASHSEED={hash_seed!r} exited "
            f"{error.returncode}:\n{error.stderr or error.stdout}"
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"computing identifiers under PYTHONHASHSEED={hash_seed!r} did not finish "
            "within 120 s"
        )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("hash_seed", sorted(HASH_SEEDS))
def test_identifiers_do_not_depend_on_pythonhashseed(hash_seed: str) -> None:
    """A separate interpreter under a different hash seed produces the same identifiers.

    The suffix is a content digest, never ``hash()``: a ``hash()``-derived component would
    differ between the two child processes and between two machines, and the evidence
    directories it names are promised to be byte-stable (AAP §0.5.4).
    """
    expected = [failure_id(*case) for case in SUBPROCESS_CASES]
    assert _identifiers_in_subprocess(hash_seed) == expected


# --------------------------------------------------------------------------- #
# One definition only
# --------------------------------------------------------------------------- #


def _relative(path: Path) -> str:
    """The repository-relative POSIX form an offender entry starts with."""
    return path.relative_to(_REPOSITORY_ROOT).as_posix()


def _package_sources() -> tuple[tuple[Path, str], ...]:
    """Every ``.py`` file under ``qrcheck/``, in sorted path order, with its source."""
    return tuple(
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(path for path in _PACKAGE_DIR.rglob("*.py") if path.is_file())
    )


def _failure_id_definitions() -> tuple[str, ...]:
    """Every ``def`` or ``async def`` under ``qrcheck/`` whose name holds ``failure_id``.

    A parameter named ``failure_id`` is an ``ast.arg`` and is deliberately not collected:
    the two layout helpers take an identifier as an argument, they do not define one.
    """
    definitions: list[str] = []
    for path, source in _package_sources():
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                "failure_id" in node.name
            ):
                definitions.append(f"{_relative(path)}:{node.name}")
    return tuple(sorted(definitions))


def test_there_is_exactly_one_definition_of_a_failure_id() -> None:
    """A second definition would let two artefacts name one failure differently."""
    assert _failure_id_definitions() == (f"{FAILURE_ID_OWNER}:failure_id",)


def test_the_owner_publishes_the_identifier() -> None:
    """``failure_id`` is in the model's ``__all__``, so it is a declared public surface."""
    assert "failure_id" in tuple(model_module.__all__)


def _naming_module_usage(source: str) -> tuple[tuple[str, ...], int]:
    """The names ``naming.py`` imports from the model, and how often it calls one."""
    tree = ast.parse(source, filename=str(_NAMING_PATH))
    imported: set[str] = set()
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "qrcheck.report.model":
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id == "failure_id":
                calls += 1
            elif isinstance(function, ast.Attribute) and function.attr == "failure_id":
                calls += 1
    return tuple(sorted(imported)), calls


def test_the_evidence_layout_consumes_the_single_definition() -> None:
    """``naming.py`` imports and calls ``failure_id`` rather than re-deriving an ID.

    The import sits inside the two functions that need it, for the cycle-breaking reason
    the module documents, so the scan walks the whole tree instead of the module body
    alone.
    """
    imported, calls = _naming_module_usage(_NAMING_PATH.read_text(encoding="utf-8"))
    assert "failure_id" in imported, (
        f"{EVIDENCE_NAMING_MODULE} does not import failure_id from qrcheck.report.model; "
        f"it imports {imported!r}"
    )
    assert calls >= 1, f"{EVIDENCE_NAMING_MODULE} imports failure_id but never calls it"


def test_the_scan_is_not_vacuous() -> None:
    """A moved, emptied or renamed package cannot make the scan pass silently."""
    assert _PACKAGE_DIR.name == "qrcheck", _PACKAGE_DIR
    assert _MODEL_PATH == _PACKAGE_DIR / "report" / "model.py", _MODEL_PATH
    assert _NAMING_PATH == _PACKAGE_DIR / "evidence" / "naming.py", _NAMING_PATH
    files = tuple(path for path, _source in _package_sources())
    assert files, f"no Python file found under {_PACKAGE_DIR}"
    assert _MODEL_PATH in files, f"{len(files)} files scanned, none of them {_MODEL_PATH}"
    assert _NAMING_PATH in files, f"{len(files)} files scanned, none of them {_NAMING_PATH}"
