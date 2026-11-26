"""Corpus assembly: the ``CorpusRecord`` type and the 240-record corpus builder.

The corpus is the fixed input every corpus-driven check case is built from, so it must be a
pure function of the run seed: ``build_corpus(seed)`` returns the same records, in the same
order, in every process on every machine. This module owns three things and nothing else:

* the record type, ``CorpusRecord`` -- the only definition in the product, carrying the wire
  form the request fixes (``id``, ``class``, ``payload_utf8``/``payload_b64``,
  ``expected_mode``, ``notes``);
* the class inventory -- ``CLASS_COUNTS`` is the single home of the per-class record counts,
  and therefore of the corpus total and of the boundary subset success criterion 2 runs at
  EC L, Q and H through ``--class 'boundary-*'``;
* assembly, ``build_corpus`` -- which walks the payload generators of
  ``qrcheck/corpus/payloads.py``, assigns stable ids, orders the records by id, and then
  proves that the discriminating predicates the self-check targets depend on still hold.

Why the predicates are asserted here rather than only in a test: the defective target's
injected ``truncated_payload`` and ``wrong_ec_level`` defects are observable only through
records that make them so. A generator edit that quietly removed those records would leave
the suite green while proving less, so ``_verify_invariants`` counts the witnesses and
refuses to return a corpus without them.

No I/O, no printing, no configuration, no clock and no environment: writing the corpus to
disk is ``qrcheck/corpus/io.py``'s job, and the seed default and the corpus path belong to
``qrcheck/config``. Standard library plus the two corpus modules this package owns.
"""

__all__ = [
    "BOUNDARY_CLASSES",
    "BOUNDARY_RECORDS",
    "CLASS_COUNTS",
    "CLASS_NAMES",
    "CorpusError",
    "CorpusRecord",
    "MD_CHARSET_CLASSES",
    "TOTAL_RECORDS",
    "build_corpus",
]

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qrcheck.corpus import capacity, payloads

# The three boundary classes. A boundary payload is exactly as long as its version's
# encoding capacity at EC L, which makes its minimal version ``v`` at EC L and ``v + 1`` at
# EC M: that one-step difference is what lets the wrong-EC-level defect fail
# ``SY-VERSION-MIN`` as well as ``SY-EC-MATCH``, and it is why this subset is the one run at
# three error-correction levels.
BOUNDARY_CLASSES: tuple[str, ...] = (
    "boundary-alphanumeric",
    "boundary-byte",
    "boundary-numeric",
)

# The fourteen payload classes and how many records each contributes. The single home of the
# corpus size: every total below is derived from this mapping and no other module restates a
# count. Written in sorted-key order so ``repr`` and any future serialization are stable.
CLASS_COUNTS: dict[str, int] = {
    "boundary-alphanumeric": 12,
    "boundary-byte": 16,
    "boundary-numeric": 12,
    "emoji": 12,
    "mecard": 10,
    "nul-byte": 6,
    "otp": 12,
    "random-bytes": 40,
    "repeat-run": 30,
    "rtl": 8,
    "url": 30,
    "utf8-multibyte": 24,
    "vcard": 16,
    "wifi": 12,
}

# The corpus class names, sorted, as the vocabulary a record's class must belong to.
CLASS_NAMES: tuple[str, ...] = tuple(sorted(CLASS_COUNTS))

# Derived, never typed: 240 records in total, and the 40-record boundary subset that success
# criterion 2 runs at EC L, Q and H. The corpus is 200 non-boundary records plus that subset.
TOTAL_RECORDS: int = sum(CLASS_COUNTS.values())
BOUNDARY_RECORDS: int = sum(CLASS_COUNTS[class_name] for class_name in BOUNDARY_CLASSES)

# The classes ``MD-CHARSET`` draws its charset coverages from: ASCII from ``otp`` and
# ``boundary-alphanumeric``, UTF-8 multibyte from ``utf8-multibyte``, four-byte code points
# from ``emoji`` and from the widest ``utf8-multibyte`` payloads, right-to-left from ``rtl``
# and an embedded NUL from ``nul-byte``. The truncation witness below is looked for in these
# classes only, because they are the ones the truncating defect must be caught by.
MD_CHARSET_CLASSES: tuple[str, ...] = (
    "boundary-alphanumeric",
    "emoji",
    "nul-byte",
    "otp",
    "rtl",
    "utf8-multibyte",
)

# The length the ``truncated_payload`` defect cuts a longer payload back to. The corpus must
# hold at least one ``MD-CHARSET`` record whose truncation to this many bytes does not move
# its Version at every error-correction level where that is possible, or the defect is
# silently inert -- the injection would shorten the payload without changing the symbol's
# version, and no check would see it. Which levels those are is decided by the predicate
# itself, never by a list here; see ``_verify_truncation_witness``.
# The eight synthetic ``SY-MATRIX-EQUIV`` payloads are built by ``qrcheck/checks/symbol.py``
# and are 64 bytes or shorter by construction, so the injection can never reach them; that is
# another module's case set and is deliberately not represented here.
_TRUNCATION_BYTES: int = 64

# The wire keys of a record. ``class`` is the name the request fixes for the JSONL line, even
# though it cannot be a Python attribute name; ``payload_utf8`` and ``payload_b64`` are the
# mutually exclusive alternatives and exactly one of them is present.
_WIRE_KEYS: tuple[str, ...] = (
    "class",
    "expected_mode",
    "id",
    "notes",
    "payload_b64",
    "payload_utf8",
)
_REQUIRED_WIRE_KEYS: tuple[str, ...] = ("class", "expected_mode", "id")


class CorpusError(ValueError):
    """Raised when a record or a built corpus breaks one of the stated invariants.

    A ``ValueError`` subclass, so a caller that already catches the builtin -- the JSONL
    reader reconstructing records from file text, the CLI turning an unusable corpus into a
    configuration error -- needs no import of this module to handle it.
    """


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """One corpus record: a payload and the facts the checks need about it.

    The wire form spells the class key ``class``, the name the request fixes for the JSONL
    line, while the attribute here is ``class_name`` because ``class`` is a Python keyword;
    :meth:`to_dict` and :meth:`from_dict` are the only translation between the two, and
    every other key is spelled identically on both sides.

    ``expected_mode`` is the encoding mode the payload's own bytes are limited to
    (``numeric``, ``alphanumeric`` or ``byte``), and ``notes`` records how the payload was
    derived -- for a boundary record, the capacity arithmetic that fixed its length.

    Instances are immutable and validate themselves on construction: the ``id`` must be
    ``<class>-<3-digit index>`` for the record's own class, exactly one of the two payload
    fields must be set and it must carry at least one byte -- an empty ``payload_utf8``, an
    empty ``payload_b64`` and a ``payload_b64`` that decodes to nothing are all rejected --
    ``payload_utf8`` must hold no embedded NUL, ``payload_b64`` must be valid base64,
    ``expected_mode`` must be one of ``numeric``, ``alphanumeric`` and ``byte``, and ``notes``
    must be a str. A record read back from a malformed corpus line therefore fails where it is
    created rather than inside a check, and the build path and the load path share that one
    gate.
    """

    id: str
    class_name: str
    payload_utf8: str | None
    payload_b64: str | None
    expected_mode: str
    notes: str

    def __post_init__(self) -> None:
        """Raise CorpusError unless the record satisfies every stated invariant."""
        if not isinstance(self.id, str) or not self.id:
            raise CorpusError(f"id must be a non-empty str, got {_preview(self.id)}")
        if self.class_name not in CLASS_NAMES:
            raise CorpusError(
                f"record {_preview(self.id)}: class {_preview(self.class_name)} is not one of "
                f"the corpus classes {CLASS_NAMES}"
            )
        if not _id_matches_class(self.id, self.class_name):
            raise CorpusError(
                f"record id {_preview(self.id)} does not have the form "
                f"'<class>-<3-digit index>' for class {_preview(self.class_name)}"
            )
        if (self.payload_utf8 is None) == (self.payload_b64 is None):
            raise CorpusError(
                f"record {_preview(self.id)}: exactly one of payload_utf8 and payload_b64 must "
                f"be set, got payload_utf8={_preview(self.payload_utf8)}, "
                f"payload_b64={_preview(self.payload_b64)}"
            )
        if self.payload_utf8 is not None:
            if not isinstance(self.payload_utf8, str):
                raise CorpusError(
                    f"record {_preview(self.id)}: payload_utf8 must be a str, got "
                    f"{type(self.payload_utf8).__name__}"
                )
            if "\x00" in self.payload_utf8:
                raise CorpusError(
                    f"record {_preview(self.id)}: payload_utf8 holds an embedded NUL byte; a "
                    f"payload containing one must be stored as payload_b64"
                )
        if self.payload_b64 is not None:
            if not isinstance(self.payload_b64, str):
                raise CorpusError(
                    f"record {_preview(self.id)}: payload_b64 must be a str, got "
                    f"{type(self.payload_b64).__name__}"
                )
            _decode_base64(self.payload_b64, self.id)
        if not self.payload_bytes:
            raise CorpusError(
                f"record {_preview(self.id)}: payload_bytes is empty; every corpus payload "
                f"carries at least one byte"
            )
        if self.expected_mode not in capacity.MODES:
            raise CorpusError(
                f"record {_preview(self.id)}: expected_mode {_preview(self.expected_mode)} is "
                f"not one of {capacity.MODES}"
            )
        if not isinstance(self.notes, str):
            raise CorpusError(
                f"record {_preview(self.id)}: notes must be a str, got "
                f"{type(self.notes).__name__}"
            )

    @property
    def payload_bytes(self) -> bytes:
        """Return the payload as the bytes a symbol carries.

        ``payload_utf8`` is UTF-8 encoded -- precisely the bytes the encoder receives -- and
        ``payload_b64`` is decoded with validation, so a corrupt string fails loudly here
        instead of yielding the garbage a lenient decode would. Never returns ``None``;
        raises CorpusError only for a record built around the constructor's invariants.
        """
        if self.payload_b64 is not None:
            return _decode_base64(self.payload_b64, self.id)
        if self.payload_utf8 is None:
            raise CorpusError(
                f"record {_preview(self.id)}: neither payload_utf8 nor payload_b64 is set"
            )
        return self.payload_utf8.encode("utf-8")

    def to_dict(self) -> dict[str, object]:
        """Return the record in its wire form, with the class key spelled ``class``.

        Exactly five keys: ``id``, ``class``, ``expected_mode`` and ``notes``, plus exactly
        one of ``payload_utf8`` and ``payload_b64`` -- the absent alternative is omitted
        rather than written as null, so a record states its payload in one way only.
        """
        wire: dict[str, object] = {
            "id": self.id,
            "class": self.class_name,
            "expected_mode": self.expected_mode,
            "notes": self.notes,
        }
        if self.payload_utf8 is not None:
            wire["payload_utf8"] = self.payload_utf8
        if self.payload_b64 is not None:
            wire["payload_b64"] = self.payload_b64
        return wire

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CorpusRecord":
        """Rebuild a record from its wire form.

        Reads the class from the key ``class`` and requires ``id``, ``class`` and
        ``expected_mode``; ``notes`` may be absent and is then empty; exactly one of
        ``payload_utf8`` and ``payload_b64`` must carry a string, and a key spelled any other
        way is rejected rather than silently dropped.

        Raises CorpusError when the mapping has an unknown key, is missing a required key,
        holds a value of the wrong type, or describes a record that breaks an invariant.
        """
        if not isinstance(raw, Mapping):
            raise CorpusError(
                f"a corpus record must be a mapping, got {type(raw).__name__}"
            )
        unknown = sorted(str(key) for key in raw if key not in _WIRE_KEYS)
        if unknown:
            raise CorpusError(
                f"unknown corpus record key(s) {unknown}; expected keys drawn from "
                f"{_WIRE_KEYS}"
            )
        missing = [key for key in _REQUIRED_WIRE_KEYS if key not in raw]
        if missing:
            raise CorpusError(
                f"missing corpus record key(s) {missing}; expected keys drawn from "
                f"{_WIRE_KEYS}"
            )
        notes = _optional_wire_str(raw, "notes")
        return cls(
            id=_required_wire_str(raw, "id"),
            class_name=_required_wire_str(raw, "class"),
            payload_utf8=_optional_wire_str(raw, "payload_utf8"),
            payload_b64=_optional_wire_str(raw, "payload_b64"),
            expected_mode=_required_wire_str(raw, "expected_mode"),
            notes="" if notes is None else notes,
        )


def build_corpus(seed: int) -> list[CorpusRecord]:
    """Assemble the whole corpus from `seed`.

    Guarantees the record set every corpus-driven check is defined against: exactly
    ``TOTAL_RECORDS`` records, per-class counts equal to ``CLASS_COUNTS``, ids of the form
    ``<class>-<3-digit index>`` in ascending order, exactly one payload field per record, and
    both discriminating witnesses the defect-detection contract requires.

    Pure: the result is a function of `seed` and of the capacity tables alone, so two calls
    with the same `seed` return equal records in the same order, in any process, on any
    machine.

    Raises CorpusError when `seed` is not a non-negative integer, when a corpus class has no
    payload generator, or when the assembled corpus fails an invariant.
    """
    _require_seed(seed)
    _require_generator_coverage()
    records: list[CorpusRecord] = []
    for class_name in sorted(CLASS_COUNTS):
        generate = payloads.CLASS_GENERATORS[class_name]
        for index in range(CLASS_COUNTS[class_name]):
            draft = generate(seed, index)
            payload_utf8, payload_b64 = _payload_fields(draft.payload)
            records.append(
                CorpusRecord(
                    id=f"{class_name}-{index:03d}",
                    class_name=class_name,
                    payload_utf8=payload_utf8,
                    payload_b64=payload_b64,
                    expected_mode=draft.expected_mode,
                    notes=draft.notes,
                )
            )
    # The walk above is already in id order; sorting states the ordering as a contract rather
    # than leaving it to follow from the shape of the loop.
    records.sort(key=_id_key)
    _verify_invariants(records)
    return records


def _verify_invariants(records: Sequence[CorpusRecord]) -> None:
    """Raise CorpusError unless `records` is the corpus the checks depend on.

    Every clause positively counts records that satisfy its predicate, so none of them can
    pass on a corpus that merely lacks what it looks for: the counts must match the
    inventory, ids must be unique and ordered, payloads must be stated one way, and both
    discriminating witnesses must exist. Validation reads `records` as given -- sorted or
    not -- so a perturbed corpus handed to this function is rejected rather than trusted.
    """
    _verify_counts(records)
    _verify_ids(records)
    _verify_payload_fields(records)
    _verify_truncation_witness(records)
    _verify_version_level_witness(records)


def _verify_counts(records: Sequence[CorpusRecord]) -> None:
    """Raise CorpusError unless the record counts match the class inventory exactly."""
    counts: dict[str, int] = {class_name: 0 for class_name in CLASS_NAMES}
    for record in records:
        if record.class_name not in counts:
            raise CorpusError(
                f"record {_preview(record.id)}: class {_preview(record.class_name)} is not "
                f"one of the corpus classes {CLASS_NAMES}"
            )
        counts[record.class_name] += 1
    for class_name in CLASS_NAMES:
        if counts[class_name] != CLASS_COUNTS[class_name]:
            raise CorpusError(
                f"class {class_name!r} holds {counts[class_name]} record(s), CLASS_COUNTS "
                f"requires {CLASS_COUNTS[class_name]}"
            )
    if len(records) != TOTAL_RECORDS:
        raise CorpusError(
            f"corpus holds {len(records)} record(s), TOTAL_RECORDS is {TOTAL_RECORDS}"
        )
    boundary = sum(counts[class_name] for class_name in BOUNDARY_CLASSES)
    if boundary != BOUNDARY_RECORDS:
        raise CorpusError(
            f"the boundary subset holds {boundary} record(s), BOUNDARY_RECORDS is "
            f"{BOUNDARY_RECORDS}"
        )


def _verify_ids(records: Sequence[CorpusRecord]) -> None:
    """Raise CorpusError unless the ids are well formed, unique and in ascending order."""
    previous: str | None = None
    for record in records:
        if not _id_matches_class(record.id, record.class_name):
            raise CorpusError(
                f"record id {_preview(record.id)} does not have the form "
                f"'<class>-<3-digit index>' for class {_preview(record.class_name)}"
            )
        if previous is not None and record.id == previous:
            raise CorpusError(f"record id {_preview(record.id)} appears more than once")
        if previous is not None and record.id < previous:
            raise CorpusError(
                f"record id {_preview(record.id)} follows {_preview(previous)}, so the "
                f"corpus is not sorted by id"
            )
        previous = record.id


def _verify_payload_fields(records: Sequence[CorpusRecord]) -> None:
    """Raise CorpusError unless each record states one payload field the wire allows.

    The non-empty rule is the record's own invariant, enforced by
    ``CorpusRecord.__post_init__`` for a freshly built corpus and for a corpus read back from
    disk alike; this function keeps the corpus-wide payload-field checks -- exactly one field
    set, and no embedded NUL in ``payload_utf8`` -- so a record set that reached here by any
    other route is still held to the wire's two payload rules.
    """
    for record in records:
        if (record.payload_utf8 is None) == (record.payload_b64 is None):
            raise CorpusError(
                f"record {_preview(record.id)}: exactly one of payload_utf8 and payload_b64 "
                f"must be set, got payload_utf8={_preview(record.payload_utf8)}, "
                f"payload_b64={_preview(record.payload_b64)}"
            )
        if record.payload_utf8 is not None and "\x00" in record.payload_utf8:
            raise CorpusError(
                f"record {_preview(record.id)}: payload_utf8 holds an embedded NUL byte; a "
                f"payload containing one must be stored as payload_b64"
            )


def _verify_truncation_witness(records: Sequence[CorpusRecord]) -> None:
    """Raise CorpusError unless a record makes the truncating defect observable.

    The ``truncated_payload`` defect cuts a payload longer than ``_TRUNCATION_BYTES`` back to
    exactly that many bytes, under the rule ``capacity.truncation_keeps_version`` -- the same
    function the defect evaluates, so the witness cannot be computed by a second predicate
    that drifts from the injection. It stays observable only where the shortened payload
    still needs the same Version at the error-correction level in use, so that
    ``RT-ROUNDTRIP`` and ``MD-CHARSET`` fail on the payload and ``SY-VERSION-MIN`` keeps its
    single cause.

    At least one ``MD-CHARSET`` record must therefore witness the cut at every level in
    ``capacity.EC_LEVELS`` where a witness is arithmetically possible. A level is impossible
    when the window is itself the byte capacity of a Version there: the very next byte
    already needs the following Version, so no payload longer than the window can keep its
    version -- at EC H the 64-byte window fills Version 7 exactly. That condition is read
    from the same predicate applied to the shortest longer payload, never from a list of
    levels, so the clause cannot pass by silently excusing a level that could have been
    witnessed.
    """
    witnesses: dict[str, list[str]] = {}
    for record in records:
        if record.class_name not in MD_CHARSET_CLASSES:
            continue
        payload = record.payload_bytes
        if len(payload) <= _TRUNCATION_BYTES:
            continue
        for ec in capacity.EC_LEVELS:
            if capacity.truncation_keeps_version(len(payload), _TRUNCATION_BYTES, ec):
                witnesses.setdefault(ec, []).append(record.id)
    possible = [
        ec
        for ec in capacity.EC_LEVELS
        if capacity.truncation_keeps_version(_TRUNCATION_BYTES + 1, _TRUNCATION_BYTES, ec)
    ]
    missing = [ec for ec in possible if not witnesses.get(ec)]
    if missing:
        raise CorpusError(
            f"no MD-CHARSET record longer than {_TRUNCATION_BYTES} bytes keeps its Version "
            f"when truncated to {_TRUNCATION_BYTES} bytes at EC {', '.join(missing)}, so the "
            f"truncated_payload defect would be inert there (classes searched: "
            f"{MD_CHARSET_CLASSES}; levels witnessed: "
            f"{', '.join(ec for ec in capacity.EC_LEVELS if witnesses.get(ec)) or 'none'})"
        )


def _verify_version_level_witness(records: Sequence[CorpusRecord]) -> None:
    """Raise CorpusError unless a boundary record separates EC L from EC M by version.

    The ``wrong_ec_level`` defect encodes one level below the request. ``SY-EC-MATCH``
    catches that on its own, but ``SY-VERSION-MIN`` catches it only through a record whose
    minimal version differs between the requested level and the level below it -- which is
    exactly the arithmetic a boundary payload is built for. At least one such record must
    exist, or that half of the expected failure set could never be produced.
    """
    witnesses: list[str] = []
    for record in records:
        if record.class_name not in BOUNDARY_CLASSES:
            continue
        payload = record.payload_bytes
        mode = payloads.mode_of(payload)
        version_l = capacity.minimal_version(len(payload), "L", mode)
        version_m = capacity.minimal_version(len(payload), "M", mode)
        if version_l is None or version_m is None:
            continue
        if version_l != version_m:
            witnesses.append(record.id)
    if not witnesses:
        raise CorpusError(
            f"no boundary record needs a different minimal version at EC L and EC M, so the "
            f"wrong_ec_level defect could not fail SY-VERSION-MIN (boundary classes: "
            f"{BOUNDARY_CLASSES})"
        )


def _require_seed(seed: int) -> None:
    """Raise CorpusError unless `seed` is an int, and not a bool, that is non-negative."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CorpusError(f"seed must be an int, got {_preview(seed)}")
    if seed < 0:
        raise CorpusError(f"seed must be non-negative, got {seed}")


def _require_generator_coverage() -> None:
    """Raise CorpusError unless every corpus class has a payload generator and vice versa."""
    declared = sorted(CLASS_COUNTS)
    available = sorted(payloads.CLASS_GENERATORS)
    if declared == available:
        return
    raise CorpusError(
        f"CLASS_COUNTS and payloads.CLASS_GENERATORS name different classes: only in "
        f"CLASS_COUNTS {sorted(set(declared) - set(available))}, only in CLASS_GENERATORS "
        f"{sorted(set(available) - set(declared))}"
    )


def _payload_fields(payload: str | bytes) -> tuple[str | None, str | None]:
    """Return the ``(payload_utf8, payload_b64)`` pair for one generated payload.

    One mechanical rule decides which field carries it: text that a UTF-8 codec round-trips
    and that holds no NUL goes to ``payload_utf8``, so a regenerated corpus diffs readably
    and its ordinary payloads stay legible in review; ``bytes``, text with an embedded NUL
    and text no UTF-8 codec can represent go to ``payload_b64``, which keeps a line-oriented
    JSONL file free of raw NULs and of sequences that are not valid UTF-8.
    """
    if isinstance(payload, bytes):
        return None, _encode_base64(payload)
    if "\x00" not in payload and _utf8_round_trips(payload):
        return payload, None
    return None, _encode_base64(_utf8_bytes(payload))


def _utf8_round_trips(value: str) -> bool:
    """Return whether a UTF-8 codec returns `value` unchanged.

    False only for text holding a lone surrogate, which the codec refuses to encode; every
    string the corpus generators build returns True.
    """
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return encoded.decode("utf-8") == value


def _utf8_bytes(value: str) -> bytes:
    """Return the bytes a str payload contributes to a symbol.

    ``surrogatepass`` is reached only for text the strict codec refused, where it is the one
    encoding that still yields the bytes a caller could have meant; for every other string
    this is the same strict UTF-8 encoding a symbol carries.
    """
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        return value.encode("utf-8", "surrogatepass")


def _encode_base64(payload: bytes) -> str:
    """Return `payload` as the ASCII base64 text a record stores."""
    return base64.b64encode(payload).decode("ascii")


def _decode_base64(encoded: str, record_id: str) -> bytes:
    """Return the bytes `encoded` carries, raising CorpusError when it is not valid base64.

    Validation is strict, so a corrupted string fails instead of decoding to something
    plausible; ``binascii.Error`` is a ``ValueError`` subclass, so one clause covers both a
    malformed alphabet and a malformed padding.
    """
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CorpusError(
            f"record {_preview(record_id)}: payload_b64 is not valid base64 ({exc})"
        ) from exc


def _id_matches_class(record_id: str, class_name: str) -> bool:
    """Return whether `record_id` is ``<class_name>-<3-digit index>``.

    The index must be exactly three ASCII digits and must survive a round trip through
    ``int``, so a splatter such as ``url-7`` or ``url-007x`` is rejected while ``url-007``
    is accepted.
    """
    if not isinstance(record_id, str):
        return False
    prefix = f"{class_name}-"
    if not record_id.startswith(prefix):
        return False
    suffix = record_id[len(prefix) :]
    if len(suffix) != 3 or not suffix.isascii() or not suffix.isdigit():
        return False
    return f"{int(suffix):03d}" == suffix


def _id_key(record: CorpusRecord) -> str:
    """Return the key the corpus is ordered by: the record id, ascending."""
    return record.id


def _required_wire_str(raw: Mapping[str, object], key: str) -> str:
    """Return the string value at `key`, raising CorpusError when it is missing or not str."""
    value = raw[key]
    if not isinstance(value, str):
        raise CorpusError(
            f"corpus record key {key!r} must be a str, got {type(value).__name__}"
        )
    return value


def _optional_wire_str(raw: Mapping[str, object], key: str) -> str | None:
    """Return the string value at `key`, None when absent or null, else raise CorpusError."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CorpusError(
            f"corpus record key {key!r} must be a str when present, got "
            f"{type(value).__name__}"
        )
    return value


def _preview(value: object, limit: int = 60) -> str:
    """Return a short one-line repr of `value` for a diagnostic message."""
    text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text

