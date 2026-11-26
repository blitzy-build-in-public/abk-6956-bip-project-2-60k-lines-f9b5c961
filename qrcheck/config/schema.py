"""The declarative configuration schema: every key, its type, default and rule.

This module holds the single declaration of what ``qrcheck.toml`` may contain and
of the value each key accepts.  ``qrcheck.config.load`` reads the TOML document,
calls :func:`validate`, then :func:`apply_defaults`, and only then builds the
frozen records of :mod:`qrcheck.config.models`; no other module in the product
enumerates a configuration key, and no validator of an individual key lives
anywhere else.

Two rules shape the code below:

* Every default *value* is declared once, as a ``models.DEFAULT_*`` constant, and
  every entry of :data:`SCHEMA` references that name rather than restating a
  number.  This module owns the *table* -- which key, which type, which
  validator, which default constant -- never a copy of a default.  A threshold
  therefore changes in one place plus the acceptance criterion that reads it.
* The document's structure is derived from the group paths of :data:`SCHEMA`
  instead of being listed a second time.  A top-level table exists because some
  group path starts with its name, and a child of ``[checks]`` exists because a
  group path names it, so there is no second list to forget a key in when a
  table gains one.

Neither function performs I/O and neither raises anything but
:class:`ConfigError`:

* No path is touched.  ``corpus_path`` existence is selection-dependent ("must
  exist when a corpus-driven check is selected"), so the runner raises it as its
  run-prerequisite error; ``report_dir`` creation and its writability probe are
  ``load.py``'s work and surface as ``OSError``/``PermissionError``, which the CLI
  maps to exit 3 rather than to a configuration error.
* No exit code, no printing, no ``sys.exit``: exit decisions belong solely to
  ``qrcheck.cli.exit``, which builds its message from ``str(exc)``.
* No capability *shortfall* validation.  Whether a declared capability set can
  satisfy a check depends on the run-time ``--checks`` selection, so the same
  document would be invalid for one selection and correct for the next; that
  condition is a ``skipped`` result with reason ``unsupported_capability``.  An
  unknown capability *member* is a configuration error, because it is wrong
  whoever selects what.

Determinism is a contract here too: the schema is walked in its declared order
and unknown keys are reported over ``sorted()``, so the first diagnostic raised
for a given document is always the same one, on every machine and every run.
No clock, locale, ``hash()``, ``id()`` or filesystem order reaches a message, and
neither public function mutates its argument.
"""

from __future__ import annotations

import datetime
import math
import string
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from qrcheck.config import models

__all__ = ["ConfigError", "KeySpec", "SCHEMA", "apply_defaults", "validate"]


# ---------------------------------------------------------------------------
# The one error type
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """A configuration fault, attributed to the dotted path of the offending key.

    Raised by :func:`validate` for anything the document gets wrong on its own
    terms, and by ``qrcheck.config.load`` for the whole-file faults of an
    unreadable file or a TOML syntax error.  ``key`` is the dotted path a reader
    has to edit -- ``targets.foo.returns``, ``checks.damage.jpeg_quality`` -- so
    the exit-2 diagnostic built from ``str(exc)`` names exactly what to fix,
    which is the entire point of the exit-2 contract.

    ``key`` is empty only for a fault that belongs to no single key: a document
    whose root is not a table, or a file that cannot be read or parsed.
    """

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key
        self.message = message

    def __str__(self) -> str:
        return f"{self.key}: {self.message}" if self.key else self.message


# ---------------------------------------------------------------------------
# One key's declaration
# ---------------------------------------------------------------------------


class _RequiredSentinel:
    """The type of :data:`_REQUIRED`: marks a key the document has to supply."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED: Final[_RequiredSentinel] = _RequiredSentinel()


@dataclass(frozen=True)
class KeySpec:
    """Declared type, default and rule for one configuration key.

    ``kinds`` are the value types accepted *after* TOML parsing, so an integer
    is written ``(int,)``, a number that may be written either way ``(float,
    int)`` -- TOML spells ``timeout_s = 30`` as an integer -- and a table
    ``(dict,)``.  ``default`` is either a ``models.DEFAULT_*`` constant, whose
    value this module never restates, or :data:`_REQUIRED`.

    ``check`` receives the value alone and returns the error fragment that
    :class:`ConfigError` attaches to the key, or ``None`` when the value is
    valid.  Every fragment is a self-contained sentence fragment naming the
    offending value, and type checking happens before it (compare
    :func:`_validate_value`), so a ``check`` may assume its declared ``kinds``.
    The one check that has to name a *nested* leaf -- ``request_template`` --
    raises :class:`_LeafError` to do so; see :func:`_check_request_template`.
    """

    key: str
    kinds: tuple[type, ...]
    default: object
    check: Callable[[Any], str | None]

    @property
    def is_required(self) -> bool:
        """Whether the document must supply this key, because it has no default."""
        return self.default is _REQUIRED


# ---------------------------------------------------------------------------
# Diagnostic vocabulary
#
# A message says what was expected and what was found.  Value kinds are named in
# the reader's terms ("table", "array") rather than the parser's, and booleans are
# named before integers because ``bool`` subclasses ``int``.
# ---------------------------------------------------------------------------

_KIND_LABELS: Final[tuple[tuple[type, str], ...]] = (
    (bool, "boolean"),
    (datetime.datetime, "datetime"),
    (datetime.date, "date"),
    (datetime.time, "time"),
    (str, "string"),
    (int, "integer"),
    (float, "number"),
    (list, "array"),
    (Mapping, "table"),
    (type(None), "null"),
)

_KIND_LABEL_BY_TYPE: Final[Mapping[type, str]] = MappingProxyType(dict(_KIND_LABELS))


def _kind(value: object) -> str:
    """Name what ``value`` is, in the terms a configuration message uses."""
    for kind, label in _KIND_LABELS:
        if isinstance(value, kind):
            return label
    return type(value).__name__


def _label_for(kind: type) -> str:
    """Name one declared type in the reader's vocabulary, not the parser's.

    A declared type the map spells out is answered directly.  A table key is
    declared ``(dict,)`` -- the type TOML actually produces -- and ``dict`` is not a
    key of the map, so a declared type absent from it is matched to the first label
    it is a subclass of and still reads as ``table``; only a type that matches no
    label at all falls back to its own name.
    """
    label = _KIND_LABEL_BY_TYPE.get(kind)
    if label is not None:
        return label
    if isinstance(kind, type):
        for base, base_label in _KIND_LABELS:
            if issubclass(kind, base):
                return base_label
    return kind.__name__


def _expected_label(kinds: tuple[type, ...]) -> str:
    """Name the accepted types as one phrase: ``string``, ``number``, ``table``."""
    labels = list(dict.fromkeys(_label_for(kind) for kind in kinds))
    # ``(float, int)`` is one concept -- a number -- not two alternatives.
    if labels in (["integer", "number"], ["number", "integer"]):
        return "number"
    return " or ".join(labels)


def _fmt(number: float) -> str:
    """Render a bound or a value in a message without exponent noise."""
    return format(number, "g")


def _dotted(prefix: str, key: str) -> str:
    """Join a table's dotted path with one of its keys."""
    return f"{prefix}.{key}" if prefix else str(key)


# ---------------------------------------------------------------------------
# Generic validators
#
# Each takes what it needs and returns a ``Callable[[Any], str | None]`` matching
# ``KeySpec.check``, so a range, a membership set or a composition of the two is
# declared in the table instead of hand-written per key.
# ---------------------------------------------------------------------------


def _type_fragment(kinds: tuple[type, ...], value: Any) -> str | None:
    """Return the type diagnostic for ``value``, or ``None`` when it fits ``kinds``.

    ``bool`` is rejected for every key even where an integer is accepted: TOML
    ``true`` parses to ``bool``, which is an ``int`` subclass, so without this
    ``warmup = true`` would silently pass as ``1``.  A float must also be finite,
    because TOML can express ``inf`` and ``nan`` and neither is a usable bound.
    """
    if isinstance(value, bool):
        return f"expected {_expected_label(kinds)}, got boolean"
    if not isinstance(value, kinds):
        return f"expected {_expected_label(kinds)}, got {_kind(value)}"
    if isinstance(value, float) and not math.isfinite(value):
        return f"must be a finite number, got {value!r}"
    return None


def _all_of(*checks: Callable[[Any], str | None]) -> Callable[[Any], str | None]:
    """Combine validators, keeping the first fragment any of them returns."""

    def check(value: Any) -> str | None:
        for validator in checks:
            fragment = validator(value)
            if fragment is not None:
                return fragment
        return None

    return check


def _check_minimum(minimum: float, *, strict: bool) -> Callable[[Any], str | None]:
    """Require a number at least ``minimum``, or strictly greater when ``strict``."""
    relation = "greater than" if strict else "at least"

    def check(value: float) -> str | None:
        if value < minimum or (strict and value == minimum):
            return f"must be {relation} {_fmt(minimum)}, got {_fmt(value)}"
        return None

    return check


def _check_maximum(maximum: float, *, strict: bool = False) -> Callable[[Any], str | None]:
    """Require a number at most ``maximum``, or strictly less when ``strict``."""
    relation = "less than" if strict else "at most"

    def check(value: float) -> str | None:
        if value > maximum or (strict and value == maximum):
            return f"must be {relation} {_fmt(maximum)}, got {_fmt(value)}"
        return None

    return check


def _check_membership(allowed: tuple[str, ...]) -> Callable[[Any], str | None]:
    """Require a string drawn from a closed vocabulary declared in ``models``."""
    permitted = sorted(allowed)

    def check(value: str) -> str | None:
        if value not in allowed:
            return f"invalid value '{value}'; allowed: {permitted}"
        return None

    return check


def _check_non_empty_str(value: str) -> str | None:
    """Require a string with at least one non-whitespace character."""
    if not value.strip():
        return f"must be a non-empty string, got {value!r}"
    return None


def _check_path(value: str) -> str | None:
    """Require a non-empty path that a filesystem call can actually use.

    A POSIX file name may hold any byte except ``/`` and NUL, so the other control
    characters are legal in a path and are deliberately not rejected here.  A NUL is
    not: no filesystem call accepts one, and ``pathlib`` answers it with a
    ``ValueError`` -- which is not the ``OSError`` the exit-3 contract covers -- so
    an operator's ``report_dir = "rep\\u0000orts"`` would surface as an internal
    error instead of a diagnostic naming the key.  Rejecting it here keeps the fault
    on the key it belongs to.
    """
    fragment = _check_non_empty_str(value)
    if fragment is not None:
        return fragment
    if "\x00" in value:
        return f"must not contain a NUL byte, got {value!r}"
    return None


# ---------------------------------------------------------------------------
# Validators for individual keys
# ---------------------------------------------------------------------------


def _check_seed(value: int) -> str | None:
    """Require a non-negative run seed; the message is fixed by the contract."""
    if value < 0:
        return "seed must be a non-negative integer"
    return None


def _check_capabilities(value: list) -> str | None:
    """Require every declared capability to be one of the 13 member names.

    Only membership is checked.  Whether the declared set can satisfy a selected
    check is not decidable here -- it depends on ``--checks`` -- and is a
    ``skipped`` result with reason ``unsupported_capability`` instead.  A repeated
    member is accepted because ``TargetConfig`` stores a ``frozenset``.
    """
    permitted = sorted(models.CAPABILITY_NAMES)
    for member in value:
        if not isinstance(member, str):
            return f"every capability must be a string, got {_kind(member)}"
        if member not in models.CAPABILITY_NAMES:
            return f"unknown capability '{member}'; allowed: {permitted}"
    return None


def _check_callable(value: str) -> str | None:
    """Require a syntactically valid ``module:function`` import path.

    Syntax only: whether the module and attribute exist is answered by
    ``qrcheck doctor`` and, at run time, by ``TargetUnavailable`` -- a missing
    module must not fail configuration loading, because the same document is
    valid on a machine where the target is installed.
    """
    if value.count(":") != 1:
        return f"expected exactly one ':' as 'module:function', got {value!r}"
    module, attribute = value.split(":")
    if not module or not attribute:
        return f"expected both parts of 'module:function', got {value!r}"
    for part in (*module.split("."), *attribute.split(".")):
        if not part.isidentifier():
            return f"'{part}' is not a valid Python identifier in {value!r}"
    return None


def _check_command(value: list) -> str | None:
    """Require a non-empty argv list of non-empty strings."""
    if not value:
        return "must list at least one argument"
    for index, member in enumerate(value):
        if not isinstance(member, str):
            return f"argument {index} must be a string, got {_kind(member)}"
        if not member:
            return f"argument {index} must not be an empty string"
    return None


def _check_arg_template(value: str) -> str | None:
    """Require an argument template that carries a payload or names a flag.

    The rule is the contract's own: a template either contains ``{}`` for the
    value or starts with ``-`` and is passed as a flag.  The substitution itself
    belongs to the CLI adapter.
    """
    if "{}" in value or value.startswith("-"):
        return None
    return f"must contain '{{}}' or start with '-', got {value!r}"


def _check_str_map(value: Mapping[str, str]) -> str | None:
    """Require a table of non-empty strings to non-empty strings.

    Shape only: which framework field names exist is not a closed set in the
    contract, and interpreting a mapping is the shared resolution in
    ``qrcheck.targets.base``, so no vocabulary is invented here.
    """
    for key, mapped in value.items():
        if not isinstance(key, str) or not key.strip():
            return f"every key must be a non-empty string, got {key!r}"
        if not isinstance(mapped, str) or not mapped.strip():
            return f"the value for '{key}' must be a non-empty string, got {mapped!r}"
    return None


def _check_image_field(value: str) -> str | None:
    """Require ``_body`` or a dotted path of non-empty, whitespace-free segments.

    Shape only, like :func:`_check_str_map`: following the path through a decoded
    response is the HTTP adapter's work.
    """
    if value == "_body":
        return None
    segments = value.split(".")
    for segment in segments:
        if not segment or any(character.isspace() for character in segment):
            return f"expected '_body' or a dotted path of non-empty segments, got {value!r}"
    return None


# The only hosts the framework may ever talk to.  A target that is not on this
# list cannot be reached by the product at all, which is what makes the tool
# verifiably offline rather than merely configured to be.
_LOOPBACK_HOSTS: Final[tuple[str, ...]] = ("127.0.0.1", "localhost", "::1")


def _check_url(value: str) -> str | None:
    """Require a loopback ``http``/``https`` URL that carries no userinfo.

    Two security controls, both enforced at configuration time:

    * The host must be ``127.0.0.1``, ``localhost`` or ``::1``, so the framework
      cannot be aimed at a remote host and stays fully offline.  ``urlsplit``
      lower-cases the host and strips the brackets of an IPv6 literal, so
      ``http://[::1]:8000/`` is accepted and ``http://example.com/gen`` is not.
    * A ``user:password@`` prefix is rejected, because a credential in a
      configuration file is outside the product's no-credential boundary; there
      is nothing to authenticate to on loopback.
    """
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError as exc:
        return f"is not a parseable URL: {exc}"
    if parts.scheme not in ("http", "https"):
        return f"url scheme '{parts.scheme}' is not http or https; allowed: ['http', 'https']"
    if parts.username is not None or parts.password is not None:
        return "url must not carry a username or password; configuration holds no credentials"
    host = parts.hostname
    if host is None or host.lower() not in _LOOPBACK_HOSTS:
        return f"url host '{host or ''}' is not a loopback host; allowed: {sorted(_LOOPBACK_HOSTS)}"
    return None


class _LeafError(Exception):
    """A fault at one nested leaf of ``request_template``.

    ``suffix`` is appended to the key's dotted path (``.image``,
    ``.emitted[0].name``) so the diagnostic names the leaf rather than the whole
    table, and ``fragment`` is the message :class:`ConfigError` carries.  It is
    the one exception :func:`validate` catches other than its own.
    """

    def __init__(self, suffix: str, fragment: str) -> None:
        super().__init__(fragment)
        self.suffix = suffix
        self.fragment = fragment


def _check_request_template(value: Mapping[str, Any]) -> str | None:
    """Require every string leaf of the template to name a known placeholder.

    The template is sent as JSON, so the walk accepts exactly what JSON carries:
    tables, arrays, strings, numbers, booleans and null.  A TOML datetime is
    rejected by name, an infinite or NaN number is rejected, and any other value
    type is named and refused.

    A field may carry a conversion or a format spec -- ``{size:d}`` and
    ``{payload_utf8!s}`` are accepted -- because ``string.Formatter.parse``
    reports the field name alone; only the name is checked against
    ``models.TEMPLATE_FIELDS``.  Escaped braces (``{{``/``}}``) yield no field and
    are literal text.
    """
    _walk_template(value, "")
    return None


def _walk_template(node: Any, suffix: str) -> None:
    """Check one template node, raising :class:`_LeafError` at the first fault."""
    if isinstance(node, Mapping):
        for key, child in node.items():
            if not isinstance(key, str) or not key:
                raise _LeafError(suffix, "every request_template key must be a non-empty string")
            _walk_template(child, f"{suffix}.{key}")
        return
    if isinstance(node, (list, tuple)):
        for index, child in enumerate(node):
            _walk_template(child, f"{suffix}[{index}]")
        return
    if isinstance(node, str):
        _check_template_string(node, suffix)
        return
    if isinstance(node, (int, float)):
        # ``bool`` arrives here too, which is right: JSON carries booleans.
        if isinstance(node, float) and not math.isfinite(node):
            raise _LeafError(suffix, f"must be a finite number, got {node!r}")
        return
    if isinstance(node, (datetime.date, datetime.time)):
        raise _LeafError(suffix, "TOML dates and times cannot be sent as JSON; write a string")
    raise _LeafError(suffix, f"unsupported value of type {_kind(node)}")


def _check_template_string(text: str, suffix: str) -> None:
    """Require every ``{}`` field in one template string to be a known placeholder."""
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as exc:
        raise _LeafError(suffix, f"malformed placeholder string {text!r}: {exc}") from None
    for _literal, field_name, _format_spec, _conversion in parsed:
        if field_name is None:
            continue
        if field_name not in models.TEMPLATE_FIELDS:
            raise _LeafError(
                suffix,
                f"unknown placeholder '{field_name}' in request_template; "
                f"allowed: {sorted(models.TEMPLATE_FIELDS)}",
            )


# ---------------------------------------------------------------------------
# The declared tables
#
# One entry per key of the configuration contract, in the order the contract
# tabulates them.  A table is built from its ``KeySpec`` list so a key is named
# once -- in its spec -- and the mapping key can never disagree with it.
# ---------------------------------------------------------------------------


def _group(*specs: KeySpec) -> Mapping[str, KeySpec]:
    """Build one table from its declarations, keyed by each spec's own key."""
    return {spec.key: spec for spec in specs}


# Top level.  The two path keys are non-empty strings that a filesystem call can use
# -- a NUL byte is the one character refused -- and nothing more: a path is resolved
# against the configuration file's directory by ``load.py``, and neither ``~`` nor
# ``$VAR`` is expanded or promised.
TOP_LEVEL: Final[Mapping[str, KeySpec]] = _group(
    KeySpec("default_adapter", (str,), models.DEFAULT_ADAPTER, _check_non_empty_str),
    KeySpec("report_dir", (str,), models.DEFAULT_REPORT_DIR, _check_path),
    KeySpec("seed", (int,), models.DEFAULT_SEED, _check_seed),
    KeySpec("corpus_path", (str,), models.DEFAULT_CORPUS_PATH, _check_path),
    KeySpec("loc_limit", (int,), models.DEFAULT_LOC_LIMIT, _check_minimum(1, strict=False)),
)

# [render]: how RN-SVG rasterizes SVG output.
RENDER: Final[Mapping[str, KeySpec]] = _group(
    KeySpec(
        "svg_rasterizer",
        (str,),
        models.DEFAULT_SVG_RASTERIZER,
        _check_membership(models.SVG_RASTERIZERS),
    ),
)

# [checks.performance]: PF-ENCODE and PF-DECODE read every number here, so a
# floor changes in ``models`` alone.  A count may be an integer only; a latency or
# a throughput floor may be written either way and is cast to float by ``load.py``.
PERFORMANCE: Final[Mapping[str, KeySpec]] = _group(
    KeySpec(
        "encode_p95_ms", (float, int), models.DEFAULT_ENCODE_P95_MS, _check_minimum(0, strict=True)
    ),
    KeySpec(
        "decode_p95_ms", (float, int), models.DEFAULT_DECODE_P95_MS, _check_minimum(0, strict=True)
    ),
    KeySpec(
        "min_symbols_per_minute",
        (float, int),
        models.DEFAULT_MIN_SYMBOLS_PER_MINUTE,
        _check_minimum(0, strict=True),
    ),
    KeySpec(
        "encode_samples", (int,), models.DEFAULT_ENCODE_SAMPLES, _check_minimum(10, strict=False)
    ),
    KeySpec(
        "decode_samples", (int,), models.DEFAULT_DECODE_SAMPLES, _check_minimum(10, strict=False)
    ),
    KeySpec("warmup", (int,), models.DEFAULT_WARMUP, _check_minimum(0, strict=False)),
    KeySpec(
        "decode_image_px", (int,), models.DEFAULT_DECODE_IMAGE_PX, _check_minimum(64, strict=False)
    ),
)

# [checks.damage]: the eight thresholds DM-RESILIENCE, DM-SEEDED and RN-LOGO read.
# ``occlusion_margin`` is bounded above *exclusively* at 0.10, so the safety margin
# can never consume the whole documented L-level recovery capacity; the logo
# fraction, the rotation and the perspective warp are bounded above inclusively at
# the values the contract states.
DAMAGE: Final[Mapping[str, KeySpec]] = _group(
    KeySpec(
        "occlusion_margin",
        (float, int),
        models.DEFAULT_OCCLUSION_MARGIN,
        _all_of(_check_minimum(0, strict=False), _check_maximum(0.10, strict=True)),
    ),
    KeySpec(
        "blur_radius_per_module",
        (float, int),
        models.DEFAULT_BLUR_RADIUS_PER_MODULE,
        _check_minimum(0, strict=True),
    ),
    KeySpec(
        "jpeg_quality",
        (int,),
        models.DEFAULT_JPEG_QUALITY,
        _all_of(_check_minimum(1, strict=False), _check_maximum(95)),
    ),
    KeySpec(
        "noise_sigma", (float, int), models.DEFAULT_NOISE_SIGMA, _check_minimum(0, strict=False)
    ),
    KeySpec(
        "rotation_deg",
        (float, int),
        models.DEFAULT_ROTATION_DEG,
        _all_of(_check_minimum(0, strict=True), _check_maximum(45)),
    ),
    KeySpec(
        "perspective_deg",
        (float, int),
        models.DEFAULT_PERSPECTIVE_DEG,
        _all_of(_check_minimum(0, strict=True), _check_maximum(30)),
    ),
    KeySpec(
        "downscale_px_per_module",
        (float, int),
        models.DEFAULT_DOWNSCALE_PX_PER_MODULE,
        _check_minimum(0, strict=True),
    ),
    KeySpec(
        "logo_area_fraction",
        (float, int),
        models.DEFAULT_LOGO_AREA_FRACTION,
        _all_of(_check_minimum(0, strict=True), _check_maximum(0.25)),
    ),
)

# Defaults for the two optional collection keys.  ``capabilities`` is declared as
# an empty tuple and ``metadata_map`` as a read-only empty mapping, so the table
# holds no mutable default and ``_materialise`` can hand each resolution a private
# copy.  Only ``capabilities`` is reached from below: ``metadata_map`` is an
# optional *per-type* key whose default each record in ``qrcheck.config.models``
# already owns, so ``apply_defaults`` never fills it.
_EMPTY_CAPABILITIES: Final[tuple[str, ...]] = ()
_EMPTY_MAPPING: Final[Mapping[str, str]] = MappingProxyType({})

# [targets.<name>], common to every adapter kind.  ``type`` is the only required
# key and it is what selects a target's per-type table below.
TARGET_COMMON: Final[Mapping[str, KeySpec]] = _group(
    KeySpec("type", (str,), _REQUIRED, _check_membership(models.ADAPTER_TYPE_NAMES)),
    KeySpec("timeout_s", (float, int), models.DEFAULT_TIMEOUT_S, _check_minimum(0, strict=True)),
    KeySpec(
        "contrast_floor",
        (float, int),
        models.DEFAULT_CONTRAST_FLOOR,
        _check_minimum(1.0, strict=True),
    ),
    KeySpec("capabilities", (list,), _EMPTY_CAPABILITIES, _check_capabilities),
)

# type = "library": an in-process import and call.
TARGET_LIBRARY: Final[Mapping[str, KeySpec]] = _group(
    KeySpec("callable", (str,), _REQUIRED, _check_callable),
    KeySpec("kwargs_map", (dict,), _REQUIRED, _check_str_map),
    KeySpec("returns", (str,), _REQUIRED, _check_membership(models.RETURNS_KINDS)),
    KeySpec("metadata_map", (dict,), _EMPTY_MAPPING, _check_str_map),
)

# type = "cli": an argv-only subprocess.
TARGET_CLI: Final[Mapping[str, KeySpec]] = _group(
    KeySpec("command", (list,), _REQUIRED, _check_command),
    KeySpec("payload_arg", (str,), _REQUIRED, _check_arg_template),
    KeySpec("output_arg", (str,), _REQUIRED, _check_arg_template),
    KeySpec("returns", (str,), _REQUIRED, _check_membership(models.RETURNS_KINDS)),
    KeySpec("metadata_map", (dict,), _EMPTY_MAPPING, _check_str_map),
)

# type = "http": a loopback request whose vendor shape lives entirely here.
TARGET_HTTP: Final[Mapping[str, KeySpec]] = _group(
    KeySpec("url", (str,), _REQUIRED, _check_url),
    KeySpec("method", (str,), _REQUIRED, _check_membership(models.HTTP_METHODS)),
    KeySpec("request_template", (dict,), _REQUIRED, _check_request_template),
    KeySpec("image_field", (str,), _REQUIRED, _check_image_field),
    KeySpec("image_encoding", (str,), _REQUIRED, _check_membership(models.IMAGE_ENCODINGS)),
    KeySpec("metadata_map", (dict,), _EMPTY_MAPPING, _check_str_map),
)

# The two built-in self-check targets take only the common keys, so a configurable
# input on them would be a key with no effect; an empty table makes it an error.
TARGET_REFERENCE: Final[Mapping[str, KeySpec]] = _group()
TARGET_DEFECTIVE: Final[Mapping[str, KeySpec]] = _group()


# ---------------------------------------------------------------------------
# The one exported table
#
# Keyed by the group path the document itself spells, with "" for the top level.
# Everything below derives the document's structure from these keys, so adding a
# table here is the whole of adding a table to the contract.
# ---------------------------------------------------------------------------

SCHEMA: Final[Mapping[str, Mapping[str, KeySpec]]] = {
    "": TOP_LEVEL,
    "render": RENDER,
    "checks.performance": PERFORMANCE,
    "checks.damage": DAMAGE,
    "targets.*": TARGET_COMMON,
    "targets.*.library": TARGET_LIBRARY,
    "targets.*.cli": TARGET_CLI,
    "targets.*.http": TARGET_HTTP,
    "targets.*.reference": TARGET_REFERENCE,
    "targets.*.defective": TARGET_DEFECTIVE,
}

# The non-scalar top-level names, and the children of [checks], derived from the
# group paths of SCHEMA in their declared order.  ``targets`` is derived like the
# others but walked separately, because its children are the target names the
# document declares rather than names fixed here.
_TOP_LEVEL_TABLES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(path.split(".", 1)[0] for path in SCHEMA if path)
)
_CHECK_TABLES: Final[tuple[str, ...]] = tuple(
    path.split(".")[1] for path in SCHEMA if path.startswith("checks.")
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_table(dotted: str, value: Any) -> Mapping[str, Any]:
    """Return ``value`` as a table, or raise naming ``dotted`` and what arrived.

    A TOML array-of-tables written where a table belongs (``[[targets]]``) parses
    to a list and is reported as ``expected table, got array``.  Keys of a table
    are always strings once TOML has parsed them, and requiring that here is what
    lets every later walk sort them.
    """
    if not isinstance(value, Mapping):
        raise ConfigError(dotted, f"expected table, got {_kind(value)}")
    for key in value:
        if not isinstance(key, str):
            raise ConfigError(dotted, f"table keys must be strings, got {_kind(key)}")
    return value


def _validate_value(dotted: str, spec: KeySpec, value: Any) -> None:
    """Check one value against its spec: declared type first, then its rule."""
    fragment = _type_fragment(spec.kinds, value)
    if fragment is not None:
        raise ConfigError(dotted, fragment)
    try:
        fragment = spec.check(value)
    except _LeafError as exc:
        raise ConfigError(f"{dotted}{exc.suffix}", exc.fragment) from None
    if fragment is not None:
        raise ConfigError(dotted, fragment)


def _reject_unknown_keys(
    dotted: str, table: Mapping[str, Any], declared: Mapping[str, KeySpec]
) -> None:
    """Raise on the first key of ``table`` that ``declared`` does not hold.

    Sorted, so a document with two mistakes always reports the same one.
    """
    for key in sorted(table):
        if key not in declared:
            full = _dotted(dotted, key)
            raise ConfigError(full, f"unknown key '{full}'")


def _reject_missing_required(
    dotted: str, table: Mapping[str, Any], declared: Mapping[str, KeySpec]
) -> None:
    """Raise on the first required key ``table`` does not supply, in declared order.

    Declared order -- never the document's order or a set -- is what keeps the first
    diagnostic for a given document identical on every run and machine.
    """
    for key, spec in declared.items():
        if spec.is_required and key not in table:
            raise ConfigError(_dotted(dotted, key), f"missing required key '{key}'")


def _validate_table(
    dotted: str, table: Mapping[str, Any], declared: Mapping[str, KeySpec]
) -> None:
    """Validate one declared table: unknown keys, missing required keys, then values."""
    _reject_unknown_keys(dotted, table, declared)
    _reject_missing_required(dotted, table, declared)
    for key, spec in declared.items():
        if key in table:
            _validate_value(_dotted(dotted, key), spec, table[key])


def _validate_target(dotted: str, table: Mapping[str, Any]) -> None:
    """Validate one ``[targets.<name>]`` table against its ``type``.

    ``type`` is read first because it selects the per-type keys, so a key that
    belongs to another adapter kind -- ``command`` on a ``library`` target -- is
    an unknown key rather than a silently inert one.  The table's own keys are then
    rejected when unknown, its required per-type keys are rejected when absent --
    ``callable`` for a ``library`` target, ``command`` for a ``cli`` target, ``url``
    for an ``http`` target -- and only then are the values present validated.
    """
    if "type" not in table:
        raise ConfigError(f"{dotted}.type", "missing required key 'type'")
    _validate_value(f"{dotted}.type", TARGET_COMMON["type"], table["type"])
    declared = {**TARGET_COMMON, **SCHEMA.get(f"targets.*.{table['type']}", {})}
    _reject_unknown_keys(dotted, table, declared)
    _reject_missing_required(dotted, table, declared)
    for key, spec in declared.items():
        # ``type`` is both validated above and the value that chose ``declared``.
        if key != "type" and key in table:
            _validate_value(_dotted(dotted, key), spec, table[key])


def validate(raw: Mapping[str, Any]) -> None:
    """Validate a parsed ``qrcheck.toml`` document, raising :class:`ConfigError`.

    The document is checked against :data:`SCHEMA` and nothing else.  A missing
    key is not an error unless it is required: every other key has a declared
    default, so an empty document is structurally valid right up to the cross-key
    rule below.

    The order of work is fixed so the first diagnostic is stable: unknown
    top-level keys, then the top-level values in declared order, then
    ``[render]``, then ``[checks]`` and its children, then ``[targets]``, and
    last the cross-key rule.  Unknown keys are reported over ``sorted()``.

    The cross-key rule is that ``default_adapter`` must name a declared target.
    Its consequence is deliberate: a document with no ``[targets.*]`` table
    cannot satisfy it, because the fallback ``default_adapter`` names nothing, so
    a usable configuration declares at least one target -- ``qrcheck.toml``
    declares both built-in ones.

    Raises:
        ConfigError: for any key that is unknown, missing when required, of the
            wrong type, outside its enum or range, syntactically invalid, on a
            non-loopback URL, or naming an undeclared adapter.  Nothing else is
            raised: no file is opened, no directory is probed and no exit code is
            decided here.
    """
    document = _require_table("", raw)

    declared_top = {*TOP_LEVEL, *_TOP_LEVEL_TABLES}
    for key in sorted(document):
        if key not in declared_top:
            raise ConfigError(key, f"unknown key '{key}'")
    for key, spec in TOP_LEVEL.items():
        if key in document:
            _validate_value(key, spec, document[key])

    if "render" in document:
        _validate_table("render", _require_table("render", document["render"]), RENDER)

    targets: Mapping[str, Any] = {}
    if "checks" in document:
        checks = _require_table("checks", document["checks"])
        for key in sorted(checks):
            if key not in _CHECK_TABLES:
                full = _dotted("checks", key)
                raise ConfigError(full, f"unknown key '{full}'")
        for child in _CHECK_TABLES:
            if child in checks:
                path = f"checks.{child}"
                _validate_table(path, _require_table(path, checks[child]), SCHEMA[path])

    if "targets" in document:
        targets = _require_table("targets", document["targets"])
        for name in sorted(targets):
            if not name.strip():
                raise ConfigError(
                    "targets", f"target name must be a non-empty string, got {name!r}"
                )
            dotted = f"targets.{name}"
            _validate_target(dotted, _require_table(dotted, targets[name]))

    default_adapter = document.get("default_adapter", models.DEFAULT_ADAPTER)
    declared_names = sorted(targets)
    if default_adapter not in declared_names:
        raise ConfigError(
            "default_adapter",
            f"unknown adapter '{default_adapter}'; declared: {declared_names}",
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _materialise(default: object) -> object:
    """Return a private copy of a declared default, so no two results alias one value.

    A collection default reaches this function only when the table that declares it
    is one :func:`apply_defaults` fills, and the only such key today is
    ``capabilities``: declared as an empty tuple and resolved to a list per target.
    ``metadata_map`` is declared the same way -- a read-only empty mapping that
    would resolve to a plain dict -- but it is an optional per-type key, so no table
    filled here holds it and its default is its record's in
    ``qrcheck.config.models``.  Scalars are immutable and are returned unchanged,
    and no value is ever coerced to another type.
    """
    if isinstance(default, tuple):
        return list(default)
    if isinstance(default, Mapping):
        return dict(default)
    return default


def _table_or_empty(value: Any) -> Mapping[str, Any]:
    """Return ``value`` as a table, or an empty table when it is absent."""
    return value if isinstance(value, Mapping) else {}


def _apply_group(values: Mapping[str, Any], declared: Mapping[str, KeySpec]) -> dict[str, Any]:
    """Return one table with every declared default filled in, supplied values kept.

    Supplied keys keep their order and a key that has no default -- a required one
    -- stays exactly as supplied.
    """
    completed: dict[str, Any] = dict(values)
    for key, spec in declared.items():
        if not spec.is_required and key not in completed:
            completed[key] = _materialise(spec.default)
    return completed


def apply_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new document with every declared default filled in.

    The result is a fresh nested ``dict`` holding the five top-level keys, a
    complete ``[render]`` table, complete ``[checks.performance]`` and
    ``[checks.damage]`` tables, and, in :data:`SCHEMA`'s declared order, one
    ``[targets.<name>]`` table per declared target carrying every key of
    :data:`TARGET_COMMON`.  Required keys -- a target's ``type`` and its per-type
    inputs -- are carried over as supplied and are never defaulted.

    A per-type *optional* default is deliberately not filled in: the contract's
    only one, a ``library``/``cli``/``http`` target's ``metadata_map``, is owned by
    each record of ``qrcheck.config.models``, which declares its own empty mapping,
    and ``load.py`` passes the key on to that record only when the document supplies
    it.  Filling it in here would declare one default in two places.

    Pre-condition: :func:`validate` has already accepted ``raw``.  This function
    therefore validates nothing, raises nothing and fills no required key: a
    document it is handed that skipped validation resolves to whatever it holds
    plus the defaults, which keeps it total for a caller that composes the two
    in the wrong order rather than failing late with a different error type.
    ``load.py`` is its only caller.

    ``raw`` is never mutated and no value is coerced, so the loader alone decides
    that ``timeout_s`` becomes a ``float`` and ``capabilities`` a ``frozenset``.

    Args:
        raw: The parsed TOML document, already accepted by :func:`validate`.

    Returns:
        A new document: every table it builds is a new ``dict``, while a value the
        document supplies is carried over as it stands, so a supplied list or table
        (``capabilities``, ``request_template``) is one object shared with ``raw``.
    """
    document = _table_or_empty(raw)
    resolved: dict[str, Any] = {}
    for key, spec in TOP_LEVEL.items():
        resolved[key] = document[key] if key in document else _materialise(spec.default)

    resolved["render"] = _apply_group(_table_or_empty(document.get("render")), RENDER)

    checks = _table_or_empty(document.get("checks"))
    resolved["checks"] = {
        child: _apply_group(_table_or_empty(checks.get(child)), SCHEMA[f"checks.{child}"])
        for child in _CHECK_TABLES
    }

    targets = _table_or_empty(document.get("targets"))
    resolved["targets"] = {
        name: _apply_group(_table_or_empty(targets[name]), TARGET_COMMON)
        for name in sorted(targets)
    }
    return resolved
