"""Configuration vocabulary, threshold defaults and immutable records for qrcheck.

This module is the deepest foundation of the package: ``qrcheck.targets``,
``qrcheck.corpus``, ``qrcheck.checks``, ``qrcheck.registry``, ``qrcheck.cli``
and ``qrcheck.runner`` all read the records defined here, while this module
imports nothing from ``qrcheck`` at all -- not even inside a type-checking
guard -- so the dependency direction can never close into a cycle.  That is
equally why a capability travels here as its member *name* rather than as the
adapter layer's ``Capability`` enumeration: ``qrcheck.targets`` may import
``qrcheck.config`` and the reverse would close the cycle.  The adapter converts
with ``Capability(name)``, and because that enumeration is string-valued the
converted set compares and hashes equal to the ``frozenset[str]`` kept here, so
capability intersection and containment work across the seam unchanged.

Every configurable threshold has one fallback, declared as a ``DEFAULT_*``
constant below.  ``qrcheck.config.schema`` references these names instead of
restating the numbers, and no check module carries a numeric threshold
literal; the effective value a run uses is the committed ``qrcheck.toml``'s, so
changing a limit means changing that file's entry plus the acceptance criterion
that reads it.

The dataclasses are plain frozen records: they validate nothing, read neither
TOML nor the filesystem, and perform no I/O.  Validation is the gate in
``qrcheck.config.schema.validate``, invoked by ``qrcheck.config.load`` before
any record is built -- the same split as ``GenerateRequest`` in
``qrcheck.targets.base``, which validates nothing itself because
``build_request()`` is its gate.  Nothing here raises a domain error; the only
failure a caller can provoke is the ordinary
:class:`dataclasses.FrozenInstanceError` from attempting to mutate a record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Mapping, get_args

# ---------------------------------------------------------------------------
# Vocabulary
#
# Each ``Literal`` alias states a closed set once, and the runtime tuple beside
# it is derived from that alias with ``get_args`` so no member name is written
# twice.  ``qrcheck.config.schema`` validates against the tuples; the adapters
# consume the same values.
# ---------------------------------------------------------------------------

# ``[targets.<name>] type``: the five kinds the target port is implemented by.
AdapterType = Literal["library", "cli", "http", "reference", "defective"]

# ``returns`` (library and CLI adapters): how the target hands back its
# artefact -- a path to bytes, a path to a file, base64 text, or raw bytes.
ReturnsKind = Literal["bytes_path", "file_path", "base64", "raw_bytes"]

# ``method`` (HTTP adapter).
HttpMethod = Literal["GET", "POST"]

# ``image_encoding`` (HTTP adapter): how the response payload carries the
# image, as base64 text or as the raw response body.
ImageEncoding = Literal["base64", "raw"]

# ``[render] svg_rasterizer``.  ``auto`` discovers in the documented order:
# an importable ``cairosvg`` module first, then ``rsvg-convert``, ``resvg``,
# ``magick`` and ``convert`` on ``PATH``; ``none`` disables rasterizing, which
# makes RN-SVG report a skip rather than a pass.
SVGRasterizer = Literal[
    "auto",
    "cairosvg",
    "rsvg-convert",
    "resvg",
    "magick",
    "none",
]

# The 13 capabilities an adapter may declare for itself.  These are member
# names, not the adapter layer's enumeration, so this package stays free of any
# import from ``qrcheck.targets``.
CapabilityName = Literal[
    "png",
    "svg",
    "ec_level",
    "module_size",
    "quiet_zone",
    "color",
    "logo_overlay",
    "version_pin",
    "mask_pin",
    "metadata_version",
    "metadata_ec",
    "metadata_mask",
    "metadata_mode",
]

# A ``metadata_map`` key: the four framework metadata fields a target may report
# about a symbol.  The set is closed, because a field the framework does not know
# cannot be carried into evidence, so the schema rejects any other key at load
# time and ``qrcheck.targets.base`` aliases this tuple as ``METADATA_FIELDS``.
MetadataFieldName = Literal["version", "ec", "mask", "mode"]

ADAPTER_TYPE_NAMES: Final[tuple[str, ...]] = get_args(AdapterType)
RETURNS_KINDS: Final[tuple[str, ...]] = get_args(ReturnsKind)
HTTP_METHODS: Final[tuple[str, ...]] = get_args(HttpMethod)
IMAGE_ENCODINGS: Final[tuple[str, ...]] = get_args(ImageEncoding)
SVG_RASTERIZERS: Final[tuple[str, ...]] = get_args(SVGRasterizer)
CAPABILITY_NAMES: Final[tuple[str, ...]] = get_args(CapabilityName)
METADATA_FIELD_NAMES: Final[tuple[str, ...]] = get_args(MetadataFieldName)

# The eight placeholders an HTTP ``request_template`` may interpolate.  The
# schema validates every template leaf against this tuple and the HTTP adapter
# fills placeholders from the same tuple, so the two definitions cannot drift.
TEMPLATE_FIELDS: Final[tuple[str, ...]] = (
    "payload_b64",
    "payload_utf8",
    "ec_level",
    "format",
    "size",
    "margin",
    "fg",
    "bg",
)

# ---------------------------------------------------------------------------
# Outbound allow-list
# ---------------------------------------------------------------------------

# The only hosts the framework may ever talk to.  This list is the product's
# single outbound control -- an HTTP target is accepted only when its host is one
# of these, which is what makes the tool verifiably offline rather than merely
# configured to be -- so it is declared exactly once, here, and read by both
# layers that enforce it: ``qrcheck.config.schema`` when a document is validated
# and ``qrcheck.targets.http`` when an ``HTTPTarget`` is built without the
# loader.  Neither layer restates a host name, so the online/offline boundary
# cannot widen in one place while staying closed in the other.
LOOPBACK_HOSTS: Final[tuple[str, ...]] = ("127.0.0.1", "localhost", "::1")

# ---------------------------------------------------------------------------
# Threshold defaults
#
# Each constant is the fallback for exactly one configuration key, used when
# ``qrcheck.toml`` omits it, and is the one place the fallback is written down;
# the effective value a run uses is that file's whenever it supplies the key.
# ``qrcheck.config.schema`` builds its table from these names rather than
# retyping the numbers.
# ---------------------------------------------------------------------------

DEFAULT_ADAPTER: Final[str] = "reference"  # default_adapter
DEFAULT_REPORT_DIR: Final[str] = "reports"  # report_dir
DEFAULT_SEED: Final[int] = 20240101  # seed
DEFAULT_CORPUS_PATH: Final[str] = "corpus/default.jsonl"  # corpus_path
DEFAULT_LOC_LIMIT: Final[int] = 60000  # loc_limit

DEFAULT_SVG_RASTERIZER: Final[SVGRasterizer] = "auto"  # render.svg_rasterizer

# [checks.performance]
DEFAULT_ENCODE_P95_MS: Final[float] = 250.0  # encode_p95_ms
DEFAULT_DECODE_P95_MS: Final[float] = 400.0  # decode_p95_ms
DEFAULT_MIN_SYMBOLS_PER_MINUTE: Final[float] = 120.0  # min_symbols_per_minute
DEFAULT_ENCODE_SAMPLES: Final[int] = 100  # encode_samples
DEFAULT_DECODE_SAMPLES: Final[int] = 200  # decode_samples
DEFAULT_WARMUP: Final[int] = 5  # warmup
DEFAULT_DECODE_IMAGE_PX: Final[int] = 500  # decode_image_px

# [checks.damage].  ``occlusion_margin`` is the absolute safety margin
# subtracted from the documented error-correction recovery capacity, so the
# damage budget never relies on the approximate percentage alone.
DEFAULT_OCCLUSION_MARGIN: Final[float] = 0.05  # occlusion_margin
DEFAULT_BLUR_RADIUS_PER_MODULE: Final[float] = 0.6  # blur_radius_per_module
DEFAULT_JPEG_QUALITY: Final[int] = 40  # jpeg_quality
DEFAULT_NOISE_SIGMA: Final[float] = 12.0  # noise_sigma
DEFAULT_ROTATION_DEG: Final[float] = 15.0  # rotation_deg
DEFAULT_PERSPECTIVE_DEG: Final[float] = 10.0  # perspective_deg
DEFAULT_DOWNSCALE_PX_PER_MODULE: Final[float] = 3.0  # downscale_px_per_module
DEFAULT_LOGO_AREA_FRACTION: Final[float] = 0.15  # logo_area_fraction

# [targets.<name>], common to every adapter kind.  The contrast floor defaults
# to the ratio RN-CONTRAST asserts, so a target that declares nothing is still
# held to the documented floor, and IV-BAD-OPTION reads that same declared value
# to build a request measured below it.
DEFAULT_TIMEOUT_S: Final[float] = 30.0  # timeout_s
DEFAULT_CONTRAST_FLOOR: Final[float] = 7.0  # contrast_floor

# ---------------------------------------------------------------------------
# Records
#
# Frozen, so a resolved configuration cannot drift once the loader has built
# it, and free of validation, so the single gate stays
# ``qrcheck.config.schema.validate``.
# ---------------------------------------------------------------------------

# ``MappingProxyType({})`` is deliberately not written inline as a field
# default: ``dataclasses`` rejects a default whose class is unhashable, and a
# mapping proxy has ``__hash__ = None``.  The factory hands every record the
# same read-only empty mapping instead.  For the same reason a record that
# carries a mapping compares field-wise but cannot be hashed, which no caller
# needs: targets are looked up by name, never keyed by their record.
_EMPTY_MAPPING: Final[Mapping[str, str]] = MappingProxyType({})


def _empty_mapping() -> Mapping[str, str]:
    """Return the shared read-only empty mapping used as a ``metadata_map`` default."""
    return _EMPTY_MAPPING


@dataclass(frozen=True)
class RenderConfig:
    """The ``[render]`` table: how SVG output is rasterized for checking.

    Mirrors the ``[render]`` row of the configuration contract.  Constructing
    it raises nothing and probes no rasterizer: resolving ``auto`` against the
    discovery order belongs to the renderer discovery that ``doctor`` and
    RN-SVG perform, and an unknown name is rejected by
    ``qrcheck.config.schema.validate``.
    """

    svg_rasterizer: SVGRasterizer = DEFAULT_SVG_RASTERIZER


@dataclass(frozen=True)
class PerformanceLimits:
    """The ``[checks.performance]`` table: the latency and throughput floors.

    The field names are the configuration keys and the attributes
    ``qrcheck.checks.performance`` reads, so each threshold is named once here
    and read once there.  Constructing it raises nothing; the lower bounds on
    the sample counts belong to ``qrcheck.config.schema.validate``.
    """

    encode_p95_ms: float = DEFAULT_ENCODE_P95_MS
    decode_p95_ms: float = DEFAULT_DECODE_P95_MS
    min_symbols_per_minute: float = DEFAULT_MIN_SYMBOLS_PER_MINUTE
    encode_samples: int = DEFAULT_ENCODE_SAMPLES
    decode_samples: int = DEFAULT_DECODE_SAMPLES
    warmup: int = DEFAULT_WARMUP
    decode_image_px: int = DEFAULT_DECODE_IMAGE_PX


@dataclass(frozen=True)
class DamageLimits:
    """The ``[checks.damage]`` table: every damage threshold in one record.

    The field names are the configuration keys and the attributes
    ``qrcheck.checks.damage`` reads.  Constructing it raises nothing; the
    permitted range of every fraction, angle and quality belongs to
    ``qrcheck.config.schema.validate``.
    """

    occlusion_margin: float = DEFAULT_OCCLUSION_MARGIN
    blur_radius_per_module: float = DEFAULT_BLUR_RADIUS_PER_MODULE
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    noise_sigma: float = DEFAULT_NOISE_SIGMA
    rotation_deg: float = DEFAULT_ROTATION_DEG
    perspective_deg: float = DEFAULT_PERSPECTIVE_DEG
    downscale_px_per_module: float = DEFAULT_DOWNSCALE_PX_PER_MODULE
    logo_area_fraction: float = DEFAULT_LOGO_AREA_FRACTION


@dataclass(frozen=True, kw_only=True)
class TargetConfig:
    """One ``[targets.<name>]`` table for an adapter with no per-type keys.

    Used directly by the two built-in targets (``reference`` and
    ``defective``), which declare no configurable input beyond the common keys.
    ``capabilities`` holds member names drawn from ``CAPABILITY_NAMES`` and
    ``type`` one of ``ADAPTER_TYPE_NAMES``; both are validated by
    ``qrcheck.config.schema.validate``, not here.  Keyword-only construction,
    because a subclass appends required fields after these defaulted ones and
    because every call site then reads as the table it came from.
    Constructing it raises nothing.
    """

    name: str
    type: AdapterType
    timeout_s: float = DEFAULT_TIMEOUT_S
    contrast_floor: float = DEFAULT_CONTRAST_FLOOR
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True, kw_only=True)
class LibraryTarget(TargetConfig):
    """A ``type = "library"`` target: an in-process import-and-call.

    ``callable`` is the ``module:function`` path, named for its configuration
    key; ``kwargs_map`` maps framework field names to the callable's keyword
    arguments; ``returns`` says how the returned value carries the artefact.
    Constructing it raises nothing; a malformed ``callable`` is rejected by
    ``qrcheck.config.schema.validate``.
    """

    callable: str
    kwargs_map: Mapping[str, str]
    returns: ReturnsKind
    metadata_map: Mapping[str, str] = field(default_factory=_empty_mapping)


@dataclass(frozen=True, kw_only=True)
class CLITarget(TargetConfig):
    """A ``type = "cli"`` target: an argv-only subprocess invocation.

    ``command`` is an immutable argv tuple rather than a list, so a resolved
    configuration cannot be mutated into a different command line.
    ``payload_arg`` and ``output_arg`` are the argument templates that receive
    the payload and name the artefact's path.  Constructing it raises nothing;
    an empty command is rejected by ``qrcheck.config.schema.validate``.
    """

    command: tuple[str, ...]
    payload_arg: str
    output_arg: str
    returns: ReturnsKind
    metadata_map: Mapping[str, str] = field(default_factory=_empty_mapping)


@dataclass(frozen=True, kw_only=True)
class HTTPTarget(TargetConfig):
    """A ``type = "http"`` target reached over loopback HTTP.

    ``request_template`` is the parsed TOML table whose string leaves carry
    placeholders drawn from ``TEMPLATE_FIELDS``; a ``request_template`` written as a
    JSON-object string is parsed into the same read-only mapping by
    ``qrcheck.config.load``, so the record has one shape either way.  ``image_field``
    is the dotted path to the image inside the response, or ``_body`` for the raw
    body; ``image_encoding`` says how that field carries the image.  The request and
    response shape of a vendor lives entirely in this record, which is what keeps the
    HTTP adapter free of hard-coded field names.  Constructing it raises nothing; a
    non-loopback host and an unknown placeholder are rejected by
    ``qrcheck.config.schema.validate``.
    """

    url: str
    method: HttpMethod
    request_template: Mapping[str, object]
    image_field: str
    image_encoding: ImageEncoding
    metadata_map: Mapping[str, str] = field(default_factory=_empty_mapping)


@dataclass(frozen=True)
class Config:
    """The fully resolved configuration: every default already applied.

    ``path`` is the resolved configuration file path and the base against which
    every relative path in that file was resolved; ``report_dir`` and
    ``corpus_path`` are absolute.  ``targets`` maps each declared target name to
    its record in name order, and ``default_adapter`` names one of those keys.
    The three limit tables default to their ``DEFAULT_*`` values, so a ``Config``
    is constructible without a ``qrcheck.toml`` on disk.  Constructing it raises
    nothing and touches no file; ``report_dir`` is created and probed for
    writability by ``qrcheck.config.load``.
    """

    path: Path
    default_adapter: str
    report_dir: Path
    corpus_path: Path
    seed: int
    loc_limit: int
    targets: Mapping[str, TargetConfig]
    render: RenderConfig = field(default_factory=RenderConfig)
    performance: PerformanceLimits = field(default_factory=PerformanceLimits)
    damage: DamageLimits = field(default_factory=DamageLimits)


__all__ = [
    "ADAPTER_TYPE_NAMES",
    "AdapterType",
    "CAPABILITY_NAMES",
    "CLITarget",
    "CapabilityName",
    "Config",
    "DEFAULT_ADAPTER",
    "DEFAULT_BLUR_RADIUS_PER_MODULE",
    "DEFAULT_CONTRAST_FLOOR",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_DECODE_IMAGE_PX",
    "DEFAULT_DECODE_P95_MS",
    "DEFAULT_DECODE_SAMPLES",
    "DEFAULT_DOWNSCALE_PX_PER_MODULE",
    "DEFAULT_ENCODE_P95_MS",
    "DEFAULT_ENCODE_SAMPLES",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_LOC_LIMIT",
    "DEFAULT_LOGO_AREA_FRACTION",
    "DEFAULT_MIN_SYMBOLS_PER_MINUTE",
    "DEFAULT_NOISE_SIGMA",
    "DEFAULT_OCCLUSION_MARGIN",
    "DEFAULT_PERSPECTIVE_DEG",
    "DEFAULT_REPORT_DIR",
    "DEFAULT_ROTATION_DEG",
    "DEFAULT_SEED",
    "DEFAULT_SVG_RASTERIZER",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_WARMUP",
    "DamageLimits",
    "HTTPTarget",
    "HTTP_METHODS",
    "HttpMethod",
    "IMAGE_ENCODINGS",
    "ImageEncoding",
    "LOOPBACK_HOSTS",
    "LibraryTarget",
    "METADATA_FIELD_NAMES",
    "MetadataFieldName",
    "PerformanceLimits",
    "RETURNS_KINDS",
    "RenderConfig",
    "ReturnsKind",
    "SVGRasterizer",
    "SVG_RASTERIZERS",
    "TEMPLATE_FIELDS",
    "TargetConfig",
]
