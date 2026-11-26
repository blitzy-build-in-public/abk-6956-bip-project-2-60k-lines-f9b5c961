"""Read ``qrcheck.toml`` and resolve it into an immutable ``models.Config``.

This is the only module in the product that opens the configuration file, and it
owns exactly two things: *resolution* (the document's relative paths become
absolute, its omitted keys get their declared defaults) and *assembly* (the
defaulted document becomes the frozen records the rest of the product reads).  It
performs no validation of its own -- every key, type, default and rule is declared
in :mod:`qrcheck.config.schema` and every default value in
:mod:`qrcheck.config.models` -- so a threshold changes in one declaration and the
acceptance criterion that reads it, never here.

What it deliberately does not do, so nothing written below has to be undone later:

* No environment access.  Nothing reads an environment variable, a ``.env`` file, a
  secret or a telemetry endpoint: the document is the only input.
* No ``~`` expansion and no ``$VAR`` interpolation.  A configured path is resolved
  against the configuration file's own directory and nothing else, so
  ``qrcheck --config /anywhere/qrcheck.toml`` behaves the same for every user.
* Nothing is printed and no exit code is decided.  The command-line layer owns the
  exit-code vocabulary; this module raises only the two failures that vocabulary is
  built from.
* No capability check.  Whether a declared capability set can satisfy a check
  depends on the run-time ``--checks`` selection, so the same document would be
  wrong for one selection and correct for the next; a shortfall is a ``skipped``
  result with reason ``unsupported_capability`` instead.
* No check that ``corpus_path`` exists.  That condition holds "when a corpus-driven
  check is selected", which this module cannot know, so ``qrcheck.runner`` raises it
  as its own run-prerequisite error and names the path there.

Two failure channels, one exit code each:

* :class:`qrcheck.config.schema.ConfigError` -- the file cannot be read, is not
  valid UTF-8, is not valid TOML, or holds a key the schema rejects.  The CLI maps
  it to exit code 2, prints ``str(exc)`` on stderr, which names the offending
  dotted key path, and writes no report.  A whole-file fault carries an empty
  ``key``, because no single key owns it.
* :class:`OSError` -- ``report_dir`` cannot be created or is not writable.  It is
  raised unwrapped so ``str(exc)`` names the directory, and the CLI maps it to exit
  code 3, streaming whatever results exist to stdout as JSON lines.  An unwritable
  report directory must never read as a configuration fault.

Determinism: loading the same file twice returns equal records.  Nothing here reads
a clock, a hostname, a process id, ``hash()`` or the filesystem's iteration order,
and the only strings kept are those the document holds plus the resolved, absolute
paths.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, cast

from qrcheck.config import models, schema

__all__ = ["load_config"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The probe file that proves ``report_dir`` is writable.  It is created inside the
# report directory itself -- never in the system temporary directory, which the
# artifact-cleanup check inspects -- and removed before this module returns, so it
# leaves no trace: no marker file, no name a later run has to filter out.
_PROBE_PREFIX: Final[str] = "qrcheck-probe-"
_PROBE_SUFFIX: Final[str] = ".tmp"
_PROBE_CONTENT: Final[bytes] = b"qrcheck writability probe\n"

# Which configuration record each ``type`` resolves to.  This is the *record* map of
# the configuration package and is not the adapter factory map, which pairs a type
# with the implementation that consumes these records: importing the adapter layer
# from here would close a cycle, because that layer depends on this package for the
# very records built below.
_TARGET_CLASSES: Final[Mapping[str, type[models.TargetConfig]]] = MappingProxyType(
    {
        "library": models.LibraryTarget,
        "cli": models.CLITarget,
        "http": models.HTTPTarget,
        "reference": models.TargetConfig,
        "defective": models.TargetConfig,
    }
)


# ---------------------------------------------------------------------------
# The one public function
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> models.Config:
    """Load, validate and resolve one ``qrcheck.toml``.

    The returned record is fully resolved.  ``path`` is the absolute configuration
    file path and the base every relative path in that file was resolved against;
    ``report_dir`` and ``corpus_path`` are absolute; every key the document omitted
    carries the default declared in :mod:`qrcheck.config.models`; and ``report_dir``
    exists and has been proven writable by creating and removing a probe file inside
    it.

    Args:
        path: The configuration file, absolute or relative to the current working
            directory.  No ``~`` expansion and no environment interpolation is
            applied to it.

    Returns:
        The resolved configuration: frozen records, a read-only target mapping in
        name order, and a value equal to what a second load of the same file
        returns.

    Raises:
        qrcheck.config.schema.ConfigError: the file cannot be read, is not valid
            UTF-8, is not valid TOML, or holds a key the schema rejects.  The CLI
            maps this to exit code 2, prints ``str(exc)`` and writes no report.
        OSError: ``report_dir`` cannot be created or is not writable.  Raised
            unwrapped, so ``str(exc)`` names the directory; the CLI maps it to exit
            code 3.
    """
    config_path = _resolve_config_path(path)
    document = _read_document(config_path)
    schema.validate(document)
    resolved = schema.apply_defaults(document)

    base = config_path.parent
    report_dir = _resolve_path(base, resolved["report_dir"])
    corpus_path = _resolve_path(base, resolved["corpus_path"])
    _prepare_report_dir(report_dir)

    return models.Config(
        path=config_path,
        default_adapter=str(resolved["default_adapter"]),
        report_dir=report_dir,
        corpus_path=corpus_path,
        seed=int(resolved["seed"]),
        loc_limit=int(resolved["loc_limit"]),
        targets=_build_targets(resolved["targets"]),
        render=_build_render(resolved["render"]),
        performance=_build_performance(resolved["checks"]["performance"]),
        damage=_build_damage(resolved["checks"]["damage"]),
    )


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def _resolve_config_path(path: str | Path) -> Path:
    """Return the absolute configuration file path a relative ``path`` denotes.

    A relative path is joined to the current working directory and then resolved,
    which is what makes the diagnostic of an unreadable file name a path the caller
    can open, and what makes two loads of the same relative argument agree.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _read_document(config_path: Path) -> dict[str, Any]:
    """Parse the TOML document at ``config_path`` into a fresh mapping.

    The bytes are read first and decoded explicitly, rather than handing a binary
    stream to ``tomllib.load``, so the UTF-8 contract is stated in this module
    instead of being implied by the parser.

    Every fault is a whole-file fault and therefore carries the empty key: no
    single key owns a file that cannot be opened or parsed.

    Args:
        config_path: The resolved path to read.

    Returns:
        The parsed document.

    Raises:
        qrcheck.config.schema.ConfigError: the file cannot be read (a missing file,
            a directory, a denied permission or any other ``OSError`` -- every one of
            them is exit code 2, never exit 3, because only ``report_dir`` failures
            are environment faults), the bytes are not UTF-8, or the text is not
            valid TOML.
    """
    try:
        data = config_path.read_bytes()
    except OSError:
        raise schema.ConfigError(
            "", f"cannot read configuration file: {config_path}"
        ) from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise schema.ConfigError(
            "", f"configuration file is not valid UTF-8: {config_path}"
        ) from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        # The parser's own "at line N column M" text is kept verbatim: it is what
        # turns the diagnostic into an instruction.
        raise schema.ConfigError("", f"invalid TOML in {config_path}: {exc}") from None


# ---------------------------------------------------------------------------
# Paths and the report directory
# ---------------------------------------------------------------------------


def _resolve_path(base: Path, value: Any) -> Path:
    """Resolve one configured path against the configuration file's directory.

    An absolute value in the document is honoured as written; a relative one is
    rooted at ``base``, the directory holding the configuration file, and then
    resolved.  No ``~`` and no environment variable is expanded, and no path is
    checked for existence: whether ``corpus_path`` exists depends on which checks
    are selected, and whether ``report_dir`` is usable is decided by the probe
    below.
    """
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _prepare_report_dir(report_dir: Path) -> None:
    """Create ``report_dir`` if absent and prove it is writable.

    Writability is proven the only way it can be: by writing.  A permission bit, an
    access-mode guess or a mount flag can all be wrong for the reason the run will
    actually hit, so a probe file is created and removed, and a read-only directory,
    a full filesystem or a path whose parent is a regular file each surface as the
    ``OSError`` the real write would have produced.

    The probe is created inside ``report_dir`` rather than in the system temporary
    directory, so the artifact-cleanup check -- which watches that directory -- never
    sees it, and it is removed in a ``finally`` so a failure part way through leaves
    no file behind.

    Args:
        report_dir: The resolved, absolute report directory.

    Raises:
        OSError: ``mkdir`` or the probe write failed.  Raised unwrapped and never
            converted to a configuration error, because a report directory the
            process cannot write is an environment fault (exit code 3), not a
            configuration fault (exit code 2).
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=report_dir,
            prefix=_PROBE_PREFIX,
            suffix=_PROBE_SUFFIX,
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(_PROBE_CONTENT)
            probe.flush()
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Assembly
#
# The document handed to these builders has already been validated, so every value
# below is present and of its declared kind.  Values are coerced in exactly two ways:
#
# * A number is cast to the type its record declares.  TOML spells a whole number
#   as an integer, ``timeout_s = 30``, and the schema accepts either spelling for a
#   float field, so this is the one place a number key becomes the ``float`` or the
#   ``int`` its record holds.
# * A closed-vocabulary string is cast to the ``Literal`` alias that names its set,
#   which records that the schema has already restricted it to a known member.
# ---------------------------------------------------------------------------


def _build_targets(tables: Mapping[str, Any]) -> Mapping[str, models.TargetConfig]:
    """Build every declared target, keyed by name in ``sorted()`` order.

    The sorted population is what makes iteration over targets -- in ``doctor``, in
    the reports and in any evidence listing -- reproducible from one run to the next
    regardless of the order the document happened to declare its tables in.  The
    returned mapping is read-only, so a resolved configuration cannot gain, lose or
    silently re-point a target.
    """
    targets: dict[str, models.TargetConfig] = {}
    for name in sorted(tables):
        targets[name] = _build_target(name, tables[name])
    return MappingProxyType(targets)


def _build_target(name: str, table: Mapping[str, Any]) -> models.TargetConfig:
    """Build the configuration record for one ``[targets.<name>]`` table."""
    kind = str(table["type"])
    target_class = _TARGET_CLASSES.get(kind)
    if target_class is None:
        # Unreachable while ``validate`` runs first, since it rejects a ``type``
        # outside the enum.  The mapping stays total anyway so a caller that composes
        # the steps in the wrong order fails with the same named-key diagnostic the
        # schema would have raised, rather than with a bare ``KeyError``.
        raise schema.ConfigError(
            f"targets.{name}.type",
            f"unknown target type '{kind}'; allowed: {sorted(_TARGET_CLASSES)}",
        )
    return target_class(
        name=name,
        type=cast(models.AdapterType, kind),
        timeout_s=float(table["timeout_s"]),
        contrast_floor=float(table["contrast_floor"]),
        capabilities=frozenset(str(member) for member in table["capabilities"]),
        **_per_type_options(kind, table),
    )


def _per_type_options(kind: str, table: Mapping[str, Any]) -> dict[str, Any]:
    """Return the per-type keyword arguments a target's ``type`` declares.

    ``metadata_map`` is added only when the document supplies it.  Every record that
    reads it declares the same read-only empty mapping as its own default, so filling
    it here would be a second copy of a default that ``qrcheck.config.models``
    already owns.

    ``request_template`` is kept as a plain ``dict`` rather than wrapped in a mapping
    proxy like the other tables.  The HTTP adapter hands it to ``json.dumps`` to build
    the request body, and ``json`` cannot serialise a proxy; nothing in the product
    mutates it, so the copy made here is as immutable in practice as the document it
    came from.
    """
    if kind == "library":
        options: dict[str, Any] = {
            "callable": str(table["callable"]),
            "kwargs_map": _frozen_map(table["kwargs_map"]),
            "returns": cast(models.ReturnsKind, str(table["returns"])),
        }
    elif kind == "cli":
        options = {
            "command": tuple(str(argument) for argument in table["command"]),
            "payload_arg": str(table["payload_arg"]),
            "output_arg": str(table["output_arg"]),
            "returns": cast(models.ReturnsKind, str(table["returns"])),
        }
    elif kind == "http":
        options = {
            "url": str(table["url"]),
            "method": cast(models.HttpMethod, str(table["method"])),
            "request_template": dict(table["request_template"]),
            "image_field": str(table["image_field"]),
            "image_encoding": cast(models.ImageEncoding, str(table["image_encoding"])),
        }
    else:
        # ``reference`` and ``defective`` declare no per-type key of their own, so
        # their record is built from the common keys alone.
        options = {}
    if "metadata_map" in table:
        options["metadata_map"] = _frozen_map(table["metadata_map"])
    return options


def _frozen_map(value: Mapping[str, Any]) -> Mapping[str, str]:
    """Return a read-only copy of a ``kwargs_map`` or ``metadata_map`` table.

    A mapping proxy rather than a ``dict``, so no caller can widen a resolved
    configuration's field mapping after the fact, and a copy rather than the parsed
    table itself, so the record shares no mutable structure with the document.
    """
    return MappingProxyType({str(key): str(mapped) for key, mapped in value.items()})


def _build_render(table: Mapping[str, Any]) -> models.RenderConfig:
    """Build the ``[render]`` record."""
    return models.RenderConfig(
        svg_rasterizer=cast(models.SVGRasterizer, str(table["svg_rasterizer"])),
    )


def _build_performance(table: Mapping[str, Any]) -> models.PerformanceLimits:
    """Build the ``[checks.performance]`` record, reading every floor from its key."""
    return models.PerformanceLimits(
        encode_p95_ms=float(table["encode_p95_ms"]),
        decode_p95_ms=float(table["decode_p95_ms"]),
        min_symbols_per_minute=float(table["min_symbols_per_minute"]),
        encode_samples=int(table["encode_samples"]),
        decode_samples=int(table["decode_samples"]),
        warmup=int(table["warmup"]),
        decode_image_px=int(table["decode_image_px"]),
    )


def _build_damage(table: Mapping[str, Any]) -> models.DamageLimits:
    """Build the ``[checks.damage]`` record, reading every threshold from its key."""
    return models.DamageLimits(
        occlusion_margin=float(table["occlusion_margin"]),
        blur_radius_per_module=float(table["blur_radius_per_module"]),
        jpeg_quality=int(table["jpeg_quality"]),
        noise_sigma=float(table["noise_sigma"]),
        rotation_deg=float(table["rotation_deg"]),
        perspective_deg=float(table["perspective_deg"]),
        downscale_px_per_module=float(table["downscale_px_per_module"]),
        logo_area_fraction=float(table["logo_area_fraction"]),
    )
