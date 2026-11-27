"""``qrcheck doctor``: verify the target and the local toolchain before any check runs.

``doctor`` is that pre-flight verification.  It decides whether this host can test this target at
all, and it reports every prerequisite rather than stopping at the first fault, because a reader
fixing an environment needs the whole list.  It runs no check, decodes no symbol, writes no
report and asserts nothing about a generated symbol; the one symbol it asks a target for is the
probe below, which makes this module the **only** place outside a run that is allowed to call
``generate``.

What it verifies, and what each outcome means, prerequisite by prerequisite.  A blocking
prerequisite fails the command with exit 2, naming the culprit: the configuration file (the dotted
key, or the path of a file that cannot be read, since nothing else can be resolved without it), a
CPython older than 3.11, an undeclared target adapter (the diagnostic echoing the declared list), a
declared one that is unreachable, or a declared one that refuses its own configuration when it is
built (a mapping whose vocabulary belongs to the adapter, reported with its dotted key), the
primary decoder ``zxing-cpp`` (naming the distribution, the interpreter and the wheel tag of this
platform), and the oracle and core libraries ``segno``,
``numpy``, ``Pillow``, ``httpx`` and ``click`` (naming the distribution and
``requirements.lock``).  A degrading prerequisite still exits 0: a newer 3.x or a non-CPython
interpreter runs unsupported, is reported ``degraded`` and still exits 0; a missing secondary
decoder ``opencv-python-headless`` lets the run proceed in single-decoder mode, with ``agreement``
recording ``primary_only`` and ``index.html`` carrying the degradation banner; and a missing SVG
rasterizer makes ``RN-SVG`` report ``skipped`` with reason ``missing_dependency`` and the discovery
order it attempted.  A ``degraded`` entry never fails the gate, and a missing secondary decoder
never becomes a skipped or a passing check: degradation is recorded, never inferred as health.

Output contract.  One result object with exactly six keys -- ``adapter``, ``type``,
``reachable``, ``capabilities``, ``dependencies`` and ``svg_rasterizer`` -- is emitted once, on both
the passing and the failing path, so a CI consumer reads one shape and finds a machine-readable
failure instead of an empty stream.  Each dependency entry carries exactly ``name``, ``version``,
``status`` and ``blocking``, with ``status`` drawn from ``ok``/``missing``/``degraded``, and the
list is sorted by ``name``.  ``--json`` prints that object with sorted keys; without it a table is
printed.  Diagnostics always go to standard error, so standard output stays parsable, and nothing
emitted carries a timestamp, a hostname, a duration or a path other than the configuration path the
caller supplied.

What this module does not own.  SVG-rasterizer discovery is
:func:`qrcheck.checks.render.discover_svg_rasterizer` -- imported here, never re-implemented and
never re-ordered, because the tool this command reports has to be the tool ``RN-SVG`` uses.
Adapters are built through :data:`qrcheck.targets.ADAPTER_TYPES`, the one factory map.  Exit codes
come from :mod:`qrcheck.cli.exit` and are never written as literals.  Decoder presence is read from
:mod:`qrcheck.decode`, whose probes import neither ``cv2`` nor ``zxingcpp``; this command then
confirms the load itself, because a module that is present but fails to import is precisely the
failure this command exists to report.  Both the decoder's load and each core library are confirmed
through the one guarded probe in :mod:`qrcheck.dependencies`, which is also what ``qrcheck run``
refuses on, so the two commands can never disagree about whether a dependency is usable.

The public surface is two names, split exactly as ``qrcheck/cli/loc.py`` splits ``loc`` and
``loc_command``: ``doctor_command`` is the click object ``qrcheck.cli.main`` registers, and
:func:`doctor` is the function behind it, returning the exit code to a direct caller and to a test.

Exit codes, from :mod:`qrcheck.cli.exit`: ``0`` when every blocking prerequisite passes, ``2`` when
one fails -- a configuration fault included, whether the loader or the adapter itself reports it --
and ``3`` when an unexpected framework fault occurred: an exception outside both the adapter error
vocabulary and the configuration faults an adapter states by ``ConfigError`` or ``ValueError``, an
adapter that cannot be closed, or a ``report_dir`` that cannot be prepared.
"""

from __future__ import annotations

import platform
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

import click

from qrcheck.checks.render import RasterizerStatus, discover_svg_rasterizer
from qrcheck.cli.exit import EXIT_CONFIG, EXIT_INTERNAL, EXIT_OK
from qrcheck.cli.support import DEFAULT_CONFIG_PATH, ExitCodeCommand, resolve_config_path
from qrcheck.config import Config, ConfigError, load_config
from qrcheck.decode import DecoderAvailability, decoder_availability
from qrcheck.dependencies import (
    CORE_DEPENDENCIES,
    PRIMARY_DECODER_DISTRIBUTION,
    PRIMARY_DECODER_MODULE,
    SECONDARY_DECODER_DISTRIBUTION,
    SECONDARY_DECODER_MODULE,
    import_problem,
    probe,
)
from qrcheck.report.json_report import dumps_canonical
from qrcheck.targets import ADAPTER_TYPES
from qrcheck.targets.base import (
    TargetContractError,
    TargetError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    TargetUnsupported,
    build_request,
)

if TYPE_CHECKING:  # Annotation resolution only: this module constructs no record itself.
    from qrcheck.config import TargetConfig
    from qrcheck.targets.base import Adapter

__all__ = ("doctor", "doctor_command")

# Every diagnostic carries this prefix and goes to stderr, so a line on stdout is always
# either the report object or the table, and a line on stderr is always something to act on.
_DIAGNOSTIC_PREFIX: Final[str] = "qrcheck doctor: "

# The Unicode general categories that make a terminal or a CI log act instead of print: the C0
# and C1 controls and DEL (``Cc``), the invisible format characters including the bidi overrides
# (``Cf``), a lone surrogate (``Cs``) and the line and paragraph separators (``Zl``, ``Zp``). A
# character in one of these categories is escaped where a message is rendered; the note, and
# therefore what the command carries, keeps the character a target supplied.
_UNPRINTABLE_CATEGORIES: Final[frozenset[str]] = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})

# The probe request. It is a short ASCII payload at the framework's default geometry and
# colours -- a request every conforming target must encode -- built through
# :func:`qrcheck.targets.base.build_request` so an impossible probe fails loudly instead of
# reaching a target. Nothing reads the probe symbol: its arrival is the whole assertion.
# No geometry is pinned: ``module_size_px`` and ``quiet_zone_modules`` keep their ``None``
# default, so the probe asks for the output format and nothing else, and the capability set it
# demands is the one the adapter really needs. A target that omits an optional capability --
# declaring nothing but ``png``, the single capability ``RT-ROUNDTRIP`` requires -- is valid
# configuration: the checks that do need that capability are resolved to
# ``skipped``/``unsupported_capability`` by ``qrcheck/registry/resolve.py``, so such a
# shortfall is reported by the affected checks skipping and never by this command failing.
PROBE_PAYLOAD: Final[bytes] = b"qrcheck doctor probe"
PROBE_EC_LEVEL: Final[str] = "M"

# The three statuses a prerequisite can report. The vocabulary is closed: ``ok`` is usable,
# ``missing`` is absent or unusable, ``degraded`` is usable-but-unsupported or absent where
# its absence is a supported degradation.
_Status = Literal["ok", "missing", "degraded"]
STATUS_OK: Final[_Status] = "ok"
STATUS_MISSING: Final[_Status] = "missing"
STATUS_DEGRADED: Final[_Status] = "degraded"

# The supported interpreter, exactly: CPython 3.11. Below it the configuration cannot even be
# parsed on 3.10 (no ``tomllib``), and above it the product runs untested, which is reported as
# a degradation rather than as health.
_SUPPORTED_PYTHON: Final[tuple[int, int]] = (3, 11)
_LOWEST_PYTHON: Final[tuple[int, int]] = (3, 11)
_CPYTHON: Final[str] = "CPython"

# The architectures whose Linux wheels the pinned primary decoder does not publish. This is
# the one platform gap in the pinned dependency set, so its diagnostic names the missing tag
# instead of leaving a reader to deduce it from an ImportError.
_AARCH64_MACHINES: Final[frozenset[str]] = frozenset({"aarch64", "arm64"})
_AARCH64_WHEEL_TAG: Final[str] = "manylinux_2_17_aarch64"
_LINUX_WHEEL_TAG_PREFIX: Final[str] = "manylinux_2_17_"
_MACOS_WHEEL_TAG: Final[str] = "macosx_10_9_universal2"

# The table's column headers, and therefore the minimum width of each column: the human report
# is a table a reader scans, and a fixed vocabulary keeps it stable across hosts.
_NAME_HEADER: Final[str] = "NAME"
_VERSION_HEADER: Final[str] = "VERSION"
_STATUS_HEADER: Final[str] = "STATUS"
_BLOCKING_HEADER: Final[str] = "BLOCKING"


# ---------------------------------------------------------------------------
# The records this command is built from
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Note:
    """One diagnostic, and whether it blocks the run.

    ``blocking`` is what decides the exit code and what the table's verdict line names, so the
    two can never disagree about which failure stopped the command: the note that is reported
    is the note that was counted.
    """

    message: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class _Dependency:
    """One prerequisite, in the shape ``doctor --json`` publishes for it.

    ``version`` is the installed distribution version, or ``None`` when the distribution is
    absent -- never a guessed or default version. ``blocking`` says whether its absence stops
    a run, which is the whole difference between the two decoder entries.
    """

    name: str
    version: str | None
    status: _Status
    blocking: bool

    def as_dict(self) -> dict[str, object]:
        """Project this entry to the four keys the JSON contract fixes.

        The keys are written out in full rather than built by iteration, so the published
        shape cannot drift with a field rename.
        """
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class _ProbeOutcome:
    """What the one probe generate produced.

    ``reachable`` is the JSON ``reachable`` value: whether the target answered at all, which
    is true even for a declared rejection, because a target that answers is reachable.
    ``capabilities`` is the capability set the adapter itself declares, which is the effective
    set -- the two built-in targets declare all thirteen in code rather than in configuration --
    and is empty when the adapter could not be built at all.
    ``internal_failure`` marks a framework defect -- a fault outside both the adapter error
    vocabulary and the configuration faults an adapter states for its own keys -- and exits 3.
    """

    reachable: bool
    internal_failure: bool = False
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DoctorReport:
    """The result object ``doctor --json`` emits: exactly the six keys :meth:`as_dict` projects.

    Every field has an "unknown" value -- ``None`` or an empty sequence -- so a fault that
    stops the command early still emits a complete object rather than a partial one.
    """

    adapter: str | None
    target_type: str | None
    reachable: bool
    capabilities: tuple[str, ...]
    dependencies: tuple[_Dependency, ...]
    svg_rasterizer: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        """Project the report to exactly the six keys, and never a seventh."""
        return {
            "adapter": self.adapter,
            "type": self.target_type,
            "reachable": self.reachable,
            "capabilities": list(self.capabilities),
            "dependencies": [dependency.as_dict() for dependency in self.dependencies],
            "svg_rasterizer": self.svg_rasterizer,
        }


# ---------------------------------------------------------------------------
# The command's function
# ---------------------------------------------------------------------------


def doctor(
    adapter: str | None,
    as_json: bool,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> int:
    """Verify the target and the local toolchain, and return the process exit code.

    The checks run in the order a reader would run them, because a fault early in the list
    makes the later ones meaningless:

    1. the configuration file is loaded and validated;
    2. the interpreter, the two decoders and the oracle/core libraries are inventoried -- host
       facts that need no configuration, so they are collected first and reported even when the
       configuration file itself is unusable;
    3. the SVG rasterizer is resolved through the single discovery in
       :mod:`qrcheck.checks.render`;
    4. the target name is resolved against the declared targets;
    5. the target is built through :data:`qrcheck.targets.ADAPTER_TYPES` and asked for one
       probe symbol, with each of the five adapter errors mapped to its documented outcome.

    Args:
        adapter: Target name to verify, or ``None`` for the configured ``default_adapter``.
        as_json: Emit the six-key result object instead of the human table.
        config_path: Configuration file to read, supplied by the group's ``--config`` option.

    Returns:
        :data:`qrcheck.cli.exit.EXIT_OK` when every blocking prerequisite passed,
        :data:`qrcheck.cli.exit.EXIT_CONFIG` when one failed or the configuration, the adapter
        name or the probe did -- the adapter refusing that configuration at construction included
        -- and :data:`qrcheck.cli.exit.EXIT_INTERNAL` for a fault outside both vocabularies.

    Raises:
        Nothing. Every failure is reported on stderr and returned as an exit code, so a caller
        propagates one integer instead of handling this module's faults.
    """
    notes: list[_Note] = []
    dependencies = _host_dependencies(notes)
    report = _DoctorReport(
        adapter=adapter,
        target_type=None,
        reachable=False,
        capabilities=(),
        dependencies=tuple(dependencies),
        svg_rasterizer=None,
    )

    config, failure_code = _load_configuration(config_path, notes)
    if config is None:
        return _finish(report, notes, failure_code, as_json)

    # The rasterizer is resolved before the target so that a reader wiring a toolchain sees the
    # degradation in the same report as the target verdict. It is never a blocking failure.
    rasterizer = discover_svg_rasterizer(config.render)
    report = replace(report, svg_rasterizer=dict(rasterizer.as_dict()))
    _note_rasterizer(rasterizer, notes)

    name = config.default_adapter if adapter is None else adapter
    report = replace(report, adapter=name)
    target = config.targets.get(name)
    if target is None:
        # The same wording the configuration schema fixes for an undeclared adapter, so the
        # two diagnostics for one mistake read identically.
        notes.append(
            _Note(
                f"unknown adapter {name!r}; declared: {sorted(config.targets)}",
                True,
            )
        )
        return _finish(report, notes, EXIT_CONFIG, as_json)

    report = replace(
        report,
        target_type=str(target.type),
        capabilities=tuple(sorted(target.capabilities)),
    )

    outcome = _probe_target(target, name, notes)
    report = replace(report, reachable=outcome.reachable)
    if outcome.capabilities:
        report = replace(report, capabilities=outcome.capabilities)
    if outcome.internal_failure:
        return _finish(report, notes, EXIT_INTERNAL, as_json)
    if any(note.blocking for note in notes):
        return _finish(report, notes, EXIT_CONFIG, as_json)
    return _finish(report, notes, EXIT_OK, as_json)


# ---------------------------------------------------------------------------
# Step 1: the configuration file
# ---------------------------------------------------------------------------


def _load_configuration(config_path: str, notes: list[_Note]) -> tuple[Config | None, int]:
    """Load and validate ``config_path``, reporting the fault rather than raising it.

    The three failures are distinguished exactly as the loader documents them, because they
    mean different things to whoever has to fix them: a configuration fault is exit 2 and names
    the dotted key or the unreadable file, while a ``report_dir`` that cannot be created or
    proved writable is an environment fault and exit 3.

    Args:
        config_path: The configuration file the caller named.
        notes: The diagnostic list to append to.

    Returns:
        The resolved configuration and :data:`qrcheck.cli.exit.EXIT_OK`, or ``None`` and the
        exit code the fault implies.
    """
    try:
        return load_config(config_path), EXIT_OK
    except ConfigError as exc:
        # str() of a ConfigError is "<dotted key>: <message>", or the bare message for a
        # whole-file fault such as an unreadable file or invalid TOML.
        notes.append(_Note(str(exc), True))
        return None, EXIT_CONFIG
    except FileNotFoundError:
        notes.append(_Note(f"configuration file not found: {config_path}", True))
        return None, EXIT_CONFIG
    except OSError as exc:
        # The loader converts every configuration *read* fault into a ConfigError, so the only
        # OSError it lets through is the report_dir mkdir and writability probe
        # (PermissionError included). str() of it already names the directory.
        notes.append(_Note(f"cannot prepare the report directory: {exc}", True))
        return None, EXIT_INTERNAL


# ---------------------------------------------------------------------------
# Step 2: the host's inventory -- interpreter, decoders, oracle and core libraries
# ---------------------------------------------------------------------------


def _host_dependencies(notes: list[_Note]) -> list[_Dependency]:
    """Inventory every prerequisite that depends on this host and not on the configuration.

    Reported even when the configuration file cannot be read, because a broken toolchain is
    the more urgent of two faults and nothing here needs a ``qrcheck.toml`` to decide.

    Args:
        notes: The diagnostic list to append to.

    Returns:
        The entries sorted by ``name``, so two runs of this command print the same table.
    """
    availability = decoder_availability()
    dependencies = [
        _python_dependency(notes),
        _primary_decoder_dependency(availability, notes),
        _secondary_decoder_dependency(availability, notes),
    ]
    dependencies.extend(
        _distribution_dependency(distribution, module, notes)
        for distribution, module in CORE_DEPENDENCIES
    )
    return sorted(dependencies, key=lambda dependency: dependency.name)


def _python_dependency(notes: list[_Note]) -> _Dependency:
    """Report the interpreter against the CPython 3.11 contract.

    Below 3.11 the product cannot run at all -- ``tomllib`` arrived in 3.11 -- so the status is
    ``missing`` and blocks. A newer 3.x, or another implementation at a supported version, runs
    but is not a supported platform: ``degraded``, and the table says so.
    """
    version = platform.python_version()
    implementation = platform.python_implementation()
    major_minor = sys.version_info[:2]
    if major_minor < _LOWEST_PYTHON:
        notes.append(
            _Note(
                f"unsupported interpreter {implementation} {version}: qrcheck requires "
                f"CPython {_SUPPORTED_PYTHON[0]}.{_SUPPORTED_PYTHON[1]}",
                True,
            )
        )
        return _Dependency("python", version, STATUS_MISSING, True)
    if implementation != _CPYTHON or major_minor > _SUPPORTED_PYTHON:
        notes.append(
            _Note(
                f"interpreter {implementation} {version} is not a supported platform: qrcheck "
                f"supports CPython {_SUPPORTED_PYTHON[0]}.{_SUPPORTED_PYTHON[1]}; it runs here "
                f"but is untested",
                False,
            )
        )
        return _Dependency("python", version, STATUS_DEGRADED, True)
    return _Dependency("python", version, STATUS_OK, True)


def _primary_decoder_dependency(
    availability: DecoderAvailability, notes: list[_Note]
) -> _Dependency:
    """Report the primary reference decoder, whose absence blocks the run.

    ``availability`` comes from :func:`qrcheck.decode.decoder_availability`, whose probe never
    imports the binding. That probe reports a module which is present, and this function then
    confirms the module actually loads, because a binding present but unimportable is exactly
    the state this command exists to report -- and the state ``qrcheck run`` refuses on, through
    the same guarded probe (see :func:`_decoder_load_problem`).
    """
    status: _Status = STATUS_OK
    problem = _decoder_load_problem(PRIMARY_DECODER_MODULE, availability.primary)
    if problem is not None:
        notes.append(_Note(_primary_decoder_message(problem), True))
        status = STATUS_MISSING
    return _Dependency(PRIMARY_DECODER_DISTRIBUTION, availability.primary_version, status, True)


def _secondary_decoder_dependency(
    availability: DecoderAvailability, notes: list[_Note]
) -> _Dependency:
    """Report the secondary reference decoder, whose absence degrades the run.

    Its status is ``ok`` or ``degraded`` and never ``missing``: there is no state in which the
    corroborating decoder blocks anything, and no check may pass or skip on its absence.
    """
    status: _Status = STATUS_OK
    problem = _decoder_load_problem(SECONDARY_DECODER_MODULE, availability.secondary)
    if problem is not None:
        notes.append(_Note(_secondary_decoder_message(problem), False))
        status = STATUS_DEGRADED
    return _Dependency(
        SECONDARY_DECODER_DISTRIBUTION, availability.secondary_version, status, False
    )


def _decoder_load_problem(module: str, present: bool) -> str | None:
    """Return why ``module`` cannot be used, or ``None`` when it loads.

    ``present`` is what the decode package's availability probe found with
    ``importlib.util.find_spec``, which locates a module without executing it. That is the
    right probe for the decode layer, which must not pay for a native image stack, but it
    cannot see the state a reader most needs diagnosed: a module that is installed yet fails to
    import. So the load is attempted here too, through
    :func:`qrcheck.dependencies.import_problem` -- the product's one guarded import probe, which
    reports whatever the module raises as "not usable", the two failures the decoder wrappers
    themselves treat that way included: ``ImportError`` for a missing distribution and ``OSError``
    for a compiled extension built for another architecture. Sharing the probe is what keeps this
    command and ``qrcheck run``'s prerequisite check from disagreeing about whether the decoder is
    usable.
    """
    if not present:
        return f"the {module} module is not importable"
    return import_problem(module)


def _primary_decoder_message(problem: str) -> str:
    """Render the blocking diagnostic for an unusable primary decoder.

    It names the distribution, the interpreter, the platform and the wheel tag the pinned
    release would have to be installed from, and -- on the one architecture whose Linux wheel
    the pin does not publish -- says so outright, because an ``ImportError`` alone leaves a
    reader guessing whether the package is missing or the platform is unsupported.
    """
    message = (
        f"primary decoder {PRIMARY_DECODER_DISTRIBUTION} is unusable: {problem}; "
        f"interpreter {platform.python_version()} on {sys.platform}-{platform.machine()}, "
        f"wheel tag {_decoder_wheel_tag()}; it is a blocking dependency, because every "
        f"round-trip assertion is decoded by it, so no check can run without it; install the "
        f"pins in requirements.lock"
    )
    if _is_linux_aarch64():
        message += (
            f"; zxing-cpp 2.2.0 publishes no cp311 wheel for {_AARCH64_WHEEL_TAG}, so the "
            f"pinned decoder cannot be installed on this platform"
        )
    return message


def _secondary_decoder_message(problem: str) -> str:
    """Render the non-blocking diagnostic for an unusable secondary decoder.

    The wording states the degradation and what it means, so a reader neither treats it as a
    failure nor expects the corroboration the run will not have.
    """
    return (
        f"secondary decoder {SECONDARY_DECODER_DISTRIBUTION} is unusable: {problem}; the run "
        f"proceeds in single-decoder mode, agreement records primary_only, index.html carries "
        f"the degradation banner, and no check passes on the strength of the missing decoder"
    )


def _decoder_wheel_tag() -> str:
    """Return the wheel tag the pinned primary decoder publishes for this platform.

    Derived from :data:`sys.platform` and :func:`platform.machine` rather than from the
    interpreter's own supported-tag list, so the diagnostic names the tag that would actually
    be requested for this machine.
    """
    if sys.platform.startswith("linux"):
        return f"{_LINUX_WHEEL_TAG_PREFIX}{platform.machine().lower() or 'unknown'}"
    if sys.platform == "darwin":
        return _MACOS_WHEEL_TAG
    return f"{sys.platform}_{platform.machine().lower() or 'unknown'}"


def _is_linux_aarch64() -> bool:
    """Report whether this is the one platform whose pinned primary decoder has no wheel."""
    return sys.platform.startswith("linux") and platform.machine().lower() in _AARCH64_MACHINES


def _distribution_dependency(distribution: str, module: str, notes: list[_Note]) -> _Dependency:
    """Report one blocking runtime distribution and its installed version.

    The status is ``ok`` only when :func:`qrcheck.dependencies.probe` finds the pair usable --
    the distribution installed *and* its module importing -- because an install that cannot be
    imported is the same broken environment in a different disguise: ``importlib.metadata``
    answers with a version for a module that a shadowed entry on ``sys.path`` or an unloadable
    extension makes impossible to import, and a run then fails with an ``ImportError`` from
    wherever that module was first touched.  The version is read from the installed distribution
    metadata, never guessed from the lock file: this command reports the environment it is
    running in. An unusable distribution is a broken install rather than a user choice -- it is
    declared in ``requirements.lock`` -- so the diagnostic is the probe's own, which names the
    distribution, the module, the reason and the lock file a reader has to install from.
    """
    found = probe(distribution, module)
    if not found.usable:
        notes.append(_Note(found.message, True))
        return _Dependency(distribution, found.version, STATUS_MISSING, True)
    return _Dependency(distribution, found.version, STATUS_OK, True)


# ---------------------------------------------------------------------------
# Step 3: the SVG rasterizer, through the product's single discovery
# ---------------------------------------------------------------------------


def _note_rasterizer(status: RasterizerStatus, notes: list[_Note]) -> None:
    """Record the rasterizer outcome, which degrades ``RN-SVG`` and never blocks the run.

    Nothing here probes a rasterizer: :func:`qrcheck.checks.render.discover_svg_rasterizer` is
    the product's only discovery, so the tool this command names is the tool the check uses.
    The note states the order that was attempted, which is what tells a reader what to install,
    and an explicitly disabled renderer is reported as the choice it is rather than as a gap.
    """
    if status.available:
        return
    if status.configured == "none":
        notes.append(
            _Note(
                "SVG rasterizer disabled by [render] svg_rasterizer = 'none'; RN-SVG will "
                "report skipped/missing_dependency",
                False,
            )
        )
        return
    notes.append(
        _Note(
            f"no SVG rasterizer found for [render] svg_rasterizer = {status.configured!r} "
            f"after trying {_attempted_order(status)}; RN-SVG will report "
            f"skipped/missing_dependency",
            False,
        )
    )


def _attempted_order(status: RasterizerStatus) -> str:
    """Render the discovery order a rasterizer search actually probed, in order."""
    return ", ".join(status.attempted) if status.attempted else "nothing"


# ---------------------------------------------------------------------------
# Step 4: the one probe generate
# ---------------------------------------------------------------------------


def _probe_target(target: TargetConfig, name: str, notes: list[_Note]) -> _ProbeOutcome:
    """Build the target and ask it for one probe symbol, mapping each failure to its outcome.

    This is the only place outside a run that calls ``generate``. The five adapter errors are
    mapped one for one, and their meanings differ in a way the exit code alone cannot carry:

    * ``TargetUnsupported`` means the target answered nothing and did no I/O by contract, so it
      is reachable -- but the probe asks for a PNG at the framework's default geometry, so a
      target that cannot produce one skips nearly every check and still fails the gate. An
      optional capability the target does not declare does not reach the gate this way: the
      probe pins no geometry, so nothing beyond the output format is required of it, and the
      checks that do need that capability are skipped by ``qrcheck/registry/resolve.py``;
    * ``TargetRejected`` means the target answered by refusing a request the framework
      considers valid, which keeps it reachable and still fails the gate. The rejection text is
      carried **verbatim**, as the adapter contract requires, and is escaped only where it is
      rendered for a terminal or a CI log;
    * ``TargetUnavailable``, ``TargetTimeout`` and ``TargetContractError`` mean no usable
      symbol arrived, so they are unreachable as well as blocking.

    A configuration the adapter refuses for its own keys is neither a target fault nor a framework
    defect: a ``ConfigError`` or ``ValueError`` naming the dotted key is exit 2, decided before any
    check runs, exactly as ``qrcheck run`` classifies the same refusal at construction. Only a
    fault outside both vocabularies is a framework defect, mapped to
    :data:`qrcheck.cli.exit.EXIT_INTERNAL` rather than silently becoming a target fault.

    Args:
        target: The resolved ``[targets.<name>]`` record to build the adapter from.
        name: The configured target name, quoted by every diagnostic.
        notes: The diagnostic list to append to.

    Returns:
        The probe outcome: whether the target is reachable, and whether the fault was internal.
    """
    try:
        # The framework's default geometry, so the probe requires the target's output format
        # and nothing more.
        request = build_request(PROBE_PAYLOAD, PROBE_EC_LEVEL, output_format="png")
    except ValueError as exc:
        # The probe is a constant legal request, so a refusal here is a defect in the request
        # builder or the capacity tables, not something a target did.
        notes.append(_Note(f"the probe request is not a legal request: {exc}", True))
        return _ProbeOutcome(reachable=False, internal_failure=True)

    adapter_obj: Adapter | None = None
    outcome = _ProbeOutcome(reachable=False)
    try:
        adapter_obj = ADAPTER_TYPES[str(target.type)](target, name=name)
        # The adapter is the authority on what it supports, so its own set is what the report
        # publishes; the configuration's list is only the input some adapters read.
        outcome = replace(
            outcome,
            capabilities=tuple(sorted(str(capability) for capability in adapter_obj.capabilities)),
        )
        adapter_obj.generate(request)
        outcome = replace(outcome, reachable=True)
    except TargetUnsupported as exc:
        notes.append(
            _Note(f"target {name!r} cannot produce the probe symbol: {exc}", True)
        )
        outcome = replace(outcome, reachable=True)
    except TargetRejected as exc:
        # str(exc) is exactly what the target said, and it is carried unchanged: the rejection
        # text is recorded verbatim. It is escaped only where it is rendered, by _printable in
        # _finish and in _verdict_line.
        notes.append(_Note(str(exc), True))
        outcome = replace(outcome, reachable=True)
    except TargetUnavailable as exc:
        notes.append(_Note(f"target {name!r} is unreachable: {exc}", True))
    except TargetTimeout as exc:
        notes.append(
            _Note(
                f"target {name!r} did not answer within timeout_s={target.timeout_s} while "
                f"generating the probe symbol: {exc}",
                True,
            )
        )
    except TargetContractError as exc:
        notes.append(
            _Note(
                f"target {name!r} answered the probe with an unusable response: {exc}",
                True,
            )
        )
    except (ConfigError, TargetError, ValueError) as exc:
        # The adapter refused its own configuration -- a ``kwargs_map`` field the framework does
        # not have, a ``metadata_map`` beside a stdout-based ``returns`` -- and only the adapter
        # can see that: the mapping's vocabulary is its own, not the schema's. So this is the
        # exit-2 configuration fault ``qrcheck run`` calls it, naming the dotted key, and the arm
        # follows the five adapter errors because those answer a request the adapter accepted.
        notes.append(_Note(f"configuration error: {exc}", True))
    except ImportError as exc:
        # The adapter's own module could not be imported, so a dependency this host declares is
        # absent or broken. That is an environment fault (exit 2) and names the lock file, not
        # an internal error: adapters convert their own import failures, so this is the layer
        # above them.
        notes.append(
            _Note(
                f"the {str(target.type)!r} adapter could not be imported: {exc}; install the "
                f"pins in requirements.lock",
                True,
            )
        )
    except KeyError as exc:
        notes.append(
            _Note(
                f"no adapter implementation is registered for type {str(target.type)!r}: {exc}",
                True,
            )
        )
        outcome = replace(outcome, internal_failure=True)
    except Exception as exc:  # Reported below and mapped to exit 3: never swallowed.
        # An adapter may raise only the five declared errors, so anything else is a framework
        # defect. It is named with its type rather than hidden behind a generic message.
        notes.append(
            _Note(
                f"target {name!r} raised {type(exc).__name__} instead of a declared adapter "
                f"error: {exc}",
                True,
            )
        )
        outcome = replace(outcome, internal_failure=True)
    finally:
        if adapter_obj is not None and not _close_adapter(adapter_obj, name, notes):
            outcome = replace(outcome, internal_failure=True)
    return outcome


def _close_adapter(adapter_obj: Adapter, name: str, notes: list[_Note]) -> bool:
    """Close ``adapter_obj``, reporting a contract breach instead of leaking it.

    The port documents ``close()`` as safe to call twice and as not raising for the ordinary
    reasons, so a failure here means the adapter broke that contract. A client left open would
    outlive the command, so the fault is reported and the command exits 3 rather than
    pretending the probe left nothing behind.

    Returns:
        ``True`` when the adapter closed cleanly, ``False`` when closing raised.
    """
    try:
        adapter_obj.close()
    except Exception as exc:  # Reported as a blocking fault: the adapter broke its contract.
        notes.append(
            _Note(
                f"closing target {name!r} failed with {type(exc).__name__}: {exc}",
                True,
            )
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Emitting the report: once, on every path
# ---------------------------------------------------------------------------


def _printable(text: str) -> str:
    """Return ``text`` with every non-printable character escaped for a terminal or CI log.

    Printable text is returned unchanged. Every control, format, surrogate and line or paragraph
    separator character is replaced by its ``\\xNN`` / ``\\uNNNN`` spelling, so no escape sequence
    can act on a terminal or a CI log -- rewriting a log line, taking over a window title, or
    reversing the text after a bidi override -- while the message still reads as the text a target
    supplied. Nothing is dropped, so the rendering stays a faithful transcript of the note.
    """
    return "".join(
        character
        if unicodedata.category(character) not in _UNPRINTABLE_CATEGORIES
        else character.encode("unicode_escape").decode("ascii")
        for character in text
    )


def _finish(
    report: _DoctorReport,
    notes: Sequence[_Note],
    exit_code: int,
    as_json: bool,
) -> int:
    """Write the diagnostics, then the report, and return ``exit_code``.

    The report object is emitted on the failing paths too, because a CI consumer that receives
    nothing cannot tell an absent toolchain from a broken command. Diagnostics go to stderr
    first, so a caller interleaving both streams sees the reasons before the verdict, and every
    message is rendered through :func:`_printable`, because a target's own text is untrusted. The
    JSON form comes from :func:`qrcheck.report.json_report.dumps_canonical`, the product's single
    canonical formatter, so key order, separators and the non-finite-float policy have one owner
    rather than one per command.
    """
    for note in notes:
        click.echo(f"{_DIAGNOSTIC_PREFIX}{_printable(note.message)}", err=True)
    if as_json:
        click.echo(dumps_canonical(report.as_dict()))
    else:
        for line in _table_lines(report, notes):
            click.echo(line)
    return exit_code


def _table_lines(report: _DoctorReport, notes: Sequence[_Note]) -> list[str]:
    """Render the human report: one row per prerequisite, then the verdict.

    Column widths come from the values being printed, so the table is aligned and still
    byte-identical between two runs on the same host. Nothing here reads a clock, a locale or an
    environment variable.
    """
    capabilities = ",".join(report.capabilities) or "<unset>"
    lines = [
        f"adapter={report.adapter or '<unset>'} type={report.target_type or '<unset>'} "
        f"reachable={'yes' if report.reachable else 'no'}",
        f"capabilities={capabilities}",
        "",
    ]
    name_width = max([len(_NAME_HEADER), *(len(entry.name) for entry in report.dependencies)])
    version_width = max(
        [len(_VERSION_HEADER), *(len(entry.version or "-") for entry in report.dependencies)]
    )
    status_width = max([len(_STATUS_HEADER), *(len(entry.status) for entry in report.dependencies)])
    lines.append(
        f"{_NAME_HEADER:<{name_width}}  {_VERSION_HEADER:<{version_width}}  "
        f"{_STATUS_HEADER:<{status_width}}  {_BLOCKING_HEADER}"
    )
    for entry in report.dependencies:
        blocking = "yes" if entry.blocking else "no"
        lines.append(
            f"{entry.name:<{name_width}}  {entry.version or '-':<{version_width}}  "
            f"{entry.status:<{status_width}}  {blocking}"
        )
    lines.append(_rasterizer_line(report.svg_rasterizer))
    lines.append(_verdict_line(notes))
    return lines


def _rasterizer_line(status: Mapping[str, object] | None) -> str:
    """Render the rasterizer status line, including the order that was attempted."""
    if status is None:
        return "svg_rasterizer=<unset> (the configuration did not load)"
    attempted = status.get("attempted")
    entries = attempted if isinstance(attempted, list) else []
    order = ",".join(str(entry) for entry in entries) or "nothing"
    configured = status.get("configured")
    name = status.get("name") or "none"
    return f"svg_rasterizer={name} configured={configured} attempted={order}"


def _verdict_line(notes: Sequence[_Note]) -> str:
    """Render the verdict, naming the first blocking failure when there is one.

    The named message is rendered through :func:`_printable` for the same reason the diagnostics
    are: this line reaches a terminal or a CI log, and the message it quotes can be a target's own
    text.
    """
    for note in notes:
        if note.blocking:
            return f"verdict: blocking failure: {_printable(note.message)}"
    if notes:
        return f"verdict: ok with degradation: {_printable(notes[0].message)}"
    return "verdict: ok"


# ---------------------------------------------------------------------------
# The click command
# ---------------------------------------------------------------------------


@click.command(name="doctor", cls=ExitCodeCommand)
@click.option(
    "--adapter",
    type=str,
    default=None,
    help="Target to verify; defaults to default_adapter in the configuration.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the machine-readable report instead of the table.",
)
@click.pass_context
def doctor_command(ctx: click.Context, adapter: str | None, as_json: bool) -> int:
    """Verify the target and the local toolchain before any check runs."""
    return doctor(adapter, as_json, config_path=resolve_config_path(ctx))
