"""Whether each pinned dependency can actually be imported, and the diagnostic when it cannot.

Every package the product declares in ``requirements.lock`` is a *blocking* prerequisite
(AAP §0.7.2): a broken installation is an environment error -- exit code 2, decided before a single
case runs, naming the distribution -- never a silent degradation and never an internal error.
Deciding that needs one fact: **does the module the product imports load?** Installed metadata is
not that fact: a distribution present in ``site-packages`` can still fail to import (a shadowing
module on ``sys.path``, an extension built for another architecture) while
:func:`importlib.metadata.version` reports a version, so this module probes the import itself.

Standard library only, importing nothing from :mod:`qrcheck`, because it is reached *when the
environment is broken*; it reads no configuration, decides no exit code and records no state.
Whatever a module raises while being imported is that pin's failure reason -- a module that cannot
initialise is a broken install -- and only a :exc:`BaseException` propagates.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import platform
import sysconfig
from typing import Final

__all__ = [
    "CORE_DEPENDENCIES",
    "DependencyProbe",
    "LOCK_FILE",
    "PRIMARY_DECODER_DISTRIBUTION",
    "PRIMARY_DECODER_MODULE",
    "SECONDARY_DECODER_DISTRIBUTION",
    "SECONDARY_DECODER_MODULE",
    "blocking_problem",
    "import_problem",
    "probe",
    "problems",
]

#: The lock file every pin comes from, named verbatim in every diagnostic (AAP §0.7.1).
LOCK_FILE: Final[str] = "requirements.lock"

#: The primary reference decoder -- the authority for every round-trip assertion, so its absence
#: blocks the run (AAP §0.6.6 row 3) -- and the secondary one, a second opinion only, whose absence
#: degrades the run to single-decoder mode instead of failing it (AAP §0.6.6 row 4).
PRIMARY_DECODER_DISTRIBUTION: Final[str] = "zxing-cpp"
PRIMARY_DECODER_MODULE: Final[str] = "zxingcpp"
SECONDARY_DECODER_DISTRIBUTION: Final[str] = "opencv-python-headless"
SECONDARY_DECODER_MODULE: Final[str] = "cv2"

#: The oracle and the core libraries as ``(distribution, module)`` pairs, in declaration order: the
#: module is the one the product really imports -- ``PIL.Image`` rather than ``Pillow``, because
#: metadata says nothing about whether the image stack loads. Each blocks the run (AAP §0.7.2).
CORE_DEPENDENCIES: Final[tuple[tuple[str, str], ...]] = (
    ("segno", "segno"),
    ("numpy", "numpy"),
    ("Pillow", "PIL.Image"),
    ("httpx", "httpx"),
    ("click", "click"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class DependencyProbe:
    """What one ``(distribution, module)`` pair reports about this host.

    ``version`` is the installed distribution version, or ``None`` when the distribution is absent
    -- never a locked version restated. ``problem`` is why the pair is unusable, or ``None`` when
    the module imported; it is the only thing :attr:`usable` and :attr:`message` read.
    """

    distribution: str
    module: str
    version: str | None
    problem: str | None

    @property
    def usable(self) -> bool:
        """Whether the module imported, so the distribution can serve a run."""
        return self.problem is None

    @property
    def message(self) -> str:
        """Return the one-line diagnostic for this pair: the blocking one a caller writes before
        exiting 2 (AAP §0.7.2) when it is unusable, the verified sentence otherwise.
        """
        where = f"interpreter {platform.python_version()} on {sysconfig.get_platform()}"
        if self.problem is None:
            version = "an unknown version" if self.version is None else f"version {self.version}"
            return (
                f"the pinned runtime dependency {self.distribution} is usable: {version} of its "
                f"{self.module} module imports on {where}"
            )
        return (
            f"the pinned runtime dependency {self.distribution} cannot be used because its "
            f"{self.module} module cannot be imported ({self.problem}); {where}; it is a blocking "
            f"dependency, so install the pins in {LOCK_FILE}"
        )


def import_problem(module: str) -> str | None:
    """Import ``module`` and return why it failed, or ``None`` when it loaded.

    The guarded half of every probe, shared with the callers in :mod:`qrcheck.cli.doctor` and
    :mod:`qrcheck.runner`. Any :exc:`Exception` raised while importing is that dependency's failure
    reason, worded with its type so the diagnostic names what to fix; only a BaseException escapes.
    """
    try:
        importlib.import_module(module)
    except Exception as exc:
        return f"importing {module} failed: {type(exc).__name__}: {exc}"
    return None


def probe(distribution: str, module: str) -> DependencyProbe:
    """Probe one ``(distribution, module)`` pair and return what it reports.

    Two independent facts, both required: the distribution is installed and the module the product
    imports loads. The metadata is read first, so an absent distribution is reported as absent even
    where a same-named module happens to be importable.
    """
    version = _installed_version(distribution)
    reasons = (
        f"distribution {distribution} is not installed" if version is None else None,
        import_problem(module),
    )
    problem = "; ".join(reason for reason in reasons if reason is not None)
    return DependencyProbe(
        distribution=distribution, module=module, version=version, problem=problem or None
    )


def problems() -> tuple[DependencyProbe, ...]:
    """Return the unusable entries of :data:`CORE_DEPENDENCIES`, in table order."""
    return tuple(
        found
        for found in (probe(distribution, module) for distribution, module in CORE_DEPENDENCIES)
        if not found.usable
    )


def blocking_problem() -> DependencyProbe | None:
    """Return the first unusable core dependency in table order, or ``None``.

    The single decision ``qrcheck run`` needs before it starts. Table order, not alphabetical,
    so ``segno`` -- the oracle every oracle-backed assertion depends on -- comes first.
    """
    found = problems()
    return found[0] if found else None


def _installed_version(distribution: str) -> str | None:
    """Return the installed version of ``distribution``, or ``None`` when it is absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
