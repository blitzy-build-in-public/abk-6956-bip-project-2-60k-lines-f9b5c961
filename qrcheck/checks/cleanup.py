"""``CL-ARTIFACTS``: the run leaves no temporary file and no open handle behind.

One synthetic case, ``synthetic:cleanup``, observing the framework's own footprint rather than
anything a target produced (AAP §0.8.1), in two halves:

* **temporary-file half** -- the window is the run. The runner snapshots
  :func:`tempfile.gettempdir` once, through :func:`temp_entries`, immediately before the first
  check can run, and hands it to the checks as
  :attr:`~qrcheck.registry.check.RunContext.temp_baseline`; this check executes after the walk and
  compares that baseline against the names it finds then. Every framework-owned temporary path is
  created under the prefix :data:`TEMP_PREFIX` -- the subprocess adapter's scratch directory and
  the logo overlay bitmap -- and each is removed in a ``finally``.
* **handle half** -- every entry of :func:`fd_directory` (``/proc/self/fd`` on Linux, ``/dev/fd`` on
  macOS) resolved and compared against ``report_dir``, plus the images
  :func:`qrcheck.evidence.mux.open_image` registers, read through :func:`mux.open_images`. Which
  interface was enumerated is recorded as the platform token of :func:`fd_directory_token`, never as
  the directory's path, because no artefact may carry an absolute path (AAP §0.5.5).

The check executes at the end of the run because its contract declares ``observes_run_end=True``:
it asserts a property of the whole run rather than of one case, so the runner runs it after every
other check -- after a ``--fail-fast`` break too, since it never invokes the target -- and that is
what makes the two snapshots bracket the run itself (AAP §0.8.1, §0.5.3).

It relies on a per-run temporary directory: ``TMPDIR``, ``TMP`` or ``TEMP`` pointing at a directory
of this run's own. The baseline separates this run's entries from what was there before it, but a
sibling process that leaves a ``qrcheck-*`` entry in a shared system temporary directory *during*
the run would be reported as this run's leak, because the two are indistinguishable by name. A host
that runs many checkouts at once has to give each run a private temporary directory.

Zero adapter invocations: :data:`EXPECTED_ENCODES` declares ``0``, and the one dependency token is
the primary decoder, as AAP §0.3.4's row records, so a run whose target is unreachable still
reports this check as passing. That token can never turn the check into a ``missing_dependency``
skip: a run without the primary decoder exits 2 before resolution (AAP §0.6.6 row 3).

Reasons emitted: ``value_mismatch``, the single ``failed`` verdict, carrying the leaked names, the
fd paths under ``report_dir`` and the open images; and ``unsupported_platform``, the single
``skipped`` verdict, only when :func:`fd_directory` finds no usable directory *and* nothing leaked.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from qrcheck.checks import support
from qrcheck.registry.check import DEPENDENCY_PRIMARY_DECODER, Case, RunContext, check
from qrcheck.report.model import CaseResult, Reason

__all__ = [
    "CASE_ID",
    "CHECK_ID",
    "EXPECTED_ENCODES",
    "TEMP_PREFIX",
    "cases",
    "fd_directory",
    "fd_directory_token",
    "leaked_temp_entries",
    "open_paths_under",
    "temp_entries",
]

# The one check this module owns, and the one case it contributes.
CHECK_ID: Final[str] = "CL-ARTIFACTS"
CASE_ID: Final[str] = "synthetic:cleanup"

# The product-wide prefix every framework-owned temporary path carries.  Restated rather than
# imported because ``qrcheck/checks/`` may import no adapter module, and
# ``qrcheck.targets.cli.TEMP_DIR_PREFIX`` holds the same value: the two must stay equal by hand.
TEMP_PREFIX: Final[str] = "qrcheck-"

# The adapter-invocation count this check's contract fixes, declared where the negative tests
# can import it: zero, because it observes the framework and never the target.  A retry or a
# stray generate would register as a count above zero and fail that test.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType({CHECK_ID: 0})

# The platform file-descriptor directories, in discovery order: ``/proc/self/fd`` on Linux,
# ``/dev/fd`` on macOS.  A host that exposes neither gets ``None`` from :func:`fd_directory`, and
# that is what makes the handle half ``skipped``/``unsupported_platform`` per case while the
# temporary-file half still asserts.
_FD_DIRECTORIES: Final[tuple[Path, ...]] = (Path("/proc/self/fd"), Path("/dev/fd"))

# The artefact token of each descriptor directory: which platform interface was enumerated, stated
# without naming a host location, because no artefact may carry an absolute path (AAP §0.5.5).  One
# entry per member of :data:`_FD_DIRECTORIES`, so a third directory cannot be added without a token.
_FD_DIRECTORY_TOKENS: Final[Mapping[Path, str]] = MappingProxyType(
    {
        Path("/proc/self/fd"): "proc-self-fd",
        Path("/dev/fd"): "dev-fd",
    }
)

# The kernel's marker for an entry whose file was unlinked while the handle stayed open.  It is
# stripped before the path is compared, because an unlinked-but-open file under ``report_dir``
# is a leaked handle all the same.
_DELETED_SUFFIX: Final[str] = " (deleted)"

# The two values of the ``handle_half`` field of ``decoded``, so a reader of the artefact can
# tell "nothing is open" apart from "this platform cannot be asked".
_HANDLE_SUPPORTED: Final[str] = "supported"
_HANDLE_UNSUPPORTED: Final[str] = "unsupported"


def temp_entries(prefix: str = TEMP_PREFIX) -> frozenset[str]:
    """Return the names in the system temporary directory that start with `prefix`.

    The whole of the temporary-file half's observation: one directory listing, filtered by name
    prefix. No entry is stat'd, opened, joined onto a path or walked into, so the snapshot cannot
    itself create, touch or delay anything the next snapshot would see. The names come back as an
    unordered set for set arithmetic, empty when nothing matches or the directory no longer exists.
    The runner calls it once, immediately before the walk, to take the baseline
    :attr:`~qrcheck.registry.check.RunContext.temp_baseline` this check compares against at the
    run's end (AAP §0.8.1).

    Raises:
        ValueError: `prefix` is not a non-empty string. An empty prefix would match every entry and
        make the comparison meaningless.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError(f"temp_entries: prefix must be a non-empty str, got {prefix!r}")
    try:
        names = os.listdir(tempfile.gettempdir())
    except FileNotFoundError:
        # The temporary directory is gone, so it holds nothing: an empty snapshot is the honest
        # answer.  Any other OSError is a host fault the runner turns into an ``error`` result
        # carrying the traceback, which is the right outcome for a directory it cannot list.
        return frozenset()
    return frozenset(name for name in names if name.startswith(prefix))


def leaked_temp_entries(before: frozenset[str], after: frozenset[str]) -> tuple[str, ...]:
    """Return the framework-prefixed names of `after` that `before` did not hold.

    Set difference over the two snapshots, not a modification-time comparison: no clock is read, so
    the verdict is identical on every host and every run for the same sequence of events. Both
    snapshots are passed in rather than re-listed here, so the names the artefact reports are
    exactly the names the verdict compared. The leaked names come back in ascending order, empty
    when the framework cleaned up after itself.
    """
    return tuple(sorted(after - before))


def fd_directory() -> Path | None:
    """Return the platform's file-descriptor directory, or ``None`` where there is none.

    The single platform probe for this concern: Linux answers ``/proc/self/fd``, macOS answers
    ``/dev/fd``, and a host that exposes neither gets ``None``, which makes the handle half
    unsupported rather than silently clean. Returns the first existing directory of
    :data:`_FD_DIRECTORIES`.
    """
    for candidate in _FD_DIRECTORIES:
        if candidate.is_dir():
            return candidate
    return None


def fd_directory_token(directory: Path | None) -> str | None:
    """Return the artefact token of a descriptor directory, ``None`` for ``None``.

    ``/proc/self/fd`` is recorded as ``proc-self-fd`` and ``/dev/fd`` as ``dev-fd``: the token says
    which platform interface :func:`fd_directory` enumerated, which is what a reader of
    ``decoded.json`` needs to judge the handle half, and it names no host location, because no
    artefact may carry an absolute path (AAP §0.5.5). ``None`` stays ``None``: neither directory
    exists, which is what makes the handle half ``unsupported_platform``.

    Raises:
        ValueError: `directory` is not one of :data:`_FD_DIRECTORIES` and so has no token. A caller
        that reaches this has not passed what :func:`fd_directory` returned, and recording ``None``
        would report a missing platform interface instead of the defect.
    """
    if directory is None:
        return None
    token = _FD_DIRECTORY_TOKENS.get(Path(directory))
    if token is None:
        raise ValueError(
            f"fd_directory_token: {directory} is not one of "
            f"{tuple(str(candidate) for candidate in _FD_DIRECTORIES)}"
        )
    return token


def open_paths_under(directory: Path, root: Path) -> tuple[str, ...]:
    """Return the resolved paths in `directory` that lie at or under `root`.

    Every entry is resolved to the path it actually names and compared against `root`, so a handle
    left open on a file under the report directory is reported by name. Entries that name no
    resolvable path
    -- a pipe, a socket, an anonymous inode, one that vanished between listing and resolution -- are
    skipped: they are not leaks this check can assert.

    Raises:
            OSError: `directory` cannot be listed -- it was removed, or the process may not read it.
            The caller decides what that means for its verdict; this helper never guesses by
            returning an empty result that would look clean.
    """
    target_root = Path(os.path.realpath(root))
    found: set[str] = set()
    for name in sorted(os.listdir(directory)):
        target = _fd_target(Path(directory) / name)
        if target is not None and _is_under(target, target_root):
            found.add(str(target))
    return tuple(sorted(found))


def _fd_target(entry: Path) -> Path | None:
    """Return the path a descriptor-directory `entry` names, or ``None`` for a non-path entry.

    The link is read with ``os.readlink`` rather than resolved through the filesystem: the link
    target is the only statement about the object that need not be re-derived. A target that is not
    absolute names something other than a file -- ``pipe:[1234]``, ``socket:[5678]``,
    ``anon_inode:[eventfd]`` -- and resolving it would fabricate a path relative to the working
    directory, so it is skipped.
    """
    try:
        link = os.readlink(entry)
    except OSError:
        # The entry disappeared between the listing and this read, or the platform refused to
        # describe it.  Neither is a leak this check can name, and neither is worth an error.
        return None
    if link.endswith(_DELETED_SUFFIX):
        link = link[: -len(_DELETED_SUFFIX)]
    if not link or not os.path.isabs(link):
        return None
    return Path(os.path.realpath(link))


def _is_under(path: Path, root: Path) -> bool:
    """Return whether `path` is `root` itself or lies below it. Both must be resolved."""
    return path == root or root in path.parents


def _report_relative(path: str, report_dir: Path) -> str:
    """Return `path` in the portable form an artefact may carry.

    No artefact records an absolute host path (AAP §0.5.5), so a path under `report_dir` is
    recorded relative to it -- the same form ``summary.json`` gives ``evidence_dir`` -- and an
    absolute path elsewhere is recorded by its name alone. A path the caller already gave relatively
    is returned untouched, because it never carried the run's location.
    """
    if not os.path.isabs(path):
        return path
    root = Path(os.path.realpath(report_dir))
    resolved = Path(os.path.realpath(path))
    if not _is_under(resolved, root):
        return resolved.name
    return resolved.relative_to(root).as_posix()


@check(
    id=CHECK_ID,
    title="No temporary files or image handles left behind",
    family="cleanup",
    requires=frozenset(),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
    observes_run_end=True,
)
def cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the single ``synthetic:cleanup`` case, compared against the run's own baseline.

    Nothing is snapshotted here: the baseline is the one the runner took before the first check
    could run and carried on :attr:`~qrcheck.registry.check.RunContext.temp_baseline`, so
    enumeration neither reads the filesystem nor fixes a window of its own. The verdict is formed
    when the case executes, which the runner arranges to be after the walk, so what the case
    compares at either end is the run (AAP §0.8.1).

    Yields:
        Exactly one :class:`~qrcheck.registry.check.Case`, named :data:`CASE_ID`.
    """
    baseline = ctx.temp_baseline

    def execute(context: RunContext) -> CaseResult:
        """Return this case's verdict for the run-end footprint observed right now."""
        return _verdict(context, baseline)

    yield support.synthetic_case(CHECK_ID, CASE_ID, execute)


def _verdict(ctx: RunContext, baseline: frozenset[str]) -> CaseResult:
    """Return the case's result for the run's footprint, observed at the run's end.

    Both halves are observed before any verdict is formed, so ``decoded`` describes the whole
    footprint even when one half alone already decided the outcome, and it reports both windows of
    the temporary-file comparison: the names the run began with and the names present now. Returns
    ``failed`` with reason ``value_mismatch`` when anything leaked or is open, ``skipped`` with
    reason ``unsupported_platform`` when the handle half cannot be observed and nothing leaked, and
    ``passed`` when both halves are clean.
    """
    # Deferred: importing this module starts no image stack in a process that only lists checks.
    from qrcheck.evidence import mux

    report_dir = ctx.config.report_dir
    after = temp_entries()
    leaked = leaked_temp_entries(baseline, after)
    fd_dir = fd_directory()
    handle_supported = fd_dir is not None
    handle_error: str | None = None
    open_under: tuple[str, ...] = ()
    if fd_dir is not None:
        try:
            open_under = open_paths_under(fd_dir, report_dir)
        except OSError as exc:
            # An unreadable descriptor directory makes the handle half unobservable rather than
            # failing a run on a host's restriction; the reason stays in evidence either way.
            handle_supported = False
            handle_error = f"{type(exc).__name__}: {exc}"
    open_images = mux.open_images()
    violations: list[str] = []
    if leaked:
        violations.append("leaked_temp_entries")
    if open_under:
        violations.append("open_under_report_dir")
    if open_images:
        violations.append("open_images")
    decoded: dict[str, Any] = {
        "temp_prefix": TEMP_PREFIX,
        "temp_entries_before": sorted(baseline),
        "temp_entries_after": sorted(after),
        "leaked": list(leaked),
        "fd_directory": fd_directory_token(fd_dir),
        "handle_half": _HANDLE_SUPPORTED if handle_supported else _HANDLE_UNSUPPORTED,
        "open_under_report_dir": [_report_relative(path, report_dir) for path in open_under],
        "open_image_count": len(open_images),
        "open_images": [
            {"case_id": entry.case_id, "path": _report_relative(entry.path, report_dir)}
            for entry in open_images
        ],
        "violations": violations,
    }
    if handle_error is not None:
        decoded["handle_half_error"] = handle_error
    expected: dict[str, Any] = {
        "leaked_temp_entries": [],
        "open_under_report_dir": [],
        "open_images": [],
        "handle_half": _HANDLE_SUPPORTED,
    }
    verdict: dict[str, Any] = {"decoded": decoded, "expected": expected, "seed": ctx.seed}
    if violations:
        failing = support.fail(CHECK_ID, CASE_ID, reason=Reason.VALUE_MISMATCH, **verdict)
        return support.record_evidence(ctx, failing)
    if not handle_supported:
        incomplete = support.skip(CHECK_ID, CASE_ID, reason=Reason.UNSUPPORTED_PLATFORM, **verdict)
        return support.record_evidence(ctx, incomplete)
    return support.pass_(CHECK_ID, CASE_ID, **verdict)
