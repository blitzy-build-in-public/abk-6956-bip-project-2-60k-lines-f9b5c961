"""``CL-ARTIFACTS``: the run leaves no temporary file and no open handle behind.

AAP §0.8.1's ``checks/cleanup.py`` -> ``CL-ARTIFACTS`` line and the "``CL-ARTIFACTS``'s
observation" paragraph: one synthetic case, ``synthetic:cleanup``, that observes the framework's
own footprint instead of anything a target produced.  The observation has two halves.  The
**temporary-file half** snapshots :func:`tempfile.gettempdir` while this check's cases are
enumerated and compares it as a set of *names* against a second snapshot taken when the case
executes: every framework-owned temporary path is created under the product prefix
:data:`TEMP_PREFIX` -- the subprocess adapter's scratch directory
(``tempfile.mkdtemp(prefix="qrcheck-")`` in :mod:`qrcheck.targets.cli`) and the logo overlay
bitmap -- and each is removed in a ``finally``, so an entry that survives its case is exactly what
AAP §0.8.1 calls "no user-visible temporary files remain".  The **handle half** resolves every
entry of :func:`fd_directory` (``/proc/self/fd`` on Linux, ``/dev/fd`` on macOS) and compares it
against ``report_dir``: an entry that resolves under the report root is a file handle the framework
opened and did not close, and the images :func:`qrcheck.evidence.mux.open_image` registers for the
life of its context are asserted empty through the accessor that module exposes
(:func:`mux.open_images`), so an unclosed image is reported here rather than merely documented as
unobservable.

Zero adapter invocations: :data:`EXPECTED_ENCODES` declares ``0``, and the check never calls the
target, declares no capability requirement and no dependency token.  Together with ``DM-SEEDED``
and ``PF-DECODE`` it is one of the three checks that invoke no adapter, which is what lets a run
whose target is unreachable still report it as passing while every target-dependent check reports
its single error (AAP §0.6.6 row 6, ``tests/unit/test_runner_target_loss.py``).  A dependency token
is deliberately absent: a token can only turn a check into one ``missing_dependency`` skip for the
whole run, whereas this check reports its handle half per case as ``unsupported_platform`` while
its temporary-file half still asserts (AAP §0.8.1).

The ordering limitation, stated rather than papered over: AAP §0.8.1 describes the snapshot as
"before and after the run", but a check cannot see the end of a run from inside its own case, since
checks execute in ascending check-ID order (AAP §0.5.3) and ``CL-ARTIFACTS`` sorts first of the
twenty-one, so its case runs *before* every other check.  What it therefore observes is the window
from its own enumeration to its own execution, plus whatever handles happen to be open at that
moment.  Two consequences are accepted: the temporary-file half proves that start-up
(configuration load, target resolution, corpus load) and the window up to its own execution left
nothing behind and can say nothing about checks that have not run yet; and the handle half sees the
handles open at execution time, which is the moment the framework is holding its configuration,
corpus and target -- a live handle under ``report_dir`` then is still a defect, because nothing has
written there yet.  There is no end-of-run hook and no second observation point in another module,
since a faked one would put a result in the report that no case produced; the run-level end-state
assertion belongs to the end-to-end test module that drives a whole run and then inspects
``report_dir``, which is where the "after the run" half of AAP §0.8.1 lives.

Reasons emitted: ``value_mismatch`` for the single ``failed`` verdict, carrying the leaked names,
the fd paths that resolved under ``report_dir`` and the open images in ``decoded``; and
``unsupported_platform`` for the single ``skipped`` verdict, emitted only when
:func:`fd_directory` finds neither platform directory (or cannot list the one it found) *and*
nothing leaked, so the temporary-file half still asserted.  ``counts.skipped`` is its own integer
and a skip never counts as a pass.

It observes; it never cleans.  No file or directory is created, deleted or modified, no entry is
opened or stat'd, and no temporary path is joined onto anything: the temporary half is one
``os.listdir`` and set membership over the names it returned.  No clock is read, so
``duration_ms`` stays at the factory's zero instead of recording a wall-clock value no assertion
may use (AAP §0.5.5).  Nothing here calls ``hash()``, ``id()`` or ``random``, and every list it
records is sorted.  There is no bare or broad ``except``, no retry and no threshold literal: an fd
entry that cannot be resolved is skipped, and an fd directory that cannot be listed downgrades the
handle half to unsupported rather than failing the run on the host's behaviour.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from qrcheck.checks import support
from qrcheck.evidence import mux
from qrcheck.registry.check import Case, RunContext, check
from qrcheck.report.model import CaseResult, Reason

__all__ = [
    "CASE_ID",
    "CHECK_ID",
    "EXPECTED_ENCODES",
    "TEMP_PREFIX",
    "cases",
    "fd_directory",
    "leaked_temp_entries",
    "open_paths_under",
    "temp_entries",
]

# The one check this module owns, and the one case it contributes.
CHECK_ID: Final[str] = "CL-ARTIFACTS"
CASE_ID: Final[str] = "synthetic:cleanup"

# The product-wide prefix every framework-owned temporary path carries.  It is restated here
# rather than imported: ``qrcheck/checks/`` may import no adapter module, and
# ``qrcheck.targets.cli.TEMP_DIR_PREFIX`` holds the same value.  The two must stay equal by
# hand; a divergence widens the scan's blind spot by exactly the entries that stopped matching.
TEMP_PREFIX: Final[str] = "qrcheck-"

# The adapter-invocation count this check's contract fixes, declared where the negative tests
# can import it: zero, because it observes the framework and never the target.  A retry or a
# stray generate would register as a count above zero and fail that test.
EXPECTED_ENCODES: Final[Mapping[str, int]] = MappingProxyType({CHECK_ID: 0})

# The platform file-descriptor directories, in discovery order.  Linux exposes the process's
# own handles under /proc/self/fd; macOS exposes them under /dev/fd.  Neither existing means the
# handle half is unsupported on this host, which is reported per case rather than as a skip of
# the whole check.
_FD_DIRECTORIES: Final[tuple[Path, ...]] = (Path("/proc/self/fd"), Path("/dev/fd"))

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
    prefix.  No entry is stat'd, opened, joined onto a path or walked into, so the snapshot cannot
    itself create, touch or delay anything the next snapshot would then see.

    Args:
        prefix: The name prefix that marks an entry as framework-owned.  It defaults to
            :data:`TEMP_PREFIX`; a caller passes another only to observe a different product.

    Returns:
        The matching entry names, as an unordered set for set arithmetic against a later
        snapshot.  Empty when the temporary directory holds no matching entry or no longer
        exists at all.

    Raises:
        ValueError: `prefix` is not a non-empty string.  An empty prefix would match every entry
            and make the comparison meaningless.
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


def leaked_temp_entries(baseline: frozenset[str]) -> tuple[str, ...]:
    """Return the framework-prefixed names present now that were absent from `baseline`.

    Set difference over names, not a modification-time comparison: no clock is read, so the
    verdict is identical on every host and every run for the same sequence of events.

    Args:
        baseline: The snapshot :func:`temp_entries` returned when the check's cases were
            enumerated.

    Returns:
        The leaked entry names in ascending order, empty when the framework cleaned up after
        itself.
    """
    return tuple(sorted(temp_entries() - baseline))


def fd_directory() -> Path | None:
    """Return the platform's file-descriptor directory, or ``None`` where there is none.

    The single platform probe for this concern: Linux answers ``/proc/self/fd``, macOS answers
    ``/dev/fd``, and a host that exposes neither gets ``None``, which makes the handle half
    unsupported rather than silently clean.

    Returns:
        The first existing directory of :data:`_FD_DIRECTORIES`, else ``None``.
    """
    for candidate in _FD_DIRECTORIES:
        if candidate.is_dir():
            return candidate
    return None


def open_paths_under(directory: Path, root: Path) -> tuple[str, ...]:
    """Return the resolved paths in `directory` that lie at or under `root`.

    Every entry of `directory` is resolved to the path it actually names and compared against
    `root`, so a handle the framework left open on a file under the report directory is reported
    by name.  Entries that name no resolvable path -- a pipe, a socket, an anonymous inode, an
    entry that vanished between listing and resolution -- are skipped: they are not leaks this
    check can assert, and treating them as errors would fail a run on the host's behaviour.

    Args:
        directory: The file-descriptor directory, as :func:`fd_directory` returned it.
        root: The directory no open handle may lie under, normally the resolved configuration's
            ``report_dir``.

    Returns:
        The offending absolute paths in ascending order, empty when nothing is open under
        ``root``.

    Raises:
        OSError: `directory` cannot be listed -- it was removed, or the process may not read it.
            The caller decides what an unreadable descriptor directory means for its verdict;
            this helper never guesses by returning an empty result that would look clean.
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

    The link is read with ``os.readlink`` rather than resolved through the filesystem: a
    descriptor directory is a directory of symbolic links to the objects they describe, and the
    link target is the only statement about the object that does not have to be re-derived.  A
    link target that is not absolute names something other than a file -- ``pipe:[1234]``,
    ``socket:[5678]``, ``anon_inode:[eventfd]`` -- and resolving it would fabricate a path
    relative to the working directory, so it is skipped instead.

    Args:
        entry: One entry of a descriptor directory.

    Returns:
        The resolved path the entry names, or ``None`` when the entry cannot be resolved or does
        not name a path at all.
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

    No artefact records an absolute host path (AAP §0.5.5), so a path under the report directory
    is recorded relative to it -- the same form ``summary.json`` gives ``evidence_dir`` -- and an
    absolute path elsewhere is recorded by its name alone.  A path the caller already gave
    relatively is returned untouched, because it never carried the run's location.

    Args:
        path: The path as observed.
        report_dir: The run's report directory, as the resolved configuration gives it.

    Returns:
        A relative posix path when `path` lies under `report_dir`, the bare name otherwise.
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
    dependencies=(),
)
def cases(ctx: RunContext) -> Iterator[Case]:
    """Yield the single ``synthetic:cleanup`` case, its baseline already captured.

    The baseline is taken here, when the runner enumerates this check's cases, rather than when
    the case runs: that is the earliest observation the framework offers, because ``CL-ARTIFACTS``
    sorts first of the twenty-one check identifiers.  Capturing it before the case means an entry
    a sibling process left in the shared temporary directory is part of the baseline and cannot
    be mistaken for this run's leak.

    Args:
        ctx: The run's context, accepted for the calling convention every family's ``cases``
            shares.  Nothing is read from it here; the case reads it when it executes.

    Yields:
        Exactly one :class:`~qrcheck.registry.check.Case`, named :data:`CASE_ID`.
    """
    baseline = temp_entries()

    def execute(context: RunContext) -> CaseResult:
        """Return this case's verdict for the footprint observed right now."""
        return _verdict(context, baseline)

    yield support.synthetic_case(CHECK_ID, CASE_ID, execute)


def _verdict(ctx: RunContext, baseline: frozenset[str]) -> CaseResult:
    """Return the case's result for the footprint observed at execution time.

    Both halves are observed before any verdict is formed, so ``decoded`` describes the whole
    footprint even when one half alone already decided the outcome -- a reader diagnosing a leak
    should not have to guess what the other half would have said.

    Args:
        ctx: The run's context; its configuration owns ``report_dir`` and its seed is recorded
            on the result.
        baseline: The temporary-directory snapshot taken when the cases were enumerated.

    Returns:
        ``failed`` with reason ``value_mismatch`` when anything leaked or is open, ``skipped``
        with reason ``unsupported_platform`` when the handle half cannot be observed and nothing
        leaked, and ``passed`` when both halves are clean.
    """
    report_dir = ctx.config.report_dir
    leaked = leaked_temp_entries(baseline)
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
        "baseline_count": len(baseline),
        "leaked": list(leaked),
        "fd_directory": None if fd_dir is None else str(fd_dir),
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
