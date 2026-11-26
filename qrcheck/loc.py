"""The single definition of the qrcheck line budget: tracked physical lines.

``count_lines`` measures every tracked ``.py`` file beneath one root and counts
its physical lines -- blank lines, comment lines and a final line that carries no
terminator included. One implementation answers the repository-wide ceiling, so
the CI gate, the ``qrcheck loc`` table and ``RunResult.loc_count`` cannot
disagree about the number.

Lines are counted on the ``LF`` byte in the raw bytes rather than on decoded
text, so the locale, an encoding declaration, a byte-order mark or a decode error
cannot move the count, and a file that is not valid UTF-8 is counted instead of
aborting the walk. The measurement is pure and deterministic: sub-directories are
pruned before the walk descends and every emitted collection is sorted; nothing is
cached, printed or written, and the filesystem is the only thing touched, so there
is no network, no subprocess, no adapter error and no report reason here.

Failure modes: ``FileNotFoundError`` when the root does not exist and
``NotADirectoryError`` when it names anything other than a directory, each naming
the offending path (the caller maps both to exit code 2). A read error on one
file propagates unchanged rather than being skipped, so an incomplete sweep can
never be reported as the total.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["FileLines", "LocReport", "count_lines"]

# Pruned wherever they appear, matched on the directory name alone: the root may
# be a fixture tree, so an absolute prefix would tie the sweep to a location.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", ".tox", ".nox",
        ".eggs", "build", "dist", "node_modules", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", "htmlcov",
    }
)

# setuptools metadata directories are named after the package they describe, so
# only a suffix identifies them.
EXCLUDED_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)

# Matched case-sensitively: a tracked source file is ``.py``, and ``.PY`` is not.
INCLUDED_FILE_SUFFIX: str = ".py"


@dataclass(frozen=True, slots=True)
class FileLines:
    """Physical line count of one tracked file, keyed by its report path."""

    path: str
    lines: int


@dataclass(frozen=True, slots=True)
class LocReport:
    """One complete measurement of a root: per-file counts, pruned dirs, total."""

    root: Path
    files: tuple[FileLines, ...]
    excluded: tuple[str, ...]
    total: int


def count_lines(root: str | os.PathLike[str]) -> LocReport:
    """Count the physical lines of every tracked ``.py`` file beneath ``root``.

    ``files`` is sorted by ``path`` and ``total`` is the sum of their counts,
    computed once here so no later property can report a different figure.

    Raises:
        FileNotFoundError: ``root`` does not exist; the message names the path.
        NotADirectoryError: ``root`` is not a directory; the message names it.
    """
    given = Path(root)
    if not given.exists():
        raise FileNotFoundError(f"line-count root does not exist: {os.fspath(given)}")
    if not given.is_dir():
        raise NotADirectoryError(f"line-count root is not a directory: {os.fspath(given)}")

    # Enumeration starts from the lexical absolute path -- a pure string
    # operation, no symlink resolution -- so each walked dir is an unambiguous
    # prefix of the paths it yields and a report path never needs normalizing.
    base = Path(os.path.abspath(os.fspath(given)))
    files: list[FileLines] = []
    excluded: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        # Sorted in place before the walk descends, so traversal order -- not
        # only the final report -- is independent of filesystem enumeration.
        dirnames.sort()
        filenames.sort()
        kept: list[str] = []
        for name in dirnames:
            if name in EXCLUDED_DIR_NAMES or name.endswith(EXCLUDED_DIR_SUFFIXES):
                excluded.append(_report_path(Path(dirpath, name), base))
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if not name.endswith(INCLUDED_FILE_SUFFIX):
                continue
            path = Path(dirpath, name)
            # A symlink is skipped rather than followed, so a link cannot count
            # one file twice or walk the sweep into a cycle.
            if path.is_symlink():
                continue
            files.append(FileLines(_report_path(path, base), _count_file(path)))
    files.sort(key=lambda entry: entry.path)
    excluded.sort()
    return LocReport(
        root=given.resolve(),
        files=tuple(files),
        excluded=tuple(excluded),
        total=sum(entry.lines for entry in files),
    )


def _report_path(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` with forward slashes on any platform."""
    return path.relative_to(base).as_posix()


def _count_file(path: Path) -> int:
    """Count one file's physical lines by counting ``LF`` bytes.

    An empty file holds no lines. A final line with no terminator still counts as
    one line, which is what keeps a hand-written file at its real length.
    """
    data = path.read_bytes()
    if not data:
        return 0
    terminated = data.count(b"\n")
    return terminated if data.endswith(b"\n") else terminated + 1
