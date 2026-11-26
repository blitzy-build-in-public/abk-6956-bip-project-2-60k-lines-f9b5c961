"""The one line-count metric, its exclusion list, and the ``--limit`` gate on top of it.

``qrcheck/loc.py`` is the product's only definition of "tracked physical lines" and
``qrcheck/cli/loc.py`` the only place that compares that figure with the ceiling, so a second
counting rule anywhere -- in the command, in a report writer, in a test -- would let
``RunResult.loc_count`` and the CI gate disagree about one tree. Four levels are held: the counting
rule (every physical line counts, blank and comment lines included, a final unterminated line
counts too, and the ``LF`` byte rather than decoded text is counted); the walk (the name-based
exclusion list prunes generated and metadata directories, no symlink is followed, report paths are
root-relative POSIX strings, the collections are sorted and immutable and ``total`` is the sum of
the per-file counts); the gate (``0`` under the limit, ``1`` at or above it, ``2`` for a missing
path, and the configured ``loc_limit`` when no option overrides it); and the property that makes
the rest worth anything -- no other module defines a line-counting function and the command imports
the metric instead of repeating the walk.

The gate boundary is "``0`` when the total is under the limit and ``1`` when it is at or above it"
(AAP §0.3.4 and ``qrcheck/cli/loc.py``'s own docstring), which is what makes ``qrcheck loc --limit
$((count - 1))`` fail -- the negative case success criterion 11 turns into a CI gate (AAP §0.6.1
#11). All three neighbouring limits are therefore asserted, because an off-by-one in the comparison
moves exactly one of them and a single assertion could sit on the side the defect does not reach.

Every limit is computed from the tree just measured or read from
:data:`qrcheck.config.schema.SCHEMA`, never typed, and the repository's own total is never
asserted: the module stays green as the product grows, and retuning ``loc_limit`` in
``qrcheck.toml`` cannot leave a stale expectation behind it. Each tree is built under ``tmp_path``
and the one real-repository walk is a file listing rather than an import.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Final

import pytest

from qrcheck.config import load_config
from qrcheck.config.schema import SCHEMA
from qrcheck.loc import (
    EXCLUDED_DIR_NAMES,
    EXCLUDED_DIR_SUFFIXES,
    INCLUDED_FILE_SUFFIX,
    FileLines,
    LocReport,
    count_lines,
)

# The repository root, derived from this file's own location so the one real-tree walk is
# independent of the working directory pytest was launched from:
# tests/unit/test_loc.py -> tests/unit -> tests -> <root>.
_REPOSITORY: Final[Path] = Path(__file__).resolve().parents[2]

# The package under the AST scan, named from the path rather than by importing it.
_PACKAGE: Final[str] = "qrcheck"
_METRIC_MODULE: Final[str] = "qrcheck/loc.py"
_COMMAND_MODULE: Final[str] = "qrcheck/cli/loc.py"

# The modules the scan must always reach, so a moved or emptied package fails here instead of
# turning the "no second definition" assertions into a scan of nothing.
_SCANNED_MODULES: Final[tuple[str, ...]] = (
    _COMMAND_MODULE,
    _METRIC_MODULE,
    "qrcheck/report/model.py",
    "qrcheck/runner.py",
)

# A floor on the number of modules parsed, so the scan cannot pass vacuously.  It is the plan's
# tree inventory (AAP §0.9.1), not a product threshold and not a retunable limit.
_MINIMUM_SCANNED_MODULES: Final[int] = 20

# The exit codes of AAP §0.3.4, whose single owner is qrcheck/cli/exit.py: `loc` returns 0 under
# the limit, 1 at or above it, 2 for a configuration fault or an unusable path.
_EXIT_OK: Final[int] = 0
_EXIT_FAILED: Final[int] = 1
_EXIT_CONFIG: Final[int] = 2

# The directory names every sweep must prune (AAP §0.9.1, §0.5 "build/ and dist/ hold generated
# copies of the package"); ".tmp" is the local scratch directory a suite run uses as its TMPDIR and
# "env" a per-checkout virtualenv, so neither holds tracked source.  Neither list is restated by
# the metric: the assertions below check that these names are the ones it declares, then that the
# walk demonstrably skips them.
_ALWAYS_PRUNED_DIRS: Final[tuple[str, ...]] = (
    ".git", ".venv", "__pycache__", "build", "dist", ".tmp", "env",
)

# The setuptools metadata directory, identified by suffix because its name follows the package.
_EGG_INFO_DIR: Final[str] = "something.egg-info"

# Files that are not tracked source and so must contribute nothing to the count.  The upper-case
# ``.PY`` member pins the case-sensitive match: a tracked source file is ``.py``.
_UNTRACKED_FILES: Final[tuple[str, ...]] = ("notes.txt", "data.json", "module.pyc", "shout.PY")

# The call names a second counting implementation would need: a directory walk, a directory
# glob, or a byte read of a source file.  None of them may appear in the command module.
_REIMPLEMENTATION_CALLS: Final[frozenset[str]] = frozenset(
    {"walk", "rglob", "glob", "read_bytes", "read_text", "readlines", "splitlines"}
)

# One payload per counting-rule case, with the count that payload must produce.  Each expected
# value is a fact about the payload below it -- how many physical lines it holds -- and never a
# limit or a threshold of the product.
_COUNTING_CASES: Final[tuple[tuple[str, bytes, int], ...]] = (
    # Two lines, the second unterminated: the last line still counts.
    ("unterminated-last-line", b"a = 1\nb = 2", 2),
    # No bytes at all is zero lines, not one.
    ("empty-file", b"", 0),
    # One terminator and nothing else is one blank line.
    ("single-blank-line", b"\n", 1),
    # Blank lines are physical lines.
    ("blank-lines-only", b"\n\n\n\n", 4),
    # Trailing blank lines count; they are not trimmed away.
    ("trailing-blank-lines", b"x = 1\n\n\n", 3),
    # A very long single line is still one line.
    ("single-long-line", b"VALUE = '" + b"a" * 8000 + b"'", 1),
    # A CRLF file counts one line per line: the count is taken on the LF byte.
    ("crlf-terminated", b"a = 1\r\nb = 2\r\n", 2),
    ("crlf-unterminated", b"a = 1\r\nb = 2", 2),
    # A multi-byte character does not add a line, terminated or not.
    ("multi-byte-terminated", "café = 'naïve 日本語 😀'\n".encode(), 1),
    ("multi-byte-unterminated", "café = 'naïve 日本語 😀'".encode(), 1),
)

# One source file spelling out every kind of physical line the count must include: a docstring,
# a full-line comment, a trailing comment, blank runs and five code lines.
_DOCSTRING_SOURCE_LINES: Final[tuple[str, ...]] = (
    '"""A module docstring: code, and one physical line."""',
    "",
    "# a full-line comment counts",
    "import os  # a trailing comment counts",
    "",
    "",
    "VALUE = 1",
    "",
    "",
    "def double(number):",
    "    return number * 2",
    "",
    "RESULT = double(VALUE)",
)

# How many of the lines above a comment-and-blank-stripping measure would keep: the import, the
# assignment, the two definition lines and the trailing call.  The count must exceed it, which is
# what proves the blanks and the comments are included rather than filtered out.
_CODE_LINE_COUNT: Final[int] = 5


def _tree(base: Path, payloads: Mapping[str, bytes], name: str = "tree") -> Path:
    """Create a directory tree of files beneath ``base`` and return its root."""
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def _link(target: Path, link: Path) -> bool:
    """Create a symbolic link and report whether the platform allowed it. A symlink is created
    without ``Path.exists`` validation and without following the target, so a test can point
    one outside the measured tree.
    """
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        return False
    return True


def _function_names(tree: ast.Module) -> tuple[str, ...]:
    """Return every function name defined in ``tree``, including nested ones, sorted."""
    return tuple(
        sorted(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    )


def _is_line_counting(name: str) -> bool:
    """Whether ``name`` reads as a line-counting function. The test is on the snake-case segments
    rather than on a substring, because ``block`` and ``local`` contain ``loc`` while neither
    counts a line, and a substring test would flag a correct module.
    """
    lowered = name.lower()
    if "count_lines" in lowered:
        return True
    return "loc" in lowered.lstrip("_").split("_")


def _called_names(tree: ast.Module) -> frozenset[str]:
    """Return every function or method name called in ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    return frozenset(names)


def _imported_names(tree: ast.Module, module: str) -> frozenset[str]:
    """Return the names ``tree`` imports from ``module``."""
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imported.update(alias.name for alias in node.names)
    return frozenset(imported)


@lru_cache(maxsize=1)
def _package_modules() -> dict[str, ast.Module]:
    """Parse every module under ``qrcheck/`` once, keyed by repository-relative POSIX path. Parsed
    from the files on disk rather than imported, so a package that cannot be imported cannot
    hide a second counting implementation by failing first.
    """
    root = _REPOSITORY / _PACKAGE
    return {
        path.relative_to(_REPOSITORY).as_posix(): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(root.rglob(f"*{INCLUDED_FILE_SUFFIX}"))
    }


# ---------------------------------------------------------------------------
# 1. The counting rule: every physical line, counted on the LF byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "payload", "expected"),
    _COUNTING_CASES,
    ids=[case for case, _, _ in _COUNTING_CASES],
)
def test_every_physical_line_is_counted(
    tmp_path: Path, case: str, payload: bytes, expected: int
) -> None:
    """One physical line, one count -- blanks, comments and unterminated endings included."""
    tree = _tree(tmp_path, {"measured.py": payload})
    report = count_lines(tree)
    assert report.total == expected, case
    assert [entry.path for entry in report.files] == ["measured.py"], case
    assert report.files[0].lines == expected, case


def test_a_docstring_comment_and_blank_lines_all_count(tmp_path: Path) -> None:
    """The count includes the docstring, both comments and every blank line.

    Only :data:`_CODE_LINE_COUNT` of the file's lines would survive a comment-and-blank-stripping
    measure, so a count that equalled it would prove the lines were filtered rather than counted.
    """
    payload = ("\n".join(_DOCSTRING_SOURCE_LINES) + "\n").encode("utf-8")
    tree = _tree(tmp_path, {"stated.py": payload})
    report = count_lines(tree)
    assert payload.endswith(b"\n")
    assert report.total == len(_DOCSTRING_SOURCE_LINES)
    assert report.total > _CODE_LINE_COUNT


def test_the_count_is_taken_on_the_lf_byte_in_the_raw_bytes(tmp_path: Path) -> None:
    """A multi-byte character neither adds nor loses a line: the LF byte decides. The promise is
    byte-identity, so the assertion is made against ``bytes.count(b" ")`` of the payload itself
    rather than against another reading of the same decoded text.
    """
    terminated = "café = 'naïve 日本語 😀'\n".encode("utf-8")
    unterminated = terminated.rstrip(b"\n")
    assert len(terminated) > len(terminated.decode("utf-8"))
    tree = _tree(tmp_path, {"terminated.py": terminated, "unterminated.py": unterminated})
    counted = {entry.path: entry.lines for entry in count_lines(tree).files}
    assert counted["terminated.py"] == terminated.count(b"\n")
    assert counted["unterminated.py"] == unterminated.count(b"\n") + 1
    assert sum(counted.values()) == 2


# ---------------------------------------------------------------------------
# 2. The walk: what is pruned, what is counted, and how a path is reported
# ---------------------------------------------------------------------------


def test_the_exclusion_constants_name_the_untracked_directories() -> None:
    """The metric declares the generated and metadata directories the sweep must prune. ``build/``
    and ``dist/`` hold generated copies of the package, so counting them would inflate the
    metric after a wheel build, and ``.git/`` would make the figure depend on the clone (AAP
    §0.5.3, §0.12).
    """
    for name in _ALWAYS_PRUNED_DIRS:
        assert name in EXCLUDED_DIR_NAMES
    assert ".egg-info" in EXCLUDED_DIR_SUFFIXES
    assert INCLUDED_FILE_SUFFIX == ".py"


def test_generated_directories_contribute_nothing_and_are_reported(tmp_path: Path) -> None:
    """Every pruned directory is skipped and named in ``excluded``, in sorted order."""
    payloads = {
        f"{name}/inside.py": b"x = 1\ny = 2\n" for name in (*_ALWAYS_PRUNED_DIRS, _EGG_INFO_DIR)
    }
    payloads["kept.py"] = b"kept = True\n"
    tree = _tree(tmp_path, payloads)
    report = count_lines(tree)
    assert [entry.path for entry in report.files] == ["kept.py"]
    assert report.total == 1
    assert report.excluded == tuple(sorted((*_ALWAYS_PRUNED_DIRS, _EGG_INFO_DIR)))
    assert report.excluded == tuple(sorted(report.excluded))
    for entry in report.excluded:
        assert entry in EXCLUDED_DIR_NAMES or entry.endswith(EXCLUDED_DIR_SUFFIXES)


def test_only_python_files_are_counted(tmp_path: Path) -> None:
    """A tracked source file is ``.py``, matched case-sensitively; nothing else contributes."""
    payloads = {name: b"a\nb\nc\n" for name in _UNTRACKED_FILES}
    payloads["tracked.py"] = b"tracked = 1\n"
    tree = _tree(tmp_path, payloads)
    report = count_lines(tree)
    assert [entry.path for entry in report.files] == ["tracked.py"]
    assert report.total == 1


def test_a_nested_excluded_directory_is_reported_by_its_relative_path(tmp_path: Path) -> None:
    """``excluded`` holds root-relative POSIX paths, so a nested pruning site is identifiable."""
    tree = _tree(
        tmp_path,
        {
            "pkg/build/generated.py": b"generated = 1\n",
            "pkg/__pycache__/cached.py": b"cached = 1\n",
            "pkg/source.py": b"source = 1\nsource_2 = 2\n",
        },
    )
    report = count_lines(tree)
    assert [entry.path for entry in report.files] == ["pkg/source.py"]
    assert report.excluded == ("pkg/__pycache__", "pkg/build")
    assert report.total == 2


def test_symlinks_are_not_followed_and_nothing_outside_the_root_is_visited(tmp_path: Path) -> None:
    """A link is neither followed nor counted, so no file is counted twice and no cycle is walked.
    A link that is unavailable on the platform leaves the same exact file list behind, so the
    assertions hold either way and nothing is skipped.
    """
    outside = _tree(tmp_path, {"leak.py": b"leaked = 1\n"}, name="outside")
    tree = _tree(
        tmp_path,
        {"kept.py": b"kept = 1\nkept_2 = 2\n", "pkg/source.py": b"source = 1\n"},
    )
    links = {
        "kept-alias.py": _link(tree / "kept.py", tree / "kept-alias.py"),
        "pkg-alias": _link(tree / "pkg", tree / "pkg-alias"),
        "outside-alias": _link(outside, tree / "outside-alias"),
    }
    report = count_lines(tree)
    paths = [entry.path for entry in report.files]
    assert paths == ["kept.py", "pkg/source.py"]
    assert len(paths) == len(set(paths))
    assert report.total == 3
    assert not any(path.startswith(("pkg-alias/", "outside-alias/")) for path in paths)
    for name, was_created in links.items():
        if was_created:
            assert (tree / name).is_symlink()
    for entry in report.files:
        assert ".." not in Path(entry.path).parts
        assert (tree / entry.path).is_file()


# ---------------------------------------------------------------------------
# 3. The report: sorted root-relative POSIX paths, an immutable model, one total
# ---------------------------------------------------------------------------


def test_report_paths_are_sorted_root_relative_posix_strings(tmp_path: Path) -> None:
    """Paths are sorted, relative to the counted root and POSIX on any platform."""
    tree = _tree(tmp_path, {"b.py": b"b = 1\n", "a.py": b"a = 1\n", "pkg/c.py": b"c = 1\n"})
    report = count_lines(tree)
    paths = [entry.path for entry in report.files]
    assert paths == ["a.py", "b.py", "pkg/c.py"]
    for path in paths:
        assert not path.startswith("/")
        assert ".." not in Path(path).parts
        assert "\\" not in path
        assert Path(path).as_posix() == path
        assert (tree / path).is_file()
    assert report.root == tree.resolve()


def test_the_total_is_the_sum_and_the_collections_are_immutable(tmp_path: Path) -> None:
    """``total`` is computed from the reported files, and neither collection can be edited."""
    tree = _tree(
        tmp_path,
        {
            "a.py": b"a = 1\na_2 = 2\n",
            "pkg/b.py": b"b = 1\n",
            "build/c.py": b"c = 1\n",
        },
    )
    report = count_lines(tree)
    assert isinstance(report, LocReport)
    assert isinstance(report.files, tuple)
    assert isinstance(report.excluded, tuple)
    assert all(isinstance(entry, FileLines) for entry in report.files)
    assert report.total == sum(entry.lines for entry in report.files)
    assert report.total == 3
    assert report.excluded == ("build",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.files[0].lines = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.total = 0


def test_counting_the_same_tree_twice_returns_equal_reports(tmp_path: Path) -> None:
    """The measurement is a pure function of the tree: two sweeps agree in every field."""
    tree = _tree(
        tmp_path,
        {"a.py": b"a = 1\n", "pkg/b.py": b"b = 1\nb_2 = 2\n", "dist/c.py": b"c = 1\n"},
    )
    first = count_lines(tree)
    second = count_lines(tree)
    assert first == second
    assert first.files == second.files
    assert first.excluded == second.excluded
    assert first.total == second.total


def test_a_string_root_is_measured_as_its_path(tmp_path: Path) -> None:
    """The documented ``str | os.PathLike`` signature accepts a string root."""
    tree = _tree(tmp_path, {"a.py": b"a = 1\n"})
    report = count_lines(str(tree))
    assert report.total == 1
    assert report.root == tree.resolve()


def test_a_missing_root_is_reported_with_its_path(tmp_path: Path) -> None:
    """``count_lines`` names the path it could not measure; the command maps it to exit 2."""
    missing = tmp_path / "absent"
    with pytest.raises(FileNotFoundError) as raised:
        count_lines(missing)
    assert str(missing) in str(raised.value)


def test_a_file_root_is_not_a_directory(tmp_path: Path) -> None:
    """A regular file is refused rather than measured, and the diagnostic names it."""
    afile = tmp_path / "not-a-tree.py"
    afile.write_bytes(b"x = 1\n")
    with pytest.raises(NotADirectoryError) as raised:
        count_lines(afile)
    assert str(afile) in str(raised.value)


# ---------------------------------------------------------------------------
# 4. The real tree: countable, and it contains the metric and this module
# ---------------------------------------------------------------------------


def test_the_repository_tree_is_countable() -> None:
    """The product's own tree measures positive and includes the metric and this module. The total
    itself is deliberately not asserted: it changes with every feature, and the ceiling it is
    gated against is ``loc_limit`` in ``qrcheck.toml`` (AAP §0.6.2), not a number a test may
    carry.
    """
    report = count_lines(_REPOSITORY)
    paths = {entry.path for entry in report.files}
    assert report.total > 0
    assert report.total == sum(entry.lines for entry in report.files)
    assert _METRIC_MODULE in paths
    assert "tests/unit/test_loc.py" in paths
    assert INCLUDED_FILE_SUFFIX == ".py"
    # The repository root is a checkout, so ``.git`` is pruned from it; counting it would make
    # the metric depend on the clone rather than on the tracked sources (AAP §0.5.3).
    assert ".git" in report.excluded
    assert all(not path.startswith("/") and ".." not in Path(path).parts for path in paths)


# ---------------------------------------------------------------------------
# 5. The gate: `qrcheck loc --limit` and the exit-code contract
# ---------------------------------------------------------------------------


def test_the_gate_passes_under_the_limit(tmp_path: Path, invoke_cli) -> None:
    """A total below the limit exits 0, and stdout carries the per-file rows and the total."""
    tree = _tree(tmp_path, {"a.py": b"x = 1\ny = 2\n", "pkg/b.py": b"z = 3\n"})
    total = count_lines(tree).total
    limit = total + 1
    result = invoke_cli("loc", "--path", str(tree), "--limit", str(limit))
    assert result.exit_code == _EXIT_OK
    assert "\ta.py" in result.stdout
    assert "\tpkg/b.py" in result.stdout
    assert f"total\t{total}" in result.stdout
    assert f"limit\t{limit}" in result.stdout


def test_the_gate_fails_at_or_above_the_limit(tmp_path: Path, invoke_cli) -> None:
    """The boundary is "at or above", so a limit equal to the total already fails. All three
    neighbouring limits are asserted, because an off-by-one in the comparison moves exactly one
    of them: ``total + 1`` passes, ``total`` and ``total - 1`` fail, which is what makes
    ``qrcheck loc --limit $((count - 1))`` the negative case of AAP §0.6.1 #11.
    """
    tree = _tree(tmp_path, {"a.py": b"x = 1\ny = 2\n", "pkg/b.py": b"z = 3\n"})
    total = count_lines(tree).total
    passing = invoke_cli("loc", "--path", str(tree), "--limit", str(total + 1))
    at_the_limit = invoke_cli("loc", "--path", str(tree), "--limit", str(total))
    below_the_limit = invoke_cli("loc", "--path", str(tree), "--limit", str(total - 1))
    assert passing.exit_code == _EXIT_OK
    assert at_the_limit.exit_code == _EXIT_FAILED
    assert below_the_limit.exit_code == _EXIT_FAILED
    for result, limit in ((at_the_limit, total), (below_the_limit, total - 1)):
        assert f"total\t{total}" in result.stdout
        assert f"limit\t{limit}" in result.stdout
        assert "at-or-above" in result.stdout


def test_a_configured_limit_decides_the_gate_when_no_option_is_given(
    tmp_path: Path, invoke_cli, toml_config_file
) -> None:
    """Without ``--limit`` the gate reads ``loc_limit`` from the configuration file. The generous
    limit comes from :data:`qrcheck.config.schema.SCHEMA` -- the ceiling is never typed here --
    and the tight one is derived from the measured total, so retuning the budget in
    ``qrcheck.toml`` cannot leave a stale number behind in this test (AAP §0.6.2, §0.12).
    """
    tree = _tree(tmp_path, {"a.py": b"x = 1\ny = 2\n", "pkg/b.py": b"z = 3\n"})
    total = count_lines(tree).total
    generous = max(int(SCHEMA[""]["loc_limit"].default), total + 1)
    passing_config = toml_config_file(loc_limit=generous, name="generous.toml")
    assert load_config(passing_config).loc_limit == generous
    passing = invoke_cli("loc", "--path", str(tree), config=passing_config)
    assert passing.exit_code == _EXIT_OK
    assert f"limit\t{generous}" in passing.stdout
    tight = total - 1
    assert tight >= 1
    failing_config = toml_config_file(loc_limit=tight, name="tight.toml")
    assert load_config(failing_config).loc_limit == tight
    failing = invoke_cli("loc", "--path", str(tree), config=failing_config)
    assert failing.exit_code == _EXIT_FAILED
    assert f"limit\t{tight}" in failing.stdout
    assert f"total\t{total}" in failing.stdout


def test_a_missing_path_exits_with_the_configuration_code(tmp_path: Path, invoke_cli) -> None:
    """``--path`` naming nothing is exit 2, with the path in the diagnostic and no table."""
    missing = tmp_path / "absent-tree"
    result = invoke_cli("loc", "--path", str(missing))
    assert result.exit_code == _EXIT_CONFIG
    assert str(missing) in result.stderr
    assert result.stdout == ""


def test_a_file_given_as_the_path_exits_with_the_configuration_code(
    tmp_path: Path, invoke_cli
) -> None:
    """``--path`` naming a regular file is refused with exit 2, exactly like a missing one."""
    afile = tmp_path / "not-a-tree.py"
    afile.write_bytes(b"x = 1\n")
    result = invoke_cli("loc", "--path", str(afile))
    assert result.exit_code == _EXIT_CONFIG
    assert str(afile) in result.stderr
    assert result.stdout == ""


def test_a_non_positive_limit_exits_with_the_configuration_code(
    tmp_path: Path, invoke_cli
) -> None:
    """A limit below one can gate nothing, so it is refused before the tree is measured."""
    tree = _tree(tmp_path, {"a.py": b"x = 1\n"})
    result = invoke_cli("loc", "--path", str(tree), "--limit", "0")
    assert result.exit_code == _EXIT_CONFIG
    assert "limit" in result.stderr
    assert "0" in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# 6. One definition: no second line counter anywhere in the package
# ---------------------------------------------------------------------------


def test_the_scan_reaches_the_modules_it_judges() -> None:
    """The AST scan cannot pass vacuously: it parses the package and finds the metric."""
    modules = _package_modules()
    assert len(modules) >= _MINIMUM_SCANNED_MODULES
    for path in _SCANNED_MODULES:
        assert path in modules
    assert "count_lines" in _function_names(modules[_METRIC_MODULE])


def test_no_module_besides_the_metric_defines_a_line_counting_function() -> None:
    """``qrcheck/loc.py`` owns the count; the command defines the verb and nothing else."""
    definers: dict[str, tuple[str, ...]] = {}
    for path, tree in _package_modules().items():
        names = tuple(name for name in _function_names(tree) if _is_line_counting(name))
        if names:
            definers[path] = names
    assert "count_lines" in definers[_METRIC_MODULE]
    assert set(definers) <= {_METRIC_MODULE, _COMMAND_MODULE}
    assert set(definers.get(_COMMAND_MODULE, ())) <= {"loc", "loc_command"}


def test_the_loc_command_imports_the_metric_instead_of_reimplementing_it() -> None:
    """The command calls ``qrcheck.loc.count_lines`` and walks no tree of its own."""
    module = _package_modules()[_COMMAND_MODULE]
    assert "count_lines" in _imported_names(module, "qrcheck.loc")
    assert "count_lines" in _called_names(module)
    assert _called_names(module) & _REIMPLEMENTATION_CALLS == frozenset()
    assert "count_lines" not in _function_names(module)
