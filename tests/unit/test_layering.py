"""The inward-only dependency direction, held mechanically by an AST scan of the package.

The architecture's dependency direction is a contract, not a convention (AAP §0.3.2, §0.3.4
"Extension boundary", NFR-6, §0.12), and this module asserts it by reading the package's source, so
a breach fails a test instead of relying on a reviewer: ``qrcheck/checks/**`` may import
``qrcheck.targets.base`` and nothing else from ``qrcheck.targets`` -- not a concrete adapter, and
not the ``qrcheck.targets`` package itself, whose entry point is the only module allowed to know
the concrete class names, since denying only the five leaf names would leave ``import
qrcheck.targets`` and ``from qrcheck.targets import Capability`` open; ``qrcheck/targets/**`` never
imports ``qrcheck.checks`` in any spelling; nothing under ``qrcheck/`` imports anything under
``tests``; and the scan cannot pass vacuously, because both subtrees must still hold the modules
the plan's tree fixes, synthetic offenders must be flagged by the same collector the assertions
use, and the modules permitted to name a concrete adapter are one explicit, sorted allow-list.

Why the AST and not a search over the source text: a comment, a docstring or a log line that
mentions ``qrcheck.targets.cli`` is not a dependency and a text search would report it as one,
while the same search also misses the relative spelling -- every import here is read from the
parsed tree, and a relative ``from ..targets.cli import CliAdapter`` is resolved through the
importing file's own package path before it is judged.

Limitations, stated rather than discovered later: a module name computed at run time is invisible
to any static scan, though a literal adapter module name passed to ``importlib.import_module`` or
``__import__``, or stored in a table, is caught (:func:`_adapter_references`); the scan judges the
two boundaries the plan names and is not a general import-cycle detector; and no verdict is stored,
so a fresh run always re-reads the files on disk. The module imports no part of the product
(:func:`test_this_module_does_not_import_the_product` holds that), because a package that cannot be
imported must not be able to mask a layering breach by failing first.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final, NamedTuple

# The repository root, from this file's own location so the scan is independent of the working
# directory pytest is invoked from: tests/unit/test_layering.py -> tests/unit -> tests -> root.
_REPOSITORY: Final[Path] = Path(__file__).resolve().parents[2]

# The package under scrutiny, named from the path rather than by importing it.
_PACKAGE: Final[str] = "qrcheck"
_PRODUCT_ROOT: Final[Path] = _REPOSITORY / _PACKAGE

# The four scopes a reference is judged against, as dotted module paths and as directories.
_TARGETS: Final[str] = f"{_PACKAGE}.targets"
_PORT: Final[str] = f"{_TARGETS}.base"
_CHECKS: Final[str] = f"{_PACKAGE}.checks"
_TEST_SUITE: Final[str] = "tests"
_CHECKS_SCOPE: Final[str] = f"{_PACKAGE}/checks"
_TARGETS_SCOPE: Final[str] = f"{_PACKAGE}/targets"

# The five concrete adapters of the plan's tree (AAP §0.9.1), sorted, each of which a check
# must never name and only the factory table may name.
_ADAPTER_MODULES: Final[tuple[str, ...]] = (
    f"{_TARGETS}.cli",
    f"{_TARGETS}.defective",
    f"{_TARGETS}.http",
    f"{_TARGETS}.library",
    f"{_TARGETS}.reference",
)

# The module counts the plan's tree fixes (AAP §0.9.1).  A floor, not a target: qrcheck/checks/
# holds __init__, support and the nine family modules, and qrcheck/targets/ holds __init__,
# base and the five adapters.  A rename or a move that empties either subtree must fail here
# rather than turn the rules below into a scan of nothing.
_PLANNED_CHECK_MODULES: Final[int] = 11
_PLANNED_TARGET_MODULES: Final[int] = 7

# The only modules allowed to name a concrete adapter or the adapter package, sorted, each with
# the plan item that permits it.  Nothing else in the package may; adding an importer is a
# deliberate one-line change here rather than an accidental pass.
ALLOWED_ADAPTER_CONSUMERS: Final[frozenset[str]] = frozenset(
    {
        # Resolves a target through the one factory map during `qrcheck doctor`'s pre-flight
        # probe (AAP §0.8.1 "doctor" row, §0.7.2).
        "qrcheck/cli/doctor.py",
        # Resolves a target through the same map for a run (AAP §0.8.1 "run" row); listed
        # ahead of its creation because the factory's contract names it as a consumer.
        "qrcheck/cli/run.py",
        # The factory table itself: the single owner of the five concrete module names, which
        # it resolves lazily and so imports none of (AAP §0.3.4 extension boundary).
        "qrcheck/targets/__init__.py",
        # DefectiveAdapter reuses the conforming generator's encode path and injects the five
        # declared defects (AAP §0.4.2), so it builds on ReferenceAdapter directly.
        "qrcheck/targets/defective.py",
    }
)

# The only modules inside qrcheck/targets/ allowed to reach a sibling adapter.  Every other
# adapter reaches the port alone, which is what lets a fourth kind be added without touching
# an existing one; DefectiveAdapter is the declared exception above (AAP §0.4.2).
ALLOWED_SIBLING_ADAPTER_IMPORTERS: Final[frozenset[str]] = frozenset(
    {"qrcheck/targets/defective.py"}
)

# The call names that import a module named by their first argument: a literal adapter module
# name handed to one of them reaches an adapter without any import statement.
_IMPORT_HOOKS: Final[frozenset[str]] = frozenset({"import_module", "__import__"})


class _Reference(NamedTuple):
    """One module reference found in a source file."""

    path: str
    lineno: int
    module: str


def _python_files(root: Path) -> tuple[Path, ...]:
    """Return every ``.py`` file under ``root``, sorted, excluding bytecode caches. Sorted so that
    a diagnostic lists offenders in a stable order, which is what lets a reader diff two
    failures and lets this module's own messages be asserted verbatim.
    """
    return tuple(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))


def _package_of(path: str) -> str:
    """Return the dotted package a repository-relative module path lives in."""
    return ".".join(path.split("/")[:-1])


def _absolute_module(node: ast.ImportFrom, package: str) -> str:
    """Resolve an ``ImportFrom`` to the absolute module it names, relative forms included."""
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    keep = len(parts) - (node.level - 1)
    prefix = ".".join(parts[:keep]) if keep > 0 else ""
    tail = node.module or ""
    if prefix and tail:
        return f"{prefix}.{tail}"
    return prefix or tail


def _references(tree: ast.AST, path: str) -> tuple[_Reference, ...]:
    """Return every module reference in one parsed file, sorted. Only syntax reaches this
    function: a comment is absent from the AST and a docstring is an expression, so neither can
    be mistaken for a dependency.
    """
    package = _package_of(path)
    found: list[_Reference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(_Reference(path, node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_module(node, package)
            if base:
                found.append(_Reference(path, node.lineno, base))
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.append(
                    _Reference(path, node.lineno, f"{base}.{alias.name}" if base else alias.name)
                )
    return tuple(sorted(found))


def _docstring_constants(tree: ast.AST) -> frozenset[int]:
    """Return the identities of the string constants that are module, class or function
    docstrings. Docstrings talk *about* the adapters throughout this package, so they must
    never count as references; identifying them by ``id`` is exact, whereas any textual rule
    would have to guess where prose ends.
    """
    bodies = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, bodies):
            continue
        body = getattr(node, "body", ())
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return frozenset(docstrings)


def _literal_references(tree: ast.AST, path: str) -> tuple[_Reference, ...]:
    """Return the references a file makes by naming an adapter module in code, not in prose. Both
    are recorded so a module outside :data:`ALLOWED_ADAPTER_CONSUMERS` cannot reach an adapter
    that way either.
    """
    docstrings = _docstring_constants(tree)
    names = set(_ADAPTER_MODULES) | {_TARGETS}
    found: list[_Reference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            hook = node.func
            name = hook.attr if isinstance(hook, ast.Attribute) else getattr(hook, "id", "")
            if name in _IMPORT_HOOKS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.append(_Reference(path, node.lineno, first.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings and node.value in names:
                found.append(_Reference(path, node.lineno, node.value))
    return tuple(sorted(found))


@lru_cache(maxsize=None)
def _sources(root: Path) -> tuple[tuple[str, ast.AST], ...]:
    """Parse every ``.py`` file under ``root`` once per process, as (path, tree) pairs. The ASTs
    are memoised so the contract tests below cost one pass over the tree instead of one each;
    the memo holds parsed source and nothing else -- no verdict, no allow-list result, nothing
    read from disk a second time and nothing written anywhere.
    """
    base = root.parent
    parsed: list[tuple[str, ast.AST]] = []
    for path in _python_files(root):
        relative = path.relative_to(base).as_posix()
        source = path.read_text(encoding="utf-8")
        parsed.append((relative, ast.parse(source, filename=str(path))))
    return tuple(parsed)


def _scan(root: Path) -> tuple[_Reference, ...]:
    """Return every import reference under ``root``, sorted across the whole subtree."""
    found: list[_Reference] = []
    for relative, tree in _sources(root):
        found.extend(_references(tree, relative))
    return tuple(sorted(found))


def _adapter_references(root: Path) -> tuple[_Reference, ...]:
    """Return every import *and* literal reference under ``root`` that could reach an adapter."""
    found: list[_Reference] = []
    for relative, tree in _sources(root):
        found.extend(_references(tree, relative))
        found.extend(_literal_references(tree, relative))
    return tuple(sorted(found))


def _literal_scan(root: Path) -> tuple[_Reference, ...]:
    """Return only the literal references under ``root``, without the import statements. Split out
    so a control can assert what the literal half sees on its own: prose that names an adapter
    must appear in neither half.
    """
    found: list[_Reference] = []
    for relative, tree in _sources(root):
        found.extend(_literal_references(tree, relative))
    return tuple(sorted(found))



def _within(path: str, scope: str) -> bool:
    """Report whether a repository-relative path sits in ``scope`` or a subdirectory of it."""
    return path == scope or path.startswith(f"{scope}/")


def _violations(
    references: Sequence[_Reference],
    *,
    scope: str | None,
    reaches: Callable[[str], bool],
) -> list[tuple[str, int, str]]:
    """Return every reference that ``reaches`` accepts, one entry per offending statement. One
    statement can contribute several candidate module paths -- the resolved module plus each
    imported name -- so the shortest offending candidate is the one reported, and the entries
    come back sorted by ``(path, lineno)``: a failure lists every offender in a stable order
    instead of the first one found.
    """
    chosen: dict[tuple[str, int], str] = {}
    for reference in references:
        if scope is not None and not _within(reference.path, scope):
            continue
        if not reaches(reference.module):
            continue
        key = (reference.path, reference.lineno)
        best = chosen.get(key)
        if best is None or reference.module < best:
            chosen[key] = reference.module
    return sorted((path, lineno, module) for (path, lineno), module in chosen.items())


def _format(violations: Sequence[tuple[str, int, str]]) -> str:
    """Render violations as the ``path:lineno:module`` lines the failure messages carry."""
    return "\n".join(f"  {path}:{lineno}:{module}" for path, lineno, module in violations)


def _is_port(module: str) -> bool:
    """Report whether a module path is :mod:`qrcheck.targets.base` or a name inside it."""
    return module == _PORT or module.startswith(f"{_PORT}.")


def _beyond_the_port(module: str) -> bool:
    """Report whether ``module`` reaches anything of the adapter layer but the port. The
    ``qrcheck.targets`` package itself counts: it is the adapter layer's entry point and the
    one module allowed to know the concrete class names, so a check importing it is reaching
    for an implementation rather than the abstraction.
    """
    if module == _TARGETS:
        return True
    if not module.startswith(f"{_TARGETS}."):
        return False
    return not _is_port(module)


def _reaches_the_check_layer(module: str) -> bool:
    """Report whether ``module`` is :mod:`qrcheck.checks` or anything inside it."""
    return module == _CHECKS or module.startswith(f"{_CHECKS}.")


def _reaches_the_test_suite(module: str) -> bool:
    """Report whether ``module`` is the ``tests`` package or anything inside it."""
    return module == _TEST_SUITE or module.startswith(f"{_TEST_SUITE}.")


def _names_an_adapter(module: str) -> bool:
    """Report whether ``module`` names the adapter package or one of its concrete adapters."""
    if module == _TARGETS:
        return True
    return any(
        module == adapter or module.startswith(f"{adapter}.") for adapter in _ADAPTER_MODULES
    )


def test_checks_layer_reaches_only_the_adapter_port() -> None:
    """``qrcheck/checks/**`` imports :mod:`qrcheck.targets.base` and nothing else of that layer.
    AAP §0.3.4: a check depends on the abstraction and never on a concrete implementation, so a
    fourth adapter kind needs no change to any check.
    """
    violations = _violations(_scan(_PRODUCT_ROOT), scope=_CHECKS_SCOPE, reaches=_beyond_the_port)
    assert not violations, (
        "Check modules may import qrcheck.targets.base and nothing else from qrcheck.targets "
        "(AAP §0.3.4 extension boundary). Offenders, as path:lineno:module:\n"
        f"{_format(violations)}"
    )


def test_target_layer_never_reaches_the_check_layer() -> None:
    """``qrcheck/targets/**`` never imports :mod:`qrcheck.checks`, in any spelling. The seam is
    one-directional: the port is written for its consumers, never against them, so an adapter
    cannot be tailored to a check.
    """
    violations = _violations(
        _scan(_PRODUCT_ROOT), scope=_TARGETS_SCOPE, reaches=_reaches_the_check_layer
    )
    assert not violations, (
        "Adapter modules must never import the check layer (AAP §0.3.4 extension boundary). "
        "Offenders, as path:lineno:module:\n"
        f"{_format(violations)}"
    )


def test_product_never_reaches_the_test_suite() -> None:
    """Nothing under ``qrcheck/`` imports anything under ``tests``. The product ships without its
    tests: an import of the suite from the package would make the installed distribution depend
    on files that are not installed, and would let a fixture stand in for production code.
    """
    violations = _violations(_scan(_PRODUCT_ROOT), scope=None, reaches=_reaches_the_test_suite)
    assert not violations, (
        "The package must not import its own test suite. Offenders, as path:lineno:module:\n"
        f"{_format(violations)}"
    )


def test_scan_covers_the_planned_modules() -> None:
    """The scan visits the modules the plan's tree fixes, so no rule above can pass vacuously.

    A layering test that scans nothing is worse than none, and a directory rename is how that
    happens: both subtrees are counted against the plan's module inventory (AAP §0.9.1) and
    every file must parse, so an emptied or moved subtree fails here instead of passing quietly.
    """
    assert _PRODUCT_ROOT.is_dir(), f"the package directory {_PRODUCT_ROOT} is missing"
    checks = _python_files(_PRODUCT_ROOT / "checks")
    targets = _python_files(_PRODUCT_ROOT / "targets")
    assert len(checks) >= _PLANNED_CHECK_MODULES, (
        f"qrcheck/checks/ holds {len(checks)} module(s); the plan's tree fixes at least "
        f"{_PLANNED_CHECK_MODULES} (AAP §0.9.1)"
    )
    assert len(targets) >= _PLANNED_TARGET_MODULES, (
        f"qrcheck/targets/ holds {len(targets)} module(s); the plan's tree fixes at least "
        f"{_PLANNED_TARGET_MODULES} (AAP §0.9.1)"
    )
    scanned = [path for path, _ in _sources(_PRODUCT_ROOT)]
    assert scanned, "the scan parsed no module at all"
    assert len(scanned) == len(_python_files(_PRODUCT_ROOT)), (
        "the scan did not visit every .py file under qrcheck/"
    )
    assert f"{_CHECKS_SCOPE}/support.py" in scanned, (
        "the shared check support module is missing from the scan"
    )
    assert all(path.startswith(f"{_PACKAGE}/") for path in scanned), (
        "every scanned path is repository-relative and starts with the package name"
    )


def test_adapters_are_named_only_by_the_declared_modules() -> None:
    """Only :data:`ALLOWED_ADAPTER_CONSUMERS` reaches an adapter, by import or by literal name."""
    violations = _violations(
        _adapter_references(_PRODUCT_ROOT), scope=None, reaches=_names_an_adapter
    )
    assert violations, (
        "the scan found no module naming an adapter at all, so this rule would pass vacuously: "
        "qrcheck/targets/__init__.py is expected to hold the factory's five module names"
    )
    unexpected = [entry for entry in violations if entry[0] not in ALLOWED_ADAPTER_CONSUMERS]
    assert not unexpected, (
        "Only the declared consumers may name a concrete adapter or the adapter package "
        f"({', '.join(sorted(ALLOWED_ADAPTER_CONSUMERS))}). Offenders, as "
        f"path:lineno:module:\n{_format(unexpected)}"
    )


def test_adapters_import_siblings_only_where_declared() -> None:
    """Inside ``qrcheck/targets/``, an adapter reaches a sibling only where that is declared."""
    violations = _violations(_scan(_PRODUCT_ROOT), scope=_TARGETS_SCOPE, reaches=_names_an_adapter)
    unexpected = [
        entry for entry in violations if entry[0] not in ALLOWED_SIBLING_ADAPTER_IMPORTERS
    ]
    assert not unexpected, (
        "Adapter modules reach the port alone, except where declared in "
        "ALLOWED_SIBLING_ADAPTER_IMPORTERS. Offenders, as path:lineno:module:\n"
        f"{_format(unexpected)}"
    )


def _control_tree(tmp_path: Path) -> Path:
    """Write a miniature package of synthetic offenders and return its directory. The tree mirrors
    the real layout (``qrcheck/checks``, ``qrcheck/targets``, ``qrcheck/cli``) so the same
    collector, the same package-path resolution and the same scope filters judge it.
    """
    package = tmp_path / _PACKAGE
    for directory in ("", "checks", "targets", "cli"):
        (package / directory).mkdir(parents=True, exist_ok=True)
        (package / directory / "__init__.py").write_text("", encoding="utf-8")
    files = {
        "checks/absolute.py": "from qrcheck.targets.cli import CliAdapter\n",
        "checks/relative.py": "from ..targets.reference import ReferenceAdapter\n",
        "checks/package.py": "import qrcheck.targets\n",
        "checks/uses_tests.py": "from tests.conftest import report_dir\n",
        "checks/prose.py": (
            '"""qrcheck.targets.defective"""\n'
            "\n"
            "# from qrcheck.targets.cli import CliAdapter -- prose, not a dependency.\n"
            "from qrcheck.targets.base import Adapter\n"
        ),
        "targets/violator.py": "from qrcheck.checks import support\n",
        "cli/naming.py": (
            "import importlib\n"
            "\n"
            'TARGET = "qrcheck.targets.library"\n'
            "\n"
            "\n"
            "def load():\n"
            '    return importlib.import_module("qrcheck.targets.http")\n'
        ),
    }
    for relative, source in files.items():
        (package / relative).write_text(source, encoding="utf-8")
    return package


def test_scanner_flags_a_synthetic_checks_offender(tmp_path: Path) -> None:
    """The collector reports an absolute, a relative and a package-only adapter reference. The
    module's own sanity: if the resolver regressed -- a wrong package path, an unjoined alias,
    a dropped relative level -- the rules above would still pass, and this test is what fails
    instead.
    """
    package = _control_tree(tmp_path)
    violations = _violations(_scan(package), scope=_CHECKS_SCOPE, reaches=_beyond_the_port)
    assert violations == [
        ("qrcheck/checks/absolute.py", 1, "qrcheck.targets.cli"),
        ("qrcheck/checks/package.py", 1, "qrcheck.targets"),
        ("qrcheck/checks/relative.py", 1, "qrcheck.targets.reference"),
    ], f"unexpected offenders: {_format(violations)}"


def test_scanner_ignores_prose_and_the_port(tmp_path: Path) -> None:
    """A comment, a docstring and a port import are not dependencies and are never reported. The
    AST, not a text search, is what makes this true: the same words that would fail a grep sit
    in ``checks/prose.py`` beside a legal ``from qrcheck.targets.base import Adapter``.
    """
    package = _control_tree(tmp_path)
    checks = _violations(_scan(package), scope=_CHECKS_SCOPE, reaches=_beyond_the_port)
    reported = {path for path, _, _ in checks}
    assert "qrcheck/checks/prose.py" not in reported, (
        f"prose was mistaken for a dependency: {_format(checks)}"
    )
    literals = _violations(
        _literal_scan(package), scope=_CHECKS_SCOPE, reaches=_names_an_adapter
    )
    assert not literals, f"a docstring was read as a literal: {_format(literals)}"


def test_scanner_flags_synthetic_literals_and_cross_layer_spellings(tmp_path: Path) -> None:
    """A literal adapter name and both reversed directions are reported, with their lines. The
    factory keeps its entries as literal module names, so the literal half of the scan is what
    holds the allow-list over a lazily resolved table; the two reversed imports are the other
    direction of the contract, checked here in the same synthetic tree.
    """
    package = _control_tree(tmp_path)
    literals = _violations(
        _literal_scan(package), scope=f"{_PACKAGE}/cli", reaches=_names_an_adapter
    )
    assert literals == [
        ("qrcheck/cli/naming.py", 3, "qrcheck.targets.library"),
        ("qrcheck/cli/naming.py", 7, "qrcheck.targets.http"),
    ], f"unexpected literal offenders: {_format(literals)}"
    reversed_layer = _violations(
        _scan(package), scope=_TARGETS_SCOPE, reaches=_reaches_the_check_layer
    )
    assert reversed_layer == [
        ("qrcheck/targets/violator.py", 1, "qrcheck.checks")
    ], f"unexpected reversals: {_format(reversed_layer)}"
    into_tests = _violations(_scan(package), scope=None, reaches=_reaches_the_test_suite)
    assert into_tests == [
        ("qrcheck/checks/uses_tests.py", 1, "tests.conftest")
    ], f"unexpected test-suite imports: {_format(into_tests)}"


def test_this_module_does_not_import_the_product() -> None:
    """This module reads the product's source and never imports it. The scan has to hold on a
    package that cannot be imported -- a half-installed checkout, a missing native decoder --
    or a broken import would mask the layering breach it is meant to report.
    """
    own_path = Path(__file__).resolve()
    relative = own_path.relative_to(_REPOSITORY).as_posix()
    tree = ast.parse(own_path.read_text(encoding="utf-8"), filename=str(own_path))
    violations = _violations(
        _references(tree, relative),
        scope=None,
        reaches=lambda module: module == _PACKAGE or module.startswith(f"{_PACKAGE}."),
    )
    assert not violations, (
        "this test module must not import the product it inspects. Offenders, as "
        f"path:lineno:module:\n{_format(violations)}"
    )
