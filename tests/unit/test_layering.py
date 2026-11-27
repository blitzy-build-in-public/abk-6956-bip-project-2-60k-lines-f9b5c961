"""The inward-only dependency direction, held mechanically by an AST scan of the package.

The dependency direction is a contract, not a convention -- it is the extension boundary that
lets a fourth adapter kind be added with no check change -- so this module asserts it by reading
the package's source and lets a breach fail a test instead of a review: ``qrcheck/checks/**`` may
import ``qrcheck.targets.base`` and nothing else of that layer -- not a concrete adapter and not
the ``qrcheck.targets`` package itself, whose entry point is the one module allowed to know the
concrete class names; ``qrcheck/targets/**`` never imports ``qrcheck/checks``; nothing under
``qrcheck/`` imports anything under ``tests``; and the scan cannot pass vacuously, because both
subtrees must still hold at least :data:`_MIN_CHECK_MODULES` and
:data:`_MIN_TARGET_MODULES` modules.

The AST rather than a text search: a comment, a docstring or a log line naming an adapter is not
a dependency, while a relative ``from ..targets.cli import ...`` is one a text search misses. A
literal adapter module name passed to ``importlib.import_module`` or ``__import__`` is caught
anywhere in the package (:func:`_adapter_references`), and inside ``qrcheck/checks/**`` an import
hook may name its module only by a literal or by a name provably inside ``qrcheck.checks``
(:func:`_computed_imports`) -- the one dynamic import that layer has a use for, loading its own
nine family modules -- so the computed spelling that defeats a literal scan meets this one. The
rule judges a call to a hook by name; a hook bound to an alias and called through it stays
outside it, and holding that would need the scan to run the code it reads.
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

# The five concrete adapters ``qrcheck.targets`` ships, sorted, each of which a check must never
# name and only the factory table may name.
_ADAPTER_MODULES: Final[tuple[str, ...]] = (
    f"{_TARGETS}.cli",
    f"{_TARGETS}.defective",
    f"{_TARGETS}.http",
    f"{_TARGETS}.library",
    f"{_TARGETS}.reference",
)

# The floor on the modules the scan must visit, rather than a target: a rename or a move that
# empties either subtree must fail here rather than turn the rules below into a scan of nothing --
# qrcheck/checks/ holds __init__, support and nine families, qrcheck/targets/ seven.
_MIN_CHECK_MODULES: Final[int] = 11
_MIN_TARGET_MODULES: Final[int] = 7

# The only modules allowed to name a concrete adapter or the adapter package, sorted, each with the
# reason beside it; adding an importer is a deliberate change here, not an accidental pass.
ALLOWED_ADAPTER_CONSUMERS: Final[frozenset[str]] = frozenset(
    {
        # Resolves a target through the one factory map during `qrcheck doctor`'s pre-flight probe.
        "qrcheck/cli/doctor.py",
        # Resolves a target through the same map for a run.
        "qrcheck/cli/run.py",
        # The factory table itself: the single owner of the five concrete module names, which it
        # resolves lazily and so imports none of.
        "qrcheck/targets/__init__.py",
        # DefectiveAdapter reuses the conforming generator's encode path and injects the five
        # defects.
        "qrcheck/targets/defective.py",
    }
)

# The only modules inside qrcheck/targets/ allowed to reach a sibling adapter. Every other adapter
# reaches the port alone, which is what lets a fourth kind be added without touching an existing
# one.
ALLOWED_SIBLING_ADAPTER_IMPORTERS: Final[frozenset[str]] = frozenset(
    {"qrcheck/targets/defective.py"}
)

# The call names that import a module named by their first argument: a literal adapter module name
# handed to one of them reaches an adapter without any import statement.
_IMPORT_HOOKS: Final[frozenset[str]] = frozenset({"import_module", "__import__"})


class _Reference(NamedTuple):
    """One module reference found in a source file."""

    path: str
    lineno: int
    module: str


def _python_files(root: Path) -> tuple[Path, ...]:
    """Return every ``.py`` file under `root`, sorted, excluding bytecode caches."""
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
    """Return every module reference in one parsed file, sorted; comments never reach it."""
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
    """Return the ``id`` of every string constant that is a module, class or function docstring."""
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
    """Return the references a file makes by naming an adapter module in code, not in prose."""
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
    """Parse every ``.py`` file under `root` once per process, as (path, tree) pairs."""
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
    """Return only the literal references under `root`, without the import statements."""
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
    """Return every reference `reaches` accepts, one entry per offending statement, sorted."""
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
    """Report whether `module` reaches anything of the adapter layer but the port."""
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


def _hook_of(node: ast.AST) -> str:
    """Return the import hook a call target names, or ``""`` when it names none.

    A hook reached by attribute (``importlib.import_module``) and one resolved by
    ``getattr(importlib, "import_module")`` are the same dynamic import in two spellings, so both
    answer with the hook's own name.
    """
    if isinstance(node, ast.Name):
        return node.id if node.id in _IMPORT_HOOKS else ""
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr in _IMPORT_HOOKS else ""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            resolved = node.args[1].value
            if isinstance(resolved, str) and resolved in _IMPORT_HOOKS:
                return resolved
    return ""


def _module_argument(call: ast.Call) -> ast.AST | None:
    """Return the expression a dynamic import takes its module name from, or ``None``."""
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "name":
            return keyword.value
    return None


def _rooted_in_the_check_package(node: ast.AST | None) -> bool:
    """Report whether a module-name expression provably names a module inside ``qrcheck.checks``.

    Two forms can be proved without running the code: ``__name__``, which is ``qrcheck.checks`` in
    every module of the subtree, and a literal that is that package or a name inside it. A
    concatenation or an f-string is rooted by its leftmost operand, and an empty literal decides
    nothing, so ``"" + name`` and ``f"{name}.render"`` are both unproven.
    """
    if isinstance(node, ast.Name):
        return node.id == "__name__"
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and _reaches_the_check_layer(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _rooted_in_the_check_package(node.left)
    if isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                if not part.value:
                    continue
                return _reaches_the_check_layer(part.value)
            if isinstance(part, ast.FormattedValue):
                return _rooted_in_the_check_package(part.value)
            return False
    return False


def _computed_imports(root: Path, scope: str) -> list[tuple[str, int, str]]:
    """Return every unproven dynamic import under ``root`` inside ``scope``, sorted.

    Each entry is ``path``, ``lineno`` and the import hook the statement reaches, one per
    statement. A call taking a literal module name is judged by :func:`_literal_references`
    instead, and a name rooted in the check package is the layer's own modules, so neither is
    reported: what remains is a module name no reading of the code can prove.
    """
    chosen: dict[tuple[str, int], str] = {}
    for relative, tree in _sources(root):
        if not _within(relative, scope):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            hook = _hook_of(node.func)
            if not hook:
                continue
            argument = _module_argument(node)
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                continue
            if _rooted_in_the_check_package(argument):
                continue
            key = (relative, node.lineno)
            best = chosen.get(key)
            if best is None or hook < best:
                chosen[key] = hook
    return sorted((path, lineno, hook) for (path, lineno), hook in chosen.items())


def test_checks_layer_reaches_only_the_adapter_port() -> None:
    """``qrcheck/checks/**`` imports ``qrcheck.targets.base`` and nothing else of that layer."""
    violations = _violations(_scan(_PRODUCT_ROOT), scope=_CHECKS_SCOPE, reaches=_beyond_the_port)
    assert not violations, (
        "Check modules may import qrcheck.targets.base and nothing else from qrcheck.targets. "
        "Offenders, as path:lineno:module:\n"
        f"{_format(violations)}"
    )


def test_target_layer_never_reaches_the_check_layer() -> None:
    """``qrcheck/targets/**`` never imports ``qrcheck.checks``, in any spelling."""
    violations = _violations(
        _scan(_PRODUCT_ROOT), scope=_TARGETS_SCOPE, reaches=_reaches_the_check_layer
    )
    assert not violations, (
        "Adapter modules must never import the check layer. Offenders, as "
        "path:lineno:module:\n"
        f"{_format(violations)}"
    )


def test_product_never_reaches_the_test_suite() -> None:
    """Nothing under ``qrcheck/`` imports anything under ``tests``, in any spelling."""
    violations = _violations(_scan(_PRODUCT_ROOT), scope=None, reaches=_reaches_the_test_suite)
    assert not violations, (
        "The package must not import its own test suite. Offenders, as path:lineno:module:\n"
        f"{_format(violations)}"
    )


def test_scan_covers_the_required_modules() -> None:
    """The scan visits at least the declared module counts, so no rule above can pass vacuously."""
    assert _PRODUCT_ROOT.is_dir(), f"the package directory {_PRODUCT_ROOT} is missing"
    checks = _python_files(_PRODUCT_ROOT / "checks")
    targets = _python_files(_PRODUCT_ROOT / "targets")
    assert len(checks) >= _MIN_CHECK_MODULES, (
        f"qrcheck/checks/ holds {len(checks)} module(s); the counted floor is "
        f"{_MIN_CHECK_MODULES}"
    )
    assert len(targets) >= _MIN_TARGET_MODULES, (
        f"qrcheck/targets/ holds {len(targets)} module(s); the counted floor is "
        f"{_MIN_TARGET_MODULES}"
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


def test_the_check_layer_imports_no_module_by_a_computed_name() -> None:
    """No check module imports a module whose name it cannot show to be inside the check package.

    The extension boundary requires the scan to fail on a literal import of one of the five
    adapters, and :func:`_literal_references` holds that. A name assembled at run time defeats a
    literal rule,
    because the adapter's name never appears: ``importlib.import_module("qrcheck.targets." +
    name)``, ``__import__(module_name)`` and ``getattr(importlib, "import_module")(name)`` each
    reach an adapter while every literal assertion still passes. This rule constrains the form
    instead of the argument, which is the only way it survives an edit: every import hook under
    ``qrcheck/checks/**`` must take its module name from a literal or from a name provably inside
    ``qrcheck.checks``, which is all that layer loads -- its own nine family modules -- so
    reaching an adapter by a computed name now costs an edit to this gate as well.
    """
    violations = _computed_imports(_PRODUCT_ROOT, scope=_CHECKS_SCOPE)
    assert not violations, (
        "Check modules may import only a literal module name or one provably inside "
        "qrcheck.checks, so no computed name reaches an adapter. Offenders, as "
        "path:lineno:hook:\n"
        f"{_format(violations)}"
    )


def _control_tree(tmp_path: Path) -> Path:
    """Write a miniature package of synthetic offenders and return its directory."""
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
    """The collector reports an absolute, a relative and a package-only adapter reference."""
    package = _control_tree(tmp_path)
    violations = _violations(_scan(package), scope=_CHECKS_SCOPE, reaches=_beyond_the_port)
    assert violations == [
        ("qrcheck/checks/absolute.py", 1, "qrcheck.targets.cli"),
        ("qrcheck/checks/package.py", 1, "qrcheck.targets"),
        ("qrcheck/checks/relative.py", 1, "qrcheck.targets.reference"),
    ], f"unexpected offenders: {_format(violations)}"


def test_scanner_ignores_prose_and_the_port(tmp_path: Path) -> None:
    """A comment, a docstring and a port import are not dependencies and are never reported."""
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
    """A literal adapter name and both reversed directions are reported, with their lines."""
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


def test_scanner_flags_a_computed_module_name_and_permits_a_check_one(tmp_path: Path) -> None:
    """The three computed spellings are reported with their hook and line; a rooted one is not."""
    package = _control_tree(tmp_path)
    concatenated = '    return importlib.import_module("qrcheck.targets." + name)'
    by_variable = "    return __import__(module_name)"
    through_getattr = '    return getattr(importlib, "import_module")(module_name)'
    source = [
        "import importlib",
        "",
        "",
        "def concatenated(name):",
        concatenated,
        "",
        "",
        "def from_a_variable(module_name):",
        by_variable,
        "",
        "",
        "def through_getattr(module_name):",
        through_getattr,
        "",
        "",
        "def own_family(name):",
        '    return importlib.import_module(f"{__name__}.{name}")',
        "",
        "",
        "def own_module(name):",
        '    return importlib.import_module("qrcheck.checks." + name)',
        "",
        "",
        "def a_third_party_probe():",
        '    return importlib.import_module("cairosvg")',
        "",
    ]
    (package / "checks" / "computed.py").write_text("\n".join(source), encoding="utf-8")
    reported = _computed_imports(package, scope=_CHECKS_SCOPE)
    expected = [
        ("qrcheck/checks/computed.py", source.index(concatenated) + 1, "import_module"),
        ("qrcheck/checks/computed.py", source.index(by_variable) + 1, "__import__"),
        ("qrcheck/checks/computed.py", source.index(through_getattr) + 1, "import_module"),
    ]
    assert reported == expected, (
        "the scan reports every import hook whose module name it cannot prove is a check "
        f"module, and only those, as path:lineno:hook:\n{_format(reported)}"
    )


def test_this_module_does_not_import_the_product() -> None:
    """This module reads the product's source and never imports it, so it cannot fail first."""
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
