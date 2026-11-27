"""The one availability probe, and the property that makes it usable in a broken environment.

:mod:`qrcheck.dependencies` answers the question every blocking-dependency gate asks -- *does the
module the product imports actually load?* -- because installed metadata does not answer it. The
table, the probe and its diagnostic (naming the distribution, the module, the interpreter, the
platform and ``requirements.lock``) are held here. An absent pair is probed by name, and a
present-but-unimportable one by blocking its module in :data:`sys.modules`. No test skips, retries
or relaxes an assertion.
"""

from __future__ import annotations

import platform
import sys
from typing import Final

import pytest

from qrcheck.dependencies import (
    CORE_DEPENDENCIES,
    LOCK_FILE,
    blocking_problem,
    import_problem,
    probe,
    problems,
)

#: A distribution and module name that cannot exist, so a probe reports a problem on any host.
ABSENT_DISTRIBUTION: Final[str] = "qrcheck-nonexistent-dist"
ABSENT_MODULE: Final[str] = "no_such_module_xyz"

#: The core dependency whose module is blocked in ``sys.modules``: installed but unimportable.
BLOCKED: Final[str] = "httpx"


def test_the_table_names_the_module_the_product_imports() -> None:
    """The core table maps each pinned distribution to its importable module, in order.

    Breaks if a pair drifts from what the product imports -- ``Pillow``'s module is ``PIL.Image`` --
    or if a pinned library is dropped, which would stop a run refusing a broken install of it.
    """
    assert CORE_DEPENDENCIES == (
        ("segno", "segno"),
        ("numpy", "numpy"),
        ("Pillow", "PIL.Image"),
        ("httpx", "httpx"),
        ("click", "click"),
    ), f"the core dependency table has changed: {CORE_DEPENDENCIES!r}"
    for distribution, module in CORE_DEPENDENCIES:
        found = probe(distribution, module)
        assert found.usable, f"{distribution!r} is reported unusable here: {found.message}"
        assert found.version is not None and found.version in found.message, (
            f"{distribution!r} does not report its installed version: {found.message}"
        )
        assert LOCK_FILE not in found.message, f"a usable pair reports a failure: {found.message}"
    assert problems() == () and blocking_problem() is None, (
        f"the healthy host reports problems: {[item.distribution for item in problems()]}"
    )


def test_an_absent_pair_reports_a_problem_naming_what_to_fix() -> None:
    """A distribution that is not installed yields the full blocking diagnostic.

    Breaks if the message stops naming the distribution to install, the module the product imports,
    the interpreter running it or ``requirements.lock``.
    """
    found = probe(ABSENT_DISTRIBUTION, ABSENT_MODULE)

    assert not found.usable and found.version is None, (
        f"an absent pair is reported usable, or with a version: {found!r}"
    )
    for expected in (ABSENT_DISTRIBUTION, ABSENT_MODULE, platform.python_version(), LOCK_FILE):
        assert expected in found.message, f"the diagnostic omits {expected!r}: {found.message}"


def test_a_present_but_unimportable_module_is_a_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module that cannot be imported is unusable even though metadata still answers.

    The state the metadata check cannot see: ``sys.modules[name] = None`` makes ``import name``
    raise :exc:`ImportError` while the installed version is still reported. Breaks if the probe goes
    back to trusting the distribution's presence.
    """
    monkeypatch.setitem(sys.modules, BLOCKED, None)

    found = probe(BLOCKED, BLOCKED)

    assert not found.usable and found.version is not None, (
        f"{BLOCKED!r} is reported usable, or without its installed version: {found!r}"
    )
    assert BLOCKED in (found.problem or ""), (
        f"the problem does not name the module that failed to import: {found.problem!r}"
    )
    blocking = blocking_problem()
    assert blocking is not None and blocking.distribution == BLOCKED, (
        f"the blocked dependency is not the one a run would refuse on: {blocking!r}"
    )


def test_import_problem_is_none_only_when_the_module_loads() -> None:
    """The guarded import every caller shares reports a loading module and a missing one."""
    assert import_problem("json") is None, "importing the standard library's json module failed"
    absent = import_problem(ABSENT_MODULE)
    assert absent is not None and ABSENT_MODULE in absent, (
        f"the guarded import does not report {ABSENT_MODULE!r}: {absent!r}"
    )
