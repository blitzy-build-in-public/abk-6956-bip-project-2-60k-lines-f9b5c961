"""The :mod:`qrcheck` command-line package, which exports nothing and imports no submodule.

The callable both entry paths reach is :func:`qrcheck.cli.main.main`: the console script
``qrcheck = "qrcheck.cli.main:main"`` that ``pyproject.toml`` declares, and the
``python -m qrcheck`` path :mod:`qrcheck.__main__` dispatches.

This package deliberately exports nothing and imports no submodule. A re-export under the name
``main`` would rebind the package attribute that has to stay the submodule, so the dotted form
``import qrcheck.cli.main as m`` would hand back the re-exported object and ``m.main`` would no
longer be that callable. An eager import here would also make ``python -m qrcheck.cli.main``
start with the :mod:`runpy` warning about a module already in ``sys.modules``.

A consumer therefore imports the module it needs -- ``from qrcheck.cli.main import main``, the form
the console script, :mod:`qrcheck.__main__` and every in-tree caller already use. A package-level
``from qrcheck.cli import main`` names that submodule as well: the import system falls back to it
when the package itself carries no attribute of that name.

Nothing about the command line is registered, parsed or decided at package import time: no command
table, no option default, no interpreter guard and no exit code. Importing this package costs one
docstring and one empty ``__all__``, and :exc:`AttributeError` for a name it does not define is the
only thing it ever raises.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
