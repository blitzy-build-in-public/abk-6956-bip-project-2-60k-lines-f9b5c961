"""The command-line surface of ``qrcheck``, whose only public name is :func:`main`.

Both entry paths -- the ``qrcheck`` console script ``qrcheck.cli.main:main`` and ``python -m
qrcheck`` -- reach :func:`qrcheck.cli.main.main` directly, so this module re-exports that callable
and defines nothing of its own. It names the module rather than an attribute because this package is
initialised first, so a sibling written as ``from qrcheck.cli import main`` would close the cycle
the command modules leave open.
"""

from qrcheck.cli.main import main

__all__ = ("main",)
