"""The click plumbing every ``qrcheck`` command module repeats, kept in one place.

Two pieces of the command line are the same for every verb, so each is defined here once instead
of being restated by ``run``, ``doctor``, ``corpus build`` and ``loc``:

* :func:`resolve_config_path` -- how a subcommand finds the value of the group-level ``--config``
  option that :mod:`qrcheck.cli.main` owns. The group may leave the raw string in the click
  context object, keep a mapping or a namespace there, or hold the value only in its own
  parameters, and all three are read in one order here, so registering a command under the group
  needs no adaptation of either module.
* :class:`ExitCodeCommand` -- the click command class that turns a callback's integer return value
  into the process exit status, which click discards from a command that runs standalone.

Nothing else belongs here. A helper whose message, exit code or returned shape differs between two
commands stays in the command that decides it, so this module cannot become the place where two
behaviours are quietly made to agree. Its imports are the standard library and click only, which
is what lets every command module depend on it without depending on a sibling command.

The dependency direction is inward and one-way: :mod:`qrcheck.cli.main` imports the command
modules, so no command module imports it back. This module is where the option plumbing both sides
need lives without closing that cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import click

__all__ = ("DEFAULT_CONFIG_PATH", "ExitCodeCommand", "resolve_config_path")

# The configuration file a command falls back to when nothing supplies the option: the committed
# ``qrcheck.toml`` at the repository root, which is also the group option's own default.
DEFAULT_CONFIG_PATH: Final[str] = "qrcheck.toml"

# The names a group's context object, or the group's own click parameters, may carry the
# --config value under; tried in this order.
_CONFIG_KEYS: Final[tuple[str, ...]] = ("config", "config_path", "config_file")


def resolve_config_path(ctx: click.Context | None) -> str:
    """Return the configuration file the group's ``--config`` option selected.

    ``qrcheck.cli.main`` owns that option, and how it hands the value down is its contract rather
    than any command's: the group may leave the raw string in the context object, keep a mapping or
    a namespace there, or hold the value only in its own click parameters. All three are read, in
    that order. When none of them carries a value -- including a call with no click context at all,
    as a test that calls a command function directly makes -- the default configuration file name
    stands.

    Args:
        ctx: The click context the command was invoked with, or ``None`` when a caller invoked the
            command function itself instead of going through the group.

    Returns:
        The configuration file path, never empty: :data:`DEFAULT_CONFIG_PATH` when nothing in the
        context supplies one.
    """
    if ctx is not None:
        found = _config_value(ctx.obj)
        if found is not None:
            return found
        root_params = ctx.find_root().params
        for key in _CONFIG_KEYS:
            found = _config_value(root_params.get(key))
            if found is not None:
                return found
    return DEFAULT_CONFIG_PATH


def _config_value(value: object) -> str | None:
    """Return the configuration path ``value`` holds, or ``None`` when it holds none."""
    if isinstance(value, str | Path):
        text = str(value)
        return text or None
    if value is None:
        return None
    for key in _CONFIG_KEYS:
        entry = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
        if isinstance(entry, str | Path) and str(entry):
            return str(entry)
    return None


class ExitCodeCommand(click.Command):
    """A command whose integer return value becomes the process exit code.

    Click 8.1 discards a callback's return value when the command runs standalone, so a callback
    that merely returned :data:`qrcheck.cli.exit.EXIT_CONFIG` would leave the process status at 0
    however the verb went, and every gate built on that command -- the empty-selection gate above
    all -- would pass silently. The callback still returns the integer, so a caller and a test read
    it directly; this class is where that integer becomes the process status, and in
    non-standalone mode click turns it back into the very same return value.
    """

    def invoke(self, ctx: click.Context) -> object:
        """Run the callback and, when it returned an exit code, exit with it."""
        result = super().invoke(ctx)
        if isinstance(result, int):
            ctx.exit(result)
        return result
