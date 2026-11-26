"""``qrcheck list``: enumerate every registered check, and nothing else.

The registry is the single source of check identity (AAP §0.3.4), so this command adds,
filters, renames and reorders nothing. It imports the check families so that they register
themselves, then prints exactly what :func:`qrcheck.registry.check.iter_checks` returns:
every registered check, in ascending identifier order. Two runs on the same installation
therefore describe the same checks in the same order, and a check that a run executes but
this listing omits -- or the reverse -- is a defect in the registry rather than a
difference between two surfaces (see ``tests/integration/test_registry_reachability.py``).

Registration is import-time, and ``qrcheck.registry`` never imports ``qrcheck.checks``
(that would close the registration cycle), so this module imports the family package and
calls :func:`qrcheck.checks.load_families` before it reads the registry. Without that call
the listing would be empty while every other surface reported twenty-one checks, which is
the one silent failure this file can have; the call is idempotent, so it costs nothing
when ``qrcheck.cli.main`` already imported the checks.

No configuration is read here, deliberately: no ``--config`` option, no
``qrcheck.config.load_config`` call and no configurable default. ``qrcheck list`` has to
answer "which checks does this installation have?" in a checkout with no ``qrcheck.toml``,
before a target is configured and before ``qrcheck doctor`` has anything to verify. A
configuration load would also let a malformed file describing a *target* hide checks that
exist, so it must never be added.

Two output forms, one set of rows
--------------------------------
``--json`` prints the object array that AAP §0.8.3 fixes as a public surface: one object
per check carrying exactly the five keys ``id``, ``title``, ``family``, ``requires`` and
``dependencies``, sorted by ``id``, with the keys themselves in sorted order because the
product's single JSON formatter writes them
(:func:`qrcheck.report.json_report.dumps_canonical`). Tooling and
``tests/integration/test_registry_reachability.py`` compare that array against the
``check_id`` values a run's ``summary.json`` records, so the key names are a contract: a
renamed key, a wrapping object or a nesting level would make a consumer mis-parse a
correct run. The table form prints the same rows in the same order, with the capability
and dependency lists comma-joined, an empty one rendered as ``-``, and the title last so
the short, machine-relevant columns line up. Its header names are the row keys, so the
two forms describe the same five fields with the same words.

Determinism
-----------
``CheckContract.requires`` is a ``frozenset`` of :class:`qrcheck.targets.base.Capability`
members, so its iteration order is not stable across processes; both list-valued fields
are sorted before they are emitted (AAP §0.5.3). Column widths come from the values being
printed, never from the terminal, so neither form wraps, truncates an identifier or
depends on the environment.

Exit code
---------
:data:`qrcheck.cli.exit.EXIT_OK` and nothing else. A broken installation is not a listed
condition: a missing family module raises :exc:`ImportError` and a contradictory
registration raises :class:`qrcheck.registry.check.CheckRegistrationError` out of
``load_families``, which the CLI group turns into exit code 3 rather than into a short,
plausible-looking table. Click's own usage errors are mapped to exit code 2 by
``qrcheck.cli.main``, so this module decides neither.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

import click

from qrcheck.checks import load_families
from qrcheck.cli.exit import EXIT_OK
from qrcheck.registry import CheckContract, iter_checks
from qrcheck.report.json_report import dumps_canonical

__all__ = ("list_checks", "list_command")

# The table's columns, in order: the first four are AAP §0.3.4's listing contract, and
# the title -- the one-line description the registry declares as "what ``qrcheck list``
# prints" -- comes last so the short columns align. The header row prints these names
# verbatim, so they double as the row keys of the --json form.
_TABLE_COLUMNS: Final[tuple[str, ...]] = ("id", "family", "requires", "dependencies", "title")

# What an empty capability or dependency list renders as in the table; the --json form
# carries the empty list itself, which is the machine-readable equivalent.
_EMPTY_CELL: Final[str] = "-"

# Separator between the members of one list-valued cell. A comma with no space keeps a
# long capability list inside one line, which is what stops the table from wrapping.
_CELL_SEPARATOR: Final[str] = ","

# Column separator: two spaces, so a padded column never runs into the next one and the
# layout does not depend on how wide a terminal renders a tab stop.
_COLUMN_GAP: Final[str] = "  "


class _Row(TypedDict):
    """One listing row: the five keys of the ``--json`` contract, already JSON-ready.

    The field names are the contract (AAP §0.8.3), so they are spelled here once and
    nowhere else: :func:`_row` builds every row through this type, which is what keeps the
    two output forms and the machine consumer naming the same five fields.
    """

    id: str
    title: str
    family: str
    requires: list[str]
    dependencies: list[str]


def list_checks(as_json: bool) -> int:
    """Print every registered check and return the process exit code.

    Args:
        as_json: ``True`` for the object array that machine consumers read, ``False`` for
            the human table. Both describe the same rows in the same order.

    Returns:
        :data:`qrcheck.cli.exit.EXIT_OK`, always. A listing has no failing condition of its
        own: it reports what the registry holds, and a defect in the registry raises out of
        :func:`qrcheck.checks.load_families` before anything is printed.

    Raises:
        ImportError: A check family module is missing or fails to import. Deliberately not
            caught: an installation that cannot load its checks must fail the process, not
            print a shorter listing that a reader would take for the whole inventory.
        CheckRegistrationError: A check module registers an identifier twice, or registers
            one that contradicts its declaration. Also deliberately uncaught, because
            ``qrcheck list`` and a run must never disagree about which checks exist.
    """
    # Registration is import-time and the registry package never imports this one, so the
    # families are loaded here before the registry is read; the call is idempotent.
    load_families()
    rows = [_row(contract) for contract in iter_checks()]
    if as_json:
        # The product's only JSON formatter: sorted keys, two-space indent, UTF-8 literal
        # text, and exactly one trailing newline once click.echo has written the line.
        click.echo(dumps_canonical(rows))
    else:
        for line in _table_lines(rows):
            click.echo(line)
    return EXIT_OK


def _row(contract: CheckContract) -> _Row:
    """Return the listing row for ``contract``.

    Both list-valued fields are sorted before they leave this function: ``requires`` is a
    ``frozenset`` whose iteration order differs between processes, and ``dependencies`` is
    a tuple the registry already normalised but whose order is not a contract a listing may
    rely on. Sorting here rather than at emission is what makes both output forms agree
    member for member (AAP §0.5.3).

    Capability members are a :class:`~enum.StrEnum`, so ``str()`` yields the member's
    value -- ``"png"``, ``"ec_level"`` -- which is the spelling the configuration, the
    resolution table and the reports all use.
    """
    return _Row(
        id=contract.id,
        title=contract.title,
        family=contract.family,
        requires=sorted(str(capability) for capability in contract.requires),
        dependencies=sorted(contract.dependencies),
    )


def _table_lines(rows: Sequence[_Row]) -> list[str]:
    """Render the header row and one line per check, aligned on data-derived widths.

    Widths come from the values being printed, so two runs of the same installation produce
    the same bytes and no width depends on a terminal or an environment variable. Only the
    four leading columns are padded; the last one is printed as it stands, so no line ends
    in padding whitespace.
    """
    body = [_cells(row) for row in rows]
    widths = [
        max(len(_TABLE_COLUMNS[index]), *(len(cells[index]) for cells in body))
        for index in range(len(_TABLE_COLUMNS) - 1)
    ]
    lines = [_padded(_TABLE_COLUMNS, widths)]
    lines.extend(_padded(cells, widths) for cells in body)
    return lines


def _cells(row: _Row) -> tuple[str, ...]:
    """Return one row's table cells in :data:`_TABLE_COLUMNS` order."""
    return (
        row["id"],
        row["family"],
        _CELL_SEPARATOR.join(row["requires"]) or _EMPTY_CELL,
        _CELL_SEPARATOR.join(row["dependencies"]) or _EMPTY_CELL,
        row["title"],
    )


def _padded(cells: Sequence[str], widths: Sequence[int]) -> str:
    """Join ``cells`` with padding to ``widths`` for every column but the last."""
    padded = [
        f"{cell:<{widths[index]}}" if index < len(widths) else cell
        for index, cell in enumerate(cells)
    ]
    return _COLUMN_GAP.join(padded)


# ``EXIT_OK`` is 0, which is click's own success status, so this callback needs no
# exit-code plumbing of its own: unlike ``qrcheck doctor``, whose verdict can be
# ``EXIT_CONFIG``, a listing either succeeds or raises before it prints. The callback
# still returns the integer, because a caller and a test read it directly.
@click.command(name="list")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the machine-readable object array instead of the table.",
)
def list_command(as_json: bool) -> int:
    """Enumerate every registered check with its family, capabilities and dependencies."""
    return list_checks(as_json)
