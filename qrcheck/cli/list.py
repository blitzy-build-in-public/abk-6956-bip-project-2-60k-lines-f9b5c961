"""``qrcheck list``: enumerate every registered check, and nothing else.

The registry is the single source of check identity (AAP §0.3.4), so this command adds, filters,
renames and reorders nothing: it imports the check families so they register themselves, then prints
exactly what :func:`qrcheck.registry.check.iter_checks` returns, in ascending identifier order.

``--json`` prints the object array AAP §0.8.3 fixes as a public surface: one object per check
carrying exactly the five keys ``id``, ``title``, ``family``, ``requires`` and ``dependencies``,
keys sorted, because tooling and ``tests/integration/test_registry_reachability.py`` compare it
against the ``check_id`` values a run's ``summary.json`` records. The table form prints the same
rows in the same order, capability and dependency lists comma-joined with an empty one as ``-``, on
widths derived from the values rather than the terminal.

No configuration is read here, deliberately: a malformed file describing a target must never hide
checks that exist. The exit code is ``EXIT_OK`` and nothing else; a missing family module raises
``ImportError`` and a contradictory registration ``CheckRegistrationError`` out of
``load_families``, which the CLI group turns into exit code 3.
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

    The field names are the contract (AAP §0.8.3), so they are spelled here once and nowhere else.
    """

    id: str
    title: str
    family: str
    requires: list[str]
    dependencies: list[str]


def list_checks(as_json: bool) -> int:
    """Print every registered check and return the process exit code.

    Both output forms describe the same rows in the same order, and the return value is ``EXIT_OK``
    always: a listing has no failing condition of its own.

    Raises:
        ImportError: A check family module is missing or fails to import, deliberately uncaught:
            an installation that cannot load its checks must fail the process rather than print a
            shorter listing.
        CheckRegistrationError: A check module registers an identifier twice or against its own
            declaration, also uncaught, because this listing and a run must never disagree about
            which checks exist.
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

    Both list-valued fields are sorted here rather than at emission, so both output forms agree
    member for member (AAP §0.5.3); capability members are a ``StrEnum``, so ``str()`` yields the
    value.
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

    Two runs of the same installation therefore produce the same bytes. Only the four leading
    columns are padded, so no line ends in padding whitespace.
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
