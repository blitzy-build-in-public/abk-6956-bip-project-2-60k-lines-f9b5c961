"""The run-end observation: ``CL-ARTIFACTS`` runs after the walk, so its window is the run.

AAP §0.8.1 defines the check's temporary-file observation as a snapshot of
:func:`tempfile.gettempdir` "before and after the run", the two snapshots asserted equal, and
AAP §0.4.4 treats a check that cannot fail as a defect. Before its contract declared
``observes_run_end`` the baseline was taken when the check enumerated its cases and compared when
it executed -- and because checks ran in ascending identifier order, ``CL-ARTIFACTS`` executed
first, so the window held no other case at all and no entry a later case left behind could fail it.

Three properties are pinned here, through the real command line:

* an entry that appears in the system temporary directory **during** the walk fails
  ``CL-ARTIFACTS`` with reason ``value_mismatch`` and names the entry in ``decoded.leaked`` -- the
  observation the enumeration-time window could never make;
* a clean run of the same selection passes it, so the failure above is attributable to the entry
  and not to the check;
* ``decoded`` reports both windows (``temp_entries_before``, ``temp_entries_after``) and records
  which descriptor interface was enumerated as a platform token, never as an absolute path
  (AAP §0.5.5).

The entry is left by a case of another selected check, which is the only seam that puts it inside
the run's own window, and every run here is given a temporary directory of its own: a sibling clone
leaving a ``qrcheck-*`` entry in the host's shared temporary directory would otherwise decide these
results. The entry is removed after each test.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.checks import cleanup as cleanup_module
from qrcheck.checks import load_families, support
from qrcheck.cli.exit import EXIT_FAILED, EXIT_OK
from qrcheck.registry.check import REGISTRY, Case, CheckContract, RunContext
from qrcheck.report import SUMMARY_FILENAME
from qrcheck.report.model import CaseResult, Reason, Status

#: The registry module itself, reached through :func:`importlib.import_module` because the
#: ``qrcheck.registry`` package binds ``check`` to the registration decorator, so the dotted import
#: form names a function rather than this module.
_REGISTRY_MODULE: Final[Any] = importlib.import_module("qrcheck.registry.check")

#: The check under test: the one whose contract declares ``observes_run_end`` (AAP §0.8.1), so the
#: runner executes it after every ordinary check.
WITNESS_CHECK: Final[str] = "CL-ARTIFACTS"

#: The ordinary check whose case generator the leaking seam replaces. It requires only ``png``,
#: contributes one synthetic case and reads no corpus record, so the seam adds no adapter work and
#: no corpus dependency to the run it is part of.
SEAM_CHECK: Final[str] = "IV-EMPTY"

#: The case identifier the seam yields, distinct from every identifier the real check builds: a
#: result filed under a real case's identifier would describe a case that never ran.
SEAM_CASE_ID: Final[str] = "synthetic:run-end-seam"

#: The selection the two mirrored runs share: one ordinary check to carry the seam, then the
#: witness. Ordering in the selector is irrelevant; the walk decides the order.
SELECTION: Final[str] = f"{WITNESS_CHECK},{SEAM_CHECK}"

#: The name of the entry the seam leaves behind. It carries the product-wide ``qrcheck-`` prefix
#: the witness scans for, which is what makes it a leak rather than stray debris.
LEAK_NAME: Final[str] = f"{cleanup_module.TEMP_PREFIX}run-end-seam"

#: The bytes the entry holds: nothing reads them, so any content would do, and a fixed value keeps
#: the seam free of a clock or a random source.
LEAK_BYTES: Final[bytes] = b"qrcheck run-end seam"

#: The tokens ``decoded.fd_directory`` may carry for the descriptor directory the handle half
#: enumerated; ``None`` is the third case, for a host that exposes neither (AAP §0.5.5 forbids the
#: absolute path itself).
FD_TOKENS: Final[frozenset[str]] = frozenset({"proc-self-fd", "dev-fd"})

#: The two fields ``decoded`` uses to report the temporary-file windows of the run.
WINDOW_FIELDS: Final[tuple[str, ...]] = ("temp_entries_before", "temp_entries_after")


# --------------------------------------------------------------------------- #
# A temporary directory of the run's own, and the entry the seam leaves in it
# --------------------------------------------------------------------------- #


@pytest.fixture
def private_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point :func:`tempfile.gettempdir` at a directory only this test uses.

    ``tempfile.gettempdir`` answers with the module-level cache once it is set, so patching
    ``tempfile.tempdir`` -- rather than the environment alone -- is what makes every later call in
    this process answer with this directory. Neither a sibling clone's leftovers nor the harness's
    own scratch files can then decide these results, and the leak this module creates cannot reach
    the host's shared temporary directory.
    """
    private = tmp_path / "run-end-temp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    return private


@pytest.fixture
def leak_entry(private_temp: Path) -> Iterator[Path]:
    """Return the entry's path, and remove it afterwards whatever the run did to it."""
    entry = private_temp / LEAK_NAME
    try:
        yield entry
    finally:
        with contextlib.suppress(FileNotFoundError):
            entry.unlink()


def install_leaking_seam(monkeypatch: pytest.MonkeyPatch, entry: Path) -> CheckContract:
    """Replace ``SEAM_CHECK``'s cases with one that leaves `entry` behind and passes.

    The seam is a case of a selected ordinary check, so it runs inside the walk and before the
    witness -- exactly the position from which an entry can enter the run's own observation window.
    The replacement goes into the module-private store, the way
    ``tests/unit/test_exit_codes.py`` replaces a generator: the registry holds the contract built
    at import, so patching the module attribute would change nothing the walk reads. Returns the
    registered replacement.
    """

    def cases(context: RunContext) -> Iterator[Case]:
        """Yield the one case that leaves the entry behind, ignoring the run context."""
        del context

        def execute(run_context: RunContext) -> CaseResult:
            """Leave the entry in the temporary directory, then report a pass."""
            entry.write_bytes(LEAK_BYTES)
            return support.pass_(SEAM_CHECK, SEAM_CASE_ID, seed=run_context.seed)

        return iter([Case(case_id=SEAM_CASE_ID, execute=execute)])

    load_families()
    replacement = dataclasses.replace(REGISTRY[SEAM_CHECK], cases=cases)
    monkeypatch.setitem(_REGISTRY_MODULE._REGISTRY, SEAM_CHECK, replacement)
    return replacement


# --------------------------------------------------------------------------- #
# Reading what the run reported
# --------------------------------------------------------------------------- #


def read_summary(report_root: Path) -> Mapping[str, Any]:
    """Return ``summary.json`` of the run under `report_root`, parsed."""
    return json.loads((report_root / SUMMARY_FILENAME).read_text(encoding="utf-8"))


def witness_result(report_root: Path) -> Mapping[str, Any]:
    """Return the run's single ``CL-ARTIFACTS`` result.

    One result per case and one case for this check, so a second one would be the accumulation
    defect AAP §0.3.4 forbids; the diagnostic prints every identifier the run reported.
    """
    summary = read_summary(report_root)
    entries = [entry for entry in summary["results"] if entry["check_id"] == WITNESS_CHECK]
    assert len(entries) == 1, (
        "the run-end check contributes exactly one result; the run holds "
        f"{[(entry['check_id'], entry['case_id']) for entry in summary['results']]}"
    )
    return entries[0]


def assert_windows_and_token(entry: Mapping[str, Any]) -> None:
    """Assert the artefact reports both temporary-file windows and no absolute descriptor path."""
    decoded = entry["decoded"]
    for field in WINDOW_FIELDS:
        assert field in decoded, (
            f"decoded must report {field}: the artefact has to state what the check observed"
        )
        assert isinstance(decoded[field], list), f"decoded.{field} must be a list of names"
    assert "fd_directory" in decoded, "decoded must record which descriptor interface was read"
    token = decoded["fd_directory"]
    assert token is None or token in FD_TOKENS, (
        f"decoded.fd_directory must be one of the platform tokens {sorted(FD_TOKENS)} or null, "
        f"not {token!r}"
    )
    assert not (isinstance(token, str) and token.startswith("/")), (
        f"no absolute path may reach an artefact (AAP §0.5.5); decoded.fd_directory is {token!r}"
    )


# --------------------------------------------------------------------------- #
# The three properties
# --------------------------------------------------------------------------- #


def test_an_entry_created_during_the_walk_fails_the_run_end_check(
    invoke_cli: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    report_dir: Path,
    leak_entry: Path,
) -> None:
    """An entry left by a case of another check is a leak of the run, and the run's end sees it.

    The witness executes after the walk, so the entry the seam leaves part-way through it is inside
    the observation window: the snapshot taken at the check's own enumeration could never contain
    it, which is exactly the vacuity AAP §0.4.4 calls a defect.
    """
    install_leaking_seam(monkeypatch, leak_entry)

    result = invoke_cli("run", "--checks", SELECTION)

    assert result.exit_code == EXIT_FAILED, result.stderr
    entry = witness_result(report_dir)
    assert entry["status"] == Status.FAILED.value, (
        f"the run's footprint is not clean, so the check must fail; decoded is {entry['decoded']}"
    )
    assert entry["reason"] == Reason.VALUE_MISMATCH.value
    assert entry["decoded"]["leaked"] == [LEAK_NAME], (
        "the leaked list must name the entry the run left behind"
    )
    assert "leaked_temp_entries" in entry["decoded"]["violations"]
    assert entry["decoded"]["temp_entries_before"] == [], (
        "the baseline is the state the run began in, which held no entry"
    )
    assert entry["decoded"]["temp_entries_after"] == [LEAK_NAME], (
        "the second window is the state at the run's end, which holds the seam's entry"
    )
    assert_windows_and_token(entry)


def test_a_clean_run_of_the_same_selection_passes_the_run_end_check(
    invoke_cli: Callable[..., Any],
    report_dir: Path,
    private_temp: Path,
) -> None:
    """The same selection with nothing left behind passes: the failure above is the entry's.

    ``private_temp`` is requested rather than used: it is the fixture that gives the run its own
    temporary directory, and it must exist for the clean run as much as for the leaking one, or a
    sibling's leftovers would fail this control.
    """
    assert tuple(private_temp.iterdir()) == (), "the run's temporary directory starts empty"

    result = invoke_cli("run", "--checks", SELECTION)

    assert result.exit_code == EXIT_OK, result.stderr
    entry = witness_result(report_dir)
    assert entry["status"] == Status.PASSED.value, (
        f"a clean run leaves nothing to report; decoded is {entry['decoded']}"
    )
    assert entry["reason"] is None
    assert entry["decoded"]["leaked"] == []
    assert entry["decoded"]["violations"] == []
    assert entry["decoded"]["temp_entries_before"] == []
    assert entry["decoded"]["temp_entries_after"] == []
    assert_windows_and_token(entry)


def test_the_artefact_states_both_windows_and_the_platform_token(
    invoke_cli: Callable[..., Any],
    report_dir: Path,
    private_temp: Path,
) -> None:
    """What was observed is in the artefact: both windows, and the descriptor interface by token.

    A reader has to be able to tell which two states were compared and which platform interface the
    handle half enumerated, without the artefact naming a single host location (AAP §0.5.5).
    """
    assert tuple(private_temp.iterdir()) == (), "the run's temporary directory starts empty"

    result = invoke_cli("run", "--checks", WITNESS_CHECK)

    assert result.exit_code == EXIT_OK, result.stderr
    entry = witness_result(report_dir)
    assert_windows_and_token(entry)
    assert entry["decoded"]["temp_prefix"] == cleanup_module.TEMP_PREFIX, (
        "the artefact names the prefix the observation filtered on"
    )
