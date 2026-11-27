"""The run-end observation: ``CL-ARTIFACTS`` runs after the walk, so its window is the run.

``CL-ARTIFACTS``'s temporary-file observation is a snapshot of :func:`tempfile.gettempdir`
"before and after the run", the two snapshots asserted equal, because a check that cannot fail is
a defect. Before its contract declared ``observes_run_end`` the baseline was taken when the check
enumerated its cases and compared when it executed -- and because checks ran in ascending
identifier order, ``CL-ARTIFACTS`` executed first, so the window held no other case at all and no
entry a later case left behind could fail it.

Three properties are pinned here, through the real command line:

* an entry that appears in the system temporary directory **during** the walk fails
  ``CL-ARTIFACTS`` with reason ``value_mismatch`` and names the entry in ``decoded.leaked`` -- the
  observation the enumeration-time window could never make;
* a clean run of the same selection passes it, so the failure above is attributable to the entry
  and not to the check;
* ``decoded`` reports both windows (``temp_entries_before``, ``temp_entries_after``) and records
  which descriptor interface was enumerated as a platform token, never as an absolute path.

The entry is left by a case of another selected check, which is the only seam that puts it inside
the run's own window, and every run here is given a temporary directory of its own: a sibling clone
leaving a ``qrcheck-*`` entry in the host's shared temporary directory would otherwise decide these
results. The entry is removed after each test.

The same file holds the other half of the runner's containment: ``_persist`` wraps every result this
layer builds, and ``EvidenceStore.record`` documents ``ValueError``, ``TypeError`` and ``OSError``
as its refusal modes -- a value it will not serialise, and a write that fails. The write fault is
proven end to end through ``run`` in ``tests/unit/test_exit_codes.py``; the refusals a
target-influenced value reaches are pinned here against the function itself, because each has to
become one ``error`` result in the case's own place instead of an exception that aborts the walk .
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

import qrcheck.runner as runner_module
from qrcheck.checks import cleanup as cleanup_module
from qrcheck.checks import load_families, support
from qrcheck.cli.exit import EXIT_FAILED, EXIT_OK
from qrcheck.registry.check import REGISTRY, Case, CheckContract, RunContext
from qrcheck.report import SUMMARY_FILENAME
from qrcheck.report.model import TRACEBACK_KEY, CaseResult, Reason, Status

#: The registry module itself, reached through :func:`importlib.import_module` because the
#: ``qrcheck.registry`` package binds ``check`` to the registration decorator, so the dotted import
#: form names a function rather than this module.
_REGISTRY_MODULE: Final[Any] = importlib.import_module("qrcheck.registry.check")

#: The check under test: the one whose contract declares ``observes_run_end``, so the runner
#: executes it after every ordinary check.
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
#: enumerated; ``None`` is the third case, for a host that exposes neither, since no artefact
#: carries the absolute path itself.
FD_TOKENS: Final[frozenset[str]] = frozenset({"proc-self-fd", "dev-fd"})

#: The two fields ``decoded`` uses to report the temporary-file windows of the run.
WINDOW_FIELDS: Final[tuple[str, ...]] = ("temp_entries_before", "temp_entries_after")

#: The seed the refusal tests record on their case: the committed default, so nothing here depends
#: on a clock or a random source.
SEED: Final[int] = 20240101


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
    The replacement goes into the module-private store, the way ``tests/unit/test_exit_codes.py``
    replaces a generator: the registry holds the contract built at import, so patching the module
    attribute would change nothing the walk reads. Returns the registered replacement.
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

    One result per case and one case for this check, so a second one would be an accumulation
    defect; the diagnostic prints every identifier the run reported.
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
        f"no absolute path may reach an artefact; decoded.fd_directory is {token!r}"
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
    it, so a check built on that window could not fail.
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
    handle half enumerated, without the artefact naming a single host location.
    """
    assert tuple(private_temp.iterdir()) == (), "the run's temporary directory starts empty"

    result = invoke_cli("run", "--checks", WITNESS_CHECK)

    assert result.exit_code == EXIT_OK, result.stderr
    entry = witness_result(report_dir)
    assert_windows_and_token(entry)
    assert entry["decoded"]["temp_prefix"] == cleanup_module.TEMP_PREFIX, (
        "the artefact names the prefix the observation filtered on"
    )


# --------------------------------------------------------------------------- #
# The evidence store's full refusal set, contained rather than escaping the walk
# --------------------------------------------------------------------------- #


class RefusingStore:
    """A stand-in evidence store whose ``record`` raises the refusal it was built with.

    The duck-typed port, not the real store: what is under test is the containment in the runner,
    so the refusal is chosen by the test and no filesystem is involved.
    """

    def __init__(self, refusal: BaseException) -> None:
        """Hold the refusal every call raises, and count the calls."""
        self.refusal = refusal
        self.calls = 0

    def record(self, result: CaseResult) -> str | None:
        """Refuse `result` the way the real store would, after recording that it was asked."""
        del result
        self.calls += 1
        raise self.refusal


@pytest.mark.parametrize(
    "refusal",
    (
        ValueError("artefact request.json: payload holds a non-finite float (nan, inf or -inf)"),
        TypeError("artefact decoded.json: Object of type bytes is not JSON serializable"),
        UnicodeEncodeError("utf-8", "x\ud800y", 1, 2, "surrogates not allowed"),
    ),
    ids=("value-error", "type-error", "unicode-encode-error"),
)
def test_a_store_refusal_that_is_not_a_write_fault_is_contained(
    refusal: BaseException,
) -> None:
    """A refusal the store documents becomes one ``error`` result, never an escaping exception.

    ``EvidenceStore.record`` documents ``ValueError``, ``TypeError`` and ``OSError`` as its refusal
    modes, and a value a target influenced is enough to reach the first two. Containing only
    ``OSError`` let one of them escape from inside the handler that builds
    the ``error`` result, which aborted the walk: no report, no evidence, exit 3, and every result
    already accumulated discarded -- the opposite of what containment requires. The
    ``UnicodeEncodeError`` case pins the subclass: a refusal of that shape is contained as the
    ``ValueError`` it is and needs no arm of its own.
    """
    store = RefusingStore(refusal)
    case = CaseResult(
        check_id="RT-ROUNDTRIP",
        case_id="corpus:url-000",
        status=Status.FAILED,
        reason=Reason.VALUE_MISMATCH,
        duration_ms=1.5,
        decoded={"payload_len": 31},
        expected={"payload_len": 32},
        seed=SEED,
    )

    result = runner_module._persist(store, case)

    # Once, not twice: containing a refusal is not a retry.
    assert store.calls == 1
    # The case keeps the one result it is entitled to, under its own identity.
    assert (result.check_id, result.case_id) == (case.check_id, case.case_id)
    assert result.duration_ms == case.duration_ms
    assert result.seed == case.seed
    # The outcome the case had reached is replaced by the internal-error one, not duplicated.
    assert result.status is Status.ERROR
    assert result.reason is Reason.INTERNAL_ERROR
    assert result.evidence_dir is None, "nothing was written, so no directory may be named"
    assert result.expected == {"outcome": "evidence_recorded"}
    # The fault and the outcome it displaced are both in the artefact, so a reader of
    # ``summary.json`` can tell a lost failing case from a lost skipped one.
    assert result.decoded["error_type"] == type(refusal).__name__
    assert str(refusal) in result.decoded[TRACEBACK_KEY]
    assert result.decoded["prevented_status"] == Status.FAILED.value
    assert result.decoded["prevented_reason"] == Reason.VALUE_MISMATCH.value
