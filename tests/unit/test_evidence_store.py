"""The evidence store's path containment: a planted symlink is refused, never followed.

QASEC-02 (CWE-59) recorded :meth:`qrcheck.report.evidence.EvidenceStore.record` composing
``<report_dir>/evidence/<check-id>/<failure-id>/`` and then writing through it, so a symlink
planted at ``<report_dir>/evidence``, at the sanitized check-id directory, at the failure-id
directory or at one of the artefact names carried ``request.json`` and ``decoded.json`` outside the
report tree -- or overwrote whatever file an artefact link named. This module holds the refusal
that closes it: an :exc:`OSError` naming the path, raised before anything is created, leaving the
link's target empty and a linked destination's file byte for byte unchanged.

It holds the guarantees the refusal must not cost, too: the two checks are independent, so a link
the component scan does not see is still refused by the real-path comparison; a relative report
root and a report root that is itself a symlink (the operator's own choice) both still record; a
normal record writes exactly the closed artefact set under ``evidence/<CHECK-ID>/<failure-id>`` and
returns that relative posix path; and a passed result still returns ``None`` having written nothing
(AAP §0.3.4, §0.6.6 row 11).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qrcheck.evidence.naming import DECODED_JSON, EVIDENCE_DIR_NAME, REQUEST_JSON, SYMBOL_PNG
from qrcheck.report.evidence import EvidenceStore
from qrcheck.report.model import CaseResult, Reason, Status

# The identity of the case the finding was filed against and the failure ID the naming module
# derives from it, spelled out because it is the directory name the store composes.
_CHECK_ID = "RT-ROUNDTRIP"
_CASE_ID = "corpus:url-000"
_FAILURE_ID = "RT-ROUNDTRIP__corpus_url-000__value_mismatch"
_SEED = 20240101

# The request projection, the decoded evidence and the artefact bytes a case carries. The store
# writes all three verbatim, so they only have to be distinguishable from each other.
_REQUEST: dict[str, object] = {"ec_level": "M", "payload_len": 32}
_DECODED: dict[str, object] = {"agreement": "agree", "payload_len": 31}
_PNG = b"\x89PNG\r\n\x1a\n" + b"qrcheck symbol bytes"

# The bytes a file the store must never overwrite already holds.
_VICTIM = b"untouched\n"


def _case(
    status: Status = Status.FAILED,
    reason: Reason | None = Reason.VALUE_MISMATCH,
) -> CaseResult:
    """Return the case under test, carrying no evidence directory yet."""
    return CaseResult(
        check_id=_CHECK_ID,
        case_id=_CASE_ID,
        status=status,
        reason=reason,
        duration_ms=12.5,
        request=_REQUEST,
        decoded=_DECODED,
        seed=_SEED,
    )


def _symbol() -> SimpleNamespace:
    """Return the duck-typed symbol the store writes as ``symbol.png``, bytes untouched."""
    return SimpleNamespace(image_bytes=_PNG, output_format="png")


def _case_dir(report_dir: Path, failure_id: str = _FAILURE_ID) -> Path:
    """Return the evidence directory the case under test is filed under."""
    return report_dir / EVIDENCE_DIR_NAME / _CHECK_ID / failure_id


def _entries(root: Path) -> list[str]:
    """Return every entry under ``root`` as a sorted relative posix path."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def _files(root: Path) -> list[str]:
    """Return every regular file under ``root``, so a refusal can assert none was created."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


@pytest.mark.parametrize("component", ("evidence", "check-id", "failure-id"))
def test_a_symlinked_component_is_refused_and_its_target_stays_empty(
    report_dir: Path, component: str
) -> None:
    """A link planted below the report root stops the write, by name, with nothing written."""
    target = report_dir.parent / f"target-{component}"
    target.mkdir()
    link = {
        "evidence": report_dir / EVIDENCE_DIR_NAME,
        "check-id": report_dir / EVIDENCE_DIR_NAME / _CHECK_ID,
        "failure-id": _case_dir(report_dir),
    }[component]
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)

    with pytest.raises(OSError) as raised:
        EvidenceStore(report_dir).record(_case(), symbol=_symbol())

    assert str(link) in str(raised.value)
    assert "symlink" in str(raised.value)
    assert _entries(target) == []
    assert _files(report_dir) == []


@pytest.mark.parametrize("artifact_name", (REQUEST_JSON, DECODED_JSON, SYMBOL_PNG))
def test_a_symlinked_artifact_destination_is_refused(
    report_dir: Path, artifact_name: str
) -> None:
    """A link where an artefact belongs is refused, and the file it names keeps its bytes."""
    victim = report_dir.parent / "victim.bin"
    victim.write_bytes(_VICTIM)
    directory = _case_dir(report_dir)
    directory.mkdir(parents=True)
    (directory / artifact_name).symlink_to(victim)

    with pytest.raises(OSError) as raised:
        EvidenceStore(report_dir).record(_case(), symbol=_symbol())

    assert str(directory / artifact_name) in str(raised.value)
    assert victim.read_bytes() == _VICTIM
    assert sorted(path.name for path in directory.iterdir()) == [artifact_name]


def test_the_real_path_check_refuses_a_link_the_component_scan_missed(
    report_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolved-path comparison refuses on its own, so the two checks are independent."""
    target = report_dir.parent / "outside-evidence"
    target.mkdir()
    (report_dir / EVIDENCE_DIR_NAME).symlink_to(target)
    # A link planted after the component scan is the race the second check exists for: with the
    # scan blind to links, only the comparison of real paths can refuse the write.
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)

    with pytest.raises(OSError) as raised:
        EvidenceStore(report_dir).record(_case(), symbol=_symbol())

    assert "leaves the report directory" in str(raised.value)
    assert _entries(target) == []


@pytest.mark.parametrize(
    ("status", "reason", "failure_id"),
    (
        (Status.FAILED, Reason.VALUE_MISMATCH, _FAILURE_ID),
        (
            Status.SKIPPED,
            Reason.MISSING_DEPENDENCY,
            "RT-ROUNDTRIP__corpus_url-000__missing_dependency",
        ),
    ),
)
def test_a_normal_record_writes_the_closed_set_and_returns_the_relative_path(
    report_dir: Path, status: Status, reason: Reason, failure_id: str
) -> None:
    """An unlinked root records the artefact set, and the directory is named relatively."""
    case = _case(status, reason)

    recorded = EvidenceStore(report_dir).record(case, symbol=_symbol())

    assert recorded == f"{EVIDENCE_DIR_NAME}/{_CHECK_ID}/{failure_id}"
    directory = _case_dir(report_dir, failure_id)
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        (DECODED_JSON, REQUEST_JSON, SYMBOL_PNG)
    )
    assert json.loads((directory / REQUEST_JSON).read_text(encoding="utf-8")) == {
        **_REQUEST,
        "check_id": _CHECK_ID,
        "case_id": _CASE_ID,
        "seed": _SEED,
    }
    assert json.loads((directory / DECODED_JSON).read_text(encoding="utf-8")) == _DECODED
    assert (directory / SYMBOL_PNG).read_bytes() == _PNG


def test_a_report_root_that_is_a_symlink_still_records(report_dir: Path) -> None:
    """``report_dir`` is the operator's own path: a linked report root records as usual."""
    real = report_dir / "real-root"
    real.mkdir()
    linked = report_dir / "linked-root"
    linked.symlink_to(real)

    recorded = EvidenceStore(linked).record(_case(), symbol=_symbol())

    assert recorded == f"{EVIDENCE_DIR_NAME}/{_CHECK_ID}/{_FAILURE_ID}"
    names = (DECODED_JSON, REQUEST_JSON, SYMBOL_PNG)
    assert _files(real) == sorted(f"{recorded}/{name}" for name in names)


def test_a_relative_report_root_still_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root is used as configuration gave it: a relative one records below the cwd."""
    monkeypatch.chdir(tmp_path)

    recorded = EvidenceStore("reports").record(_case(), symbol=_symbol())

    assert recorded == f"{EVIDENCE_DIR_NAME}/{_CHECK_ID}/{_FAILURE_ID}"
    assert (tmp_path / "reports" / recorded / REQUEST_JSON).is_file()


def test_a_passed_case_returns_none_and_writes_nothing(report_dir: Path) -> None:
    """The persistence policy is unchanged: a green case leaves no directory and no file."""
    passed = _case(Status.PASSED, None)

    assert EvidenceStore(report_dir).record(passed) is None
    assert _entries(report_dir) == []
