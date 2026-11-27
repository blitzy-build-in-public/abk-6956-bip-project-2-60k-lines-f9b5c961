"""Every pinned runtime dependency blocks a run, and the diagnostic names it and the lock file.

``segno`` is the oracle and ``numpy``, ``Pillow``, ``httpx`` and ``click`` are imported at their
point of use, so an unusable one "blocks with a diagnostic naming the missing distribution and
pointing at ``requirements.lock``; a broken install is an environment error (exit 2), not a
silent degradation". Held at the process boundary: ``doctor`` and ``run`` both
exit 2 -- never 3, never with a traceback -- naming the distribution and the lock file, and the
run writes no report, ``httpx`` getting its own case because it loads only for an ``http``
target. Absence is the state a broken installation really has -- module present, import failing
-- produced by the :func:`unimportable_module` fixture, whose stub reaches the child process
through ``PYTHONPATH`` while the distribution metadata still reports the installed version. No
test skips or retries, and every artefact lands under ``tmp_path``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.cli.exit import EXIT_CONFIG
from qrcheck.dependencies import LOCK_FILE
from qrcheck.report import SUMMARY_FILENAME

#: The core dependencies whose failure surfaces while the verbs register, as ``(distribution a
#: reader installs, module the stub shadows)``; ``httpx`` is not here -- it loads only for ``http``.
REGISTRATION_TIME_DEPENDENCIES: Final[tuple[tuple[str, str], ...]] = (
    ("segno", "segno"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
)

#: The built-in target, which needs no socket, and a check that needs no target at all.
REFERENCE_TARGET: Final[str] = "reference"
CONTROL_CHECK: Final[str] = "CL-ARTIFACTS"

#: The declared ``http`` target's name, and a loopback URL on a port nothing listens on: the run
#: must refuse on the missing client library before any connection is attempted.
LOOPBACK_TARGET: Final[str] = "loopback"
LOOPBACK_URL: Final[str] = "http://127.0.0.1:9/generate"

#: How much of a process-level diagnostic a failure message quotes.
DIAGNOSTIC_CHARS: Final[int] = 800


def config_for(
    write_config: Callable[..., Path],
    directory: Path,
    *,
    targets: Mapping[str, Mapping[str, Any]] | None = None,
    default_adapter: str = REFERENCE_TARGET,
) -> tuple[Path, Path]:
    """Write a schema-valid configuration under ``directory``; return it with its report root.

    The report root is fresh per test, so a leaked report cannot satisfy an absence assertion.
    """
    report_dir = directory / "reports"
    config = write_config(
        directory / "qrcheck.toml",
        default_adapter=default_adapter,
        report_dir=report_dir,
        targets=targets,
    )
    return config, report_dir


def diagnostic(result: Any) -> str:
    """Return one invocation's outcome, bounded, for a failure message."""
    return (
        f"exit code {result.exit_code}, stdout tail {result.stdout[-DIAGNOSTIC_CHARS:]!r}, "
        f"stderr tail {result.stderr[-DIAGNOSTIC_CHARS:]!r}"
    )


def assert_refusal(result: Any, distribution: str) -> None:
    """Assert ``result`` is the exit-2 environment diagnostic naming ``distribution``.

    A traceback or an ``internal error`` line means the fault reached the last-resort handler, and a
    missing lock-file reference leaves the reader no install to fix.
    """
    assert result.exit_code == EXIT_CONFIG, (
        f"the command exited {result.exit_code} with {distribution!r} unimportable, expected "
        f"{EXIT_CONFIG}; {diagnostic(result)}"
    )
    for marker in ("internal error", "Traceback"):
        assert marker not in result.stderr, (
            f"the outcome carries {marker!r}, so a broken install is reported as a framework fault "
            f"rather than an environment one; {diagnostic(result)}"
        )
    for expected in (distribution, LOCK_FILE):
        assert expected in result.stderr, (
            f"the diagnostic omits {expected!r}; {diagnostic(result)}"
        )


@pytest.mark.parametrize(("distribution", "module"), REGISTRATION_TIME_DEPENDENCIES)
def test_both_verbs_refuse_an_unimportable_core_dependency(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    unimportable_module: Callable[[str], Path],
    tmp_path: Path,
    distribution: str,
    module: str,
) -> None:
    """``doctor`` and ``run`` exit 2 naming the distribution, and the run writes no report.

    Breaks if the ``ImportError`` reaches the last-resort arm again -- exit 3 with a traceback is
    what the review found -- or if the run completes, turning a broken install into results.
    """
    unimportable_module(module)
    config, report_dir = config_for(write_config, tmp_path)

    assert_refusal(run_cli("doctor", "--json", config=config), distribution)
    result = run_cli(
        "run", "--adapter", REFERENCE_TARGET, "--checks", CONTROL_CHECK, config=config
    )
    assert_refusal(result, distribution)
    assert list(report_dir.rglob(SUMMARY_FILENAME)) == [], (
        f"the run wrote a {SUMMARY_FILENAME} although it refused to start, so a report describes a "
        f"run that must not have happened; {diagnostic(result)}"
    )


def test_a_module_that_raises_anything_but_ImportError_is_a_named_dependency_fault(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    unimportable_module: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """A pinned module that cannot even initialise still exits 2 naming it, not 3 with a traceback.

    Two faults at once: the oracle's ``ImportError`` is the one the command line meets, and probing
    the second pin raises ``RuntimeError`` from inside that very handler. Breaks if the probe lets
    that fault escape, or if the diagnostic names only the first unusable dependency.
    """
    stub = unimportable_module("segno")
    (stub / "numpy.py").write_text(
        'raise RuntimeError("numpy cannot initialise")\n', encoding="utf-8", newline="\n"
    )
    config, report_dir = config_for(write_config, tmp_path)

    assert_refusal(run_cli("doctor", "--json", config=config), "numpy")
    result = run_cli(
        "run", "--adapter", REFERENCE_TARGET, "--checks", CONTROL_CHECK, config=config
    )
    assert_refusal(result, "numpy")
    assert not list(report_dir.rglob(SUMMARY_FILENAME)), diagnostic(result)


def test_an_unimportable_client_library_blocks_doctor_and_an_http_target_run(
    run_cli: Callable[..., Any],
    write_config: Callable[..., Path],
    unimportable_module: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """``httpx`` is reported ``missing`` and blocking, and an ``http`` run refuses naming it.

    Breaks if ``doctor`` derives the entry from the installed distribution instead of importability
    -- the review found ``status: "ok"`` published while the module was unimportable -- or if the
    ``http`` adapter's ``ImportError`` escapes as an internal error.
    """
    unimportable_module("httpx")
    config, report_dir = config_for(
        write_config,
        tmp_path,
        default_adapter=LOOPBACK_TARGET,
        targets={
            LOOPBACK_TARGET: {
                "type": "http",
                "url": LOOPBACK_URL,
                "method": "POST",
                "image_field": "_body",
                "image_encoding": "raw",
                "capabilities": ["png"],
                "request_template": {"payload_b64": "{payload_b64}", "format": "{format}"},
            }
        },
    )

    doctor = run_cli("doctor", "--json", config=config)
    assert_refusal(doctor, "httpx")
    document = json.loads(doctor.stdout)
    entries = [entry for entry in document["dependencies"] if entry["name"] == "httpx"]
    assert len(entries) == 1, (
        f"the report holds {len(entries)} entries named 'httpx', expected exactly one; reported: "
        f"{sorted(entry['name'] for entry in document['dependencies'])}"
    )
    entry = entries[0]
    assert entry["status"] == "missing" and entry["blocking"] is True, (
        f"httpx is reported as {entry!r}, so an unimportable library is not a blocking missing "
        f"dependency"
    )
    assert entry["version"] is not None, (
        f"the installed version disappeared from an entry whose distribution is present; {entry}"
    )

    result = run_cli(
        "run", "--adapter", LOOPBACK_TARGET, "--checks", CONTROL_CHECK, config=config
    )
    assert_refusal(result, "httpx")
    assert list(report_dir.rglob(SUMMARY_FILENAME)) == [], (
        f"the run wrote a {SUMMARY_FILENAME} although it refused to start, so an unreachable "
        f"target would be reported as though the network were at fault; {diagnostic(result)}"
    )
