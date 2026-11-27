"""The ``cli`` adapter against its shipped example target, over a real process boundary.

This module is the integration gate for AAP §0.3.3's ``cli`` target contract and for AAP
§0.9.2's rule that ``examples/cli_target.py`` is load-bearing: the example's documented
configuration is executed here, so it either keeps working or CI fails. Nothing is mocked --
every case spawns a real process and the decoders that judge the artefact are the pinned real
ones.

The cases: a legal request produces a PNG decoding back to the requested bytes; shell
metacharacters reach the target byte for byte (AAP §0.3.2's argv-only control); a declared
rejection is ``TargetRejected`` with the target's own text while a non-zero exit that
declares nothing is ``TargetContractError``; a target outliving ``timeout_s`` is
``TargetTimeout``; an unspawnable command is ``TargetUnavailable``; a bounded run drives
every ``otp`` record through the pipeline; an stdout-based ``returns`` is a contract
error on this example; and a child that streams to stdout or to stderr without end is cut
off at the adapter's documented byte bound and refused as ``TargetContractError``. No
duration is compared with a product threshold (AAP §0.5.5) -- the two capture-bound cases
only measure that the call returned far inside their own configured ``timeout_s``, which is
what proves the child was cut off at the bound and not killed at the ceiling -- the module
carries no ``e2e`` marker, and a missing example script fails rather than skips (AAP §0.4.3)."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.config.load import load_config
from qrcheck.corpus.capacity import minimal_version
from qrcheck.decode import decode_symbol
from qrcheck.targets import ADAPTER_TYPES, cli as cli_module
from qrcheck.targets.base import (
    HEX_PREFIX_LIMIT,
    Adapter,
    GenerateRequest,
    TargetContractError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    build_request,
)

# The name the target under test is declared under, passed explicitly through ``--adapter`` in case
# 7 so the assertion names the target it drove.
_TARGET_NAME: Final[str] = "example_cli"

# The configuration file's name and the report root's, both created under ``tmp_path``.
_CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"
_REPORT_DIR_NAME: Final[str] = "reports"

# The PNG signature: the first assertion that proves the example handed back an image rather than,
# say, an error page behind a ``.png`` path.
_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"

# The geometry case 1 pins, supplied to the example as static words of its ``command`` so the symbol
# really carries them and the declared capabilities are honest rather than aspirational.
_MODULE_SIZE_PX: Final[int] = 4
_QUIET_ZONE_MODULES: Final[int] = 4

# The timeout case 5 configures: a harness value rather than a product threshold, being the
# per-target ``timeout_s`` key inside the schema's ``> 0`` rule, and no other case reads it.
_TIMEOUT_S: Final[float] = 0.5

# How long the one direct ``subprocess.run`` in case 3 may take: a guard against a hang in the
# capture run, not a measurement and not a product threshold.
_SUBPROCESS_GUARD_S: Final[float] = 120.0

# The one exit status ``examples/cli_target.py`` uses to declare a rejection.
_REJECTED_EXIT_CODE: Final[int] = 2

# Case 2's payload: shell metacharacters a shell-interpreting invocation would act on -- a quote, a
# redirect-and-command sequence, a command substitution, a backtick and a single quote. Strictly
# ASCII and NUL-free, so it can travel as one argv word.
_METACHARACTER_PAYLOAD: Final[bytes] = b'"; rm -rf / #' + b"$(echo pwned)" + b"`id`" + b"'"

# Case 7's selection: ``otp`` is one of the corpus's fourteen classes and holds exactly twelve
# records (AAP §0.8.2), which is the pass count the assertion expects.
_OTP_CLASS: Final[str] = "otp"
_OTP_CLASS_RECORDS: Final[int] = 12

# The single check case 7 selects, so the run is bounded to twelve encodes and twelve decodes
# instead of the 1,287 cases of a full run (AAP §0.8.1).
_ROUNDTRIP_CHECK_ID: Final[str] = "RT-ROUNDTRIP"

# The one payload each of cases 1, 2, 4-8 sends: lowercase letters keep it in byte mode, so its
# expected version comes from the byte row of the capacity tables.
_DEFAULT_PAYLOAD: Final[bytes] = b"qrcheck-cli-adapter"
_CONTRACT_PAYLOAD: Final[bytes] = b"qrcheck-cli-contract-error"
_TIMEOUT_PAYLOAD: Final[bytes] = b"qrcheck-cli-timeout"
_UNAVAILABLE_PAYLOAD: Final[bytes] = b"qrcheck-cli-unavailable"
_BASE64_PAYLOAD: Final[bytes] = b"qrcheck-cli-base64"
_STREAM_BOUND_PAYLOAD: Final[bytes] = b"qrcheck-cli-stream-bound"

# The ``timeout_s`` the two capture-bound cases configure. A harness value, not a product
# threshold: it only has to sit far above the fraction of a second the byte bound needs, so that a
# child cut off at the bound and a child killed at this ceiling cannot be confused.
_STREAM_BOUND_TIMEOUT_S: Final[float] = 20.0

# A child that writes to one stream without end, one pipe buffer at a time, so it passes any bound
# in a few writes. Neither the payload nor the artefact path matters to it: the adapter refuses the
# call on the bytes it captured, before it reads anything the child produced.
_ENDLESS_STDOUT_CHILD: Final[str] = (
    "import sys\n"
    "chunk = 'x' * 65536\n"
    "while True:\n"
    "    sys.stdout.write(chunk)\n"
    "    sys.stdout.flush()\n"
)
_ENDLESS_STDERR_CHILD: Final[str] = _ENDLESS_STDOUT_CHILD.replace("sys.stdout", "sys.stderr")


def _table(base: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return the shipped example's target table with ``overrides`` applied, as a fresh mapping."""
    table = dict(base)
    table.update(overrides)
    return table


def _write_cli_config(
    write_config: Callable[..., Path],
    directory: Path,
    table: Mapping[str, Any],
    *,
    name: str = _TARGET_NAME,
) -> tuple[Path, Path]:
    """Write a schema-valid document declaring ``name`` as a ``cli`` target."""
    report_dir = directory / _REPORT_DIR_NAME
    config_path = write_config(
        directory / _CONFIG_FILE_NAME,
        default_adapter=name,
        report_dir=report_dir,
        targets={name: dict(table)},
    )
    return config_path, report_dir


@contextlib.contextmanager
def _open_adapter(
    write_config: Callable[..., Path],
    directory: Path,
    table: Mapping[str, Any],
    *,
    name: str = _TARGET_NAME,
) -> Iterator[Adapter]:
    """Build the ``cli`` adapter from a real configuration and close it on the way out."""
    config_path, _ = _write_cli_config(write_config, directory, table, name=name)
    config = load_config(config_path)
    adapter = ADAPTER_TYPES["cli"](config.targets[name], name=name)
    assert isinstance(adapter, Adapter), (
        f"ADAPTER_TYPES['cli'] returned {type(adapter).__name__}, which does not satisfy the "
        "Adapter port of qrcheck.targets.base"
    )
    try:
        yield adapter
    finally:
        adapter.close()


def test_generate_returns_a_decodable_png(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """A legal request yields a PNG that decodes back to the requested bytes."""
    request = build_request(
        _DEFAULT_PAYLOAD,
        "M",
        output_format="png",
        module_size_px=_MODULE_SIZE_PX,
        quiet_zone_modules=_QUIET_ZONE_MODULES,
        foreground_rgb=(0, 0, 0),
        background_rgb=(255, 255, 255),
    )
    table = _table(
        cli_target.target_config,
        command=[
            sys.executable,
            str(cli_target.script),
            "--size",
            str(_MODULE_SIZE_PX),
            "--margin",
            str(_QUIET_ZONE_MODULES),
        ],
        capabilities=["png", "module_size", "quiet_zone"],
    )

    with _open_adapter(write_config, tmp_path, table) as adapter:
        symbol = adapter.generate(request)

    # The port requires close() to be safe to call twice: the runner closes every adapter
    # once the run ends, including after a failure.
    adapter.close()

    assert symbol.image_bytes.startswith(_PNG_SIGNATURE)
    assert symbol.output_format == "png"
    assert symbol.declared_version is None
    assert symbol.declared_ec is None
    assert symbol.declared_mask is None
    assert symbol.declared_mode is None
    assert isinstance(symbol.elapsed_ms, float)
    assert symbol.elapsed_ms >= 0.0

    # The lowest version that holds this payload at this level in byte mode is the version a
    # conforming encoder must have used, read back from the sampled module count rather than
    # from anything the target claims.
    expected_version = minimal_version(len(request.payload), request.ec_level, "byte")
    assert expected_version is not None
    decoded = decode_symbol(symbol.image_bytes, request)
    assert decoded.payload_bytes == request.payload
    assert decoded.version == expected_version


def test_payload_reaches_the_target_verbatim_through_argv(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """Shell metacharacters reach the target byte for byte, as one literal argv word."""
    payload = _METACHARACTER_PAYLOAD
    assert b"$(echo pwned)" in payload
    assert b"`id`" in payload
    request = build_request(payload, "M")
    table = _table(cli_target.target_config, payload_arg="{}")

    with _open_adapter(write_config, tmp_path, table) as adapter:
        symbol = adapter.generate(request)

    assert symbol.image_bytes.startswith(_PNG_SIGNATURE)
    decoded = decode_symbol(symbol.image_bytes, request)
    assert decoded.payload_bytes == payload


def test_declared_rejection_becomes_TargetRejected_with_the_text_verbatim(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """A declared rejection becomes ``TargetRejected`` carrying the target's own text."""
    request = GenerateRequest(payload=b"", ec_level="M", output_format="png")
    captured = tmp_path / "captured.png"
    completed = subprocess.run(
        [sys.executable, str(cli_target.script), "--payload-b64=", f"--output={captured}"],
        capture_output=True,
        timeout=_SUBPROCESS_GUARD_S,
        check=False,
    )

    assert completed.returncode == _REJECTED_EXIT_CODE, completed.stderr
    lines = completed.stdout.decode("utf-8").splitlines()
    assert len(lines) == 1, f"a rejection is exactly one line of JSON: {lines!r}"
    expected_text = json.loads(lines[0])["error"]
    assert expected_text

    with _open_adapter(write_config, tmp_path, cli_target.target_config) as adapter:
        with pytest.raises(TargetRejected) as rejection:
            adapter.generate(request)

    assert str(rejection.value) == expected_text


def test_non_zero_exit_without_a_rejection_is_a_contract_error(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """A non-zero exit declaring no rejection is a ``TargetContractError``."""
    table = _table(
        cli_target.target_config,
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
    )
    request = build_request(_CONTRACT_PAYLOAD, "M")

    with _open_adapter(write_config, tmp_path, table) as adapter:
        with pytest.raises(TargetContractError) as contract:
            adapter.generate(request)

    message = str(contract.value)
    assert "status 3" in message
    assert not isinstance(contract.value, TargetRejected)


def test_hung_target_times_out(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """A target outliving ``timeout_s`` is killed and reported as ``TargetTimeout``."""
    table = _table(
        cli_target.target_config,
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=_TIMEOUT_S,
    )
    request = build_request(_TIMEOUT_PAYLOAD, "M")

    with _open_adapter(write_config, tmp_path, table) as adapter:
        with pytest.raises(TargetTimeout) as timeout:
            adapter.generate(request)

    message = str(timeout.value)
    assert f"timeout_s = {_TIMEOUT_S:g} s" in message
    assert not isinstance(timeout.value, TargetContractError)


def test_missing_executable_is_TargetUnavailable(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """A command that cannot be spawned is ``TargetUnavailable``, never an escaped ``OSError``."""
    missing = "/nonexistent/qrcheck-no-such-target"
    table = _table(cli_target.target_config, command=[missing])
    request = build_request(_UNAVAILABLE_PAYLOAD, "M")

    with _open_adapter(write_config, tmp_path, table) as adapter:
        with pytest.raises(TargetUnavailable) as unavailable:
            adapter.generate(request)

    message = str(unavailable.value)
    assert "could not be spawned" in message
    assert missing in message
    assert not isinstance(unavailable.value, OSError)


def test_bounded_run_through_the_cli_adapter_passes(
    cli_target: Any,
    write_config: Callable[..., Path],
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """A bounded run drives every ``otp`` record through adapter, decoders and writers."""
    config_path, report_dir = _write_cli_config(
        write_config, tmp_path, cli_target.target_config
    )
    result = run_cli(
        "run",
        "--adapter",
        _TARGET_NAME,
        "--ec",
        "M",
        "--checks",
        _ROUNDTRIP_CHECK_ID,
        "--class",
        _OTP_CLASS,
        config=config_path,
    )

    assert result.exit_code == 0, result.stderr
    summary_path = report_dir / "summary.json"
    assert summary_path.is_file(), f"the run wrote no {summary_path.name}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    counts = summary["counts"]
    assert counts["passed"] == _OTP_CLASS_RECORDS
    assert counts["failed"] == 0
    assert counts["error"] == 0
    assert counts["skipped"] == 0

    results = summary["results"]
    assert len(results) == _OTP_CLASS_RECORDS
    assert {entry["check_id"] for entry in results} == {_ROUNDTRIP_CHECK_ID}
    assert {entry["status"] for entry in results} == {"passed"}


def test_stdout_based_returns_is_a_contract_error_on_this_example(
    cli_target: Any, write_config: Callable[..., Path], tmp_path: Path
) -> None:
    """An stdout-based ``returns`` is refused rather than read from the wrong stream."""
    table = _table(cli_target.target_config, returns="base64")
    request = build_request(_BASE64_PAYLOAD, "M")

    with _open_adapter(write_config, tmp_path, table) as adapter:
        with pytest.raises(TargetContractError) as contract:
            adapter.generate(request)

    message = str(contract.value)
    assert "base64" in message
    assert "empty artefact" in message


# The capture bound: one stream of the child cannot grow this process without limit


@pytest.mark.parametrize(
    ("stream", "child"),
    [("stdout", _ENDLESS_STDOUT_CHILD), ("stderr", _ENDLESS_STDERR_CHILD)],
)
def test_an_endless_stream_is_refused_at_the_capture_bound(
    cli_target: Any,
    write_config: Callable[..., Path],
    tmp_path: Path,
    stream: str,
    child: str,
) -> None:
    """A child that writes to `stream` without end is cut off at the documented byte bound and
    refused as a ``TargetContractError`` naming that stream and the bound it passed.

    ``_STREAM_BOUND_TIMEOUT_S`` is a harness value, not a product threshold: it is this case's own
    ``timeout_s``, and a child that had to be killed at that ceiling would consume the whole of it.
    Returning far inside it is what proves the child was cut off at the bound instead.
    """
    table = _table(
        cli_target.target_config,
        command=[sys.executable, "-c", child],
        payload_arg="{}",
        returns="raw_bytes",
        timeout_s=_STREAM_BOUND_TIMEOUT_S,
    )
    request = build_request(_STREAM_BOUND_PAYLOAD, "M")

    started = time.perf_counter()
    with _open_adapter(write_config, tmp_path, table) as adapter:
        with pytest.raises(TargetContractError) as contract:
            adapter.generate(request)
    elapsed_s = time.perf_counter() - started

    message = str(contract.value)
    assert stream in message, f"the diagnostic must name the over-bound stream: {message!r}"
    assert str(cli_module._MAX_CAPTURED_STREAM_BYTES) in message, (
        f"the diagnostic must name the byte bound the stream passed: {message!r}"
    )
    assert len(contract.value.body_prefix) == HEX_PREFIX_LIMIT, (
        "the over-bound stream's first bytes must travel as evidence; got "
        f"{len(contract.value.body_prefix)} bytes"
    )
    assert elapsed_s < _STREAM_BOUND_TIMEOUT_S / 2, (
        "the child must be cut off at the capture bound rather than killed at "
        f"timeout_s = {_STREAM_BOUND_TIMEOUT_S:g} s; the call took {elapsed_s:.3f} s"
    )
