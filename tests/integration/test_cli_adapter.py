"""The ``cli`` adapter against its shipped example target, over a real process boundary.

The integration gate for AAP §0.3.3's ``cli`` contract -- ``command``, ``payload_arg``,
``output_arg``, ``returns`` and ``metadata_map`` -- and for AAP §0.9.2's rule that
``examples/cli_target.py`` is load-bearing: the example's documented configuration executes
here, so it either keeps working or CI fails. Nothing is mocked -- every case spawns the example
or ``python -m qrcheck`` as a subprocess and judges the artefact with the pinned real decoders.

The eight cases: a legal request yielding a PNG that decodes back to the requested bytes at the
capacity tables' version, with ``close()`` safe twice; shell metacharacters reaching the target
byte for byte, the behavioural proof of the argv-only ``shell = False`` control (AAP §0.3.2); a
declared rejection becoming ``TargetRejected`` carrying the target's own text; a non-zero exit
with no rejection as ``TargetContractError``, which keeps that trigger "exactly one"; an
over-time target killed and reported ``TargetTimeout`` naming the timeout; an unspawnable
command as ``TargetUnavailable``, never an escaped ``OSError``; a bounded run driving every
``otp`` record through adapter, decoders and writers; and an stdout-based ``returns`` refused as
a contract error.

Only ``returns = "file_path"`` can be exercised end to end, because the example writes its
symbol to ``--output`` and nothing to stdout; ``bytes_path``, ``raw_bytes`` and ``base64`` are
held at unit level by ``tests/unit/test_targets_transport.py`` rather than by a second example
invented here. ``CliAdapter.generate`` gates every optional field the caller actually set, so a
request pinning ``module_size_px`` or ``quiet_zone_modules`` needs the matching declared
capability: the example's templates carry neither and its canonical table declares only ``png``,
which is why case 1 supplies both as static words of its own ``command`` and declares the two
capabilities it honours.

Every case spawns its target by the absolute path in ``cli_target.script`` and writes under
``tmp_path``, no duration is asserted (AAP §0.5.5 keeps wall-clock assertions to the ``PF-*``
checks), every adapter is closed by its ``with`` block, and the process table is never inspected
because another checkout on this host may run the same example. No ``e2e`` marker: the module is
bounded by ``--checks``/``--class`` and belongs in the fast job (AAP §0.6.4), and a missing
example script fails rather than skips. AAP §0.12's ``MUST``/``MUST NOT`` list and §0.4.3's
ban on skipping, ``xfail``, retry and threshold relaxation bind every case.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from qrcheck.config.load import load_config
from qrcheck.corpus.capacity import minimal_version
from qrcheck.decode import decode_symbol
from qrcheck.targets import ADAPTER_TYPES
from qrcheck.targets.base import (
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


def _table(base: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return the shipped example's target table with ``overrides`` applied.

    A fresh mapping, so a case that replaces the command line or an argument template cannot reach
    the fixture's table or another case's. ``base`` mirrors the canonical block in
    ``examples/cli_target.py``'s own docstring; the overrides are the keys this case changes -- a
    different ``command``, or the bare-text ``payload_arg`` case 2 sends metacharacters through.
    """
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
    """Write a schema-valid document declaring ``name`` as a ``cli`` target.

    The ``write_config`` fixture fills every key the caller did not choose from
    :data:`qrcheck.config.schema.SCHEMA`'s declared defaults, so no threshold, sample count or
    timeout is restated in this module (AAP §0.12).
    """
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
    """Build the ``cli`` adapter from a real configuration and close it on the way out.

    The adapter is resolved through :func:`qrcheck.config.load.load_config` and
    ``ADAPTER_TYPES["cli"]`` exactly as ``qrcheck run`` and ``qrcheck doctor`` resolve one, so every
    case exercises the configuration schema and the factory as well as the transport: a table the
    schema rejects fails here instead of being hand-constructed past it. The adapter is asserted to
    satisfy the :class:`~qrcheck.targets.base.Adapter` port, so a factory returning something else
    fails before a single request is built; nothing is opened at construction time, and ``close``
    runs in a ``finally`` so no case can leave a child behind.
    """
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
    """A legal request yields a PNG that decodes back to the requested bytes.

    Catches an adapter that assembles the wrong argv, a template whose placeholders are never
    substituted, a missing artefact read, a symbol rendered at a version the capacity tables do not
    prescribe, and a ``close`` that is not safe to call twice. The geometry comes from the case's
    own ``command`` as two static words and the two capabilities it implies are declared, because
    the example's templates carry neither (see the module docstring); the ``declared_*`` fields are
    asserted ``None``, because the example writes nothing to stdout and a target that reports no
    metadata is recorded as reporting none rather than believed.
    """
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
    """Shell metacharacters reach the target byte for byte, as one literal argv word.

    Catches the defect that matters most in this layer: an adapter that renders a command *string*
    and lets a shell interpret it, which AAP §0.3.2 forbids (argv only, ``shell = False``). A shell
    would have expanded ``$(echo pwned)`` and the backtick before the target ever saw the payload,
    so the decoded bytes would differ from the requested ones. The payload travels through the bare
    ``{}`` placeholder rather than ``{payload_b64}``, because that spelling puts the raw characters
    into the argument vector and the base64 form would encode them first and prove nothing about a
    shell; nothing here inspects the adapter's internals, because the round trip is the evidence.
    """
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
    """A declared rejection becomes ``TargetRejected`` carrying the target's own text.

    Catches an adapter that maps exit 2 to a generic failure, paraphrases, prefixes or truncates the
    target's message, or looks for the rejection on the wrong stream. The expected text is captured
    by running the example once with the argument vector the adapter builds, so the assertion stays
    verbatim without freezing a message the example owns. The request is the deliberate
    out-of-contract probe of AAP §0.8.2 -- an empty payload -- constructed through
    :class:`GenerateRequest` and not :func:`build_request`, because ``IV-EMPTY`` exists to prove a
    target refuses exactly that, so the builder's validation is bypassed on purpose.
    """
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
    """A non-zero exit that declares no rejection is a ``TargetContractError``.

    Catches an adapter that treats every non-zero exit as a declared rejection, which would make the
    ``IV-*`` checks vacuous: they pass on a rejection, so a fabricated one would let any broken
    target through. With the case above it pins the trigger to "exactly one" -- a rejection is what
    a target *says*, never what its exit status implies.
    """
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
    """A target outliving ``timeout_s`` is killed and reported as ``TargetTimeout``.

    Catches an adapter that lets ``subprocess.TimeoutExpired`` escape unwrapped, that passes no
    timeout at all -- in which case this case would hang for the sleeper's full length instead of
    failing -- or that reports the failure without the configured number a reader needs to fix it.
    No elapsed duration is asserted: the case finishing at all is the evidence that the timeout
    fired, because AAP §0.5.5 keeps wall-clock assertions out of all but the ``PF-*`` checks.
    """
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
    """A command that cannot be spawned is ``TargetUnavailable``, not an escaped ``OSError``.

    Catches an adapter that lets :class:`FileNotFoundError` -- which ``subprocess.run`` raises from
    ``execvp`` alone, before any pipe is read -- reach the caller unwrapped. The five-error
    vocabulary of AAP §0.3.4 is closed and "the process cannot be spawned" is
    ``TargetUnavailable``'s single trigger, so the exception type is asserted to be that one and
    explicitly not an :class:`OSError`.
    """
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
    """A bounded run drives every ``otp`` record through the adapter and passes all twelve.

    Catches a break anywhere on the real path -- the document a run loads, the registry's selection,
    the runner's capability resolution, the adapter's spawn and artefact read, the decoders that
    judge the symbol, and the three writers -- none of which a per-request test of the adapter alone
    can see. ``otp`` holds exactly twelve corpus records (AAP §0.8.2) and the selection is bounded
    to one check, so the module stays in the fast CI job instead of paying for a full run. The
    counts are asserted as four separate integers, so a ``skipped`` or an ``error`` result cannot
    hide behind a passing tally: every record must genuinely round-trip.
    """
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
    """An stdout-based ``returns`` is refused rather than read from the wrong stream.

    Catches an adapter that would hand a zero-exit, path-writing target's empty stdout to a decoder
    and report the resulting confusion as a symbol defect: ``examples/cli_target.py`` writes its PNG
    to ``--output`` and nothing to stdout when it succeeds, so ``returns = "base64"`` miswires the
    target, and AAP §0.3.4 lists "empty body" among ``TargetContractError``'s triggers -- a loud
    failure rather than a symbol read from the wrong place.

    This is the honest scope of the ``returns`` coverage: ``file_path`` is the one delivery mode the
    shipped example can serve end to end, and ``bytes_path``, ``raw_bytes`` and ``base64`` are
    decoded by :func:`qrcheck.targets.base.decode_returned_image` and held at unit level by
    ``tests/unit/test_targets_transport.py``. No second example script is invented here to widen
    that coverage, because it would document a shape the shipped example does not have.
    """
    table = _table(cli_target.target_config, returns="base64")
    request = build_request(_BASE64_PAYLOAD, "M")

    with _open_adapter(write_config, tmp_path, table) as adapter:
        with pytest.raises(TargetContractError) as contract:
            adapter.generate(request)

    message = str(contract.value)
    assert "base64" in message
    assert "empty artefact" in message
