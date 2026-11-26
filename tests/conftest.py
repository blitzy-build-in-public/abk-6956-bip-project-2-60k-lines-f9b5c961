"""The shared fixture surface for the ``qrcheck`` test suite: fixtures only, never tests.

Every module under ``tests/unit/`` and ``tests/integration/`` consumes what is defined
here, so no test module re-implements a report directory, a configuration record, a stub
adapter, a decoded result or the offline guard.  Nothing in this module is collected as a
test: it defines no ``test_*`` function and no ``Test*`` class, registers no marker (the
``e2e`` marker belongs to ``pyproject.toml``), and installs no retry, re-run, quarantine
or skip hook -- a retry has to surface as the declared adapter-invocation count plus one,
which is what the counting stub exists to make visible.

What the fixtures guarantee
---------------------------
* A fresh, writable report root per test (:func:`report_dir`), without pre-creating the
  ``evidence/`` subtree: ``qrcheck.report.evidence.EvidenceStore`` owns that layout.
* A fully populated :class:`qrcheck.config.models.Config` built **in memory** from the
  defaults declared in ``qrcheck/config/schema.py`` (:func:`config_factory`), so no test
  reads, mutates or depends on the committed ``qrcheck.toml``; plus one writer that puts a
  minimal valid document into ``tmp_path`` (:func:`toml_config_file`) for the tests that
  must drive the real CLI through a real ``--config`` path.
* Protocol-conforming target stubs (:class:`StubAdapter`, :class:`CountingAdapter`,
  :class:`RaisingAdapter`), recognised by ``isinstance(stub, Adapter)``, whose request log
  is the evidence of how many times a check called its target.
* The decoded-evidence stubs every check's negative case is driven through
  (:func:`make_decode_result`, :func:`make_decoder_outcome`) and the injection that puts
  one in front of the checks without a target, an image or a decoder (:func:`patch_decode`).
* A :class:`qrcheck.registry.check.RunContext` a check's ``cases(ctx)`` can be called with
  (:func:`make_run_context`), the committed and freshly built corpora, and a fixed small
  subset of them (:func:`corpus_subset`) so a unit test stays at millisecond cost.
* The determinism comparator (:func:`mask_volatile`) that masks exactly the three volatile
  fields of the result model, and the autouse :func:`no_outbound_network` guard that makes
  any outbound socket connect fail loudly, enforcing the product's fully-offline MUST.

Thresholds never appear in this module
--------------------------------------
Every numeric default a fixture supplies is read from ``qrcheck/config/schema.py``
(:data:`~qrcheck.config.schema.SCHEMA`) or from the ``models.DEFAULT_*`` constant that
table references, so retuning a threshold in ``qrcheck.toml`` cannot leave a fixture
asserting a stale value, and a test that needs a different threshold passes one override
instead of restating the record.

Fixtures that are also plain callables
--------------------------------------
The builders and the masking comparator are module-level functions *and* fixtures of the
same name, so a test may either request them by name (``def test_x(make_decode_result)``)
or import the function directly; both paths reach one implementation, and the duplicate
declaration is a name binding, not a second definition.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import ipaddress
import json
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy
import pytest
from click.testing import CliRunner

# Every local import below names a single owner of something a fixture hands out, so nothing
# is re-implemented here. Three of them are worth stating, because a reader cannot recover
# the reason from the code:
#
# * ``qrcheck.report.evidence`` is imported for ``EvidenceStore``, which
#   ``RunContext.evidence`` is typed as (AAP §0.8.1). The run-context fixture cannot exist
#   without it, so it is part of this file's scope even where a dependency list names only
#   ``qrcheck/report/model.py`` from that package.
# * ``qrcheck.decode`` -- the package facade -- is imported only to capture the *real*
#   ``decode_symbol``: ``patch_decode`` replaces that module attribute with a stub, and
#   ``real_decode`` has to keep the original callable reachable alongside it.
# * ``click.testing.CliRunner`` is the harness the exit-code conditions are driven through
#   (AAP §0.6.1 #7, §0.6.6 rows 1, 2, 10, 11) and ``click`` 8.1.8 is one of the product's
#   seven pinned runtime dependencies, so using it adds no dependency of its own.
from qrcheck.config import models, schema
from qrcheck.corpus.builder import CorpusRecord, build_corpus
from qrcheck.corpus.io import filter_by_class, load_corpus
from qrcheck.decode import decode_symbol as _REAL_DECODE_SYMBOL
from qrcheck.decode.base import DECODER_ORDER, DecodeResult, DecoderOutcome, Matrix
from qrcheck.registry.check import (
    DEPENDENCY_PRIMARY_DECODER,
    DEPENDENCY_SECONDARY_DECODER,
    DEPENDENCY_SVG_RASTERIZER,
    RunContext,
)
from qrcheck.report.evidence import EvidenceStore
from qrcheck.report.model import CaseResult, Counts, Reason, RunResult, Status
from qrcheck.targets.base import (
    Adapter,
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    TargetTimeout,
    TargetUnavailable,
    TargetUnsupported,
)

#: The module's public surface: the stubs, the builders and the fixtures, written out
#: sorted so ``dir()``, ``import *`` and any surface assertion are stable across runs. Every
#: name here exists as a module attribute; a fixture declared under a different function
#: name (the ``name=`` form) appears under the name a test requests.
__all__ = [
    "CountingAdapter",
    "OutboundNetworkBlocked",
    "RaisingAdapter",
    "StubAdapter",
    "adapter_raising",
    "baseline_decode_result",
    "boundary_corpus",
    "built_corpus",
    "cli_runner",
    "committed_corpus",
    "config",
    "config_factory",
    "config_file",
    "corpus_subset",
    "counting_adapter",
    "error_adapter",
    "invoke_cli",
    "make_case_result",
    "make_decode_result",
    "make_decoder_outcome",
    "make_run_context",
    "make_run_result",
    "mask_volatile",
    "no_outbound_network",
    "patch_decode",
    "real_decode",
    "report_dir",
    "run_context",
    "stub_adapter",
    "target_errors",
    "toml_config_file",
    "unwritable_report_dir",
]

# ---------------------------------------------------------------------------
# Locations and fixed values
#
# The repository root is needed for one purpose only: the committed corpus is a
# repository file (AAP §0.9.2), not package data, so the fixtures that load it name it
# relative to this module rather than relative to the current working directory.
# ---------------------------------------------------------------------------

#: The repository root, derived from this file's own location (``<root>/tests/conftest.py``).
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: The name of the configuration file a project writes; also the name of the minimal
#: document :func:`toml_config_file` creates under the test's ``tmp_path``.
_CONFIG_FILE_NAME: Final[str] = "qrcheck.toml"

#: How many records the fast subset fixtures carry. A sample size for keeping unit tests at
#: millisecond cost, not a product threshold: no check reads it and no configuration key
#: declares it.
_SUBSET_SIZE: Final[int] = 8

#: The payload the decoded-evidence baseline carries. Plain bytes on purpose: the negative
#: suite stubs the decode layer, so a real symbol's payload buys nothing there.
_STUB_PAYLOAD: Final[bytes] = b"qrcheck stub payload"

#: The bytes the target stub answers with by default. They are **not** a decodable image
#: and are not meant to be: a test that needs decoding stubs the decode layer instead of
#: paying for PNG encoding, and a test that needs real bytes passes its own.
_STUB_IMAGE_BYTES: Final[bytes] = b"\x89PNG\r\n\x1a\nqrcheck-stub"

#: The value the two run timestamps are masked to by :func:`mask_volatile`. Deliberately
#: not a timestamp: a masked summary must never parse as one, so a consumer that forgets to
#: mask cannot mistake the value for a real reading.
_MASKED_TIMESTAMP: Final[str] = "<masked>"

#: The value ``duration_ms`` is masked to by :func:`mask_volatile`. Kept numeric so the
#: masked projection still has the type the summary schema fixes for that field.
_MASKED_DURATION_MS: Final[float] = 0.0

#: The one volatile field of a result, masked by :func:`mask_volatile` (AAP §0.5.4).
_RESULT_VOLATILE_FIELD: Final[str] = "duration_ms"

#: The two volatile fields of a run, masked by :func:`mask_volatile` (AAP §0.5.4).
_RUN_VOLATILE_FIELDS: Final[tuple[str, str]] = ("started_at", "finished_at")

#: Fixed timestamps the run-result factory starts from. Constants, never a clock reading:
#: the result model requires a timezone-aware ISO-8601 pair and refuses one that would put
#: ``finished_at`` before ``started_at``, so the factory needs *some* valid pair, and a
#: constant one keeps two runs of the same factory equal.
_STUB_STARTED_AT: Final[str] = "2024-01-01T00:00:00+00:00"
_STUB_FINISHED_AT: Final[str] = "2024-01-01T00:00:01+00:00"

#: The package whose modules import the decode entry themselves, so :func:`patch_decode`
#: can reach the binding each of them actually resolves.
_CHECKS_PACKAGE: Final[str] = "qrcheck.checks"

#: The attribute name of the decode entry point in every module that exposes it.
_DECODE_ATTRIBUTE: Final[str] = "decode_symbol"


# ---------------------------------------------------------------------------
# Configuration defaults, read from the one declaration
# ---------------------------------------------------------------------------


def _schema_default(group: str, key: str) -> Any:
    """Return the declared default of one configuration key.

    The factory's only source of a default value: ``qrcheck/config/schema.py`` references
    the ``models.DEFAULT_*`` constants, so reading the table here is what keeps a fixture
    from carrying a second copy of a threshold (AAP §0.12).

    Args:
        group: The schema group path -- ``""`` for the top level, otherwise ``render``,
            ``checks.performance``, ``checks.damage`` or ``targets.*``.
        key: The key within that group.

    Returns:
        The declared fallback for the key, exactly as the schema holds it.

    Raises:
        KeyError: ``group`` or ``key`` is not declared, which is a defect in this module
            rather than a configuration fault.
    """
    return schema.SCHEMA[group][key].default


def _group_defaults(group: str) -> dict[str, Any]:
    """Return every declared default of one schema group, keyed by the configuration key.

    The result can be splatted straight into the frozen record that mirrors the group
    (``RenderConfig``, ``PerformanceLimits``, ``DamageLimits``), which is how the config
    factory proves it supplies what the schema declares and nothing retyped.

    Args:
        group: The schema group path, as :func:`_schema_default` takes it.

    Returns:
        One entry per declared key of the group, in the schema's declared order.
    """
    return {key: spec.default for key, spec in schema.SCHEMA[group].items()}


def _default_targets() -> Mapping[str, models.TargetConfig]:
    """Return the two built-in self-check targets, keyed by name in name order.

    ``reference`` and ``defective`` take only the common keys (AAP §0.3.4), so their
    records are :class:`~qrcheck.config.models.TargetConfig` instances; neither declares
    ``capabilities`` because both adapters declare all thirteen themselves. The mapping is
    read-only and sorted, matching the loader's contract, and it names both targets so a
    test can switch ``default_adapter`` without rebuilding the mapping.

    Returns:
        A read-only mapping of target name to its resolved record.
    """
    common = {
        "timeout_s": float(_schema_default("targets.*", "timeout_s")),
        "contrast_floor": float(_schema_default("targets.*", "contrast_floor")),
    }
    return MappingProxyType(
        {
            "defective": models.TargetConfig(name="defective", type="defective", **common),
            "reference": models.TargetConfig(name="reference", type="reference", **common),
        }
    )


def _toml_scalar(value: object) -> str:
    """Render one value as a TOML scalar for the minimal configuration document.

    Only the shapes the minimal document needs are handled -- a string, an integer and a
    float -- because the writer exists to produce a *valid* document; a test that needs an
    invalid one writes the text itself.

    Args:
        value: The value to render.

    Returns:
        The TOML literal for the value, quoted and escaped where it is a string.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _minimal_config_text(**overrides: Any) -> str:
    """Build the smallest configuration document that loads and validates.

    Five top-level keys plus one ``[targets.reference]`` table, every value taken from the
    schema's declared defaults unless overridden. ``report_dir`` and ``corpus_path`` are
    written as *relative* paths -- ``reports`` and ``corpus/default.jsonl`` -- because
    ``qrcheck/config/load.py`` resolves them against the configuration file's directory, so
    a document written into a test's ``tmp_path`` keeps all of its output there.

    Args:
        **overrides: Top-level key overrides, applied to the default document.

    Returns:
        The document text, newline-terminated.
    """
    values: dict[str, Any] = {
        key: _schema_default("", key)
        for key in ("default_adapter", "report_dir", "seed", "corpus_path", "loc_limit")
    }
    values.update(overrides)
    lines = [f"{key} = {_toml_scalar(value)}" for key, value in values.items()]
    lines.extend(("", "[targets.reference]", 'type = "reference"'))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Deterministic digests and stub payloads
# ---------------------------------------------------------------------------


def _content_hash(*parts: str | int) -> str:
    """Return a stable hexadecimal digest over `parts`, used for synthetic run identities.

    SHA-256, never the builtin ``hash()``: a run identity is compared across processes and
    machines, and ``hash()`` is salted per process (AAP §0.5.5). The digest is hex, so it
    can never be mistaken for the UUID or the timestamp the result model refuses.

    Args:
        *parts: The identity components, in a fixed order.

    Returns:
        The hexadecimal digest of the joined components.
    """
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _payload_text(payload: bytes | None) -> str | None:
    """Return `payload` as text when it round-trips through UTF-8, else ``None``.

    Args:
        payload: The payload bytes, or ``None`` when nothing was decoded.

    Returns:
        The decoded text, or ``None`` for absent or non-UTF-8 bytes.
    """
    if payload is None:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _stub_matrix(version: int = 2) -> Matrix:
    """Return a deterministic, synthetic module grid for a Model 2 symbol of `version`.

    The grid is ``4V + 17`` modules on a side (ISO/IEC 18004 Model 2), dark = ``True``,
    with a dark border and a single dark centre module. It is a *stub*: it is not a valid
    QR symbol and is never decoded. Its only jobs are to be a real
    :data:`~qrcheck.decode.base.Matrix` so ``matrix_sha256`` is exercised and to give a
    negative-case test something well-defined to replace with the defect it asserts against.

    Args:
        version: The symbol version the grid dimension should correspond to.

    Returns:
        A read-only-by-convention ``numpy`` boolean grid of shape ``(N, N)``.
    """
    side = 4 * version + 17
    grid = numpy.zeros((side, side), dtype=numpy.bool_)
    grid[0, :] = True
    grid[side - 1, :] = True
    grid[:, 0] = True
    grid[:, side - 1] = True
    grid[side // 2, side // 2] = True
    return grid


# ---------------------------------------------------------------------------
# Target stubs: the Adapter protocol implemented in four members
#
# Nothing here duplicates a production type: the records, the capability enumeration and
# the error classes are imported from ``qrcheck/targets/base.py``, which is the single
# declaration of each, so a rename there breaks this module rather than ten tests.
# ---------------------------------------------------------------------------


class StubAdapter:
    """A protocol-conforming target that produces canned bytes and records every request.

    Satisfies :class:`qrcheck.targets.base.Adapter` exactly -- ``name``, ``capabilities``,
    ``generate(request)`` and ``close()`` -- so ``isinstance(adapter, Adapter)`` holds and
    a check cannot tell it from a real target. ``capabilities`` defaults to all thirteen
    members, because a stub that left one out would turn the affected checks into
    ``skipped`` results and hide the assertion a test meant to exercise.

    ``generate`` appends its request to :attr:`requests` and returns a
    :class:`~qrcheck.targets.base.GeneratedSymbol` built from the configured bytes, or the
    symbol ``responder`` returns when one is set. The ``declared_*`` fields are the
    metadata a target *claims*; they are evidence only and are never an assertion.

    ``close`` is idempotent and only counts its calls, so a runner that closes an adapter
    twice is observable rather than exceptional.

    Raises:
        Nothing of its own: ``generate`` raises only what ``responder`` raises, which is how
        a test injects one of the five adapter errors without a second stub class.
    """

    def __init__(
        self,
        *,
        name: str = "stub",
        capabilities: Sequence[Capability] | Sequence[str] | None = None,
        image_bytes: bytes = _STUB_IMAGE_BYTES,
        output_format: str = "png",
        declared_version: int | None = None,
        declared_ec: str | None = None,
        declared_mask: int | None = None,
        declared_mode: str | None = None,
        elapsed_ms: float = 0.0,
        responder: Callable[[GenerateRequest], GeneratedSymbol] | None = None,
    ) -> None:
        """Configure the stub's identity, declared capabilities and canned answer.

        Args:
            name: The configured target name every diagnostic quotes.
            capabilities: What the stub declares it can do, as members or their string
                values. ``None`` declares all thirteen, so no check is skipped by accident.
            image_bytes: The artefact the stub returns.
            output_format: The format the stub claims it produced.
            declared_version: The version the stub claims, as evidence only.
            declared_ec: The EC level the stub claims, as evidence only.
            declared_mask: The mask the stub claims, as evidence only.
            declared_mode: The mode the stub claims, as evidence only.
            elapsed_ms: The target-side timing the stub reports, never asserted.
            responder: An optional callable producing the symbol for each request, so one
                stub can serve a whole check family.
        """
        self.name: str = name
        declared = Capability if capabilities is None else capabilities
        self.capabilities: frozenset[Capability] = frozenset(
            Capability(member) for member in declared
        )
        self.image_bytes: bytes = bytes(image_bytes)
        self.output_format: str = output_format
        self.declared_version: int | None = declared_version
        self.declared_ec: str | None = declared_ec
        self.declared_mask: int | None = declared_mask
        self.declared_mode: str | None = declared_mode
        self.elapsed_ms: float = elapsed_ms
        self.responder: Callable[[GenerateRequest], GeneratedSymbol] | None = responder
        self.requests: list[GenerateRequest] = []
        self.close_calls: int = 0
        self.closed: bool = False

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Record `request` and return the stub's answer.

        Args:
            request: The request the framework built for one case.

        Returns:
            The configured symbol, or what ``responder`` returned for `request`.

        Raises:
            BaseException: Whatever ``responder`` raises, unchanged.
        """
        self.requests.append(request)
        return self._produce(request)

    def _produce(self, request: GenerateRequest) -> GeneratedSymbol:
        """Build the symbol for a request already logged, or delegate to ``responder``."""
        if self.responder is not None:
            return self.responder(request)
        return GeneratedSymbol(
            image_bytes=self.image_bytes,
            output_format=self.output_format,
            declared_version=self.declared_version,
            declared_ec=self.declared_ec,
            declared_mask=self.declared_mask,
            declared_mode=self.declared_mode,
            elapsed_ms=self.elapsed_ms,
        )

    def close(self) -> None:
        """Mark the stub closed; safe to call any number of times."""
        self.close_calls += 1
        self.closed = True


class CountingAdapter(StubAdapter):
    """A stub that counts its invocations, which is how a declared count is held to.

    The count is cumulative over the whole case walk of a check and is reset only when a
    test calls :meth:`reset`, so AAP §0.4.3's declared invocation counts -- two for
    ``DT-DETERMINISTIC``, eight for ``DM-RESILIENCE``, one hundred and five for
    ``PF-ENCODE``, none for the three target-independent checks and one per case otherwise
    -- are asserted against a number a retry cannot hide: a retry registers as the declared
    count plus one.

    ``raise_on_call`` schedules a failure at a chosen invocation, so one adapter can serve a
    check that must survive a target loss after succeeding once.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Build the counting stub, deferring every option to :class:`StubAdapter`."""
        super().__init__(**kwargs)
        self._scheduled: dict[int, BaseException] = {}

    @property
    def count(self) -> int:
        """Return how many times ``generate`` was called since the last reset."""
        return len(self.requests)

    @property
    def encodes(self) -> int:
        """Return :attr:`count`; the name the case matrix's ``Encodes`` column uses."""
        return len(self.requests)

    @property
    def calls(self) -> tuple[GenerateRequest, ...]:
        """Return the recorded requests in call order."""
        return tuple(self.requests)

    def reset(self) -> None:
        """Forget every recorded request and every scheduled failure."""
        self.requests.clear()
        self._scheduled.clear()

    def raise_on_call(self, number: int, error: BaseException) -> None:
        """Schedule `error` to be raised by the `number`-th invocation, counted from one.

        Args:
            number: The 1-based invocation that should fail.
            error: The exception instance to raise.

        Raises:
            ValueError: ``number`` is not a positive integer.
        """
        if number < 1:
            raise ValueError(f"raise_on_call needs a 1-based invocation, got {number}")
        self._scheduled[number] = error

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Record `request`, then raise a scheduled failure or return the stub's answer.

        The request is recorded before the failure, so the counted invocation includes the
        one that failed -- which is what keeps a retry visible as count plus one.

        Args:
            request: The request the framework built for one case.

        Returns:
            The configured symbol for an invocation that was not scheduled to fail.

        Raises:
            BaseException: The scheduled exception, when this invocation is the one.
        """
        number = len(self.requests) + 1
        self.requests.append(request)
        failure = self._scheduled.get(number)
        if failure is not None:
            raise failure from None
        return self._produce(request)


class RaisingAdapter(StubAdapter):
    """A stub whose ``generate`` raises one of the five adapter errors.

    The error vocabulary is closed (AAP §0.3.4), so a test proving a consumer handles a
    target fault needs exactly this and no more. With ``fail_on_call`` unset every
    invocation raises; with it set, only that 1-based invocation raises and later ones
    answer normally, which is what lets a test prove a run continues past a target loss.
    """

    def __init__(
        self,
        error: BaseException,
        *,
        fail_on_call: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the raising stub around `error`.

        Args:
            error: The exception instance ``generate`` raises. One of the five adapter
                errors, though nothing here enforces that: the class is imported from the
                port and instantiated by the caller.
            fail_on_call: The 1-based invocation that raises, or ``None`` for all of them.
            **kwargs: Every :class:`StubAdapter` option.
        """
        super().__init__(**kwargs)
        self.error: BaseException = error
        self.fail_on_call: int | None = fail_on_call

    def generate(self, request: GenerateRequest) -> GeneratedSymbol:
        """Record `request` and raise, or answer once the scheduled failure has passed.

        Args:
            request: The request the framework built for one case.

        Returns:
            The configured symbol, when ``fail_on_call`` names an earlier invocation.

        Raises:
            BaseException: The configured adapter error, when this invocation is the one.
        """
        number = len(self.requests) + 1
        self.requests.append(request)
        if self.fail_on_call is None or number == self.fail_on_call:
            raise self.error from None
        return self._produce(request)


def adapter_raising(
    error: BaseException,
    *,
    fail_on_call: int | None = None,
    name: str = "raising",
) -> Adapter:
    """Return a protocol-conforming adapter whose ``generate`` raises `error`.

    A plain function as well as a fixture (:func:`error_adapter`), because a test that
    parametrises over the five adapter errors builds them inline. The error class comes
    from the port, so a renamed error breaks this module rather than every test.

    Args:
        error: The exception instance to raise; one of ``TargetUnavailable``,
            ``TargetTimeout``, ``TargetRejected``, ``TargetContractError`` or
            ``TargetUnsupported``.
        fail_on_call: The 1-based invocation that raises, or ``None`` for every one.
        name: The configured target name the stub reports.

    Returns:
        A :class:`RaisingAdapter`, which satisfies the ``Adapter`` protocol.
    """
    return RaisingAdapter(error, fail_on_call=fail_on_call, name=name)


# ---------------------------------------------------------------------------
# Decoded-evidence stubs
#
# Every check consumes one DecodeResult, so a negative case is "a valid result with exactly
# one field changed to the defect this check asserts against". The factories below build the
# valid baseline and apply the change through dataclasses.replace, keeping the fields that
# must agree with each other -- the payload and its per-decoder digests, the grid and its
# digest -- consistent whatever the caller overrides.
# ---------------------------------------------------------------------------


def _decoder_outcome(**overrides: Any) -> DecoderOutcome:
    """Return one decoder outcome, valid by default, with `overrides` applied.

    The status defaults to ``decoded`` and the payload to the stub's payload; overriding
    ``status`` to ``undetected`` or ``error`` clears the payload for that status rather than
    leaving a decoded payload beside a status that says nothing was decoded.

    Args:
        **overrides: Any :class:`~qrcheck.decode.base.DecoderOutcome` field.

    Returns:
        The constructed outcome.

    Raises:
        ValueError: An override puts a field outside its closed vocabulary, or supplies a
            ``payload_sha256`` that disagrees with ``payload``.
        TypeError: An override names a field the record does not have.
    """
    status = overrides.get("status", "decoded")
    payload = overrides.pop("payload", _STUB_PAYLOAD if status == "decoded" else None)
    fields: dict[str, Any] = {
        "decoder": DECODER_ORDER[-1],
        "status": status,
        "polarity": "original",
        "payload_sha256": None,
        "reported_ec": "M" if payload is not None else None,
        "reported_version": None,
        "error_text": None,
        "payload": payload,
    }
    fields.update(overrides)
    return DecoderOutcome(**fields)


#: The decoder-outcome factory under its public name, for the symmetry with
#: :func:`make_decode_result`; the fixture :func:`_make_decoder_outcome_fixture` returns this
#: same callable under the name ``make_decoder_outcome``.
make_decoder_outcome: Final[Callable[..., DecoderOutcome]] = _decoder_outcome


def _baseline_decode_result(payload: bytes | None = _STUB_PAYLOAD) -> DecodeResult:
    """Return a valid, self-consistent :class:`~qrcheck.decode.base.DecodeResult`.

    Both decoders are present, in the lexicographic order ``DECODER_ORDER`` fixes, and both
    agree on the payload, so ``agreement`` is ``agree``; the EC level agrees between the
    decoder's metadata and the format information, the version and mask are in range, the
    grid is a real ``Matrix`` and the digest is computed by the record itself from that grid.
    Passing ``None`` as the payload yields the single consistent alternative: neither decoder
    decoded anything, ``agreement`` is ``undetected``, and the payload-derived fields are
    ``None``.

    Args:
        payload: The payload both decoders returned, or ``None`` when neither did.

    Returns:
        The baseline result.
    """
    decoded = payload is not None
    outcomes = {
        name: _decoder_outcome(
            decoder=name,
            status="decoded" if decoded else "undetected",
            payload=payload,
            reported_ec="M" if decoded else None,
        )
        for name in DECODER_ORDER
    }
    return DecodeResult(
        payload_bytes=payload,
        text=_payload_text(payload),
        symbology_identifier="]Q1" if decoded else None,
        content_type="Text" if decoded else None,
        ec_level_decoder="M" if decoded else None,
        ec_level_format="M" if decoded else None,
        version=2 if decoded else None,
        mask=3 if decoded else None,
        matrix=_stub_matrix(2) if decoded else None,
        matrix_sha256=None,
        per_decoder=outcomes,
        agreement="agree" if decoded else "undetected",
        polarity_applied="original",
        rejection=None,
        metrics={},
    )


def make_decode_result(**overrides: Any) -> DecodeResult:
    """Return a valid decoded result with `overrides` applied to exactly those fields.

    The negative suite's whole mechanism: flip one field to the defect a check asserts
    against -- a wrong ``ec_level_format``, a payload differing in one byte, ``mask=9``,
    ``agreement="disagree"``, an undetected primary -- and the check under test must fail.

    Two consistency guarantees hold whatever is overridden. Overriding ``payload_bytes``
    rebuilds the per-decoder digests from the new payload, so no outcome names bytes that
    are not the payload. Overriding ``matrix`` without naming ``matrix_sha256`` clears the
    digest, so the record recomputes it from the new grid instead of refusing the pair.

    Args:
        **overrides: Any :class:`~qrcheck.decode.base.DecodeResult` field.

    Returns:
        The constructed result.

    Raises:
        ValueError: An override puts ``agreement`` or ``polarity_applied`` outside its
            vocabulary, names an unknown decoder in ``per_decoder``, or supplies a digest
            that disagrees with the bytes or the grid beside it.
        TypeError: An override names a field the record does not have.
    """
    payload = overrides.pop("payload_bytes", _STUB_PAYLOAD)
    baseline = _baseline_decode_result(payload)
    if "matrix" in overrides and "matrix_sha256" not in overrides:
        overrides["matrix_sha256"] = None
    return dataclasses.replace(baseline, **overrides)


class _DecodePatcher:
    """Installs a stubbed decode entry in front of the checks, and controls what it returns.

    The patch targets are the bindings the checks actually resolve. ``qrcheck/checks/support.py``
    imports ``qrcheck.decode.decode_symbol`` inside its own ``decode()`` call, so patching
    that module attribute reaches every family that goes through the shared helper; a family
    that imported the entry itself holds its own attribute, which is why every module in the
    checks package that carries a ``decode_symbol`` attribute is patched too.

    ``qrcheck.decode.consensus.decode_symbol`` is deliberately **not** patched: the captured
    real function delegates to it, so leaving it intact is what keeps :func:`real_decode`
    working while a patch is installed.

    The instance is callable (:meth:`__call__`) and its :attr:`result` is assignable, so a
    test may either install a result inline or swap it between cases of one walk.
    """

    def __init__(self, result: DecodeResult) -> None:
        """Store the result the stub returns and start an empty call log.

        Args:
            result: The decoded evidence every stubbed call returns.
        """
        self._result: DecodeResult = result
        self.calls: list[tuple[bytes, GenerateRequest]] = []

    @property
    def result(self) -> DecodeResult:
        """Return the result the stub currently returns."""
        return self._result

    @result.setter
    def result(self, value: DecodeResult) -> None:
        """Replace the result every later stubbed call returns."""
        self._result = value

    def set(self, result: DecodeResult) -> None:
        """Replace the result, for a caller that prefers a method to the property."""
        self._result = result

    def __call__(self, result: DecodeResult) -> _DecodePatcher:
        """Install `result` and return this patcher, so a call can be an expression."""
        self._result = result
        return self

    @property
    def requests(self) -> tuple[GenerateRequest, ...]:
        """Return the requests the stubbed calls were made for, in call order."""
        return tuple(request for _, request in self.calls)

    @property
    def count(self) -> int:
        """Return how many stubbed decode calls were made."""
        return len(self.calls)

    @property
    def real(self) -> Callable[..., DecodeResult]:
        """Return the real decode entry, callable even while this patch is installed."""
        return _REAL_DECODE_SYMBOL

    def stub(
        self,
        image_bytes: bytes,
        request: GenerateRequest,
        *,
        secondary: bool | None = None,
    ) -> DecodeResult:
        """Stand in for ``qrcheck.decode.decode_symbol``: log the call, return the result.

        The signature is the real entry's, including the keyword-only ``secondary``, so a
        caller cannot tell the stub from the implementation by how it calls it. ``secondary``
        is accepted and ignored: the canned result already records both decoders.

        Args:
            image_bytes: The artefact the caller wanted decoded; recorded, not read.
            request: The request the artefact was produced for; recorded.
            secondary: Single-decoder-mode selection, ignored.

        Returns:
            The result the patcher currently holds.
        """
        del secondary
        self.calls.append((image_bytes, request))
        return self._result


def _decode_patch_targets() -> tuple[tuple[Any, str], ...]:
    """Return the modules, and the attribute on each, that :func:`patch_decode` replaces.

    ``qrcheck.decode`` is always patched. Every module of the checks package that already
    holds a ``decode_symbol`` attribute is patched as well, because a module-level
    ``from qrcheck.decode import decode_symbol`` binds a name of its own that patching the
    source module would leave untouched. The package is imported first so the walk sees it.

    Returns:
        One ``(module, attribute)`` pair per patch site, in a stable order.
    """
    targets: list[tuple[Any, str]] = [
        (importlib.import_module("qrcheck.decode"), _DECODE_ATTRIBUTE)
    ]
    for name in sorted(sys.modules):
        if name != _CHECKS_PACKAGE and not name.startswith(f"{_CHECKS_PACKAGE}."):
            continue
        module = sys.modules.get(name)
        if module is not None and getattr(module, _DECODE_ATTRIBUTE, None) is not None:
            targets.append((module, _DECODE_ATTRIBUTE))
    return tuple(targets)


# ---------------------------------------------------------------------------
# Result-model factories and the determinism comparator
# ---------------------------------------------------------------------------


def make_case_result(**overrides: Any) -> CaseResult:
    """Return a valid :class:`~qrcheck.report.model.CaseResult`, passed by default.

    The status defaults to ``passed`` with no reason and no evidence directory, which is
    the only shape the model accepts for a pass. A non-passing status supplied without a
    reason gets :attr:`~qrcheck.report.model.Reason.VALUE_MISMATCH`, so no override can
    produce the pair the model refuses. Every other field is overridable, including the
    two the model validates against each other.

    Args:
        **overrides: Any ``CaseResult`` field.

    Returns:
        The constructed result.

    Raises:
        ValueError: An override breaks one of the model's invariants -- a reason or an
            evidence directory on a pass, a missing reason on any other status, a
            non-mapping request, a negative or non-finite duration, an absolute
            evidence directory.
        TypeError: An override names a field the record does not have.
    """
    status = Status(overrides.get("status", Status.PASSED))
    fields: dict[str, Any] = {
        "check_id": "RT-ROUNDTRIP",
        "case_id": "corpus:url-000",
        "status": status,
        "reason": None,
        "duration_ms": 0.0,
        "request": {},
        "decoded": {},
        "expected": None,
        "evidence_dir": None,
        "seed": int(_schema_default("", "seed")),
    }
    if status is not Status.PASSED:
        fields["reason"] = Reason.VALUE_MISMATCH
    fields.update(overrides)
    return CaseResult(**fields)


def make_run_result(results: Sequence[CaseResult] = (), **overrides: Any) -> RunResult:
    """Return a valid :class:`~qrcheck.report.model.RunResult` over `results`.

    The identity is a content hash over the run's parameters and its cases, never a UUID or
    a timestamp, so two calls with the same arguments agree. The tallies are computed from
    `results`, which is what the model re-derives and compares. The timestamps are the
    module's fixed constants, so a test that wants a difference passes its own -- and the
    pair a determinism test compares differs only in the fields :func:`mask_volatile` masks.

    Args:
        results: The case results the run holds, in any order; the model sorts them.
        **overrides: Any other ``RunResult`` field.

    Returns:
        The constructed run result.

    Raises:
        ValueError: An override supplies a run identity, timestamp pair or tally the model
            refuses, or two of `results` share a case identity.
        TypeError: An override names a field the record does not have.
    """
    ordered = tuple(results)
    adapter = str(overrides.get("adapter", "reference"))
    seed = int(overrides.get("seed", _schema_default("", "seed")))
    ec_level = str(overrides.get("ec_level", "M"))
    fields: dict[str, Any] = {
        "run_id": _content_hash(
            adapter,
            seed,
            ec_level,
            *(result.check_id for result in ordered),
            *(result.case_id for result in ordered),
        ),
        "started_at": _STUB_STARTED_AT,
        "finished_at": _STUB_FINISHED_AT,
        "adapter": adapter,
        "seed": seed,
        "ec_level": ec_level,
        "loc_count": 0,
        "counts": Counts.from_results(ordered),
        "results": ordered,
    }
    fields.update(overrides)
    return RunResult(**fields)


def mask_volatile(value: RunResult | Mapping[str, Any]) -> dict[str, Any]:
    """Return a summary-shaped copy with exactly the volatile fields masked.

    The three volatile fields are the complete inventory AAP §0.5.4 names --
    ``started_at``, ``finished_at`` and every ``results[].duration_ms`` -- and nothing else
    is touched: every other key keeps its value and every list keeps its order, so two runs
    compare equal only when they agree about everything reproducible.

    A :class:`~qrcheck.report.model.RunResult` is projected through ``as_dict()`` first,
    because a masked timestamp is not a timestamp and would not satisfy the record's own
    validation; the return value is therefore always the mapping ``summary.json`` is built
    from, which is also what a writer test has in hand.

    Args:
        value: A run result, or the mapping a summary writer produced from one.

    Returns:
        A new mapping: the two run timestamps replaced by a non-timestamp sentinel, and each
        result's ``duration_ms`` replaced by zero. The argument is never mutated.
    """
    summary = copy.deepcopy(value.as_dict() if isinstance(value, RunResult) else dict(value))
    for key in _RUN_VOLATILE_FIELDS:
        if key in summary:
            summary[key] = _MASKED_TIMESTAMP
    results = summary.get("results")
    if isinstance(results, list):
        masked: list[Any] = []
        for entry in results:
            if isinstance(entry, Mapping):
                item = dict(entry)
                if _RESULT_VOLATILE_FIELD in item:
                    item[_RESULT_VOLATILE_FIELD] = _MASKED_DURATION_MS
                masked.append(item)
            else:
                masked.append(entry)
        summary["results"] = masked
    return summary


# ---------------------------------------------------------------------------
# The offline guard
# ---------------------------------------------------------------------------


class OutboundNetworkBlocked(RuntimeError):
    """A test tried to connect to a destination that is not loopback.

    Raised by the autouse :func:`no_outbound_network` guard. A ``RuntimeError`` rather than
    an ``OSError`` on purpose: a client library may translate an ``OSError`` into its own
    connection error and report a merely-unreachable host, and the failure this guard
    reports must stay recognisable as the product's offline contract being broken.
    """


#: Host names that resolve to this machine. ``127.0.0.0/8``, ``::1`` and the loopback name
#: are the product's whole allow-list (``qrcheck.config.models.LOOPBACK_HOSTS``); the
#: address check below covers the rest.
_LOOPBACK_HOSTNAMES: Final[frozenset[str]] = frozenset({"localhost"})


def _is_loopback_destination(address: object) -> bool:
    """Return whether a socket address names a loopback destination.

    ``127.0.0.0/8`` in any spelling, ``::1`` and the ``localhost`` name are loopback; a
    non-tuple address is an ``AF_UNIX`` path and therefore local IPC rather than a network
    destination; anything the classifier cannot read is refused, because the safe direction
    for a guard is to block.

    Args:
        address: The address a ``connect`` call received.

    Returns:
        ``True`` to let the call through, ``False`` to block it.
    """
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    text = host.strip().strip("[]").lower()
    if not text:
        return False
    if text in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False

    return grid



# ---------------------------------------------------------------------------
# The offline guard (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block every socket connection to a destination outside loopback, for every test.

    Enforces the product's fully-offline MUST (AAP §0.12: no host other than
    ``127.0.0.1``/``localhost``/``::1``, no paid service, no cloud credential, no
    telemetry). A stray outbound call fails loudly as :class:`OutboundNetworkBlocked`
    instead of silently depending on a network that a CI runner may not have -- and instead
    of passing locally and failing there.

    Loopback is untouched, because the HTTP adapter's integration test drives a real server
    on ``127.0.0.1`` and the connect must succeed there. ``create_connection`` needs no
    separate patch: it funnels through ``socket.connect``, which is the call that would put
    a packet on the wire. ``connect_ex`` is patched too, since it is the non-raising spelling
    of the same operation and an unguarded path through it would let a call slip out.

    Args:
        monkeypatch: The fixture that unwinds the patches when the test ends.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def blocked(sock: socket.socket, address: object) -> OutboundNetworkBlocked:
        return OutboundNetworkBlocked(
            "test attempted an outbound network connection to "
            f"{address!r} from {sock.family!s}; the framework is offline and "
            "loopback-only (127.0.0.0/8, ::1, localhost)"
        )

    def guarded_connect(sock: socket.socket, address: object) -> None:
        if not _is_loopback_destination(address):
            raise blocked(sock, address)
        real_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if not _is_loopback_destination(address):
            raise blocked(sock, address)
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


# ---------------------------------------------------------------------------
# Report directories and configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    """Return a fresh, existing, writable report root unique to the test.

    The directory is created but its ``evidence/`` subtree is not: the store owns that
    layout -- ``<report_dir>/evidence/<check-id>/<failure-id>/`` -- and a fixture that
    pre-created it would let a test assert a directory the code never made.

    Args:
        tmp_path: The per-test temporary directory.

    Returns:
        ``<tmp_path>/reports``, existing and empty.
    """
    path = tmp_path / str(_schema_default("", "report_dir"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def config_factory(tmp_path: Path, report_dir: Path) -> Callable[..., models.Config]:
    """Return a callable building a fully populated configuration **in memory**.

    Nothing here opens ``qrcheck.toml``: every value comes from
    :data:`qrcheck.config.schema.SCHEMA`'s declared defaults, splatted into the frozen
    records of :mod:`qrcheck.config.models`, so a test that retunes nothing agrees with the
    committed document's fallbacks and a test that retunes one value states only that value.
    The report directory is the test's, the corpus path is the committed corpus, and the two
    built-in self-check targets are both declared so ``default_adapter`` can be switched
    between them.

    The returned callable takes keyword overrides that replace single top-level fields --
    ``seed``, ``default_adapter``, ``report_dir``, ``targets``, ``performance`` -- and
    refuses a name that is not one of them, so a mistyped override fails instead of being
    silently ignored.

    Args:
        tmp_path: The per-test temporary directory, used for the record's own config path.
        report_dir: The test's report root.

    Returns:
        A callable ``build(**overrides) -> Config``.

    Raises:
        TypeError: The callable is given an override naming no ``Config`` field.
    """
    config_path = (tmp_path / _CONFIG_FILE_NAME).resolve()
    corpus_path = (_REPO_ROOT / str(_schema_default("", "corpus_path"))).resolve()
    defaults: dict[str, Any] = {
        "path": config_path,
        "default_adapter": str(_schema_default("", "default_adapter")),
        "report_dir": Path(report_dir),
        "corpus_path": corpus_path,
        "seed": int(_schema_default("", "seed")),
        "loc_limit": int(_schema_default("", "loc_limit")),
        "targets": _default_targets(),
        "render": models.RenderConfig(**_group_defaults("render")),
        "performance": models.PerformanceLimits(**_group_defaults("checks.performance")),
        "damage": models.DamageLimits(**_group_defaults("checks.damage")),
    }

    def build(**overrides: Any) -> models.Config:
        unknown = sorted(set(overrides) - set(defaults))
        if unknown:
            raise TypeError(f"config_factory got unknown override(s) {unknown}")
        fields = dict(defaults)
        fields.update(overrides)
        return models.Config(**fields)

    return build


@pytest.fixture
def config(config_factory: Callable[..., models.Config]) -> models.Config:
    """Return the default in-memory configuration, for a test that needs only a valid one.

    Args:
        config_factory: The factory fixture.

    Returns:
        A :class:`~qrcheck.config.models.Config` carrying every declared default.
    """
    return config_factory()


@pytest.fixture
def toml_config_file(tmp_path: Path) -> Callable[..., Path]:
    """Return a writer that puts a configuration document under the test's ``tmp_path``.

    The document is minimal and valid by default -- the five top-level keys plus
    ``[targets.reference]``, every value taken from the schema's declared defaults -- and
    its ``report_dir`` and ``corpus_path`` are written **relative** so that
    ``qrcheck/config/load.py``, which resolves paths against the configuration file's own
    directory, keeps every artefact of the run inside ``tmp_path``. The committed corpus is
    copied to the same relative location for the same reason: a CLI test must neither depend
    on nor mutate the repository file.

    Passing ``text`` writes that document verbatim instead, which is how a test builds a
    deliberately invalid one (an unknown key, a non-loopback ``url``) without this module
    carrying a catalogue of broken documents.

    Args:
        tmp_path: The per-test temporary directory that becomes the config file's base.

    Returns:
        A callable ``write(text=None, *, name="qrcheck.toml", **overrides) -> Path``.
    """
    corpus_relative = Path(str(_schema_default("", "corpus_path")))

    def write(
        text: str | None = None,
        *,
        name: str = _CONFIG_FILE_NAME,
        **overrides: Any,
    ) -> Path:
        corpus_target = tmp_path / corpus_relative
        if not corpus_target.exists():
            corpus_target.parent.mkdir(parents=True, exist_ok=True)
            corpus_target.write_bytes((_REPO_ROOT / corpus_relative).read_bytes())
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        document = _minimal_config_text(**overrides) if text is None else text
        path.write_text(document, encoding="utf-8")
        return path

    return write


@pytest.fixture
def config_file(toml_config_file: Callable[..., Path]) -> Path:
    """Return the path of the default minimal ``qrcheck.toml`` written under ``tmp_path``.

    Args:
        toml_config_file: The writer fixture.

    Returns:
        The configuration file's path, for ``--config``.
    """
    return toml_config_file()


# ---------------------------------------------------------------------------
# Target and decode stubs
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_adapter() -> StubAdapter:
    """Return a protocol-conforming target stub declaring all thirteen capabilities.

    Returns:
        A fresh :class:`StubAdapter` with an empty request log.
    """
    return StubAdapter()


@pytest.fixture
def counting_adapter() -> CountingAdapter:
    """Return a target stub that counts its invocations and can fail on a chosen one.

    Returns:
        A fresh :class:`CountingAdapter` with a zero count.
    """
    return CountingAdapter()


@pytest.fixture
def error_adapter() -> Callable[..., Adapter]:
    """Return :func:`adapter_raising`, the factory of always-or-once failing adapters.

    Returns:
        The factory, so a test builds one stub per adapter error it exercises.
    """
    return adapter_raising


@pytest.fixture
def target_errors() -> tuple[BaseException, ...]:
    """Return one instance of each of the five adapter errors, in the documented order.

    Constructed from the real classes, so this fixture is the one place the closed
    five-error vocabulary is enumerated for tests: a renamed or removed error breaks this
    module rather than the tests that assume it. The order is the order of the error table
    in AAP §0.3.4.

    Returns:
        ``TargetUnavailable``, ``TargetTimeout``, ``TargetRejected``,
        ``TargetContractError`` (carrying a body prefix) and ``TargetUnsupported``.
    """
    return (
        TargetUnavailable("cannot spawn target process"),
        TargetTimeout("no response within timeout_s"),
        TargetRejected("declared rejection: payload rejected by the target"),
        TargetContractError("response body is not an image", body_prefix=b"\x00\x01\x02"),
        TargetUnsupported("target does not declare capability 'svg'"),
    )


@pytest.fixture(name="make_decode_result")
def _make_decode_result_fixture() -> Callable[..., DecodeResult]:
    """Return :func:`make_decode_result`, requestable as a fixture.

    Returns:
        The decoded-result factory.
    """
    return make_decode_result


@pytest.fixture(name="make_decoder_outcome")
def _make_decoder_outcome_fixture() -> Callable[..., DecoderOutcome]:
    """Return the decoder-outcome factory, requestable as a fixture.

    Returns:
        A callable ``(**overrides) -> DecoderOutcome``.
    """
    return make_decoder_outcome


@pytest.fixture
def baseline_decode_result() -> DecodeResult:
    """Return the unmodified valid decoded result the factories start from.

    A test that wants to change several fields, or to compare a stubbed result against the
    baseline, requests this instead of calling the factory with no arguments.

    Returns:
        The baseline :class:`~qrcheck.decode.base.DecodeResult`.
    """
    return _baseline_decode_result()


@pytest.fixture
def patch_decode(
    monkeypatch: pytest.MonkeyPatch,
    baseline_decode_result: DecodeResult,
) -> _DecodePatcher:
    """Install a stubbed decode entry in front of the checks, without a target or an image.

    This is how a check's negative case is driven: the check calls the decode entry it always
    calls, and receives the canned :class:`~qrcheck.decode.base.DecodeResult` instead of a
    real one. The patch covers ``qrcheck.decode.decode_symbol`` -- the binding
    ``qrcheck/checks/support.py`` imports inside its own ``decode()``, so every family that
    goes through the shared helper is covered -- plus, for a family that imported the entry
    itself, the ``decode_symbol`` attribute of that checks module.

    ``qrcheck.decode.consensus.decode_symbol`` is not patched, which is what keeps
    :func:`real_decode` usable side by side with the stub.

    The returned patcher is callable (``patch_decode(result)`` installs and returns it), its
    ``result`` is assignable, and it logs every stubbed call, so a test can walk a whole
    case matrix and change the result per case. ``monkeypatch`` unwinds the patch at the end
    of the test.

    Args:
        monkeypatch: The fixture that owns the patch's lifetime.
        baseline_decode_result: The result the stub returns until a test replaces it.

    Returns:
        The installed :class:`_DecodePatcher`.
    """
    patcher = _DecodePatcher(baseline_decode_result)
    for module, attribute in _decode_patch_targets():
        monkeypatch.setattr(module, attribute, patcher.stub, raising=True)
    return patcher


@pytest.fixture
def real_decode() -> Callable[..., DecodeResult]:
    """Return the real decode entry, callable even while :func:`patch_decode` is installed.

    The function object is captured when this module is imported, and the patch never
    touches ``qrcheck.decode.consensus``, which the captured function delegates to -- so a
    test can compare real and stubbed behaviour in one process.

    Returns:
        ``qrcheck.decode.decode_symbol`` as it was before any patch.
    """
    return _REAL_DECODE_SYMBOL


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def committed_corpus() -> tuple[CorpusRecord, ...]:
    """Return the committed corpus, read through its canonical reader, in ``id`` order.

    The file is the repository's ``corpus/default.jsonl`` (AAP §0.9.2: a repository file,
    not package data), parsed by ``qrcheck.corpus.io.load_corpus`` so the JSONL contract --
    sorted keys, LF, one trailing newline -- is that module's and not a second parser's.

    Returns:
        The 240 records, sorted by ``id`` ascending.
    """
    path = _REPO_ROOT / str(_schema_default("", "corpus_path"))
    return tuple(sorted(load_corpus(path), key=lambda record: record.id))


@pytest.fixture(scope="session")
def built_corpus() -> tuple[CorpusRecord, ...]:
    """Return the corpus the builder produces in memory from the default seed.

    For a test that needs records without the committed file -- proving regeneration, or
    driving a check on records the builder just made.

    Returns:
        The records built from ``models.DEFAULT_SEED``, sorted by ``id`` ascending.
    """
    seed = int(_schema_default("", "seed"))
    return tuple(sorted(build_corpus(seed), key=lambda record: record.id))


@pytest.fixture
def corpus_subset(committed_corpus: tuple[CorpusRecord, ...]) -> tuple[CorpusRecord, ...]:
    """Return a small, fixed slice of the committed corpus for millisecond-cost unit tests.

    A fixed rule -- the first records in ``id`` order -- never a random sample, because a
    random one would stop the suite being deterministic (AAP §0.5).

    Args:
        committed_corpus: The corpus to slice.

    Returns:
        The first ``_SUBSET_SIZE`` records in ``id`` order.
    """
    return committed_corpus[:_SUBSET_SIZE]


@pytest.fixture(scope="session")
def boundary_corpus(committed_corpus: tuple[CorpusRecord, ...]) -> tuple[CorpusRecord, ...]:
    """Return the boundary classes of the committed corpus: the 40-record subset.

    The same view success criterion 2 runs at EC levels L, Q and H is invoked with on the
    command line (``--class 'boundary-*'``), produced through
    ``qrcheck.corpus.io.filter_by_class`` rather than by a second filter.

    Args:
        committed_corpus: The corpus to filter.

    Returns:
        The records whose class matches ``boundary-*``, in ``id`` order.
    """
    return filter_by_class(committed_corpus, ["boundary-*"])


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@pytest.fixture
def make_run_context(
    config: models.Config,
    corpus_subset: tuple[CorpusRecord, ...],
    stub_adapter: StubAdapter,
    report_dir: Path,
) -> Callable[..., RunContext]:
    """Return a callable building the :class:`~qrcheck.registry.check.RunContext` a check sees.

    Every field a check's ``cases(ctx)`` may read is supplied: the adapter (the port, never
    a concrete adapter), the resolved configuration that owns every threshold, the filtered
    corpus, the EC level and the seed, the three dependency-availability facts, and an
    evidence store rooted in the report directory. The defaults are the fixtures above, so
    ``make_run_context()`` is a complete context, and each keyword replaces one field.

    Availability defaults to all three dependencies present -- primary decoder, secondary
    decoder and SVG rasterizer -- because a missing one turns the affected checks into
    ``skipped`` results, which would hide the assertion a test meant to exercise. A test
    modelling a degradation passes ``availability={"secondary-decoder": False}``; an absent
    token reads as unavailable, exactly as the context's own contract states.

    Args:
        config: The default configuration.
        corpus_subset: The default corpus.
        stub_adapter: The default adapter.
        report_dir: The report root the evidence store writes under.

    Returns:
        A callable ``build(*, adapter, config, corpus, ec_level, seed, availability,
        evidence) -> RunContext``.
    """
    default_config = config
    default_corpus = corpus_subset
    default_adapter = stub_adapter
    default_evidence = EvidenceStore(report_dir)
    default_availability = MappingProxyType(
        {
            DEPENDENCY_PRIMARY_DECODER: True,
            DEPENDENCY_SECONDARY_DECODER: True,
            DEPENDENCY_SVG_RASTERIZER: True,
        }
    )

    def build(
        *,
        adapter: Adapter | None = None,
        config: models.Config | None = None,
        corpus: Sequence[CorpusRecord] | None = None,
        ec_level: str = "M",
        seed: int | None = None,
        availability: Mapping[str, bool] | None = None,
        evidence: EvidenceStore | None = None,
    ) -> RunContext:
        resolved_config = default_config if config is None else config
        return RunContext(
            adapter=default_adapter if adapter is None else adapter,
            config=resolved_config,
            corpus=tuple(default_corpus if corpus is None else corpus),
            ec_level=ec_level,
            seed=int(resolved_config.seed if seed is None else seed),
            availability=default_availability
            if availability is None
            else MappingProxyType(dict(availability)),
            evidence=default_evidence if evidence is None else evidence,
        )

    return build


@pytest.fixture
def run_context(make_run_context: Callable[..., RunContext]) -> RunContext:
    """Return the default run context, for a test that needs only a usable one.

    Args:
        make_run_context: The context factory fixture.

    Returns:
        A :class:`~qrcheck.registry.check.RunContext` over the stub adapter, the default
        configuration and the corpus subset.
    """
    return make_run_context()


# ---------------------------------------------------------------------------
# Result-model factories and the determinism comparator, as fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="make_case_result")
def _make_case_result_fixture() -> Callable[..., CaseResult]:
    """Return :func:`make_case_result`, requestable as a fixture.

    Returns:
        The case-result factory.
    """
    return make_case_result


@pytest.fixture(name="make_run_result")
def _make_run_result_fixture() -> Callable[..., RunResult]:
    """Return :func:`make_run_result`, requestable as a fixture.

    Returns:
        The run-result factory.
    """
    return make_run_result


@pytest.fixture(name="mask_volatile")
def _mask_volatile_fixture() -> Callable[[RunResult | Mapping[str, Any]], dict[str, Any]]:
    """Return :func:`mask_volatile`, requestable as a fixture.

    Returns:
        The comparator that masks exactly the three volatile fields of a run.
    """
    return mask_volatile


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _cli_group() -> Any:
    """Return the click object the ``qrcheck`` console script is built from.

    Imported on use rather than at module import, so a test session that never touches the
    command line does not need the whole CLI chain resolved, and so the import error of an
    incomplete installation names the module that is missing. ``cli`` is preferred over
    ``main``: AAP §0.8.1 names the group ``cli`` and the entry point ``main``, and a group is
    what ``CliRunner`` invokes.

    Returns:
        The click group.

    Raises:
        RuntimeError: ``qrcheck.cli.main`` defines neither name, which is a defect in the
            CLI rather than a test problem.
    """
    module = importlib.import_module("qrcheck.cli.main")
    for attribute in ("cli", "main"):
        candidate = getattr(module, attribute, None)
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "qrcheck.cli.main defines neither 'cli' nor 'main', so the command line "
        "cannot be invoked"
    )


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a ``CliRunner`` that keeps the two streams apart.

    ``mix_stderr=False`` so a test can assert on ``result.stderr`` -- several exit-code and
    selection conditions are reported there and nowhere else (AAP §0.6.6 rows 1, 2, 10) --
    and so ``result.stdout`` holds only what the command printed on stdout, which is what the
    JSON-lines contract on an unwritable report directory is asserted against.

    Returns:
        A fresh runner.
    """
    return CliRunner(mix_stderr=False)


@pytest.fixture
def invoke_cli(
    cli_runner: CliRunner,
    toml_config_file: Callable[..., Path],
) -> Callable[..., Any]:
    """Return a callable invoking a ``qrcheck`` subcommand through a real ``--config`` path.

    The configuration is the minimal valid document written under ``tmp_path`` unless the
    test passes its own, so a CLI test never reads, mutates or depends on the committed
    ``qrcheck.toml``. Exceptions are **not** caught: the subcommands are supposed to return
    exit codes, and a click exception escaping is itself a finding, so it reaches the test as
    a failure rather than being turned into an exit code 1 the test might mistake for the
    behaviour it was checking. No exit code is asserted here either -- each test asserts the
    one its case requires.

    Args:
        cli_runner: The runner the invocation goes through.
        toml_config_file: The writer that supplies the default ``--config`` path.

    Returns:
        A callable ``invoke(*args, config=None, env=None, input=None) -> Result``.
    """

    def invoke(
        *args: str,
        config: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        input: str | None = None,
    ) -> Any:
        config_path = toml_config_file() if config is None else Path(config)
        argv = ["--config", str(config_path), *(str(argument) for argument in args)]
        return cli_runner.invoke(
            _cli_group(),
            argv,
            catch_exceptions=False,
            env=None if env is None else dict(env),
            input=input,
        )

    return invoke


# ---------------------------------------------------------------------------
# An unusable report directory
# ---------------------------------------------------------------------------


@pytest.fixture
def unwritable_report_dir(tmp_path: Path) -> Path:
    """Return a path a report directory can never be, whatever user the tests run as.

    A **regular file** stands where the directory should be, so creating the directory fails
    for root as well: ``chmod 0o500`` proves nothing in a container that runs as root, while
    ``mkdir`` over an existing file raises ``FileExistsError`` (an ``OSError``) for every
    user. That is the condition AAP §0.6.6 row 10 requires exit code 3 for, with every line
    the command writes on stdout still parsing as a JSON object.

    Args:
        tmp_path: The per-test temporary directory.

    Returns:
        The path of the existing regular file to use as ``report_dir``.
    """
    path = tmp_path / "report-dir-is-a-regular-file"
    path.write_text("this path is a file, not a directory\n", encoding="utf-8")
    return path
