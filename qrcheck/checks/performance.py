"""The two performance checks: encode latency with throughput, and decode latency.

Plan item served (AAP §0.8.1, the component-to-check-family table line
``checks/performance.py`` -> ``PF-ENCODE``, ``PF-DECODE``), together with the case-matrix
rows it fixes -- 1 synthetic case, **105 encodes** and 0 decodes for ``PF-ENCODE``; 1
synthetic case, **0 encodes** and **205 decodes** of one fixed 500 x 500 PNG for
``PF-DECODE`` -- and the numeric budgets of AAP §0.6.2 with the environment they are
defined against in §0.6.3 (CPython 3.11, Linux x86_64, 2 vCPU, the pinned lock).

``PF-ENCODE`` and ``PF-DECODE`` are the **only** two checks in the product permitted to
assert on a duration (AAP §0.12).  Every other case's ``duration_ms`` is recorded and
never compared: timing is environment-bound, so a duration that reached any other
assertion would make the suite flaky by construction (AAP §0.5.5).  That monopoly is why
this module reads a monotonic clock at all, and why it never reads a wall clock -- a
``perf_counter()`` difference can only produce a duration, never a date.

Every threshold comes from the configuration
--------------------------------------------
This file carries no threshold, sample size or warm-up count as a literal.  All seven
numbers -- ``encode_p95_ms``, ``decode_p95_ms``, ``min_symbols_per_minute``,
``encode_samples``, ``decode_samples``, ``warmup`` and ``decode_image_px`` -- are read
from :attr:`qrcheck.config.models.Config.performance`, the ``[checks.performance]`` table
of the committed ``qrcheck.toml`` (AAP §0.6.2, §0.12), and each threshold is copied into
the result's ``expected`` so the report proves which budget the case was measured
against.  Changing a floor therefore means changing one declared default and the
acceptance criterion that reads it, never editing a check.

What each check measures
------------------------
``PF-ENCODE`` asks the target for one symbol of a declared fixed payload ``warmup`` times
without measuring, then ``encode_samples`` times measuring each call with
:func:`time.perf_counter`.  It asserts the 95th-percentile encode duration against
``encode_p95_ms`` and the measured throughput against ``min_symbols_per_minute``.  The
invocation count is therefore exactly ``warmup + encode_samples`` (5 + 100 = 105 under
the committed configuration), and :data:`EXPECTED_ENCODES` declares it: a retry would
register as one more invocation and fail the counting-stub test rather than improve a
figure (AAP §0.4.3, §0.5.6).

``PF-DECODE`` draws its probe image **inside the framework**, with no adapter call at
all: it encodes a declared fixed payload with segno at the run's error level, scales the
module grid by nearest neighbour to exactly ``decode_image_px`` pixels square, and
encodes the PNG through :mod:`qrcheck.evidence.mux`.  That is why the check declares no
capability -- it must be runnable against a target that cannot encode anything at all,
including one that is unreachable -- and why its own declared count is zero encodes and
``warmup + decode_samples`` (5 + 200 = 205) primary decodes of that one PNG:
:data:`EXPECTED_DECODES` declares that half.  The first decode happens before any sample
is timed and must return the payload the probe was drawn from; a probe that cannot decode
would make the measurement meaningless, so that condition fails the case instead of
timing an unreadable image.

The percentile rule
-------------------
``encode_p95_ms`` and ``decode_p95_ms`` are the **nearest-rank** percentile of the sample
durations: the samples are sorted ascending and the reported value is the one at index
``ceil(0.95 * n) - 1`` (the 95th of 100 sorted samples, the 190th of 200).  It is stated
here because an undocumented percentile is not reproducible -- linear interpolation,
inverted CDFs and nearest-rank agree on no two data sets -- and because the rule must not
change silently between two runs whose figures are compared by a reader.  A slow outlier
becomes a reported p95; it is never re-run, and no sample is discarded.

Determinism, and the one honest exception
-----------------------------------------
Nothing here reads ``random``, ``hash()``, ``id()``, the environment or a CPU model: the
payloads and the probe grid are declared constants, so two runs draw the same image and
compare the same budgets.  A **passing** case writes no evidence at all (the persistence
policy of :mod:`qrcheck.report.evidence`), so the byte-identity promise for
``request.json`` and ``decoded.json`` is untouched for every other case.  A *failing* PF
case necessarily files the measured figures in its ``decoded.json``: the measurement is
the finding, and a failure whose evidence omitted the numbers would be unreadable.

Reasons this module emits
-------------------------
:attr:`qrcheck.report.model.Reason.VALUE_MISMATCH` when a floor is breached or the probe
image does not decode, with the measured value and the configured threshold named in
``decoded["message"]``.  Adapter failures travel through
:func:`qrcheck.checks.support.case_status_for_error`, which reports a declared rejection
as ``target_rejected`` and an unusable response as ``contract_error``;
:class:`~qrcheck.targets.base.TargetUnavailable` and
:class:`~qrcheck.targets.base.TargetTimeout` are deliberately **not** caught here -- the
runner owns the one-error-result-per-check policy on target loss (AAP §0.6.6 row 6), so
this module lets both propagate unchanged.

Import cost and layering
------------------------
This module imports the standard library, :mod:`qrcheck.checks.support`, the registry's
case and context types, the result vocabulary, the configuration record and the adapter
port's capability type -- all at module scope, because it is a check family and the
registry is expected to hold it.  The three heavy dependencies (``segno``, ``numpy`` and
:mod:`qrcheck.evidence.mux`, which is the product's only Pillow importer) are imported
inside :func:`_probe_png`, at the point of use, so ``qrcheck list`` does not pay for an
encoder and an image library to print two titles.  Nothing here imports a concrete
adapter or a decoder family: the port is the only adapter-side name in the file.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from qrcheck.checks import support
from qrcheck.config.models import PerformanceLimits
from qrcheck.registry.check import (
    DEPENDENCY_PRIMARY_DECODER,
    Case,
    RunContext,
    check,
)
from qrcheck.report.model import CaseResult, Reason
from qrcheck.targets.base import (
    Capability,
    GenerateRequest,
    GeneratedSymbol,
    TargetContractError,
    TargetRejected,
    TargetUnsupported,
)

if TYPE_CHECKING:  # Imported for type checking only; support.decode returns this type.
    from qrcheck.decode import DecodeResult

__all__ = [
    "EXPECTED_DECODES",
    "EXPECTED_ENCODES",
    "pf_decode_cases",
    "pf_encode_cases",
]

# The two identifiers, spelled once each: the registry validates them against its own
# closed tuple, the case identifiers below are built from the same literals, and
# EXPECTED_* is keyed on them, so a typo cannot register a check under a name the
# accounting table does not describe.
_ENCODE_CHECK_ID: Final[str] = "PF-ENCODE"
_DECODE_CHECK_ID: Final[str] = "PF-DECODE"

# The constructed cases AAP §0.3.4 fixes: ``synthetic:<name>`` for a case the framework
# builds rather than draws from the corpus.
_ENCODE_CASE_ID: Final[str] = "synthetic:pf-encode"
_DECODE_CASE_ID: Final[str] = "synthetic:pf-decode"

# The adapter-invocation counts each check declares, derived from the configuration
# defaults rather than typed in: ``PF-ENCODE`` asks the target for one symbol
# ``warmup + encode_samples`` times, and ``PF-DECODE`` asks it for nothing at all because
# it draws its own probe image.  tests/unit/test_checks_negative.py counts the calls a
# case makes against these numbers with a stub adapter, so a retry inside either check
# registers as one more invocation and fails that test instead of quietly improving a
# figure (AAP §0.4.3, §0.5.6).
_EXPECTED_DEFAULTS: Final[PerformanceLimits] = PerformanceLimits()
EXPECTED_ENCODES: Final[dict[str, int]] = {
    _ENCODE_CHECK_ID: _EXPECTED_DEFAULTS.warmup + _EXPECTED_DEFAULTS.encode_samples,
    _DECODE_CHECK_ID: 0,
}

# The decode half of the same accounting: ``PF-DECODE`` decodes one fixed PNG
# ``warmup + decode_samples`` times and ``PF-ENCODE`` decodes nothing.  Neither check
# retries a sample, so these counts are exact rather than upper bounds.
EXPECTED_DECODES: Final[dict[str, int]] = {
    _ENCODE_CHECK_ID: 0,
    _DECODE_CHECK_ID: _EXPECTED_DEFAULTS.warmup + _EXPECTED_DEFAULTS.decode_samples,
}

# The declared payload of each probe.  Fixed ASCII text, short enough to sit comfortably
# inside the capacity of the smallest symbol at every error level, so neither check ever
# measures a capacity boundary instead of the encode or decode path.  They are distinct so
# a reader of an evidence file can tell which probe produced it.
_ENCODE_PAYLOAD: Final[bytes] = b"qrcheck performance probe: encode path"
_DECODE_PAYLOAD: Final[bytes] = b"qrcheck performance probe: decode path"

# The quiet zone of the framework-drawn decode probe, in modules.  Four modules is the
# width the standard specifies around a QR symbol, and a decoder needs one to find the
# symbol at all; it is a property of the probe image rather than a threshold, which is why
# it lives here and not in qrcheck.toml.
_PROBE_QUIET_ZONE_MODULES: Final[int] = 4

# The two pixel values of the probe image: dark modules and light modules.  A grayscale
# image keeps the probe free of colour handling, so the decode latency measured is the
# pipeline's and not a colour-conversion artefact.
_PROBE_DARK_PIXEL: Final[int] = 0
_PROBE_LIGHT_PIXEL: Final[int] = 255

# Unit conversions and the quantile, named so the arithmetic below reads as the rule the
# module docstring states rather than as a bare number.
_MS_PER_SECOND: Final[float] = 1000.0
_SECONDS_PER_MINUTE: Final[float] = 60.0
_P95: Final[float] = 0.95

# The three adapter errors :func:`qrcheck.checks.support.case_status_for_error` converts
# into a status.  ``TargetUnavailable`` and ``TargetTimeout`` are absent on purpose: the
# runner owns the target-loss policy and can only apply it if it sees them itself.
_TRANSLATED_ERRORS: Final[tuple[type[BaseException], ...]] = (
    TargetRejected,
    TargetContractError,
    TargetUnsupported,
)


@check(
    id=_ENCODE_CHECK_ID,
    title="Encode p95 latency and throughput against the configured floors",
    family="performance",
    requires=frozenset({Capability.png}),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def pf_encode_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield ``PF-ENCODE``'s single constructed case.

    The case needs nothing from `ctx` at enumeration time -- every value it uses is on
    the context it is handed when the runner executes it, which is what keeps a case a
    pure function of the run rather than of the moment it was listed.  The generator
    takes the context because :data:`qrcheck.registry.check.CasesFn` requires it.

    Args:
        ctx: The run's context; unread here, passed on by the runner.

    Yields:
        Exactly one case, ``synthetic:pf-encode``.
    """
    yield support.synthetic_case(_ENCODE_CHECK_ID, _ENCODE_CASE_ID, _run_encode)


def _run_encode(ctx: RunContext) -> CaseResult:
    """Time ``warmup + encode_samples`` encodes and assert the two configured floors.

    The warm-up invocations are deliberately unmeasured: a reported p95 has to describe a
    warmed process rather than interpreter start-up, which is why the measured window
    starts after them (AAP §0.6.3).  The window's own elapsed time is what the throughput
    is computed from, so the figure is the rate the samples were actually produced at
    rather than an extrapolation from the mean.

    Args:
        ctx: The run's context: its adapter, configuration, error level and seed.

    Returns:
        A passed result carrying the measured metrics, or a failed result carrying them
        beside the breached threshold.  A failure records evidence -- the last artefact
        the target returned, the request that produced it and the measured figures --
        because the measurement is the finding.

    Raises:
        TargetUnavailable: The target could not be reached; propagated unchanged so the
            runner can report one error result for the check instead of one per sample.
        TargetTimeout: No response within the configured timeout; propagated unchanged.
        ValueError: The request or the measurement is unusable, which is a framework
            defect the runner reports as an errored case with its traceback.
    """
    limits = ctx.config.performance
    request = support.build(_ENCODE_PAYLOAD, ctx.ec_level)
    started_at = time.perf_counter()
    try:
        samples, window_seconds, symbol = _time_encodes(ctx, request, limits)
    except _TRANSLATED_ERRORS as error:
        return support.case_status_for_error(
            _ENCODE_CHECK_ID,
            _ENCODE_CASE_ID,
            error,
            request=request,
            duration_ms=_elapsed_ms(started_at),
            seed=ctx.seed,
        )
    metrics: dict[str, float | int] = {
        "encode_p95_ms": _nearest_rank_percentile(samples, _P95),
        "symbols_per_minute": _symbols_per_minute(len(samples), window_seconds),
        "samples": len(samples),
        "warmup": limits.warmup,
    }
    expected: dict[str, float | int] = {
        "encode_p95_ms": limits.encode_p95_ms,
        "encode_samples": limits.encode_samples,
        "min_symbols_per_minute": limits.min_symbols_per_minute,
        "warmup": limits.warmup,
    }
    duration_ms = _elapsed_ms(started_at)
    breaches = _encode_breaches(metrics, limits)
    if breaches:
        failed = support.fail(
            _ENCODE_CHECK_ID,
            _ENCODE_CASE_ID,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded={"message": " ".join(breaches), "metrics": metrics},
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
        return support.record_evidence(
            ctx,
            failed,
            symbol=symbol,
            request_extra=_encode_probe_parameters(limits),
        )
    return support.pass_(
        _ENCODE_CHECK_ID,
        _ENCODE_CASE_ID,
        request=request.as_dict(),
        decoded={"metrics": metrics},
        expected=expected,
        duration_ms=duration_ms,
        seed=ctx.seed,
    )


def _time_encodes(
    ctx: RunContext,
    request: GenerateRequest,
    limits: PerformanceLimits,
) -> tuple[list[float], float, GeneratedSymbol | None]:
    """Run the warm-up and measured encode loops, returning the samples and the window.

    Exactly ``limits.warmup`` untimed invocations followed by exactly
    ``limits.encode_samples`` timed ones, and no other call to the target: the counts in
    :data:`EXPECTED_ENCODES` are a promise the counting-stub test checks, so a retry of a
    slow sample would be an extra invocation rather than a better figure.

    Args:
        ctx: The run's context; only its adapter is read, through ``support.generate``.
        request: The single request every invocation repeats.
        limits: The resolved performance table, read for the two counts.

    Returns:
        The measured durations in milliseconds in call order, the elapsed seconds of the
        measured window, and the last artefact the target returned (``None`` only for a
        configuration that measures no samples, which ``qrcheck.config.schema`` refuses
        at load time).

    Raises:
        TargetUnsupported, TargetRejected, TargetContractError: As ``support.generate``
            raises them; the caller translates them into a status.
        TargetUnavailable, TargetTimeout: Propagated by the caller unchanged.
    """
    for _ in range(limits.warmup):
        support.generate(ctx, request)
    samples: list[float] = []
    symbol: GeneratedSymbol | None = None
    window_started = time.perf_counter()
    for _ in range(limits.encode_samples):
        sample_started = time.perf_counter()
        symbol = support.generate(ctx, request)
        samples.append(_elapsed_ms(sample_started))
    return samples, time.perf_counter() - window_started, symbol


def _encode_breaches(
    metrics: Mapping[str, float | int],
    limits: PerformanceLimits,
) -> tuple[str, ...]:
    """Return one message per configured floor `metrics` breaches, in a fixed order.

    Every breached floor is reported, not only the first: a run that is both too slow and
    too slow per symbol should say so once, in one result, rather than force a second run
    to discover the second breach.  The messages name the measured value and the
    configured threshold, which is what makes the failure actionable from the report
    alone.

    Args:
        metrics: The measured figures, as recorded in the result.
        limits: The resolved floors.

    Returns:
        The breach messages, empty when every floor holds.
    """
    breaches: list[str] = []
    encode_p95 = metrics["encode_p95_ms"]
    if encode_p95 > limits.encode_p95_ms:
        breaches.append(
            f"encode p95 {encode_p95:.3f} ms exceeds the configured floor "
            f"{limits.encode_p95_ms:.3f} ms."
        )
    throughput = metrics["symbols_per_minute"]
    if throughput < limits.min_symbols_per_minute:
        breaches.append(
            f"throughput {throughput:.1f} symbols/minute is below the configured floor "
            f"{limits.min_symbols_per_minute:.1f} symbols/minute."
        )
    return tuple(breaches)


def _encode_probe_parameters(limits: PerformanceLimits) -> dict[str, Any]:
    """Return the resolved measurement parameters to record beside the request.

    ``expected`` travels in ``summary.json`` but not in the evidence directory, so the
    probe's own resolved parameters are written into ``request.json`` as well: a reader
    holding only the evidence of a failed encode case can then see which budget was
    applied and over how many samples, without the summary that sat beside it.  Every
    value is configuration-derived, so the artefact stays byte-identical for the same
    configuration.

    Args:
        limits: The resolved performance table.

    Returns:
        The parameters as a JSON-ready mapping.
    """
    return {
        "encode_p95_ms": limits.encode_p95_ms,
        "encode_samples": limits.encode_samples,
        "min_symbols_per_minute": limits.min_symbols_per_minute,
        "probe": "encode",
        "warmup": limits.warmup,
    }


@check(
    id=_DECODE_CHECK_ID,
    title="Decode p95 latency of one fixed probe image against the configured floor",
    family="performance",
    requires=frozenset(),
    dependencies=(DEPENDENCY_PRIMARY_DECODER,),
)
def pf_decode_cases(ctx: RunContext) -> Iterator[Case]:
    """Yield ``PF-DECODE``'s single constructed case.

    No capability is declared, and that is a contract rather than an omission: the check
    draws its own probe image and never asks the target for anything, so it must stay
    runnable against a target that cannot encode, cannot render or cannot be reached.  A
    requirement here would turn a target-independent measurement into a ``skipped``
    result for a target that simply lacks a capability.

    Args:
        ctx: The run's context; unread here, passed on by the runner.

    Yields:
        Exactly one case, ``synthetic:pf-decode``.
    """
    yield support.synthetic_case(_DECODE_CHECK_ID, _DECODE_CASE_ID, _run_decode)


def _run_decode(ctx: RunContext) -> CaseResult:
    """Time ``warmup + decode_samples`` decodes of one framework-drawn PNG.

    The image is drawn by :func:`_probe_png`, so the measurement needs nothing from the
    target and the same bytes are decoded every time.  The first decode is asserted to
    return the payload the image was drawn from *before* any sample is timed: timing an
    image the pipeline cannot read would report a meaningless p95, and the condition is
    therefore a failed case rather than a surprising number.

    Args:
        ctx: The run's context: its configuration, error level and seed.  The adapter is
            never touched, which is why no adapter error can arise here.

    Returns:
        A passed result carrying the measured metrics and the probe's own facts, or a
        failed result naming the breached floor or the unreadable probe.  Both failures
        record evidence: the probe PNG, the request and the numbers.

    Raises:
        ValueError: The request is unusable or a measured window is degenerate -- both
            framework-side defects the runner reports as an errored case with its
            traceback.
    """
    limits = ctx.config.performance
    request = support.build(_DECODE_PAYLOAD, ctx.ec_level)
    started_at = time.perf_counter()
    probe_bytes = _probe_png(limits.decode_image_px, ctx.ec_level)
    probe_symbol = GeneratedSymbol(image_bytes=probe_bytes, output_format="png")
    probe: DecodeResult | None = None
    for index in range(limits.warmup):
        decoded = support.decode(ctx, probe_bytes, request)
        if index == 0:
            probe = decoded
    samples: list[float] = []
    if probe is None:
        # A zero warm-up is a legal configuration, and the probe still has to be proven
        # readable before anything is measured.  That decode becomes the first measured
        # sample rather than an extra invocation, so the declared decode count stays
        # exactly ``warmup + decode_samples`` for every configuration.
        sample_started = time.perf_counter()
        probe = support.decode(ctx, probe_bytes, request)
        samples.append(_elapsed_ms(sample_started))
    if probe.payload_bytes != request.payload:
        return _probe_failure(ctx, probe, request, probe_symbol, limits, started_at)
    for _ in range(limits.decode_samples - len(samples)):
        sample_started = time.perf_counter()
        support.decode(ctx, probe_bytes, request)
        samples.append(_elapsed_ms(sample_started))
    metrics: dict[str, float | int] = {
        "decode_p95_ms": _nearest_rank_percentile(samples, _P95),
        "image_px": limits.decode_image_px,
        "samples": len(samples),
        "warmup": limits.warmup,
    }
    expected: dict[str, float | int] = {
        "decode_image_px": limits.decode_image_px,
        "decode_p95_ms": limits.decode_p95_ms,
        "decode_samples": limits.decode_samples,
        "warmup": limits.warmup,
    }
    duration_ms = _elapsed_ms(started_at)
    decoded_probe = _probe_facts(probe)
    if metrics["decode_p95_ms"] > limits.decode_p95_ms:
        failed = support.fail(
            _DECODE_CHECK_ID,
            _DECODE_CASE_ID,
            reason=Reason.VALUE_MISMATCH,
            request=request.as_dict(),
            decoded={
                "message": (
                    f"decode p95 {metrics['decode_p95_ms']:.3f} ms exceeds the configured "
                    f"floor {limits.decode_p95_ms:.3f} ms over {metrics['samples']} "
                    f"samples of a {limits.decode_image_px} x {limits.decode_image_px} PNG"
                ),
                "metrics": metrics,
                "probe": decoded_probe,
            },
            expected=expected,
            duration_ms=duration_ms,
            seed=ctx.seed,
        )
        return support.record_evidence(
            ctx,
            failed,
            symbol=probe_symbol,
            request_extra=_decode_probe_parameters(limits),
        )
    return support.pass_(
        _DECODE_CHECK_ID,
        _DECODE_CASE_ID,
        request=request.as_dict(),
        decoded={"metrics": metrics, "probe": decoded_probe},
        expected=expected,
        duration_ms=duration_ms,
        seed=ctx.seed,
    )


def _probe_png(image_px: int, ec_level: str) -> bytes:
    """Return a PNG of the decode probe payload, exactly `image_px` pixels square.

    The image is the framework's own: the payload is encoded by segno at the run's error
    level with ``boost_error=False``, so the symbol occupies the smallest version that
    holds it at that level and the oracle's two expectation-corrupting defaults are both
    off (AAP §0.3.5).  ``segno.make_qr`` is Model 2 by construction and has no ``micro``
    parameter in the pinned 1.6.6 release -- passing one would be a ``TypeError`` -- so
    Micro QR cannot be drawn here and needs no argument to exclude.

    The module grid is padded with the quiet zone a decoder needs, then sampled to the
    requested size by nearest neighbour: output pixel ``i`` takes source pixel
    ``i * side // image_px``, which preserves module edges exactly and involves no
    interpolation, resampling filter or colour conversion.  The pixels are drawn as
    grayscale dark/light values and encoded through :mod:`qrcheck.evidence.mux`, which
    writes no ancillary chunk, so equal grids produce equal bytes on every run.

    Args:
        image_px: The side length of the image in pixels, from ``decode_image_px``.
        ec_level: The run's error-correction level, so the probe exercises the same
            symbol shape the run's other cases do.

    Returns:
        The PNG bytes, non-empty and exactly `image_px` pixels on each side.

    Raises:
        ValueError: `image_px` is not a positive integer -- a configuration the schema
            already refuses, so reaching it here is a framework defect.
        TypeError: segno rejects the level or the payload; a framework defect, since the
            level comes from a validated run and the payload is a declared constant.
    """
    # Imported here rather than at module scope: this is the only function that needs an
    # encoder or an image library, so importing the module to register the check must not
    # load them (``qrcheck list`` imports every check family).
    import numpy
    import segno

    from qrcheck.evidence import mux

    if isinstance(image_px, bool) or not isinstance(image_px, int) or image_px <= 0:
        raise ValueError(f"image_px must be a positive integer, got {image_px!r}")
    code = segno.make_qr(_DECODE_PAYLOAD, error=ec_level, boost_error=False)
    modules = numpy.array(code.matrix, dtype=bool)
    quiet = _PROBE_QUIET_ZONE_MODULES
    side = modules.shape[0] + 2 * quiet
    padded = numpy.zeros((side, side), dtype=bool)
    padded[quiet : quiet + modules.shape[0], quiet : quiet + modules.shape[1]] = modules
    index = (numpy.arange(image_px) * side) // image_px
    scaled = padded[numpy.ix_(index, index)]
    pixels = numpy.where(scaled, _PROBE_DARK_PIXEL, _PROBE_LIGHT_PIXEL).astype(numpy.uint8)
    # Pillow is reached only through the image boundary, as the layering rule requires:
    # ``mux`` is the one module that imports it, and its ``Image`` supplies the single
    # conversion that boundary has no function for -- a module grid to pixels.  This
    # module therefore imports no image library of its own.
    return mux.encode_png(mux.Image.fromarray(pixels))


def _probe_facts(probe: DecodeResult) -> dict[str, Any]:
    """Return the probe decode's own facts, for the result and for evidence.

    Recorded on a passing case as well as a failing one, so the figure a report shows can
    be read together with what was decoded to produce it: the version the grid dimension
    implies, the EC level from both independent sources, the mask, the grid digest and the
    payload length.  Every value is a function of the declared probe payload and the run's
    error level, so a passing case's record stays stable from run to run.

    Args:
        probe: The first decode of the probe image.

    Returns:
        The facts as a JSON-ready mapping.
    """
    return {
        "agreement": probe.agreement,
        "ec_level_decoder": probe.ec_level_decoder,
        "ec_level_format": probe.ec_level_format,
        "mask": probe.mask,
        "matrix_sha256": probe.matrix_sha256,
        "payload_len": None if probe.payload_bytes is None else len(probe.payload_bytes),
        "version": probe.version,
    }


def _probe_failure(
    ctx: RunContext,
    probe: DecodeResult,
    request: GenerateRequest,
    probe_symbol: GeneratedSymbol,
    limits: PerformanceLimits,
    started_at: float,
) -> CaseResult:
    """Return the failed result for a probe image the pipeline could not read.

    The condition is stated as a failure rather than measured around: a p95 computed from
    an image that does not decode describes nothing, and the run would otherwise report a
    comfortable number for a pipeline that had stopped working.  The whole decode result
    is persisted with the failure, so the per-decoder outcomes, the polarity and the
    format-information reading show why the pipeline could not read the framework's own
    image.

    Args:
        ctx: The run's context.
        probe: The decode result the pipeline produced for the probe image.
        request: The request the probe payload was built for.
        probe_symbol: The probe artefact, recorded as this case's symbol.
        limits: The resolved performance table, for the parameters written to evidence.
        started_at: When the case began, so the failed result carries its own duration.

    Returns:
        The failed result, carrying its evidence directory.
    """
    decoded: dict[str, Any] = dict(probe.as_dict())
    decoded["message"] = (
        "the framework's own probe image did not decode to the payload it was drawn from: "
        f"decoded {0 if probe.payload_bytes is None else len(probe.payload_bytes)} of "
        f"{len(_DECODE_PAYLOAD)} bytes with agreement "
        f"{probe.agreement!r}"
    )
    failed = support.fail(
        _DECODE_CHECK_ID,
        _DECODE_CASE_ID,
        reason=Reason.VALUE_MISMATCH,
        request=request.as_dict(),
        decoded=decoded,
        expected={
            "decode_image_px": limits.decode_image_px,
            "outcome": "probe_decodes",
            "payload_len": len(_DECODE_PAYLOAD),
        },
        duration_ms=_elapsed_ms(started_at),
        seed=ctx.seed,
    )
    return support.record_evidence(
        ctx,
        failed,
        symbol=probe_symbol,
        request_extra=_decode_probe_parameters(limits),
    )


def _decode_probe_parameters(limits: PerformanceLimits) -> dict[str, Any]:
    """Return the resolved decode measurement parameters to record beside the request.

    The counterpart of :func:`_encode_probe_parameters`: the floor, the sample plan and
    the image size all reach ``request.json`` so a failed decode case is readable from its
    own evidence directory.  ``probe`` and ``probe_image`` state that the artefact in the
    directory is the framework's drawing rather than a target's output, which a reader of
    ``symbol.png`` would otherwise have no way to know.

    Args:
        limits: The resolved performance table.

    Returns:
        The parameters as a JSON-ready mapping.
    """
    return {
        "decode_image_px": limits.decode_image_px,
        "decode_p95_ms": limits.decode_p95_ms,
        "decode_samples": limits.decode_samples,
        "probe": "decode",
        "probe_image": "framework-generated",
        "warmup": limits.warmup,
    }


def _nearest_rank_percentile(samples: Sequence[float], quantile: float) -> float:
    """Return the nearest-rank `quantile` of `samples`, in the samples' own unit.

    The rule the module docstring states: sort ascending, take index
    ``ceil(quantile * n) - 1``, clamped into range.  No interpolation, so the reported
    figure is always one of the durations actually measured and two readers of the same
    samples always agree on it.

    Args:
        samples: The measured durations, in any order.
        quantile: The quantile to take, in ``(0, 1]``.

    Returns:
        The sample at the nearest rank.

    Raises:
        ValueError: `samples` is empty or `quantile` is outside ``(0, 1]``.  The schema
            guarantees at least ten samples, so an empty sequence is a caller defect.
    """
    if not samples:
        raise ValueError("a percentile needs at least one sample")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile!r}")
    ordered = sorted(samples)
    rank = math.ceil(quantile * len(ordered)) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


def _symbols_per_minute(count: int, window_seconds: float) -> float:
    """Return the throughput of `count` encodes produced in `window_seconds`.

    Args:
        count: How many symbols the measured window produced.
        window_seconds: How long that window took, measured with the same clock the
            samples were.

    Returns:
        The rate in symbols per minute.

    Raises:
        ValueError: `window_seconds` is not positive, which would make the rate infinite.
            A monotonic clock difference over a hundred sampled calls cannot be zero, so
            this is a framework defect rather than a measurement to report.
    """
    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError(
            f"the measured window must be a positive duration, got {window_seconds!r}"
        )
    return count / window_seconds * _SECONDS_PER_MINUTE


def _elapsed_ms(started_at: float) -> float:
    """Return the milliseconds elapsed since `started_at` on the monotonic clock.

    The one place this module computes a duration from a clock reading, so every
    ``duration_ms`` and every sample is produced the same way.  A wall clock is never
    read: a monotonic difference can only be a duration, which is the only thing a report
    records (AAP §0.5.5).

    Args:
        started_at: A prior :func:`time.perf_counter` reading.

    Returns:
        The elapsed milliseconds, never negative.
    """
    return max(time.perf_counter() - started_at, 0.0) * _MS_PER_SECOND
