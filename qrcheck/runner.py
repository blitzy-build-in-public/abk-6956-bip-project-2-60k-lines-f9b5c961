"""Orchestration of exactly one sequential run, and nothing else.

:func:`run_suite` is the single public entry point of the product's execution layer.  It asserts
nothing, builds no request, encodes nothing, decodes nothing, writes no report and decides no exit
code; it performs this sequence and nothing else: capture the run's start time once; probe
what the host provides -- both reference decoders and the SVG rasterizer, with **zero** target
invocations -- and refuse to start when the blocking primary decoder is unusable or any other
pinned dependency cannot be imported; resolve the ``--checks`` selection, because which checks a run
selected is what decides what the run needs; load the committed corpus when that selection reads it
-- a ``--class`` filter was given, or a selected check draws its cases from the corpus -- and apply
the filter, refusing to start when it leaves nothing; build the one
:class:`~qrcheck.registry.check.RunContext` the checks of this run share; resolve each selected
check's requirements against the target's declared capabilities, so every check is decided runnable
or skipped *before* the first case runs; walk the runnable checks in check-identifier order and
each check's cases in case-identifier order, containing every per-case fault, and finish with the
run-scoped checks -- the ones whose subject is the whole run -- so that they observe the run's end;
apply the one-error-result-per-check policy when the target is lost; and accumulate one
:class:`~qrcheck.report.model.RunResult` before closing the adapter.

What this module deliberately does not own.  **Exit codes**: ``qrcheck.cli.exit`` is the only place
a code is decided, from the returned model's tallies, and nothing here prints, calls ``sys.exit`` or
maps a count.  **Reports**: ``qrcheck.cli.run`` calls ``qrcheck.report.write_all_reports`` after
:func:`run_suite` returns, and nothing here serialises a result.  **Evidence layout**: every check
persists its own case evidence through :func:`qrcheck.checks.support.record_evidence`, which routes
to the single store in :mod:`qrcheck.report.evidence`; this module creates that store, hands it to
the check layer on the context, and records evidence itself only for the results **it** builds,
because re-recording a check's result here would overwrite ``request.json`` with a body that lacks
the ``out_of_contract`` marker and the resolved case parameters the check added.  **Adapter-error
translation**: ``TargetRejected``, ``TargetContractError`` and ``TargetUnsupported`` become case
outcomes inside the check layer (:func:`qrcheck.checks.support.case_status_for_error`), and the
capability gate was already decided by ``qrcheck.registry.resolve``.  **Retries**: there is no
retry, re-run, quarantine or ``xfail`` path here and there must never be one, because each check's
declared adapter-invocation count is asserted against a single pass, so a retry would surface as
that count plus one.

This module raises **none** of the five adapter errors itself; that closed vocabulary belongs to
:mod:`qrcheck.targets`.  It produces exactly four kinds of outcome.  :class:`RunPrerequisiteError`
for a fault found before any case runs, after which the command ends with exit code 2 and writes no
report, raised for exactly four conditions: the blocking primary decoder (``zxing-cpp``) is
unusable, another pinned runtime dependency cannot be imported, the configured corpus file cannot be
read or is malformed, or the ``--class`` filter leaves zero records -- the message always naming the
offending dependency, path or glob.  An ``error`` result, created here and only here, for an
exception that escaped a case's ``execute``, a check's case generator or an evidence write --
each contained with its traceback while the run continues, the evidence one also keeping every
result the walk had already accumulated, which is what lets the writer path stream them when the
report directory turns out to be unwritable -- or for a lost target, a fact about the run rather
than a verdict on a case.  A ``skipped``
result, built by :func:`qrcheck.registry.resolve.skip_case_results`, one per check the target's
capabilities or the host's dependencies cannot serve, counted separately and never folded into
``passed``.  And a ``passed`` or ``failed`` result, produced by a check and accumulated
verbatim.

Target loss is a run-level condition, not a per-case one:
:class:`~qrcheck.targets.base.TargetUnavailable` and
:class:`~qrcheck.targets.base.TargetTimeout` propagate out of a check unchanged, because only this
layer can tell one dead target from one defective symbol.  On the first such error the check that
met it receives exactly **one** ``error`` result filed under ``synthetic:target``, its remaining
cases are abandoned, and every later check that requires a capability of the target receives its
own single ``synthetic:target`` error without the target being called again.  Checks that declare
no capability requirement are target-independent (``PF-DECODE`` and ``CL-ARTIFACTS`` declare
none) and still run, so the report describes what was measurable rather than only what broke.

Determinism: the walk order is check identifier then case identifier, and the results are sorted by
that pair before they leave; ``run_id`` is a content hash over the target name, the seed, the EC
level, the selected check identifiers and a digest of the filtered corpus, so a ``--class`` run has
a different identity from a whole-corpus one.  Nothing here reads ``hash()``, ``id()``, the
environment, a locale, a set's iteration order or a filesystem listing, and the only wall-clock
readings are the two run timestamps and the ``time.perf_counter`` deltas that become
``duration_ms`` -- the fields every determinism comparison masks because the result model requires
them.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import platform
import sysconfig
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import qrcheck
from qrcheck.checks import load_families
from qrcheck.checks.cleanup import temp_entries
from qrcheck.checks.render import discover_svg_rasterizer
from qrcheck.config.models import Config
from qrcheck.corpus.builder import CorpusRecord
from qrcheck.corpus.io import filter_by_class, load_corpus
from qrcheck.decode import decoder_availability
from qrcheck.dependencies import (
    PRIMARY_DECODER_MODULE,
    blocking_problem,
    import_problem,
)
from qrcheck.loc import count_lines
from qrcheck.registry.check import (
    DEPENDENCY_PRIMARY_DECODER,
    DEPENDENCY_SECONDARY_DECODER,
    DEPENDENCY_SVG_RASTERIZER,
    REGISTRY,
    Case,
    CheckContract,
    RunContext,
)
from qrcheck.registry.resolve import Resolution, resolve, skip_case_results
from qrcheck.registry.selection import Selection, select
from qrcheck.report.evidence import EvidenceStore
from qrcheck.report.model import (
    TRACEBACK_KEY,
    CaseResult,
    Counts,
    Reason,
    RunResult,
    Status,
)
from qrcheck.targets.base import Adapter, TargetTimeout, TargetUnavailable

__all__ = ["RunPrerequisiteError", "run_suite"]

#: The case identifier every target-loss result is filed under.
#:
#: One string, defined once: ``summary.json`` carries it as ``case_id``, ``junit.xml``
#: renders it inside ``name="<CHECK-ID>[<case_id>]"`` and the evidence directory is derived
#: from it.  It is deliberately distinct from ``synthetic:capability`` (a pre-execution
#: skip, owned by :mod:`qrcheck.registry.resolve`) and from every ``synthetic:*`` name a
#: check builds for itself, so a reader can tell "this target never promised this" from
#: "this target was reachable when the run started and then broke".
TARGET_LOSS_CASE_ID: Final[str] = "synthetic:target"

#: The case identifier a result for a broken case *generator* is filed under.
#:
#: Deliberately distinct from :data:`TARGET_LOSS_CASE_ID` -- a lost target is a fact about the
#: run's environment, described by one result per affected check, while a generator that raises
#: is a fault of one check's own enumeration -- and from every ``synthetic:*`` name a check
#: builds for itself (``capability``, ``cleanup``, ``determinism-*``, ``iv-*``, ``pf-*``,
#: ``target``), so a reader can tell "this check could not say what it would run" from "this
#: target never promised this" and from "this case did not behave".
CASES_FAILED_CASE_ID: Final[str] = "synthetic:cases"

#: The two statuses that make a check count as having gone wrong, for ``--fail-fast``.
#: A ``skipped`` result is not one of them: it describes a target that never promised the
#: capability, not a check that produced a bad verdict.
_STOPPING_STATUSES: Final[frozenset[Status]] = frozenset({Status.FAILED, Status.ERROR})

#: The machine names for which zxing-cpp 2.2.0 publishes no CPython 3.11 wheel.  Named
#: so the blocking diagnostic can say that the *platform* is the obstruction rather than
#: a broken installation; the check is combined with ``platform.system()`` because macOS
#: arm64 does have a universal2 wheel.
_WHEELLESS_MACHINES: Final[tuple[str, ...]] = ("aarch64", "arm64", "armv7l", "armv8l")


class RunPrerequisiteError(Exception):
    """A run cannot start, so its command ends with exit code 2 and writes no report.

    The carrier for every fault this layer detects **before** the first case runs, and the
    only exception :func:`run_suite` raises of its own.  Four conditions use it:

    * the blocking primary decoder (``zxing-cpp``) is not usable -- every round-trip assertion
      is decoded with it, so ``RT-ROUNDTRIP`` must never be silently skipped in its place;
    * another pinned runtime dependency cannot be imported -- a broken installation is an
      environment error, exit 2 naming the distribution and ``requirements.lock``;
    * the configured corpus file cannot be read, or holds a line that is not a valid
      record, when the run selects a corpus-driven check or gives a ``--class`` filter -- and
      only then;
    * the ``--class`` filters leave zero corpus records, so the run would describe an empty
      selection as if it were a complete one.

    It carries a message and nothing else: no exit code (``qrcheck.cli.exit`` owns those),
    no :class:`~qrcheck.report.model.Reason` (the model's vocabulary describes a finished
    case, and no case exists), and no partial result.  ``qrcheck.cli.run`` maps this type
    -- together with :class:`qrcheck.config.schema.ConfigError` and
    :class:`qrcheck.registry.selection.SelectionError` -- to exit code 2.

    It is deliberately **not** a :class:`qrcheck.targets.base.TargetError`: that hierarchy
    is the closed failure vocabulary of a target implementation, and a missing decoder or an
    empty corpus filter is a condition of the framework's own environment.
    """


def run_suite(
    config: Config,
    selection: Selection,
    adapter: Adapter,
    ec_level: str,
    seed: int,
    *,
    fail_fast: bool = False,
) -> RunResult:
    """Execute one run and return its accumulated model.

    Args:
        config: The resolved configuration; it owns every threshold, the corpus path -- read
            only when this run's selection reads the corpus (:func:`_corpus_for_run`) -- the
            report directory and the ``[render]`` table.
        selection: What the run was asked to select -- the raw ``--checks`` selector and the
            normalised ``--class`` globs.
        adapter: The target, reached only through the :class:`~qrcheck.targets.base.Adapter`
            port.  It is closed exactly once, after the walk, whatever happened during it.
        ec_level: The requested error-correction level, ``L``, ``M``, ``Q`` or ``H``.  It is
            recorded on every result and on the run, and passed to the checks on the context.
        seed: The run's seed.  Every derived seed and every reproducibility claim in the
            report is derived from it via :func:`qrcheck.config.seeding.derive_seed`.
        fail_fast: Stop launching further ordinary checks once a check has produced a
            ``failed`` or ``error`` result.  The in-flight check always completes all of its
            cases, so the report never describes a check whose cases were cut off half-way.
            The run-scoped check still runs, because it never invokes the target and its
            subject is the finished run.  Additive and
            keyword-only because the CLI's ``--fail-fast`` option has to reach the runner
            while the positional interface stays as specified.

    Returns:
        The finished :class:`~qrcheck.report.model.RunResult`, whose ``results`` are sorted by
        ``(check_id, case_id)`` and whose ``counts`` are the four separate tallies the model
        re-derives and compares.  A run whose target was lost still returns normally, with
        one ``error`` result per affected check, and a case whose evidence could not be written
        is reported in its own place as one ``error`` result (:func:`_persist`), so the CLI can
        always write a complete report -- or stream one -- and exit 3.

    Raises:
        RunPrerequisiteError: The primary decoder is unusable, another pinned runtime
            dependency cannot be imported, the corpus file cannot be read or is malformed when
            this run's selection reads it (:func:`_corpus_for_run`), or the
            ``--class`` filter leaves no record.
        SelectionError: The ``--checks`` selector matches no check (raised by
            :func:`qrcheck.registry.selection.select`), which is exit code 2 upstream.
        OSError: The previous run's evidence tree cannot be removed
            (:meth:`qrcheck.report.evidence.EvidenceStore.clear`), which happens before the walk
            and before any result exists, and is exit code 3 upstream.  An evidence write, or any
            other refusal the store documents, once the walk has begun raises here no more:
            :func:`_persist` contains it as an ``error`` result, because aborting would discard the
            results accumulated so far.
    """
    started_at = datetime.datetime.now(datetime.timezone.utc)
    # Idempotent: the family modules register themselves while they import, and calling this
    # again re-binds the modules Python already holds rather than registering a second set.
    load_families()
    probes = _probe_availability(config)
    _require_primary_decoder(probes.available)
    # Before the corpus is read and long before an evidence store or a report exists: a broken
    # installation is an exit-2 environment error, not thirty fabricated generator defects.
    _require_blocking_dependencies()
    # Resolved before the corpus is read: which checks this run selected is what decides
    # whether it needs the corpus at all, and an empty selector is an exit-2 fault of its own,
    # raised by ``select`` before anything else here runs.
    selected = select(REGISTRY, selection.checks)
    corpus = _corpus_for_run(config, selection, selected)
    # Bound before the walk so that a target lost, or a check that failed, cannot change the
    # identity of the run that reports it.
    adapter_name = adapter.name
    capabilities = adapter.capabilities
    evidence = EvidenceStore(config.report_dir)
    # The snapshot ``CL-ARTIFACTS`` calls "before the run", taken at the last instant before a
    # check can enumerate a case -- every exit-2 fault is already decided -- through the
    # observation that check documents, so this layer restates neither the prefix nor the listing.
    temp_baseline = temp_entries()
    context = RunContext(
        adapter=adapter,
        config=config,
        corpus=corpus,
        ec_level=ec_level,
        seed=seed,
        availability=MappingProxyType(probes.available),
        evidence=evidence,
        temp_baseline=temp_baseline,
    )
    resolution = resolve(
        selected,
        capabilities,
        probes.available,
        attempted=probes.attempted,
    )
    # After every exit-2 fault is decided: a reused root then holds this run alone.
    evidence.clear()
    # A skipped-with-reason case carries evidence like any other, and only this layer stores the
    # pre-execution skips: through :func:`_persist` like every other result this layer builds, so
    # a store that cannot write the skip reports it as an error result rather than raising.
    results: list[CaseResult] = [
        _persist(evidence, result) for result in skip_case_results(resolution, seed)
    ]
    try:
        results.extend(_walk(resolution, context, fail_fast=fail_fast))
    finally:
        # In a finally, so a case that raised an uncontained exception still releases the
        # target before the exception reaches the CLI.
        _close_adapter(adapter)
    ordered = tuple(
        sorted(results, key=lambda result: (result.check_id, result.case_id))
    )
    loc_count = count_lines(_repository_root()).total
    finished_at = datetime.datetime.now(datetime.timezone.utc)
    return RunResult(
        run_id=_run_identifier(
            adapter_name,
            ec_level,
            seed,
            tuple(contract.id for contract in selected),
            corpus,
        ),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        adapter=adapter_name,
        seed=seed,
        ec_level=ec_level,
        loc_count=loc_count,
        counts=Counts.from_results(ordered),
        results=ordered,
    )


# ---------------------------------------------------------------------------
# Pre-flight probe: what the host provides, with no target invocation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _Probes:
    """What the host provides, and the order each ordered probe walked to find out."""

    available: dict[str, bool]
    attempted: dict[str, tuple[str, ...]]


def _probe_availability(config: Config) -> _Probes:
    """Return whether each runtime dependency a check may declare is available, and what was tried.

    Three probes, all of them facts about the host rather than about the target, and none of
    them touching the target: adding a probe encode here would make every declared
    adapter-invocation count wrong by one.

    * the primary decoder is read from :func:`qrcheck.decode.decoder_availability`, which
      locates the module with ``importlib.util.find_spec`` and never imports it;
    * the secondary decoder comes from the same probe; its absence degrades the run to
      single-decoder mode instead of failing it;
    * the SVG rasterizer comes from the product's only rasterizer discovery, resolved against
      the ``[render] svg_rasterizer`` setting so that ``auto`` and a forced name both land
      here; it is the one dependency probed in an ordered candidate list.

    The keys are the :data:`qrcheck.registry.check.DEPENDENCY_TOKENS` values, which is what
    :func:`qrcheck.registry.resolve.resolve` reads and what
    :attr:`qrcheck.registry.check.RunContext.availability` carries to the checks.

    Args:
        config: The resolved configuration; only its ``render`` table is read.

    Returns:
        The availability per token in token-name order, and the rasterizer's discovery order.
    """
    decoders = decoder_availability()
    rasterizer = discover_svg_rasterizer(config.render)
    return _Probes(
        available={
            DEPENDENCY_PRIMARY_DECODER: bool(decoders.primary),
            DEPENDENCY_SECONDARY_DECODER: bool(decoders.secondary),
            DEPENDENCY_SVG_RASTERIZER: bool(rasterizer.available),
        },
        attempted={DEPENDENCY_SVG_RASTERIZER: rasterizer.attempted},
    )


def _require_primary_decoder(availability: Mapping[str, bool]) -> None:
    """Refuse the run when the blocking primary decoder is not usable.

    ``zxing-cpp`` is the authority for every round-trip assertion, so a run without it would
    either fail every decode-dependent check for the wrong reason or, worse, let the checks
    be skipped and the run look green.  It is therefore a pre-execution fault, decided before
    any case runs and before a report exists.

    Two states are refused, because :func:`qrcheck.decode.decoder_availability` locates the module
    with ``importlib.util.find_spec`` and never imports it, and so cannot see the second: the
    binding is absent, and the binding is present but fails to import.  Without the second check the
    run would start, every decode would fail, and the secondary decoder would silently become the
    authority that reports a conforming target as defective.  The load is confirmed through
    :func:`qrcheck.dependencies.import_problem`, the probe ``qrcheck doctor`` uses.

    Args:
        availability: The probe result from :func:`_probe_availability`.

    Raises:
        RunPrerequisiteError: The primary decoder is absent or cannot be imported.  The message
            names ``zxing-cpp``, the reason, the interpreter version and the platform tag.
    """
    if not availability.get(DEPENDENCY_PRIMARY_DECODER, False):
        problem: str | None = f"the {PRIMARY_DECODER_MODULE} module is not importable"
    else:
        problem = import_problem(PRIMARY_DECODER_MODULE)
    if problem is None:
        return
    raise RunPrerequisiteError(_primary_decoder_message(problem))


def _require_blocking_dependencies() -> None:
    """Refuse the run when any pinned runtime dependency cannot be imported.

    ``segno`` is the oracle and ``numpy``, ``Pillow``, ``httpx`` and ``click`` are imported at the
    point of use, so an unusable one is a broken installation rather than a user's choice: it is
    declared in ``requirements.lock``, and the exit-code contract counts it an environment error,
    exit 2 naming the distribution, decided before execution -- so before the corpus is read and
    before any evidence, result or report can exist.

    Raises:
        RunPrerequisiteError: One of :data:`qrcheck.dependencies.CORE_DEPENDENCIES` is not installed
            or its module does not import. The message is the probe's own diagnostic.
    """
    problem = blocking_problem()
    if problem is None:
        return
    raise RunPrerequisiteError(problem.message)


def _primary_decoder_message(detail: str | None = None) -> str:
    """Return the blocking diagnostic for an unusable primary decoder.

    It names the distribution a reader has to install, the reason it is unusable, the interpreter
    and the platform tag the wheel would have to match, and -- for the one platform zxing-cpp 2.2.0
    does not publish, Linux arm -- that the platform itself is the obstruction.

    Args:
        detail: Why the decoder is unusable, as the presence probe or the guarded import worded it.
            ``None`` renders the absent-module wording.

    Returns:
        The one-line diagnostic ``qrcheck run`` prints on stderr before exiting 2.
    """
    interpreter = f"CPython {platform.python_version()}"
    tag = sysconfig.get_platform()
    problem = (
        f"the {PRIMARY_DECODER_MODULE} module is not importable" if detail is None else detail
    )
    message = (
        f"the primary reference decoder zxing-cpp cannot be used for {interpreter} on {tag} "
        f"because {problem}; every round-trip assertion is decoded with it, so the run cannot "
        f"start and RT-ROUNDTRIP is never silently skipped -- install zxing-cpp==2.2.0 from "
        f"requirements.lock"
    )
    return message + _platform_note(tag)


def _platform_note(tag: str) -> str:
    """Return the "no wheel for this platform" clause, or an empty string.

    Args:
        tag: The platform tag from :func:`sysconfig.get_platform`.

    Returns:
        A clause naming ``tag`` when the interpreter runs on Linux arm, where zxing-cpp 2.2.0
        publishes no CPython 3.11 wheel; otherwise an empty string, because every other
        supported platform does have one.
    """
    machine = platform.machine().lower()
    if platform.system() != "Linux" or not machine.startswith(_WHEELLESS_MACHINES):
        return ""
    return (
        f"; zxing-cpp 2.2.0 publishes no wheel for {tag}, so the platform itself is the "
        f"obstruction rather than a broken installation"
    )


# ---------------------------------------------------------------------------
# Corpus: load, filter, and refuse an empty selection
# ---------------------------------------------------------------------------


def _corpus_for_run(
    config: Config,
    selection: Selection,
    selected: tuple[CheckContract, ...],
) -> tuple[CorpusRecord, ...]:
    """Return the corpus records this run covers, reading the file only when the run uses it.

    This module conditions ``corpus_path``'s "must exist" rule on a corpus-driven check being
    selected, so the file is read for exactly two kinds of run and neither of them otherwise:

    * any run given a ``--class`` filter, because that filter is a statement about the records:
      :func:`_filtered_corpus` is what refuses a filter that leaves nothing;
    * any run that selected a check declaring ``corpus_driven``, because those cases are drawn
      from the records the run loaded.

    Every other selection -- a run of the synthetic-only checks such as ``IV-EMPTY``,
    ``PF-DECODE`` or ``CL-ARTIFACTS`` -- proceeds with no corpus at all, which is what lets it
    complete on a machine where no corpus has ever been built.

    Args:
        config: The resolved configuration; its ``corpus_path`` names the file.
        selection: The run's selection; its ``class_globs`` decide whether a filter applies.
        selected: The contracts this run selected, in identifier order.

    Returns:
        The matching records in ``id`` order and never empty, or ``()`` when this run reads no
        corpus record.

    Raises:
        RunPrerequisiteError: A corpus is needed and the file cannot be read, holds a malformed
            line, or the filter leaves no record. Every message names the path or the globs.
    """
    if not selection.class_globs and not any(
        contract.corpus_driven for contract in selected
    ):
        return ()
    return _filtered_corpus(config, selection)


def _filtered_corpus(config: Config, selection: Selection) -> tuple[CorpusRecord, ...]:
    """Return the corpus records this run covers, refusing an empty selection.

    Called only for a run that reads the corpus -- a ``--class`` filter was given, or a selected
    check declares ``corpus_driven`` (:func:`_corpus_for_run` decides that).  Two faults are the
    runner's rather than the loader's, because both depend on the run and neither can be decided
    from the configuration alone: the corpus file itself, which
    :func:`qrcheck.corpus.io.load_corpus` reads but whose absence means no run can happen, and
    the ``--class`` filter, which only this layer applies.  A filter that leaves nothing must
    never be reported as a completed run: the suite would look clean while covering no record
    at all.

    Args:
        config: The resolved configuration; its ``corpus_path`` names the file.
        selection: The run's selection; its ``class_globs`` narrow the records.

    Returns:
        The matching records in ``id`` order, never empty.

    Raises:
        RunPrerequisiteError: The file cannot be read, holds a malformed line, or the filter
            leaves no record.  Every message names the path or the globs it was given.
    """
    try:
        records = load_corpus(config.corpus_path)
    except OSError as error:
        raise RunPrerequisiteError(
            f"the corpus file {config.corpus_path} cannot be read ({error}); build it with "
            f"'qrcheck corpus build --seed <seed>' or point corpus_path at an existing file"
        ) from error
    except ValueError as error:
        # CorpusError is a ValueError, and so is a decode fault in the file's text; both mean
        # the corpus on disk is not the corpus this run can describe.
        raise RunPrerequisiteError(
            f"the corpus file {config.corpus_path} is not a usable corpus: {error}"
        ) from error
    filtered = filter_by_class(records, selection.class_globs)
    if filtered:
        return filtered
    raise RunPrerequisiteError(_empty_corpus_message(config, selection))


def _empty_corpus_message(config: Config, selection: Selection) -> str:
    """Return the diagnostic for a corpus that leaves this run no record.

    Args:
        config: The resolved configuration, naming the file in the diagnostic.
        selection: The run's selection; its globs are quoted verbatim when there are any.

    Returns:
        The one-line diagnostic, which names the ``--class`` globs exactly as the run's
        selection holds them, or the corpus file when no filter was given.
    """
    if selection.class_globs:
        globs = " ".join(selection.class_globs)
        return (
            f"--class {globs} selected no corpus record in {config.corpus_path}; a filter "
            f"must leave at least one record, and no report is written for an empty "
            f"selection"
        )
    return (
        f"the corpus file {config.corpus_path} holds no records; build it with 'qrcheck "
        f"corpus build --seed <seed>'"
    )


# ---------------------------------------------------------------------------
# The walk: sorted checks, sorted cases, contained faults, the target-loss policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _CheckRun:
    """What one check contributed to the run, and whether it lost the target.

    A private record, not a result type: ``results`` holds the check's accumulated
    :class:`~qrcheck.report.model.CaseResult` objects in case-identifier order, and
    ``target_loss`` is the error that ended the check early, or ``None`` when the check
    completed.  The two travel together because only the walk, not the check, decides what
    the next check is allowed to do about a lost target.
    """

    results: tuple[CaseResult, ...]
    target_loss: TargetUnavailable | TargetTimeout | None


def _walk(
    resolution: Resolution,
    context: RunContext,
    *,
    fail_fast: bool,
) -> tuple[CaseResult, ...]:
    """Execute every runnable check of `resolution` in order and return the results.

    The order is fixed twice: checks by identifier ascending within each of the two groups below
    (which :class:`~qrcheck.registry.resolve.Resolution` has already sorted) and each check's cases
    by case identifier ascending.  Import order, registry insertion order and the order a check
    happened to yield its cases therefore never reach a report.

    Three run-level policies live here:

    * **Target loss.** Once any check has lost the target, every later check that requires a
      capability gets exactly one ``synthetic:target`` error and is never asked to produce a
      case, so one dead target yields one result per affected check rather than one per
      corpus record.  A check that requires no capability is target-independent and still
      runs.
    * **``--fail-fast``.** After a check finishes, a ``failed`` or ``error`` result stops
      further checks from launching.  The check that produced it always finished all of its
      cases, and the pre-execution skip results are kept, because they were decided before
      execution and describe checks that were never going to run.
    * **Run-scoped observation.** A check whose contract sets
      :attr:`~qrcheck.registry.check.CheckContract.observes_run_end` asserts a property of the
      whole run rather than of one case, so it executes after every ordinary check -- still
      through :func:`_run_check`, so the per-case containment and the one-result-per-check rule
      apply unchanged.  It always runs, even after a ``--fail-fast`` break, because it never
      invokes the target and the footprint it reports is only complete once the run is.
      Running it out of identifier order changes no report: :func:`run_suite` sorts every
      result by ``(check_id, case_id)`` before it returns.

    Args:
        resolution: The selected checks, split into the runnable ones and the skips.
        context: The run's single context, shared by every check and every case.
        fail_fast: Whether to stop launching checks after the first failed or errored one.

    Returns:
        Every result this walk produced, in execution order: the ordinary checks by identifier
        with their cases in case-identifier order, then the run-scoped checks in identifier order.
        :func:`run_suite` sorts the accumulated tuple by ``(check_id, case_id)``.
    """
    ordinary = tuple(
        contract for contract in resolution.runnable if not contract.observes_run_end
    )
    run_scoped = tuple(
        contract for contract in resolution.runnable if contract.observes_run_end
    )
    results: list[CaseResult] = []
    target_loss: TargetUnavailable | TargetTimeout | None = None
    for contract in ordinary:
        if target_loss is not None and contract.requires:
            # The target died earlier and this check cannot run without it: one result, no
            # target call, no case enumerated.
            results.append(
                _target_loss_result(contract.id, target_loss, context, duration_ms=0.0)
            )
            continue
        outcome = _run_check(contract, context)
        results.extend(outcome.results)
        if outcome.target_loss is not None:
            target_loss = outcome.target_loss
        if fail_fast and any(
            result.status in _STOPPING_STATUSES for result in outcome.results
        ):
            break
    # After the walk, and after a ``--fail-fast`` break: a run-scoped check never invokes the target
    # and its subject is the footprint of the finished run.
    for contract in run_scoped:
        if target_loss is not None and contract.requires:
            results.append(
                _target_loss_result(contract.id, target_loss, context, duration_ms=0.0)
            )
            continue
        results.extend(_run_check(contract, context).results)
    return tuple(results)


def _run_check(contract: CheckContract, context: RunContext) -> _CheckRun:
    """Execute every case of `contract`, containing each fault it meets.

    A case's ``execute`` is called exactly once per case: there is no retry path, so a
    check's declared adapter-invocation count comes out right on a single pass and any
    attempt to re-run a case would surface as that count plus one.

    Four outcomes, the last two of them faults this layer contains rather than re-raises:

    * a returned result is accumulated as the check built it, evidence directory and all;
    * :class:`~qrcheck.targets.base.TargetUnavailable` or
      :class:`~qrcheck.targets.base.TargetTimeout` ends the check: one ``synthetic:target``
      error result is recorded and the remaining cases are abandoned, because they would all
      report the same dead target;
    * any other exception escaping a case's ``execute`` is contained: one ``error`` result
      carrying the traceback is recorded and the walk continues with the next case, which is
      what keeps one broken case from hiding every result after it;
    * an exception escaping the check's own case *generator* -- raised while its cases are
      enumerated, before any case can run -- is contained the same way: one ``error`` result
      under :data:`CASES_FAILED_CASE_ID` carrying the traceback is recorded and the walk
      continues with the next **check**, so a check that cannot even say what it would run
      never destroys the report for the others. Both containments hold the same contract: the
      check reports ``error`` with its traceback, the run continues, and the process exits 3
      at the end, after the report is written.

    Args:
        contract: The registered check to execute.
        context: The run's single context.

    Returns:
        The check's results in case-identifier order, and the target error that ended it
        early when there was one.
    """
    results: list[CaseResult] = []
    try:
        cases: list[Case] = sorted(contract.cases(context), key=lambda case: case.case_id)
    except Exception as error:  # Containment is this layer's purpose.
        # A check that cannot enumerate its cases has produced no result at all, so its whole
        # contribution is one error result: letting this escape would abort the run before a
        # RunResult exists, and one broken generator would destroy the report for the other
        # twenty checks.
        results.append(
            _internal_error_result(
                contract.id,
                CASES_FAILED_CASE_ID,
                error,
                context,
                duration_ms=0.0,
            )
        )
        return _CheckRun(tuple(results), None)
    for case in cases:
        started = time.perf_counter()
        try:
            result = case.execute(context)
        except (TargetUnavailable, TargetTimeout) as error:
            results.append(
                _target_loss_result(
                    contract.id,
                    error,
                    context,
                    duration_ms=_elapsed_ms(started),
                )
            )
            return _CheckRun(tuple(results), error)
        except Exception as error:  # Containment is this layer's purpose.
            results.append(
                _internal_error_result(
                    contract.id,
                    case.case_id,
                    error,
                    context,
                    duration_ms=_elapsed_ms(started),
                )
            )
            continue
        results.append(result)
    return _CheckRun(tuple(results), None)


def _target_loss_result(
    check_id: str,
    error: TargetUnavailable | TargetTimeout,
    context: RunContext,
    *,
    duration_ms: float,
) -> CaseResult:
    """Return the single ``error`` result a lost target contributes to one check.

    The reason is read from the error's type rather than its message, so the machine-readable
    cause is exact while ``decoded["error_text"]`` still carries the adapter's own diagnostic
    verbatim.  The configured ``timeout_s`` of the target is recorded alongside it when the
    configuration declares that target, so a reader of the evidence can tell a slow target
    from a hung one without going back to ``qrcheck.toml``.

    Args:
        check_id: The check that could not run, which is the identifier the result is filed under:
            a per-check result keeps the registry-reachability gate honest, because every result
            then names a registered check.
        error: The error the target raised; it reaches here unchanged from the check layer.
        context: The run's context, read for the target name, the seed and the evidence store.
        duration_ms: How long the case took before the target was lost; ``0.0`` for a check
            that was never asked to produce a case.

    Returns:
        The ``error`` result, with its evidence directory when the store wrote one, or -- when
        the store could not write it -- the evidence-fault error result that replaces it in the
        same case (:func:`_persist`).
    """
    reason = (
        Reason.TARGET_TIMEOUT
        if isinstance(error, TargetTimeout)
        else Reason.TARGET_UNAVAILABLE
    )
    decoded: dict[str, object] = {
        "adapter": context.adapter.name,
        "error_type": type(error).__name__,
        "error_text": str(error),
    }
    timeout_s = _configured_timeout(context.config, context.adapter.name)
    if timeout_s is not None:
        decoded["timeout_s"] = timeout_s
    result = CaseResult(
        check_id=check_id,
        case_id=TARGET_LOSS_CASE_ID,
        status=Status.ERROR,
        reason=reason,
        duration_ms=duration_ms,
        request={},
        decoded=decoded,
        expected={"outcome": "reachable_target"},
        seed=context.seed,
    )
    return _persist(context.evidence, result)


def _internal_error_result(
    check_id: str,
    case_id: str,
    error: BaseException,
    context: RunContext,
    *,
    duration_ms: float,
) -> CaseResult:
    """Return the ``error`` result an exception escaping a case is contained as.

    The traceback is captured with :func:`traceback.format_exc` and stored under
    :data:`qrcheck.report.model.TRACEBACK_KEY`, the one key the model normalises so that no
    artefact records the directory the run happened in: the result keeps naming the module
    and the line that failed, on any clone and any machine.

    Args:
        check_id: The check whose case raised.
        case_id: The case that raised, kept so the failure is filed under its own case rather
            than under a synthetic one.
        error: The exception itself, quoted in the result's ``error_type``.
        context: The run's context, read for the seed and the evidence store.
        duration_ms: How long the case ran before it raised.

    Returns:
        The ``error`` result, with its evidence directory when the store wrote one, or -- when
        the store could not write it -- the evidence-fault error result that replaces it in the
        same case (:func:`_persist`).
    """
    result = CaseResult(
        check_id=check_id,
        case_id=case_id,
        status=Status.ERROR,
        reason=Reason.INTERNAL_ERROR,
        duration_ms=duration_ms,
        request={},
        decoded={
            "error_type": type(error).__name__,
            TRACEBACK_KEY: traceback.format_exc(),
        },
        expected={"outcome": "completed_case"},
        seed=context.seed,
    )
    return _persist(context.evidence, result)


def _persist(store: EvidenceStore, result: CaseResult) -> CaseResult:
    """Persist `result` and return it, containing an evidence refusal as an error result.

    The containment this layer owes its callers, at the point where the fault
    happens.  :class:`~qrcheck.report.evidence.EvidenceStore` documents three refusal modes, and
    every one of them is contained here rather than only the write fault: ``OSError`` when the case
    directory or one of its artefacts cannot be written -- a full disk, a permission that changed
    under the run, a path component that became a file, or a symlink in the way of it -- plus the
    ``ValueError`` and ``TypeError`` of an artefact it will not serialise, with a
    ``UnicodeEncodeError`` contained as the ``ValueError`` it is and needing no arm of its own.
    Letting a refusal escape would abort the walk with every result accumulated so far discarded,
    so ``qrcheck.cli.run`` would report an empty stream over a run that had produced plenty -- and
    a value a target influenced is enough to reach one, which is exactly the case that must not
    cost the run its report.  Contained, the fault becomes one ``error`` result in its own case,
    the walk continues with the next case, and the report the other results describe is still
    written -- or, when the report directory itself is what failed, still streamed as JSON lines,
    so the run continues and the process still exits 3 at the end.

    A containment and never a retry: the store is called exactly once per result whether it
    succeeds or fails, so no check's declared adapter-invocation count changes and no retry
    path appears.  Every result this layer builds takes this path -- the
    two error builders and the pre-execution skips -- and a result a check built is persisted by
    the check itself, inside its own case, where the same refusal reaches :func:`_run_check`
    as a case fault.

    Args:
        store: The run's evidence store.
        result: The result this layer built, already carrying its ``check_id``, ``case_id``,
            ``duration_ms`` and ``seed``.

    Returns:
        ``result`` itself when the store wrote its evidence or declined to, or the ``error``
        result :func:`_evidence_fault_result` builds in its place when the store raised.
    """
    try:
        return _with_evidence(store, result)
    except (OSError, ValueError, TypeError) as error:
        return _evidence_fault_result(result, error, traceback.format_exc())


def _evidence_fault_result(
    result: CaseResult,
    error: OSError | ValueError | TypeError,
    traceback_text: str,
) -> CaseResult:
    """Return the ``error`` result that replaces `result` when its evidence could not be written.

    The case keeps the one result it is entitled to rather than gaining a second: a run must
    never hold two results for one ``(check_id, case_id)`` pair, because that pair is the case's
    identity and the failure ID and evidence directory are derived from it.  So the outcome the
    case had already reached is *replaced*, not repeated -- ``check_id``, ``case_id``,
    ``duration_ms`` and ``seed`` are carried over unchanged, ``evidence_dir`` is unset because
    no evidence exists, and ``decoded`` records both the fault and the outcome it displaced.
    That last part is why the displaced outcome is spelled out rather than dropped: one
    ``reason`` cannot say both "the framework failed" and "this case had already skipped" or
    "this case had already failed", so a reader of ``summary.json`` would otherwise be unable to
    tell a lost failing case from a lost skipped one.

    Args:
        result: The result whose evidence the store could not write.
        error: The refusal the store raised, one of the three types :func:`_persist` contains; its
            type name reaches ``decoded["error_type"]``, so a reader can tell a write fault from a
            value the store would not serialise.
        traceback_text: The traceback captured at the fault, stored under
            :data:`qrcheck.report.model.TRACEBACK_KEY` -- the one key the result model
            normalises, so no artefact records the directory the run happened in.

    Returns:
        The replacement ``error`` result -- status ``error``, reason
        :attr:`qrcheck.report.model.Reason.INTERNAL_ERROR` -- whose ``expected`` names the write
        the case was prevented from reaching.
    """
    decoded: dict[str, object] = {
        "error_type": type(error).__name__,
        TRACEBACK_KEY: traceback_text,
        "prevented_status": result.status.value,
        "prevented_reason": None if result.reason is None else result.reason.value,
    }
    return dataclasses.replace(
        result,
        status=Status.ERROR,
        reason=Reason.INTERNAL_ERROR,
        decoded=decoded,
        expected={"outcome": "evidence_recorded"},
        evidence_dir=None,
    )


def _with_evidence(store: EvidenceStore, result: CaseResult) -> CaseResult:
    """Persist `result` through the evidence store and return it carrying its directory.

    The one place this module writes anything, and it writes through the store rather than
    naming a file: the layout, the closed artefact set and the sanitization belong to
    :mod:`qrcheck.evidence.naming` and the persistence policy to
    :class:`~qrcheck.report.evidence.EvidenceStore`.  Results **this** layer builds take this
    path: target-loss, internal-error and pre-execution skips.  A result a check
    returned already carries the directory
    :func:`qrcheck.checks.support.record_evidence` gave it, together with artefacts -- the
    symbol, the damaged variants, the overlay -- that this layer does not hold.

    Args:
        store: The run's evidence store.
        result: The result to persist; a passing result is left untouched by the store.

    Returns:
        ``result`` itself, or a copy whose ``evidence_dir`` is the directory relative to the
        report directory.

    Raises:
        OSError: The case directory or one of its artefacts cannot be written.
        ValueError: The store refuses the artefact -- an undeclared damage kind, an unknown symbol
            format, or a value it will not serialise as canonical JSON.
        TypeError: An image artefact is not bytes-like, or ``request``/``decoded`` holds a value
            that is not JSON-serialisable.

    All three propagate to :func:`_persist`, its only caller, which contains each of them as an
    ``error`` result so the walk continues and the run's results still reach the report -- or the
    JSON-lines stream an unwritable report directory falls back to.
    """
    directory = store.record(result)
    if directory is None:
        return result
    return dataclasses.replace(result, evidence_dir=directory)


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds elapsed since `started`, never negative.

    The clock is :func:`time.perf_counter`, whose readings are monotonic, so the clamp guards
    a malformed input rather than an expected one; a negative duration would be refused by
    the result model anyway.

    Args:
        started: A :func:`time.perf_counter` reading taken before the work began.

    Returns:
        The elapsed time in milliseconds.
    """
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _configured_timeout(config: Config, adapter_name: str) -> float | None:
    """Return the configured ``timeout_s`` of `adapter_name`, or ``None`` if undeclared.

    A target the configuration does not declare -- the port is only asked for a name, so a
    test double or a target constructed elsewhere has one that no table names -- simply
    contributes no ``timeout_s`` to the evidence instead of failing the run.

    Args:
        config: The resolved configuration.
        adapter_name: The name the adapter reports.

    Returns:
        The configured timeout in seconds, or ``None``.
    """
    target = config.targets.get(adapter_name)
    return None if target is None else float(target.timeout_s)


# ---------------------------------------------------------------------------
# Run identity, the LOC measurement root, and closing the target
# ---------------------------------------------------------------------------


def _run_identifier(
    adapter_name: str,
    ec_level: str,
    seed: int,
    check_ids: tuple[str, ...],
    corpus: tuple[CorpusRecord, ...],
) -> str:
    """Return the run's content hash: the same inputs always give the same identifier.

    The identity is over what the run *was*, never over what it produced: the target name,
    the seed, the EC level, the selected check identifiers and a digest of the filtered
    corpus.  A ``--class`` run therefore has a different identity from a whole-corpus one with
    the same seed, and two runs of one configuration agree field for field.  Nothing here
    reads a clock, a UUID source, ``hash()`` or a random generator, and the canonical encoding
    fixes the JSON form (sorted keys, no whitespace, ASCII-escaped) so the digest cannot
    depend on a Python version's separators.

    Args:
        adapter_name: The name the adapter reports.
        ec_level: The requested error-correction level.
        seed: The run's seed.
        check_ids: The selected check identifiers, in any order.
        corpus: The filtered records, in any order; each contributes its ``id`` and a digest
            of its payload, so two corpora that differ only in payload still differ.

    Returns:
        The SHA-256 hex digest of the canonical encoding.
    """
    corpus_digest = hashlib.sha256()
    for record in sorted(corpus, key=lambda item: item.id):
        corpus_digest.update(record.id.encode("utf-8"))
        corpus_digest.update(hashlib.sha256(record.payload_bytes).digest())
    identity = {
        "adapter": adapter_name,
        "checks": sorted(check_ids),
        "corpus": corpus_digest.hexdigest(),
        "ec_level": ec_level,
        "seed": seed,
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _repository_root() -> Path:
    """Return the directory :func:`qrcheck.loc.count_lines` measures.

    The root is derived from a module's own location, never from the current working
    directory: ``summary.json`` carries the resulting ``loc_count``, and a value that changed
    with the shell's directory would break the determinism promise the report makes.  The
    package directory's parent is the repository root under the package's flat layout.

    The anchor is ``qrcheck.__file__``.  It is ``None`` while the package is only a namespace
    package -- a checkout in which ``qrcheck/__init__.py`` has not been written yet -- so this
    module's own ``__file__``, which sits directly inside ``qrcheck/``, is the fallback; both
    derivations name the same directory once the package's ``__init__`` exists.

    Returns:
        The repository root.
    """
    package_file = getattr(qrcheck, "__file__", None)
    if package_file:
        return Path(package_file).resolve().parent.parent
    return Path(__file__).resolve().parent.parent


def _close_adapter(adapter: Adapter) -> None:
    """Release the target, and never let a close failure replace the run's outcome.

    A close failure is swallowed deliberately.  The run has already been measured, and an
    ``error`` result for it would have to name a check: attributing a transport teardown fault
    to a registered check would make the report claim that check produced it, and inventing an
    unregistered identifier would break the gate that every result names a registered check.
    The failures a close can raise for the ordinary reasons are outside the adapter contract
    anyway -- :meth:`qrcheck.targets.base.Adapter.close` is required to be safe to call twice
    and not to raise once its child or socket is already gone -- so what is left is a fault
    the run cannot act on and must not report as a check's verdict.

    Args:
        adapter: The target to release.

    Returns:
        ``None``, whatever happened.
    """
    try:
        adapter.close()
    except Exception:  # See the docstring: it must not replace the run.
        return

