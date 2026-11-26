"""The product's own JUnit XML writer: one deterministic ``xunit2`` document per run.

This is the artefact a CI system gates on, so it is written from the in-memory
:class:`~qrcheck.report.model.RunResult` and never by re-reading ``summary.json``. Two runs
of the same workload produce files whose bytes differ only inside the ``time`` attributes:
no hostname, no timestamp, no absolute path and no locale-dependent formatting ever reach
the document, and every collection it iterates is already ordered.

Shape
-----
``testsuites > testsuite name="qrcheck" > testcase``, one ``testcase`` per case result, in
the order :class:`~qrcheck.report.model.RunResult` stores them - sorted by
``(check_id, case_id)`` - which is by construction the order ``summary.json`` presents the
same results in (AAP §0.5.3).

The five aggregate attributes are ``tests``, ``failures``, ``errors``, ``skipped`` and
``time``. They appear on the suite, which is what AAP §0.3.4 fixes, and again on the root,
which most JUnit consumers read; the two copies are written from one mapping so they can
never disagree, and the tallies come from :class:`~qrcheck.report.model.Counts` rather than
from a recount here, so this file cannot contradict ``summary.json`` about the same run.

A passed case carries **no** child element: an empty ``<testcase ... />`` is how JUnit
records success. A skipped case carries exactly one ``<skipped message="..."/>``, a failed
case exactly one ``<failure message="...">`` whose text is the decoded-versus-expected
detail, and an errored case exactly one ``<error message="...">`` whose text is the
traceback the runner recorded. A skip therefore never reads as a pass (AAP §0.3.4), because
it is neither element-less nor counted in ``failures``.

Deliberately absent
-------------------
No ``hostname``, no ``timestamp``, no ``properties``, no ``system-out``/``system-err``, and
no ``package``, ``file``, ``line``, ``assertions`` or ``id`` (AAP §0.3.4, §0.5.5). pytest's
own ``--junitxml`` emits ``hostname`` and ``timestamp``; omitting both is what makes this
file reproducible, and it confines the volatile content to ``time`` - which is also why a
determinism comparison masks that one attribute and nothing else (AAP §0.5.4).

XML 1.0 safety
--------------
The corpus deliberately contains embedded NUL bytes, right-to-left text and four-byte emoji,
and those characters arrive here inside a case identifier, a decoded payload or a
traceback. ``ElementTree`` escapes ``&``, ``<``, ``>`` and quotes but emits control
characters raw, so a document carrying a NUL would be ill-formed and no parser would accept
it. Every text node and every attribute value therefore passes through :func:`_xml_text`
first.

Layering
--------
The one fact this writer needs that the result model does not carry is a check's family,
which is the ``classname`` suffix. That family is owned by
:mod:`qrcheck.registry.check`, the single owner of check identity, and it is looked up
there - never re-derived from an identifier prefix and never tabulated here. The lookup is
deferred into :func:`_family_for` so that this module can be imported without pulling the
registry, and through it the target port, into a process that only wants to write XML.

Nothing here prints, logs, reads a clock, the environment or the filesystem, and nothing
retries. The only effect is the file :func:`write_junit` is handed.
"""

from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

from qrcheck.report.json_report import dumps_canonical
from qrcheck.report.model import CaseResult, RunResult, Status

__all__ = ["write_junit"]

# The XML 1.0 §2.2 ``Char`` production admits #x9, #xA, #xD, #x20-#xD7FF, #xE000-#xFFFD and
# #x10000-#x10FFFF, and nothing else. Excluded: the C0 controls other than tab/LF/CR (the
# corpus really does carry an embedded NUL), the surrogate range that cannot even be
# encoded as UTF-8, and the two non-characters at the top of the BMP. Every code point
# outside the production is *replaced* rather than dropped, because dropping would silently
# shorten a case identifier that a CI trend line keys on.
_ILLEGAL_CODEPOINTS: Final[tuple[int, ...]] = (
    *range(0x00, 0x09),
    0x0B,
    0x0C,
    *range(0x0E, 0x20),
    *range(0xD800, 0xE000),
    0xFFFE,
    0xFFFF,
)

# Built once at import and reused for every node, so sanitising a document costs one
# ``str.translate`` call per string and no regular expression. U+FFFD is the replacement
# character, which keeps the text's length readable while making the document well-formed.
_ILLEGAL_XML_CHARS: Final[dict[int, str]] = str.maketrans(
    {codepoint: "\ufffd" for codepoint in _ILLEGAL_CODEPOINTS}
)

# The key an errored result carries its traceback under. The runner and
# ``qrcheck.checks.support`` put it in ``decoded["traceback"]``; that key is the agreed
# contract between the two, so it is named once here rather than spelled out per call site.
_TRACEBACK_KEY: Final[str] = "traceback"

# ``testsuite name="qrcheck"`` is the suite identity every consumer keys on, and the
# ``classname`` of a case is this prefix followed by its check's family.
_SUITE_NAME: Final[str] = "qrcheck"
_CLASSNAME_PREFIX: Final[str] = "qrcheck."

# The family reported for an identifier the registry does not hold. Unreachable in a real
# run (every executed identifier is registered), but a writer must never be the thing that
# loses a run's results, so an unregistered identifier degrades its ``classname`` instead.
_UNKNOWN_FAMILY: Final[str] = "unknown"


def write_junit(run_result: RunResult, path: Path) -> Path:
    """Write ``run_result`` as a JUnit ``xunit2`` document to ``path`` and return ``path``.

    One ``testcase`` per result, in the stored ``(check_id, case_id)`` order, each carrying
    ``classname`` (``qrcheck.<family>``), ``name`` (``<CHECK-ID>[<case-id>]``, the two
    identifiers verbatim: this is a display string, not a path) and ``time`` in seconds to
    three decimals. A passed case gets no child element; a skipped, failed or errored one
    gets exactly one, so a skip can never be read as a pass.

    The document is serialised compactly with an XML declaration and exactly one trailing
    newline, and it is written directly rather than through a temporary file and a rename: a
    stray file inside the report directory would be an artefact outside the closed evidence
    layout, which is exactly what ``CL-ARTIFACTS`` inspects for (AAP §0.3.4).

    Args:
        run_result: The completed run. Its results are already sorted by
            :class:`~qrcheck.report.model.RunResult`, so they are not re-sorted here; a
            second ordering rule for the same artefact would let this file and
            ``summary.json`` disagree.
        path: Destination file, normally ``<report_dir>/junit.xml``, used exactly as given
            and never resolved to an absolute path here.

    Returns:
        ``path``, unchanged, so a caller can record what it wrote.

    Raises:
        TypeError: a result field holds a value that :func:`dumps_canonical` cannot
            serialise, which is a check's defect and must fail loudly.
        OSError: the parent directory cannot be created or the file cannot be written.
            Propagated untouched - ``qrcheck.cli.run`` is the single place that turns an
            unwritable report directory into exit code 3, and the product has no retry path
            anywhere (AAP §0.3.4, §0.5.6).
    """
    aggregate = _aggregate_attributes(run_result)
    root = ET.Element("testsuites", aggregate)
    suite = ET.SubElement(root, "testsuite", {"name": _SUITE_NAME, **aggregate})
    for case in run_result.results:
        _append_testcase(suite, case)
    document = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(document + b"\n")
    return path


def _append_testcase(suite: ET.Element, case: CaseResult) -> None:
    """Append one ``testcase`` for ``case`` to ``suite``, with its status child.

    The three attributes are set in the fixed order ``classname``, ``name``, ``time`` and
    the child element, if any, is appended last, so the serialised bytes are stable for a
    given run. ``CaseResult`` guarantees that every status other than ``passed`` carries a
    reason, which is why ``case.reason.value`` is read without a ``None`` branch.
    """
    testcase = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": _xml_text(f"{_CLASSNAME_PREFIX}{_family_for(case.check_id)}"),
            "name": _xml_text(f"{case.check_id}[{case.case_id}]"),
            "time": _seconds(case.duration_ms),
        },
    )
    if case.status is Status.PASSED:
        # Success is the absence of a child element: nothing more may be written here.
        return
    message = {"message": _xml_text(case.reason.value)}
    if case.status is Status.SKIPPED:
        ET.SubElement(testcase, "skipped", message)
    elif case.status is Status.FAILED:
        ET.SubElement(testcase, "failure", message).text = _failure_body(case)
    elif case.status is Status.ERROR:
        ET.SubElement(testcase, "error", message).text = _error_body(case)


def _aggregate_attributes(run_result: RunResult) -> dict[str, str]:
    """Return the five aggregate attributes, in the order they are serialised.

    ``tests`` is the number of results, which ``RunResult`` has already proven equal to the
    sum of the tallies; the three outcome counts are read from
    :class:`~qrcheck.report.model.Counts` rather than recounted, so this document and
    ``summary.json`` cannot describe the same run differently. ``time`` is the total of the
    case durations, summed in milliseconds once and divided once - summing pre-rounded
    strings would let rounding error accumulate into the total.
    """
    counts = run_result.counts
    total_ms = sum(case.duration_ms for case in run_result.results)
    return {
        "tests": str(len(run_result.results)),
        "failures": str(counts.failed),
        "errors": str(counts.error),
        "skipped": str(counts.skipped),
        "time": _seconds(total_ms),
    }


def _seconds(milliseconds: float) -> str:
    """Return ``milliseconds`` as whole seconds with exactly three decimals.

    ``time`` attributes are the only volatile content of the document, and a fixed
    three-decimal rendering is what makes them comparable character by character once a
    determinism check masks them (AAP §0.5.4).
    """
    return f"{milliseconds / 1000.0:.3f}"


def _family_for(check_id: str) -> str:
    """Return the check family that owns ``check_id``, or ``"unknown"`` for a stranger.

    The registry is imported **inside** this function on purpose. It is the single owner of
    check identity (AAP §0.8.1), and it depends on :mod:`qrcheck.report.model`; importing it
    at module scope from a report module would close the report-to-registry cycle as soon as
    the registry needs a report symbol at run time, and would drag the target port into
    every process that merely writes XML. Deferring the import keeps the dependency one-way
    and costs one module lookup for the whole run.

    Args:
        check_id: The check identifier a result carries.

    Returns:
        The owning family name, which becomes the JUnit ``classname`` suffix, or
        ``"unknown"`` when no contract is registered under ``check_id``. The fallback exists
        because a report writer must never lose a run's results over a lookup; in a real run
        it is unreachable, since every executed identifier is registered.
    """
    from qrcheck.registry.check import REGISTRY

    contract = REGISTRY.get(check_id)
    return _UNKNOWN_FAMILY if contract is None else contract.family


def _failure_body(case: CaseResult) -> str:
    """Return the text of a ``failure`` element: the decoded-versus-expected detail.

    The two mappings are rendered by :func:`qrcheck.report.json_report.dumps_canonical`,
    the product's only JSON formatter, so a failure body and the evidence beside it agree
    character for character. ``expected`` may be ``None``, which renders as ``null`` - an
    honest "this check asserts nothing beyond the decoded evidence" rather than an omitted
    section.
    """
    return _xml_text(
        f"check: {case.check_id}\n"
        f"case: {case.case_id}\n"
        f"reason: {case.reason.value}\n"
        f"decoded:\n{dumps_canonical(case.decoded)}\n"
        f"expected:\n{dumps_canonical(case.expected)}"
    )


def _error_body(case: CaseResult) -> str:
    """Return the text of an ``error`` element: the traceback, or the decoded mapping.

    An internal framework exception reaches the model as an ``error`` result whose
    ``decoded`` carries the traceback under ``traceback``, and that text is what a reader
    needs verbatim. A result without it still renders its whole decoded mapping, so an
    error is never reported as an empty element.
    """
    traceback_text = case.decoded.get(_TRACEBACK_KEY)
    if isinstance(traceback_text, str):
        return _xml_text(traceback_text)
    return _xml_text(dumps_canonical(case.decoded))


def _xml_text(value: str) -> str:
    """Return ``value`` with every character XML 1.0 cannot carry replaced by ``U+FFFD``.

    Implements the ``Char`` production of XML 1.0 §2.2 (see
    :data:`_ILLEGAL_CODEPOINTS`). ``ElementTree`` escapes markup but not control characters,
    so without this filter a case identifier with an embedded NUL - which the ``nul-byte``
    corpus class produces on purpose - would yield a document no parser accepts.

    Args:
        value: Any text destined for an attribute value or a text node.

    Returns:
        The same text with each illegal code point replaced; the length is preserved.
    """
    return value.translate(_ILLEGAL_XML_CHARS)
