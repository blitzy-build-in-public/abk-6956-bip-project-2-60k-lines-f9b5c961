"""The product's own JUnit XML writer: one deterministic ``xunit2`` document per run.

This is the artefact a CI system gates on, so it is written from the in-memory
:class:`~qrcheck.report.model.RunResult` and never by re-reading ``summary.json``. Two runs of the
same workload differ only inside the ``time`` attributes: no hostname, no timestamp, no absolute
path and no locale-dependent formatting ever reach the document.

Shape: ``testsuites > testsuite name="qrcheck" > testcase``, one ``testcase`` per case result in the
order :class:`~qrcheck.report.model.RunResult` stores them -- sorted by ``(check_id, case_id)``, the
order ``summary.json`` presents the same results in. The five aggregate attributes ``tests``,
``failures``, ``errors``, ``skipped`` and ``time`` appear on the suite, as the ``xunit2`` shape
requires, and on the root for the consumers that read it; both copies come from one mapping and the
tallies from :class:`~qrcheck.report.model.Counts`, so this file cannot contradict ``summary.json``.
A passed case carries **no** child element; a skipped case exactly one ``<skipped message="..."/>``,
a failed case one ``<failure message="...">`` with the decoded-versus-expected detail, an errored
case one ``<error message="...">`` with the traceback -- so a skip never reads as a pass.

Deliberately absent: ``hostname``, ``timestamp``, ``properties``, ``system-out``/``system-err``,
``package``, ``file``, ``line``, ``assertions`` and ``id``. pytest's own ``--junitxml`` emits the
first two, and omitting them confines the volatile content to ``time``, the one attribute a
determinism comparison masks.

XML 1.0 safety: the corpus deliberately contains NUL bytes, right-to-left text and four-byte emoji,
arriving here inside a case identifier, a decoded payload or a traceback. ``ElementTree`` escapes
markup but emits control characters raw, so every text node and attribute value passes through
:func:`_xml_text` first.

Layering: the one fact this writer needs that the result model does not carry is a check's family,
the ``classname`` suffix, owned by :mod:`qrcheck.registry.check` and read from the contract
:data:`~qrcheck.registry.check.REGISTRY` holds for the identifier, otherwise from
:data:`~qrcheck.registry.check.FAMILY_BY_PREFIX`, which maps the identifier's own prefix segment
(``RT`` in ``RT-ROUNDTRIP``) to its family. That table is a constant of the registry module, so a
process that has not imported the check families still emits ``classname="qrcheck.<family>"`` rather
than a placeholder; both lookups sit inside :func:`_family_for`, so importing this
module pulls no registry into such a process. Nothing here prints, logs, reads a clock, the
environment or the filesystem, and nothing retries.
"""

from pathlib import Path
from typing import Final
from xml.etree import ElementTree as ET

from qrcheck.report.json_report import dumps_canonical
from qrcheck.report.model import TRACEBACK_KEY, CaseResult, RunResult, Status

__all__ = ["write_junit"]

# The XML 1.0 §2.2 ``Char`` production admits #x9, #xA, #xD, #x20-#xD7FF, #xE000-#xFFFD and
# #x10000-#x10FFFF, and nothing else. Excluded: the C0 controls other than tab/LF/CR (the corpus
# really does carry an embedded NUL), the surrogate range that cannot even be encoded as UTF-8,
# and the two non-characters at the top of the BMP. Every code point outside the production is
# *replaced* rather than dropped, because dropping would silently shorten a case identifier that a
# CI trend line keys on.
_ILLEGAL_CODEPOINTS: Final[tuple[int, ...]] = (
    *range(0x00, 0x09),
    0x0B,
    0x0C,
    *range(0x0E, 0x20),
    *range(0xD800, 0xE000),
    0xFFFE,
    0xFFFF,
)

# Built once at import and reused for every node, so sanitizing a document costs one
# ``str.translate`` call per string and no regular expression. U+FFFD is the replacement
# character, which keeps the text's length readable while making the document well-formed.
_ILLEGAL_XML_CHARS: Final[dict[int, str]] = str.maketrans(
    {codepoint: "\ufffd" for codepoint in _ILLEGAL_CODEPOINTS}
)

# ``testsuite name="qrcheck"`` is the suite identity every consumer keys on, and the ``classname``
# of a case is this prefix followed by its check's family.
_SUITE_NAME: Final[str] = "qrcheck"
_CLASSNAME_PREFIX: Final[str] = "qrcheck."

# The family reported for an identifier whose prefix appears in neither the registry nor
# ``FAMILY_BY_PREFIX``. Unreachable for the twenty-one registered identifiers: each of their nine
# prefixes is a key of that table, which resolves whether or not the check family modules have
# imported. A writer must nevertheless never be the thing that loses a run's results, so a
# stranger degrades its ``classname`` rather than failing the document.
_UNKNOWN_FAMILY: Final[str] = "unknown"


def write_junit(run_result: RunResult, path: Path) -> Path:
    """Write ``run_result`` as a JUnit ``xunit2`` document to ``path`` and return ``path``.

    One ``testcase`` per result, in the stored ``(check_id, case_id)`` order, each carrying
    ``classname`` (``qrcheck.<family>``), ``name`` (``<CHECK-ID>[<case-id>]``, the two identifiers
    verbatim: a display string, not a path) and ``time`` in seconds to three decimals. The document
    is serialised compactly with an XML declaration and exactly one trailing newline, and written
    directly rather than through a temporary file and a rename: a stray file inside the report
    directory would be an artefact outside the closed evidence layout that ``CL-ARTIFACTS`` inspects
    for.

    Args:
        run_result: The completed run; its results are stored already sorted and are not re-sorted.
        path: Destination file, normally ``<report_dir>/junit.xml``, used exactly as given.

    Returns:
        ``path``, unchanged, so a caller can record what it wrote.

    Raises:
        TypeError: a result field holds a value :func:`dumps_canonical` cannot serialise.
        ValueError: a result mapping holds a non-finite float (``nan``, ``inf`` or ``-inf``).
        OSError: the parent directory cannot be created or the file cannot be written, propagated
            untouched for ``qrcheck.cli.run`` to turn into exit code 3.
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

    The three attributes are set in the fixed order ``classname``, ``name``, ``time`` and the child
    element, if any, is appended last, so the serialised bytes are stable for a given run.
    ``CaseResult`` guarantees that every status other than ``passed`` carries a reason, which is why
    ``case.reason.value`` is read without a ``None`` branch.
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

    ``tests`` is the number of results, which ``RunResult`` has already proven equal to the sum of
    the tallies; the three outcome counts are read from :class:`~qrcheck.report.model.Counts` rather
    than recounted, so this document and ``summary.json`` cannot describe the same run differently.
    ``time`` is the total of the case durations, summed in milliseconds once and divided once --
    summing pre-rounded strings would let rounding error accumulate.
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

    ``time`` attributes are the only volatile content of the document, and a fixed three-decimal
    rendering is what makes them comparable character by character once a determinism check masks
    them.
    """
    return f"{milliseconds / 1000.0:.3f}"


def _family_for(check_id: str) -> str:
    """Return the check family that owns ``check_id``, or ``"unknown"`` for a stranger.

    A contract registered under ``check_id`` supplies the family outright -- registration rejects an
    identifier whose prefix belongs to another family, so a registered family and the prefix table
    can never disagree -- and otherwise :data:`~qrcheck.registry.check.FAMILY_BY_PREFIX` supplies
    the family of the identifier's own prefix segment, the text before the first ``-`` (``RT`` for
    ``RT-ROUNDTRIP``). That table is a constant of the registry module while
    :data:`~qrcheck.registry.check.REGISTRY` is populated only as a side effect of the check family
    modules importing, so the second step gives a process that has not imported them the real family
    instead of a placeholder. The import sits **inside** this function because
    :mod:`qrcheck.registry.check` depends on :mod:`qrcheck.report.model`: a module-scope import
    would close the report-to-registry cycle. The family returned becomes the JUnit ``classname``
    suffix, and ``"unknown"`` covers an identifier whose prefix is in neither source, or a
    non-``str``, both unreachable for a registered check.
    """
    from qrcheck.registry.check import FAMILY_BY_PREFIX, REGISTRY

    contract = REGISTRY.get(check_id)
    if contract is not None:
        return contract.family
    if not isinstance(check_id, str):
        # ``CaseResult`` does not restrict the field to ``str``; such a value has no
        # prefix segment and degrades exactly as it did while the registry was the only
        # source, rather than raising out of the writer.
        return _UNKNOWN_FAMILY
    return FAMILY_BY_PREFIX.get(check_id.split("-", 1)[0], _UNKNOWN_FAMILY)


def _failure_body(case: CaseResult) -> str:
    """Return the text of a ``failure`` element: the decoded-versus-expected detail.

    Both mappings are rendered by :func:`qrcheck.report.json_report.dumps_canonical`, the product's
    only JSON formatter, so a failure body and the evidence beside it agree character for character.
    ``expected`` may be ``None``, which renders as ``null`` -- an honest "this check asserts nothing
    beyond the decoded evidence" rather than an omitted section.
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

    An internal framework exception reaches the model as an ``error`` result whose ``decoded``
    carries the traceback under :data:`qrcheck.report.model.TRACEBACK_KEY`, and that text is what a
    reader needs; a result without it still renders its whole decoded mapping, so an error is never
    an empty element. The key is imported from the model rather than spelled out here, because a
    second copy of the name could read a value the model never normalised.
    """
    traceback_text = case.decoded.get(TRACEBACK_KEY)
    if isinstance(traceback_text, str):
        return _xml_text(traceback_text)
    return _xml_text(dumps_canonical(case.decoded))


def _xml_text(value: str) -> str:
    """Return ``value`` with every character XML 1.0 cannot carry replaced by ``U+FFFD``.

    Implements the ``Char`` production of XML 1.0 §2.2 (see :data:`_ILLEGAL_CODEPOINTS`).
    ``ElementTree`` escapes markup but not control characters, so without this filter a case
    identifier with an embedded NUL -- which the ``nul-byte`` corpus class produces on purpose --
    would yield a document no parser accepts. The length of the text is preserved.
    """
    return value.translate(_ILLEGAL_XML_CHARS)
