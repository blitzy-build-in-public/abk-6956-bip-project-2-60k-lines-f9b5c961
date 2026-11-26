"""The ``--checks`` grammar, the ``--class`` filter and the two empty-result diagnostics.

AAP §0.3.4 ("Identity and selection") fixes three token forms for ``--checks`` -- an exact
identifier (``RT-ROUNDTRIP``), an identifier glob (``SY-*``) and a family selector
(``family:render``) -- makes a selection that matches nothing an exit-2 condition that echoes
the selector **verbatim** with no report written, and makes ``--class`` a repeatable glob over
the corpus record's class name whose zero-record outcome is a second, differently worded exit-2
condition.  AAP §0.6.6 rows 1 and 2 name this module as the gate for exactly those two states,
and §0.11.1 lists it as a deliverable.  Four properties are pinned:

* **the three token forms resolve as the plan says**, are unioned, de-duplicated and returned
  **sorted by identifier ascending**.  A report whose ``results[]`` order depended on how the
  user typed ``--checks`` would break the determinism promise of AAP §0.5.3, so the sort is
  asserted against a selector whose token order is deliberately the reverse of the answer;
* **every token must select something.**  A mistyped token raises :class:`SelectionError`
  carrying the selector as typed instead of being dropped, because a dropped token would stop
  a check from executing while the run still reported a clean pass -- the vacuous suite
  AAP §0.4.1 forbids;
* **the two empty states are proved separately**, since collapsing them into one message would
  lose the "name the filter" half of §0.6.6 row 2: ``--checks NOPE-*`` names the token,
  ``--class 'no-such-*'`` names the glob, and neither writes a report;
* **``--class 'boundary-*'`` selects the 40-record boundary subset** that success criterion 2
  runs at EC L, Q and H, and repeated ``--class`` options combine as a union.

Registration, and why it is forced here
---------------------------------------
``qrcheck.registry`` deliberately leaves :data:`~qrcheck.registry.check.REGISTRY` empty when it
is imported: a package that populated it would make the executed set depend on whether some
unrelated module happened to import it (AAP §0.5).  Registration therefore belongs to the entry
points, so this module imports :func:`qrcheck.checks.load_families` once and asserts the
registry holds exactly ``CHECK_IDS`` before any selector is resolved -- a grammar test written
against a half-populated registry would pass while selecting nothing.

What this module deliberately does not do
-----------------------------------------
It decides no exit code: ``qrcheck.registry.selection`` raises ``SelectionError`` and
``qrcheck.cli.run`` maps it to exit 2, so both halves are asserted separately -- the raise here,
the process status through the real command line.  It re-implements neither policy: the corpus
filtering is ``qrcheck.corpus.io.filter_by_class``'s and the zero-record condition is
``qrcheck.runner``'s, which is why the exit-2 assertion for an empty ``--class`` belongs to the
CLI case rather than to a filter unit test.  No threshold, sample count or timeout appears
anywhere in this file -- the only numbers are contract facts from the plan: the exit code, the
40-record boundary subset, the 42-record ``url`` + ``otp`` union, and the twenty-one
identifiers.  Nothing is marked ``skip`` or ``xfail``ed and nothing is retried: the product has
no retry path (AAP §0.5.6), and a flaky selection would be a defect to fix rather than to
suppress.

Cost and determinism
--------------------
The grammar cases touch the registry only -- no target, no decoder, no image.  The three
command-line cases are held to the same rule: the two exit-2 paths stop before any case runs,
and the one run that executes is restricted with ``--checks RT-ROUNDTRIP`` and two ``--class``
globs, so it exercises 42 cases instead of the 1,283 of a full run.  Every expectation is
computed from sorted data or from the plan's own tables, never from a set's iteration order.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from qrcheck.checks import load_families
from qrcheck.cli.main import cli
from qrcheck.corpus.builder import (
    BOUNDARY_CLASSES,
    BOUNDARY_RECORDS,
    CLASS_COUNTS,
    CorpusRecord,
)
from qrcheck.corpus.io import filter_by_class
from qrcheck.registry.check import (
    CHECK_IDS,
    FAMILY_NAMES,
    REGISTRY,
    CheckContract,
    iter_checks,
    registered_ids,
)
from qrcheck.registry.selection import (
    Selection,
    SelectionError,
    matches,
    parse_selection,
    select,
)

if TYPE_CHECKING:  # Annotation-only: the CLI framework is not imported by this module.
    from click.testing import CliRunner, Result

# The nine families and the checks each owns, from AAP §0.8.1's component-to-check-family
# mapping.  This is the only place in this file where check identifiers are written as
# literals: every other expectation below is computed from the registry through ``matches`` or
# ``iter_checks``, so a check that is added, renamed or moved between families fails in one
# named place instead of in a dozen, and the literal sets stay reviewable in a diff.  Each
# tuple is in ascending identifier order, which is the order ``select`` promises.
FAMILY_CHECKS: Final[Mapping[str, tuple[str, ...]]] = {
    "cleanup": ("CL-ARTIFACTS",),
    "damage": ("DM-RESILIENCE", "DM-SEEDED"),
    "determinism": ("DT-DETERMINISTIC",),
    "invalid": ("IV-BAD-OPTION", "IV-EMPTY", "IV-OVERCAPACITY"),
    "mode": ("MD-CHARSET", "MD-MODE-OPTIMAL"),
    "performance": ("PF-DECODE", "PF-ENCODE"),
    "render": (
        "RN-CONTRAST",
        "RN-LOGO",
        "RN-MODULE-SIZE",
        "RN-QUIET-ZONE",
        "RN-SVG",
    ),
    "roundtrip": ("RT-ROUNDTRIP",),
    "symbol": ("SY-EC-MATCH", "SY-MASK", "SY-MATRIX-EQUIV", "SY-VERSION-MIN"),
}

# The exit code AAP §0.3.4 assigns to a configuration, selection or corpus-filter fault.  A
# literal on purpose: ``qrcheck.cli.exit`` owns the vocabulary, and a test that imported the
# constant it is checking would follow a renumbering instead of catching it.
EXIT_CONFIG: Final[int] = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _populated_registry() -> tuple[str, ...]:
    """Import the nine check families once so the registry holds all twenty-one checks.

    Autouse and module-scoped because every case in this file resolves selectors against the
    real registry: a grammar assertion made against an empty one would pass while selecting
    nothing, which is the failure mode :func:`select` itself exists to prevent.

    Returns:
        Every registered identifier, ascending.

    Raises:
        AssertionError: The registry does not hold exactly :data:`CHECK_IDS`.
    """
    identifiers = load_families()
    assert identifiers == CHECK_IDS, (
        "the selection grammar is asserted against the full registry; registered "
        f"{list(identifiers)}"
    )
    return identifiers


def _ids(contracts: Sequence[CheckContract]) -> tuple[str, ...]:
    """Return the identifiers of `contracts`, in the order they were returned.

    The order is preserved rather than sorted, so a test can assert that ``select`` returned
    them in identifier order instead of re-establishing that order itself.

    Args:
        contracts: The contracts a selection resolved to.

    Returns:
        Their identifiers.
    """
    return tuple(contract.id for contract in contracts)


def _classes(records: Sequence[CorpusRecord]) -> tuple[str, ...]:
    """Return the distinct class names of `records`, sorted.

    Args:
        records: Records selected from a corpus.

    Returns:
        The class names, ascending and de-duplicated.
    """
    return tuple(sorted({record.class_name for record in records}))


def _report_entries(report_root: Path) -> tuple[str, ...]:
    """Return the entry names under `report_root`, sorted, or ``()`` when it does not exist.

    "No report was written" has to hold for both shapes the command can leave: a report root
    the writability probe created and left empty, and one that was never created at all.

    Args:
        report_root: The ``--report-dir`` the invocation was given.

    Returns:
        The directory's entry names in ascending order, empty when the path is absent.
    """
    if not report_root.exists():
        return ()
    return tuple(sorted(entry.name for entry in report_root.iterdir()))


def _invoke_run(
    runner: CliRunner, config_path: Path, report_root: Path, *arguments: str
) -> Result:
    """Invoke ``qrcheck run`` through the real command line and return click's result.

    ``--config`` is always passed explicitly: the group's default is the repository's own
    ``qrcheck.toml``, whose relative ``report_dir`` would write into the checkout.  ``--report-dir``
    is always passed too, so the "no report written" assertion targets a path this test owns
    instead of whatever the configuration's relative default resolves to.
    ``catch_exceptions=False`` keeps a genuine exception a test failure rather than folding it
    into an exit code the assertion might mistake for the behaviour under test.

    Args:
        runner: The ``CliRunner`` fixture, which keeps stdout and stderr apart.
        config_path: The minimal configuration document written under the test's ``tmp_path``.
        report_root: The report root this invocation is given.
        *arguments: The remaining ``qrcheck run`` options.

    Returns:
        click's result record.
    """
    argv = [
        "--config",
        str(config_path),
        "run",
        "--report-dir",
        str(report_root),
        *arguments,
    ]
    return runner.invoke(cli, argv, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Registration: the registry is full before any selector is resolved
# ---------------------------------------------------------------------------


def test_registry_holds_exactly_the_declared_check_identifiers(
    _populated_registry: tuple[str, ...],
) -> None:
    """The registry holds the twenty-one declared identifiers, ascending, and nothing else.

    ``registered_ids()`` is the set a report's executed checks are compared against, so it has
    to equal the product's own declaration rather than a subset of it; and the nine families of
    :data:`FAMILY_CHECKS` have to account for every one of them, so no check can exist in a
    family the plan does not name.
    """
    assert _populated_registry == CHECK_IDS
    assert registered_ids() == CHECK_IDS
    assert sorted(REGISTRY) == list(CHECK_IDS)
    declared = tuple(
        identifier for family in sorted(FAMILY_CHECKS) for identifier in FAMILY_CHECKS[family]
    )
    assert sorted(declared) == list(CHECK_IDS)


# ---------------------------------------------------------------------------
# Exact identifiers, comma lists, whitespace and duplicates
# ---------------------------------------------------------------------------


def test_an_exact_identifier_selects_that_contract_and_nothing_else() -> None:
    """One exact identifier resolves to the registered contract itself, not to a copy."""
    selected = select(REGISTRY, "RT-ROUNDTRIP")
    assert selected == (REGISTRY["RT-ROUNDTRIP"],)
    assert selected[0] is REGISTRY["RT-ROUNDTRIP"]
    assert _ids(selected) == FAMILY_CHECKS["roundtrip"]


def test_a_comma_list_is_returned_sorted_by_identifier() -> None:
    """A comma list resolves to both contracts, in identifier order whatever the token order.

    The selector is written with the later identifier first: a ``select`` that preserved token
    order would fail here, and a report built from it would carry a different ``results[]``
    order for the same set of checks (AAP §0.5.3).
    """
    selected = select(REGISTRY, "SY-MASK,RT-ROUNDTRIP")
    assert _ids(selected) == ("RT-ROUNDTRIP", "SY-MASK")
    assert _ids(selected) == tuple(sorted(_ids(selected)))


def test_surrounding_whitespace_on_each_term_is_tolerated() -> None:
    """Spaces around a token are trimmed, and the answer is the same as the compact form."""
    spaced = select(REGISTRY, " RT-ROUNDTRIP , SY-MASK ")
    assert _ids(spaced) == _ids(select(REGISTRY, "RT-ROUNDTRIP,SY-MASK"))
    assert _ids(spaced) == ("RT-ROUNDTRIP", "SY-MASK")


def test_duplicate_tokens_do_not_duplicate_results() -> None:
    """The three ways of naming one check collapse into a single result.

    An exact identifier, its family and a glob over it are a union, not a concatenation: a
    duplicate would make one check run -- and one check's cases appear in the report -- twice.
    """
    selected = select(REGISTRY, "RT-ROUNDTRIP,family:roundtrip,RT-*,RT-ROUNDTRIP")
    assert _ids(selected) == FAMILY_CHECKS["roundtrip"]
    assert len(selected) == len(FAMILY_CHECKS["roundtrip"])


@pytest.mark.parametrize("selector", [None, "", "   "], ids=("absent", "empty", "blank"))
def test_an_absent_or_blank_selector_selects_every_check(selector: str | None) -> None:
    """No selector, an empty one and a whitespace-only one all mean "every check".

    ``--checks ""`` cannot be allowed to select nothing: an option a user believed was a
    no-op would silently run an empty suite.
    """
    assert _ids(select(REGISTRY, selector)) == CHECK_IDS


def test_a_trailing_comma_is_tolerated() -> None:
    """A trailing comma leaves an empty token, which is skipped rather than reported."""
    assert _ids(select(REGISTRY, "RT-ROUNDTRIP,")) == FAMILY_CHECKS["roundtrip"]
    assert _ids(select(REGISTRY, "SY-MASK,,RT-ROUNDTRIP")) == ("RT-ROUNDTRIP", "SY-MASK")


def test_a_selector_of_nothing_but_commas_is_an_empty_selection() -> None:
    """Commas with no token between them select nothing, which is exit 2 rather than a run.

    Every token is empty and is skipped, so the union stays empty; the condition reported is the
    empty-selection one and the selector is echoed verbatim, commas included.
    """
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, ",")
    assert raised.value.selector == ","
    assert "matched no registered check" in raised.value.detail


def test_an_empty_registry_cannot_satisfy_a_whole_run_selection() -> None:
    """A registry with no checks reports the empty-selection condition, not a clean run.

    The whole-run selection is the one path that does not go through a token, so it is asserted
    against a registry of its own rather than against the product's -- which must never be
    empty.
    """
    with pytest.raises(SelectionError) as raised:
        select({}, None)
    assert raised.value.selector == ""
    assert "matched no registered check" in raised.value.detail
    assert isinstance(raised.value, ValueError)


# ---------------------------------------------------------------------------
# Identifier globs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "family"),
    [("SY-*", "symbol"), ("RN-*", "render"), ("*-ROUNDTRIP", "roundtrip")],
    ids=lambda value: value,
)
def test_a_glob_token_selects_every_matching_identifier(token: str, family: str) -> None:
    """A glob over identifiers selects exactly the plan's checks of the owning family.

    The expectation is AAP §0.8.1's literal set, so a glob that over- or under-matched -- the
    ``SY-*`` form matching four checks and not two or five -- fails here.
    """
    selected = select(REGISTRY, token)
    assert _ids(selected) == FAMILY_CHECKS[family]
    assert {contract.family for contract in selected} == {family}


def test_a_glob_can_span_families() -> None:
    """A single glob is matched against identifiers, not against families.

    ``?T-*`` names one check from each of two families, and the expectation is computed with
    :func:`matches` -- the product's only token predicate -- so this case pins the predicate's
    reading of a wildcard rather than restating a list.
    """
    expected = tuple(sorted(contract.id for contract in iter_checks() if matches(contract, "?T-*")))
    assert _ids(select(REGISTRY, "?T-*")) == expected
    assert set(expected) == set(FAMILY_CHECKS["determinism"]) | set(FAMILY_CHECKS["roundtrip"])


def test_a_glob_that_matches_no_identifier_raises_naming_the_selector() -> None:
    """An identifier-shaped glob that matches nothing is an error, not an empty selection.

    This is the typo the diagnostic exists for: ``SY-VERSIN-*`` is one character from a real
    family's identifiers, and dropping it silently would stop those checks from running while
    the run still reported success.
    """
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, "SY-VERSIN-*")
    assert raised.value.selector == "SY-VERSIN-*"
    assert "SY-VERSIN-*" in raised.value.detail


# ---------------------------------------------------------------------------
# Family selectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY_CHECKS), ids=lambda value: value)
def test_a_family_selector_selects_that_family_in_identifier_order(family: str) -> None:
    """``family:<name>`` selects exactly the checks AAP §0.8.1 files under that family.

    Every one of the nine families is driven, so a family that matches nothing -- a renamed
    module, a mistyped table entry -- fails instead of silently selecting an empty set.
    """
    selected = select(REGISTRY, f"family:{family}")
    assert _ids(selected) == FAMILY_CHECKS[family]
    assert {contract.family for contract in selected} == {family}
    assert _ids(selected) == tuple(sorted(_ids(selected)))


def test_every_named_family_matches_at_least_one_check() -> None:
    """The plan's family table and the product's family table describe the same nine families."""
    assert sorted(FAMILY_CHECKS) == list(FAMILY_NAMES)
    for family in FAMILY_NAMES:
        assert select(REGISTRY, f"family:{family}"), f"family {family!r} selected nothing"


def test_an_unknown_family_raises_with_the_whole_selection_echoed() -> None:
    """An unknown family name is a typo to report, and the diagnostic quotes what was typed.

    The error is raised by :func:`matches`, which is handed the enclosing selector so that a
    caller reading stderr sees the whole ``--checks`` value rather than a fragment of it.
    """
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, "SY-*,family:renderr")
    assert raised.value.selector == "SY-*,family:renderr"
    assert "family:renderr" in raised.value.detail
    assert isinstance(raised.value, ValueError)


# ---------------------------------------------------------------------------
# Mixtures of the three forms, and case handling
# ---------------------------------------------------------------------------


def test_mixed_tokens_are_unioned_deduplicated_and_sorted() -> None:
    """A selector mixing all three forms resolves to their union, once each, in id order."""
    selector = "family:render,SY-MASK,RT-*"
    selected = select(REGISTRY, selector)
    expected = tuple(sorted({*FAMILY_CHECKS["render"], "SY-MASK", *FAMILY_CHECKS["roundtrip"]}))
    assert _ids(selected) == expected
    assert len(selected) == len(expected)


@pytest.mark.parametrize(
    "selector",
    ["rt-roundtrip", "sy-mask", "family:Render", "Family:symbol"],
    ids=lambda value: value,
)
def test_tokens_are_matched_case_sensitively(selector: str) -> None:
    """Identifiers and family names are compared case-sensitively, as the product documents.

    The identifiers are uppercase by contract and the globs are matched with
    ``fnmatch.fnmatchcase``, so a case-fold that followed the platform would make a selection
    depend on the operating system.  A lower-case near-miss therefore fails loudly instead of
    selecting something the caller did not name.
    """
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, selector)
    assert raised.value.selector == selector


# ---------------------------------------------------------------------------
# ``matches``: the single token predicate
# ---------------------------------------------------------------------------


def test_matches_reads_each_token_form() -> None:
    """One contract is selected by its identifier, by a glob over it and by its family."""
    contract = REGISTRY["SY-MASK"]
    assert matches(contract, "SY-MASK") is True
    assert matches(contract, "SY-*") is True
    assert matches(contract, "family:symbol") is True
    assert matches(contract, "RT-ROUNDTRIP") is False
    assert matches(contract, "SY-*QUIV") is False


@pytest.mark.parametrize("token", ["", "   "], ids=("empty", "blank"))
def test_matches_ignores_a_blank_token(token: str) -> None:
    """A blank token selects nothing; skipping it is ``select``'s decision, not an error here."""
    assert matches(REGISTRY["CL-ARTIFACTS"], token) is False


def test_matches_quotes_the_enclosing_selection_it_was_given() -> None:
    """The error's selector is the enclosing ``--checks`` value when one was supplied.

    Called on its own, ``matches`` has no enclosing selection to quote and echoes the token;
    called from ``select``, it echoes what the caller typed.  Both shapes are exercised, because
    the echoed value is what a CI consumer greps to find the failing invocation.
    """
    contract = REGISTRY["CL-ARTIFACTS"]

    with pytest.raises(SelectionError) as alone:
        matches(contract, "family:cleanupp")
    assert alone.value.selector == "family:cleanupp"

    with pytest.raises(SelectionError) as enclosed:
        matches(contract, "family:cleanupp", selector="CL-ARTIFACTS,family:cleanupp")
    assert enclosed.value.selector == "CL-ARTIFACTS,family:cleanupp"
    assert "family:cleanupp" in enclosed.value.detail


# ---------------------------------------------------------------------------
# The empty-selection diagnostic, and its exit code through the command line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector",
    ["NOPE-*", "  NOPE-*  ", "SY-*,nope:thing", "RT-ROUNDTRIP|SY-MASK"],
    ids=("glob", "padded", "mixed", "punctuated"),
)
def test_the_empty_selection_diagnostic_carries_the_selector_verbatim(selector: str) -> None:
    """The selector is echoed exactly as typed, so a normalising bug cannot hide here.

    The padded and mixed forms are the point: a diagnostic built from trimmed or re-joined
    tokens would read plausibly and still be useless to a CI consumer grepping for its own
    invocation.  A selection that resolves some tokens and fails another is a failure of the
    whole selection, never a partial run.
    """
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, selector)
    assert raised.value.selector == selector
    assert str(raised.value).startswith(selector)
    assert isinstance(raised.value, ValueError)


def test_the_cli_reports_an_empty_selection_as_exit_two_and_writes_no_report(
    cli_runner: CliRunner, toml_config_file: Callable[..., Path], tmp_path: Path
) -> None:
    """``run --checks NOPE-*`` exits 2, echoes the selector on stderr and leaves no report.

    AAP §0.6.6 row 1's gate.  The exit status is read from the process-facing contract, not from
    a return value, and the report root is asserted empty rather than absent: the writability
    probe may create it, and an empty directory is the honest statement that nothing was
    written.
    """
    config_path = toml_config_file()
    report_root = tmp_path / "reports-empty-selection"

    result = _invoke_run(cli_runner, config_path, report_root, "--checks", "NOPE-*")

    assert result.exit_code == EXIT_CONFIG
    assert "NOPE-*" in result.stderr
    assert "unknown token" in result.stderr
    assert result.stdout.strip() == "", "an exit-2 path publishes no result and no summary"
    assert _report_entries(report_root) == ()


def test_the_cli_reports_an_empty_class_filter_as_exit_two_naming_the_filter(
    cli_runner: CliRunner, toml_config_file: Callable[..., Path], tmp_path: Path
) -> None:
    """``run --class 'no-such-*'`` exits 2, names the glob on stderr and writes no report.

    AAP §0.6.6 row 2's gate, and a distinct state from the empty selection: the diagnostic names
    the ``--class`` option and the glob, which is the half a merged message would lose.
    """
    config_path = toml_config_file()
    report_root = tmp_path / "reports-empty-class"

    result = _invoke_run(cli_runner, config_path, report_root, "--class", "no-such-*")

    assert result.exit_code == EXIT_CONFIG
    assert "no-such-*" in result.stderr
    assert "--class" in result.stderr
    assert "unknown token" not in result.stderr, "a corpus filter is not a selector token"
    assert result.stdout.strip() == ""
    assert _report_entries(report_root) == ()


# ---------------------------------------------------------------------------
# ``--class``: the corpus subset and the repeatable union
# ---------------------------------------------------------------------------


def test_the_boundary_class_glob_selects_the_forty_record_subset(
    built_corpus: tuple[CorpusRecord, ...],
) -> None:
    """``--class 'boundary-*'`` is the 40-record subset success criterion 2 runs at L, Q and H.

    The selector is the one ``qrcheck run --class`` builds -- a :class:`Selection`, whose globs
    are normalised -- and the records are the builder's, so the assertion covers the path from
    the option value to the corpus rather than either end alone.
    """
    selection = parse_selection(None, ("boundary-*",))
    selected = filter_by_class(built_corpus, selection.class_globs)

    assert selection.class_globs == ("boundary-*",)
    assert len(selected) == BOUNDARY_RECORDS == 40
    assert _classes(selected) == tuple(sorted(BOUNDARY_CLASSES))


def test_repeated_class_options_run_the_union_of_their_classes(
    cli_runner: CliRunner, toml_config_file: Callable[..., Path], tmp_path: Path
) -> None:
    """Two ``--class`` options are one union: ``url`` and ``otp`` together drive 42 cases.

    The count comes from the builder's own class table rather than a literal, and the run is
    restricted to ``RT-ROUNDTRIP`` so the assertion costs 42 cases instead of a full run's 1,283.
    Every case passing is what makes the count a statement about the corpus subset rather than
    about how many results survived.
    """
    config_path = toml_config_file()
    report_root = tmp_path / "reports-class-union"

    result = _invoke_run(
        cli_runner,
        config_path,
        report_root,
        "--checks",
        "RT-ROUNDTRIP",
        "--class",
        "url",
        "--class",
        "otp",
    )

    assert result.exit_code == 0, result.stderr
    summary = json.loads((report_root / "summary.json").read_text(encoding="utf-8"))
    expected_cases = CLASS_COUNTS["url"] + CLASS_COUNTS["otp"]
    assert expected_cases == 42
    check_ids = [entry["check_id"] for entry in summary["results"]]
    assert check_ids == ["RT-ROUNDTRIP"] * expected_cases
    assert summary["counts"] == {
        "passed": expected_cases,
        "failed": 0,
        "skipped": 0,
        "error": 0,
    }


# ---------------------------------------------------------------------------
# The ``Selection`` value object
# ---------------------------------------------------------------------------


def test_parse_selection_defaults_to_every_check_and_no_class_globs() -> None:
    """With no options, the selection asks for every check and narrows no corpus."""
    selection = parse_selection()
    assert isinstance(selection, Selection)
    assert selection.checks is None
    assert selection.class_globs == ()
    assert selection.selects_all_checks is True
    assert "class=" not in str(selection)


def test_parse_selection_keeps_the_selector_verbatim_and_normalises_the_globs() -> None:
    """The raw selector is stored as typed; the globs are stripped, de-duplicated and sorted.

    Both halves matter for opposite reasons: a verbatim selector is what makes the exit-2
    diagnostic reproducible from the command line, and an order-stable glob tuple is what keeps
    two command lines differing only in ``--class`` order from describing two corpora.
    """
    selection = parse_selection(" RT-ROUNDTRIP ", (" url ", "otp", "url", "  "))

    assert selection.checks == " RT-ROUNDTRIP "
    assert selection.class_globs == ("otp", "url")
    assert selection.selects_all_checks is False
    # The verbatim selector still resolves: trimming happens per token inside ``select``.
    assert _ids(select(REGISTRY, selection.checks)) == FAMILY_CHECKS["roundtrip"]
    assert str(selection).endswith("class=otp url")


def test_a_blank_selector_means_every_check_but_is_not_stored_as_one() -> None:
    """A blank ``--checks`` is normalised to ``None``; a real one is never rewritten."""
    assert parse_selection("").checks is None
    assert parse_selection("   ").checks is None
    assert parse_selection("").selects_all_checks is True
    assert parse_selection("RT-ROUNDTRIP").selects_all_checks is False


def test_selection_as_dict_is_json_safe_and_preserves_the_order() -> None:
    """``as_dict`` renders the globs as a list, so the record serialises unchanged."""
    selection = parse_selection("RT-ROUNDTRIP", ("url", "otp"))

    record = selection.as_dict()

    assert record == {"checks": "RT-ROUNDTRIP", "class_globs": ["otp", "url"]}
    assert json.loads(json.dumps(record)) == record


def test_selection_is_frozen_and_hashable() -> None:
    """A selection cannot be mutated after it is handed to a runner, and two equal ones agree."""
    selection = parse_selection("RT-ROUNDTRIP", ("otp",))

    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.checks = "SY-MASK"

    assert selection == Selection(checks="RT-ROUNDTRIP", class_globs=("otp",))
    assert hash(selection) == hash(Selection(checks="RT-ROUNDTRIP", class_globs=("otp",)))


@pytest.mark.parametrize("value", [1, (), b"RT-ROUNDTRIP", ["RT-ROUNDTRIP"]], ids=repr)
def test_a_non_string_selector_is_rejected(value: object) -> None:
    """A selector that is not a string is a caller defect, reported by both entry points.

    A tuple of tokens in particular is refused rather than joined: accepting it would give the
    selector two spellings, and the verbatim echo would then quote something the user cannot
    type.
    """
    with pytest.raises(TypeError):
        Selection(checks=value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_selection(checks=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [3, 3.5, object()], ids=["int", "float", "object"])
def test_a_non_iterable_class_glob_container_is_rejected(value: object) -> None:
    """Class globs are a string or an iterable of strings; anything else cannot be a filter."""
    with pytest.raises(TypeError):
        Selection(class_globs=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, b"url", None], ids=repr)
def test_a_non_string_class_glob_is_rejected(value: object) -> None:
    """One entry of the glob container must be a string, and the diagnostic quotes it."""
    with pytest.raises(TypeError):
        Selection(class_globs=(value,))  # type: ignore[arg-type]


def test_a_bare_string_class_glob_is_one_pattern() -> None:
    """A string is a single glob, never one glob per character.

    ``filter_by_class`` reads a bare string the same way, so the two layers cannot disagree
    about ``--class url``.
    """
    assert Selection(class_globs="url").class_globs == ("url",)
    assert Selection(class_globs=" boundary-* ").class_globs == ("boundary-*",)
