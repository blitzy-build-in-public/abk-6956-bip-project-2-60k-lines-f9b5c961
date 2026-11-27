"""The ``--checks`` grammar, the ``--class`` filter and the two empty-result diagnostics.

AAP §0.3.4 ("Identity and selection") fixes three ``--checks`` token forms -- an exact
identifier, an identifier glob and a family selector -- makes a selection matching nothing an
exit-2 condition that echoes the selector **verbatim** with no report written, and makes
``--class`` a repeatable glob whose zero-record outcome is a second, differently worded exit-2
condition; §0.6.6 rows 1 and 2 name this module as the gate for both states. Also pinned here:
the corpus file is a prerequisite only of a run that reads it — a ``--class`` filter, or a check
whose cases are drawn from the corpus — because §0.3.4 conditions it on a corpus-driven check
being selected, so a synthetic-only selection completes with no corpus file at all.

Pinned: the forms are unioned, de-duplicated and returned sorted by identifier, asserted against
a selector written in reverse order, because an order following the user's typing would break
§0.5.3's determinism promise; and a mistyped token raises :class:`SelectionError` carrying the
selector as typed rather than being dropped, since a dropped token would stop a check running
while the run still reported a clean pass. Registration is forced here -- ``qrcheck.registry``
leaves the registry empty on import (AAP §0.5) -- so no grammar case can pass against a
half-populated one.
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

# The nine families and the checks each owns, from AAP §0.8.1's component-to-check-family mapping.
# This is the only place check identifiers are written as literals here; every other expectation is
# computed from the registry through ``matches`` or ``iter_checks``. Each tuple is in ascending
# identifier order, which is the order ``select`` promises.
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

# The exit code AAP §0.3.4 assigns to a configuration, selection or corpus-filter fault. A literal
# on purpose: a test that imported the constant it is checking would follow a renumbering.
EXIT_CONFIG: Final[int] = 2

# A corpus path that does not exist, relative to the temporary configuration that names it. Every
# minimal configuration written by ``toml_config_file`` has the committed corpus copied beside it,
# so only this override can leave the file unreadable.
MISSING_CORPUS: Final[str] = "corpus/absent.jsonl"

# The checks whose cases are constructed rather than drawn from the corpus. Selecting one of them
# reads no corpus record, so AAP §0.3.4's "``corpus_path`` must exist when a corpus-driven check is
# selected" -- a condition that is then false -- cannot make such a run an exit-2 fault.
CORPUS_INDEPENDENT_CHECKS: Final[tuple[str, ...]] = ("CL-ARTIFACTS", "IV-EMPTY", "PF-DECODE")


# --- Helpers ---


@pytest.fixture(scope="module", autouse=True)
def _populated_registry() -> tuple[str, ...]:
    """Import the nine families once so the registry holds all twenty-one checks."""
    identifiers = load_families()
    assert identifiers == CHECK_IDS, (
        "the selection grammar is asserted against the full registry; registered "
        f"{list(identifiers)}"
    )
    return identifiers


def _ids(contracts: Sequence[CheckContract]) -> tuple[str, ...]:
    """Return the identifiers of `contracts` in the order returned, never sorted."""
    return tuple(contract.id for contract in contracts)


def _classes(records: Sequence[CorpusRecord]) -> tuple[str, ...]:
    """Return the distinct class names of `records`, sorted."""
    return tuple(sorted({record.class_name for record in records}))


def _report_entries(report_root: Path) -> tuple[str, ...]:
    """Return the entry names under `report_root`, sorted, or ``()`` when it does not exist."""
    if not report_root.exists():
        return ()
    return tuple(sorted(entry.name for entry in report_root.iterdir()))


def _invoke_run(
    runner: CliRunner, config_path: Path, report_root: Path, *arguments: str
) -> Result:
    """Invoke ``qrcheck run`` through the real command line, always passing ``--config``."""
    argv = [
        "--config",
        str(config_path),
        "run",
        "--report-dir",
        str(report_root),
        *arguments,
    ]
    return runner.invoke(cli, argv, catch_exceptions=False)


# --- Registration: the registry is full before any selector is resolved ---


def test_registry_holds_exactly_the_declared_check_identifiers(
    _populated_registry: tuple[str, ...],
) -> None:
    """The registry holds the twenty-one declared identifiers, ascending, and nothing else."""
    assert _populated_registry == CHECK_IDS
    assert registered_ids() == CHECK_IDS
    assert sorted(REGISTRY) == list(CHECK_IDS)
    declared = tuple(
        identifier for family in sorted(FAMILY_CHECKS) for identifier in FAMILY_CHECKS[family]
    )
    assert sorted(declared) == list(CHECK_IDS)


# --- Exact identifiers, comma lists, whitespace and duplicates ---


def test_an_exact_identifier_selects_that_contract_and_nothing_else() -> None:
    """One exact identifier resolves to the registered contract itself, not to a copy."""
    selected = select(REGISTRY, "RT-ROUNDTRIP")
    assert selected == (REGISTRY["RT-ROUNDTRIP"],)
    assert selected[0] is REGISTRY["RT-ROUNDTRIP"]
    assert _ids(selected) == FAMILY_CHECKS["roundtrip"]


def test_a_comma_list_is_returned_sorted_by_identifier() -> None:
    """A comma list resolves in identifier order whatever the token order."""
    selected = select(REGISTRY, "SY-MASK,RT-ROUNDTRIP")
    assert _ids(selected) == ("RT-ROUNDTRIP", "SY-MASK")
    assert _ids(selected) == tuple(sorted(_ids(selected)))


def test_surrounding_whitespace_on_each_term_is_tolerated() -> None:
    """Spaces around a token are trimmed, and the answer is the same as the compact form."""
    spaced = select(REGISTRY, " RT-ROUNDTRIP , SY-MASK ")
    assert _ids(spaced) == _ids(select(REGISTRY, "RT-ROUNDTRIP,SY-MASK"))
    assert _ids(spaced) == ("RT-ROUNDTRIP", "SY-MASK")


def test_duplicate_tokens_do_not_duplicate_results() -> None:
    """Three ways of naming one check collapse into a single result."""
    selected = select(REGISTRY, "RT-ROUNDTRIP,family:roundtrip,RT-*,RT-ROUNDTRIP")
    assert _ids(selected) == FAMILY_CHECKS["roundtrip"]
    assert len(selected) == len(FAMILY_CHECKS["roundtrip"])


@pytest.mark.parametrize("selector", [None, "", "   "], ids=("absent", "empty", "blank"))
def test_an_absent_or_blank_selector_selects_every_check(selector: str | None) -> None:
    """No selector, an empty one and a blank one all mean every check."""
    assert _ids(select(REGISTRY, selector)) == CHECK_IDS


def test_a_trailing_comma_is_tolerated() -> None:
    """A trailing comma leaves an empty token, which is skipped rather than reported."""
    assert _ids(select(REGISTRY, "RT-ROUNDTRIP,")) == FAMILY_CHECKS["roundtrip"]
    assert _ids(select(REGISTRY, "SY-MASK,,RT-ROUNDTRIP")) == ("RT-ROUNDTRIP", "SY-MASK")


def test_a_selector_of_nothing_but_commas_is_an_empty_selection() -> None:
    """Commas with no token between them select nothing: the empty-selection condition."""
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, ",")
    assert raised.value.selector == ","
    assert "matched no registered check" in raised.value.detail


def test_an_empty_registry_cannot_satisfy_a_whole_run_selection() -> None:
    """A registry with no checks reports the empty-selection condition, not a clean run."""
    with pytest.raises(SelectionError) as raised:
        select({}, None)
    assert raised.value.selector == ""
    assert "matched no registered check" in raised.value.detail
    assert isinstance(raised.value, ValueError)


# --- Identifier globs ---


@pytest.mark.parametrize(
    ("token", "family"),
    [("SY-*", "symbol"), ("RN-*", "render"), ("*-ROUNDTRIP", "roundtrip")],
    ids=lambda value: value,
)
def test_a_glob_token_selects_every_matching_identifier(token: str, family: str) -> None:
    """A glob selects exactly §0.8.1's literal set, so a wildcard that over-matched fails."""
    selected = select(REGISTRY, token)
    assert _ids(selected) == FAMILY_CHECKS[family]
    assert {contract.family for contract in selected} == {family}


def test_a_glob_can_span_families() -> None:
    """One glob is matched against identifiers, not families, computed with :func:`matches`."""
    expected = tuple(sorted(contract.id for contract in iter_checks() if matches(contract, "?T-*")))
    assert _ids(select(REGISTRY, "?T-*")) == expected
    assert set(expected) == set(FAMILY_CHECKS["determinism"]) | set(FAMILY_CHECKS["roundtrip"])


def test_a_glob_that_matches_no_identifier_raises_naming_the_selector() -> None:
    """An identifier-shaped glob matching nothing is a reported typo, never an empty selection."""
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, "SY-VERSIN-*")
    assert raised.value.selector == "SY-VERSIN-*"
    assert "SY-VERSIN-*" in raised.value.detail


# --- Family selectors ---


@pytest.mark.parametrize("family", sorted(FAMILY_CHECKS), ids=lambda value: value)
def test_a_family_selector_selects_that_family_in_identifier_order(family: str) -> None:
    """``family:<name>`` selects exactly §0.8.1's checks, driven over all nine families."""
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
    """An unknown family is reported with the whole enclosing ``--checks`` value quoted."""
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, "SY-*,family:renderr")
    assert raised.value.selector == "SY-*,family:renderr"
    assert "family:renderr" in raised.value.detail
    assert isinstance(raised.value, ValueError)


# --- Mixtures of the three forms, and case handling ---


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
    """Identifiers and families compare case-sensitively, never by platform case-folding."""
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, selector)
    assert raised.value.selector == selector


# --- ``matches``: the single token predicate ---


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
    """The error's selector is the enclosing ``--checks`` value a CI consumer greps for."""
    contract = REGISTRY["CL-ARTIFACTS"]

    with pytest.raises(SelectionError) as alone:
        matches(contract, "family:cleanupp")
    assert alone.value.selector == "family:cleanupp"

    with pytest.raises(SelectionError) as enclosed:
        matches(contract, "family:cleanupp", selector="CL-ARTIFACTS,family:cleanupp")
    assert enclosed.value.selector == "CL-ARTIFACTS,family:cleanupp"
    assert "family:cleanupp" in enclosed.value.detail


# --- The empty-selection diagnostic, and its exit code through the command line ---


@pytest.mark.parametrize(
    "selector",
    ["NOPE-*", "  NOPE-*  ", "SY-*,nope:thing", "RT-ROUNDTRIP|SY-MASK"],
    ids=("glob", "padded", "mixed", "punctuated"),
)
def test_the_empty_selection_diagnostic_carries_the_selector_verbatim(selector: str) -> None:
    """The selector is echoed exactly as typed, padded and mixed forms included."""
    with pytest.raises(SelectionError) as raised:
        select(REGISTRY, selector)
    assert raised.value.selector == selector
    assert str(raised.value).startswith(selector)
    assert isinstance(raised.value, ValueError)


def test_the_cli_reports_an_empty_selection_as_exit_two_and_writes_no_report(
    cli_runner: CliRunner, toml_config_file: Callable[..., Path], tmp_path: Path
) -> None:
    """``run --checks NOPE-*`` exits 2, echoes the selector and leaves the report root empty."""
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
    """``run --class 'no-such-*'`` exits 2 naming the option and the glob, not merged."""
    config_path = toml_config_file()
    report_root = tmp_path / "reports-empty-class"

    result = _invoke_run(cli_runner, config_path, report_root, "--class", "no-such-*")

    assert result.exit_code == EXIT_CONFIG
    assert "no-such-*" in result.stderr
    assert "--class" in result.stderr
    assert "unknown token" not in result.stderr, "a corpus filter is not a selector token"
    assert result.stdout.strip() == ""
    assert _report_entries(report_root) == ()


# --- The corpus prerequisite is conditioned on the selection (AAP §0.3.4) ---


@pytest.mark.parametrize("check_id", CORPUS_INDEPENDENT_CHECKS, ids=lambda value: value)
def test_a_selection_that_reads_no_corpus_record_needs_no_corpus_file(
    check_id: str,
    cli_runner: CliRunner,
    toml_config_file: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A synthetic-only selection completes with an unreadable ``corpus_path``.

    AAP §0.3.4 conditions the corpus file's existence on a corpus-driven check being selected, so
    a run whose checks build their own cases must not be refused for a file it never reads: the
    command exits 0, writes its report and records that check's results alone.
    """
    config_path = toml_config_file(corpus_path=MISSING_CORPUS)
    report_root = tmp_path / "reports-without-corpus"

    result = _invoke_run(cli_runner, config_path, report_root, "--checks", check_id)

    assert result.exit_code == 0, result.stderr
    summary = json.loads((report_root / "summary.json").read_text(encoding="utf-8"))
    assert {entry["check_id"] for entry in summary["results"]} == {check_id}
    assert summary["counts"]["error"] == 0


def test_a_corpus_driven_selection_still_requires_the_corpus_file(
    cli_runner: CliRunner,
    toml_config_file: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``RT-ROUNDTRIP`` draws every case from the corpus, so that same missing file is exit 2.

    The other half of the condition: the file is a prerequisite exactly when a selected check
    reads it, and the diagnostic names the path a reader has to build.
    """
    config_path = toml_config_file(corpus_path=MISSING_CORPUS)
    report_root = tmp_path / "reports-corpus-required"

    result = _invoke_run(cli_runner, config_path, report_root, "--checks", "RT-ROUNDTRIP")

    assert result.exit_code == EXIT_CONFIG
    assert Path(MISSING_CORPUS).name in result.stderr
    assert _report_entries(report_root) == ()


# --- ``--class``: the corpus subset and the repeatable union ---


def test_the_boundary_class_glob_selects_the_forty_record_subset(
    built_corpus: tuple[CorpusRecord, ...],
) -> None:
    """``--class 'boundary-*'`` is the 40-record subset criterion 2 runs at L, Q and H."""
    selection = parse_selection(None, ("boundary-*",))
    selected = filter_by_class(built_corpus, selection.class_globs)

    assert selection.class_globs == ("boundary-*",)
    assert len(selected) == BOUNDARY_RECORDS == 40
    assert _classes(selected) == tuple(sorted(BOUNDARY_CLASSES))


def test_repeated_class_options_run_the_union_of_their_classes(
    cli_runner: CliRunner, toml_config_file: Callable[..., Path], tmp_path: Path
) -> None:
    """Two ``--class`` options are one union, counted from the builder's class table."""
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


# --- The ``Selection`` value object ---


def test_parse_selection_defaults_to_every_check_and_no_class_globs() -> None:
    """With no options, the selection asks for every check and narrows no corpus."""
    selection = parse_selection()
    assert isinstance(selection, Selection)
    assert selection.checks is None
    assert selection.class_globs == ()
    assert selection.selects_all_checks is True
    assert "class=" not in str(selection)


def test_parse_selection_keeps_the_selector_verbatim_and_normalises_the_globs() -> None:
    """The raw selector is stored as typed while the globs are stripped, deduped, sorted."""
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
    """A selector that is not a string is a caller defect, refused by both entry points."""
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
    """A string is a single glob, never one glob per character."""
    assert Selection(class_globs="url").class_globs == ("url",)
    assert Selection(class_globs=" boundary-* ").class_globs == ("boundary-*",)
