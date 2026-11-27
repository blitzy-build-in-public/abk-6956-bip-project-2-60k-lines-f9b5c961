"""The single owner of the ``--checks`` and ``--class`` selection grammar.

Two options of a run decide *what* runs rather than *how*: ``--checks`` narrows the twenty-one
registered checks, and ``--class`` narrows the corpus to records of chosen payload classes. Both
grammars are defined here and nowhere else, so ``qrcheck run``, its tests and any later command
agree about what a selector means. A ``--checks`` token is an exact identifier (``RT-ROUNDTRIP``),
compared case-sensitively because the identifiers are uppercase by contract; a glob over identifiers
(``SY-*``, ``DM-?EEDED``, ``[RS][TY]-*``), matched with :func:`fnmatch.fnmatchcase` and never
:func:`fnmatch.fnmatch`, whose case-folding follows the platform and would let a selection depend on
the operating system; or a family selector (``family:render``), matching every check whose
:attr:`~qrcheck.registry.check.CheckContract.family` is the named member of
:data:`~qrcheck.registry.check.FAMILY_NAMES`. Tokens are a union and duplicates collapse, so
``RT-ROUNDTRIP,family:roundtrip`` is one check, and the returned tuple is always sorted by
identifier ascending, so neither the typed order nor the registry's insertion order reaches a run's
execution order or a report. Every token must select something: a mistyped token
raises :class:`SelectionError` rather than being silently dropped, because a dropped token would
stop a check executing while the run still reported a clean pass, which would leave the suite
vacuous. One predicate, :func:`matches`, decides what a token means.

``--class`` globs are only *normalised* here -- stripped, de-duplicated and sorted -- and carried on
:attr:`Selection.class_globs` for the runner to hand to :func:`qrcheck.corpus.io.filter_by_class`;
matching records and raising on a zero-record filter belongs to ``qrcheck.runner``.
This module decides no exit code, prints nothing, writes nothing, reads no file and mutates no
registry: ``qrcheck.cli.run`` maps a raised :class:`SelectionError` to exit code 2 with no report
written, the diagnostic carrying the selector **as the caller typed it**. Importing it pulls in
nothing beyond :mod:`fnmatch` and the registry.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from qrcheck.registry.check import FAMILY_NAMES, REGISTRY, CheckContract

__all__ = [
    "Selection",
    "SelectionError",
    "matches",
    "parse_selection",
    "select",
]

# The characters that make a ``--checks`` token a glob rather than an exact identifier. Presence
# of any one of them routes a token to ``fnmatch.fnmatchcase``; their absence makes it an exact,
# case-sensitive identifier comparison.
_GLOB_CHARS: Final[str] = "*?["

# The one keyword that introduces a family selector, colon included. Every other use of a colon
# leaves the token an identifier or glob, which no check identifier can satisfy.
_FAMILY_PREFIX: Final[str] = "family:"

# What a diagnostic echoes when there is no selector to quote. A whole-run selection has none --
# its ``checks`` is ``None`` -- and the message must still read as a sentence.
_ALL_CHECKS_ECHO: Final[str] = "<all checks>"


class SelectionError(ValueError):
    """A selector cannot be resolved against the registry.

    Raised for a selection that matches no registered check and for an individual token that matches
    none, an unknown family name included. :attr:`selector` holds the selector **verbatim**, exactly
    as the caller typed it, so ``qrcheck run`` can echo it back unchanged; a token-level error
    quotes the whole selection whenever the caller's is known --
    :func:`select` passes it down to :func:`matches` -- and ``detail`` names the offending token
    either way, the message starting with the echo and then the detail. It subclasses
    :class:`ValueError` so a caller that already handles the builtin needs no special case, and
    nothing here decides an exit code.

    Attributes:
        selector: The selector as given, or ``""`` for a whole-run selection, which has none to
            quote.
        detail: Why the selection failed, phrased to follow the echoed selector.
    """

    def __init__(self, selector: str | None, detail: str) -> None:
        """Record the offending selector and build the diagnostic from it.

        Args:
            selector: The selector as the caller supplied it, kept verbatim; ``None`` for a
            whole-run
                selection, stored as an empty string because there is nothing to echo.
            detail: Why the selection failed, for example ``"matched no registered check"``.
        """
        self.selector: str = "" if selector is None else selector
        self.detail: str = detail
        echoed = self.selector if self.selector.strip() else _ALL_CHECKS_ECHO
        super().__init__(f"{echoed}: {detail}")


def _normalise_class_globs(value: object) -> tuple[str, ...]:
    """Return `value` as a sorted, duplicate-free tuple of non-empty globs.

    Normalising in one place is what makes a :class:`Selection` order-stable and hashable, so two
    command lines differing only in the order or spacing of their ``--class`` options produce the
    same selection, the same diagnostics and the same ``summary.json``. A bare string
    is one pattern rather than one pattern per character, the same reading
    :func:`qrcheck.corpus.io.filter_by_class` applies. Raises :exc:`TypeError` when `value` is
    neither a string nor an iterable, or an entry is not a string.
    """
    declared: Iterable[object]
    if isinstance(value, str):
        declared = (value,)
    elif isinstance(value, Iterable):
        declared = value
    else:
        raise TypeError(
            f"class globs must be a string or an iterable of strings, got {value!r}"
        )
    patterns: set[str] = set()
    for item in declared:
        if not isinstance(item, str):
            raise TypeError(
                f"class glob {item!r} is not a str; --class takes corpus class-name globs"
            )
        pattern = item.strip()
        if pattern:
            patterns.add(pattern)
    return tuple(sorted(patterns))


@dataclass(frozen=True)
class Selection:
    """What a run was asked to select, resolved far enough to hand to the runner.

    Two fields, each a stable diagnostic surface: ``checks``, the raw ``--checks`` selector stored
    **verbatim** -- untrimmed, unsplit and unmatched, because keeping it exactly as typed is what
    lets a failure echo it back, with ``None`` meaning "every registered check" and
    resolution happening in :func:`select` -- and ``class_globs``, normalised on construction to a
    sorted, duplicate-free tuple by :func:`_normalise_class_globs`, which the runner hands to
    :func:`qrcheck.corpus.io.filter_by_class` unchanged. The instance is frozen and, both fields
    being immutable, hashable. Raises :exc:`TypeError` when ``checks`` is neither ``None`` nor a
    string, or a class glob is not a string.
    """

    checks: str | None = None
    class_globs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a mis-typed selector and normalise the class globs.

        Raises:
            TypeError: ``checks`` is neither ``None`` nor a ``str``, or an entry of
                ``class_globs`` is not a ``str``.
        """
        if self.checks is not None and not isinstance(self.checks, str):
            raise TypeError(
                f"checks selector must be a str or None, got {self.checks!r}"
            )
        # The field is typed with its normalised form; coercion happens here so every
        # construction path -- ``parse_selection``, a test building one directly -- ends
        # up with the same sorted tuple.
        object.__setattr__(self, "class_globs", _normalise_class_globs(self.class_globs))

    @property
    def selects_all_checks(self) -> bool:
        """Return whether the selection asks for every registered check.

        ``True`` when no selector was given, and also when one was given but holds only whitespace:
        a blank option means the same thing to a user as an absent one, and treating the two
        differently would make ``--checks ""`` select nothing at all.
        """
        return self.checks is None or not self.checks.strip()

    def as_dict(self) -> dict[str, object]:
        """Return the selection as a JSON-serialisable record for diagnostics and tests.

        Returns:
            ``{"checks": <the raw selector or None>, "class_globs": [...]}``, with the globs as a
            list so :func:`json.dumps` accepts the record unchanged.
        """
        return {"checks": self.checks, "class_globs": list(self.class_globs)}

    def __str__(self) -> str:
        """Return a one-line, locale-independent rendering of the selection.

        Returns:
            The selector as typed, or ``<all checks>`` when none was given, followed by
            ``class=<glob> <glob>`` when the selection narrows the corpus.
        """
        selector = self.checks
        if selector is None or not selector.strip():
            selector = _ALL_CHECKS_ECHO
        if not self.class_globs:
            return selector
        return f"{selector} class={' '.join(self.class_globs)}"


def parse_selection(
    checks: str | None = None, class_globs: Iterable[str] = ()
) -> Selection:
    """Turn the ``--checks`` and ``--class`` options into a :class:`Selection`.

    This function only *normalises* the two options; it resolves no check and reads no corpus. A
    blank or absent ``--checks`` becomes ``None`` -- the whole-run selection -- while a non-blank
    one is stored verbatim, because :func:`select` reports failures against the text the caller
    typed. The ``--class`` globs are stripped, de-duplicated and sorted by :class:`Selection`, so
    their order on the command line cannot reach a report. Corpus semantics are not decided here:
    :func:`qrcheck.corpus.io.filter_by_class` applies the globs to records and ``qrcheck.runner``
    raises the zero-record condition naming the filter, and a malformed selector is
    :func:`select`'s to reject. Raises :exc:`TypeError` when ``checks`` is neither ``None`` nor a
    string, or a class glob is not a string.
    """
    if checks is not None and not isinstance(checks, str):
        raise TypeError(f"checks selector must be a str or None, got {checks!r}")
    selector = None if checks is None or not checks.strip() else checks
    return Selection(checks=selector, class_globs=class_globs)


def matches(
    contract: CheckContract, token: str, *, selector: str | None = None
) -> bool:
    """Return whether the single ``--checks`` token `token` selects `contract`.

    The three token forms are decided here and nowhere else, which keeps the grammar reviewable in
    one place: ``family:<name>`` selects every check of that family, compared with
    :data:`~qrcheck.registry.check.FAMILY_NAMES` and raising on an unknown name rather than matching
    nothing, because ``family:renderr`` is a typo the caller wants told about; a token holding
    ``*``, ``?`` or ``[`` is a glob over identifiers, matched with :func:`fnmatch.fnmatchcase` so
    case-folding cannot vary by platform; and anything else is an exact identifier, compared
    case-sensitively. A colon anywhere other than behind ``family`` leaves the token an identifier
    or glob, and no identifier contains one, so such a token matches nothing and :func:`select`
    reports it as unknown. ``selector`` is the enclosing selection echoed verbatim in a
    :class:`SelectionError`, or ``None`` when the token is all the caller supplied; that error is
    raised for a family selector naming a family outside
    :data:`~qrcheck.registry.check.FAMILY_NAMES`.
    """
    candidate = token.strip()
    if not candidate:
        return False
    if candidate.startswith(_FAMILY_PREFIX):
        family = candidate[len(_FAMILY_PREFIX) :].strip()
        if family not in FAMILY_NAMES:
            raise SelectionError(
                candidate if selector is None else selector,
                f"unknown check family {family!r} in token {candidate!r};"
                f" the families are {list(FAMILY_NAMES)}",
            )
        return contract.family == family
    if any(character in candidate for character in _GLOB_CHARS):
        return fnmatch.fnmatchcase(contract.id, candidate)
    return contract.id == candidate


def select(
    registry: Mapping[str, CheckContract] | None = None,
    selector: str | None = None,
) -> tuple[CheckContract, ...]:
    """Resolve a ``--checks`` selector against a registry of check contracts.

    The selector is a comma-separated list of tokens in the grammar :func:`matches` defines, and
    ``None`` or a blank selector selects every registered check. Each token's matches are unioned,
    duplicates collapse, and the result is sorted by identifier ascending, so neither the order of
    the tokens nor the registry's insertion order can reach a run's execution order or a report.

    Three conditions raise :class:`SelectionError`, each with the selector echoed verbatim so
    ``qrcheck run`` can exit 2 quoting it: a token selects nothing,
    reported as ``unknown token <token>``, because dropping it silently would let a check stop
    executing while the run still reported a clean pass, which would leave the suite vacuous; a
    family selector names an unknown family, raised by :func:`matches`; and nothing at all is
    selected -- an empty selector made only of commas, or a registry with no checks -- reported as
    ``matched no registered check``, since a run that would execute nothing is a failure to report.
    ``registry`` may be passed explicitly, which is how a test exercises the grammar without
    registering checks.
    """
    contracts = REGISTRY if registry is None else registry
    if selector is None or not selector.strip():
        every = sorted(contracts.values(), key=lambda contract: contract.id)
        if not every:
            raise SelectionError(selector, "matched no registered check")
        return tuple(every)
    matched: dict[str, CheckContract] = {}
    for token in selector.split(","):
        candidate = token.strip()
        if not candidate:
            # A trailing or doubled comma is an empty token; it selects nothing anyone meant and
            # dropping it keeps ``RT-ROUNDTRIP,`` usable while a real typo still names itself below.
            continue
        hits = [
            contract
            for contract in contracts.values()
            if matches(contract, candidate, selector=selector)
        ]
        if not hits:
            raise SelectionError(selector, f"unknown token {candidate!r}")
        for contract in hits:
            matched[contract.id] = contract
    if not matched:
        raise SelectionError(selector, "matched no registered check")
    return tuple(sorted(matched.values(), key=lambda contract: contract.id))
