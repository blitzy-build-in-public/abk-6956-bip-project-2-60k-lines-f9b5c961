"""The configuration contract: every key, every rule, its dotted-key diagnostic, and the one
deliberate exception.

Every case is derived from :data:`qrcheck.config.schema.SCHEMA`, from ``models`` or from the
committed document, so a key added without a kind, a rule or a default fails here, and each
rejection asserts the offending **dotted** key path the exit-2 diagnostic prints. A subset of
the faults is also driven through ``qrcheck run`` with a real ``--config`` path: exit 2 and no
report artefact at all.

The one deliberate exception: a capability *shortfall* is a ``skipped`` result with reason
``unsupported_capability``, not a configuration error, because whether a declared set can
satisfy a check depends on the run-time ``--checks`` selection (AAP §0.12, fourth resolved
conflict; ``tests/unit/test_registry_contract.py`` asserts that half). An unknown capability
*member* is still exit 2.

Nothing here loads the repository's own ``qrcheck.toml`` through ``load_config``, which would
create ``reports/`` inside the checkout, and the unusable-``report_dir`` cases never use
``chmod``, which proves nothing as root."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pytest

import qrcheck.config
from qrcheck.cli.main import cli as qrcheck_cli
from qrcheck.config import load as load_module
from qrcheck.config import models, schema
from qrcheck.config.load import load_config


# The repository root is derived from the imported module rather than from the working
# directory, so the committed-configuration test finds ``qrcheck.toml`` from any cwd (the same
# derivation ``qrcheck/config/seeding.py``'s tests use).
_SCHEMA_PATH: Final[Path] = Path(schema.__file__).resolve()
_PACKAGE_DIR: Final[Path] = _SCHEMA_PATH.parents[1]  # .../qrcheck
_REPOSITORY_ROOT: Final[Path] = _SCHEMA_PATH.parents[2]  # the directory holding qrcheck/
_COMMITTED_CONFIG: Final[Path] = _REPOSITORY_ROOT / "qrcheck.toml"

#: The target name the probe document declares when the defaults of ``[targets.*]`` are read.
_PROBE_TARGET: Final[str] = "probe"

#: The target name every valid document built by this module declares.
_TARGET_NAME: Final[str] = "target"


# Reading the one declaration. Nothing below restates a group path, a key name or a default, so
# this module cannot disagree with the contract it is testing.


def _group_defaults(group: str) -> dict[str, Any]:
    """Return every declared default of one schema group, keyed by its configuration key."""
    return {key: spec.default for key, spec in schema.SCHEMA[group].items()}


def _declared_default(group: str, key: str) -> Any:
    """Return the declared fallback of one key."""
    return schema.SCHEMA[group][key].default


def _materialised(default: Any) -> Any:
    """Return a declared default as ``apply_defaults`` resolves it, converted here rather than
    imported, so the expectation is independent of the implementation it checks."""
    if isinstance(default, tuple):
        return list(default)
    if isinstance(default, Mapping):
        return dict(default)
    return default


def _is_record_group(group: str) -> bool:
    """Whether ``group`` is a per-target-*type* table rather than one ``apply_defaults`` fills; a
    per-type table's optional keys belong to the record the loader builds."""
    parts = group.split(".")
    return len(parts) == 3 and parts[0] == "targets"


def _defaults_document() -> dict[str, Any]:
    """Return the resolved document for one probe target: ``apply_defaults`` cannot invent a
    ``[targets.<name>]`` table, so the name is declared here to make the common keys observable."""
    return schema.apply_defaults({"targets": {_PROBE_TARGET: {"type": "reference"}}})


def _filled_table(group: str, document: Mapping[str, Any]) -> Any:
    """Return the resolved table for ``group`` inside a defaulted document."""
    if group == "":
        return {key: document[key] for key in schema.SCHEMA[""] if key in document}
    if group == "targets.*":
        return document.get("targets", {}).get(_PROBE_TARGET)
    node: Any = document
    for part in group.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


#: Every group of the contract, sorted so the parametrised ids are stable.
GROUPS: Final[tuple[str, ...]] = tuple(sorted(schema.SCHEMA))

#: The groups whose defaults ``apply_defaults`` fills, in sorted order.
TABLE_GROUPS: Final[tuple[str, ...]] = tuple(
    group for group in GROUPS if not _is_record_group(group)
)

#: The per-target-type groups, in sorted order.
RECORD_GROUPS: Final[tuple[str, ...]] = tuple(
    group for group in GROUPS if _is_record_group(group)
)

#: Every ``(group, key)`` pair, sorted by both, so a key with no kind fails on its own id.
KEY_CASES: Final[tuple[tuple[str, ...], ...]] = tuple(
    sorted(
        ((group, key) for group in GROUPS for key in schema.SCHEMA[group]),
        key=lambda pair: (pair[0], pair[1]),
    )
)

#: The pairs that declare a fallback, i.e. the ones ``apply_defaults`` must supply.
DEFAULTED_KEY_CASES: Final[tuple[tuple[str, ...], ...]] = tuple(
    pair for pair in KEY_CASES if not schema.SCHEMA[pair[0]][pair[1]].is_required
)

#: The pairs the document has to supply, which ``apply_defaults`` must never invent.
REQUIRED_KEY_CASES: Final[tuple[tuple[str, ...], ...]] = tuple(
    pair for pair in KEY_CASES if schema.SCHEMA[pair[0]][pair[1]].is_required
)

#: The per-type tables that declare at least one key, keyed by the ``type`` value in their path.
CONFIGURABLE_KINDS: Final[tuple[str, ...]] = tuple(
    sorted(
        group.rsplit(".", 1)[-1] for group in RECORD_GROUPS if schema.SCHEMA[group]
    )
)

#: Every required per-type input, so a type that gained one is demanded here automatically.
REQUIRED_PER_TYPE_CASES: Final[tuple[tuple[str, ...], ...]] = tuple(
    sorted(
        (
            (kind, key)
            for kind in CONFIGURABLE_KINDS
            for key, spec in schema.SCHEMA[f"targets.*.{kind}"].items()
            if spec.is_required
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
)

#: The per-type tables that declare a ``returns`` key, derived from the table itself.
_RETURNS_KINDS: Final[tuple[str, ...]] = tuple(
    kind for kind in CONFIGURABLE_KINDS if "returns" in schema.SCHEMA[f"targets.*.{kind}"]
)

#: Every optional per-type input, whose default the assembled record owns.
OPTIONAL_PER_TYPE_CASES: Final[tuple[tuple[str, ...], ...]] = tuple(
    sorted(
        (
            (kind, key)
            for kind in CONFIGURABLE_KINDS
            for key, spec in schema.SCHEMA[f"targets.*.{kind}"].items()
            if not spec.is_required
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
)

#: ``default_adapter``'s fallback constant is ``DEFAULT_ADAPTER``, not ``DEFAULT_DEFAULT_ADAPTER``;
#: every other defaulted key follows the mechanical ``DEFAULT_<KEY>`` spelling, so the lookup is
#: derived rather than a hand-maintained map of twenty names.
_DEFAULT_CONSTANT_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {"default_adapter": "DEFAULT_ADAPTER"}
)


def _default_constant_name(key: str) -> str:
    """Return the ``models`` constant a key's declared fallback is expected to live in."""
    return _DEFAULT_CONSTANT_ALIASES.get(key, f"DEFAULT_{key.upper()}")


# Valid documents, one per adapter kind: the worked examples every rejection below perturbs once,
# asserted to cover ``models.ADAPTER_TYPE_NAMES``. Where the schema declares the ``dict`` kind a
# plain ``dict`` is required, which is how TOML itself produces a table.

_VALID_TARGET_TABLES: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "cli": MappingProxyType(
            {
                "type": "cli",
                "command": ["generate", "--ec", "M", "-o", "symbol.png"],
                "payload_arg": "{}",
                "output_arg": "-o",
                "returns": "file_path",
            }
        ),
        "defective": MappingProxyType({"type": "defective"}),
        "http": MappingProxyType(
            {
                "type": "http",
                "url": "http://127.0.0.1:8000/generate",
                "method": "POST",
                "request_template": {"payload": "{payload_b64}", "ec": "{ec_level}"},
                "image_field": "data.image",
                "image_encoding": "base64",
            }
        ),
        "library": MappingProxyType(
            {
                "type": "library",
                "callable": "example_target:generate",
                "kwargs_map": {"payload": "text", "ec_level": "error"},
                "returns": "base64",
            }
        ),
        "reference": MappingProxyType({"type": "reference"}),
    }
)


def _target_document(kind: str = "reference", **overrides: Any) -> dict[str, Any]:
    """Return a valid document declaring one target of ``kind``, perturbed by ``overrides``;
    ``default_adapter`` names that target, so a case exercises the key it means to."""
    table = dict(_VALID_TARGET_TABLES[kind])
    table.update(overrides)
    return {"default_adapter": _TARGET_NAME, "targets": {_TARGET_NAME: table}}


def _target_document_without(kind: str, *omitted: str) -> dict[str, Any]:
    """Return a valid document for ``kind`` with the named keys of its table removed."""
    table = {
        key: value
        for key, value in _VALID_TARGET_TABLES[kind].items()
        if key not in omitted
    }
    return {"default_adapter": _TARGET_NAME, "targets": {_TARGET_NAME: table}}


def _http_document(**overrides: Any) -> dict[str, Any]:
    """Return a valid ``http`` target document with the named keys replaced."""
    return _target_document("http", **overrides)


# TOML documents for the command-line cases

#: The five top-level keys, in the table's declared order, which is the order they are written in.
_TOP_LEVEL_KEYS: Final[tuple[str, ...]] = tuple(schema.SCHEMA[""])

#: The target name the fallback ``default_adapter`` names, read from the schema, never restated.
_REFERENCE_TARGET: Final[str] = _declared_default("", "default_adapter")


def _toml_literal(value: object) -> str:
    """Render one scalar as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _toml_value(value: object) -> str:
    """Render one configuration value as TOML, arrays included."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    return _toml_literal(value)


def _emit_table(lines: list[str], prefix: str, table: Mapping[str, Any]) -> None:
    """Write one table into ``lines``: its own values first, then one section per sub-table, since
    TOML reads a key after a table header as a member of that table."""
    children: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in table.items():
        if isinstance(value, Mapping):
            children.append((key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, child in children:
        path = f"{prefix}.{key}" if prefix else key
        lines.extend(("", f"[{path}]"))
        _emit_table(lines, path, child)


def _document_text(document: Mapping[str, Any]) -> str:
    """Render a raw configuration document as the TOML text written under ``tmp_path``, exactly as
    the case built it, for :func:`schema.validate` and for ``qrcheck run --config`` alike."""
    lines: list[str] = []
    _emit_table(lines, "", document)
    return "\n".join(lines) + "\n"


def _minimal_document(**overrides: Any) -> dict[str, Any]:
    """Return the smallest document that loads and validates, so a case states only its own key."""
    values: dict[str, Any] = {key: _declared_default("", key) for key in _TOP_LEVEL_KEYS}
    for key, value in overrides.items():
        if key not in values:
            raise KeyError(f"_minimal_document got an undeclared top-level key {key!r}")
        values[key] = value
    return {**values, "targets": {_REFERENCE_TARGET: {"type": "reference"}}}


def _report_artefacts(report_dir: Path) -> list[str]:
    """Return every artefact present under ``report_dir``: the three fixed writer outputs, so an
    exit-2 path that wrote any of them is caught here."""
    if not report_dir.exists():
        return []
    return sorted(entry.name for entry in report_dir.iterdir())


def _key_ids(pairs: tuple[tuple[str, ...], ...]) -> list[str]:
    """Return one readable parametrize id per ``(group, key)`` pair."""
    return [f"{group or 'top'}/{key}" for group, key in pairs]


def _kind_ids(kinds: tuple[str, ...]) -> list[str]:
    """Return one parametrize id per adapter kind."""
    return list(kinds)


# The table itself: every key carries a kind, a rule and a fallback


@pytest.mark.parametrize("group, key", KEY_CASES, ids=_key_ids(KEY_CASES))
def test_every_declared_key_carries_a_kind_a_rule_and_a_default(group: str, key: str) -> None:
    """Every key declares what it accepts, how it is judged and its fallback: no kind accepts
    anything, no rule accepts what should be refused, neither fallback nor required is silent."""
    spec = schema.SCHEMA[group][key]
    named = f"{group or 'top level'}:{key}"
    assert isinstance(spec, schema.KeySpec), f"{named} is not declared as a KeySpec"
    assert spec.key == key, f"{named} is declared under the name {spec.key!r}"
    assert spec.kinds, f"{named} declares no accepted type"
    assert all(isinstance(kind, type) for kind in spec.kinds), f"{named} declares a non-type kind"
    assert callable(spec.check), f"{named} declares no rule to judge its value by"
    assert spec.is_required or spec.default is not None, f"{named} declares no usable default"


@pytest.mark.parametrize("group, key", DEFAULTED_KEY_CASES, ids=_key_ids(DEFAULTED_KEY_CASES))
def test_the_declared_default_is_the_models_constant(group: str, key: str) -> None:
    """Every declared fallback is the ``models.DEFAULT_*`` constant, so retuning a default cannot
    leave the schema behind (AAP §0.12, NFR-3: a threshold is declared once)."""
    spec = schema.SCHEMA[group][key]
    named = f"{group or 'top level'}:{key}"
    constant = _default_constant_name(key)
    if hasattr(models, constant):
        assert spec.default == getattr(models, constant), (
            f"{named} declares {spec.default!r}, but models.{constant} is "
            f"{getattr(models, constant)!r}"
        )
        return
    assert isinstance(spec.default, (tuple, Mapping)), (
        f"{named} has no models.{constant} constant and its default is no collection"
    )
    assert not spec.default, f"{named} declares a non-empty collection default away from models"


@pytest.mark.parametrize(
    "group", TABLE_GROUPS, ids=[group or "top-level" for group in TABLE_GROUPS]
)
def test_apply_defaults_fills_exactly_the_declared_defaults(group: str) -> None:
    """A resolved document carries every declared fallback of the group and invents nothing: a
    missing key leaves a record unset, and an extra one was never declared."""
    filled = _filled_table(group, _defaults_document())
    named = group or "top level"
    assert isinstance(filled, Mapping), f"{named} is not reachable through apply_defaults"
    declared = schema.SCHEMA[group]
    extra = sorted(set(filled) - set(declared))
    assert not extra, f"{named} gained key(s) the schema never declared: {extra}"
    for key, spec in declared.items():
        if spec.is_required:
            assert key in filled, f"{named}:{key} is required, so the document must supply it"
            continue
        assert key in filled, f"{named}:{key} declares a default that apply_defaults did not fill"
        assert filled[key] == _materialised(spec.default), (
            f"{named}:{key} resolved to {filled[key]!r}, declared {spec.default!r}"
        )


@pytest.mark.parametrize("group, key", REQUIRED_KEY_CASES, ids=_key_ids(REQUIRED_KEY_CASES))
def test_apply_defaults_never_invents_a_required_key(group: str, key: str) -> None:
    """A required key is never defaulted: only the document can supply it."""
    filled = _filled_table(group, _defaults_document())
    named = f"{group or 'top level'}:{key}"
    if filled is None:
        # A per-type table is never filled by apply_defaults at all, so its required keys cannot
        # be invented either -- the loader reads them from the assembled record's table.
        assert _is_record_group(group), f"{named} sits in a group apply_defaults should fill"
        return
    if key in filled:
        expected = _VALID_TARGET_TABLES["reference"][key]
        assert group == "targets.*", f"{named} was invented in a group that supplies nothing"
        assert filled[key] == expected, f"{named} resolved to {filled[key]!r}, not {expected!r}"


def test_every_group_is_either_filled_or_a_per_type_table() -> None:
    """The two classifications cover the contract, so a group that stopped being reachable through
    ``apply_defaults`` cannot drop out of the parametrisation below unchecked."""
    assert set(TABLE_GROUPS).isdisjoint(RECORD_GROUPS), "a group is classified twice"
    assert set(TABLE_GROUPS) | set(RECORD_GROUPS) == set(GROUPS), "a group is classified nowhere"
    assert "" in TABLE_GROUPS, "the top level must be one of the filled groups"
    document = _defaults_document()
    for group in TABLE_GROUPS:
        assert isinstance(_filled_table(group, document), Mapping), (
            f"{group or 'top level'} is classified as filled but apply_defaults omits it"
        )
    for group in RECORD_GROUPS:
        assert _filled_table(group, document) is None, (
            f"{group} is classified as a per-type table but apply_defaults fills it"
        )


@pytest.mark.parametrize("kind", CONFIGURABLE_KINDS, ids=_kind_ids(CONFIGURABLE_KINDS))
def test_the_optional_per_type_default_is_owned_by_the_record(
    kind: str, toml_config_file: Any
) -> None:
    """An omitted optional per-type key resolves to its record's fallback: ``metadata_map`` is
    filled in :mod:`qrcheck.config.models`, not in the schema, so the default is declared once."""
    document = _target_document_without(kind, *(key for k, key in OPTIONAL_PER_TYPE_CASES))
    config = load_config(toml_config_file(_document_text(document)))
    record = config.targets[_TARGET_NAME]
    for optional_kind, key in OPTIONAL_PER_TYPE_CASES:
        if optional_kind != kind:
            continue
        declared = _materialised(schema.SCHEMA[f"targets.*.{kind}"][key].default)
        resolved = getattr(record, key)
        assert dict(resolved) == dict(declared), (
            f"targets.{kind}.{key} resolved to {resolved!r}, declared {declared!r}"
        )


def test_the_two_built_in_targets_take_no_per_type_key() -> None:
    """A configurable input on a built-in self-check target would be a key with no effect: both
    built-in kinds take only the common keys, so any other key there is an unknown key."""
    builtin = sorted(
        kind for kind in models.ADAPTER_TYPE_NAMES if kind not in CONFIGURABLE_KINDS
    )
    assert builtin, "every adapter kind declares per-type keys, so this contract is untested"
    for kind in builtin:
        assert schema.SCHEMA[f"targets.*.{kind}"] == {}, (
            f"targets.*.{kind} declares keys, but the built-in target takes none"
        )


def test_the_worked_examples_cover_every_adapter_kind() -> None:
    """The valid documents below cover the whole ``type`` enum, so none of the tests is untested."""
    assert set(_VALID_TARGET_TABLES) == set(models.ADAPTER_TYPE_NAMES), (
        "the worked example documents do not cover every adapter kind: "
        f"{sorted(set(models.ADAPTER_TYPE_NAMES) - set(_VALID_TARGET_TABLES))} missing"
    )


@pytest.mark.parametrize("kind", sorted(_VALID_TARGET_TABLES), ids=_kind_ids(
    tuple(sorted(_VALID_TARGET_TABLES))
))
def test_a_worked_example_document_is_accepted(kind: str) -> None:
    """Each worked example validates, which is what makes every rejection a single perturbation."""
    schema.validate(_target_document(kind))


@pytest.mark.parametrize(
    "kind, key", REQUIRED_PER_TYPE_CASES, ids=_key_ids(REQUIRED_PER_TYPE_CASES)
)
def test_every_required_per_type_input_is_demanded(kind: str, key: str) -> None:
    """A per-type table missing one of its required inputs is rejected, naming that key."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document_without(kind, key))
    dotted = f"targets.{_TARGET_NAME}.{key}"
    assert excinfo.value.key == dotted, f"expected {dotted}, got {excinfo.value.key!r}"
    assert f"missing required key '{key}'" in str(excinfo.value), str(excinfo.value)


# The threshold records read the one declaration


def test_the_limit_records_hold_the_thresholds_the_schema_declares(
    toml_config_file: Any,
) -> None:
    """A document omitting every threshold resolves to the schema's declared fallbacks. The records
    are compared with the table, never with literals, so a retuned default cannot drift apart."""
    config = load_config(toml_config_file(_document_text(_minimal_document())))
    assert config.render == models.RenderConfig(**_group_defaults("render"))
    assert config.performance == models.PerformanceLimits(
        **_group_defaults("checks.performance")
    )
    assert config.damage == models.DamageLimits(**_group_defaults("checks.damage"))

    defaults = schema.apply_defaults({})
    assert config.render.svg_rasterizer == defaults["render"]["svg_rasterizer"]
    for key, value in defaults["checks"]["performance"].items():
        assert getattr(config.performance, key) == value, (
            f"checks.performance.{key} resolved to {getattr(config.performance, key)!r}, "
            f"declared {value!r}"
        )
    for key, value in defaults["checks"]["damage"].items():
        assert getattr(config.damage, key) == value, (
            f"checks.damage.{key} resolved to {getattr(config.damage, key)!r}, declared {value!r}"
        )


def test_a_supplied_threshold_reaches_its_record(toml_config_file: Any) -> None:
    """A supplied threshold wins over the fallback and the rest keep theirs: one key per limit table
    is moved one unit inside its declared range, from the declared default and never a literal."""
    performance_key = "warmup"
    damage_key = "jpeg_quality"
    performance_value = _declared_default("checks.performance", performance_key) + 1
    damage_value = _declared_default("checks.damage", damage_key) + 1
    document = _minimal_document()
    document["checks"] = {
        "performance": {performance_key: performance_value},
        "damage": {damage_key: damage_value},
    }
    config = load_config(toml_config_file(_document_text(document)))

    assert getattr(config.performance, performance_key) == performance_value
    assert getattr(config.damage, damage_key) == damage_value
    for key, spec in schema.SCHEMA["checks.performance"].items():
        if key != performance_key:
            assert getattr(config.performance, key) == spec.default, (
                f"checks.performance.{key} left its declared default"
            )
    for key, spec in schema.SCHEMA["checks.damage"].items():
        if key != damage_key:
            assert getattr(config.damage, key) == spec.default, (
                f"checks.damage.{key} left its declared default"
            )


def test_the_package_facade_publishes_exactly_its_declared_surface() -> None:
    """The package façade reports exactly its declared surface, so a name added without ``__all__``
    -- or a documented name no longer bound -- fails here instead of disagreeing with the docs."""
    assert dir(qrcheck.config) == sorted(qrcheck.config.__all__), (
        f"dir() reports {dir(qrcheck.config)}, __all__ declares "
        f"{sorted(qrcheck.config.__all__)}"
    )
    assert qrcheck.config.ConfigError is schema.ConfigError, (
        "the façade's ConfigError is not the schema's own error type"
    )
    assert qrcheck.config.load_config is load_config, (
        "the façade's load_config is not the loader's own function"
    )


# Unknown keys, at every level

#: One document per level an unknown key can be written at, with the dotted path the diagnostic
#: has to name. Sorted by that path, so the ids do not depend on the order they were written in.
_UNKNOWN_KEY_CASES: Final[tuple[tuple[Mapping[str, Any], str], ...]] = tuple(
    sorted(
        (
            ({"nonsense": 1}, "nonsense"),
            (_target_document("reference", bogus=1), f"targets.{_TARGET_NAME}.bogus"),
            (
                _target_document("library", command=["generate"]),
                f"targets.{_TARGET_NAME}.command",
            ),
            ({"render": {"bogus": 1}}, "render.bogus"),
            ({"checks": {"performance": {"bogus": 1}}}, "checks.performance.bogus"),
            ({"checks": {"damage": {"bogus": 1}}}, "checks.damage.bogus"),
            ({"checks": {"bogus": {}}}, "checks.bogus"),
        ),
        key=lambda case: (case[1], repr(case[0])),
    )
)


@pytest.mark.parametrize(
    "document, dotted",
    _UNKNOWN_KEY_CASES,
    ids=[dotted for _document, dotted in _UNKNOWN_KEY_CASES],
)
def test_an_unknown_key_is_rejected_at_every_level(
    document: Mapping[str, Any], dotted: str
) -> None:
    """An unknown key is a configuration error naming its dotted path at every level; a key that
    belongs to another adapter kind is unknown here, never silently inert."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(document)
    assert excinfo.value.key == dotted, f"expected key {dotted!r}, got {excinfo.value.key!r}"
    assert dotted in str(excinfo.value), str(excinfo.value)
    assert "unknown key" in str(excinfo.value), str(excinfo.value)


def test_a_target_name_that_is_blank_is_rejected() -> None:
    """A ``[targets.<name>]`` name has to be a non-empty string: a blank name is not a key, so the
    fault belongs to the ``[targets]`` table whose identity cannot be reported."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate({"targets": {" ": {"type": "reference"}}})
    assert excinfo.value.key == "targets", f"expected key 'targets', got {excinfo.value.key!r}"
    assert "target name must be a non-empty string" in str(excinfo.value), str(excinfo.value)


def test_a_document_that_is_not_a_table_is_rejected() -> None:
    """A document whose root is not a table belongs to no single key and says so."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate([])
    assert excinfo.value.key == "", f"a whole-document fault carries no key: {excinfo.value.key!r}"
    assert "expected table" in str(excinfo.value), str(excinfo.value)


def test_a_table_written_as_an_array_is_rejected() -> None:
    """``[[targets]]`` parses to a list, which is reported as what arrived rather than crashing."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate({"targets": [{"type": "reference"}]})
    assert excinfo.value.key == "targets"
    assert "got array" in str(excinfo.value), str(excinfo.value)


# Kinds, closed vocabularies and the capability member

#: One wrong-kind value per shape the type check has to name, with the key and the diagnostic
#: fragment it must contain. Sorted by (key, fragment).
_WRONG_KIND_CASES: Final[tuple[tuple[Mapping[str, Any], str, str], ...]] = tuple(
    sorted(
        (
            ({"default_adapter": ["reference"]}, "default_adapter", "expected string, got array"),
            ({"corpus_path": None}, "corpus_path", "expected string, got null"),
            ({"loc_limit": 1.5}, "loc_limit", "expected integer, got number"),
            ({"report_dir": 5}, "report_dir", "expected string, got integer"),
            ({"seed": "abc"}, "seed", "expected integer, got string"),
            ({"seed": True}, "seed", "expected integer, got boolean"),
            ({"render": []}, "render", "expected table, got array"),
            (
                {"render": {"svg_rasterizer": 1}},
                "render.svg_rasterizer",
                "expected string, got integer",
            ),
            ({"checks": "no"}, "checks", "expected table, got string"),
            (
                {"checks": {"performance": {"encode_samples": "ten"}}},
                "checks.performance.encode_samples",
                "expected integer, got string",
            ),
            (
                {"checks": {"performance": {"warmup": True}}},
                "checks.performance.warmup",
                "expected integer, got boolean",
            ),
            (
                {"checks": {"performance": {"encode_p95_ms": float("inf")}}},
                "checks.performance.encode_p95_ms",
                "finite",
            ),
            (
                {"checks": {"damage": {"jpeg_quality": 40.5}}},
                "checks.damage.jpeg_quality",
                "expected integer, got number",
            ),
            (
                _target_document("reference", capabilities="png"),
                f"targets.{_TARGET_NAME}.capabilities",
                "expected array, got string",
            ),
            (
                _target_document("reference", timeout_s="30"),
                f"targets.{_TARGET_NAME}.timeout_s",
                "expected number, got string",
            ),
        ),
        key=lambda case: (case[1], case[2]),
    )
)


@pytest.mark.parametrize(
    "document, dotted, fragment",
    _WRONG_KIND_CASES,
    ids=[f"{dotted}:{fragment}" for _document, dotted, fragment in _WRONG_KIND_CASES],
)
def test_a_value_of_the_wrong_kind_is_rejected(
    document: Mapping[str, Any], dotted: str, fragment: str
) -> None:
    """A value of the wrong kind is rejected naming the key and what arrived: ``bool`` is refused
    for an int key despite being an ``int`` subclass, and a non-finite float is refused too."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(document)
    assert excinfo.value.key == dotted, f"expected key {dotted!r}, got {excinfo.value.key!r}"
    assert fragment in str(excinfo.value), (
        f"{dotted}: expected {fragment!r} in {str(excinfo.value)!r}"
    )


def test_an_absent_target_type_is_rejected() -> None:
    """A ``[targets.<name>]`` table with no ``type`` cannot select its per-type keys."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate({"default_adapter": _TARGET_NAME, "targets": {_TARGET_NAME: {}}})
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.type"
    assert "missing required key 'type'" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "value",
    ["queue", "Library", "reference ", ""],
    ids=["queue", "Library", "trailing-space", "empty"],
)
def test_a_type_outside_the_five_kinds_is_rejected(value: str) -> None:
    """Only the five declared kinds are accepted, compared exactly and case-sensitively."""
    document = {"default_adapter": _TARGET_NAME, "targets": {_TARGET_NAME: {"type": value}}}
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(document)
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.type"
    assert str(sorted(models.ADAPTER_TYPE_NAMES)) in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "kind", _RETURNS_KINDS, ids=_kind_ids(_RETURNS_KINDS)
)
def test_a_returns_value_outside_its_enum_is_rejected(kind: str) -> None:
    """``returns`` is a closed vocabulary for every kind that declares it."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document(kind, returns="stream"))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.returns"
    assert str(sorted(models.RETURNS_KINDS)) in str(excinfo.value), str(excinfo.value)


def test_an_image_encoding_outside_its_enum_is_rejected() -> None:
    """``image_encoding`` accepts ``base64`` or ``raw`` and nothing else."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(image_encoding="png"))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.image_encoding"
    assert sorted(models.IMAGE_ENCODINGS).__repr__() in str(excinfo.value), str(excinfo.value)


def test_a_method_outside_its_enum_is_rejected() -> None:
    """``method`` accepts ``GET`` or ``POST``, spelled as the enum spells them."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(method="PUT"))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.method"
    assert sorted(models.HTTP_METHODS).__repr__() in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "value", ["inkscape", "rsvg", "AUTO", ""], ids=["inkscape", "rsvg", "AUTO", "empty"]
)
def test_an_svg_rasterizer_outside_the_six_values_is_rejected(value: str) -> None:
    """``[render] svg_rasterizer`` is one of the six documented names, exactly spelled."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate({"render": {"svg_rasterizer": value}})
    assert excinfo.value.key == "render.svg_rasterizer"
    assert sorted(models.SVG_RASTERIZERS).__repr__() in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "member", ["teleport", "PNG", "", "png "], ids=["teleport", "PNG", "empty", "trailing-space"]
)
def test_an_unknown_capability_member_is_rejected(member: str) -> None:
    """A capability *member* outside the 13 names is a configuration error naming it --
    membership is the only part of a capability shortfall that is decidable here."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("reference", capabilities=["png", member]))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.capabilities"
    assert f"unknown capability {member!r}" in str(excinfo.value), str(excinfo.value)


def test_every_declared_capability_member_is_accepted() -> None:
    """All 13 capability names are accepted, so the vocabulary is not narrower than declared."""
    schema.validate(
        _target_document("reference", capabilities=sorted(models.CAPABILITY_NAMES))
    )


def test_a_capability_that_is_not_a_string_is_rejected() -> None:
    """A non-string member is named as the kind that arrived rather than crashing the loop."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("reference", capabilities=[1]))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.capabilities"
    assert "every capability must be a string" in str(excinfo.value), str(excinfo.value)



def _document_for(dotted: str, value: Any) -> dict[str, Any]:
    """Return a document declaring ``dotted`` with ``value`` and nothing else that is wrong, so a
    table below states a key, a value and the expected wording without repeating the nesting."""
    parts = dotted.split(".")
    if len(parts) == 3 and parts[0] == "targets":
        return _target_document("reference", **{parts[2]: value})
    document = _minimal_document()
    node = document
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return document


# Syntax and ranges

#: One malformed ``module:function`` path per shape the rule names, with its wording. Sorted by
#: value, so the ids are stable.
_MALFORMED_CALLABLE_CASES: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        (
            ("modulefunction", "expected exactly one ':' as 'module:function'"),
            ("a:b:c", "expected exactly one ':' as 'module:function'"),
            (":func", "expected both parts of 'module:function'"),
            ("mod:", "expected both parts of 'module:function'"),
            ("1mod:func", "is not a valid Python identifier"),
            ("mod:1func", "is not a valid Python identifier"),
        ),
        key=lambda case: (case[0], case[1]),
    )
)


@pytest.mark.parametrize(
    "value, fragment",
    _MALFORMED_CALLABLE_CASES,
    ids=[value for value, _fragment in _MALFORMED_CALLABLE_CASES],
)
def test_a_malformed_callable_is_rejected(value: str, fragment: str) -> None:
    """A ``callable`` must be a syntactically valid ``module:function`` path. Syntax only: a missing
    module must not fail loading -- ``doctor`` and ``TargetUnavailable`` own that answer."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("library", callable=value))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.callable"
    assert fragment in str(excinfo.value), (
        f"callable {value!r}: expected {fragment!r} in {str(excinfo.value)!r}"
    )


@pytest.mark.parametrize(
    "value",
    ["example_target:generate", "pkg.module:generate", "pkg.mod:Class.make"],
    ids=["plain", "dotted-module", "dotted-attribute"],
)
def test_a_well_formed_callable_is_accepted(value: str) -> None:
    """The documented shapes are accepted, so the rule is not narrower than the contract."""
    schema.validate(_target_document("library", callable=value))


#: One out-of-range value per numeric rule: the dotted key, a value on the wrong side of its
#: bound, and the wording the diagnostic must carry. Neither the value nor the wording is a
#: threshold this module invented -- each bound is the one the schema declares and reports.
_RANGE_CASES: Final[tuple[tuple[str, Any, str], ...]] = tuple(
    sorted(
        (
            ("seed", -1, "must be a non-negative integer"),
            ("seed", -(2**63), "must be a non-negative integer"),
            ("loc_limit", 0, "must be at least 1"),
            ("loc_limit", -5, "must be at least 1"),
            ("targets.target.timeout_s", 0, "must be greater than 0"),
            ("targets.target.timeout_s", -1.5, "must be greater than 0"),
            ("targets.target.contrast_floor", 1.0, "must be greater than 1"),
            ("targets.target.contrast_floor", 0.5, "must be greater than 1"),
            ("checks.performance.encode_p95_ms", 0, "must be greater than 0"),
            ("checks.performance.decode_p95_ms", 0.0, "must be greater than 0"),
            ("checks.performance.min_symbols_per_minute", 0, "must be greater than 0"),
            ("checks.performance.encode_samples", 9, "must be at least 10"),
            ("checks.performance.decode_samples", 9, "must be at least 10"),
            ("checks.performance.warmup", -1, "must be at least 0"),
            ("checks.performance.decode_image_px", 63, "must be at least 64"),
            ("checks.damage.occlusion_margin", -0.01, "must be at least 0"),
            ("checks.damage.occlusion_margin", 0.10, "must be less than 0.1"),
            ("checks.damage.jpeg_quality", 0, "must be at least 1"),
            ("checks.damage.jpeg_quality", 96, "must be at most 95"),
            ("checks.damage.noise_sigma", -1, "must be at least 0"),
            ("checks.damage.rotation_deg", 0, "must be greater than 0"),
            ("checks.damage.rotation_deg", 46, "must be at most 45"),
            ("checks.damage.perspective_deg", 31, "must be at most 30"),
            ("checks.damage.blur_radius_per_module", 0, "must be greater than 0"),
            ("checks.damage.downscale_px_per_module", 0, "must be greater than 0"),
            ("checks.damage.logo_area_fraction", 0, "must be greater than 0"),
            ("checks.damage.logo_area_fraction", 0.26, "must be at most 0.25"),
        ),
        key=lambda case: (case[0], case[2], repr(case[1])),
    )
)


@pytest.mark.parametrize(
    "dotted, value, fragment",
    _RANGE_CASES,
    ids=[f"{dotted}={value!r}" for dotted, value, _fragment in _RANGE_CASES],
)
def test_a_value_outside_its_range_is_rejected(
    dotted: str, value: Any, fragment: str
) -> None:
    """Every numeric rule refuses the value on the wrong side of its bound, naming the key."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_document_for(dotted, value))
    assert excinfo.value.key == dotted, f"expected key {dotted!r}, got {excinfo.value.key!r}"
    assert fragment in str(excinfo.value), (
        f"{dotted}: expected {fragment!r} in {str(excinfo.value)!r}"
    )


#: Every value at or inside a bound the contract states, so the rules are not narrower than the
#: declaration. Sorted by (key, wording) like the table above.
_ACCEPTED_RANGE_CASES: Final[tuple[tuple[str, Any], ...]] = tuple(
    sorted(
        (
            ("seed", 0),
            ("loc_limit", 1),
            ("targets.target.timeout_s", 0.001),
            ("targets.target.contrast_floor", 1.5),
            ("checks.performance.encode_p95_ms", 0.5),
            ("checks.performance.encode_samples", 10),
            ("checks.performance.warmup", 0),
            ("checks.performance.decode_image_px", 64),
            ("checks.damage.occlusion_margin", 0),
            ("checks.damage.occlusion_margin", 0.099),
            ("checks.damage.jpeg_quality", 1),
            ("checks.damage.jpeg_quality", 95),
            ("checks.damage.noise_sigma", 0),
            ("checks.damage.rotation_deg", 45),
            ("checks.damage.perspective_deg", 30),
            ("checks.damage.logo_area_fraction", 0.25),
        ),
        key=lambda case: (case[0], repr(case[1])),
    )
)


@pytest.mark.parametrize(
    "dotted, value",
    _ACCEPTED_RANGE_CASES,
    ids=[f"{dotted}={value!r}" for dotted, value in _ACCEPTED_RANGE_CASES],
)
def test_a_value_inside_its_range_is_accepted(dotted: str, value: Any) -> None:
    """The inclusive bounds the contract states are inside the range: ``occlusion_margin`` is
    the one exclusive upper bound (0.10 would consume all L-level recovery), hence 0.099 here."""
    schema.validate(_document_for(dotted, value))


def test_a_negative_seed_renders_the_contract_diagnostic_exactly() -> None:
    """The reason states what is expected and :class:`schema.ConfigError` supplies the dotted key,
    so the rendered diagnostic carries the key once -- ``seed: must be a non-negative integer`` --
    exactly as every other rule does. A fragment that repeats the key would print it twice."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_document_for("seed", -1))
    assert excinfo.value.key == "seed", excinfo.value.key
    assert str(excinfo.value) == "seed: must be a non-negative integer", str(excinfo.value)


@pytest.mark.parametrize(
    "command, fragment",
    [
        ([], "must list at least one argument"),
        ([1], "argument 0 must be a string, got integer"),
        ([""], "argument 0 must not be an empty string"),
    ],
    ids=["empty", "non-string", "empty-string"],
)
def test_a_command_that_is_not_a_non_empty_argv_is_rejected(
    command: list[Any], fragment: str
) -> None:
    """``command`` must be a non-empty argv of non-empty strings."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("cli", command=command))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.command"
    assert fragment in str(excinfo.value), str(excinfo.value)


def test_a_command_argument_that_is_not_a_string_is_rejected() -> None:
    """A non-string member is named by index and kind rather than crashing the walk."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("cli", command=["generate", 1]))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.command"
    assert "argument 1 must be a string, got integer" in str(excinfo.value), str(excinfo.value)


def test_an_empty_command_argument_is_rejected() -> None:
    """An empty string would spawn nothing meaningful, so it is refused by index."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("cli", command=[""]))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.command"
    assert "argument 0 must not be an empty string" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("key", ["payload_arg", "output_arg"], ids=["payload-arg", "output-arg"])
def test_an_argument_template_with_neither_a_placeholder_nor_a_flag_is_rejected(key: str) -> None:
    """A template either carries ``{}`` for the value or starts with ``-`` as a flag."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("cli", **{key: "argument"}))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.{key}"
    assert "must contain '{}' or start with '-'" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "value",
    ["{}", "--payload={}", "-o", "--output"],
    ids=["positional", "flag-inline", "flag-short", "flag-long"],
)
def test_an_argument_template_with_a_placeholder_or_a_flag_is_accepted(value: str) -> None:
    """Both documented template shapes are accepted."""
    schema.validate(_target_document("cli", payload_arg=value, output_arg=value))


# The argument templates: the same one vocabulary, refused at load
#
# ``payload_arg`` and ``output_arg`` are templates over the placeholders a request carries, while
# ``command`` is a literal argv. Both facts are decided when the document is read, so a typo is the
# exit-2 configuration fault that names its key rather than an adapter ``ValueError`` raised once
# checks are already running.

#: Every placeholder an argument template may carry: the bare ``{}`` and one token per field of
#: ``models.TEMPLATE_FIELDS``, built from that one declaration so a field added there is covered.
_ARG_TEMPLATE_PLACEHOLDERS: Final[tuple[str, ...]] = tuple(
    f"{{{field}}}" for field in sorted(models.TEMPLATE_FIELDS)
)


@pytest.mark.parametrize(
    "placeholder",
    ["{}", *_ARG_TEMPLATE_PLACEHOLDERS],
    ids=["bare", *sorted(models.TEMPLATE_FIELDS)],
)
def test_every_placeholder_of_the_one_vocabulary_is_accepted_in_an_argument_template(
    placeholder: str,
) -> None:
    """A flag template may interpolate the bare ``{}`` or any of the eight named placeholders."""
    schema.validate(
        _target_document(
            "cli", payload_arg=f"--payload={placeholder}", output_arg=f"--out={placeholder}"
        )
    )


@pytest.mark.parametrize("key", ["payload_arg", "output_arg"], ids=["payload-arg", "output-arg"])
@pytest.mark.parametrize("token", ["{payload_base64}", "{payload}"], ids=["near-miss", "unknown"])
def test_an_unknown_placeholder_in_an_argument_template_is_rejected(key: str, token: str) -> None:
    """A token outside the vocabulary is refused at load, naming the key and the token: the CLI
    adapter substitutes only the eight fields and the bare ``{}``, so anything else would reach it
    as a ``ValueError`` after checks had started, and the run would report an internal error with a
    written report instead of this diagnostic. The message names the allowed vocabulary once and
    takes its key from :class:`schema.ConfigError`, so the dotted path appears exactly once."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("cli", **{key: f"--argument={token}"}))

    dotted = f"targets.{_TARGET_NAME}.{key}"
    assert excinfo.value.key == dotted, excinfo.value.key
    message = str(excinfo.value)
    assert token in message, message
    for allowed in ("{}", *_ARG_TEMPLATE_PLACEHOLDERS):
        assert allowed in message, message
    assert message.count(dotted) == 1, message


@pytest.mark.parametrize(
    ("command", "index", "token"),
    [
        (["generate", "--ec", "{ec_level}"], 2, "{ec_level}"),
        (["generate", "{payload_b64}"], 1, "{payload_b64}"),
        (["generate", "{}"], 1, "{}"),
    ],
    ids=["known-name", "known-name-alone", "bare"],
)
def test_a_command_word_that_carries_a_placeholder_is_rejected(
    command: list[str], index: int, token: str
) -> None:
    """``command`` is a literal argv, so a word carrying any placeholder -- a known name or the
    bare ``{}`` -- would reach the target as its own braces, which no adapter renders. It is
    refused by index and token, and the message says where a placeholder does belong."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("cli", command=command))

    dotted = f"targets.{_TARGET_NAME}.command"
    assert excinfo.value.key == dotted, excinfo.value.key
    message = str(excinfo.value)
    assert f"argument {index}" in message, message
    assert f"'{token}'" in message, message
    assert "payload_arg" in message and "output_arg" in message, message
    assert message.count(dotted) == 1, message


@pytest.mark.parametrize(
    "word",
    ['--config={"a":1}', "--spec={size:d}", "{not-a-placeholder}", "--name=}"],
    ids=["json-object", "format-spec", "hyphenated", "stray-brace"],
)
def test_a_command_word_whose_braces_are_not_placeholder_shaped_is_accepted(word: str) -> None:
    """Only a brace-delimited name of ``[A-Za-z0-9_]`` is a placeholder; every other brace sequence
    is literal text a real command line carries, so it stays accepted."""
    schema.validate(_target_document("cli", command=["generate", word]))


def test_an_unknown_metadata_field_is_rejected() -> None:
    """``metadata_map``'s keys are a closed vocabulary: a field the framework does not report has
    nowhere to go, so naming one is refused while the document is still a document."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("library", metadata_map={"version": "v", "width": "w"}))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.metadata_map"
    assert "'width'" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "member", sorted(models.METADATA_FIELD_NAMES), ids=sorted(models.METADATA_FIELD_NAMES)
)
def test_every_metadata_field_the_framework_reports_is_accepted(member: str) -> None:
    """Each of the four framework metadata fields can be mapped."""
    schema.validate(_target_document("library", metadata_map={member: "payload.field"}))


# The cross-key rule: default_adapter names a declared target


def test_an_undeclared_default_adapter_is_rejected_and_lists_the_targets_sorted() -> None:
    """``default_adapter`` must name a declared target, and the message lists them sorted: the text
    reaches stderr and CI logs are compared, so it never follows the document's own order."""
    document = {
        "default_adapter": "missing",
        "targets": {
            "zeta": {"type": "reference"},
            "alpha": {"type": "reference"},
        },
    }
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(document)
    assert excinfo.value.key == "default_adapter"
    message = str(excinfo.value)
    expected = f"declared: {sorted(document['targets'])}"
    assert expected in message, f"expected {expected!r} in {message!r}"
    assert message.index("'alpha'") < message.index("'zeta'"), (
        f"the declared targets are not listed in sorted order: {message!r}"
    )


def test_a_document_declaring_no_target_at_all_is_rejected() -> None:
    """The fallback ``default_adapter`` names nothing, so a usable document declares a target."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate({})
    assert excinfo.value.key == "default_adapter"
    assert "declared: []" in str(excinfo.value), str(excinfo.value)


def test_validation_and_defaulting_never_mutate_the_document() -> None:
    """Both entry points are pure: the parsed document is left exactly as it arrived."""
    document = _target_document("http")
    snapshot = copy.deepcopy(document)
    schema.validate(document)
    schema.apply_defaults(document)
    assert document == snapshot, "validate or apply_defaults mutated the document it was given"


# The offline boundary: loopback hosts only, http or https, no credentials

#: URLs whose host is one of ``models.LOOPBACK_HOSTS``, in both schemes and both address families.
_ACCEPTED_URLS: Final[tuple[str, ...]] = tuple(
    sorted(
        (
            "http://127.0.0.1/generate",
            "http://127.0.0.1:8000/generate",
            "https://127.0.0.1:8443/generate",
            "http://localhost/generate",
            "http://localhost:8000/generate",
            "https://localhost:8443/generate",
            "http://[::1]/generate",
            "http://[::1]:8000/generate",
            "HTTP://127.0.0.1/generate",
        )
    )
)

#: URLs the offline boundary refuses, with the wording the diagnostic must carry.
_REJECTED_URLS: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        (
            ("https://example.com/generate", "url host 'example.com' is not a loopback host"),
            ("http://192.168.0.10:8000/generate", "is not a loopback host"),
            ("http://127.0.0.1.example.com/generate", "is not a loopback host"),
            ("file:///tmp/symbol.png", "url scheme 'file' is not http or https"),
            ("ftp://127.0.0.1/symbol.png", "url scheme 'ftp' is not http or https"),
            ("", "url scheme '' is not http or https"),
            ("https://user:not-a-real-password@127.0.0.1/generate", "must not carry a username"),
        ),
        key=lambda case: (case[0], case[1]),
    )
)


@pytest.mark.parametrize("url", _ACCEPTED_URLS, ids=_ACCEPTED_URLS)
def test_a_loopback_url_is_accepted(url: str) -> None:
    """The three loopback hosts are accepted over http and https, ports and brackets included: this
    is the product's only outbound control, so the framework is verifiably offline."""
    schema.validate(_http_document(url=url))


@pytest.mark.parametrize(
    "url, fragment", _REJECTED_URLS, ids=[url or "empty" for url, _fragment in _REJECTED_URLS]
)
def test_a_url_outside_the_offline_boundary_is_rejected(url: str, fragment: str) -> None:
    """A remote host, a non-http scheme or an embedded credential is a configuration error."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(url=url))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.url"
    assert fragment in str(excinfo.value), f"{url!r}: expected {fragment!r} in {excinfo.value}"


def test_every_declared_loopback_host_is_accepted() -> None:
    """Every host ``models.LOOPBACK_HOSTS`` declares is accepted in URL form, read from its one
    declaration, so a host added there is exercised here without an edit."""
    for host in models.LOOPBACK_HOSTS:
        bracketed = f"[{host}]" if ":" in host else host
        schema.validate(_http_document(url=f"http://{bracketed}:8000/generate"))


# The request template: placeholders come from the one vocabulary


@pytest.mark.parametrize(
    "field", sorted(models.TEMPLATE_FIELDS), ids=sorted(models.TEMPLATE_FIELDS)
)
def test_every_documented_placeholder_is_accepted(field: str) -> None:
    """Each placeholder of the one vocabulary may appear in a template leaf."""
    schema.validate(_http_document(request_template={"payload": "{" + field + "}"}))


def test_a_template_using_every_placeholder_at_once_is_accepted() -> None:
    """One template carrying every placeholder is accepted, which is the HTTP adapter's contract."""
    template = {field: "{" + field + "}" for field in sorted(models.TEMPLATE_FIELDS)}
    schema.validate(_http_document(request_template=template))


@pytest.mark.parametrize(
    "template, suffix",
    [
        ({"payload": "{nope}"}, "payload"),
        ({"data": {"image": "{nope}"}}, "data.image"),
        ({"items": ["{nope}"]}, "items[0]"),
    ],
    ids=["flat", "nested", "array"],
)
def test_an_unknown_or_malformed_placeholder_names_its_leaf(
    template: Mapping[str, Any], suffix: str
) -> None:
    """An unknown placeholder is reported at the leaf that carries it, not at the whole table."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template=template))
    expected = f"targets.{_TARGET_NAME}.request_template.{suffix}"
    assert excinfo.value.key == expected, f"expected key {expected!r}, got {excinfo.value.key!r}"
    assert "unknown placeholder 'nope'" in str(excinfo.value), str(excinfo.value)


def test_a_malformed_placeholder_string_names_its_leaf() -> None:
    """An unclosed placeholder is reported at its own leaf, with the parser's reason kept."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template={"payload": "{payload_b64"}))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.request_template.payload"
    assert "malformed placeholder" in str(excinfo.value), str(excinfo.value)


def test_an_unknown_placeholder_in_a_json_string_template_is_rejected() -> None:
    """A template written as JSON text is held to exactly the rule a table is."""
    text = json.dumps({"payload": "{nope}"})
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template=text))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.request_template.payload"
    assert "unknown placeholder 'nope'" in str(excinfo.value), str(excinfo.value)


def test_a_json_string_template_is_accepted() -> None:
    """The string form resolves to the same mapping the table form does, so it is accepted."""
    schema.validate(_http_document(request_template=json.dumps({"payload": "{payload_b64}"})))


@pytest.mark.parametrize(
    "template, fragment",
    [
        ("{not json", "is not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        (42, "expected table or string"),
    ],
    ids=["bad-json", "json-array", "wrong-kind"],
)
def test_a_template_that_is_not_a_json_object_is_rejected(template: Any, fragment: str) -> None:
    """A template must be a table or the JSON text of an object, and nothing else."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template=template))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.request_template"
    assert fragment in str(excinfo.value), str(excinfo.value)


def test_a_template_carrying_a_toml_datetime_is_rejected() -> None:
    """A TOML date cannot be sent as JSON, so it is refused at its own leaf with that reason."""
    template = {"emitted": datetime.datetime(2024, 1, 1, 0, 0, 0)}
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template=template))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.request_template.emitted"
    assert "TOML dates and times cannot be sent as JSON" in str(excinfo.value), str(excinfo.value)


def test_a_template_without_placeholders_or_inside_an_array_is_accepted() -> None:
    """Literal text, escaped braces and a placeholder inside an array are all accepted: a leaf is a
    placeholder only when it carries a ``{}`` field, and an array is walked element by element."""
    schema.validate(
        _http_document(
            request_template={
                "static": "no placeholders here",
                "escaped": "{{literal braces}}",
                "items": ["{payload_b64}", "{ec_level}"],
            }
        )
    )


def test_a_template_leaf_carrying_a_non_finite_number_is_rejected() -> None:
    """An infinite or NaN number cannot be sent as JSON, so it is refused at its own leaf."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template={"payload": float("inf")}))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.request_template.payload"
    assert "must be a finite number" in str(excinfo.value), str(excinfo.value)


def test_an_image_field_is_checked_for_shape_only() -> None:
    """``_body`` or a dotted path of non-empty segments is accepted; a blank segment is not."""
    for accepted in ("_body", "data.image", "result.0.image"):
        schema.validate(_http_document(image_field=accepted))
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(image_field="data..image"))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.image_field"
    assert "dotted path" in str(excinfo.value), str(excinfo.value)



# The remaining shape rules

#: Every path key, each with a value no filesystem call can use and the wording it produces.
_PATH_KEY_CASES: Final[tuple[tuple[str, str, str], ...]] = tuple(
    sorted(
        (key, value, fragment)
        for key in ("corpus_path", "report_dir")
        for value, fragment in (
            ("   ", "must be a non-empty string"),
            ("rep\u0000orts", "must not contain a NUL byte"),
        )
    )
)


@pytest.mark.parametrize(
    "key, value, fragment",
    _PATH_KEY_CASES,
    ids=[f"{key}={fragment}" for key, _value, fragment in _PATH_KEY_CASES],
)
def test_a_path_no_filesystem_call_can_use_is_rejected(
    key: str, value: str, fragment: str
) -> None:
    """A path key must be a non-empty string carrying no NUL byte: no filesystem call accepts
    one and ``pathlib`` raises ``ValueError``, surfacing as an internal error, not a named key."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_minimal_document(**{key: value}))
    assert excinfo.value.key == key, f"expected key {key!r}, got {excinfo.value.key!r}"
    assert fragment in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs_map, fragment",
    [
        ({1: "value"}, "every key must be a non-empty string"),
        ({"payload": ""}, "the value for 'payload' must be a non-empty string"),
    ],
    ids=["non-string-key", "empty-value"],
)
def test_a_kwargs_map_maps_non_empty_strings_to_non_empty_strings(
    kwargs_map: Mapping[Any, Any], fragment: str
) -> None:
    """``kwargs_map`` is checked for shape only: its keys are whatever the callable takes."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_target_document("library", kwargs_map=kwargs_map))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.kwargs_map"
    assert fragment in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "template, expected_key, fragment",
    [
        (
            {1: "{payload_b64}"},
            f"targets.{_TARGET_NAME}.request_template",
            "every request_template key must be a non-empty string",
        ),
        (
            {"payload": object()},
            f"targets.{_TARGET_NAME}.request_template.payload",
            "unsupported value of type",
        ),
    ],
    ids=["non-string-key", "unsupported-value"],
)
def test_a_template_key_and_leaf_value_are_checked(
    template: Mapping[Any, Any], expected_key: str, fragment: str
) -> None:
    """A template is sent as JSON, so only what JSON carries is accepted: a non-string key or a leaf
    JSON cannot represent is refused while the document is still a document."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(request_template=template))
    assert excinfo.value.key == expected_key, (
        f"expected key {expected_key!r}, got {excinfo.value.key!r}"
    )
    assert fragment in str(excinfo.value), str(excinfo.value)


def test_a_table_key_that_is_not_a_string_is_rejected() -> None:
    """A document whose table has a non-string key is refused before it is walked."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate({1: "value"})
    assert excinfo.value.key == "", f"a whole-document fault carries no key: {excinfo.value.key!r}"
    assert "table keys must be strings" in str(excinfo.value), str(excinfo.value)


def test_an_unparseable_url_is_rejected() -> None:
    """A URL the parser cannot split is named as unparseable rather than raising ``ValueError``."""
    with pytest.raises(schema.ConfigError) as excinfo:
        schema.validate(_http_document(url="http://[::1/generate"))
    assert excinfo.value.key == f"targets.{_TARGET_NAME}.url"
    assert "not a parseable URL" in str(excinfo.value), str(excinfo.value)



# Exit 2 through the real command line, with no report written


def _cli_case_documents() -> tuple[tuple[Mapping[str, Any], str], ...]:
    """Return one complete document per fault the command line is driven with: the valid minimal
    document with a single fault, so a case is never about a document broken twice."""
    unknown_top_level = _minimal_document()
    unknown_top_level["nonsense"] = 1
    unknown_checks_key = _minimal_document()
    unknown_checks_key["checks"] = {"damage": {"bogus": 1}}
    return tuple(
        sorted(
            (
                (unknown_top_level, "nonsense"),
                (unknown_checks_key, "checks.damage.bogus"),
                (
                    _target_document("http", url="https://example.com/generate"),
                    f"targets.{_TARGET_NAME}.url",
                ),
                (
                    _target_document("library", returns="stream"),
                    f"targets.{_TARGET_NAME}.returns",
                ),
                (_minimal_document(default_adapter="gateway"), "default_adapter"),
                (_minimal_document(seed=-1), "seed"),
            ),
            key=lambda case: (case[1], repr(case[0])),
        )
    )


#: A representative subset of the faults above, as complete documents the command line loads.
_CLI_CASES: Final[tuple[tuple[Mapping[str, Any], str], ...]] = _cli_case_documents()

#: The three artefacts the report writers produce; an exit-2 path must leave none of them behind.
_REPORT_ARTEFACT_NAMES: Final[tuple[str, ...]] = ("index.html", "junit.xml", "summary.json")


@pytest.mark.parametrize(
    "document, dotted",
    _CLI_CASES,
    ids=[dotted for _document, dotted in _CLI_CASES],
)
def test_a_configuration_fault_exits_two_and_writes_no_report(
    cli_runner: Any,
    toml_config_file: Any,
    document: Mapping[str, Any],
    dotted: str,
) -> None:
    """A configuration fault reaches the shell as exit 2, naming the key, with no report written:
    the code is decided before any check runs, so no half-written report can reach a consumer."""
    path = toml_config_file(_document_text(document))
    result = cli_runner.invoke(
        qrcheck_cli, ["--config", str(path), "run"], catch_exceptions=False
    )

    assert result.exit_code == 2, (
        f"expected exit 2 for {dotted}, got {result.exit_code}; stderr={result.stderr!r}"
    )
    assert dotted in result.stderr, (
        f"expected {dotted!r} on stderr, got {result.stderr!r}"
    )
    assert "configuration error" in result.stderr, result.stderr
    assert result.stdout == "", (
        f"a configuration fault must print nothing on stdout: {result.stdout!r}"
    )

    report_dir = path.parent / _declared_default("", "report_dir")
    for name in _REPORT_ARTEFACT_NAMES:
        assert not (report_dir / name).exists(), f"{dotted}: {name} was written before the gate"
    assert _report_artefacts(report_dir) == [], (
        f"{dotted}: the report directory is not empty: {_report_artefacts(report_dir)}"
    )


def test_a_valid_configuration_is_not_a_configuration_fault(
    cli_runner: Any, toml_config_file: Any
) -> None:
    """The same command line accepts the committed defaults, so exit 2 is not unconditional."""
    path = toml_config_file(_document_text(_minimal_document()))
    result = cli_runner.invoke(
        qrcheck_cli, ["--config", str(path), "doctor", "--json"], catch_exceptions=False
    )
    assert result.exit_code == 0, (
        f"a valid document was refused with exit {result.exit_code}: {result.stderr!r}"
    )
    assert "configuration error" not in result.stderr, result.stderr
    report = json.loads(result.stdout)
    assert report["adapter"] == _REFERENCE_TARGET, (
        f"doctor resolved the adapter to {report['adapter']!r}"
    )


# The deliberate negative: a capability shortfall is not a configuration error


def test_a_capability_shortfall_is_not_a_configuration_error(toml_config_file: Any) -> None:
    """A target declaring fewer capabilities than a check needs still loads, and is not exit 2:
    sufficiency depends on the run-time ``--checks`` selection (AAP §0.12, fourth conflict)."""
    shortfall = "png"
    missing = "svg"
    assert missing in models.CAPABILITY_NAMES, (
        "the shortfall asserted here is not a capability the adapter layer knows"
    )
    document = _target_document("reference", capabilities=[shortfall])

    schema.validate(document)  # must not raise: the document is right on its own terms
    config = load_config(toml_config_file(_document_text(document)))

    target = config.targets[_TARGET_NAME]
    assert target.capabilities == frozenset({shortfall}), (
        f"the declared capability set resolved to {target.capabilities!r}"
    )
    assert missing not in target.capabilities, "the shortfall under test has disappeared"


# load_config: resolution, creation, immutability and the two failure channels


def test_a_relative_path_is_resolved_against_the_configuration_files_directory(
    toml_config_file: Any,
) -> None:
    """Both paths resolve against the document's own directory, never the working directory."""
    path = toml_config_file(_document_text(_minimal_document()))
    config = load_config(path)
    base = path.parent
    assert config.path == path.resolve(), f"config path resolved to {config.path}"
    assert config.report_dir == (base / _declared_default("", "report_dir")).resolve(), (
        f"report_dir resolved to {config.report_dir}, not against {base}"
    )
    assert config.corpus_path == (base / _declared_default("", "corpus_path")).resolve(), (
        f"corpus_path resolved to {config.corpus_path}, not against {base}"
    )
    for resolved in (config.report_dir, config.corpus_path):
        assert resolved.is_absolute(), f"{resolved} is not absolute"
        assert resolved.is_relative_to(base), f"{resolved} escaped the document's directory"


def test_an_absolute_path_is_honoured_as_written(tmp_path: Path) -> None:
    """An absolute report directory in the document is used unchanged, not re-rooted."""
    absolute = tmp_path / "absolute-reports"
    document = _minimal_document(report_dir=str(absolute))
    path = tmp_path / "qrcheck.toml"
    path.write_text(_document_text(document), encoding="utf-8")
    config = load_config(path)
    assert config.report_dir == absolute.resolve(), (
        f"an absolute report_dir was re-rooted to {config.report_dir}"
    )


def test_the_report_directory_is_created_and_left_empty(toml_config_file: Any) -> None:
    """The loader creates ``report_dir`` and removes its writability probe: the probe proves the
    directory is writable for the reason the run will hit, and must leave no trace behind."""
    path = toml_config_file(_document_text(_minimal_document()))
    config = load_config(path)
    assert config.report_dir.is_dir(), f"{config.report_dir} was not created"
    assert _report_artefacts(config.report_dir) == [], (
        f"the loader left {_report_artefacts(config.report_dir)} behind"
    )


def test_the_loaded_records_are_immutable(toml_config_file: Any) -> None:
    """Every resolved record refuses mutation, so a run cannot drift once the file is read: a
    read-only target mapping, an argv tuple, a capability ``frozenset`` and the frozen records."""
    config = load_config(toml_config_file(_document_text(_target_document("cli"))))
    target = config.targets[_TARGET_NAME]
    assert isinstance(target, models.CLITarget), f"resolved a {type(target).__name__}"
    assert isinstance(target.command, tuple), f"command resolved to {type(target.command).__name__}"
    assert isinstance(target.capabilities, frozenset), (
        f"capabilities resolved to {type(target.capabilities).__name__}"
    )
    with pytest.raises(TypeError):
        config.targets["another"] = target
    with pytest.raises(TypeError):
        target.command[0] = "other"
    with pytest.raises(TypeError):
        target.metadata_map[models.METADATA_FIELD_NAMES[0]] = "payload.field"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.timeout_s = 1.0


def test_a_report_directory_that_cannot_be_created_raises_oserror(tmp_path: Path) -> None:
    """An uncreatable ``report_dir`` is an environment fault, never a ``ConfigError``: a regular
    file stands where the parent must be, so ``mkdir`` fails for every user, root included."""
    blocker = tmp_path / "a-regular-file"
    blocker.write_text("a directory cannot be created below this\n", encoding="utf-8")
    document = _minimal_document(report_dir=str(blocker / _declared_default("", "report_dir")))
    path = tmp_path / "qrcheck.toml"
    path.write_text(_document_text(document), encoding="utf-8")

    with pytest.raises(OSError) as excinfo:
        load_config(path)
    assert not isinstance(excinfo.value, schema.ConfigError), (
        "an unusable report directory must not read as a configuration fault"
    )
    assert isinstance(excinfo.value, OSError)


def test_a_failing_writability_probe_raises_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing probe write is an environment fault too, found by the probe itself: the loader's
    own ``tempfile`` is replaced, so only the probe is affected and ``chmod`` is not used."""
    path = tmp_path / "qrcheck.toml"
    path.write_text(_document_text(_minimal_document()), encoding="utf-8")

    class _RefusingTempfile:
        """A stand-in for :mod:`tempfile` whose probe write always fails."""

        @staticmethod
        def NamedTemporaryFile(**_kwargs: Any) -> Any:
            """Raise the ``PermissionError`` a read-only directory produces."""
            raise PermissionError("report directory is not writable")

    monkeypatch.setattr(load_module, "tempfile", _RefusingTempfile)

    with pytest.raises(OSError) as excinfo:
        load_config(path)
    assert not isinstance(excinfo.value, schema.ConfigError), (
        "a failing writability probe must not read as a configuration fault"
    )


def test_a_missing_configuration_file_is_a_configuration_fault(tmp_path: Path) -> None:
    """A file that cannot be read is exit 2 and carries no key, because no key owns it."""
    missing = tmp_path / "absent.toml"
    with pytest.raises(schema.ConfigError) as excinfo:
        load_config(missing)
    assert excinfo.value.key == "", f"a whole-file fault carries no key: {excinfo.value.key!r}"
    assert str(missing.resolve()) in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "content, fragment",
    [
        ("seed = \n", "invalid TOML"),
        (b"seed = \xff\xfe\n", "not valid UTF-8"),
    ],
    ids=["invalid-toml", "not-utf8"],
)
def test_an_unparsable_document_is_a_configuration_fault(
    tmp_path: Path, content: Any, fragment: str
) -> None:
    """A document that cannot be parsed is exit 2, naming the file rather than a key."""
    path = tmp_path / "qrcheck.toml"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(schema.ConfigError) as excinfo:
        load_config(path)
    assert excinfo.value.key == ""
    assert fragment in str(excinfo.value), str(excinfo.value)


def test_loading_the_same_file_twice_returns_equal_records(toml_config_file: Any) -> None:
    """Loading is deterministic: the same document resolves to equal records every time."""
    path = toml_config_file(_document_text(_minimal_document()))
    assert load_config(path) == load_config(path)


# The committed qrcheck.toml


def test_the_committed_configuration_validates_and_holds_the_declared_defaults() -> None:
    """The repository's own document is valid and carries the declared fallbacks, read with
    :mod:`tomllib` rather than ``load_config``, which would create ``reports/`` in the checkout."""
    assert _COMMITTED_CONFIG.is_file(), f"{_COMMITTED_CONFIG} is missing"
    raw = tomllib.loads(_COMMITTED_CONFIG.read_text(encoding="utf-8"))
    schema.validate(raw)

    defaults = schema.apply_defaults({})
    for key in _TOP_LEVEL_KEYS:
        assert key in raw, f"the committed configuration omits the top-level key {key!r}"
        assert raw[key] == defaults[key], (
            f"the committed {key} is {raw[key]!r}, the declared default is {defaults[key]!r}"
        )
    assert raw["render"] == defaults["render"], (
        f"the committed [render] is {raw['render']!r}, the declared default is "
        f"{defaults['render']!r}"
    )
    for child in sorted(defaults["checks"]):
        assert raw["checks"][child] == defaults["checks"][child], (
            f"the committed [checks.{child}] differs from the declared defaults: "
            f"{sorted(set(raw['checks'].get(child, {})) ^ set(defaults['checks'][child]))}"
        )

    declared_targets = raw["targets"]
    assert raw["default_adapter"] in declared_targets, (
        f"the committed default_adapter {raw['default_adapter']!r} names no declared target"
    )
    for name, table in sorted(declared_targets.items()):
        assert "type" in table, f"[targets.{name}] declares no type"
        for key, spec in schema.SCHEMA["targets.*"].items():
            if spec.is_required or key not in table:
                continue
            assert table[key] == _materialised(spec.default), (
                f"[targets.{name}] {key} is {table[key]!r}, the declared default is "
                f"{spec.default!r}"
            )
