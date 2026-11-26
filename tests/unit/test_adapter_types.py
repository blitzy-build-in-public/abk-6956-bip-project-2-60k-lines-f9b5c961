"""Contract test: the adapter factory covers every target type the configuration declares.

``qrcheck/targets/__init__.py`` is the product's single extension point, so the table's keys must
be exactly ``models.ADAPTER_TYPE_NAMES``, each entry must be the class the extension model names
(compared by **identity**), every entry given a record of its own dataclass type must return an
object satisfying the :class:`~qrcheck.targets.base.Adapter` protocol (``isinstance``, never
``issubclass``: the protocol has non-method members and CPython 3.11 refuses that), an undeclared
type must raise ``KeyError`` rather than resolve to ``None``, and the capability vocabulary must
match on both sides of the configuration seam.

The capability half is load-bearing beyond this module: the three configurable adapters declare
exactly what their record carries and the two built-in self-check targets declare all thirteen
themselves, which is what lets the clean-run gate demand zero skips (AAP §0.4.1) and keeps an
expected failure of the ``defective`` target from being answered with a skip (AAP §0.4.2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pytest

from qrcheck.config import models
from qrcheck.targets import ADAPTER_TYPES
from qrcheck.targets.base import Adapter, Capability
from qrcheck.targets.cli import CliAdapter
from qrcheck.targets.defective import DefectiveAdapter
from qrcheck.targets.http import HttpAdapter
from qrcheck.targets.library import LibraryAdapter
from qrcheck.targets.reference import ReferenceAdapter

# The five classes the extension model names, keyed by the ``type`` value that must resolve to each.
# Compared by identity, so a factory entry returning a subclass or a wrapper fails even though its
# name would match; these keys are the only place this module writes the five type names.
_EXPECTED_CLASSES: Final[Mapping[str, type]] = MappingProxyType(
    {
        "cli": CliAdapter,
        "defective": DefectiveAdapter,
        "http": HttpAdapter,
        "library": LibraryAdapter,
        "reference": ReferenceAdapter,
    }
)

# The capability declaration the three configurable adapters are configured with. It is data, not a
# class attribute: it rides in the resolved record and is what ``require_capabilities`` gates on, so
# an adapter declares exactly what its configuration says, never a built-in superset.
_CONFIGURED_CAPABILITIES: Final[frozenset[str]] = frozenset({"png", "svg"})

# The three configurable adapter kinds, which declare only what their record carries, and the two
# built-in self-check targets, which declare the full set themselves. Sorted, unique and exhaustive:
# the two groups partition ``models.ADAPTER_TYPE_NAMES``.
_CONFIGURABLE_TYPES: Final[tuple[str, ...]] = ("cli", "http", "library")
_BUILT_IN_TYPES: Final[tuple[str, ...]] = ("defective", "reference")

# A loopback URL on a fixed, non-ephemeral port. Nothing here binds or dials it: the address only
# has to pass the loopback rule the schema and the HTTP adapter enforce, and a fixed port keeps the
# record byte-stable across runs.
_LOOPBACK_URL: Final[str] = "http://127.0.0.1:18100/generate"


def _record(kind: str) -> models.TargetConfig:
    """Return a minimal valid resolved record for `kind`: every value is one the adapter's own
    constructor validates, and nothing here is imported, spawned or dialled.
    """
    if kind == "library":
        return models.LibraryTarget(
            name=kind,
            type="library",
            capabilities=_CONFIGURED_CAPABILITIES,
            callable="myapp.qrgen:render_png",
            kwargs_map={"payload": "data", "ec_level": "ec"},
            returns="raw_bytes",
        )
    if kind == "cli":
        return models.CLITarget(
            name=kind,
            type="cli",
            capabilities=_CONFIGURED_CAPABILITIES,
            command=("python3", "examples/cli_target.py"),
            payload_arg="--payload-b64={payload_b64}",
            output_arg="--output={}",
            returns="file_path",
        )
    if kind == "http":
        return models.HTTPTarget(
            name=kind,
            type="http",
            capabilities=_CONFIGURED_CAPABILITIES,
            url=_LOOPBACK_URL,
            method="POST",
            request_template=MappingProxyType(
                {"payload_b64": "{payload_b64}", "ec_level": "{ec_level}"}
            ),
            image_field="image_b64",
            image_encoding="base64",
        )
    if kind in _BUILT_IN_TYPES:
        return models.TargetConfig(name=kind, type=kind)
    raise AssertionError(
        f"no minimal record is declared for target type {kind!r}: add one to _record() with that "
        "kind's per-type keys, and add its class to _EXPECTED_CLASSES"
    )


def _adapter(kind: str, record: models.TargetConfig) -> Adapter:
    """Build the adapter `kind` resolves to, as a run builds one, performing no I/O."""
    return ADAPTER_TYPES[kind](record, name=record.name)


@pytest.fixture
def target_records(
    config_factory: Callable[..., models.Config],
) -> Mapping[str, models.TargetConfig]:
    """Return one minimal valid target record per declared type: a sixth declared type fails in
    :func:`_record` by name.
    """
    records = {kind: _record(kind) for kind in sorted(models.ADAPTER_TYPE_NAMES)}
    config = config_factory(targets=MappingProxyType(records))
    return config.targets


def test_the_factory_table_covers_every_configured_type() -> None:
    """The table's keys and the expected-class table are both exactly the declared type set."""
    declared = sorted(models.ADAPTER_TYPE_NAMES)
    assert sorted(ADAPTER_TYPES) == declared, (
        "ADAPTER_TYPES and models.ADAPTER_TYPE_NAMES disagree about the target kinds: the table "
        f"declares {sorted(ADAPTER_TYPES)}, the configuration declares {declared}"
    )
    assert sorted(_EXPECTED_CLASSES) == declared, (
        f"this module's expected-class table declares {sorted(_EXPECTED_CLASSES)} but the "
        f"configuration declares the target kinds {declared}: a kind added to or removed from "
        "models.AdapterType must be added to or removed from _EXPECTED_CLASSES and ADAPTER_TYPES "
        "in the same change"
    )
    assert len(declared) == len(set(declared)), (
        f"models.ADAPTER_TYPE_NAMES repeats a target kind: {declared}"
    )


@pytest.mark.parametrize("adapter_type", sorted(_EXPECTED_CLASSES))
def test_the_factory_entry_is_the_class_the_extension_model_names(adapter_type: str) -> None:
    """Each entry resolves to the adapter class the extension model names, by identity."""
    factory = ADAPTER_TYPES[adapter_type]
    assert factory is _EXPECTED_CLASSES[adapter_type], (
        f"ADAPTER_TYPES[{adapter_type!r}] resolved to {factory!r}, not to the declared "
        f"{_EXPECTED_CLASSES[adapter_type]!r}"
    )


@pytest.mark.parametrize("adapter_type", sorted(models.ADAPTER_TYPE_NAMES))
def test_each_factory_entry_returns_a_protocol_conforming_adapter(
    adapter_type: str,
    target_records: Mapping[str, models.TargetConfig],
) -> None:
    """Each entry, given its own record, returns an object satisfying the ``Adapter`` port."""
    record = target_records[adapter_type]
    assert record.type == adapter_type, (
        f"the minimal record for {adapter_type!r} carries type {record.type!r}"
    )
    factory = ADAPTER_TYPES[adapter_type]
    assert callable(factory), f"ADAPTER_TYPES[{adapter_type!r}] is not callable: {factory!r}"

    adapter = _adapter(adapter_type, record)
    try:
        # ``Adapter`` is a runtime-checkable Protocol, so ``isinstance`` checks presence of the four
        # members only -- on CPython 3.11 ``issubclass`` raises for a protocol with non-method
        # members. The types of ``name`` and ``capabilities`` are asserted separately, since a
        # ``list`` would pass as well.
        assert isinstance(adapter, Adapter), (
            f"the {adapter_type!r} factory returned {adapter!r}, which does not satisfy the "
            "Adapter protocol"
        )
        assert isinstance(adapter.name, str) and adapter.name, (
            f"the {adapter_type!r} adapter's name is not a non-empty str: {adapter.name!r}"
        )
        assert adapter.name == record.name, (
            f"the {adapter_type!r} adapter reports the name {adapter.name!r}, not the configured "
            f"target name {record.name!r}"
        )
        assert isinstance(adapter.capabilities, frozenset), (
            f"the {adapter_type!r} adapter declares its capabilities as "
            f"{type(adapter.capabilities).__name__}, not a frozenset"
        )
        for capability in sorted(adapter.capabilities, key=str):
            assert isinstance(capability, Capability), (
                f"the {adapter_type!r} adapter declares {capability!r}, which is not a Capability "
                "member"
            )
        assert callable(adapter.generate), (
            f"the {adapter_type!r} adapter's generate attribute is not callable"
        )
        assert callable(adapter.close), (
            f"the {adapter_type!r} adapter's close attribute is not callable"
        )
        # Closing twice must not raise: the runner closes every adapter once the run ends, including
        # after a failure, so a raise on an already-released client would turn a completed run into
        # an error.
        adapter.close()
        adapter.close()
    finally:
        adapter.close()


@pytest.mark.parametrize("adapter_type", sorted(models.ADAPTER_TYPE_NAMES))
def test_the_constructor_defaults_the_name_to_the_declared_type(
    adapter_type: str,
    target_records: Mapping[str, models.TargetConfig],
) -> None:
    """A factory call without a name falls back to the record's ``type``."""
    adapter = ADAPTER_TYPES[adapter_type](target_records[adapter_type])
    try:
        assert adapter.name == adapter_type, (
            f"the {adapter_type!r} adapter built without a name reports {adapter.name!r}, not its "
            f"declared type {adapter_type!r}"
        )
    finally:
        adapter.close()


@pytest.mark.parametrize("adapter_type", _CONFIGURABLE_TYPES)
def test_the_configurable_adapters_declare_exactly_the_configured_capabilities(
    adapter_type: str,
    target_records: Mapping[str, models.TargetConfig],
) -> None:
    """A configurable adapter declares exactly the capability set its record carries."""
    record = target_records[adapter_type]
    assert record.capabilities == _CONFIGURED_CAPABILITIES, (
        f"the minimal record for {adapter_type!r} carries {sorted(record.capabilities)}, not the "
        f"configured {sorted(_CONFIGURED_CAPABILITIES)}"
    )
    configured = frozenset(Capability(name) for name in _CONFIGURED_CAPABILITIES)
    adapter = _adapter(adapter_type, record)
    try:
        assert adapter.capabilities == configured, (
            f"the {adapter_type!r} adapter declares {sorted(adapter.capabilities, key=str)}, not "
            f"exactly the configured {sorted(configured, key=str)}"
        )
    finally:
        adapter.close()


@pytest.mark.parametrize("adapter_type", _BUILT_IN_TYPES)
def test_the_built_in_targets_declare_every_capability(
    adapter_type: str,
    target_records: Mapping[str, models.TargetConfig],
) -> None:
    """Both built-in self-check targets declare all thirteen capabilities themselves."""
    record = target_records[adapter_type]
    assert record.capabilities == frozenset(), (
        f"the record for the built-in {adapter_type!r} target must declare no capabilities, so "
        f"the declared set is the adapter's own; it carries {sorted(record.capabilities)}"
    )
    assert len(Capability) == 13, (
        f"the capability vocabulary has {len(Capability)} members; the built-in targets and the "
        "zero-skip reference gate are specified against thirteen"
    )
    adapter = _adapter(adapter_type, record)
    try:
        assert adapter.capabilities == frozenset(Capability), (
            f"the {adapter_type!r} adapter declares "
            f"{sorted(adapter.capabilities, key=str)}; it must declare all thirteen, or a check "
            "needing one of the missing members would report skipped and the zero-skip reference "
            "gate would fail for the wrong reason"
        )
    finally:
        adapter.close()


def test_the_capability_groups_partition_the_declared_types() -> None:
    """The configurable types and the built-in types together are exactly the declared types."""
    configurable = set(_CONFIGURABLE_TYPES)
    built_in = set(_BUILT_IN_TYPES)
    assert configurable.isdisjoint(built_in), (
        f"{sorted(configurable & built_in)} is declared both configurable and built in"
    )
    assert configurable | built_in == set(models.ADAPTER_TYPE_NAMES), (
        "the capability assertions leave "
        f"{sorted(set(models.ADAPTER_TYPE_NAMES) - (configurable | built_in))} uncovered and claim "
        f"{sorted((configurable | built_in) - set(models.ADAPTER_TYPE_NAMES))} that no type "
        "declares"
    )


def test_an_undeclared_type_raises_key_error() -> None:
    """An undeclared ``type`` raises, rather than resolving to ``None``."""
    undeclared = "grpc"
    assert undeclared not in ADAPTER_TYPES, (
        f"{undeclared!r} is declared as a target type, so this test's premise no longer holds"
    )
    with pytest.raises(KeyError):
        ADAPTER_TYPES[undeclared]


def test_the_factory_table_cannot_be_mutated() -> None:
    """The table is configuration vocabulary, so no caller can redirect a run by editing it."""
    with pytest.raises(TypeError):
        ADAPTER_TYPES["grpc"] = object()  # type: ignore[index]
    assert sorted(ADAPTER_TYPES) == sorted(models.ADAPTER_TYPE_NAMES), (
        "a refused mutation still changed the table: it now declares "
        f"{sorted(ADAPTER_TYPES)} instead of {sorted(models.ADAPTER_TYPE_NAMES)}"
    )


def test_the_capability_vocabulary_matches_the_configuration() -> None:
    """``models.CAPABILITY_NAMES`` and the ``Capability`` enum hold the same members."""
    member_names = tuple(member.value for member in Capability)
    assert len(member_names) == 13, (
        f"the Capability enum declares {len(member_names)} members, not thirteen; SY-MATRIX-EQUIV, "
        "the RN-* checks and IV-OVERCAPACITY all key on the declared thirteen"
    )
    assert set(models.CAPABILITY_NAMES) == set(member_names), (
        "the configuration's capability vocabulary and the adapter layer's enum disagree on "
        f"{sorted(set(models.CAPABILITY_NAMES) ^ set(member_names))}"
    )
    # A StrEnum member *is* its value, so the configuration's frozenset[str] and the adapter layer's
    # frozenset[Capability] compare and hash equal across the seam without either side importing the
    # other; otherwise capability intersection would quietly stop matching and every gated check
    # skip.
    assert frozenset(Capability) == frozenset(models.CAPABILITY_NAMES), (
        "Capability members no longer compare equal to their configured names: "
        f"{sorted(frozenset(Capability) ^ frozenset(models.CAPABILITY_NAMES), key=str)}"
    )
    for name in sorted(models.CAPABILITY_NAMES):
        assert Capability(name).value == name, (
            f"the configured capability {name!r} does not convert to a Capability member of the "
            "same name"
        )

