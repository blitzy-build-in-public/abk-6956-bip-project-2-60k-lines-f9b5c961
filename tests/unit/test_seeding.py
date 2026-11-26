"""Determinism proofs for ``qrcheck/config/seeding.py``, the only seed-derivation point.

``derive_seed`` turns a run seed into the seed of one draw, and two properties make a run
reproducible from the ``seed`` its report records: the derivation is a stated formula,
recomputed here from :func:`hashlib.sha256` over the same ``"|"``-joined, UTF-8-encoded parts,
so a change to it fails here instead of silently re-seeding every derived draw; and it reads no
ambient process state, returning the same values under two different ``PYTHONHASHSEED`` values,
while no module under ``qrcheck/`` reaches for ``random``, ``hash()``, ``id()``, ``os.urandom``
or the wall clock. An integer part and its decimal string derive the same seed, because every
part is stringified with ``str()``.

The parts are joined without escaping, so ``("a", "b")`` and ``("a|b")`` collide -- a documented
limitation rather than a defect, because the call sites pass identifiers (check IDs,
``corpus:<record-id>`` case IDs, class names, damage-kind tokens) that cannot contain ``"|"``,
and escaping would change every derived seed and therefore every committed
``damage-<kind>-<seed>.png`` name. Every value lies in ``0 <= seed < 2**64``, the range
``numpy.random.PCG64`` accepts directly.
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qrcheck.config import seeding as seeding_module
from qrcheck.config.seeding import derive_seed

# Resolved from the imported module, never from the working directory: a submodule's
# __file__ is always the real source file, so the package directory and the repository
# root above it are derived from the module rather than from the shell's cwd.
_SEEDING_PATH = Path(seeding_module.__file__).resolve()
_PACKAGE_DIR = _SEEDING_PATH.parents[1]  # .../qrcheck
_REPOSITORY_ROOT = _SEEDING_PATH.parents[2]  # the directory that holds qrcheck/

# Call shapes worth pinning: no parts, one integer part, several mixed parts, a
# non-ASCII part, a long part, the configured default seed and the largest run seed
# a JSON-configurable integer realistically reaches. Sorted by repr so the
# parametrisation order does not depend on the order the literal was edited in.
_DEFINITION_SPECS: tuple[tuple[object, ...], ...] = (
    (0,),
    (1, 2),
    (1, "url", 3),
    (5, "a", 0, "b"),
    (7, "DM-RESILIENCE", "corpus:url-000", "occluders"),
    (9, "a" * 300),
    (10, "RN-LOGO", "synthetic:logo-h", "logo"),
    (42, "utf8-multibyte", "caf\u00e9"),
    (20240101,),
    (20240101, "repeat-run", 17, ""),
    (12345678901234567890,),
)

DEFINITION_CASES: tuple[tuple[object, ...], ...] = tuple(sorted(_DEFINITION_SPECS, key=repr))

# Pairs that differ in exactly one way: the run seed, one part's text, one part's
# namespace, an extra part, or the order of two parts. An integer part and its
# decimal string are deliberately absent, because the derivation makes them equal and
# the equivalence is asserted rather than perturbed. Sorted by repr for the same
# reason the definition table is.
_PERTURBATION_SPECS: tuple[tuple[tuple[object, ...], tuple[object, ...]], ...] = (
    ((0, "url", 3), (1, "url", 3)),
    ((7, "utf8-multibyte", "caf\u00e9"), (7, "utf8-multibyte", "cafe")),
    (
        (20240101, "DM-RESILIENCE", "corpus:url-000", "noise"),
        (20240101, "DM-RESILIENCE", "corpus:url-000", "occluders"),
    ),
    ((20240101, "url", 3), (20240101, "url", 3, "extra")),
    ((20240101, "url", 3), (20240101, "url", 4)),
    ((20240101, "url", 3), (20240101, "vcard", 3)),
    ((20240101, "url", 3), (20240101, 3, "url")),
)

PERTURBATIONS: tuple[tuple[tuple[object, ...], tuple[object, ...]], ...] = tuple(
    sorted(_PERTURBATION_SPECS, key=repr)
)

# The integer-part shapes the product actually uses: a corpus record index passed as
# an int by qrcheck/corpus/payloads.py, and a run seed derived before it is used.
_INT_PART_SPECS: tuple[tuple[int, str, int], ...] = (
    (0, "random-bytes", 0),
    (20240101, "boundary-byte", 15),
    (20240101, "url", 29),
    (12345678901234567890, "utf8-multibyte", 23),
)

INT_PART_CASES: tuple[tuple[int, str, int], ...] = tuple(sorted(_INT_PART_SPECS, key=repr))

# The two hash seeds the subprocess runs under. PYTHONHASHSEED perturbs only
# CPython's own string hashing, so a derivation that agrees under both cannot be
# leaning on hash or dict order.
_HASH_SEEDS: tuple[str, ...] = ("0", "12345")


def reference_seed(run_seed: int, *parts: str | int) -> int:
    """Recompute the documented derivation independently of the module under test. The expectation
    is the definition written out again, never a value copied from a run: a formula change has
    to be a deliberate change in two places, which is the whole point of pinning it.
    """
    joined = "|".join(str(part) for part in (run_seed, *parts))
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")


def _unpack(case: tuple[object, ...]) -> tuple[int, tuple[str | int, ...]]:
    """Split one table entry into the run seed and the ordered parts."""
    run_seed, *parts = case
    assert isinstance(run_seed, int), f"table run seed must be an int: {case!r}"
    for part in parts:
        assert isinstance(part, (str, int)), f"table part must be a str or an int: {case!r}"
    return run_seed, tuple(parts)


def _label(value: object) -> str:
    """Render a table entry as a bounded, ASCII, collision-free test id."""
    rendered = repr(value).encode("ascii", "backslashreplace").decode("ascii")
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:8]
    return f"{rendered[:60]}-{digest}"


def _expected_value() -> list[int]:
    """The documented derivation applied to every definition-table case, in table order."""
    values: list[int] = []
    for case in DEFINITION_CASES:
        run_seed, parts = _unpack(case)
        values.append(reference_seed(run_seed, *parts))
    return values


def _subprocess_source() -> str:
    """Build the probe payload: the definition table in, derived seeds out. The table is embedded
    with ``!a`` -- an ASCII-only ``repr`` -- so the payload survives an ``argv`` entry whatever
    characters a part holds.
    """
    return (
        "import json\n"
        "\n"
        "from qrcheck.config.seeding import derive_seed\n"
        f"\nCASES = {tuple(DEFINITION_CASES)!a}\n"
        "\nprint(json.dumps([derive_seed(case[0], *case[1:]) for case in CASES]))\n"
    )


def _derive_in_subprocess(hash_seed: str) -> list[int]:
    """Derive every definition-table case in a fresh interpreter under one hash seed."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    # qrcheck is not installed into the interpreter's site-packages here, so the
    # checkout root must be reachable both as the working directory and on the path;
    # neither depends on who launched pytest or from where.
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(_REPOSITORY_ROOT), environment.get("PYTHONPATH", "")) if entry
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _subprocess_source()],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_REPOSITORY_ROOT),
            env=environment,
            timeout=120,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"deriving under PYTHONHASHSEED={hash_seed!r} exited {error.returncode}:\n"
            f"{error.stderr or error.stdout}"
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"deriving under PYTHONHASHSEED={hash_seed!r} did not finish within 120 s")
    return json.loads(completed.stdout)


@pytest.mark.parametrize("case", DEFINITION_CASES, ids=_label)
def test_derivation_matches_the_documented_definition(case: tuple[object, ...]) -> None:
    """``derive_seed`` returns the value the documented formula produces."""
    run_seed, parts = _unpack(case)
    assert derive_seed(run_seed, *parts) == reference_seed(run_seed, *parts)


def test_repeated_calls_return_identical_values() -> None:
    """A seed derivation is pure: the same call derives the same seed every time."""
    for case in DEFINITION_CASES:
        run_seed, parts = _unpack(case)
        repeats = {derive_seed(run_seed, *parts) for _ in range(4)}
        assert len(repeats) == 1, f"{case!r} derived {len(repeats)} distinct seeds"


@pytest.mark.parametrize("hash_seed", sorted(_HASH_SEEDS))
def test_values_do_not_depend_on_pythonhashseed(hash_seed: str) -> None:
    """A separate interpreter under a different hash seed derives the same seeds."""
    assert _derive_in_subprocess(hash_seed) == _expected_value()


@pytest.mark.parametrize("pair", PERTURBATIONS, ids=_label)
def test_one_perturbation_changes_the_derived_seed(
    pair: tuple[tuple[object, ...], tuple[object, ...]],
) -> None:
    """Changing the run seed, any one part, or the part order changes the seed."""
    base, perturbed = pair
    base_seed, base_parts = _unpack(base)
    other_seed, other_parts = _unpack(perturbed)
    derived = derive_seed(base_seed, *base_parts)
    assert derived == reference_seed(base_seed, *base_parts)
    assert derived != derive_seed(other_seed, *other_parts), f"{base!r} and {perturbed!r} collided"


def test_appending_an_empty_part_is_a_distinct_namespace() -> None:
    """An empty trailing part is a part: the join keeps its separator. ``(1,)`` joins to ``"1"``
    while ``(1, "")`` joins to ``"1|"``, so the two namespaces differ and so do their seeds.
    """
    assert derive_seed(1) == reference_seed(1)
    assert derive_seed(1, "") == reference_seed(1, "")
    assert derive_seed(1) != derive_seed(1, "")
    assert derive_seed(20240101, "url", 3) != derive_seed(20240101, "url", 3, "")


def test_the_documented_join_collision_is_preserved() -> None:
    """Parts are joined unescaped, so an embedded separator collides with a boundary. The product
    promises nothing stronger, and escaping would change every derived seed and every committed
    ``damage-<kind>-<seed>.png`` name, so the behaviour is pinned as documented instead of
    being asserted away.
    """
    assert derive_seed(1, "a", "b") == derive_seed(1, "a|b")
    assert derive_seed(1, "a|b") == reference_seed(1, "a|b")


@pytest.mark.parametrize("case", INT_PART_CASES, ids=_label)
def test_an_integer_part_and_its_decimal_string_derive_the_same_seed(
    case: tuple[int, str, int],
) -> None:
    """``str()`` inside the derivation makes an int part and its text equivalent.
    ``qrcheck/corpus/payloads.py`` passes a corpus record index as an ``int`` while the other
    call sites pass tokens, so the two shapes must not diverge.
    """
    run_seed, word, index = case
    assert derive_seed(run_seed, word, index) == derive_seed(run_seed, word, str(index))
    assert derive_seed(run_seed, word, index) == reference_seed(run_seed, word, str(index))


def test_a_derived_seed_is_usable_as_a_part() -> None:
    """A seed derived for one draw behaves exactly like its decimal string as a part."""
    derived = derive_seed(20240101, "logo")
    assert derive_seed(20240101, "RN-LOGO", derived) == derive_seed(
        20240101, "RN-LOGO", str(derived)
    )


def test_every_derived_seed_is_a_legal_pcg64_seed() -> None:
    """Every value lies in ``0 <= seed < 2**64``, the range ``PCG64`` accepts.
    ``numpy.random.PCG64`` rejects a negative or oversized seed, and
    ``Generator(PCG64(derive_seed(...)))`` is the only entropy path the product allows, so an
    out-of-range value would break every seeded draw downstream.
    """
    values = sorted(_expected_value())
    assert values, "the definition table is empty"
    for value in values:
        assert 0 <= value < 2**64, value


def test_a_derived_seed_reproduces_a_pcg64_bit_stream() -> None:
    """The same derived seed yields the same draws in a freshly constructed generator."""
    for value in sorted(_expected_value()):
        first = np.random.Generator(np.random.PCG64(value)).random(4)
        second = np.random.Generator(np.random.PCG64(value)).random(4)
        assert first.tolist() == second.tolist(), value


# --------------------------------------------------------------------------- #
# The package-wide prohibition on ambient entropy
# --------------------------------------------------------------------------- #

# Modules whose import lets a non-deterministic value into the product. ``uuid`` and
# ``secrets`` belong here for the same reason as ``random``: the summary's
# ``run_id`` is a content hash and no artefact carries a random token, so neither has
# a sanctioned use anywhere in the package.
_PROHIBITED_IMPORTS: frozenset[str] = frozenset({"random", "secrets", "uuid"})

# Builtin calls that return a value differing between processes: ``hash()`` is
# salted per interpreter run and ``id()`` is an address inside one.
_PROHIBITED_BUILTIN_CALLS: frozenset[str] = frozenset({"hash", "id"})

# Wall-clock readers. A wall clock has no sanctioned use in the product: the
# summary's ``started_at``/``finished_at`` are built from ``datetime`` in UTC, and
# every other timing figure is a duration measured with a monotonic clock.
_WALL_CLOCK_ATTRS: frozenset[str] = frozenset({"time", "time_ns"})

# Monotonic readers are deliberately NOT prohibited. A monotonic clock can only
# produce a duration, the result schema records ``duration_ms`` for every case, and
# AAP 0.5.4 masks that field in every determinism comparison. The performance checks
# read one for their p95 figures and every other case is timed somewhere outside
# ``qrcheck/checks/performance.py``, so restricting the read to that one module would
# fail a conforming package. What is prohibited outright is a *wall-clock* read.

# OS-provided ambient identity. Each value differs between machines or between
# processes, and none may reach an artefact (AAP 0.5.5): no hostname, user name or
# process ID in a report, an evidence file or a filename.
_AMBIENT_IDENTITY_ATTRS: dict[str, frozenset[str]] = {
    "getpass": frozenset({"getuser"}),
    "os": frozenset({"getlogin", "getpid", "getrandom", "urandom"}),
    "platform": frozenset({"node"}),
    "socket": frozenset({"gethostname"}),
    "uuid": frozenset({"uuid1", "uuid4"}),
}

# Modules whose aliases are resolved, so ``import time as clock`` is caught.
_INSPECTED_MODULES: frozenset[str] = frozenset(
    {"getpass", "numpy", "os", "platform", "socket", "time", "uuid"}
)

# numpy's *global* random API, which seeds itself from OS entropy at first use. The
# sanctioned path is ``Generator(PCG64(derive_seed(...)))``; ``Generator``, ``PCG64``
# and the bit-generator classes are attributes of ``numpy.random`` but are absent
# from this set, as are the instance methods a seeded generator exposes.
_NUMPY_GLOBAL_RNG_ATTRS: frozenset[str] = frozenset(
    {
        "bytes",
        "choice",
        "get_state",
        "permutation",
        "rand",
        "randint",
        "randn",
        "random",
        "random_integers",
        "random_sample",
        "ranf",
        "sample",
        "seed",
        "set_state",
        "shuffle",
        "standard_normal",
    }
)

# Constructors that seed themselves from OS entropy when called without arguments.
_ENTROPY_DEFAULT_CALLS: frozenset[str] = frozenset(
    {"PCG64", "RandomState", "SeedSequence", "default_rng"}
)

# A source carrying every prohibited pattern once, with an aliased clock, an aliased
# global numpy RNG and an ordinary ``import os`` that must stay silent, so the
# self-test proves both that the detectors fire and that they are not blanket bans.
_OFFENDING_SOURCE = """\
import os
import random
import time as clock
import uuid

from secrets import token_hex
from numpy import random as global_rng
from numpy.random import default_rng
from time import time as wall_clock


def probe(value):
    return (
        token_hex(4),
        hash(value),
        id(value),
        os.urandom(1),
        clock.time(),
        clock.time_ns(),
        wall_clock(),
        global_rng.seed(1),
        global_rng.random(),
        default_rng(),
        uuid.uuid4(),
        os.getpid(),
        getpass.getuser(),
        socket.gethostname(),
        platform.node(),
        numpy.random.default_rng(),
    )
"""

_OFFENDING_EXPECTED: frozenset[str] = frozenset(
    {
        "from secrets import token_hex",
        "getpass.getuser",
        "hash()",
        "id()",
        "import random",
        "import uuid",
        "numpy.random.default_rng() without a seed",
        "numpy.random.random",
        "numpy.random.seed",
        "os.getpid",
        "os.urandom",
        "platform.node",
        "socket.gethostname",
        "time.time",
        "time.time()",
        "time.time_ns",
        "uuid.uuid4",
    }
)

# The primitives the product does allow, so a detector that fires on everything would
# fail here instead of passing the package scan by accident: ``hashlib``, a UTC
# ``datetime`` timestamp, an explicitly seeded ``PCG64`` generator, a seeded
# ``default_rng``, generator instance methods and a monotonic clock read.
_SANCTIONED_SOURCE = """\
import datetime
import hashlib
import json
import time

import numpy as np

from numpy.random import default_rng as seeded_rng

GENERATOR = np.random.Generator(np.random.PCG64(1234))


def derive(run_seed, part):
    generator = np.random.Generator(np.random.PCG64(run_seed))
    helper = np.random.default_rng(run_seed)
    another = seeded_rng(run_seed)
    elapsed = time.perf_counter()
    stamp = datetime.datetime.now(datetime.timezone.utc)
    payload = json.dumps({"seed": run_seed, "part": part}).encode("utf-8")
    return (
        generator.random(2),
        generator.choice([1, 2]),
        helper.integers(0, 3),
        another.integers(0, 3),
        elapsed,
        stamp,
        hashlib.sha256(payload).hexdigest(),
    )
"""


def _root_and_chain(node: ast.AST) -> tuple[str | None, tuple[str, ...]]:
    """Split a dotted expression into its root name and its attribute chain."""
    chain: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        chain.append(current.attr)
        current = current.value
    root = current.id if isinstance(current, ast.Name) else None
    return root, tuple(reversed(chain))


def _alias_maps(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """Map bound names to the inspected module, or to a direct import's canonical name. Aliasing
    is honoured because ``import time as clock`` followed by ``clock.time()`` is the same
    wall-clock read as ``time.time()``, and a scan that knew only the canonical module names
    would miss it.
    """
    modules: dict[str, str] = {module: module for module in sorted(_INSPECTED_MODULES)}
    attributes: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _INSPECTED_MODULES:
                    modules[alias.asname or root] = root
                if alias.name == "numpy.random" and alias.asname:
                    modules[alias.asname] = "numpy.random"
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue  # a relative import names a local module, never a stdlib one
            root = node.module.split(".")[0]
            for alias in node.names:
                bound = alias.asname or alias.name
                if root in _INSPECTED_MODULES:
                    modules[bound] = root
                if node.module == "numpy" and alias.name == "random":
                    modules[bound] = "numpy.random"
                if node.module == "numpy.random" and (
                    alias.name in _NUMPY_GLOBAL_RNG_ATTRS
                    or alias.name in _ENTROPY_DEFAULT_CALLS
                ):
                    attributes[bound] = f"numpy.random.{alias.name}"
                if node.module == "time" and alias.name in _WALL_CLOCK_ATTRS:
                    attributes[bound] = f"time.{alias.name}"
                if alias.name in _AMBIENT_IDENTITY_ATTRS.get(root, frozenset()):
                    attributes[bound] = f"{root}.{alias.name}"
    return modules, attributes


def _attribute_offenders(node: ast.Attribute, modules: dict[str, str]) -> list[str]:
    """The prohibited reference one dotted expression makes, if any."""
    root, chain = _root_and_chain(node)
    module = modules.get(root) if root is not None else None
    if module is None or not chain:
        return []
    if module == "numpy" and len(chain) == 2 and chain[0] == "random":
        if chain[1] in _NUMPY_GLOBAL_RNG_ATTRS:
            return [f"numpy.random.{chain[1]}"]
        return []
    if len(chain) != 1:
        return []
    owner = chain[0]
    if module == "numpy.random" and owner in _NUMPY_GLOBAL_RNG_ATTRS:
        return [f"numpy.random.{owner}"]
    if module == "time" and owner in _WALL_CLOCK_ATTRS:
        return [f"time.{owner}"]
    if owner in _AMBIENT_IDENTITY_ATTRS.get(module, frozenset()):
        return [f"{module}.{owner}"]
    return []


def _call_offenders(node: ast.Call, attributes: dict[str, str]) -> list[str]:
    """The prohibited call one call expression makes, if any."""
    function = node.func
    if isinstance(function, ast.Name):
        if function.id in _PROHIBITED_BUILTIN_CALLS:
            return [f"{function.id}()"]
        canonical = attributes.get(function.id)
        if canonical is None:
            return []
        leaf = canonical.rsplit(".", 1)[-1]
        if leaf in _ENTROPY_DEFAULT_CALLS:
            if node.args or node.keywords:
                return []  # explicitly seeded, which the sanctioned path permits
            return [f"{canonical}() without a seed"]
        return [f"{canonical}()"]
    if isinstance(function, ast.Attribute) and not node.args and not node.keywords:
        _root, chain = _root_and_chain(function)
        if chain and chain[-1] in _ENTROPY_DEFAULT_CALLS:
            return [f"numpy.random.{chain[-1]}() without a seed"]
    return []


def _node_offenders(
    node: ast.AST, modules: dict[str, str], attributes: dict[str, str]
) -> list[str]:
    """The prohibited names one AST node introduces, if any."""
    if isinstance(node, ast.Import):
        return [
            f"import {alias.name}"
            for alias in node.names
            if alias.name.split(".")[0] in _PROHIBITED_IMPORTS
        ]
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module:
            return []
        if node.module.split(".")[0] in _PROHIBITED_IMPORTS:
            imported = ", ".join(sorted(alias.name for alias in node.names))
            return [f"from {node.module} import {imported}"]
        return []
    if isinstance(node, ast.Attribute):
        return _attribute_offenders(node, modules)
    if isinstance(node, ast.Call):
        return _call_offenders(node, attributes)
    return []


def offenders_in_source(source: str, filename: str) -> list[str]:
    """Every ambient-entropy reference in one module's source, as ``file:line:name``."""
    tree = ast.parse(source, filename=filename)
    modules, attributes = _alias_maps(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        for name in _node_offenders(node, modules, attributes):
            found.add(f"{filename}:{node.lineno}:{name}")
    return sorted(found)


def _sha256_call_sites(source: str, filename: str) -> int:
    """Count ``<something>.sha256(...)`` calls, the primitive the derivation uses."""
    tree = ast.parse(source, filename=filename)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sha256"
    )


def _relative(path: Path) -> str:
    """The repository-relative POSIX form an offender entry starts with."""
    return path.relative_to(_REPOSITORY_ROOT).as_posix()


def _scan_package() -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Scan every ``.py`` file under ``qrcheck/`` in sorted path order."""
    files = tuple(sorted(path for path in _PACKAGE_DIR.rglob("*.py") if path.is_file()))
    offenders: set[str] = set()
    for path in files:
        source = path.read_text(encoding="utf-8")
        offenders.update(offenders_in_source(source, _relative(path)))
    return files, tuple(sorted(offenders))


def test_the_analyser_reports_every_prohibited_pattern() -> None:
    """The scan's own detectors are exercised on a source carrying every pattern."""
    entries = offenders_in_source(_OFFENDING_SOURCE, "sample.py")
    for entry in entries:
        filename, line, _name = entry.split(":", 2)
        assert filename == "sample.py", entry
        assert line.isdigit(), entry
    assert {entry.split(":", 2)[2] for entry in entries} == _OFFENDING_EXPECTED


def test_the_analyser_accepts_the_sanctioned_patterns() -> None:
    """The sanctioned primitives are not reported, so the scan cannot pass by banning all."""
    assert offenders_in_source(_SANCTIONED_SOURCE, "sample.py") == []


def test_no_module_under_qrcheck_draws_ambient_entropy() -> None:
    """Nothing in the package may derive a value from the process, the clock or a global RNG."""
    _files, offenders = _scan_package()
    assert not offenders, (
        f"{len(offenders)} ambient-entropy reference(s) in qrcheck/; each value would differ "
        "between runs:\n" + "\n".join(offenders)
    )


def test_the_scan_is_not_vacuous() -> None:
    """A moved, emptied or renamed package cannot make the scan pass silently."""
    assert _PACKAGE_DIR.name == "qrcheck", _PACKAGE_DIR
    assert _SEEDING_PATH == _PACKAGE_DIR / "config" / "seeding.py", _SEEDING_PATH
    files, _offenders = _scan_package()
    assert files, f"no Python file found under {_PACKAGE_DIR}"
    assert _SEEDING_PATH in files, f"{len(files)} files scanned, none of them {_SEEDING_PATH}"
    source = _SEEDING_PATH.read_text(encoding="utf-8")
    assert _sha256_call_sites(source, _relative(_SEEDING_PATH)) >= 1, (
        f"{_relative(_SEEDING_PATH)} contains no hashlib.sha256 call, so the derivation the "
        "scan is guarding is not where it is claimed to be"
    )
