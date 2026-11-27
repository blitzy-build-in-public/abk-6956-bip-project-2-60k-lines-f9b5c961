"""The only seed-derivation point in ``qrcheck``.

Every seeded draw the product makes -- corpus payload generation, occluder placement and size,
additive-noise patterns and the logo overlay bitmap -- is derived here and consumed only through a
PCG64-backed ``numpy`` generator (``Generator(PCG64(seed))``), whose bit stream is stable across
numpy releases. No other module in the package may derive, salt or draw a seed of its own. The
module reads no configuration, performs no I/O, prints nothing and raises nothing, so a run is
reproducible from the ``seed`` value its report records alone.
"""

import hashlib

__all__ = ["derive_seed"]


def derive_seed(run_seed: int, *parts: str | int) -> int:
    """Derive one seed for a single draw from a run seed and ordered parts.

    The same ``(run_seed, *parts)`` yields the same integer on every machine, in every process and
    every Python run: the parts are joined with a literal ``"|"``, encoded as UTF-8 and hashed with
    SHA-256, so the result is independent of ``PYTHONHASHSEED``, locale, timezone and filesystem
    order. Declared callers: ``corpus/payloads.py`` with ``(seed, class_name, record_index)``;
    ``checks/damage.py`` with ``(seed, "DM-RESILIENCE", case_id, kind)``, recording that seed in
    ``results[].seed`` and in the ``damage-<kind>-<seed>.png`` name, which makes a damaged image
    regenerable from the record alone; and ``checks/render.py`` with ``(seed, "RN-LOGO", case_id,
    "logo")``.

    Deliberate limitation, which must not be "fixed": the parts are joined with ``"|"`` and nothing
    is escaped, so callers pass identifiers that cannot contain ``"|"``; adding escaping would
    rename every committed ``damage-<kind>-<seed>.png``.
    """
    joined = "|".join(str(part) for part in (run_seed, *parts))
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")
