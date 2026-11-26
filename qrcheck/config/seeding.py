"""The only seed-derivation point in ``qrcheck``.

Every seeded draw the product makes -- corpus payload generation, occluder
placement and size, additive-noise patterns and the logo overlay bitmap -- is
derived here and consumed only through a PCG64-backed ``numpy`` generator
(``Generator(PCG64(seed))``), whose bit stream is stable across numpy releases.
No other module in the package may derive, salt or draw a seed of its own.

The module reads no configuration, performs no I/O, prints nothing and raises
nothing, so a run is reproducible from the ``seed`` value its report records
alone.
"""

import hashlib

__all__ = ["derive_seed"]


def derive_seed(run_seed: int, *parts: str | int) -> int:
    """Derive one seed for a single draw from a run seed and ordered parts.

    The same ``(run_seed, *parts)`` yields the same integer on every machine, in
    every process and every Python run: the parts are joined with a literal
    ``"|"``, encoded as UTF-8 and hashed with SHA-256, so the result is
    independent of ``PYTHONHASHSEED``, locale, timezone and filesystem order.
    The value comes from SHA-256 alone, never from ambient process state.

    Declared callers, and the part convention each of them owns:

    * ``qrcheck/corpus/payloads.py`` -- ``derive_seed(seed, class_name,
      record_index)``.
    * ``qrcheck/checks/damage.py`` -- ``derive_seed(seed, "DM-RESILIENCE",
      case_id, kind)``, where ``kind`` is one of the seven damage kinds of
      ``qrcheck/evidence/naming.py``. The same seed is recorded in
      ``results[].seed`` and in the ``damage-<kind>-<seed>.png`` filename, which
      is what makes a damaged image regenerable from the record alone.
    * ``qrcheck/checks/render.py`` -- ``derive_seed(seed, "RN-LOGO", case_id,
      "logo")``.

    Integer parts are stringified inside this function with ``str()``, so an
    integer and its decimal string derive the same seed
    (``derive_seed(1, 2) == derive_seed(1, "2")``); a call site may therefore
    pass a corpus record index as an ``int``.

    Deliberate limitation, which must not be "fixed": the parts are joined with
    ``"|"`` and nothing is escaped, so ``derive_seed(1, "a", "b")`` and
    ``derive_seed(1, "a|b")`` are the same seed. Callers must pass identifiers
    that cannot contain ``"|"``; check IDs, case IDs (``corpus:<record-id>``,
    ``synthetic:<name>``), class names, damage-kind tokens and ``"logo"`` all
    satisfy that. Adding escaping would change every derived seed in the
    product and, with them, every committed ``damage-<kind>-<seed>.png`` name.

    ``run_seed`` is validated to be ``>= 0`` by ``qrcheck/config/schema.py``
    before it reaches this function, so this function validates nothing: staying
    total and branch-free is what makes it exhaustively testable.

    Args:
        run_seed: The run's seed, a non-negative integer.
        *parts: Ordered namespace components that distinguish this draw from
            every other draw of the same run.

    Returns:
        A 64-bit unsigned integer, ``0 <= seed < 2**64``, which is the range
        ``numpy``'s ``PCG64`` consumes directly.
    """
    joined = "|".join(str(part) for part in (run_seed, *parts))
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")
