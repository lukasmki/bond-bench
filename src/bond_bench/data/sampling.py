"""Deterministic, seeded sampling helpers."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def derive_seed(*parts: object) -> int:
    """Stable 64-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def sample(
    items: Sequence[T],
    n: int,
    seed: int,
    key: Callable[[T], str],
) -> list[T]:
    """Draw up to ``n`` items without replacement; order-independent of input."""
    ordered = sorted(items, key=key)
    if n >= len(ordered):
        return ordered
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(ordered), size=n, replace=False)
    return [ordered[i] for i in sorted(picked)]


def stratified_sample(
    items: Sequence[T],
    n: int,
    seed: int,
    stratum: Callable[[T], Hashable],
    key: Callable[[T], str],
) -> list[T]:
    """Sample ``n`` items spread as evenly as possible across strata.

    Strata with too few items contribute all of them and the remainder is
    redistributed to the others.
    """
    groups: dict[Hashable, list[T]] = defaultdict(list)
    for item in items:
        groups[stratum(item)].append(item)
    names = sorted(groups, key=str)
    quota = {g: 0 for g in names}
    remaining = min(n, len(items))
    open_groups = list(names)
    while remaining > 0 and open_groups:
        share, extra = divmod(remaining, len(open_groups))
        for i, g in enumerate(open_groups):
            take = min(share + (i < extra), len(groups[g]) - quota[g])
            quota[g] += take
            remaining -= take
        open_groups = [g for g in open_groups if quota[g] < len(groups[g])]
    out: list[T] = []
    for g in names:
        out.extend(sample(groups[g], quota[g], derive_seed(seed, "strat", g), key))
    return sorted(out, key=key)


def tertile_bounds(values: Sequence[float]) -> tuple[float, float]:
    """Cut points splitting ``values`` into three equally populated bins."""
    ordered = sorted(values)
    return ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]


def tertile(value: float, bounds: tuple[float, float]) -> int:
    return 0 if value < bounds[0] else 1 if value < bounds[1] else 2
