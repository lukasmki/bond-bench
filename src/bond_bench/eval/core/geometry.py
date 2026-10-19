"""P1 invariant featurization and P2 contact map."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .elements import covalent_radius

#: Default contact tolerance: τ_ab = scale · (r_a + r_b), as in RDKit DetermineConnectivity.
CONTACT_SCALE = 1.3


def distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Pairwise distances (Å). Squared distances Δ are ``distance_matrix(X) ** 2``."""
    x = np.asarray(positions, dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def contact_threshold(z_a: np.ndarray, z_b: np.ndarray, scale: float = CONTACT_SCALE) -> np.ndarray:
    radii = np.vectorize(covalent_radius, otypes=[float])
    return scale * (radii(z_a) + radii(z_b))


def candidate_pairs(positions: np.ndarray, cutoff: float) -> np.ndarray:
    """All pairs i < j within ``cutoff`` Å, sorted (cell-list search)."""
    x = np.asarray(positions, dtype=np.float64)
    if len(x) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = cKDTree(x).query_pairs(cutoff, output_type="ndarray").astype(np.int64)
    pairs.sort(axis=1)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))] if len(pairs) else pairs.reshape(0, 2)


def contact_map(
    atomic_numbers: np.ndarray, positions: np.ndarray, scale: float = CONTACT_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """P2: contacts E = {(i, j): ‖x_i − x_j‖ ≤ τ_{z_i z_j}} and their margins |d − τ|.

    Returns ``(edges, margin)`` where ``edges`` is (m, 2) sorted with i < j and
    ``margin`` is the distance (Å) of each contact from its threshold.
    """
    z = np.asarray(atomic_numbers)
    x = np.asarray(positions, dtype=np.float64)
    max_r = max((covalent_radius(int(a)) for a in set(z.tolist())), default=0.0)
    pairs = candidate_pairs(x, 2 * scale * max_r)
    if not len(pairs):
        return pairs, np.zeros(0)
    d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    tau = contact_threshold(z[pairs[:, 0]], z[pairs[:, 1]], scale)
    keep = d <= tau
    return pairs[keep], (tau - d)[keep]


def nearest_noncontact_margin(
    atomic_numbers: np.ndarray, positions: np.ndarray, scale: float = CONTACT_SCALE
) -> float:
    """Smallest |d − τ| over non-contacts, i.e. how close A(X) is to gaining an edge."""
    z = np.asarray(atomic_numbers)
    x = np.asarray(positions, dtype=np.float64)
    max_r = max((covalent_radius(int(a)) for a in set(z.tolist())), default=0.0)
    pairs = candidate_pairs(x, 2 * scale * max_r + 1.0)
    if not len(pairs):
        return float("inf")
    d = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    tau = contact_threshold(z[pairs[:, 0]], z[pairs[:, 1]], scale)
    gap = d - tau
    gap = gap[gap > 0]
    return float(gap.min()) if len(gap) else float("inf")


def angle(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle ∠ijk at j in degrees."""
    a = positions[i] - positions[j]
    b = positions[k] - positions[j]
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
