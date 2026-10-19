"""Canonical R: explicit semi-algebraic partition (§4.1).

Windows on bond lengths give candidate order sets, a hybridization test on
angles prunes them, a greedy cascade in certainty order picks one order per
contact, and the result is projected onto 𝒢 (Hamming-nearest valid
structure) when it is not already valid. Aromatic windows (k = 1.5) survive to
the projection, which resolves them into a Kekulé form.
"""

from __future__ import annotations

import numpy as np

from ..core.canon import ranks_from_colors, wl_colors
from ..core.elements import states
from ..core.geometry import CONTACT_SCALE, angle, contact_map
from ..core.graph import Graph
from ..core.validity import is_valid
from ..prediction import Prediction, Status
from ..query import Query
from ..tables import Window, bond_windows
from .base import Capabilities, Method, register
from .wrappers import project_to_valid

CLASSES = (1.0, 1.5, 2.0, 3.0)
_SP3_PRUNED = frozenset({5, 6, 7, 8})  # second-row atoms; hypervalent S/P are tetrahedral with π bonds


def _window_distance(d: float, w: Window) -> float:
    return max(w.lo - d, 0.0, d - w.hi)


def max_valence(z: int, table: str, c_max: int) -> int:
    vals = [s.valence for s in states(z, table, c_max)]
    return max(vals) if vals else 0


@register
class CanonicalR(Method):
    name = "canonical-R"
    family = "rule"
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=True)
    defaults = {
        "table": "octet",
        "c_max": 1,
        "contact_scale": CONTACT_SCALE,
        "sp_angle": 155.0,
        "sp2_angle": 115.0,
        "max_pool": 64,
        "time_limit": 30.0,
    }

    def hybridization(self, query: Query, i: int, nbrs: list[int]) -> str | None:
        if len(nbrs) < 2:
            return None
        x = query.positions
        angles = [angle(x, a, i, b) for k, a in enumerate(nbrs) for b in nbrs[k + 1 :]]
        mean = float(np.mean(angles))
        if mean >= self.params["sp_angle"]:
            return "sp"
        if mean >= self.params["sp2_angle"]:
            return "sp2"
        return "sp3"

    @staticmethod
    def carbon_hydrogens(z: np.ndarray, hyb: list[str | None], nbrs: list[list[int]]) -> np.ndarray:
        """H-free regime: implied H count of carbons from hybridization (NaN = unknown)."""
        steric = {"sp3": 4, "sp2": 3, "sp": 2}
        out = np.full(len(z), np.nan)
        for i, h in enumerate(hyb):
            if int(z[i]) == 6 and h is not None:
                out[i] = max(steric[h] - len(nbrs[i]), 0)
        return out

    def predict(self, query: Query) -> Prediction:
        p = self.params
        z = query.atomic_numbers
        x = query.positions
        edges, _ = contact_map(z, x, p["contact_scale"])
        windows = bond_windows()
        n, m = query.n_atoms, len(edges)
        nbrs: list[list[int]] = [[] for _ in range(n)]
        for i, j in edges.tolist():
            nbrs[i].append(j)
            nbrs[j].append(i)

        # 1. window tests → candidate sets and margins
        cands: list[list[float]] = []
        margins = np.zeros(m)
        for e, (i, j) in enumerate(edges.tolist()):
            a, b = sorted((int(z[i]), int(z[j])))
            d = float(np.linalg.norm(x[i] - x[j]))
            ws = {k: windows[(a, b, k)] for k in CLASSES if (a, b, k) in windows}
            if a == 1 or not ws:
                cands.append([1.0])
                margins[e] = np.inf if a == 1 else 0.0
                continue
            inside = [k for k, w in ws.items() if w.lo <= d <= w.hi]
            if not inside:
                inside = [min(ws, key=lambda k: _window_distance(d, ws[k]))]
            cands.append(inside)
            margins[e] = min(min(abs(d - w.lo), abs(d - w.hi)) for w in ws.values())

        # 2. hybridization pruning
        hyb = [self.hybridization(query, i, nbrs[i]) for i in range(n)]
        for e, (i, j) in enumerate(edges.tolist()):
            keep = set(cands[e])
            for a in (i, j):
                if hyb[a] == "sp3" and int(z[a]) in _SP3_PRUNED:
                    keep &= {1.0}
                elif hyb[a] == "sp2":
                    keep -= {3.0}
            if keep:
                cands[e] = sorted(keep)

        # 3. greedy cascade, most certain first, equivariant order
        rank = ranks_from_colors(wl_colors(z, x, edges))
        order = sorted(
            range(m),
            key=lambda e: (len(cands[e]), -margins[e], *sorted((rank[edges[e, 0]], rank[edges[e, 1]]))),
        )
        residual = np.array([max_valence(int(a), p["table"], p["c_max"]) for a in z], dtype=np.float64)
        guess = np.ones(m)
        residual -= np.bincount(edges.reshape(-1), minlength=n)  # every contact is at least single
        for e in order:
            i, j = edges[e]
            fits = [k for k in cands[e] if k - 1 <= min(residual[i], residual[j]) + 1e-9]
            k = max(fits) if fits else min(cands[e])
            guess[e] = k
            residual[[i, j]] -= k - 1

        # 4. projection onto 𝒢 when needed
        flags: list[str] = []
        integral = np.all(guess == np.round(guess))
        if integral and query.hydrogens_observed:
            g = Graph(n, edges, guess.astype(np.int64), np.zeros(n), np.zeros(n), np.zeros(n))
            if is_valid(z, query.total_charge, g, table=p["table"], c_max=p["c_max"], r_max=0):
                return Prediction.ok(g, margin=margins)
        target_h = None if query.hydrogens_observed else self.carbon_hydrogens(z, hyb, nbrs)
        best, pool, reason = project_to_valid(
            query, edges, guess, target_h, table=p["table"], c_max=p["c_max"],
            max_pool=p["max_pool"], time_limit=p["time_limit"],
        )
        if best is None:
            return Prediction.fail(Status.INFEASIBLE, reason)
        flags.append("projected")
        if len(pool) > 1:
            flags.append("projection_tied")
        bond_margin = dict(zip(map(tuple, edges.tolist()), margins))
        return Prediction.ok(
            best,
            alternatives=pool if len(pool) > 1 else [],
            margin=np.array([bond_margin.get(tuple(b), 0.0) for b in best.bonds.tolist()]),
            flags=flags,
            extra={"n_aromatic_guess": int((guess == 1.5).sum())},
        )
