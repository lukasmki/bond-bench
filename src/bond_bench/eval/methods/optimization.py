"""Canonical O: coarse-grain the geometry to A(X), then select inside the fibre (§4.2).

Geometry enters only through the contact map. The prior is a lexicographic
list of linear objectives solved exactly by P4; all co-optimal structures are
returned and the point estimate is chosen equivariantly (P5).
"""

from __future__ import annotations

import numpy as np

from ..core.canon import pick, tiebreak_objectives, wl_colors
from ..core.geometry import CONTACT_SCALE, contact_map
from ..core.ilp import DecodeSpec, Objective, decode
from ..prediction import Prediction, Status
from ..query import Query
from ..tables import MISSING_PENALTY, atom_penalties
from .base import Capabilities, Method, register


def aps_contexts(z: np.ndarray, edges: np.ndarray) -> list[set[str]]:
    """Context qualifiers of the Wang et al. (2006) table that apply to each atom.

    Connectivities are counted on the contact graph, so they include H only
    when H are observed (H-free queries use heavy-atom connectivity here).
    """
    n = len(z)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, j in np.asarray(edges).reshape(-1, 2).tolist():
        nbrs[i].append(j)
        nbrs[j].append(i)
    deg = [len(v) for v in nbrs]
    out: list[set[str]] = [set() for _ in range(n)]
    for i in range(n):
        zi, nb = int(z[i]), nbrs[i]
        terminal_os = sum(int(z[j]) in (8, 16) and deg[j] == 1 for j in nb)
        terminal_o = sum(int(z[j]) == 8 and deg[j] == 1 for j in nb)
        if zi == 6:
            if deg[i] == 1 and int(z[nb[0]]) == 7 and deg[nb[0]] == 2:
                out[i].add("in C≡N–R")
            if terminal_o == 2:
                out[i].add("in COO−")
        elif zi == 7:
            if any(int(z[j]) == 7 and {deg[i], deg[j]} == {1, 2} for j in nb):
                out[i].add("in N=N=R")
            if deg[i] == 3 and terminal_o == 2:
                out[i].add("in nitro")
            elif deg[i] == 3 and terminal_o == 1:
                out[i].add("in pyridine-1-oxide, etc.")
        elif zi == 8 and deg[i] == 1 and int(z[nb[0]]) == 7 and deg[nb[0]] == 3:
            out[i].add("in pyridine-1-oxide, etc.")
        elif zi == 16 and deg[i] == 1 and int(z[nb[0]]) == 7 and deg[nb[0]] == 3:
            out[i].add("in pyridine-1-thiol anion, etc.")
        if zi in (15, 16) and deg[i] == 4:
            word = {2: "two", 3: "three", 4: "four"}.get(terminal_os)
            if word:
                out[i].add(f"bonded to {word} O(X1) or S(X1)")
    return out


def prior_objectives(preset: str, edges: np.ndarray, z: np.ndarray) -> list[Objective]:
    edges = np.asarray(edges).reshape(-1, 2)
    if preset == "kim2015":
        return [
            Objective(edge=np.tile(np.arange(4.0), (len(edges), 1)), name="max ΣB"),
            Objective(atom=lambda i, st, eta: -abs(st.charge), name="min Σ|q|"),
        ]
    if preset == "atom_penalty":
        table = atom_penalties()
        degree = np.bincount(edges.reshape(-1), minlength=len(z))
        contexts = aps_contexts(z, edges)

        def ap(i: int, st, eta: int) -> float:
            row = table.row(int(z[i]), int(degree[i] + eta), contexts[i])
            return -(row.penalty(st.valence) if row else MISSING_PENALTY)

        return [
            Objective(atom=ap, name="min ΣAP"),
            Objective(atom=lambda i, st, eta: -abs(st.charge), name="min Σ|q|"),
        ]
    raise ValueError(f"unknown preset {preset!r}")


@register
class CanonicalO(Method):
    name = "canonical-O"
    family = "optimization"
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=True)
    defaults = {
        "preset": "kim2015",
        "table": "octet",
        "c_max": 1,
        "contact_scale": CONTACT_SCALE,
        "max_pool": 64,
        "time_limit": 30.0,
    }

    def unsupported_reason(self, query: Query) -> str | None:
        reason = super().unsupported_reason(query)
        if reason is None and self.params["preset"] == "atom_penalty":
            missing = set(np.unique(query.atomic_numbers).tolist()) - atom_penalties().elements
            if missing:
                return f"no atomic penalty scores for elements {sorted(missing)}"
        return reason

    def predict(self, query: Query) -> Prediction:
        p = self.params
        z = query.atomic_numbers
        edges, margin = contact_map(z, query.positions, p["contact_scale"])
        allowed = np.ones((len(edges), 4), dtype=bool)
        allowed[:, 0] = False  # fibre constraint: supp B = A(X)
        spec = DecodeSpec(
            z,
            query.total_charge,
            edges,
            allowed=allowed,
            hydrogens=np.zeros(query.n_atoms, dtype=np.int64) if query.hydrogens_observed else None,
            radicals=query.unpaired_electrons,
            table=p["table"],
            c_max=p["c_max"],
            time_limit=p["time_limit"],
        )
        prior = prior_objectives(p["preset"], edges, z)
        res = decode(spec, prior, pool="all", max_pool=p["max_pool"])
        if res.status != "ok":
            return Prediction.fail(Status(res.status), res.reason)
        pool = [s.graph for s in res.solutions]
        colors = wl_colors(z, query.positions, edges)
        if res.truncated:  # pick inside the full co-optimal set, not the truncated pool
            tied = decode(spec, [*prior, *tiebreak_objectives(edges, query.positions, colors)], pool="all", max_pool=p["max_pool"])
            best = pick([s.graph for s in tied.solutions], colors, query.positions)
        else:
            best = pick(pool, colors, query.positions)
        flags = []
        if len(pool) > 1:
            flags.append("prior_tied")
        if res.truncated:
            flags.append("pool_truncated")
        bond_margin = dict(zip(map(tuple, edges.tolist()), margin))
        return Prediction.ok(
            best,
            alternatives=pool if len(pool) > 1 else [],
            margin=np.array([bond_margin[tuple(b)] for b in best.bonds.tolist()]),
            flags=flags,
            extra={"objective": list(res.solutions[0].values)},
        )
