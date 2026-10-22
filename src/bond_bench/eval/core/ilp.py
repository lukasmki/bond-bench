"""P4: the constrained linear decoder, solved exactly with OR-Tools CP-SAT.

Variables are one-hot: ``y[e, k]`` picks the order k ∈ {0..3} of candidate
pair e, and ``s[i, t]`` picks atom i's (state, η) pair, where η is the implicit
H count (fixed when hydrogens are observed). Constraints enforce
D_i = (B·1)_i + η_i, Σ q = Q and Σ r = r_total, so every feasible point lies
in 𝒢(z, Q) with supp B ⊆ E.

Objectives are linear in the one-hot variables. CP-SAT needs integer
coefficients, so float scores are multiplied by :data:`SCALE` and rounded
(integer scores are used as is).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .elements import AtomState, max_hydrogens, states
from .graph import Graph

SCALE = 10_000
NEG_INF = -np.inf

#: atom score u_i(state, η)
AtomScore = Callable[[int, AtomState, int], float]


@dataclass
class Objective:
    """A linear objective (maximized): per-pair order scores plus per-atom scores."""

    edge: np.ndarray | None = None  # (m, 4), score of order k on pair e
    atom: AtomScore | None = None
    name: str = ""


@dataclass
class DecodeSpec:
    z: np.ndarray
    total_charge: int
    edges: np.ndarray  # (m, 2) candidate pairs E
    allowed: np.ndarray | None = None  # (m, 4) bool; default: orders 0..3
    hydrogens: np.ndarray | None = None  # fixed h per atom; None = free η (H-free regime)
    radicals: int = 0  # Σ r (unpaired electrons)
    table: str = "octet"
    c_max: int = 1
    time_limit: float = 30.0

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=np.int64)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        if self.allowed is None:
            self.allowed = np.ones((len(self.edges), 4), dtype=bool)
        self.allowed = np.asarray(self.allowed, dtype=bool).reshape(len(self.edges), 4)

    @property
    def n(self) -> int:
        return len(self.z)


@dataclass
class Solution:
    graph: Graph
    values: tuple[float, ...]  # objective values, one per lexicographic stage


@dataclass
class DecodeResult:
    status: str  # "ok" | "infeasible" | "timeout"
    solutions: list[Solution] = field(default_factory=list)
    reason: str = ""
    truncated: bool = False  # solution pool hit its cap or the enumeration hit the time limit

    @property
    def best(self) -> Solution | None:
        return self.solutions[0] if self.solutions else None


def _lin(variables: list, coefs: list) -> cp_model.LinearExpr:
    return cp_model.LinearExpr.weighted_sum(variables, [int(c) for c in coefs])


def _is_integral(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(np.all(finite == np.round(finite)))


class _Model:
    def __init__(self, spec: DecodeSpec, relax_charge: bool = False):
        self.spec = spec
        m = cp_model.CpModel()
        self.model = m
        n, edges = spec.n, spec.edges
        deg_min = np.zeros(n, dtype=np.int64)
        deg_max = np.zeros(n, dtype=np.int64)
        self.y: list[dict[int, cp_model.IntVar]] = []
        for e, (i, j) in enumerate(edges):
            ks = [k for k in range(4) if spec.allowed[e, k]]
            row = {k: m.new_bool_var(f"y{e}_{k}") for k in ks}
            m.add_exactly_one(row.values())
            self.y.append(row)
            lo, hi = min(ks), max(ks)
            deg_min[[i, j]] += lo
            deg_max[[i, j]] += hi

        self.cand: list[list[tuple[AtomState, int]]] = []
        self.s: list[list[cp_model.IntVar]] = []
        for i in range(n):
            zi = int(spec.z[i])
            r_max = spec.radicals
            if spec.hydrogens is None:
                etas = range(max_hydrogens(zi, spec.table, spec.c_max) + 1)
            else:
                etas = [int(spec.hydrogens[i])]
            cands = [
                (st, eta)
                for st in states(zi, spec.table, spec.c_max, r_max)
                for eta in etas
                if deg_min[i] + eta <= st.valence <= deg_max[i] + eta
            ]
            self.cand.append(cands)
            row = [m.new_bool_var(f"s{i}_{t}") for t in range(len(cands))]
            if row:
                m.add_exactly_one(row)
            else:
                m.add_bool_or([])  # infeasible: no admissible state
            self.s.append(row)

        incident: list[list[tuple[int, int, cp_model.IntVar]]] = [[] for _ in range(n)]
        for e, (i, j) in enumerate(edges):
            for k, var in self.y[e].items():
                if k:
                    incident[i].append((k, e, var))
                    incident[j].append((k, e, var))
        for i in range(n):
            bond_vars = [var for _, _, var in incident[i]]
            bond_coefs = [k for k, _, _ in incident[i]]
            need = [-(st.valence - eta) for st, eta in self.cand[i]]
            m.add(_lin(bond_vars + self.s[i], bond_coefs + need) == 0)
        all_s = [v for row in self.s for v in row]
        all_states = [st for cands in self.cand for st, _ in cands]
        if not relax_charge:
            m.add(_lin(all_s, [st.charge for st in all_states]) == spec.total_charge)
        m.add(_lin(all_s, [st.radicals for st in all_states]) == spec.radicals)

    def objective_expr(self, obj: Objective) -> tuple[cp_model.LinearExpr, float]:
        terms, coefs = [], []
        if obj.edge is not None:
            w = np.asarray(obj.edge, dtype=np.float64)
            for e, row in enumerate(self.y):
                for k, var in row.items():
                    terms.append(var)
                    coefs.append(w[e, k])
        if obj.atom is not None:
            for i, cands in enumerate(self.cand):
                for (st, eta), var in zip(cands, self.s[i]):
                    terms.append(var)
                    coefs.append(obj.atom(i, st, eta))
        c = np.asarray(coefs, dtype=np.float64)
        if len(c) and not np.isfinite(c).all():
            raise ValueError("objective has non-finite scores; use DecodeSpec.allowed to forbid orders")
        scale = 1.0 if _is_integral(c) else float(SCALE)
        ints = np.round(c * scale).astype(np.int64)
        return _lin(terms, ints.tolist()), scale

    def literals(self) -> list[cp_model.IntVar]:
        return [v for row in self.y for v in row.values()] + [v for row in self.s for v in row]

    def extract(self, solver) -> Graph:
        spec = self.spec
        orders = np.zeros(len(spec.edges), dtype=np.int64)
        for e, row in enumerate(self.y):
            for k, var in row.items():
                if solver.value(var):
                    orders[e] = k
        q = np.zeros(spec.n, dtype=np.int64)
        h = np.zeros(spec.n, dtype=np.int64)
        r = np.zeros(spec.n, dtype=np.int64)
        for i, cands in enumerate(self.cand):
            for (st, eta), var in zip(cands, self.s[i]):
                if solver.value(var):
                    q[i], h[i], r[i] = st.charge, eta, st.radicals
        return Graph(spec.n, spec.edges, orders, q, h, r)


def _solver(spec: DecodeSpec, enumerate_all: bool = False) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = spec.time_limit
    solver.parameters.num_workers = 1  # deterministic
    solver.parameters.random_seed = 0
    if enumerate_all:
        solver.parameters.enumerate_all_solutions = True
    return solver


class _Collector(cp_model.CpSolverSolutionCallback):
    def __init__(self, model: _Model, cap: int):
        super().__init__()
        self._model, self._cap = model, cap
        self.graphs: list[Graph] = []
        self.truncated = False

    def on_solution_callback(self) -> None:
        if len(self.graphs) >= self._cap:
            self.truncated = True
            self.stop_search()
            return
        self.graphs.append(self._model.extract(self))


def diagnose(spec: DecodeSpec) -> str:
    """Name the constraint that makes 𝒢 ∩ {supp B ⊆ E} empty (FEASIBILITY_GATE)."""
    ok = (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if _solver(spec).solve(_Model(spec, relax_charge=True).model) in ok:
        return "charge"
    widened = DecodeSpec(**{**spec.__dict__, "allowed": np.ones_like(spec.allowed)})
    if not spec.allowed.all() and _solver(widened).solve(_Model(widened, relax_charge=True).model) in ok:
        return "contact_map"
    return "states"


def decode(
    spec: DecodeSpec,
    objectives: Sequence[Objective],
    pool: str | int = "none",
    max_pool: int = 64,
) -> DecodeResult:
    """Lexicographically maximize ``objectives`` over 𝒢 ∩ {supp B ⊆ E}.

    ``pool``: ``"none"`` returns one optimum; ``"all"`` enumerates every
    co-optimal structure of the last stage (up to ``max_pool``); an integer K
    returns the top-K structures of the last stage via no-good cuts.
    """
    mdl = _Model(spec)
    m = mdl.model
    if not objectives:
        objectives = [Objective(name="feasibility")]
    exprs = []
    values: list[float] = []
    for stage, obj in enumerate(objectives):
        expr, scale = mdl.objective_expr(obj)
        exprs.append((expr, scale))
        m.maximize(expr)
        solver = _solver(spec)
        status = solver.solve(m)
        if status == cp_model.INFEASIBLE:
            return DecodeResult("infeasible", reason=diagnose(spec))
        if status != cp_model.OPTIMAL:
            return DecodeResult("timeout", reason=f"stage {stage} ({obj.name}) not proven optimal")
        best = int(round(solver.objective_value))
        values.append(best / scale)
        if stage < len(objectives) - 1 or pool == "all":
            m.add(expr == best)
        best_graph = mdl.extract(solver)

    if pool == "none":
        return DecodeResult("ok", [Solution(best_graph, tuple(values))])

    if pool == "all":
        m.clear_objective()
        collector = _Collector(mdl, max_pool)
        status = _solver(spec, enumerate_all=True).solve(m, collector)
        # OPTIMAL means the enumeration finished; otherwise (time limit) the pool is incomplete
        truncated = collector.truncated or status != cp_model.OPTIMAL
        graphs = collector.graphs or [best_graph]  # the proven optimum is co-optimal by construction
        sols = [Solution(g, tuple(values)) for g in graphs]
        return DecodeResult("ok", sols, truncated=truncated)

    k_best = int(pool)
    last_expr, last_scale = exprs[-1]
    sols = []
    lits = mdl.literals()
    for _ in range(k_best):
        solver = _solver(spec)
        status = solver.solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        g = mdl.extract(solver)
        sols.append(Solution(g, (*values[:-1], solver.value(last_expr) / last_scale)))
        true_lits = [v for v in lits if solver.value(v)]
        m.add(sum(true_lits) <= len(true_lits) - 1)
    return DecodeResult("ok", sols)
