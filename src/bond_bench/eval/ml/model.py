"""f_θ of canonical M: an E(3)-invariant, S_n-equivariant network (PaiNN, Schütt et al. 2021).

Atoms carry scalar features s_i and vector features v_i; messages run over all
pairs within ``r_msg``. Every output is a function of s and of inner products
of v, so it is invariant to rotations, reflections and translations, and the
sums over neighbours make it permutation-equivariant. The total charge Q
enters the initial scalars. Heads:

- edge head on candidate pairs (i, j) ∈ E: symmetric in i ↔ j, logits over
  orders {0, 1, 2, 3};
- atom head: logits over the (formal charge, implicit H) grid of
  ``prediction.ATOM_CHARGES × ATOM_HYDROGENS``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .labels import IGNORE, N_ATOM_CLASSES

MAX_Z = 54


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 128
    layers: int = 4
    r_msg: float = 5.0  # Å, message-passing radius
    n_rbf: int = 20  # radial basis of the messages
    n_pair_rbf: int = 64  # Gaussian basis of the candidate-pair distance
    pair_rbf_range: tuple[float, float] = (0.6, 3.6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = dict(d)
        d["pair_rbf_range"] = tuple(d["pair_rbf_range"])
        return cls(**d)


# ---------------------------------------------------------------- inputs


def message_pairs(positions: np.ndarray, r_msg: float) -> np.ndarray:
    """Directed pairs (src, dst) within ``r_msg``, both directions."""
    x = np.asarray(positions, dtype=np.float64)
    if len(x) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    p = cKDTree(x).query_pairs(r_msg, output_type="ndarray").astype(np.int64).reshape(-1, 2)
    return np.concatenate([p, p[:, ::-1]])


def graph_inputs(z: np.ndarray, positions: np.ndarray, total_charge: int, pairs: np.ndarray, r_msg: float) -> dict:
    return {
        "z": np.asarray(z, dtype=np.int64),
        "pos": np.asarray(positions, dtype=np.float64),
        "charge": float(total_charge),
        "msg": message_pairs(positions, r_msg),
        "pairs": np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
    }


def collate(items: list[dict], dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Concatenate molecules into one disconnected graph (atom indices offset per molecule)."""
    offsets = np.cumsum([0] + [len(it["z"]) for it in items[:-1]])
    n_atoms = [len(it["z"]) for it in items]
    cat = lambda key: np.concatenate([it[key] + o for it, o in zip(items, offsets)])  # noqa: E731
    return {
        "z": torch.from_numpy(np.concatenate([it["z"] for it in items])),
        "pos": torch.from_numpy(np.concatenate([it["pos"] for it in items])).to(dtype),
        "charge": torch.tensor(np.repeat([it["charge"] for it in items], n_atoms), dtype=dtype),
        "msg": torch.from_numpy(cat("msg").reshape(-1, 2)),
        "pairs": torch.from_numpy(cat("pairs").reshape(-1, 2)),
        "mol": torch.from_numpy(np.repeat(np.arange(len(items)), n_atoms)),
        "pair_mol": torch.from_numpy(np.repeat(np.arange(len(items)), [len(it["pairs"]) for it in items])),
    }


# ---------------------------------------------------------------- network


def _sinc_basis(d: torch.Tensor, n: int, cutoff: float) -> torch.Tensor:
    k = torch.arange(1, n + 1, device=d.device, dtype=d.dtype)
    return torch.sin(k * math.pi * d[:, None] / cutoff) / d[:, None]


def _cosine_cutoff(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    return 0.5 * (torch.cos(math.pi * d / cutoff) + 1.0) * (d < cutoff)


class _Message(nn.Module):
    def __init__(self, f: int, n_rbf: int):
        super().__init__()
        self.f = f
        self.phi = nn.Sequential(nn.Linear(f, f), nn.SiLU(), nn.Linear(f, 3 * f))
        self.w = nn.Linear(n_rbf, 3 * f)

    def forward(self, s, v, src, dst, rbf, unit, fcut):
        x = self.phi(s)[src] * self.w(rbf) * fcut[:, None]
        ds, dvv, dvs = x.split(self.f, dim=-1)
        dv = dvv[:, None, :] * v[src] + dvs[:, None, :] * unit[:, :, None]
        return s.index_add(0, dst, ds), v.index_add(0, dst, dv)


class _Update(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.f = f
        self.u = nn.Linear(f, f, bias=False)
        self.v = nn.Linear(f, f, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * f, f), nn.SiLU(), nn.Linear(f, 3 * f))

    def forward(self, s, v):
        uv, vv = self.u(v), self.v(v)
        norm = torch.sqrt((vv**2).sum(dim=1) + 1e-8)
        avv, asv, ass = self.mlp(torch.cat([s, norm], dim=-1)).split(self.f, dim=-1)
        return s + asv * (uv * vv).sum(dim=1) + ass, v + avv[:, None, :] * uv


class BondOrderNet(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        f = config.hidden
        self.embed = nn.Embedding(MAX_Z + 1, f)
        self.charge = nn.Linear(1, f, bias=False)
        self.messages = nn.ModuleList(_Message(f, config.n_rbf) for _ in range(config.layers))
        self.updates = nn.ModuleList(_Update(f) for _ in range(config.layers))
        lo, hi = config.pair_rbf_range
        self.register_buffer("centers", torch.linspace(lo, hi, config.n_pair_rbf), persistent=False)
        self.gamma = 0.5 / ((hi - lo) / (config.n_pair_rbf - 1)) ** 2
        self.edge_head = nn.Sequential(
            nn.Linear(3 * f + config.n_pair_rbf, f), nn.SiLU(), nn.Linear(f, f), nn.SiLU(), nn.Linear(f, 4)
        )
        self.atom_head = nn.Sequential(nn.Linear(f, f), nn.SiLU(), nn.Linear(f, N_ATOM_CLASSES))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        z, pos = batch["z"].clamp(max=MAX_Z), batch["pos"]
        s = self.embed(z) + self.charge(batch["charge"][:, None])
        v = torch.zeros(len(z), 3, cfg.hidden, dtype=pos.dtype, device=pos.device)
        src, dst = batch["msg"][:, 0], batch["msg"][:, 1]
        vec = pos[src] - pos[dst]
        d = vec.norm(dim=-1)
        rbf, fcut, unit = _sinc_basis(d, cfg.n_rbf, cfg.r_msg), _cosine_cutoff(d, cfg.r_msg), vec / d[:, None]
        for message, update in zip(self.messages, self.updates):
            s, v = message(s, v, src, dst, rbf, unit, fcut)
            s, v = update(s, v)

        i, j = batch["pairs"][:, 0], batch["pairs"][:, 1]
        dij = (pos[i] - pos[j]).norm(dim=-1)
        pair_rbf = torch.exp(-self.gamma * (dij[:, None] - self.centers) ** 2)
        edge_in = torch.cat([s[i] + s[j], s[i] * s[j], (v[i] * v[j]).sum(dim=1), pair_rbf], dim=-1)
        return self.edge_head(edge_in), self.atom_head(s)


# ---------------------------------------------------------------- loss


def member_labels(edge_labels: list[np.ndarray], atom_labels: list[np.ndarray]) -> dict[str, torch.Tensor]:
    """Flatten per-molecule (M, m) edge and (M, n) atom labels for :func:`class_nll`.

    Entry t of the flat arrays scores row ``*_row[t]`` of the batch against
    label ``*_label[t]`` for member ``*_member[t]``; member r belongs to
    molecule ``member_mol[r]`` at slot ``member_slot[r]``.
    """
    rows = {"edge": [], "atom": []}
    labels = {"edge": [], "atom": []}
    members = {"edge": [], "atom": []}
    member_mol, member_slot = [], []
    offset = {"edge": 0, "atom": 0}
    first = 0
    for k, (el, al) in enumerate(zip(edge_labels, atom_labels)):
        n_members = len(el)
        for kind, lab in (("edge", el), ("atom", al)):
            width = lab.shape[1]
            rows[kind].append(np.tile(np.arange(width) + offset[kind], n_members))
            labels[kind].append(lab.reshape(-1))
            members[kind].append(np.repeat(np.arange(n_members) + first, width))
            offset[kind] += width
        member_mol.extend([k] * n_members)
        member_slot.extend(range(n_members))
        first += n_members
    out = {f"{kind}_{name}": torch.from_numpy(np.concatenate(src[kind]).astype(np.int64))
           for name, src in (("row", rows), ("label", labels), ("member", members)) for kind in ("edge", "atom")}
    out["member_mol"] = torch.tensor(member_mol, dtype=torch.int64)
    out["member_slot"] = torch.tensor(member_slot, dtype=torch.int64)
    return out


def class_nll(edge_logits: torch.Tensor, atom_logits: torch.Tensor, lab: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-molecule min over label-class members of −log q(G') (and its argmax mismatch count)."""
    n_members = len(lab["member_mol"])
    n_mol = int(lab["member_mol"].max()) + 1
    nll = edge_logits.new_zeros(n_members)
    wrong = edge_logits.new_zeros(n_members)
    for kind, logits in (("edge", edge_logits), ("atom", atom_logits)):
        label, row, member = lab[f"{kind}_label"], lab[f"{kind}_row"], lab[f"{kind}_member"]
        keep = label != IGNORE
        label, row, member = label[keep], row[keep], member[keep]
        logp = torch.log_softmax(logits, dim=-1)
        nll = nll.index_add(0, member, -logp[row, label])
        wrong = wrong.index_add(0, member, (logits[row].argmax(dim=-1) != label).to(logits.dtype))
    slots = int(lab["member_slot"].max()) + 1
    grid = edge_logits.new_full((n_mol, slots), float("inf"))
    grid = grid.index_put((lab["member_mol"], lab["member_slot"]), nll)
    miss = edge_logits.new_full((n_mol, slots), float("inf"))
    miss = miss.index_put((lab["member_mol"], lab["member_slot"]), wrong)
    return {"nll": grid.min(dim=1).values, "exact": miss.min(dim=1).values == 0}
