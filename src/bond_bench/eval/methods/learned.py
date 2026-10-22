"""Canonical M: discriminative posterior, decoded entrywise (§4.3).

A network trained by ``scripts/train_canonical_m.py`` gives q_θ(B_ij = k | X)
on the candidate pairs E (d_ij ≤ r_c(z_i, z_j), A6) and q_θ(q_i, h_i | X) per
atom. The decoder takes the entrywise argmax, the Hamming-loss Bayes estimator
over 𝓜: validity is reported (flag ``invalid``), not enforced. When hydrogens
are observed, h_i is fixed to 0 and only q_i is decoded. ``+hybrid`` decodes the
same posterior by constrained MAP over 𝒢 instead (canonical H).

Flags: ``ood_elements`` / ``ood_states`` when the query's elements or the
predicted (z, q, D) states do not occur in the training labels. ``extra``
records whether the structure itself was in the training set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...data.download import sha256sum
from ...data.manifest import REPO_ROOT
from ..core.graph import Graph
from ..core.validity import is_valid
from ..prediction import ATOM_CHARGES, ATOM_HYDROGENS, Prediction
from ..query import Query
from .base import Capabilities, Method, register

#: Trained checkpoints live in ``models/<name>.pt``; ``checkpoint`` takes a name or a ``.pt`` path.
CHECKPOINT_DIR = REPO_ROOT / "models"
DEFAULT_CHECKPOINT = "canonical-M"


def checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.suffix != ".pt":  # a name (keeps the results tag free of path separators)
        return CHECKPOINT_DIR / f"{checkpoint}.pt"
    return path if path.is_absolute() else REPO_ROOT / path


@register
class CanonicalM(Method):
    name = "canonical-M"
    family = "ml"
    capabilities = Capabilities(stripped_h=True, charged=True, radicals=False)
    defaults = {"checkpoint": DEFAULT_CHECKPOINT, "device": "cpu", "threads": 1}

    def __init__(self, **params):
        super().__init__(**params)
        self._model = None

    @property
    def checkpoint_path(self) -> Path:
        return checkpoint_path(str(self.params["checkpoint"]))

    def check_available(self) -> str | None:
        try:
            import torch  # noqa: F401
        except ImportError:
            return "torch not installed (uv sync --extra ml)"
        if not self.checkpoint_path.exists():
            return f"no checkpoint at {self.checkpoint_path} (scripts/train_canonical_m.py)"
        return None

    def version(self) -> str:
        path = self.checkpoint_path
        return f"{super().version()}+ckpt.{sha256sum(path)[:12] if path.exists() else 'missing'}"

    def _load(self):
        if self._model is None:
            import torch

            from ..ml.train import load_checkpoint, load_model

            torch.set_num_threads(int(self.params["threads"]))
            ckpt = load_checkpoint(self.checkpoint_path)
            self._model = load_model(ckpt).to(self.params["device"])
            self._seen_elements = set(ckpt["seen"]["elements"])
            self._seen_states = {tuple(s) for s in ckpt["seen"]["states"]}
            self._train_ids = set(ckpt["train"]["train_ids"])
            self._cutoff_scale = float(ckpt["train"]["config"]["cutoff_scale"])
        return self._model

    def posterior(self, query: Query) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(E, q(B_ij = k), q(q_i, h_i)) with shapes (m, 2), (m, 4), (n, |charges|, |H counts|)."""
        import torch

        from ..ml.labels import candidate_pairs
        from ..ml.model import collate, graph_inputs

        model = self._load()
        pairs = candidate_pairs(query.atomic_numbers, query.positions, self._cutoff_scale)
        inputs = graph_inputs(query.atomic_numbers, query.positions, query.total_charge, pairs, model.config.r_msg)
        batch = {k: v.to(self.params["device"]) for k, v in collate([inputs], dtype=torch.float64).items()}
        with torch.no_grad():
            edge_logits, atom_logits = model(batch)
        edge_probs = torch.softmax(edge_logits, dim=-1).cpu().numpy()
        atom_probs = torch.softmax(atom_logits, dim=-1).cpu().numpy()
        return pairs, edge_probs, atom_probs.reshape(query.n_atoms, len(ATOM_CHARGES), len(ATOM_HYDROGENS))

    def predict(self, query: Query) -> Prediction:
        z, n = query.atomic_numbers, query.n_atoms
        pairs, edge_probs, atom_probs = self.posterior(query)
        orders = edge_probs.argmax(axis=1)
        grid = atom_probs[:, :, :1] if query.hydrogens_observed else atom_probs  # η ≡ 0 when H are observed
        a, b = np.divmod(grid.reshape(n, -1).argmax(axis=1), grid.shape[2])
        g = Graph(n, pairs, orders, np.asarray(ATOM_CHARGES)[a], np.asarray(ATOM_HYDROGENS)[b], np.zeros(n))

        logp = np.sort(np.log(np.clip(edge_probs, 1e-300, 1.0)), axis=1)
        margin = (logp[:, -1] - logp[:, -2])[orders > 0]  # log-odds of the chosen order over the runner-up
        flags = []
        if not is_valid(z, query.total_charge, g):
            flags.append("invalid")
        if not set(np.unique(z).tolist()) <= self._seen_elements:
            flags.append("ood_elements")
        states = zip(z.tolist(), g.charges.tolist(), g.valences().tolist())
        if any(s not in self._seen_states for s in states):
            flags.append("ood_states")
        return Prediction.ok(
            g, edge_pairs=pairs, edge_probs=edge_probs, atom_probs=atom_probs, margin=margin, flags=flags,
            extra={"in_training_set": query.structure_id in self._train_ids},
        )
