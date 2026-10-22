"""TRAIN M: fit f_θ by the (resonance-aware) cross-entropy over a training measure P (A4).

The training measure is a set of strata files. Structures are split into
train/validation by a hash of their id; the checkpoint with the lowest
validation loss is kept. Optional augmentations follow the algorithm: Gaussian
coordinate noise (σ_aug) and the H-free regime (``regimes``).
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ...data.builders.base import git_commit
from ...data.download import sha256sum
from ...data.sampling import derive_seed
from ...data.schema import MoleculeRecord
from ...data.store import StratumReader
from ..core.graph import Graph
from .labels import CUTOFF_SCALE, Target, candidate_pairs, targets, trainable
from .model import BondOrderNet, ModelConfig, class_nll, collate, graph_inputs, member_labels

log = logging.getLogger(__name__)

CHECKPOINT_FORMAT = 1


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 2
    sigma_aug: float = 0.0  # Å, Gaussian coordinate noise per sample and epoch
    regimes: tuple[str, ...] = ("explicit_h", "stripped")
    resonance: bool = True  # min over the label class (False: the reference graph only)
    max_members: int = 64
    cutoff_scale: float = CUTOFF_SCALE
    val_fraction: float = 0.1
    seed: int = 0
    loader_workers: int = 4
    label_workers: int = 8
    max_samples: int | None = None  # cap on structures (smoke runs)
    model: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------- data


def _targets_of(args: tuple[MoleculeRecord, tuple[str, ...], bool, int]) -> tuple[str, list[Target] | str]:
    record, regimes, resonance, max_members = args
    reason = trainable(record)
    if reason:
        return record.structure_id, reason
    return record.structure_id, targets(record, regimes, resonance, max_members)


def build_targets(records: Sequence[MoleculeRecord], cfg: TrainConfig) -> tuple[list[Target], Counter]:
    """Targets for every trainable record; ``skipped`` counts the rest by reason."""
    jobs = [(r, tuple(cfg.regimes), cfg.resonance, cfg.max_members) for r in records]
    out: list[Target] = []
    skipped: Counter = Counter()
    with ProcessPoolExecutor(max(cfg.label_workers, 1)) as pool:
        for _, res in pool.map(_targets_of, jobs, chunksize=16):
            if isinstance(res, str):
                skipped[res] += 1
            else:
                out.extend(res)
    return out, skipped


def is_validation(structure_id: str, cfg: TrainConfig) -> bool:
    return derive_seed(cfg.seed, "canonical-M-split", structure_id) % 10_000 < cfg.val_fraction * 10_000


class TargetDataset(torch.utils.data.Dataset):
    """Featurizes on access, so coordinate noise is redrawn every epoch."""

    def __init__(self, items: list[Target], cfg: TrainConfig, noise: bool):
        self.items, self.cfg, self.noise = items, cfg, noise
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> tuple[dict, np.ndarray, np.ndarray, int]:
        t, cfg = self.items[k], self.cfg
        x = t.positions
        if self.noise and cfg.sigma_aug > 0:
            rng = np.random.default_rng(derive_seed(cfg.seed, "aug", t.structure_id, t.observation, self.epoch))
            x = x + rng.normal(0.0, cfg.sigma_aug, x.shape)
        pairs = candidate_pairs(t.z, x, cfg.cutoff_scale)
        return graph_inputs(t.z, x, t.total_charge, pairs, cfg.model.r_msg), t.edge_labels(pairs), t.atom_labels, t.missed_bonds(pairs)


def collate_targets(batch: list) -> dict:
    inputs, edge_labels, atom_labels, missed = zip(*batch)
    return {"inputs": collate(list(inputs)), "labels": member_labels(list(edge_labels), list(atom_labels)), "missed": sum(missed)}


def _to(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in d.items()}


# ---------------------------------------------------------------- training


def seen_states(records: Sequence[MoleculeRecord]) -> dict[str, list]:
    """Elements and Λ-states (z, q, D) of the training labels, for out-of-distribution flags."""
    states = set()
    for r in records:
        g = Graph.from_record(r)
        states.update(zip(r.atomic_numbers.tolist(), g.charges.tolist(), g.valences().tolist()))
    return {
        "elements": sorted({int(z) for z, _, _ in states}),
        "states": sorted([int(z), int(q), int(d)] for z, q, d in states),
    }


def _evaluate(model: BondOrderNet, loader, device) -> dict[str, float]:
    model.eval()
    nll, exact, n, missed = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            edge_logits, atom_logits = model(_to(batch["inputs"], device))
            out = class_nll(edge_logits, atom_logits, _to(batch["labels"], device))
            nll += float(out["nll"].sum())
            exact += int(out["exact"].sum())
            n += len(out["nll"])
            missed += batch["missed"]
    return {"loss": nll / max(n, 1), "exact": exact / max(n, 1), "missed_bonds": missed}


def train(
    strata: Sequence[Path],
    cfg: TrainConfig,
    device: str = "cuda",
    log_every: int = 1,
) -> dict:
    """Train on ``strata`` and return the checkpoint (see :func:`save_checkpoint`)."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    records: list[MoleculeRecord] = []
    for path in strata:
        records.extend(StratumReader(path))
    if cfg.max_samples is not None:
        records = sorted(records, key=lambda r: derive_seed(cfg.seed, "subsample", r.structure_id))[: cfg.max_samples]
    t0 = time.perf_counter()
    items, skipped = build_targets(records, cfg)
    n_members = [len(t.bond_orders) for t in items]
    log.info("%d targets from %d records in %.0f s (skipped %s); label-class size mean %.2f, max %d",
             len(items), len(records), time.perf_counter() - t0, dict(skipped), np.mean(n_members), max(n_members))
    train_items = [t for t in items if not is_validation(t.structure_id, cfg)]
    val_items = [t for t in items if is_validation(t.structure_id, cfg)]
    train_ids = sorted({t.structure_id for t in train_items})

    loader_kw = {"batch_size": cfg.batch_size, "collate_fn": collate_targets, "num_workers": cfg.loader_workers,
                 "persistent_workers": cfg.loader_workers > 0, "pin_memory": dev.type == "cuda"}
    train_ds = TargetDataset(train_items, cfg, noise=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed), drop_last=True, **loader_kw
    )
    val_loader = torch.utils.data.DataLoader(TargetDataset(val_items, cfg, noise=False), shuffle=False, **loader_kw)

    model = BondOrderNet(cfg.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total, warmup = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    best = {"loss": float("inf")}
    best_state, history = None, []
    for epoch in range(cfg.epochs):
        train_ds.epoch = epoch  # workers are persistent: noise reseeding needs fresh workers
        if cfg.sigma_aug > 0 and cfg.loader_workers > 0:
            train_loader = torch.utils.data.DataLoader(
                train_ds, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed + epoch), drop_last=True,
                **{**loader_kw, "persistent_workers": False},
            )
        model.train()
        t0, running, count = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            edge_logits, atom_logits = model(_to(batch["inputs"], dev))
            loss = class_nll(edge_logits, atom_logits, _to(batch["labels"], dev))["nll"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            running += loss.item() * cfg.batch_size
            count += cfg.batch_size
        val = _evaluate(model, val_loader, dev)
        row = {"epoch": epoch, "train_loss": running / max(count, 1), **{f"val_{k}": v for k, v in val.items()},
               "seconds": time.perf_counter() - t0}
        history.append(row)
        if val["loss"] < best["loss"]:
            best = {**val, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            log.info("epoch %3d  train %.4f  val %.4f  val exact %.3f  (%.1f s)",
                     epoch, row["train_loss"], val["loss"], val["exact"], row["seconds"])

    in_train = set(train_ids)
    train_records = [r for r in records if r.structure_id in in_train]
    return {
        "format": CHECKPOINT_FORMAT,
        "model_config": cfg.model.to_dict(),
        "state_dict": best_state,
        "seen": seen_states(train_records),
        "train": {
            "config": {k: v for k, v in asdict(cfg).items() if k != "model"},
            "strata": [{"file": Path(p).name, "sha256": sha256sum(Path(p))} for p in strata],
            "n_records": len(records),
            "skipped": dict(skipped),
            "n_train_targets": len(train_items),
            "n_val_targets": len(val_items),
            "train_ids": train_ids,
            "best": best,
            "history": history,
            "git_commit": git_commit(),
            "torch": str(torch.__version__),
        },
    }


def save_checkpoint(ckpt: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: checkpoint format {ckpt.get('format')!r}, expected {CHECKPOINT_FORMAT}")
    return ckpt


def load_model(ckpt: dict, dtype: torch.dtype = torch.float64) -> BondOrderNet:
    model = BondOrderNet(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(dtype).eval()
