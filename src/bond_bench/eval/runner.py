"""Run a method over strata and write predictions plus per-structure scores.

Layout under ``results/<method_tag>/``::

    <stratum>__<obs>.parquet               predictions (one row per structure)
    <stratum>__<obs>.scores.parquet        metric rows (``metrics.grade``)
    <stratum>__<obs>__noise.*.parquet      on-the-fly σ-grid replicates (optional)
    <stratum>__<obs>__equivariance.parquet (optional)
    provenance.json
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
import multiprocessing as mp
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..data.builders.base import git_commit
from ..data.download import sha256sum
from ..data.manifest import REPO_ROOT, StratumSpec
from ..data.sampling import derive_seed
from ..data.schema import MoleculeRecord
from ..data.store import StratumReader
from .core.canon import same_resonance_class, wl_colors
from .metrics import grade
from .methods.base import Method, base_method, get_method
from .methods.external import ExternalMethod
from .prediction import Prediction, Status, from_row, to_row
from .query import Observation, Query, random_euclidean

log = logging.getLogger(__name__)

DEFAULT_RESULTS = REPO_ROOT / "results"


def observations_for(spec: StratumSpec | None, requested: str = "auto") -> list[Observation]:
    if requested != "auto":
        return [Observation(requested)]
    default = (spec.hydrogens_default if spec else "Present").lower()
    if default.startswith("stripped"):
        return [Observation.STRIPPED]
    if "both" in default or "stripped" in default:
        return [Observation.EXPLICIT_H, Observation.STRIPPED]
    return [Observation.EXPLICIT_H]


def group_key(record: MoleculeRecord) -> str:
    """Records that must be sampled together (C1 noise ladders, B5 tautomer groups)."""
    return str(record.meta.get("parent_id") or record.meta.get("group_id") or record.structure_id)


def select(reader: StratumReader, n: int | None, seed: int) -> list[MoleculeRecord]:
    records = list(reader)
    if n is None or n >= len(records):
        return records
    groups: dict[str, list[MoleculeRecord]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    order = sorted(groups, key=lambda g: derive_seed(seed, "eval-select", reader.stratum_id, g))
    out: list[MoleculeRecord] = []
    for g in order:
        if len(out) >= n:
            break
        out.extend(groups[g])
    return out


# ---------------------------------------------------------------- execution

#: Default wall-clock limit per structure for in-process methods (seconds).
DEFAULT_TIMEOUT = 120.0


def _timed(method: Method, query: Query) -> Prediction:
    reason = method.unsupported_reason(query)
    if reason:
        return Prediction.fail(Status.UNSUPPORTED, reason)
    t0 = time.perf_counter()
    try:
        pred = method.predict(query)
    except Exception as exc:  # a method failure is a result, not a crash
        pred = Prediction.fail(Status.ERROR, f"{type(exc).__name__}: {exc}")
    pred.runtime_s = pred.runtime_s or time.perf_counter() - t0
    return pred


def _worker_loop(conn: Connection, spec: str, params: dict[str, Any]) -> None:
    method = get_method(spec, params)
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        k, query = msg
        conn.send((k, _timed(method, query)))


class TimeoutPool:
    """Worker processes with a per-structure wall-clock limit.

    ``ProcessPoolExecutor`` cannot cancel a running task, so a method that
    hangs on one molecule would stall the run. Here a worker that exceeds the
    limit is terminated and replaced, and its structure is recorded as
    ``timeout``. The clock starts when a task is sent to a ready worker, so
    worker start-up is not charged to the first structure.
    """

    def __init__(self, n_workers: int, spec: str, params: dict[str, Any], timeout: float):
        self.n_workers, self.spec, self.params, self.timeout = max(n_workers, 1), spec, params, timeout
        self.ctx = mp.get_context("spawn")

    def _start(self) -> tuple[mp.Process, Connection]:
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_worker_loop, args=(child, self.spec, self.params), daemon=True)
        proc.start()
        child.close()
        return proc, parent

    def map(self, queries: Sequence[Query]) -> list[Prediction]:
        out: list[Prediction | None] = [None] * len(queries)
        pending = list(range(len(queries)))[::-1]
        workers = [self._start() for _ in range(min(self.n_workers, len(queries)))]
        busy: dict[int, tuple[int, float]] = {}  # worker slot -> (task, start time)
        ready: set[int] = set()
        try:
            while pending or busy:
                for slot in sorted(ready - set(busy)):
                    if not pending:
                        break
                    k = pending.pop()
                    workers[slot][1].send((k, queries[k]))
                    busy[slot] = (k, time.monotonic())
                conns = {workers[s][1]: s for s in range(len(workers))}
                for conn in wait(list(conns), timeout=0.5):
                    slot = conns[conn]
                    try:
                        tag, pred = conn.recv()
                    except EOFError:  # worker died (e.g. segfault in native code)
                        self._replace(workers, slot, busy, ready, out, Status.ERROR, "worker crashed")
                        continue
                    if tag == "ready":
                        ready.add(slot)
                        continue
                    _, t0 = busy.pop(slot)
                    elapsed = time.monotonic() - t0
                    if elapsed > self.timeout:  # finished, but over the limit: same verdict as a kill
                        pred = Prediction.fail(Status.TIMEOUT, f"> {self.timeout:g} s", runtime_s=elapsed)
                    out[tag] = pred
                now = time.monotonic()
                for slot, (k, t0) in list(busy.items()):
                    if now - t0 > self.timeout:
                        self._replace(workers, slot, busy, ready, out, Status.TIMEOUT, f"> {self.timeout:g} s")
        finally:
            for proc, conn in workers:
                try:
                    conn.send(None)
                except (BrokenPipeError, OSError):
                    pass
                proc.join(timeout=1)
                if proc.is_alive():
                    proc.terminate()
        return out  # type: ignore[return-value]

    def _replace(self, workers, slot, busy, ready, out, status: Status, reason: str) -> None:
        proc, conn = workers[slot]
        if slot in busy:
            k, t0 = busy.pop(slot)
            out[k] = Prediction.fail(status, reason, runtime_s=time.monotonic() - t0)
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        conn.close()
        ready.discard(slot)
        workers[slot] = self._start()


@dataclass
class Runner:
    method_spec: str
    params: dict[str, Any]
    workers: int = 1
    batch_size: int = 256
    timeout: float | None = DEFAULT_TIMEOUT  # per structure; None = no limit (runs in-process)

    def __post_init__(self) -> None:
        self.method = get_method(self.method_spec, self.params)

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        if not queries:
            return []
        if isinstance(base_method(self.method), ExternalMethod):  # batched: one adapter process per batch
            return self._predict_external(queries)
        if self.timeout is None and self.workers <= 1:
            return [_timed(self.method, q) for q in queries]
        return TimeoutPool(self.workers, self.method_spec, self.params, self.timeout or float("inf")).map(queries)

    def _predict_external(self, queries: Sequence[Query]) -> list[Prediction]:
        out: list[Prediction | None] = [None] * len(queries)
        todo = []
        for k, q in enumerate(queries):
            reason = self.method.unsupported_reason(q)
            if reason:
                out[k] = Prediction.fail(Status.UNSUPPORTED, reason)
            else:
                todo.append(k)
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            for k, pred in zip(chunk, self.method.predict_batch([queries[k] for k in chunk])):
                out[k] = pred
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------- stratum runs


def _tautomer_lookup(records: Iterable[MoleculeRecord]):
    groups: dict[str, set[str]] = defaultdict(set)
    pref: dict[str, str] = {}
    for r in records:
        gid = r.meta.get("group_id")
        if gid is None:
            continue
        groups[gid].add(r.smiles)
        if int(r.meta.get("pref_label", 0) or 0) == 1:
            pref[gid] = r.smiles
    return groups, pref


def _meta_columns(records: Sequence[MoleculeRecord]) -> dict[str, list]:
    keys = ("res_bin", "sigma", "parent_id", "group_id", "pref_label")
    cols = {}
    for key in keys:
        if any(key in r.meta for r in records):
            cols[f"meta_{key}"] = [None if r.meta.get(key) is None else str(r.meta.get(key)) for r in records]
    return cols


def score_frame(records: Sequence[MoleculeRecord], obs: Observation, preds: Sequence[Prediction]) -> pl.DataFrame:
    groups, pref = _tautomer_lookup(records)
    rows = []
    for rec, pred in zip(records, preds):
        gid = rec.meta.get("group_id")
        rows.append(grade(rec, obs, pred, groups.get(gid) if gid else None, pref.get(gid) if gid else None))
    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.with_columns(**{k: pl.Series(v, dtype=pl.String) for k, v in _meta_columns(records).items()})


def run_stratum(
    runner: Runner,
    reader: StratumReader,
    spec: StratumSpec | None,
    out_dir: Path,
    observation: str = "auto",
    n: int | None = None,
    seed: int = 0,
    noise: Sequence[float] = (),
    reps: int = 1,
    equivariance: int = 0,
) -> list[Path]:
    records = select(reader, n, seed)
    written = []
    for obs in observations_for(spec, observation):
        stem = f"{reader.stratum_id}__{obs}"
        try:
            queries = [Query.from_record(r, obs) for r in records]
        except ValueError as exc:
            log.warning("[%s] skipping %s: %s", reader.stratum_id, obs, exc)
            continue
        log.info("[%s/%s] %s on %d structures", reader.stratum_id, obs, runner.method.tag, len(queries))
        preds = runner.predict(queries)
        pl.DataFrame([to_row(q.structure_id, p) for q, p in zip(queries, preds)], infer_schema_length=None).write_parquet(
            out_dir / f"{stem}.parquet"
        )
        score_frame(records, obs, preds).write_parquet(out_dir / f"{stem}.scores.parquet")
        written.append(out_dir / f"{stem}.parquet")

        for sigma in noise:
            noisy, noisy_records = [], []
            for rec, q in zip(records, queries):
                for r in range(reps):
                    rng = np.random.default_rng(derive_seed(seed, "noise", q.structure_id, sigma, r))
                    noisy.append(q.noisy(sigma, rng))
                    noisy_records.append(rec)
            npreds = runner.predict(noisy)
            df = score_frame(noisy_records, obs, npreds).with_columns(
                pl.lit(float(sigma)).alias("noise_sigma"),
                pl.Series("noise_rep", [r for _ in records for r in range(reps)]),
            )
            df.write_parquet(out_dir / f"{stem}__noise.s{sigma:.3f}.parquet")

        if equivariance:
            check_equivariance(runner, queries[:equivariance], preds[:equivariance], seed).write_parquet(
                out_dir / f"{stem}__equivariance.parquet"
            )
    return written


def check_equivariance(runner: Runner, queries: Sequence[Query], preds: Sequence[Prediction], seed: int) -> pl.DataFrame:
    """Φ((σ,g)·x) vs σ·Φ(x) for one random (σ, g) per query."""
    transformed, perms = [], []
    for q in queries:
        rng = np.random.default_rng(derive_seed(seed, "equivariance", q.structure_id))
        perm = rng.permutation(q.n_atoms)
        rot, t = random_euclidean(rng)
        transformed.append(q.transformed(perm, rot, t))
        perms.append(perm)
    tpreds = runner.predict(transformed)
    rows = []
    for q, perm, p, tp in zip(queries, perms, preds, tpreds):
        colors = wl_colors(q.atomic_numbers, q.positions)
        row = {
            "structure_id": q.structure_id, "status": str(p.status), "status_transformed": str(tp.status),
            "symmetric": len(set(colors.tolist())) < q.n_atoms,  # exact symmetry: strict equivariance is impossible
        }
        if p.graph is not None and tp.graph is not None:
            expected = p.graph.permuted(perm)
            row["strict"] = tp.graph == expected
            row["set"] = row["strict"] or any(a == expected for a in tp.alternatives)
            row["resonance"] = same_resonance_class(tp.graph, expected)
        else:
            row["strict"] = row["set"] = row["resonance"] = p.status == tp.status
        rows.append(row)
    return pl.DataFrame(rows)


def dataset_label(path: Path) -> str:
    """``path`` relative to the repository when inside it, so provenance carries no local paths."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else path.name


def run(
    method_spec: str,
    strata: Sequence[str],
    dataset_dir: Path,
    specs: dict[str, StratumSpec],
    results_dir: Path = DEFAULT_RESULTS,
    params: dict[str, Any] | None = None,
    observation: str = "auto",
    n: int | None = None,
    workers: int = 1,
    seed: int = 0,
    noise: Sequence[float] = (),
    reps: int = 1,
    equivariance: int = 0,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> Path:
    runner = Runner(method_spec, params or {}, workers=workers, timeout=timeout)
    out_dir = Path(results_dir) / runner.method.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_path = out_dir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"runs": {}}
    prov["method"] = runner.method.provenance()
    for stratum in strata:
        path = Path(dataset_dir) / f"{stratum}.h5"
        if not path.exists():
            log.warning("[%s] not built (%s); skipping", stratum, path)
            continue
        reader = StratumReader(path)
        run_stratum(runner, reader, specs.get(stratum), out_dir, observation, n, seed, noise, reps, equivariance)
        prov["runs"][stratum] = {
            "dataset": dataset_label(path),
            "dataset_sha256": sha256sum(path),
            "dataset_built_at": str(reader.attrs.get("built_at", "")),
            "dataset_git_commit": str(reader.attrs.get("git_commit", "")),
            "n": n,
            "seed": seed,
            "observation": observation,
            "noise": list(noise),
            "reps": reps,
            "equivariance": equivariance,
            "timeout_s": timeout,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        }
    prov_path.write_text(json.dumps(prov, indent=2, default=str) + "\n")
    return out_dir


def rescore(results_dir: Path, dataset_dir: Path) -> int:
    """Recompute ``*.scores.parquet`` from stored predictions (e.g. after a metrics change)."""
    count = 0
    for pred_path in sorted(Path(results_dir).glob("*/*__*.parquet")):
        name = pred_path.name
        if name.endswith(".scores.parquet") or "__noise" in name or "__equivariance" in name:
            continue
        stratum, obs = pred_path.stem.split("__")
        reader = StratumReader(Path(dataset_dir) / f"{stratum}.h5")
        df = pl.read_parquet(pred_path)
        records = [reader.get(sid) for sid in df["structure_id"]]
        observation = Observation(obs)
        preds = [
            from_row(row, rec.strip_hydrogens().n_atoms if observation is Observation.STRIPPED else rec.n_atoms)
            for row, rec in zip(df.iter_rows(named=True), records)
        ]
        score_frame(records, observation, preds).write_parquet(pred_path.with_suffix(".scores.parquet"))
        count += 1
    return count
