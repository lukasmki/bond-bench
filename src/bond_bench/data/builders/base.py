"""Common machinery for stratum builders."""

from __future__ import annotations

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from ..chem import LabelError, validate_record
from ..manifest import REPO_ROOT, StratumSpec
from ..schema import MoleculeRecord
from ..store import write_stratum

log = logging.getLogger(__name__)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class StratumBuilder(ABC):
    """Builds one stratum file.

    Subclasses implement :meth:`fetch` (idempotent downloads into
    ``self.raw_dir``) and :meth:`records` (the final, selected records). Use
    :meth:`accept` on every candidate so the label is validated and the
    rejection funnel is recorded in the build report.
    """

    stratum_id: ClassVar[str]
    #: Default builder parameters, overridable via ``params``.
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        spec: StratumSpec,
        cache_dir: Path,
        out_dir: Path,
        seed: int = 0,
        n: int | None = None,
        params: dict[str, Any] | None = None,
    ):
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.out_dir = Path(out_dir)
        self.seed = seed
        self.n = spec.n_structures if n is None else n
        self.params = {**self.defaults, **(params or {})}
        self.funnel: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.extra_attrs: dict[str, Any] = {}

    @property
    def raw_dir(self) -> Path:
        return self.cache_dir / self.source_name

    @property
    def source_name(self) -> str:
        return self.stratum_id

    @property
    def output_path(self) -> Path:
        return self.out_dir / f"{self.stratum_id}.h5"

    @abstractmethod
    def fetch(self) -> None: ...

    @abstractmethod
    def records(self) -> list[MoleculeRecord]: ...

    def reject(self, reason: str) -> None:
        self.rejections[reason] += 1

    def accept(self, record: MoleculeRecord) -> bool:
        """Validate a candidate's label; count and drop it if inconsistent."""
        try:
            validate_record(record)
        except LabelError as exc:
            self.reject(f"label: {str(exc).split(':')[0]}")
            return False
        return True

    def build(self) -> Path:
        log.info("[%s] fetching sources", self.stratum_id)
        self.fetch()
        log.info("[%s] building records (target n=%d)", self.stratum_id, self.n)
        records = self.records()
        if len(records) < self.n:
            log.warning(
                "[%s] only %d of %d requested structures available",
                self.stratum_id, len(records), self.n,
            )
        attrs = {
            **self.spec.as_attrs(),
            "n_requested": self.n,
            "seed": self.seed,
            "params": self.params,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            **self.extra_attrs,
        }
        path = write_stratum(self.output_path, records, attrs)
        self.write_report(len(records))
        log.info("[%s] wrote %d structures to %s", self.stratum_id, len(records), path)
        return path

    def write_report(self, n_written: int) -> Path:
        report = {
            "stratum_id": self.stratum_id,
            "n_requested": self.n,
            "n_written": n_written,
            "seed": self.seed,
            "params": self.params,
            "funnel": dict(self.funnel),
            "rejections": dict(self.rejections.most_common()),
            **self.extra_attrs,
        }
        path = self.out_dir / "reports" / f"{self.stratum_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return path
