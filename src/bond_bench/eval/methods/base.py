"""Method interface and registry.

A method maps a :class:`Query` to a :class:`Prediction`. It declares its
:class:`Capabilities`, so the runner can mark queries it cannot handle as
``unsupported`` without calling it. Composite methods are named with ``+``:
``openbabel+project`` wraps ``openbabel`` in the valid-set projection.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np

from ..core.elements import SUPPORTED_ELEMENTS
from ..prediction import Prediction
from ..query import Query


@dataclass(frozen=True)
class Capabilities:
    stripped_h: bool = False  # handles π_H(X) (infers implicit H)
    charged: bool = True  # accepts Q ≠ 0
    radicals: bool = False  # accepts multiplicity > 1
    elements: frozenset[int] | None = SUPPORTED_ELEMENTS  # None = any element
    max_atoms: int | None = None


class Method(ABC):
    name: ClassVar[str]
    family: ClassVar[str]  # rule | optimization | ml | hybrid | external
    capabilities: ClassVar[Capabilities] = Capabilities()
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}

    @property
    def tag(self) -> str:
        """Results directory name; non-default params are appended."""
        changed = {k: v for k, v in self.params.items() if v != self.defaults.get(k)}
        if not changed:
            return self.name
        return self.name + "@" + ",".join(f"{k}={v}" for k, v in sorted(changed.items()))

    def unsupported_reason(self, query: Query) -> str | None:
        caps = self.capabilities
        if not query.hydrogens_observed and not caps.stripped_h:
            return "stripped hydrogens"
        if query.total_charge != 0 and not caps.charged:
            return "charged"
        if query.multiplicity > 1 and not caps.radicals:
            return "open shell"
        if caps.elements is not None:
            extra = set(np.unique(query.atomic_numbers).tolist()) - caps.elements
            if extra:
                return f"elements {sorted(extra)}"
        if caps.max_atoms is not None and query.n_atoms > caps.max_atoms:
            return "too many atoms"
        return None

    @abstractmethod
    def predict(self, query: Query) -> Prediction: ...

    def predict_batch(self, queries: Sequence[Query]) -> list[Prediction]:
        return [self.predict(q) for q in queries]

    def version(self) -> str:
        from importlib.metadata import version

        return version("bond-bench")

    def provenance(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "tag": self.tag,
            "family": self.family,
            "version": self.version(),
            "params": json.loads(json.dumps(self.params, default=str)),
            "capabilities": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in asdict(self.capabilities).items()},
        }


class Wrapper(Method):
    """A method that post-processes another method's output."""

    suffix: ClassVar[str]

    def __init__(self, inner: Method, **params: Any):
        super().__init__(**params)
        self.inner = inner

    @property
    def tag(self) -> str:
        own = super().tag
        return f"{self.inner.tag}+{own}"

    def unsupported_reason(self, query: Query) -> str | None:
        return self.inner.unsupported_reason(query)

    @abstractmethod
    def finish(self, query: Query, pred: Prediction) -> Prediction:
        """Post-process the inner method's prediction for ``query``."""

    def predict(self, query: Query) -> Prediction:
        return self.finish(query, self.inner.predict(query))

    def predict_batch(self, queries: Sequence[Query]) -> list[Prediction]:
        # the inner method sees the whole batch (one adapter process per batch)
        return [self.finish(q, p) for q, p in zip(queries, self.inner.predict_batch(queries))]

    def version(self) -> str:
        return self.inner.version()

    def provenance(self) -> dict[str, Any]:
        return {**super().provenance(), "inner": self.inner.provenance()}


def base_method(method: Method) -> Method:
    """The innermost method of a ``name+wrapper...`` composite."""
    while isinstance(method, Wrapper):
        method = method.inner
    return method


METHODS: dict[str, type[Method]] = {}
WRAPPERS: dict[str, type[Wrapper]] = {}


def register(cls):
    if issubclass(cls, Wrapper):
        WRAPPERS[cls.name] = cls
    else:
        METHODS[cls.name] = cls
    return cls


def _load_all() -> None:
    from . import adapters, external, hybrid, optimization, rdkit_determine, rule, wrappers  # noqa: F401


def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def get_method(spec: str, params: dict[str, Any] | None = None) -> Method:
    """Build a method from ``name[+wrapper...]``; ``params`` go to the base method."""
    _load_all()
    name, *wrappers = spec.split("+")
    try:
        method: Method = METHODS[name](**(params or {}))
    except KeyError:
        raise KeyError(f"unknown method {name!r}; available: {sorted(METHODS)}") from None
    for w in wrappers:
        try:
            method = WRAPPERS[w](method)
        except KeyError:
            raise KeyError(f"unknown wrapper {w!r}; available: {sorted(WRAPPERS)}") from None
    return method


def available_methods() -> dict[str, type[Method]]:
    _load_all()
    return dict(METHODS)
