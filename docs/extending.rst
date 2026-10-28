Adding methods and strata
=========================

There are three ways to benchmark a new method, from least to most
integrated:

1. an external program, run with ``--param command=...``;
2. an adapter under ``adapters/<name>/`` with its own environment;
3. an in-process :class:`~bond_bench.eval.methods.base.Method` subclass.

External programs
-----------------

Any program that reads queries from a JSONL file and writes predictions to
another can be run without touching the code:

.. code-block:: bash

   uv run bond-bench run external --param command='my-prog {input} {output}' --param label=my-prog

``{input}`` and ``{output}`` are replaced by the paths of the two files.
``label`` names the results directory (``results/my-prog/``). The program
is called once per batch, and each batch has a time limit of one hour
(``--param batch_timeout=<seconds>`` changes it).

Each input line is one query:

.. code-block:: json

   {"structure_id": "B1-000001",
    "atomic_numbers": [6, 8, 1, 1],
    "positions": [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0], [-0.54, 0.94, 0.0], [-0.54, -0.94, 0.0]],
    "total_charge": 0,
    "multiplicity": 1,
    "observation": "explicit_h"}

Methods that assign bond orders to a given connectivity, rather than
perceive it, can also receive ``"contacts"``: the atom-index pairs of the
contact map A(X) that ``canonical-O`` uses.

Each output line is one prediction:

.. code-block:: json

   {"structure_id": "B1-000001",
    "status": "ok",
    "atomic_numbers": [6, 8, 1, 1],
    "bonds": [[0, 1], [0, 2], [0, 3]],
    "bond_orders": [2, 1, 1],
    "formal_charges": [0, 0, 0, 0],
    "implicit_hydrogens": [0, 0, 0, 0],
    "radical_electrons": [0, 0, 0, 0],
    "runtime_s": 0.002}

Rules:

- Atom order must be preserved. ``atomic_numbers`` is echoed back and
  checked; a mismatch is an ``error``.
- ``bond_orders`` are Kekulé orders (1, 2, 3). Aromatic orders are not
  accepted.
- ``implicit_hydrogens`` matters only for ``stripped`` queries; for
  ``explicit_h`` queries it is ignored.
- On failure, write ``{"structure_id": ..., "status": "error", "reason": ...}``
  (or ``"infeasible"``). A structure missing from the output is an error.
- Optionally, ``edge_pairs`` (m × 2) and ``edge_probs`` (m × 4,
  P(order = 0, 1, 2, 3)) let the method be decoded with ``+hybrid``.

The protocol is defined in ``bond_bench/eval/methods/external.py``.

Adapters
--------

A method with heavy or conflicting dependencies gets its own directory
``adapters/<name>/`` with

- ``pyproject.toml`` (uv) or ``pixi.toml`` (pixi) for its environment,
- ``run.py``, which speaks the protocol above and also answers
  ``run.py --version``, and
- optionally ``setup.sh``, which fetches pinned upstream code and weights
  into ``adapters/<name>/vendor/`` (not committed).

``adapters/_shared/bbadapter.py`` handles the file I/O, timing and per-molecule
error reporting. ``run.py`` only needs a function that handles one query:

.. code-block:: python

   import sys
   from pathlib import Path

   sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
   from bbadapter import serve


   def perceive(q):
       ...  # q is one input line as a dict
       return {"atomic_numbers": ..., "bonds": ..., "bond_orders": ...,
               "formal_charges": ..., "implicit_hydrogens": ..., "radical_electrons": ...}


   if __name__ == "__main__":
       serve(perceive, version="my-method 1.0")

Then register it in ``bond_bench/eval/methods/adapters.py``:

.. code-block:: python

   @register
   class MyMethod(AdapterMethod):
       """One-line description and license."""

       name = "my-method"
       family = "ml"
       adapter = "my-method"            # directory under adapters/
       runner = "uv"                    # or "pixi"
       requires = ("vendor/weights.pt",)  # files setup.sh must create
       capabilities = Capabilities(stripped_h=True, charged=False)

Document the adapter's upstream pins and known issues in
``adapters/<name>/README.md``; ``adapters/README.md`` covers the shared
conventions.

In-process methods
------------------

Subclass :class:`~bond_bench.eval.methods.base.Method`, declare its
:class:`~bond_bench.eval.methods.base.Capabilities`, implement ``predict``
and decorate the class with ``@register``. Then import its module in
``_load_all`` in ``bond_bench/eval/methods/base.py`` so the registry sees it.

.. code-block:: python

   from bond_bench.eval.core.geometry import CONTACT_SCALE, contact_map
   from bond_bench.eval.core.graph import Graph
   from bond_bench.eval.methods.base import Capabilities, Method, register
   from bond_bench.eval.prediction import Prediction, Status
   from bond_bench.eval.query import Query


   @register
   class AllSingle(Method):
       """Every contact is a single bond (a baseline)."""

       name = "all-single"
       family = "rule"
       capabilities = Capabilities(stripped_h=False, charged=False)
       defaults = {"scale": CONTACT_SCALE}

       def predict(self, query: Query) -> Prediction:
           bonds, _ = contact_map(query.atomic_numbers, query.positions, self.params["scale"])
           if len(bonds) == 0:
               return Prediction.fail(Status.INFEASIBLE, "no contacts")
           n = query.n_atoms
           graph = Graph(n, bonds, [1] * len(bonds), [0] * n, [0] * n, [0] * n)
           return Prediction.ok(graph)

- Parameters are declared in ``defaults``; ``--param`` rejects unknown keys.
- The harness decides ``unsupported`` from the capabilities before calling
  ``predict``, so ``predict`` only sees queries it claims to handle.
- Do not check validity yourself; the harness does.
- Return co-optimal structures in ``alternatives`` and probabilities in
  ``edge_pairs``/``edge_probs`` (and ``atom_probs``) when the method has them.
- Override ``version()`` if the method depends on an external library, so the
  library version ends up in ``provenance.json``.

Adding a stratum
----------------

1. Add or edit the stratum's row in ``assets/benchmark_strata_manifest.csv``.
2. Subclass :class:`~bond_bench.data.builders.base.StratumBuilder` in
   ``bond_bench/data/builders/`` and set its ``stratum_id``. Implement ``fetch`` (idempotent downloads
   into ``self.raw_dir``) and ``records`` (the final, selected records), and
   pass every candidate through ``accept`` so its label is validated and the
   rejection funnel ends up in the build report.
3. Add the builder to ``BUILDERS`` in ``bond_bench/data/builders/registry.py``.
4. ``uv run bond-bench build <id> --n 30 && uv run bond-bench validate``.

Keep builds deterministic: derive every random choice from ``self.seed``.
