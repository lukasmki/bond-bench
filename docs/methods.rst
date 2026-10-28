Methods
=======

``uv run bond-bench methods`` lists every registered method and whether it
can run on this machine; when it cannot, it names what is missing.

A method is named on the command line as ``name[+wrapper...]``, e.g.
``mambax+hybrid``. ``--param`` values go to the base method.

Canonical methods
-----------------

The four canonical algorithms are specified in
:doc:`theory/canonical-algorithms`, one per estimator family of
:doc:`theory/formulation`. They share the primitives in
``bond_bench.eval.core`` (element state tables, contact map, validity oracle,
CP-SAT decoder, equivariant tie-breaking).

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - name
     - family
     - notes
   * - ``canonical-O``
     - optimization
     - Contact map A(X), then a lexicographic prior solved exactly with
       CP-SAT. The ``kim2015`` preset maximizes ΣB, then minimizes Σ|q|. The
       ``atom_penalty`` preset minimizes the Wang et al. (2006) atomic penalty
       scores, then Σ|q|. Returns every co-optimal structure; the
       ``prior_tied`` flag marks resonance or tautomer ambiguity.
   * - ``canonical-R``
     - rule
     - Bond-length windows (centered on MMFF94 reference lengths) and
       angle-based hybridization feed a greedy cascade in certainty order,
       followed by Hamming-nearest projection onto 𝒢. ``--param windows=b2``
       uses windows fitted on B2 instead.
   * - ``canonical-M``
     - ml
     - A PaiNN network predicting bond-order probabilities on candidate pairs
       and (charge, H count) probabilities per atom; returns the entrywise
       argmax without enforcing validity. Needs ``uv sync --extra ml`` and a
       checkpoint in ``models/`` (see :doc:`full-benchmark`). Trained on B2,
       so its B2 scores are not held out.
   * - ``canonical-M+hybrid``
     - hybrid
     - Canonical H: constrained MAP over 𝒢 under canonical M's probabilities.

Other built-in methods
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - name
     - family
     - notes
   * - ``rdkit-determinebonds``
     - optimization
     - RDKit's xyz2mol (Kim & Kim 2015). Requires explicit H.
   * - ``external``
     - external
     - Any program that speaks the JSONL protocol (see :doc:`extending`):
       ``--param command='prog {input} {output}'``.

Adapters
--------

Published methods with GPL code, PyTorch or Uni-Core models, or conda-forge
builds run in their own environment under ``adapters/<name>/`` and talk to
the harness over the JSONL protocol. Each needs a one-time ``setup.sh``
(see :doc:`full-benchmark`).

.. list-table::
   :header-rows: 1
   :widths: 16 12 72

   * - name
     - family
     - notes
   * - ``openbabel``
     - external
     - Open Babel ``ConnectTheDots`` + ``PerceiveBondOrders`` (GPL-2.0).
   * - ``antechamber``
     - optimization
     - AmberTools antechamber bond typing (Wang et al. 2006). Explicit H only;
       assigns no formal charges. pixi.
   * - ``indigo``
     - optimization
     - indigo-bondorder (Welsh & Allison 2019) on the harness contact map.
       Explicit H only. ``--param algorithm=fpt|astar|local_optimisation``.
       pixi.
   * - ``ball``
     - optimization
     - BALL ``AssignBondOrderProcessor`` (Dehof et al. 2011) on the harness
       contact map. Explicit H only; no formal charges.
       ``--param strategy=fpt|a_star|ilp``. pixi, Linux.
   * - ``mambax``
     - ml
     - Loschen's random forest over pairs within 3 Å. Reports
       ``edge_probs``; no formal charges or H counts.
   * - ``yuelbond``
     - ml
     - YuelBond edge GNN on heavy atoms. Reports ``edge_probs``; no formal
       charges or H counts. Upstream has no license, so code and weights are
       fetched by ``setup.sh`` and never shipped.
   * - ``bondnet``
     - ml
     - BondNet v4 joint PaiNN models. Reports ``edge_probs``; no formal
       charges or H counts. ``--param seed=42|43|44``.
   * - ``mperformer``
     - ml
     - MPerformer (GPL-3.0) on heavy atoms; predicts formal charges and H
       counts.

Methods that do not predict formal charges or H counts produce invalid
outputs for charged or H-stripped structures. Run them with ``+project`` or,
for those reporting ``edge_probs``, ``+hybrid``.

Known issues are documented with each adapter: ``adapters/README.md`` covers
the shared conventions, and ``adapters/<name>/README.md`` covers upstream
bugs, version pins and limits.

Wrappers
--------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - suffix
     - effect
   * - ``+project``
     - Maps an invalid output to the nearest valid structure on the same
       support.
   * - ``+hybrid``
     - Constrained MAP over 𝒢 from the inner method's ``edge_probs``, plus its
       ``atom_probs`` as atom scores when it reports them.

Capabilities
------------

Each method declares a
:class:`~bond_bench.eval.methods.base.Capabilities`. The runner marks a query
``unsupported`` without calling the method when the query needs a capability
the method lacks (see :doc:`evaluation`). A wrapped method keeps the
capabilities of the method inside it.

The default element set is H, B, C, N, O, F, Al, Si, P, S, Cl, Ga, Ge, As,
Se, Br, In, Sn, Sb, Te, I.

.. list-table::
   :header-rows: 1
   :widths: 20 10 9 9 30 22

   * - name
     - stripped H
     - charged
     - open shell
     - elements
     - other limits
   * - ``canonical-O``
     - ✅
     - ✅
     - ✅
     - default
     -
   * - ``canonical-R``
     - ✅
     - ✅
     - ✅
     - default
     -
   * - ``canonical-M``
     - ✅
     - ✅
     - ❌
     - default
     -
   * - ``rdkit-determinebonds``
     - ❌
     - ✅
     - ❌
     - default
     -
   * - ``external``
     - ✅
     - ✅
     - ✅
     - any
     -
   * - ``openbabel``
     - ✅
     - ✅
     - ❌
     - default
     -
   * - ``antechamber``
     - ❌
     - ✅
     - ❌
     - H, C, N, O, F, Si, P, S, Cl, Br, I
     -
   * - ``indigo``
     - ❌
     - ✅
     - ❌
     - H, C, N, O, F, P, S, Cl, Br
     - one connected contact graph
   * - ``ball``
     - ❌
     - ✅
     - ❌
     - H, C, N, O, F, Si, P, S, Cl, Br, I
     - one connected contact graph
   * - ``mambax``
     - ✅
     - ✅
     - ❌
     - H, B, C, N, O, F, Si, P, S, Cl, Br, I
     -
   * - ``yuelbond``
     - ✅
     - ✅
     - ❌
     - H, C, N, O, F, P, S, Cl, Br, I
     - at most 127 heavy atoms
   * - ``bondnet``
     - ✅
     - ✅
     - ❌
     - H, B, C, N, O, F, Si, P, S, Cl, As, Br, I, Hg, Bi
     -
   * - ``mperformer``
     - ✅
     - ✅
     - ❌
     - H, Li, B, C, N, O, F, Na, Al, Si, P, S, Cl, K, Ca, Cr, Fe, Zn, As, Se,
       Br, Sn, I, Gd, Au, Hg
     - at most 256 heavy atoms
