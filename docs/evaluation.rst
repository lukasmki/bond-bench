Evaluation
==========

The task
--------

A method maps a :class:`~bond_bench.eval.query.Query` to a
:class:`~bond_bench.eval.prediction.Prediction`.

The query holds what a method is allowed to see: atomic numbers,
coordinates (Å), total charge, multiplicity and the *observation*. The
observation is either ``explicit_h`` (hydrogens are atoms with coordinates)
or ``stripped`` (hydrogens are deleted and their counts must be inferred).
The query never contains the reference label.

The prediction carries a :class:`~bond_bench.eval.core.graph.Graph`: bond
orders B, formal charges q, implicit hydrogen counts h and radical electrons
r, index-aligned with the query's atoms. It may also carry co-optimal
alternatives, per-pair bond-order probabilities (``edge_probs``), per-atom
charge and H probabilities (``atom_probs``) and flags. A method that cannot
produce a graph returns a failure status:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - status
     - meaning
   * - ``ok``
     - a graph was returned
   * - ``infeasible``
     - ⊥: no valid structure exists on the method's candidate support
   * - ``unsupported``
     - the method lacks a needed capability (stripped H, charge, element, …);
       decided by the harness before calling the method
   * - ``error``
     - the method raised or produced malformed output
   * - ``timeout``
     - the per-structure (or per-batch, for adapters) time limit was exceeded

Running
-------

.. code-block:: bash

   uv run bond-bench run canonical-O --strata B1 A1 --n 200 --workers 8
   uv run bond-bench run openbabel+project --strata A1          # wrappers are appended with +
   uv run bond-bench run canonical-O --param preset=atom_penalty

``--strata`` defaults to every built stratum. ``--obs auto`` (the default)
follows the manifest's ``hydrogens_default``: ``Stripped`` strata run
H-free, ``Present`` strata run with explicit H, and strata marked for both
run both ways.

With ``--n``, records are sampled by group, so C1 noise ladders and B5
tautomer groups stay whole. ``--param KEY=VALUE`` passes method parameters;
values are parsed as JSON when possible. Non-default parameters are appended
to the results directory name, e.g. ``canonical-O@preset=atom_penalty``.

Validity
--------

Every output is checked by the harness against the valid set 𝒢(z, Q): the
chemical graphs whose element valence states, charges and hydrogen counts
are allowed by the broad Lewis table in
``bond_bench.eval.core.elements`` and whose formal charges sum to Q. Methods
do not self-report validity. Two wrappers turn any method's output into a
valid one:

- ``+project`` maps an invalid output to the nearest valid structure on the
  same support.
- ``+hybrid`` decodes a method's ``edge_probs`` (and ``atom_probs``) as a
  constrained MAP over 𝒢.

Metrics
-------

Each structure gets one row of metrics in a ``.scores.parquet`` file.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - metric
     - meaning
   * - ``valid``
     - output ∈ 𝒢(z, Q) under the broad Lewis table
   * - ``exact``
     - B, q and h equal to the label, index by index
   * - ``resonance_match``
     - valid, with the same support and the same h as the label, i.e. in the
       label's resonance class
   * - ``smiles_match``
     - same RDKit canonical SMILES (resonance-invariant only up to
       aromaticity)
   * - ``connectivity_match``
     - same set of bonded pairs as the label
   * - ``f1_1``, ``f1_2``, ``f1_3``, ``f1_macro``
     - per-pair bond-order F1 over heavy-atom pairs, per class and averaged
   * - ``conn_precision``, ``conn_recall``
     - connectivity precision and recall over heavy-atom pairs
   * - ``taut_accept``, ``taut_pref``
     - B5 only: the prediction is any member of the tautomer group, or the
       preferred member
   * - ``coverage``
     - fraction of structures the method supports

Rates are computed over *supported* structures: ``infeasible``, ``error``
and ``timeout`` count as wrong, while ``unsupported`` structures are
excluded and show up as reduced coverage. ``bond-bench score`` reports
Wilson 95% intervals for rates and a bootstrap interval for F1.

Robustness and equivariance
---------------------------

Robustness to coordinate noise is measured on C1 (reported per ``sigma``
subgroup) or with on-the-fly noise:

.. code-block:: bash

   uv run bond-bench run canonical-R --strata B1 --noise 0.05,0.1 --reps 3

``--equivariance N`` applies a random atom permutation σ and a random
Euclidean transform g (possibly improper) to N structures per stratum and
compares Φ((σ, g)·x) with σ·Φ(x) at three levels:

- ``strict``: the graphs are identical.
- ``set``: the expected graph is the output or one of its alternatives.
- ``resonance``: the graphs are in the same resonance class.

Inputs with exact geometric symmetry are flagged ``symmetric`` and reported
separately, because strict equivariance is impossible for them.

Results layout
--------------

.. code-block:: text

   results/
     <method_tag>/
       <stratum>__<obs>.parquet                      predictions
       <stratum>__<obs>.scores.parquet               per-structure metrics
       <stratum>__<obs>__noise.s<σ>.parquet          scores under on-the-fly noise (--noise)
       <stratum>__<obs>__equivariance.parquet        equivariance checks (--equivariance)
       provenance.json                               method version and parameters, dataset SHA-256, seed, …
     summary.parquet                     written by `bond-bench score`
     plots/                              written by `bond-bench plot`

Rerunning a method replaces its results for the strata it covers.
``bond-bench score --rescore`` recomputes the score files from the stored
predictions, for example after a metric changes.

Plots
-----

``bond-bench plot`` (needs ``uv sync --extra plot``) writes, for each metric
(default: ``smiles_match``, ``valid``, ``f1_macro``):

- ``strata_<metric>.png``: one row per method, sorted by overall score, with
  an overall panel and one panel per stratum, with 95% CIs.
- ``top_<metric>.png``: the 5 best methods plus the canonical methods, ranked
  by the mean over strata of metric × coverage (unsupported structures count
  as failures). Only methods run on every stratum are ranked.
- ``noise_c1_<metric>.png``: the same layout with one panel per σ of the C1
  ladder, plus ``noise_onthefly_*`` when ``--noise`` was used.

and, across all metrics:

- ``overall.png``: every method's overall score on each metric side by side,
  one panel per metric, with its rank on that metric beside each dot. Rows are
  sorted by mean rank over the metrics.
- ``overall_raw.png``: the same, with ``+project`` and ``+hybrid`` methods
  dropped, so base methods are ranked among themselves.
- ``equivariance.png``: violation rates per method at the three levels of
  strictness.
