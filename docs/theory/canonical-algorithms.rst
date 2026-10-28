Canonical algorithms
====================

This page gives one canonical algorithm for each estimator family of
:doc:`formulation`: rule-based (R), global optimization (O), machine learning
(M) and the hybrid (H). Each algorithm uses as few assumptions as possible.
Every choice the formalism does not force is either

(a) pulled out as a named parameter, or
(b) fixed to the option that keeps the three requirements of the
    deterministic statement: equivariance, soundness and correctness.

The four algorithms share one set of primitives, so they differ only where
the family comparison table says they differ.

The source of this page is ``assets/boa_canonical_algorithms.md``. Boxes
marked *In bond-bench* describe how each piece is implemented and where it
departs from or fixes a choice in the specification.

Shared assumptions and primitives
---------------------------------

Assumption ledger
~~~~~~~~~~~~~~~~~

Each algorithm uses only the assumptions it lists from this ledger.

.. list-table::
   :header-rows: 1
   :widths: 6 20 52 22

   * - id
     - assumption
     - object
     - used by
   * - A0
     - element state table
     - :math:`\Lambda(a)\subset\mathbb{Z}\times\mathbb{N}` for each
       :math:`a\in\mathbb{E}`, derived from :math:`\nu(a)`, :math:`o(a)`
     - R, O, M (optional), H
   * - A1
     - contact tolerance
     - :math:`\tau_{ab}` for each element pair
     - R, O
   * - A2
     - local geometric windows
     - :math:`W_{abk}` (distance), :math:`W'` (angle)
     - R
   * - A3
     - prior objective
     - :math:`\log\pi_\theta(G)`, linear in :math:`(B,q,h)`
     - O
   * - A4
     - training measure
     - samples :math:`(G,X)\sim P`
     - M, H
   * - A5
     - canonicalizer
     - :math:`\kappa:\mathcal{G}/\!\sim\ \to\mathcal{G}`
     - all (only when one representative must be reported)
   * - A6
     - candidate cutoff
     - :math:`r_c>\max_{ab}\tau_{ab}`
     - M, H

With :math:`\Lambda` derived from the Lewis conditions, A0 needs only
:math:`\nu(a)` and :math:`o(a)`:

.. math::

   \Lambda(a)=\{(c,d)\in\mathbb{Z}\times\mathbb{N}:\ \nu(a)-c-d\ge 0,\ \
   \nu(a)-c-d\ \text{even},\ \ \nu(a)-c+d\le 2o(a),\ \ |c|\le c_{\max}\},

where :math:`c_{\max}` is a bound on the formal charge per atom, e.g. 1 or 2.

.. admonition:: In bond-bench

   ``bond_bench.eval.core.elements.states(z, table, c_max, r_max)`` builds
   :math:`\Lambda`, with radicals :math:`r` as a third coordinate. Two
   tables are provided:

   - ``lewis``: the conditions above, with expanded shells for period ≥ 3.
     It backs the harness validity oracle.
   - ``octet``: one valence per (charge, radicals), that of the isoelectronic
     octet configuration, plus the hypervalent valences of the xyz2mol table
     (P(V), S(VI), …). It excludes carbenes, nitrenes and the spurious
     S(IV)/Cl(III) states that a bond-maximizing prior would otherwise pick.
     The canonical methods use it by default (``--param table=octet``,
     ``c_max=1``).

   A1 is :math:`\tau_{ab}=1.3\,(r_a+r_b)` with covalent radii :math:`r`, as in
   RDKit ``DetermineConnectivity`` (``--param contact_scale``). A6 is
   :math:`r_c(a,b)=1.5\,(r_a+r_b)`.

Primitives
~~~~~~~~~~

**P1. Invariant featurization.** The only input any algorithm reads is
:math:`(z,\Delta(X),Q)`, plus angles computed from :math:`\Delta` by the law
of cosines. By construction this makes every step E(3)-invariant.

**P2. Contact map.**
:math:`A(X)_{ij}=\mathbb{1}[\Delta_{ij}\le\tau_{z_iz_j}^2]`. Use a cell list
so the cost is :math:`O(n)`.

**P3. Validity oracle.** :math:`\mathrm{valid}(G)` returns true iff
:math:`B=B^\top`, :math:`\mathrm{diag}B=0`,
:math:`(q_i,(B\mathbf 1+h)_i)\in\Lambda(z_i)` for all :math:`i`, and
:math:`\mathbf 1^\top q=Q`. Cost is :math:`O(n+|\mathrm{supp}B|)`.

**P4. Constrained linear decoder (the core ILP).** Given an edge set
:math:`E\subseteq\binom{[n]}{2}`, edge scores :math:`w_{ij}(k)` for
:math:`k\in\{0,1,2,3\}` and atom scores :math:`u_i(c,d,\eta)`, solve

.. math::

   \max\ \sum_{(i,j)\in E}\sum_{k}w_{ij}(k)\,y_{ijk}+\sum_i\sum_{(c,d,\eta)}u_i(c,d,\eta)\,s_{i,c,d,\eta}

subject to

.. math::

   \begin{aligned}
   &\textstyle\sum_k y_{ijk}=1 &&\forall (i,j)\in E &&\text{(one order per pair)}\\
   &\textstyle\sum_{(c,d,\eta)} s_{i,c,d,\eta}=1 &&\forall i &&\text{(one state per atom)}\\
   &\textstyle\sum_{j:(i,j)\in E}\sum_k k\,y_{ijk}+\sum \eta\, s_{i,c,d,\eta}=\sum d\,s_{i,c,d,\eta} &&\forall i &&(D_i=B\mathbf1+h)\\
   &\textstyle\sum_i\sum c\,s_{i,c,d,\eta}=Q &&&&\text{(total charge)}\\
   &y,s\in\{0,1\},
   \end{aligned}

where :math:`(c,d)\in\Lambda(z_i)` and
:math:`\eta\in\{0,\dots,h_{\max}(z_i)\}`, with :math:`\eta\equiv 0` when
hydrogens are observed. Read off :math:`B_{ij}=\sum_k k\,y_{ijk}`,
:math:`q_i=\sum c\,s`, :math:`h_i=\sum\eta\,s`. Pairs outside :math:`E` have
:math:`B_{ij}=0`. Any feasible point lies in :math:`\mathcal G(z,Q)`, so P4
searches exactly :math:`\mathcal G\cap\{\mathrm{supp}B\subseteq E\}`.
Because the objective is linear in one-hot variables, every factorized score
(a sum of per-edge and per-atom terms) is optimized exactly.

- *Infeasible case:* return ⊥ with a certificate (the solver's infeasibility
  proof). Do not relax silently; see the feasibility gate below.
- *Solution pool:* collect all optima, or the top :math:`K`, by adding no-good
  cuts :math:`\sum_{(ijk):y^\star_{ijk}=1}y_{ijk}\le|E|-1` and re-solving.
  This yields set-valued and probabilistic outputs.

**P5. Equivariant tie-breaking.** Tie-breaking by atom index breaks
:math:`S_n`-equivariance. Instead:

1. Compute a canonical atom ranking :math:`\rho` from :math:`(z,\Delta)`,
   either by canonical labeling of the weighted complete graph or by
   Weisfeiler–Lehman refinement on :math:`(z,A,\text{binned }\Delta)` with a
   canonical-labeling fallback for residual ties.
2. Break ties lexicographically in the order :math:`\rho`.

If :math:`X` has a nontrivial symmetry, the co-optimal set is a union of
symmetry orbits. In that case return the whole set, or apply :math:`\kappa`,
which is defined on isomorphism classes and is therefore equivariant.

**P6. Canonicalization.** :math:`\kappa` maps any :math:`G` to the
representative of its class :math:`[G]_\sim`. A minimal-assumption choice:

1. Enumerate the class: resonance moves are alternating-path / π-shift
   transformations that leave :math:`\mathrm{supp}B` and :math:`h` unchanged
   and stay inside :math:`\mathcal G`. Implement by repeated P4 calls with
   :math:`E=\mathrm{supp}B`, fixed :math:`h`, and no-good cuts.
2. Pick the member that minimizes
   :math:`\bigl(\sum_i|q_i|,\ \text{canonical string under }\rho\bigr)`
   lexicographically.
3. Optionally rewrite the result in aromatic notation.

.. admonition:: In bond-bench

   .. list-table::
      :widths: 12 88

      * - P1, P2
        - ``bond_bench.eval.core.geometry``: ``distance_matrix``,
          ``contact_map`` (a k-d tree in place of a cell list; it also
          returns each contact's margin :math:`|d-\tau|`) and
          ``candidate_pairs``.
      * - P3
        - ``bond_bench.eval.core.validity.check``, which returns the reason a
          structure is invalid, or ``None``.
      * - P4
        - ``bond_bench.eval.core.ilp.decode``, solved with OR-Tools CP-SAT.
          Objectives are a lexicographic list; each stage is solved with the
          earlier optima fixed. Float scores are scaled by
          :math:`10^4` and rounded, because CP-SAT needs integer
          coefficients. ``diagnose`` names the cause of ⊥.
      * - P5
        - ``bond_bench.eval.core.canon``: WL colors on :math:`(z,` binned
          :math:`\Delta)` with a 0.001 Å bin, so only exactly symmetric atoms
          share a color. ``pick`` chooses among co-optimal structures by
          :math:`\sum|q|`, then the colored structure key, then the
          E(3)-invariant score :math:`\sum k_{ij}d_{ij}` (multiple bonds on
          shorter contacts). When the pool is too large to enumerate, the
          same rules run as extra ILP stages (``tiebreak_objectives``).
      * - P6
        - ``resonance_class`` and ``canonicalize`` in the same module.

Common pre- and post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   PREPROCESS(z, X, Q, obs):
     Δ ← squared-distance matrix of X                  # P1
     if obs = periodic: Δ ← minimum-image distances on R^3/L
     hfree ← (obs = hydrogens missing)                 # turns on η-variables in P4
     return z, Δ, Q, hfree

   FEASIBILITY_GATE(z, Q, E):
     if P4 with zero objective on E is infeasible:
         return ⊥, reason ∈ {charge Q inconsistent, Λ too small, E too small}

The gate makes soundness testable. If
:math:`\mathcal G(z,Q)\cap\{\mathrm{supp}B\subseteq E\}=\emptyset`, every
algorithm must report ⊥ rather than return an invalid structure. Optional
fallbacks, each declared in the output: widen :math:`E` (raise :math:`\tau`),
widen :math:`c_{\max}`, or treat :math:`Q` as unknown by dropping the charge
constraint and reporting the inferred :math:`Q`.

.. admonition:: In bond-bench

   ⊥ is the ``infeasible`` status. No fallback is applied automatically; the
   reason is reported in the prediction's ``reason`` field.

R. Rule-based estimator
-----------------------

R partitions configuration space explicitly into semi-algebraic decision
regions.

**Assumptions:** A0, A1, A2 (A5 if one representative must be reported).
**Search set:** :math:`\mathcal M`, followed by projection onto
:math:`\mathcal G`.

.. code-block:: text

   ALGORITHM R(z, X, Q):
     z, Δ, Q, hfree ← PREPROCESS(z, X, Q, obs)
     E ← supp A(X)                                         # P2

     # 1. Local window tests → candidate sets (each test is a polynomial inequality in X)
     for (i,j) ∈ E:
         C_ij ← { k ∈ {1,2,3} : Δ_ij ∈ W_{z_i z_j k} }
         if C_ij = ∅: C_ij ← { argmin_k dist(Δ_ij, W_{z_i z_j k}) }   # nearest window
     for i ∈ [n]:
         H_i ← hybridization class from deg_E(i) and angles ∠_{jik}, j,k ∈ N_E(i), tested against W'
         prune each C_ij that conflicts with H_i (e.g. sp3 ⇒ C_ij ← C_ij ∩ {1})

     # 2. Greedy cascade (deterministic, equivariant order)
     order ← edges sorted by (|C_ij|, margin to nearest competing window, ρ)   # most certain first
     B ← 0
     for (i,j) in order:
         B_ij ← max{ k ∈ C_ij : residual valence of i and j still admits k under Λ }
         update residual valences of i and j

     # 3. Projection M → G (replaces ad hoc valence repair)
     if not valid(B, q, h):
         G ← P4(E, w_ij(k) = −|k − B_ij|, u_i = −|c|)       # nearest valid structure in Hamming distance
     return P5/P6(G), plus a validity flag (true after step 3, or ⊥ if the gate fails)

**Properties.**

- *Decision regions:* every branch is a finite Boolean combination of
  polynomial inequalities in :math:`X`, so each :math:`R_G` is
  semi-algebraic. Boundaries sit at the hand-set window edges, not at
  posterior equal-probability contours.
- *Margin:* computable in closed form. It is the minimum over tested pairs of
  the distance from :math:`\Delta_{ij}` to the nearest window edge that would
  change the output. Report it as a per-bond confidence.
- *Minimal choices made here:* candidate *sets* in place of hard single
  assignments; cascade order determined by certainty plus :math:`\rho`, which
  keeps the cascade equivariant; repair defined as the Hamming-nearest point
  of :math:`\mathcal G`, the unique canonical projection, instead of a
  hand-tuned repair.
- *Cost:* :math:`O(n)` for tests and cascade, plus an ILP that is small in
  practice because only the edges with conflicts are free.

.. admonition:: In bond-bench: ``canonical-R``

   ``bond_bench.eval.methods.rule.CanonicalR``. The windows are centered on
   MMFF94 reference bond lengths (``eval/tables/bond_windows_mmff.csv``,
   generated by ``scripts/mmff_windows.py``), so no benchmark stratum is used
   to fit them. ``--param windows=b2`` selects windows fitted on B2
   (``scripts/fit_windows.py``), or pass a CSV path. An aromatic window
   (:math:`k=1.5`) survives to the projection step, which resolves it into a
   Kekulé form. The projection is the same routine as the ``+project``
   wrapper (``bond_bench.eval.methods.wrappers.project_to_valid``). The
   per-bond window margin is returned as ``margin``.

O. Global-optimization estimator
--------------------------------

O coarse-grains the geometry to a contact map, then selects a structure.

**Assumptions:** A0, A1, A3 (A5 for a single representative).
**Search set:** :math:`\mathcal G\cap\{\mathrm{supp}B=A(X)\}`, exactly.

.. code-block:: text

   ALGORITHM O(z, X, Q):
     z, Δ, Q, hfree ← PREPROCESS(z, X, Q, obs)
     E ← supp A(X)                                         # the only use of geometry
     FEASIBILITY_GATE(z, Q, E)

     # Fiber constraint supp B = E: forbid order 0 on contacts
     w_ij(0) ← −∞  for (i,j) ∈ E

     # Prior objective (A3), written as a lexicographic list of linear objectives f_1 ≻ f_2 ≻ …
     #   Kim 2015:   f_1 = Σ B_ij  (maximal bonding),   f_2 = −Σ |q_i|
     #   Dehof 2011: f_1 = −Σ AP(z_i, D_i)              (atom penalties via u_i)
     for ℓ = 1..L:
         G*_ℓ, v_ℓ ← P4(E, objective f_ℓ, extra constraints f_1 = v_1, …, f_{ℓ−1} = v_{ℓ−1})
     Opt ← solution pool of the last stage                  # all co-optimal structures

     return  set-valued:     Opt
             point:          P6(argmin_ρ Opt)              # equivariant tie-break
             diagnostic:     |Opt| > 1 ⇒ flag "prior-tied" (resonance or tautomer ambiguity)

**Properties.**

- *Factorization:* :math:`\Phi=\mathrm{select}\circ A`. The output is
  constant on each fiber :math:`A^{-1}(\text{graph})`, and validity holds by
  construction.
- *Exactness:* P4 solves the stated objective exactly. With fixed per-atom
  states, the problem reduces to a degree-constrained subgraph (b-matching)
  on the π-bond surplus :math:`d_i-\deg_E(i)`, which is the Labute-type
  special case and is solvable in polynomial time. A general ILP solver
  handles the enumeration over states.
- *Minimal choices made here:* lexicographic objectives rather than
  hand-weighted sums, so no relative weights need to be assumed; the full
  co-optimal set is returned, so tie-breaking is explicit and not left to
  solver order.
- *Margin:* the distance to the nearest contact-threshold crossing,
  :math:`\min_{ij}\bigl|\sqrt{\Delta_{ij}}-\tau_{z_iz_j}\bigr|`. Report it.
  Any error in :math:`E` passes directly into the output, so this number is
  the method's whole robustness profile.
- *Optional robustness extension (no new assumptions):* if some
  :math:`\sqrt{\Delta_{ij}}` lies within :math:`\delta` of :math:`\tau`,
  re-solve with that pair toggled and return the union of results with a
  flag.

.. admonition:: In bond-bench: ``canonical-O``

   ``bond_bench.eval.methods.optimization.CanonicalO``. Presets
   (``--param preset=...``):

   - ``kim2015`` (default): maximize :math:`\sum B_{ij}`, then minimize
     :math:`\sum|q_i|`.
   - ``atom_penalty``: minimize the Wang et al. (2006) atomic penalty scores
     in ``eval/tables/atom_penalties.csv``, then :math:`\sum|q_i|`. Molecules
     with elements outside the table are reported as unsupported.

   The co-optimal pool (up to ``max_pool=64``) is returned as
   ``alternatives``, the ``prior_tied`` flag is set when it has more than one
   member, and the per-bond contact margin is returned as ``margin``. The
   robustness extension is not implemented.

M. Machine-learning estimator
-----------------------------

M estimates the posterior discriminatively.

**Assumptions:** A4, A6. A0 is used only to compute the validity flag.
**Search set:** :math:`\mathcal M`; the decoder takes an entrywise argmax.

.. code-block:: text

   TRAIN M(dataset D = {(z, X, Q, G*)} ~ P):
     for each sample:
         Δ ← P1(X);  E ← { (i,j) : Δ_ij ≤ r_c² }           # candidate pairs, superset of true bonds
         optional augmentation: X ← X + ε, ε ~ N(0, σ_aug² I)   # targets robustness at chosen σ
         optional: delete H points and move their count into h targets   # trains the H-free regime
     model f_θ: (z, Δ|_E, Q) ↦ edge logits ℓ_ij ∈ R^4 for (i,j) ∈ E,
                               atom logits  m_i ∈ R^{|states|}
         requirement: invariant to E(3) (consumes only Δ and angles) and equivariant to S_n
                      (message passing over E)
         (two-stage variant: ℓ_ij = [logit P(B_ij > 0)] ⊕ [logits P(B_ij = k | B_ij > 0)])
     loss(θ) = − Σ_{(i,j)∈E} log softmax(ℓ_ij)[B*_ij] − Σ_i log softmax(m_i)[(q*_i, h*_i)]
         resonance-aware variant (targets G/∼):  loss = min_{G' ∈ [G*]∼} NLL(G'),
             with [G*]∼ enumerated once per training molecule by P6
     θ* ← argmin_θ E_D[loss]                                # cross-entropy ≡ expected KL

   INFER M(z, X, Q):
     z, Δ, Q, hfree ← PREPROCESS(z, X, Q, obs);  E ← candidate pairs (r_c)
     ℓ, m ← f_θ*(z, Δ|_E, Q)
     B_ij ← argmax_k ℓ_ij(k);  (q_i, h_i) ← argmax m_i      # Hamming-loss Bayes estimator over M
     return G, per-entry probabilities, valid(G)           # validity reported, not enforced

**Properties.**

- *Bayes target:* the entrywise posterior mode, which is optimal for per-bond
  F1 and can lie outside :math:`\mathcal G`.
- *Minimal choices made here:* a candidate graph defined by a generous cutoff
  :math:`r_c`, so connectivity is learned rather than fixed by :math:`\tau`
  and avoids the contact-map error of O; invariance obtained from
  :math:`\Delta` alone, so no architecture choice is forced; resonance-aware
  loss as an option. The resonance-aware loss removes the probability
  splitting caused by labeling conventions, and with it the main way
  entrywise decoding mixes resonance forms.
- *Not controlled:* behavior outside the support of :math:`P`, e.g. metals
  or exotic valences. Report the out-of-distribution status of a query by
  checking whether its elements and :math:`\Lambda`-states appear in the
  training set.
- *Margin:* no closed form. Estimate it empirically (see the evaluation
  protocol below) or by gradient-based adversarial search on :math:`X`.

.. admonition:: In bond-bench: ``canonical-M``

   ``bond_bench.eval.methods.learned.CanonicalM``, with the network, targets
   and training loop in ``bond_bench.eval.ml`` (``model.py``, ``labels.py``,
   ``train.py``). :math:`f_\theta` is a PaiNN network. It is trained by
   ``scripts/train_canonical_m.py`` on B2 with the resonance-aware loss: the
   minimum NLL over the label's Kekulé and charge-placement alternatives.
   ``--sigma-aug`` turns on noise augmentation. When hydrogens are observed,
   :math:`h_i` is fixed to 0 and only :math:`q_i` is decoded. The
   ``ood_elements`` and ``ood_states`` flags report the out-of-distribution
   status, and ``extra.in_training_set`` marks B2 structures seen in
   training. The prediction carries ``edge_probs`` and ``atom_probs``, so
   ``+hybrid`` can decode it.

H. Hybrid estimator
-------------------

H computes the constrained MAP under a learned likelihood.

**Assumptions:** A0, A4, A6 (A5 for a single representative). This is the
union of the assumptions of O and M, minus A1 and A3, which the learned
scores replace.
**Search set:** :math:`\mathcal G\cap\{\mathrm{supp}B\subseteq E\}`, exactly.

.. code-block:: text

   ALGORITHM H(z, X, Q):
     z, Δ, Q, hfree ← PREPROCESS(z, X, Q, obs);  E ← candidate pairs (r_c)
     ℓ, m ← f_θ*(z, Δ|_E, Q)                                # model trained as in M
     FEASIBILITY_GATE(z, Q, E)

     # Factorized log-posterior is linear in the one-hot variables → P4 is exact
     w_ij(k)     ← log softmax(ℓ_ij)[k]          for k ∈ {0,1,2,3}   # order 0 allowed: connectivity is decided jointly
     u_i(c,d,η)  ← log softmax(m_i)[(c, η)]      for (c,d) ∈ Λ(z_i)  # d is fixed by the valence constraint
     G* , Pool_K ← P4(E, w, u) with top-K solution pool

     return  point:          P6(G*)
             set-valued:     { G ∈ Pool_K : score(G*) − score(G) ≤ log(1/α) }   # structures within likelihood ratio α
             probabilistic:  K̂(G) ∝ exp(score(G)) over Pool_K                   # renormalized on G
             diagnostics:    gap score(G*) − score(G_2);  Hamming(G*, entrywise argmax of M)

**Properties.**

- *Bayes target:* MAP over :math:`\mathcal G` under :math:`q_\theta`, which
  is the 0–1-loss estimator. Soundness and the use of bond lengths both hold.
- *Exactness:* with a factorized :math:`q_\theta`, the objective is a sum of
  per-edge and per-atom log-probabilities, so P4 returns the global
  constrained MAP. A
  non-factorized :math:`q_\theta`, e.g. with pairwise edge–edge terms, makes
  the objective quadratic. In that case either linearize the products
  (standard ILP lifting) or fall back to beam search, and declare the
  approximation.
- *Resonance handling:* score the equivalence class instead of the
  representative, :math:`\log\sum_{G'\in[G]_\sim}q_\theta(G')`. The exact
  version enumerates :math:`[G]_\sim` for the pool members (P6) and re-ranks
  by summed probability. The cheap version re-ranks the pool after applying
  :math:`\kappa`, adding up the probabilities of members that map to the same
  representative.
- *Reduction checks:* H specializes to each of the other three.

  - with :math:`w_{ij}(0)=-\infty` on :math:`A(X)` and :math:`w,u` set to
    :math:`\log\pi_\theta`, H becomes O;
  - with the constraints dropped, H becomes M;
  - with :math:`w` set to window indicators, H becomes R followed by
    projection.
- *Gap diagnostic:* the score gap between the best and second-best valid
  structure is the hybrid's counterpart of a margin, and it is cheap to
  compute from the pool. A small gap marks inputs where the answer is decided
  by the prior or by convention.

.. admonition:: In bond-bench: ``+hybrid``

   ``bond_bench.eval.methods.hybrid.Hybrid`` is a wrapper, so H applies to
   any method that reports ``edge_pairs``/``edge_probs``.
   ``canonical-M+hybrid`` is canonical H; ``mambax+hybrid``,
   ``yuelbond+hybrid`` and ``bondnet+hybrid`` decode the published models the
   same way. The atom scores are :math:`\log q(q_i=c, h_i=\eta)` when the
   inner method reports ``atom_probs``; otherwise they are 0, or
   :math:`-\lambda|c|` with ``--param charge_penalty=λ``. Defaults:
   ``top_k=16``, ``alpha=0.01``. Without learned atom scores, the decoder
   can return charge-separated aromatic rings (see ``adapters/README.md``).

Common evaluation protocol
--------------------------

.. code-block:: text

   EVALUATE(Φ, test set T, κ, σ-grid):
     for (z, X, Q, G*) ∈ T:
         G ← Φ(z, X, Q)
         record: valid(G); [G = G*]; [κ(G) = κ(G*)]; per-bond confusion of B vs B*
         for σ in σ-grid, r = 1..R:
             record [κ(Φ(X + ε_r)) = κ(G*)],  ε_r ~ N(0, σ² I)
         check equivariance: Φ(Pz, gPX, Q) = P·Φ(z, X, Q) for random P ∈ S_n, g ∈ E(3)
     report: validity rate, exact match, resonance-aware exact match, per-bond F1,
             robustness curve σ ↦ accuracy (its decay locates the margin distribution),
             equivariance violation rate (must be 0),  ⊥ rate with reasons,
             stratified by observation operator O and by in/out-of-distribution status

.. admonition:: In bond-bench

   This is what ``bond-bench run`` and ``bond-bench score`` do; see
   :doc:`../evaluation`. The σ-grid is the C1 ladder or ``--noise`` with
   ``--reps``, and the equivariance check is ``--equivariance N``.

Summary
-------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 16 21 21 21 21

   * -
     - R
     - O
     - M
     - H
   * - assumptions
     - A0 A1 A2
     - A0 A1 A3
     - A4 A6
     - A0 A4 A6
   * - geometry used
     - windows on :math:`\Delta`, angles
     - :math:`A(X)` only
     - full :math:`\Delta` (learned)
     - full :math:`\Delta` (learned)
   * - edge set
     - :math:`A(X)`
     - :math:`A(X)` (fixed support)
     - cutoff :math:`r_c`
     - cutoff :math:`r_c` (support decided jointly)
   * - decoder
     - greedy cascade + P4 projection
     - lexicographic P4
     - entrywise argmax
     - P4 with learned scores
   * - validity
     - by projection
     - by construction
     - reported only
     - by construction
   * - native output
     - point
     - set (co-optimal)
     - Markov kernel on :math:`\mathcal M`
     - point, set, or kernel on :math:`\mathcal G`
   * - margin / confidence
     - closed form (window edges)
     - closed form (:math:`\tau` crossings)
     - empirical
     - score gap from the solution pool
   * - equivariance
     - via :math:`\rho` in the cascade order
     - via P5
     - by architecture
     - by architecture + P5
   * - bond-bench method
     - ``canonical-R``
     - ``canonical-O``
     - ``canonical-M``
     - ``canonical-M+hybrid``
