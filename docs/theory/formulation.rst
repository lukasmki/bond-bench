Problem statement
========================

This page states bond order assignment (BOA) in general mathematical terms.
The input is a finite cloud of labeled points in :math:`\mathbb{R}^3`,
considered up to rigid motion and relabeling. The output is an integer
bond-order matrix that satisfies valence constraints. Solving the problem
means constructing an equivariant map between these two spaces, which amounts
to inverting the process by which a molecular graph produces a geometry.
Rule-based, optimization and machine-learning methods are three ways of
building such a map.

The source of this page is ``assets/bond_order_formalization.md``. Boxes
marked *In bond-bench* point to the code that implements each object.

.. _formulation-input:

Input space
-----------

**Labeled point clouds.** Let :math:`\mathbb{E}` be a finite alphabet of
elements. A molecule with :math:`n` atoms is presented as

.. math::

   (z,X)\in\mathbb{E}^n\times\mathrm{Conf}_n(\mathbb{R}^3),\qquad
   \mathrm{Conf}_n(\mathbb{R}^3)=\{X=(x_1,\dots,x_n)\in(\mathbb{R}^3)^n : x_i\neq x_j\ \text{for } i\neq j\},

together with scalar side information :math:`Q\in\mathbb{Z}` (total charge)
and, optionally, :math:`m\in\mathbb{N}` (number of unpaired electrons).

**Symmetry.** The group :math:`\Gamma=S_n\times E(3)` acts on inputs by

.. math::

   (\sigma,g)\cdot(z,X)=\bigl(z\circ\sigma^{-1},\;(g\,x_{\sigma^{-1}(1)},\dots,g\,x_{\sigma^{-1}(n)})\bigr).

The physically meaningful input is therefore the orbit
:math:`[z,X]\in\mathcal{X}_n=(\mathbb{E}^n\times\mathrm{Conf}_n)/\Gamma`.

**Reduction to distances.** Bond orders do not change under reflection, so
the full group :math:`E(3)`, including improper motions, applies. A
configuration is determined up to :math:`E(3)` by its Euclidean distance
matrix

.. math::

   \Delta(X)_{ij}=\lVert x_i-x_j\rVert^2,\qquad \Delta(X)\in\mathrm{EDM}_n^{(3)},

where :math:`\mathrm{EDM}_n^{(3)}` is the set of Euclidean distance matrices
of embedding dimension at most 3. Every admissible solution can therefore be
written as a function of :math:`(z,\Delta(X))`, and the input can be
identified with :math:`(\mathbb{E}^n\times\mathrm{EDM}_n^{(3)})/S_n`. A
permutation :math:`P` acts on it by :math:`\Delta\mapsto P\Delta P^{\top}`.

**Partial observation.** The other input conditions are observation
operators applied to the full cloud :math:`X`:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - condition
     - observation :math:`\mathcal{O}(X)`
   * - full 3D with H
     - :math:`X`
   * - hydrogens missing (PDB ligands)
     - :math:`\pi_{\mathrm{H}}(X)`, the projection that deletes the H points
   * - noisy geometry (MD, generative models)
     - :math:`X+\varepsilon` with :math:`\varepsilon\sim\mathcal{N}(0,\sigma^2 I)`
   * - connectivity only
     - :math:`A(X)`, the contact graph (see :ref:`formulation-optimization`)
   * - periodic crystal
     - :math:`X` modulo a lattice :math:`L`, i.e. points on the torus :math:`\mathbb{R}^3/L`

.. admonition:: In bond-bench

   A :class:`~bond_bench.eval.query.Query` is :math:`(z, X, Q, m)` plus the
   observation operator: ``explicit_h`` is :math:`X` and ``stripped`` is
   :math:`\pi_{\mathrm{H}}(X)`. Noisy geometry is stratum C1 or ``--noise``.
   ``--equivariance`` tests the :math:`\Gamma`-action with
   :meth:`Query.transformed <bond_bench.eval.query.Query.transformed>`.

.. _formulation-output:

Output space
------------

**Lewis structures as integer matrices.** On the same index set
:math:`[n]`, a chemical graph is a triple :math:`G=(B,q,h)`:

- :math:`B\in\{0,1,2,3\}^{n\times n}`, symmetric with zero diagonal.
  :math:`B_{ij}` is the bond order between atoms :math:`i` and :math:`j`,
  and :math:`B_{ij}=0` means no bond. The support :math:`\mathrm{supp}(B)` is
  the connectivity.
- :math:`q\in\mathbb{Z}^n`, formal charges.
- :math:`h\in\mathbb{N}^n`, implicit hydrogen counts. :math:`h=0` when
  hydrogens are among the points.

Unpaired electrons :math:`r\in\mathbb{N}^n` can be added in the same way.

**Validity.** Let :math:`D=B\mathbf{1}+h` be the vector of bond valences.
The **valid set** is

.. math::

   \mathcal{G}(z,Q)=\Bigl\{(B,q,h):\ B=B^{\top},\ \mathrm{diag}(B)=0,\
   \bigl(q_i,D_i\bigr)\in\Lambda(z_i)\ \forall i,\ \mathbf{1}^{\top}q=Q\Bigr\}.

Here :math:`\Lambda(a)\subset\mathbb{Z}\times\mathbb{N}` is the finite table
of admissible (charge, valence) states of element :math:`a`; for example,
:math:`\Lambda(\mathrm{N})\ni(0,3),(+1,4),(-1,2)`. For main-group elements
the table follows from the Lewis conditions: the number of non-bonding
electrons :math:`\nu(a)-q_i-D_i` is non-negative and even, and the shell
count :math:`\nu(a)-q_i+D_i` is at most :math:`2o(a)`, where :math:`\nu(a)`
is the valence-electron count and :math:`o(a)` the octet capacity.

:math:`\mathcal{G}(z,Q)` is therefore a finite set of lattice points cut
out by linear equalities and membership constraints, i.e. the integer points
of a union of polytopes. :math:`S_n` acts on it by
:math:`(B,q,h)\mapsto(PBP^{\top},Pq,Ph)`.

**Non-identifiable structures.** Two valid structures :math:`G\neq G'` can
correspond to the same physical state, for example the Kekulé forms of
benzene or the two resonance forms of a carboxylate (made precise in
:ref:`formulation-statistical`). The well-posed output space is a quotient
:math:`\mathcal{G}/\!\sim`, and a canonicalizer
:math:`\kappa:\mathcal{G}/\!\sim\ \to\mathcal{G}` picks one representative per
class (for example the form with minimal :math:`\sum|q_i|`, written with
aromatic notation).

.. admonition:: In bond-bench

   :math:`G=(B,q,h,r)` is :class:`~bond_bench.eval.core.graph.Graph`, stored
   as an edge list of the nonzero entries of :math:`B`. :math:`\Lambda` is
   ``bond_bench.eval.core.elements.states``. The harness judges validity
   (the ``valid`` metric) with ``bond_bench.eval.core.validity.check`` under
   the broad ``lewis`` table (expanded shells for period ≥ 3,
   :math:`c_{\max}=3`). The resonance class :math:`[G]_\sim` is computed by
   ``bond_bench.eval.core.canon.resonance_class``.

The problem
-----------

Deterministic statement
~~~~~~~~~~~~~~~~~~~~~~~

Find a map

.. math::

   \Phi:\ \mathbb{E}^n\times\mathrm{Conf}_n(\mathbb{R}^3)\times\mathbb{Z}\longrightarrow\mathcal{G}

that satisfies

1. **Equivariance.**
   :math:`\Phi\bigl((\sigma,g)\cdot(z,X),Q\bigr)=\sigma\cdot\Phi(z,X,Q)`.
   Equivalently, :math:`\Phi` descends to a map
   :math:`\mathcal{X}_n\to\mathcal{G}/S_n` from point-cloud orbits to
   isomorphism classes of chemical graphs.
2. **Soundness.** :math:`\Phi(z,X,Q)\in\mathcal{G}(z,Q)` whenever
   :math:`\mathcal{G}(z,Q)\neq\emptyset`.
3. **Correctness.** :math:`\Phi(z,X,Q)\sim G^{*}`, where :math:`G^{*}` is the
   graph that produced :math:`X`.

The input space is a continuous manifold and the output space is finite. Any
such :math:`\Phi` is therefore piecewise constant: it partitions
configuration space into **decision regions**

.. math::

   R_G=\Phi^{-1}(G),\qquad \mathrm{Conf}_n=\bigsqcup_{G}R_G,

and it must be discontinuous on the region boundaries :math:`\partial R_G`.
The robustness of a method at :math:`X` is the **margin**
:math:`\operatorname{dist}(X,\partial R_{\Phi(X)})`: the smallest
perturbation of the coordinates that changes the output.

.. _formulation-statistical:

Statistical statement: an inverse problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Forward model.** A chemical graph :math:`G` does not fix a geometry; it
induces a probability measure on configurations, :math:`X\sim p(X\mid G)`.
This measure is the Boltzmann distribution of conformers on the potential
energy surface of :math:`G`, convolved with whatever measurement or model
noise is present (crystallographic error, MD fluctuations, generative-model
error). A data source corresponds to a joint measure
:math:`P(G,X)=\pi(G)\,p(X\mid G)`, where :math:`\pi` is the prior over
chemistry, e.g. drug-like molecules for GEOM or PDB ligands for
crystallographic sets.

**BOA is the inverse problem.** Given :math:`X`, recover :math:`G`. With a
loss :math:`L:\mathcal{G}\times\mathcal{G}\to\mathbb{R}_{\ge0}`, the
Bayes-optimal assignment is

.. math::

   \Phi^{*}(X)=\arg\min_{G\in\mathcal{G}(z,Q)}\ \mathbb{E}_{G'\sim P(\cdot\mid X)}\bigl[L(G,G')\bigr],
   \qquad P(G\mid X)\propto p(X\mid G)\,\pi(G).

Which loss is used determines what the estimate is:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - loss
     - Bayes estimator
     - remark
   * - 0–1 loss on :math:`\mathcal{G}`
     - :math:`\arg\max_{G\in\mathcal{G}}p(X\mid G)\,\pi(G)` (MAP)
     - exact-match criterion; valid by construction
   * - Hamming loss on entries of :math:`B`
     - entrywise posterior mode :math:`\hat B_{ij}=\arg\max_k P(B_{ij}=k\mid X)`
     - per-bond F1 criterion; may lie outside :math:`\mathcal{G}`
   * - 0–1 loss on :math:`\mathcal{G}/\!\sim`
     - MAP over equivalence classes
     - resonance-aware exact match

**Identifiability.** Define :math:`G\sim G'` if
:math:`p(\cdot\mid G)=p(\cdot\mid G')`, meaning no geometry can tell them
apart. Resonance forms satisfy this by definition: they are alternative
descriptions of a single electronic state with a single geometry. Their
posterior ratio is then fixed by the prior,
:math:`P(G\mid X)/P(G'\mid X)=\pi(G)/\pi(G')`, whatever :math:`X` is.
Recovery is possible only on the quotient :math:`\mathcal{G}/\!\sim`. Within
a class, the "correct" answer is a matter of convention and is fixed by
:math:`\kappa`, not by the data. Tautomers are different: they are distinct
physical states with different :math:`p(X\mid G)`, so they are identifiable
in principle. In practice their likelihoods are close, and without hydrogens
they are only weakly separated.

**Partial observation.** If only :math:`Y=\mathcal{O}(X)` is observed, the
likelihood becomes the pushforward
:math:`p(Y\mid G)=\int\mathbb{1}[\mathcal{O}(X)=Y]\,p(X\mid G)\,dX`. Without
hydrogens this means integrating over the H positions, which flattens the
likelihood and makes the prior more important. With noise it is a
convolution. With connectivity only, the likelihood is constant on the fibers
of the contact map.

Unified form of a solution
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every method computes

.. math::

   \Phi(X)=\operatorname*{arg\,max}_{G\in\mathcal{S}}\ S_\theta(G;z,X)

with

- a **score** :math:`S_\theta` that stands in for
  :math:`\log p(X\mid G)+\log\pi(G)`,
- a **search set** :math:`\mathcal{S}` that is either :math:`\mathcal{G}`
  (valid only) or the unconstrained matrix space
  :math:`\mathcal{M}=\{0,1,2,3\}^{n\times n}_{\mathrm{sym}}\times\mathbb{Z}^n\times\mathbb{N}^n`,
  and
- an **optimizer** that is exact or approximate.

The families differ in which part of the posterior they model and how.

The three families as estimators
--------------------------------

Rule-based: explicit semi-algebraic partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Likelihood model.** A product of local factors with hard windows,

  .. math::

     p(X\mid G)\;\approx\;\prod_{i<j}\mathbb{1}\bigl[\Delta_{ij}\in W_{z_iz_jB_{ij}}\bigr]\
     \prod_{(i,j,k)}\mathbb{1}\bigl[\angle_{ijk}\in W'_{\dots}\bigr]\cdots,

  with windows tabulated from CSD/PDB statistics. The prior :math:`\pi` is
  ignored, apart from functional-group templates.
- **Optimizer.** A sequential greedy cascade over atoms and groups, followed
  optionally by a valence repair that maps :math:`\mathcal{M}\to\mathcal{G}`.
- **Geometry of the decision regions.** Each window test is a polynomial
  inequality in the coordinates, because
  :math:`\Delta_{ij}=\lVert x_i-x_j\rVert^2` is quadratic and cosines of
  angles are rational in the coordinates. The regions :math:`R_G` are
  therefore explicit semi-algebraic sets, with boundaries fixed by
  hand-set thresholds rather than by the posterior.
- **Consequence.** Margins are small exactly where windows for different bond
  orders nearly touch, as for C–C single/double or C–O single/double bonds.
  Noise of comparable size moves :math:`X` across a boundary, which makes
  these methods brittle on noisy input.

.. _formulation-optimization:

Global optimization: coarse-grain the geometry, then choose a structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Likelihood model.** The geometry enters only through the **contact map**

  .. math::

     A:\mathrm{Conf}_n\to\{0,1\}^{n\times n},\qquad
     A(X)_{ij}=\mathbb{1}\bigl[\lVert x_i-x_j\rVert\le\tau_{z_iz_j}\bigr],

  where :math:`\tau` is a tolerance on summed covalent radii, and the
  likelihood is replaced by :math:`p(X\mid G)\approx\mathbb{1}[\mathrm{supp}(B)=A(X)]`.
- **Selection inside the fiber.** The estimate is

  .. math::

     \Phi(X)=\operatorname*{arg\,max}_{G\in\mathcal{G}(z,Q),\ \mathrm{supp}(B)=A(X)}\ \log\pi_\theta(G).

  This is an integer linear program over :math:`(B,q,h)` with an objective
  that encodes a chemical prior:

  .. list-table::
     :header-rows: 1
     :widths: 35 65

     * - method
       - :math:`\log\pi_\theta(G)`
     * - xyz2mol / RDKit ``DetermineBonds`` (Kim 2015)
       - :math:`\sum_{i<j}B_{ij}`, then :math:`-\sum_i\lvert q_i\rvert`
         (maximal bonding, minimal charge separation)
     * - Antechamber, Dehof 2011
       - :math:`-\sum_i \mathrm{AP}(z_i,D_i)`, a sum of atomic valence
         penalties
     * - Labute 2005
       - statistically weighted bond terms; solved as a maximum-weight
         degree-constrained matching over extra π bonds
     * - Froeyen 2005
       - stability score under linearized octet constraints
     * - Welsh 2019
       - :math:`-\bigl(\sum E_{\mathrm{bond}}+\sum E_{\mathrm{atom}}\bigr)`
         from quantum-chemistry tables

- **Structure.** :math:`\Phi=\mathrm{select}\circ A` factors through a
  discrete invariant. Two configurations with the same contact map receive
  the same structure: the region :math:`R_G` is a union of fibers of
  :math:`A`, and bond lengths play no further role. Validity holds by
  construction.
- **Consequences.** (a) When :math:`\pi_\theta` takes the same value on
  several members of a resonance or tautomer class, the argmax is a set and
  the reported answer depends on tie-breaking. (b) An error in :math:`A(X)`,
  from a distorted geometry, passes straight to the output, because no later
  step looks at the coordinates.

Machine learning: discriminative estimation of the posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Model.** A :math:`\Gamma`-equivariant parametric family
  :math:`q_\theta(G\mid z,X)`, implemented by an E(3)-invariant,
  permutation-equivariant network (PaiNN, SE(3)-Transformer). It is usually
  factorized over entries,
  :math:`q_\theta(G\mid X)=\prod_{i<j}q_\theta(B_{ij}\mid X)\prod_i q_\theta(q_i,h_i\mid X)`.
- **Fitting.** Minimize the expected cross-entropy under the data measure,
  which is equivalent to minimizing
  :math:`\mathbb{E}_X\,\mathrm{KL}\bigl(P(\cdot\mid X)\,\Vert\,q_\theta(\cdot\mid X)\bigr)`
  over the training distribution :math:`P`:

  .. math::

     \theta^{*}=\arg\min_\theta\ \mathbb{E}_{(G,X)\sim P}\bigl[-\log q_\theta(G\mid X)\bigr].

  Two-stage architectures (BondNet, MPerformer, YuelBond) use the chain-rule
  factorization
  :math:`q_\theta(B\mid X)=q_\theta(\mathrm{supp}B\mid X)\,q_\theta(B\mid\mathrm{supp}B,X)`:
  connectivity first, then bond type.
- **Decoder.** Taking the entrywise argmax gives the Hamming-loss Bayes
  estimator over :math:`\mathcal{M}`, not the MAP estimate over
  :math:`\mathcal{G}`. That is why validity is not guaranteed.
- **Consequences.** The method learns the full likelihood from data, so bond
  lengths are used and the decision boundaries track the posterior, which
  explains the robustness to noise. It is also tied to the training measure
  :math:`P`: generalization outside the training chemistry (e.g. to metals)
  is not controlled. Within a non-identifiable class, :math:`q_\theta` splits
  probability according to labeling conventions in the training set, and
  entrywise decoding can then combine bonds from different resonance forms
  into an invalid matrix, as in BondNet's residual errors on N–O and carboxyl
  groups.

Comparison and the hybrid estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 18 16 16 18 18

   * -
     - likelihood :math:`p(X\mid G)`
     - prior :math:`\pi(G)`
     - search set
     - optimizer
     - Bayes target
   * - rule-based
     - hand-set local windows
     - implicit (templates)
     - :math:`\mathcal{M}`, then repair
     - greedy
     - none (heuristic)
   * - optimization
     - coarse-grained to :math:`A(X)`
     - explicit objective
     - :math:`\mathcal{G}`
     - exact ILP / matching / DP
     - MAP under a flat likelihood
   * - ML
     - learned
     - learned (from data)
     - :math:`\mathcal{M}`
     - entrywise argmax
     - Hamming loss
   * - hybrid
     - learned
     - learned
     - :math:`\mathcal{G}`
     - exact ILP / DP
     - MAP under a learned likelihood

NAOMI-type systems combine rule-derived scores with search over
:math:`\mathcal{G}`. The last row pairs a learned likelihood with exact
search over the valid set:

.. math::

   \Phi(X)=\operatorname*{arg\,max}_{G\in\mathcal{G}(z,Q)}\ \log q_\theta(G\mid X).

This is the constrained MAP estimate, which combines the geometric sensitivity
of the ML family with the validity guarantee of the optimization family.

.. admonition:: In bond-bench

   Each family has one canonical implementation, described in
   :doc:`canonical-algorithms`: ``canonical-R`` (rule-based),
   ``canonical-O`` (optimization), ``canonical-M`` (ML) and
   ``canonical-M+hybrid`` (hybrid). The ``+hybrid`` wrapper applies the
   constrained MAP to any method that reports ``edge_probs``.

Input/output summary
--------------------

**Input.** A labeled point cloud :math:`(z,X)` with :math:`z\in\mathbb{E}^n`
and :math:`X\in\mathrm{Conf}_n(\mathbb{R}^3)`, considered modulo
:math:`S_n\times E(3)` (equivalently :math:`(z,\Delta(X))` modulo
:math:`S_n`), together with the total charge :math:`Q` and optionally the
number of unpaired electrons. Any of the observation operators
:math:`\mathcal{O}` may have been applied to it.

**Output.** An element of the finite set :math:`\mathcal{G}(z,Q)` of
valence-valid integer bond-order matrices with charge and hydrogen vectors,
or of its quotient :math:`\mathcal{G}/\!\sim` by non-identifiable
structures. A method can return:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - output type
     - mathematical object
   * - point estimate
     - a measurable map :math:`\Phi:\mathcal{X}_n\to\mathcal{G}`
   * - set-valued estimate (all co-optimal structures)
     - a correspondence :math:`\Phi:\mathcal{X}_n\rightrightarrows\mathcal{G}`
   * - probabilistic estimate
     - a Markov kernel :math:`K:\mathcal{X}_n\to\Delta(\mathcal{G})`, or
       :math:`\Delta(\mathcal{M})` if validity is not enforced

**Evaluation** is the empirical risk under the chosen loss:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - metric
     - corresponding loss
     - bond-bench metric
   * - per-molecule exact match
     - 0–1 loss on :math:`\mathcal{G}`
     - ``exact``
   * - resonance-aware exact match
     - 0–1 loss on :math:`\mathcal{G}/\!\sim`
     - ``resonance_match``
   * - per-bond F1
     - Hamming loss
     - ``f1_*``
   * - validity rate
     - :math:`\Pr[\Phi(X)\in\mathcal{G}]`
     - ``valid``
   * - robustness
     - :math:`\Pr_\varepsilon\bigl[\Phi(X+\varepsilon)\sim G^{*}\bigr]` as a
       function of :math:`\sigma`, i.e. the distribution of margins
     - C1 ``sigma`` subgroups, ``--noise``

.. admonition:: In bond-bench

   A :class:`~bond_bench.eval.prediction.Prediction` holds all three output
   types: the point estimate ``graph``, the set-valued ``alternatives``, and
   the probabilistic ``edge_probs``/``atom_probs``.

Extensions
----------

- **Transition metals.** Enlarge the output alphabet with a dative label, and
  add metal oxidation states :math:`\mathrm{ox}_i` to the unknowns, with
  :math:`\mathbf{1}^{\top}q+\sum_{i\in\mathrm{M}}\mathrm{ox}_i=Q`. Multicenter
  (hapto) bonding replaces :math:`B` with a symmetric tensor on subsets of
  atoms, i.e. a hypergraph. (The dataset reserves bond order 9 for
  coordination bonds.)
- **Crystals.** Points lie on the torus :math:`\mathbb{R}^3/L`, the symmetry
  group becomes the space group, and the output is a periodic graph: a finite
  quotient graph labeled with lattice translation vectors.
- **Protonation.** In the hydrogen-free setting, :math:`h` is part of the
  unknown and the likelihood is marginalized over hydrogen positions, so
  tautomer and protonation choices are decided mainly by the prior
  :math:`\pi` (as in Protoss).
