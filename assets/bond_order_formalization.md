# Bond order assignment as graph assignment to a 3D point cloud

This note states bond order assignment (BOA) in general mathematical terms. The input is a finite cloud of labelled points in $\mathbb{R}^3$, considered up to rigid motion and relabelling. The output is an integer bond-order matrix satisfying valence constraints. Solving the problem means constructing an equivariant map between these two spaces, which amounts to inverting the process by which a molecular graph produces a geometry. Rule-based, optimization and machine-learning methods are three ways of building such a map. References are to the attached review.

---

## 1. Input space

**Labelled point clouds.** Let $\mathbb{E}$ be a finite alphabet of elements. A molecule with $n$ atoms is presented as
$$
(z,X)\in\mathbb{E}^n\times\mathrm{Conf}_n(\mathbb{R}^3),\qquad
\mathrm{Conf}_n(\mathbb{R}^3)=\{X=(x_1,\dots,x_n)\in(\mathbb{R}^3)^n : x_i\neq x_j\ \text{for } i\neq j\},
$$
together with scalar side information $Q\in\mathbb{Z}$ (total charge) and, optionally, $m\in\mathbb{N}$ (number of unpaired electrons).

**Symmetry.** The group $\Gamma=S_n\times E(3)$ acts on inputs by
$$
(\sigma,g)\cdot(z,X)=\bigl(z\circ\sigma^{-1},\;(g\,x_{\sigma^{-1}(1)},\dots,g\,x_{\sigma^{-1}(n)})\bigr).
$$
The physically meaningful input is therefore the orbit $[z,X]\in\mathcal{X}_n=(\mathbb{E}^n\times\mathrm{Conf}_n)/\Gamma$.

**Reduction to distances.** Bond orders do not change under reflection, so the full group $E(3)$, including improper motions, applies. A configuration is determined up to $E(3)$ by its Euclidean distance matrix
$$
\Delta(X)_{ij}=\lVert x_i-x_j\rVert^2,\qquad \Delta(X)\in\mathrm{EDM}_n^{(3)},
$$
where $\mathrm{EDM}_n^{(3)}$ is the set of Euclidean distance matrices of embedding dimension at most 3. Every admissible solution can therefore be written as a function of $(z,\Delta(X))$, and the input can be identified with $(\mathbb{E}^n\times\mathrm{EDM}_n^{(3)})/S_n$. A permutation $P$ acts on it by $\Delta\mapsto P\Delta P^{\top}$.

**Partial observation.** The other input conditions are observation operators applied to the full cloud $X$:

| Condition | Observation $\mathcal{O}(X)$ |
|---|---|
| full 3D with H | $X$ |
| hydrogens missing (PDB ligands) | $\pi_{\mathrm{H}}(X)$, the projection that deletes the H points |
| noisy geometry (MD, generative models) | $X+\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2 I)$ |
| connectivity only | $A(X)$, the contact graph (§4.2) |
| periodic crystal | $X$ modulo a lattice $L$, i.e. points on the torus $\mathbb{R}^3/L$ |

---

## 2. Output space

**Lewis structures as integer matrices.** On the same index set $[n]$, a chemical graph is a triple $G=(B,q,h)$:

- $B\in\{0,1,2,3\}^{n\times n}$, symmetric with zero diagonal. $B_{ij}$ is the bond order between atoms $i$ and $j$, and $B_{ij}=0$ means no bond. The support $\mathrm{supp}(B)$ is the connectivity.
- $q\in\mathbb{Z}^n$, formal charges.
- $h\in\mathbb{N}^n$, implicit hydrogen counts. $h=0$ when hydrogens are among the points.

(Unpaired electrons $r\in\mathbb{N}^n$ can be added in the same way.)

**Validity.** Let $D=B\mathbf{1}+h$ be the vector of bond valences. The **valid set** is
$$
\mathcal{G}(z,Q)=\Bigl\{(B,q,h):\ B=B^{\top},\ \mathrm{diag}(B)=0,\ \bigl(q_i,D_i\bigr)\in\Lambda(z_i)\ \forall i,\ \mathbf{1}^{\top}q=Q\Bigr\}.
$$
Here $\Lambda(a)\subset\mathbb{Z}\times\mathbb{N}$ is the finite table of admissible (charge, valence) states of element $a$; for example, $\Lambda(\mathrm{N})\ni(0,3),(+1,4),(-1,2)$. For main-group elements the table follows from the Lewis conditions: the number of non-bonding electrons $\nu(a)-q_i-D_i$ is non-negative and even, and the shell count $\nu(a)-q_i+D_i$ is at most $2o(a)$, where $\nu(a)$ is the valence-electron count and $o(a)$ the octet capacity.

$\mathcal{G}(z,Q)$ is therefore a **finite set of lattice points** cut out by linear equalities and membership constraints, i.e. the integer points of a union of polytopes. $S_n$ acts on it by $(B,q,h)\mapsto(PBP^{\top},Pq,Ph)$.

**Non-identifiable structures.** Two valid structures $G\neq G'$ can correspond to the same physical state, for example the Kekulé forms of benzene or the two resonance forms of a carboxylate. §3 makes this precise. The well-posed output space is a quotient $\mathcal{G}/\!\sim$, and a canonicalizer $\kappa:\mathcal{G}/\!\sim\ \to\mathcal{G}$ picks one representative per class (for example the form with minimal $\sum|q_i|$, written with aromatic notation).

---

## 3. The problem

### 3.1 Deterministic statement
Find a map
$$
\Phi:\ \mathbb{E}^n\times\mathrm{Conf}_n(\mathbb{R}^3)\times\mathbb{Z}\longrightarrow\mathcal{G}
$$
that satisfies

1. **Equivariance.** $\Phi\bigl((\sigma,g)\cdot(z,X),Q\bigr)=\sigma\cdot\Phi(z,X,Q)$. Equivalently, $\Phi$ descends to a map $\mathcal{X}_n\to\mathcal{G}/S_n$ from point-cloud orbits to isomorphism classes of chemical graphs.
2. **Soundness.** $\Phi(z,X,Q)\in\mathcal{G}(z,Q)$ whenever $\mathcal{G}(z,Q)\neq\emptyset$.
3. **Correctness.** $\Phi(z,X,Q)\sim G^{*}$, where $G^{*}$ is the graph that produced $X$ (§3.2).

The input space is a continuous manifold and the output space is finite. Any such $\Phi$ is therefore piecewise constant: it partitions configuration space into **decision regions**
$$
R_G=\Phi^{-1}(G),\qquad \mathrm{Conf}_n=\bigsqcup_{G}R_G,
$$
and it must be discontinuous on the region boundaries $\partial R_G$. The robustness of a method at $X$ is the **margin** $\operatorname{dist}(X,\partial R_{\Phi(X)})$: the smallest perturbation of the coordinates that changes the output.

### 3.2 Statistical statement: an inverse problem
**Forward model.** A chemical graph $G$ does not fix a geometry; it induces a probability measure on configurations,
$$
X\sim p(X\mid G).
$$
This measure is the Boltzmann distribution of conformers on the potential energy surface of $G$, convolved with whatever measurement or model noise is present (crystallographic error, MD fluctuations, generative-model error). A data source corresponds to a joint measure $P(G,X)=\pi(G)\,p(X\mid G)$, where $\pi$ is the prior over chemistry, e.g. drug-like molecules for GEOM or PDB ligands for crystallographic sets.

**BOA is the inverse problem.** Given $X$, recover $G$. With a loss $L:\mathcal{G}\times\mathcal{G}\to\mathbb{R}_{\ge0}$, the Bayes-optimal assignment is
$$
\Phi^{*}(X)=\arg\min_{G\in\mathcal{G}(z,Q)}\ \mathbb{E}_{G'\sim P(\cdot\mid X)}\bigl[L(G,G')\bigr],
\qquad P(G\mid X)\propto p(X\mid G)\,\pi(G).
$$
Which loss is used determines what the estimate is:

| Loss | Bayes estimator | Remark |
|---|---|---|
| 0–1 loss on $\mathcal{G}$ | $\arg\max_{G\in\mathcal{G}}p(X\mid G)\,\pi(G)$ (MAP) | exact-match criterion; valid by construction |
| Hamming loss on entries of $B$ | entrywise posterior mode $\hat B_{ij}=\arg\max_k P(B_{ij}=k\mid X)$ | per-bond F1 criterion; **may lie outside $\mathcal{G}$** |
| 0–1 loss on $\mathcal{G}/\!\sim$ | MAP over equivalence classes | resonance-aware exact match |

**Identifiability.** Define $G\sim G'$ if $p(\cdot\mid G)=p(\cdot\mid G')$, meaning no geometry can tell them apart. Resonance forms satisfy this by definition: they are alternative descriptions of a single electronic state with a single geometry. Their posterior ratio is then fixed by the prior, $P(G\mid X)/P(G'\mid X)=\pi(G)/\pi(G')$, whatever $X$ is. Recovery is possible only on the quotient $\mathcal{G}/\!\sim$. Within a class, the "correct" answer is a matter of convention and is fixed by $\kappa$, not by the data. Tautomers are different: they are distinct physical states with different $p(X\mid G)$, so they are identifiable in principle. In practice their likelihoods are close, and without hydrogens they are only weakly separated.

**Partial observation.** If only $Y=\mathcal{O}(X)$ is observed, the likelihood becomes the pushforward $p(Y\mid G)=\int\mathbb{1}[\mathcal{O}(X)=Y]\,p(X\mid G)\,dX$. Without hydrogens this means integrating over the H positions, which flattens the likelihood and makes the prior more important. With noise it is a convolution. With connectivity only, the likelihood is constant on the fibres of the contact map.

### 3.3 Unified form of a solution
Every method in the review computes
$$
\boxed{\ \Phi(X)=\operatorname*{arg\,max}_{G\in\mathcal{S}}\ S_\theta(G;z,X)\ }
$$
with a **score** $S_\theta$ that stands in for $\log p(X\mid G)+\log\pi(G)$, a **search set** $\mathcal{S}$ that is either $\mathcal{G}$ (valid only) or the unconstrained matrix space $\mathcal{M}=\{0,1,2,3\}^{n\times n}_{\mathrm{sym}}\times\mathbb{Z}^n\times\mathbb{N}^n$, and an **optimizer** that is exact or approximate. The families differ in which part of the posterior they model and how.

---

## 4. The three families as estimators

### 4.1 Rule-based: explicit semi-algebraic partition
- **Likelihood model.** A product of local factors with hard windows,
$$
p(X\mid G)\;\approx\;\prod_{i<j}\mathbb{1}\bigl[\Delta_{ij}\in W_{z_iz_jB_{ij}}\bigr]\ \prod_{(i,j,k)}\mathbb{1}\bigl[\angle_{ijk}\in W'_{\dots}\bigr]\cdots,
$$
with windows tabulated from CSD/PDB statistics. The prior $\pi$ is ignored, apart from functional-group templates.
- **Optimizer.** A sequential greedy cascade over atoms and groups, followed optionally by a valence repair that maps $\mathcal{M}\to\mathcal{G}$.
- **Geometry of the decision regions.** Each window test is a polynomial inequality in the coordinates, because $\Delta_{ij}=\lVert x_i-x_j\rVert^2$ is quadratic and cosines of angles are rational in the coordinates. The regions $R_G$ are therefore **explicit semi-algebraic sets**, with boundaries fixed by hand-set thresholds rather than by the posterior.
- **Consequence.** Margins are small exactly where windows for different bond orders nearly touch, as for C–C single/double or C–O single/double bonds. Noise of comparable size moves $X$ across a boundary. This is the brittleness the review describes for noisy input.

### 4.2 Global optimization: coarse-grain the geometry, then choose a structure
- **Likelihood model.** The geometry enters only through the **contact map**
$$
A:\mathrm{Conf}_n\to\{0,1\}^{n\times n},\qquad A(X)_{ij}=\mathbb{1}\bigl[\lVert x_i-x_j\rVert\le\tau_{z_iz_j}\bigr],
$$
where $\tau$ is a tolerance on summed covalent radii, and the likelihood is replaced by $p(X\mid G)\approx\mathbb{1}[\mathrm{supp}(B)=A(X)]$.
- **Selection inside the fibre.** The estimate is
$$
\Phi(X)=\operatorname*{arg\,max}_{G\in\mathcal{G}(z,Q),\ \mathrm{supp}(B)=A(X)}\ \log\pi_\theta(G).
$$
This is an integer linear program over $(B,q,h)$ with an objective that encodes a chemical prior:

| Method | $\log\pi_\theta(G)$ |
|---|---|
| xyz2mol / RDKit `DetermineBonds` (Kim 2015) | $\sum_{i<j}B_{ij}$, then $-\sum_i\lvert q_i\rvert$ (maximal bonding, minimal charge separation) |
| Antechamber, Dehof 2011 | $-\sum_i \mathrm{AP}(z_i,D_i)$, a sum of atomic valence penalties |
| Labute 2005 | statistically weighted bond terms; solved as a maximum-weight degree-constrained matching over extra π bonds |
| Froeyen 2005 | stability score under linearized octet constraints |
| Welsh 2019 | $-\bigl(\sum E_{\mathrm{bond}}+\sum E_{\mathrm{atom}}\bigr)$ from quantum-chemistry tables |

- **Structure.** $\Phi=\mathrm{select}\circ A$ factors through a discrete invariant. Two configurations with the same contact map receive the same structure: the region $R_G$ is a union of fibres of $A$, and bond lengths play no further role. Validity holds by construction.
- **Consequences.** (a) When $\pi_\theta$ takes the same value on several members of a resonance or tautomer class, the argmax is a set and the reported answer depends on tie-breaking. (b) An error in $A(X)$, from a distorted geometry, passes straight to the output, because no later step looks at the coordinates.

### 4.3 Machine learning: discriminative estimation of the posterior
- **Model.** A $\Gamma$-equivariant parametric family $q_\theta(G\mid z,X)$, implemented by an E(3)-invariant, permutation-equivariant network (PaiNN, SE(3)-Transformer). It is usually factorized over entries, $q_\theta(G\mid X)=\prod_{i<j}q_\theta(B_{ij}\mid X)\prod_i q_\theta(q_i,h_i\mid X)$.
- **Fitting.** Minimize the expected cross-entropy under the data measure, which is equivalent to minimizing $\mathbb{E}_X\,\mathrm{KL}\bigl(P(\cdot\mid X)\,\Vert\,q_\theta(\cdot\mid X)\bigr)$ over the training distribution $P$:
$$
\theta^{*}=\arg\min_\theta\ \mathbb{E}_{(G,X)\sim P}\bigl[-\log q_\theta(G\mid X)\bigr].
$$
The two-stage architectures (BondNet, MPerformer, YuelBond) use the chain-rule factorization $q_\theta(B\mid X)=q_\theta(\mathrm{supp}B\mid X)\,q_\theta(B\mid\mathrm{supp}B,X)$: connectivity first, then bond type.
- **Decoder.** Taking the entrywise argmax gives the **Hamming-loss** Bayes estimator over $\mathcal{M}$, not the MAP estimate over $\mathcal{G}$. That is why validity is not guaranteed.
- **Consequences.** The method learns the full likelihood from data, so bond lengths are used and the decision boundaries track the posterior, which explains the robustness to noise. It is also tied to the training measure $P$: generalization outside the training chemistry (e.g. to metals) is not controlled. Within a non-identifiable class, $q_\theta$ splits probability according to labelling conventions in the training set, and entrywise decoding can then combine bonds from different resonance forms into an invalid matrix. This matches BondNet's residual errors on N–O and carboxyl groups.

### 4.4 Comparison and the hybrid estimator

| | Likelihood $p(X\mid G)$ | Prior $\pi(G)$ | Search set | Optimizer | Bayes target |
|---|---|---|---|---|---|
| Rule-based | hand-set local windows | implicit (templates) | $\mathcal{M}$, then repair | greedy | none (heuristic) |
| Optimization | coarse-grained to $A(X)$ | explicit objective | $\mathcal{G}$ | exact ILP / matching / DP | MAP under a flat likelihood |
| ML | learned | learned (from data) | $\mathcal{M}$ | entrywise argmax | Hamming loss |
| **Hybrid** | learned | learned | $\mathcal{G}$ | exact ILP / DP | **MAP under a learned likelihood** |

NAOMI-type systems combine rule-derived scores with search over $\mathcal{G}$. The pairing the review flags as unevaluated (its open problem 3) is the last row:
$$
\Phi(X)=\operatorname*{arg\,max}_{G\in\mathcal{G}(z,Q)}\ \log q_\theta(G\mid X).
$$
This is the constrained MAP estimate, and it combines the geometric sensitivity of the ML family with the validity guarantee of the optimization family.

---

## 5. Input/output summary

**Input.** A labelled point cloud $(z,X)$ with $z\in\mathbb{E}^n$ and $X\in\mathrm{Conf}_n(\mathbb{R}^3)$, considered modulo $S_n\times E(3)$ (equivalently $(z,\Delta(X))$ modulo $S_n$), together with the total charge $Q$ and optionally the number of unpaired electrons. Any of the observation operators $\mathcal{O}$ of §1 may have been applied to it.

**Output.** An element of the finite set $\mathcal{G}(z,Q)$ of valence-valid integer bond-order matrices with charge and hydrogen vectors, or of its quotient $\mathcal{G}/\!\sim$ by non-identifiable structures. A method can return:

| Output type | Mathematical object |
|---|---|
| point estimate | a measurable map $\Phi:\mathcal{X}_n\to\mathcal{G}$ |
| set-valued estimate (all co-optimal structures) | a correspondence $\Phi:\mathcal{X}_n\rightrightarrows\mathcal{G}$ |
| probabilistic estimate | a Markov kernel $K:\mathcal{X}_n\to\Delta(\mathcal{G})$, or $\Delta(\mathcal{M})$ if validity is not enforced |

**Evaluation** is the empirical risk under the loss chosen in §3.2:

| Metric | Corresponding loss |
|---|---|
| per-molecule exact match | 0–1 loss on $\mathcal{G}$ |
| resonance-aware exact match | 0–1 loss on $\mathcal{G}/\!\sim$ |
| per-bond F1 | Hamming loss |
| validity rate | $\Pr[\Phi(X)\in\mathcal{G}]$ |
| robustness | $\Pr_\varepsilon\bigl[\Phi(X+\varepsilon)\sim G^{*}\bigr]$ as a function of $\sigma$, i.e. the distribution of margins |

---

## 6. Extensions

- **Transition metals.** Enlarge the output alphabet with a dative label, and add metal oxidation states $\mathrm{ox}_i$ to the unknowns, with $\mathbf{1}^{\top}q+\sum_{i\in\mathrm{M}}\mathrm{ox}_i=Q$. Multicentre (hapto) bonding replaces $B$ with a symmetric tensor on subsets of atoms, i.e. a hypergraph.
- **Crystals.** Points lie on the torus $\mathbb{R}^3/L$, the symmetry group becomes the space group, and the output is a periodic graph: a finite quotient graph labelled with lattice translation vectors.
- **Protonation.** In the hydrogen-free setting, $h$ is part of the unknown and the likelihood is marginalized over hydrogen positions, so tautomer and protonation choices are decided mainly by the prior $\pi$ (Protoss).
