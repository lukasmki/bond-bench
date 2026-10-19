# Canonical algorithms for bond order assignment

Companion to *bond_order_formalization.md*. Section numbers (§) refer to that note.

This document gives one canonical algorithm for each estimator family in §4: rule-based (R), global optimization (O), machine learning (M) and the hybrid (H). Each algorithm is written to use as few assumptions as possible. Everything that is not forced by the formalism is either (a) pulled out as a named parameter, or (b) fixed to the choice that keeps the three requirements of §3.1 (equivariance, soundness, correctness). The four algorithms share one set of primitives (§0), so their differences are only the differences in the comparison table of §4.4.

---

## 0. Shared assumptions and primitives

### 0.1 Assumption ledger

Each algorithm uses only the assumptions it lists from this ledger.

| ID | Assumption | Object | Used by |
|---|---|---|---|
| A0 | Element state table | $\Lambda(a)\subset\mathbb{Z}\times\mathbb{N}$ for each $a\in\mathbb{E}$ (§2), derived from $\nu(a)$, $o(a)$ | R, O, M (optional), H |
| A1 | Contact tolerance | $\tau_{ab}$ for each element pair (§4.2) | R, O |
| A2 | Local geometric windows | $W_{abk}$ (distance), $W'$ (angle) (§4.1) | R |
| A3 | Prior objective | $\log\pi_\theta(G)$, linear in $(B,q,h)$ (§4.2 table) | O |
| A4 | Training measure | samples $(G,X)\sim P$ (§3.2) | M, H |
| A5 | Canonicalizer | $\kappa:\mathcal{G}/\!\sim\ \to\mathcal{G}$ (§2) | all (only when one representative must be reported) |
| A6 | Candidate cutoff | $r_c>\max_{ab}\tau_{ab}$ | M, H |

With $\Lambda$ derived from the Lewis conditions of §2, A0 needs only $\nu(a)$ and $o(a)$:
$$
\Lambda(a)=\{(c,d)\in\mathbb{Z}\times\mathbb{N}:\ \nu(a)-c-d\ge 0,\ \ \nu(a)-c-d\ \text{even},\ \ \nu(a)-c+d\le 2o(a),\ \ |c|\le c_{\max}\},
$$
where $c_{\max}$ is a bound on the formal charge per atom, e.g. 1 or 2.

### 0.2 Primitives

**P1 — Invariant featurization.** The only input any algorithm reads is $(z,\Delta(X),Q)$, plus angles computed from $\Delta$ by the law of cosines. By construction this makes every step E(3)-invariant (§1).

**P2 — Contact map.** $A(X)_{ij}=\mathbb{1}[\Delta_{ij}\le\tau_{z_iz_j}^2]$. Use a cell list so the cost is $O(n)$.

**P3 — Validity oracle.** $\mathrm{valid}(G)$ returns true iff $B=B^\top$, $\mathrm{diag}B=0$, $(q_i,(B\mathbf 1+h)_i)\in\Lambda(z_i)$ for all $i$, and $\mathbf 1^\top q=Q$. Cost is $O(n+|\mathrm{supp}B|)$.

**P4 — Constrained linear decoder (the core ILP).** Given an edge set $E\subseteq\binom{[n]}{2}$, edge scores $w_{ij}(k)$ for $k\in\{0,1,2,3\}$ and atom scores $u_i(c,d,h)$, solve
$$
\max\ \sum_{(i,j)\in E}\sum_{k}w_{ij}(k)\,y_{ijk}+\sum_i\sum_{(c,d,\eta)}u_i(c,d,\eta)\,s_{i,c,d,\eta}
$$
subject to
$$
\begin{aligned}
&\textstyle\sum_k y_{ijk}=1 &&\forall (i,j)\in E &&\text{(one order per pair)}\\
&\textstyle\sum_{(c,d,\eta)} s_{i,c,d,\eta}=1 &&\forall i &&\text{(one state per atom)}\\
&\textstyle\sum_{j:(i,j)\in E}\sum_k k\,y_{ijk}+\sum \eta\, s_{i,c,d,\eta}=\sum d\,s_{i,c,d,\eta} &&\forall i &&(D_i=B\mathbf1+h)\\
&\textstyle\sum_i\sum c\,s_{i,c,d,\eta}=Q &&&&\text{(total charge)}\\
&y,s\in\{0,1\},
\end{aligned}
$$
where $(c,d)\in\Lambda(z_i)$ and $\eta\in\{0,\dots,h_{\max}(z_i)\}$, with $\eta\equiv 0$ when hydrogens are observed. Read off $B_{ij}=\sum_k k\,y_{ijk}$, $q_i=\sum c\,s$, $h_i=\sum\eta\,s$. Pairs outside $E$ have $B_{ij}=0$. Any feasible point lies in $\mathcal G(z,Q)$, so P4 searches exactly $\mathcal G\cap\{\mathrm{supp}B\subseteq E\}$. Because the objective is linear in one-hot variables, every factorized score (a sum of per-edge and per-atom terms) is optimized exactly.
- *Infeasible case:* return $\bot$ with a certificate (the solver's infeasibility proof). Do not relax silently; see 0.3.
- *Solution pool:* collect all optima, or the top $K$, by adding no-good cuts $\sum_{(ijk):y^\star_{ijk}=1}y_{ijk}\le|E|-1$ and re-solving. This yields set-valued and probabilistic outputs (§5).

**P5 — Equivariant tie-breaking.** Tie-breaking by atom index breaks $S_n$-equivariance. Instead:
1. Compute a canonical atom ranking $\rho$ from $(z,\Delta)$, either by canonical labelling of the weighted complete graph or by Weisfeiler–Lehman refinement on $(z,A,\text{binned }\Delta)$ with a canonical-labelling fallback for residual ties.
2. Break ties lexicographically in the order $\rho$.

If $X$ has a nontrivial symmetry, the co-optimal set is a union of symmetry orbits. In that case return the whole set, or apply $\kappa$, which is defined on isomorphism classes and is therefore equivariant.

**P6 — Canonicalization.** $\kappa$ maps any $G$ to the representative of its class $[G]_\sim$. A minimal-assumption choice:
1. Enumerate the class: resonance moves are alternating-path / π-shift transformations that leave $\mathrm{supp}B$ and $h$ unchanged and stay inside $\mathcal G$. Implement by repeated P4 calls with $E=\mathrm{supp}B$, fixed $h$, and no-good cuts.
2. Pick the member that minimizes $\bigl(\sum_i|q_i|,\ \text{canonical string under }\rho\bigr)$ lexicographically.
3. Optionally rewrite the result in aromatic notation.

### 0.3 Common pre- and post-processing

```
PREPROCESS(z, X, Q, obs):
  Δ ← squared-distance matrix of X                  # P1
  if obs = periodic: Δ ← minimum-image distances on R^3/L
  hfree ← (obs = hydrogens missing)                 # turns on η-variables in P4
  return z, Δ, Q, hfree

FEASIBILITY_GATE(z, Q, E):
  if P4 with zero objective on E is infeasible:
      return ⊥, reason ∈ {charge Q inconsistent, Λ too small, E too small}
```

The gate makes condition (2) of §3.1 testable. If $\mathcal G(z,Q)\cap\{\mathrm{supp}B\subseteq E\}=\emptyset$, every algorithm must report $\bot$ rather than return an invalid structure. Optional fallbacks, each declared in the output: widen $E$ (raise $\tau$), widen $c_{\max}$, or treat $Q$ as unknown by dropping the charge constraint and reporting the inferred $Q$.

---

## R. Rule-based estimator (explicit semi-algebraic partition, §4.1)

**Assumptions:** A0, A1, A2 (A5 if one representative must be reported).
**Search set:** $\mathcal M$, followed by projection onto $\mathcal G$.

```
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
```

**Properties (from §4.1).**
- *Decision regions:* every branch is a finite Boolean combination of polynomial inequalities in $X$, so each $R_G$ is semi-algebraic. Boundaries sit at the hand-set window edges, not at posterior equal-probability contours.
- *Margin:* computable in closed form. It is the minimum over tested pairs of the distance from $\Delta_{ij}$ to the nearest window edge that would change the output. Report it as a per-bond confidence.
- *Minimal choices made here:* candidate *sets* in place of hard single assignments; cascade order determined by certainty plus $\rho$, which keeps the cascade equivariant; repair defined as the Hamming-nearest point of $\mathcal G$, the unique canonical projection, instead of a hand-tuned repair.
- *Cost:* $O(n)$ for tests and cascade, plus an ILP that is small in practice because only the edges with conflicts are free.

---

## O. Global-optimization estimator (coarse-grain, then select, §4.2)

**Assumptions:** A0, A1, A3 (A5 for a single representative).
**Search set:** $\mathcal G\cap\{\mathrm{supp}B=A(X)\}$, exactly.

```
ALGORITHM O(z, X, Q):
  z, Δ, Q, hfree ← PREPROCESS(z, X, Q, obs)
  E ← supp A(X)                                         # the only use of geometry
  FEASIBILITY_GATE(z, Q, E)

  # Fibre constraint supp B = E: forbid order 0 on contacts
  w_ij(0) ← −∞  for (i,j) ∈ E

  # Prior objective (A3), written as a lexicographic list of linear objectives f_1 ≻ f_2 ≻ …
  #   Kim 2015:   f_1 = Σ B_ij  (maximal bonding),   f_2 = −Σ |q_i|
  #   Dehof 2011: f_1 = −Σ AP(z_i, D_i)              (atom penalties via u_i)
  for ℓ = 1..L:
      G*_ℓ, v_ℓ ← P4(E, objective f_ℓ, extra constraints f_1 = v_1, …, f_{ℓ−1} = v_{ℓ−1})
  Opt ← solution pool of the last stage                  # all co-optimal structures

  return  set-valued:     Opt
          point:          P6(argmin_ρ Opt)              # equivariant tie-break
          diagnostic:     |Opt| > 1 ⇒ flag "prior-tied" (resonance or tautomer ambiguity, §4.2a)
```

**Properties.**
- *Factorization:* $\Phi=\mathrm{select}\circ A$. The output is constant on each fibre $A^{-1}(\text{graph})$, and validity holds by construction.
- *Exactness:* P4 solves the stated objective exactly. With fixed per-atom states, the problem reduces to a degree-constrained subgraph (b-matching) on the π-bond surplus $d_i-\deg_E(i)$, which is the Labute-type special case and is solvable in polynomial time. A general ILP solver handles the enumeration over states.
- *Minimal choices made here:* lexicographic objectives rather than hand-weighted sums, so no relative weights need to be assumed; the full co-optimal set is returned, so tie-breaking is explicit and not left to solver order.
- *Margin:* the margin is the distance to the nearest contact-threshold crossing, $\min_{ij}\bigl|\sqrt{\Delta_{ij}}-\tau_{z_iz_j}\bigr|$. Report it. Any error in $E$ passes directly into the output (§4.2b), so this number is the method's whole robustness profile.
- *Optional robustness extension (no new assumptions):* if some $\sqrt{\Delta_{ij}}$ lies within $\delta$ of $\tau$, re-solve with that pair toggled and return the union of results with a flag.

---

## M. Machine-learning estimator (discriminative posterior, §4.3)

**Assumptions:** A4, A6. A0 is used only to compute the validity flag.
**Search set:** $\mathcal M$; the decoder takes an entrywise argmax.

```
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
  θ* ← argmin_θ E_D[loss]                                # cross-entropy ≡ expected KL (§4.3)

INFER M(z, X, Q):
  z, Δ, Q, hfree ← PREPROCESS(z, X, Q, obs);  E ← candidate pairs (r_c)
  ℓ, m ← f_θ*(z, Δ|_E, Q)
  B_ij ← argmax_k ℓ_ij(k);  (q_i, h_i) ← argmax m_i      # Hamming-loss Bayes estimator over M
  return G, per-entry probabilities, valid(G)           # validity reported, not enforced
```

**Properties.**
- *Bayes target:* the entrywise posterior mode, which is optimal for per-bond F1 and can lie outside $\mathcal G$.
- *Minimal choices made here:* a candidate graph defined by a generous cutoff $r_c$, so connectivity is learned rather than fixed by $\tau$ and avoids the contact-map error of O; invariance obtained from $\Delta$ alone, so no architecture choice is forced; resonance-aware loss as an option. The resonance-aware loss removes the probability splitting caused by labelling conventions (§4.3), and with it the main way entrywise decoding mixes resonance forms.
- *Not controlled:* behaviour outside the support of $P$, e.g. metals or exotic valences. Report the out-of-distribution status of a query by checking whether its elements and $\Lambda$-states appear in the training set.
- *Margin:* no closed form. Estimate it empirically (§E) or by gradient-based adversarial search on $X$.

---

## H. Hybrid estimator (constrained MAP under a learned likelihood, §4.4)

**Assumptions:** A0, A4, A6 (A5 for a single representative). This is the union of the assumptions of O and M, minus A1 and A3, which the learned scores replace.
**Search set:** $\mathcal G\cap\{\mathrm{supp}B\subseteq E\}$, exactly.

```
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
```

**Properties.**
- *Bayes target:* MAP over $\mathcal G$ under $q_\theta$, which is the 0–1-loss estimator of §3.2. Soundness and the use of bond lengths both hold.
- *Exactness:* with a factorized $q_\theta$, the objective is a sum of per-edge and per-atom log-probabilities, so P4 returns the global constrained MAP; no approximation is introduced at decoding time. A non-factorized $q_\theta$, e.g. with pairwise edge–edge terms, makes the objective quadratic. In that case either linearize the products (standard ILP lifting) or fall back to beam search, and declare the approximation.
- *Resonance handling:* score the equivalence class instead of the representative, $\log\sum_{G'\in[G]_\sim}q_\theta(G')$. The exact version enumerates $[G]_\sim$ for the pool members (P6) and re-ranks by summed probability. The cheap version re-ranks the pool after applying $\kappa$, adding up the probabilities of members that map to the same representative.
- *Reduction checks:*
  - with $w_{ij}(0)=-\infty$ on $A(X)$ and $w,u$ set to $\log\pi_\theta$, H becomes O;
  - with the constraints dropped, H becomes M;
  - with $w$ set to window indicators, H becomes R followed by projection.
  So H is the general case and the other three are its specializations.
- *Gap diagnostic:* the score gap between the best and second-best valid structure is the hybrid's counterpart of a margin, and it is cheap to compute from the pool. A small gap marks inputs where the answer is decided by the prior or by convention.

---

## E. Common evaluation protocol (§5)

```
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
```

---

## Summary

| | R | O | M | H |
|---|---|---|---|---|
| Assumptions | A0 A1 A2 | A0 A1 A3 | A4 A6 | A0 A4 A6 |
| Geometry used | windows on $\Delta$, angles | $A(X)$ only | full $\Delta$ (learned) | full $\Delta$ (learned) |
| Edge set | $A(X)$ | $A(X)$ (fixed support) | cutoff $r_c$ | cutoff $r_c$ (support decided jointly) |
| Decoder | greedy cascade + P4 projection | lexicographic P4 | entrywise argmax | P4 with learned scores |
| Validity | by projection | by construction | reported only | by construction |
| Native output | point | set (co-optimal) | Markov kernel on $\mathcal M$ | point, set, or kernel on $\mathcal G$ |
| Margin / confidence | closed form (window edges) | closed form ($\tau$ crossings) | empirical | score gap from the solution pool |
| Equivariance | via $\rho$ in the cascade order | via P5 | by architecture | by architecture + P5 |
