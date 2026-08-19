# fTC h=0 sign form: locality, translation invariance, and the even-L obstruction

*(2026-08-07, plan item ② outcome. Tool: `analysis/scripts/stencil_phase_head.py`;
companion: `analysis/scripts/prefit_phase_head.py` ①-results, `notes/fermionic_next_steps.md`.)*

**Question.** The analytic GF(2) phase head solved per size is dense
(N_p + N_p² parameters). Does a *local, translation-invariant* (TI) stencil
representative exist — solve once at small L, deploy at any L, O(1) couplings?

**Answer: no — and the way it fails is physics, not noise.**

## The one place the sign IS local and TI: application variables

Let x ∈ F₂^{3L³} record *which decorated plaquette operators* B̃_p were applied
to |0…0⟩. Because the B̃_p commute, the accumulated sign is the well-defined
quadratic form

    sign(x) = (−1)^{Σ_{p<q} C_pq x_p x_q},   C_pq = |∂p ∩ xpair_q| mod 2,

with C symmetric (commutation ⇒ C_pq = C_qp), **range exactly 1 lattice unit,
translation invariant, and analytic** — no solving. Verified by sampling:
3000/3000 at L = 2, 3, 4; every plaquette couples to exactly 8 neighbors
(nonzero pairs = 4·N_p). This is the operational (bosonization-phase) form.

## State variables: the inversion is nonlocal

The state only exposes x through linear screens: flux tokens t = Mx, or the
bit string s (x up to ker M ⊕ star span). Writing the sign as a form *in the
state variables* means pulling C back through a (pseudo)inverse of a local map
— generically nonlocal, like ∇⁻². Empirically (sampled class equations, joint
GF(2) feasibility, fresh-sample certificates; all in `stencil_phase_head.py`):

| test | verdict |
|---|---|
| tokens, range 1, L=3 alone | feasible, 5000/5000 — but a *folded gauge accident* |
| the same stencil deployed at L=4/5 | ~50% = chance |
| joint L=3+4 (tokens or bits), range 1 | infeasible (~2000 inconsistent rows) |
| bits, L=4 alone, range 1 / 1.5 / 2 (= whole torus) | infeasible at every range: **no TI form at L=4 at all** |
| bits, L=4, full range, period-2 cell-parity enrichment | still infeasible (1210 bad rows) |
| bits, L=5 alone, range 1 | infeasible |
| bits, L=5, full range | **feasible** (residual 0, 1000/1000) with couplings out to \|Δ\| ≈ L/2 — TI but nonlocal |

## The parity theorem (why odd L is special)

The state is TI, so every translate q∘T_g of a valid sign form is valid, and
differences of valid forms vanish on the class space V. Hence
Σ_g q∘T_g = |G|·q + (form vanishing on V) over GF(2). For |G| = L³ **odd** the
sum is a manifestly TI *valid* representative — existence guaranteed (L=3, L=5
confirm). For even L the argument collapses, and empirically no TI
representative exists at L=4 — not even allowing period-2 sublattice structure.

## Reading: a Kasteleyn/spin-structure obstruction

Encoding fermion signs in local lattice data classically requires a Kasteleyn
orientation; on even tori no translation-invariant one exists — one must insert
a codimension-1 defect seam (the antiperiodic spin structure). Our GF(2) form
reproduces exactly this pattern: locally the sign data is trivial to write
down (the C_pq form), but globally, as a function of the state, it demands
either odd system size or TI-breaking seam-like data. The sign structure's
"hardness" is a *global topological* feature, not a local one.

## Practical consequence for the NQS

A universal O(1) convolutional phase stencil in state variables does not
exist. The production design stands as-is: **per-size dense token-quadratic
head, θ set analytically by the sampled-class GF(2) solve** (poly(L): rank
saturates at d(d+1)/2, d = dim of the token class space — 6 at L=2, 36 at
L=3). Certified exact at L=2 (full enumeration) and L=3 (10⁴ fresh classes);
`--dump_form` stores each size's (l, Q). A seam-decorated stencil (TI stencil
+ Kasteleyn-style defect plane) could still compress the head at even L — an
interesting follow-up, not a blocker.
