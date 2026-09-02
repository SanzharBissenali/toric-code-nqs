# The fTC h=0 sign is a lattice cup product: `s = ∫ a ∪ δa`

*(2026-09-02, theory note. Labels: **[V]** verified numerically here, **[D]** derived at
cochain level, **[C]** conjecture, **[O]** open. Scripts: scratchpad
`cup_and_correction.py`, `lift_test.py`, `lift_closedform.py`, `final_checks.py`.)*

## 0. Notation

Primal cubic lattice, qubits on edges. `b ∈ C¹(T³;F₂)` = flipped-edge set (a 1-cochain);
`F = δb ∈ C²` = flux tokens (`t_p = −1`), a 2-cocycle = dual flux loop; star gauge is
`b → b + δλ`, `λ ∈ C⁰`. `x ∈ C²` = applied pair moves.

**Key identification [D].** `p ↦ e₊(p) = centre(p) + ½(1,1,1)` is a *bijection*
plaquettes → edges preserving the direction label (normal `ĉ` ↦ edge direction `ĉ`).
This is the standard hypercubic identification of the dual complex with the primal complex
shifted by the half body diagonal. Write `a ∈ C¹` for the shift-dual of `x`. Since
`e₋(p) = e₊(p) − (1,1,1)`,

    b = (1 + S) a ,        (S u)(e) := u(e + (1,1,1)) .

## A. The classical structure: the cubical (Serre) cup product

Define, for `u ∈ C¹`,

    Q(u) := ∫_{T³} u ∪ δu = Σ_{3-cells c} Σ_{A} u(edge_B at min in A) · (δu)(face_A at max in B),

`A` a 2-subset of {x,y,z}, `B` its complement — the standard cubical Alexander–Whitney /
Serre diagonal with the coordinate vertex order. Then, **exactly**,

    Q(u) = Σ_p (δu)(p) · u(e₊(p)) = Σ_p (δu)(p) · u(e₋(p)) .

* **[V]** `c₊ = c₋ = Q(b) = ∫ F ∪ b` for *arbitrary* 1-cochains `b`, 30/30 at L = 2,3,4.
  So the body-diagonal push-off **is** the cup product; `c₊ ≡ c₋` because
  `∫F∪b = ∫b∪F` on a closed manifold (they differ by `∫δ(b∪b)=0`). This is exactly
  Chen–Tata's geometric definition, "α∪β is dual to `α^∨ ∩ β^∨_shifted`, the shift
  determined by a vector field" (arXiv:[2106.05274](https://arxiv.org/abs/2106.05274),
  J. Math. Phys. 64, 091902).

**Main result [V].** The exact fermionic sign is the cup product of the *application*
cochain with itself:

    s(x) = Q(a) = ∫_{T³} a ∪ δa ,        a = shift-dual of x.

Verified 200/200 at L = 2, 3, 4 and 60/60 at L = 5 (PBC). This is translation invariant,
strictly local (one 3-cell), and analytic — no GF(2) solve, any L.

**Q is a quadratic refinement of the mod-2 linking form [D].** Its polarization is

    B(u,v) := Q(u+v)+Q(u)+Q(v) = ∫ (u∪δv + v∪δu) = ∫ δu ∪₁ δv = Lk₂(δu, δv),

using `α∪β + β∪α = δ(α∪₁β) + δα∪₁β + α∪₁δβ`. Hence
`Q(u+v) = Q(u) + Q(v) + Lk₂(δu, δv)` — the Arf/Brown–Kervaire pattern. `Q` is *not* `Sq¹`
(`∫Sq¹F = ∫w₁∪F = 0` on an orientable 3-manifold); it is the 3d spatial analogue of the
`∫(B∪B + B∪₁δB)` "generalized Steenrod square" term that Chen–Kapustin identify as the
(3+1)D replacement of Chern–Simons (arXiv:[1807.07081](https://arxiv.org/abs/1807.07081),
PRB 100, 245127). Their fermionic Gauss law
`G_e = (∏_{f⊃e} X_f) ∏_{f'} Z_{f'}^{∫δe∪₁f'}` and hopping
`U_f = X_f ∏_{f'} Z_{f'}^{∫f'∪₁f}` are the same `∪₁`-framing data on a triangulation;
we are its hypercubic avatar. Related: 2d version arXiv:[1711.00515](https://arxiv.org/abs/1711.00515);
all-d arXiv:[1911.00017](https://arxiv.org/abs/1911.00017); Kapustin–Thorngren
arXiv:[1701.08264](https://arxiv.org/abs/1701.08264); the phase is "FcBl", the 3D fermionic
surface code (Levin–Wen cond-mat/0302460, Walker–Wang arXiv:1104.2632,
Fidkowski–Haah–Hastings arXiv:2110.14654, vKBS arXiv:1208.5128).

**Star-gauge and Seifert independence [D], matches [V].** `Q(b+δλ) = Q(b) + ∫F∪δλ` and
`∫F∪δλ = ∫δ(F∪λ) = 0`; changing the Seifert surface at fixed `F` changes `b` by a cocycle
`z` and `∫δb∪z = ∫δ(b∪z) = 0`. So `c₊ = q(F)` is a function of the flux loop alone, and a
sheet wrapping a non-contractible cycle is a *cocycle* — killed by the pairing with `F = δb`.
No large-sheet ambiguity. **[D]** The star sector carries no sign at all: the extra factor is
`(−1)^{⟨δ*x, δλ⟩} = (−1)^{⟨x, δδλ⟩} = +1`.

## B. Why even L fails, and the correction (solved)

The state only sees `b = (1+S)a`; recovering `a` means integrating `b` along the
(1,1,1) lines. `ker(1+S) = span{1_ℓ}`, the 3L² body-diagonal line indicators.

**Lift independence [V, exhaustive L = 2…6].** Every `1_ℓ` satisfies `Q(1_ℓ) = 0` and
`B(u,1_ℓ) = 0 ∀u` (checked as an exact GF(2) matrix identity: `span{1_ℓ} ⊆ rad(B)`).
Therefore `Q(a)` depends only on `b`. Hence the **all-L state-variable closed form**:

    s = Q(a)  for ANY a with  a(e) + a(e+(1,1,1)) = b(e).

*Seam lift*: on each line `ℓ = {e₀, e₀+d, …}` set `a(e₀)=0`, `a(e_{k+1}) = a(e_k) + b(e_k)`.
**[V] s == Q(seam lift) 200/200 at L = 2,3,4 and 40/40 at L = 5,6.** The even-L defect is gone.

**What kind of correction is it?** Not a fixed 2-cocycle twist and not a different quadratic
refinement — `Q` is the same at all `L`. It is a **change of lift/framing on a seam**: a
codimension-1 cut transverse to (1,1,1), one base edge per diagonal line. Algebraically, a
translation-covariant lift is `a = W(S) b` with `(1+S)W = 1+P` in `F₂[S]/(S^L+1)`,
`P = Σ_k S^k`; the solution `W = Σ_{k odd, 0<k<L} S^k` exists **iff L is odd** [D].
This is the Kasteleyn/spin-structure statement: the framing vector field wraps a diagonal
cycle of length `L`, and on an even torus no translation-invariant framing/orientation
exists — one must insert a seam (the parity theorem of `fermionic_stencil_obstruction.md`,
`Σ_{g∈Z_L} q∘T_g = |G|q + …`, is the same coin: for `L` odd an odd number of translates
survives, for `L` even they cancel in pairs). Chen–Kapustin's map likewise "depends
explicitly on the choice of a spin structure" (a 2-chain `E` with `∂E = w₂`).

**Correction to a previous reading [V].** `s = c₊` is **an L = 3 accident, not an odd-L law.**
At L = 3, `W = S` (a single translation) and `Q` is translation invariant, so
`Q(Sb) = Q(b) = c₊`. At L = 5, `W = S + S³` and `s = Q(Sb + S³b) ≠ Q(b)`.
Measured `s == Q(b)`: **L=2 129/200, L=3 200/200, L=4 114/200, L=5 32/60** — i.e. chance
everywhere but L=3. Measured `s == Q(W(S)b)`: **L=3 200/200, L=5 60/60.** This reproduces the
old stencil scan exactly: range-1 TI form feasible at L=3 (the "folded gauge accident"),
TI-but-nonlocal with reach `|Δ| ≈ L/2` at L=5 (indeed `W = S+S³`), infeasible at L=4.

**Practical [C, testable].** At **OBC** the diagonal lines are cut open and boundary faces
carry a single X partner, so the boundary *is* the seam: the lift has a canonical base point
and a bulk-local, size-independent stencil sign head should exist at every `L`. This is also
why the OBC syndrome space collapses (below).

## C. The (1,1,1) diagonal-line detectors

**[D, exact]** `e₊(p)` and `e₋(p)` are parallel and differ by `(1,1,1)`, so they lie on the
*same* diagonal line. Every pair move therefore flips exactly two edges of one line, and the
`3L²` parities `n_ℓ(b) = Σ_{e∈ℓ} b_e` are conserved (`3L² × L = 3L³ = N` ✓). A single `σ^x_e`
flips exactly one of them — precisely the observed "one diagonal-line detector per flip".
The measuring operator is the diagonal Wilson line `Z_ℓ = ∏_{e∈ℓ} σ^z_e`: it commutes with
every `B̃_p` but *anticommutes* with `A_v` (a star's 6 edges lie on 6 distinct lines), which
is why the star-gauge-invariant syndrome space is strictly smaller than `3L²` (12→8 at L=2,
27→16 at L=3). At OBC the lines are broken and boundary pair moves flip only one edge of a
line, destroying the charge — hence 1 bit at L=2 OBC, 2 at L=3 OBC, and zero-syndrome
boundary flips. **[C]** This is a genuine **subsystem (lineon-like) symmetry along body
diagonals**; it is the hypercubic shadow of the `∪₁` dressing in the CKR/Chen–Kapustin
fermionic Gauss law, whose Z-string support is fixed by exactly the same shift vector field.

## D. The physics in one sentence

*The fermionic minus sign is the mod-2 self-linking number of the flux loop with the framing
supplied by the decoration's body diagonal.* The emergent fermion is a charge bound to a small
flux loop; a 2π twist costs `−1`, and on the lattice the twist is bookkept by the constant
`(1,1,1)` framing vector field. Concretely `Q(u) = ∫u∪δu` computes
`Lk₂(δu, δu^{push-off})` as the intersection of the (1,1,1) push-off of the flux loop with the
Seifert surface `u` — and the push-off lands on primal edges, which pierce dual 2-cells exactly
at themselves. The polarization `B = Lk₂` is the *mutual* statistics; `Q` is its quadratic
refinement, so `Q(F) = 1` on a single elementary loop is literally the statement "the charge is
a fermion". Gauge/Seifert independence is the statement that coboundaries integrate to zero on
a closed manifold; the *only* real ambiguity is the diagonal lift, and `Q` is blind to it.

## Open

* **[O]** Verify `s = Q(seam lift)` on the full star⊗pair orbit (I verified on the pair-move
  representative `b ∈ im(1+S)`; the star sector is signless [D], so this should be automatic
  after gauge-fixing to zero line parities — worth one check).
* **[O]** OBC: does the boundary-anchored lift give a bulk-local TI stencil at all `L`?
* **[O]** Is `Q(1_ℓ) = 0`, `1_ℓ ∈ rad(B)` provable (it looks like `δ1_ℓ` bounds a "diagonal
  tube" that links nothing), or is it an L-independent accident of the (1,1,1) shift?
* **[O]** NQS payoff: the sign head can become a fixed, parameter-free `∫a∪δa` layer —
  a line-integral (prefix-XOR along body diagonals) followed by one local cubic stencil.
