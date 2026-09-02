# Even-L (and OBC) closed form for the fTC h=0 sign — numerics report

Scratch: `.../scratchpad/evenL/` — `common.py`, `geom.py` (any Lx,Ly,Lz + PBC/OBC),
`task1*.py`, `seam.py`, `solve_full.py`, `cup.py`, `cup2.py`, `starfix.py`, `fluxfn.py`,
`gaugefix.py`, `final.py`, `extras.py`.  Conventions from `linking_test.py`:
`s(x)=x^T triu(M,1) x`, `M=I X^T`, `b=X^T x`, `F=Mx=I b`, `v=(1,1,1)`.

## 1. The defect of `c+` is exactly `D = M (1 + T_v + T_v^{-1})`   [EXACT, all L]

Polar forms: `A_s = M`; `A_c+ = C+C^T` with `C = M P_+ X^T`; `D = A_s + A_c+`.

* `D == M(1+T+T^{-1})` where `T` = plaquette translation by `(1,1,1)`: **True at L=2,3,4,5,6**
  (exact matrix identity, not sampling).
* `rank(D)`: L=2 **6**, L=3 **0**, L=4 **94**, L=5 **200**, L=6 **214**. `nnz(D)=8·NP` for all L≠3.
* `D` **is** fully translation invariant (all unit and double translations) at every L.
* `ker(M) ⊆ ker(D)` at every L, and `D = M K M` is solvable → `D` factors through the tokens.
* Support: only the off-diagonal normal blocks (`D_{cc}=0`), 4 partners in each of the two other
  normals. Stencil = the `M` stencil with its "−" half pushed by `−v` and its "+" half by `+v`.
* **Why L=3 is special.** The `M` stencil is `S ∪ (S+v)`; so `D` stencil `= (S−v) ∪ (S+2v)`,
  which collapses iff `3v ≡ 0`, i.e. **L | 3**.  Algebraically `M = W(1+T)` and
  `D = M(1+T+T^{-1}) = W T^{-1}(1+T^3)`.

**Consequence — correction to the premise:** `s = c+` is an **L=3 accident, not an odd-L law**.
Measured `c+ == s`: L=2 1119/2000, **L=3 2000/2000**, L=4 1013/2000, **L=5 1007/2000**, L=6 1049/2000.

## 2. Hypothesis tests (all negative for even L) — and the proof of why

* **(b) 8 body-diagonal push-offs and every GF(2) sum of them.** A subset with polar form `M`
  exists only at **L=3** (there both `e+` and `e−` work alone). L=2,4,5,6: **none**.
* **(a) seam framing along the diagonal.** Family `f(b)=Σ_p F_p Σ_{k∈K(c,j)} b[e+(p)+kv]`,
  `j` = the `(1,1,1)` coordinate `n_z`, `K` free per `(normal, j mod m)`:
  solvable at **L=3** (`K={0}` = `c+`) and **L=5** (`K={1}`, i.e. push-off by `2v`);
  **infeasible at L=2,4,6 for every seam period m = 1, 2, L** (polar condition alone already fails).
* **(d) line-winding terms in `n_ℓ`** cannot repair it either: they are *linear-space* corrections,
  while the obstruction is in the polar (bilinear) part.
* **Proof (group algebra `F₂[Z_L³]`).** With `X_c = t^{−e_c}(1+t^v)` and
  `M_{cc'} = (1+t^{e_{c''}}) t^{e_{c'}}(1+t^{−v})`, a TI edge-space form `A` obeys
  `X A X^T = M` iff `(1+t^v)[ t^{−e_c}(1+t^v)A_{cc'} + (1+t^{e_{c''}}) ] = 0`.
  Summing coefficients along each `⟨v⟩`-coset (`Φ`) gives obstruction
  `Φ(t^{e_c}(1+t^{e_{c''}})) = [e_c]+[e_c+e_{c''}] ≠ 0`, repairable only by `ν_v·h`, and
  `Φ(ν_v h) = (L mod 2)·H`. **So a translation-invariant closed form exists iff L is odd** —
  the Kasteleyn / spin-structure statement. Even L *must* carry a seam.

## 3. Existence of an exact star-invariant state formula  [brute force]

Solving `W A W^T = diag(M,0)`, `W=[X;Σ]`, `A` symmetric zero-diagonal, then the linear
counterterm `μ` from `W μ = λ`:

| geometry | unknowns | affine solution dim | verify `f(b)=s(x)` |
|---|---|---|---|
| 2×2×2 PBC | 276 | **171** | 3000/3000 |
| 2×2×2 OBC | 66 | **11** | 3000/3000 |
| 3×3×3 OBC | 1431 | **105** | 3000/3000 |
| 2×2×3 OBC | 190 | **19** | 3000/3000 |

So an exact star-invariant quadratic form always exists; it is just never TI at even L.

## 4. THE FORMULA (verified, all L, PBC **and** OBC, full star⊗pair orbit)

Cup product (Serre/Alexander–Whitney, body-diagonal front/back rule):
`Q(u) = ∫ u ∪ δu = Σ_p (δu)_p · u(e₊(p))`.

**Detectors.** `Φ = basis of null(X)` = 1-cochains constant along the `(1,1,1)` lines.
`dim null(X)`: PBC `3L²` (12/27/48/75/108 at L=2..6); OBC `3L(L−1)` (6/18/36/60 at L=2..5),
9 at 2×2×3, 19 at 2×3×4. At PBC `rowspace(X) = {b : n_ℓ(b)=0 ∀ℓ}` (verified).

**Recipe.**
1. gauge-fix: add stars to get `b'` with `Φ·b' = 0` (always feasible on the orbit);
2. lift: `a(e) = Σ_{k≥0} b'(e + k·v)` — suffix-XOR along the `(1,1,1)` line, anchored at the
   `+v` end (PBC: a seam; **OBC: the boundary itself, canonical**);
3. `s = Q(a)`.

| geometry | `s == Q(lift(gauge-fixed b))` |
|---|---|
| PBC L=2,3,4,5,6 | **3000/3000 each** |
| OBC 2³,3³,4³,5³ | **3000/3000 each** |
| OBC 2×2×3, 2×3×4 | **3000/3000 each** |

Independent of the gauge choice: 2000/2000 at 4³ PBC, 3³ OBC, 4³ OBC (random elements of the
gauge-freedom subspace added).

**Without the gauge fix** (pair-move orbit only, `b = X^T x`): `Q₊(lift₊)` is already exact —
3000/3000 at PBC L=2,4,6 and OBC 2³,3³,4³,2×2×3.  On OBC the pairing is rigid: `Q` must use
`e₊` **and** the lift must be anchored at the `+v` end (`Q₋`, or the `−v` anchor, gives chance).
`Q(shift-dual of x)` is exact at PBC for every L (3000/3000, L=2..6) but **fails at OBC**
(≈1500/3000) — the plaquette↦`e₊` map is not a bijection there.

## 5. OBC specifics asked for

* **Boundary term needed?** Yes, but only as the gauge fix. Without it the defect
  `g(b) = Q(lift(b)) + s` is **exactly a GF(2) quadratic function of the diagonal-line parities
  `n_ℓ(b)`** — 0 inconsistencies over 6000 samples at 2×2×2 (20 terms), 2×2×3 (40), 3×3×3 (132)
  OBC. (At PBC `g` is *not* a function of `n` alone: 2592/6000 inconsistencies at L=2.)
* **Off-support configs.** At OBC the lift is always defined and `(1+S)a = b` holds for an
  **arbitrary** `b` (500/500 random `b` at 3³ and 4³ OBC) — no seam ambiguity. At PBC the same
  check gives 0/500 unless `n_ℓ=0`. The OBC "defect" instead lands on the single-X boundary faces
  at the **−v end** of each line: there `X^T x_rec ≠ b`, which is exactly what the `null(X)`
  gauge fix removes.
* **Star-invariance / flux dependence.** On the physical orbit `s` is a function of the flux
  `F = δb` **alone**: 0 inconsistencies over 6000 samples at OBC 2³,2×2×3,3³,4³ and PBC 2³,4³.
  `[δ, ray]` is nonzero on 0/6 (2³ OBC), 2/11 (2×2×3), 24/36 (3³), 90/108 (4³) plaquettes —
  the non-commutation is what the gauge fix compensates.
