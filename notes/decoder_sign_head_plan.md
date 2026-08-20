# Decoder-extended sign head ("head v2") — design spec & staged plan

2026-08-13. Status: DESIGN — Stage 1 prototype pending. Owner: user + Claude
(coordinator); implementation by delegated agents; adversarial audit gate applies.

## 0. Motivation

The production analytic phase head (token-quadratic, `analysis/scripts/prefit_phase_head.py
--analytic_C`) is exact **only on the h=0 support** V = orbit of |0…0⟩ under
{stars, decorated-plaquette pair-flips}. Off-support it still outputs a definite
±1 — the solver-gauge "shadow" extrapolation of the RREF pivot choice
(`prefit_phase_head.py:188`: "off-V values are irrelevant (flux penalty)") — an
*arbitrary checkerboard* the trunk must dress against. That is harmless on the
electric line (σ^z is diagonal: the state never leaves V) and fatal-in-principle
for the north star: NQS under **arbitrary perturbations**, where off-support
weight is physical (h_x populates error sectors) and its sign structure is the
hard part (SR never repairs signs — BLOG 2026-08-07).

**Proposal (user, 2026-08-13):** extend the exact fixed-point sign to *all* 2^N
configurations by a canonical error decomposition: decode σ to the support via a
canonical error E(syndrome), then read the sign from the stabilizer algebra:
sign(σ) = sign( E · ∏B̃_p^{x_p} |0…0⟩ ). MWPM was the illustrative decoder; we
adopt the **fixed GF(2)-linear decoder** form instead (one right-inverse, built
once) so the whole construction collapses to an analytic form.

## 1. Key structural result

Everything in the construction is linear or quadratic over GF(2) in the edge
bits b (spins s_i = (−1)^{b_i}):

1. support membership / syndrome: canonical coset representative e(b) = the
   residual of b after RREF-elimination against the flip-generator image Im(F)
   (equivalently e = R·K·b, K = check matrix, R = fixed right-inverse);
2. application pattern: x(b) = P·(b ⊕ e(b)), P = fixed pivot right-inverse of
   the generator map (full-RREF with preimage tracking, exactly the
   `prefit_phase_head.py:202-217` pattern — including the reduce-against-EVERY-
   pivot bug note);
3. sign: the verified application-variable C-form q_C(x) = Σ_{p<q} C_pq x_p x_q,
   C_pq = |∂p ∩ xpair_q| mod 2, plus (convention-dependent) a cross term.

⇒ **S(b) = (−1)^{q(b)} with q a single quadratic form over GF(2) in edge bits.**
No decoder runs inside logψ; the decoder is baked into the coefficients at build
time. Evaluation = one matmul on raw spins, φ = s·λ + sᵀΛs, frozen constants —
same code shape and budget as the token head (N=3L³=192 at L=4 → ~37k frozen
entries). Strictly more expressive: a flux token is a 4-edge parity, i.e.
GF(2)-LINEAR in bits, so every token-quadratic is a bit-quadratic (v1 ⊂ v2).
This does not contradict the stencil obstruction (no *local* TI form in state
variables): Λ is non-local through R and P.

## 2. Gauge conventions — the fork the prototype must settle

Two principled orderings, both exact bit-quadratics:

- **ERROR-LAST** ("PT-injection gauge"): S = sign of W(x)|0⟩ with X(e) applied
  last (pure X, no sign). Matches first-order injection: coefficient of σ′ in
  σ^x_e|GS⟩ is the parent's sign. But intra-sector B̃ transport is off by
  (−1)^{|∂p ∩ e|} → B̃ matrix elements are NOT all stoquastic off-support.
- **ERROR-FIRST** ("H0-stoquastic gauge"): apply X(e) to |0⟩ first, then the
  plaquettes; each application's z-parity then sees e, adding the cross term
  Σ_p x_p·|∂p ∩ e| mod 2 (bilinear in (x,e), still bit-quadratic in b). Makes
  EVERY B̃ off-diagonal stoquastic in every sector (same argument as on-support);
  shifts the σ^x injection sign by a linear-in-x factor.

If the two conflict in a sector, that sector is intrinsically sign-frustrated
(the fermion-ness); the trunk keeps that residual and we want to *measure* it.
Expectation (to be tested, not assumed): error-first wins at J ≫ h because
intra-sector H0 relaxation dominates injection.

Second convention axis: the choice of R fixes homology-class representatives of
errors. **No logical/homology machinery exists in the repo** (orientation audit
2026-08-13); Stage 1 accepts the RREF-implied class and *measures* the damage;
minimal-weight/anchored decoders are a later refinement, decided on data.

## 3. Stage 1 — standalone prototype (delegated implementation)

`analysis/scripts/decoder_sign_prototype.py`, pure numpy + Python-int bitmasks, imports
tc3d ONLY for geometry + `fermionic_decoration` (mirror `prefit_phase_head.py`
setup and ordering conventions exactly). No NetKet, no eigsh, no ED. L=2 PBC
default (N=24, |V|=2^15=32768 — all enumerable locally in seconds), `--L 3`
sampled spot-check. Deterministic `--seed`. Both gauges behind `--gauge
{error_last,error_first}`.

Validation ladder (print PASS/FAIL table + metrics; dump JSON to
`results/fermionic_h0/decoder_gauge_L{L}.json`):

- **V0 well-definedness**
  a. generator relations: for random k ∈ ker(F) and random x: q-sign invariant
     (operator identities carry phase +1; BFS "dep" consistency implies it);
  b. star invariance: S(b) = S(b ⊕ star) for all stars × random b;
  c. pivot-gauge dependence: rebuild with permuted edge order → S′; verify
     S′·S restricted to each syndrome sector is an affine-linear (−1)^{ℓ(b)}
     pattern (the predicted Pauli-transport form), and report it.
- **V1 on-support exactness**: S == BFS sign on all 32768 support states
  (BFS reimplemented as in `prefit_phase_head.py:155-177`). Must be 100% —
  hard FAIL otherwise.
- **V2 PT-gauge match** (weight-1 sectors, exhaustive at L=2): for every b with
  e-weight 1 reachable by a single flip from V:
  a. *frustration*: do all single-flip support parents of b transport the same
     sign? Report the frustrated fraction (property of the model, not the gauge);
  b. *match*: on unfrustrated states, does S(b) equal the common parent sign?
     Report per gauge. Also compute the first-order vector on each sector
     (H0-eigenbasis-correct: project V|GS₀⟩ onto the sector and relax with the
     sector-restricted H0 sign structure if feasible cheaply; else report the
     raw injection-sign comparison and say so).
- **V3 stoquasticity / violation audit — the go/no-go**: over support ∪ all
  weight-1 ∪ sampled weight-2 sector states, for each off-diagonal matrix
  element (B̃_p terms, coef −J; σ^x terms, coef −h_x): count sign-violating
  elements (S(b′)·M_{b′b}·S(b) > 0), raw and PT-weighted (h^{|e|}), for gauge ∈
  {v1 shadow (replicate the analytic_C token pullback), v2 error-last,
  v2 error-first}. Two field points: (h_x=0.3, h_z=0) and (0.3, 0.15).
  Success criterion: v2 (best gauge) strictly and substantially below v1.
- **V4 scale check**: L=3 (N=81, Python ints are fine), sampled: V0b, V1 on
  sampled classes (as `--sampled` prefit mode), V2a/b on sampled weight-1 states.

## 4. Stage 2 — wiring (AFTER adversarial audit of Stage 1 + user go)

- `networks.py`: `phase_head_bits` option — frozen `constants` (λ: (N,),
  Λ: (N,N)) evaluated on raw spins; replaces (not stacks with) the token head;
  checkpoint-tree-compatible story same as `phase_head_frozen`.
- Build-time constructor: extend `prefit_phase_head.py` with `--analytic_decoder`
  (emits λ, Λ from the Stage-1 construction; asserts on-support agreement with
  `--analytic_C` head).
- κ policy: syndrome-weight penalty vs current Gauss-mask κ — on the magnetic
  line the perturbed GS *needs* off-support weight, so κ must not fight PT
  weight. Policy decision deferred to Stage 3 data (flagged in
  `notes/fermionic_next_steps.md` as the magnetic-line question).
- `tests/test_fermionic.py`: head-v2 == BFS on support; head evaluation ==
  prototype S on random states (both gauges); v1⊂v2 consistency check.

## 5. Stage 3 — physics validation (gated: cluster/Colab, per-request approval)

- Extend `analysis/scripts/ed_electric_line.py` to the magnetic line at L=2, with
  degenerate-manifold sector projection fixed (the banked
  `ed_L2_electric.json` h=0 row is contaminated: `sign_match_weighted=0.78`
  from an unprojected eigsh vector — known artifact, do not reuse as-is).
- A/B training runs L=2/L=3 on the magnetic line: v1 head + complex trunk vs
  v2 head (+ real-vs-complex trunk ablation). Metrics: Vscore, E vs ED,
  sign-match of sampled configs.

## 6. Open questions / decision points (user owns)

1. Gauge convention (error-first vs error-last) — decide on V2/V3 data.
2. Homology-representative policy for R at PBC (accept RREF class vs anchored/
   minimal-weight canonical routing) — decide on V2/V3 data + near-h_c needs.
3. κ policy on the magnetic line (off / Gauss-only / syndrome-weight, fixed vs
   annealed).
4. Whether the T³ 8-fold degeneracy needs an explicit logical-sector choice for
   production PBC training (no machinery exists today).

## 6b. Stage 1 results (2026-08-13)

Prototype (`analysis/scripts/decoder_sign_prototype.py`) + two adversarial audit passes
(code lens, physics lens; results banked in
`results/fermionic_h0/decoder_gauge_L{2,3}.json`). One CRUCIAL sampling bug found
and fixed (weight-1/weight-2 pools weren't filtered on syndrome popcount —
contaminated ~25-45% of the pool); post-fix numbers below are clean.

**Construction validated**: gauge formulas match literal operator application
(0 mismatches, both orderings, L=2/L=3); v1_shadow baseline reproduces BFS
exactly on-support; RREF/preimage machinery exact. `error_last` on true
weight-1 sectors: **common_parent_match = 1.0000 exactly at both L=2 and L=3**
— proven exact by construction (not empirical), since the RREF control flow
branches only on pivot columns and a weight-1 residual's free bit doesn't
touch it, so the reduction path for a state and its parent are identical.
`error_first`'s gap is isolated entirely to its cross-term (never the base
quadratic form) — clean argument for preferring **error_last**.

**Two findings that temper the Stage-2 case**:
1. Frustration (parents disagree — an intrinsic model property, gauge-
   independent, confirmed: it *is* exactly the sign of the one B̃_p element
   connecting the two parents) is not a fixed floor: parent count per weight-1
   sector grows with L (2 at L=2 → 3 at L=3), so the achievable ceiling itself
   changes shape, and `error_last` (a FLAT/sample-independent gauge) no longer
   saturates its own ceiling at L=3 (0.671 vs 0.751 achievable by an oracle
   majority vote).
2. Go/no-go margin (PT-weighted violation ratio error_last/v1_shadow) erodes
   with L: 0.739 (L=2) → 0.818 (L=3, computed by the auditor ad hoc — the
   shipped ladder has no L≥3 V3 section, a ladder gap to close). Still PASSES
   at L=3, but the trend, if it continues, threatens the case at production L.

**Decision point (open, user-owned)**: get L=4 trend data (same bitmask
machinery, still local-safe) before committing to Stage 2 wiring, vs. proceed
now and iterate, vs. pause and investigate a richer/sample-dependent gauge
given the flat-gauge ceiling result. **RESOLVED by L=4 data below.**

## 6c. L=4 trend data (2026-08-13) — resolves 6b's decision point

Extended ladder (`analysis/scripts/decoder_sign_prototype.py`, sampled at L≥3; L=2
regression confirmed bit-identical to the pre-fix bank). Full trend:

| | L=2 | L=3 | L=4 |
|---|---|---|---|
| parent count / weight-1 state | 2 | 3 | 4 |
| frustration fraction (intrinsic, gauge-free) | 0.500 | 0.746 | 0.878 |
| error_last bond_match | 0.750 | 0.668 | 0.625 |
| oracle ceiling | 0.750 | 0.751 | 0.686 |
| gap to ceiling (error_last) | 0.000 (exact) | 0.083 | 0.062 |
| go/no-go ratio, **Bp family** (lower=better) | 0.692 | 0.577 | 0.488 |
| go/no-go ratio, **sx family** | 0.754 | 0.827 | 0.886 |
| error_first (either family) | ~0.96-1.08 throughout — never better than the current head |

**The single-ratio framing in 6b was misleading** — the trend splits by physical
channel. The **B̃_p (stabilizer/plaquette) family's advantage over the current
production head GROWS with L** (0.69→0.58→0.49): plaquette transport stays tied
to local geometry the gauge captures. The **σ^x (bare field) family's advantage
erodes toward parity** (0.75→0.83→0.89) but never crosses below 1 (never worse)
at the L tested: a generic single-edge flip carries no plaquette structure, so
as L grows it increasingly hits configurations no FLAT gauge can distinguish —
this tracks the frustration floor rising (50%→75%→88%), a property of the
*model*, not a defect of this construction. `error_last` sits at or within
6-8% of its own oracle ceiling at every L — a flat gauge is doing close to as
well as any flat gauge could. `error_first` is conclusively ruled out (no
advantage at any L, any family) — **error_last is the gauge convention**.

Caveat (implementer's own honest flag): L=4 V3 used n_v3=600 pool states
(effective ~1800 independent draws, not enough time budget for more) — the
Bp-improves / error_first-useless conclusions are robust, but the exact sx
ratio (0.886) could shift a few % under a different seed. Directional trend
is trustworthy; the precise crossover point is not final.

**Verdict: proceed to Stage 2.** The construction helps monotonically-or-better
across all tested L on the topologically-relevant channel, never underperforms
the current head on the generic-flip channel, and is bug-verified through two
independent audit rounds. The eroding sx ratio reflects a physical ceiling
(rising frustration), not a fixable flaw — a richer sample-dependent gauge
could theoretically chip at that specific floor, but is not a blocker to
shipping `error_last` now.

## 7. Risks

- Sector frustration may be large (fermionic exchange between defect pairs) —
  then no flat gauge exists and the payoff shrinks; V2a measures exactly this.
- Wrap/homology-class mismatches near h_c: canonical reps in the wrong class on
  a measure-small but growing set of sectors.
- Implementation traps: partial-RREF pivot bug (see `prefit_phase_head.py:205-207`),
  star-leak in P (V0b), edge-order gauge artifacts (V0c).
