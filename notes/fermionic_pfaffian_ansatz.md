# A whole-phase-diagram ansatz for the 3D fermionic toric code: X-basis gauge CNN × neural Pfaffian

Status: **proposal** (written 2026-10-20 after the DMRG L=3 OBC benchmark). Nothing here is
implemented. The fermionic code lives in the `../toric-code-nqs-fsign` worktree
(branch `feat/dmrg-L3-ftc`); this note is the design brief for the next session.

## 1. Why the current stack cannot cover the plane

All three benchmarked fermionic arms (A real net, B complex net, R real net + perturbative
amplitudes; see BLOG 2026-10-19 night and 2026-10-20) share one construction: work in the
σ^z basis, fix the sign of every amplitude by the exact h=0 rule (cup product
∫a∪δa, `tc3d/sign_frame.py` / `sign_decoders.py`), and let a CNN learn positive amplitudes.
Where that rule is exact the arms match DMRG (whole hz axis to 1.4, hx to ≈0.1); where it is
not they fail:

| region | what happens to the sign | result |
|---|---|---|
| hz axis | σ^z is diagonal; fermion pairs are created but the cup sign stays exact | A/B match DMRG to 1e-5, map the electric transition at hz≈0.9–1.2 |
| hx ≲ 0.4 | single flips leave the flux sector; amplitude = hx·D(s), D an interfering sum over 12 pre-images; sign(D) ≠ cup sign | A/B stuck on the κ floor; R (first-order head) exact to 0.1, 2e-4 at 0.2, 1.7e-3 at 0.4 |
| hx ≳ 0.8 | many flips deep, many flux sectors with comparable weight; sign non-local | every arm wrong; R broken |
| hy ≠ 0 | complex phases, no stoquastic frame with three fields on | not attempted for fTC; bTC hy axis also unresolved (saddle) |

The failure is structural: the sign structure changes character across the diagram (cup
product in the code space → interfering pre-image signs in the crossover → trivial product
phases at polarization) and no single analytic frame on Z bit strings covers it. A learned
sign network on bit strings is the known-hard case (fermionic signs are non-factorizable;
cf. the L=2 OBC decoder ceiling, `results/fermionic_obc_L2/`).

## 2. The idea: make the fermion explicit

The fermionic toric code is 3D bosonization (Chen–Kapustin) of fermions coupled to a Z₂
gauge field. In the **σ^x basis** every term of our Hamiltonian is a standard fermion–gauge
term, and the fermion occupations are read off the bit string:

| our operator | in the X basis | meaning |
|---|---|---|
| A_v = ∏_{e∋v} σ^x_e | diagonal, = (−1)^{n_v} | Gauss law: **n_v = (1 − A_v(x))/2 is the fermion number at v** |
| hx σ^x_e | diagonal | gauge electric term (the axis that broke the Z-basis stack is now trivial) |
| hz σ^z_e | flips x_e, toggles A_v at both endpoints | fermion hopping c†_v c_v' + pairing c†_v c†_v' along e, times a gauge string |
| hy σ^y_e | same flip with phase i·x_e | the same bilinear with a phase |
| B̃_p = Z⁴ X_{e+} X_{e−} | flips the 4 plaquette edges × sign x_{e+} x_{e−} | magnetic plaquette dressed by the two decoration edges; the dressing is what makes the charge a fermion (`tc3d/fermionic_decoration.py`) |

Consequences:
- The h=0 code space is the **no-fermion sector** (all A_v = +1), and the ground state is
  ∏_p (1+B̃_p)|x=+1⟩ up to the usual boundary/flux bookkeeping: a sum over closed magnetic
  configurations with signs from the decoration. (Check at L=2 OBC ED whether this sign is
  again the cup product in the dual cell complex; expected yes — it is the same Arf-type term.)
- The three polarized endpoints are product states in this basis: hx→∞ is |x=sgn hx⟩ (one
  bit string), hz→∞ is uniform positive over all x (|z=+⟩ = ∏(|+⟩+|−⟩)/√2), hy→∞ is a
  product phase i^{#(x=−1)}. All are trivially representable.
- hz, the term that is "easy" in the Z basis, is now the term that moves fermions; the sign
  it generates is the **fermion exchange sign**, and that is exactly what a Pfaffian
  reproduces without learning.

## 3. The ansatz

For a bit string x with fermion set F(x) = {v : A_v(x) = −1} (even size by construction),

    ψ(x) = exp[ CNN_gauge(x) ] · Pf[ M(x) ],        M_{vv'} = f_θ(v, v'; x) antisymmetric, v,v' ∈ F(x)

- **CNN_gauge**: the existing `ToricCNN_gridinv` trunk on X tokens (the bosonic dual lane
  already does this: `--dual_basis` star tokens), output real or complex. It carries the
  bosonic gauge amplitudes (flux loop fugacity, electric polarization).
- **Pf[M]**: a neural Pfaffian (BCS/pairing form) over the fermions present in x. The pairing
  function f_θ(v, v'; x) takes the relative position (gridinv-symmetric) and a
  gauge-invariant dressing: the product of x along a canonical path from v to v' (or a small
  learned combination of such strings), so the pairing amplitude sees the gauge background.
  Antisymmetrize explicitly, M_{vv'} = g(v,v';x) − g(v',v;x). Empty F → Pf = 1.
- **Signs**: fermion exchange sign exact from Pf; the gauge sign either (i) the h=0 analytic
  frame transported to the X basis (preferred: keeps the "SR learns amplitudes only" property
  of the current stack) or (ii) learned by a complex CNN_gauge. Start with (i) if the L=2
  check in §2 confirms the cup form; (ii) is the fallback.
- **hy**: enters only as phases on the hopping; a complex f_θ (or complex CNN_gauge) covers it.
  No σ^y rotation, no third frame.

Cost: Pf of a |F|×|F| matrix per sample, |F| ≤ #vertices (27 at L=3, 64 at L=4); O(|F|³)
with a log-Pfaffian via Parlett–Reid or Householder tridiagonalization, differentiable in JAX
(NetKet has no Pfaffian primitive; write one, ~60 lines, test against `scipy`/pfapack on random
antisymmetric matrices). Negligible next to the CNN at these sizes. Sampling: cluster/star
moves in the X basis as in the dual bosonic lane plus single flips (hz connects sectors).

## 4. Why this should cover the whole plane

- h=0: exact if the gauge sign frame is exact (the Pfaffian is trivial, F = ∅).
- hz axis: pairs of fermions on neighbouring vertices with a pairing amplitude ~hz/ΔE; the BCS
  Pfaffian is exactly the natural form for a paired fermion gas, through the electric
  transition (fermion condensation) to the z-polarized product state.
- hx axis: diagonal field; the CNN alone handles it, including the first-order magnetic
  transition, exactly like the bosonic dual lane does.
- mixed and hy: both ingredients present simultaneously; no frame change needed anywhere.

The one thing that is *not* guaranteed is trainability near a continuous critical point with
both fields on (Pfaffian NQS are known to work there in Hubbard-type models, but this is a
gauge theory). That is the research question; everything else is engineering.

## 5. Staged plan (each stage has an exact certificate)

1. **Dictionary check (L=2 OBC ED, local is fine):** build H in the X basis, verify
   A_v = (−1)^{n_v} bookkeeping, that hz σ^z_e acts as hopping+pairing with the decoration
   sign, and that the h=0 ground-state sign in the X basis is a cup-type form (write
   `sign_frame` for the X basis or reuse the Z one on the dual complex). Certificate: fidelity
   1 of the frame-rotated state with ED at h=0.
2. **Pfaffian module:** `tc3d/pfaffian.py` (log|Pf|, sign, JAX-differentiable) + unit test.
3. **Ansatz:** `ToricCNN_gridinv` + Pfaffian factor as a new arch (flag `--pfaffian`), X basis,
   gauge frame fixed. Certify at L=2 OBC against ED on the hz axis (where the current stack is
   exact, so any loss is the Pfaffian's), then on hx (where the current stack fails), then at
   a mixed point and at hy ≠ 0.
4. **L=3 OBC vs DMRG:** rerun the notebook axes (`analysis/notebooks/dmrg_L3_benchmark.ipynb`,
   add a lane "P"); the pass criterion is the existing `matches()` rule. Then ask the
   collaborator for mixed (hx,hz) DMRG points.
5. Only then: L=4 and the phase diagram.

Adversarial audit gate (CLAUDE.md) applies after stage 3.

## 6. Cheaper alternatives, kept for reference

- **Gated two-arm T⁺**, ψ = e^c ψ_topo + ψ_pol: endpoints exact, resolves first-order jumps, not
  a continuous crossover. Worth doing for the magnetic transition only if the Pfaffian route
  stalls.
- **Basis-swap arms** (Z-basis stack for hz + weak hx, the same construction rebuilt in the X
  basis for hx + weak hz): duplicates the machinery, still leaves a gap near a two-field
  critical region. Not recommended.
