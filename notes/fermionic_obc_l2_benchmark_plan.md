# fTC L=2 OBC benchmark plan — NQS (sign-aware) vs dense ED, fully local

Status: APPROVED 2026-08-19 (all open points resolved by user, see §Decisions).
Branch: `feat/fermionic-obc`.
Everything below runs on the dev Mac — no cluster involvement (Perlmutter down).

## Goal

Certify the sign-aware fermionic architecture against exact results at finite
external fields. Venue: **3D fermionic toric code, L=2, OBC** (12 edge qubits,
2^12 = 4096 states — dense ED is trivial locally; unique gapped GS at h=0,
E0 = −14, gap 4, verified 2026-08-19).

**Benchmark rectangle** (h_x, h_z): `(0,0)`, `(0.2,0)`, `(0,0.2)`, `(0.2,0.2)`.
Tiered difficulty by construction: h=0 (head should be exact) → electric
(0,0.2) (head exact if the superselection argument transfers to OBC) →
magnetic (0.2,0) and mixed (0.2,0.2) (head approximate — the actual test).

**Protocol:** at each point, train NQS **150 steps** with observable
**snapshots every 25 steps** (steps 25..150); dense ED at the same points;
compare **energy + local observables** (magnetizations, stabilizers) via
relative error and check convergence across snapshots.

## Phase 0 — OBC decoration in `tc3d` (prereq, code)

Adopt the **truncated decoration** as the OBC model (decision 2026-08-19):
keep whichever of the two body-diagonal σ^x partners exists; boundary B̃_p
become 5-body ZZZZX. Verified already (scratch scripts): 0 commutation
violations at L=2/3/4; coincides with PBC decoration on all interior faces;
L=2 h=0 model is consistent (rank 12 → unique GS at −14).

1. `fermionic_decoration.py`: make `_idx`/`fermionic_plaquettes` bc-aware —
   OBC: no wrap, skip incomplete faces, keep existing diag partners; PBC path
   byte-identical to today.
2. Re-derive `flux_constraint_masks` on the truncated stabs (code is generic
   over the stabs list; verify what it returns at OBC — 6 tokens → ? masks).
   Outcome decides whether `--flux_penalty` + `--chains_up` are needed at OBC
   at all (if the constraint structure trivializes, drop both; if ghost
   sectors exist, keep κ=6 per the PBC precedent).
3. `builders.py`: remove any PBC assumption in the fermionic branch
   (H + sampler cluster rules must consume the OBC stabs cleanly).
4. Gates: `verify_xz_commutation` = 0 violations; h=0 dense ED reproduces
   E0 = −14, unique GS; existing PBC tests untouched (regression).

## Phase 1 — analytic sign head at OBC (h=0)

Derive the token-quadratic C-form on the OBC masks via the existing GF(2)
construction (`prefit_phase_head.py --analytic_C`; un-hardcode `bc="PBC"` at
:89). **Certificate is stronger than at PBC:** check the head sign against
the dense-ED ground state on the FULL 4096-dim space (not a 10k-sample
audit — exhaustive).

Fallback if the analytic form does not transfer: solve the exact h=0 sign map
from the ED wavefunction directly (GF(2) least squares over token bilinears —
trivially conditioned at 4096 states). Either way the deliverable is a frozen
prefit init `results/fermionic_obc_L2/prefit_anaC_L2_OBC.mpack` + a
pass/fail certificate.

## Phase 2 — ED referee at the 4 points

Extend `analysis/scripts/ed_electric_line.py` (or a sibling
`ed_ftc_point.py`) with `--hx` and `--bc OBC`; at 2^12 switch to **dense
`eigh`** (full spectrum: exact gap, exact degeneracy — kills the eigsh-k=1
contamination trap that poisoned the PBC hz=0 row).

Per point, bank: `E0`, gap, per-site `M_x`, `M_z`, mean and per-operator
`⟨A_v⟩`, `⟨B̃_p⟩` (helpers exist: `exact_diag.expect_x_string` /
`expect_xz_string`), plus the full-space **sign vector in the head-token
frame** and the GS vector itself (small .npz, gitignored) for fidelity /
sign-match metrics. JSON schema: compatible with `tc3d/validation.py`'s
`exact_diag_*.json` reader so the automated NQS-vs-ED harness works.
Output: `results/fermionic_obc_L2/ed_L2_OBC_rect.json`.

Anchors: (0,0) must give exactly −14, unique GS, M_x = M_z = 0,
⟨A_v⟩ = ⟨B̃_p⟩ = 1. Hermiticity + eigsh-vs-eigh cross-check at one point.

## Phase 3 — NQS trainings (local CPU)

Config per point: `--model fermionic --bc OBC --L 2`, arch `ToricCNN_gridinv`
(complex), `--phase_head_frozen --init_from <Phase-1 prefit>`, flux
penalty/chains_up per Phase-0 outcome, `--n_iter 150 --snapshot_every 25`.
Sampler/optimizer: start from the fermionic-ladder L=2 recipe (n2x4 nh4-8
inv8-8 k=2 heritage); calibrate samples + s/step with a 10-step smoke first.
One seed per point initially.

Observables at snapshots — **two evaluation modes**:
- **Exact full-space eval** (unique to this system size): contract ψ_NQS over
  all 4096 configs → noise-free E, M_x, M_z, ⟨A_v⟩, ⟨B̃_p⟩, plus
  **fidelity |⟨ψ_NQS|ψ_ED⟩|²** and weighted sign-match. Primary convergence
  diagnostic.
- **MC-sampled eval** (production-representative): the same observables
  through the standard sampled estimators, so the benchmark also certifies
  the estimator pipeline, not just the ansatz.

Optional control (recommended, +1 run): sign-blind ansatz (no head) at
(0.2, 0.2) to show the sign head is doing real work at the hardest point.

## Phase 4 — comparison + report

Per point × snapshot: relative error |O_NQS − O_ED| / |O_ED| for E and for
observables with |O_ED| above a floor; **absolute error where O_ED ≈ 0**
(M_x on the h_x=0 edge, M_z on the h_z=0 edge, both at the origin — relative
error is ill-defined there). Convergence = flat last-2-snapshot observables +
final-block E consistent with ED in σ units (Phase-A criterion; the
Vscore≤1e-7 gate is h=0-only).

Deliverables: snapshot-eval JSONs + final-state JSONs + ED refs committed
(per-run commit policy); summary notebook
`analysis/notebooks/fermionic_obc_L2_benchmark.ipynb` (savefig commented out)
with E/observable convergence panels + NQS-vs-ED relative-error table.

## Audit gate (CLAUDE.md) — RESULTS 2026-08-19

(i) **Model/sampler audit**: code fully validated (both H constructions
identical to 5.6e-17 on 2^12; orderings/token conventions exact end-to-end;
sampler kernel clean; cluster orbit = h=0 support, LocalRule gives full-space
ergodicity at finite field). ONE CRUCIAL protocol finding: **fixed κ=6 at
h_x≠0 is a designed-in variational floor** — the flux sector is not conserved
under σ^x, the true GS carries violated-sector weight 4.8e-3/1.5e-2/2.66e-2
at h_x=0.1/0.15/0.2, and hard sector projection costs
ΔE_proj = +1.79e-2 (0.1,0) / +9.05e-2 (0.2,0) / +9.03e-2 (0.2,0.2), ≈
1.8–2.3·h_x², h_z-independent (electric line: <4e-15, κ=6 stays exact+
mandatory there). No estimator bias — ⟨H⟩ stays a true upper bound; the
penalty restricts the variational family (e^{-24} sector suppression at L=2
OBC, where the 2 token masks are ONE physical constraint bit, u0≡u1).
**Protocol amendment:** at h_x≠0 points run BOTH κ=6 (expect ≈ E0_proj) and
κ=0 (targets true E0 — the genuine sign-aware test); interpret against both
references. Projected-E0 references: (0.2,0) → −14.06606, (0.2,0.2) →
−14.16682, (0.1,0) → −14.01576.
(ii) **ED referee audit**: zero CRUCIAL/MAJOR — all 4 points independently
recomputed (from-scratch kron H) to ≤7e-15; decorated-vs-bare unambiguous;
training operator ≡ referee operator across the full spectrum; npz vectors
sound; schema resolves. Two minor legacy-path regressions FIXED + re-verified
(incremental --out dump restored; observables gated to dense mode;
byte-identical artifact rerun).
(iii) **Full-space eval**: self-gated (⟨ψ_ED|H|ψ_ED⟩=E0 to 5e-15; frozen-head
init gives sign_match=1.0, fidelity=0.5000=2048/4096 exactly); focused audit
to run alongside Phase 3.

## Run matrix (amended)

| # | (h_x,h_z) | head | κ | note |
|---|---|---|---|---|
| 1 | (0, 0) | frozen anaC | 6 | head exact (theorem) |
| 2 | (0, 0.2) | frozen anaC | 6 | electric: head+penalty exact |
| 3 | (0.2, 0) | frozen anaC | 6 | expect ≈ E0_proj = −14.06606 |
| 4 | (0.2, 0.2) | frozen anaC | 6 | expect ≈ E0_proj = −14.16682 |
| 5 | (0.2, 0) | frozen anaC | 0 | targets true E0 = −14.15654 |
| 6 | (0.2, 0.2) | frozen anaC | 0 | targets true E0 = −14.25710 |
| 7 | (0.2, 0.2) | none (control) | 0 | sign-blind, cold start |

All: L=2 OBC, ToricCNN_gridinv complex k=2 n2x4 nh[4,8] inv[8,8],
--chains_up, --init_from prefit_anaC_k2_L2_OBC (runs 1–6), 150 steps,
--snapshot_every 25, seed 0, local CPU.

## Decisions (user, 2026-08-19)

1. **Truncated decoration = the OBC model.** Confirmed.
2. **Both eval modes:** exact full-space contraction AND MC-sampled
   estimators at every snapshot.
3. **Sign-blind control run at (0.2, 0.2): include** (5 runs total).
4. **1 seed per point** first pass; add seeds only where convergence looks
   marginal.
5. Observable set: E, M_x, M_z, ⟨A_v⟩, ⟨B̃_p⟩ — banked as means AND
   per-operator (default, unobjected).
