# Research blog — toric-code-nqs

Living log of research progress. Newest entries first. Supersedes
`notes/log_and_plan.md`, which is frozen as the historical design record of the
architecture-extension era (old-layout paths there resolve at tag `2d-final`).

## Current goal — NQS hyperparameter tuning in the dual basis

The active work is **track 1**: tune the dual-basis NQS
(`python -m tc3d.train --dual_basis` — Hadamard-rotated inputs + star-token
`ToricCNN_gridinv`) in the sign-problem-free regime, across system sizes.

- **Accuracy yardstick:** `--ref_E/--ref_sig` streams the signed per-step gap
  against the QMC benchmark. Canonical point so far: L=4 OBC,
  (h_x=0.2, h_z=0.2), combined ParaToric reference **E = −174.5957(147)**
  (`results/qmc_hx0.2_hz0.2/`; PMRQMC cross-check at h_z=0.1).
- **Status:** tune-rect campaign COMPLETE (2026-08-06, below) — canonical
  architecture locked: **dual · nh(4→8) → inv(8,8) · 15-tap · kernel=L−1 ·
  dt=0.02→0.002 · ds=1e-3 (3e-3 at strong field for L=5)**; validated vs QMC
  at 4 points × L∈{4,5,6}. Summary: `analysis/tune_rect_summary.ipynb`.
- **Track 2 (support):** QMC validation via `analysis/paratoric_driver.py`
  (ParaToric primary; `--validate` ladder mandatory) +
  `analysis/export_pmrqmc.py --verify` (PMRQMC cross-check). New benchmark
  points get generated here as tuning moves through the phase diagram.

---

## 2026-08-19 (overnight) — fermionic h=0 architecture ladder: CNN vs approx-symm vs sign-head, L=2/3/4 × 3 seeds — the presentation figure

**Headline: the three-tier hierarchy is now measured, multi-seed, at three sizes,
in one figure (`analysis/figs/fermionic_arch_ladder.png`, built by
`analysis/fermionic_arch_ladder.ipynb` from `results/fermionic_ladder/`).**
Campaign: `nersc/launch_fermionic_ladder.sh` (prefit/smoke/full stages; 45
completed runs + 2 banked-diverged, ~30 GPU-h shared QOS, production gated on 3
L=2 smokes via `--dependency=afterok`). Protocols reproduce the historical ones
exactly (cold tiers: opened guard 1e6/50, dt=0.02, k=2; sign-head: stock guard,
frozen anaC head + fp6 + chains_up, `--init_from` prefit).

| tier | L=2 rel.err | L=3 | L=4 |
|---|---|---|---|
| GeoCNN (symmetry-unaware, complex) | 0.36–0.40 | 0.48–0.53 | 0.50–0.53 |
| gridinv, no head (approx-symm) | 0.296–0.306 | 0.35–0.50 | 0.49–0.50 |
| gridinv + frozen analytic sign head | **2.2e-8–1.3e-7** | **1.1e-7–3.7e-7** | **1.4e-8–5.8e-8** |

- **Sign-head: exact at every size, every seed, ~size-independent** (9/9 runs,
  100 SR steps each; seed spread from prefit trunk inits — new `--seed`
  threading in `analysis/prefit_phase_head.py`, cfg previously dropped it).
  L=2 trap −22.521 reproduced to 6 decimals by the no-head tier; L=3/4 traps
  are coset-dependent (−54.0 attractor = ⟨A⟩=1, ⟨B̃⟩=1/3; deeper basins to
  −70.2/−131.5), and asymm's best seed beats CNN's best at every L.
- **GeoCNN complex enablement** (2 lines in `builders.py`, adversarially
  audited + empirically verified: complex grads flow, NetKet jacobian mode
  identical to gridinv). GeoCNN trains stably at (4,4,4); the (8,8)
  wide-shallow variant **diverges 3/3 seeds under the opened guard** at L=2
  (one blow-up hit spread 7e72) and 2/5 even under the stock guard — when it
  survives it lands exactly on the (4,4,4) plateau (−19.28..−19.31).
  gridinv inv(2,2,2) similarly needs the stock guard at L=3 (opened-guard runs
  end sign-corrupted, Im⟨E⟩~5). Cold fermionic + opened guard is safe ONLY for
  the canonical widths — g-variant reruns are the banked data, opened-guard
  failures kept as instability evidence (loader skips `diverged=True`).
- **Audit gate earned its keep again**: the pre-launch adversarial pass caught
  that a missing prefit checkpoint silently cold-starts with θ=0 — a no-op
  sign head still labeled `phase_head_frozen` (now hard-gated in the launcher).
  Also caught pre-launch: jobs must pin `REPO` to the campaign worktree or they
  import the wrong-branch tc3d.

Branch: `feat/fermionic-arch-ladder` (worktree; main checkout untouched).

---

## 2026-08-15/16 (night) — Right-cut instability traced to the LR schedule, then fixed with dt=0.01; multi-L convergence confirmed to grow with L; cross-session collaboration with a peer working the identical L=6 problem

**Headline: the warmup+gentler-floor schedule tested as a follow-up to the
2026-08-14 ablation actively destabilizes the first-order (right-cut)
transition — a real, guard-invisible energy "bump" the old schedule never
had — and halving `dt` (0.02→0.01) fixes it cleanly, in one case (L5,
h_x=0.85) also dramatically improving QMC agreement, not just stability.**

**Run C — multi-L (L=4/5/6) + 5% warmup + floor=20%·dt, 500 steps
(`--warmup_frac`/`--guard_warmup`, new opt-in flags in `tc3d/train.py`/
`builders.py`).** Confirms training-length requirement grows with L on the
continuous (up-cut) transition: L=4 converges to ≤3σ by step ~200-350 (new
schedule slightly faster than old), L=5 plateaus around 8-9σ by step 500
(not converged), L=6 is still falling at 4.6σ by step 500 (not converged
either, though closer). At the first-order (right-cut) transition the new
schedule is **worse than the 2026-08-14 baseline**: h_x=0.80 (L4) shows a real
non-monotonic energy regression (E worse from step 100→150, −197.08→−196.79)
that the old schedule's curve never showed anywhere; h_x=0.85 (L5) is far more
dramatic — E swings −416.4→−391.5 over steps 72-80, spread jumping 3.7× over
its local baseline, still under the guard's 10× trigger so **entirely
invisible to the divergence guard**. `chains_up` was confirmed OFF in these
runs, ruling that out as the mechanism; likeliest explanation is the cold-init
network's near-zero-order default state persisting longer under warmup before
real learning kicks in — free bonus where OP≈0 is still correct, a debt to
repay abruptly where it isn't.

**Run D — fixing the right-cut instability (L4 h_x=0.8, L5 h_x=0.85, 500
steps each), run overnight in direct collaboration with a peer Claude Code
session (`toric-code-nqs-c5`) independently chasing the identical phenomenon
at L=6.** Split the work by L to avoid duplicate shared-QOS compute (peer:
diag_shift-up + dt-down at L=6 h_x∈{1.05,1.175,1.25}; here: dt-down +
LR-floor-tightening at L4/L5) and agreed a shared numeric bar with the user
mid-investigation (kick ≤0.10% of |E| acceptable, ≤0.05% ideal, checked via
**both** curve smoothness *and* rollback count — not either alone).

- **dt=0.01 (no warmup, floor unchanged at 10%·dt) is the clean, verified
  fix.** Zero guard rollbacks at both L4 and L5. At L4 it restores exactly the
  old schedule's own baseline stability (~0.10% kick) without moving QMC
  convergence. At L5 it does *both*: kick stays at the noise floor **and**
  QMC agreement improves sharply — B_p pull 37.9σ→8.5σ, σ_x pull
  −38.3σ→−10.8σ, order-parameter pull −15.4σ→**−3.6σ** (nearly inside the
  ±3σ consistency band). n=2 points, so not claiming this scales with L, but
  it is the one lever that helped in every test tonight without a single
  counterexample — including on the peer's independent L=6 runs, where their
  diag_shift-up lever made the visible kick at h_x=1.175 **2× worse** while
  simultaneously making the QMC gap 2.4× *better* — the sharpest demonstration
  yet that kick-size and QMC-convergence are two separate axes, not one.
- **Floor-tightening (5%·dt, dt unchanged) is a trap, not a fix — caught by
  checking rollback count, not just the surviving curve.** L4 was clean;
  the *identical* config at L5 needed **26 separate guard rollbacks in the
  first 54 steps**, with energy swinging as low as **−1044** against the
  exact physical minimum of −365 — a genuinely corrupted trajectory, not
  noise — before settling into smooth descent from step ~60 onward and
  landing in a good final basin anyway. The final numbers looked fine; the
  path there was a lucky escape from real divergence, and a different seed
  could easily not recover. The `is_bad_step` cosine-decay math says the
  floor value barely affects `dt` this early in training regardless — so this
  is likely stochastic (the same GPU float non-associativity the peer
  independently flagged), not a deterministic floor effect. Lesson: a single
  run per config isn't enough to certify a fix as safe; needs a
  multi-seed failure-rate check before this becomes a default, not one draw.
- A pull-computation bug (wrong error-field derivation for `O_FM_membrane_R1`
  specifically — its error key is a plain `_err` suffix, not the `_mean`→
  `_err` substitution that works for every other observable) produced a
  falsely optimistic −0.5σ read before being caught and corrected to the real
  −19.2σ — worth remembering when writing quick one-off pull scripts instead
  of reusing `analysis/ablation_report*.py`'s explicit key-pair tables.

**Process note:** the cross-session collaboration held up well over several
hours unattended — shared vocabulary (kick-%, rollback count, the numeric
bar), each side flagging its own mistakes to the other (the pull bug here,
"warmup_frac was on our shortlist too, dropping it" from the peer) rather than
either side declaring victory on partial evidence.

---

## 2026-08-14 (later) — Decoder-extended sign head: Stage 1 complete, go for Stage 2; cross-session puzzle with 2D fTC peer resolved (twice)

**Decoder-extended sign head (fermionic, off-h=0-support signs).** User's proposal:
extend the exact analytic h=0 sign (currently token-quadratic, exact only on the
stabilizer orbit of |0…0⟩) to arbitrary off-orbit configurations via a canonical
QEC-style decoding — decompose σ′ = error · stabilizers · |0…0⟩, transport the sign
through the algebra. Worked out analytically first: this collapses to a **single
frozen GF(2)-quadratic form in the raw spin bits** (RREF of the generator image with
preimage tracking → a commutation-derived quadratic form pulled back through it) —
same cost class as the existing token-quadratic head, no decoder ever runs at
inference time. Two gauge conventions considered (error-applied-last vs
error-applied-first); design + validation ladder in `notes/decoder_sign_head_plan.md`.

- **Stage 1 prototype** (`analysis/decoder_sign_prototype.py`, pure bitmask, no
  NetKet/ED): built, then two adversarial audit rounds. First caught a real
  sampling bug (weight-1/weight-2 sector filters checked syndrome≠0 instead of
  popcount==1, contaminating ~25–45% of the pool) — fixed, numbers re-banked
  (`results/fermionic_h0/decoder_gauge_L{2,3,4}.json`). Second audit round proved
  `error_last`'s match is **exact by construction, not empirical** (the RREF
  reduction path for a weight-1 state and its designated parent are byte-identical),
  confirmed via literal brute-force operator application (0 mismatches, both
  gauges, L=2/3).
- **L=2→3→4 trend** (the actual go/no-go): `error_last` beats `error_first`
  decisively at every size (error_first never below ratio ~0.96 — ruled out as a
  convention). Against the current production (token-quadratic) head, PT-weighted
  violation ratio splits by channel: **stabilizer (B̃_p) channel improves with L**
  (0.69→0.58→0.49), **generic-flip (σx) channel erodes but never crosses 1**
  (0.75→0.83→0.89) — tracks a real, gauge-independent **frustration floor** (some
  off-orbit configs have disagreeing valid decodings; frustration fraction climbs
  50%→75%→88% across L=2/3/4, an intrinsic model property, not a gauge defect).
  `error_last` is exact at L=2 and sits within 6–8% of its own provable ceiling at
  L=3/4.
- **Verdict: proceed to Stage 2** — wire `error_last` into `tc3d/networks.py` as a
  new frozen bit-quadratic head (mirrors `phase_head_frozen`'s pattern).
- **New design, not yet implemented:** rather than suppress amplitude at frustrated
  configs (flux_kappa-style), **gate only the head's phase contribution to zero**
  there and let the trunk own the whole phase — the frustration flag itself is
  cheap (reuses `error_last`'s own formula, evaluated at two fixed shifted inputs
  per residual class; a precomputable partner-edge lookup, not a per-sample
  search). Decided against also exposing the flag as an explicit trunk input for
  now (implicit only). Rationale: a wrong committed prior is worse for SR than a
  blank slate (matches the h=0 sign-trap lesson: SR pays for sign defects, never
  fixes them).

**Cross-session: 2d-tc-06 (2D fermionic toric code, peer session on this machine).**
Reported CNN converging to exact ED (E=−25.0000, L=4) "for free" on what looked
like a sign-problem-full model — proposed and sent a Kasteleyn/even-odd-L parity
hypothesis, **which they correctly refuted**: proved `A'_v·B_NE(v) = A_v` as an
operator identity, so their h=0 stabilizer group is literally identical to the
plain (undecorated) toric code's — the "free" convergence was an
identity-initialized architecture starting at an already-known-trivial answer, no
optimization involved.

- **The real, still-open puzzle they found instead:** their hx-perturbed cut
  (hx∈[0.1,1.5], L=2/3) has an exactly sign-free ground state despite H being
  provably, structurally non-stoquastic. Jointly verified by building their model
  from exact generator masks on both ends: genuine positive off-diagonal elements
  exist (traced to dressed-star terms — an off-diagonal sign equal to the pre-flip
  flux of the star's own partner plaquette); **no global diagonal gauge fixes it**
  (their exhaustive 65,536-gauge search at L=2 + our independently-found frustrated
  cycles at both L=2 and L=3 — gauge-equivalence to stoquastic, our own 3D
  mechanism, is fully ruled out); the frustrated configurations carry
  non-negligible ground-state weight (not a corner the GS avoids). Mechanism
  unknown to either session. Practical read: their Phase-4 NQS probably doesn't
  need a complex/sign-aware ansatz — they're confirming with an actual L=4 ED
  point before committing (correctly declined to act on the L=2/3 extrapolation
  alone).
- **Separately worked out (not yet relayed) why 2D h=0 is trivial while 3D h=0
  isn't:** 2D's dressed star borrows a plaquette that's *already* independently
  pinned to +1 by the same stabilizer set, so within the physical sector it
  collapses back to a bare (sign-free) star. 3D's decorated plaquette *replaces*
  rather than supplements the bare plaquette — there's no independently-fixed
  partner for its Z-boundary to cancel against, so its sign is irreducible.
  Supersedes the (wrong) Kasteleyn-dimensionality guess as the real explanation
  for h=0; doesn't touch the still-open hx-cut mystery. Worth sending to
  2d-tc-06 next session.

---

## 2026-08-14 — Phase B order-parameter gap explained: training length fixes the continuous transition, cannot fix the first-order one, and depth is not a shortcut

**Headline: Phase B's stabilizer/magnetization/order-parameter mismatch (energy
fine, everything else off by 10–100σ near both transitions) is the CAMPAIGN.md
guard-blind-spot showing up in order-parameter space, not a QMC bug — and the
fix depends entirely on the transition's order.** `analysis/phaseB_summary.ipynb`
(built by a peer session from the completed 86/87-point campaign) showed energy
agreeing between QMC and NQS almost everywhere, but stabilizers, magnetization,
and the order parameters disagreeing sharply near h_z≈0.22–0.30 (up cut,
electric, continuous transition) and h_x≈0.65–0.95 (right cut, magnetic,
first-order transition), at every L, growing worse with L. A read-only audit of
`tc3d/hamiltonian.py`/`validation.py`/`fm.py` against `analysis/paratoric_driver.py`
confirmed every observable definition matches between the two pipelines
(stabilizers, magnetization, and both order-parameter families use the same
rep_x/rep_z convention on both sides) — ruling out a measurement/mapping bug.
The known pending "dual-magnetic O_FM dispatch bug" (task #9) lives in
`train.py`'s inline eval block, which Phase B never runs (`--final_eval_rounds
8` always skips it) — irrelevant here.

Four L=4 points were picked by ranking actual pull (not eyeballing the plots):
**(h_x=0.2, h_z=0.26), (0.2, 0.28)** on the continuous up cut and
**(h_x=0.75, 0.1), (0.8, 0.1)** on the first-order right cut — the latter pair
the worst in the whole campaign (B_p/σ_x off by 0.09–0.17 in raw value). Two
ablations ran at all four: **Run A** — same winner architecture, 750 steps
instead of 150, with a new opt-in `--snapshot_every` flag (`tc3d/train.py`)
persisting a never-overwritten weights snapshot every 50 steps, replayed
post-hoc through `analysis/eval_snapshots.py` (reuses `pooled_final_observables`
exactly, so every snapshot's stats are directly comparable to a normal
campaign point). **Run B** — the normal 150-step budget, but deeper
(`noninv_hidden=(4,8,8)`, `inv_hidden=(8,8,8)`) — a combination the tune-rect
campaign never tested at a hard corner, only at the home point.

| point | transition | order-param pull @150 (Run A) | @750 (Run A) | converges ≤3σ by | Run B @150 (deeper) |
|---|---|---|---|---|---|
| h_z=0.26 | continuous | +8.0σ | +2.4σ | step ~250 | +17.2σ (worse) |
| h_z=0.28 | continuous | +7.9σ | +0.7σ | step ~350 | +14.0σ (worse) |
| h_x=0.75 | first-order | −28.3σ | −24.0σ | never | −25.2σ (~same) |
| h_x=0.80 | first-order | −27.4σ | −18.5σ | plateaus ~−15σ to −19σ from step ~300, never converges | −62.1σ (much worse) |

**Training length resolves the continuous transition** — both up-cut points
fall inside ±3σ by step ~250–350, so the campaign's 150-step default is simply
short there, not wrong. **Training length only partly helps the first-order
transition and barely at all at h_x=0.75**: energy stays fine (|pull|<3) the
entire 750 steps at both right-cut points while the order parameter stays
stuck tens of σ off — a cold start lands on a locally competitive energy on
the wrong side of the first-order jump, and more SR steps from that *same*
starting point cannot cross back out; this needs a different remedy (multiple
seeds, annealing/warm-start), not more of the same optimizer. **Depth is not a
shortcut**: at the matched 150-step budget the deeper network is worse than
the shallow one at 3 of 4 points (worst at h_x=0.80, where it barely moves off
its own step-50 values) — more parameters need more SR steps to converge in a
fixed budget, not fewer, and depth does nothing for the first-order basin
problem regardless.

Full per-step table + convergence charts (canvas, all four points):
`analysis/ablation_report.py`; code additions: `--snapshot_every` in
`tc3d/train.py`, `analysis/eval_snapshots.py`. Data:
`results/phaseB_ablationA/`, `results/phaseB_ablationB/` (both local, not yet
committed — `results/phaseB/` itself is still untracked pending a consolidated
commit).

---

## 2026-08-12 — Phase B COMPLETE: full L=4 electric-line table, 10/10 points; hz≥0.7 divergence traced to dt, not diag_shift

**Headline: the electric-line recipe scales cleanly to L=4 across the whole
h_z∈[0.1,1.0] window, once one lever is turned.** Jobs 56695196/56695197
converged 7/10 points on the first pass (h_z=0.1,0.2,0.3,0.4,0.5,0.6,0.9); three
points (h_z=0.7, 0.8, 1.0) genuinely diverged — guard exhausted
`max_rollbacks=5` each time, signature E≈−256→≈−168 by step 2, full blowup by
step 3–4. A seed=1 retry reproduced the seed=0 trajectory bit-for-bit through
the pre-divergence steps (`--chains_up` resets every chain to the deterministic
all-up state regardless of seed, so seed alone never explores a different
path) — ruling out stochastic tail-risk and pointing at the optimizer step
itself. Two isolated single-variable retries settled it: **dt=0.01 (half the
default 0.02) fixes all three points cleanly**, with diag_shift left at the
unchanged 1e-3; a parallel ds=3e-3 retry (motivated by the bosonic L=6/7
session hitting an identical pattern, fixed the same way) also converged
hz=0.7 cleanly but to a ~2× higher Vscore than the dt fix, so dt=0.01 became
the canonical recipe for h_z≥0.7 and the ds=3e-3 branch was kept only as a
backup (never promoted). All 10 points banked via the early-stopping rule
(step>100, Vscore plateaued, energy flat over the last 20 steps) rather than
run to full `n_iter=250` — GPU-hour savings ranged 12–44% per point.

| h_z | E0 | E_err | Vscore | M_z | M_x | ⟨A_v⟩ | ⟨B̃_p⟩ | dt | banked@ |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | −256.239809 | 4.4e-4 | 2.1e-6 | 0.0266 | 6.5e-12 | 0.9979 | 0.9993 | 0.02 | 150 |
| 0.2 | −256.960421 | 6.1e-4 | 2.8e-6 | 0.0484 | 6.5e-12 | 0.9930 | 0.9977 | 0.02 | 250 |
| 0.3 | −258.167166 | 1.0e-3 | 8.4e-6 | 0.0776 | 6.5e-12 | 0.9821 | 0.9940 | 0.02 | 170 |
| 0.4 | −259.870552 | 1.5e-3 | 1.9e-5 | 0.1011 | 6.5e-12 | 0.9699 | 0.9898 | 0.02 | 220 |
| 0.5 | −262.071503 | 2.0e-3 | 3.2e-5 | 0.1284 | 6.4e-12 | 0.9516 | 0.9836 | 0.02 | 190 |
| 0.6 | −264.788964 | 2.6e-3 | 5.9e-5 | 0.1550 | 6.4e-12 | 0.9303 | 0.9760 | 0.02 | 140 |
| 0.7 | −268.056137 | 3.9e-3 | 1.1e-4 | 0.1821 | 6.4e-12 | 0.9048 | 0.9671 | **0.01** | 160 |
| 0.8 | −271.842319 | 6.8e-3 | 3.0e-4 | 0.2129 | 6.4e-12 | 0.8735 | 0.9543 | **0.01** | 180 |
| 0.9 | −276.265446 | 7.0e-3 | 3.1e-4 | 0.2448 | 6.4e-12 | 0.8370 | 0.9395 | 0.02 | 250 |
| 1.0 | −281.290740 | 1.1e-2 | 6.8e-4 | 0.2822 | 6.4e-12 | 0.7896 | 0.9197 | **0.01** | 170 |

(h_z=0.2 and 0.9 ran to full n_iter rather than early-stopping — they converged
in the first pass, before the banking mechanism existed.)

**Sanity checks all pass.** E0 sits below the exact h=0 anchor
(−4L³ = −256 for L=4 PBC) at every point and decreases monotonically with h_z,
as it must (h_z is a uniform negative bias on ⟨σ_z⟩). M_z rises smoothly
0.027→0.282; M_x stays pinned at ~6.4e-12 — machine zero at *every* field
strength, the expected structural signature of the frozen analytic head
keeping the state in the real diagonal-sign manifold across the whole electric
line (not just at h=0). ⟨A_v⟩ falls smoothly from 0.998→0.790 (h_z frustrates
the star term via composite defects); ⟨B̃_p⟩ falls much more gently,
0.999→0.920 (only indirectly coupled). No discontinuities or non-monotonicity
anywhere in the table.

**First transition read: no signature of a sharp transition in this window at
L=4.** Every observable — E0, M_z, ⟨A_v⟩, ⟨B̃_p⟩ — varies smoothly and
monotonically across h_z∈[0.1,1.0]; nothing kinks. That's consistent with
either a genuinely smooth crossover in this range, a transition beyond h_z=1.0,
or a second-order transition too weak to see as a kink at a single L (finite-
size rounding). Locating any h_z^c properly needs the multi-L FSS ladder
(L=2 already done for Phase A; L=3,5,6 next) — this L=4 table is one column of
that eventual grid, not a standalone transition claim.

**wandb:** all 24 offline runs (7×L=2 Phase A + 17×L=4 Phase B, including
diverged attempts and the ds=3e-3 backup) synced to wandb.ai under groups
`fermionic-electric-line-L2` / `-L4` (models-california-institute-of-technology-caltech/approx-sym-3D-TC).
Resumed/re-banked runs share a wandb run ID across resubmissions, so each
h_z point shows one continuous learning curve rather than fragments.

Artifacts: `results/fermionic_eline/` (10 final JSONs + curve.json checkpoints,
L=4); `analysis/bank_point.py` (new — re-derives the final-JSON schema from a
periodic checkpoint so a plateaued point can be banked without burning the
remaining `n_iter`, committed this session and used for 8 of the 10 points).

---

## 2026-08-11 (evening) — Phase A PASS: L=2 electric line matches ED at all 7 points; Phase B (L=4) launched

**Headline: the electric-line recipe works in practice, not just in theorems.**
Job 56679620 (tc-eline-L2, 1:14 wall) trained 7 *independent* cold-start points
(h_z ∈ {0, 0.05, 0.1, 0.194, 0.3, 0.5, 0.8}, h_x=0, PBC, frozen C-form head +
κ=6 penalty + chains_up + k=2, `--init_from prefit_anaC_k2_L2`, 8192 samples,
200 iters, ~2.2 s/step). Every point is statistically consistent with the
banked ED energies (`results/fermionic_h0/ed_L2_electric.json`):

| h_z | E_NQS | δ = E_NQS−E_ED | δ/σ | Vscore |
|---|---|---|---|---|
| 0.00 | −32.0000062 | −6.2e-6 | −0.8 | 1.3e-8 |
| 0.05 | −32.0075153 | −1.3e-5 | −0.1 | 4.7e-6 |
| 0.10 | −32.0300783 | −3.7e-5 | −0.2 | 6.0e-6 |
| 0.194 | −32.1133110 | +1.9e-4 | +0.7 | 1.3e-5 |
| 0.30 | −32.2730789 | +4.2e-4 | +0.9 | 4.2e-5 |
| 0.50 | −32.7801599 | +1.3e-4 | +0.3 | 3.6e-5 |
| 0.80 | −34.1974063 | −3.4e-4 | −0.4 | 1.2e-4 |

No systematic drift with field; M_z rises 0→0.27, M_x pinned at ~1e-11
(machine zero — the frozen head keeps the state in the diagonal-sign manifold),
⟨A_v⟩/⟨B̃_p⟩ fall smoothly (1.0→0.817/0.934). Two lessons: (i) the nominal
Vscore≤1e-7 gate is an h=0 artifact — at finite field the amplitudes are
genuinely nontrivial and the floor is set by samples×iters; the real criterion
is δ-consistency-with-ED, which passes everywhere. (ii) mid-training running
energies can sit *below* E₀ transiently (h_z=0.194 read −2.7σ at step 82 —
chain-equilibration bias); only the fresh-statistics final block is trustworthy.

**Phase B launched:** jobs 56695196/56695197 (tc-eline-L4a/b), L=4 PBC,
h_z 0.1..1.0 step 0.1, two sweeps × 5 independent cold points, same recipe with
`prefit_anaC_k2_L4`, 4096 samples, 512 chains, CHUNK 256, 250 iters, 2:45
walltime. Timing reality from the h=0 L=4 curve (~68 s/step at 8192 samples →
~35 s/step here): ~2.4 h/point, so each walltime window finishes ~1 point and
the requeue-safe sweep + resubmit loop carries the rest. Deliverable: first
transition read (M_z, stabilizers, O_FM) across the fermionic electric line —
does the fermion condense above the bosonic h_z_c ≈ 0.194?

---

## 2026-08-11 (day) — electric line first: head-rotated stoquasticity peer-verified; ③ design settles

**Headline: the finite-field campaign has an order, and its first leg is
protected by theorems.** The §6.2 observation in
`notes/fermionic_architecture.tex` — that on the electric line (h_x=0, sweep
h_z) the frozen head + flux penalty stay *exact* — went through adversarial
review by the fermionic-ladder session. Split verdict:

- **Claim 1 (exact superselection at h_x=0) — airtight, unconditional.** The
  Gauss parities u_c commute with every A_v (even star overlap of any
  boundary XOR), every B̃_p (Mᵀc = 0 is the defining equation of the masks),
  and the diagonal h_z term. The penalty is an **exact sector projection at
  every h_z**, not an approximation.
- **Claim 2 (head-rotated stoquasticity in the physical sector) — airtight.**
  The A_v worry dissolves: star flips are token-invariant, so the head phase
  cancels identically on star matrix elements; tidy fact ε_p(s) = t_p(s)
  makes the B̃ cancellation manifest. All rotated off-diagonal elements ≤ 0
  → Perron–Frobenius → **frozen-head signs exact for the *sector* ground
  state at every h_z**; training there is genuinely amplitude-only, and any
  sign drift is a red flag, not physics.
- **The one open gap:** whether the *global* GS remains in the u≡+1 sector at
  intermediate h_z. Endpoints safe (h=0 by construction; h_z→∞ polarized has
  t≡+1 ∈ V); flux costs B̃ energy and gains nothing from a flux-neutral
  diagonal field — expected yes, not yet excluded by generalities.

**Gate: L=2 fermionic ED along h_z** (grid over [0, ~0.6];
`tests/colab_exact_diag.py` fermionic toggle, cluster job, 2²⁴ states),
extracting per point: (1) **sector-resolved E0** (u≡+1 vs lowest other
sector) — closes the gap; (2) **phase audit** ψ_ED(s)·(−1)^{q(t(s))} > 0 on
the support — end-to-end test of claim 2 including convention slips; (3) the
**E(h_z) curve** — first exact finite-field benchmark for the NQS electric
sweeps. Gate *trust* on ED; build sweep infrastructure in parallel.

**Design decisions recorded** (folded into `notes/fermionic_architecture.tex`
§6–§7 and `notes/fermionic_next_steps.md` ③):

- **Electric first, magnetic second.** Tiered exactness: h=0 exact (theorem)
  → electric line: discrete scaffolding exact, amplitude-only training
  (bosonic-grade stoquastic in the rotated frame) → h_x≠0: fixed-κ bias
  (~6e-6/violation vs perturbative ghost weight) + genuine trunk phase work —
  the empirical bet, and where the new physics lives.
- **Penalty stays MANDATORY on the electric line, κ=6 frozen:** sector
  conservation is a property of H, not of the variational state — the
  token-blind trunk would leak ghost weight exactly as at h=0.
- Physics bonus: a σ^z insertion both creates star defects *and* frustrates
  the decorated plaquettes containing that edge, so the h_z transition probes
  condensation of the genuinely fermionic composite — the right foil for
  bosonic h_z^c ≈ 0.194.

Also this session, from a close read of the companion doc: the missing
derivations are now in it — the eigenvalue-equation → two-amplitude-relation
step (new eq:propagation), the collision-counting proof that the sign is
GF(2)-quadratic in tokens (C-form moved to §2; Dehaene–De Moor ref), "frozen
≠ rigid" (§6.1: the continuity objection retired — total phase = frozen head
+ trainable trunk phase, so freezing removes no smooth directions), and why
zero-init correction heads are saddle-free once the frozen head breaks the
phase-flip symmetry (§6.2).

---

## 2026-08-11 (night) — L=8–12 QMC FM capability CERTIFIED; the ParaToric τ-warning segfault

**Headline: the QMC-only FSS tail is open.** Both frozen operator families
evaluated in production at L=8/10/12 (β=24, ×4 decorrelation, 4×4 blocks,
pooled = mean(num)/√|mean(den)| with chain jackknife; pooled ≡ naive to
<0.05σ everywhere):

| L | Z-string (hx=0.2, hz≈.27–.28) | membrane pt-cube (hx=0.88, hz=0) | membrane R1 | wall |
|---|---|---|---|---|
| 8  | 0.5906(35), den_z=1295 | 0.1843(94)  | 0.5103(124) | 33/10 min |
| 10 | 0.5738(43), den_z=902  | 0.1117(92)  | 0.3838(157) | 37/23 min |
| 12 | 0.5656(42), den_z=1125 | 0.0475(50)  | 0.2903(205) | 86/56 min |

Structural findings: at hz=0 the closed membrane is the coboundary of
conserved A_v's, so **⟨closed⟩ ≡ 1 exactly** (den_z=∞) — the magnetic mirror
of the string's hx=0 line; and string den_z stays ~900–1300 even at two-field
near-critical points, so the FM denominator is never the limiting factor in
the FSS range. Campaign caveats: membrane χ²_red ran 3–6 at hx=0.88 → use
**NBS_MULT=8 near h_c** in the x basis; den_z=∞ is specific to hz=0 — probe
den_z once before any two-field membrane campaign.

**The segfault saga (why membranes were blocked).** Five identical
multiprocess failures (BrokenProcessPool, ~chain-end, no oom_kill) at L=8
x-basis while every single-process probe passed. Refuted in order: OOM, stack
size, thread explosion, fork-vs-spawn. The worker `faulthandler` (kept from
48ad08b) finally showed **SIGSEGV inside the C++ `get_sample`**; a debug-QOS
rerun with `ulimit -c unlimited` + post-mortem gdb put the crash in stock
ParaToric's `calculate_autocorrelation_time_with_warning` — specifically the
**Boost.Log record it emits when τ > 0.1·N_samples**. Mechanism: the cluster
.so is built `-static-libstdc++` while conda-forge's boost_log links the
dynamic libstdc++ — two C++ runtimes in one process; formatting the record
across that boundary dies in `_M_insert<double>`. That threshold explains the
"stochastic" signature exactly: near-critical x-basis chains trip the warning
~20% of the time (plaquette_z τ≈116 at ns=1000), so 16-chain runs always
contained a crasher while the lone probed seed (12345) never fired it —
seed-dependence was probability, not mechanism. Falsification steps that
mattered: the warning prints fine on the login node (killed the
"any-first-record crashes" theory) and the τ-estimator code audit came back
memory-safe, isolating the emission path.

**Fix** (`external/paratoric_stdio_taulog.patch`, commit c709c7b, wired into
both build scripts after the membrane patch): replace the library's ONLY
reachable non-debug log record with plain `fprintf(stderr, …)` — identical
message, no C++ stream/locale state, zero statistics change (τ was always
returned separately). **Proven in anger:** pilotX7fix reran the exact
twice-crashed seed family — 16/16 chains, a live τ warning printing
harmlessly mid-run (pre-fix pass probability ≲3%). Driver keeps spawn-context
workers + faulthandler as permanent tripwires.

**Handoff:** CERTIFIED ping sent to the QMC-validation session (matrix +
caveats); queue freed for their 87-run Phase-B fleet (launch gated on user
approval by design). The one open box before "NQS ≡ QMC" is a *measured*
statement: the numerical same-point comparison at L=4–6, running now on their
side (L=4 pooled eval PASSED; L=5 re-running; L=6 blocked on an XLA
compile-RAM OOM they're re-shaping around). Operator-level identity is
already a theorem of the test suite (edge-for-edge C++ replication,
L=4–8,10,12).

## 2026-08-10 (night) — fermionic h=0 ladder COMPLETE: exact GS at L=2..6, one analytic recipe

**Every exact anchor hit** (E₀ = −4L³, PBC; all early-stopped at the
sampling-resolution floor):

| L | E₀ | final E | δ | run |
|---|---|---|---|---|
| 2 | −32 | −32.0000042(100) | 1.3e-7 | ph_polishANA |
| 3 | −108 | −107.99999592(6) | 3.8e-8 | ph_fp6_polishANA |
| 4 | −256 | −255.999986(17) | 5.3e-8 | k2_phf_fp6_anaC |
| 5 | −500 | −500.000017(122) | 3.5e-8 | k2_phf_fp6_anaC |
| 6 | −864 | −863.99939(32) | 7.1e-7 | k2_phf_fp6_anaC (3 SR steps!) |

The L≥4 recipe (commits 4a43811..8986de0): **C-form pullback** θ (analytic at
any L — q̃(x)=ΣC_pq x_p x_q through a GF(2) right-inverse of the token-flip
map; RREF-with-preimages bug found and fixed: 10⁴/10⁴ certificates at
L=2..6), **frozen head** (θ in the flax 'constants' collection — no
parameters, no QGT blow-up; a trainable head at L=6 would be 420k params),
**flux penalty as one matmul** (cos π·b@W — per-mask prod kernels cost 78
min/step at L=6), **--chains_up** (random chain inits land in ghost cosets
with single-flip local minima; L=4 froze at δ=7.3e-3 for 120 steps until
chains start IN the physical sector), and k=2 kernels (h=0 on-sector state
is uniform; k=L−1 buys nothing but 4-16× cost). Ops lessons: CHUNK sizes the
per-chunk host-transfer overhead, not just memory — grad 408→118 s at L=5
going 64→256; cold trunk + analytic structure starts at δ ≈ 5e-3..5e-6 at
EVERY size (L=6 needed 3 SR steps total). Explainer with full failure-mode
map: notes/fermionic_architecture.tex (+ results table). Next: item ③ —
finite fields, where the h_x=0 electric line may keep exact signs
(conjecture, §7 of the explainer; L=2 ED check proposed).

## 2026-08-10 (later) — observable-level QMC validation: every expectation value at 4 points × L=4–6

The tune-rect winner is now validated against ParaToric on **every observable
both methods can measure** — not just energy. Everything in
`analysis/tune_rect_summary.ipynb` §6 (table + 3 figures: stabilizers and
magnetizations as relative deviations with 2σ upper limits, order parameters
raw). Grid: 4 points × L ∈ {4,5,6}, NQS re-evals + dedicated z-basis QMC.

- **A_v/B_p/M_x/M_z joined for the first time** (both sides always measured
  them; nobody compared). Weak field (h_x=0.2): everything within |z| ≲ 3,
  stabilizers mostly 2σ upper limits. Strong field (h_x=0.6): a coherent,
  L-growing variational systematic — E high, stabilizers high, magnetizations
  low, **sharpest in M_z (z ≈ −11, 10–25% relative)**. The §4 energy gap is
  field-channel-dominated; capacity, not sampling.
- **Z-string O_FM: the headline.** ParaToric's native `fredenhagen_marcu`
  wired into the driver (`--fm`, z-basis only — the x-basis cubic loop branch
  doesn't exist upstream and silently returns 1) + `--validate_fm` anchors
  (symmetry-exact zero at h_z=0; deep-trivial →1: PASS). NQS side scores the
  identical loop edge-for-edge (`fm.paratoric_fm_edges`, upper-half-U
  convention). **11/12 (point, L) agree at |z| ≤ 1.2**, both methods tracing
  the same field-monotone finite-R tail (0.002→0.021); the 12th
  ((0.6,0.15) L=6) is an unconverged heavy-tailed NQS ratio (0.53(52)), a
  statistics limitation, not a discrepancy. QMC cross-basis energies |z| ≤ 2.2
  everywhere (diagonal ↔ kink estimators swap roles: certifies both).
- **S₂**: no ParaToric estimator exists — NQS-internal column, hovering at the
  h=0 anchor 3·ln2 across the rectangle.
- **X-membrane**: ParaToric had no magnetic order parameter; we patched one in
  (`external/paratoric_membrane.patch`, VertexPair storage — Lattice is copied
  per chain, boost Edge descriptors go stale; unconditional throws — NDEBUG
  guards vanish in Release). Convention then frozen mid-campaign to the
  corner-rule families (entry below); ladders PASS on the merged build
  (corner-rule L=5/6 + R=1 L=4, local values reproduce the conventions
  session's exactly). **Production membrane runs are gated** on the cluster
  ladder + user go.
- **Audit caveat (label correction, findings by the conventions session's
  adversarial fleet, confirmed here independently):** `fm._load_weights`
  restores the checkpoint's sampling config, so all `.eval65k`/`.fm65k`
  re-evals actually ran at the training budget n=8192 with a seed-independent
  restored sampler (`.eval65k` ≡ `.fm65k` bit-identical). Error bars are
  honest for the true budget — every pull above stands — but means are ~2.8×
  noisier than the label claimed; notebook rows re-tagged `8k†`. True-65k
  re-run after the loader fix.
- Ops: QMC on shared-QOS 16 cores ≈ 9 min per L=6 point (~4× the contended
  local Mac) — wrapper walltime right-sized 3h→1h with measured per-L `-t`
  guidance; eval jobs are compile-dominated on shared nodes (~1h20 cold);
  `submit_eval_ckpt.sh` added (batch checkpoint re-evals, exports the JAX
  cache extract-style wrappers miss).


---

## 2026-08-10 — FM convention freeze: ParaToric geometry is THE cross-method family (string + membranes)

**Decision (with user):** for the NQS↔QMC FSS-consistency program (agree at
L=4–6, then trust QMC alone at L=8–12), both codes measure **bit-identical FM
operators**, frozen as:

- **Z-string:** ParaToric's stock loop at every L — corners s=(L−1)//4,
  e=3(L−1)//4 in plane z=(L−1)//2, upper half-U open string; boundary-touching
  at L=4 *by convention*. R = e−s grows ≈L/2 (2,2,2,3,4,4,4,5,6 for L=4..12) —
  a fixed-aspect-½ family whose ℓ→∞ limit is the genuine order parameter (the
  "R jump" at L=7 is the family doing its job). Odd-R caveat: the open U is
  2R+1 edges at L=7,11 (a smooth O(1) factor, harmless for h_c).
- **X-membrane, growing family:** cube on the SAME corners cubed ([s,e]³,
  vertical=z, single orientation), L≥5 — at L=4 the side-2 cube's coboundary is
  boundary-truncated (27 edges, odd: no exact half exists). **Changes the QMC
  patch geometry at L=6 (R 3→2) and L=10 (5→4)** vs the old R=L//2.
- **X-membrane, R=1 anchor family:** centered 2×2×2-vertex cube, L≥4 — the
  cheapest operator, existing at every size both methods reach; the
  cross-method anchor. Each (sector, family) is its OWN FSS curve — never mix
  families in one fit (plot_phase_diagram now warns).

**Implementation** (branch `feat/fm-paratoric-convention`, worktree
`../toric-code-nqs-fmconv` — main checkout stays with the QMC-validation
session): fm.py gains `paratoric_corner_rule`/`paratoric_membrane_kwargs` +
`placement="paratoric"` (electric stock string; magnetic corner-rule or
`--R 1` anchor) + `verify_paratoric_fm_geometry`; eval_ckpt scores both
membrane families per checkpoint (`--fm_membrane_R pt,1`). ParaToric patch
rewritten: corner-rule membrane + new `fredenhagen_marcu_membrane_r1`
observable (both fit in ONE x-basis run); driver gains `--fm_membrane_r1` and
ladders `--validate_fm_membrane` (L=5+6) / `--validate_fm_membrane_r1` (L=4).
Wrappers: `PLACEMENT=paratoric` + TAGs ptstring/ptcube/ptR1; eval/QMC submit
knobs FM_MEMBRANE_PARATORIC / FM_MEMBRANE_R1.

**Verification:** new `tests/test_fm_paratoric.py` replicates the C++
vertex-pair constructions in pure Python and asserts **edge-for-edge equality
at L=4–8** (the translation-layer certificate the old docstring-only claim
lacked) + coboundary brute-force + constructibility matrix. Local ParaToric
rebuild: import + smoke (L=4 r1 measures, L=4 corner-rule THROWS, both
families in one run, trivial anchor = 1.0000) and the full membrane ladders
**PASSED** locally (zeros |z|<1.4; trivials >0.91; regenerated patch verified
to apply to pristine upstream). Coordination: stale-convention L=6 x-basis
membrane refs invalidated (QMC session cancelled its pending jobs); L=5 refs
remain valid (corner-rule cube ≡ old centered R=2 cube there). Cluster next:
rebuild + full 4-ladder validation + L=4 spot check; L=8 noise pilot gates
L=10/12 (⟨closed⟩ decays exponentially in operator size).

---

## 2026-08-07 (night) — fermionic L=3 EXACT (δ = 3.8e-8): sampled classes, the ghost-sector trap, and the flux-penalty head

**Headline: E = −107.99999592(6) at L=3 PBC** (exact −108; Vscore 2.0e-7,
B̃_p = 1.000000025; run `…_ph_fp6_polishANA`, job 56478278) — the L=2 recipe
scaled, but only after discovering that the **flux-sector channel needs the
same analytic treatment as the sign channel**.

**① Sampled-class solve (no enumeration).** `analysis/prefit_phase_head.py`
generalized to any L: dataset = uniform-random subsets of the 3L³ decorated
plaquettes applied to |0…0⟩ with the BFS sign rule (stars provably change
neither tokens nor signs, and commutation makes the order irrelevant). Online
augmented-GF(2) elimination gives rank-saturation stopping and a contradiction
detector. L=3: rank saturates at **666 = d(d+1)/2 with d = 36** (dim of the
token class space; 21 → d=6 at L=2 — 64 classes ✓) after 1168 draws, zero
contradictions, residual 0; certificate **10,000/10,000 fresh samples, every
one a never-before-seen class** (~7×10¹⁰ classes exist). The sign is an exact
token-quadratic form at L=3.

**The ghost-sector trap.** The polish (analytic head, guard open) plateaued at
E = −89.97 with a PERFECT on-orbit state: 100% sign accuracy on fresh support
samples, uniform support amplitudes, 0.4% head drift. MCMC census: **99.8% of
sampled weight in wrong flux sectors** at on-orbit amplitude. Mechanism: the
trunk is a function of flux tokens and cannot separate the 2^45 flux cosets;
chains start in a random coset; cluster moves conserve the coset; SR converges
faithfully in the wrong superselection sector (B̃ = 0.777 is near-optimal
there). A_v = 1 is vacuous for this architecture (star flips don't change
tokens), and Im⟨E⟩ = 0 because the head generalizes consistently into ghost
sectors — the standard diagnostics were all blind to it.

**Flux-penalty head** (`--flux_penalty κ`, commit 43415f4):
`fermionic_decoration.flux_constraint_masks` computes the GF(2) null space of
the pair-move token-flip map = the conserved closed-surface flux parities
(lattice Gauss laws; 18 at L=2, 45 at L=3); the network subtracts a fixed κ
from Re logψ per violated parity. Zero parameters (checkpoint-compatible),
never trained. **Training-free validation:** penalty attached to the plateaued
checkpoint → on-orbit weight 0.2% → 100.0%, ⟨H⟩ = −107.916(31) with no
retraining; a 150-step SR polish then closed it to δ = 3.8e-8 in ~40 steps.
Doctrine, twice confirmed: *every discrete/topological channel — signs AND
flux sectors — is GF(2)-solved analytically; the optimizer touches only the
smooth amplitude channel.*

**② Stencil obstruction** (notes/fermionic_stencil_obstruction.md, 8b4ec59):
no universal local translation-invariant sign form exists in state variables.
In *application* variables the sign is analytic, TI, range-1
(C_pq = |∂p ∩ xpair_q|, verified 3000/3000 at L=2,3,4); as a function of the
state, a TI representative exists **iff L is odd** (parity theorem: Σ_g q∘T_g
is valid iff |G| odd; empirically L=3,5 feasible at full range, L=4 infeasible
even with period-2 enrichment) and locality fails across sizes — a
Kasteleyn/spin-structure-type obstruction. Vindicates the per-size dense
analytic head as the production design.

**④ Dressed observables** (f7c48bc, worktree subagent + review): fermionic
O_FM ported into fm.py (`dressed_electric_edges`, model= threading,
`topological_observables` now returns the dressed ⟨W̃⟩) with two upstream
`dressed_string` fixes — a σ^y guard that fired on contractible loops
(line-free re-solve) and endpoint-localization of the open-string flux
residual (3-plaquette body-diagonal cluster per endpoint, 6 total,
size-independent; weight-2 proven impossible).

**Infra finds** (the OOM trilogy, all at the first L=3 gradient, 78 GB):
the phase-head einsum was a red herring (rewritten to (t@Q)*t anyway,
e3dba3c); the real bug — **io.load_weights restored the CHECKPOINT's sampling
config** (NetKet serializes n_samples/n_discard/chunk_size), so every
`--init_from` run silently inherited `chunk_size=None` from CPU-built prefit
checkpoints and ran the unchunked forces kernel (fixed 4d3d479; fingerprints:
chunk-independent allocation + `jit(forces_expect_hermitian)` sans `_chunked`
in the XLA log). Fermionic runs need CHUNK ∝ 1/L³ (256 at L=3): n_conn ≈ 4L³.

**Next:** ③ finite-field program (analytic θ + penalty init, `--init_from`
chaining, fermionic ED references; QMC is sign-blocked — NQS is the only
method) and L=4 (E₀ = −256, same recipe, CHUNK ~64–128).

---

## 2026-08-07 (later) — token-quadratic phase head: exact fermionic GS, δ = 1.3e-7

Follow-up to the morning's sign-trap study: implemented the **token-quadratic
phase head** and closed the ladder. `ToricCNN_gridinv` gains `--phase_head`
(`PHASE_HEAD=1`): adds i·(θ_p t_p + θ_pq t_p t_q) over the EXACT flux tokens of
the raw input, as a parallel branch summed into log ψ (zero-init = inactive;
complex/primal-gridinv only). Final run `ph_polishANA`:
**E = −32.0000, δ = 1.3e-7, Vscore 2e-8** — the exact GS, through the old trap
region without a wobble. Ladder: `analysis/fermionic_h0_prefit_ladder.ipynb`.

The path there was a chain of instructive FAILED predictions, each isolating
one requirement (all runs L=2, h=0, 16-of-64-class hold-out as the
extrapolation probe):

- **Joint fit, no head**: 14/16 unseen classes; polish −31.64 (δ 1.1%).
- **Joint fit + head**: 13/16 (!), polish −30.15 (δ 5.8%) — gradient descent
  smears signs across trunk+head; the smeared representation is *fragile*
  under the early off-support amplitude drain. Adding the right hypothesis
  class is useless unless the fit is FORCED into it.
- **Head-only over a random trunk**: 11/16 — the trunk's O(0.3 rad) init
  phases aren't quadratic in tokens; the head absorbs their projection and
  extrapolates the bias.
- **Head-only + exactly real trunk**: frozen at baseline — b ≡ 0 is a
  stationary point of every (even-in-b) sign loss; gradient identically zero.
- **+ noise-seeded head**: escapes the saddle but plateaus at 62/64 — the
  cosine loss of a 600-parameter linear phase model is a phase-retrieval
  landscape with local minima; the big-network fits only ever worked via
  overparameterization.
- **ANALYTIC head** (no optimizer in the sign channel at all): GF(2)-solve the
  quadratic form from 48 class labels → set θ directly → **16/16 held-out,
  100% on all 32,768 support states, zero training steps**. SR polish from it:
  −31.9569 by step 20, −32.0000 at convergence. Explicit-θ sign storage is
  robust — SR's amplitude work never touches it.

Cold start WITH the head still traps at −22.52 (control): the head fixes
representability and storage, not the energy landscape's descent path.

**The recipe** (every ingredient poly(L)): sample support configurations +
exact signs from the stabilizer algebra → GF(2)-solve the token quadratic
form → write θ into the phase head → SR for amplitudes only. Next: L=3
(E0=−108) without enumeration, dense→local-stencil head compression, and
finite-field runs where the analytic θ becomes the inductive bias.

## 2026-08-07 — fermionic TC at h=0: the sign trap, and what sign quality buys

First benchmark of the production stack on the **fermionic** toric code at zero
field (branch `feat/fermionic-h0`). Headline: the ansatz is expressive enough,
**SR is the failing component** — it neither creates nor repairs sign structure.
Full background: `notes/fermionic_sign_problem.{tex,pdf}` (2D warm-up, mechanism,
literature); diagnosis chain: `notes/fermionic_plateau_diagnosis.md`; ladder
plot: `analysis/fermionic_h0_prefit_ladder.ipynb`.

- **Wiring verified, sign-fullness proven** (no ED): commutation clean at L=2/3,
  dressed Wilson anchors reproduced (closed loop flux-free, open string = 2
  endpoint fluxes), and the exact h=0 stabilizer BFS shows the GS support
  (2^15 states at L=2) splits 56/44 positive/negative — with the sign an exact
  **function of the flux tokens** (64 classes; a GF(2) quadratic form). In the
  dual frame the sign is token-invisible (faces overlap stars evenly), so
  `--dual_basis` stays bosonic-only for a structural reason. New fast suite:
  `tests/test_fermionic.py`.
- **Cold-start VMC traps at the positive-sector optimum**: both the tune-rect
  winner and the old workhorse converge to E=−22.521 vs exact −32 (δ=29.6%,
  identical to 4 digits) — the optimal state confined to the positive-sign half
  (all samples on-support/positive, phases flat, ⟨B̃_p⟩=0.605 uniform, energy
  closes as −8−24·0.605). A guard-off escape probe (400 hot iters) attempted
  crossings and relaxed back: **a genuine optimization barrier, not a guard
  artifact** (though the guard's 10× spike heuristic also kills escapes — its
  plateau-median baseline makes any phase restructuring look like divergence).
- **Supervised phase pre-fit** (exact BFS labels, fit Im logψ → sign): the
  production CNN hits **100% sign accuracy on all 32,768 support states in
  ~100 Adam steps** — capacity was never the problem. Class-held-out variant:
  trains to 100% on 48/64 token classes, generalizes to **14/16 unseen classes**
  (87.5%) — learns the rule approximately, not the exact quadratic form, even
  though 48 classes over-determine it.
- **Polish ladder** (SR warm-started from pre-fits, guard open): cold −22.52
  (δ 29.6%) → 62/64-sign warm start **−31.303** (δ 2.2%) → 64/64 warm start
  **−31.644** (δ 1.1%). Both polished runs blow straight through the old trap —
  and both freeze short of −32: SR pays for sign defects but never fixes them,
  and even perfect signs lose fidelity during the violent early off-support
  amplitude drain (pre-fit shapes the support only; raw pre-fit E ≈ −10.9).
- **Next**: (a) token-pair quadratic phase head in gridinv — exact at h=0 by the
  stabilizer quadratic-form theorem, now doubly motivated (extrapolation +
  drift-resistance); (b) off-support amplitude suppression in the pre-fit and/or
  a gentler polish schedule to close the last 1.1%; (c) decoration annealing
  (1−λ)B_p+λB̃_p as the no-new-architecture alternative.
- Infra: `--model fermionic` now auto-derives complex weights (was hy-only — a
  real-logψ fermionic run was silently impossible); fermionic sampler gains the
  B̃_p x-pair cluster moves (the only star-suborbit-crossing moves at h=0);
  `submit_nqs_gridinv.sh` gains MODEL / EXACT_E0 / EXTRA_ARGS knobs;
  `--init_from` fixed for complex weights; bare-loop O_FM gated off for
  fermionic runs; stale `.venv` editable path repaired.

## 2026-08-06 — tune-rect campaign: dual-basis architecture locked, scaled to L=6

Full campaign in ~24 h (75+ GPU jobs, ~35 GPU-h): hyperparameter search for the
dual-basis `ToricCNN_gridinv` on the rectangle hx {0.2,0.6} × hz {0.1,0.15}, judged
against dedicated ParaToric references on all observables, then the winner scaled to
L=5/6. Everything in `analysis/tune_rect_summary.ipynb` +
`results/tune_rect/tuning_table_L4_all.{md,json}`; 59+ runs on W&B (`tune_rect_L4`).

- **QMC reference grid built**: 3 new L=4 corners + L=5/6 at all corners (14 runs,
  every sum rule ≤0.06σ, β-drift z=+1.18 at the hardest corner). GPU-shared trick:
  ParaToric on 1-GPU shared QOS (16 cores), ~1–3 min per L=4 point.
- **Dual beats primal at equal budget**: winner nh(4→8)→inv(8,8) k3 @ dt=0.02/ds=1e-3
  hits −173.4099(48) at (0.2,0.1) vs QMC −173.4493(179) with Vscore 2.9e-4 (primal:
  7.2e-3 at +2.5σ, 5,319 params vs 5,345). Enablers: invariant width (2,2,2)→(8,8)
  cut the gap 4× (capacity, as predicted 2026-07-29); hot schedule dt=0.02 converges
  in 150 steps (0.01 still descending; 0.04 spikes). Damping ladder monotone: 1e-3 best.
- **r0.9 NN-only stencil fails at strong field** (+0.21…0.28 vs +0.09…0.13 at hx=0.6;
  one divergence; ds=3e-3 retry stays bad → capacity, not fragility). The dropped
  d=1.0 taps are the plaquette-adjacent correlations that matter once ⟨B_p⟩≈0.85.
  User's r0.9 tie-breaker overridden on this evidence (robustness-first rule).
- **Scaling (kernel=L−1 → params 5,345/10,377/18,673, i.e. ≈35 per spin at every L)**:
  L=6 statistically consistent with QMC
  at both weak-field points (+1.3σ / **+0.7σ**, rel 1.0e-4 / 2.9e-5), degrading
  smoothly to +4.8σ (6.1e-4) at (0.6,0.15). L=6 beats L=5 everywhere — kernel span +
  iters co-scale against fixed width. **L=5 needed ds=3e-3 at hx=0.6** (both diverged
  at 1e-3); L=6 (k=5) stable at 1e-3.
- Ops: timing smokes before scaling (L=5 15.6 s/step, L=6 59.8, nodes ±40%);
  AUTO_RESUBMIT chains carried all four L=6 runs across the 5 h cap; Phase-B name
  collision taught: training knobs belong in the checkpoint name (SEED/NAME tags);
  inline O_FM returned nothing at L=6 (R=3) — post-hoc via `tc3d.fm` if needed.

## 2026-08-05 — post-promotion hardening; cluster back online

- **Adversarial audit of the promotion** (3-agent sweep, `99b0a8f`): fixed the
  real defects it surfaced — `WALLTIME` env knob now survives AUTO_RESUBMIT
  requeues (a bare `sbatch --time` silently reset on every chain link); JAX
  cache export hoisted above the requeue trap; `setup_conda_gpu.sh` now
  `pip install -e .`; `run_phase_campaign.sh` cds to repo root and exports
  `REPO`. Tree hygiene (stale `model/ simulation/ utils/` husks removed,
  `*.egg-info` untracked) and a docs-truthfulness pass (`nersc/README.md`
  rewritten to match the actual scripts; README/CLAUDE.md corrections).
- **JAX compilation cache unified** (`212c091`): all five training submit
  wrappers default `JAX_COMPILATION_CACHE_DIR=$PSCRATCH/tc_nqs/jax_cache`.
  Measured: cold dual L=4 compile ~20 min → cached seconds. Never disable it.
- **NERSC back from maintenance a day early.** Login + `gpu_ss11` healthy;
  `$PSCRATCH/tc_nqs` survived intact (the pre-maintenance local mirror wasn't
  needed); cluster clone `~/toric-code-nqs` at parity with `main`.

## 2026-08-04 — repo promotion: ThreeD_TC → toric-code-nqs

The project moved from the old working tree to this repo, **keeping full git
history** (in-place restructure, never a fresh-history copy):

- **New canonical remote:** `github.com/SanzharBissenali/toric-code-nqs`;
  cluster clone path repointed to `~/toric-code-nqs` (`48f7778` — 15 nersc
  wrappers, Colab clone URL).
- **Package promotion** (`9a318ab`): `Three_TC/` became the flat installable
  **`tc3d/`** package (`pip install -e ".[analysis]"`); tests run against the
  installed package with no path shims.
- **2D legacy frozen at tag `2d-final`:** the 2D surface-code implementation,
  factored-attention transformer, and all dead/ED-era code were dropped from
  the tree. What survived the cut is exactly what serves the two tracks:
  geometry/Hamiltonian/networks, `builders.py`, cluster-update sampler,
  `train.py`/`sweep.py`, `fm.py`/`renyi.py` extraction, QMC drivers,
  analytic benchmarks (42 self-checks), NERSC wrappers.
- **Dual-basis branch merged:** `--dual_basis` (Hadamard + star tokens) is now
  a first-class `train.py` flag rather than a feature branch.

Everything before this point lives in `notes/log_and_plan.md` (frozen) and the
git history — highlights: 2D→3D architecture extension and L=2 arch comparison
(symmetry-aware ~100× lower energy error than plain CNN), the 2026-07 phase
campaigns (`phase_hx*`/`phase_hz*` sweeps, FM + S2 extraction, x–z magnetic
line), and the QMC benchmark pipeline build-out.
