# 3D bosonic toric-code phase-diagram campaign — config reference

> **SUPERSEDED for protocol & architecture (2026-08-19).** The tables below
> record the July 2026 campaign *as it was run* (pre-tune-rect architecture,
> cold starts everywhere, β=12 references) and stay as frozen provenance for
> `results/phaseB*` and the archived `results/phase_*` data. For any NEW run,
> `notes/transition_mapping_recipes.md` §0–§B is authoritative: production
> flags `DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=$((L-1))`, warm chains
> through first-order windows, β≥24 QMC referees near crossings.

Maps the topological→trivial boundary of the perturbed 3D bosonic toric code
(`ToricCNN_gridinv`, OBC) in the (hx, hz) plane. Fixed-hx vertical cuts, hz swept,
finite-size scaling over L. Hamiltonian
$H=-J\sum_v A_v-J\sum_p B_p-h_x\sum_i\sigma^x_i-h_z\sum_i\sigma^z_i$, $J{=}1$, $h_y{=}0$.

Launch: `nersc/run_phase_campaign.sh` (per (hx, L) it submits one hz-array via
`nersc/submit_nqs_hz_sweep.sh`). Last updated 2026-07-08.

## Fixed across all L (architecture + optimizer + sampler)

| Setting | Value |
|---|---|
| Architecture | `ToricCNN_gridinv`, OBC |
| noninv block | 4 channels × 2 layers (`noninv_channels=4`, `n_noninv=2`) |
| inv block | `inv_hidden=(2,2,2)` — 3 grid-conv layers, width 2 |
| LR schedule | cosine, `dt=0.01` → `lr_min=0.001` |
| Sampler | `n_samples=8192`, `n_chains=1024`, `n_sweeps=48` |
| Solver | dense `QGTJacobianDense` (SRt opt-in via `--qgt srt`) |
| chunk_size | 2048 |
| checkpoint_every | 10 (resume-safe) |

## Per-L hyperparameters (FSS set)

| L | qubits N | #A_v | #B_p | E0(h=0) bound | kernel | n_params | diag_shift | n_iter | ~s/step | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 144 | 64 | 108 | −172 | 3 | 5,319 | 1e-3 | 150 | ~15* | 1:00 |
| 5 | 300 | 125 | 240 | −365 | 4 | 11,313 | 1e-3 | 150 | ~25* | 1:30 |
| 6 | 540 | 216 | 450 | −666 | 4 | 11,313 | 5e-3 | 150 | ~45 | 3:00 |
| 7 | 882 | 343 | 756 | −1099 | 5 | 21,195 | 5e-3 | 175 | ~105 | 6:00 |

\* L=4/L=5 per-step are estimates; L=6 (~43–45 s) and L=7 (104.7 s at k5) are measured.
n_params is set by kernel+channels, not N — L=5 and L=6 (both k4) coincide at 11,313.
A converged run must land strictly below its bound (L=7 reached −1113.5 < −1099 ✓).
L=3 exists in the scripts (kernel 2) but is validation-only, not in the campaign.

### Notes on the L=6/L=7 choices
- **kernel:** L=7 uses 5 (smoke showed k4 too small, k5 the sweet spot — k6 converges
  no lower for +16 %/step, +70 % params). L≤6 capped at ≤4 for cost.
- **diag_shift:** L=6,7 at 5e-3 (1e-2 over-regularized the descent). The divergence
  guard is the backstop for the large-L post-convergence SR blow-up that originally
  motivated 1e-2. L≤5 keep the proven 1e-3.
- **n_iter:** L=7 needs 175 (converges ~150–175 at 882 sites); others 150.

## Sweep / campaign details

| Aspect | Value |
|---|---|
| hx cuts | {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} — 6 |
| hz window (every hx) | [0.1, 0.4], 13 points, step 0.025 |
| L (FSS) | {4, 5, 6, 7} — 4 |
| Total runs | 6 × 4 × 13 = 312 |
| Launch | `run_phase_campaign.sh` → 24 array jobs (one per hx×L), 13 tasks each |
| Robustness | `--resume` always on; `AUTO_RESUBMIT=1`; divergence guard on (dense path) |
| Output | `$PSCRATCH/tc_nqs/phase_hx{HX}/L{L}/bosonic_gridinv_L{L}_hx{HX}_hz{HZ}.{mpack,json,curve.json}` |
| QA gate | `analysis/scripts/check_convergence.py --tree` (finite E, E<bound, ⟨A_v⟩≤1, not diverged, missing list) |

**Budget:** per hz-point across all 4 L ≈ 8.6 GPU-h → ~675 GPU-h total. L=7 is ~60 %
(5.1 h/run × 78 runs ≈ 400 h) — the biggest lever if the L set is ever trimmed.

**Caveats:**
- hz window is fixed [0.1, 0.4] at every hx, but hz_c drifts down as hx grows — at
  hx=0.8/1.0 the peak may slide below 0.1 (QA scan flags no-crossing; cheap re-launch
  with a lower `HZ_MIN`).
- L=4/L=5 wall times assume the estimated per-step; widen if the first medians run hot.
- Solver is dense for the whole campaign; SRt (`--qgt srt`) is a memory/scaling tool
  (no per-step win at k5/k6) reserved for k7+ where the dense S-matrix OOMs.

## Phase B — QMC-validation cuts (spec frozen 2026-08-11)

Two cuts from the anchor **(h_x, h_z) = (0.2, 0.1)** (the tune-rect corner, validated
on every observable at L=4–6), each crossing its phase boundary; **L ∈ {4, 5, 6}**.
Goal: NQS-vs-QMC pull ribbons along both cuts + the L-shaped phase-diagram visual.

| cut | fixed | swept | grid |
|---|---|---|---|
| up (electric) | h_x=0.2 | h_z | 0.10, 0.15, **0.18–0.36 at Δ=0.02**, 0.40, 0.45, 0.50 (15 pts) |
| right (magnetic) | h_z=0.1 | h_x | 0.20, 0.35, 0.50, 0.65, **0.75–1.10 at Δ=0.05**, 1.175, 1.25 (14 pts) |

Fine windows sit on the measured finite-size crossings (up: FM crossings 0.29–0.33
for L=4–7 from `phase_hx0.2_bulkR1`; right: first-order jump bracketed 0.84–1.0 by the
old family-mixed xz_line data — Phase B regenerates that line under the frozen
convention). Optional wave 2: after per-L sigmoid/peak fits, add 2–3 points within
±Δ of each inflection (sweep.py batches → compile amortized).

- **NQS**: winner arch (tune-rect spec above), `--final_eval_rounds 8` → 65k-equivalent
  end-of-training observables (E, A_v, B_p, M_x, M_z, Z-string, both membrane
  families) in the training job — no separate eval pass. Watch `E_err_scatter` vs
  `E_err`: scatter ≫ pooled ⇒ τ blow-up (near-critical); raise `--n_sweeps` there.
  Every point trains independently (cold start) — no `--init_from` chaining
  (user decision 2026-08-11: no bidirectional/hysteresis protocol).
- **QMC**: per point — up cut: z-basis `FM=1`; right cut: x-basis `FM_MEMBRANE=1
  FM_MEMBRANE_R1=1`; β=12, ×4 recipe, fresh seeds, `nbs_mult` escalation when
  `chi2_red ≫ 1` near the crossings. Never mix membrane R-families in one FSS fit.
- **Budget**: ~87 NQS runs ≈ 120–150 GPU-h (L=6-dominated) + QMC ≈ 20 GPU-h.
- **Launch gate**: the FM-conventions session's certification ping (CERTIFIED or
  PARTIAL) — L≥8 QMC tooling is theirs; this campaign is L≤6 and independent of
  that outcome, but the queue handoff waits for the ping.
- **Launcher**: `bash nersc/launch_phaseB.sh` on Perlmutter (7 sbatch calls, 45
  array tasks). CERTIFIED received 2026-08-11; campaign-scale submission requires
  the user's own hands (session permission layer blocks autonomous fleet sbatch).

### diag_shift default: 3e-3 at L=6 (and presumptively L=7), not 1e-3

Overnight run (2026-08-11→12) found the right cut's near-transition window
(h_x≈0.65–1.25) badly divergence-prone at L=6 with the campaign default
`diag_shift=1e-3`: **8 of 12 attempted points needed escalation** to 3e-3
before converging (0.65, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.175, 1.25 all
diverged at 1e-3; only 0.75 survived unaided, and even that one later turned
out to have converged to an *unphysical* state — see the guard blind-spot
below). By contrast the equivalent L=5 window diverged only at the exact
transition point (h_x=1.0). Read as: L=6's larger Hilbert space makes the
near-transition SR landscape harder to navigate at the old default, and this
will very plausibly be worse, not better, at L=7. **Any future near-transition
sweep at L=6/L=7 should start at `diag_shift=3e-3`**, escalating to 5e-3/1e-2
only if that still diverges — not 1e-3→3e-3→5e-3→1e-2 from scratch.

**Guard blind spot (important, independent of the ds fix):** the divergence
guard's "sane" check is spread-based relative to a rolling baseline, not
energy-value-based — it can accept a wrong-regime state as "sane" and never
trip the 5-consecutive-rollback kill switch, so `diverged=True` does **not**
reliably fire on every bad run. One L=6 point (h_x=0.75) converged with a
clean guard history (no rollbacks past warmup) to E0 above the exact h=0
bound — variationally impossible, since any finite field can only lower the
ground energy. **Always cross-check a landed point's E0 against the h=0 bound
and its neighbors, not just the `diverged` flag** — this is now baked into
`analysis/scripts/check_convergence.py`'s bound check but was missed for several points
during an unattended stretch (2026-08-12 03:00–06:30) before being caught and
re-run.
