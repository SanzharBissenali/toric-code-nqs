# 3D bosonic toric-code phase-diagram campaign — config reference

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
| QA gate | `analysis/check_convergence.py --tree` (finite E, E<bound, ⟨A_v⟩≤1, not diverged, missing list) |

**Budget:** per hz-point across all 4 L ≈ 8.6 GPU-h → ~675 GPU-h total. L=7 is ~60 %
(5.1 h/run × 78 runs ≈ 400 h) — the biggest lever if the L set is ever trimmed.

**Caveats:**
- hz window is fixed [0.1, 0.4] at every hx, but hz_c drifts down as hx grows — at
  hx=0.8/1.0 the peak may slide below 0.1 (QA scan flags no-crossing; cheap re-launch
  with a lower `HZ_MIN`).
- L=4/L=5 wall times assume the estimated per-step; widen if the first medians run hot.
- Solver is dense for the whole campaign; SRt (`--qgt srt`) is a memory/scaling tool
  (no per-step win at k5/k6) reserved for k7+ where the dense S-matrix OOMs.
