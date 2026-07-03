# NQS-on-Perlmutter timing & scaling — work log

Terse lab-log for the NERSC A100 timing/scaling study of `ToricCNN_gridinv`
(3D bosonic toric code). Companion narrative: `nqs_nersc_timing_blog.md`.
Env: `tc-nqs` conda (jax 0.5.2 / netket 3.16.1), 1× A100-40GB, `jax_enable_x64`.

## 2026-07-03

### Failures diagnosed & fixed
- **Env ruled out.** `python -c "import jax_cuda12_plugin, nvidia.cudnn, nvidia.cublas"`
  → wheels OK; no `cudatoolkit` on `LD_LIBRARY_PATH`; `lax.conv_general_dilated`
  → `CONV OK` on the compute node. Not the [[tc-nqs-gpu-cudnn-fix]] issue.
- **int32 overflow @ L=6** (compile-time abort). Fatal line hidden above the
  backtrace: `F shape_util.cc:359] INVALID_ARGUMENT: invalid shape … dims=[3,60,-2062483456]`
  (2³¹ overflow of a dot operand). Fix: `--chunk_size`.
- **f64 conv OOM @ L=4** (runtime). `forces_expect_hermitian` conv
  `f64[1712128,12,7,7,7]` → tried 56 GB on 40 GB. `1712128 = n_samples(8192) ×
  n_conn(209)`, where `n_conn = #A_v(=L³=64) + N(=144) + 1`. Fix: `--chunk_size`
  (needed at ALL L≥4; it's a memory knob, not speed). Grid dim = `2L−1` (L4→7³).

### Code changes (committed)
- `Three_TC/builders.py`: `run_loop(time_phases=True, on_timing=...)` splits step
  into sample / grad / qgt / update, each behind `jax.block_until_ready`; prints
  `[t]` per-step + `[timing]` median. Verified identical to `driver.advance`
  (max |ΔE| = 0). Defaults: `n_samples 4096→8192`, `n_sweeps None(2N)→48`.
- `Three_TC/train.py`: logs `timing` into curve JSON + wandb; removed GPU
  sample-doubling.
- `nersc/submit_nqs_gridinv.sh`: `N_SAMPLES 16384→8192`, new `N_SWEEPS=48`,
  `CHUNK` default 2048 (all L).

### Per-step timing (s), gridinv, dense QGT, f64, n_samples 8192, n_chains 1024, n_sweeps 48
```
L   N     chunk  sample  grad   qgt   total     notes
4   144   2048    1.41    3.61   ~2.0   ~7.0     hx=hz=0.2 (near-converged)
5   300   2048    2.63   12.95   ~1.3  ~16.8     hx=hz=0.2 (step ~30)
6   540   2048    4.44   38.85   ~0.46 ~43.7     hx=hz=0.2 (step 1)
```
Doubling n_samples @ L4: total 5.20→9.7 (×1.86); grad ×2.00 (linear, compute-bound);
sample ×1.49 (sub-linear, GPU headroom); qgt tiny.

### Scaling fit (2 points, L4→L5) & projection
- `sample ~ N^0.85`, `grad ~ N^1.74` (theory ceiling N²; drifts →N² at large L).
- `qgt` ~ n_params-bound (const solve) + small N-linear Jacobian; inflates near
  convergence (ill-conditioned S at small diag_shift).
```
L   N     per-step (fit .. N²)   300-step run       5h-slots
6   540   ~48 .. ~63 s           ~4.0 .. ~5.2 h      1-2
7   882   ~100 .. ~155 s         ~8.6 .. ~12.9 h     2-3
8   1344  ~200 .. ~345 s         ~17 .. ~28.5 h      4-6
```
L=6 measured 43.7 s ⇒ on the fit. `chunk` must scale ∝1/N²: ≈2048/512/256 @ L6/7/8.

### Physics validation — E0(h) ≤ E0(0) = -(#A_v+#B_p) (strict for h≠0)
Anchor: OBC `-(L³+3(L−1)²L)`. Converged NQS above the anchor = failure.
```
L   anchor E0(0)   run                     result
4   -172           h=0                     E=-172.000000  delta 1e-9  spread→2e-4  (trivial: stabilizer state; PLUMBING ok, n_sweeps=48 unbiased)
4   -172           hx=hz=0.2, 300 it       E→-174.55 < -172  ✓  spread~0.4-0.7 (entangled floor)
5   -365           hx=hz=0.2, 200 it       crossed -365 @ step 32  ✓
6   -666           hx=hz=0.2, 10 it        TIMING only; diag_shift 1e-3 UNSTABLE (E -324→-286, spread 21→112) → use 1e-2 @ L≥6
```
Recipe that converges L4/L5: `--diag_shift 1e-3 --dt 0.01 --lr_min 0.001`, cosine
over full `n_iter`, ≥ few hundred steps. Pitfall: 30 steps + cosine horizon=n_iter
self-throttles → looks falsely plateaued. Reference-free quality: `Vscore =
N·Var(H)/⟨H⟩² → 0`.

### Validation CLIs (interactive A100, `-q interactive -A m5340_g`)
```
# L=4 finite-field benchmark (< -172)
python -u -m Three_TC.train --L 4 --bc OBC --model bosonic --arch ToricCNN_gridinv \
  --hx 0.2 --hz 0.2 --noninv_channels 4 --n_noninv 2 --inv_hidden 2 2 2 --kernel_size 4 \
  --dt 0.01 --lr_min 0.001 --diag_shift 1e-3 --qgt dense --n_iter 300 \
  --n_samples 8192 --n_chains 1024 --n_sweeps 48 --chunk_size 2048 \
  --out_dir $PSCRATCH/tc_nqs/bench --name L4_hx02_hz02 --no_wandb
# L=5: --L 5 ... --n_iter 200 ; L=6: --L 6 ... --diag_shift 1e-2 --chunk_size 2048
```

### Open items
- Confirm `diag_shift 1e-2` stabilizes L=6 over a full run.
- Make submit script `chunk_size` (and diag_shift) L-aware.
- Vscore/variance convergence study at L=5/6 to quantify ansatz quality.
