# Per-step speed levers for the 3D-TC NQS (branch `p3d/speed-research`, 2026-09-09)

Goal: make one VMC+SR step much cheaper at L=4,5,6 (complex ansatz first) without changing
the variational family or the converged energies. Everything measured on Perlmutter A100s
(40 GB nodes unless noted) at the production config — dual basis, OBC, `ToricCNN_gridinv`
nh 4→8 / inv 8,8 / kernel L−1, 8192 samples, 1024 chains, 48 sweeps, chunk 2048, dense QGT,
`diag_shift` 1e-3, dt 0.02 — at the production point (h_x, h_z) = (0.2, 0.26), h_y = 0.4
(complex) or 0 (real). Raw JSONs: `results/speed_bench/`, `results/speed_equiv/`; tables:
`analysis/scripts/summarize_speed_bench.py`, `compare_equiv.py`. Local gates:
`tests/test_speed_levers.py` (6 tests, all pass).

## 1. Where the time goes

A step = sampling (16 sweeps/chain of forward passes) + `expect_and_grad` ("grad") + QGT/SR.
"grad" is **network forward passes over the connected states**: n_conn = 1 + #B_p + N
(h_z and h_y flips share a state; measured `max_conn_size` 253/541/991 at L=4/5/6 OBC),
i.e. 2.1M / 4.4M / 8.1M evaluations per step. The energy VJP (1 backward over 8192
configs) and the QGT Jacobian (2×8192 backward passes) are negligible next to that.

Per-evaluation cost of the production model (`bench_hy_speed.py --microbench`, forward at
the E_loc batch size; HLO census of the compiled forward):

| L | ansatz | invariant block | compute | µs/config | TFLOP/s | E_loc forward s/step | lowering |
|---|---|---|---|---|---|---|---|
| 5 | complex | conv (nn.Conv) | c128 | 11.55 | 0.87 | 51.2 | complex conv → 45 cuDNN fp64 real convs |
| 5 | complex | dense GEMM | c128 | 7.34 | 1.36 | 32.5 | cuBLAS zgemm |
| 5 | complex | dense GEMM | c64 strict | 1.96 | 5.1 | 8.7 | cuBLAS cgemm |
| 5 | complex | dense GEMM | c64 TF32 | 1.01 | 9.9 | 4.5 | cuBLAS cgemm, TF32 |
| 6 | complex | dense GEMM | c128 | 13.3 | 2.3 | 108 | zgemm |
| 6 | complex | dense GEMM | c64 TF32 | 2.02 | 15.7 | 16.4 | cgemm, TF32 |
| 6 | real | conv | f64 | 6.18 | 1.28 | 50.2 | 15 cuDNN fp64 convs |
| 6 | real | dense GEMM | f64 | 2.43 | 3.3 | 19.7 | dgemm |
| 6 | real | dense GEMM | f32 strict | 1.83 | 4.3 | 14.9 | sgemm |
| 6 | real | dense GEMM | f32 TF32 | 0.72 | 11.0 | 5.8 | sgemm, TF32 |

The baseline's cost is a **kernel-quality** problem: the kernel-(L−1) 3-D convolution in
float64/complex128 has no fast GPU path (<1 TFLOP/s ≈ 5 % of the A100). Two orthogonal
fixes, both leaving the parameter tree and checkpoints untouched:

* **`--inv_impl dense`** (`networks.UnfoldedConv3D`): each invariant-block conv evaluated as
  ONE GEMM `(B, P·C_in) @ (P·C_in, P·C_out)` against the block-Toeplitz matrix unfolded from
  the same `(k,k,k,C_in,C_out)` kernel (rebuilt per call; 48 MB at L=6). Same parameter
  names (`Conv_i/kernel,bias`), same init from the same seed, log ψ identical to 1e-15.
* **`--compute_dtype float32`**: parameters stay complex128/float64; sampling, local energies
  and the energy VJP run in complex64/float32 (strict: `Precision.HIGHEST`); log ψ is cast
  back; the dense-QGT Jacobian and SR solve run on an exact-precision twin of the model
  (`builders.exact_qgt_apply_fun`). `--compute_dtype tf32` = XLA's Ampere default (TF32
  tensor cores, 10-bit mantissa) as an explicit opt-in.
* **`--qgt_solver kernel`** (`builders.kernel_solve`): the same regularised SR solution via the
  Woodbury identity on the (2 n_samples)² Gram matrix + one refinement step, never forming
  the n_params² S matrix — at L=6 complex S is 11 GB and NetKet's `cholesky` copies it
  (the 20.8 GiB single allocation that OOMed chunk 2048 on a 40 GB node).

`get_conn_padded` (NetKet's generic PauliStrings kernel) is the second item once the forward
is fast (in situ ≤5 s/step of 26 at L=6); chunk size does not touch it (§3).

## 2. Measured s/step (median, compile step excluded; "peak" = jax peak_bytes_in_use)

Complex ansatz (h_y = 0.4). † = QGT stage measured through the GEMM model before the
conv-twin fix; with the twin it is the baseline's value. ‡ = 80 GB node.

| L | impl / compute / solver | sample | grad | qgt | **total** | speed-up | peak GiB |
|---|---|---|---|---|---|---|---|
| 4 | conv / f64 / cholesky (**baseline**) | 3.93 | 9.29 | 0.33 | **13.6** | 1.0× | 3.7 |
| 4 | dense / f64 | 3.18 | 7.34 | 0.33 | **10.9** | 1.25× | 3.7 |
| 4 | conv / f32 strict | 1.19 | 3.67 | 0.33 | **5.2** | 2.6× | 3.7 |
| 4 | **dense / f32 strict** | 0.64 | 2.23 | 0.33 | **3.2** | **4.2×** | 3.7 |
| 4 | conv / tf32 | 0.61 | 2.04 | 0.33 | **3.0** | 4.5× | 3.7 |
| 4 | dense / tf32 | 0.43 | 1.62 | 0.33 | **2.4** | 5.7× | 3.7 |
| 5 | conv / f64 / cholesky (**baseline**)‡ | 10.9 | 53.9 | 1.26 | **66.1** | 1.0× | 9.0 |
| 5 | dense / f64‡ | 7.34 | 34.9 | 1.85† | **44.1** | 1.5× | 9.0 |
| 5 | conv / tf32‡ | 1.53 | 10.2 | 1.26 | **13.0** | 5.1× | 9.0 |
| 5 | dense / tf32‡ | 0.83 | 6.21 | 1.85† | **8.9** | 7.4× | 9.0 |
| 5 | **dense / f32 strict**, chunk 8192 / 16384 | 1.87 | 12.5 / 11.3 | 1.33 | **15.7 / 14.5** | **4.2–4.6×** | 14.3 |
| 5 | dense / f32 strict, chunk 2048 | — | — | — | *unmeasured (job queued); expect ≈15* | | |
| 6 | conv / f64 / cholesky, chunk 512 (**baseline**, smoke 58113537) | 22.4 | 380.8 | 4.2 | **407** | 1.0× | (chunk 2048 OOM: 20.8 GiB S+copy) |
| 6 | dense / f64 / kernel | 14.6 | 126.7 | 2.17 | **143.5** | 2.8× | 14.3 |
| 6 | conv / f32 strict / kernel‡ | 9.51 | 102.4 | 2.12 | **114.0** | 3.6× | 14.2 |
| 6 | **dense / f32 strict / kernel**‡ | 3.55 | 42.5 | 2.10 | **48.1** | **8.5×** | 14.2 |
| 6 | conv / tf32 / kernel | 2.56 | 31.7 | 2.13 | **36.4** | 11× | 14.3 |
| 6 | dense / tf32 / kernel | 1.61 | 21.9 | 2.13 | **25.7** | 16× | 14.3 |

Real ansatz (h_y = 0; the earlier "L6 real 28 s" was at h_z = 0 where n_conn = 451, not 991):

| L | impl / compute / solver | sample | grad | qgt | **total** | speed-up | peak GiB |
|---|---|---|---|---|---|---|---|
| 6 | conv / f64 / cholesky (**baseline**) | 4.83 | 54.1 | 0.64 | **59.5** | 1.0× | 7.6 |
| 6 | dense / f64 | 1.86 | 23.7 | 0.63 | **26.2** | 2.3× | 6.5 |
| 6 | **dense / f32 strict**, chunk 8192 / 16384 | 1.50 | 19.4 / 19.6 | 0.65 | **21.6 / 21.8** | **2.8×** | 11.3 |
| 6 | conv / tf32 | 1.02 | 14.8 | 0.63 | **16.5** | 3.6× | 6.4 |
| 6 | dense / tf32 | 0.72 | 11.5 | 0.64 | **12.9** | 4.6× | 6.4 |
| 4, 5 | all variants | | | | *unmeasured (L4-real job queued; L5-real Pauli cache cold)* | | |

## 3. Chunk size is not a lever

L=6 complex dense/tf32/kernel: 25.7 / 25.8 / 26.0 s at chunk 2048 / 8192 / 16384 while peak
memory grows 14 → 24 GiB; L=5/L=6 strict f32 likewise flat. Keep 2048.

## 4. Memory at L=6 complex is the S matrix, not the network

n_params = 18 673 complex = 37 346 real ⇒ S = 11.2 GB, `cholesky` needs two copies; the
network path at chunk 2048 peaks at ~14 GiB with the levers on. `--qgt_solver kernel` is
mandatory at L=6 complex (dp identical to cholesky: 2.4e-12 at L=4, 3.9e-13 at L=2).
The QGT twin must also be the *conv* implementation: NetKet's dense Jacobian vmaps the
backward pass per sample, so the unfolded matrix's cotangent would be chunk × (P·C)² × 16 B
= 98 GB at L=6 (measured OOM). `exact_qgt_apply_fun` handles both automatically.

## 5. Equivalence

### 5a. Identical samples, identical parameters — L=4 complex, production config, after 5 warm steps
(`check_equivalence.py`, `results/speed_equiv/check_L4_hy0.4_strict.json`; max-norm relative
deviations from conv/f64; E = −159.243 ± 0.093)

| lever | log ψ | E | gradient | SR update dp | S |
|---|---|---|---|---|---|
| `--qgt_solver kernel` | — | — | — | **2.4e-12** | — |
| `--inv_impl dense` (f64) | 3.8e-16 | 0 | 1.5e-15 | **8.4e-13** | 5.6e-17 |
| `--compute_dtype float32` strict, dense | 4.5e-7 | 7e-6 × err | 2.2e-6 | 1.9e-3 | 5.6e-17 (twin) |
| `--compute_dtype float32` strict, conv | 4.6e-7 | 3e-6 × err | 1.7e-6 | 1.4e-3 | 5.6e-17 (twin) |
| `--compute_dtype tf32`, dense | 4.0e-4 | 1e-2 × err | 9.3e-4 | 0.44 | 5.6e-17 (twin) |

Exact levers are exact to double roundoff. For the float levers dp deviates by the gradient
deviation × up to 1/`diag_shift` along the flat directions of S (the amplification the MC
noise gets too); strict fp32 stays at 1e-3, TF32 does not.

### 5b. Same-seed 60-step trajectories (L=4 complex; per-step error bars ≈ 0.03)
GPU GEMM/conv kernels are not bitwise deterministic, so even the exact kernel-vs-cholesky
pair decorrelates within a few steps; two runs differ by the *optimisation* noise. The
seed-0 → seed-1 baseline control sets that scale: late-window E −176.421 vs −176.577
(Δ = 0.16). Every lever sits inside it, strict fp32 well inside:

| run (60 steps) | s/step | E last-10 (± 0.009) | Vscore | final E0 |
|---|---|---|---|---|
| conv / f64 / cholesky, seed 0 (baseline) | 13.6 | −176.421 | 2.86e-2 | −176.443 |
| conv / f64 / cholesky, seed 1 (control) | 14.0 | −176.577 | 3.01e-2 | −176.576 |
| dense / f32 strict / cholesky | 3.2 | −176.408 | 2.89e-2 | −176.425 |
| dense / f32 strict / kernel | 3.4 | −176.388 | 2.88e-2 | −176.447 |
| dense / tf32 / cholesky | 2.4 | −176.295 | 2.83e-2 | −176.337 |
| dense / tf32 / kernel | 2.6 | −176.358 | 2.88e-2 | −176.373 |

### 5c. Converged energy vs the production run (500 steps, 8 final rounds)
Reference `results/hy_cuts_L4/up/hy0.4/L4/…hx0.2_hz0.26_hy0.4…k3.json` (conv/f64, same
dt/lr_min/diag_shift): E0 = −177.2006 ± 0.014, Vscore 0.042. 500-step runs of dense/tf32
(seeds 0–2) and dense/f32-strict (seeds 0–1) were queued (`speed_equiv_job.sh` with
CKPT_EVERY=10 resume chaining); at write time seed 1 of each was mid-flight and seed 0 /
seed 2 still queued behind other debug jobs — **unmeasured here**; evaluate with
`compare_equiv.py --final REF results/speed_equiv/*it500*.json` when they land (JSONs are
written to `$PSCRATCH/tc_nqs/speed_equiv/`).

## 6. Not levers
Holomorphic Jacobian (split Re/Im activations ⇒ non-holomorphic); minSR/SRt (QGT stage is
0.3–2 s; the L=6 memory problem is solved by `kernel` inside the guarded dense path);
incremental log ψ for single flips (the kernel-(L−1) block couples every cell); n_samples
4096 (halves everything, √2 larger error bars — trajectory job queued, unmeasured).

## 7. Recommendation

| L | ansatz | settings | s/step now → new | gain | status of the evidence |
|---|---|---|---|---|---|
| 4 | complex | `--inv_impl dense --compute_dtype float32` | 13.6 → 3.2 | 4.2× | gate + 60-step ✓; 500-step pending |
| 5 | complex | `--inv_impl dense --compute_dtype float32` | 66 → ≈15 | ≈4.4× | speed at chunk 8192/16384; chunk-2048 point queued |
| 6 | complex | `--inv_impl dense --compute_dtype float32 --qgt_solver kernel` | 407 → 48 (80 GB node) | 8.5× | `kernel` needed for memory; strict-f32 timing on 40 GB node not measured (tf32: 25.7) |
| 4–6 | real | `--inv_impl dense` (exact); add `--compute_dtype float32` for a further ~20 % | L6: 59.5 → 26.2 (→ 21.6) | 2.3–2.8× | exact lever gate ✓; L4/L5 real timings unmeasured |
| any | either | `--compute_dtype tf32` | another 1.7–2× on top | | log ψ 4e-4, dp 44 % off per step; 60-step E inside seed spread but 500-step statistics pending — do NOT enable in production yet |

Wrappers: `INV_IMPL=dense COMPUTE_DTYPE=float32 [QGT_SOLVER=kernel]` on
`submit_nqs_gridinv.sh` / `submit_nqs_batch.sh` (carried through AUTO_RESUBMIT); the campaign
launcher's env dicts in `phase3d_grid.py` need the same three keys. No checkpoint, `--resume`
or resume-config-guard change (the keys are not in `_RESUME_CHECK_KEYS`, so a chunk can switch
mid-chain). Risks: (i) strict fp32 perturbs the gradient at 2e-6 and the SR update at ~2e-3
per step — smaller than MC noise, but not bitwise; (ii) `kernel` divides by `diag_shift`:
keep `diag_shift ≥ 1e-3` (refinement step included; equality to cholesky 1e-12 measured);
(iii) the GEMM twin rule means the QGT stage keeps the conv cost (0.3–2 s); (iv) all L=6
complex timings with strict fp32 are from an 80 GB node.

### 5c (update) — converged runs that landed before the write-up
reference: gridinv_dual_L4_OBC_hx0.2_hz0.26_hy0.4_n2x4_nh4-8_inv8-8_k3  E0 = -177.2006 ± 0.0140  Vscore 4.200e-02  (500 steps, 8 final rounds)
| run | impl | compute | solver | n_s | seed | steps | s/step | E0 ± err | ΔE0 vs ref | z | Vscore |
|---|---|---|---|---|---|---|---|---|---|---|---|
| equiv_L4_dense_tf32_cholesky_n8192_it500_s1 | dense | tf32 | cholesky | 8192 | 1 | 500 | 2.33 | -177.1830 ± 0.0144 | +0.0177 | +0.88 | 4.390e-02 |
