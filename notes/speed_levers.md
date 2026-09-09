# Per-step speed levers for the 3D-TC NQS (branch `p3d/speed-research`, 2026-09-09)

Goal: make one VMC+SR step much cheaper at L=4,5,6 — complex ansatz (h_y≠0) first —
without changing the variational family or the converged energies. Everything below is
measured on Perlmutter A100s at the production config (dual basis, OBC, `ToricCNN_gridinv`
nh 4→8, inv 8,8, kernel L−1, 8192 samples / 1024 chains / 48 sweeps, chunk 2048, dense
QGT + Cholesky, `diag_shift` 1e-3) at the production point (h_x, h_z) = (0.2, 0.26),
h_y = 0.4 (complex) or 0 (real). Raw JSONs: `results/speed_bench/`, `results/speed_equiv/`;
tables via `analysis/scripts/summarize_speed_bench.py`, `compare_equiv.py`.

## 1. Where the time goes (profile)

One step = sample (16 sweeps/chain × 1024 chains of forward passes) + `expect_and_grad`
("grad": local energies over the connected states + one VJP) + QGT/SR.
The `grad` stage is **network forward passes over the connected states**:
n_conn = 1 + #B_p + N (h_z and h_y single-flips share a connected state; verified
`max_conn_size` = 19/91/253/541/991 at L=2..6 OBC) ⇒ n_samples·n_conn = 2.1M / 4.4M / 8.1M
forward evaluations per step at L=4/5/6. Nothing else is close: the energy VJP is one
backward over 8192 configs, the QGT Jacobian 2×8192 backward passes.

Per-evaluation cost (`--microbench`, one forward call at the E_loc batch size, HLO census):

| L | ansatz | impl | compute | µs/config | achieved TFLOP/s | E_loc forward est. s/step | how the invariant block is lowered |
|---|---|---|---|---|---|---|---|
| 5 | complex | conv | c128 | 11.55 | 0.87 | 51.2 | complex conv → 45 cuDNN real-conv calls, fp64 |
| 5 | complex | dense | c128 | 7.34 | 1.36 | 32.5 | 5 cuBLAS zgemm |
| 5 | complex | conv | c64 (tf32) | 1.96 | 5.10 | 8.7 | cuDNN fp32 |
| 5 | complex | dense | c64 (tf32) | 1.01 | 9.94 | 4.5 | 5 cuBLAS cgemm |
| 6 | complex | dense | c128 | 13.28 | 2.39 | 107.8 | cuBLAS zgemm |
| 6 | complex | dense | c64 (tf32) | 2.02 | 15.70 | 16.4 | cuBLAS cgemm |
| 6 | complex | conv | c64 (tf32) | 3.29 | 9.64 | 26.7 | cuDNN fp32 (45 real-conv calls) |
| 6 | real | conv | f64 | 6.18 | 1.28 | 50.2 | 15 cuDNN fp64 convs |
| 6 | real | dense | f32 (tf32) | 0.72 | 11.02 | 5.8 | cuBLAS sgemm |

So the baseline `grad` time is a **kernel-quality** problem, not an algorithmic one: the
kernel-(L−1) complex128 3-D convolution has no fast GPU path (cuDNN fp64 real convs at
<1 TFLOP/s ≈ 5 % of the A100). Two orthogonal fixes, both keeping the same parameters:

* **`--inv_impl dense`** (`networks.UnfoldedConv3D`): with kernel L−1 on an L³ grid the conv
  is a dense linear map in disguise; evaluate each invariant-block layer as ONE GEMM
  `(B, P·C_in) @ (P·C_in, P·C_out)` against the block-Toeplitz matrix unfolded from the same
  `(k,k,k,C_in,C_out)` kernel (rebuilt per call: 48 MB at L=6, negligible). ~2× the MACs of
  the conv, but cuBLAS at tensor-core rates. Same parameter names (`Conv_i/kernel,bias`),
  same init from the same seed, log ψ identical to 1e-15 (tests) ⇒ checkpoints interchangeable.
* **`--compute_dtype float32`**: parameters stay complex128/float64; the forward/backward
  pass (sampling, E_loc, energy VJP) runs in complex64/float32 and log ψ is cast back. The
  dense-QGT Jacobian and the SR solve use an exact-precision twin of the model
  (`builders.exact_qgt_apply_fun`, same params), so the SR geometry is unchanged.

`get_conn_padded` (NetKet's generic PauliStrings kernel) is the second-largest item once
the forward is fast: ~1.2 / 3.4 / 6 s/step at L=4/5/6 with chunk 2048 (4096 two-sample
calls per step at L=6) — see §3 for the chunk-size lever.

## 2. Measured s/step (median over steps, compile step excluded)

Complex ansatz, h_y = 0.4:

| L | variant | sample | grad | qgt | **total** | speed-up | peak GiB |
|---|---|---|---|---|---|---|---|
| 4 | conv / f64 / cholesky (baseline) | 3.93 | 9.29 | 0.33 | **13.55** | 1.0× | 3.7 |
| 4 | dense / f64 | 3.18 | 7.34 | 0.33 | **10.85** | 1.25× | 3.7 |
| 4 | conv / tf32 | 0.61 | 2.04 | 0.33 | **2.98** | 4.5× | 3.7 |
| 4 | dense / tf32 | 0.43 | 1.62 | 0.33 | **2.39** | 5.7× | 3.7 |
| 5 | conv / f64 / cholesky (baseline) | 10.90 | 53.90 | 1.26 | **66.07** | 1.0× | 9.0 |
| 5 | dense / f64 | 7.34 | 34.85 | 1.85† | **44.05** | 1.5× | 9.0 |
| 5 | conv / tf32 | 1.53 | 10.22 | 1.26 | **13.00** | 5.1× | 9.0 |
| 5 | dense / tf32 | 0.83 | 6.21 | 1.85† | **8.90** | 7.4× | 9.0 |
| 6 | conv / f64 / cholesky, chunk 512 (baseline, smoke 58113537) | 22.4 | 380.8 | 4.2 | **407** | 1.0× | — (chunk 2048 OOMs: 20.8 GiB alloc = S + Cholesky copy) |
| 6 | dense / f64 / kernel, chunk 2048 | 14.62 | 126.74 | 2.17 | **143.5** | 2.8× | 14.3 (40 GB node) |
| 6 | conv / tf32 / kernel, chunk 2048 | 2.56 | 31.71 | 2.13 | **36.4** | 11.2× | 14.3 |
| 6 | dense / tf32 / kernel, chunk 2048 | 1.61 | 21.92 | 2.13 | **25.7** | 15.9× | 14.3 |

† run before the twin fix (QGT Jacobian still through the GEMM model); with the conv twin
the qgt stage is the baseline's 1.26 s.

Real ansatz, h_y = 0 (note: the earlier "L6 real 28 s" figure was at h_z = 0, where
n_conn = 451 instead of 991; these are at the production point):

| L | variant | sample | grad | qgt | **total** | speed-up | peak GiB |
|---|---|---|---|---|---|---|---|
| 6 | conv / f64 / cholesky (baseline) | 4.83 | 54.05 | 0.64 | **59.52** | 1.0× | 7.6 |
| 6 | dense / f64 | 1.86 | 23.68 | 0.63 | **26.18** | 2.3× | 6.5 |
| 6 | conv / tf32 | 1.02 | 14.82 | 0.63 | **16.47** | 3.6× | 6.4 |
| 6 | dense / tf32 | 0.72 | 11.54 | 0.64 | **12.89** | 4.6× | 6.4 |

## 3. Chunk size is not a lever

L=6 complex, dense/tf32/kernel: 25.7 / 25.8 / 26.0 s/step at chunk 2048 / 8192 / 16384
while peak memory grows 14.3 → 24.2 GiB. The E_loc kernel is compute-bound, not
launch-bound, so `get_conn_padded` (NetKet's generic PauliStrings kernel, ~20 % of the
fast step at L=6) would need a leaner operator, not bigger chunks. Keep chunk 2048.

## 3b. Single precision: TF32 is NOT single precision

XLA's default matmul/conv precision on Ampere is TF32 (10-bit mantissa). Measured on
identical samples at L=4 complex (`check_equivalence.py`, 5 warm steps): log ψ relative
deviation 4.1e-4, energy −4.5e-4 (0.5 % of the error bar), gradient 8e-4, and — because
the SR solve amplifies gradient noise along the flat directions by up to 1/`diag_shift` —
the update vector dp deviates by 44 % in max-norm (the same amplification MC noise gets,
but avoidable). `--compute_dtype float32` therefore means strict single precision
(`lax.Precision.HIGHEST` on the ansatz's matmuls/convs: log ψ to ~1e-6, as on CPU);
`--compute_dtype tf32` is the opt-in for the faster arithmetic. All "f32" rows measured
before this change are labelled tf32 in `results/speed_bench/` and in the tables here.

## 4. Memory at L=6 complex: the S matrix, not the network

n_params = 18 673 complex ⇒ 37 346 real ⇒ S is 11.2 GB; NetKet's `cholesky` materialises S
and `cho_factor` copies it (the 20.8 GiB single allocation that OOMed chunk 2048 on a 40 GB
node). `--qgt_solver kernel` (`builders.kernel_solve`) returns the SAME regularised solution
through the Woodbury identity on the (2 n_samples)² = 16384² Gram matrix (2.1 GB), plus one
refinement step reusing the Cholesky factor: dp agrees with `cholesky` to 4e-13 (L=2 SR step)
and TODO at L=4. Use it whenever n_params_real > 2·n_samples (L=6 complex).

NetKet's dense Jacobian vmaps the backward pass per sample, so the unfolded GEMM matrix
must never be in that graph (its per-sample cotangent is chunk × (P·C)² × 16 B = 98 GB at
L=6, measured OOM): `exact_qgt_apply_fun` therefore hands the QGT a double-precision,
conv-implementation twin whenever either lever is on. The energy VJP reduces over the batch
first and is fine with the GEMM path.

## 5. Equivalence — TODO from jobs 58115449 / 58116348

## 6. Not levers

* Holomorphic Jacobian: the ansatz uses split (Re/Im) ELU/sigmoid ⇒ non-holomorphic; the
  complex-mode Jacobian is required (and is a negligible cost anyway).
* minSR/SRt (`--qgt srt`): the QGT stage is 0.3–4 s; the memory problem it would solve is
  solved by `--qgt_solver kernel` inside the guarded, timed dense path.
* Incremental log ψ for single-site flips: the kernel-(L−1) invariant block makes every
  output cell depend on every flip; only the first conv layer is linear in the flip.

## 7. How to turn it on

`python -m tc3d.train ... --inv_impl dense --compute_dtype float32 [--qgt_solver kernel]`
(same flags on `tc3d.sweep`); wrappers `submit_nqs_gridinv.sh` / `submit_nqs_batch.sh` take
`INV_IMPL=dense COMPUTE_DTYPE=float32 QGT_SOLVER=kernel` and carry them through
AUTO_RESUBMIT; the campaign launcher's env dicts (`phase3d_grid.py`) need the same three
keys. Nothing changes in checkpoints, `--resume`, or the resume-config guard. Tests:
`cd tests && ../.venv/bin/python test_speed_levers.py`.

## 8. Recommendation — TODO
