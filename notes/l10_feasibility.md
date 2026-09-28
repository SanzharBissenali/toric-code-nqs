# L=8 / L=10 feasibility — dual-basis `ToricCNN_gridinv` (3D toric code, OBC)

Can we afford a 500-step point and a 7-point warm chain (500 + 6×200 = 1700 steps)
at L=8 / L=10 on Perlmutter A100s?

Structure **derived** from `networks.py::ToricCNN_gridinv_dual` + `geometry.py`,
checked to the digit against `results/speed_bench`: `N = 3L²(L−1)`,
`N_p = 3L(L−1)²`, `n_conn = 1 + N_p + N`, `n_params = 1673 + 136k³`,
FLOPs/config `= 272L³k³ + 1080N`, `get_conn_padded = n_samples·n_conn·N·1.35 ns`.
Runtime calibrated at 1.19 TFLOP/s — the measured L=6 conv/float64 grad
throughput, i.e. the path `feat/phase3d-campaign` actually runs. h_y ≠ 0 leaves
`n_conn` unchanged but makes weights complex: ≈4× FLOPs, 2× memory.

| L | N | n_conn | k | n_params | dense QGT | SRt QGT | max chunk (int32) | act@chunk | fwd | get_conn | s/step real | s/step cplx | 500 steps | 7-pt chain | GPU-h/cut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 540 | 991 | 5 | 18,673 | 2.6 GB | 0.5 GB | 9,942 | 6.6 GB | 54 s | 6 s | 70 s | 232 s | 9.7 h | 33 h | 33 |
| 7 | 882 | 1639 | 5 | 18,673 | 2.6 GB | 0.5 GB | 6,260 | 10.5 GB | 142 s | 16 s | 168 s | 595 s | 23 h | 3.3 d | 79 |
| 7 | 882 | 1639 | 6 | 31,049 | 7.2 GB | 0.5 GB | 3,623 | 18.1 GB | 238 s | 16 s | 264 s | 978 s | 37 h | 5.2 d | 125 |
| **8** | 1344 | 2521 | **5** | 18,673 | 2.6 GB | 0.5 GB | 4,194 | 15.6 GB | 327 s | 38 s | **375 s** | 1357 s | 2.2 d | 7.4 d | **177** |
| 8 | 1344 | 2521 | 7 | 48,321 | **17.4 GB** | 0.5 GB | 1,528 | 32.0 GB | 854 s | 38 s | 902 s | 3464 s | 5.2 d | 17.7 d | 426 |
| **10** | 2700 | 5131 | **5** | 18,673 | 2.6 GB | 0.5 GB | 2,147 | 30.5 GB | 1304 s | 153 s | **1467 s** | 5379 s | 8.5 d | 28.9 d | **693** |
| 10 | 2700 | 5131 | 9 | 100,817 | **75.7 GB** | 0.5 GB | 368 | 32.0 GB | 7107 s | 153 s | 7270 s | 28591 s | 42 d | 143 d | 3433 |

A plane = 8 cuts ⇒ ×8 the GPU-h column. n_samples 8192, `--qgt dense`.

**Binding constraint.** Activation = XLA's implicit-GEMM unfold of the k³ conv,
`2·chunk·L³·k³·C·8 B` (predicts 6.6 GB vs **6.47 GB measured** at L=6/k=5/chunk 2048).
Chunkable, so never fatal — but it forces the chunk down, and
`chunk·L³·k³·C < 2³¹` is the known int32 compile wall ("max chunk"). So: L≤7 nothing
binds; L=8/k=5 and L=10/k=5 are **time-bound only**; L=8/k=7 is bound by the 17.4 GB
dense QGT (needs `hbm80g` or SRt); **L=10/k=9 is impossible on any node** (75.7 GB
QGT + a Cholesky copy).

**Three walls found, in the order they bite.**
1. *Not the GPU* — `create_hamiltonian`'s `LocalOperator` `+=` assembly: measured
   257.8/660.3/1356.4 s at L=4/5/6 (O(n^1.24)) ⇒ **~70 min at L=8, ~2.8 h at L=10**,
   single-core, before step 1, on every cold start. Fixed:
   `analysis/scripts/prime_pauli_cache.py` enumerates the strings directly —
   L=7/8/10 both dtypes prime in **4.9 s**. Gate is exact, no tolerance: multiset
   match vs `create_hamiltonian` at L=2/3; production-cache match at L=4/5/6
   (h_y sign included); and end-to-end at L=2 OBC `max|ΔH| = max|ΔE| = 0`, with
   identical `max_conn_size` and local energies at L=3/4.
2. `MAX_RESUBMITS` defaults to **8**, but one L=8/k=5 point needs **11** links
   against the 5 h cap (L=8 chain 36; L=10 point 41, chain **139**). Silently
   truncates today.
3. **Amdahl:** `get_conn_padded` is index/copy work, immune to precision and kernel
   width. It is 11% of a L=6 step but 31% at L=10 — and **65%** once float32 +
   `inv_impl=dense` land. The full lever stack takes L=10 to ~234 s/step and stops.

**Dense QGT is rank-deficient everywhere here** (`n_params > n_samples = 8192` for
all k ≥ 5), so `--qgt srt` solves the identical regularized problem: at L=10/k=9,
0.50 GB vs 75.7 GB and 1.4 s vs 50.8 s. It is already wired.

## Verdict

**L = 8 — feasible at campaign cost, only with the kernel capped at 5.** 375 s/step
⇒ 2.2 d per 500-step point (~11 chained links), 7.4 d and **~177 GPU-h per 7-point
cut**, ~1400 GPU-h per plane. A campaign request, not a run. At full-span k=7 it is
426 GPU-h/cut and needs 80 GB nodes — not affordable. h_y ≠ 0 at L=8 is ~4× worse
(~640 GPU-h/cut) — out.

**L = 10 — not feasible as the code stands.** Capped at k=5 it is 1467 s/step ⇒
**8.5 d for one 500-step point** (~41 chained links), **693 GPU-h/cut**,
~5500 GPU-h/plane. At k=9 it is dead twice: 75.7 GB dense QGT and 42 d/point.
Complex at L=10 is out at any kernel.

*Reliability:* the work counts are exact; the one fitted number is throughput, still
climbing at L=6 (0.56/0.82/1.19 TFLOP/s at L=4/5/6), so the `fwd` column is an upper
bound good to ~2×. `get_conn` does not move with it, so an L=10 step cannot fall
below ~250 s without new code. Neither verdict flips under a 2× speedup.

## What must change to make L=10 run at all

Each is a physics/accuracy decision to validate at L=6 first.

1. **Cap the kernel at 5** (mandatory) — k=L−1=9 is dead on memory *and* time.
   Open question: does a 5-tap invariant block still span a 10³ grid well enough for
   topological long-range order? (L=7 found k5 ≡ k6 in energy — encouraging, not L=10.)
2. **`--qgt srt`** (mandatory above k=5) — 150× less QGT memory, and the correct
   choice anyway when n_params > n_samples.
3. **float32/tf32 compute** — 3.6× at L=6 (16.47 vs 59.52 s/step), same peak memory.
   Knob absent on `feat/phase3d-campaign`; needs a precision-vs-Vscore study.
4. **Port the speed clone's `inv_impl=dense`** — 2.3× at L=6/f64; 4.6× with (3).
5. **Narrow to `INV="2 2 2"`** (the L=7 production choice) — 5.2× fewer FLOPs in the
   dominant block; n_params 18,673 → 4,913.
6. **Prime the Pauli cache** — done, `prime_pauli_cache.py`.
7. **Attack `get_conn_padded`** — the new bottleneck once 3–5 land (65% of a step).
   Not implemented anywhere.
8. **`n_sweeps` ∝ N** — 48 sweeps decorrelate 8.9% of the lattice at L=6, 1.8% at
   L=10. Cheap; a correctness issue, not a speed one.
9. **Re-examine 500 steps** — L=6 needed 500; larger L will not need fewer.

Levers 1–5 put L=10 at ~234 s/step ⇒ 32 h/point, **~111 GPU-h/cut**, ~890 GPU-h/plane
— ~5× cheaper than L=8 is today, and the floor until lever 7.

**Recommendation.** L=8 at k=5 is fundable today (raise `MAX_RESUBMITS`). Do not
schedule L=10 until levers 3–5 are implemented and A/B'd at L=6, where the answer is
already known.
