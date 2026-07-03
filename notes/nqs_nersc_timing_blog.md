# Getting 3D toric-code NQS onto Perlmutter: debugging, timing, and scaling

*A narrative write-up of bringing the `ToricCNN_gridinv` variational ansatz to
NERSC GPUs — what broke, how we instrumented per-step cost, how it scales with
system size, and how we know the results are trustworthy.*

## The goal

Run the neural-quantum-state (NetKet/JAX) study of the perturbed 3D bosonic
toric code on Perlmutter A100s, and answer three questions before committing to
long production sweeps:

1. Can we even get the GPU jobs to run reliably?
2. How much wall-clock does one gradient step cost, and how does it scale with
   linear size `L`?
3. Is the ansatz actually good — does it beat a trivial reference?

## Act 1 — three ways a GPU job dies (and one knob that fixes two of them)

The batch jobs kept aborting. Peeling back the failures:

- **Not the environment.** The abort *looked* like the known cuDNN mismatch
  (`jax-cuda12-plugin[with-cuda]`), but a minimal `lax.conv_general_dilated`
  compiled and ran fine on the compute node (`CONV OK`, 4 A100s visible). The GPU
  stack was healthy. Lesson: reproduce with a one-line probe before blaming the env.

- **int32 overflow at L=6.** The real fatal line, hidden above a bare C++
  backtrace, was `INVALID_ARGUMENT: invalid shape … dims=[3,60,-2062483456]`. That
  negative dimension is a **2³¹ integer overflow** — a matmul operand exceeding
  what XLA can represent, during layout assignment (compile time). That's why a
  bare conv compiled but the full program didn't.

- **float64 conv OOM at L=4.** With the overflow chunked away, L=4 then hit
  `RESOURCE_EXHAUSTED: 56 GB`. `expect_and_grad` evaluates the network on *every
  Hamiltonian-connected configuration of every sample at once* —
  `n_samples × n_conn ≈ 8192 × 209 ≈ 1.7M` configs — and in float64 that
  convolution is 56 GB on a 40 GB card.

Both the overflow and the OOM are fixed by the same knob: **`--chunk_size`**,
which tiles the sample batch. The important mental correction: **`chunk_size` is
a *memory* knob, not a speed knob** — it changes how the work is tiled, never the
total FLOPs. It's needed at *every* `L ≥ 4`, not just large `L`.

## Act 2 — instrumenting the step

To make budget decisions from data rather than theory, we split each VMC+SR step
into its three real phases — **sample**, **grad** (local energy + gradient), and
**qgt** (the SR/QGT solve) — each timed behind a `jax.block_until_ready` barrier
(JAX is async; without the barrier you time dispatch, not compute). The
instrumented loop was verified **byte-for-byte identical** to NetKet's
`driver.advance` (max |ΔE| = 0 on a tiny proxy), and as a bonus it dropped a
redundant second energy evaluation the original loop did purely for logging.

We also fixed the sampling cost model: `n_sweeps` was defaulting to `2N`
(proposals ∝ N → **O(N²)** sampling). Pinning it to a constant (48) makes
sampling **O(N)**. Confirmed harmless: at `h=0` the run still converged to machine
precision with `n_sweeps=48`.

## Act 3 — what a step actually costs

Two clean data points (identical config, only `L` differs), per phase:

| L | N | sample | grad | total |
|---|---|---|---|---|
| 4 | 144 | 1.4 s | 3.6 s | ~7 s |
| 5 | 300 | 2.6 s | 12.9 s | ~17 s |
| 6 | 540 | 4.4 s | **38.8 s** | **43.7 s** |

**`grad` dominates (~70–90%)** and drives everything. Fitting the two points
gives `grad ∝ N^1.74` (theory ceiling is `N²` — the gap is the A100 getting more
efficient at bigger batches, an advantage that taps out as you scale). Sampling
is sub-linear (`N^0.85`, GPU has headroom). The QGT solve is small but *grows near
convergence* — with a small `diag_shift` the S-matrix becomes ill-conditioned and
the dense solve does more work; since `n_params` is fixed by weight-sharing, it
stays roughly L-independent.

**Projection (budget with the pessimistic N² end for L ≥ 7):**

| L | N | per-step | 300-step run | 5 h-slots |
|---|---|---|---|---|
| 6 | 540 | ~48–63 s | ~4–5 h | 1–2 |
| 7 | 882 | ~100–155 s | ~9–13 h | 2–3 |
| 8 | 1344 | ~200–345 s | ~17–29 h | 4–6 |

The L=6 measurement (43.7 s) landed on the fit, validating the extrapolation.
Practical consequences: L=6 is cheap; L=7 needs auto-resubmit; **L=8 is a
~1-day-of-GPU-per-point commitment** — so do a dense `h_z` sweep at L≤6 and only a
few near-transition points at L=7/8 for the finite-size scaling. Also,
**`chunk_size` must shrink ∝ 1/N²** (≈ 2048 / 512 / 256 at L = 6 / 7 / 8) or the
larger jobs OOM.

## Act 4 — is the ansatz any good? A reference-free red line

There is no exact diagonalization past L=2, so we need reference-free checks.

The cleanest is a **variational lower bound**. For `H(h) = H₀ − hₓΣσˣ − h_zΣσᶻ`,
the `h=0` toric-code ground state has `⟨σˣ⟩ = ⟨σᶻ⟩ = 0` on every spin, so
evaluating `H(h)` on it gives exactly `E₀(0) = −(#A_v+#B_p)`. By the variational
principle the *true* ground state can only be lower:

> **E₀(h) ≤ E₀(0) for any field, strictly below for h ≠ 0.**

So a converged NQS that sits *above* `E₀(0)` has failed to even match a trivial
product-of-stabilizers state. The anchor is exact at any L:
`−(L³+3(L−1)²L)` (OBC), `−4L³` (PBC).

Results:

- **`h=0` is a trivial test** — the GS is a stabilizer state, exactly
  representable. It converged to `E = −172.000000` (delta ~1e−9, spread → 2e−4) and
  only validates the *plumbing*.
- **L=4, `hₓ=h_z=0.2`: E → −174.55 < −172 ✓** — the ansatz genuinely captures the
  *entangled* finite-field state.
- **L=5, `hₓ=h_z=0.2`: crossed −365 ✓** (anchor −365).
- **L=6:** timing confirmed; but at `diag_shift 1e-3` the energy destabilized early
  (spread exploding) — **L≥6 needs `diag_shift 1e-2`** for a stable SR step.

For L≥4, ongoing quality is judged by `Vscore = N·Var(H)/⟨H⟩² → 0` (and its
per-step proxy, the printed `spread`): a decreasing Vscore means keep training; a
nonzero floor means an ansatz-*capacity* limit rather than a training one.

## Takeaways

1. `chunk_size` is the master knob for large-L feasibility — memory, not speed —
   and must scale down with L.
2. `grad` (local energy) is the wall-clock driver, ~`N^1.74`–`N²`; L=8 is
   expensive enough to plan the sweep around.
3. The ansatz clears the `E₀(h) < E₀(0)` red line at L=4 and L=5 — the first
   evidence it captures the entangled 3D toric code, not just the trivial state.
4. Reference-free judging (the `E₀(0)` bound + `Vscore`) is essential past L=2.

## Open items

- `diag_shift 1e-2` for L≥6 stability (needs a longer confirmation run).
- Make the submit script's `chunk_size` (and possibly `diag_shift`) L-aware.
- Full `Vscore`/variance convergence study at L=5/6 to quantify ansatz quality.
