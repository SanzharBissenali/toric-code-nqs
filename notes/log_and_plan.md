# Log & Plan: Approximately-Symmetric NQS for the 3D Toric Code

> Formerly `3D_extension_plan.md`. The architecture-extension plan below is
> now **mostly executed** (and in places redirected). This top section is the
> living log; the original plan is kept underneath as the design record.

## Current research direction — mapping out the phase diagram

The architecture extension and the training/validation infrastructure are in
place. The active goal is now to **map out the 3D toric-code phase diagram**
(topological → trivial transition under a uniform field) with the NQS ansatz —
starting from faithfully reproducing **bosonic TC at L=2** before scaling up and
moving to the sign-full fermionic TC.

**Immediate goal (updated 2026-06-29):** the L=2 architecture comparison is done
— the symmetry-aware `ToricCNN_full` reaches a relative energy error **~100×
lower** than the symmetry-unaware pure-CNN (`GeoCNN`) baseline. Now **scale to
larger L** (L≥3 PBC), where exact diagonalisation is no longer available, and
show that (i) training stays **stable** and (ii) the symmetry-aware and
symmetry-unaware nets **converge to distinguishable energies** (the gap is the
order parameter once `delta` is gone).

## Work done so far

- **Architecture extended 2D → 3D.** `KernelManager3D`, `GeoConv3D`,
  `CNN_invariant_3D` / `CNN_noninvariant_3D`, and the `ToricCNN` /
  `ToricCNN_full` ansätze in `Three_TC/model/networks.py`. The Wilson 4-product
  enforces A_v invariance unchanged in 3D; the symmetric-only net reaches the
  exact h=0 ground state at L=2/3/4 PBC (see Checkpoint 1 in
  `notes/progress_log.md`).
- **Exact 3D kernel manager.** `KernelManager3D` removes the 2D ring-traversal
  approximation: for each output site it scans the exact **15-spin
  neighbourhood** — the site itself + 8 nearest neighbours + 6 next-nearest —
  via precomputed gather/mask/scatter index arrays (generalising the 2D
  hor/vert kernels to a 3×3 orientation-pair structure).
- **Training + validation pipeline.** `Three_TC/validation.py` scores ansätze
  against the Colab L=2 exact reference (`eps_E`, V-score, stabiliser/
  magnetisation deviations with MC pulls, parameter/runtime cost), for both
  bosonic and fermionic models. See `notes/pipeline.md` and Checkpoint 2 in
  `notes/progress_log.md`.

## Daily log

### 2026-06-24
- **Met with Norm — research direction approved.** Three-part program:
  (1) generalise the CNN NQS architecture to 3D, (2) reproduce the 3D **bosonic**
  TC phase diagram, (3) extend the architecture + phase diagram to the 3D
  **fermionic** TC (the sign-full problem).
- Training + validation pipeline (against exact L=2) is in place.
- **Open problem: unstable training.** `tau_corr` and `R̂` spike during training.
  Suspected to need hyperparameter tuning on three fronts: optimiser dynamics
  (`lr`, `diag_shift`), MCMC sampling, and possibly the architecture itself.
- Implemented the **exact `KernelManager3D`**: scans the 15-spin neighbourhood
  one spin at a time (1 self + 8 nearest + 6 next-nearest).

### 2026-06-26
- **MCMC acceptance is phase-dependent, not a bug.** Bit-flip + vertex-flip
  sampling works well throughout the **topological** phase in both 2D and 3D.
  The ~1% acceptance rate deep in the **trivial** phase is expected physics
  (the wavefunction sharpens), *not* a code bug — judge convergence by
  `R̂` / `tau_corr` / energy instead. (See [[mcmc-acceptance-phase-dependent]].)
- **Replicate before innovate: chasing Dom's relative error.** Struggling to
  reproduce Dom's `1e-7`–`1e-8` relative-error results. Currently sitting at
  **2.2e-4**. Next: sweep architectures and hyperparameters to get the error
  down to at least `1e-6`–`1e-5` reliably before adding anything new.
- **Code changes supporting the above:**
  - Added **`VanillaCNN`** and **`VanillaWilsonCNN`** baseline ansätze
    (`Three_TC/model/networks.py`) — plain grid CNNs that bypass
    `KernelManager3D`, for the replicate-first comparison.
  - `Three_TC/train.py` now exposes `--arch`
    (`ToricCNN` | `ToricCNN_full` | `VanillaCNN` | `VanillaWilsonCNN`) plus
    vanilla depth / kernel-extent flags; documented in `notes/training_cli.md`.
  - `simulation/optimizer.py` now logs **live MCMC acceptance rate** and `R̂`
    in the progress bar (the diagnostic that settled the acceptance question).
  - New Colab `colab/2D_TC_ED_NQS_colab.ipynb` (2D ED + NQS reference);
    removed the stale `notes/claude_handoff.md`.

### 2026-06-26 — OBC enabled for the symmetry-aware CNN (bosonic, L=2)

The pipeline was PBC-only in practice (OBC `--bc OBC` existed but the CNN hard-
rejected it, and several OBC paths were latently buggy). OBC often gives *lower*
relative error, so it's now a first-class option via the same `--bc` flag — no new
flag, **PBC behaviour byte-identical**. Bosonic only this pass; fermionic OBC
deferred (`fermionic_decoration._idx` still PBC-hardcoded).

- **Geometry** (`geometry.py`): OBC now keeps only **complete 4-edge plaquettes**
  (incomplete boundary faces dropped — they corrupt both ED Z-strings and the
  Wilson product), and exposes `plaq_centers` + a `_plaq_center_to_idx` map.
- **`KernelManager3D`** (`networks.py`): OBC support — edge stencils **mask**
  out-of-box neighbours instead of wrapping; plaquette stencils are now
  coordinate-based (via the centre map). `build_model` errors on Vanilla\*+OBC
  (CIRCULAR padding is intrinsically PBC). See `notes/nqs_architecture.md`.
- **Sampler** (`builders.py`): vertex clusters strip `-1` and pad to width 6 —
  fixes the L=2 OBC divide-by-zero (no full bulk stars) and `-1` flips.
- **`bc` threaded** through `validation.py` / `optimize.py` (default PBC).
- **New Colab** `colab/3D_TC_OBC_ED_colab.ipynb`: self-contained OBC ED over a 5×5
  `(hx,hz)∈{0,.1,.2,.3,.4}²` grid → per-point reference JSONs (validation schema)
  + E₀/gap heatmaps. L=2 OBC is N=12 (2¹²), so ED is local-trivial.
- **Validated**: geometry/commutation (L=2,3 OBC) in `test_geometry.py`; inline
  notebook ED matches repo `model/exact_diag.py` exactly (`E₀(.2,.2)=−14.279396`);
  `ToricCNN_full` under OBC trains to **`eps_E=1.1e-4`** (150 iters, dense QGT).

### 2026-06-26 — Conceptual: extending the symmetry-aware net to the *fermionic* TC

Analysis only (no code yet), ahead of program step 3 (fermionic phase diagram).
**Where the architecture transfers and where it breaks** — see `nqs_architecture.md`
("Extending to the fermionic TC") for the full argument.

- **Wilson nonlinearity + A_v machinery transfer unchanged.** Vertex stars are
  identical in the fermionic model (`fermionic_decoration.py:9`); only plaquettes
  are decorated. ∏σᶻ over a boundary is still exactly A_v-invariant, and the
  fermionic GS is still exactly A_v-symmetric *including phase*. **Keep Wilson.**
- **No "decorated Wilson" is possible — or needed.** B̃_p carries σˣ (off-diagonal),
  so it isn't a function of one bitstring. But the σˣ adds **no new diagonal
  A_v-invariant**: the bare flux features already form a complete invariant basis.
- **What actually breaks: the sign.** −J·B̃_p has mixed-sign off-diagonal matrix
  elements (∓J), so the fermionic GS is **non-stoquastic** — the real-`log ψ`
  (h_y=0) sector the 3D net is built for fails. The entire fermionic content lives
  in the wavefunction **sign/phase**, which the current real ansatz throws away.
- **Plan:** (1) **highest leverage** — test whether a finite-depth **Clifford
  disentangler** U maps fermionic→bosonic stabilisers (cheap GF(2), reuses
  `_gf2_solve`): if yes, conjugate inputs by U† and the real net works verbatim; if
  no, the sign is topologically intrinsic. (2) If needed, **complexify** `GeoConv3D`
  (port the 2D complex branch), keep Wilson+invariant CNN as the A_v-invariant
  complex backbone, consider a separate phase head. (3) Add B̃_p 2-edge-flip moves
  to `E_loc`/sampler. (4) Validate on FM ratio + gap + ⟨Mz⟩ vs ED (L=2 local; L≥3
  Colab only).

### 2026-06-29 — Scaling the bosonic NQS past L=2 (no exact reference)

- **L=2 result (user-reported, recorded here):** symmetry-aware `ToricCNN_full`
  achieves a relative energy error **~100× lower** than the symmetry-unaware pure
  CNN stack (`GeoCNN`, the no-Wilson control at matched params/depth). This is the
  payoff of the A_v-invariant Wilson change-of-coordinates. → **scale up.**
- **What we want to observe at L≥3:** (1) training is **stable** (R̂≈1, spread
  `√Var` shrinking, energy converging — *not* `mcmc_acceptance`, which is
  phase-dependent, see [[mcmc-acceptance-phase-dependent]]); (2) the symmetry-aware
  and symmetry-unaware nets **converge to different energies** (lower = better by
  the variational principle; the gap replaces `delta` as the figure of merit).
- **Reviewed the pipeline for L-scaling — the stack scales with just `--L`.**
  `Three_TC/builders.py` and `train.py` are clean: params are **L-independent**
  (weights shared across sites — verified `ToricCNN_full`=2571, `GeoCNN`=1839 at
  both L=2 and L=3), the sampler `n_sweeps` auto-scales to `2N`, dense QGT stays
  cheap. VMC at L=3/4 is fine on a laptop (Checkpoint 1 ran L=4 h=0); the 8 GB OOM
  rule is about **ED**, not VMC. Cheap-proxy smoke: both archs build + `expect(H)`
  at L=3 PBC (N=81).
- **The only L=2-specific machinery is the exact reference**, and it's now guarded
  in `nqs_sweep_colab_exp.ipynb`: ED is tractable only at L=2 (PBC N=24 / OBC N=12);
  at L≥3 (PBC 2⁸¹, OBC 2⁵⁴) there is no `E_exact`. The configure cell sets
  `HAS_GROUND_TRUTH = (L==2)` — L=2 keeps the `--hz_preset` (PBC) / on-the-fly ED
  (OBC) path; **L≥3 passes `--hz` directly with no `--exact_E0`**, so `train.py`
  prints `E ± ΔE` + spread and `delta=None` (handled gracefully, `train.py:124`).
  Also: `N_SWEEPS=0` → code-default `2N`; intro markdown documents the regime.
  `HZ_PRESETS` (`train.py:54`) stays L=2-only; do **not** use it at L>2.
- **Next:** on Colab, A/B `ToricCNN_full` vs `GeoCNN` at L=3 PBC, matched params
  (use the param-count helper cell), same (hx,hz); confirm stable training and a
  clear energy gap. Likely **raise `diag_shift`** for the larger SR solve.

### 2026-06-30 — `ToricCNN_gridinv`: grid-conv invariant block (kernel → L)

The geometry-exact invariant conv (`GeoConv3D(lattice="plaq")`) reaches the
topological long-range order only by spanning `Θ(L)` — expensive in 3D because the
plaquette stencil carries an `O=3` orientation axis and a 15-tap footprint per layer.
Added **`ToricCNN_gridinv`** (`Three_TC/model/networks.py:621`): the **2D-paper
architecture generalised directly to 3D** — keep the Wilson sandwich, but make the
*invariant* block a standard `nn.Conv3D` whose kernel scales to `L`.

- **The fold** (`plaq_grid_layout`): after the per-channel Wilson product, each
  plaquette is anchored to cube cell `floor(centre)` and placed in orientation-channel
  `o` → dense `(L,L,L,3·C)` grid + occupancy mask. The 2D "plaquettes on dual-lattice
  vertices" picture, except a 3D cell holds 3 plaquettes (one per normal) folded into
  channels — collapsing the within-cell ½-offsets (the **half-offset approximation**
  `GeoConv3D` avoids). Trades geometry-exactness for a fast conv + trivial kernel→L
  scaling; pre-Wilson (noninv) block stays geometry-exact.
- **Conv + readout**: `nn.Conv3D(kernel_size=L)` (override `--kernel_size`), CIRCULAR
  pad (PBC) / zero pad (OBC); final conv → `O` channels then a **masked mean** over
  occupied cells (OBC boundary cells excluded) → real `log ψ`. Supports both BC.
- **Wired**: `builders.py` dispatch + `plaq_grid_layout`; `train.py` `--arch
  ToricCNN_gridinv` and `--kernel_size` documented for it (`--noninv_channels /
  --n_noninv / --inv_hidden` reused). See `notes/nqs_architecture.md`
  ("Grid-conv invariant alternative") and `notes/training_cli.md`.
- **Verified**: builds + forward + sampler at L=2 OBC (N=12, params 3543) and PBC
  (N=24); **Colab confirmed it trains to the same ~1e-3→1e-5 `delta`** as
  `ToricCNN_full` at L=2 OBC. `nqs_sweep_colab_exp.ipynb` now **defaults to
  `ToricCNN_gridinv`**, routes runs to the arch-named folder
  `outputs/gridinv/colab-<bc>-<point>/`, and exposes `NONINV`/`INV`/`KERNEL` knobs.
- **Why it matters for scaling:** this is the cheap path to `kernel ∝ L` coverage in
  3D. A/B `ToricCNN_gridinv` (grid invariant) vs `ToricCNN_full` (geometry-exact
  invariant) isolates how much the half-offset exactness is worth vs. the speed/scaling
  of a plain `nn.Conv3D`.

---

### 2026-06-30 — Cluster (NERSC) pipeline + first L=6 gridinv kernel sweep

Made `Three_TC/train.py` **timeout-safe** for NERSC and launched the first L=6
runs. Goal of these runs: not full convergence, but a coarse probe of whether
`ToricCNN_gridinv` can even **land near the ground state at L=6** — on Colab it was
getting "nowhere closer."

- **Checkpoint/resume + offline wandb** (`train.py`, `builders.run_loop`,
  `utils/io.load_weights`): `--checkpoint_every N` atomically writes
  `{name}.ckpt.mpack` (weights + sampler RNG) + `{name}.curve.json` (step count +
  full curve) every N steps; `--resume` reloads them and continues the cosine-LR
  schedule (new `start_step`/`total_iter` args); `--wandb_offline` +
  deterministic run-id so requeued chunks merge into one run on `wandb sync`.
  Verified resume locally at L=2 OBC (warm start continued the energy, curve kept
  all steps). See `notes/training_cli.md` + `nersc/README.md` Phase 5.
- **Submit wrapper** `nersc/submit_nqs_gridinv.sh`: single long gridinv run, every
  knob an env var (`L/BC/HX/HZ/DT/DIAG_SHIFT/NONINV/N_NONINV/INV/KERNEL/N_ITER/...`),
  `--resume` always on, `AUTO_RESUBMIT=1` self-requeues ~3 min before the wall limit
  (Slurm `--signal` trap) for multi-slot runs. `qos=shared`, 1 A100. **Name now
  encodes `INV`** (`inv4-4-4`) after two k=5 runs differing only in `--inv_hidden`
  collided on the same checkpoint.
- **`n_params` (cheap local build, the dense-QGT gate):** L=6 OBC (N=540),
  noninv 2×4, inv `[4,4,4]` — kernel 5 → 59k, kernel 4 → 31k, kernel 3 → 14k
  params. All dense-QGT-safe on one A100 (Jacobian ≤ 7.8 GB). **Use `dense`, not
  onthefly** — direct solve beats CG here and CG is the path that flakes on GPU.
  (Full-span kernel 6 with inv `[4,4,4,4]` blows up to 133k → would OOM dense;
  cap the kernel or stack depth instead.)
- **First batch (running, L=6 OBC, hx=hz=0.2, dt=0.02, diag_shift=1e-3, 100 iters,
  AUTO_RESUBMIT):** kernels {5,4,3} at inv `[4,4,4]`, plus inv `[4,4,4,2]` at
  kernel 5. Submitted at the **perturbed** point (h=0 is a trivial stabilizer
  target); judged **relative** to each other + the ~−666 h=0 anchor as a ballpark
  (true `E0(0.2)` sits a bit below −666, no exact value at L=6) + `⟨A_v⟩→1` / low
  spread / `R_hat≈1` for "stayed in the topological state."
- **Next:** rank kernels by landed energy; if one lands close, extend its `n_iter`
  (just re-`sbatch`, `--resume` continues) and then move the winner to a `(hx,hz)`
  line for the transition. If none land close, the gridinv half-offset
  approximation may be too lossy at L=6 → compare against `ToricCNN_full`.

---














## Original plan (design record)

### Context

The user has a working understanding of the 2D toric-code paper
([arXiv:2405.17541](https://arxiv.org/abs/2405.17541)) and the repository
that implements it. Their actual research direction is **extending the
approximately-symmetric NQS architecture from the 2D to the 3D toric code**
(and adjacent 3D Z₂-symmetric topological models). This plan captures (a) a
literature scout of prior NQS work on 3D toric / gauge models, and (b) a
concrete implementation plan for the 3D extension, anchored to specific
files and patterns in the existing 2D codebase that should be re-used or
generalised.

### Literature scout: what already exists

#### Direct prior art (must be aware of)

- **Luo, Liu, Halverson, Hsu et al. (2021, arXiv:2101.07243; PRR 5, 013216,
  2023): "Gauge Invariant and Anyonic Symmetric Transformer and RNN Quantum
  States for Quantum Lattice Models."**
  Provides *exact* representation of ground and excited states of the **2D
  and 3D toric codes and the X-cube fracton model** using
  **autoregressive** (Transformer/RNN) NQS with gauge constraints baked
  into the conditional probability factorisation. Allows exact sampling
  (no MCMC). This is the closest prior 3D-TC NQS work; the proposed
  extension of Kufel et al. is methodologically distinct
  (approximately-symmetric CNN with TDVP + MCMC, not autoregressive),
  so the contribution is the *architecture class* and the *perturbed
  regimes*, not the unperturbed model.

- **Luo, Chen, Hu, Hsu, Karki, Halverson, Bhowmik (2020/2021,
  arXiv:2012.05232; PRL 127, 276402): "Gauge Equivariant Neural Networks
  for Quantum Lattice Gauge Theories."**
  Z₂ gauge theory on 2D square lattices up to 12×12, exact loop-gas
  solution as a special case, demonstrated perimeter-to-area Wilson
  loop transition. Methodologically very close to Kufel et al. but 2D.

#### Competing methods in 3D

- **Tensor networks for (3+1)d toric code** (e.g., Schwarz et al.,
  arXiv:2012.15631, Quantum 2021; PRB 104, 235151 on TN stability).
  PEPS captures the fixed-point exactly; bond dimension explodes away
  from it. **Niche for NQS: moderate perturbation strength.**
- **QMC**: sign-problem-free for parallel-field (hx, hz) perturbations;
  this is the gold-standard baseline for diagonal perturbations.
  **Niche for NQS: hy (Y-field) or other non-stoquastic perturbations.**

#### Strategic implication

The novelty bar is *not* "represent the unperturbed 3D toric code" —
that's been done. Headline-worthy contributions live in:
1. **Sign-problem regimes** (h_y or arbitrary non-stoquastic
   perturbations) — QMC fails, autoregressive NQS works but is
   parameter-heavy, tensor networks struggle in 3D.
2. **Mixed-field phase diagrams** of the 3D TC — less mapped than 2D.
3. **Generalisation to X-cube / fracton models** — symmetry trick may
   port; even fewer methods work there.

### Physics scope of the extension

3D toric code, **PBC, qubits on edges of cubic lattice**:
- N = 3·L³ qubits
- Vertex stabilisers A_v = ∏σ_x over the **6 edges** at vertex v
- Plaquette stabilisers B_p = ∏σ_z over the **4 edges** of a face, with
  **3 orientations** (xy, yz, xz)
- L³ vertices, 3L³ plaquettes
- Ground-state degeneracy on a 3-torus: 2³ = 8 (three independent
  non-contractible 1-cycles)

**Key symmetry observation** (the reason this is tractable): an A_v flip
hits 6 edges at v; any plaquette intersects those 6 edges in either 0 or 2
edges → ∏σ_z over 4 plaquette edges is **still A_v-invariant in 3D**. The
Wilson nonlinearity from the 2D paper generalises *unchanged*.

### Implementation plan (anchored to existing code)

#### Module-by-module change table

| Module | Status | Change |
|---|---|---|
| `model/geometry.py` | rewrite (new file) | New `ToricCodeGeometry3D` class (PBC only initially). 3 edge orientations, 3 plaquette orientations, 6-neighbour vertex enumeration. Re-use the `_mapping2Dto1D` pattern with a 3D analogue. |
| `model/hamiltonian.py` | reuse as-is | The loops in `create_hamiltonian` iterate `len(vertex_all[v])`. Feed 6-tuples and 4-tuples; works unchanged. Pauli-string conversion will be slower but correct. |
| `model/networks.py` → `KernelManager` | rewrite | All shift logic (`_generate_pos_shifts`, `shifts_plaq`, `_generate_kernel_shifts`) is 2D-specific (ring traversal, hard-coded shift lists). Needs full 3D rewrite. Generate three edge-orientation kernel sets and three plaquette-orientation kernel sets. |
| `model/networks.py` → `CNN_noninvariant` | adapt | The split `Wconv_hor` / `Wconv_vert` becomes `Wconv_x`, `Wconv_y`, `Wconv_z`. Strong recommendation: **tie weights across orientations** (enforces cubic point-group symmetry, fewer parameters). Identity-initialisation pattern from `identity_initializer_CNN_links` carries over. |
| `model/networks.py` → `WilsonNonlinearity` / `_Wilson_4spin_plaq` | reuse, multiply | Same 4-spin product, applied **three times** (once per plaquette orientation). Either concatenate the three outputs along channel axis or sum-pool them. |
| `model/networks.py` → `CNN_invariant` | adapt | Currently operates on plaquette positions in 2D. In 3D, plaquettes live on a richer 2-complex; cleanest is a 3D conv on cubic lattice with weight-sharing across orientations. |
| `simulation/custom_sampler.py` | reuse as-is | `MultiRule` is generic over cluster size. 6-spin vertex clusters from the new 3D geometry plug in directly. |
| `simulation/optimizer.py` | reuse as-is | TDVP loop is dimension-agnostic. May need to *raise* `diag_shift` for larger systems. |
| `simulation/observables.py` | partial rewrite | 1D Wilson loops still work (electric strings). The *closed-surface* analogue (magnetic-loop BFFM in 3D) requires new enumeration code. Magnetisation/Rényi work unchanged. |
| `utils/config.py` | minor | Add `Lz` parameter; update `N` calculation for 3D PBC: `N = 3*Lx*Ly*Lz`. |
| `main.py` | minor | Switch on `--dimension 2/3` to select geometry class; everything downstream stays. |

#### Sequenced development order

1. **`geometry3d.py`** (PBC only, no OBC). Validate at L=2 with hand-checks:
   24 qubits, 8 vertices (each touching 6 distinct qubits), 24 plaquettes
   (each 4 qubits). **No code from `geometry.py` is salvageable beyond the
   `_mapping*Dto1D` pattern** — but read it line-by-line first to absorb
   the conventions.

2. **Plug into existing Hamiltonian, run exact diag at L=2**. With h=0:
   E₀ = −(N_v + N_p) = −(8 + 24) = −32. GS degeneracy 8. Lanczos on 2²⁴
   states is sparse-doable. This is the gateway check.

3. **Minimal network: symmetric block only**, no non-symmetric CNN. Verify
   it hits E₀ = −32 at h=0 with identity-ish initialisation.

4. **Custom sampler at L=2, L=3.** Single-flip will fail worse than in 2D;
   vertex-update `MultiRule` is the primary tool. Measure acceptance to
   verify.

5. **Add non-symmetric block**, weight-tied across orientations. Test on
   small h_z, h_x perturbations at L=2. Compare against (i) exact diag at
   L=2, (ii) QMC reference at L=3 (cite literature).

6. **Push to L=3, L=4 PBC**. Lose exact diag. Sanity checks:
   - ⟨A_v⟩, ⟨B_p⟩ → 1 in topological phase
   - Var(H) per site → 0 if eigenstate
   - Cross-seed consistency
   - QMC literature values for h_z

7. **Push into novel regime**: non-stoquastic perturbation (h_y), or
   approach to e-confinement transition. **This is where the paper-shaped
   contribution lives.**

#### Critical re-use patterns from the 2D code

- The **identity initialisation idiom** in `identity_initializer_CNN_links`
  is the secret sauce that makes training start near the symmetric fixed
  point. Port it exactly to the 3D CNN.
- The **`jax.lax.scan` + masked-conv pattern** in
  `CNN_noninvariant.__call__` handles boundary masking elegantly — keep
  this shape even though PBC won't need the `-1` sentinels.
- The **`WeightedRule` / `MultiRule` composition** in `custom_sampler.py`
  is plug-and-play.
- The **JSON+mpack output format** in `utils/config.py` / `utils/io.py`
  is fine to reuse; just bump the `kind` annotation to `"G-NonInv-3D"`.

### Verification

End-to-end checks the implementation must pass before committing to the
real experiments:

1. **Geometry self-consistency at L=2 PBC**: print all stabilisers,
   verify each qubit appears in exactly 4 plaquettes (one per
   orientation × 2 sides) and exactly 2 vertices (its two endpoints).
2. **Hamiltonian exact-diag at L=2 PBC, h=0**: E₀ = −32, degeneracy 8.
3. **Network symmetry test**: pick a random σ, evaluate log ψ(σ); apply
   a bulk A_v (flip 6 spins); evaluate log ψ(A_v σ). Must agree to
   machine precision. This is **the** confirmation that the symmetry
   trick generalised correctly.
4. **TDVP convergence at L=2 PBC, h=0**: NN energy → −32 within MC
   noise; Var(H) → ~0.
5. **TDVP at L=2 PBC, h_z=0.1**: NN E within ~10⁻³ of exact diag.
6. **L=3 PBC with custom sampler**: acceptance ratio stays > 0.05;
   ⟨A_v⟩, ⟨B_p⟩ > 0.95 deep in topological phase.
7. **L=3 PBC, h_z scan**: cross-check ⟨σ_z⟩, energy density against
   published QMC values (need to find the right reference — likely
   Vidal et al. for 3D TC + parallel field).

### Estimated effort

- Steps 1–3 (geometry + minimal symmetric net): **1–2 weeks**
- Steps 4–5 (sampler + non-symmetric block + small perturbations):
  **2–3 weeks**
- Steps 6–7 (scaling + novel physics): **3–6+ weeks**

Total: ~2–3 months for a working prototype, ~6 months to a publishable
result. The architectural concept ports cleanly; the work is in the
3D plumbing and in losing exact-diag as a safety net.

### Immediate next action (before any 3D code)

Finish reading the 2D codebase — specifically `model/geometry.py` and
`model/networks.py` — until each function's purpose can be predicted in
one sentence before reading the body. The 3D rewrite cannot start
before the 2D conventions are second nature. Suggested verification
exercise: pick a Lx=3 OBC run, hand-trace the network output for one
fixed σ on paper, confirm vertex-flip invariance numerically.

---

## hz phase-diagram sweep — config decisions (2026-07-04)

Mapping the topological→trivial boundary by fixing hx=0.2 and sweeping hz
(16 pts, 0.0→0.9), one `ToricCNN_gridinv` NQS per (L, hz), transition = peak of
`dO_FM/dhz`. Pipeline: `nersc/submit_nqs_hz_sweep.sh` (GPU array, index→hz) →
`Three_TC/fm.py` CLI extractor (cluster) → `analysis/plot_phase_diagram.py`
(local multi-L overlay + FSS). See `nersc/README.md` §4.7.

### Fixed run config
- Arch: `noninv_channels 4`, `n_noninv 2`, `inv_hidden "2 2 2"` (n_params ≈ 11,313,
  L-independent via weight sharing), bc OBC.
- Training: SGD+SR, `qgt dense`, `dt 0.01 → lr_min 0.001` (cosine over n_iter),
  **`diag_shift 1e-3` for ALL L**, **`n_iter 300`**.
- MCMC: `n_samples 8192`, `n_chains 1024`, `n_sweeps 48` (fixed), `n_discard 8`,
  `chunk_size 2048`.

### kernel_size per L (LRE vs cost)
Physics: to capture long-range entanglement the conv receptive field should span
the lattice → `kernel ≈ L-1` is the natural choice. Cost: a 3D conv is ∝ k³, so
`kernel = L-1` adds a ~(L-1)³ ≈ N factor on top of grid-volume scaling — it makes
small L cheaper but large L much more expensive.

Timing rule of thumb (measured baseline: L=6, kernel 4 = **44 s/step**; L=4,
kernel 4 = ~7 s): **kernel k→k+1 multiplies conv work by ((k+1)/k)³** (4→5 ≈
1.95×), diluting to **~1.5–1.9× on the whole step** (conv is ~50–90% of it). So
L=6 at kernel 5 would be ~70–85 s/step; a secondary bump comes from the larger
param count inflating the dense-QGT solve (∝ n_params³, but that phase is small).

Chosen map (compromise): **L=3→2, L=4→3, L=5→4, L=6→4**. L≤5 use L-1; L=6 is
held at 4 (not 5) to avoid the ~2× step-time blow-up, betting that network *depth*
(2 non-inv + 3 inv layers, stacked receptive field ≈ Σ(kᵢ−1)+1) captures the LRE
instead. Baked into `submit_nqs_hz_sweep.sh` as a per-L `case` (override via KERNEL=).

### Caveats to watch (from last session)
- L=6 destabilized at `diag_shift 1e-3` (E rose, spread blew up in ~2 steps); we
  keep 1e-3 by choice — if L=6 diverges, re-submit that L with `DIAG_SHIFT=1e-2`.
- `n_samples (8192) < n_params (11313)` → dense QGT is under-determined, leans on
  diag_shift; if the near-transition derivative peak looks ragged, raise N_SAMPLES.
- `inv_hidden "2 2 2"` is small capacity — watch `Vscore` near the peak; a nonzero
  floor = capacity limit (widen inv_hidden), not a training limit.

### Vscore vs hz — cross-L quality record (2026-07-04)

Reference-free quality of the hz-sweep runs (`Vscore = N·Var(H)/⟨H⟩² → 0` for an
eigenstate). Tracking this vs L is the capacity test: n_params ≈ 11,313 is
**L-independent** (weight-shared), so if Vscore climbs sharply with L the fixed
net is running out of capacity near the transition. hx=0.2, gridinv, diag_shift
1e-3, n_iter 300, n_samples 8192, n_sweeps 48.

**L=3 (kernel 2, anchor E0(0) = −63):** all sub-1e-2; Vscore **peaks at the
transition** (hz≈0.42, 7.8e-3) — coincides with the O_FM derivative peak
h_c≈0.40. Every E0 below −63. ✓

| hz | Vscore | E0 |
|----|--------|-----|
| 0.000 | 1.71e-04 | −63.4546 |
| 0.060 | 5.28e-04 | −63.5027 |
| 0.120 | 4.02e-04 | −63.6478 |
| 0.180 | 5.29e-04 | −63.9055 |
| 0.240 | 1.23e-03 | −64.2890 |
| 0.300 | 2.01e-03 | −64.8654 |
| 0.360 | 4.30e-03 | −65.8505 |
| 0.420 | 7.76e-03 | −67.5146 |  ← peak (≈ h_c)
| 0.480 | 7.59e-03 | −69.6718 |
| 0.540 | 6.95e-03 | −72.0566 |
| 0.600 | 5.94e-03 | −74.6499 |
| 0.660 | 4.49e-03 | −77.3417 |
| 0.720 | 4.73e-03 | −80.1259 |
| 0.780 | 2.91e-03 | −82.9802 |
| 0.840 | 4.02e-03 | −85.8868 |
| 0.900 | 2.68e-03 | −88.8436 |
L=3: min 1.7e-4, max **7.8e-3** @ hz=0.42, median ~4.3e-3.

**L=4 (kernel 3, anchor E0(0) = −172):** mostly sub-1e-2; broad elevated plateau
(~5–8e-3) across hz≈0.3–0.6 rather than L=3's single sharp peak. Every E0 below
−172 (hz=0 → −173.03). ✓

| hz | Vscore | E0 |
|----|--------|-----|
| 0.000 | 6.21e-04 | −173.0281 |
| 0.060 | 1.03e-03 | −173.1568 |
| 0.120 | 6.87e-04 | −173.5604 |
| 0.180 | 8.07e-04 | −174.2527 |
| 0.240 | 2.02e-03 | −175.3203 |
| 0.300 | 4.57e-03 | −177.3163 |
| 0.360 | 7.07e-03 | −181.5804 |
| 0.420 | 6.07e-03 | −187.1905 |
| 0.480 | 6.22e-03 | −193.4516 |
| 0.540 | 5.85e-03 | −200.3938 |
| 0.600 | 8.29e-03 | −207.4921 |
| 0.660 | 4.32e-03 | −215.0760 |
| 0.720 | **1.36e-02** | −222.5978 |  ← outlier (deep in trivial phase; likely MC-variance / mild under-convergence, not the transition)
| 0.780 | 3.87e-03 | −230.5655 |
| 0.840 | 5.22e-03 | −238.4736 |
| 0.900 | 2.71e-03 | −246.5601 |
L=4: min 6.2e-4, max 1.36e-2 @ hz=0.72 (outlier); near-transition ~6–8e-3.

**L=5 (kernel 4, anchor −365):** _incomplete_ — array timed out at step 270/300
(1:30 walltime too tight). O_FM extractable from the step-270 checkpoints, but the
final `observables.Vscore` isn't written until a run completes → resume to 300
(`L=5 sbatch --time=03:00:00 --array=0-15 …`) to record. (Rough proxy meanwhile:
`Vscore ≈ N·energy_spread²/energy²` from the step-270 `curve.json`.)
**L=6 (kernel 4, anchor −666):** _pending_

**Trend so far (peak Vscore):** L=3 ≈ 7.8e-3, L=4 ≈ 6–8e-3 across the transition
band (ignoring the 0.72 outlier) — roughly flat, so the fixed "2 2 2" capacity
(n_params ≈ 11,313) is holding through L=4. Watch whether L=5/6 break sharply
upward. Flag: re-check L=4 hz=0.72 if that point matters (its 1.36e-2 is 2–3× its
neighbours and sits deep in the trivial phase where Vscore should be *low*).

Watch: does the peak Vscore stay ≲1e-2 as L grows, or climb (→ widen inv_hidden)?
Extract per L on a login node: read `observables.Vscore` from each
`$PSCRATCH/tc_nqs/phase_hx0.2/L<L>/*_hz*.json` (skip *.curve.json).

---

## 2026-07-20 — XZ-plane magnetic (h_x) 1st-order line: L=4 map + L≥5 launch

Mapping the topological→trivial *magnetic* transition (h_x-driven, fixed h_z) across
the XZ plane, from the fixed-h_z run trees `phase_hz{0.0…1.1}/L*` (hx-SWEEPS — the
e↔m mirror of the `phase_hx*` electric cuts).

- **L=4 line located, all clean.** `check_convergence.py` over the 10 h_z cuts:
  **138/138 runs `ok`** (finite E<bound, ⟨A_v⟩≤1, Vscore<0.05, none diverged). Login-node
  dumps (E, ⟨M_x⟩, ⟨B_p⟩, Vscore) pulled to `results/xz_line_L4/`; analysed in the new
  `analysis/xz_line_L4.ipynb` (curves + susceptibility + boundary + window finalizer).
  Mag/Vscore consensus h_x^c: ≈0.67 flat for h_z≤0.4, rising to ≈1.54 at h_z=1.1.
  **Caveat:** ⟨M_x⟩ is a smooth local quantity → it UNDERestimates the true transition;
  the FM order parameter sits ~+0.2 higher (phase_hz0.0 L=5 O_FM h_c=0.886 vs mag 0.67).
  So windows/boundary must be finalized off **O_FM**, not magnetization.
- **Inline O_FM + S₂ at end of training** (`validation.topological_observables`, wired in
  `train.py`, `--fm_sector`/`--no_topological`): single-state mirror of the fm/renyi sweep
  extractors, sector auto (h_x→magnetic membrane / h_z→electric loop), R clamped to the
  bulk (L=4→R=1). Best-effort/guarded. **Committed on `feat/xz-line-fm-s2`; NOT yet
  deployed to the cluster** (would touch the live h_y campaign) — L≥5 runs below use the
  current cluster code + the sweep extractor.
- **L=4 FM+S₂ extraction launched** (job 56203028, `nersc/{extract_fm_s2_hxline,submit_extract_hxline}.sh`,
  magnetic R=1 + central-plaquette S₂, all 10 h_z, 4 h wall, SKIP_EXISTING).
- **L=5/6/7 campaign launched** (`submit_nqs_hx_sweep.sh`, 22 arrays × 7 hx = **154 runs**):
  L=5,6 over h_z {0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.0,1.1}; **L=7 only 4 representative cuts**
  {0.3,0.5,0.9,1.1} (h_z=0.0 already done all L in `phase_hz0.0`; 1st-order FSS needs only
  jump-sharpening, not exponents → L=7 ≈143 GPU-h not 357). Windows = proven L=4 brackets,
  7 pts each. WALLTIME 1:30 (L5) / 2:00 (L6,L7) with AUTO_RESUBMIT (L=7 → ~3× 2 h chunks;
  short chunks backfill faster on shared QOS).
- **Measured per-run timing** (median runtime_s over the real runs, replacing CAMPAIGN.md
  L4/L5 estimates): L4 ~10 min (4.1 s/step), L5 ~45 min (18), L6 ~1.9 h (45), L7 ~5.1 h (105).
- **Next:** when 56203028 lands, pull FM/S₂ → re-plot the L=4 boundary off O_FM → finalize
  L≥5 windows (widen/shift up if O_FM sits near an edge) → per-L FM/S₂ extraction for L≥5.

---

## 2026-07-09 — full (hx, hz) campaign: first QA snapshot

Campaign launched 2026-07-08 (`nersc/run_phase_campaign.sh`, 24 arrays × 13 hz =
**312 runs**; grid hx {0,0.2,0.4,0.6,0.8,1.0} × L {4,5,6,7} × hz [0.1,0.4] step
0.025). First `check_convergence.py --tree --progress --summary` snapshot:
**88 finished, 4 in-flight, 220 not-started.** L=4 complete for all six hx (78
runs); L=5/6/7 filling in from hx=0.0 upward.

**One genuine failure — rerun later:** `L=5 hx=0.4 hz=0.1` **DIVERGED** —
E=−373.94 looks fine (below the −365 bound) but Vscore = **1.45e+06**; the variance
blew up and the in-run guard tripped (`diverged:true`). Skipped by `fm.py`. Re-run
as a one-off (its own array index) once the queue drains.

**Not failures (classifier fixed):** the two cells that first showed `!`
ABOVE-BOUND — `L=6 hx=0.4 hz=0.1` (E=−583 vs −666) and `L=7 hx=0.2 hz=0.1`
(E=−1073 vs −1099) — were **in-flight checkpoints still descending**, not converged
failures (tell: `<A_v>=None`, i.e. read from a `.curve.json`). `classify()` now
labels in-flight-above-bound `descending` (informational), reserving the `!` flag
for DIVERGED / BAD-ESTIMATOR / **finished**-above-bound.

**L=4 physics (complete cut, all hx):** clean transition band in every hx. <A_v>
falls 0.99→~0.6 and <M_z> rises ~0.05→~0.6 across the window; peak Vscore grows
mildly with hx (hx=0: 1.38e-2 @ hz=0.375; hx=0.4: 1.57e-2; hx=0.6: 2.8e-2).

**Boundary-window correction (data-backed):** the earlier guess "hz_c *decreases*
with hx (peak may drop below 0.1)" is **wrong at the top end**. From L=4, the
midpoint (where <A_v>≈0.8, <M_z>≈0.4) sits ~hz≈0.33 for hx≤0.4, then rises:
hx=0.6 ≈0.38, and hx=0.8/1.0 stay ordered across the *whole* window (hx=1.0:
<A_v> 0.997→0.948, <M_z> 0.034→0.156, Vscore still climbing at hz=0.4). So **hz_c
> 0.4 for hx≥0.8** — the high-hx cuts need HZ_MAX *extended upward* (~0.5–0.6),
not down. Revisit after L=5/6 confirm the same trend; then relaunch hx=0.8,1.0
with a higher HZ_MAX.

---

## 2026-07-29 — exact/analytic energy benchmarks (DMRG & QMC scoped out)

Goal was to validate the architecture against DMRG and QMC. A four-agent literature
review settled the scope, and the answer turned out to be **neither**: the exact
dualities plus boundary-aware perturbation series give a stronger, free benchmark.

### Literature findings (all in OUR normalization, J=1)

- **There is no published QMC / tensor-network / high-order-series energy for the 3D TC
  in a field, by anyone.** The sole 3D reference is Reiss & Schmidt
  ([arXiv:1902.03908](https://arxiv.org/abs/1902.03908)) — exact dualities + 2nd-order
  pCUT + a variational bound, no lattice numerics. Our L=4–7 curves have no competitor
  *and* no reference; hence this work.
- **CONVENTION TRAP:** the Vidal/Dusuel/Schmidt/Reiss school uses J=1/2, so **every field
  of theirs is half of ours**. This produced a genuine factor-2 disagreement between two
  agents; settled in favour of the values below by our own h_x data.
- Exactly known: **h_x^c = 1 exactly, FIRST order** (self-duality to 4D Wegner Z₂ gauge
  theory) and **h_z^c = 0.193869, second order, (3+1)D-Ising\*** (duality to the 3D TFIM,
  (Γ/J)_c = 5.158136). Our first-order h_x line is therefore *correct and exactly pinned*.
- Sign structure verified numerically (L=2 OBC dense): stoquastic for h_x, h_z ≥ 0 → QMC
  is applicable; h_y gives complex off-diagonals → no QMC route. h_x<0 looks signful but
  is curable (A_v has an even number of σˣ, so σˣ→−σˣ flips only the field).
- Applicable-but-deferred QMC codes: **ParaToric** ([arXiv:2510.14781](https://arxiv.org/abs/2510.14781),
  our exact H on the cubic lattice, MIT, no public repo URL found) and **PMRQMC**
  ([github.com/LevBarash/PMRQMC](https://github.com/LevBarash/PMRQMC), takes an arbitrary
  Pauli-string list — which our `PauliStrings` object already is).
- **DMRG is dead here**, independently of taste: an exact stabilizer-rank calculation on
  our own geometry gives the *minimal* MPS bond dimension χ = 2^(L²−1) for our OBC — 8 at
  L=2, 256 at L=3, **33 050 at L=4**, 4.3e9 for L=4 PBC. The TC entanglement spectrum is
  exactly flat, so there is no cheap-truncation regime (at L=2 PBC dropping *one* of 256
  Schmidt values already costs 2.4e-4 relative). ED beats it at L=2; L≥4 is hopeless.

### Built: `analysis/exact_benchmarks.py` + `analysis/energy_benchmarks.ipynb`

Boundary-aware series from our own `vertex_all`/`plaq_all` (39 self-checks, run
`.venv/bin/python analysis/exact_benchmarks.py`). All coefficients reduce to *counting how
many stabilizers the local field operator violates*:

- **low field** (exact through O(h²)):
  `E = -(#A_v+#B_p) - Σ_i [hz²/(2 n_v) + hx²/(2 n_p) + hy²/(2(n_v+n_p))]`.
  n_v = 2 always; n_p = 4 in the bulk but **2–3 on the open surface** — so at L=4 OBC the
  h_x² coefficient is **25.5**, not the naive bulk 18.0. Bulk PBC per spin:
  `e0 = -4/3 - hz²/4 - hx²/8 - hy²/12`.
  Coefficients (L: E0, hz², hx², hy²): 4: −172, 36, 25.5, 14.7 · 5: −365, 75, 49.5, 29.4 ·
  6: −666, 135, 85, 51.5 · 7: −1099, 220.5, 134.25, 82.5.
- **high field** (exact through O(1/h)), derived here:
  `E = -hx N - #A_v - #B_p/(8 hx)` about |+⟩^N and
  `E = -hz N - #B_p - Σ_v 1/(2 hz n_star(v))` about |↑⟩^N.
  Both denominators are pure Zeeman cost because a plaquette meets each of its vertices in
  exactly *two* edges, so the other stabilizer family is left untouched.
- **energy crossing** → analytic 1st-order h_x^c: 0.7051 / 0.7665 / 0.8066 / 0.8349 for
  L=4/5/6/7, extrapolating in 1/L to **1.0084** vs exact 1.0 (max residual 5e-4).
  Select the crossing by *direction*: both truncated branches contribute a spurious
  **downward** root (the −#B_p/(8h) divergence at h→0, and the parabola eventually diving
  under the high-field line); the physical level crossing is the unique **upward** zero.

### Results

- **4th order DERIVED — the certificate is now non-circular** (2026-07-29, later same day).
  Exact pure-h_z coefficient, zero fitted parameters, zero NQS input:
  **`c_z4 = (5/16)·#B_p + N_adj/32 − N/64`**, `N_adj = Σ_v C(n_star(v),2)`.
  Derivation: in the σ^z sector every state is labelled by its syndrome S with
  `E = E₀ + 2|S|`, and every σ^z_i matrix element is exactly **+1** (two edge sets with the
  same boundary differ by a contractible loop = a product of B_p, eigenvalue +1). So 4th-order
  Rayleigh–Schrödinger is pure combinatorics over ordered 4-tuples of edges returning to the
  vacuum. Two families: (a) four distinct edges must form a closed 4-cycle, and in a cubic
  lattice **every 4-cycle is a plaquette** → 5/2 per plaquette over its 24 orderings; (b)
  coincident pairs {a,a,b,b} → 1/d(a,b) per pair. Family (b) is O(N²) and **cancels exactly**
  against the −E⁽²⁾Σ|V₀ₖ|²/Δₖ² renormalization term, leaving the extensive result.
  OBC only — the argument needs simple connectivity (syndrome ⇒ state) and a non-degenerate
  ground state; PBC's eightfold degeneracy would need degenerate PT.
  Values (L: c_z4): 4: 48.0 · 5: 107.34 · 6: 201.94 · 7: 339.94.
- **Verified three independent ways.** (i) matches a brute-force sum over all 4-tuples at L=2
  OBC (2.4375); (ii) exact ED at L=2 OBC gives `E = −14 − 3h_z² − 2.4375h_z⁴ + O(h_z⁶)` with
  the residual/h_z⁶ converging to a constant (−5.416, so c₆(L=2) ≈ 5.416 as a bonus);
  (iii) **`c_z4/N` extensivity now PASSES** — 0.3333, 0.3578, 0.3740, 0.3854 for L=4–7, fitting
  `a + b/L` to 1e-4 and extrapolating to a = 0.45496 against the *independently computed* PBC
  bulk value **29/64 = 0.453125** (0.4%). Contrast the earlier *fitted* c₄/N: −0.196, −0.305,
  +1.74, +9.15 — the circularity diagnostic, now retired.
- **E_NQS is a SINGLE `vs.expect(Ham)` on the final state** (`validation.py:169`), *not* an
  average over optimization steps — so it carries an MC error of the mean (`E_err`, stored as
  `E_spread` in the aggregates; NetKet's autocorrelation-corrected value). **dev4 means nothing
  until divided by it.** At the campaign's `n_samples=8192`:
  | L | dev4 | E_err | sigma | statement |
  |---|---|---|---|---|
  | 4 | +0.0044 | 0.0036 | **1.2** | unresolved → **ε < 1.7e-5** (2σ bound) |
  | 5 | +0.0004 | 0.0010 | **0.5** | unresolved → **ε < 6.1e-6** (2σ bound) |
  | 6 | +0.199 | 0.048 | **5.4** | **MEASURED ε = 5.2e-4 — UNDER-CONVERGED** |
  | 7 | +1.726 | 0.147 | **13.0** | **MEASURED ε = 1.8e-3 — UNDER-CONVERGED** |
  So the L=6/L=7 under-convergence is strongly significant, but **there is no measured accuracy
  at L=4/L=5** — their deviations are consistent with zero. The useful statement there is the
  2σ upper bound (the ansatz is *at least* that good), computed conservatively as
  `(dev4 + 2·E_err + c₆h_z⁶)/|E|` and only where the estimated truncation sits below the MC
  noise (`c6_scale < E_err`) — otherwise the crude `0.45·N` c₆ guess dominates and the bound
  can even go negative.
- **Cheap fix to MEASURE ε at L=4/5:** re-evaluate the saved `.mpack` checkpoints with more
  samples — `E_err ∝ 1/√n`, so L=4 needs ~6× (49k samples) and L=5 ~43× (348k) to clear 3σ.
  One `vs.expect` call per point, **not retraining**: minutes of GPU, plausibly local at L=4.
- **LEARNING CURVES settle the diagnosis — "under-converged" was the WRONG label** (user pushed
  back; they were right). Plotting the per-step residual `|E−E_pert4|/|E_pert4|` against each
  run's own MC error floor (`analysis/energy_benchmarks.ipynb` §3, hz=0.1, last-30-step slope of
  log₁₀ residual):
  | L | final residual | MC floor | ratio | slope/step | reading |
  |---|---|---|---|---|---|
  | 4 | 1.6e-6 | 4.0e-6 | 0.4× | −1.7e-3 | **at the noise floor** (oscillating) |
  | 5 | 5.5e-7 | 1.4e-6 | 0.4× | −2.1e-2 | **at the noise floor** |
  | 6 | 3.0e-4 | 3.0e-5 | **9.8×** | −1.2e-3 | **FLAT plateau above the floor** |
  | 7 | 1.6e-3 | 5.0e-5 | **31.8×** | −5.0e-4 | **FLAT plateau above the floor** |
  L=6/L=7 have **stopped descending** — dead flat for the last ~50–75 of 150/175 steps — so this
  is *not* "needs more iterations". A flat plateau above the noise floor at fixed capacity
  (~11,313 params, weight-shared and therefore **L-independent** while N grows 144→882) points to
  an **ansatz-capacity / optimization-landscape limit**, which is exactly the failure mode the
  hz-sweep Vscore notes predicted ("a nonzero floor = capacity limit, not a training limit").
  Correct wording for these results: *resolved nonzero variational error, growing with L*.
  Next diagnostic to separate capacity from optimizer: widen `inv_hidden` at fixed n_iter and see
  if the plateau drops (capacity) or not (optimizer/diag_shift).
