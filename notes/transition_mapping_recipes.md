# Transition-mapping recipes (distilled from the Phase-B reconciliation campaign, 2026-08-17/19)

Executable playbook for mapping phase-diagram cuts with the dual-basis NQS.
Written for agents in future sessions: follow the steps literally; every
threshold and flag here was validated against β-converged QMC at L=4–6 on the
h_z (2nd-order) and h_x (1st-order) cuts. Working assumption for new cuts
(including h_y ≠ 0): transitions of the same order occur at similar field
values as the sign-free ones — seed your grids from the existing map.

## 0 · Invariants (every launch, both recipes)

- **Architecture flags are MANDATORY on every submit** — the wrapper defaults
  do NOT reproduce the production ansatz:
  `DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=$((L-1))`.
  Omission ⇒ instant broadcast `ValueError` on any warm start
  (conv-shape mismatch = missing INV widths; bias `(8,)vs(24,)` = missing DUAL).
- **Warm starts**: `--init_from` takes the BASE path — no `.mpack` extension.
  A wrong path **silently cold-starts**. After every link starts, grep the log
  for `warm start: loaded` AND check E[step 0] is in the warm range (cold L6
  starts near −500s; warm sits near the seed's energy). Cold line ⇒ scancel
  the rest of the chain, fix, relaunch.
- **Always**: `--snapshot_every 50 --final_eval_rounds 8`; `--checkpoint_every 10`;
  5 h walltime cap with `AUTO_RESUBMIT=1` for anything > ~280 L6-steps.
- **Divergence forensics**: diverged runs look CLEAN on W&B (the guard rolls
  back before logging). Truth source = `grep "GENUINE DIVERGENCE" <log>` and
  the JSON's `diverged` flag. A diverged JSON's observables are the last-sane
  snapshot; its `E0` is numerical garbage — never plot either as physics.
- **Error convention**: pull = (NQS−QMC)/σ_comb; NQS errors are underestimated
  ~×3 ⇒ corrected pull = raw/3, threshold 3. The ×3 masks coherent
  systematics: ALWAYS also scan raw pulls across adjacent points — three
  same-sign neighbors is a systematic regardless of per-point σ.
- **Per-L step timing** (A100 shared QOS): L4 ≈ 4.5 s, L5 ≈ 16.5 s, L6 ≈ 61 s.

## A · Recipe: SECOND-ORDER transition (continuous; e.g. h_z sweep at fixed h_x)

1. **Grid**: 0.02-spaced points through the expected transition window
   (± 0.06 around the prior estimate), coarser (0.05–0.15) outside.
2. **Runs**: cold starts, `DT=0.02 LR_MIN=0.002 DIAG_SHIFT=1e-3`
   (**3e-3 for L≥5 near the transition** — 1e-3 diverges).
   **Budget: fixed 500 steps for every L.** Extend to **750** (one +250
   resume) ONLY IF the snapshot series shows an obvious drift over the last
   100 steps in the observables — order parameter or locals (A_v/σ_z)
   still moving monotonically rather than fluctuating about a level.
   No extensions beyond 750 without explicit user sign-off.
   (Context: 500 was measured-sufficient at L4; L5/L6 peak points were
   still drifting at step 500 — the drift test is what catches them.)
3. **Convergence check per point** (from the `.snapshots.json` replay,
   `analysis/eval_snapshots.py --rounds 4..8`):
   - energy: last-quarter drift < 2× its error bar;
   - the "hotter-state" signature = NOT converged: A_v still rising, σ_z
     still falling toward their asymptotes while O_FM already flat.
     O_FM converges FIRST at 2nd order; locals last. If locals still drift
     at the end, extend the run (resume, +250–500 steps) — do not accept.
4. **Extraction**: plot Z-string O_FM vs field (per L); locate the transition
   from the O_FM inflection (sigmoid/tanh fit, `analysis/vertical_line_hz.ipynb`
   machinery) and/or dO/dh peak; S2-Rényi locator as an independent
   cross-check. FSS via the existing crossing pipelines.
5. **Plots** (house style, see §D): four figures per cut — energy/spin,
   stabilizers (A_v+B_p), magnetization, order parameter.
6. **QMC referee (sign-free cuts only)**: β=12 z-basis refs are fine AWAY
   from the peak; within ±0.04 of the peak run a β=24 check once per L.
   Loader must use the highest-β subset only.

## B · Recipe: FIRST-ORDER transition (e.g. h_x sweep at fixed h_z)

**Never map the window with cold starts.** Cold + dt<0.02 inside the
coexistence window diverges (15/15 at L6; wall ∝ 1/dt — any off-branch state
must reorganize through VMC-fatal configurations). Cold dt=0.02 survives but
lags in O_FM. No cold-retry ladders: after one cold failure, go to chains.

1. **Anchors**: converge one cold point deep in each phase (≥0.15 from the
   expected crossing; dt=0.02, 500 steps). These seed the chains.
2. **Two chains through the window, 0.05 field steps, Slurm `afterok`-chained**:
   - **up-chain**: topological anchor carried toward the trivial phase;
   - **down-chain**: trivial/ordered anchor carried toward topological.
   Link protocol: `DT=0.005 LR_MIN=0.0005 DIAG_SHIFT=3e-3 N_ITER=200`,
   `--init_from <previous link's base path>`; verify warm-load per §0.
3. **Expect and RECORD, per link, one of four outcomes** (all are data):
   - *relaxes as GS* — flat E at/below every other state at that field;
   - *metastable hold* — flat E plateau ABOVE a known-lower state, O_FM keeps
     the branch value (robust even to 2×dt — don't bother kicking it);
   - *smooth shed* — O_FM decays during the run (branch's minimum vanished);
   - *VMC crash* — E rises, spread balloons, `GENUINE DIVERGENCE`
     (the branch destabilized mid-crossing). Crash/shed = the branch's
     **spinodal**; cancel the chain's remaining links (they'd inherit a
     corrupted or off-branch state).
4. **Per-point winner**: the branch state with the **lowest energy** is the
   plotted/GS value. Metastable-branch data goes ONLY into the hysteresis
   figure, never into sweep figures.
5. **Extraction**: crossing = energy branch-crossing bracket (where the
   lower-E branch switches); spinodals = last field each branch survived;
   plot the hysteresis loop (both branches' O_FM vs field, crash marker,
   reference values open-marker).
6. **The resonance blind spot — budget for it, don't fight it**: within
   ±0.05 of the crossing expect the best single-branch energy to sit
   ~1(L4)–3(L6) above the true GS, with locals tilted toward the state's
   origin. Proven-useless remedies: more steps (flat at L5, saturating at
   L6), 2×dt kicks (hold). Label such points "resonance window" in outputs.
   Optional resolution experiment: two-state subspace diagonalization from
   the two saved branch checkpoints (predicts subspace E₀ drops toward the
   referee value).
7. **QMC referee (sign-free only)**: within ±0.1 of the crossing, β=12
   x-basis refs are ~10–25σ thermally biased — **β≥24 with ×8 decorrelation
   is mandatory**, and even those need a χ²_red gate (reject/re-run > ~2;
   near-crossing runs strain decorrelation). β-ladder (12/24/48) once per L
   to demonstrate convergence (per-doubling decay r ≈ 0.1 ⇒ β=48 converged).

## C · Sign-full addendum (h_y ≠ 0, fermionic): no QMC referee

Use the same two recipes with the complex ansatz, replacing the QMC column by
the **internal trust ladder** (each validated against QMC in the sign-free
campaign):
1. **Variational energy ordering** between deliberately prepared states
   (cold, up-carried, down-carried) — the referee for "which branch is GS".
2. **Branch bracketing**: prepared states straddle the truth in the local
   observables from opposite sides (origin-tilt fingerprint); the true value
   lies between the brackets. Report the bracket, not one state's value.
3. **Hysteresis self-consistency**: spinodals must order as
   h_sp,ordered < h_c < h_sp,topological; the crossing from energy-ordering
   must fall between the spinodals. Violations ⇒ un-converged states.
4. **Im⟨E⟩ ≈ 0** check every run (complex-ansatz sanity), plus the h=0 /
   analytic anchors where the cut touches them.
5. Assume the resonance blind spot exists near any first-order feature;
   the subspace-diag check is the built-in mitigation.

## D · Plotting standard (all figures)

Plasma colormap keyed by L (0.15/0.5/0.8 for L=4/5/6); open axes (top/right
spines off); NQS filled markers with **bars ×3, labeled "(bars ×3)"** on
order-parameter panels; referee values as open markers ±1σ; ms=5, capsize=2;
wide content in its own container. Trajectory figures: corrected pull vs step
from snapshot replays, ±3σ gray band. Notebook `plt.savefig` lines stay
commented — the user promotes figures manually.

## E · Known traps (cost real time in the campaign)

- QMC dir names are not `:g`-uniform (`qmc_hx1.0_hz0.1`): glob+parse, never
  format-and-look-up.
- Never equal-weight-combine QMC files across β (highest-β subset only).
- FORCE W&B sync resets run groups — regroup via `wandb.Api` after every sync.
- nbstripout can't run in bare worktrees (`.venv` path) — bypass with
  `-c filter.nbstripout.clean=cat` for merge commits only.
- macOS `head` is an HTTP tool — use `sed -n`/`grep -m`.
- Jupyter saves clobber concurrent disk edits to open notebooks — reload from
  disk before saving, or expect to re-apply.
