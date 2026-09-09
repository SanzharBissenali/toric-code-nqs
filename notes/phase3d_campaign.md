# phase3d — mapping the 3D bTC phase diagram (h_x, h_y, h_z), L = 4, 5, 6

One page. Full design + decision log: `~/.claude/plans/hello-claude-how-are-floating-blossom.md` (orchestrator).
Recipes are authoritative: `notes/transition_mapping_recipes.md` §0 (invariants) §A (2nd order) §B (1st order) §C (h_y≠0).

## Scope (approved 2026-09-09)
- Planes h_y ∈ {0, 0.2, 0.4} first; 0.6/0.8 only after these are complete and reviewed.
- Per plane 10 cuts, each at L = 4, 5, 6:
  electric (fixed h_x ∈ {0.0, 0.2, 0.5, 0.8}, sweep h_z; 2nd order; cold, parallel points);
  magnetic (fixed h_z ∈ {0.0, 0.1, 0.2}) and tail (h_z ∈ {0.4, 0.7, 1.0}), sweep h_x; 1st order; two warm chains.
  h_x=0.2 / h_z=0.1 already exist (Phase B at h_y=0, hy-cuts at 0.2/0.4, L=4) — registered with `lane="phase3d"` only if redone.
- Every run: dual basis, `DUAL=1 NONINV_HIDDEN="4 8" INV="8 8" KERNEL=L-1`, `--snapshot_every 50 --final_eval_rounds 8`,
  checkpoint every 10, 5 h wall cap + AUTO_RESUBMIT. Cold points 500 steps (dt 0.02, ds 1e-3 L4 / 3e-3 L5,6).
  Chains: anchor 500 steps cold, links 200 steps (dt 0.005, lr_min 5e-4, ds 3e-3), one resume-safe job per branch
  (`tc3d.sweep --warm_start --anchor_overrides`), stop on divergence or E0 above the h=0 bound.

## Per-plane program (one Sonnet agent per plane)
1. Wave 1 — L=4 on all cuts: `HY=<hy> LS=4 bash nersc/launch_phase3d.sh` (31 jobs). Grids: 7 electric points around a seeded
   centre; chains anchors ≥0.15 outside the window, 0.1/0.05 link spacing.
2. Watch every 30–60 min: `nersc/watch_phase3d.sh` (GENUINE DIVERGENCE, `warm start: loaded` on every link, E0 < −(#A_v+#B_p),
   TIMEOUT without resubmit). Electric divergence → resubmit with ds 3e-3 → 5e-3. Chain crash → record spinodal, branch stops.
3. Pull + look: `analysis/scripts/pull_phase3d.sh`; `analysis/notebooks/phase3d_progress.ipynb` (§1 `HY` selects the plane;
   coverage, E vs bound, Vscore, learning curves, partial h_c(L)); `analysis/scripts/phase3d_status.py` → `results/phase3d/STATUS.md`.
   W&B project `tc3d-phase3d`, groups `hy{hy}/{cut}/L{L}`, synced every 30 min by scrontab `p3d-wandb-sync`.
4. Refine L=4: electric — if h_c_err > 0.01 or the inflection is > 0.02 from a grid point, add 2 cold points (±0.02, then ±0.01);
   if the rise is not bracketed, extend outward by 0.06. Chains — if the energy-crossing bracket > 0.05, add 0.025 links from the
   nearest saved link checkpoint. Two rounds max.
5. Wave 2 — L=5,6 with windows recentred on the L=4 result: `phase3d_grid.py --json --from_l4 results/phase3d/hy<hy>` →
   `LS="5 6" GRID_JSON=... bash nersc/launch_phase3d.sh` (62 jobs). Same watch/refine loop.
6. Bank: finals + snapshots → `results/phase3d/hy{hy}/{electric_hx*|magnetic_hz*}/L{L}/` (curves stay on scratch / W&B).
   h_y≠0 trust ladder (§C): E below the same-(h_x,h_z) h_y=0 QMC value where one exists, E monotone along the cut, Im E ≈ 0,
   Vscore ≈ 0.5·h_y² + baseline, one ±h_y pair per cut per L.
7. Locate + register: electric cuts → one `_cut(...),` line in the `CAMPAIGN = [ ]` list of `analysis/notebooks/transition_fss.ipynb`
   (obs `O_FM_paratoric`, S2 from the in-job final-state replay); first-order cuts → `analysis/scripts/firstorder_fit.py`
   (energy branch crossing primary; M_x, ⟨A_v⟩, ⟨B_p⟩, M_z jumps secondary; `kind` topo-trivial for h_z ≤ 0.2,
   trivial-trivial for h_z ≥ 0.4). Run §7 headlessly → `results/transitions/<tag>.json` → `phase_diagram_3d_btc.ipynb`.
8. Report per plane: h_c(L=4,5,6) and FSS h_c(∞) per cut, quality flags, divergences, resonance-window points, GPU-h used.

## Gates
Stage 1 h_y=0 (+ QMC referee at 2–3 points per new cut, β=24) → Stage 2 h_y=0.2/0.4 at L=4,5 → Stage 3 L=6 at h_y≠0 only
after the measured L=6 complex step time (unmeasured; extrapolated ≈250 s/step). ≤40 queued jobs per plane, ≤120 total.
Production submissions need the user's explicit go per stage; gpu_debug smokes are autonomous.
