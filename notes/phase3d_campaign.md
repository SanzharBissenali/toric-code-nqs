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

## Per-plane program (one Sonnet agent per plane; the launcher decides, the agent re-runs it)
`HY=<hy> bash nersc/launch_phase3d.sh` is idempotent and state-driven: re-run it any time (≈hourly) and it submits exactly what
has become launchable from the manifest + the finals on disk + the locator fits, in priority order under the queue ceiling.
- t=0: L=4 everything (7 electric points per cut, both chain branches); L=5/6 electric flanks+centre from seeds; L=5/6 chain
  anchors as cold jobs named after their branch.
- L=4 electric fit exists (≥5 points, err < 0.02) → remaining L=5/6 electric points recentred on h_c(L4) + per-L offset.
- L=4 chain verdict exists (crossing or merged) → L=5/6 chain jobs: links recentred (+0.02 L5, +0.06 L6), warm-started from
  their anchors (`INIT_FROM`), ordered after them by `--dependency=singleton` on the shared branch job name.
- Refinement per L once that L's own fit meets the trigger (electric: err > 0.01 or inflection > 0.02 off-grid → ±0.02 then
  ±0.01 points; rise not bracketed → extend by 0.06; chains: crossing bracket > 0.05 → 0.025 links from the nearest saved
  link checkpoint). Two rounds electric, one round chains.
Between re-runs the agent: watches (`nersc/watch_phase3d.sh`: GENUINE DIVERGENCE, `warm start: loaded` on every link,
E0 < −(#A_v+#B_p), TIMEOUT without resubmit; electric divergence → resubmit with ds 3e-3 → 5e-3; chain crash → spinodal,
branch stops), pulls (`analysis/scripts/pull_phase3d.sh`), looks (`analysis/notebooks/phase3d_progress.ipynb`, §1 `HY`
selects the plane; `phase3d_status.py` → `results/phase3d/STATUS.md`; **Viewer (Artifact, private): https://claude.ai/code/artifact/212eb390-6a83-4046-bf3c-b8526c2dc12e** — rebuild: `phase3d_status.py --export-viewer HY` per plane → `phase3d_viewer_build.py` → republish to that URL. W&B project `tc3d-phase3d`, groups
`hy{hy}/{cut}/L{L}`, synced every 30 min by scrontab `p3d-wandb-sync`).
When a cut is complete at all L: bank finals + snapshots into `results/phase3d/hy{hy}/{electric_hx*|magnetic_hz*}/L{L}/`
(curves stay on scratch / W&B); h_y≠0 trust ladder (§C: E below the same-(h_x,h_z) h_y=0 QMC value where one exists,
E monotone along the cut, Im E ≈ 0, Vscore ≈ 0.5·h_y² + baseline, one ±h_y pair per cut per L); locate + register:
electric → one `_cut(...),` line in the `CAMPAIGN = [ ]` list of `analysis/notebooks/transition_fss.ipynb` (obs
`O_FM_paratoric`, S2 from the in-job final-state replay); first-order → `analysis/scripts/firstorder_fit.py` (energy branch
crossing primary; M_x, ⟨A_v⟩, ⟨B_p⟩, M_z jumps secondary; `kind` topo-trivial for h_z ≤ 0.2, trivial-trivial for h_z ≥ 0.4);
run §7 headlessly → `results/transitions/<tag>.json` → `phase_diagram_3d_btc.ipynb`. Report per plane: h_c(L=4,5,6) and
FSS h_c(∞) per cut, quality flags, divergences, resonance-window points, GPU-h used.

## Gates
Stage 1 h_y=0 (+ QMC referee at 2–3 points per new cut, β=24) → Stage 2 h_y=0.2/0.4 at L=4,5 → Stage 3 L=6 at h_y≠0 only
after the measured L=6 complex step time (unmeasured; extrapolated ≈250 s/step). ≤40 queued jobs per plane, ≤120 total.
Production submissions need the user's explicit go per stage; gpu_debug smokes are autonomous.
