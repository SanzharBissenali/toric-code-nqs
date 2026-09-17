# phase3d — L=4 multi-plane mapping plan (agreed 2026-09-17)

The durable plan for the pivoted campaign. Read this after a context compaction; it supersedes the
Stage-1/Stage-2 scope in `notes/phase3d_handoff.md` (which still holds the mechanics: cluster setup,
launcher, manifests, monitors, fix recipes). Status of each item is kept in the checklist at the end.

## 0. Decisions (user, 2026-09-17)

- **L=4 only.** L5/L6 are dropped; L=4 maps the diagram well enough and costs ~1.5–2 h/point.
- **h_y ≥ 0 only** (H(h_y) and H(−h_y) are time-reversal partners; the diagram is mirrored).
- **Locators (frozen):**
  - electric cuts (fix h_x, sweep h_z): `O_FM_paratoric` inflection (richards/logistic) — 2nd order;
  - magnetic cuts with h_z ≤ 0.3 (topological → trivial): `O_FM_membrane_R1` inflection on the
    winner curve; energy crossing is a secondary check only (200-step links lag);
  - magnetic cuts with h_z > 0.3 (trivial → trivial): **steepest step of ⟨σ^x⟩ on the winner curve**
    (h_c = bracket midpoint, err = half spacing; accepted as a jump only if |step| ≥ 0.2 and the step
    slope ≥ 2× the curve's median slope, else "crossover"); ⟨A_v⟩/⟨B_p⟩ must jump in the same
    bracket. The energy branch crossing compares two separately optimized ansätze and is only as good
    as the worse-converged branch — demoted to a check. Implemented in
    `analysis/scripts/phase3d_status.py` (`step_locator`, `jump_entry`, `cut["jump"]`).
- **Chain recipe changes for every future chain:** links **300 steps** (was 200); link spacing
  **0.05 across a ±0.15 window around the seeded crossing, ≤ 0.1 everywhere else** (anchor → first
  link included); **both branches cover the whole window** so they overlap.
- Item 6 of the 2026-09-17 proposal (hysteresis test along h_z at h_x = 0.8) is **not** wanted.

## 1. What exists (as of 2026-09-17)

Planes h_y = 0, 0.2, 0.4 at L=4: 3 electric cuts (h_x = 0, 0.5, 0.8) + 5 magnetic cuts
(h_z = 0, 0.2, 0.4, 0.7, 1.0), 280 points, zero divergences. Plus the older `hy_cuts_L4` cuts
(h_x = 0.2 electric, h_z = 0.1 magnetic) and the pure h_y axis (cold points, `results/hy_axis_L4`).

| locator | h_y=0 | 0.2 | 0.4 |
|---|---|---|---|
| electric h_z,c at h_x = 0 / 0.5 / 0.8 | 0.295 / 0.300 / 0.335 | 0.291 / 0.297 / 0.352 | 0.274 / 0.282 / 0.351 |
| envelope h_x,c at h_z = 0 / 0.2 (membrane O_FM) | 0.815 / 0.828 | 0.815 / 0.819 | 0.790 / 0.808 |
| h_z = 0.4 M_x jump | 0.888(13) | 0.825(25) | 0.825(25) |
| h_z = 0.7 M_x jump | 1.20(5) | 1.20(5) | 1.20(5) |
| h_z = 1.0 | crossover | crossover | crossover |
| h_y axis (h_x=h_z=0) | h_y,c ≈ 1.16 (branch-crossing estimate; raw jump bracket [1.2, 1.3]) | | |

Reading: the h_y field barely moves the boundary through 0.4; the first-order (trivial→trivial) line
ends between h_z = 0.7 and 1.0; the topological lobe closes in h_y between the 1.0 and 1.2 planes.
Analysis notebook: `analysis/notebooks/phase3d_L4_planes.ipynb` (input `results/phase3d/summary.json`
from `phase3d_status.py --export-summary`). Viewer artifact (Cuts + Phase-diagram views):
https://claude.ai/code/artifact/eb8e3881-84e0-4c28-bf03-6827dfc5a554 — always republish to this URL.

## 2. Work items

### A. Improve the three existing planes (h_y = 0, 0.2, 0.4)

| # | what | why | jobs |
|---|---|---|---|
| A1 | h_z = 0.7: links at h_x = 1.2, both branches | jump sits in the 1.15→1.25 gap; halves ±0.05, lets branches overlap | 6 links |
| A2 | h_z = 0 and 0.2: links at h_x = 0.75, both branches | membrane O_FM rises across the 0.7→0.8 gap → ±0.05–0.08 errors | 12 links |
| A3 | new electric cut **h_x = 0.65** and new magnetic cut **h_z = 0.25** (topological → trivial, membrane O_FM) | pin the corner where the 2nd-order line meets the 1st-order line (h_x ≈ 0.8, h_z ≈ 0.33); h_z = 0.3 would start *on* the electric line at h_x ≤ 0.5, so 0.25 | 7 cold + 2 chains per plane |
| A4 | new tail cut **h_z = 0.85** (anchors 0.7 / 1.7, window centred 1.3) | bracket the endpoint of the first-order line (0.7 jumps, 1.0 does not) | 2 chains per plane |
| A5 | optional: redo the h_y = 0 polarized (dn) anchors with the gentle recipe and re-run the dn links at h_z = 0.4 / 0.7 | dn Vscore 0.06–0.12 there vs 0.02–0.05 elsewhere; makes the energy crossing a usable check | 2 chains |

A1/A2 are realised by the new link rule: the planner sees 1.2 and 0.75 as missing links of an existing
branch and submits them as warm-started inserts. Cost of A1–A4 for three planes ≈ 25 GPU-h.

### B. New planes h_y = 0.6 and 0.8 and 1.0 (user, 2026-09-17)

All cuts of §A's final list (electric h_x = 0, 0.5, 0.65, 0.8; magnetic h_z = 0, 0.2, 0.25, 0.4, 0.7,
0.85, 1.0) with the new chain rule. Electric window shift ≈ −0.14·h_y² (−0.05 / −0.09 / −0.14 at
0.6 / 0.8 / 1.0), refined automatically once the L4 fit lands. ≈ 50 GPU-h per plane. Launch:
`HY=<hy> LS=4 MAX_QUEUE=200 bash nersc/launch_phase3d.sh` on the cluster (+ scrontab line per plane,
see handoff §2). Quality watch: Vscore floor ≈ 0.5·h_y² (0.18 / 0.32 / 0.5).

### C. y-cuts: sweep h_y at fixed (h_x, h_z) — the roof and the two remaining first-order lines

Organize the 3D diagram as fixed-h_x slices in the (h_y, h_z) plane (and fixed-h_z slices in (h_x, h_y)).
Each slice's pocket has two walls: the electric (or magnetic) wall, already given point by point by the
h_y planes, and the roof h_y,c(h_z), measured by h_y sweeps at fixed (h_x, h_z) sitting on the same h_x and
h_z values as the existing cuts, so everything superimposes. Beyond the walls the same sweeps find the
trivial→trivial first-order lines that leave the pocket's tip (y/z-polarized at h_x = 0, y/x-polarized at
h_z = 0), ending in crossovers like the x/z line.

Cut type `ycut_hx{hx}_hz{hz}` (kind "ycut", sweep h_y): up chain from the anchor h_y = 0.6, dn chain from
h_y = 1.5, seeded centre 1.15 (±0.15 window at 0.05, coarse 0.1 outside, 300-step links, complex lane).
Locators: roof cuts (topological → trivial) = steepest step of ⟨σ^y⟩ on the winner curve with ⟨A_v⟩/⟨B_p⟩
agreement, O_FM (loop) as the topological check; trivial→trivial cuts the same without O_FM. Energy crossing
secondary. Vscore floor ≈ 0.5·h_y² (0.6 at 1.1): trust energy/stabilizer jumps over ⟨σ^y⟩.

| family | cuts | probes |
|---|---|---|
| roof, slices h_x = 0, 0.5, 0.8 | h_z = 0, 0.1, 0.2 at each h_x (9) | h_y,c(h_z) per slice; (0, 0) = the axis point ≈ 1.16 |
| y/z first-order line, slice h_x = 0 | h_z = 0.4, 0.55, 0.7 (3) | where the line runs, where it ends |
| y/x first-order line, slice h_z = 0 | h_x = 1.0, 1.2, 1.4 (3) | same for the other corner |

15 cuts × 2 chains ≈ 120 GPU-h. Outputs live in `$PSCRATCH/tc_nqs/phase3d/ycuts/<cut>/L4/`, launched as
the pseudo-plane `HY=y` (`LS=4 HY=y MAX_QUEUE=220 bash nersc/launch_phase3d.sh`), viewer tab "y-cuts".
Smoke-test one dn anchor (h_y = 1.5, y-polarized, never run before) on gpu_debug before the 30 chains.
Later additions: h_x = 0.5 for the y/z line, h_z = 0.2 for the y/x line, slice h_x = 0.65.

### D. Later

- Plane h_y = 1.2 as a closure check (expected fully trivial).
- Electric grid rebalancing wherever a new plane's L4 fit lands skewed (≥ 3 points on each side).
- QMC referee at h_y = 0 for the new cuts (approved earlier; ParaToric β=24 ×8, `--validate` first).

## 3. Implementation checklist

- [x] planner `analysis/scripts/phase3d_grid.py`: HY_VALUES + `_DHY` (0.6, 0.8, 1.0); ELECTRIC_HX += 0.65
      (`_DHX[0.65] = 0.04`); MAGNETIC_HZ += 0.25; TAIL_HZ += 0.85 (`_ANCHORS[0.85] = (0.7, 1.7)`);
      generic chain-link rule (0.1 outside / 0.05 inside a ±0.15 window, both branches cover it);
      links N_ITER 300; L4 chain walltime raised; self-tests updated; `--dry` per plane checked.
- [x] topological threshold: h_z ≤ 0.3 counts as topological → trivial (status `TOPO_TRIVIAL_HZ_MAX`,
      viewer `isTopo`/`TOPO_HZ`, notebook `TOPO_HZ_MAX`).
- [ ] cluster: pull the branch into `~/toric-code-nqs`, re-sync `phase3d_grid.py`, dry-run, then
      (after approval per plane) launch A for 0/0.2/0.4, then B for 0.6, 0.8.
- [ ] y-cut type (C): planner (kind ycut, `HY=y` pseudo-plane) → launcher → watcher → pull → status export
      (jump on sy) → viewer tab; dry-run; gpu_debug smoke of a dn anchor; submit 30 chains.
- [ ] closure check h_y = 1.2 (D).

## 4. Monitoring (re-arm in a new session)

Local tick script (scratchpad, session-specific; recreate from this description if lost):
pull (`analysis/scripts/pull_phase3d.sh` with LOCAL_RESULTS/LOCAL_DATA pointing at the main checkout's
`results/phase3d` / `data/tc_nqs/phase3d`) → `phase3d_status.py` STATUS.md + `--export-summary` +
`--export-viewer` per plane → `phase3d_viewer_build.py` → republish the artifact; then one ssh line with
squeue counts (excluding the cron jobs), `sacct` failures of the last 6 h, GENUINE DIVERGENCE / CHAIN STOPPED
in `~/toric-code-nqs/p3d_*.out` and `slurm_logs/` modified in the last 6 h, and the last driver log lines.
Armed as a Monitor (30-min cap, re-armed on expiry; pull every other tick).

## 5. Approval log

- 2026-09-17: user approved A1–A4 (A5 optional), B (0.6, 0.8, then 1.0 added the same day), C, the 300-step links and the
  spacing rule; declined the h_x = 0.8 hysteresis test. User gave the go for everything at once the same
  evening ("submit as many jobs right now as possible"); submitted 2026-09-18 00:10 (+05).
