# ★ RESUME HERE (written 2026-09-18 00:55 +05, before a context compaction) ★

You are the orchestrator of the running L=4 multi-plane campaign. Everything below this block is the plan;
`notes/phase3d_handoff.md` has the cluster mechanics and fix recipes; memory `phase3d-campaign-plan` points here.

State
- 189 campaign jobs submitted 00:10 (+05) (planes 0/0.2/0.4 upgrades: h_x=0.65 electric, h_z=0.25/0.85 chains,
  warm inserts; planes 0.6/0.8/1.0: 11 cuts each). First runs started ~00:40. scrontab: six hourly drivers
  `LS=4 HY=<hy> MAX_QUEUE=220` (:15..:40) + W&B sync. Job `debug_grayanchor` in the queue is NOT ours — leave it.
- y-cuts (plan §C): **SUBMITTED 2026-09-18 02:36** after smoke 58478685 passed (E −223.8, Vscore 0.21, 3.07 s/step,
  ~14 min compile; only the post-training final eval was cut by the 25-min gpu_debug cap). 27/30 chains in the
  queue (cap 220); the seventh scrontab driver `45 * * * * LS=4 HY=y MAX_QUEUE=220 …` (-J p3d-driver-hyy) fills the
  rest. Results land in `$PSCRATCH/tc_nqs/phase3d/ycuts/ycut_hx*_hz*/L4/` (pull: `HY=y` → local `results/phase3d/ycuts`).
  Gotcha: run the launcher with the conda python (`export PATH=$HOME/.conda/envs/tc-nqs/bin:$PATH`); the login
  node's system python3 breaks the planner's fit-module import (launcher now prefers the conda python itself).
- **Cadence (user decision 2026-09-18): every 2 h, not more often** — check jobs, pull, republish the viewer.
  Armed as a session CronCreate job (`23 */2 * * *`, 7-day expiry, re-create in a new session) whose prompt runs
  `DO_PULL=1 SINCE_MIN=125 bash analysis/scripts/phase3d_local_tick.sh` (worktree; zsh) and also checks the y-cut
  smoke job (submit the 30 chains when healthy). After a pull with new finals: republish
  `$VIEWER_DIR/phase3d_viewer.html` to the artifact URL in §1 (read it once with the Artifact tool first in a new
  session; omit favicon/capabilities). Also export per-plane JSONs for every `hy*` dir AND `--export-viewer y`
  once `ycuts/` exists locally (the tick script loops over `hy*` only — add `y` there when y-cuts land).
- Analysis notebook `analysis/notebooks/phase3d_L4_planes.ipynb` (main checkout + worktree copy) has no y-cut
  section yet: add one (roof h_y,c(h_z) per h_x slice + the two trivial lines) once y-cut finals exist; its
  generator lived in the session scratchpad — rebuild edits in the notebook itself.
- Repo: worktree `/Users/sanzhar123/Desktop/toric-code-nqs-p3d/integration` on `feat/phase3d-campaign` (pushed);
  cluster clone `~/toric-code-nqs` at the same commit (`git pull -q origin feat/phase3d-campaign` after every push).
  Main checkout `/Users/sanzhar123/Desktop/toric-code-nqs` is the peer's branch; `results/phase3d/` and the
  notebook there are untracked — ask the user where to bank them. Never commit to main.
- User preferences today: concise, lead with what changed; report only changes on routine ticks; they look at
  the viewer themselves; approvals for everything in §2 are given (per-plane go no longer needed).

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

### 0.b Plane-by-plane review decisions (2026-09-19, with the user)
- **h_y = 0, electric h_x = 0.8**: at L=4 the S₂ curve does not show the topological plateau, but L=5/6 do (finite-size
  effect only), so the point stays topo→trivial. CAVEAT for the higher planes: their h_x = 0.8 cuts are L=4 only and the
  data there suggest the cut is no longer topo→trivial at larger h_y — re-examine plane by plane, do not assume.
- **Tail-cut rule refined**: trivial→trivial cuts (h_z > 0.3, y-cuts) use the M jump when sharp; if the jump test fails
  but the up/down branches still cross in E/N (a small hysteresis loop, h_z = 0.85 at h_y = 0: 1.316 ± 0.018), the
  E/N crossing locates the transition and the point enters the phase diagram. Merged branches (h_z = 1.0 at h_y = 0,
  all points overlap) remain a crossover, no point.
- **h_y = 0.2 review**: electric h_x = 0.8 stays topo→trivial (sits just right of the envelope 0.78–0.79 but the
  curves still read topo→trivial). The energy-crossing detector was wrong on both tail cuts: at h_z = 1.0 it centred
  three noise sign-flips (|ΔE| ≤ 0.001/site) into a "crossing" at 1.475; at h_z = 0.85 the dn branch is lower by
  ~0.001/site (≈10σ_raw) at EVERY overlap point (a branch-quality offset), so no crossing although the M_x loop is
  visible. Fixes: (i) crossing needs a NET sign flip between the significant ends of the overlap (else merged);
  (ii) new hysteresis-loop locator = centre of the run of ≥3 consecutive points with |M_dn − M_up| > 0.015 (and
  > 3σ_inflated), err = half the run width; tail-cut priority = M jump → net E/N crossing → loop centre → crossover.
  Result at h_y = 0.2: h_z = 0.85 → loop 1.35 ± 0.05, h_z = 1.0 → crossover. (h_y = 0 unchanged.)
- **h_y = 0.4 review**: electric h_x = 0.8 sits right of the envelope (0.755/0.782/0.764 at h_z = 0/0.2/0.25): S₂ flat
  at 1.00 (no 3 ln 2 plateau), O_FM 0.06 → 0.64 in one link → FIRST-ORDER trivial→trivial (x-pol → z-pol), located
  by the M_z jump: h_z = 0.345 ± 0.015 (M_x, A_v, B_p agree). Encoded as `ELECTRIC_FIRST_ORDER = {(0.4, 0.8)}` in
  phase3d_status.py; the point joins the trivial→trivial line, not the envelope. h_z = 0.85 → loop 1.40 ± 0.08 (ok).
  h_z = 1.0 → CROSSOVER by review (`TAIL_CROSSOVER = {(0.4, 1.0)}`): the 0.063 spike at 1.45 is under-convergence and
  the remaining ~0.027 separation runs flat to the overlap edge. Loop rule also hardened: isolated spikes smoothed,
  separation must peak inside the run.
- **h_y = 0.6 review**: electric h_x = 0.8 is again a first-order trivial→trivial step (added to ELECTRIC_FIRST_ORDER);
  everything else fine. Plot rule (user): the trivial→trivial dashed line starts at the envelope's corner = the
  midpoint between the last electric and the first magnetic envelope point — ONLY on planes whose h_x = 0.8 cut is
  first-order (h_y ≥ 0.4). At h_y = 0 / 0.2 the h_x = 0.8 point is the apex of the topological region and the tail
  line starts from it.

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
- [x] y-cut type (C) built end to end (commit 1143ad2, 2026-09-18): planner kind ycut + `HY=y` pseudo-plane,
      launcher/watcher/pull, status export (M_y jump primary), viewer y-cuts tab + 3D stars; cluster dry-run = 30 jobs.
- [x] y-cuts: gpu_debug smoke of the dn anchor (job 58478685, h_x=h_z=0, h_y=1.5) PASSED 2026-09-18 (E −223.8, Vscore 0.21, 3.07 s/step; final eval cut by the 25-min cap only). 27/30 chains submitted 02:36 (manifest_20260917_143645), HY=y scrontab driver at :45 fills the last 3. →
      `LS=4 HY=y MAX_QUEUE=220 bash nersc/launch_phase3d.sh` (30 chains) + scrontab line `HY=y` at :45.
- [ ] closure check h_y = 1.2 (D).

## 4. Monitoring (re-arm in a new session)

Local tick script (scratchpad, session-specific; recreate from this description if lost):
pull (`analysis/scripts/pull_phase3d.sh` with LOCAL_RESULTS/LOCAL_DATA pointing at the main checkout's
`results/phase3d` / `data/tc_nqs/phase3d`) → `phase3d_status.py` STATUS.md + `--export-summary` +
`--export-viewer` per plane → `phase3d_viewer_build.py` → republish the artifact; then one ssh line with
squeue counts (excluding the cron jobs), `sacct` failures of the last 6 h, GENUINE DIVERGENCE / CHAIN STOPPED
in `~/toric-code-nqs/p3d_*.out` and `slurm_logs/` modified in the last 6 h, and the last driver log lines.
Committed as `analysis/scripts/phase3d_local_tick.sh`. Cadence: **every 2 h** (user decision 2026-09-18; 15/30 min
was too often) via a session CronCreate job at :23 on even hours, DO_PULL=1 SINCE_MIN=125 every tick; the same tick
checks the y-cut smoke job and submits the y-cut chains once it passes. Report only changes.

## 5. Approval log

- 2026-09-17: user approved A1–A4 (A5 optional), B (0.6, 0.8, then 1.0 added the same day), C, the 300-step links and the
  spacing rule; declined the h_x = 0.8 hysteresis test. User gave the go for everything at once the same
  evening ("submit as many jobs right now as possible"); submitted 2026-09-18 00:10 (+05).
- 2026-09-18 10:40 — RETRY (autonomous, single point): plane 0.6 electric h_x=0.65 h_z=0.32 GENUINE DIVERGENCE (dt 0.02 wall: guard rollback every other step, 203 rollbacks). Parked as redo_58477412, resubmitted with DIAG_SHIFT=5e-3 DT=0.01 (manifest_20260917_223536). Neighbours 0.17/0.23 healthy.
- 2026-09-18 12:40 — RETRY (autonomous, single point): plane 0.8 electric h_x=0.8 h_z=0.17 GENUINE DIVERGENCE at step 54 (27 rollbacks, same dt 0.02 wall). Parked as redo_58477487, resubmitted with DIAG_SHIFT=5e-3 DT=0.01. Other 6 points of the cut healthy.
- 2026-09-18 14:05 — RETRY ×2 (autonomous): plane 1.0 electric (0.65, h_z 0.14) and (0.8, h_z 0.24) GENUINE DIVERGENCE (dt 0.02 wall, 6 rollbacks then gave up). Parked as redo_58477527 / redo_58477536, resubmitted with DIAG_SHIFT=5e-3 DT=0.01. First retry (plane 0.6, 0.65/0.32) landed clean → cut complete, h_z,c = 0.278(8). Watch: plane 1.0 (0.8, 0.30) finished non-diverged but with 178 rollbacks, Vscore 0.32.
- 2026-09-18 16:45 — RETRY (autonomous): plane 0.6 magnetic h_z=0 dn ANCHOR (h_x=1.25) GENUINE DIVERGENCE at step 38 → CHAIN STOPPED, 9 dn links never ran. Anchor parked as redo_58477440, resubmitted with DIAG_SHIFT=5e-3 DT=0.01; the 9 never-run link rows of job 58477440 were dropped from manifest_20260917_121117 (backup .bak_58477440) so the hourly driver re-plans the dn chain once the anchor lands healthy. Plane 1.0 retries (0.65/0.14, 0.8/0.24) landed clean.
- 2026-09-18 21:40 — BUG FIXED (04ff27d): since 1143ad2 the planner's `plan --hy` parsed hy as str → TypeError in the L4 gap-fill tier → the launcher read the crash as '0 jobs'. All hourly drivers for numeric planes were silent no-ops from ~00:30 to 21:40 (chains/cold points were pre-submitted, so no data lost; refine/gap-fill tiers idle). Retry tool now names L4 anchor retries with the branch suffix (e9af1db); the plane 0.6 hz=0 dn anchor (retried healthy, E −261.9) was renamed by hand and its 7 window links resubmitted (manifest_20260918_093450). Two more dn anchors diverged with the dt 0.02 recipe — plane 0.8 hz=0.85 (h_x 1.7, step 37) and plane 1.0 hz=0 (h_x 1.25, step 135) — parked (redo_58477506, redo_58477541), retried gentle, link rows dropped. Pattern: dn (x-polarized) anchors at h_y ≥ 0.6 need the gentle recipe at L4 too — consider gentle = dn and (L≥5 or hy≥0.6) for future submissions. Lost by the drop: dn coarse points 1.15/1.05 outside the window (chain tier only refills the window).
- 2026-09-18 22:40 — both remaining dn-anchor retries landed healthy (plane 0.8 hz=0.85 h_x=1.7: E −343.9, Vscore 0.024; plane 1.0 hz=0 h_x=1.25: E −274.9, Vscore 0.081); their window links are re-planned by the gap-fill tier on the next driver pass. Revived drivers submitted the refine tiers: chain up/dn refine pairs on planes 0.2 (hz 0.2/0.4/0.7/1.0), 0.4 (hz 0.2/0.25/0.4/0.7/0.85), 0.6 (hz 0.2) + electric refine ×2 on plane 1.0 h_x=0 (14 jobs).
- 2026-09-19 00:45 — RETRY (autonomous): plane 1.0 magnetic h_z=1.0 dn ANCHOR (h_x=1.7) GENUINE DIVERGENCE at step 10 → CHAIN STOPPED. Parked redo_58477554, retried gentle (DIAG_SHIFT=5e-3 DT=0.01), 7 never-run link rows dropped. 4th dn-anchor divergence, all at h_y ≥ 0.6 — the gentle recipe is 3/3 so far. Drivers re-planned the dn links for plane 0.8 hz=0.85 and plane 1.0 hz=0 (7 each). y-cuts started: first anchors healthy (Vscore 0.07–0.17).
- 2026-09-19 01:20 — user: add magnetic_hz0.1 (chain pair, ~20 pts) to planes 0.6/0.8/1.0 so the envelope has the same rungs as 0/0.2/0.4 (where the older hy_cuts_L4 campaign provides it via phase3d_extras.json). Launcher DEFAULT_CUTS gains magnetic_hz0.1 for HY ∉ {0, 0.2, 0.4, y}; submitted right away.
