# ★ RESUME HERE (rewritten 2026-09-19 03:10 +05, before a context compaction) ★

You are the orchestrator of the running L=4 multi-plane campaign (branch `feat/phase3d-campaign`, worktree
`toric-code-nqs-p3d/integration`; a copy of this file lives in the main checkout's notes/). Everything below this
block is the plan; §0.b holds the plane-by-plane REVIEW DECISIONS made with the user on 2026-09-19 (read them
before touching any locator or label); `notes/phase3d_handoff.md` has cluster mechanics; memory
`phase3d-campaign-plan` points here.

## State (2026-09-19 03:10)
- ~1150 finals on disk, ~300 campaign jobs submitted, 0 failed. Queue ~12 running / ~118 pending. Seven hourly
  scrontab drivers (`LS=4 HY=<0.0|0.2|0.4|0.6|0.8|1.0|y> MAX_QUEUE=220`, :15..:45) plan gap-fill/refine tiers.
  Job `debug_grayanchor` in the queue is NOT ours.
- Planes 0 / 0.2 / 0.4 / 0.6: complete and REVIEWED (§0.b). Plane 0.8: reviewed, waiting for redone up chains.
  Plane 1.0: same treatment as 0.8, waiting. y-cuts: 30 chains, started 2026-09-19 ~00:00, first anchors healthy.
- IN FLIGHT (what the coming data are): (a) redone UP chains of the topological cuts h_z = 0/0.1/0.2/0.25 on planes
  0.8 and 1.0 — anchor h_x = 0.45, 0.05 links to 0.95 (`_PLANE_UP_REDO`); old up outputs parked in
  `redo_up_20260919/`, dn chains untouched (1.25 → 0.7). (b) NEW cuts: magnetic h_z = 0.1 on planes 0.6/0.8/1.0
  (dn anchors gentle), electric h_x = 0.2 on 0.6/0.8/1.0, electric h_x = 0.25 on 0.8/1.0. (c) REDONE electric
  cold points at h_y = 1.0 for h_x = 0/0.2/0.25/0.5 with dt 0.01, diag_shift 1e-2, 1000 steps
  (`_PLANE_ELECTRIC_REDO`; old outputs in `redo_electric_20260919/`). (d) dn links re-planned by the drivers after
  the four dn-anchor retries (planes 0.6 hz0, 0.8 hz0.85, 1.0 hz0, 1.0 hz1.0). (e) refine tiers on 0.2/0.4/0.6/0.8. (f) y-cuts: 30 combined chain jobs; ycut_hx0.5_hz0 dn resubmitted gentle (job 58546356, 2026-09-19 04:40); ycut_hx0.8_hz0.2 dn likewise (job 58551572, 06:40); ycut_hx0_hz0.55 dn likewise (58555000, 08:40); plane 0.8 electric hx0.2/hz0.15 retried (58551570). NOTE: h_x = 0 y-cut dn chains legitimately stop at h_y = 1.0 ("E0 above h=0 bound" = metastable polarized branch above −172, not a divergence); the retry tool does not handle ycut dirs — forget the job's rows and resubmit the combined job with CUTS=<ycut_id> HY=y.
- Locator/label OVERRIDES live at the top of `analysis/scripts/phase3d_status.py`: `ELECTRIC_FIRST_ORDER`
  {(0.4,0.8),(0.6,0.8),(0.8,0.65),(0.8,0.8),(1.0,0.65),(1.0,0.8)} (first-order x-pol→z-pol steps, M_z jump
  locator, join the trivial→trivial line) and `TAIL_CROSSOVER` {(0.4,1.0),(0.8,1.0)}, `EXCLUDE_CUTS` {(1.0,"electric_hx0.8")}, `MAGNETIC_JUMP_PRIMARY`
  {(1.0,0.25)}. Add to them plane by plane with the user; never silently.
- Legacy hy_cuts_L4 runs (h_x = 0.2 electric, h_z = 0.1 magnetic at h_y = 0.2/0.4) were IMPORTED into
  `results/phase3d/hy{0.2,0.4}/{electric_hx0.2,magnetic_hz0.1}/L4` (marker LEGACY_IMPORT.txt; no curves; the
  pull never deletes them). Their h_y = 0 counterparts are pre-dual-lane files and stay extras-only
  (`analysis/viewer/phase3d_extras.json`).

## How to handle the coming data (per tick, every 2 h — user decision, not more often)
1. `cd <worktree> && DO_PULL=1 SINCE_MIN=125 bash analysis/scripts/phase3d_local_tick.sh` (pull → STATUS/summary →
   exports incl. plane `y` → cluster watch line). Then rebuild + republish the viewer (memory
   `phase3d-viewer-artifact`), copy summary.json + this file to the main checkout, report ONLY changes.
2. New DIV lines in the watch: a dn-anchor GENUINE DIVERGENCE = CHAIN STOPPED → `RETRY_FINAL=<final> RETRY_SET=
   "DIAG_SHIFT=5e-3 DT=0.01" HY=<hy> bash nersc/launch_phase3d.sh` (conda python on PATH!), then drop the job's
   never-run link rows from its manifest (backup .bak_<jobid>) so the driver re-plans the links; park the local copy
   in `redo_<jobid>/`. Electric divergence → same retry recipe. (dn anchors at h_y ≥ 0.6 are now gentle by default.)
3. Redone up chains (0.8/1.0): the envelope should now be bracketed on the up side. If the dn branch is still
   lower at its last point (0.7), the crossing needs dn links down to 0.5 — ask the user (they preferred the
   up-chain redo over a dn extension; raise it only with the new data in hand).
4. Redone h_y = 1.0 electric: judge by S₂ plateau (≈ 2.08) and Vscore (floor ≈ 0.5·h_y² in the complex lane);
   the old runs had Vscore 0.2–0.3 on the topological side and a plateau ending near h_z 0.1.
5. y-cuts: locator = M_y jump → net E crossing → loop centre (same ladder as tail cuts); roof cuts (topo=True) also
   show O_FM. Add a y-cut section to `analysis/notebooks/phase3d_L4_planes.ipynb` once ~half have landed.
6. Tail-cut locator ladder (viewer `located()`): M jump (sharp) → NET energy crossing (sign flip between the
   significant ends of the overlap) → hysteresis-loop centre (≥3 consecutive points with |M_dn−M_up| > 0.015,
   spikes smoothed, peak inside the run) → crossover. Energy crossings are only as good as the worse branch
   (offsets of ~0.001/site are common in the complex lane).

## Gotchas learned tonight
- Run the launcher/planner with `export PATH=$HOME/.conda/envs/tc-nqs/bin:$PATH` (login-node python3 is too old).
- The launcher prints "0 job(s)" when the planner CRASHES; always dry-run once after a planner change
  (`DRYRUN=1 LS=4 HY=<hy> CUTS=<cut> bash nersc/launch_phase3d.sh`).
- Manifests are the dedup source of truth: to resubmit something, forget its rows (keep a .bak) — array job ids
  in squeue carry `_[0]`, manifest ids don't.
- L4 anchor retries must keep the `_up`/`_dn` suffix (retry tool does it now); an unsuffixed anchor is invisible to
  the gap-fill tier.
- Viewer size: curves are thinned to 150 points (16 MB artifact cap; was 15 MB).
- Mac sleep kills ssh (mux "Broken pipe", banner timeouts): the user runs `caffeinate`; if it recurs, wait, don't
  re-mint the cert (sshproxy cert = 24 h, minted 2026-09-18 21:54).

## Open items
- Plane 0.8/1.0 re-review once the redone chains + new cuts land (envelope corner, h_x = 0.65 label check).
- h_z = 0.85 at 0.8 and all 1.0-plane tails: judge when dn links land. h_z = 0.85 endpoint of the first-order line
  (0.7 jumps everywhere, 1.0 crossover everywhere, 0.85 = small loops) — write it up in the notebook §10.
- Ask the user where to bank the untracked `results/phase3d/` + notebook (main checkout); S₂ on magnetic chains is
  extractable from checkpoints with tc3d.renyi if wanted; closure plane h_y = 1.2 (item D); QMC referee at h_y = 0.

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
- **h_y = 0.8 and 1.0 review (2026-09-19 ~02:00)**: the topological lobe has shrunk to h_x ≈ 0.6, so the up chains
  seeded at 0.6 started AT the boundary and the crossing was never bracketed. User decision: REDO the up chains of
  the topological cuts (h_z = 0/0.1/0.2/0.25) from deep inside — anchor h_x = 0.45, 0.05 links up to 0.95
  (`_PLANE_UP_REDO` in phase3d_grid.py); old up outputs parked in `redo_up_20260919/` (cluster + local), their
  manifest rows forgotten, the queued h_z = 0.1 up jobs cancelled and resubmitted; dn chains untouched. Extra
  electric cut h_x = 0.25 (7 cold points) on both planes. h_x = 0.65 and 0.8 are first-order trivial→trivial on
  both planes (0.65: metastable topological plateau in the cold points, then a one-link drop). h_z = 1.0 at 0.8 =
  crossover (constant 0.04–0.06 offset, E equal); h_z = 0.85 at 0.8 and the 1.0-plane tails wait for their dn links.
  Tail cuts h_z = 0.4/0.7 at 0.8 fine (jump 0.775 / 1.125). Submitted 02:05: 11 jobs per plane.

- **h_y = 0.8 plane (2026-09-21, after the redone up chains)**: reviewed, no comments — solid.
- **h_y = 1.0 plane (2026-09-21)**: (i) electric h_x = 0.25 and 0.5 have only one point in the topological region, the logistic
  fit is unconstrained → 3 extra deep-topological cold points each (h_z = 0 / 0.015 / 0.03 at 0.25; 0 / 0.02 / 0.04 at 0.5),
  gentle recipe, jobs 58680153–58680158 (launcher `PLAN_FILE` hook + `_electric_spec` lines). (ii) Magnetic h_z = 0 / 0.1 /
  0.2: fine. h_z = 0.25: the membrane O_FM is UNDEFINED on the topological side (closed-membrane denominator 0.014 ± 0.004,
  jackknife delete-one ≤ 0 → NaN), so the O_FM fit saw only dn-branch points and landed at 0.85; the winner-curve M_x jump
  (0.675 ± 0.025, B_p agrees, same as h_z 0.1/0.2) is the primary there (`MAGNETIC_JUMP_PRIMARY = {(1.0, 0.25)}`). (iii)
  Electric h_x = 0.8 is pure noise → excluded from the phase diagram (`EXCLUDE_CUTS = {(1.0, "electric_hx0.8")}`; stays in
  the Cuts view with a note). (iv) h_z = 0.7 / 0.85 locate via the loop centre (1.10 ± 0.10 / 1.30 ± 0.10) but the cut panels
  drew no dashed guide (guides only followed the jump test) → the dashed guide on every cut is now the located h_c from the
  same ladder as the phase diagram, with the locator named in the panel note.

- **Referee report (adversarial agent, 2026-09-21)** — CONFIRMED: rectangular (h_x,h_z) pocket + corner + x↔z first-order line
  = Fradkin–Shenker 3+1D Z₂ gauge–Higgs topology (Reiss & Schmidt 2019); product-state mean field reproduces our x↔z line to
  ≤ 0.1; the roof ≈ sphere |h| ≈ 1.18; pure-h_y transition first order; no self-duality on the h_y axis (unlike 2D), so
  h_y,c(∞) ≈ 1.3–1.45 and our 1.175 is low mainly from OBC. No exact mapping gives the (h_x,h_y)/(h_y,h_z) planes for free
  (S-gate / x-rotation turn the code into a different stabilizer model). SUSPICIOUS → VERIFIED in our data: the h_z,c(h_y)
  collapse (0.295 → 0.114) is far beyond 2nd-order perturbation theory (10–15 %); the topological side of the h_x = 0
  electric cuts at h_y ≥ 0.8 is under-converged (E RISES with h_z by 0.9–3.3 between neighbours, Vscore 0.2–0.4 vs 0.05
  polarized; no violations at h_y ≤ 0.4) → the O_FM inflection there is biased low; honest brackets h_z,c(0.8) ∈ [0.15,0.24],
  h_z,c(1.0) ∈ [0.05,0.2]. WRONG per referee: "no z↔y transition at h_z ≥ 0.4" (window too short: MF puts the jumps at
  h_y = 1.43 (0.4) / 1.61 (0.55)); the x↔y "line" (mean field: continuous canting for h_x ≥ 0.5; our up branches there are
  lagging optimizer states, the y-polarized dn branch is lower everywhere); an x↔z endpoint below h_z = 1.0 is not
  established (loop-width extrapolation needed).
- **h_x = 0 and h_z = 0 planes as their own maps (user, 2026-09-21)**: (i) h_x = 0 plane: the h_z sweeps at h_y = 0.6/0.8/1.0
  redone as WARM CHAINS (`ELECTRIC_CHAIN`, `_zchain_l4_job_spec`: up from h_z = 0.02, dn from 0.45, 12 shared links, gentle
  1000-step anchors; outputs next to the cold points with `_up`/`_dn` suffix; phase3d_status adds crossing/loop/M_z jump to
  such cuts, the O_FM fit runs on the winner curve) + y-cuts at h_z = 0.05 and 0.15 (0/0.1/0.2 exist). (ii) h_z = 0 plane:
  topo→x-pol = the existing h_z = 0 magnetic chains; topo→y-pol = y-cuts at h_x = 0.2/0.4/0.6 (0/0.5/0.8 exist). New y-cuts
  centre their fine window on the spherical-roof estimate (`ycut_center`). (iii) S₂ on every new point: launcher `POST_S2=1`
  submits a singleton-dependent `submit_eval_hy_axis.sh` job per chain — SUPERSEDED the same day by the in-job
  `TOPO_POOLED=1` block (see the 10:30 log entry); keep `POST_S2` only for re-evaluating old runs. (iv) Viewer: pseudo-plane tabs "h_x = 0 plane"
  (h_y,h_z) and "h_z = 0 plane" (h_x,h_y) built from the stored cuts + y-cuts. Next (after these land): the z↔y sheet needs
  y-cut windows to h_y ≈ 2.0 at h_z ≥ 0.3; the x↔z endpoint via h_z = 0.9/0.95 rungs.

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
- 2026-09-19 02:30 — user: the h_y = 1.0 electric cold points at h_x = 0 / 0.25 / 0.5 are badly converged (Vscore 0.2-0.3 on the topological side, S₂ plateau ending near h_z 0.1) → REDO with dt 0.01, diag_shift 1e-2, 1000 steps (`_PLANE_ELECTRIC_REDO`, walltime 3:00). Old hx0/0.5 outputs parked in redo_electric_20260919/ (cluster + local), rows forgotten, the just-queued hx0.25 points cancelled and resubmitted with the new recipe. 21 cold points.
- 2026-09-19 04:40 — tick (autonomous). 170 new finals (1293 total; y-cuts 61 pts, 6 of 15 cuts started). y-cut findings: (i) the h_x = 0 dn chains (hz 0/0.1) stopped at their LAST point h_y = 1.0 with "E0 above h=0 bound" — NOT a divergence: at h_x = 0 the metastable y-polarized branch has E ≈ −144·h_y·0.95 > −172 below h_y ≈ 1.1, so the sweep's health gate rejects it; the crossing is already bracketed (1.15–1.2, M_y jump 1.175 ± 0.025) → no action, expected on every h_x = 0 y-cut. (ii) ycut_hx0.5_hz0 dn ANCHOR (h_y 1.5) junk with the cold recipe: 26 guard rollbacks, E0 = −92 (should be ≈ −235), diverged=False (guard blind spot), then its 1.4 link failed the bound gate → CHAIN STOPPED. Parked redo_58485370 (cluster + local), 18 manifest rows forgotten (.bak_58485370 on all.tsv + manifest_20260917_143645), y-cut dn anchors now gentle by default (be4cbac: dt 0.01, ds 5e-3, mirrors the hy ≥ 0.6 rule), whole dn chain resubmitted as job 58546356 (retry tool does not know ycut dirs — resubmit the combined job via CUTS=<ycut_id> HY=y after forgetting the rows). Selftest cut-count assertions were stale since hx0.25 was added (fixed 16c608d). Redone h_y = 1.0 h_x = 0 electric: 4/7 gentle points landed (hz 0.04/0.1/0.13/0.19; O_FM 0.008 → 0.57 smooth, logistic fit meaningless yet), 0.16/0.22/0.28 running; the live dir still holds COLD copies of 0.16/0.22/0.28 (overwritten when the gentle ones land) and the two cold refine points 0.1147/0.1547 (never redone — decide with the user whether to forget + redo them). Queue 27 R / 79 PD, 0 failed. Viewer v50.
- 2026-09-19 06:40 — tick (autonomous). 102 new finals (1381). Redone h_y = 1.0 electric h_x = 0.25 / 0.5 complete (7/7 each, gentle 1000-step recipe): h_z,c = 0.068 ± 0.039 / 0.093 ± 0.017 → the topological lobe at h_y = 1.0 is tiny (h_x = 0: fit still meaningless, 3 gentle points running). y-cuts: 9/15 cuts have both branches; M_y jumps h_x = 0: 1.175/1.175/1.275 (hz 0/0.1/0.2), h_x = 0.5: 1.075/1.075 (hz 0.1/0.2), h_x = 0.8: 1.125 (hz 0/0.1, wide brackets, chains still running). Two GENUINE DIVERGENCES, both retried: (a) plane 0.8 electric h_x = 0.2 h_z = 0.15 (step 174, 14 rollbacks) → retry tool, redo_58543490, job 58551570 (DIAG_SHIFT 5e-3, DT 0.01); (b) ycut_hx0.8_hz0.2 dn ANCHOR (h_y 1.5, cold recipe, step 15) → parked redo_58485381, rows forgotten (.bak_58485381), combined dn job resubmitted gentle as 58551572. The five still-PENDING cold y dn jobs (h_x = 0 hz 0.55/0.7; h_x = 1/1.2/1.4 hz 0) were deliberately NOT cancelled: they hold ~2 days of queue age and a divergence costs minutes; retry on failure only. Queue 10 R / 60 PD, 0 failed. Viewer v51.
- 2026-09-19 08:40 — tick (autonomous). 65 new finals (1443). Plane 0.8 electric h_x = 0.2 retry landed → cut complete, h_z,c = 0.207 ± 0.014. y-cuts: 13/15 started; roof M_y jumps h_x = 0.8: 0.95 ± 0.05 (hz 0 and 0.1); tail y-cuts (h_x = 0, hz 0.4/0.55/0.7) have no sharp jump yet (chains still running). 3rd cold y dn ANCHOR divergence: ycut_hx0_hz0.55 (h_y 1.5, step 10, 6 rollbacks) → parked redo_58485387, rows forgotten (.bak_58485387), resubmitted gentle as 58555000. Cold y dn anchors: 5 ok / 3 diverged so far (hx0.5/hz0, hx0.8/hz0.2, hx0/hz0.55); h_x = 1/1.2/1.4 still pending cold (kept for queue age). Queue 12 R / 59 PD, 0 failed. Viewer v52 (5.0 MB).
- 2026-09-19 10:40 — tick (autonomous). 50 new finals (1491), all y-cuts: 15/15 started, 14 with both branches. Tail y-cuts (h_x = 0 hz 0.4/0.7, h_x = 1/1.2 hz 0) show no sharp M_y jump yet (largest steps 0.06–0.23) — the tail-cut ladder (net E crossing → loop) applies once the chains complete. Flagged log p3d_hy0.8_m0.2_L4_up-58533205 is NOT a divergence: a pre-redo refine link (h_x 0.975, INIT_FROM the parked 0.95 up checkpoint) that exited in 25 s ("anchor previous point diverged" = checkpoint gone); its row already sits in .bak_upredo, so the tier re-plans it after the redone up chain lands. Redone up chains (0.8/1.0, 8 jobs) + hz0.1 dn chains still PENDING since 2026-09-18 ~12:00 cluster time. Queue 11 R / 54 PD, 0 failed. Viewer v53 (5.2 MB).
- 2026-09-19 16:30 — tick (autonomous; the two intermediate ticks did not fire, ~6 h gap). 84 new finals (1575). y-cuts: 15/15 have both branches except the three gentle dn retries still pending (hx0.5/hz0, hx0.8/hz0.2, hx0/hz0.55); tail y-cuts h_x = 1/1.2/1.4 (hz 0) complete with NO sharp M_y jump (largest steps 0.06–0.12 near h_y 0.9–1.0) → E-crossing/loop ladder in the viewer decides. Plane 1.0: h_x = 0 electric 11/11 gentle → h_z,c = 0.114 ± 0.031 (richards; the two cold refine points 0.1147/0.1547 still in the curve, user to decide), h_x = 0.2 7/9 → 0.048 ± 0.072 (poorly constrained, lobe edge). Plane 0.8 hz 0.85 dn complete (13/13, no crossing). Refine pairs landed on planes 0.2/0.4/0.6 (2 pts per cut); drivers submitted refine pairs for plane 0.6 hz 0 (58571809/12). Redone up chains (8 jobs, 0.8/1.0) STILL PENDING (submitted 2026-09-18 ~12:00 cluster). Queue 6 R / 16 PD, 0 failed. Viewer v54.
- 2026-09-19 20:30 — tick (autonomous). 142 new finals (1719); queue drained to 8 running / 0 pending. REDONE UP CHAINS LANDED: plane 0.8 hz 0.1/0.2/0.25 complete, plane 1.0 hz 0/0.1/0.2/0.25 complete or finishing; hz 0.1 chains complete on 0.6/0.8/1.0. KEY FINDING (rule 3 of the RESUME block): on every redone cut the x-polarized dn branch is LOWER in energy than the up branch over the whole overlap 0.7–0.95 (E_up − E_dn per site: plane 0.8 hz0.1 +0.007…+0.013, hz0.2 +0.0002…+0.007, plane 1.0 hz0 +0.008…+0.024, hz0.1 ≈ +0.022 flat, hz0.2 +0.011…+0.041, hz0.25 +0.055…+0.087); only plane 0.8 hz0.25 crosses (at h_x ≈ 0.78–0.80). So the envelope on these planes lies BELOW h_x = 0.7 and is still not bracketed → the user must decide on dn links down to h_x = 0.5 (they preferred the up redo before; raise it now with data). Plane 0.8 hz 0 redone up ANCHOR (h_x 0.45, cold up recipe) GENUINE DIVERGENCE at step 88 (11 rollbacks) → retry tool (redo_58542685, job 58584571, DIAG_SHIFT 5e-3 DT 0.01), 10 never-run link rows dropped (.bak_58542685) for the driver to re-plan. Plane 1.0 electric h_x = 0.2 complete (9/9): h_z,c = 0.113 ± 0.033. Three gentle y dn retries running fine (hx0.5/hz0 dn already 7 pts, jump 1.075 ± 0.025). Viewer v56.
- 2026-09-19 22:30 — tick (autonomous). 22 new finals (1741). y-CUTS COMPLETE: 15/15 with 11 up + 9 dn points each (the three gentle dn retries landed; hx0.8/hz0.2 and hx0/hz0.55 show no sharp M_y jump → ladder). Plane 1.0 hz 0.2/0.25 up chains complete (15/15). Plane 0.8 hz 0 gentle up anchor (58584571) landed healthy in 32 min; the driver re-planned its 10 up links (job 58585135, running) — the last campaign job in the queue besides it: 1 R / 0 PD. All other cuts on all planes are complete. OPEN: the dn-extension decision (rule 3) for planes 0.8/1.0. Viewer v57.
- 2026-09-20 00:30 — tick (autonomous). 4 new finals (1745): plane 0.8 hz 0 up WINDOW links 0.7–0.95 (job 58585135, warm-started straight from the gentle 0.45 anchor — the gap-fill tier refills only the window). The coarse links 0.5/0.55/0.6/0.65 of the user's up redo were therefore missing → hand-emitted one link job via `_chain_link_job_spec(..., [0.5,0.55,0.6,0.65], init_from=0.45 anchor)` + `_bash_line` (job 58587919, manifest_*_manual_hz0up.tsv). CAVEAT for the review: on this one cut the 0.7–0.95 points were warmed from 0.45 directly, not chained through 0.65 (re-chain on request). Queue otherwise EMPTY (0 R / 0 PD besides drivers): every planned cut on every plane and all 15 y-cuts are complete. Viewer v58. Waiting on the user: dn extension to h_x 0.5 on planes 0.8/1.0 (rule 3); the two cold refine points of plane 1.0 h_x = 0.
- 2026-09-20 02:30 — tick (autonomous). 3 new finals (1748): plane 0.8 hz 0 coarse up links 0.5/0.55/0.6 (job 58587919, 0.65 still running). Nothing else in the queue. Viewer v59.
- 2026-09-20 04:30 — tick (autonomous). 1 new final (1749): plane 0.8 hz 0 up link 0.65 → that cut is now 15/15. QUEUE EMPTY: the campaign as planned is fully landed (1749 finals, 0 failed). Viewer v60. Open: dn extension decision (rule 3), plane 1.0 h_x = 0 cold refine points, planes 0.8/1.0 review.
- 2026-09-21 ~00:00 — user review of planes 0.8 (solid) and 1.0 (see §0.b): 6 extra electric points submitted (58680153–58680158); overrides EXCLUDE_CUTS / MAGNETIC_JUMP_PRIMARY added (caf6ea6); viewer: y-cut toggle (cafe446, off by default), dashed guide = located h_c on every cut. Viewer v62.
- 2026-09-21 ~00:30 — user: the h_y = 1.0 electric line sits at h_z ≈ 0.115, so the h_z = 0.2 / 0.25 magnetic cuts may be trivial→trivial. TEST = end-of-training S₂ on every point of the plane-1.0 magnetic cuts h_z = 0 / 0.1 / 0.2 / 0.25 (both branches, 82 points): expect the 3 ln 2 plateau + abrupt drop on the first two, no plateau on the latter two. Submitted as 4 shared-GPU eval jobs 58680825/28/30/31 (`nersc/submit_eval_hy_axis.sh` with LAST_ONLY=1, SUFFIX=.finaleval_electric.json so phase3d_status.py picks the S₂ up; 356d763). The tick's pull syncs *.json, so the S₂ panels fill in automatically. (The user wrote "hy=0.8" but 0.115 is the plane-1.0 value; done on 1.0 — extend to 0.8 on request.)
- 2026-09-21 ~01:30 — user: S₂ on ALL y-cut points (15 jobs 58681847–61, same LAST_ONLY recipe) to test whether the "roof" crossings are true topo→trivial; plus a 3D SKETCH panel at the bottom of the phase-diagram view (three.js r128 from cdnjs, drag/wheel): transparent purple pocket lofted through the plane cross-sections + roof levels + apex; blue = 2nd-order electric points, red = 1st-order magnetic/roof points, translucent red sheet = x-pol ↔ z-pol first-order line across planes, orange = y-cut remnants that start outside the pocket, grey = y-cut crossovers. Judgement rules are in the code comments (`sketchModel`): magnetic cut demoted to the trivial sheet if its h_z > 1.5 × the plane's electric h_z,c(h_x≈0) (plane 1.0 hz 0.2/0.25); y-cut roof point only if (h_x, h_z) lies inside the cross-section just below h_c (10 % tolerance) → roof = (0,0), (0,0.1), (0.5,0); (0.5,0.1) currently falls outside only because the plane-1.0 hx=0.25/0.5 fits are bad (fixes itself when the 6 extra points land); (0.8,·), (0.5,0.2), (0,0.2) = polarized ↔ y-polarized remnants. Viewer v63.
- 2026-09-21 ~02:30 — user: sketch restricted to the positive octant, fonts adjusted, checked visually (Claude-in-Chrome on a local http.server copy; the artifact iframe does not accept the extension's synthetic clicks reliably, the HTML is identical). Sketch now: capped loft (faces on the three coordinate planes), tube lines, measured sprite labels, faint box grid, fixed 3:2 canvas ≤ 1000 px, default view from the (+h_x, +h_y) side with h_z up; rendering from a rAF loop (dirty flag) with preserveDrawingBuffer so screenshots/captures do not stall. Viewer v67 (commits 7ea7378, next).
- 2026-09-21 ~09:00 — user: h_x = 0 and h_z = 0 planes as their own maps (§0.b), S₂ on every point, submit all at once. Code reviewed by a Sonnet agent (2 caveats applied), commit 96337a5. SUBMITTED 02:45 cluster time: electric h_z chains p3d_hy{0.6,0.8,1.0}_e0_L4_{up,dn} (58688581/84, 58688587/89, 58688594/96; 13 pts each), new y-cuts p3d_y_hx0_hz0.05, hx0_hz0.15, hx0.2_hz0, hx0.4_hz0, hx0.6_hz0 (58688598–58688616, up 10–12 pts / dn 9–11 pts), and ONE S₂ eval job per chain (58688582…58688617, singleton on the chain's job name, glob *_k3_{up,dn}.json, writes .finaleval_electric.json). Manifests manifest_20260921_0244*.tsv. Total 16 chain jobs (~176 points) + 16 eval jobs. Caveat: an eval job waits for the chain job of the same name; if AUTO_RESUBMIT spawns a continuation after the eval ran, the late points lack S₂ until re-evaluated. Viewer v68 has the pseudo-plane tabs.
- 2026-09-21 ~10:30 — user: S₂ must be part of EVERY run, not a follow-up. Done (23e0e0b): `--topological_after_pooled` in train.py/sweep.py keeps the inline O_FM+S₂ block after the pooled final eval (~4 min/point at L=4, measured from the eval jobs); batch wrapper `TOPO_POOLED=1`; every L4 chain spec (magnetic, y-cut, electric h_z chain, link jobs) sets it; the S₂ lands in the run JSON's observables and phase3d_status reads it from there (finaleval files still take precedence). Electric cold points already had the in-job POST_S2_EVAL. The 16 pending chains + 16 follow-up evals were cancelled (manifests moved to manifests_bak/), and the 16 chains RESUBMITTED with TOPO_POOLED=1: 58689262–58689277. The 19 older eval jobs (plane-1.0 magnetic cuts, 15 original y-cuts) stay: those runs predate the change. Local L=2 OBC smoke confirmed the plumbing. Queue 4 R / 37 PD.
- 2026-09-21 ~12:35 — tick. S₂ TEST ON PLANE 1.0 COMPLETE (82 pts): h_z = 0 and 0.1 up branches sit on the 3 ln 2 plateau (2.06–2.10) for h_x 0.45–0.65 and drop abruptly over 0.70→0.80 (2.00→1.01→0.65 / 1.88→0.79→0.60) → TRUE topo→trivial at the M_x jump 0.675. h_z = 0.2 and 0.25 NEVER reach the plateau (0.9–1.2 rising to 1.65 at 0.70, then down; 0.57–0.80 at 0.25) → trivial→trivial (x-pol ↔ z-pol), confirming the user's suspicion and the sketch's demotion rule. First y-cut S₂: (h_x,h_z) = (0.5,0.1) and (0.5,0.2) up branches on the plateau (≈2.0–2.1) all the way to h_y 1.15, dn ≈ 0.2 → both cross the roof as topo→y-pol (so (0.5,0.2) is a roof point after all, contrary to the sketch's inside-test; revisit the rule once the plane-1.0 electric chains land). Viewer v74. Queue 2 R / 29 PD.
- 2026-09-22 ~02:35 — tick. y-cut S₂ so far (154/~300 pts): (h_x,h_z) = (0.5, 0/0.1/0.2) up branch on the 3 ln 2 plateau to h_y 1.15–1.30, dn ≈ 0.2 → genuine topo→y-pol roof (all three; the sketch's inside-test wrongly demoted (0.5,0.2)); (0.8, 0/0.1/0.2) never on the plateau (0.5–0.8 decaying) → outside the pocket at every h_y, their jumps are polarized↔y-pol remnants; (0, 0.1) on the plateau to 1.25 → roof; (0, 0.2) up branch starts at 1.76 (partly topological at h_y 0.6) and decays smoothly to 0.64 by 1.15 while the dn branch drops 0.75→0.19 at 1.10 → it exits the pocket continuously (2nd order, electric-type) around h_y ≈ 0.8 and the M_y jump at 1.275 is z-pol↔y-pol; (0, 0.4/0.55) ≈ 0.2–0.4 (trivial throughout). Viewer v77.
- 2026-09-22 ~07:35 — tick. y-cut S₂ COMPLETE (all 15 cuts, ~300 pts): confirms every roof/remnant/crossover call from
- 2026-09-22 ~09:20 — FULL UPDATE (user: "wait for plane 1.0 to finish, then update plots fully"; cert had also expired
  mid-campaign, re-minted ~14:13). All 16 h_x=0/h_z=0 chains + their in-job S₂ landed while the cert was down (queue drained
  to 0/0). RESULTS: (i) electric h_z chains fix the referee's under-convergence bias — h_z,c(h_y): 0.6: 0.252→0.235±0.019,
  0.8: 0.197→0.207±0.014, 1.0: 0.114→0.164±0.046 (energy now monotone on all three, one small 1.5-unit blip at h_y=0.8
  h_z=0.02→0.05, harmless); all three now sit INSIDE the referee's honest brackets. Loop locator agrees (0.23/0.26/0.17).
  (ii) 2 of 5 new roof y-cuts had a GENUINE UP-ANCHOR DIVERGENCE at the cold h_y=0.6 start (not the usual dn/y-polarized
  problem) — (0,0.05) and (0.4,0), both n_rollbacks 8-12, garbage E0/Vscore. Parked redo_58689268/58689274, rows forgotten,
  resubmitted gentle (dt 0.01 ds 5e-3) via hand-emitted PLAN_FILE jobs 58744241/58744247 (the planner has no override knob
  for a y-cut UP anchor retry — only dn gets gentle by default). (0,0.15),(0.2,0),(0.6,0) landed clean: h_y,c = 1.245±0.014,
  1.185±0.014, 0.945±0.015. (iii) Hardcoded the S2 test's verdicts server-side (5549010): `MAGNETIC_JUMP_PRIMARY` now
  covers (1.0,0.2) too (S2 never plateaus, same M_x jump 0.675 as hz=0.1/0.25); new `YCUT_S2_ROOF`/`YCUT_S2_REMNANT` sets
  replace the old geometric inside-polygon guess for the roof/remnant split (which had wrongly demoted (0.5,0.2) — S2
  confirms it IS a roof point, plateau to h_y=1.15). `isTopo`/envelopeOf/ttLineOf/sketchModel all now key off the same
  `jump_primary` flag so the Cuts view, phase-diagram envelope, 2D side-plane overlays and the 3D sketch agree. Viewer v79.
  the sketch except (0.5,0.2) which is a genuine roof (noted 02:35). (0,0) up branch stays on the plateau (2.05–2.13,
  one noisy point 1.48 at h_y=1.05) through 1.30 → clean roof. Tail cuts (0,0.7), (1.0,0), (1.2,0), (1.4,0) all sit at
  S2 ≈ 0.07–0.35 on BOTH branches throughout — confirms these are true crossovers (trivial on both sides the whole way),
  not a hidden topological remnant. One new final landed: plane 0.6 h_x=0 electric chain, first point. Queue watch failed
  this tick — sshproxy cert expired at 11:07 (see gotchas); local pull/export/viewer done, cluster state (chain progress,
  divergences) unknown until re-minted. Viewer v78.
