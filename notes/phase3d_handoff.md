# phase3d campaign — session handoff (written 2026-09-11 12:45 +05)

Read this first, then the RESUME block at the top of
`~/.claude/plans/hello-claude-how-are-floating-blossom.md` (event log lives below it) and the
memory file `phase3d-campaign-plan.md`. You are the ORCHESTRATOR of a running, self-driving
cluster campaign: watch, fix, report concisely. Do not implement big things yourself; small
fixes are fine. The user is a physicist, wants short updates that lead with what changed.

## 0. Hard rules (do not relax)

- Never run 3D toric-code ED/sweeps locally (8 GB Mac). L=2 OBC only.
- Cluster autonomy: monitor, `sbatch`, `scancel`/`scontrol hold|release|update` ONLY our own jobs
  (user `sanzharb`). gpu_debug smokes are autonomous; production runs are per-stage approval only.
- APPROVED so far: Stage 1 = h_y=0 plane, L=4/5/6, all 8 cuts (running); Stage 2 = h_y=0.2 and 0.4
  at L=4 ONLY (queued, first points landed). NOT approved: L5/L6 at h_y≠0, planes h_y=0.6/0.8
  (needs "drop LS=4 from the scrontab lines" after the user says go). QMC referee at h_y=0 approved
  (ParaToric β=24 ×8, `--validate` first) once new cuts have L4 h_c — not started.
- Never propose walltime > 5 h. Never commit to `main`. Never `git add analysis` wholesale
  (the peer's untracked `analysis/scripts/transition_fit.py` and `firstorder_fit.py` live there).
- Never install/download on the user's behalf. Cert re-mint is the user's: `! sshproxy -u sanzharb`
  (MFA); the cert lasts 24 h — current one expires 2026-09-12 12:16 (+05). "Too many authentication
  failures" = expired cert.
- Main checkout `/Users/sanzhar123/Desktop/toric-code-nqs` is the PEER's dirty tree; ours there are only
  `results/phase3d/` and `data/tc_nqs/phase3d/`. Research clones must not touch `~/toric-code-nqs`
  on the cluster or `$PSCRATCH/tc_nqs/phase3d`.

## 1. Where everything is

Local
- Worktree with the campaign code: `/Users/sanzhar123/Desktop/toric-code-nqs-p3d/integration`,
  branch `feat/phase3d-campaign` (origin). `.venv` symlinked; run tools as `.venv/bin/python`.
  Tests need `PYTHONPATH=<worktree>`.
- Mirror of cluster outputs: `results/phase3d/` (finals, STATUS.md, viewer.html, watch_state.json,
  manifests/) and `data/tc_nqs/phase3d/` (per-step `.curve.json`, gitignored) under the main checkout.
  rsync never deletes: after parking a run on the cluster, delete its local copies by hand.
- Viewer artifact (drill-down: plane → cut → point → learning curve):
  https://claude.ai/code/artifact/eb8e3881-84e0-4c28-bf03-6827dfc5a554 — republish to THIS url
  from `/Users/sanzhar123/.claude/jobs/7546b659/tmp/viewer/phase3d_viewer.html` (built by the
  pull loop, see §2); omit favicon on redeploys. In a new session, first `Artifact action=read url=…`
  once, then publish with `url=`.
- Program docs: `notes/phase3d_campaign.md` (design), `notes/speed_levers.md`, `notes/l10_feasibility.md`,
  `notes/transition_mapping_recipes.md` (playbook), this file.

Cluster (`ssh perlmutter`; user sanzharb; account m5340_g; QOS gpu_shared, 1 GPU per job)
- Code clone `~/toric-code-nqs` on `feat/phase3d-campaign` (keep it at origin HEAD: `git pull -q origin
  feat/phase3d-campaign`). Drivers run from there with `PATH=$HOME/.conda/envs/tc-nqs/bin:$PATH` and
  `PYTHONPATH=$HOME/toric-code-nqs` (system python3 is too old for the planner).
- Outputs `$PSCRATCH/tc_nqs/phase3d/` = `/pscratch/sd/s/sanzharb/tc_nqs/phase3d/`:
  `hy{0.0,0.2,0.4}/{electric_hx*,magnetic_hz*}/L{4,5,6}/<run>.json` (final), `.curve.json`,
  `.mpack`/`.ckpt.mpack`/`.stepNN.mpack`, `.finaleval_electric.json`, `.snapshots.json`, `wandb/`;
  parked bad runs in `<L dir>/redo_<oldjid>/` (and one `diverged_58114578/`).
  `manifests/manifest_*.tsv` (columns jobid hy cut L role h name out_dir submitted_at; one row per
  point; the planner dedupes against them), `manifests/all.tsv` (union, rebuilt by the driver),
  `manifests_bak/` (originals of edited manifests), `watch_state.json`, `driver_hy{0.0,0.2,0.4}.log`,
  `defaults.env` (INV_IMPL=dense COMPUTE_DTYPE=float32 — the speed levers, applied on a job's first start).
- Slurm logs: single-point jobs `~/toric-code-nqs/p3d_hy0.0_<cut>_L5-<jid>.out`; array/chain jobs
  `~/toric-code-nqs/slurm_logs/p3d_hy0.0_m0.2_L5_up-<jid>_0.out`. `scontrol show job <jid> | grep StdOut`.
- W&B project `tc3d-phase3d` (offline runs, synced by the `p3d-wandb-sync` scrontab job every 30 min).

Run naming: `gridinv_dual_L{L}_OBC_hx{hx}_hz{hz}[_hy{hy}]_n2x4_nh4-8_inv8-8_k{L-1}[_up|_dn]`.
Electric cuts fix hx and sweep hz (`electric_hx0.8`); magnetic cuts fix hz and sweep hx
(`magnetic_hz0.2`), each with an `up` chain (from the topological anchor at low hx) and a `dn` chain
(from the polarized anchor at high hx); links are 200-step warm starts inside one array job.

Job names: `p3d_hy0.0_e0.8_L5` (electric cold point), `p3d_hy0.0_m0.2_L5_up` (chain anchor AND its
link train share the name; anchors are plain job ids, trains are arrays `<jid>_[0]`).

## 2. What runs without you

Cluster (`scrontab -l`): `p3d-wandb-sync` :00/:30; `p3d-driver-hy0.0` :15 (`HY=0.0 MAX_QUEUE=200`);
`p3d-driver-hy0.2` and `-hy0.4` :35 (`LS=4 HY=… MAX_QUEUE=200`). Driver = `nersc/phase3d_cron_driver.sh`
= idempotent launcher (`analysis/scripts/phase3d_grid.py plan` → `nersc/launch_phase3d.sh`, submits
whatever is launchable and not yet in a manifest, up to MAX_QUEUE non-dependency-held jobs) + watcher
(`nersc/watch_phase3d.sh` → `watch_state.json`, flags diverged / above-bound / warm_loaded=False).
Refinement rounds (≤2 electric inserts per cut, ≤1 chain insert per branch) are planned automatically
once a cut has enough landed points. L5/L6 link trains are submitted right after their anchor with
`--dependency=afterok:<anchor>,singleton`; the sweep re-checks the anchor JSON at start.

Local Monitors (session-bound; re-arm in a new session — they die with the old one):
1. Pull loop (2-hourly, clock-based so it survives Mac sleep; macOS has NO `timeout`): pulls with
   `analysis/scripts/pull_phase3d.sh`, writes STATUS.md, exports the three planes, builds the viewer,
   prints one `PULL hh:mm: finals=N …` line. After each PULL line: republish the artifact, report
   only what changed. Command (run from the worktree):
   ```
   cd /Users/sanzhar123/Desktop/toric-code-nqs-p3d/integration
   R=/Users/sanzhar123/Desktop/toric-code-nqs/results/phase3d; DATA=/Users/sanzhar123/Desktop/toric-code-nqs/data/tc_nqs/phase3d; V=/Users/sanzhar123/.claude/jobs/<JOB>/tmp/viewer; mkdir -p $V; last=0
   while true; do now=$(date +%s); if [ $((now-last)) -ge 7200 ]; then last=$now
     LOCAL_RESULTS=$R LOCAL_DATA=$DATA bash analysis/scripts/pull_phase3d.sh >/dev/null 2>&1 || echo "PULL FAILED $(date +%H:%M) (ssh/cert?)"
     n=$(find $R -name '*.json' ! -name '*snapshots*' ! -name 'watch_state*' ! -name '*finaleval*' | wc -l | tr -d ' ')
     .venv/bin/python analysis/scripts/phase3d_status.py --root $R --out $R/STATUS.md >/dev/null 2>&1
     rm -f $V/viewer_hy*.json; planes=""
     for hy in 0.0 0.2 0.4; do [ -d $R/hy$hy ] || continue; .venv/bin/python analysis/scripts/phase3d_status.py --export-viewer $hy --root $R --curves-root $DATA --out $V/viewer_hy$hy.json >/dev/null 2>&1 && planes="$planes $hy"; done
     .venv/bin/python analysis/scripts/phase3d_viewer_build.py $V/phase3d_viewer.html $V/viewer_hy*.json >/dev/null 2>&1 && cp $V/phase3d_viewer.html $R/viewer.html
     echo "PULL $(date +%H:%M): finals=$n planes=[$planes ] viewer rebuilt -> $V/phase3d_viewer.html"; fi; sleep 600; done
   ```
2. Cluster watch (15-min): new `GENUINE DIVERGENCE` / `CHAIN STOPPED` lines in `~/toric-code-nqs/*.out`
   and `slurm_logs/*.out`, failed jobs (`sacct … -s F,TO,OOM,CA`), the drivers' last `[launch] N job(s)`
   + `[watch] flagged: K` lines, and `squeue` counts; print only changes.
3. Fill-up ping (user request): fire ONCE when ≥8 of our jobs run or ≥4 start within 5 min, then send
   PushNotification + a chat line. Script: `$CLAUDE_JOB_DIR/tmp/fillup_watch.sh` (5-min poll).
   It fired the equivalent overnight (11 running at 04:00); re-arm only if the user asks again.

## 3. Daily loop (commands that work)

```
ssh perlmutter 'squeue -u sanzharb -h -o "%T" | sort | uniq -c; squeue -u sanzharb -h -t R -o "%i %j %M %l"'
ssh perlmutter 'tail -n 30 $PSCRATCH/tc_nqs/phase3d/driver_hy0.0.log'        # [launch]/[watch] summary
ssh perlmutter 'sinfo -p gpu_ss11 -h -o "%D %T" | sort -k2'                    # 1664 nodes total
ssh perlmutter 'sprio -u sanzharb -o "%i %Y %A" | sort -k2 -rn | head -3'      # who accrues age
# health of finals landed in the last N minutes (E, Vscore, rollbacks, diverged, hours):
ssh perlmutter 'cd $PSCRATCH/tc_nqs/phase3d; for f in $(find hy0.0 hy0.2 hy0.4 -name "gridinv_dual_*.json" -mmin -600 | grep -v "snapshots\|finaleval\|curve\|wandb\|redo_"); do python3 -c "
import json,sys; d=json.load(open(sys.argv[1])); o=d[\"observables\"]; c=d[\"curve\"]
print(sys.argv[1][:60].ljust(60),\"E %9.2f\"%c[\"energy\"][-1],\"Vs %.1e\"%o[\"Vscore\"],\"rb\",d[\"n_rollbacks\"],\"DIV\" if d[\"diverged\"] else \"ok\",\"%.2fh\"%(d[\"runtime_s\"]/3600))" "$f"; done | sort'
```
Final JSON: `config` (L hx hy hz dt diag_shift n_iter compute_dtype inv_impl qgt_solver), `observables`
(Vscore, E_err, E_var, A_v_mean, B_p_mean, O_FM_paratoric[_den], O_FM_membrane_R1[_den], …), `curve`
(dict of lists: step energy energy_err energy_spread delta energy_im timing{total,…}), `n_rollbacks`,
`diverged`, `runtime_s`. Vscore = N·Var(E)/E² with N = number of edges (L5 OBC: 300, L6: 540).
h=0 stabilizer bound (E must be below): L4 −172, L5 −365, L6 −666.

Trust criteria (user): E below the bound + a smooth curve by eye; anchors Vscore ≲1e-2 (up anchors land
at 2–7e-3, polarized dn anchors are worse: 1–4e-2 at hz≥0.7 is the ansatz floor); 200-step links
Vscore 1e-3…4e-2; electric cold points 2e-4…6e-3 (near-transition at hx=0.8: 1–3e-2); h_y≠0 points
Vscore ≈ 0.5·h_y² on top (h_y=0.2 → ~2e-2, confirmed).

Recipes: cold points dt 0.02, lr_min 0.002, 500 steps, diag_shift 1e-3 (L4) / 3e-3 (L≥5); L≥5 dn
anchors GENTLE: dt 0.01, diag_shift 1e-2, 600 steps (3 of 4 polarized anchors under dt 0.02 failed);
links dt 0.005, lr_min 5e-4, diag_shift 3e-3, 200 steps. Electric GENUINE DIVERGENCE near the
transition at L≥5 → retry with DIAG_SHIFT=5e-3 (see §5.1).

Speed (fast path = dense conv + float32 forward, double QGT twin; exact to 1e-15 in logψ): s/step
L4 real ~1.5, L4 complex 3.2, L5 real 6.2–6.6, L6 real 21.9; L6 complex needs QGT_SOLVER=kernel
(planner sets it). Measured walls: L5 electric point 1:35–1:46 (54 min train + ~50 min POST_S2_EVAL
pass, electric only), L5 anchor 1:11, L5 8-link train 1:57 (!), L6 cold point 3.0–3.1 h, L6 anchor
3:05–3:50, L4 complex cold point 26 min.

Walltimes (planner `walltime_for`): L4 h_y=0 1:30, L4 h_y≠0 2:00, L5 electric 2:30, L5 chains/anchors
3:00 (was 2:00 — the ~20 already-queued L5 jobs still carry 2:00 and will auto-resubmit if they overrun:
the wrapper's USR1 trap 3 min before the limit sbatches a continuation, which then waits in the queue
again), L6 5:00. Owners may LOWER a queued job's limit (`scontrol update JobId=… TimeLimit=…`), never raise.

## 4. Queue physics (why nothing runs, 2026-09-10/11)

- Perlmutter GPU is saturated by other users' full-node jobs (1500–1630 of 1664 nodes allocated;
  the shared lane we use gets ~45–55 running slots vs ~3400 pending shared jobs).
- Priority = QOS base 67679 + age only (fairshare weight 0). gpu_shared has MaxJobsAccruePerUser=2:
  only our 2 OLDEST pending job ids accrue age; the other ~100 sit at zero age. Our starts come from
  backfill of short jobs (hence the shorter walltimes) and from the two accruers reaching the top.
  Bursts happen (03:30–10:00 on 09-11 ran ~40 jobs), then nothing for hours.
- MAX_QUEUE (our launcher knob, not a NERSC limit; NERSC allows 5000 submitted) = 200 now.
- NERSC FULL MAINTENANCE 2026-09-16 06:00 → 09-23 06:00 PDT. Nothing runs that week; Slurm won't
  start a job that would run into it (short jobs backfill best in the final hours).
- Option offered to the user, undecided: premium QOS (priority 73440 beats every aged job; max 5
  submitted per user; NERSC bills ~2×) for the 62 h_y=0.2/0.4 L4 jobs via a 5-slot feeder.

## 5. Open problems and the exact fix procedures

5.1 Retry a single point with different knobs (electric divergence, bad anchor) — the tool:
```
ssh perlmutter
export PATH="$HOME/.conda/envs/tc-nqs/bin:$PATH"; export PYTHONPATH="$HOME/toric-code-nqs"; cd ~/toric-code-nqs
RETRY_FINAL=$PSCRATCH/tc_nqs/phase3d/hy0.0/electric_hx0.0/L5/<run>.json RETRY_SET="DIAG_SHIFT=5e-3" HY=0.0 bash nersc/launch_phase3d.sh
```
It parks `<run>.*` into `<L dir>/redo_<oldjid>/`, drops the point's manifest rows (backup in
`manifests_bak/`), rebuilds the planner's own spec (gentle recipe for L≥5 dn anchors, fast path, current
walltime), applies the overrides, sbatches, logs a manifest row. Then delete the local mirror copies of
that run (results/ and data/). For a dn anchor redo with the now-default gentle recipe: `RETRY_SET=""`.
NOT for chain links (it refuses) — see 5.2.

5.2 Re-emit a chain after a link diverged / CHAIN STOPPED (done once for hz0.2 L5 up):
park the bad link's files (`mv <L dir>/<link>_up.* <L dir>/redo_<jid>/`), drop the manifest rows of that
branch for h from the bad link outward (match `name == p3d_hy0.0_m0.2_L5_up` and h ≥ h_bad; keep the
other branch's rows — both branches share out_dir and h values), then run
`HY=0.0 MAX_QUEUE=200 bash nersc/launch_phase3d.sh` — the planner re-emits the train warm-started from
the outermost healthy checkpoint. Chain crash near coexistence (hx≈0.9–1.15 at hz≤0.2) is the spinodal:
record only, do not loop. Deep in a stable phase: re-emit.

5.3 Currently pending redos (verify when landed: E below bound, Vscore per §3, smooth curve; if bad,
report, don't loop):
- 58188982 L5 e0.0 hz0.20, 58188984 L5 e0.8 hz0.23, 58188986 L6 e0.5 hz0.16 — DIAG_SHIFT=5e-3 retries.
- 58188987 hz0.2 L5 UP train 0.75–1.0 re-emitted from the 0.7 link (3 h limit).
- 58165621 L5 hz1.0 dn anchor (hx1.7, gentle recipe, f32); its train 58117290 is HELD by us →
  `scontrol release 58117290` once the anchor lands healthy (its afterok is already satisfied; the
  sweep start-gate reads the NEW anchor JSON).
- 58127185 L6 hx1.25 hz0, 58127183 L6 hx1.3 hz0.4, 58127182 L5 hx1.5 hz0.7 (gentle, f64) — earlier redos;
  their trains 58117272 / 58137003 / 58117285 wait on afterok.
- L6 hz1.0 dn anchor (hx1.7, job 58116356, old dt 0.02 recipe) landed with Vscore 0.11, 0 rollbacks,
  energy converged (E −1237.3). Its train 58117292 is HELD. Decide after 58165621 (same field, L5, gentle)
  lands: if the gentle recipe gives a clearly lower Vscore than the old-recipe L5 result (0.097), redo the
  L6 anchor with `RETRY_FINAL=<its json> RETRY_SET="" HY=0.0 …` (5 h) and keep the train held until it
  lands; otherwise accept it as the ansatz floor and `scontrol release 58117292`.

5.4 hz=0 L5 dn chain went spinodal below hx≈0.95: dn links 0.75–0.9 have Vscore 0.09–70 and 0.75 sits
above the bound (E −255 > −365). Expected physics ("record only"). The winner-curve locators pick the
lower-energy branch per hx, so the garbage loses automatically; the STATUS "above-bound 1" flag on that
row is this point. Same will happen on hz=0.2 L5 dn (its continuation chunk 58186890 is queued).

5.5 Known-benign watcher flags: hz0.0 L4 up hx0.6/0.7 `warm_loaded=False` (leftovers of the parked
diverged job 58114578, energies fine); "E0=None warm_loaded=False" rows for dn links at hx 0.6–0.75 =
points of a train whose continuation chunk has not run yet. Old DIV/STOP lines the watch reports:
58114578 (fixed), 58115317/58115319 (redone), 58116748/58116742/58116739 (retried 09-11),
58117274_0 (hz0.2 up, re-emitted), 58117267_0 (hz0.0 dn spinodal stop — fine).

5.6 Pre-existing gap: the wrappers' `srun … & wait` doesn't propagate the training exit code (job
COMPLETED while the step failed). Safety nets don't depend on it. Fix when convenient.

5.7 Local mirror hygiene: after parking on the cluster, `rm` the run's `.json` under
`results/phase3d/...` and its `.curve.json` under `data/tc_nqs/phase3d/...`, then rebuild STATUS/viewer.

## 6. Results so far (h_y=0 unless noted; ± from the logistic locator, L4 membrane-O_FM for hz≤0.2)

| cut | L4 | L5 (partial) |
|---|---|---|
| electric hx=0 | hz_c 0.295(6) | 0.261(12), 6/7 |
| electric hx=0.5 | 0.300(13) | 0.266(7), 7/7 |
| electric hx=0.8 | 0.345(11) | 0.26(13) poor, 5/8 |
| magnetic hz=0 | hx_c 0.815(70) (membrane) | crossing in [0.95, 1.0] (spinodal-limited) |
| magnetic hz=0.2 | 0.829(54) | no crossing yet |
| magnetic hz=0.4 | crossing 0.90(3) | no crossing yet (8/8 landed) |
| hz=0.7 / 1.0 | merged at L4 | links clean (Vscore 2–9e-3) |
| h_y=0.2 electric hx=0 (L4) | hz_c 0.291(8) | — |

L≥8 verdict: L=10 is not a campaign (k=5 + SRt, 4–8 d/point); L=8 k=5 ~100–180 GPU-h per cut. See
`notes/l10_feasibility.md`.

## 7. Reporting conventions the user likes

Lead with what changed; tables for numbers; one line for routine ticks; flag anything that needs their
decision (approvals, allocation). Ping (PushNotification) only for what they asked: cluster filling up.
After each pull with new finals: republish the artifact and say what landed and whether it is healthy.
