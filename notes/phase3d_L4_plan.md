# ★ RESUME HERE (rewritten 2026-09-22 ~11:45 +05, end of session — user is closing this session; full handoff) ★

You are the orchestrator of the running L=4 multi-plane phase3d campaign (branch `feat/phase3d-campaign`, worktree
`toric-code-nqs-p3d/integration`; a copy of this file lives in the main checkout's `notes/`). Everything below this
block is the durable plan; §0.b holds every plane-by-plane REVIEW DECISION made with the user (never change a
locator/label without them). `notes/phase3d_handoff.md` has cluster mechanics. Memory `phase3d-campaign-plan` and
`phase3d-referee-findings` point here — read both before doing anything.

## State (2026-09-23 ~15:10 +05, 2-hourly ticks running)
- 2121 finals on disk, 0 failed jobs. Viewer v95 (same url). Queue 0 R / 15 PD, all logged below. sshproxy cert valid to 09-24 10:06.
- **In flight right now:**
  - 6 chains `p3d_hy1.0_e{0.2,0.25,0.5}_L4_{up,dn}` (58744879/80, 58744882/89, 58744890/91): h_y=1.0 electric
    redo at h_x=0.2/0.25/0.5. **Tick 09-23 00:53:** hx=0.2 COMPLETE (O_FM 0.189, jump 0.21, E/N merge 0.19–0.21);
    hx=0.25 (0.18–0.21) and hx=0.5 (jump 0.205, E/N crossing ~0.22) need one last point per branch — AUTO_RESUBMIT
    continuations 58762254/456 (+ hx=0.5's pending). Electric line at h_y=1.0 ≈ 0.17/0.20/0.19/0.21 at h_x
    0/0.2/0.25/0.5 — flat, as on the h_y=0 Higgs surface. **For the user's review (label change = user decision):** at
    h_y=1.0 every chained electric cut looks FIRST-ORDER — topological up branch (S2 = 3 ln 2) and z-pol dn branch
    cross with O_FM 0.01→0.6, S2 2.05→0.8, M_z 0.08→0.5 discontinuous, hysteresis ~0.08–0.26; the logistic O_FM fit
    degenerates to a step there (hx=0.5: ±0.85, meaningless) — jump/E-crossing are the meaningful locators.
    2 driver refine cold points on hx=0.5 (58754370/71) landed 3–4 above the up chain (under-converged, lose in the
    winner curve, harmless).
  - **8 trivial↔trivial probes (user, 2026-09-22 evening)** via `analysis/scripts/phase3d_tt_diag_probe.py`
    (PLAN_FILE, 5 h cap): h_x=0 plane fixed h_y=1.4 / 1.5, sweep h_z (`hy1.4|hy1.5/electric_hx0.0`, jobs
    58754397/98/99/404); h_z=0 plane fixed h_x=0.8 / 0.9, sweep h_y 0.6→1.4 / 0.7→1.5 at 0.05 (`ycuts/ycut_hx0.8_hz0`,
    `ycut_hx0.9_hz0`, jobs `p3d_yhc_*` 58754408/11/15/19). `ELECTRIC_FIRST_ORDER` has (1.4,0)/(1.5,0) pre-declared.
    Tick 02:53: hy1.4 running since 12:34 PDT, 5+5 points (up 0.05–0.25: y-pol, M_z 0.04→0.26 curving up; dn
    1.2–0.8: z-pol, M_z 0.85–0.90), all healthy, E slopes = −N⟨σz⟩. Its "0.525" locator is the branch-gap
    artefact (no overlap yet); E extrapolation puts a crossing ~0.33–0.37 (tip-line extrapolation ~0.41). Both
    branches need one AUTO_RESUBMIT cycle → finish after 08:00. hy1.5 / hx0.8 / hx0.9 / (0.4,0) still queued.
    Tick 04:54: hy1.4 up 0.05→0.60, dn 1.2→0.40 (12+12, all healthy). In the overlap 0.40–0.60 the branches are
    MERGED (M_z sep 0.03 at 0.40 → 0.01 above), NO E crossing (up lower by 0.3 throughout), M_z/B_p a steep but
    continuous S-curve (max M_z step 0.165/0.05 < JUMP_MIN 0.2, between 0.30 and 0.40) → crossover-like, centred
    h_z≈0.35 — not "confidently located" as first order. Decisive: does dn stay z-pol below 0.40 (hysteresis) or
    follow up? (dn continues to 0.10 after its AUTO_RESUBMIT). Reading so far: the z↔y first-order line (sharp at
    (0.2, 1.275)) has softened into a crossover by h_y=1.4 — consistent with the y-cuts' "fully merged at h_z≥0.4".
    No extension (rule: both probes complete + confident); for the user: a bisecting fixed-h_y≈1.33 h_z sweep
    between the last sharp point (1.275) and 1.4 is the natural next probe.
    **Tick 06:54 — REVISES the 04:54 reading:** dn (1st chunk done, continuation 58774719 queued) reached 0.35/0.30/
    0.25 and stays z-polarized: M_z dn/up = 0.67/0.53, 0.63/0.37, 0.57/0.26; B_p 0.50/0.27, 0.45/0.14, 0.36/0.09 →
    a HYSTERESIS LOOP opening below 0.40 (merged above). Jump locator 0.325±0.025 ok=True (sharp M_z step on the
    winner curve 0.30→0.35); E_dn−E_up = +1.69/+0.10/−0.35/+0.34 at 0.25/0.30/0.35/0.40 → crossing ~0.30–0.35 but
    within the ~0.3 convergence offset seen where the branches coincide, so the net-crossing test still says
    "merged"; loop test not closed yet (dn must come back below 0.25). Reading now: weakly first-order at h_y=1.4,
    h_z,c ≈ 0.325. The line tip (0.15,1.245) → (0.2,1.275) → (0.325,1.4) runs monotonically outward and steepens
    (Δh_y/Δh_z 0.6 → 1.0), so it bends up toward larger h_y. hy1.5 started 17:48 PDT (1+1 points: up y-pol M_z 0.03,
    dn z-pol M_z 0.90). Overnight extension window closed unused (probes incomplete at 08:00).
    **Tick 08:54:** queue opened — hy1.5, hx0.8, hx0.9 running. hy1.5: up 0.05→0.50 (M_z 0.03→0.63, steepening, B_p
    takes off above 0.35), dn 1.3→0.75 (M_z 0.90→0.81); no overlap yet. **hx=0.8 y-cut (h_z=0 plane): the old
    "x↔y jump 0.95" looks like a branch-gap artefact** — the old coarse dn branch stopped at 1.0; the new dn points
    0.85–0.95 already sit at/below the up branch in E. dn is a smoothly canting state (M_y 0.32→0.75, A_v 0.79→0.30
    over 0.85→1.5); the up branch stays x-pol (M_y 0.26 at 1.3, S2 0.45–0.8) and lies 16 above dn at 1.3 — a lagging
    optimizer state, as the referee predicted ("MF: continuous canting for h_x ≥ 0.5"). Decisive next: does dn
    descend smoothly to 0.6 and join the up branch (crossover, no x↔y line at hx=0.8) or jump (first order, crossing
    ≲ 0.85)? Running now.
    **10:10 (morning overview):** hx=0.8 dn reached 0.65 → CLEAN net E crossing at h_y = 0.798 (E_dn−E_up = +1.88
    at 0.65 … +0.74 at 0.75, −0.03 at 0.80 … −19 at 1.35, monotone), branches distinct over the whole overlap
    (M_y 0.19 vs 0.30, A_v 0.90 vs 0.82, S2 0.57 vs 0.28 at the crossing). Caveat for the user: the line would then
    slope DOWN from the tip (~1.0 at h_x 0.67 → 0.80 at 0.8), and the up branch is a correlated x-pol state the
    referee suspected of lagging — cold-start points at (0.8, h_y 0.75/0.85) would decide. hy1.5: overlap 0.50–0.75,
    merged ≥0.55, split opening at 0.50 (M_z 0.71 vs 0.63, dn −0.88 lower); M_z steepest 0.40–0.50 → h_z,c ~0.45;
    dn continues below 0.50 after its AUTO_RESUBMIT. hx=0.9 just started; (0.4,0) rerun started 10:04.
    **11:10:** hx=0.8 COMPLETE (up to 1.40, dn to 0.60) — left alone per the user. hy1.5 up COMPLETE (0.85); dn at
    0.40, continuation 58782232 queued for 0.35/0.30/0.25. Loop now clearly open: M_z dn/up 0.62/0.39 (0.40),
    0.67/0.53 (0.45), 0.71/0.63 (0.50); dn LOWER in E over 0.40–0.55 → the crossing is ≤ 0.40. The viewer's "0.375"
    is a branch-gap artefact (dn stops at 0.40). hx=0.9: up 0.6→0.9 (x-pol), dn 1.6→1.3 (canted), no overlap yet.
    h_y=1.0 hx=0.25 COMPLETE. h_x=1.0 fine y-sweep PROPOSED to the user (h_y 0.4→1.4, up anchor 0.3) — the old
    (1.0,0) coarse dn branch stops at 1.0 while already 1.5 below up, same artefact as hx=0.8 — awaiting go-ahead.
    **13:10:** h_y=1.0 electric line COMPLETE at all h_x (0.5 dn landed). hx=0.9 overlap 0.95–1.25: E_dn−E_up =
    +0.12/−0.24/−0.54/…/−1.12 → net crossing h_y ≈ 0.97, BUT the branches are the same smooth canting curve offset by
    ~0.07 in h_y (constant ΔM_y 0.07–0.08, ΔA_v similar, E/N nearly coincide, S2 trivial on both) — reads as a
    crossover with optimizer lag, not two phases; viewer "1.000" = winner hopping between offset branches (dn not
    yet below 0.95). Contrast hx=0.8 (rigid up branch, ΔE to −19): the two cuts together (0.80 at hx 0.8, ~0.97 at
    0.9) are non-monotonic from the tip — flagged for the user, not acted on. hx=0.9 continues (up→1.5, dn→0.7).
    **15:09:** hx=0.9 dn reached 0.90 (E_dn−E_up = +0.41/+0.12/−0.24 at 0.90/0.95/1.00) → net E crossing bracketed,
    viewer 0.967; still parallel offset branches (ΔM_y ≈ 0.07). Continuations 58784396/97 queued. Nothing running; the
    hy1.4/hy1.5 dn continuations are our two priority accruers. Viewer: 3D boundary fixes (pocket closes at the roof
    tip, z↔y line from the tip; zoom at cursor, shift-drag pan, dbl-click reset) — v93/v94.
    **15:20 (user):** h_x=0 plane judged on track (z↔y line leaves the tip, runs diagonally up) → submitted h_y=1.6/1.7/
    1.8 h_z sweeps, up+dn (58785717–22): windows centred 0.45/0.50/0.55 ±0.25 at 0.05 (extrapolated; slope Δh_z/Δh_y
    fell 1.7→1.0→0.5), up anchor 0.05, dn anchor window-top+0.1 (user: the 1.5 dn anchor at 1.3 was too far right).
    13/13/14 up + 12×3 dn points. ELECTRIC_FIRST_ORDER += (1.6,0),(1.7,0),(1.8,0). The cron's conditional 1.6/1.7-or-1.45
    rule is retired (job 0f0c752c replaces 6753a563). Next: discuss the h_z=0 plane with the user.
    **15:30 (user) — h_z=0 plane re-designed:** h_y-sweeps equilibrate slowly (the state must rotate x → y); the
    h_x=0.8/0.9 y-sweep points are judged noise. Instead: fixed-h_y h_x sweeps above the roof (max 1.185 at h_z=0), both
    trains starting deep in their own polarized phase, never crossing the pocket — h_y=1.3: window 0.70–1.30, up anchor
    0.4 (M_y≈0.88), dn anchor 1.6; h_y=1.4: 0.80–1.40, up 0.5, dn 1.7; 16 points per train (58785793–96). Estimates:
    mean field (fully polarized) h_x,c ≈ h_y − 0.44 → 0.86/0.96; extrapolated h_x=0.8/0.9 y-cut points → ~1.1/1.16.
    At h_y≈1.3 the existing winner states cant smoothly in h_x (M_x 0→0.88 over h_x 0→1.4, A_v 0.19→0.67) — those are
    y-cut dn states; the x-locked branch has never been followed downward in h_x. MAGNETIC_JUMP_PRIMARY += (1.3,0),(1.4,0);
    viewer draws these x↔y points dashed from the h_z=0 roof tip, off the topological edge (v96). Cron → job 7383fa13.
    2026-09-23 15:50: user → h_x=0.9 continuations 58784396/97 CANCELLED (h_x=0.8/0.9 y-sweeps = noise). Viewer v98:
    phase-diagram projection "(h_y, h_z) planes at fixed h_x" (h_x = 0/0.2/0.5 toggles) replaces the single h_x=0 plane;
    Cuts view gains h_x = 0.2 / 0.5 pseudo-plane tabs (d362a3a, e3b059c).
    2026-09-23 ~16:30 (user go): fixed-h_x pocket mapping, 14 jobs via phase3d_tt_diag_probe.py (df29fc4) --
    h_x=0.2 roof y-cuts h_z=0.1/0.15/0.2 (windows 1.05-1.35 / 1.05-1.35 / 1.10-1.40, anchors 0.6/1.5):
    58786076/77, 58786078/80, 58786081/82; z<->y h_z sweeps (up 0.05, dn 0.1 above the window, half 0.25):
    h_x=0.2 h_y=1.4 (0.10-0.60) 58786083/84, h_y=1.5 (0.15-0.65) 58786086/87; h_x=0.5 h_y=1.4 (0.25-0.75)
    58786088/90, h_y=1.5 (0.30-0.80) 58786091/92. Expect shift vs h_x=0 (0.325/0.375): MF +0.01/+0.07,
    tip-continuation (user) ~+0.15 at h_x=0.5. ELECTRIC_FIRST_ORDER += (1.4|1.5, 0.2|0.5) (fddf863).
    When landed: classify the new y-cuts roof/remnant (YCUT_S2_ROOF/REMNANT) and check the viewer's fixed-h_x overlay
    + 3D (z<->y dashed line now per h_x slice, ebc1722).
    2026-09-23 (user): scrontab trimmed to the single `p3d-driver-hyy` entry (keeps watch_state.json fresh; idle
    launcher). Removed p3d-wandb-sync (timing out, W&B unused by the viewer) and the six h_y-plane drivers (no
    submissions since 09-15..22; all new work is PLAN_FILE + in-job AUTO_RESUBMIT). Backup:
    $PSCRATCH/tc_nqs/phase3d/scrontab_backup_20260923.txt (`scrontab <file>` restores).
    Direction (user, 2026-09-23 evening): let the current runs finish, then decide. Candidate next steps if the
    in-plane-sweep test holds: extend the x<->y / z<->y lines to their endpoints (jump size + loop width -> 0), then map
    the x<->y sheet at h_z = 0.1/0.2 (watch for the z<->y sheet / triple line). L=5 spot checks agreed useful but
    compute-heavy -- only by explicit user decision. Cron tick -> job 98450c7c.
  - TICK 2026-09-23 17:20 (v99): h_y=1.4/1.5 dn continuations landed (1.4 dn to h_z=0.1, 1.5 dn to 0.25) -> full
    M_z hysteresis loop at 1.4 (dn drops 0.2->0.15, up jumps 0.3->0.35; jump 0.325, 1.5: 0.375). (0.4,0) last retry:
    up reached 1.08 (healthy), diverged at 1.11 (25 rollbacks, E -28.6) -> CHAIN STOPPED; its "jump 1.095" is the
    branch-gap artefact, extrapolated E crossing ~1.16 (biased high, see below). 1.6-1.8 z<->y trains running.
    **FINDING: y-polarized NQS states are bimodal and under-converged.** 2nd-order strong-field series (around the
    product state along (0,h_y,h_z); validated vs L=2 OBC ED: series sits 0.08-0.16 above exact, i.e. ~-1.2 at N=144)
    puts E(h_y=1.4, h_z=0.1) ~ -218.3. NQS y-pol points fall in two families: "good" <B_p> ~0.12-0.15, E ~+2.5..3 above
    exact; "bad" <B_p> ~0.05-0.08, E ~+7..8 above (perturbative <B_p> = 1/(4h) ~ 0.18). Bad: both h_z-sweep up
    branches (1.4, 1.5), roof y-cut dn branches (0,0.05), (0,0.15), (0.4,0), partly (0,0.2); good: (0,0), (0,0.1),
    (0.2,0), (0.5,0.1), (0.5,0.2) [(0.5,0), (0.6,0) start bad at 1.5, recover to good by h_y~1.2]. The hx=0 roof
    zig-zag (1.175 / 1.205 / 1.175 / 1.245 at h_z 0/0.05/0.1/0.15) tracks the family -> the tip at (0.15, 1.245) is
    likely an artefact (roof ~flat 1.175-1.185); z<->y crossings are biased low in h_z (y-pol too high). Reported to
    the user with a reseed proposal (warm-start bad y-pol branches from a good state at the same field); no jobs
    changed pending the decision.
    User (18:50): keep chains single-variable -> fix at the ANCHOR: best-of-3 seeds (phase3d_reseed.py, be7442b).
    Anchor trials submitted: ycut (0,0.15) dn at h_y=1.5 -> 58788131/32/33 (seeds 101-103); h_y=1.4 h_z-sweep up at
    h_z=0.05 -> 58788137/40/41. When they land (tick): on the cluster
      python analysis/scripts/phase3d_reseed.py select --base $PSCRATCH/tc_nqs/phase3d --apply --emit $PF
    (winner = lowest E0 incl. the original seed-0 anchor, gate dE <= 4 vs the strong-field estimate) then
      PLAN_FILE=$PF/ycut_hx0_hz0.15_dn_chain.tsv HY=y / PLAN_FILE=$PF/hy1.4_e0_up_chain.tsv HY=1.4 launch_phase3d.sh
    and mirror the parking locally: phase3d_reseed.py park --label <l> --base <main>/results/phase3d --stamp <S>.
    If no seed passes the gate: report, don't launch. Queued h_x=0.2/0.5 y-pol trains untouched (user not asked to hold).
  - DATA ARCHIVAL (user, 2026-09-23): do it AT THE END, once the 3D bosonic phase diagram is final. Then: (1) pull the
    final trained networks (one <name>.mpack per point; phase3d ~557 MB / ~2130 files, + phaseB/phaseB_rerun/hy_cuts/
    hy_axis/tune_rect ~120 MB) into the main checkout's gitignored data/archive/<campaign>/ (mirror the scratch layout,
    verify counts); (2) result JSONs + curves are already local (results/phase3d, data/tc_nqs/phase3d) -- commit
    curve-stripped finals (~6 MB) + summary.json + transitions/hy_axis summaries to git; (3) skip step snapshots (3.9 GB),
    resume ckpts, W&B. Ask the user then about a second copy on NERSC CFS (scratch purges ~8 weeks unaccessed).
    Publication plots: later, separately. Cleanup branch chore/publication-cleanup (tag pre-publication-cleanup) in flight.
  - TICK 2026-09-24 00:22 (new session after the usage-limit stop; viewer v100): +200 finals. Physics:
    * z<->y (h_x=0, h_z sweeps): clean jump+loop at h_y 1.4 (0.325) / 1.5 (0.375); loop narrows at 1.6 (jump ~0.475),
      marginal at 1.7 (~0.525), at 1.8 the branches share one M_z curve (differ only in energy = stuck-anchor effect)
      -> the z<->y line ends near h_y ~1.7-1.8 at L=4.
    * x<->y (h_z=0, h_x sweeps, the in-plane-sweep test): at h_y=1.3 NO jump and NO hysteresis -- up/dn agree to
      |dM_x|~0.01, |dE|~0.2 over h_x 0.75-1.25; smooth canting from both ends -> crossover; the x<->y line ends
      between the h_z=0 pocket tip (h_y~1.19) and 1.3. h_y=1.4 dn still landing (same picture so far).
    * STUCK y-pol anchors: 8/9 new ones (h_x-sweep up @1.3/1.4, h_z-sweep up @1.6/1.7/1.8 and h_x=0.2 @1.4, y-cut dn
      (0.2,0.1)/(0.2,0.15)); (0.2,0.2) dn ok. Best-of-3 at (0,0.15): 0.32x/0.27x/0.56x -> same-recipe reseed FAILS
      (stuck is the typical cold y-pol outcome, not a coin flip). (0.2,0.1)/(0.2,0.15) dn end at h_y=1.05 above the
      h=0 bound (unhealthy last points, excluded).
    RECIPE EXPERIMENT (00:40, all single-variable, existing points or branch extensions; phase3d_reseed.py 3d7c684):
      (0,0.15) y-cut dn anchor @h_y=1.5: c103 = continue s103 +1000 steps (58796090); L201/L202 = cold 1500 steps
      ds 3e-3 (58796093/94); a301 = cold at h_y=3.0 then 2.5/2.0/1.75/1.5 on the same line (58796095); round trip =
      up branch extended 1.32->1.4->1.5 (58796096). h_y=1.4 h_z-sweep up anchor @h_z=0.05: L201/L202 (58796100/01);
      round trip = dn branch extended 0.1->0.05 (58796103); s101-103 (old recipe) still queued.
      select now also scores the OPPOSITE branch at the anchor field (round trip) -- `select` (dry) shows all.
    HELD (scontrol hold, release with `scontrol release`): 58786086 (h_y=1.5 h_x=0.2 up), 58786088 (1.4 h_x=0.5 up),
      58786091 (1.5 h_x=0.5 up) -- y-pol-start trains that would land stuck; release once a recipe works.
      `scontrol top` is not permitted for users on Perlmutter.
    00:55 user: RELEASED the 3 held trains again (let the y-pol-start h_x-plane sweeps run and see) -- nothing held now.
  - TICK 01:18 (v101): +6 finals (h_y=1.4 h_x=0.2 up to h_z=0.6; y-cut (0.2,0.2) up 1.4); no new y-pol anchors, no
    failures. No p3d job running: the user-level queue (MaxJobsAccrue=2) is shared with ~27 jobs of OTHER sessions
    (hc_sgnb x18, hc_signfid, hc_pretrain, tc-signbench -- not ours, untouched), so p3d throughput is low tonight.
    The core-cleanup agent's debug jobs (pc-base-*, pc-core-*) run on gpu_debug.
  - CLEANUP (02:10): chore/publication-cleanup = pc-nersc + pc-analysis + pc-core merged + fixes (1b2f37d, pushed; tag
    pre-publication-cleanup pushed). tc3d removals verified bit-identical on Perlmutter (13 run JSONs + 22 mpacks,
    ~/tc-nqs-pc-jobs/COMPARISON.txt). Docs agent + adversarial tc3d audit running. DEPLOY NOTES (after the campaign):
    export WANDB_ENTITY=models-california-institute-of-technology-caltech in the cluster env (train.py now defaults to
    $WANDB_ENTITY); hamiltonian.py/geometry.py changed -> Pauli-cache code hash changes -> first jobs rebuild the cache
    (~200 s at L=4) -- re-prime before a campaign. Incident: the core agent ran `rm -rf <macOS $TMPDIR>/tmp.*`
    (may have removed other processes' mktemp dirs) -- tell the user.
  - TICK 03:16: 0 new finals, 0 p3d running / 22 pending (all "Priority"). Perlmutter GPU pool mostly drained tonight
    (~133 drained, 51 draining, 32 planned; gpu_shared 25 R / ~1760 PD) -- nothing wrong on our side. Recipe trials all
    still queued; select unchanged (best so far s103 0.56x, gate FAIL).
    CLEANUP: docs refreshed + ARCHIVE.md (c55e937); adversarial tc3d audit: NO CRUCIAL (12 extra configs bit-identical
    old vs new; resume keys unaffected); follow-ups be78cc8 (tests/test_renyi_exact.py exact 3ln2 anchor, fit error-bar
    rtol 0.15, merge/cache/error-bar notes). Pauli-cache priming on deploy: L4 ~3-11 min, L5 ~9-30 min, L6 ~20 min per
    (L,bc,dual,dtype) key, no lock -> prime before releasing a campaign. Transition error bars reproduce only to ~12%
    across scipy builds (ill-conditioned Richards covariance) -> record env or bootstrap for publication.
  - TICK 05:16: GPU pool recovering (gpu_shared 57 R); 3 p3d chains running (h_x=0.2 dn @1.4/1.5 healthy, drift ~0;
    h_x=0.2 up @1.5 anchor at step ~100/1000, still descending -- normal). **RECIPE RESULT: ROUND TRIP WORKS for the
    h_z-sweep anchor** -- the h_y=1.4 dn branch continued 0.1->0.05 on the same line landed GOOD (E0 -214.714, <B_p>
    0.123 = 0.69x lead, dE +3.15) vs the stuck original up anchor (0.31x, +7.7). Applied at 05:17 (did not wait for the
    queued L/s trials -- they could only gain <~0.6): up branch parked -> redo_reseed_202609231717 (cluster + local),
    winner copied in as the up anchor, original up chain relaunched = job 58806874 (0.05 -> 0.7, same 14 points).
    L201/L202/s101-103 trials for this label still queued (informational: does a cold long recipe also work?).
    (0,0.15) y-cut label: best still s103 0.56x (FAIL); its c103/L/a301/round-trip trials queued.
  - TICK 07:16 (v102): GPU pool back (11 of ours running). +19 finals: h_x=0.2 dn @1.4 (0.4-0.7) and @1.5 (0.5-0.75);
    h_x=0.2 up @1.5 and h_x=0.5 up @1.4 started -- both anchors STUCK (0.27x / 0.35x), as expected (left running per
    user). Reseeded h_y=1.4 up anchor shows as `ok` (0.69x); its chain 58806874 queued. c103 trial failed instantly:
    my INIT_FROM was absolute but the launcher prefixes OUT_DIR -> fixed (help text, relative ../s103/<name>),
    resubmitted as 58809926 (old dir parked in anchor_trials/redo_c103_badpath). s101-103 trials for h_y=1.4 running.
  - TICK 09:16 (v103): +31 finals. z<->y at h_x=0.2: jump 0.375 @h_y=1.4 (h_x=0: 0.325) and 0.425 @1.5 (0.375) -> shift
    +0.05 at both; loops visible (1.4: up jumps 0.40->0.45, dn drops 0.25->0.20). h_x=0.5 @1.4: up-only jump ~0.475
    (+0.15, the user's estimate; dn pending, locator not ok yet). Recipe tally at the h_y=1.4 h_z=0.05 anchor: old recipe
    3 new seeds -> s101 GOOD (0.67x, +3.5), s102 stuck (0.30x), s103 badly stuck (0.05x, +10.5): cold starts hit ~1/3;
    round trip GOOD. (0,0.15) y-cut: c103 resubmit / L / a301 / round trip still queued. sshproxy cert expires 10:06.
  - TICK 11:16: SKIPPED -- sshproxy cert expired 10:06 (Permission denied); needs `! sshproxy -u sanzharb` from the user.
  - `p3d_y_hx0.4_hz0_L4_up` extension (58755647): the gentle retry's up branch diverged at h_y=1.11 (15 rollbacks)
    and CHAIN STOPPED; the 1.085 "jump" was the branch-gap artefact (up at 1.06 is 8.5 below dn; crossing
    extrapolates to ~1.14). Re-run from the 1.01 checkpoint over 1.06→1.26, ds 1e-2, 500 steps/link; old
    1.06/1.11 parked in `redo_58744247/` (cluster + local mirror).
    **Tick 11:10:** that rerun (58755647) retrained 1.06 CLEANLY (E0 −186.64, 0 rollbacks, S2 1.81, A_v 0.95 — kept) but
    1.11 blew up AT STEP 0 (E = +7e18 before any update; 37 rollbacks, CHAIN STOPPED) — the 2nd death on 1.06→1.11.
    Not physics: extrapolated, the up branch would still be ~4.4 BELOW dn at 1.11, and a ground-state branch has no
    spinodal. E extrapolation puts the crossing at h_y ≈ 1.16. 1.11 parked in `redo_58755647/`. ONE last attempt
    (58782510): from the new 1.06 via INIT_FROM (fresh sampler), finer steps 1.08/1.11/1.13/1.16/1.18/1.21 (land on
    the dn grid at 1.11/1.16/1.21), dt 0.0025, ds 1e-2, 400 steps. If it dies again: stop, record the extrapolated 1.16.
- (0,0.05) y-cut resolved: branches overlap, net E crossing 1.185±0.015 (jump 1.205) — between (0,0)=1.175 and
  (0,0.15)=1.245.
- **User decisions, 2026-09-23 morning:** (a) h_y=1.0 electric cuts STAY labelled 2nd order (the first-order look
  is noted, not acted on). (b) h_x=0.8 y-cut: leave as is — no cold-start test, no extension; wait for h_x=0.9 and
  (if approved) h_x=1.0. (c) h_x=0 plane next step APPROVED once h_y=1.5 lands and makes physical sense: still
  first-order → h_y=1.6/1.7 on the linear extrapolation; already a crossover → one bisecting h_y=1.45 sweep;
  unclear → report. Exact rules in the session cron prompt (job 7383fa13). The overnight authorization (08:00)
  lapsed unused. `phase3d_tt_diag_probe.py --only <labels>` emits just new points; a y-sweep at an (h_x,h_z) with an
  existing y-cut reuses its run names (e.g. h_x=0.8 reused 20 old points, trained only the interleaved h_y).
- **`phase3d_grid.py extend`** (new): warm-started continuation of a chain branch from a healthy final — the fix
  path for diverged/stopped/unconverged chain links (retry refuses links). **PLAN_FILE gotcha:** the launcher writes
  the shell `HY` into every manifest row — pass the plane value, or `y` for y-cuts (the probes were first logged
  under 1.0 by mistake; rows fixed, originals in `manifests_bak/*.hyfix_bak`).
- Looking at plots: Chrome's screenshot capture stalls on the 7 MB viewer. Serve it with
  `<scratchpad>/viewer_srv.py` (static + POST /save sink), pull each chart's SVG out of the live tab (inline computed
  styles), `rsvg-convert` to PNG, Read.
- Old: the 4 cancelled y-cut guesses (commit `bb171f4`, (0.7,0)/(0.9,0)/(0,0.25)/(0,0.3)) stay unused; the user's
  own probes above replaced them.

## Scrontab automation on the cluster — READ THIS BEFORE TOUCHING THE QUEUE
`scrontab -l` on Perlmutter shows **1 job, intentional infrastructure** (trimmed 2026-09-23 by the user; the old
8-entry table is in `$PSCRATCH/tc_nqs/phase3d/scrontab_backup_20260923.txt`, restore with `scrontab <file>`):
- `p3d-driver-hyy` (`cron` QOS: CPU-only NERSC queue for scheduled scripts, no GPU hours), hourly at :45. Re-runs
  `nersc/launch_phase3d.sh` for HY=y (idempotent, manifest-deduped; idle since 09-17) then `nersc/watch_phase3d.sh`
  over ALL manifests -> `watch_state.json` (the live-state column of phase3d_status.py). Kept only for that watcher.
- Removed: the six h_y-plane drivers (no submissions since 09-15..22 -- new work goes through PLAN_FILE, timeouts
  resubmit in-job via AUTO_RESUBMIT) and `p3d-wandb-sync` (never cleared its offline backlog; W&B unused by the viewer).
- `p3d-driver-hyy` sometimes hits its 25-min limit in the watch step (~3000+ files) -- harmless, the launch step
  finishes first; raise `-t` if the watch state goes stale.
- Job `debug_grayanchor` (if present in `squeue`) is **NOT ours** — never touch it. Only `scancel`/inspect jobs
  whose name starts `p3d_` or `p3d-`.

## Physics summary — what we actually know now
The three planes (h_x=0, h_y=0..1.0 sweeps; h_z=0; and the six h_x–h_z planes) are all mapped and cross-checked
with end-of-training S₂ on every point (see below). An adversarial referee agent reviewed the whole picture on
2026-09-21 (full report: memory `phase3d-referee-findings`); its verdicts, now folded into the data:
- **Confirmed sound:** the (h_x,h_z) pocket is the Fradkin–Shenker 3+1D Z₂ gauge–Higgs topology (flat 2nd-order
  Higgs surface + 1st-order confinement surface meeting at a corner, first-order line into the interior); a
  product-state mean field reproduces the x↔z line to ≤0.1. The roof (topo→y-polarized) is close to a sphere
  |h|≈1.18 at L=4. No exact/self-dual mapping gives the (h_x,h_y) or (h_y,h_z) planes for free (unlike 2D, where
  h_y,c=1 exactly) — expect h_y,c(L→∞)≈1.3–1.45; the L=4 OBC value (~1.175–1.275 depending on cut) is low mainly
  from open boundaries removing 1/8 of the stabilizer energy.
- **Convergence bias found and partially fixed:** cold 7-point electric fits at h_y≥0.6 violated exact energy
  monotonicity on the topological side (E rising by 0.9–3.3 between neighbours, Vscore 0.2–0.4 vs 0.05 polarized).
  Fix = warm h_z chains (`ELECTRIC_CHAIN` in `phase3d_grid.py`: up from h_z=0.02, dn from 0.45, gentle 1000-step
  anchors, `TOPO_POOLED=1`). Already run for h_x=0 at h_y=0.6/0.8/1.0: h_z,c moved 0.252→0.235 (0.6, within 1σ),
  0.197→0.207 (0.8, within 1σ), **0.114→0.164 (1.0, the only significant shift)**. The same redo for h_x=0.2/0.25/0.5
  at h_y=1.0 is the 6 chains currently in flight (expect h_z,c to land near 0.16–0.18 per the user's own estimate).
- **S₂ test (all ~600 points across the plane-1.0 magnetic cuts + all 15+3 landed y-cuts) confirms every
  roof/remnant/trivial↔trivial call**, now hardcoded server-side in `phase3d_status.py` instead of guessed
  geometrically: `MAGNETIC_JUMP_PRIMARY = {(1.0,0.2),(1.0,0.25)}` (S₂ never reaches 3 ln 2 → trivial↔trivial, M_x
  jump at 0.675 is the real locator, not the broken O_FM fit), `YCUT_S2_ROOF`/`YCUT_S2_REMNANT` (explicit sets
  replacing the old inside-polygon geometry guess — it had wrongly demoted (0.5,0.2), which S₂ confirms IS a roof
  point). `EXCLUDE_CUTS = {(1.0,"electric_hx0.8")}` stays (pure noise, no fit possible).
- **Open, unresolved:** where exactly the z↔y line (h_x=0 plane) ends between h_z=0.2 (confirmed jump, h_y,c=1.275)
  and h_z=0.4 (branches fully merged) — the cancelled probes above were aimed at this. Same for the x↔y line
  (h_z=0 plane): h_x=0.8 clean jump (0.95), h_x=1.0/1.2/1.4 same nominal value but NOT sharp and weakening
  (separation 0.09→0.055→0.027, not vanishing) rather than a clean endpoint. Also unresolved: the exact x↔z line
  endpoint (referee wants a loop-width extrapolation from h_z=0.4/0.7/0.85, not yet done).

## Viewer (Artifact) — what's in it as of v80
Rebuild recipe is in memory `phase3d-viewer-artifact` (unchanged). Contents as of today:
- Cuts view: per-h_y-plane tabs + a "y-cuts" tab, unchanged structure.
- Phase-diagram view: **two new pseudo-plane tabs**, "h_x = 0 plane" and "h_z = 0 plane" (views over the same
  stored cuts, not new data) — click one to see the (h_y,h_z) or (h_x,h_y) cross-section directly.
- "Planes overlaid" panel: a **projection selector** — `(h_x,h_z) planes at fixed h_y` (original), `h_x=0 plane ·
  (h_y,h_z)`, `h_z=0 plane · (h_y,h_x)` (axes intentionally swapped vs the pseudo-plane tab's own chart so h_y is
  always the horizontal axis in this panel).
- "Boundary in (h_x,h_y,h_z)" 3D SVG panel: **mouse-driven now** (drag to rotate, wheel to zoom — sliders removed);
  y-cut roof/remnant points are red/orange **squares** now (were an unlabelled star), matching the "Sketch" panel's
  palette; "show y-cut 1st-order boundary points" toggle defaults ON (used to default off and silently hide them
  here too — that was the bug that prompted this fix).
- "Sketch" panel (three.js, bottom of page): unchanged mechanics, automatically reflects the same S₂-confirmed
  roof/remnant classification.
- Rotation direction is now the same convention in both 3D views (drag right = scene turns right).

## Per-tick procedure (every 2 h — user decision; re-create the session CronCreate job if this is a fresh session)
1. `cd toric-code-nqs-p3d/integration && DO_PULL=1 SINCE_MIN=125 bash analysis/scripts/phase3d_local_tick.sh`
   (pull → STATUS/summary → exports incl. plane `y` → cluster watch line).
2. If new finals landed: rebuild + republish the viewer to the SAME url (memory `phase3d-viewer-artifact`), copy
   `summary.json` + this file to the main checkout, report ONLY what changed.
3. New DIV line in the watch = a GENUINE DIVERGENCE (check `n_rollbacks`/garbage E0-Vscore, not just the flag):
   for a magnetic/electric-chain anchor, `RETRY_FINAL=<final> RETRY_SET="DIAG_SHIFT=5e-3 DT=0.01" HY=<hy> bash
   nersc/launch_phase3d.sh` then drop the job's never-run link rows from its manifest (backup `.bak_<jobid>`),
   park the old outputs in `redo_<jobid>/`. **The retry tool does NOT handle `ycuts/` dirs or the electric h_z
   chains** — for those, forget the job's manifest rows by hand and resubmit via a hand-built `PLAN_FILE` (see the
   09-21/09-22 approval-log entries below for worked examples) or by re-running the launcher with `CUTS=<cut_id>`
   scoped narrowly (the planner regenerates the same chain job automatically once the anchor is gone from the
   manifest — no need to hand-craft anything for a plain re-submit of a whole chain).
4. sshproxy cert is 24h; if `ssh perlmutter` starts failing with auth errors (not a timeout), the cert expired —
   tell the user, don't keep retrying.
5. **Never** `git commit` to `main`, never `git add analysis/` wholesale (a peer's untracked
   `analysis/scripts/transition_fit.py`/`firstorder_fit.py` live there), never touch `debug_grayanchor`.

## Still open / ask the user before doing
- Where to bank the untracked `results/phase3d/` + `analysis/notebooks/phase3d_L4_planes.ipynb` (main checkout).
- The two unresolved endpoint questions above (z↔y line, x↔y line) — reconsider the cancelled probes with fresh
  eyes and the user's buy-in.
- x↔z line endpoint via loop-width extrapolation (h_z=0.4/0.7/0.85 loop widths → 0 crossing).
- S₂ on the magnetic chains outside plane 1.0, if ever wanted (extractable from checkpoints via `tc3d.renyi`).
- Closure plane h_y=1.2, QMC referee at h_y=0 (both from the original 2026-09-17 plan, never started).

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

**Every 2 h** (session CronCreate at :43 on even hours; re-arm in a new session). One command does the
mechanics: `VIEWER_DIR=<scratchpad>/viewer DO_PULL=1 SINCE_MIN=125 zsh analysis/scripts/phase3d_local_tick.sh`
(pull -> STATUS.md / summary / viewer JSONs + build -> `phase3d_tick_checks.py` -> one ssh line: queue counts,
failures of the last 6 h, DIV / CHAIN STOPPED logs, the y-cut driver, 3 random job samples).

**Checklist (user, 2026-09-23 -- numbers first, plots only on a flag):**
1. **What's running** -- the WATCH line: running / pending counts, failed jobs, DIV / CHAIN STOPPED logs.
   Any failure or stopped chain -> diagnose (log tail) and fix (retry / extend / rerun from the last healthy point).
2. **Random samples** -- the 3 SAMPLE lines (`nersc/phase3d_sample_jobs.sh`): current point, step, E, drift and
   spread over the last <=50 steps, rollbacks. Healthy: |drift| << spread, few rollbacks. A bad sample -> read
   that job's full log.
3. **Y-polarized anchors -- ALWAYS hand-check** every newly landed one (the `anchor ok / STUCK` lines): read
   <B_p> (vs 0.6x leading order) and dE (h_x = 0), and look at its learning-curve tail numerically. STUCK ->
   best-of-3 reseed (`phase3d_reseed.py`, add the chain to CHAINS) -- rerunning existing points is within the
   fix mandate; tell the user in the report.
4. **New data** -- republish the viewer (same URL). Physical sense from the numeric flags on touched branches:
   E below -172, E not rising with the swept field, HF on h_x/h_z sweeps, branch crossings/hysteresis where
   expected (`scratchpad/cutdiag.py <plane> <cut>` tabulates both branches). Render a plot (SVG extraction +
   rsvg-convert) ONLY when a check flags something or a transition is newly located -- at most one or two.
5. **Log** the tick in the RESUME block (commit + push + copy to the main checkout's notes/); report only changes.

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
- 2026-09-22 ~10:05 — user: the h_x=0 chain fix (0.114->0.164) should apply to h_x=0.2/0.25/0.5 too, expected h_z,c ~0.16-0.18.
- 2026-09-22 ~11:30 — user: the y-cut points closing the pocket in the "Boundary in (h_x,h_y,h_z)" 3D view were
- 2026-09-22 ~11:15 — user: trace the trivial-trivial lines off both plane corners using the boundaries already mapped.
- 2026-09-22 ~12:35 — tick. The 2 gentle up-anchor retries landed clean: y-cut (0,0.05) -> h_y,c = 0.99 ± 0.09,
  (0.4,0) -> h_y,c = 0.88 ± 0.08 (both roof points, stabilizers agree). All 7 h_x=0 roof rungs now complete
  (0,0/0.05/0.1/0.15 -> 1.175/0.99/1.175/1.245) and all 4 h_z=0 roof rungs complete (0,0.2,0.4,0.6 ->
  1.175/1.185/0.88/0.945) — note (0.4,0)=0.88 breaks the otherwise-smooth decreasing trend (0.6->0.945 is HIGHER
  than 0.4->0.88), worth a second look once eyes are back on this. Viewer v81. Queue 2 R / 6 PD (the 6 hy=1.0
  electric chains at hx=0.2/0.25/0.5, still running), 0 failed.
- 2026-09-22 ~13:10 — CORRECTION to the entry above, read this first: the user checked (0,0.05) and (0.4,0) by eye
  and was right to distrust them — the up and down branches do NOT overlap yet (up: 0.6-0.9 / 0.6-0.8; dn: 1.08+ /
  0.96+), so the "located" h_y,c (0.99, 0.88) is just the midpoint of the current GAP, not a real crossing. Verified
  this is NOT a divergence: jobs 58744241 / 58744247 are still RUNNING (no rollback/CHAIN STOPPED lines), 1h50m into
  a 4h30m walltime, and their own FIELD_VALUES lists already extend to 1.33 / 1.26 respectively — both will pass
  through the down branch's start (1.08 / 0.96) once they get there, closing the gap on their own. User decision:
  DO NOT submit an extension yet — wait for these two jobs to finish (they should on their own within a tick or
  two), then re-check. Only if the gap is STILL open once they finish should you extend each branch by +0.20 h_y
  (a chain-link-style warm-started job from each branch's own last checkpoint; no such y-cut "add more links"
  helper exists yet in phase3d_grid.py — the closest precedent is the hand-emitted `PLAN_FILE` link job used for
  the plane-0.8 h_z=0 "coarse up links" fix in the 2026-09-20 log, adapted to SWEEP=hy). General lesson for next
  time: before trusting ANY newly-landed y-cut's located h_c, check that both branches actually overlap in h — a
  jump/crossing locator run on a winner-take-all curve across a real GAP will report the gap's midpoint as if it
  were a transition, indistinguishable from a real result unless you look at the branch ranges.
  h_z=0 (x-pol<->y-pol): hx=0.6 roof, hx=0.8 clean jump 0.95, hx=1.0/1.2/1.4 same nominal 0.95 but NOT sharp (sep
  0.09/0.055/0.027, weakening not vanishing) -> added hx=0.7 (pin the corner, window 0.82-1.12) and hx=0.9 (bracket
  the weakening, window 0.9-1.2). h_x=0 (z-pol<->y-pol): only hz=0.2 -> 1.275 confirmed; hz=0.4/0.55/0.7 FULLY merged
  (sep 0.003-0.018) -> the line likely ends between 0.2 and 0.4 -> added hz=0.25 (window 1.25-1.55, dn anchor
  extended 1.5->2.0) and hz=0.3 (window 1.4-1.7, dn anchor 2.0). New `YCUT_CENTER_OVERRIDE`/`YCUT_DN_ANCHOR_OVERRIDE`
  in phase3d_grid.py (bb171f4) since these are physically-guessed line points, not roof points (the spherical-roof
  centre formula and the fixed 1.5 anchor don't apply off the roof). Submitted 8 chains: 58747688-95.
  drawn as an unlabelled 5-pointed star, confusing. Fixed (17fb83d): star -> red square (roof, 1st order, bounds the
  pocket) / orange square (polarized<->y-polarized remnant), same palette as the three.js sketch panel; hy-axis exact
  anchor -> hollow circle. Found + fixed along the way: the "show y-cuts" toggle (added 2026-09-21 to declutter the
  2D overlay) was ALSO stripping these points from the SVG 3D boundary view by default via the shared chartRows()
  filter -- defaulted it ON now that the points read as meaningful boundary markers instead of generic triangles.
  Confirmed the 3 newly-landed roof y-cuts ((0,0.15)->1.245, (0.2,0)->1.185, (0.6,0)->0.945) with real in-job S2
  (plateau 2.0-2.2 throughout). Visually verified (rotate + zoom, both the SVG boundary panel and the sketch) in a
  local Chrome copy before publishing v80 -- Chrome's screenshot capture was intermittently unresponsive this
  session (recovered on retry each time; not a page issue, confirmed via DOM/data inspection as a backup).
  Extended `ELECTRIC_CHAIN` to (1.0,0.2),(1.0,0.25),(1.0,0.5) (e43cab2). Submitted 6 chains (up 0.02->0.40, dn 0.45->0.05,
  gentle 1000-step anchors, TOPO_POOLED=1): 58744879/80 (hx=0.2), 58744882/89 (hx=0.25), 58744890/91 (hx=0.5). The old
  cold-point fits stay on disk (winner-take-all keeps whichever point is lower energy) and will be superseded once the
  chains land. Not yet done: the same redo for hy=0.6/0.8 hx=0.2/0.25/0.5 -- their hx=0 fits barely moved, so lower
  priority; ask before extending further.
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
