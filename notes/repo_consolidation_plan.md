# Repo consolidation plan — APPROVED by the user 2026-09-28 (execute after /compact)

Goal: one checkout, one branch, ending in `main`, that holds all code + committed results for the 3D bTC phase
diagram, such that a fresh clone regenerates the phase-diagram viewer data byte-identically with two commands.
Fermionic track (`toric-code-nqs-fsign`, `toric-code-nqs-ladder`, fermionic branches/results) is OUT OF SCOPE — leave untouched.

## Facts measured 2026-09-28 (re-check before acting)
- One GitHub repo `origin = github.com/SanzharBissenali/toric-code-nqs`. `main` (873f28a) has NONE of the phase3d work.
- `feat/phase3d-campaign` (worktree `toric-code-nqs-p3d/integration`, pushed, head ec7f376): all campaign code; 290
  commits ahead of main. Missing `analysis/scripts/transition_fit.py` and `analysis/notebooks/transition_fss.ipynb`
  (both on the cleanup branch; transition_fit.py is untracked in the p3d worktree too — phase3d_status imports it).
- `chore/publication-cleanup` (worktree `toric-code-nqs-pc/main`, pushed, head be78cc8, tag `pre-publication-cleanup`
  marks its base): 37 cleanup commits; lacks the last campaign commits. Merge-base diff: campaign changed only 9 files
  (cutdiag, local_tick, reseed, status, tt_diag_probe, pull_phase3d, viewer extras/html, plan); cleanup changed 102.
  Overlap = `analysis/scripts/phase3d_status.py` only. `git merge-tree` of the two: ZERO textual conflicts.
- Main checkout `toric-code-nqs` is on `feat/phase3d-transition-fss` (= c7d2af2, no phase3d commits) with UNTRACKED:
  `results/phase3d/` (341 MB: 2545 finals = 332.6 MB because each final JSON embeds the network `weights`; 706
  snapshot/finaleval 2.8 MB; 657 other 15.6 MB), `results/transitions/` (132 K; the h_y=0 L4-6 older-lane locators the
  viewer shows via phase3d_extras.json "boundary"), `results/hy_cuts_L4/`, `results/hy_axis_L4/`, notebooks
  (`transition_fss`, `phase3d_L4_planes`, `hy_cuts_L4_transitions`, `hy_axis_L4_S2`, `cut_fss_explorer`,
  `phase_diagram_manual`, b8's `phase3d_trivial_wall.ipynb`), `analysis/scripts/transition_fit.py`, figs, notes
  (`phase3d_L4_plan.md` copy, `phase3d_handoff.md`, `l10_feasibility.md`), modified `BLOG.md`, `CLAUDE.md`, `README.md`,
  `notes/transition_mapping_recipes.md` (b8's 2026-09-27 BLOG entry lives here). Also fermionic/arch-compare untracked
  files — NOT ours, leave.
- `data/tc_nqs/phase3d/` (225 MB, gitignored): curves/checkpoint mirror. Cluster: repo `~/toric-code-nqs` tracks
  `origin/feat/phase3d-campaign`; raw outputs + networks + `manifests/*.tsv` under `$PSCRATCH/tc_nqs/phase3d`.
- Worktrees: `toric-code-nqs-p3d/{A1-chain-runner,A2-pauli-cache,A3-launcher,A4-progress,A7-firstorder,close-medium,
  speed}` = feature branches ALL merged into feat/phase3d-campaign; `toric-code-nqs-pc/{analysis,core,nersc}` merged
  into chore/publication-cleanup; `.claude/worktrees/agent-a640050e99c369032` merged; `/private/tmp/.../hy-cuts`
  prunable. Only `p3d/integration` and `pc/main` are live.

## Steps (user approvals noted)
1. **Merge code.** New branch from `chore/publication-cleanup`, merge `feat/phase3d-campaign` (name the branch e.g.
   `phase3d-publication`; ask if unsure). Add `transition_fit.py` if missing. Verify: `tests/` suite passes (NOT the
   cluster-only ED tests; never run TC training/ED locally); viewer per-plane JSONs from the merged branch are
   byte-identical to the current v117 exports (scratchpad `viewer/viewer_hy*.json`), apart from the `generated` stamp.
2. **Loose files → commit** (USER: include session b8's files — BLOG.md 2026-09-27 entry, `phase3d_trivial_wall.ipynb`,
   plotly in the `[analysis]` extra). Triage every untracked/modified phase3d item listed above; skip fermionic /
   arch_compare. Notebooks: nbstripout filter; savefig lines stay commented.
3. **Results.** Commit `results/phase3d` finals with the `weights` field stripped (expected ~5-6 MB) + snapshot/finaleval
   JSONs; commit `results/transitions`, `results/hy_cuts_L4`, `results/hy_axis_L4`; pull + commit the small
   `manifests/*.tsv` (per-point run records). Full JSONs WITH weights → gitignored `data/archive/phase3d/` (the approved
   end-of-campaign archive of the final networks). Verify: every stripped file == original minus `weights`; viewer export
   from the stripped tree identical; archive has one network per final.
4. **Repoint** the main checkout + viewer build to the new branch; republish the viewer (same artifact URL) unchanged.
   Prepare a PR into `main` — the USER opens/approves it (outward-facing). Cluster repo stays on the old branch until the
   next launch; at that launch: export WANDB_ENTITY, re-prime the Pauli cache (cleanup changed the hamiltonian hash).
5. **Remove merged worktrees** (`git worktree remove` for the p3d/*, pc/*, agent leftovers; prune the tmp one) only
   after verification; branches stay in git + origin. Leave fermionic worktrees.

## Adversarial verification wave (after step 5, read-only agents, one lens each; fix CRUCIAL findings, re-verify)
1. Code equivalence (Opus): merged branch == campaign exports + tests pass; cleanup's verified tc3d unchanged by merge.
2. Data completeness (Sonnet): every $PSCRATCH final has a stripped twin in git with identical observables; archive complete.
3. Reproducibility (Sonnet): fresh clone of the branch → two commands → byte-identical viewer data, without cluster or
   `data/`; imports/paths/plotly OK.
4. Git hygiene (Sonnet): no large files/secrets (W&B keys), no fermionic/other-track files swept in, .gitignore covers
   `data/`, only merged worktrees removed.

## Reproducibility statement (for the user)
From committed results: deterministic regeneration of the viewer / 3D plot (review decisions live in phase3d_status.py
sets + phase3d_extras.json + the viewer's located()); fit error bars need a pinned numpy/scipy env (~12% otherwise).
From scratch: code + manifests rerun every point on Perlmutter; NQS is stochastic → same diagram within error bars,
not bit-for-bit (~850+ GPU-h at L=4). A publication-quality 3D figure script is still to be written (b8's plotly notebook
is a start).

## Execution record (2026-09-28, same session after /compact)
Done on `feat/phase3d-publication` (pushed): 6283ce9 merge (0 conflicts) · a07c604 loose files (b8's BLOG entry +
`phase3d_trivial_wall.ipynb` + plotly, `l10_feasibility.md`, 3 hy figures) · 55e6acc pull pipeline · bee981a results.
The main checkout now sits on this branch; 12 merged worktrees removed (+ the tmp one pruned); viewer republished as v118.

Corrections to the plan's facts, found while executing:
- **The bulk of a final JSON is the inline per-step learning curve `curve` (~260 KB), not network weights**; `weights`
  is only the `.mpack` path on $PSCRATCH (networks were never local). So the committed finals drop `curve` (a duplicate
  of `<name>.curve.json` in data/tc_nqs, 2470/2471 equal on step/energy/spread; the 74 legacy hy_cuts imports have no
  mirror file) and keep `weights`. results/phase3d = 16.4 MB; git pack 48 MiB.
- **Two stale finals** (h_y=1.0 electric h_x=0 cold points h_z 0.1147/0.1547, + finaleval): parked on the cluster by the
  2026-09-19 redo, copied instead of moved locally. Dropped (identical copies stay in redo_electric_20260919/). Only
  viewer change vs v117: that cut's O_FM h_c 0.164 ± 0.046 → 0.169 ± 0.027 (M_z jump unchanged at 0.18).
- Future pulls: plane JSONs → gitignored `data/archive/phase3d/` (raw) → `phase3d_strip_sync.py` → `results/phase3d/`;
  `phase3d_reseed.py park --base results/phase3d data/archive/phase3d`. Manifest backups (*.bak_*, manifests_bak/) are
  gitignored and kept in the archive only; W&B run dirs likewise.
- Fresh-clone reproducibility = identical viewer data except the stamps `generated` and `data_as_of` (max file mtime ->
  checkout time in a clone) and the learning-curve panels (need data/tc_nqs). Checked on a local clone.
- Tests: the 12 laptop-safe tests pass + phase3d_status selftest; test_grad_guard / test_resume_guard /
  test_speed_levers take VMC steps (cluster-only by rule) — the merge touched no tc3d/tests/nersc file.
- `.claude/worktrees/agent-a640050e99c369032` had 261 lines of uncommitted tc3d edits (Aug 7, early fermionic
  phase-head work) matching no branch: snapshotted as 7da0340 on its own branch `worktree-agent-a640050e99c369032`
  (+ patch in data/archive/main_checkout_pre_consolidation_20260928/), then the worktree was removed (user OK).
  Fermionic worktrees (`toric-code-nqs-fsign`, `toric-code-nqs-ladder`) untouched.
- Main checkout's old local state: tracked edits in `git stash` ("main checkout local edits before consolidation …";
  PR#5-era doc drafts superseded on this branch, the fermionic 2026-08-20 BLOG entry, notebook outputs); colliding
  untracked copies in `data/archive/main_checkout_pre_consolidation_20260928/`.

## Adversarial wave (2026-09-28) — no CRUCIAL/MAJOR findings; all claims hold
- Code equivalence (Opus): merge touched nothing in tc3d/tests/nersc; phase3d_status.py = both sides (AST-equal to the
  campaign version modulo docstrings); transition_fit.py a strict superset; 12 tests + selftest pass; viewer = v117 except
  the two dropped points' cut (points 37→35, O_FM fit, jump sharpness; jump h_c unchanged). Fixed its MINORs: a mid-write
  (non-parsing) final no longer aborts the pull (skipped, retried next pull); plane-root JSONs sync; park docs name both
  trees. Left: data_as_of / STATUS "last" come from mtimes (checkout time in a fresh clone); 1 of 2543 raw finals was
  compact JSON and is re-indented (content equal). Informational: cleanup removed train/sweep flags (--hidden,
  --vanilla_depth, --noninv_random, --radius_plaq, --hz_preset; --arch limited) — nothing in the campaign uses them.
- Data completeness (Sonnet): every archive/cluster final has an exact curve-stripped twin; 0 stale duplicates left;
  manifests complete; summary/STATUS regenerate identically; the 74 legacy imports never had curves.
- Fresh-clone reproducibility (Sonnet): GitHub clone, no data/ → all 17 planes + summary byte-identical (modulo stamps,
  curve); the 3 consuming notebooks run headless with 0 errors. Fixed: analysis/notebooks/figures/ gitignored.
- Git hygiene (Sonnet): no large/secret/out-of-scope files; notebooks stripped; merge into main is clean (main is an
  ancestor). Fixed: prime_pauli_cache.py carried over from p3d/l10-feasibility (its --check passes at L=2,3 OBC).
