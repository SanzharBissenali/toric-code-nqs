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
