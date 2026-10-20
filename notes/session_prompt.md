# Session prompt — 3D fermionic toric code NQS, DMRG-benchmarked (written 2026-10-19 night)

Read first: `CLAUDE.md`, `BLOG.md` entries dated 2026-10-19 (three of them: the DMRG wave, the
hx-floor fix, the architecture explainer), `notes/transition_mapping_recipes.md`.

## State of play
- **Reference.** Collaborator DMRG, 3×3×3 OBC (N=54), bosonic + fermionic, single-field axes,
  DMRG h ∈ {0.1,0.2,0.4,0.8,1.6} = ours {0.05,0.1,0.2,0.4,0.8}. CONVENTION: h_ours = h_DMRG/2,
  **hx_ours NEGATIVE** (odd-weight corner stars), hz/hy positive; obs: sx↔avgX, sy↔−avgY,
  sz↔−avgZ, A/B↔−avgA/−avgB. Banked: `results/dmrg_L3_obc/dmrg_L3_obc_reference.json`.
- **Notebook (the deliverable).** `analysis/notebooks/dmrg_L3_benchmark.ipynb` — energy/N, ΔE/N,
  |ΔE|/|E| and the field-responsive observables per axis, NQS arms vs DMRG. Config knobs:
  SHOW_ARMS, JITTER, N_SITES, SHARE_Y, AXES_FOR, XMAX. Table: `analysis/scripts/dmrg_compare.py`
  (rule |ΔE| < max(3σ, 1e-5|E|)). Pull: `bash analysis/scripts/dmrg_pull.sh`.
- **Bosonic.** hx, hz match (≤2.3σ; hz=0.2 is 3–7σ high = 500-step under-convergence). hy axis
  FAILS in every lane (complex trunk pinned at the real code state: exact stationary point;
  ⟨σy⟩ ≈ 0). Fix proposals in BLOG (saddle-breaking warm start; σ^y-basis sign frame).
- **Fermionic.** Arms: A = pt2 sign frame + κ=6 (exact on hz; projector floor on hx);
  B = frozen analytic head + κ=6 (hz only; exact there); C = sign-blind complex (control);
  R/Rs = sign frame + amplitude head `--amp_head pt2 [--amp_eval recover|self] [--amp_learn]`,
  κ=0, warm from the A checkpoint — Rs matches DMRG at hx ≤ 0.1, rel 1.7e-4 at 0.2.
- **Code.** Bosonic/analysis side: branch `feat/dmrg-L3-benchmark` (main checkout, pushed).
  Fermionic side: branch `feat/dmrg-L3-ftc` in the worktree `../toric-code-nqs-fsign`
  (amplitude head `tc3d/amplitude_head.py`, hx-sign decoder fix, launcher
  `nersc/submit_dmrg_L3_ftc.sh` with ARMS / HX_PTS / HZ_PTS / DEP / MODE=smoke). The cluster
  copy is `~/toric-code-nqs-fsign` (shipped by git bundle). **Push this branch to GitHub** if
  not yet done (`git push -u origin feat/dmrg-L3-ftc` from the worktree).
- **Running (2026-10-19 night).** Strong-field fTC wave: arrays 60025730 (A), 60025731 (B),
  60025733 (Rs, afterok A), 60025735 (C). Outputs `$PSCRATCH/tc_nqs/dmrg_L3/ftc/`. On landing:
  pull, raise XMAX to 1.5 in the notebook, inspect the hx/hz rows, BLOG the result.

## Rules that bit us (keep)
- Laptop: one python process at a time, no test suites, no ED above L=2 OBC (it crashed once).
- Every submission gets a 30-min log-tail tick (CronCreate) that reads energies, not just squeue.
- Walltimes just enough (A 1:00, B 1:45, C 2:00 for 200 steps); framed final eval ≈ 19 min/run.
- Never run B or A with κ=6 on the hx axis expecting DMRG agreement; use Rs.

## Next
1. Mixed (hx,hz) points — ask the collaborator for (3,3,Lz) DMRG at chosen (hx,hz); run Rs.
2. hz axis beyond 1.4 toward the fermionic electric transition (frame exact; ask for DMRG at
   DMRG h = 2.4–4.0).
3. Two-arm gated ansatz (T⁺ with in-network decoder) for the hx crossover; then complex head /
   σ^y frame for hy.
