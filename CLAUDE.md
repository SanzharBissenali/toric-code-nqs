# CLAUDE.md — project orientation

Neural-quantum-state + exact-diagonalization study of perturbed toric codes
(2D surface code and 3D toric code) and their topological→trivial transitions
under a uniform field, $H = -J\sum_v A_v - J\sum_p B_p - h_x\sum_i\sigma^x_i
- h_z\sum_i\sigma^z_i$.

## Layout

| Path | Role |
|---|---|
| `model/` | **2D** surface/toric code + shared numerics. |
| `model/geometry.py` | 2D toric / surface-code geometry. |
| `model/exact_diag.py` | **Shared** matrix-free (Numba) Hamiltonian + Pauli-string expectations. Geometry-agnostic: consumes any object with `.N`, `.vertex_all`, `.plaq_all`. Used by both the 2D and 3D sweeps. |
| `Three_TC/model/` | **3D** toric code (geometry, NetKet Hamiltonian, CNN). |
| `Three_TC/model/fermionic_decoration.py` | 3D **fermionic** toric code: plaquette decoration + dressed Wilson-loop order parameter. |
| `colab/fermionic_TC_colab.ipynb` | Sole remaining Colab notebook: self-contained 3D **fermionic** numba sweep (unique code not yet in the repo; open TODO in `nersc/README.md` to port it to `run_fermionic_sweep.py`). |
| `2D_TC_phase_diag.ipynb` | Main analysis notebook (2D + 3D bosonic + 3D fermionic). |
| `notes/handoff_fermionic_tc.md` | Detailed physics write-up of the fermionic model + order parameter. |

2D and 3D are separate packages; `exact_diag.py` is the one shared module and
stays in `model/`. The 3D fermionic code is self-contained (numpy + bit ops).

## Fermionic toric code (one-paragraph summary)

The bosonic plaquette $B_p=\prod_{e\in\partial p}\sigma^z_e$ is decorated to
$\tilde B_p = B_p\,\sigma^x_{e_+}\sigma^x_{e_-}$, with the two $\sigma^x$ on the
perpendicular corner edges at the $(+a,+b)$ corner $/{+}$perp side and the
$(-a,-b)$ corner $/{-}$perp side (a body diagonal). This is the minimal
decoration that stays a commuting-stabilizer model and makes the point
excitation a **fermion**. The bare $\sigma^z$ Wilson string is no longer
conserved, so it is **dressed with $\sigma^x$** (`dressed_string`, a small GF(2)
solve): the closed loop becomes a conserved Wilson loop $W$; the open string
provably cannot be made flux-free — each endpoint carries a charge **and** a flux
(the fermion). Detection uses the Fredenhagen–Marcu ratio
$O_{FM}=\langle S\rangle/\sqrt{|\langle W\rangle|}$ plus the gap and
$\langle M_z\rangle$. See `notes/handoff_fermionic_tc.md` for the derivation.

## Working rules

- **Never run 3D toric-code ED/sweeps locally.** $L=2$ PBC is $2^{24}$ states
  (~2.7 GB Lanczos workspace) — it OOMs the 8 GB dev machine. Verify 3D work with
  cheap proxies only: geometry construction, `verify_xz_commutation`,
  `dressed_string` flux counts, tiny-$N$ checks of `expect_*`. Run the actual
  `eigsh` sweeps on Colab (`colab/fermionic_TC_colab.ipynb`).
- Code style: concise, readable, one clear purpose per function; comments only
  where they add signal. Prefer editing existing modules over new files.
- Validate physics with a small inline check rather than asserting it works.
- The `.venv/` here has numpy/scipy/numba/netket; invoke as `.venv/bin/python`.
- **Tests** (self-contained, no pytest config — each runs standalone via a `_path.py`
  sys-path shim): `cd Three_TC/tests && ../../.venv/bin/python test_geometry.py` (likewise
  `test_exact_diag.py`, `test_fm.py`, `test_fidelity.py`, `test_renyi_units.py`,
  `test_hamiltonian.py`). Cheap ones run locally; 3D ED tests obey the no-3D-locally rule above.
- **VMC entry points** (argparse `--help` on each): `Three_TC/train.py` (single run,
  checkpoint/resume-safe), `Three_TC/sweep.py` (batch N field points per process),
  `Three_TC/fm.py` (extract `O_FM(field)` + transition fit for one L). Scale by `--L`.
- Notebook outputs are stripped on commit by an nbstripout filter (`.gitattributes`); a fresh
  clone needs `nbstripout --install` (in `.venv`) for the filter to run.

## Cluster (NERSC Perlmutter) I/O

- Login `sanzharb@perlmutter.nersc.gov` (NERSC user is `sanzharb`, NOT the local
  Mac user `sanzhar123` — always prefix the host or ssh prompts the wrong account);
  scratch `$PSCRATCH = /pscratch/sd/s/sanzharb`; local repo
  `/Users/sanzhar123/Desktop/Approximate-Symmetries-TC-main`.
- Auth is NERSC **sshproxy** (`sshproxy -u sanzharb` → 24h cert at `~/.ssh/nersc`); it is
  MFA, so a human re-mints it when it expires. GPU jobs charge account `m5340_g`; conda
  envs `tc-nqs` (GPU NQS) / `2dtc` (2D), built by `nersc/setup_conda_{gpu,cpu}.sh`.
- Submit through the `nersc/submit_*.sh` wrappers (`sbatch`, env-var driven, resume-safe);
  `nersc/CAMPAIGN.md` is the canonical FSS config spec, `nersc/README.md` the how-to.
- FM curves live at `$PSCRATCH/tc_nqs/phase_hx${HX}/fm_L*_hx${HX}_${PLACEMENT}.json`
  (extracted by `nersc/extract_fm.sh`). Pull each **placement** into its own local
  dir — `analysis/plot_phase_diagram.py` globs `fm_L*.json`, so mixing placements
  in one dir double-counts an L. Pull format (quote the remote path — zsh would
  otherwise try to expand the `*` locally and abort with `no matches found`):
  ```
  rsync -avz 'sanzharb@perlmutter.nersc.gov:/pscratch/sd/s/sanzharb/tc_nqs/phase_hx${HX}/fm_L*_hx${HX}_${PLACEMENT}.json' \
    /Users/sanzhar123/Desktop/Approximate-Symmetries-TC-main/results/phase_hx${HX}_${PLACEMENT}/
  ```
  (Legacy runs used `fm_L*_hx${HX}.json` with no `_${PLACEMENT}` suffix, pulled into
  `results/phase_hx${HX}/`.)

### Cluster autonomy (agreed permission charter)

When Claude has cluster access it may act autonomously within these bounds:

- **Jobs — full autonomy.** May `sbatch` submit, tail/monitor, auto-resubmit on timeout
  (AUTO_RESUBMIT), and `scancel` **its own** jobs, then report outcomes. Never touch jobs
  it did not launch.
- **Compute — moderate ceiling.** Single runs and small sweeps proceed freely (≤ the 5h
  walltime cap + chained resubmits). **Ask first** before launching a full multi-L phase
  campaign or running more than ~4 concurrent jobs.
  *Current gate (until lifted): autonomous submission is limited to `gpu_debug` smokes;
  production/campaign runs need explicit per-request approval.*
- **Script edits — commit to a feature branch.** May edit/create `nersc/` scripts and commit
  to the current feature branch. **Never commit to `main`.**
- **Data — rsync back, commit summaries only.** Write job outputs under `$PSCRATCH`, rsync
  results into `results/`, and commit only small derived artifacts (curve/fit JSONs, plots).
  Raw checkpoints / large data stay gitignored (`*.mpack`, `outputs/`, `wandb/`) — never committed.
- The reminders above still bind: NERSC user is `sanzharb` (prefix the host); one placement per
  results dir; quote remote globs against zsh expansion.
