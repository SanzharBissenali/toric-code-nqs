# CLAUDE.md — project orientation

Neural-quantum-state study of the **3D toric code** (bosonic & fermionic) under
uniform fields, $H = -J\sum_v A_v - J\sum_p B_p - h_x\sum_i\sigma^x_i - h_z\sum_i\sigma^z_i$,
mapping topological→trivial transitions with an approximately-symmetric CNN ansatz.

**The two first-class experiment tracks** (everything in the tree serves these):
1. **NQS hyperparameter tuning** — `tc3d/train.py --dual_basis` (+ `sweep.py`) in the
   sign-problem-free regime across system sizes; `--ref_E/--ref_sig` streams the signed
   per-step gap against a benchmark energy.
2. **QMC validation** — ParaToric (primary) + PMRQMC (cross-check) via
   `analysis/paratoric_driver.py` / `analysis/export_pmrqmc.py`; computes energy,
   stabilizer, and magnetization expectations. References in `results/qmc_*/`.

The 2D surface-code implementation this grew out of lives at git tag **`2d-final`**
(not in the tree). North-star extraction (FM order parameter, S2-Rényi) is kept:
`tc3d/fm.py`, `tc3d/renyi.py`.

## Layout

| Path | Role |
|---|---|
| `tc3d/` | Single flat package. Geometry/Hamiltonian/networks + `builders.py` (config → geometry+H+ansatz+sampler+vstate, arch registry, shared `run_loop`), `sampler.py` (cluster-update MCMC rules), `train.py`/`sweep.py` (entry points, checkpoint/resume-safe), `fm.py`/`renyi.py` (order-parameter extraction), `validation.py`, `exact_diag.py` (matrix-free Numba ED, used by QMC `--verify`), `io.py`/`config.py` (checkpoint I/O, device probe). |
| `tests/` | Standalone tests: `cd tests && ../.venv/bin/python test_geometry.py` (package is pip-installed editable; no path shims). **Exception:** `test_exact_diag.py` is an ED *reference generator* (L=2 PBC Lanczos, ~2.7 GB) — cluster/Colab only, never local. |
| `analysis/` | Pure post-processing over `results/` JSONs + QMC drivers + `exact_benchmarks.py` (analytic series, 42 self-checks). `plot_phase_diagram.py` consumes `tc3d.fm`'s output JSONs (NetKet-free; not imported by tc3d). |
| `nersc/` | Submit wrappers (`submit_nqs_gridinv.sh` single run, `submit_nqs_batch.sh` batched, `submit_nqs_{hz,hx}_sweep.sh` arrays), campaign/extract drivers, `CAMPAIGN.md` (canonical FSS config spec), `README.md` (how-to), `check_hxsweep.sh` + `analysis/check_convergence.py` (QA gate before extraction). |
| `colab/` | `dual_basis_colab.ipynb` (L=4 tuning/AB), `qmc_benchmarks_colab.ipynb`, `fermionic_TC_colab.ipynb` (unique fermionic numba sweep, not yet ported). |
| `paper/` | Manuscript; PDF gitignored. |
| `notes/` | `log_and_plan.md` = frozen historical design record (living log is root `BLOG.md` — read it first); `nqs_architecture.md` (authoritative arch write-up), `handoff_fermionic_tc.md` (fermionic model + dressed Wilson loop), `training_cli.md`, `training_gotchas.md`, `session_kickoff.md`. |

## Working rules

- **Never run 3D toric-code ED/sweeps locally.** $L=2$ PBC is $2^{24}$ states
  (~2.7 GB Lanczos workspace) — it OOMs the 8 GB dev machine. L=2 **OBC** (N=12) is
  fine. Verify with cheap proxies: geometry construction, `verify_xz_commutation`,
  tiny-$N$ checks, the `tests/` suite, analytic anchors in `analysis/exact_benchmarks.py`.
- Code style: concise, readable, one clear purpose per function; comments only
  where they add signal. Prefer editing existing modules over new files.
- **Replies: be concise and to the point.** Lead with the answer/result.
- Validate physics with a small inline check rather than asserting it works.
- **Adversarial audit gate:** after any LARGE/significant change to NQS training,
  measurement/observable code, or the QMC stack (estimators, drivers, ParaToric
  C++ patches), launch independent adversarial agents — tight one-lens prompts,
  minimal context, read-only on the tree — to break-or-validate the change
  BEFORE production runs depend on it. Fix CRUCIAL findings and re-verify.
  (Precedent: the 2026-08-10/11 FM audits caught 3 crucial defects pre-campaign.)
- **Notebook figures are NOT auto-saved**: plotting cells end with `plt.show()` and
  keep their `plt.savefig(<FIGS>/<name>.png, dpi=300, bbox_inches="tight")` line
  present but **commented out** — the user uncomments and runs it themselves when a
  figure is worth committing. Never enable savefig by default.
- The `.venv/` has numpy/scipy/numba/netket and `tc3d` installed editable
  (`pip install -e ".[analysis]"` — the extra carries matplotlib/jupyter/nbstripout);
  invoke as `.venv/bin/python`.
- **Entry points** (argparse `--help` on each): `python -m tc3d.train` (single run,
  checkpoint/resume-safe, `--dual_basis`, `--ref_E/--ref_sig`), `python -m tc3d.sweep`
  (batch N field points per process), `python -m tc3d.fm` / `tc3d.renyi` (extraction).
- Notebook outputs are stripped on commit by an nbstripout filter (`.gitattributes`);
  a fresh clone needs `nbstripout --install` (in `.venv`). The filter is configured
  with a repo-relative path — do not replace it with an absolute one.
- macOS ships an HTTP `head` that shadows GNU head — piping to `head` errors.
  Use `grep -m N`, `sed -n`, or the Read tool instead.
- Background/Monitor scripts run under **zsh**: unquoted `$VAR` does NOT
  word-split (`for x in $JOBS` sees one word) and bash-isms like `declare -A`
  misbehave — use literal lists in loops or `printf | while read`.
- **Editing notebooks:** NotebookEdit trips "File modified since read" when the notebook
  is open in Jupyter. Edit via a small `json.load → replace → json.dump` script; verify
  headlessly with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb>`.
  nbstripout is **not idempotent** on old-format notebooks — a freshly normalized
  notebook may show phantom `git status` modifications until committed once.

## QMC benchmark pipeline (track 2 specifics)

- ParaToric + PMRQMC clones live in gitignored `external/`; rebuild with the
  force-added `external/build_paratoric_local.sh` (brew LLVM + boost, libc++ —
  do not mix with gcc). Import check must touch
  `paratoric.extended_toric_code.get_sample` (the `__init__` swallows load failures).
- **Sampler hygiene is the trap:** under-decorrelated runs finish cleanly and return
  biased energies with confident error bars. Always pass the exact-anchor validation
  ladder (`paratoric_driver.py --validate`) before trusting new numbers; N_BETWEEN
  ≈ 120 updates/edge and thermalization must scale with β.
- Driver requirements for precision work: equal-weight combine, N_BETWEEN ∝ β,
  fresh `--seed0` per run, β=24 for <1e-13 thermal bias.
- **Never emit Boost.Log/iostreams from the ParaToric .so** — the cluster build
  is -static-libstdc++ vs dynamic-libstdc++ conda Boost (two C++ runtimes):
  formatting a log record segfaults batch workers (SIGSEGV in _M_insert<double>,
  seed-dependent via the tau>0.1·N warning gate). Diagnostics go through
  fprintf(stderr) — `external/paratoric_stdio_taulog.patch`, applied after the
  membrane patch in both build scripts.
- Near-h_c x-basis points need `NBS_MULT=8` (χ²_red ran 3–6 at ×4); membrane
  den_z=∞ holds only on the hz=0 line (conserved-A_v coboundary) — probe den_z
  once before any two-field membrane campaign. Strings: den_z ~900–1300 at
  two-field near-critical points, L=8–12.

## Cluster (NERSC Perlmutter) I/O

- Login `sanzharb@perlmutter.nersc.gov` (NERSC user is `sanzharb`, NOT the local Mac
  user — always prefix the host); scratch `$PSCRATCH = /pscratch/sd/s/sanzharb`.
- Auth is NERSC **sshproxy** (`sshproxy -u sanzharb` → 24h cert at `~/.ssh/nersc`); MFA,
  so a human re-mints it. GPU jobs charge account `m5340_g`; conda env `tc-nqs`
  (built by `nersc/setup_conda_gpu.sh`).
- Submit through `nersc/submit_*.sh` wrappers (`sbatch`, env-var driven, resume-safe);
  `nersc/CAMPAIGN.md` is the canonical FSS config spec.
- **Debug fast, produce gated:** reproduce crashes on `-q debug` (full node,
  30 min, minutes-scale queue); submit production with
  `--dependency=afterok:<validation-job>` so it accrues queue age but only
  starts on a pass. Post-mortem recipe: `srun bash -c 'ulimit -c unlimited;
  exec python …'` — cores land in cwd (`core.%e.%p.%h`); gdb at
  /global/common/software/nersc/bin/gdb.
- `sacct -X --format=SubmitLine%220` recovers past sbatch commands; env-var
  knobs are NOT recorded (they travel via sbatch's environment export) —
  reconstruct them from the wrapper's defaults.
- **Sweep families:** `phase_hz{HZ}/L*` = fixed h_z, sweep h_x (magnetic cut);
  `phase_hx{HX}/L*` = fixed h_x, sweep h_z (electric cut). Aggregate per-run JSONs on
  the login node into `energy_L*.json`; pull each **placement** into its own local
  `results/` dir (mixing placements double-counts an L). Quote remote globs — zsh
  expands `*` locally otherwise.

### Cluster autonomy (agreed permission charter)

- **Jobs — full autonomy.** May `sbatch`, monitor, auto-resubmit on timeout
  (AUTO_RESUBMIT), and `scancel` **its own** jobs. Never touch jobs it did not launch.
- **Compute — moderate ceiling.** Single runs / small sweeps proceed freely (≤5h
  walltime cap + chained resubmits; never propose more). **Ask first** for multi-L
  campaigns or >~4 concurrent jobs.
  *Current gate: autonomous submission limited to `gpu_debug` smokes;
  production/campaign runs need explicit per-request approval.*
- **Script edits — commit to a feature branch. Never commit to `main`.**
- **Data — rsync back, commit summaries only.** Outputs under `$PSCRATCH`, rsync into
  `results/`, commit only small derived artifacts. Raw checkpoints stay gitignored.
