"""
Minimal Weights & Biases logger for 3D toric code NQS experiments.

Usage:
    from tc3d.wandb_logger import init_run, log_step, finish_run

    run = init_run(
        project="approx-sym-3D-TC",
        entity="YOUR_WANDB_USERNAME",
        config={"Lx": 2, "Ly": 2, "Lz": 2, "bc": "PBC", "hx": 0.0, ...},
        name="cnn_inv_L2_h0",
        tags=["toy", "L=2", "h=0", "CNN_inv_only"],
    )

    for step in range(n_iter):
        driver.advance(1)
        E = vs.expect(Ham)
        log_step(run, step, E, vs)

    finish_run(run, vs, Ham, geo,
               extra={"runtime_s": elapsed,
                      "exact_E0": -32.0,
                      "vertex_flip_diff": 1e-7,
                      "translation_diff": 1e-7})

Install once:  pip install wandb
Auth once:     wandb login   (or set WANDB_API_KEY env var)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def init_run(
    project: str,
    entity: Optional[str],
    config: Dict[str, Any],
    name: Optional[str] = None,
    tags: Optional[list] = None,
    group: Optional[str] = None,
    id: Optional[str] = None,
    resume: Optional[str] = None,
    dir: Optional[str] = None,
):
    """Initialize a wandb run. Returns the run object.

    `group` ties all runs of a sweep together (one wandb group) so they can be
    compared/plotted side by side — pass the SLURM job name for an array sweep.

    `id`/`resume` make a requeued (timed-out) job continue the *same* wandb run
    instead of opening a new one: pass a deterministic `id` (e.g. a hash of the
    run name) with `resume="allow"`. Works in offline mode too — the resumed
    chunk merges into the same run on `wandb sync`.

    `dir` sets the parent for wandb's own `wandb/` folder (where offline runs are
    written). Pass the run's `out_dir` so offline runs collect predictably under
    `$OUT_DIR/wandb/` — otherwise wandb uses the process CWD, scattering them into
    whatever directory each SLURM job started in and breaking `wandb sync $OUT_DIR/wandb`.
    """
    import os

    # The wandb-core subprocess inherits these macOS allocator-debug vars and
    # spams stderr ("MallocStackLogging: can't turn off ...") which interleaves
    # with training stdout. Scrub them before wandb.init spawns the subprocess.
    for _v in ("MallocStackLogging", "MallocStackLoggingNoCompact",
               "MallocScribble", "MallocPreScribble"):
        os.environ.pop(_v, None)

    import wandb

    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=name,
        tags=tags,
        group=group,
        id=id,
        resume=resume,
        dir=dir,
        reinit=True,
    )


def log_step(run, step: int, E, vs, exact_E0: Optional[float] = None,
             ref_E: Optional[float] = None, ref_sig: Optional[float] = None) -> None:
    """Log per-step VMC scalars.

    Args:
        run: wandb run object
        step: integer iteration index
        E:   netket Stats object from vs.expect(H)
        vs:  variational state (for sampler/acceptance info)
        exact_E0: if given, also log the figure of merit
                  delta = |E - E_exact| / |E_exact| (and the absolute energy
                  error) so the FOM is plotted live and comparable across runs.
    """
    acc = float(vs.sampler_state.n_accepted) / max(1, float(vs.sampler_state.n_steps))

    e_mean = float(np.real(E.mean))
    e_var = float(np.real(E.variance))
    metrics = {
        "step": step,
        "energy":              e_mean,
        "energy_error":        float(np.real(E.error_of_mean)),
        "energy_variance":     e_var,
        # reference-free per-step quality: Vscore = N*Var[H]/E^2 -> 0 for an eigenstate
        "vscore":              (vs.hilbert.size * e_var / e_mean**2
                                if e_mean != 0 else float("nan")),
        "tau_corr":            float(np.real(E.tau_corr)),
        "R_hat":               float(np.real(E.R_hat)),
        "mcmc_acceptance":     acc,
    }
    if exact_E0 is not None:
        e = float(np.real(E.mean))
        metrics["energy_abs_err"] = abs(e - exact_E0)
        metrics["delta"] = abs(e - exact_E0) / abs(exact_E0)
    if ref_E is not None:
        # SIGNED benchmark gap (e.g. vs QMC): + above the reference, - below.
        metrics["dE_ref"] = e_mean - ref_E
        if ref_sig:
            metrics["dE_ref_sig"] = (e_mean - ref_E) / ref_sig

    run.log(metrics, step=step)


def finish_run(run, vs, Ham, geo, extra: Optional[Dict[str, Any]] = None,
               observables: Optional[Dict[str, Any]] = None) -> None:
    """End-of-run logging: stabilizer expectations + any extras you compute.

    If `observables` is given (e.g. from `validation.nqs_observables`), those are
    logged directly and the per-stabilizer recompute below is skipped — this is
    the cheap path used by the training pipeline. Otherwise <A_v>/<B_p> are
    computed here with ~1 vs.expect per stabilizer (keep the system small).
    """
    if observables is not None:
        summary = dict(observables)
        if extra is not None:
            summary.update(extra)
        for k, v in summary.items():
            run.summary[k] = v
        run.finish()
        return

    import netket as nk

    A_means = []
    for v in geo.vertex_all:
        op = 1
        for i in v:
            if i == -1:
                continue
            op = op * nk.operator.spin.sigmax(vs.hilbert, int(i))
        A_means.append(float(np.real(vs.expect(op).mean)))

    B_means = []
    for p in geo.plaq_all:
        op = 1
        for i in p:
            if i == -1:
                continue
            op = op * nk.operator.spin.sigmaz(vs.hilbert, int(i))
        B_means.append(float(np.real(vs.expect(op).mean)))

    summary = {
        "A_v_mean": float(np.mean(A_means)),
        "A_v_min":  float(np.min(A_means)),
        "A_v_max":  float(np.max(A_means)),
        "B_p_mean": float(np.mean(B_means)),
        "B_p_min":  float(np.min(B_means)),
        "B_p_max":  float(np.max(B_means)),
        "n_params": int(vs.n_parameters),
    }
    if extra is not None:
        summary.update(extra)

    for k, v in summary.items():
        run.summary[k] = v

    run.finish()
