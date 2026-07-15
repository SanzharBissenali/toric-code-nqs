"""
Three_TC/train.py
─────────────────────────────────────────────────────────────────────────────
Standalone, config-driven training pipeline for the 3D toric code (bosonic +
fermionic) — the 3D analogue of the inherited 2D `main.py`.

Inputs: hyperparameters, system size L, Hamiltonian fields (hx, hy, hz), and
naming. Outputs: model weights (`.mpack`), a local run JSON (config + final
expectation values + training curve), and W&B curves/observables.

Usage (notebook / Python):
    from Three_TC.train import train
    res = train({"L": 2, "model": "fermionic", "arch": "ToricCNN_full",
                 "hx": 0.2, "hz": 0.2, "n_iter": 200, "wandb": False})

Usage (CLI / cluster):
    python -m Three_TC.train --L 2 --model fermionic --arch ToricCNN_full \
        --hx 0.2 --hz 0.2 --n_iter 200 --no_wandb

Construction and the optimization loop are shared with `validation.py` via
`Three_TC.builders`, so the trained model is exactly what validation scores.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)  # float64 SR/QGT (esp. on GPU)

from Three_TC.builders import build_state, run_loop, with_defaults, DivergenceError
from Three_TC.validation import nqs_observables
from Three_TC.utils.wandb_logger import init_run, log_step, finish_run
from utils.config import setup_environment
from utils.io import save_model, load_weights


TRAIN_DEFAULTS: Dict[str, Any] = {
    "n_iter": 100, "dt": 2e-2, "diag_shift": 2e-4, "lr_min": 2e-3,
    "out_dir": "outputs", "wandb": True,
    "wandb_project": "approx-sym-3D-TC",
    "wandb_entity": "models-california-institute-of-technology-caltech",
    "tags": None, "name": None,
    # Cluster/timeout robustness: checkpoint the weights + energy curve to disk
    # every `checkpoint_every` steps (0 disables) so a killed job keeps its
    # progress; `--resume` continues from the last checkpoint; `wandb_offline`
    # logs to a local dir (NERSC compute nodes have no outbound network).
    "checkpoint_every": 10, "resume": False, "wandb_offline": False,
    # Self-healing divergence guard (see run_loop): detect the SR blow-up, roll
    # back to the last sane params, re-seed chains + boost diag_shift, retry.
    "grad_guard": True, "spike_factor": 10.0, "max_rollbacks": 5,
    "rollback_shift_boost": 10.0, "rollback_cooldown": 20, "baseline_window": 20,
}

# Hardcoded reference points from threed_bosonic.json (L=2 PBC bosonic, hx=0.2,
# J=1): label -> (h_z, E_exact, gap). Selected with --hz_preset; sets both the
# field and the E_exact used for the delta figure of merit.
HZ_PRESETS: Dict[str, tuple] = {
    "hard": (0.1184210526315789, -32.2968435820, 0.062),   # small gap (hardest)
    "mid":  (0.3157894736842105, -33.9620095053, 0.943),   # validated point
    "easy": (0.5526315789473684, -38.5935624665, 3.452),   # large gap (easiest)
}


def _run_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("name") or (
        f"{cfg['model']}_{cfg['arch']}_L{cfg['L']}_hx{cfg['hx']}_hz{cfg['hz']}")


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train one NQS run from a config dict; return a results dict.

    Side effects: writes `{out_dir}/{name}.mpack` (weights) and
    `{out_dir}/{name}.json` (config + observables + curve); logs to W&B if
    `config['wandb']`.
    """
    cfg = with_defaults({**TRAIN_DEFAULTS, **config})
    if cfg["hy"] != 0.0:
        raise NotImplementedError(
            "hy != 0 (sign problem) needs a complex ansatz; not supported yet.")

    # h_z preset -> set the field AND the E_exact used for the delta FOM.
    # --exact_E0 (or config["exact_E0"]) is the manual fallback at any h_z.
    if config.get("hz_preset"):
        hz, e0, _gap = HZ_PRESETS[config["hz_preset"]]
        cfg["hz"], cfg["exact_E0"] = hz, e0
    else:
        cfg["exact_E0"] = config.get("exact_E0")
    exact_E0 = cfg.get("exact_E0")

    # Device detection (reused util): picks GPU if present and returns the
    # default chain count (1024 GPU / 16 CPU). An explicit --n_chains still wins.
    _gpu, _node, n_chains_auto = setup_environment()
    is_gpu = n_chains_auto > 16          # setup_environment: 1024 GPU / 16 CPU
    if "n_chains" not in config:
        cfg["n_chains"] = n_chains_auto

    name = _run_name(cfg)
    cfg["name"] = name
    os.makedirs(cfg["out_dir"], exist_ok=True)
    weights_base = os.path.join(cfg["out_dir"], name)
    ckpt_base    = f"{weights_base}.ckpt"          # periodic weights checkpoint
    curve_path   = f"{weights_base}.curve.json"    # live energy curve + step count

    # NERSC compute nodes have no outbound network -> log W&B to a local dir and
    # `wandb sync` later from a login node. Must be set before wandb is imported.
    if cfg.get("wandb_offline"):
        os.environ["WANDB_MODE"] = "offline"

    geo, hi, Ham, vs, xz_stabs = build_state(cfg)

    # --- warm start from a NEIGHBOUR's weights (hysteresis sweeps) --------------
    # Unlike --resume (which continues THIS run from its own checkpoint), --init_from
    # seeds the parameters from another converged run's {base}.mpack so a directed
    # field sweep carries its phase across points. Applied before the resume block,
    # so an own-checkpoint resume (timeout requeue) still wins and overwrites this.
    init_from = cfg.get("init_from")
    if init_from and os.path.exists(f"{init_from}.mpack"):
        vs = load_weights(vs, init_from)
        print(f"[train] warm start: loaded weights from {init_from}.mpack", flush=True)
    elif init_from:
        print(f"[train] --init_from {init_from}.mpack not found; cold start.", flush=True)

    # Resolved run metadata -> W&B config (and saved JSON): the param count and the
    # ACTUAL sampler sweep size. build_sampler defaults n_sweeps to geo.N*2 when it
    # is unset, so without this the raw config would log n_sweeps=None.
    cfg["n_params"] = int(vs.n_parameters)
    cfg["n_sweeps"] = int(vs.sampler.sweep_size)
    print(f"[train] {name}: N={geo.N}  n_params={cfg['n_params']}  model={cfg['model']}"
          f"  n_chains={cfg['n_chains']}  n_sweeps={cfg['n_sweeps']}"
          + (f"  E_exact={exact_E0}" if exact_E0 is not None else ""))

    curve = {"step": [], "energy": [], "energy_err": [], "energy_spread": [], "delta": [],
             "timing": []}   # per-step {sample,grad,qgt,update,total} wall-clock (s)

    # --- resume a timed-out run from the last on-disk checkpoint ---------------
    # The checkpoint is {name}.ckpt.mpack (weights + sampler RNG) + {name}.curve.json
    # (completed step count + the energy history so far). We reload both, continue
    # the cosine-LR schedule from `start_step`, and append to the existing curve.
    start_step = 0
    if cfg.get("resume") and os.path.exists(curve_path):
        with open(curve_path) as f:
            ck = json.load(f)
        start_step = int(ck.get("completed_steps", 0))
        curve = ck.get("curve", curve)
        curve.setdefault("timing", [])   # checkpoints predating phase timing
        if os.path.exists(f"{ckpt_base}.mpack"):
            vs = load_weights(vs, ckpt_base)
        print(f"[train] resuming '{name}' from step {start_step}/{cfg['n_iter']}"
              f"  (loaded {len(curve['step'])} curve points)", flush=True)

    run = None
    if cfg["wandb"]:
        import hashlib
        wandb_id = hashlib.md5(name.encode()).hexdigest()[:12]  # stable across requeues
        run = init_run(project=cfg["wandb_project"], entity=cfg["wandb_entity"],
                       config=cfg, name=name, group=cfg.get("wandb_group"),
                       tags=cfg["tags"] or [cfg["model"], cfg["arch"], f"L={cfg['L']}"],
                       id=wandb_id, resume="allow", dir=cfg["out_dir"])

    ckpt_every = int(cfg.get("checkpoint_every", 0) or 0)

    def _write_checkpoint(step):
        """Persist weights + the energy curve so a kill/timeout loses nothing.

        Sanity-gated: never overwrite the last good checkpoint with a non-finite
        state, so `--resume` always restarts from a sane point (independent of the
        in-run guard, which normally keeps bad points out of `curve` entirely)."""
        if curve["energy"] and not (np.isfinite(curve["energy"][-1])
                                    and np.isfinite(curve["energy_spread"][-1])):
            print(f"  [ckpt] step {step}: last curve point non-finite; skip checkpoint.",
                  flush=True)
            return
        save_model(vs, ckpt_base, verbose=False)
        tmp = curve_path + ".tmp"                      # atomic: never a half-written file
        with open(tmp, "w") as f:
            json.dump({"completed_steps": step, "name": name, "config": cfg,
                       "curve": curve}, f)
        os.replace(tmp, curve_path)

    def on_step(step, E, vs):
        e   = float(np.real(E.mean))
        de  = float(np.real(E.error_of_mean))      # delta_E (MC error on the mean)
        var = float(np.real(E.variance))
        delta = abs(e - exact_E0) / abs(exact_E0) if exact_E0 is not None else None
        curve["step"].append(step)
        curve["energy"].append(e)
        curve["energy_err"].append(de)
        curve["energy_spread"].append(np.sqrt(var))
        curve["delta"].append(delta)
        msg = (f"  step {step:4d}/{cfg['n_iter']}:  E = {e:+.6f} ± {de:.6f}"
               f"   (spread, sqrt(var) = {np.sqrt(var):.4f})")
        if delta is not None:
            msg += f"   delta = {delta:.3e}"
        print(msg, flush=True)
        if run is not None:
            log_step(run, step, E, vs, exact_E0=exact_E0)
        if ckpt_every and ((step + 1) % ckpt_every == 0):
            _write_checkpoint(step + 1)

    def on_timing(step, td):
        """Persist the per-step phase breakdown so it's in the curve JSON (works
        offline) and on W&B for side-by-side comparison across n_sweeps/n_samples."""
        curve["timing"].append({"step": step, **td})
        if run is not None:
            run.log({f"time/{k}": v for k, v in td.items()}, step=step)

    t0 = time.time()
    remaining = max(0, cfg["n_iter"] - start_step)
    diverged = False
    try:
        if remaining > 0:                              # 0 only if a resume is already complete
            run_loop(vs, Ham, n_iter=remaining, dt=cfg["dt"],
                     diag_shift=cfg["diag_shift"], on_step=on_step, lr_min=cfg["lr_min"],
                     qgt=cfg.get("qgt", "auto"), start_step=start_step,
                     total_iter=cfg["n_iter"], time_phases=True, on_timing=on_timing,
                     grad_guard=cfg["grad_guard"], spike_factor=cfg["spike_factor"],
                     max_rollbacks=cfg["max_rollbacks"],
                     rollback_shift_boost=cfg["rollback_shift_boost"],
                     rollback_cooldown=cfg["rollback_cooldown"],
                     baseline_window=cfg["baseline_window"])
        else:
            print(f"[train] '{name}' already complete at {start_step} steps; finalizing.")
    except DivergenceError as ex:
        diverged = True
        print(f"[train] GENUINE DIVERGENCE: {ex}; persisting last sane state and "
              f"finalizing.", flush=True)
        _write_checkpoint(start_step)      # vs already restored to last_good; gate passes
    runtime_s = time.time() - t0

    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz_stabs)
    if exact_E0 is not None:                               # final FOM -> run.summary
        obs["E_exact"] = exact_E0
        obs["delta"] = abs(obs["E0"] - exact_E0) / abs(exact_E0)
    print(f"[train] done in {runtime_s:.1f}s  E={obs['E0']:.4f}  Vscore={obs['Vscore']:.2e}  "
          + (f"delta={obs['delta']:.3e}  " if exact_E0 is not None else "")
          + f"<A_v>={obs['A_v_mean']:.3f}  <sz>={obs['sz_mean']:.3f}")

    # --- artifacts: model weights (.mpack) + local run JSON ---
    weights_base = os.path.join(cfg["out_dir"], name)
    save_model(vs, weights_base)                       # writes {weights_base}.mpack

    result = {
        "name": name, "config": cfg, "n_params": int(vs.n_parameters),
        "runtime_s": runtime_s, "observables": obs, "curve": curve,
        "weights": f"{weights_base}.mpack", "diverged": diverged,
    }
    with open(f"{weights_base}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[train] saved {weights_base}.json and {weights_base}.mpack")

    if run is not None:
        try:                                           # weights as W&B artifact —
            import wandb                                # must run BEFORE finish_run,
            art = wandb.Artifact(name.replace("/", "_"), type="model")  # which calls
            art.add_file(f"{weights_base}.mpack")       # run.finish() (closes the run)
            run.log_artifact(art)
        except Exception as e:                         # noqa: BLE001
            print(f"[train] W&B artifact upload skipped: {e}")
        finish_run(run, vs, Ham, geo,
                   extra={"runtime_s": runtime_s, "n_params": int(vs.n_parameters)},
                   observables=obs)

    return result


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> Dict[str, Any]:
    # default=SUPPRESS means an omitted flag is ABSENT from the parsed dict, so it
    # falls through to the code defaults (TRAIN_DEFAULTS + builders.DEFAULTS, applied
    # in train()). This keeps those dicts the single source of truth: editing
    # TRAIN_DEFAULTS actually changes the CLI behavior (and what W&B logs). Pass a
    # flag explicitly only when you want to override the code default for that run.
    D = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description="Train a 3D toric-code NQS (bosonic or fermionic). Omitted "
                    "options fall back to TRAIN_DEFAULTS / builders.DEFAULTS.")
    # System
    p.add_argument("--L", type=int, required=True, help="linear size (Lx=Ly=Lz)")
    p.add_argument("--bc", choices=["PBC", "OBC"], default=D)
    p.add_argument("--model", choices=["bosonic", "fermionic"], default=D)
    # Hamiltonian
    p.add_argument("--hx", type=float, default=D)
    p.add_argument("--hy", type=float, default=D)
    p.add_argument("--hz", type=float, default=D)
    p.add_argument("--J", type=float, default=D)
    p.add_argument("--hz_preset", choices=list(HZ_PRESETS), default=D,
                   help="set h_z AND E_exact from a hardcoded ED reference point "
                        "(hard/mid/easy); enables the delta figure of merit")
    p.add_argument("--exact_E0", type=float, default=D,
                   help="E_exact for the delta FOM at a custom h_z (alternative to "
                        "--hz_preset)")
    # Architecture
    p.add_argument("--arch",
                   choices=["ToricCNN", "ToricCNN_full", "ToricCNN_gridinv",
                            "GeoCNN", "VanillaCNN", "VanillaWilsonCNN"],
                   default=D)
    p.add_argument("--hidden", type=int, default=D)
    p.add_argument("--vanilla_depth", type=int, default=D,
                   help="VanillaCNN: number of hidden conv layers (default 2)")
    p.add_argument("--kernel_size", type=int, default=D,
                   help="VanillaCNN/VanillaWilsonCNN: cubic conv kernel extent (default 3); "
                        "ToricCNN_gridinv: invariant grid-conv kernel (default auto = L)")
    p.add_argument("--noninv_random", action="store_true",
                   help="VanillaWilsonCNN: random-init the noninv block instead of "
                        "identity warm start (default is identity pass-through)")
    p.add_argument("--noninv_channels", type=int, default=D,
                   help="ToricCNN_full: edge channels C in each pre-Wilson block")
    p.add_argument("--n_noninv", type=int, default=D,
                   help="ToricCNN_full: number of non-invariant blocks before Wilson")
    p.add_argument("--inv_hidden", type=int, nargs="*", default=D,
                   help="ToricCNN_full: post-Wilson hidden widths, e.g. --inv_hidden 16 16")
    p.add_argument("--cnn_hidden", type=int, nargs="*", default=D,
                   help="GeoCNN: edge-conv channel widths (no Wilson), e.g. "
                        "--cnn_hidden 8 8 8; a width-1 readout is appended")
    # Training
    p.add_argument("--n_iter", type=int, default=D)
    p.add_argument("--dt", type=float, default=D, help="(initial) learning rate")
    p.add_argument("--lr_min", type=float, default=D,
                   help="if set, cosine-decay lr from --dt down to this over n_iter")
    p.add_argument("--diag_shift", type=float, default=D)
    p.add_argument("--qgt", choices=["auto", "dense", "onthefly", "srt", "minsr"], default=D,
                   help="SR solver: dense (QGTJacobianDense — fast, robust; wants "
                        "n_samples >= n_params), onthefly (matrix-free CG), srt/minsr "
                        "(VMC_SRt kernel trick — sample-space solve, best when "
                        "n_params >> n_samples; no in-run guard/phase split), or auto "
                        "(dense iff n_params <= 8192). Use 'dense' on GPU — the "
                        "onthefly/CG path is the one that fails there.")
    p.add_argument("--seed", type=int, default=D)
    # Sampling
    p.add_argument("--n_samples", type=int, default=D)
    p.add_argument("--n_chains", type=int, default=D)
    p.add_argument("--n_sweeps", type=int, default=D,
                   help="Metropolis sweeps between recorded samples (default 2N = 48 at L=2)")
    p.add_argument("--n_discard", type=int, default=D)
    p.add_argument("--chunk_size", type=int, default=D)
    # Output / logging
    p.add_argument("--name", default=D, help="run name (default auto from params)")
    p.add_argument("--out_dir", default=D)
    p.add_argument("--wandb_project", default=D)
    p.add_argument("--wandb_entity", default=D)
    p.add_argument("--wandb_group", default=D,
                   help="wandb group tying a sweep's runs together for comparison "
                        "(e.g. the SLURM job name)")
    p.add_argument("--no_wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--wandb_offline", action="store_true",
                   help="log W&B to a local dir (WANDB_MODE=offline) for compute "
                        "nodes with no network; `wandb sync` it later from a login node")
    # Checkpoint / resume (cluster timeout robustness)
    p.add_argument("--checkpoint_every", type=int, default=D,
                   help="write weights + energy curve to disk every N steps "
                        "(default 10; 0 disables) so a timed-out job keeps its progress")
    p.add_argument("--resume", action="store_true",
                   help="continue from {out_dir}/{name}.ckpt.mpack + .curve.json if "
                        "present (resumes the LR schedule and appends to the curve); "
                        "re-submit the SAME command to keep going after a timeout")
    p.add_argument("--init_from", default=D, metavar="WEIGHTS_BASE",
                   help="warm start parameters from another run's {base}.mpack (path "
                        "WITHOUT the .mpack extension) — for directed hysteresis "
                        "sweeps that carry a phase across neighbouring field points. "
                        "Overridden by --resume when this run's own checkpoint exists.")
    # Divergence guard / self-healing rollback (default ON; see run_loop)
    p.add_argument("--no_grad_guard", action="store_true",
                   help="disable the divergence guard / self-healing rollback (default ON)")
    p.add_argument("--spike_factor", type=float, default=D,
                   help="rollback if sqrt(var) exceeds this x the median of recent "
                        "spreads (default 10)")
    p.add_argument("--max_rollbacks", type=int, default=D,
                   help="give up (persist last sane state, exit nonzero) after this "
                        "many rollbacks (default 5)")
    p.add_argument("--rollback_shift_boost", type=float, default=D,
                   help="multiply diag_shift by this during the post-rollback cooldown "
                        "(default 10)")
    p.add_argument("--rollback_cooldown", type=int, default=D,
                   help="steps to keep the boosted diag_shift after a rollback (default 20)")
    p.add_argument("--baseline_window", type=int, default=D,
                   help="window (in sane steps) for the running spread median (default 20)")

    cfg = vars(p.parse_args())
    # --no_wandb only forces wandb off; otherwise leave it to TRAIN_DEFAULTS.
    if cfg.pop("no_wandb", False):
        cfg["wandb"] = False
    # --no_grad_guard flips the guard off; omission falls through to TRAIN_DEFAULTS (ON).
    if cfg.pop("no_grad_guard", False):
        cfg["grad_guard"] = False
    # --noninv_random flips the default identity warm start off (store_true always
    # present in the dict; only act when set so omission falls through to defaults).
    if cfg.pop("noninv_random", False):
        cfg["noninv_identity"] = False
    return cfg


if __name__ == "__main__":
    import sys
    res = train(_parse_args())
    sys.exit(1 if res.get("diverged") else 0)
