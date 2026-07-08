"""
model/train_2d.py
─────────────────────────────────────────────────────────────────────────────
Standalone, config-driven training pipeline for the 2D toric-code **transformer**
NQS (Step 1: factored attention, no symmetry). The 2D analogue of
`Three_TC/train.py`, sharing nothing with the legacy `main.py` TDVP loop.

Usage (Python):
    from model.train_2d import train
    res = train({"L": 2, "hx": 0.2, "hz": 0.5, "n_iter": 500, "wandb": False})

Usage (CLI):
    python -m model.train_2d --L 3 --hx 0.2 --hz 0.5 --n_iter 500 --no_wandb

Construction and the optimization loop live in `model.builders_2d`. At L<=3 a
local exact-diagonalization reference (N<=12) is computed automatically for the
delta figure of merit; at L=4 (N=24 = 2^24 states) local ED is forbidden
(CLAUDE.md), so pass --exact_E0 or rely on the V-score.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)  # float64 SR/QGT

import netket as nk

from model.builders_2d import build_state, run_loop, with_defaults
from utils.config import setup_environment
from utils.io import save_model, load_weights
from Three_TC.utils.wandb_logger import init_run, log_step, finish_run


TRAIN_DEFAULTS: Dict[str, Any] = {
    "n_iter": 500, "dt": 2e-2, "diag_shift": 2e-4, "lr_min": 2e-3, "qgt": "auto",
    "out_dir": "outputs", "wandb": True,
    "wandb_project": "approx-sym-2D-TC-transformer",
    "wandb_entity": "models-california-institute-of-technology-caltech",
    "tags": None, "name": None,
    "checkpoint_every": 10, "resume": False, "wandb_offline": False,
    "ed_max_N": 14,          # auto local-ED reference only for N <= this (L<=3)
}


def _run_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("name") or (
        f"{cfg['arch']}_L{cfg['L']}_hx{cfg['hx']}_hz{cfg['hz']}_{cfg['bc']}")


def exact_ground_energy(geo, hx: float, hz: float, J: float = 1.0) -> Optional[float]:
    """Local ED ground energy, guarded to small N (never runs the 2^24 regime)."""
    from model.exact_diag import hamiltonian_linop
    from scipy.sparse.linalg import eigsh
    H, _basis = hamiltonian_linop(geo, hx=hx, hy=0.0, hz=hz, J=J)
    evals = eigsh(H, k=1, which="SA", return_eigenvectors=False)
    return float(evals[0])


def nqs_observables(vs, Ham, geo) -> Dict[str, Any]:
    """E0, variance, V-score and simple order parameters (all from MC samples)."""
    E = vs.expect(Ham)
    e0 = float(np.real(E.mean))
    var = float(np.real(E.variance))
    vscore = geo.N * var / e0 ** 2 if e0 != 0 else float("nan")

    def _stab_mean(groups, pauli):
        vals = []
        for g in groups:
            op = 1
            for i in g:
                if i == -1:
                    continue
                op = op * pauli(vs.hilbert, int(i))
            vals.append(float(np.real(vs.expect(op).mean)))
        return float(np.mean(vals))

    A_v = _stab_mean(geo.vertex_all, nk.operator.spin.sigmax)
    B_p = _stab_mean(geo.plaq_all, nk.operator.spin.sigmaz)
    sz = _stab_mean([[i] for i in range(geo.N)], nk.operator.spin.sigmaz)
    sx = _stab_mean([[i] for i in range(geo.N)], nk.operator.spin.sigmax)
    return {"E0": e0, "Var": var, "Vscore": vscore,
            "A_v_mean": A_v, "B_p_mean": B_p, "sz_mean": sz, "sx_mean": sx}


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train one transformer NQS run from a config dict; return a results dict."""
    cfg = with_defaults({**TRAIN_DEFAULTS, **config})

    # Device detection: GPU -> 1024 chains, CPU -> 16 (explicit --n_chains wins).
    _gpu, _node, n_chains_auto = setup_environment()
    is_gpu = n_chains_auto > 16
    if "n_chains" not in config:
        cfg["n_chains"] = n_chains_auto
    if "n_samples" not in config and is_gpu:
        cfg["n_samples"] = 2 * cfg["n_samples"]

    name = _run_name(cfg)
    cfg["name"] = name
    os.makedirs(cfg["out_dir"], exist_ok=True)
    weights_base = os.path.join(cfg["out_dir"], name)
    ckpt_base    = f"{weights_base}.ckpt"
    curve_path   = f"{weights_base}.curve.json"

    if cfg.get("wandb_offline"):
        os.environ["WANDB_MODE"] = "offline"

    geo, hi, Ham, vs = build_state(cfg)
    cfg["n_params"] = int(vs.n_parameters)
    cfg["n_sweeps"] = int(vs.sampler.sweep_size)

    # Exact reference for the delta FOM: explicit --exact_E0, else auto local ED
    # for small N only (never the forbidden 2^24 regime).
    exact_E0 = config.get("exact_E0")
    if exact_E0 is None and geo.N <= cfg["ed_max_N"]:
        exact_E0 = exact_ground_energy(geo, cfg["hx"], cfg["hz"], cfg["J"])
        print(f"[train] local ED reference (N={geo.N}): E_exact = {exact_E0:.6f}")
    cfg["exact_E0"] = exact_E0

    print(f"[train] {name}: N={geo.N}  n_params={cfg['n_params']}"
          f"  n_chains={cfg['n_chains']}  n_sweeps={cfg['n_sweeps']}  qgt={cfg['qgt']}"
          + (f"  E_exact={exact_E0:.6f}" if exact_E0 is not None else ""))

    curve = {"step": [], "energy": [], "energy_err": [], "energy_spread": [],
             "delta": [], "vscore": []}

    # --- resume a timed-out run from the last on-disk checkpoint ---
    start_step = 0
    if cfg.get("resume") and os.path.exists(curve_path):
        with open(curve_path) as f:
            ck = json.load(f)
        start_step = int(ck.get("completed_steps", 0))
        curve = ck.get("curve", curve)
        if os.path.exists(f"{ckpt_base}.mpack"):
            vs = load_weights(vs, ckpt_base)
        print(f"[train] resuming '{name}' from step {start_step}/{cfg['n_iter']}"
              f"  (loaded {len(curve['step'])} curve points)", flush=True)

    run = None
    if cfg["wandb"]:
        import hashlib
        wandb_id = hashlib.md5(name.encode()).hexdigest()[:12]
        run = init_run(project=cfg["wandb_project"], entity=cfg["wandb_entity"],
                       config=cfg, name=name, group=cfg.get("wandb_group"),
                       tags=cfg["tags"] or ["factored_transformer", f"L={cfg['L']}"],
                       id=wandb_id, resume="allow")

    ckpt_every = int(cfg.get("checkpoint_every", 0) or 0)

    def _write_checkpoint(step):
        save_model(vs, ckpt_base, verbose=False)
        tmp = curve_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"completed_steps": step, "name": name, "config": cfg,
                       "curve": curve}, f)
        os.replace(tmp, curve_path)

    _t_prev = []                       # wall clock of the previous on_step call
    def on_step(step, E, vs):
        now = time.time()              # first call includes the JIT compile; the
        dt_step = (now - _t_prev[0]) if _t_prev else None   # inter-call gap is one clean step
        _t_prev[:] = [now]
        e   = float(np.real(E.mean))
        de  = float(np.real(E.error_of_mean))
        var = float(np.real(E.variance))
        vscore = geo.N * var / e ** 2 if e != 0 else float("nan")
        delta = abs(e - exact_E0) / abs(exact_E0) if exact_E0 is not None else None
        curve["step"].append(step)
        curve["energy"].append(e)
        curve["energy_err"].append(de)
        curve["energy_spread"].append(np.sqrt(var))
        curve["delta"].append(delta)
        curve["vscore"].append(vscore)
        msg = (f"  step {step:4d}/{cfg['n_iter']}:  E = {e:+.6f} ± {de:.6f}"
               f"   (spread = {np.sqrt(var):.4f}, Vscore = {vscore:.2e})")
        if delta is not None:
            msg += f"   delta = {delta:.3e}"
        msg += f"   [{dt_step*1e3:.0f} ms/step]" if dt_step is not None else "   [compiling]"
        print(msg, flush=True)
        if run is not None:
            log_step(run, step, E, vs, exact_E0=exact_E0)
            run.log({"Vscore": vscore}, step=step)
        if ckpt_every and ((step + 1) % ckpt_every == 0):
            _write_checkpoint(step + 1)

    t0 = time.time()
    remaining = max(0, cfg["n_iter"] - start_step)
    if remaining > 0:
        run_loop(vs, Ham, n_iter=remaining, dt=cfg["dt"],
                 diag_shift=cfg["diag_shift"], on_step=on_step, lr_min=cfg["lr_min"],
                 qgt=cfg["qgt"], start_step=start_step, total_iter=cfg["n_iter"])
    else:
        print(f"[train] '{name}' already complete at {start_step} steps; finalizing.")
    runtime_s = time.time() - t0

    obs = nqs_observables(vs, Ham, geo)
    if exact_E0 is not None:
        obs["E_exact"] = exact_E0
        obs["delta"] = abs(obs["E0"] - exact_E0) / abs(exact_E0)
    print(f"[train] done in {runtime_s:.1f}s  E={obs['E0']:.4f}  Vscore={obs['Vscore']:.2e}  "
          + (f"delta={obs['delta']:.3e}  " if exact_E0 is not None else "")
          + f"<A_v>={obs['A_v_mean']:.3f}  <sz>={obs['sz_mean']:.3f}")

    save_model(vs, weights_base)
    result = {
        "name": name, "config": cfg, "n_params": int(vs.n_parameters),
        "runtime_s": runtime_s, "observables": obs, "curve": curve,
        "weights": f"{weights_base}.mpack",
    }
    with open(f"{weights_base}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[train] saved {weights_base}.json and {weights_base}.mpack")

    if run is not None:
        try:
            import wandb
            art = wandb.Artifact(name.replace("/", "_"), type="model")
            art.add_file(f"{weights_base}.mpack")
            run.log_artifact(art)
        except Exception as e:  # noqa: BLE001
            print(f"[train] W&B artifact upload skipped: {e}")
        finish_run(run, vs, Ham, geo,
                   extra={"runtime_s": runtime_s, "n_params": int(vs.n_parameters)},
                   observables=obs)

    return result


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> Dict[str, Any]:
    D = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description="Train a 2D toric-code transformer NQS (factored attention). "
                    "Omitted options fall back to TRAIN_DEFAULTS / builders_2d.DEFAULTS.")
    # System
    p.add_argument("--L", type=int, required=True, help="linear size (Lx=Ly)")
    p.add_argument("--bc", choices=["OBC", "PBC"], default=D)
    # Hamiltonian
    p.add_argument("--hx", type=float, default=D)
    p.add_argument("--hz", type=float, default=D)
    p.add_argument("--J", type=float, default=D)
    p.add_argument("--exact_E0", type=float, default=D,
                   help="E_exact for the delta FOM (required at L>=4; auto-ED at L<=3)")
    # Architecture
    p.add_argument("--arch", choices=["factored_transformer", "Combo", "RPP"], default=D,
                   help="ansatz (default factored_transformer)")
    # transformer
    p.add_argument("--d", type=int, default=D, help="embed dim (default 16)")
    p.add_argument("--n_heads", type=int, default=D, help="attention heads (default 4)")
    p.add_argument("--n_layers", type=int, default=D, help="transformer blocks (default 4)")
    p.add_argument("--mlp_ratio", type=int, default=D, help="MLP hidden = mlp_ratio*d (default 2)")
    # Combo / RPP baseline
    p.add_argument("--channels_noninv", type=int, nargs="+", default=D,
                   help="Combo: non-invariant CNN channel widths, e.g. --channels_noninv 1 16")
    p.add_argument("--channels_inv", type=int, nargs="+", default=D,
                   help="Combo: invariant CNN channel widths, e.g. --channels_inv 16 8 1")
    p.add_argument("--kernel_size", type=int, default=D, help="Combo: non-invariant conv kernel")
    p.add_argument("--rescale", type=float, default=D, help="Combo: Wilson rescale (default 1.0)")
    # Training
    p.add_argument("--n_iter", type=int, default=D)
    p.add_argument("--dt", type=float, default=D, help="(initial) learning rate")
    p.add_argument("--lr_min", type=float, default=D,
                   help="if set, cosine-decay lr from --dt down to this over n_iter")
    p.add_argument("--diag_shift", type=float, default=D)
    p.add_argument("--qgt", choices=["minsr", "dense", "onthefly", "auto"], default=D,
                   help="SR solver: minsr (VMC_SRt, sample-space; best when "
                        "n_params>n_samples — the transformer default), dense, onthefly, auto")
    p.add_argument("--seed", type=int, default=D)
    # Sampling
    p.add_argument("--n_samples", type=int, default=D)
    p.add_argument("--n_chains", type=int, default=D)
    p.add_argument("--n_sweeps", type=int, default=D)
    p.add_argument("--n_discard", type=int, default=D)
    p.add_argument("--chunk_size", type=int, default=D)
    # Output / logging
    p.add_argument("--name", default=D)
    p.add_argument("--out_dir", default=D)
    p.add_argument("--wandb_project", default=D)
    p.add_argument("--wandb_entity", default=D)
    p.add_argument("--wandb_group", default=D)
    p.add_argument("--no_wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--wandb_offline", action="store_true",
                   help="log W&B to a local dir (WANDB_MODE=offline)")
    # Checkpoint / resume
    p.add_argument("--checkpoint_every", type=int, default=D)
    p.add_argument("--resume", action="store_true")

    cfg = vars(p.parse_args())
    if cfg.pop("no_wandb", False):
        cfg["wandb"] = False
    return cfg


if __name__ == "__main__":
    train(_parse_args())
