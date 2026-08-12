"""
tc3d/sweep.py
─────────────────────────────────────────────────────────────────────────────
Batch many field points through ONE long-lived process, so the ~10 min JAX/XLA
compile is paid ONCE and reused across the whole chunk — the 3D analogue of
running `tc3d.train` per point, minus the per-point process spawn.

Why this works: the magnetic field enters the Hamiltonian only as a Pauli-string
*weight* (`model/hamiltonian.py`), so the compiled model-apply / QGT / observable
kernels are field-agnostic. They are keyed on the `vs` (flax model) instance and
the sample shape, which we hold FIXED across points — only the Hamiltonian's
numeric weights change. We therefore build the geometry / ansatz / sampler /
variational state ONCE and, per field point, rebuild only the (cheap) Hamiltonian,
(re)initialise the weights, and hand the reused `vs` to `tc3d.train.train`
via its `state=` hook. `train` writes the SAME per-point `{name}.{json,mpack,
curve.json,ckpt.mpack}` as a standalone run, so all downstream extraction
(`check_convergence.py`, `fm.py`, `renyi.py`) is untouched.

Usage (CLI / cluster) — one chunk of the hz phase sweep at L=4:
    python -m tc3d.sweep --L 4 --bc OBC --model bosonic --arch ToricCNN_gridinv \
        --field hz --field_values 0.1 0.125 0.15 0.175 --fixed_field_value 0.2 \
        --name_template "bosonic_gridinv_L{L}_hx{hx}_hz{hz}" \
        --noninv_channels 4 --n_noninv 2 --inv_hidden 2 2 2 --kernel_size 3 \
        --dt 0.01 --lr_min 0.001 --diag_shift 1e-3 --qgt dense --n_iter 150 \
        --n_samples 8192 --n_chains 1024 --n_sweeps 48 --chunk_size 2048 \
        --checkpoint_every 10 --out_dir $PSCRATCH/tc_nqs/phase_hx0.2/L4 --wandb_offline

Restartability: a chunk is requeue-safe. Points whose `{name}.json` already exists
are skipped, and each point is trained with `resume=True`, so a wall-limit requeue
of the same chunk continues from the last on-disk checkpoint.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import jax
jax.config.update("jax_enable_x64", True)   # match train.py: float64 SR/QGT
import jax.numpy as jnp

from tc3d.builders import build_state, build_hamiltonian, with_defaults
from tc3d.train import train
from tc3d.validation import build_eval_operators
from tc3d.io import load_weights

# The swept field and its complementary (fixed) field.
_OTHER = {"hz": "hx", "hx": "hz"}


def _copy_tree(tree):
    """Deep, device-resident copy of a pytree (params or sampler state) — the
    same idiom `builders.run_loop` uses for its rollback snapshots."""
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)


def _load_final_params(out_dir: str, name: str, vs):
    """Copied parameters from a point's final `{name}.mpack` (written by `train`).

    Read from disk — NOT the in-memory `vs` — so the warm-start hand-off is
    authoritative and requeue-safe: on a resume `train` reassigns its *local* vs
    to a fresh checkpoint load, which would leave the loop's `vs` stale.
    """
    base = os.path.join(out_dir, name)
    return _copy_tree(load_weights(vs, base).parameters)


def init_point_weights(vs, *, cold, prev_params, warm_start):
    """Per-point weight-initialisation policy (the one real design choice here).

    cold        : (params, sampler_state) snapshot captured right after build_state
                  — the fixed-seed init a standalone seed=0 process would start from.
    prev_params : the previous point's converged params (None on the first point).
    warm_start  : if True, chain each point off its neighbour (directed / hysteresis
                  sweep); if False, reset every point to the cold init (independent
                  phase-sweep points — the default, purely for compile amortisation).

    Cold reset restores BOTH params and the (unthermalised) sampler state so the
    trajectory matches a fresh process; warm start carries params forward and lets
    the chains keep their thermalisation from the neighbour.
    """
    if warm_start and prev_params is not None:
        vs.parameters = _copy_tree(prev_params)          # chained init
    else:
        cold_params, cold_sampler = cold
        vs.parameters = _copy_tree(cold_params)          # == per-task seed=0 init
        vs.sampler_state = _copy_tree(cold_sampler)
    return vs


def sweep(base_config: Dict[str, Any], field: str, field_values: List[float], *,
          name_template: str, warm_start: bool = False) -> List[Dict[str, Any]]:
    """Run a field sweep in one process, reusing a single `vs` across all points.

    base_config carries the fixed field + all structural/optimisation knobs; each
    point overrides only `field` and derives its `name` from `name_template`.
    """
    if field not in _OTHER:
        raise ValueError(f"--field must be one of {list(_OTHER)}, got {field!r}")

    base_cfg = with_defaults(base_config)
    # Build geometry / hilbert / ansatz / sampler / variational state ONCE. The
    # Hamiltonian is field-dependent, so it is (re)built per point below.
    geo, hi, _, vs, _ = build_state(base_cfg, build_ham=False)
    cold = (_copy_tree(vs.parameters), _copy_tree(vs.sampler_state))

    print(f"[sweep] built once: N={geo.N}  n_params={int(vs.n_parameters)}  "
          f"arch={base_cfg['arch']}  n_chains={int(vs.sampler.n_chains)}  "
          f"field={field}  {len(field_values)} points  warm_start={warm_start}",
          flush=True)

    results: List[Dict[str, Any]] = []
    prev_params = None
    eval_ops = None            # field-independent; built once on the first point
    for i, val in enumerate(field_values):
        cfg = {**base_cfg, field: float(val), "resume": True}
        cfg["name"] = name_template.format(**cfg)
        done = os.path.join(cfg["out_dir"], f"{cfg['name']}.json")

        if os.path.exists(done):
            print(f"[sweep] ({i+1}/{len(field_values)}) skip {field}={val}: "
                  f"{cfg['name']}.json exists", flush=True)
            if warm_start:                                # keep the chain alive
                prev_params = _load_final_params(cfg["out_dir"], cfg["name"], vs)
            continue

        # Cheap per-point rebuild: only the Pauli-string weights change; xz_stabs
        # is field-independent (fermionic decoration) but rebuilt for correctness.
        t_pt = time.time()
        Ham, xz = build_hamiltonian(cfg, geo, hi)
        print(f"[t] build_hamiltonian: {time.time() - t_pt:.1f}s", flush=True)
        if eval_ops is None:
            # 1150 s per call inside a warm process (2026-08-11 [t] data); the
            # operators depend only on geometry/basis/model, never on (hx, hz).
            eval_ops = build_eval_operators(hi, geo, cfg, xz_stabs=xz)

        init_point_weights(vs, cold=cold, prev_params=prev_params,
                           warm_start=warm_start)

        print(f"[sweep] ({i+1}/{len(field_values)}) === {field}={val}  "
              f"name={cfg['name']} ===", flush=True)
        res = train(cfg, state=(geo, hi, Ham, vs, xz), eval_ops=eval_ops)
        print(f"[t] point {field}={val} wall total: {time.time() - t_pt:.1f}s",
              flush=True)
        results.append(res)

        if warm_start:
            prev_params = _load_final_params(cfg["out_dir"], cfg["name"], vs)
    return results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> Dict[str, Any]:
    # Mirror train.py's SUPPRESS convention: an omitted flag is ABSENT from the
    # dict and falls through to builders.DEFAULTS / TRAIN_DEFAULTS, so those stay
    # the single source of truth. Only the sweep-specific flags are required.
    D = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description="Batch a chunk of field points through one process "
                    "(amortises the JAX compile). Omitted options fall back to "
                    "TRAIN_DEFAULTS / builders.DEFAULTS, exactly like tc3d.train.")
    # Sweep control
    p.add_argument("--field", required=True, choices=["hz", "hx"],
                   help="which magnetic field is swept across the chunk")
    p.add_argument("--field_values", type=float, nargs="+", required=True,
                   help="the chunk of field values (already rounded by the submitter)")
    p.add_argument("--fixed_field_value", type=float, default=0.0,
                   help="value of the complementary (constant) field")
    p.add_argument("--name_template", required=True,
                   help="per-point run name, e.g. 'bosonic_gridinv_L{L}_hx{hx}_hz{hz}' "
                        "(formatted with the resolved config: {model},{L},{hx},{hz},...)")
    p.add_argument("--warm_start", action="store_true",
                   help="chain each point off the previous point's converged weights "
                        "(directed / hysteresis sweep); default is a cold reset per "
                        "point (independent phase-sweep points)")
    # System
    p.add_argument("--L", type=int, required=True, help="linear size (Lx=Ly=Lz)")
    p.add_argument("--bc", choices=["PBC", "OBC"], default=D)
    p.add_argument("--model", choices=["bosonic", "fermionic"], default=D)
    p.add_argument("--dual_basis", action="store_true",
                   help="Hadamard-conjugated (dual) basis — see tc3d.train --help")
    # Fermionic analytic-structure knobs (see tc3d.train --help). init_from is
    # applied PER POINT inside train(), over the cold reset — with the analytic
    # prefit artifact this injects the frozen head/penalty structure identically
    # and independently at every field point (no neighbour warm-starting).
    p.add_argument("--phase_head_frozen", action="store_true")
    p.add_argument("--flux_penalty", type=float, default=D)
    p.add_argument("--chains_up", action="store_true")
    p.add_argument("--init_from", default=D)
    p.add_argument("--J", type=float, default=D)
    # Architecture (same knobs train.py exposes)
    p.add_argument("--arch",
                   choices=["ToricCNN", "ToricCNN_full", "ToricCNN_gridinv",
                            "GeoCNN", "VanillaCNN", "VanillaWilsonCNN"], default=D)
    p.add_argument("--hidden", type=int, default=D)
    p.add_argument("--vanilla_depth", type=int, default=D)
    p.add_argument("--kernel_size", type=int, default=D)
    p.add_argument("--noninv_random", action="store_true")
    p.add_argument("--noninv_channels", type=int, default=D)
    p.add_argument("--noninv_hidden", type=str, nargs="*", default=D)
    p.add_argument("--radius_edge", type=float, default=D)
    p.add_argument("--radius_plaq", type=float, default=D)
    p.add_argument("--n_noninv", type=int, default=D)
    p.add_argument("--inv_hidden", type=int, nargs="*", default=D)
    p.add_argument("--cnn_hidden", type=int, nargs="*", default=D)
    # Training
    p.add_argument("--n_iter", type=int, default=D)
    p.add_argument("--dt", type=float, default=D)
    p.add_argument("--lr_min", type=float, default=D)
    p.add_argument("--diag_shift", type=float, default=D)
    p.add_argument("--qgt", choices=["auto", "dense", "onthefly", "srt", "minsr"], default=D)
    p.add_argument("--seed", type=int, default=D)
    # Sampling
    p.add_argument("--n_samples", type=int, default=D)
    p.add_argument("--n_chains", type=int, default=D)
    p.add_argument("--n_sweeps", type=int, default=D)
    p.add_argument("--n_discard", type=int, default=D)
    p.add_argument("--chunk_size", type=int, default=D)
    # Output / logging
    p.add_argument("--out_dir", default=D)
    p.add_argument("--wandb_project", default=D)
    p.add_argument("--wandb_entity", default=D)
    p.add_argument("--wandb_group", default=D)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_offline", action="store_true")
    # Checkpoint (resume is forced per point; --checkpoint_every kept)
    p.add_argument("--checkpoint_every", type=int, default=D)
    p.add_argument("--final_eval_rounds", type=int, default=D,
                   help="K pooled sampling rounds for end-of-training observables")
    p.add_argument("--no_topological", action="store_true",
                   help="skip the inline O_FM/S2 block (same flag as train.py)")
    # Divergence guard (same flags as train.py)
    p.add_argument("--no_grad_guard", action="store_true")
    p.add_argument("--spike_factor", type=float, default=D)
    p.add_argument("--max_rollbacks", type=int, default=D)
    p.add_argument("--rollback_shift_boost", type=float, default=D)
    p.add_argument("--rollback_cooldown", type=int, default=D)
    p.add_argument("--baseline_window", type=int, default=D)

    cfg = vars(p.parse_args())
    # Same store_true → default translations as train._parse_args.
    if cfg.pop("no_wandb", False):
        cfg["wandb"] = False
    if cfg.pop("no_grad_guard", False):
        cfg["grad_guard"] = False
    if cfg.pop("no_topological", False):
        cfg["compute_topological"] = False
    if cfg.pop("noninv_random", False):
        cfg["noninv_identity"] = False
    if not cfg.get("dual_basis", False):
        cfg.pop("dual_basis", None)
    if isinstance(cfg.get("noninv_hidden"), list):   # same tolerant parse as train.py
        cfg["noninv_hidden"] = [int(t) for tok in cfg["noninv_hidden"]
                                for t in str(tok).replace(",", " ").split()]
    return cfg


def main() -> None:
    cfg = _parse_args()
    field = cfg.pop("field")
    field_values = cfg.pop("field_values")
    name_template = cfg.pop("name_template")
    warm_start = cfg.pop("warm_start", False)
    # The complementary field is held fixed for the whole chunk.
    cfg[_OTHER[field]] = cfg.pop("fixed_field_value")
    sweep(cfg, field, field_values, name_template=name_template, warm_start=warm_start)


if __name__ == "__main__":
    main()
