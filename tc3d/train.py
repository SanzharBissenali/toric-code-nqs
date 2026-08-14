"""
tc3d/train.py
─────────────────────────────────────────────────────────────────────────────
Standalone, config-driven training pipeline for the 3D toric code (bosonic +
fermionic) — the 3D analogue of the inherited 2D `main.py`.

Inputs: hyperparameters, system size L, Hamiltonian fields (hx, hy, hz), and
naming. Outputs: model weights (`.mpack`), a local run JSON (config + final
expectation values + training curve), and W&B curves/observables.

Usage (notebook / Python):
    from tc3d.train import train
    res = train({"L": 2, "model": "fermionic", "arch": "ToricCNN_full",
                 "hx": 0.2, "hz": 0.2, "n_iter": 200, "wandb": False})

Usage (CLI / cluster):
    python -m tc3d.train --L 2 --model fermionic --arch ToricCNN_full \
        --hx 0.2 --hz 0.2 --n_iter 200 --no_wandb

Construction and the optimization loop are shared with `validation.py` via
`tc3d.builders`, so the trained model is exactly what validation scores.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)  # float64 SR/QGT (esp. on GPU)

from tc3d.builders import build_state, run_loop, with_defaults, DivergenceError
from tc3d.validation import (nqs_observables, pooled_final_observables,  # noqa: F401
                             topological_observables)
from tc3d.wandb_logger import init_run, log_step, finish_run
from tc3d.config import setup_environment
from tc3d.io import save_model, load_weights


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
    # Opt-in: ALSO persist a never-overwritten, step-suffixed weights snapshot
    # ({name}.step{N}.mpack) every `snapshot_every` steps, for post-hoc
    # mid-training observable ablations. 0 disables (default); does not alter
    # the fixed-path `.ckpt` used by --resume.
    "snapshot_every": 0,
    # Self-healing divergence guard (see run_loop): detect the SR blow-up, roll
    # back to the last sane params, re-seed chains + boost diag_shift, retry.
    "grad_guard": True, "spike_factor": 10.0, "max_rollbacks": 5,
    "rollback_shift_boost": 10.0, "rollback_cooldown": 20, "baseline_window": 20,
    # End-of-training topological order parameters (O_FM + central-plaquette S₂),
    # recorded in `observables` alongside the magnetisations/stabilisers. sector
    # "auto" picks the one the dominant field breaks (h_x→magnetic, h_z→electric).
    # Needs L>=4 (bulk placement); disable with --no_topological.
    "compute_topological": True, "fm_sector": "auto",
    # K pooled sampling rounds for the final observable block (1 = single-shot).
    "final_eval_rounds": 1,
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
    dual = "_dual" if cfg.get("dual_basis") else ""
    return cfg.get("name") or (
        f"{cfg['model']}_{cfg['arch']}{dual}_L{cfg['L']}_hx{cfg['hx']}_hz{cfg['hz']}")


def train(config: Dict[str, Any],
          *, state: Optional[Tuple[Any, Any, Any, Any, Any]] = None,
          eval_ops=None
          ) -> Dict[str, Any]:
    """Train one NQS run from a config dict; return a results dict.

    `eval_ops` is the chunk-level prebuilt (mean_ops, string_ops) from
    `validation.build_eval_operators` — field-independent, so a batch runner
    builds it once and passes it to every point (else built here per run).

    Side effects: writes `{out_dir}/{name}.mpack` (weights) and
    `{out_dir}/{name}.json` (config + observables + curve); logs to W&B if
    `config['wandb']`.

    `state` is an optional pre-built `(geo, hi, Ham, vs, xz_stabs)` tuple (as
    returned by `build_state`). When given, `build_state` is skipped and the
    injected objects are used verbatim — this is how the batch runner
    (`tc3d.sweep`) reuses ONE `vs` across many field points so the ~10 min
    JAX/XLA compile is paid once (the costly model/QGT kernels are keyed on the
    `vs` instance + sample shape, not on the field). The caller is responsible
    for building `Ham` at this point's field and resetting/warm-starting `vs`'s
    parameters before the call. With `state=None` this is byte-identical to the
    standalone single-run behaviour.
    """
    cfg = with_defaults({**TRAIN_DEFAULTS, **config})
    # h_y != 0 is the sign-full regime: with_defaults sets dtype="complex", build_model
    # returns a complex log ψ ansatz (ToricCNN/ToricCNN_full), and the SRt/SR paths use
    # the non-holomorphic complex QGT. Supported for the workhorse archs only.

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
        # An injected `state` already fixes the sampler's chain count; adopt it so
        # the logged config matches the actual `vs` (never silently overwrite the
        # reused sampler's n_chains with the device auto-default).
        cfg["n_chains"] = (int(state[3].sampler.n_chains) if state is not None
                           else n_chains_auto)

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

    geo, hi, Ham, vs, xz_stabs = state if state is not None else build_state(cfg)

    # --- warm start from a NEIGHBOUR's weights (hysteresis sweeps) --------------
    # Unlike --resume (which continues THIS run from its own checkpoint), --init_from
    # seeds the parameters from another converged run's {base}.mpack so a directed
    # field sweep carries its phase across points. Applied before the resume block,
    # so an own-checkpoint resume (timeout requeue) still wins and overwrites this.
    init_from = cfg.get("init_from")
    if init_from and os.path.exists(f"{init_from}.mpack"):
        p_fresh = jax.tree_util.tree_leaves(vs.parameters)   # this run's cold init
        vs = load_weights(vs, init_from)
        p_warm = jax.tree_util.tree_leaves(vs.parameters)    # the neighbour's weights
        num = float(np.sqrt(sum(float(np.sum(np.abs(np.asarray(a) - np.asarray(b)) ** 2))
                                for a, b in zip(p_warm, p_fresh))))
        den = float(np.sqrt(sum(float(np.sum(np.abs(np.asarray(b)) ** 2))
                                for b in p_fresh))) or 1.0
        # Distance from the *cold* init verifies the load actually took effect: a
        # warm start moves the starting point far from the (fixed-seed) random init,
        # so this is >> 0; ~0 would mean the checkpoint silently failed to load. The
        # step-0 energy logged just below is the empirical confirmation (it should sit
        # near the neighbour's CONVERGED energy, not a random-init value).
        print(f"[train] warm start: loaded {init_from}.mpack   "
              f"||theta_warm - theta_coldinit|| / ||theta|| = {num/den:.3f}  "
              f"(>> 0 = warm start applied)", flush=True)
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

    ref_E, ref_sig = cfg.get("ref_E"), cfg.get("ref_sig")
    curve = {"step": [], "energy": [], "energy_err": [], "energy_spread": [], "delta": [],
             "dE_ref": [],  # signed E - ref_E (None when --ref_E unset)
             "energy_im": [],   # Im⟨E⟩: ∼0 expected (Hermitian H); a free sign/convention check (pt 13)
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
        curve.setdefault("energy_im", [])   # checkpoints predating the Im⟨E⟩ diagnostic
        if os.path.exists(f"{ckpt_base}.mpack"):
            vs = load_weights(vs, ckpt_base)
        print(f"[train] resuming '{name}' from step {start_step}/{cfg['n_iter']}"
              f"  (loaded {len(curve['step'])} curve points)", flush=True)

    # --- start every chain in the physical flux sector -------------------------
    # AFTER init_from/resume, so a restored sampler state cannot reintroduce
    # ghost-sector chains. All-up (s=0) satisfies every closed-surface parity;
    # with --flux_penalty the chains then never have to FIND the sector — random
    # inits land in ghost cosets whose violation count has local minima that
    # single flips cannot descend (L=4: ~20% of chains stuck, delta frozen at 7e-3).
    if cfg.get("chains_up"):
        ss = vs.sampler_state
        vs.sampler_state = ss.replace(σ=jax.numpy.ones_like(ss.σ))
        print("[train] chains_up: all chains initialized in the physical sector",
              flush=True)

    run = None
    if cfg["wandb"]:
        import hashlib
        wandb_id = hashlib.md5(name.encode()).hexdigest()[:12]  # stable across requeues
        try:
            run = init_run(project=cfg["wandb_project"], entity=cfg["wandb_entity"],
                           config=cfg, name=name, group=cfg.get("wandb_group"),
                           tags=cfg["tags"] or [cfg["model"], cfg["arch"], f"L={cfg['L']}"],
                           id=wandb_id, resume="allow", dir=cfg["out_dir"])
            # summary at t=0, not just finish_run: crashed/running runs would
            # otherwise show a blank n_params column in the dashboard.
            run.summary.update({"n_params": cfg["n_params"], "N": int(geo.N)})
        except Exception as e:                              # noqa: BLE001
            # A W&B outage (CommError timeout, network, wedged run id) must never
            # cost GPU time: train without live logging — the local JSON/curve and
            # checkpoints carry everything and can be synced/re-uploaded later.
            print(f"[train] W&B init failed ({type(e).__name__}: {e}); "
                  f"continuing WITHOUT live logging.", flush=True)
            run = None

    ckpt_every = int(cfg.get("checkpoint_every", 0) or 0)
    snapshot_every = int(cfg.get("snapshot_every", 0) or 0)

    def _write_checkpoint(step, snapshot=False):
        """Persist weights + the energy curve so a kill/timeout loses nothing.

        Sanity-gated: never overwrite the last good checkpoint with a non-finite
        state, so `--resume` always restarts from a sane point (independent of the
        in-run guard, which normally keeps bad points out of `curve` entirely).

        `snapshot=True` ALSO writes a step-suffixed weights copy
        ({weights_base}.step{N}.mpack) that is never overwritten — a post-hoc
        snapshot history for mid-training observable ablations, independent of
        the fixed-path `.ckpt` used by --resume."""
        if curve["energy"] and not (np.isfinite(curve["energy"][-1])
                                    and np.isfinite(curve["energy_spread"][-1])):
            print(f"  [ckpt] step {step}: last curve point non-finite; skip checkpoint.",
                  flush=True)
            return
        save_model(vs, ckpt_base, verbose=False)
        if snapshot:
            save_model(vs, f"{weights_base}.step{step}", verbose=False)
        tmp = curve_path + ".tmp"                      # atomic: never a half-written file
        with open(tmp, "w") as f:
            json.dump({"completed_steps": step, "name": name, "config": cfg,
                       "curve": curve}, f)
        os.replace(tmp, curve_path)

    def on_step(step, E, vs):
        e   = float(np.real(E.mean))
        de  = float(np.real(E.error_of_mean))      # delta_E (MC error on the mean)
        var = float(np.real(E.variance))
        e_im = float(np.imag(E.mean))              # ∼0 for Hermitian H; sign/convention check (pt 13)
        delta = abs(e - exact_E0) / abs(exact_E0) if exact_E0 is not None else None
        dref = (e - ref_E) if ref_E is not None else None
        curve["step"].append(step)
        curve["energy"].append(e)
        curve["energy_err"].append(de)
        curve["energy_spread"].append(np.sqrt(var))
        curve["energy_im"].append(e_im)
        curve["delta"].append(delta)
        curve["dE_ref"].append(dref)
        vscore = geo.N * var / e**2 if e != 0 else float("nan")
        msg = (f"  step {step:4d}/{cfg['n_iter']}:  E = {e:+.6f} ± {de:.6f}"
               f"   (spread, sqrt(var) = {np.sqrt(var):.4f})   Vscore = {vscore:.2e}")
        if abs(e_im) > max(10 * de, 1e-6):         # imaginary energy above MC noise -> flag
            msg += f"   [!] Im(E) = {e_im:+.3e}"
        if delta is not None:
            msg += f"   delta = {delta:.3e}"
        if dref is not None:
            msg += (f"   dE_ref = {dref:+.4f}"
                    + (f" ({dref/ref_sig:+.1f}sig)" if ref_sig else "")
                    + (" BELOW ref" if dref < 0 else " above ref"))
            if ref_sig and dref < -2 * ref_sig:
                msg += "  [!] below ref-2sig: unbiased-QMC bound violated, suspect a bug"
        print(msg, flush=True)
        if run is not None:
            log_step(run, step, E, vs, exact_E0=exact_E0, ref_E=ref_E, ref_sig=ref_sig)
        due_snap = snapshot_every and ((step + 1) % snapshot_every == 0)
        if (ckpt_every and (step + 1) % ckpt_every == 0) or due_snap:
            _write_checkpoint(step + 1, snapshot=due_snap)

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

    # K-round pooled final evaluation (K=1 == the old single-shot + the frozen
    # ParaToric FM operators): 65k-equivalent statistics through the training
    # kernels — no recompile, training-budget memory (no separate eval job).
    t_eval = time.time()
    obs = pooled_final_observables(vs, Ham, geo, cfg, xz_stabs=xz_stabs,
                                   rounds=cfg.get("final_eval_rounds", 1),
                                   eval_ops=eval_ops)
    print(f"[t] final-eval: K={cfg.get('final_eval_rounds', 1)} pooled rounds "
          f"in {time.time() - t_eval:.1f}s", flush=True)
    if exact_E0 is not None:                               # final FOM -> run.summary
        obs["E_exact"] = exact_E0
        obs["delta"] = abs(obs["E0"] - exact_E0) / abs(exact_E0)
    if ref_E is not None:                                  # signed benchmark gap (QMC)
        obs["ref_E"] = ref_E
        obs["dE_ref"] = obs["E0"] - ref_E
        if ref_sig is not None:
            obs["ref_sig"], obs["dE_ref_sig"] = ref_sig, (obs["E0"] - ref_E) / ref_sig
    print(f"[train] done in {runtime_s:.1f}s  E={obs['E0']:.4f}  Vscore={obs['Vscore']:.2e}  "
          + (f"delta={obs['delta']:.3e}  " if exact_E0 is not None else "")
          + (f"dE_ref={obs['dE_ref']:+.4f}  " if ref_E is not None else "")
          + f"<A_v>={obs['A_v_mean']:.3f}  <sz>={obs['sz_mean']:.3f}")

    # Topological order parameters (O_FM + central-plaquette S₂) alongside the
    # magnetisations/stabilisers — best-effort single-state mirror of the sweep
    # extractors, never fatal (see validation.topological_observables). L<4 or
    # fm_sector="none" -> {}; the sweep extractors remain the authoritative curves.
    # Pooled-eval runs (final_eval_rounds > 1, e.g. Phase B) skip this block: its
    # 16-chain eval clone samples ~25k sequential MCMC steps at ~2% GPU batch
    # utilization (2026-08-12 profiling — tens of wall-minutes per point), its
    # inline estimators are not in the Phase-B comparison spec, and the sweep
    # extractors (fm.py / renyi.py) remain the authoritative curves.
    if cfg.get("final_eval_rounds", 1) > 1:
        if cfg.get("compute_topological", True):
            print("[train] inline topological block skipped (pooled final eval "
                  "carries the campaign observables)")
    elif cfg.get("compute_topological", True):
        t_topo = time.time()
        try:
            topo = topological_observables(vs, geo, cfg)
            if topo:
                obs.update(topo)
                _fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
                print(f"[train] topological ({topo.get('fm_sector')}, R={topo.get('fm_R')}): "
                      f"O_FM={_fmt(topo.get('O_FM'))}  S2={_fmt(topo.get('S2'))}")
        except Exception as e:                             # noqa: BLE001 — never lose a run
            print(f"[train] topological observables skipped ({type(e).__name__}: {e})")
        print(f"[t] topological block: {time.time() - t_topo:.1f}s", flush=True)

    # --- artifacts: model weights (.mpack) + local run JSON ---
    t_io = time.time()
    weights_base = os.path.join(cfg["out_dir"], name)
    save_model(vs, weights_base)                       # writes {weights_base}.mpack

    result = {
        "name": name, "config": cfg, "n_params": int(vs.n_parameters),
        "runtime_s": runtime_s, "observables": obs, "curve": curve,
        "weights": f"{weights_base}.mpack", "diverged": diverged,
    }
    with open(f"{weights_base}.json", "w") as f:
        # O_FM_* can be NaN under the den<=0 convention -> keep JSON RFC-safe
        from tc3d.fm import _json_nonfinite_safe
        json.dump(_json_nonfinite_safe(result), f, indent=2)
    print(f"[train] saved {weights_base}.json and {weights_base}.mpack "
          f"([t] artifacts: {time.time() - t_io:.1f}s)")

    if run is not None:
        t_wb = time.time()
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
        print(f"[t] wandb finish: {time.time() - t_wb:.1f}s", flush=True)

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
    p.add_argument("--phase_head", action="store_true",
                   help="token-quadratic phase head on gridinv (exact fTC h=0 "
                        "sign class; complex dtype only)")
    p.add_argument("--flux_penalty", type=float, default=D,
                   help="fermionic gridinv: Re logpsi -= kappa per violated "
                        "closed-surface flux parity (analytic ghost-sector "
                        "suppression; adds no parameters; try 6.0)")
    p.add_argument("--chains_up", action="store_true",
                   help="initialize every MCMC chain at the all-up state (inside "
                        "the physical flux sector; applied after init_from/resume "
                        "so restored sampler states cannot reintroduce stuck "
                        "ghost-sector chains)")
    p.add_argument("--phase_head_frozen", action="store_true",
                   help="phase head with theta in the 'constants' collection: "
                        "carried by checkpoints, excluded from gradients/QGT "
                        "(mandatory at L>=4 — N_p^2 head params would explode "
                        "the dense QGT; load theta via --init_from)")
    p.add_argument("--dual_basis", action="store_true",
                   help="Hadamard-conjugated (dual) basis: stars A_v become the "
                        "diagonal Z-family, the ansatz coarse-grains over vertex-star "
                        "tokens on the vertex grid, and the cluster sampler flips "
                        "plaquettes. Bosonic + hy=0 + ToricCNN_gridinv only. "
                        "Observables/JSON keys stay physical.")
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
    p.add_argument("--ref_E", type=float, default=D,
                   help="benchmark reference energy (e.g. QMC): print + log the SIGNED "
                        "per-step dE_ref = E - ref_E (+ above, - below the reference)")
    p.add_argument("--ref_sig", type=float, default=D,
                   help="1-sigma of --ref_E; reports dE_ref in sigma units and flags "
                        "runs below ref - 2*sigma (impossible vs an unbiased QMC ref)")
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
    p.add_argument("--noninv_hidden", type=str, nargs="*", default=D,
                   help="gridinv archs: per-layer noninv widths, e.g. "
                        "--noninv_hidden 1 2 4 (spins -> 1 -> 2 -> 4 -> Wilson); "
                        "a single quoted token '1 2 4' also works; "
                        "overrides --noninv_channels/--n_noninv")
    p.add_argument("--radius_edge", type=float, default=D,
                   help="noninv GeoConv3D stencil radius (default 1.05 -> the 15-tap "
                        "stencil: self + 8 perpendicular NN + 6 same-orientation "
                        "next-NN); larger radii pull in further edge shells")
    p.add_argument("--radius_plaq", type=float, default=D,
                   help="ToricCNN/ToricCNN_full plaquette-stencil radius (the gridinv "
                        "archs use a grid conv for the invariant block instead)")
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
    p.add_argument("--snapshot_every", type=int, default=D,
                   help="ALSO persist a never-overwritten weights snapshot "
                        "{name}.step{N}.mpack every N steps (default 0 = off); "
                        "for post-hoc mid-training observable ablations, independent "
                        "of the --resume checkpoint")
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
    # End-of-training topological order parameters (O_FM + central-plaquette S₂)
    p.add_argument("--no_topological", action="store_true",
                   help="skip the end-of-training O_FM + S₂ observables (default ON for "
                        "L>=4; they mirror the sweep extractors on the final state)")
    p.add_argument("--fm_sector", default=D, choices=["auto", "electric", "magnetic", "none"],
                   help="FM sector for the inline O_FM: auto (default) picks the one the "
                        "dominant field breaks (h_x→magnetic, h_z→electric); none also skips it")
    p.add_argument("--final_eval_rounds", type=int, default=D,
                   help="pool K sampling rounds for the final observables (K x n_samples "
                        "statistics through the compiled training kernels — K=8 at "
                        "n_samples=8192 is the 65k-equivalent eval; default 1)")

    cfg = vars(p.parse_args())
    # --no_topological forces the inline O_FM/S₂ off; omission falls through to ON.
    if cfg.pop("no_topological", False):
        cfg["compute_topological"] = False
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
    # --dual_basis: store_true is always present; drop the False so omission falls
    # through to builders.DEFAULTS (and a resumed config keeps its own value).
    if not cfg.get("dual_basis", False):
        cfg.pop("dual_basis", None)
    # --noninv_hidden tolerates both separate ints and one quoted/comma token
    # ("1 2 4" or 1 2 4 or 1,2,4) — callers that pass the whole string as a
    # single argv entry (notebook **extra passthrough) then still parse.
    if isinstance(cfg.get("noninv_hidden"), list):
        cfg["noninv_hidden"] = [int(t) for tok in cfg["noninv_hidden"]
                                for t in str(tok).replace(",", " ").split()]
    return cfg


if __name__ == "__main__":
    import sys
    res = train(_parse_args())
    sys.exit(1 if res.get("diverged") else 0)
