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
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
jax.config.update("jax_enable_x64", True)   # match train.py: float64 SR/QGT
import jax.numpy as jnp

from tc3d.builders import build_state, build_hamiltonian, with_defaults
from tc3d.train import train
from tc3d.validation import build_eval_operators
from tc3d.io import load_weights

# The swept field and its complementary (fixed) field. hy is a separate fixed
# passthrough (its own --hy flag, see below) — NOT listed here, so `sweep()`'s
# `field not in _OTHER` guard keeps raising for anything but hz/hx.
_OTHER = {"hz": "hx", "hx": "hz"}

# A directed chain's trailing branch tag (+ optional seed suffix), e.g. "_up",
# "_dn_s3". The analysis side classifies a name by its LAST token matching
# this same pattern -- an hy tag appended AFTER it would displace it and read
# as cold (see _tag_hy).
_CHAIN_TAG_RE = re.compile(r"_(up|dn)(?:_s\d+)?$")

# Resume-mismatch guard (see _check_resume_config): training-relevant keys
# that must match between an in-progress checkpoint and the current
# invocation before it's safe to continue it.
_RESUME_CHECK_KEYS = ("dt", "lr_min", "n_iter", "diag_shift", "hx", "hy", "hz", "L")


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


def _load_final_state(out_dir: str, name: str, vs) -> Tuple[Any, Any]:
    """(params, sampler_state) copied from a point's final `{name}.mpack`.

    Used on the SKIP path (a point already done, never trained in THIS
    process): `vs.sampler_state` there is whatever cold-init or an unrelated
    prior point left behind, not this checkpoint's thermalized chains --
    carrying params forward alone leaves the next point's chains
    re-thermalizing from a mismatched configuration (~0.03 energy bias,
    measured). See `_load_final_params` for why disk, not the live `vs`, is
    authoritative.
    """
    base = os.path.join(out_dir, name)
    loaded = load_weights(vs, base)
    return _copy_tree(loaded.parameters), _copy_tree(loaded.sampler_state)


def _tag_hy(name: str, hy: float, name_template: str) -> str:
    """Append the fixed-passthrough hy tag to an auto name -- INSERTED before
    a trailing directional chain suffix (_up/_dn[_sN], `_CHAIN_TAG_RE`) rather
    than appended after it. Appending after would make hy the LAST token, and
    the up/dn analysis regex (which matches the trailing token) would then
    read an hy!=0 name as cold. No-op at hy==0 (byte-identical names) or when
    the template already places {hy} itself."""
    if hy == 0.0 or "{hy}" in name_template:
        return name
    hy_tag = f"_hy{hy}"
    m = _CHAIN_TAG_RE.search(name)
    return name[:m.start()] + hy_tag + name[m.start():] if m else name + hy_tag


def _prev_point_status(out_dir: str, name: str, *, model: str,
                        h0_bound: Optional[float]) -> Optional[str]:
    """Health gate for a warm-started chain link's predecessor. None if
    healthy; else the reason for a `CHAIN STOPPED` message.

    `h0_bound` = -(len(vertex_all)+len(plaq_all)), the EXACT h=0 energy: any
    finite field can only LOWER E0 below it, so a bosonic point that
    converged (diverged=False) ABOVE it locked onto the wrong branch -- a
    mis-converged, not numerically-blown-up, state that the divergence flag
    alone never catches. Skipped for the fermionic model (no such exact
    bound here).
    """
    path = os.path.join(out_dir, f"{name}.json")
    if not os.path.exists(path):
        return "previous point diverged"
    with open(path) as f:
        res = json.load(f)
    if res.get("diverged"):
        return "previous point diverged"
    e0 = res.get("observables", {}).get("E0")
    if e0 is None:
        return "previous point diverged"
    if model == "bosonic" and h0_bound is not None and e0 > h0_bound + 0.05:
        return "E0 above h=0 bound"
    return None


def _check_resume_config(out_dir: str, name: str, cfg: Dict[str, Any], *,
                         is_anchor: bool, allow_mismatch: bool) -> None:
    """Guard against silently CONTINUING an in-progress checkpoint under
    different knobs than produced it (e.g. a stale partial run left over from
    an earlier, different chain attempt at this same name). A no-op when no
    checkpoint exists yet (`{name}.curve.json` absent -- a genuinely fresh
    point). Compares the training-relevant keys (+ `anchor_overrides` for
    point 0); any mismatch prints the differing keys and exits(2) unless
    `allow_mismatch`.
    """
    curve_path = os.path.join(out_dir, f"{name}.curve.json")
    if not os.path.exists(curve_path):
        return
    with open(curve_path) as f:
        saved = json.load(f).get("config", {})
    keys = _RESUME_CHECK_KEYS + (("anchor_overrides",) if is_anchor else ())
    diffs = {k: (saved.get(k), cfg.get(k)) for k in keys if saved.get(k) != cfg.get(k)}
    if not diffs:
        return
    detail = "; ".join(f"{k}: checkpoint={a!r} now={b!r}" for k, (a, b) in diffs.items())
    print(f"[sweep] CONFIG MISMATCH resuming {name}: {detail}", flush=True)
    if not allow_mismatch:
        print("[sweep] aborting (pass --allow_config_mismatch to resume anyway)", flush=True)
        sys.exit(2)


def init_point_weights(vs, *, cold, prev_state, warm_start):
    """Per-point weight-initialisation policy (the one real design choice here).

    cold       : (params, sampler_state) snapshot captured right after build_state
                 — the fixed-seed init a standalone seed=0 process would start from.
    prev_state : (params, sampler_state) from the previous point, or None on the
                 first point. `sampler_state` may itself be None, meaning "keep
                 vs's LIVE sampler_state" -- the just-trained in-process hand-off,
                 already thermalized for these exact params. A concrete
                 sampler_state (the skip-path hand-off, loaded from the
                 neighbour's OWN checkpoint via `_load_final_state`) always
                 overwrites -- vs's live state there predates this checkpoint.
    warm_start : if True, chain each point off its neighbour (directed / hysteresis
                 sweep); if False, reset every point to the cold init (independent
                 phase-sweep points — the default, purely for compile amortisation).

    Cold reset restores BOTH params and the (unthermalised) sampler state so the
    trajectory matches a fresh process; warm start carries params forward and lets
    the chains keep their thermalisation from the neighbour.
    """
    if warm_start and prev_state is not None:
        params, sampler_state = prev_state
        vs.parameters = _copy_tree(params)                # chained init
        if sampler_state is not None:
            vs.sampler_state = _copy_tree(sampler_state)  # explicit hand-off (skip path)
        # Same marker family as train.py's "warm start: loaded" so the campaign
        # watcher can verify every in-process chain link warm-started.
        print(f"[sweep] warm start: carried params{'+sampler' if sampler_state is not None else ''} "
              f"from the previous point", flush=True)
    else:
        cold_params, cold_sampler = cold
        vs.parameters = _copy_tree(cold_params)          # == per-task seed=0 init
        vs.sampler_state = _copy_tree(cold_sampler)
    return vs


def sweep(base_config: Dict[str, Any], field: str, field_values: List[float], *,
          name_template: str, warm_start: bool = False,
          anchor_overrides: Optional[Dict[str, Any]] = None,
          allow_config_mismatch: bool = False) -> List[Dict[str, Any]]:
    """Run a field sweep in one process, reusing a single `vs` across all points.

    base_config carries the fixed field + all structural/optimisation knobs; each
    point overrides only `field` and derives its `name` from `name_template`.

    `anchor_overrides`: knobs (e.g. dt/lr_min/n_iter/diag_shift) applied ONLY to
    point i==0 — the cold anchor of a first-order chain, trained slower/longer
    than the warm-started links that follow. Recorded verbatim under
    `cfg["anchor_overrides"]` so the point's JSON shows they were deliberate.
    Must not include sampler-shape keys (n_samples/n_chains/chunk_size) — `vs`
    is built once, below, from `base_config`, before any point-level override.

    `warm_start` additionally gates several chain-safety behaviours:
    (1) `--init_from` (a base_config-wide key) is dropped for i>0, so an
    external seed checkpoint seeds only the anchor — later links inherit via
    the in-process warm start, not a repeated reload of the same external file;
    (2) before point i>0 is even considered for skip/run, point i-1's JSON is
    checked (`_prev_point_status`) and the chain stops (prints, returns what's
    done so far) if it diverged, is missing E0, or (bosonic only) converged
    above the exact h=0 energy bound — the remaining points would otherwise
    inherit a corrupted or wrong-branch state; (3) on the skip path, BOTH
    params and sampler_state are reloaded from the skipped point's own
    checkpoint (`_load_final_state`) — its live sampler_state predates that
    checkpoint and would otherwise seed mismatched, unthermalized chains.

    `allow_config_mismatch=False` (default) aborts (exit 2) before resuming any
    point whose in-progress checkpoint's saved config disagrees with this
    invocation's on a training-relevant key (`_check_resume_config`) — a stale
    checkpoint from a different chain attempt must never be silently continued.
    """
    if field not in _OTHER:
        raise ValueError(f"--field must be one of {list(_OTHER)}, got {field!r}")

    base_cfg = with_defaults(base_config)
    # Build geometry / hilbert / ansatz / sampler / variational state ONCE. The
    # Hamiltonian is field-dependent, so it is (re)built per point below.
    geo, hi, _, vs, _ = build_state(base_cfg, build_ham=False)
    cold = (_copy_tree(vs.parameters), _copy_tree(vs.sampler_state))
    # Exact h=0 energy (see notes/exact-h0-energies): any finite field can only
    # LOWER E0 below this -- the bosonic branch-health bound in _prev_point_status.
    h0_bound = -(len(geo.vertex_all) + len(geo.plaq_all))

    print(f"[sweep] built once: N={geo.N}  n_params={int(vs.n_parameters)}  "
          f"arch={base_cfg['arch']}  n_chains={int(vs.sampler.n_chains)}  "
          f"field={field}  {len(field_values)} points  warm_start={warm_start}",
          flush=True)

    results: List[Dict[str, Any]] = []
    prev_state = None
    prev_name = None           # previous point's resolved name, for the divergence gate
    eval_ops = None            # field-independent; built once on the first point
    for i, val in enumerate(field_values):
        cfg = {**base_cfg, field: float(val), "resume": True}
        if warm_start and i > 0:
            # The chain inherits its state from the PREVIOUS point in-process
            # (below); an external --init_from seeds only the anchor (i==0) --
            # re-applying it every point would reload that same fixed file and
            # silently overwrite the warm-started params train() is about to see.
            cfg.pop("init_from", None)
        if i == 0 and anchor_overrides:
            cfg.update(anchor_overrides)
            cfg["anchor_overrides"] = dict(anchor_overrides)   # provenance in the JSON
        cfg["name"] = name_template.format(**cfg)
        # hy isn't a template field a caller is expected to know about (it's a
        # fixed passthrough, not swept) -- tag it on so two sweeps at different
        # hy over the same (field, values) grid don't collide. Mirrors train.py's
        # _run_name convention; hy=0 names are untouched (byte-identical).
        cfg["name"] = _tag_hy(cfg["name"], cfg.get("hy", 0.0), name_template)

        if warm_start and i > 0:
            reason = _prev_point_status(cfg["out_dir"], prev_name,
                                        model=base_cfg.get("model", "bosonic"),
                                        h0_bound=h0_bound)
            if reason:
                print(f"[sweep] CHAIN STOPPED at {cfg['name']}: {reason}", flush=True)
                return results
        prev_name = cfg["name"]

        done = os.path.join(cfg["out_dir"], f"{cfg['name']}.json")
        if os.path.exists(done):
            print(f"[sweep] ({i+1}/{len(field_values)}) skip {field}={val}: "
                  f"{cfg['name']}.json exists", flush=True)
            if warm_start:                                # keep the chain alive
                prev_state = _load_final_state(cfg["out_dir"], cfg["name"], vs)
            continue

        _check_resume_config(cfg["out_dir"], cfg["name"], cfg, is_anchor=(i == 0),
                             allow_mismatch=allow_config_mismatch)

        # Cheap per-point rebuild: only the Pauli-string weights change; xz_stabs
        # is field-independent (fermionic decoration) but rebuilt for correctness.
        t_pt = time.time()
        Ham, xz = build_hamiltonian(cfg, geo, hi)
        print(f"[t] build_hamiltonian: {time.time() - t_pt:.1f}s", flush=True)
        if eval_ops is None:
            # 1150 s per call inside a warm process (2026-08-11 [t] data); the
            # operators depend only on geometry/basis/model, never on (hx, hz).
            eval_ops = build_eval_operators(hi, geo, cfg, xz_stabs=xz)

        init_point_weights(vs, cold=cold, prev_state=prev_state,
                           warm_start=warm_start)

        print(f"[sweep] ({i+1}/{len(field_values)}) === {field}={val}  "
              f"name={cfg['name']} ===", flush=True)
        res = train(cfg, state=(geo, hi, Ham, vs, xz), eval_ops=eval_ops)
        print(f"[t] point {field}={val} wall total: {time.time() - t_pt:.1f}s",
              flush=True)
        results.append(res)

        if warm_start:
            # Params reloaded from disk (requeue-safe, see _load_final_params);
            # sampler_state left None -- keep vs's LIVE state, already
            # thermalized for these params by the training that just ran.
            prev_state = (_load_final_params(cfg["out_dir"], cfg["name"], vs), None)
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
    p.add_argument("--anchor_overrides", type=str, default=D,
                   help="JSON dict of knobs applied ONLY to point i==0, e.g. "
                        "'{\"dt\":0.02,\"lr_min\":0.002,\"n_iter\":500,\"diag_shift\":1e-3}' "
                        "-- the first-order-chain cold anchor (slower/longer than the "
                        "warm-started links that follow, which use the base knobs)")
    p.add_argument("--allow_config_mismatch", action="store_true",
                   help="continue an in-progress checkpoint even if its saved config "
                        "disagrees with this invocation's on a training-relevant key "
                        "(default: abort with exit 2 -- a stale checkpoint from a "
                        "different chain attempt must never be silently resumed)")
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
    p.add_argument("--hy", type=float, default=D,
                   help="fixed passthrough Y field (never swept; see --field)")
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
    p.add_argument("--qgt_solver", default=D, metavar="{cg,cgN,cholesky,solve,kernel}",
                   help="dense-QGT linear solver override — see tc3d.train --help "
                        "(default 'cholesky' for qgt=dense, inherited via "
                        "with_defaults/train() even when this flag is omitted)")
    p.add_argument("--compute_dtype", choices=["float64", "float32"], default=D,
                   help="ansatz arithmetic precision (same flag as train.py)")
    p.add_argument("--inv_impl", choices=["conv", "dense"], default=D,
                   help="invariant block: nn.Conv or unfolded GEMM (same flag as train.py)")
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
    p.add_argument("--snapshot_every", type=int, default=D,
                   help="ALSO persist a never-overwritten {name}.step{N}.mpack "
                        "weights snapshot every N steps (same flag as train.py)")
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
    allow_config_mismatch = cfg.pop("allow_config_mismatch", False)
    anchor_overrides_json = cfg.pop("anchor_overrides", None)
    anchor_overrides = json.loads(anchor_overrides_json) if anchor_overrides_json else None
    # The complementary field is held fixed for the whole chunk.
    cfg[_OTHER[field]] = cfg.pop("fixed_field_value")
    sweep(cfg, field, field_values, name_template=name_template, warm_start=warm_start,
          anchor_overrides=anchor_overrides, allow_config_mismatch=allow_config_mismatch)


if __name__ == "__main__":
    main()
