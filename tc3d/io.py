"""Checkpoint I/O: save/restore NetKet variational states (.mpack)."""

import json
import os
import sys
from typing import Any, Dict

import flax
import netket as nk

# Resume-mismatch guard (see check_resume_config): training-relevant keys that
# must match between an in-progress checkpoint and the current invocation
# before it's safe to continue it. Shared by tc3d.sweep and tc3d.train.
RESUME_CHECK_KEYS = ("dt", "lr_min", "n_iter", "diag_shift", "hx", "hy", "hz", "L",
                     "compute_dtype", "inv_impl", "qgt_solver")

# Speed-lever keys (2026-09, p3d/speed-research) that pre-date this guard: a
# checkpoint written before they existed has them ABSENT from its saved config
# (== None on .get), which must compare equal to an explicit request for the
# value they always behaved as -- not flag a spurious mismatch. See
# builders.DEFAULTS/resolve_compute_dtype (compute_dtype None=="float64") and
# builders.with_defaults (qgt_solver "cholesky" for qgt=="dense", the only
# literal default this repo has ever shipped).
_LEVER_LEGACY_DEFAULTS = {"compute_dtype": "float64", "inv_impl": "conv",
                          "qgt_solver": "cholesky"}


def _normalize_config_value(key: str, value: Any) -> Any:
    """Canonicalize a speed-lever config value for comparison; a no-op for
    every other key."""
    if key == "compute_dtype":
        return "float64" if value in (None, "", "none", "double") else value
    if key in ("inv_impl", "qgt_solver") and value in (None, ""):
        return _LEVER_LEGACY_DEFAULTS[key]
    return value


def check_resume_config(curve_path: str, name: str, cfg: Dict[str, Any], *,
                        keys=RESUME_CHECK_KEYS, is_anchor: bool = False,
                        allow_mismatch: bool = False, caller: str = "train") -> None:
    """Guard against silently CONTINUING an in-progress checkpoint under
    different knobs than produced it (e.g. a stale partial run left over from
    an earlier, different attempt at this same name). A no-op when no
    checkpoint exists yet at `curve_path` (a genuinely fresh run/point).
    Compares `keys` (+ `anchor_overrides` for a sweep chain's point 0) between
    the checkpoint's saved config and `cfg`, normalizing the pre-guard speed-
    lever keys (compute_dtype/inv_impl/qgt_solver) so a checkpoint written
    before they existed (absent -> None) compares equal to an explicit request
    for their legacy default (float64/conv/cholesky) -- only a REAL effective
    difference is flagged. Any mismatch prints the differing keys and
    exits(2) unless `allow_mismatch`.

    Shared by `tc3d.sweep._check_resume_config` and `tc3d.train`'s own
    `--resume` path (`caller` only tags the printed messages).
    """
    if not os.path.exists(curve_path):
        return
    with open(curve_path) as f:
        saved = json.load(f).get("config", {})
    all_keys = tuple(keys) + (("anchor_overrides",) if is_anchor else ())
    diffs = {k: (saved.get(k), cfg.get(k)) for k in all_keys
             if _normalize_config_value(k, saved.get(k)) != _normalize_config_value(k, cfg.get(k))}
    if not diffs:
        return
    detail = "; ".join(f"{k}: checkpoint={a!r} now={b!r}" for k, (a, b) in diffs.items())
    print(f"[{caller}] CONFIG MISMATCH resuming {name}: {detail}", flush=True)
    if not allow_mismatch:
        print(f"[{caller}] aborting (pass --allow_config_mismatch to resume anyway)", flush=True)
        sys.exit(2)


def save_model(vstate: nk.vqs.VariationalState, filename: str,
               verbose: bool = True) -> None:
    """
    Save a variational state to a file.

    Args:
        vstate: NetKet variational state
        filename: Name of the file to save to (without extension)
        verbose: print a confirmation line (set False for periodic checkpoints
                 that would otherwise spam the log)
    """
    with open(f"{filename}.mpack", 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

    if verbose:
        print(f"Model saved to {filename}.mpack")


def load_weights(vstate: nk.vqs.VariationalState, filename: str) -> nk.vqs.MCState:
    """
    Restore parameters (and sampler state) into an already-built variational
    state from a `.mpack` written by `save_model` — the resume path. Unlike
    a full rebuild, this keeps the caller's exact sampler / Hamiltonian wiring;
    the saved sampler RNG state is restored too, so a resumed run continues the
    Markov chains where the checkpoint left off.

    Args:
        vstate: a freshly built MCState with matching sampler / model
        filename: checkpoint path without the `.mpack` extension

    Returns:
        The variational state with checkpointed parameters loaded.
    """
    with open(f"{filename}.mpack", 'rb') as file:
        data = file.read()
    # from_bytes also restores the CHECKPOINT's sampling config (n_samples,
    # n_discard_per_chain, chunk_size), not just parameters. A warm start from
    # a state built with different settings silently clobbers the caller's —
    # e.g. chunk_size=None from a CPU prefit checkpoint disabled chunking on
    # the GPU run (unchunked forces at L=3 fermionic = 78 GB OOM). Keep ours.
    keep = (vstate.n_samples, vstate.n_discard_per_chain, vstate.chunk_size)
    vstate = flax.serialization.from_bytes(vstate, data)
    vstate.n_samples, vstate.n_discard_per_chain, vstate.chunk_size = keep
    return vstate
