"""Checkpoint I/O: save/restore NetKet variational states (.mpack)."""

import flax
import jax
import netket as nk
import numpy as np


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

    A checkpoint saved from a COMPLEX build (e.g. a phase-head/h_y!=0 run)
    loaded into a REAL target (e.g. a `--sign_frame` build via `--init_from`)
    would otherwise silently promote the target's params to complex: flax's
    `from_bytes` restores each leaf at the DTYPE IT WAS SAVED WITH, ignoring the
    target's dtype (only the pytree *structure* is checked). We compare against
    the target's dtypes below and refuse that promotion unless the imaginary
    part is negligible.
    """
    with open(f"{filename}.mpack", 'rb') as file:
        data = file.read()
    # from_bytes also restores the CHECKPOINT's sampling config (n_samples,
    # n_discard_per_chain, chunk_size), not just parameters. A warm start from
    # a state built with different settings silently clobbers the caller's —
    # e.g. chunk_size=None from a CPU prefit checkpoint disabled chunking on
    # the GPU run (unchunked forces at L=3 fermionic = 78 GB OOM). Keep ours.
    keep = (vstate.n_samples, vstate.n_discard_per_chain, vstate.chunk_size)
    target_dtypes = [np.asarray(p).dtype for p in jax.tree_util.tree_leaves(vstate.parameters)]
    new_vstate = flax.serialization.from_bytes(vstate, data)
    loaded_leaves, treedef = jax.tree_util.tree_flatten(new_vstate.parameters)
    if len(loaded_leaves) == len(target_dtypes):
        fixed, changed = [], False
        for i, (leaf, tdt) in enumerate(zip(loaded_leaves, target_dtypes)):
            arr = np.asarray(leaf)
            if np.iscomplexobj(arr) and not np.issubdtype(tdt, np.complexfloating):
                im = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
                if im <= 1e-12:
                    arr = arr.real.astype(tdt)
                    changed = True
                    print(f"[io] load_weights({filename!r}): checkpoint param #{i} "
                          f"is complex with negligible Im (max|Im|={im:.2e}) -> "
                          f"cast to real {tdt} to match the target model", flush=True)
                else:
                    raise TypeError(
                        f"load_weights({filename!r}): checkpoint param #{i} is "
                        f"complex with max|Im|={im:.3e} (NOT negligible) but the "
                        f"target model expects real dtype {tdt} -- refusing a "
                        "silent complex->real promotion (likely --init_from a "
                        "complex/phase-head checkpoint into a real --sign_frame "
                        "build)")
            fixed.append(arr)
        if changed:
            new_vstate.parameters = jax.tree_util.tree_unflatten(treedef, fixed)
    vstate = new_vstate
    vstate.n_samples, vstate.n_discard_per_chain, vstate.chunk_size = keep
    return vstate
