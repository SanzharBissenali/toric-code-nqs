"""Checkpoint I/O: save/restore NetKet variational states (.mpack)."""

import flax
import netket as nk


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
