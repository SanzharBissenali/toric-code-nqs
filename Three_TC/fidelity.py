"""
Three_TC/fidelity.py
─────────────────────────────────────────────────────────────────────────────
L=2 fidelity of a trained NQS against exact diagonalisation — the guardrail the
sign-full (h_y != 0) campaign needs, because **energy is forgiving of a wrong sign
structure while fidelity is not** (a complex ansatz can hit the right |ψ| and the
wrong phase and still look converged in E). QMC is unavailable in the h_y regime, so
small-L ED is the only unconditional reference; in 3D that means L=2 only (N=24, 2^24
states — Colab-only, never the 8 GB dev box, per CLAUDE.md).

Two subtleties this module handles explicitly:

  • Basis alignment (pt 21). The ED reference is `colab_exact_diag.py`, whose basis is
    `arange(2^N)` with the convention **bit i = 1 → σ_z = −1** (see `z_string_eigvals`)
    and qubit index == bit position. We evaluate the NQS on spin configs built with that
    *same* convention (`spin_configs_from_basis`), so the ED eigenvectors and the NQS
    amplitude vector live in one aligned basis — no reliance on NetKet's internal
    `all_states()` ordering matching scipy's.

  • Topological degeneracy (pt 22). The 3D toric code has an (up to exponentially split)
    ground-space degeneracy (2^3 = 8 on the 3-torus). A single-eigenvector fidelity is
    ill-defined against a near-degenerate manifold: the NQS may equal a valid in-manifold
    superposition yet score low overlap with `evecs[:,0]` alone. So we report the
    **subspace-projector** fidelity  F = ‖P_manifold |ψ_NQS⟩‖² = Σ_i |⟨e_i|ψ_NQS⟩|²,
    with the manifold detected adaptively from the largest spectral gap (run ED with
    k ≥ expected degeneracy, e.g. k ≥ 8, so the gap to the excited states is captured).

Usage on Colab (both ED and the trained NQS in memory):
    from Three_TC.tests.colab_exact_diag import ThreeD_ToricCodeGeometry_PBC, make_hamiltonian_op
    ... run eigsh to get (evals, evecs) ...
    from Three_TC.builders import build_state
    geo, hi, Ham, vs, _ = build_state(cfg); vs = load_weights(vs, ckpt)
    from Three_TC.fidelity import fidelity_report
    rep = fidelity_report(vs, evals, evecs, N=geo.N, chunk=cfg.get("chunk_size"))
    print(rep["subspace_fidelity"], rep["manifold_dim"])
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def spin_configs_from_basis(basis: np.ndarray, N: int) -> np.ndarray:
    """(M, N) int8 array of ±1 spins for each basis integer `b`, using the ED convention
    σ_i = 1 − 2·bit_i(b)  (bit = 1 → σ_z = −1), qubit index == bit position — identical to
    `colab_exact_diag.z_string_eigvals`. This is what pins the NQS amplitudes to the ED
    eigenvector basis."""
    b = np.asarray(basis, dtype=np.int64)[:, None]
    i = np.arange(N, dtype=np.int64)[None, :]
    return (1 - 2 * ((b >> i) & 1)).astype(np.int8)


def _chunked_log_value(vs, sigma: np.ndarray, chunk: Optional[int]) -> np.ndarray:
    """logψ over (M, N) spins → flat (M,) complex, in `chunk`-row blocks (L≥6 needs this;
    at L=2 the full 2^24 pass is fine but chunking bounds peak memory)."""
    if not chunk or chunk >= sigma.shape[0]:
        return np.asarray(vs.log_value(sigma))
    return np.concatenate([np.asarray(vs.log_value(sigma[i:i + chunk]))
                           for i in range(0, sigma.shape[0], chunk)])


def nqs_amplitudes(vs, basis: np.ndarray, N: int,
                   chunk: Optional[int] = None) -> np.ndarray:
    """Normalised NQS state vector ψ_NQS over `basis`, aligned to the ED convention.

    Evaluates logψ on the explicit spin configs and exponentiates with a real-part shift
    for numerical stability (the overall constant cancels under normalisation). Complex
    for a complex ansatz, real for the h_y=0 ansatz."""
    sigma = spin_configs_from_basis(basis, N)
    logpsi = _chunked_log_value(vs, sigma, chunk)

    # Non-finite logψ on never-sampled tail configs (an unbounded conv stack can emit
    # ±inf/nan there) would poison the whole vector via the real-part shift below
    # (inf - inf = nan). Treat non-finite entries as zero amplitude (logψ = -inf), but
    # REPORT their fraction: a negligible tail is a numerical artifact on ~0-probability
    # configs (fidelity still valid); a large fraction means the trained state is
    # genuinely pathological (fidelity meaningless) -> raise.
    finite = np.isfinite(np.real(logpsi)) & np.isfinite(np.imag(logpsi))
    n_bad = int((~finite).sum())
    if n_bad:
        frac = n_bad / logpsi.size
        re = np.real(logpsi)[finite]
        print(f"  [fidelity] WARNING: {n_bad}/{logpsi.size} ({frac:.2e}) non-finite logψ "
              f"(finite Re logψ range [{re.min():.2f}, {re.max():.2f}]); masking to 0 amplitude",
              flush=True)
        if frac > 1e-3:
            raise FloatingPointError(
                f"{frac:.2e} of logψ non-finite (> 1e-3): trained state is pathological, "
                "not a negligible tail — fidelity would be meaningless")
        logpsi = np.where(finite, logpsi, -np.inf)

    logpsi = logpsi - np.max(np.real(logpsi[finite]))  # stabilise exp; cancels in the norm
    psi = np.exp(logpsi)                               # non-finite entries -> exp(-inf)=0
    nrm = np.linalg.norm(psi)
    if nrm == 0 or not np.isfinite(nrm):
        raise FloatingPointError(f"NQS amplitude vector has norm {nrm}")
    return psi / nrm


def degenerate_manifold(evals: np.ndarray, *, k_manifold: Optional[int] = None,
                        etol: Optional[float] = None) -> np.ndarray:
    """Indices of the (near-)degenerate ground manifold among sorted `evals`.

    Precedence: explicit `k_manifold` (take the lowest k) → `etol` (all within `etol` of
    E0) → **auto**: everything below the largest gap in the computed spectrum. Auto adapts
    both ways — a non-degenerate ground state (largest gap right after E0) yields a single
    vector; a near-degenerate one groups the cluster. Run ED with k ≥ the expected
    degeneracy so the gap to the *excited* band is the one that's captured."""
    ev = np.sort(np.real(np.asarray(evals)))
    if k_manifold is not None:
        return np.arange(min(k_manifold, len(ev)))
    if etol is not None:
        return np.where(ev - ev[0] <= etol)[0]
    if len(ev) < 2:
        return np.array([0])
    cut = int(np.argmax(np.diff(ev)))                # manifold = indices [0 .. cut]
    return np.arange(cut + 1)


def subspace_fidelity(psi: np.ndarray, evecs_manifold: np.ndarray) -> float:
    """F = Σ_i |⟨e_i|ψ⟩|² for orthonormal manifold columns `evecs_manifold` (dim, m).
    Equals ‖P_manifold|ψ⟩‖²; for m=1 this is the ordinary single-state fidelity."""
    ov = evecs_manifold.conj().T @ psi               # (m,)
    return float(np.sum(np.abs(ov) ** 2))


def fidelity_report(vs, evals: np.ndarray, evecs: np.ndarray, N: int, *,
                    chunk: Optional[int] = None,
                    k_manifold: Optional[int] = None,
                    etol: Optional[float] = None) -> Dict[str, Any]:
    """Full L=2 fidelity of `vs` against an ED spectrum (evals, evecs from eigsh).

    `evecs` columns are the eigenvectors in the `arange(2^N)` basis of
    `colab_exact_diag.make_hamiltonian_op`. Returns single-vector and subspace-projector
    fidelities, the detected manifold, and the spectral gaps (so the manifold choice is
    auditable)."""
    dim = 1 << N
    if evecs.shape[0] != dim:
        raise ValueError(f"evecs has leading dim {evecs.shape[0]} != 2^N = {dim}; the ED "
                         "vectors must be in the full arange(2^N) basis for this N")
    basis = np.arange(dim, dtype=np.int64)
    psi = nqs_amplitudes(vs, basis, N, chunk=chunk)

    manifold_idx = degenerate_manifold(evals, k_manifold=k_manifold, etol=etol)
    P = evecs[:, manifold_idx]
    F_sub = subspace_fidelity(psi, P)
    F_gs = subspace_fidelity(psi, evecs[:, :1])       # overlap with E0 eigenvector alone
    ev = np.sort(np.real(np.asarray(evals)))
    return {
        "single_vector_fidelity": F_gs,
        "subspace_fidelity": F_sub,
        "infidelity": 1.0 - F_sub,
        "manifold_dim": int(len(manifold_idx)),
        "evals": ev.tolist(),
        "gaps": np.diff(ev).tolist(),
        "note": ("subspace_fidelity = ‖P_manifold|ψ⟩‖² is the trustworthy number under "
                 "topological degeneracy; single_vector_fidelity < it only means eigsh "
                 "returned a rotated in-manifold basis, not real infidelity"),
    }


# =============================================================================
# Optional persistence: save/load the ED manifold from a Colab ED run so the
# NQS-side overlap can be computed later / elsewhere. Vectors are large
# (2^24 · complex128 · k ≈ 0.27·k GB at L=2), so this is opt-in.
# =============================================================================

def save_ed_vectors(path: str, evals: np.ndarray, evecs: np.ndarray,
                    meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist (evals, evecs) + metadata (Lx/Ly/Lz/hx/hy/hz/N, basis convention) as .npz."""
    md = dict(meta or {})
    md.setdefault("basis_convention", "arange(2^N); sigma_i = 1 - 2*bit_i(b); qubit==bit")
    np.savez_compressed(path, evals=np.asarray(evals), evecs=np.asarray(evecs),
                        meta=np.array(md, dtype=object))


def load_ed_vectors(path: str):
    """Inverse of `save_ed_vectors` → (evals, evecs, meta_dict)."""
    z = np.load(path, allow_pickle=True)
    return z["evals"], z["evecs"], z["meta"].item()
