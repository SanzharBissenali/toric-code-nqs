"""
Hamiltonian smoke tests at L=2 PBC.

CLUSTER/COLAB ONLY — despite doing no diagonalization, each `H.to_sparse()`
call materialises the full 2^24-row sparse matrix (~75 min and ~1.7 GB on an
8 GB laptop), and this file calls it three times. Do not run locally; the
locally-safe Hamiltonian regression is tests/test_hamiltonian_cache.py.

Run directly (cluster):
    python test_hamiltonian.py
"""

import numpy as np
import netket as nk

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.hamiltonian import create_hamiltonian


def build_h0(L=2):
    geom = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc="PBC")
    hi = nk.hilbert.Spin(s=1/2, N=geom.N)
    H = create_hamiltonian(
        hi,
        vertex_all=geom.vertex_all,
        plaq_all=geom.plaq_all,
        bonds=geom.bonds,
        hx=0.0, hy=0.0, hz=0.0,
        J=1.0,
        dtype=float,
    )
    return geom, hi, H


def all_up_state(N):
    """|0...0⟩ as a length-2^N state vector (index 0 is all spins up)."""
    psi = np.zeros(2**N)
    psi[0] = 1.0
    return psi


def test_diagonal_action_on_all_up(geom, H):
    """
    On |all-up⟩:
        - Every Z gives +1, so each B_p contributes -J = -1  → total -N_p.
        - Every A_v flips 6 bits → orthogonal basis state, no diagonal piece.
        - h=0, no field terms.
    Therefore ⟨all-up| H |all-up⟩ = -N_p.
    """
    Hs = H.to_sparse()
    psi = all_up_state(geom.N)
    expval = float(psi @ (Hs @ psi))
    expected = -len(geom.plaq_all)
    assert np.isclose(expval, expected), f"⟨H⟩ = {expval}, expected {expected}"


def test_offdiagonal_structure_on_all_up(geom, H):
    """
    H|all-up⟩ = (-N_p)|all-up⟩ + sum_v (-1) |A_v · all-up⟩.

    The N_v vertex flips land in N_v distinct orthogonal basis states with
    amplitude -1 each; the diagonal slot accumulates all N_p plaquette
    contributions (each +1 on the all-up state) with prefactor -J, giving -N_p.
    => H|all-up⟩ has exactly 1 + N_v nonzero entries.
       Diagonal amplitude = -N_p; the rest have amplitude -1.
    """
    Hs = H.to_sparse()
    psi = all_up_state(geom.N)
    out = np.asarray(Hs @ psi).ravel()
    N_v, N_p = len(geom.vertex_all), len(geom.plaq_all)

    nnz = int(np.sum(np.abs(out) > 1e-12))
    assert nnz == 1 + N_v, f"nnz = {nnz}, expected {1 + N_v}"

    assert np.isclose(out[0], -N_p), f"out[0] = {out[0]}, expected {-N_p}"

    off = out.copy()
    off[0] = 0.0  # remove the diagonal slot
    off = off[np.abs(off) > 1e-12]
    assert np.allclose(off, -1.0), f"off-diagonal amps not all -1: {off}"


def test_hermitian(H):
    """H built from real Pauli sums should be exactly Hermitian."""
    Hs = H.to_sparse()
    diff = (Hs - Hs.conj().T).tocoo()
    # stay sparse: densifying a 2^N-dim operator is impossible past ~N=20
    max_abs = np.max(np.abs(diff.data)) if diff.data.size else 0.0
    assert max_abs < 1e-12


def run_all():
    geom, hi, H = build_h0(L=2)
    print(f"N = {geom.N},  N_v = {len(geom.vertex_all)},  N_p = {len(geom.plaq_all)}")
    for name, fn in [
        ("⟨all-up|H|all-up⟩ = -N_p",   lambda: test_diagonal_action_on_all_up(geom, H)),
        ("H|all-up⟩ has 1+N_v nnz",    lambda: test_offdiagonal_structure_on_all_up(geom, H)),
        ("H is Hermitian",             lambda: test_hermitian(H)),
    ]:
        fn()
        print(f"  ok  {name}")


if __name__ == "__main__":
    run_all()
    print("All Hamiltonian tests passed.")
