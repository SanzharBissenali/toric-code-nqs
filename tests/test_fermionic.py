"""
Fermionic 3D toric code: decoration wiring + zero-field sign structure.

All checks are exact bitmask/GF(2) algebra on the stabilizer group — no ED, no
netket Hilbert space beyond geometry construction. Safe on the 8 GB dev machine.

At h=0 the model is a commuting stabilizer Hamiltonian, so the ground state is a
stabilizer state and relative amplitudes between X-connected basis states are
fixed exactly:  A_v: psi(s^x_v) = psi(s);  B~_p: psi(s^x_p) = (-1)^{|z_p&s|} psi(s).
BFS sign propagation from |0...0> therefore decides sign-fullness with no
diagonalization. The bosonic model (pure-X stars only) is the all-positive control.

Run directly:
    python test_fermionic.py
"""

from collections import deque

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import (
    fermionic_plaquettes, verify_xz_commutation, dressed_string, _mask)


def bfs_signs(x_generators, xz_generators, seed=0, max_states=200_000):
    """Exact stabilizer-state signs over the X-orbit of `seed`.

    Returns (n_explored, n_negative, contradiction). A contradiction means the
    seed has zero amplitude in P|seed> (never expected for |0...0> here).
    """
    sign = {seed: +1}
    q = deque([seed])
    n_neg, contra = 0, False
    while q and len(sign) < max_states:
        s = q.popleft()
        moves = [(xm, +1) for xm in x_generators]
        moves += [(xm, -1 if bin(zm & s).count("1") & 1 else +1)
                  for zm, xm in xz_generators]
        for xm, rel in moves:
            t, st = s ^ xm, sign[s] * rel
            if t in sign:
                contra |= sign[t] != st
            else:
                sign[t] = st
                n_neg += st < 0
                q.append(t)
    return len(sign), n_neg, contra


def z_column(geom, x=0, y=0):
    """sigma^z edges along z at column (x, y) — the canonical Wilson-line protocol."""
    Lz2 = 2 * geom.Lz
    return [geom._coord_to_idx[(2 * x % (2 * geom.Lx), 2 * y % (2 * geom.Ly),
                                (2 * z + 1) % Lz2)] for z in range(geom.Lz)]


def test_stabilizers(geom, stabs):
    """Counts, commutation, z/x disjointness, coefficients."""
    L = geom.Lx
    assert len(stabs) == 3 * L**3, "one decorated plaquette per face"
    rep = verify_xz_commutation(stabs, geom.vertex_all)
    assert rep["ok"], f"commutation violations: {rep['violations']}"
    assert rep["n_stabilizers"] == 4 * L**3
    assert all(not (_mask(z) & _mask(x)) for z, x, _ in stabs), "z/x overlap (sigma^y)"
    assert all(c == -1.0 for _, _, c in stabs), "coef must be -J"


def test_dressed_strings(geom, stabs):
    """Closed wrap line -> conserved (flux-free, 2L-edge dressing); open
    half-string -> exactly 2 endpoint fluxes (the charge+flux fermion)."""
    cz = z_column(geom)
    _, xw, flux_w = dressed_string(geom, stabs, cz)
    assert flux_w == [], f"closed loop not conserved: flux={flux_w}"
    assert len(xw) == 2 * geom.Lz, f"dressing {len(xw)} != {2 * geom.Lz}"
    oz = cz[: max(1, len(cz) // 2)]
    _, _, flux_o = dressed_string(geom, stabs, oz)
    assert len(flux_o) == 2, f"open string flux {len(flux_o)} != 2"


def test_nonstoquastic(geom, stabs):
    """<s^x_p|(-J B~_p)|s> = -J(-1)^{|z_p&s|}: both signs must occur."""
    rng = np.random.default_rng(0)
    elems = set()
    for _ in range(500):
        s = int.from_bytes(rng.bytes(16), "little") % (1 << geom.N)
        zb, _, c = stabs[rng.integers(len(stabs))]
        elems.add(c * (-1) ** (bin(_mask(zb) & s).count("1") & 1))
    assert elems == {-1.0, +1.0}, f"expected mixed-sign off-diagonals, got {elems}"


def test_sign_structure(geom, stabs):
    """Fermionic h=0 GS is sign-full; bosonic control is all-positive."""
    stars = [_mask(v) for v in geom.vertex_all]
    ferm = [(_mask(z), _mask(x)) for z, x, _ in stabs]
    n, n_neg, contra = bfs_signs(stars, ferm)
    assert not contra, "seed |0> has zero amplitude?"
    assert n_neg > 0, "fermionic GS should have negative amplitudes"
    nb, nb_neg, bcontra = bfs_signs(stars, [])
    assert not bcontra and nb_neg == 0, "bosonic control must be all-positive"
    if 2 ** (len(stars) - 1) <= 200_000:                   # orbit fits under the BFS cap
        assert nb == 2 ** (len(stars) - 1), "bosonic orbit = independent stars"
    return n, n_neg, nb


def test_sampler_pairs():
    """Fermionic sampler gains the B~_p x-pair clusters (padded to star width)."""
    import netket as nk
    from tc3d.builders import build_sampler
    geom = ThreeD_ToricCodeGeometry(2, 2, 2, bc="PBC")
    hi = nk.hilbert.Spin(s=1 / 2, N=geom.N)
    cb = build_sampler({"model": "bosonic"}, hi, geom).rule.rules[1].update_clusters
    cf = build_sampler({"model": "fermionic"}, hi, geom).rule.rules[1].update_clusters
    assert cb.shape == (8, 6), "bosonic clusters must stay stars-only"
    assert cf.shape == (8 + 24, 6), "fermionic clusters = stars + one pair per B~_p"
    stabs = fermionic_plaquettes(geom)
    for (_, x, _), row in zip(stabs, np.asarray(cf)[8:]):
        assert set(row.tolist()) == set(x), "padded pair must flip exactly its 2 edges"


def test_dtype_derivation():
    """--model fermionic must derive complex weights (sign-full at ANY field)."""
    from tc3d.builders import with_defaults
    assert with_defaults({"L": 2, "model": "fermionic"})["dtype"] == "complex"
    assert with_defaults({"L": 2})["dtype"] == "float64"
    assert with_defaults({"L": 2, "hy": 0.1})["dtype"] == "complex"


if __name__ == "__main__":
    for L in (2, 3):
        geom = ThreeD_ToricCodeGeometry(L, L, L, bc="PBC")
        stabs = fermionic_plaquettes(geom)
        test_stabilizers(geom, stabs)
        test_dressed_strings(geom, stabs)
        test_nonstoquastic(geom, stabs)
        n, n_neg, nb = test_sign_structure(geom, stabs)
        print(f"L={L}: stabilizers/strings/stoquasticity OK; GS orbit explored "
              f"{n} states, {n_neg} negative (bosonic control {nb}, all +)")
    test_dtype_derivation()
    print("dtype derivation OK (fermionic -> complex)")
    test_sampler_pairs()
    print("sampler clusters OK (stars + B~_p x-pairs)")
    print("ALL FERMIONIC TESTS PASSED")
