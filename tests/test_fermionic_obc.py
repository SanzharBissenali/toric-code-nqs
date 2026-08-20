"""
OBC support for the fermionic decoration (truncation rule) -- Phase 0 gates.

See notes/fermionic_obc_l2_benchmark_plan.md SS Phase 0. All checks except the
end-to-end smoke are pure bitmask/GF(2) algebra or a matrix-free dense ED at
2^12 -- safe on the 8 GB dev machine (never touches L>=3 3D ED).

Run directly:
    cd tests && ../.venv/bin/python test_fermionic_obc.py
"""

import hashlib
import json
from collections import Counter

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import (
    fermionic_plaquettes, flux_constraint_masks, verify_xz_commutation)

# (a) PBC regression fingerprints, captured from the UNEDITED fermionic_plaquettes
# (sha256 of the json-serialized (z_edges, x_edges, coef) triples, order included).
PBC_FINGERPRINT = {
    2: ("df252dc1a557e3d78b47cd8d2b3de1cf2a0b0c667bbb0c7056ccfbcbc6043cd7", 24),
    3: ("6b4022d5a58194af984c486bbe480224f58af82710d8f26ecde8f73e6b71649a", 81),
}


def _fingerprint(stabs) -> str:
    payload = json.dumps([[z, x, c] for z, x, c in stabs])
    return hashlib.sha256(payload.encode()).hexdigest()


def test_pbc_regression(L):
    """PBC output must be byte-identical (order included) to the pre-change code."""
    geom = ThreeD_ToricCodeGeometry(L, L, L, bc="PBC")
    stabs = fermionic_plaquettes(geom)
    exp_hash, exp_n = PBC_FINGERPRINT[L]
    assert len(stabs) == exp_n, f"L={L} PBC: n={len(stabs)} != {exp_n}"
    got = _fingerprint(stabs)
    assert got == exp_hash, f"L={L} PBC fingerprint changed: {got} != {exp_hash}"


def test_obc_counts_and_xlists(L, n_faces, hist):
    """(b) face count + x_edges length histogram at OBC."""
    geom = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    stabs = fermionic_plaquettes(geom)
    assert len(stabs) == n_faces, f"L={L} OBC: n_faces={len(stabs)} != {n_faces}"
    got_hist = Counter(len(x) for _, x, _ in stabs)
    assert got_hist == Counter(hist), f"L={L} OBC x_edges histogram {got_hist} != {hist}"
    assert 0 not in got_hist, f"L={L} OBC: a face with 0 x-partners (should never happen)"


def test_ordering_matches_plaq_all(L):
    """The docstring of flux_constraint_masks assumes fermionic_plaquettes' face
    order matches geom.plaq_all row-for-row (the phase-head token order) -- check
    the sorted 4-edge z-sets line up exactly, not just as an unordered set."""
    geom = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    stabs = fermionic_plaquettes(geom)
    z_sets = [frozenset(z) for z, _, _ in stabs]
    plaq_sets = [frozenset(p) for p in geom.plaq_all]
    assert z_sets == plaq_sets, f"L={L} OBC: face order diverges from geom.plaq_all"


def test_commutation_obc(L):
    """(c) 0 violations at OBC (vertex_all rows carry -1 padding; verify_xz_commutation
    must skip it via _mask, not choke on it)."""
    geom = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    stabs = fermionic_plaquettes(geom)
    rep = verify_xz_commutation(stabs, geom.vertex_all)
    assert rep["ok"], f"L={L} OBC commutation violations: {rep['violations']}"


def test_flux_masks_obc(L, n_masks, expect_masks=None):
    """(d) flux_constraint_masks token-set count (and exact content at L=2)."""
    geom = ThreeD_ToricCodeGeometry(L, L, L, bc="OBC")
    stabs = fermionic_plaquettes(geom)
    masks = flux_constraint_masks(stabs)
    assert len(masks) == n_masks, f"L={L} OBC: {len(masks)} masks != {n_masks}"
    if expect_masks is not None:
        assert set(map(frozenset, masks)) == set(map(frozenset, expect_masks)), \
            f"L={L} OBC mask content {masks} != {expect_masks}"


def test_h0_dense_ed_obc_l2():
    """(e) h=0 dense ED at OBC L=2: E0=-14 (unique), first excited -10."""
    from tc3d.exact_diag import hamiltonian_linop

    geom = ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC")
    stabs = fermionic_plaquettes(geom)
    H, _basis = hamiltonian_linop(geom, xz_stabs=stabs, dtype=complex)
    dim = H.shape[0]
    assert dim == 4096, f"OBC L=2 dim={dim} != 4096"

    # H is a matvec-only LinearOperator -> build the dense matrix column by column.
    Hd = np.zeros((dim, dim), dtype=complex)
    e = np.zeros(dim, dtype=complex)
    for i in range(dim):
        e[:] = 0
        e[i] = 1.0
        Hd[:, i] = H.matvec(e)
        e[i] = 0

    herm_err = np.max(np.abs(Hd - Hd.conj().T))
    assert herm_err < 1e-12, f"H not Hermitian: max err {herm_err}"

    evals = np.linalg.eigvalsh(Hd)
    assert abs(evals[0] - (-14.0)) < 1e-9, f"E0={evals[0]} != -14"
    deg = int(np.sum(np.abs(evals - evals[0]) < 1e-8))
    assert deg == 1, f"ground state degeneracy {deg} != 1"
    assert abs(evals[1] - (-10.0)) < 1e-9, f"E1={evals[1]} != -10"


def test_sampler_pairs_obc_l2():
    """Sampler cluster construction (tc3d.builders.build_sampler) at OBC L=2:
    all 6 B~_p pairs are 1-edge, padded to the star width (3) by repeating that
    single index -- confirm each padded row still flips EXACTLY its one edge
    (the padding trap: a naive sequential toggle would double-flip a repeated
    index back to identity; the actual kernel, sampler.MultiRule.transition's
    `sigma.at[cluster].set(-sigma.at[cluster].get())`, gathers-then-scatters
    from the pre-transition state, so repeats are inert, not a double-flip)."""
    import netket as nk
    from tc3d.builders import build_sampler

    geom = ThreeD_ToricCodeGeometry(2, 2, 2, bc="OBC")
    hi = nk.hilbert.Spin(s=1 / 2, N=geom.N)
    clusters = build_sampler({"model": "fermionic"}, hi, geom).rule.rules[1].update_clusters
    n_stars = len(geom.vertex_all)
    assert clusters.shape == (n_stars + 6, 3), f"cluster shape {clusters.shape}"

    stabs = fermionic_plaquettes(geom)
    pair_rows = np.asarray(clusters)[n_stars:]
    for (_, x, _), row in zip(stabs, pair_rows):
        assert len(x) == 1, "every OBC L=2 B~_p pair is 1-edge"
        assert set(row.tolist()) == set(x), \
            f"padded row {row.tolist()} must reduce to the single edge {x}"

    # construction check: gather-then-scatter flip of an all-duplicate cluster
    # touches exactly that one index once (not zero, not twice).
    import jax.numpy as jnp
    sigma = jnp.array([1.0] * geom.N)
    cluster = jnp.array([pair_rows[0][0]] * 3)
    flipped = sigma.at[cluster].set(-sigma.at[cluster].get())
    diff = np.asarray(flipped) - np.asarray(sigma)
    n_changed = int(np.sum(np.abs(diff) > 1e-9))
    assert n_changed == 1, f"all-duplicate cluster changed {n_changed} entries, expected 1"


def test_smoke_build_state_obc_l2():
    """(f) end-to-end: build_state for L=2 OBC fermionic complex + frozen phase
    head + flux penalty, then a tiny sampled expectation must be finite."""
    from tc3d.builders import build_state

    config = {
        "L": 2, "bc": "OBC", "model": "fermionic", "arch": "ToricCNN_gridinv",
        "phase_head_frozen": True, "flux_penalty": 6.0,
        "n_samples": 32, "n_chains": 4, "n_discard": 0, "chunk_size": None,
        "hx": 0.0, "hz": 0.0, "seed": 0,
    }
    geo, hi, Ham, vs, xz_stabs = build_state(config)
    assert geo.N == 12 and len(xz_stabs) == 6
    E = vs.expect(Ham)
    em, ev = complex(E.mean), float(E.variance)
    assert np.isfinite(em.real) and np.isfinite(em.imag) and np.isfinite(ev), \
        f"non-finite expectation: mean={em}, var={ev}"


if __name__ == "__main__":
    for L in (2, 3):
        test_pbc_regression(L)
        print(f"  ok  PBC regression fingerprint (L={L})")

    test_obc_counts_and_xlists(2, 6, {1: 6})
    print("  ok  OBC L=2: 6 faces, all x-lists length 1")
    test_obc_counts_and_xlists(3, 36, {2: 12, 1: 24})
    print("  ok  OBC L=3: 36 faces, 12 two-edge + 24 one-edge")

    for L in (2, 3):
        test_ordering_matches_plaq_all(L)
        print(f"  ok  OBC face order matches geom.plaq_all (L={L})")

    for L in (2, 3):
        test_commutation_obc(L)
        print(f"  ok  OBC commutation: 0 violations (L={L})")

    test_flux_masks_obc(2, 2, expect_masks=[(0, 2, 4), (1, 3, 5)])
    print("  ok  OBC L=2 flux masks: 2, = {0,2,4}/{1,3,5}")
    test_flux_masks_obc(3, 10)
    print("  ok  OBC L=3 flux masks: 10")

    test_h0_dense_ed_obc_l2()
    print("  ok  h=0 dense ED OBC L=2: E0=-14 (unique), E1=-10")

    test_sampler_pairs_obc_l2()
    print("  ok  sampler pairs OBC L=2: 1-edge padding flips exactly its edge")

    test_smoke_build_state_obc_l2()
    print("  ok  end-to-end build_state smoke (OBC L=2 fermionic, finite E)")

    print("ALL FERMIONIC-OBC TESTS PASSED")
