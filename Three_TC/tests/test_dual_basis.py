"""
Dual-basis (Hadamard-conjugated) toric code: dense small-cell proof + ansatz
invariance gate. Everything runs locally: the dense checks use L=2 OBC
(N=12, 4096-dim) — the no-3D-locally rule is about L=2 PBC (2^24).

Covers, in order:
  1. create_hamiltonian(dual=True) == W · H · W with W = Hadamard^{⊗N}
     (matrix identity ⇒ identical spectrum; W built representation-
     independently as ⊗(σx+σz)/√2 from NetKet's own operators).
  2. Structural mirror of test_hamiltonian.py on the dual H: all-up diagonal
     = −(J·N_v + hx·N), off-diagonal nnz = 1 + N_p (star/face roles swapped).
  3. star_wilson_product (masked fixed-width gather) == ragged brute force,
     float features + OBC truncated stars — the #1 silent-wrongness risk.
  4. Token/flip pairing: face (B_p) flips preserve every star token and star
     (A_v) flips preserve every face token (even overlap); a single-edge flip
     does move star tokens (that's where the field physics lives).
  5. Flip-invariance of the ansatz at init: dual gridinv under every B_p flip,
     and the primal gridinv counterpart under every A_v flip (|Δlog ψ| ≈ 0).
  6. Guards: dual+hy, dual+fermionic, dual+non-gridinv all raise.

Run directly:
    python test_dual_basis.py
"""

import _path  # noqa: F401
from functools import reduce

import numpy as np
import netket as nk

from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.model.hamiltonian import create_hamiltonian
from Three_TC.model.networks import star_index_arrays, star_wilson_product, \
    vertex_grid_layout
from Three_TC.builders import build_model, build_hamiltonian, with_defaults

RNG = np.random.default_rng(7)


def build_pair(L=2, bc="OBC", hx=0.3, hz=0.1):
    geo = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=bc)
    hi = nk.hilbert.Spin(s=1 / 2, N=geo.N)
    kw = dict(vertex_all=geo.vertex_all, plaq_all=geo.plaq_all, bonds=geo.bonds,
              hx=hx, hz=hz, J=1.0, dtype=float)
    return geo, hi, create_hamiltonian(hi, **kw), create_hamiltonian(hi, dual=True, **kw)


def hadamard_all(hi):
    """W = ⊗_i (σx_i + σz_i)/√2 as a dense matrix — the algebraic form holds in
    any local-basis convention, so no assumption on NetKet's state ordering."""
    hi1 = nk.hilbert.Spin(s=1 / 2, N=1)
    w1 = ((nk.operator.spin.sigmax(hi1, 0, dtype=float)
           + nk.operator.spin.sigmaz(hi1, 0, dtype=float)).to_dense()) / np.sqrt(2)
    return reduce(np.kron, [w1] * hi.size)


def ragged_star_products(geo, feats):
    """Reference: per-channel star products over the -1-stripped ragged stars.
    feats: (..., C, N) → (..., C, N_v)."""
    out = []
    for star in geo.get_vertex_all_hetero():
        out.append(np.prod(feats[..., star], axis=-1))
    return np.stack(out, axis=-1)


def ragged_plaq_products(geo, feats):
    out = []
    for p in geo.plaq_all:
        out.append(np.prod(feats[..., list(p)], axis=-1))
    return np.stack(out, axis=-1)


# ── 1. conjugation identity ──────────────────────────────────────────────────

def test_conjugation_identity():
    geo, hi, Hp, Hd = build_pair(L=2, bc="OBC", hx=0.3, hz=0.1)
    assert geo.N == 12, f"L=2 OBC should have 12 edges, got {geo.N}"
    W = hadamard_all(hi)
    assert np.allclose(W @ W, np.eye(2 ** geo.N)), "W must be an involution"
    lhs = Hd.to_sparse().toarray()
    rhs = W @ Hp.to_sparse().toarray() @ W
    assert np.allclose(lhs, rhs, atol=1e-10), \
        f"H_dual != W H W (max dev {np.max(np.abs(lhs - rhs)):.3e})"


# ── 2. structural mirror on all-up ───────────────────────────────────────────

def test_structural_all_up():
    geo, hi, _Hp, Hd = build_pair(L=2, bc="OBC", hx=0.3, hz=0.0)
    N_v, N_p = len(geo.vertex_all), len(geo.plaq_all)
    Hs = Hd.to_sparse()
    psi = np.zeros(2 ** geo.N)
    psi[0] = 1.0                                  # index 0 = all spins up
    out = np.asarray(Hs @ psi).ravel()
    # dual: stars are diagonal Z-products (+1 each on all-up), the physical hx
    # field is diagonal too (rep_x = σz), faces are the 4-edge flips
    expected_diag = -(N_v + 0.3 * geo.N)
    assert np.isclose(out[0], expected_diag), f"diag {out[0]} != {expected_diag}"
    nnz = int(np.sum(np.abs(out) > 1e-12))
    assert nnz == 1 + N_p, f"nnz = {nnz}, expected {1 + N_p}"


# ── 3. masked star product == ragged brute force ─────────────────────────────

def test_star_product_masked_vs_ragged():
    import jax.numpy as jnp
    for L, bc in ((2, "OBC"), (3, "OBC"), (2, "PBC")):
        geo = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=bc)
        idx, mask = star_index_arrays(geo)
        idx, mask = jnp.asarray(idx), jnp.asarray(mask, dtype=jnp.float64)
        # float features (the noninv block outputs floats, not ±1) + a spin batch
        feats = RNG.normal(size=(5, 3, geo.N))
        spins = RNG.choice([-1.0, 1.0], size=(4, 1, geo.N))
        for x in (feats, spins):
            got = np.asarray(star_wilson_product(jnp.asarray(x), idx, mask))
            ref = ragged_star_products(geo, x)
            assert got.shape == ref.shape, f"{bc} L={L}: shape {got.shape} != {ref.shape}"
            assert np.allclose(got, ref, atol=1e-12), \
                f"{bc} L={L}: masked product != ragged (max dev " \
                f"{np.max(np.abs(got - ref)):.3e})"


# ── 4. token/flip pairing (even-overlap rule) ────────────────────────────────

def test_token_flip_pairing():
    for L, bc in ((2, "OBC"), (3, "OBC"), (2, "PBC")):
        geo = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=bc)
        x = RNG.choice([-1.0, 1.0], size=(geo.N,))
        star0 = ragged_star_products(geo, x[None, :])
        plaq0 = ragged_plaq_products(geo, x[None, :])
        for p in geo.plaq_all:                       # face flips fix star tokens
            xf = x.copy()
            xf[list(p)] *= -1
            assert np.array_equal(ragged_star_products(geo, xf[None, :]), star0), \
                f"{bc} L={L}: B_p flip moved a star token"
        for v in geo.get_vertex_all_hetero():        # star flips fix face tokens
            xf = x.copy()
            xf[v] *= -1
            assert np.array_equal(ragged_plaq_products(geo, xf[None, :]), plaq0), \
                f"{bc} L={L}: A_v flip moved a face token"
        xf = x.copy()                                # a single edge flip does not
        xf[0] *= -1
        assert not np.array_equal(ragged_star_products(geo, xf[None, :]), star0), \
            f"{bc} L={L}: single-edge flip left all star tokens fixed"


# ── 5. ansatz flip-invariance at init ────────────────────────────────────────

def _init_logpsi(model, geo, n_cfg=4):
    import jax
    x = np.asarray(RNG.choice([-1.0, 1.0], size=(n_cfg, geo.N)))
    params = model.init(jax.random.PRNGKey(0), x)
    return params, x, (lambda xx: np.asarray(model.apply(params, xx)))


def test_dual_ansatz_Bp_invariance_at_init():
    """Dual gridinv is an exact function of the star tokens at the identity
    warm start → exactly invariant under every B_p (face) flip."""
    for L, bc in ((2, "OBC"), (3, "OBC"), (2, "PBC")):
        cfg = with_defaults(dict(L=L, bc=bc, arch="ToricCNN_gridinv",
                                 dual_basis=True, kernel_size=2))
        geo = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=bc)
        model = build_model(cfg, geo)
        _params, x, logpsi = _init_logpsi(model, geo)
        base = logpsi(x)
        worst = 0.0
        for p in geo.plaq_all:
            xf = x.copy()
            xf[:, list(p)] *= -1
            worst = max(worst, float(np.max(np.abs(logpsi(xf) - base))))
        assert worst < 1e-12, f"{bc} L={L}: max |Δlog ψ| under B_p flips = {worst:.3e}"


def test_primal_ansatz_Av_invariance_at_init():
    """Primal counterpart (closes the pre-existing test gap): gridinv at the
    identity warm start is an exact function of the plaquette fluxes → exactly
    invariant under every A_v (star) flip."""
    for L, bc in ((2, "OBC"), (3, "OBC"), (2, "PBC")):
        cfg = with_defaults(dict(L=L, bc=bc, arch="ToricCNN_gridinv",
                                 kernel_size=2))
        geo = ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=bc)
        model = build_model(cfg, geo)
        _params, x, logpsi = _init_logpsi(model, geo)
        base = logpsi(x)
        worst = 0.0
        for v in geo.get_vertex_all_hetero():
            xf = x.copy()
            xf[:, v] *= -1
            worst = max(worst, float(np.max(np.abs(logpsi(xf) - base))))
        assert worst < 1e-12, f"{bc} L={L}: max |Δlog ψ| under A_v flips = {worst:.3e}"


# ── 6. guards ────────────────────────────────────────────────────────────────

def test_guards():
    geo = ThreeD_ToricCodeGeometry(Lx=2, Ly=2, Lz=2, bc="OBC")
    hi = nk.hilbert.Spin(s=1 / 2, N=geo.N)
    try:
        create_hamiltonian(hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all,
                           bonds=geo.bonds, hy=0.2, dtype="complex", dual=True)
        raise RuntimeError("dual + hy != 0 did not raise")
    except AssertionError:
        pass
    try:
        build_hamiltonian(with_defaults(dict(L=2, bc="OBC", model="fermionic",
                                             dual_basis=True)), geo, hi)
        raise RuntimeError("dual + fermionic did not raise")
    except NotImplementedError:
        pass
    for arch in ("ToricCNN_full", "GeoCNN", "VanillaCNN"):
        try:
            build_model(with_defaults(dict(L=2, bc="OBC", arch=arch,
                                           dual_basis=True)), geo)
            raise RuntimeError(f"dual + {arch} did not raise")
        except NotImplementedError:
            pass
    # sanity: vertex_grid_layout really is a permutation at a non-cubic-safe L
    dims, lin = vertex_grid_layout(geo)
    assert sorted(lin) == list(range(np.prod(dims)))


def run_all():
    steps = [
        ("H_dual == W H W (L=2 OBC dense)",       test_conjugation_identity),
        ("dual all-up: diag −(N_v+hx·N), 1+N_p",  test_structural_all_up),
        ("masked star product == ragged",          test_star_product_masked_vs_ragged),
        ("face/star flips fix the dual/primal tokens", test_token_flip_pairing),
        ("dual ansatz B_p-invariant at init",      test_dual_ansatz_Bp_invariance_at_init),
        ("primal ansatz A_v-invariant at init",    test_primal_ansatz_Av_invariance_at_init),
        ("guards (hy / fermionic / non-gridinv)",  test_guards),
    ]
    pending = 0
    for name, fn in steps:
        try:
            fn()
            print(f"  ok       {name}")
        except NotImplementedError as e:
            pending += 1
            print(f"  PENDING  {name} — {e}")
    if pending:
        print(f"{pending} test(s) pending the star_wilson_product user contribution.")
    else:
        print("All dual-basis tests passed.")


if __name__ == "__main__":
    run_all()
