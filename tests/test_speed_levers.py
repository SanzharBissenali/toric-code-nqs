"""
Speed levers (branch p3d/speed-research) -- correctness gates, all local, seconds.

  1. UnfoldedConv3D == nn.Conv: same parameter tree from the same rng and the same
     output (1e-12) for SAME and CIRCULAR padding, even/odd kernels, real/complex.
  2. ToricCNN_gridinv(_dual) with inv_impl="dense" has the IDENTICAL parameter
     tree as "conv" (same seed) and identical log psi (1e-12), OBC and PBC, primal
     and dual, real and complex.
  3. compute_dtype="float32": params stay complex128/float64, log psi comes back in
     the parameter dtype and agrees with the double forward pass to single precision.
  4. kernel_solve == cholesky on the dense QGT: random O with n_params > rows
     (the L=6-complex regime), and a real L=2 OBC SR step (identical dp).
  5. exact_qgt_apply_fun: None for a double model; for a float32 model the twin's
     QGT equals the double model's QGT (1e-12) while the fast model's own QGT
     only agrees to ~1e-6 -- i.e. the hook does keep the SR geometry in double.
  6. 3-step L=2 OBC run_loop: conv vs dense give the same energy trajectory (1e-8).

Run:  cd tests && ../.venv/bin/python test_speed_levers.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk

jax.config.update("jax_enable_x64", True)

from tc3d.networks import UnfoldedConv3D
from tc3d.builders import (with_defaults, build_geometry, build_model, build_state,
                           run_loop, kernel_solve, exact_qgt_apply_fun,
                           _qgt_dense_with_apply)

RNG = np.random.default_rng(11)


def _tree_equal(a, b):
    la, lb = jax.tree_util.tree_leaves_with_path(a), jax.tree_util.tree_leaves_with_path(b)
    if [k for k, _ in la] != [k for k, _ in lb]:
        return False
    return all(x.shape == y.shape and x.dtype == y.dtype and bool(jnp.array_equal(x, y))
               for (_, x), (_, y) in zip(la, lb))


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-300))


def _cfg(L, bc, hy, **kw):
    return with_defaults(dict(L=L, bc=bc, dual_basis=kw.pop("dual", True), hx=0.2, hz=0.26,
                              hy=hy, arch="ToricCNN_gridinv", noninv_hidden=[4, 8],
                              inv_hidden=[8, 8], kernel_size=L - 1, n_samples=256,
                              n_chains=16, chunk_size=None, **kw))


def _spins(n, N):
    return jnp.asarray(RNG.choice([-1, 1], size=(n, N)), dtype=jnp.int8)


def _perturb(params, scale=0.3):
    """Random complex/real kick so the noninv block is NOT the identity (exercises
    every layer, not just the conv stack)."""
    leaves, tree = jax.tree_util.tree_flatten(params)
    out = []
    for x in leaves:
        r = RNG.standard_normal(x.shape)
        if jnp.iscomplexobj(x):
            r = r + 1j * RNG.standard_normal(x.shape)
        out.append(x + scale * jnp.asarray(r, dtype=x.dtype))
    return jax.tree_util.tree_unflatten(tree, out)


# 1 ---------------------------------------------------------------------------
def test_unfolded_conv_matches_nn_conv():
    cases = [(3, (4, 4, 4), "SAME"), (4, (5, 5, 5), "SAME"), (5, (6, 6, 6), "SAME"),
             (2, (3, 3, 3), "SAME"), (3, (4, 4, 4), "CIRCULAR"), (4, (4, 4, 4), "CIRCULAR"),
             (2, (3, 3, 3), "CIRCULAR")]
    worst = 0.0
    for k, dims, pad in cases:
        for dtype in (jnp.float64, jnp.complex128):
            for cin, cout in ((1, 8), (8, 8), (8, 1)):
                x = RNG.standard_normal((6, *dims, cin))
                if dtype == jnp.complex128:
                    x = x + 1j * RNG.standard_normal(x.shape)
                x = jnp.asarray(x, dtype=dtype)
                ref = nn.Conv(cout, (k,) * 3, padding=pad, param_dtype=dtype)
                alt = UnfoldedConv3D(cout, k, dims, padding=pad, param_dtype=dtype)
                key = jax.random.PRNGKey(3)
                pr, pa = ref.init(key, x), alt.init(key, x)
                assert _tree_equal(pr, pa), f"param tree differs {k, dims, pad, dtype}"
                pr = _perturb(pr)                         # non-trivial bias too
                err = _rel(alt.apply(pr, x), ref.apply(pr, x))
                worst = max(worst, err)
                assert err < 1e-12, f"unfolded != conv: {err:.2e} at {k, dims, pad, dtype, cin, cout}"
    print(f"[1] UnfoldedConv3D == nn.Conv over {len(cases)}x2x3 cases; worst rel err {worst:.1e}")


# 2 ---------------------------------------------------------------------------
def test_gridinv_dense_equals_conv():
    cases = [(3, "OBC", 0.4, True), (4, "OBC", 0.0, True), (3, "PBC", 0.4, True),
             (3, "OBC", 0.4, False), (3, "PBC", 0.0, False)]
    for L, bc, hy, dual in cases:
        cfg = _cfg(L, bc, hy, dual=dual)
        geo = build_geometry(cfg)
        m_conv = build_model({**cfg, "inv_impl": "conv"}, geo)
        m_dense = build_model({**cfg, "inv_impl": "dense"}, geo)
        x = _spins(64, geo.N)
        key = jax.random.PRNGKey(5)
        p1, p2 = m_conv.init(key, x), m_dense.init(key, x)
        assert _tree_equal(p1, p2), f"param trees differ (L={L} {bc} hy={hy} dual={dual})"
        p1 = _perturb(p1)
        err = _rel(m_dense.apply(p1, x), m_conv.apply(p1, x))
        assert err < 1e-12, f"dense != conv log psi: {err:.2e} (L={L} {bc} hy={hy} dual={dual})"
        print(f"[2] L={L} {bc} hy={hy} dual={dual}: same param tree; |dlogpsi| rel {err:.1e}")


# 3 ---------------------------------------------------------------------------
def test_compute_dtype_float32():
    for L, bc, hy in [(3, "OBC", 0.4), (3, "OBC", 0.0)]:
        cfg = _cfg(L, bc, hy)
        geo = build_geometry(cfg)
        x = _spins(64, geo.N)
        key = jax.random.PRNGKey(5)
        m64 = build_model(cfg, geo)
        p = _perturb(m64.init(key, x))
        y64 = m64.apply(p, x)
        for impl in ("conv", "dense"):
            m32 = build_model({**cfg, "compute_dtype": "float32", "inv_impl": impl}, geo)
            assert _tree_equal(m32.init(key, x), m64.init(key, x)), "float32 changed the params"
            y32 = m32.apply(p, x)
            assert y32.dtype == y64.dtype, f"output dtype {y32.dtype} != param dtype {y64.dtype}"
            err = _rel(y32, y64)
            assert 0 < err < 1e-4, f"float32 forward off by {err:.2e} ({impl}, hy={hy})"
            print(f"[3] L={L} hy={hy} {impl}: compute_dtype=float32 rel err {err:.1e} "
                  f"(output {y32.dtype})")
            assert m32.precision == jax.lax.Precision.HIGHEST, "float32 must be strict fp32"
            mt = build_model({**cfg, "compute_dtype": "tf32", "inv_impl": impl}, geo)
            assert mt.precision is None and mt.compute_dtype == m32.compute_dtype
            assert _rel(mt.apply(p, x), y64) < 1e-4       # CPU: tf32 == float32 numerics


# 4 ---------------------------------------------------------------------------
def test_kernel_solver_matches_cholesky():
    from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT
    for mode, shape in (("complex", (64, 2, 300)), ("real", (64, 300)), ("real", (300, 64))):
        O = jnp.asarray(RNG.standard_normal(shape)) / 8
        A = QGTJacobianDenseT(O=O, scale=None, mode=mode, diag_shift=1e-3)
        b = jnp.asarray(RNG.standard_normal(shape[-1]))
        xc, _ = nk.optimizer.solver.cholesky(A, b)
        xk, _ = kernel_solve(A, b)
        err = _rel(xk, xc)
        assert err < 1e-9, f"kernel_solve != cholesky: {err:.2e} ({mode}, {shape})"
        print(f"[4a] random O {shape} mode={mode}: kernel vs cholesky rel err {err:.1e}")

    # the real thing: one SR step at L=2 OBC, complex ansatz (n_params_real 3618 > 512 rows)
    cfg = _cfg(2, "OBC", 0.4)
    geo, hi, Ham, vs, _ = build_state(cfg)
    vs.parameters = _perturb(vs.parameters, 0.1)
    vs.sample()
    _, grad = vs.expect_and_grad(Ham)
    QGT = nk.optimizer.qgt.QGTJacobianDense
    dps = {}
    for name, solver in (("cholesky", nk.optimizer.solver.cholesky), ("kernel", kernel_solve)):
        sr = nk.optimizer.SR(qgt=QGT, diag_shift=1e-3, holomorphic=False, solver=solver)
        dps[name] = jax.tree_util.tree_leaves(sr(vs, grad, 0))
    err = max(_rel(a, b) for a, b in zip(dps["kernel"], dps["cholesky"]))
    assert err < 1e-8, f"L=2 SR step: kernel vs cholesky dp rel err {err:.2e}"
    print(f"[4b] L=2 OBC hy=0.4 SR step (n_params={vs.n_parameters} complex, "
          f"{2 * cfg['n_samples']} rows): kernel vs cholesky dp rel err {err:.1e}")


# 5 ---------------------------------------------------------------------------
def test_exact_qgt_twin():
    cfg = _cfg(2, "OBC", 0.4)
    geo, hi, Ham, vs, _ = build_state(cfg)
    assert exact_qgt_apply_fun(vs) is None, "double model must not get a twin"
    _, _, _, vs32, _ = build_state({**cfg, "compute_dtype": "float32", "inv_impl": "dense"})
    twin = exact_qgt_apply_fun(vs32)
    assert twin is not None
    vs.parameters = _perturb(vs.parameters, 0.1)
    vs.sample()
    S_base = nk.optimizer.qgt.QGTJacobianDense(vs, diag_shift=1e-3, holomorphic=False).to_dense()
    S_twin = _qgt_dense_with_apply(vs, apply_fun=twin, diag_shift=1e-3,
                                   holomorphic=False).to_dense()
    S_fast = _qgt_dense_with_apply(vs, apply_fun=vs32._apply_fun, diag_shift=1e-3,
                                   holomorphic=False).to_dense()
    e_twin, e_fast = _rel(S_twin, S_base), _rel(S_fast, S_base)
    assert e_twin < 1e-12, f"double twin QGT differs from the double model's: {e_twin:.2e}"
    assert 1e-9 < e_fast < 1e-3, f"fast-model QGT should differ at fp32 level, got {e_fast:.2e}"
    print(f"[5] QGT(double twin) vs QGT(double model): {e_twin:.1e}; "
          f"QGT(float32 model) vs same: {e_fast:.1e}  -> twin keeps SR geometry in double")
    # the dense-GEMM model (double) must ALSO get a twin (conv impl) -- NetKet's
    # per-sample Jacobian would otherwise materialise chunk x (P*C)^2 cotangents
    _, _, _, vsd, _ = build_state({**cfg, "inv_impl": "dense"})
    twin_d = exact_qgt_apply_fun(vsd)
    assert twin_d is not None and vsd.model.inv_impl == "dense"
    S_twin_d = _qgt_dense_with_apply(vs, apply_fun=twin_d, diag_shift=1e-3,
                                     holomorphic=False).to_dense()
    e_d = _rel(S_twin_d, S_base)
    assert e_d < 1e-12, f"conv twin of the dense model differs: {e_d:.2e}"
    print(f"[5] QGT(conv twin of inv_impl=dense model) vs QGT(double model): {e_d:.1e}")


# 6 ---------------------------------------------------------------------------
def test_run_loop_conv_vs_dense():
    traj = {}
    for impl in ("conv", "dense"):
        cfg = _cfg(2, "OBC", 0.4, inv_impl=impl, seed=3)
        geo, hi, Ham, vs, _ = build_state(cfg)
        Es = []
        run_loop(vs, Ham, n_iter=3, dt=0.02, diag_shift=1e-3, qgt="dense",
                 qgt_solver="cholesky", on_step=lambda s, E, v: Es.append(complex(E.mean)))
        traj[impl] = np.array(Es)
    err = float(np.max(np.abs(traj["conv"] - traj["dense"])) / np.max(np.abs(traj["conv"])))
    assert err < 1e-8, f"conv vs dense trajectories differ: {err:.2e}"
    print(f"[6] 3-step L=2 run_loop conv vs dense: E trajectory rel diff {err:.1e}  "
          f"E={np.real(traj['conv']).round(4)}")


if __name__ == "__main__":
    test_unfolded_conv_matches_nn_conv()
    test_gridinv_dense_equals_conv()
    test_compute_dtype_float32()
    test_kernel_solver_matches_cholesky()
    test_exact_qgt_twin()
    test_run_loop_conv_vs_dense()
    print("ALL PASSED")
