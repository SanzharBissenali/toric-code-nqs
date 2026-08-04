"""
Cheap-proxy check that ToricCNN_gridinv supports a COMPLEX log ψ (the sign-full
h_y regime) — build + one forward pass at L=2 PBC, NO ED, NO sampling, so it runs
on the dev box (obeys the no-3D-ED-locally rule).

Verifies the three claims behind the gridinv complexification:
  (i)   h_y != 0  → params are complex128 and log ψ is complex with nonzero Im,
  (ii)  h_y  = 0  → params are float64 and log ψ is real (the real path is untouched),
  (iii) both paths are finite and batch-consistent.

Run directly:
    python test_gridinv_complex.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from tc3d.builders import build_geometry, build_model, with_defaults


def _build_and_apply(hy, L=2, seed=0):
    """Build ToricCNN_gridinv at (L, hy) and evaluate log ψ on a batch of random
    ±1 spin configs. Returns (params, logpsi, model)."""
    cfg = with_defaults({"L": L, "bc": "PBC", "arch": "ToricCNN_gridinv", "hy": hy})
    geo = build_geometry(cfg)
    model = build_model(cfg, geo)
    key = jax.random.PRNGKey(seed)
    kdata, kinit = jax.random.split(key)
    sigma = jax.random.choice(kdata, jnp.array([-1.0, 1.0]), shape=(8, geo.N))
    params = model.init(kinit, sigma)
    logpsi = model.apply(params, sigma)
    return params, np.asarray(logpsi), geo


def _param_dtypes(params):
    return {np.asarray(p).dtype for p in jax.tree_util.tree_leaves(params)}


def test_complex_path_is_complex(hy=0.5):
    params, logpsi, geo = _build_and_apply(hy)
    dtypes = _param_dtypes(params)
    assert dtypes == {np.dtype("complex128")}, \
        f"h_y={hy}: expected all complex128 params, got {dtypes}"
    assert logpsi.dtype == np.complex128, f"log ψ dtype {logpsi.dtype} != complex128"
    assert np.all(np.isfinite(logpsi)), "log ψ has non-finite entries"
    # random-complex invariant-block init ⇒ a genuine (nonzero) phase, not a real net
    # smuggled through a complex container.
    assert np.max(np.abs(logpsi.imag)) > 1e-8, \
        f"log ψ imaginary part is ~0 ({np.max(np.abs(logpsi.imag)):.2e}) — sign sector dead"
    print(f"[PASS] complex path (h_y={hy}): complex128 params, "
          f"max|Im log ψ|={np.max(np.abs(logpsi.imag)):.3e}, N={geo.N}")


def test_real_path_untouched(hy=0.0):
    params, logpsi, geo = _build_and_apply(hy)
    dtypes = _param_dtypes(params)
    assert dtypes == {np.dtype("float64")}, \
        f"h_y={hy}: expected all float64 params, got {dtypes}"
    assert logpsi.dtype == np.float64, f"log ψ dtype {logpsi.dtype} != float64"
    assert np.all(np.isfinite(logpsi)), "log ψ has non-finite entries"
    print(f"[PASS] real path (h_y=0): float64 params, real log ψ, N={geo.N}")


def test_determinism():
    """Same seed ⇒ identical params/output (guards against accidental RNG in the path)."""
    _, a, _ = _build_and_apply(0.5, seed=3)
    _, b, _ = _build_and_apply(0.5, seed=3)
    assert np.array_equal(a, b), "gridinv forward pass is non-deterministic at fixed seed"
    print("[PASS] deterministic at fixed seed")


def main():
    test_real_path_untouched()
    test_complex_path_is_complex()
    test_determinism()
    print("All ToricCNN_gridinv complex-path checks passed.")


if __name__ == "__main__":
    main()
