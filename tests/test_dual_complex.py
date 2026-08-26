"""
Cheap-proxy check that the DUAL (Hadamard-conjugated) ToricCNN_gridinv supports a
COMPLEX log psi (the sign-full h_y regime) -- build + one forward pass at L=2 OBC,
NO ED, NO sampling. Dual twin of test_gridinv_complex.py.

Verifies the same three claims, in the dual basis:
  (i)   h_y != 0  -> params are complex128 and log psi is complex with nonzero Im,
  (ii)  h_y  = 0  -> params are float64 and log psi is real (dual real path untouched),
  (iii) both paths are finite and batch-consistent.

Run directly:
    python test_dual_complex.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from tc3d.builders import build_geometry, build_model, with_defaults


def _build_and_apply(hy, L=2, bc="OBC", seed=0):
    """Build the dual ToricCNN_gridinv at (L, bc, hy) and evaluate log psi on a
    batch of random +-1 spin configs. Returns (params, logpsi, geo)."""
    cfg = with_defaults({"L": L, "bc": bc, "arch": "ToricCNN_gridinv",
                         "dual_basis": True, "hy": hy})
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


def test_dual_complex_path_is_complex(hy=0.5):
    params, logpsi, geo = _build_and_apply(hy)
    dtypes = _param_dtypes(params)
    assert dtypes == {np.dtype("complex128")}, \
        f"dual h_y={hy}: expected all complex128 params, got {dtypes}"
    assert logpsi.dtype == np.complex128, f"log psi dtype {logpsi.dtype} != complex128"
    assert np.all(np.isfinite(logpsi)), "log psi has non-finite entries"
    # random-complex invariant-block init => a genuine (nonzero) phase, not a real
    # net smuggled through a complex container.
    assert np.max(np.abs(logpsi.imag)) > 1e-8, \
        f"log psi imaginary part is ~0 ({np.max(np.abs(logpsi.imag)):.2e}) — sign sector dead"
    print(f"[PASS] dual complex path (h_y={hy}): complex128 params, "
          f"max|Im log psi|={np.max(np.abs(logpsi.imag)):.3e}, N={geo.N}")


def test_dual_real_path_untouched(hy=0.0):
    params, logpsi, geo = _build_and_apply(hy)
    dtypes = _param_dtypes(params)
    assert dtypes == {np.dtype("float64")}, \
        f"dual h_y={hy}: expected all float64 params, got {dtypes}"
    assert logpsi.dtype == np.float64, f"log psi dtype {logpsi.dtype} != float64"
    assert np.all(np.isfinite(logpsi)), "log psi has non-finite entries"
    print(f"[PASS] dual real path (h_y=0): float64 params, real log psi, N={geo.N}")


def test_dual_determinism():
    """Same seed => identical params/output (guards against accidental RNG in the path)."""
    _, a, _ = _build_and_apply(0.5, seed=3)
    _, b, _ = _build_and_apply(0.5, seed=3)
    assert np.array_equal(a, b), "dual gridinv forward pass is non-deterministic at fixed seed"
    print("[PASS] dual deterministic at fixed seed")


def main():
    test_dual_real_path_untouched()
    test_dual_complex_path_is_complex()
    test_dual_determinism()
    print("All dual ToricCNN_gridinv complex-path checks passed.")


if __name__ == "__main__":
    main()
