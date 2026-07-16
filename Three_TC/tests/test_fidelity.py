"""Unit tests for Three_TC/fidelity.py — the L=2 fidelity-to-ED guardrail.

Validated on a 2-qubit toy (no 3D ED), exercising the pieces that matter for the real
L=2 run: the bit→spin convention that aligns the NQS vector to the ED basis, and the
subspace-projector fidelity under a degenerate ground manifold (pt 22).
"""
import _path  # noqa: F401
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import netket as nk

from Three_TC.fidelity import (
    spin_configs_from_basis, nqs_amplitudes, subspace_fidelity,
    degenerate_manifold, fidelity_report,
)


def test_spin_config_convention():
    # arange(4), N=2: bit i = 1 -> sigma_i = -1, qubit index == bit position
    sig = spin_configs_from_basis(np.arange(4), 2)
    assert np.array_equal(sig, np.array([[+1, +1],   # b=0
                                         [-1, +1],   # b=1 (bit0=1)
                                         [+1, -1],   # b=2 (bit1=1)
                                         [-1, -1]]))  # b=3


def test_manifold_detection():
    # 2-fold degenerate ground manifold, then a gap
    assert list(degenerate_manifold(np.array([-1., -1., 1., 1.]))) == [0, 1]
    # non-degenerate: largest gap right after E0 -> single vector
    assert list(degenerate_manifold(np.array([-3., -1., 0.]))) == [0]
    # explicit override
    assert list(degenerate_manifold(np.array([-1., -1., 1.]), k_manifold=1)) == [0]


class _UpUp(nn.Module):
    """|σ0=+1, σ1=+1⟩ = basis state b=0."""
    @nn.compact
    def __call__(self, x):
        d = self.param("d", nn.initializers.zeros, (1,), jnp.complex128)
        up = (x[..., 0] > 0) & (x[..., 1] > 0)
        return jnp.where(up, 0.0 + 0j, -30.0 + 0j) + 0.0 * d.sum()


def _vs_upup():
    hi = nk.hilbert.Spin(s=0.5, N=2)
    return nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), _UpUp(), n_samples=64, seed=0)


def test_subspace_fidelity_under_degeneracy():
    # Ground manifold = span{e0, e2}; eigsh may return any orthonormal basis of it.
    # Use a rotated basis u0=(e0+e2)/√2, u1=(e0-e2)/√2 to prove rotation-invariance.
    e = np.eye(4, dtype=complex)
    u0 = (e[:, 0] + e[:, 2]) / np.sqrt(2)
    u1 = (e[:, 0] - e[:, 2]) / np.sqrt(2)
    evecs = np.column_stack([u0, u1, e[:, 1], e[:, 3]])   # (4, 4)
    evals = np.array([-1.0, -1.0, 1.0, 1.0])

    rep = fidelity_report(_vs_upup(), evals, evecs, N=2)
    # |↑↑⟩ = e0 lies fully in span{u0,u1} -> subspace F = 1, but overlap with u0 alone = 1/2
    assert abs(rep["subspace_fidelity"] - 1.0) < 1e-9, rep
    assert abs(rep["single_vector_fidelity"] - 0.5) < 1e-9, rep
    assert rep["manifold_dim"] == 2
    print(f"OK degeneracy: subspace F={rep['subspace_fidelity']:.6f} "
          f"vs single-vector {rep['single_vector_fidelity']:.6f} (manifold dim {rep['manifold_dim']})")


def test_orthogonal_state_zero_fidelity():
    # Manifold {e1, e3} (qubit0 down); |↑↑⟩=e0 is orthogonal -> F ≈ 0
    e = np.eye(4, dtype=complex)
    evecs = np.column_stack([e[:, 1], e[:, 3], e[:, 0], e[:, 2]])
    evals = np.array([-1.0, -1.0, 1.0, 1.0])
    rep = fidelity_report(_vs_upup(), evals, evecs, N=2)
    assert rep["subspace_fidelity"] < 1e-9, rep
    print(f"OK orthogonal: subspace F={rep['subspace_fidelity']:.2e}")


if __name__ == "__main__":
    test_spin_config_convention()
    test_manifold_detection()
    test_subspace_fidelity_under_degeneracy()
    test_orthogonal_state_zero_fidelity()
    print("test_fidelity PASSED")
