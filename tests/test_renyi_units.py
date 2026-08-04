"""Regression test for the S₂ bits-vs-nats convention (tc3d/renyi.py).

NetKet 3.16.1's Renyi2EntanglementEntropy returns S₂ = −log₂ Tr(ρ²) in BITS, while
renyi.py reports/compares in NATS (anchors S2_EXACT_HZ0 = 3 ln2, bound 4 ln2, GF(2)
check). `_s2_of_state` must convert bits → nats (×ln2). This test runs the ACTUAL
NetKet estimator through `_s2_of_state` on a 2-qubit state with an analytically known
S₂ and asserts the nats value — so it fails if the ×ln2 conversion is dropped.

No 3D geometry / ED needed (a 2-qubit toy state), so it is safe on the dev box.
"""
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from netket.experimental.observable import Renyi2EntanglementEntropy

from tc3d.renyi import _s2_of_state


def _analytic_s2_nats(t: float) -> float:
    """S₂ (nats) of psi[s0,s1] with (++)=(--)=1, (+-)=(-+)=t; subsystem A={0}.
    rho_A eigenvalues (1±t)²/(2+2t²)  ->  Tr rho_A² = [(1+t)^4+(1-t)^4]/(2+2t²)²."""
    trrho2 = ((1 + t) ** 4 + (1 - t) ** 4) / (2 + 2 * t ** 2) ** 2
    return float(-np.log(trrho2))


class _Psi(nn.Module):
    t: float

    @nn.compact
    def __call__(self, x):                       # x: (..., 2) spins ±1
        d = self.param("d", nn.initializers.zeros, (1,), jnp.complex128)  # dummy param
        same = x[..., 0] * x[..., 1]             # +1 equal, -1 unequal
        return jnp.where(same > 0, 0.0 + 0j, np.log(self.t) + 0j) + 0.0 * d.sum()


def test_s2_reported_in_nats():
    t = 0.5
    s2_nats = _analytic_s2_nats(t)               # 0.19845
    s2_bits = s2_nats / np.log(2.0)              # 0.28630  (what a missing ×ln2 gives)

    hi = nk.hilbert.Spin(s=0.5, N=2)
    vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), _Psi(t=t),
                        n_samples=2 ** 15, seed=0)
    obs = [("A0", Renyi2EntanglementEntropy(hi, partition=[0]))]
    s2, s2e, _ = _s2_of_state(vs, obs)

    # nats (correct) within MC tolerance; and clearly NOT the bits value.
    assert abs(s2 - s2_nats) < 0.03, f"S2={s2:.4f} not ~ {s2_nats:.4f} nats (err {s2e:.4f})"
    assert abs(s2 - s2_bits) > 0.05, f"S2={s2:.4f} looks like BITS ({s2_bits:.4f}) — ×ln2 missing"
    print(f"OK: S2={s2:.4f}±{s2e:.4f} nats (analytic {s2_nats:.4f}; bits would be {s2_bits:.4f})")


if __name__ == "__main__":
    test_s2_reported_in_nats()
    print("test_renyi_units PASSED")
