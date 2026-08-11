"""Regression tests for the fold→NaN FM-ratio convention (2026-08-11 audit).

The frozen cross-method convention: a non-positive ⟨closed⟩ makes the FM ratio
UNDEFINED — every estimator returns a loud NaN, never the silent |·|-fold that
biases the ratio up near transitions; undefined jackknife replicates make the
ERROR NaN (nan-skipping would bias it down exactly in the marginal regime);
`fit_transition` drops NaN values and must never hand a NaN error bar maximum
weight. Mirrors analysis/paratoric_driver.pooled_fm — keep the two in sync.

Standalone: cd tests && ../.venv/bin/python test_fm_nan_convention.py
Synthetic inputs only (no lattice, no sampling) — safe on the dev machine.
"""
import numpy as np

from tc3d.fm import _jackknife_fm_ratio, fit_transition, fm_ratio


class _Stat:
    def __init__(self, mean, err):
        self.mean, self.error_of_mean = mean, err


class _FakeVS:
    """vstate stub: `expect(op)` unpacks op == (mean, err)."""
    n_samples = 4096

    def expect(self, op):
        return _Stat(*op)


def test_fm_ratio_healthy_and_backwards_compatible():
    vs = _FakeVS()
    o2 = fm_ratio(vs, (0.5, 0.01), (0.81, 0.02))
    o3 = fm_ratio(vs, (0.5, 0.01), (0.81, 0.02), return_den=True)
    assert len(o2) == 2 and len(o3) == 3
    assert o2 == o3[:2] and o3[2] == (0.81, 0.02)
    assert abs(o2[0] - 0.5 / np.sqrt(0.81)) < 1e-12
    print("[PASS] fm_ratio healthy value + return_den back-compat")


def test_fm_ratio_nan_at_nonpositive_den():
    vs = _FakeVS()
    for den in (-0.5, 0.0):
        val, err, (dm, de) = fm_ratio(vs, (0.3, 0.01), (den, 0.02),
                                      return_den=True)
        assert np.isnan(val) and np.isnan(err), (den, val, err)
        assert (dm, de) == (den, 0.02)          # den still travels for the gate
    print("[PASS] fm_ratio <closed> <= 0 -> NaN, den preserved (no |.|-fold)")


def test_jackknife_healthy_matches_direct():
    rng = np.random.default_rng(7)
    ro = rng.choice([-1.0, 1.0], 4096, p=[0.2, 0.8])
    rc = rng.choice([-1.0, 1.0], 4096, p=[0.1, 0.9])
    O, Oe = _jackknife_fm_ratio(ro, rc)
    direct = np.mean(ro) / np.sqrt(np.mean(rc))
    assert abs(O - direct) < 1e-12 and np.isfinite(Oe) and Oe > 0
    print("[PASS] jackknife healthy ratio == direct estimator")


def test_jackknife_nan_at_negative_den():
    rng = np.random.default_rng(7)
    ro = rng.choice([-1.0, 1.0], 4096, p=[0.2, 0.8])
    rc = rng.choice([-1.0, 1.0], 4096, p=[0.9, 0.1])   # mean < 0
    O, Oe = _jackknife_fm_ratio(ro, rc)
    assert np.isnan(O), O
    print("[PASS] jackknife mean(closed) < 0 -> NaN value")


def test_jackknife_marginal_den_gives_nan_error():
    # mean(closed) barely > 0: delete-one replicates cross <= 0 -> loud NaN err
    rc = np.array([1.0] * 9 + [-1.0] * 8, float)
    ro = np.ones_like(rc)
    O, Oe = _jackknife_fm_ratio(ro, rc, n_blocks=17)
    assert np.isfinite(O) and np.isnan(Oe), (O, Oe)
    print("[PASS] marginal den: undefined replicates -> err = NaN (not skipped)")


def test_fit_transition_nan_robust():
    rng = np.random.default_rng(3)
    h = np.linspace(0.1, 0.5, 15)
    O = 0.02 + 0.5 / (1 + np.exp(-(h - 0.3) / 0.03)) + rng.normal(0, 4e-3, 15)
    Oe = np.full(15, 4e-3)
    O[7], Oe[3], Oe[11] = np.nan, np.nan, 0.0    # NaN value, NaN bar, saturated
    fit = fit_transition(h, O, Oe)
    assert abs(fit["h_c"] - 0.3) < 0.01, fit["h_c"]
    # the NaN-bar point must NOT dominate: refit without it barely moves h_c
    keep = np.isfinite(O) & (np.arange(15) != 3)
    ref = fit_transition(h[keep], O[keep], Oe[keep])
    assert abs(fit["h_c"] - ref["h_c"]) < 5e-3, (fit["h_c"], ref["h_c"])
    print(f"[PASS] fit_transition NaN-robust (h_c = {fit['h_c']:.4f})")


def test_fit_transition_degenerate_guard():
    h = np.linspace(0.1, 0.5, 5)
    fit = fit_transition(h, np.full(5, np.nan), None)
    assert np.isnan(fit["h_c"]) and fit["popt"] is None
    print("[PASS] fit_transition all-NaN guard")


if __name__ == "__main__":
    test_fm_ratio_healthy_and_backwards_compatible()
    test_fm_ratio_nan_at_nonpositive_den()
    test_jackknife_healthy_matches_direct()
    test_jackknife_nan_at_negative_den()
    test_jackknife_marginal_den_gives_nan_error()
    test_fit_transition_nan_robust()
    test_fit_transition_degenerate_guard()
    print("All fold->NaN convention tests passed.")
