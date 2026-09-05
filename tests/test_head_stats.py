"""Head-timing instrumentation on `SignFramedOperator` (t_head/n_head_configs)
and its plumbing through `tc3d.builders.run_loop`'s instrumented path.

L=2 OBC fermionic, --sign_frame anaC (safe on the dev machine, well under a
minute).

Run directly:
    cd tests && PYTHONPATH=.. ../.venv/bin/python test_head_stats.py
"""

import numpy as np

from tc3d.builders import build_state, run_loop

# Same config family as tests/test_sign_frame.py's BASE (L=2 OBC fermionic
# gridinv), just a smaller n_samples for a fast run_loop pass below.
BASE = dict(L=2, bc="OBC", model="fermionic", arch="ToricCNN_gridinv", kernel_size=2,
            noninv_hidden=[4, 8], inv_hidden=[8, 8], n_noninv=2, noninv_channels=4,
            hx=0.2, hz=0.1, n_samples=256, n_chains=8, n_discard=4, seed=1,
            sign_frame="anaC")


def test_get_conn_padded_head_stats(geo, Ham):
    """One get_conn_padded call touches B (sample) + B*n_conn (connected-config)
    rows of the sign head; pop_head_stats reports that row count and a positive
    wall-clock, and a second pop (no intervening call) drains to zero."""
    B = 32
    x = 1.0 - 2.0 * np.random.default_rng(0).integers(0, 2, size=(B, geo.N)).astype(float)
    Ham.get_conn_padded(x)
    stats = Ham.pop_head_stats()
    expected = B * (1 + Ham.max_conn_size)
    assert stats["n_head_configs"] == expected, \
        f"n_head_configs = {stats['n_head_configs']}, expected {expected}"
    assert stats["t_head"] > 0.0, f"t_head = {stats['t_head']} (expected > 0)"

    stats2 = Ham.pop_head_stats()
    assert stats2["t_head"] == 0.0 and stats2["n_head_configs"] == 0, \
        f"second pop not drained to zero: {stats2}"
    return stats, stats2


def test_run_loop_head_stats(geo, vs, Ham):
    """run_loop(time_phases=True) merges Ham.pop_head_stats() into `td` every
    step: t_head/n_head_configs ride along on_timing, and n_head_configs
    matches n_samples*(1+max_conn) (the sample call + the connected-config
    call inside expect_and_grad's get_conn_padded)."""
    collected = []

    def collect(step, td):
        collected.append(dict(td))

    run_loop(vs, Ham, n_iter=3, dt=0.02, diag_shift=1e-3, time_phases=True,
             on_timing=collect)

    assert len(collected) == 3
    expected = vs.n_samples * (1 + Ham.max_conn_size)
    checked = [td for step, td in enumerate(collected) if step > 0]
    assert checked, "no steps>0 collected to check"
    for td in checked:
        assert "t_head" in td and "n_head_configs" in td, f"missing head keys: {td.keys()}"
        assert td["n_head_configs"] == expected, \
            f"n_head_configs = {td['n_head_configs']}, expected {expected}"
        assert td["t_head"] > 0.0
        for k in ("sample", "grad", "qgt", "update", "total"):
            assert k in td, f"missing pre-existing timing key {k!r}"
    return collected


if __name__ == "__main__":
    geo0, hi0, Ham0, vs0, _ = build_state(BASE)
    print(f"  ok  built L=2 OBC fermionic sign_frame='anaC' state "
          f"(N={geo0.N}, max_conn_size={Ham0.max_conn_size})")

    stats, stats2 = test_get_conn_padded_head_stats(geo0, Ham0)
    print(f"  ok  get_conn_padded: n_head_configs={stats['n_head_configs']} "
          f"t_head={stats['t_head']:.3e}s; second pop drained to {stats2}")

    geo1, hi1, Ham1, vs1, _ = build_state({**BASE, "n_samples": 64})
    collected = test_run_loop_head_stats(geo1, vs1, Ham1)
    print(f"  ok  run_loop(time_phases=True) merges t_head/n_head_configs into td "
          f"every step (n_samples={vs1.n_samples}, max_conn_size={Ham1.max_conn_size}); "
          f"step1 td = {collected[1]}")
