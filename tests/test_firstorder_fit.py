"""Tests for analysis/scripts/firstorder_fit.py (task A7, first-order transition
locators). NetKet-free (numpy/scipy only) -- run directly:

    cd tests && PYTHONPATH=<repo> ../.venv/bin/python test_firstorder_fit.py

(1)/(2) are synthetic and self-contained. (3)/(4) read real per-run JSONs already in
the repo tree (results/phaseB*/, results/hy_cuts_L4/) and are report-style: they print
the numbers and assert only loose sanity bounds, per the task -- a systematic offset
against the banked O_FM record is a FINDING, not a test failure, because the energy
crossing is a different estimator from an O_FM inflection.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "scripts"))
import firstorder_fit as ff                                    # noqa: E402
import transition_fit as tf                                    # noqa: E402


def _synthetic_table(h, E, E_err):
    h = np.asarray(h, float)
    return ff.Table(L=4, h=h, E0=np.asarray(E, float), E_err=np.asarray(E_err, float),
                     obs={}, names=[f"pt{i}" for i in range(len(h))],
                     diverged=np.zeros(len(h), bool))


# ----------------------------------------------------------------------------- (1)
def test_synthetic_crossing_recovered():
    """E_up = a + b*h, E_dn = c + d*h, Gaussian noise sigma=0.02; the true crossing
    h* = (c-a)/(b-d) must fall within 2*h_c_err of the estimate in almost every one of
    200 independent noise draws."""
    h = np.round(np.arange(0.0, 4.01, 0.2), 6)
    a, b, c, d = -10.0, -1.0, -8.0, -2.0
    true_hc = (c - a) / (b - d)          # = 2.0
    sigma = 0.02
    n_seeds = 200
    passed, hcs = 0, []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        Eu = a + b * h + rng.normal(0, sigma, len(h))
        Ed = c + d * h + rng.normal(0, sigma, len(h))
        up = _synthetic_table(h, Eu, np.full(len(h), sigma))
        dn = _synthetic_table(h, Ed, np.full(len(h), sigma))
        h_c, h_c_err, bracket, info = ff.energy_crossing(up, dn)
        assert h_c is not None, f"seed {seed}: expected a crossing, got info={info}"
        hcs.append(h_c)
        passed += abs(h_c - true_hc) < 2 * h_c_err
    frac = passed / n_seeds
    print(f"OK: true h_c={true_hc}; recovered mean={np.mean(hcs):.4f} std={np.std(hcs):.4f}; "
          f"{passed}/{n_seeds} ({frac:.1%}) within 2 sigma")
    assert frac >= 0.90, f"2-sigma coverage too low: {frac:.1%}"


# ----------------------------------------------------------------------------- (2)
def test_synthetic_merged_branches():
    """One branch strictly below the other everywhere on the common grid -> no sign
    change in Delta_E -> (None, None, None, info['reason']=='branches merged')."""
    h = np.round(np.arange(0.0, 4.01, 0.2), 6)
    up = _synthetic_table(h, -10.0 - h, np.full(len(h), 0.02))     # always lower
    dn = _synthetic_table(h, -5.0 - h, np.full(len(h), 0.02))      # always higher
    h_c, h_c_err, bracket, info = ff.energy_crossing(up, dn)
    assert h_c is None and h_c_err is None and bracket is None
    assert info["reason"] == "branches merged", info
    print(f"OK: merged branches -> None, info={info}")

    # disjoint grids (no common field value at all) -> "no overlap", distinct reason
    up2 = _synthetic_table([0.0, 0.2, 0.4], [-1.0, -1.2, -1.4], [0.02] * 3)
    dn2 = _synthetic_table([5.0, 5.2, 5.4], [-1.0, -1.2, -1.4], [0.02] * 3)
    h_c2, _, _, info2 = ff.energy_crossing(up2, dn2)
    assert h_c2 is None and info2["reason"] == "no overlap", info2
    print(f"OK: disjoint grids -> None, info={info2}")


# ----------------------------------------------------------------------------- (3)
def test_real_data_hz0p1_sweep_hx():
    """results/phaseB_rerun/right/L{4,5,6} + results/phaseB/right/L{4,5,6}, hy=0,
    fixed hz=0.1, sweep hx. Phase B is COLD-ONLY (no _up/_dn run names) so up/dn are
    empty and energy_crossing must report every L "no overlap"/merged; the fallback is
    the winner-curve jump locators (--ofm turns on O_FM_membrane_R1 for a direct
    comparison against the banked O_FM-based record hy0_hz0.1_sweep-hx.json).

    Banked per_L reference (relayed by the orchestrator from that record, 2 marker
    policies logged historically for the same data):
      central=richards (current banked file): 0.803(36) / 0.834(26) / 0.873(77) at
        L=4/5/6, FSS(x=1) h_inf = 0.98(19).
      central=logistic (superseded policy):    0.829     / 0.844     / 0.886,
        FSS(x=1) h_inf = 1.03(8).
    """
    dirs = [f"results/{d}/right/L{L}" for d in ("phaseB_rerun", "phaseB") for L in (4, 5, 6)]
    dirs = [os.path.join(_ROOT, d) for d in dirs]
    rows = ff.locate_cut(dirs, "hx", {"hz": 0.1, "hy": 0.0}, want_ofm=True)
    assert len(rows) == 3, f"expected L=4,5,6, got {[r['L'] for r in rows]}"

    richards_ref = {4: 0.8028997112969953, 5: 0.8339788212672546, 6: 0.8733438483340492}
    logistic_ref = {4: 0.8287488997457574, 5: 0.8436823826526136, 6: 0.8864606302185647}

    print("\nL  n_up n_dn merged | OFM logistic | OFM richards | vs richards-ref | vs logistic-ref")
    hc_log, hce_log = [], []
    for r in rows:
        assert r["merged"] is True, f"L={r['L']}: expected merged=True (cold-only dirs), got {r}"
        assert r["n_up"] == 0 and r["n_dn"] == 0
        ofm = r["secondary"].get("O_FM_membrane_R1", {})
        log_fit, rich_fit = ofm.get("logistic"), ofm.get("richards")
        assert log_fit is not None and log_fit.ok(), f"L={r['L']}: O_FM logistic locator did not converge"
        d_rich = (rich_fit.h_c - richards_ref[r["L"]]) if rich_fit is not None and rich_fit.ok() else np.nan
        d_log = log_fit.h_c - logistic_ref[r["L"]]
        print(f"{r['L']}  {r['n_up']:>4} {r['n_dn']:>4}   {r['merged']!s:>5} | "
              f"{log_fit.h_c:.4f}      | {rich_fit.h_c if rich_fit else float('nan'):.4f}       | "
              f"{d_rich:+.4f}          | {d_log:+.4f}")
        hc_log.append(log_fit.h_c)
        hce_log.append(log_fit.h_c_err)
        # sanity bound, not a strict-agreement requirement: the crossing-fallback
        # jump locator and the banked value are the SAME estimator (O_FM logistic
        # inflection) on the SAME winner curve, so they should differ only by fit
        # numerics, not by a physical offset.
        assert abs(d_log) < 0.02, f"L={r['L']}: unexpectedly large offset from the banked logistic value"

    fss_log = tf.fss_fit([4, 5, 6], hc_log, hce_log, x=1.0)
    print(f"my O_FM-logistic FSS(x=1): h_inf = {fss_log['h_inf']:.4f} +/- {fss_log['h_inf_err']:.4f} "
          f"(banked richards-policy h_inf = 0.98(19); banked logistic-policy h_inf = 1.03(8))")


# ----------------------------------------------------------------------------- (4)
def test_hy_cuts_L4_right():
    """results/hy_cuts_L4/right/hy{0.2,0.4}/L4 -- these DO have _up/_dn chain runs.
    Recipe notes SC (2026-08-29 campaign lesson) already documents that at L=4 the two
    branches produce no surviving hysteresis and MERGE above the lag zone with the
    dn-carried branch sitting persistently lower (~0.05-0.3) near the window -- so
    energy_crossing is expected to report "branches merged" here, not a crossing. That
    is itself the deliverable of this test: confirming the tool reproduces the
    documented physics rather than mis-reporting a spurious crossing."""
    for hy in (0.2, 0.4):
        d = os.path.join(_ROOT, "results", "hy_cuts_L4", "right", f"hy{hy}", "L4")
        rows = ff.locate_cut([d], "hx", {"hz": 0.1, "hy": hy}, want_ofm=False)
        assert len(rows) == 1 and rows[0]["L"] == 4
        r = rows[0]
        assert r["n_up"] >= 5 and r["n_dn"] >= 5, f"hy={hy}: too few chain points, {r}"
        print(f"\nhy={hy}: n_up={r['n_up']} n_dn={r['n_dn']} merged={r['merged']} "
              f"crossing_info={r['crossing_info']} hf_dev(max_rel)={r['hf_dev']:.4f}")
        print(f"  spinodals: {r['spinodals']}")
        assert r["merged"] is True, f"hy={hy}: expected the documented merge, got a crossing at {r['h_c']}"
        assert r["crossing_info"]["reason"] == "branches merged"
        assert r["hf_dev"] < 0.5, f"hy={hy}: Hellmann-Feynman deviation implausibly large: {r['hf_dev']}"
        for branch in ("up", "dn"):
            sp = r["spinodals"][branch]
            assert sp is not None and sp["first_diverged"] is None, (
                f"hy={hy}: branch {branch} unexpectedly diverged: {sp}")


if __name__ == "__main__":
    test_synthetic_crossing_recovered()
    test_synthetic_merged_branches()
    test_real_data_hz0p1_sweep_hx()
    test_hy_cuts_L4_right()
    print("\ntest_firstorder_fit PASSED")
