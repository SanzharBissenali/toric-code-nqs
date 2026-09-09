"""Tests for analysis/scripts/firstorder_fit.py (task A7, first-order transition
locators). NetKet-free (numpy/scipy only) -- run directly:

    cd tests && PYTHONPATH=<repo> ../.venv/bin/python test_firstorder_fit.py

(1)/(2)/(2b)/(5) are synthetic and self-contained. (3)/(4) read real per-run JSONs
already in the repo tree (results/phaseB*/, results/hy_cuts_L4/) and are report-style:
they print the numbers and assert only loose sanity bounds, per the task -- a
systematic offset against the banked O_FM record is a FINDING, not a test failure,
because the energy crossing is a different estimator from an O_FM inflection.

(2b) and (5) were added after an adversarial audit found: a first-sign-flip bias in a
near-degenerate zone, hellmann_feynman() mixing branches on the winner() curve, NQS
E_err needing the house x3 inflation, and the branch-suffix regex missing seed-repeat
names -- see firstorder_fit.py's ERR_INFLATE / energy_crossing / hellmann_feynman(s) /
_branch_of for the fixes.

(6) is a user-decision follow-up: for kind="topo-trivial" cuts (hz <= 0.2, the default
for (3)/(4) below) the O_FM_membrane_R1 inflection is now the PRIMARY per-L marker
(transition_fit.locate_all + combine_default) and the energy crossing is a SECONDARY
consistency check only -- see locate_cut's kind dispatch.
"""
import json
import os
import sys
import tempfile
import warnings

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


# ----------------------------------------------------------------------------- (2b)
def test_synthetic_multiflip_bias_removed():
    """Audit finding: with a near-degenerate Delta_E zone, noise flips the sign of
    several adjacent points and "take the first flip" is biased low (audit: sigma=0.3
    over ~3 points, ~49% multi-flip draws, bias ~-0.05). Reproduce that setup and check
    the region-centred estimator's mean is within 1 sigma of the truth over 200 seeds,
    with a substantial (order the audit's ~49%) multi-flip fraction actually occurring
    -- i.e. this test exercises the multi-flip path, not accidentally avoiding it --
    and contrast against what "always take the first flip" would have given."""
    h = np.array([0.0, 1.0, 1.6, 1.8, 2.0, 2.2, 2.4, 3.0, 4.0])
    k = 1.0
    delta_true = k * (h - 2.0)          # true crossing at h*=2.0, shallow near it
    sigma = 0.3
    true_hc = 2.0
    n_seeds = 200
    hcs, hces, multi = [], [], 0
    naive_vals = []                     # what "always take the first flip" would give
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        Eu = 0.5 * delta_true + rng.normal(0, sigma, len(h))
        Ed = -0.5 * delta_true + rng.normal(0, sigma, len(h))
        up = _synthetic_table(h, Eu, np.full(len(h), sigma))
        dn = _synthetic_table(h, Ed, np.full(len(h), sigma))
        h_c, h_c_err, bracket, info = ff.energy_crossing(up, dn)
        assert h_c is not None, f"seed {seed}: expected a crossing, got info={info}"
        hcs.append(h_c)
        hces.append(h_c_err)
        multi += info["n_flips"] > 1

        delta = Eu - Ed
        flips = np.where(np.diff(np.sign(delta)) != 0)[0]
        i = flips[0]
        h0, h1, d0, d1 = h[i], h[i + 1], delta[i], delta[i + 1]
        naive_vals.append(h0 - d0 * (h1 - h0) / (d1 - d0) if d1 != d0 else 0.5 * (h0 + h1))

    hcs, hces = np.array(hcs), np.array(hces)
    naive_bias = float(np.mean(naive_vals) - true_hc)
    region_bias = float(np.mean(hcs) - true_hc)
    multi_frac = multi / n_seeds
    print(f"OK: multi-flip fraction={multi_frac:.1%} (audit reported ~49%); "
          f"naive-first-flip bias={naive_bias:+.4f}; region-centred bias={region_bias:+.4f}; "
          f"mean h_c_err={np.mean(hces):.4f}")
    assert multi_frac > 0.25, f"multi-flip path barely exercised: {multi_frac:.1%}"
    assert abs(naive_bias) > 3 * abs(region_bias), (
        "expected the naive first-flip estimator to be markedly more biased than the fix")
    assert abs(region_bias) < np.mean(hces), (
        f"region-centred mean {np.mean(hcs):.4f} not within 1 sigma ({np.mean(hces):.4f}) of "
        f"truth {true_hc}")


# ----------------------------------------------------------------------------- (3)
def test_real_data_hz0p1_sweep_hx():
    """results/phaseB_rerun/right/L{4,5,6} + results/phaseB/right/L{4,5,6}, hy=0,
    fixed hz=0.1, sweep hx. hz=0.1 <= 0.2 -> kind defaults to "topo-trivial", so the
    PRIMARY per-L marker is now the O_FM_membrane_R1 inflection via
    transition_fit.locate_all + combine_default on the winner curve -- called through
    the SAME window (0.5, 1.3) the banked record itself stores as metadata -- so this
    must reproduce results/transitions/hy0_hz0.1_sweep-hx.json's per_L values BY
    CONSTRUCTION (exact match, not just "within errors"): h_c, h_c_err, stat, syst,
    spread_over, n_points and amp all bit-identical, central="richards", FSS(x=1)
    h_inf = 0.9821(1873).

    Phase B is COLD-ONLY (no _up/_dn run names): the energy crossing (now SECONDARY
    under topo-trivial) has no up/dn branches to compare, so crossing_h_c is None and
    "merged" reflects only the PRIMARY (O_FM) locator -- which converges fine, so
    merged=False even though the crossing itself found nothing."""
    dirs = [f"results/{d}/right/L{L}" for d in ("phaseB_rerun", "phaseB") for L in (4, 5, 6)]
    dirs = [os.path.join(_ROOT, d) for d in dirs]
    window = (0.5, 1.3)
    rows = ff.locate_cut(dirs, "hx", {"hz": 0.1, "hy": 0.0}, window=window)
    assert len(rows) == 3, f"expected L=4,5,6, got {[r['L'] for r in rows]}"
    assert all(r["kind"] == "topo-trivial" for r in rows), "hz=0.1 must default to topo-trivial"

    banked = {
        4: dict(h_c=0.8028997112969953, h_c_err=0.04413497588133703, stat=0.036024742739802884,
                syst=0.025497333322863802, n_points=14, amp=0.8049643654538599,
                spread=[0.8028997112969953, 0.8287488997457574, 0.7777542331000298]),
        5: dict(h_c=0.8339788212672546, h_c_err=0.0271363621290046, stat=0.02613673531998434,
                syst=0.007297480141086943, n_points=14, amp=0.7726781184114643,
                spread=[0.8339788212672546, 0.8436823826526136, 0.8290874223704398]),
        6: dict(h_c=0.8733438483340492, h_c_err=0.07713745833888426, stat=0.0768581484764733,
                syst=0.006558390942257741, n_points=12, amp=0.760876436503899,
                spread=[0.8733438483340492, 0.8864606302185647, 0.8763057350118731]),
    }

    print("\nL  h_c(O_FM)       h_c_err      central  crossing_h_c  merged")
    Ls, hc, hce = [], [], []
    for r in rows:
        b = banked[r["L"]]
        assert r["central"] == "richards"
        assert r["h_c"] == b["h_c"], f"L={r['L']}: h_c {r['h_c']} != banked {b['h_c']}"
        assert r["h_c_err"] == b["h_c_err"]
        assert r["stat"] == b["stat"] and r["syst"] == b["syst"]
        assert r["spread_over"] == b["spread"]
        assert r["n_points"] == b["n_points"]
        assert r["amp"] == b["amp"]
        assert r["merged"] is False, f"L={r['L']}: O_FM converged, merged must be False"
        assert r["crossing_h_c"] is None, "Phase B is cold-only -- crossing has nothing to report"
        print(f"{r['L']}  {r['h_c']:.10f}  {r['h_c_err']:.10f}  {r['central']:>8}  "
              f"{r['crossing_h_c']}         {r['merged']}")
        Ls.append(r["L"]); hc.append(r["h_c"]); hce.append(r["h_c_err"])

    fss = tf.fss_fit(Ls, hc, hce, x=1.0)
    print(f"FSS(x=1): h_inf = {fss['h_inf']:.10f} +/- {fss['h_inf_err']:.10f} "
          f"(banked: 0.9821370020824981 +/- 0.1873160734618147)")
    assert fss["h_inf"] == 0.9821370020824981 and fss["h_inf_err"] == 0.1873160734618147
    print("OK: per-L O_FM primary + FSS(x=1) reproduce the banked record bit-for-bit")


# ----------------------------------------------------------------------------- (4)
def test_hy_cuts_L4_right():
    """results/hy_cuts_L4/right/hy{0.2,0.4}/L4 -- these DO have _up/_dn chain runs, and
    hz=0.1 <= 0.2 so kind defaults to "topo-trivial" here too: the PRIMARY marker is
    the O_FM_membrane_R1 inflection (converges fine -- merged=False), and the energy
    crossing is a SECONDARY field only. Recipe notes SC (2026-08-29 campaign lesson)
    already documents that at L=4 the two branches produce no surviving hysteresis and
    MERGE above the lag zone with the dn-carried branch sitting persistently lower
    (~0.05-0.3) near the window -- so crossing_h_c must be None (crossing_info["reason"]
    == "branches merged") even though the row overall is NOT merged (the O_FM primary
    converged). That distinction -- row merged vs. crossing merged -- is itself the
    deliverable of this test.

    hf_dev is PER BRANCH (audit fix: the old single number, computed on the winner()
    curve, was dominated by the up->dn branch handoff at hx=0.75->0.80 -- not a real
    Hellmann-Feynman violation, just two different states compared across a field
    step). Each branch's own within-branch deviation must be checked instead, and
    every branch present must report a value."""
    for hy in (0.2, 0.4):
        d = os.path.join(_ROOT, "results", "hy_cuts_L4", "right", f"hy{hy}", "L4")
        rows = ff.locate_cut([d], "hx", {"hz": 0.1, "hy": hy})
        assert len(rows) == 1 and rows[0]["L"] == 4
        r = rows[0]
        assert r["kind"] == "topo-trivial"
        assert r["n_up"] >= 5 and r["n_dn"] >= 5, f"hy={hy}: too few chain points, {r}"
        print(f"\nhy={hy}: h_c(O_FM)={r['h_c']:.4f}+/-{r['h_c_err']:.4f} central={r['central']} "
              f"crossing_h_c={r['crossing_h_c']} row_merged={r['merged']} "
              f"n_flips={r['crossing_info']['n_flips']} flags={r['crossing_info']['flags']} "
              f"hf_dev(by branch)={r['hf_dev']}")
        print(f"  spinodals: {r['spinodals']}")
        assert r["merged"] is False, f"hy={hy}: expected the O_FM primary to converge, got {r}"
        assert 0.5 < r["h_c"] < 1.2, f"hy={hy}: O_FM h_c {r['h_c']} outside the plausible sweep range"
        assert r["crossing_h_c"] is None, f"hy={hy}: expected the documented branch merge, got {r}"
        assert r["crossing_info"]["reason"] == "branches merged"
        assert "crossing_disagree" not in r["crossing_info"]["flags"], (
            "no crossing value exists here -- it must never be flagged as disagreeing")
        assert set(r["hf_dev"]) >= {"up", "dn"}, f"hy={hy}: missing per-branch HF entries: {r['hf_dev']}"
        for branch, dev in r["hf_dev"].items():
            assert np.isfinite(dev) and dev < 0.5, (
                f"hy={hy}: branch {branch} HF deviation implausibly large: {dev}")
        # the old (buggy) winner-curve number mixed the up->dn handoff into every
        # branch's deviation; the per-branch dn value in particular should now be much
        # smaller than that mixed number ever was (0.086 at hy=0.2, 0.121 at hy=0.4).
        old_mixed = {0.2: 0.086, 0.4: 0.121}[hy]
        assert r["hf_dev"]["dn"] < 0.5 * old_mixed, (
            f"hy={hy}: dn-branch HF deviation {r['hf_dev']['dn']} not much smaller than "
            f"the old branch-mixing estimate {old_mixed}")
        for branch in ("up", "dn"):
            sp = r["spinodals"][branch]
            assert sp is not None and sp["first_diverged"] is None, (
                f"hy={hy}: branch {branch} unexpectedly diverged: {sp}")


# ----------------------------------------------------------------------------- (6)
def test_topo_trivial_crossing_ofm_disagreement():
    """Synthetic topo-trivial cut, via real per-run JSONs written to a temp dir (so the
    full locate_cut pipeline -- load_branches/winner/energy_crossing/tf.locate_all/
    combine_default -- is exercised, not a hand-rolled shortcut). Energy branches cross
    cleanly at h_c_x=0.55 (steep slope, tiny E_err=0.01 -> a tight crossing_h_c_err);
    O_FM_membrane_R1 is IDENTICAL on both branches, a sharp sigmoid centred at h0=0.95.
    The two markers disagree by 0.4, far more than their combined error -> the PRIMARY
    (O_FM) h_c must land at ~0.95, the SECONDARY crossing_h_c at ~0.55, and
    crossing_info["flags"] must contain "crossing_disagree"."""
    h_grid = np.round(np.arange(0.0, 1.61, 0.1), 6)
    h_c_x_true, h_c_ofm_true = 0.55, 0.95
    delta_true = 4.0 * (h_grid - h_c_x_true)
    baseline = -50.0 - 10.0 * h_grid
    E_up, E_dn = 0.5 * delta_true + baseline, -0.5 * delta_true + baseline
    ofm_vals = 1.0 / (1.0 + np.exp(-(h_grid - h_c_ofm_true) / 0.05))

    def _run(name, hx, E0, ofm_v):
        return {"name": name, "config": {"L": 4, "hx": float(hx), "hy": 0.0, "hz": 0.1},
                "observables": {"E0": float(E0), "E_err": 0.01,
                                 "sx_mean": 0.5, "sx_err": 0.01, "sz_mean": 0.1, "sz_err": 0.01,
                                 "A_v_mean": 0.9, "A_v_err": 0.01, "B_p_mean": 0.5, "B_p_err": 0.01,
                                 "O_FM_membrane_R1": float(ofm_v), "O_FM_membrane_R1_err": 0.01},
                "diverged": False}

    with tempfile.TemporaryDirectory() as td:
        for i, hx in enumerate(h_grid):
            for suffix, E in (("up", E_up[i]), ("dn", E_dn[i])):
                path = os.path.join(td, f"pt_{i}_{suffix}.json")
                with open(path, "w") as f:
                    json.dump(_run(f"pt_{i}_{suffix}", hx, E, ofm_vals[i]), f)

        rows = ff.locate_cut([td], "hx", {"hz": 0.1, "hy": 0.0}, kind="topo-trivial")
        assert len(rows) == 1
        r = rows[0]
        print(f"\nOK: h_c(O_FM)={r['h_c']:.4f} (truth {h_c_ofm_true}), "
              f"crossing_h_c={r['crossing_h_c']:.4f} (truth {h_c_x_true}), "
              f"flags={r['crossing_info']['flags']}")
        assert r["merged"] is False
        assert abs(r["h_c"] - h_c_ofm_true) < 0.02, f"O_FM primary off: {r['h_c']}"
        assert abs(r["crossing_h_c"] - h_c_x_true) < 0.02, f"crossing secondary off: {r['crossing_h_c']}"
        combined_err = (r["h_c_err"] or 0.0) + (r["crossing_h_c_err"] or 0.0)
        assert abs(r["h_c"] - r["crossing_h_c"]) > 3 * combined_err, (
            "the two markers should disagree by far more than their combined error")
        assert "crossing_disagree" in r["crossing_info"]["flags"], (
            f"expected the disagreement flag, got flags={r['crossing_info']['flags']}")


# ----------------------------------------------------------------------------- (5)
def test_branch_suffix_seed_repeat_and_warning():
    """Audit fix: `_BRANCH_RE` must accept an optional seed-repeat suffix
    ("..._up_s1" -> "up", not "cold"), and must warn (never silently reclassify) for a
    name that contains "_up"/"_dn" WITHOUT matching the end-suffix convention."""
    assert ff._branch_of("gridinv_L4_hx0.8_up") == "up"
    assert ff._branch_of("gridinv_L4_hx0.8_dn") == "dn"
    assert ff._branch_of("gridinv_L4_hx0.8_up_s1") == "up"
    assert ff._branch_of("gridinv_L4_hx0.8_dn_s12") == "dn"
    assert ff._branch_of("phaseB2_dt01n500_L4_hx0.8_hz0.1_s1") == "cold"   # plain rerun seed, unrelated
    assert ff._branch_of("phaseB2_L4_hx0.8_hz0.1") == "cold"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        branch = ff._branch_of("gridinv_L4_hx0.8_up_wc11")   # "_up" present but NOT at the end
        assert branch == "cold"
        assert len(caught) == 1 and "up" in str(caught[0].message).lower()
    print("OK: '_up_s<seed>' suffix classified correctly; a stray mid-name '_up' "
          "falls back to 'cold' AND emits exactly one warning (never silent)")

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        ff._branch_of("phaseB2_dt01n500_L4_hx0.8_hz0.1_s1")   # unambiguous, no "_up"/"_dn" substring
        assert len(caught2) == 0, "unrelated seed suffix must not warn"
    print("OK: an unrelated '_s<seed>' rerun name warns zero times")


if __name__ == "__main__":
    test_synthetic_crossing_recovered()
    test_synthetic_merged_branches()
    test_synthetic_multiflip_bias_removed()
    test_real_data_hz0p1_sweep_hx()
    test_hy_cuts_L4_right()
    test_branch_suffix_seed_repeat_and_warning()
    test_topo_trivial_crossing_ofm_disagreement()
    print("\ntest_firstorder_fit PASSED")
