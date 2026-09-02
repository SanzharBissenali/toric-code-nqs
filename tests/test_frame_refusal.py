"""
Audit-gap test: psi-level estimators (fm.py's FM ratios/membranes, renyi.py's
Renyi-S2 swap kernel) act on psi = S*A directly and cannot be fixed by operator
framing (tc3d/sign_frame.py); a checkpoint trained with `--sign_frame != none`
stores the positive trunk A, so `load_vstate`/`fm_sweep`/`renyi_sweep` must
refuse it instead of silently scoring A as if it were psi.

No ED, no sampling, no real weights: every check here fires before
`build_state`/weight loading, so a bare fake `{name}.json` is enough.

Run directly:
    python test_frame_refusal.py
"""

import json
import os
import tempfile

from tc3d.fm import _refuse_if_framed as fm_guard, load_vstate, fm_sweep
from tc3d.renyi import _refuse_if_framed as renyi_guard, renyi_sweep

# Mirrors the config sub-dict schema written by train.py / read by fm.load_vstate
# (inspected from results/fermionic_hx_ladder/*.json).
_BASE_CFG = {
    "L": 2, "bc": "OBC", "model": "fermionic", "arch": "ToricCNN_gridinv",
    "hx": 1.0, "hz": 0.0, "hy": 0.0, "name": "fake_run",
}


def test_guards_are_the_same_function():
    """renyi.py imports the guard from fm.py -- one implementation, two entry points."""
    assert fm_guard is renyi_guard


def test_direct_guard_raises_on_framed():
    cfg = dict(_BASE_CFG, sign_frame="anaC")
    for guard in (fm_guard, renyi_guard):
        try:
            guard(cfg, "fake_run")
        except ValueError as e:
            msg = str(e)
            assert "fake_run" in msg and "anaC" in msg and "refusing" in msg, msg
        else:
            raise AssertionError(f"{guard} did not raise on sign_frame=anaC")


def test_direct_guard_passes_on_none():
    for guard in (fm_guard, renyi_guard):
        guard(dict(_BASE_CFG, sign_frame="none"), "fake_run")   # no raise
        guard(dict(_BASE_CFG), "fake_run")                      # missing key defaults to none


def _write_fake_checkpoint(dir_, name, sign_frame):
    jp = os.path.join(dir_, f"{name}.json")
    doc = {"name": name, "config": dict(_BASE_CFG, name=name, sign_frame=sign_frame),
           "observables": {}, "diverged": False}
    with open(jp, "w") as f:
        json.dump(doc, f)
    return jp


def test_load_vstate_refuses_framed_checkpoint(tmp_dir):
    """End-to-end through the real loader: raises before ever looking for the
    sibling .mpack weights (none exist here -- proves the guard runs first)."""
    jp = _write_fake_checkpoint(tmp_dir, "fake_framed", "anaC")
    try:
        load_vstate(jp)
    except ValueError as e:
        assert "fake_framed" in str(e) and "anaC" in str(e)
    else:
        raise AssertionError("load_vstate did not raise on a sign_frame checkpoint")


def test_sweeps_refuse_framed_checkpoint(tmp_dir):
    """fm_sweep/renyi_sweep refuse as soon as they see a framed checkpoint in the
    directory, before building any state (no .mpack needed here either)."""
    _write_fake_checkpoint(tmp_dir, "fake_framed", "anaC")
    for sweep, kw in ((fm_sweep, {}), (renyi_sweep, {})):
        try:
            sweep(tmp_dir, L=2, hx=None, model="fermionic", bc="OBC",
                  verbose=False, **kw)
        except ValueError as e:
            assert "fake_framed" in str(e) and "anaC" in str(e)
        else:
            raise AssertionError(f"{sweep.__name__} did not raise on a framed checkpoint")


def main():
    test_guards_are_the_same_function()
    print("[PASS] fm/renyi guard is one shared function")
    test_direct_guard_raises_on_framed()
    print("[PASS] guard raises on a framed config (fm + renyi)")
    test_direct_guard_passes_on_none()
    print("[PASS] guard passes sign_frame=none / missing key (fm + renyi)")
    with tempfile.TemporaryDirectory() as d:
        test_load_vstate_refuses_framed_checkpoint(d)
    print("[PASS] load_vstate refuses a sign_frame checkpoint end-to-end")
    with tempfile.TemporaryDirectory() as d:
        test_sweeps_refuse_framed_checkpoint(d)
    print("[PASS] fm_sweep/renyi_sweep refuse a sign_frame checkpoint")
    print("All frame-refusal tests passed.")


if __name__ == "__main__":
    main()
