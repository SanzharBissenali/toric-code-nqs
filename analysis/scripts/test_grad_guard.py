"""Unit test for the divergence detector `builders.is_bad_step` against the REAL
L=6 spreads that blew up (hz=0.24, hz=0.36).

Runs two ways:
  * locally, no cluster data -- against the spread tail reconstructed from the
    `check_convergence.py --trace` output, asserting the detector fires exactly at
    the detonation step (110 / 243) and NOT on the sane steps just before it;
  * on the cluster, if the quarantined curve.json files are present -- against the
    FULL 300-step spread series, which additionally proves NO false rollback fires
    during the volatile early-convergence phase.

    python analysis/scripts/test_grad_guard.py
    python analysis/scripts/test_grad_guard.py --dir $PSCRATCH/tc_nqs/phase_hx0.2/L6/diverged

The replay below mirrors run_loop's guard bookkeeping exactly: only SANE spreads
enter the rolling median window, so a spike never poisons its own baseline.
"""
import argparse
import glob
import json
import os

import numpy as np
import jax.tree_util as jtu

from tc3d.builders import is_bad_step

SPIKE_FACTOR, GUARD_WARMUP, BASELINE_WINDOW = 10.0, 5, 20


def _trees_equal(a, b):
    la, lb = jtu.tree_leaves(a), jtu.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y))
                                      for x, y in zip(la, lb))


def snapshot_restore_roundtrip():
    """The mechanic the fixed guard relies on: snapshot (params, sampler_state),
    blow the params up, restore BOTH, and confirm the warm state round-trips
    exactly and the energy recovers (vs. the cold reseed that death-spiraled)."""
    import jax.numpy as jnp
    import netket as nk

    hi = nk.hilbert.Spin(0.5, 6)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=8)
    vs = nk.vqs.MCState(sa, nk.models.RBM(alpha=1), n_samples=256, seed=0)
    H = nk.operator.Ising(hi, graph=nk.graph.Chain(6, pbc=True), h=1.0)

    _ = vs.samples                                     # warm the chains
    E0 = float(np.real(vs.expect(H).mean))
    copy = lambda t: jtu.tree_map(lambda x: jnp.array(x, copy=True), t)
    snap = (copy(vs.parameters), copy(vs.sampler_state))   # <- guard's _snapshot()

    vs.parameters = jtu.tree_map(lambda x: x * 1e6, vs.parameters)   # corrupt
    vs.reset()
    E_bad = float(np.real(vs.expect(H).mean))

    p, s = snap                                        # <- guard's _restore()
    vs.parameters = p
    vs.sampler_state = s
    # check the round-trip BEFORE resampling (vs.reset()+expect advances sigma/rng)
    params_ok = _trees_equal(vs.parameters, snap[0])
    sampler_ok = _trees_equal(vs.sampler_state, snap[1])
    vs.reset()
    E_rec = float(np.real(vs.expect(H).mean))
    energy_ok = np.isfinite(E_rec) and abs(E_rec - E0) < abs(E_bad - E0)
    ok = params_ok and sampler_ok and energy_ok
    print("snapshot/restore round-trip:")
    print(f"  params round-trip: {params_ok}   sampler_state round-trip: {sampler_ok}")
    print(f"  E0={E0:.4f}  E_corrupted={E_bad:.4g}  E_restored={E_rec:.4f}  "
          f"(recovered={energy_ok})  {'PASS' if ok else 'FAIL'}")
    return ok


def first_bad(spreads):
    """Index of the first step the guard would flag (or None), replaying the
    sane-only rolling window from run_loop."""
    hist = []
    for i, s in enumerate(spreads):
        if is_bad_step(float(s), hist, SPIKE_FACTOR, GUARD_WARMUP):
            return i
        hist.append(float(s))
        if len(hist) > BASELINE_WINDOW:
            hist.pop(0)
    return None


# Spread tails copied from the check_convergence --trace output (step -> spread).
RECON = {
    "hz0.24": {"step0": 103, "expect": 110,
               "spread": [1.208, 1.270, 1.228, 1.172, 1.336, 1.435, 1.187,
                          137342.442, np.inf, np.inf]},
    "hz0.36": {"step0": 237, "expect": 243,   # caught at the x20 first-warning, before the overflow
               "spread": [2.160, 2.056, 2.010, 1.916, 1.955, 2.168, 41.608,
                          3.38e42, np.inf, np.inf]},
}


def check(label, spreads, step0, expect):
    idx = first_bad(spreads)
    got = None if idx is None else step0 + idx
    ok = got == expect
    print(f"  {label}: first flagged step = {got}  (expect {expect})  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None,
                   help="dir with the quarantined *hz0.24*/*hz0.36* curve.json "
                        "(full-series test); omit for the reconstructed-tail test")
    a = p.parse_args(argv)
    passed = True

    print(f"spike_factor={SPIKE_FACTOR} guard_warmup={GUARD_WARMUP} "
          f"baseline_window={BASELINE_WINDOW}")

    # 1. clean series (jitter up to ~3x) must never fire
    clean = [2.0, 1.5, 3.0, 1.0, 2.5, 1.8, 2.2, 1.2, 2.8, 1.6] * 5
    ok = first_bad(clean) is None
    print(f"  clean/no-divergence: first flagged = {first_bad(clean)}  "
          f"(expect None)  {'PASS' if ok else 'FAIL'}"); passed &= ok

    # 2. reconstructed tails from the --trace
    print("reconstructed tails (from --trace):")
    for label, r in RECON.items():
        passed &= check(label, r["spread"], r["step0"], r["expect"])

    # 2b. warm snapshot/restore round-trip (the recovery mechanic)
    passed &= snapshot_restore_roundtrip()

    # 3. full real series, if the quarantined curve.json files are reachable
    if a.dir:
        print(f"full real series (from {a.dir}):")
        for label, expect in (("hz0.24", 110), ("hz0.36", 243)):
            hits = glob.glob(os.path.join(a.dir, f"*_{label}.curve.json"))
            if not hits:
                print(f"  {label}: no curve.json found -- skip")
                continue
            with open(hits[0]) as f:
                curve = json.load(f)["curve"]
            spreads = [s if s is not None else np.inf for s in curve["energy_spread"]]
            steps = curve["step"]
            idx = first_bad(spreads)
            got = None if idx is None else steps[idx]
            ok = got == expect
            print(f"  {label}: first flagged step = {got}  (expect {expect}, "
                  f"{len(steps)} steps)  {'PASS' if ok else 'FAIL'}"); passed &= ok

    print("ALL PASS" if passed else "SOME FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
