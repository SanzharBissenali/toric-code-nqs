"""Best-of-N anchor reseed for stuck y-polarized chain branches (user, 2026-09-23).

Some y-polarized anchors land in a stuck state: <B_p> ~0.05 instead of ~0.2 (exact, L=2 ED;
leading order 1/(4 h_y)), E ~5 above the good state at the same field. Chains stay single-variable
lines; only the anchor is redone:

  1. trials  -- the chain's ORIGINAL anchor spec, N seeds, each into <L4>/anchor_trials/s<seed>/
                (a subdir, so phase3d_status/the viewer never see them).       [local: emit PLAN_FILEs]
  2. select  -- candidates = the original anchor + the trials; winner = lowest E0 among healthy runs
                (variational principle), gated by anchor_verdict: <B_p> >= 0.6x its leading order and, at
                h_x = 0, E0 within GATE of the strong-field series.            [cluster: reads $PSCRATCH]
     --apply -- park the old branch (every point) into <L4>/redo_reseed_<stamp>/, copy the winner in
                under the anchor's name, emit the ORIGINAL chain spec: sweep.py SKIPS the anchor (its
                JSON exists), loads its final state and re-trains the same links in order.
  3. park    -- mirror step 2's parking on the local pulled copy (pull_phase3d.sh never deletes).

    python analysis/scripts/phase3d_reseed.py trials --emit DIR
    python analysis/scripts/phase3d_reseed.py select --base $PSCRATCH/tc_nqs/phase3d [--apply --emit DIR]
    PLAN_FILE=DIR/<label>_chain.tsv HY=<launch_hy> bash nersc/launch_phase3d.sh
    python analysis/scripts/phase3d_reseed.py park --label L --base <local results/phase3d> --stamp S
"""
import argparse
import copy
import glob
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from phase3d_grid import _bash_line, _ycut_l4_job_spec
from phase3d_tt_diag_probe import hz_sweep_spec

SEEDS = (101, 102, 103)          # the original anchors ran with the default seed 0
GATE = 4.0                       # accept E0 - estimate <= GATE (good family ~+2.5-3, stuck ~+7-8)
H0_BOUND = -172.0                # L=4 OBC exact h=0 energy

# label -> (original chain spec, launcher HY, trial walltime). Specs are regenerated from the same
# functions that submitted them, so FIELD_VALUES / names / recipes are identical to the first run.
CHAINS = {
    "ycut_hx0_hz0.15_dn": (lambda: _ycut_l4_job_spec("ycut_hx0_hz0.15", 0.0, 0.15, "dn"), "y", "01:30:00"),
    "hy1.4_e0_up": (lambda: hz_sweep_spec(1.4, 0.40, 0.30, 0.05, 1.2, "up"), "1.4", "02:00:00"),
}


def strong_field_estimate(hy, hz, L=4):
    """Exact-energy estimate deep in the y-polarized phase at h_x = 0 (L=4 OBC): 2nd-order series around
    the product state along (0, h_y, h_z), minus the ~1.2 residual the series leaves at N=144 (it sits
    0.08-0.16 above exact L=2 OBC ED at h_y 1.2-1.5, scaled by N)."""
    if L != 4:
        raise ValueError("estimate tabulated for L=4 OBC only")
    N, Np = 144, 108
    h = math.hypot(hy, hz); c = hz / h; s = math.sqrt(1 - c * c)
    e = -N * h - Np * c ** 4
    e -= Np * sum(math.comb(4, k) * c ** (2 * (4 - k)) * s ** (2 * k) / (2 * k * h) for k in range(1, 5))
    e -= sum(n / (2 * d * h) for n, d in ((8, 6), (24, 5), (24, 4), (8, 3)))   # A_v, by vertex degree
    return e - 1.2


def bp_leading(hx, hy, hz):
    """Leading-order <B_p> around the product state along h: n_z^4 + s_z^8/(4|h|). Holds at any h_x (B_p = ZZZZ
    fully flips spins in the xy-plane); exact L=2 OBC ED sits 1.1-1.45x above it for h_y 1.4-1.5, h_x 0-1.2."""
    h = math.sqrt(hx * hx + hy * hy + hz * hz); nz = hz / h
    return nz ** 4 + (1 - nz * nz) ** 4 / (4 * h)


def anchor_verdict(hx, hy, hz, E0, B_p):
    """(ok, dE, bp_ratio) for a deep-y-polarized anchor. ok = <B_p> >= 0.6x leading order (stuck ~0.3x, good
    0.7-0.9x) AND, where the energy series is validated (h_x = 0), E0 within GATE of the estimate."""
    ratio = B_p / bp_leading(hx, hy, hz)
    dE = E0 - strong_field_estimate(hy, hz) if hx == 0 else None
    return ratio >= 0.6 and (dE is None or dE <= GATE), dE, ratio


def trial_spec(spec, seed, walltime, tag="s", anchor_ov=None, values=None, init_from=None):
    """One anchor trial into <L4>/anchor_trials/<tag><seed>/. Default: the chain's own anchor spec, new seed.
    anchor_ov: replace ANCHOR_OVERRIDES (recipe test). values: a warm chain along the SAME line ending at the
    anchor value (e.g. start deeper in the y-polarized phase). init_from: warm-start the anchor from a checkpoint
    base path (e.g. continue an earlier trial)."""
    s = copy.deepcopy(spec)
    a = s["h_list"][0]
    vals = list(values) if values else [a]
    if abs(vals[-1] - a) > 1e-9:
        raise SystemExit(f"--values must end at the anchor {a}, got {vals}")
    s["h_list"] = vals
    s["env"].update({s["env"]["SWEEP"].upper(): str(vals[0])})
    s["env"].update(FIELD_VALUES=" ".join(str(v) for v in vals), CHUNK_POINTS=str(len(vals)),
                    EXTRA_ARGS=f"{s['env']['EXTRA_ARGS']} --seed {seed}",
                    WANDB_GROUP=f"{s['jobname']}_{tag}{seed}")
    if anchor_ov:
        s["env"]["ANCHOR_OVERRIDES"] = anchor_ov
    if init_from:
        s["env"]["INIT_FROM"] = init_from
    s["jobname"] = f"{s['jobname']}_{tag}{seed}"
    s["role"] = "anchor_trial"
    s["out_dir_rel"] = f"{s['out_dir_rel']}/anchor_trials/{tag}{seed}"
    s["walltime"] = walltime
    s["dependency"] = None
    return s


def _branch(spec):
    return spec["jobname"].rsplit("_", 1)[-1]                   # "up" | "dn"


def _branch_glob(spec, branch=None):
    return f"*_{branch or _branch(spec)}.*"                     # e.g. *_dn.* -- every point of the branch


def _is_final(f):
    return f.endswith(".json") and not f.endswith((".curve.json", ".snapshots.json")) and "finaleval" not in f


def _read(path):
    with open(path) as fh:
        d = json.load(fh)
    o, cfg = d["observables"], d["config"]
    return {"path": path, "name": d["name"], "E0": o.get("E0"), "B_p": o.get("B_p_mean"),
            "sy": o.get("sy_mean"), "Vscore": o.get("Vscore"), "diverged": d.get("diverged"),
            "seed": cfg.get("seed"), "hx": cfg.get("hx"), "hy": cfg.get("hy"), "hz": cfg.get("hz")}


def candidates(base, spec):
    """The original anchor, every anchor trial, and the OPPOSITE branch's point at the anchor field (a round trip
    along the same line: e.g. the z-polarized dn chain continued down into the y-polarized phase) -- all at the
    anchor value only (trial chains also leave their intermediate points)."""
    l4 = os.path.join(base, spec["out_dir_rel"])
    sweep, a = spec["env"]["SWEEP"], spec["h_list"][0]
    other = "up" if _branch(spec) == "dn" else "dn"
    files = [(f, "original") for f in glob.glob(os.path.join(l4, _branch_glob(spec)))]
    files += [(f, "opposite") for f in glob.glob(os.path.join(l4, _branch_glob(spec, other)))]
    files += [(f, os.path.basename(os.path.dirname(f))) for f in sorted(glob.glob(os.path.join(l4, "anchor_trials", "*", "*.json")))]
    runs = []
    for f, src in files:
        if _is_final(f):
            r = _read(f)
            if abs(r[sweep] - a) < 1e-9:
                r["src"] = src
                runs.append(r)
    for r in runs:
        r["healthy"] = bool(r["E0"] is not None and not r["diverged"] and r["E0"] < H0_BOUND)
        r["pass"], r["dE"], r["bp_ratio"] = (anchor_verdict(r["hx"], r["hy"], r["hz"], r["E0"], r["B_p"])
                                             if r["healthy"] else (False, None, None))
    return l4, runs


def park(l4, spec, stamp):
    dest = os.path.join(l4, f"redo_reseed_{stamp}")
    files = [f for f in glob.glob(os.path.join(l4, _branch_glob(spec))) if os.path.isfile(f)]
    os.makedirs(dest, exist_ok=True)
    for f in files:
        shutil.move(f, dest)
    return dest, len(files)


def main(argv):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trials"); t.add_argument("--emit", required=True)
    t.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    t.add_argument("--only", nargs="+", choices=sorted(CHAINS))
    t.add_argument("--tag", default="s", help="trial dir/jobname prefix, e.g. L for a long-recipe test")
    t.add_argument("--anchor-ov", help="ANCHOR_OVERRIDES JSON replacing the chain's recipe")
    t.add_argument("--values", type=float, nargs="+", help="warm chain along the same line ending at the anchor")
    t.add_argument("--init-from", help="checkpoint base name to warm-start the anchor from, RELATIVE to the trial's "
                                       "own dir (the launcher prefixes OUT_DIR), e.g. ../s103/<name>")
    t.add_argument("--walltime")
    s = sub.add_parser("select"); s.add_argument("--base", required=True)
    s.add_argument("--apply", action="store_true"); s.add_argument("--emit")
    s.add_argument("--only", nargs="+", choices=sorted(CHAINS))
    k = sub.add_parser("park"); k.add_argument("--label", required=True, choices=sorted(CHAINS))
    k.add_argument("--base", required=True); k.add_argument("--stamp", required=True)
    a = p.parse_args(argv)

    if a.cmd == "trials":
        os.makedirs(a.emit, exist_ok=True)
        for label, (spec_fn, launch_hy, wall) in CHAINS.items():
            if a.only and label not in a.only:
                continue
            fp = os.path.join(a.emit, f"{label}_trials_{a.tag}.tsv")
            with open(fp, "w") as fh:
                for seed in a.seeds:
                    fh.write(_bash_line(trial_spec(spec_fn(), seed, a.walltime or wall, a.tag, a.anchor_ov,
                                                   a.values, a.init_from)) + "\n")
            print(f"wrote {fp}  ({len(a.seeds)} anchor trials, launch with HY={launch_hy})")
        return

    if a.cmd == "park":
        spec = CHAINS[a.label][0]()
        dest, n = park(os.path.join(a.base, spec["out_dir_rel"]), spec, a.stamp)
        print(f"[park] {a.label}: {n} files -> {dest}")
        return

    stamp = time.strftime("%Y%m%d%H%M")
    for label in a.only or sorted(CHAINS):
        spec_fn, launch_hy, _ = CHAINS[label]
        spec = spec_fn()
        l4, runs = candidates(a.base, spec)
        print(f"== {label}  anchor {spec['env']['SWEEP']}={spec['h_list'][0]}  "
              f"estimate {strong_field_estimate(runs[0]['hy'], runs[0]['hz']) if runs else float('nan'):.2f}")
        for r in sorted(runs, key=lambda r: (r["E0"] is None, r["E0"])):
            dE = f"{r['dE']:+6.2f}" if r["dE"] is not None else "   n/a"
            print(f"   {r['src']:>9} seed {r['seed']:>4}  E0 {r['E0']:9.3f}  dE {dE}  B_p {r['B_p']:.3f} ({r['bp_ratio'] or 0:.2f}x)  "
                  f"sy {r['sy']:.3f}  V {r['Vscore']:.3f}  {'ok' if r['healthy'] else 'UNHEALTHY'}")
        ok = [r for r in runs if r["healthy"]]
        if not ok:
            print("   -> no healthy candidate"); continue
        win = min(ok, key=lambda r: r["E0"])
        passed = win["pass"]
        is_orig = win["src"] == "original"
        print(f"   -> winner seed {win['seed']} (gate {'PASS' if passed else 'FAIL'}: <B_p> >= 0.6x leading, dE <= {GATE})"
              f"{' = the original anchor, nothing to reseed' if is_orig else ''}")
        n_trials = sum(1 for r in runs if r["src"] != "original")
        if not (a.apply and passed and not is_orig):
            continue
        if n_trials < 1:
            print("   -> trials not landed yet, not applying"); continue
        dest, n = park(l4, spec, stamp)
        src_dir = os.path.dirname(win["path"])              # an "opposite" winner sits in l4 (park moved only this branch)
        name = win["name"]
        if win["src"] == "opposite":                         # same field point: take this branch's name suffix
            name = name.rsplit("_", 1)[0] + "_" + _branch(spec)
        for f in glob.glob(os.path.join(src_dir, f"{win['name']}.*")):
            if ".step" in f or ".ckpt." in f:
                continue
            shutil.copy2(f, os.path.join(l4, os.path.basename(f).replace(win["name"], name, 1)))
        new_json = os.path.join(l4, f"{name}.json")
        with open(new_json) as fh:
            d = json.load(fh)
        d["name"] = name
        d["weights"] = os.path.join(l4, f"{name}.mpack")
        d["reseed"] = {"from": win["path"], "src": win["src"], "seed": win["seed"], "dE": win["dE"], "parked": dest}
        with open(new_json, "w") as fh:
            json.dump(d, fh)
        print(f"   -> parked {n} files -> {dest}; winner copied in")
        if a.emit:
            os.makedirs(a.emit, exist_ok=True)
            fp = os.path.join(a.emit, f"{label}_chain.tsv")
            with open(fp, "w") as fh:
                fh.write(_bash_line(spec) + "\n")
            print(f"   -> wrote {fp}  (launch with HY={launch_hy}; local mirror: park --stamp {stamp})")


if __name__ == "__main__":
    main(sys.argv[1:])
