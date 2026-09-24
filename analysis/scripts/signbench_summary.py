"""Learned-vs-gated sign-head benchmark: per-cell table + the two 3x3 heatmaps.

Joins, per (h_x, h_z, arm, seed):
  run          {dir}/signbench_{box}_hx{hx}_hz{hz}_{arm}_s{seed}.json (tc3d.train)
  exact eval   the matching .snapshots.json (eval_snapshots.py --exact): the LAST
               snapshot's full-sum E and fidelity F = |<psi_ED|psi>|^2
  referee      signbench_prep_{box}.json (signbench_prep.py): E0_ED + ceilings
and reports rel = (E - E0)/|E0|, 1 - F and the arm's ceiling (T: 1 - F_s(pt2)
per the spec, and the representability ceiling T_gate the 2D side adopted too;
M / M-pre: 0). Rows without an ED-matched fidelity carry no 1 - F (never 0). Verdict per cell and arm
pair: an arm WINS if its 1 - F is >= 3x lower than the other's on every seed
present (spec section 6; one seed = provisional).

    python analysis/scripts/signbench_summary.py --dir results/fermionic_signbench \\
        --box L2x2x3_OBC
"""
import argparse
import itertools
import json
import os
import re

import numpy as np

ARMS = ("M", "Mp", "T", "Tp", "H")    # Tp (a = e^c > 0) and H (a = 0) are controls, not exported
ARM_LABEL = {"M": "M (MLP, cold)", "Mp": "M-pre (MLP, ED-pretrained)",
             "T": "T (two-branch, pt2)", "Tp": "T+ (a = e^c > 0, control)",
             "H": "H (head-only pt2, control)"}
FLOOR = 1e-12


def load_rows(root, box):
    with open(os.path.join(root, f"signbench_prep_{box}.json")) as f:
        prep = json.load(f)
    ref = {(p["hx"], p["hz"]): p for p in prep["points"]}
    pat = re.compile(rf"^signbench_{re.escape(box)}_hx([\d.]+)_hz([\d.]+)_(\w+?)_s(\d+)\.json$")
    rows = []
    for fn in sorted(os.listdir(root)):
        m = pat.match(fn)
        if not m:
            continue
        hx, hz, arm, seed = float(m[1]), float(m[2]), m[3], int(m[4])
        p = ref.get((hx, hz))
        with open(os.path.join(root, fn)) as f:
            run = json.load(f)
        row = {"box": box, "hx": hx, "hz": hz, "arm": arm, "seed": seed,
               "diverged": bool(run.get("diverged")), "n_params": run.get("n_params"),
               "E0_ED": p["E0"] if p else None,
               "ceiling": (p["ceilings"]["T_head"] if arm in ("T", "Tp", "H") else 0.0) if p else None,
               "ceiling_T_gate": (p["ceilings"]["T_gate_plus" if arm == "Tp" else "T_gate"]
                                  if p and arm in ("T", "Tp") else None),
               "ceiling_T_gate_pm": ([p["ceilings"].get("T_gate_plus"),
                                      p["ceilings"].get("T_gate_minus")]
                                     if p and arm == "T" else None),
               "ceiling_plus": p["ceilings"]["plus"] if p else None,
               "pretrain_1mF": (p.get("pretrain", {}).get("one_minus_F_s")
                                if p and arm == "Mp" else None)}
        snap = os.path.join(root, fn[:-len(".json")] + ".snapshots.json")
        if os.path.exists(snap):
            with open(snap) as f:
                series = [s for s in json.load(f)["series"] if "exact" in s]
            if series:
                last = series[-1]
                row["step"] = last["step"]
                if arm in ("T", "Tp"):
                    row["mix"] = final_mix(root, fn[:-len(".json")], last["step"])
                row["E"] = last["exact"]["E0"]
                fid = last["exact"].get("fidelity")
                if fid is not None and np.isfinite(fid):     # no ED match -> no score
                    row["one_minus_F"] = max(0.0, 1.0 - fid)
                if p:
                    row["rel"] = (row["E"] - p["E0"]) / abs(p["E0"])
                row["curve"] = [{"step": s["step"], "E": s["exact"]["E0"],
                                 "one_minus_F": (max(0.0, 1.0 - s["exact"]["fidelity"])
                                                 if np.isfinite(s["exact"].get("fidelity", np.nan))
                                                 else None)}
                                for s in series]
        rows.append(row)
    return prep, rows


def final_mix(root, name, step):
    """T arm: the signed mix a at the evaluated snapshot (None if weights absent)."""
    path = os.path.join(root, f"{name}.step{step}.mpack")
    if not os.path.exists(path):
        return None
    from flax import serialization
    with open(path, "rb") as f:
        v = serialization.msgpack_restore(f.read())
    for key in ("variables", "params"):
        v = v.get(key, v) if isinstance(v, dict) else v
    if isinstance(v, dict) and "mix" in v:
        return float(v["mix"])
    if isinstance(v, dict) and "log_mix" in v:                 # Tp: a = e^c
        return float(np.exp(v["log_mix"]))
    return None


def verdicts(rows):
    """{(hx, hz): {"A>B": bool, ...}} over every arm pair (3x rule, all seeds)."""
    out = {}
    cells = sorted({(r["hx"], r["hz"]) for r in rows})
    for c in cells:
        by = {}
        for r in rows:
            if (r["hx"], r["hz"]) == c and "one_minus_F" in r:
                by.setdefault(r["arm"], {})[r["seed"]] = max(r["one_minus_F"], FLOOR)
        v = {}
        for a, b in itertools.permutations(sorted(by), 2):
            seeds = sorted(set(by[a]) & set(by[b]))
            if seeds:
                v[f"{a} beats {b}"] = all(3.0 * by[a][s] <= by[b][s] for s in seeds)
        out[f"{c[0]},{c[1]}"] = {"n_seeds": len({s for d in by.values() for s in d}),
                                 **v}
    return out


TOKEN_2D = {"M": "cnnqM", "Mp": "cnnqMp", "T": "cnnqT"}   # 2D-TC arm tokens


def export_2d(rows, prep, root, path, tail=20):
    """The 2D-TC notebook's 3D-panel contract (analysis/07_signbench_heatmaps.ipynb:
    results/diagnostics/signbench_3d.json). rel_err is from the EXACT full-sum
    energy of the last snapshot (|E - E0|/|E0|); E_tail / rel_err_tail are the
    2D-style median of the last `tail` MC training energies. ceiling = T_gate
    for T (the representability ceiling both sides adopted), 0 for M / M-pre."""
    size = "x".join(map(str, prep["geometry"]["Lxyz"]))
    recs = []
    for r in rows:
        if r["arm"] not in TOKEN_2D:               # the H control stays out of the 3-arm panel
            continue
        with open(os.path.join(root, f"signbench_{r['box']}_hx{r['hx']}_hz{r['hz']}_"
                                     f"{r['arm']}_s{r['seed']}.json")) as f:
            curve = json.load(f).get("curve", {})
        e = np.asarray(curve.get("energy", [])[-tail:], dtype=float)
        sp = np.asarray(curve.get("energy_spread", [])[-tail:], dtype=float)
        N = prep["geometry"]["N"]
        vs = N * sp ** 2 / e ** 2 if e.size else np.array([])
        e0 = r["E0_ED"]
        E_tail = float(np.median(e)) if e.size else None
        recs.append({
            "size": size, "hx": r["hx"], "hz": r["hz"], "arm": TOKEN_2D.get(r["arm"], r["arm"]),
            "seed": r["seed"], "missing": "one_minus_F" not in r,
            "E_tail": E_tail, "E_tail_std": float(np.std(e)) if e.size else None,
            "Vscore_tail": float(np.median(vs)) if vs.size else None,
            "nan_tail": bool(e.size and not np.isfinite(e).all()),
            "E0": e0, "rel_err": abs(r["rel"]) if "rel" in r else None,
            "rel_err_tail": abs(E_tail - e0) / abs(e0) if (E_tail is not None and e0) else None,
            "F": 1.0 - r["one_minus_F"] if "one_minus_F" in r else None, "F_trunk": None,
            "one_minus_F": r.get("one_minus_F"),
            "ceiling": r["ceiling_T_gate"] if r["arm"] == "T" else r["ceiling"],
            "ceiling_kind": ("T_gate (signed-a representability)" if r["arm"] == "T"
                             else "0: (eps, x) injective"),
            "mix": r.get("mix"), "warm_1mFs": r.get("pretrain_1mF"),
            "ceiling_T_head": r["ceiling"] if r["arm"] == "T" else None,
            "T_gate_plus_minus": r.get("ceiling_T_gate_pm"),
            "log_mix": r.get("mix"), "n_params": r["n_params"], "diverged": r["diverged"]})
    arms = [TOKEN_2D[a] for a in ARMS if a in TOKEN_2D]
    out = {"Lx": None, "Ly": None, "size": size,
           "points": sorted([p["hx"], p["hz"]] for p in prep["points"]),   # full planned grid
           "arms": arms,
           "tail": tail, "records": recs, "verdicts": verdicts_2d(recs, arms),
           "note": "3D fermionic TC 2x2x3 OBC (toric-code-nqs-fsign signbench); dense SR + "
                   "cosine dt 0.02->0.002, 300 steps, 8192 samples; T = a A_triv + s_pt2 A_top "
                   "(signed a); rel_err from exact full-sum energy"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[export_2d] {path}")


def verdicts_2d(recs, arms):
    """2D-TC signbench_summary.verdicts shape: pairwise, needs seeds 0 AND 1."""
    by = {}
    for r in recs:
        if r["one_minus_F"] is not None:
            by.setdefault((r["hx"], r["hz"], r["arm"]), {})[r["seed"]] = r["one_minus_F"]
    out = []
    for hx, hz in sorted({(h, z) for (h, z, _a) in by}):
        here = [a for a in arms if {0, 1} <= by.get((hx, hz, a), {}).keys()]
        for i, a in enumerate(here):
            for b in here[i + 1:]:
                fa, fb = by[(hx, hz, a)], by[(hx, hz, b)]
                a_w = all(3 * fa[s] <= fb[s] for s in (0, 1))
                b_w = all(3 * fb[s] <= fa[s] for s in (0, 1))
                out.append({"size": recs[0]["size"], "hx": hx, "hz": hz, "arm_a": a,
                            "arm_b": b, "one_minus_F_a_seed0": fa[0],
                            "one_minus_F_a_seed1": fa[1], "one_minus_F_b_seed0": fb[0],
                            "one_minus_F_b_seed1": fb[1],
                            "winner": a if a_w else (b if b_w else None),
                            "rule": "1-F >= 3x lower on both seeds"})
    return out


def heatmaps(rows, prep, path, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    hxs = sorted({r["hx"] for r in rows})
    hzs = sorted({r["hz"] for r in rows})
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    metrics = [("rel", "relative energy error $(E-E_0)/|E_0|$"),
               ("one_minus_F", r"infidelity $1-|\langle\psi_{ED}|\psi\rangle|^2$")]
    fig, axes = plt.subplots(2, len(arms), figsize=(3.9 * len(arms), 8.2), squeeze=False,
                             layout="constrained")
    for i, (key, title) in enumerate(metrics):
        vals = [max(abs(r[key]), FLOOR) for r in rows if key in r and r["seed"] == seed]
        norm = LogNorm(vmin=min(vals), vmax=max(vals)) if vals else None
        for j, arm in enumerate(arms):
            ax = axes[i, j]
            Z = np.full((len(hzs), len(hxs)), np.nan)
            for r in rows:
                if r["arm"] == arm and r["seed"] == seed and key in r:
                    Z[hzs.index(r["hz"]), hxs.index(r["hx"])] = max(abs(r[key]), FLOOR)
            im = ax.imshow(Z, origin="lower", cmap="magma", norm=norm, aspect="auto")
            for (a, b), z in np.ndenumerate(Z):
                if np.isfinite(z):
                    r = next(r for r in rows if r["arm"] == arm and r["seed"] == seed
                             and r["hz"] == hzs[a] and r["hx"] == hxs[b])
                    txt = f"{z:.1e}" + ("\n(div)" if r["diverged"] else "")
                    if key == "one_minus_F" and arm == "T":
                        txt += f"\ngate {r['ceiling_T_gate']:.0e}\nhead {r['ceiling']:.0e}"
                    ax.text(b, a, txt, ha="center", va="center", fontsize=7.5,
                            color="w" if norm and z < np.sqrt(norm.vmin * norm.vmax) else "k")
            ax.set_xticks(range(len(hxs)), [f"{h:g}" for h in hxs])
            ax.set_yticks(range(len(hzs)), [f"{h:g}" for h in hzs])
            ax.set_xlabel(r"$h_x$")
            ax.set_ylabel(r"$h_z$")
            ax.set_title(f"{ARM_LABEL.get(arm, arm)}", fontsize=10)
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.85, label=title)
    g = prep["geometry"]
    fig.suptitle(f"fermionic TC {'x'.join(map(str, g['Lxyz']))} {g['bc']} (N={g['N']}), "
                 f"seed {seed}: learned vs gated sign heads", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[fig] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--box", default="L2x2x3_OBC")
    ap.add_argument("--out", default=None, help="default {dir}/signbench_summary_{box}.json")
    ap.add_argument("--fig", default=None, help="default {dir}/signbench_heatmaps_{box}.png")
    ap.add_argument("--no_fig", action="store_true")
    ap.add_argument("--export_2d", default=None, metavar="PATH",
                    help="also write the 2D-TC notebook's 3D-panel JSON (signbench_3d.json)")
    args = ap.parse_args()

    prep, rows = load_rows(args.dir, args.box)
    print(f"{'hx':>4} {'hz':>4} {'arm':>3} {'s':>2} {'step':>5} {'rel':>10} {'1-F':>10} "
          f"{'ceil':>9} {'T_gate':>9} {'mix':>7} div")
    for r in sorted(rows, key=lambda r: (r["hx"], r["hz"], r["arm"], r["seed"])):
        f = lambda k: f"{r[k]:10.3e}" if r.get(k) is not None else f"{'-':>10}"
        print(f"{r['hx']:4g} {r['hz']:4g} {r['arm']:>3} {r['seed']:2d} {r.get('step', '-'):>5} "
              f"{f('rel')} {f('one_minus_F')} {f('ceiling')[1:]} {f('ceiling_T_gate')[1:]} "
              f"{r['mix'] if r.get('mix') is not None else float('nan'):7.3f} "
              f"{'DIV' if r['diverged'] else ''}")
    out = {"box": args.box, "geometry": prep["geometry"], "head": prep["head"],
           "rows": rows, "verdicts": verdicts(rows)}
    path = args.out or os.path.join(args.dir, f"signbench_summary_{args.box}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {path}")
    if args.export_2d and rows:
        export_2d(rows, prep, args.dir, args.export_2d)
    if not args.no_fig and rows:
        heatmaps(rows, prep, args.fig or os.path.join(args.dir,
                                                      f"signbench_heatmaps_{args.box}.png"))


if __name__ == "__main__":
    main()
