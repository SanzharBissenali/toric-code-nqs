"""
analysis/scripts/compare_equiv.py
────────────────────────────────────────────────────────────────────────────
Compare same-seed training runs (tc3d.train result JSONs) of the speed-lever
variants against a baseline: per-step energy pull z = (E_v - E_b)/sqrt(err_v^2
+ err_b^2), late-window energy, Vscore (= N Var/E^2 from the spread), final
observables, and the median step time -> one markdown table.

    python analysis/scripts/compare_equiv.py --dir results/speed_equiv \
        --base equiv_L4_conv_float64_cholesky_n8192 equiv_L4_dense_float32_cholesky_n8192 ...
"""
import argparse
import json
import os
import sys

import numpy as np


def load(d, name):
    with open(os.path.join(d, f"{name}.json")) as f:
        return json.load(f)


def summary(r, window):
    c = r["curve"]
    E = np.asarray(c["energy"], float)
    err = np.asarray(c["energy_err"], float)
    sp = np.asarray(c["energy_spread"], float)
    N = r["config"]["n_params"] and r.get("geo_N") or None
    steps = np.asarray(c["step"])
    w = slice(max(0, len(E) - window), len(E))
    tim = [t["total"] for t in c.get("timing", []) if t["step"] > steps[0]]
    obs = r.get("observables", {})
    return dict(E=E, err=err, spread=sp, steps=steps,
                E_late=float(E[w].mean()), E_late_err=float(np.sqrt((err[w] ** 2).sum()) / max(1, len(E[w]))),
                spread_late=float(sp[w].mean()),
                E0=obs.get("E0"), Vscore=obs.get("Vscore"),
                t_step=float(np.median(tim)) if tim else float("nan"),
                runtime=r.get("runtime_s"), n_params=r["config"].get("n_params"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("names", nargs="+")
    p.add_argument("--window", type=int, default=10, help="late-window length (steps)")
    a = p.parse_args()

    rb = load(a.dir, a.base)
    N = rb["config"].get("N") or 3 * rb["config"]["L"] ** 2 * (rb["config"]["L"] - 1)
    b = summary(rb, a.window)
    rows = []
    hdr = ("| run | steps | median s/step | speedup | mean z | rms z | max abs z | "
           f"E (last {a.window}) | Vscore (last {a.window}) | final E0 | final Vscore |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))

    def vscore(s):
        return N * s["spread_late"] ** 2 / s["E_late"] ** 2

    def row(name, s, z):
        zs = ("—", "—", "—") if z is None else (f"{z.mean():+.2f}", f"{np.sqrt((z**2).mean()):.2f}",
                                                 f"{np.abs(z).max():.2f}")
        sp = "1.00x" if z is None else f"{b['t_step'] / s['t_step']:.2f}x"
        print(f"| {name} | {len(s['E'])} | {s['t_step']:.2f} | {sp} | {zs[0]} | {zs[1]} | {zs[2]} | "
              f"{s['E_late']:.4f} ± {s['E_late_err']:.4f} | {vscore(s):.2e} | "
              f"{s['E0'] if s['E0'] is None else f'{s[chr(69)+chr(48)]:.4f}'} | "
              f"{s['Vscore'] if s['Vscore'] is None else f'{s[chr(86)+chr(115)+chr(99)+chr(111)+chr(114)+chr(101)]:.2e}'} |")

    row(a.base + " (base)", b, None)
    for name in a.names:
        s = summary(load(a.dir, name), a.window)
        n = min(len(s["E"]), len(b["E"]))
        z = (s["E"][:n] - b["E"][:n]) / np.sqrt(s["err"][:n] ** 2 + b["err"][:n] ** 2)
        row(name, s, z)
        dmax = float(np.max(np.abs(s["E"][:n] - b["E"][:n])))
        print(f"|   ↳ max abs dE over {n} steps = {dmax:.3e} (exact levers should be ~1e-8 or below) |")


if __name__ == "__main__" and "--final" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# Converged comparison: final pooled observables (E0 ± E_err, Vscore) of full-length
# runs against a reference run (e.g. the production hy_cuts_L4 point), by file path.
#
#   python analysis/scripts/compare_equiv.py --final REF.json RUN1.json RUN2.json ...
def final_table(ref_path, paths):
    def fin(path):
        r = json.load(open(path)); o = r["observables"]; c = r["config"]; cur = r["curve"]
        tim = [t["total"] for t in cur.get("timing", []) if t["step"] > cur["step"][0]]
        return dict(name=os.path.basename(path)[:-5], E0=o["E0"], err=o.get("E_err", float("nan")),
                    V=o.get("Vscore", float("nan")), rounds=o.get("final_eval_rounds", c.get("final_eval_rounds")),
                    steps=len(cur["energy"]), t=float(np.median(tim)) if tim else float("nan"),
                    seed=c.get("seed"), dtype=c.get("compute_dtype") or "float64", impl=c.get("inv_impl") or "conv",
                    solver=c.get("qgt_solver") or "cg", ns=c.get("n_samples"))
    ref = fin(ref_path)
    print(f"reference: {ref['name']}  E0 = {ref['E0']:.4f} ± {ref['err']:.4f}  Vscore {ref['V']:.3e}  "
          f"({ref['steps']} steps, {ref['rounds']} final rounds)")
    print("| run | impl | compute | solver | n_s | seed | steps | s/step | E0 ± err | ΔE0 vs ref | z | Vscore |")
    print("|" + "---|" * 12)
    for path in paths:
        s = fin(path)
        d = s["E0"] - ref["E0"]; z = d / np.sqrt(s["err"] ** 2 + ref["err"] ** 2)
        print(f"| {s['name']} | {s['impl']} | {s['dtype']} | {s['solver']} | {s['ns']} | {s['seed']} | {s['steps']} | "
              f"{s['t']:.2f} | {s['E0']:.4f} ± {s['err']:.4f} | {d:+.4f} | {z:+.2f} | {s['V']:.3e} |")


if __name__ == "__main__" and "--final" in sys.argv:
    i = sys.argv.index("--final")
    final_table(sys.argv[i + 1], sys.argv[i + 2:])
    sys.exit(0)
