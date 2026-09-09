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


if __name__ == "__main__":
    main()
