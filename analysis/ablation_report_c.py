"""Phase-B ablation-C report: multi-L (L=4,5,6) + warmup/relative-floor schedule.

Same methodology as ablation_report.py, extended to:
  - L=5, L=6 (not just L=4)
  - the new schedule: --warmup_frac 0.05, --lr_min = 20% of dt (vs the old
    ablation-A schedule: no warmup, --lr_min = 10% of dt)
For the two L=4 points, also prints the OLD-schedule Run-A series (already
collected) alongside, for a direct old-vs-new-schedule read at matched L.

Inputs:
  results/phaseB/{cut}/L{L}/*.json                     baseline: winner arch, campaign n_iter
  results/phaseB_ablationA/{cut}/L4/*.snapshots.json    old schedule, L4 only (750 steps)
  results/phaseB_ablationC/{cut}/L{L}/*.snapshots.json  new schedule, L4/5/6 (500 steps)
  results/qmc_hx*_hz*/paratoric_L{L}_*.json             QMC reference
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (cut, L, hx, hz, qmc_basis, nqs_stab_key, qmc_stab_key, nqs_mag_key, qmc_mag_key,
#  nqs_op_key, qmc_op_key, label, has_old_run_a)
POINTS = [
    ("up", 4, 0.2, 0.26, "z", "A_v_mean", "star_x", "sz_mean", "sigma_z",
     "O_FM_paratoric", "fredenhagen_marcu", "Z-string", True),
    ("up", 5, 0.2, 0.24, "z", "A_v_mean", "star_x", "sz_mean", "sigma_z",
     "O_FM_paratoric", "fredenhagen_marcu", "Z-string", False),
    ("up", 6, 0.2, 0.22, "z", "A_v_mean", "star_x", "sz_mean", "sigma_z",
     "O_FM_paratoric", "fredenhagen_marcu", "Z-string", False),
    ("right", 4, 0.8, 0.1, "x", "B_p_mean", "plaquette_z", "sx_mean", "sigma_x",
     "O_FM_membrane_R1", "fredenhagen_marcu_membrane_r1", "X-membrane R1", True),
    ("right", 5, 0.85, 0.1, "x", "B_p_mean", "plaquette_z", "sx_mean", "sigma_x",
     "O_FM_membrane_R1", "fredenhagen_marcu_membrane_r1", "X-membrane R1", False),
]


def load_qmc_point(hx, hz, basis, L):
    qdir = os.path.join(ROOT, "results", f"qmc_hx{hx:g}_hz{hz:g}")
    means, sems, n = {}, {}, 0
    for f in sorted(glob.glob(os.path.join(qdir, f"paratoric_L{L}_*.json"))):
        d = json.load(open(f))
        if d.get("basis") != basis or "combined" not in d:
            continue
        n += 1
        for k, v in d["combined"].items():
            if isinstance(v, dict) and "mean" in v:
                means.setdefault(k, []).append(v["mean"])
                sems.setdefault(k, []).append(v["sem"])
    if n == 0:
        return None
    return {k: (float(np.mean(means[k])), float(np.sqrt(np.sum(np.square(sems[k]))) / n))
            for k in means}


def pull(nqs_m, nqs_e, qmc):
    qmc_m, qmc_e = qmc
    return (nqs_m - qmc_m) / np.sqrt(nqs_e ** 2 + qmc_e ** 2)


def row(obs, qmc, keys):
    out = []
    for nk, nek, qk in keys:
        if qk not in qmc:
            out.append((float("nan"), float("nan")))
            continue
        out.append((obs[nk], pull(obs[nk], obs[nek], qmc[qk])))
    return out


def plateau_step(series_vals, steps, tol=3.0):
    """First step after which every remaining pull stays within `tol` sigma."""
    for i in range(len(series_vals)):
        if all(abs(x) <= tol for x in series_vals[i:]):
            return steps[i]
    return None


def load_series(path):
    return json.load(open(path))["series"] if os.path.exists(path) else None


def main():
    for cut, L, hx, hz, basis, stab_k, stab_qk, mag_k, mag_qk, op_k, op_qk, op_label, has_old in POINTS:
        qmc = load_qmc_point(hx, hz, basis, L)
        keys = [(stab_k, stab_k.replace("_mean", "_err"), stab_qk),
                (mag_k, mag_k.replace("_mean", "_err"), mag_qk),
                (op_k, op_k + "_err", op_qk),
                ("E0", "E_err", "energy")]
        labels = [stab_k.split("_")[0], mag_k[:2], op_label, "E/N"]

        base_path = os.path.join(ROOT, "results", "phaseB", cut, f"L{L}",
                                 f"phaseB_dual_L{L}_hx{hx:g}_hz{hz:g}.json")
        base_obs = json.load(open(base_path))["observables"] if os.path.exists(base_path) else None

        new_series = load_series(os.path.join(
            ROOT, "results", "phaseB_ablationC", cut, f"L{L}",
            f"phaseB_ablationC_dual_L{L}_hx{hx:g}_hz{hz:g}.snapshots.json"))

        old_series = None
        if has_old:
            old_series = load_series(os.path.join(
                ROOT, "results", "phaseB_ablationA", cut, "L4",
                f"phaseB_ablationA_dual_L4_hx{hx:g}_hz{hz:g}.snapshots.json"))

        print(f"\n{'='*104}\nL={L}  {cut} cut  hx={hx} hz={hz}  (order param = {op_label})\n{'='*104}")
        header = f"{'step':>10} | " + " | ".join(f"{l+' val':>12} {l+' pull':>9}" for l in labels)
        print(header)
        print("-" * len(header))

        if base_obs is not None:
            r = row(base_obs, qmc, keys)
            print(f"{'orig-camp':>10} | " + " | ".join(f"{v:>12.4f} {p:>+9.1f}" for v, p in r))

        if old_series is not None:
            print(f"  --- OLD schedule (no warmup, lr_min=10%dt), 750 steps ---")
            for s in old_series:
                r = row(s, qmc, keys)
                print(f"{'old@'+str(s['step']):>10} | " + " | ".join(f"{v:>12.4f} {p:>+9.1f}" for v, p in r))

        print(f"  --- NEW schedule (warmup 5%, lr_min=20%dt), 500 steps ---")
        for s in new_series:
            r = row(s, qmc, keys)
            print(f"{'new@'+str(s['step']):>10} | " + " | ".join(f"{v:>12.4f} {p:>+9.1f}" for v, p in r))

        steps = [s["step"] for s in new_series]
        op_pulls = [row(s, qmc, keys)[2][1] for s in new_series]
        pstep = plateau_step(op_pulls, steps, tol=3.0)
        print(f"\n  NEW-schedule order-param pull converges (<=3 sigma) by step >= {pstep}")


if __name__ == "__main__":
    main()
