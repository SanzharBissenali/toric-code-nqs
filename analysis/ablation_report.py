"""Phase-B ablation report: does longer training or more capacity close the
QMC-vs-NQS gap on stabilizer / magnetization / order-parameter at the 4 worst
near-transition L=4 points?

Inputs (all local, already rsynced):
  results/phaseB/{cut}/L4/*.json                    baseline: winner arch, 150 steps
  results/phaseB_ablationA/{cut}/L4/*.snapshots.json Run A:    winner arch, step 50..750
  results/phaseB_ablationB/{cut}/L4/*.json           Run B:    deeper arch, 150 steps
  results/qmc_hx*_hz*/paratoric_L4_*.json            QMC reference

Prints one table per point: step -> (value, pull) for the order parameter,
stabilizer, and magnetization that discriminate that cut, plus where the Run A
series' pull first settles within 1.0 sigma of its step-750 value ("plateau
step") and how Run B (deeper, 150 steps) compares to Run A's own step-150
snapshot at matched training length.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (cut, hx, hz, qmc_basis, nqs_stab_key, qmc_stab_key, nqs_mag_key, qmc_mag_key,
#  nqs_op_key, qmc_op_key, label)
POINTS = [
    ("up", 0.2, 0.26, "z", "A_v_mean", "star_x", "sz_mean", "sigma_z",
     "O_FM_paratoric", "fredenhagen_marcu", "Z-string"),
    ("up", 0.2, 0.28, "z", "A_v_mean", "star_x", "sz_mean", "sigma_z",
     "O_FM_paratoric", "fredenhagen_marcu", "Z-string"),
    ("right", 0.75, 0.1, "x", "B_p_mean", "plaquette_z", "sx_mean", "sigma_x",
     "O_FM_membrane_R1", "fredenhagen_marcu_membrane_r1", "X-membrane R1"),
    ("right", 0.8, 0.1, "x", "B_p_mean", "plaquette_z", "sx_mean", "sigma_x",
     "O_FM_membrane_R1", "fredenhagen_marcu_membrane_r1", "X-membrane R1"),
]


def load_qmc_point(hx, hz, basis, L=4):
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
    """(value, pull) for each (nqs_key, nqs_err_key, qmc_key) triple."""
    out = []
    for nk, nek, qk in keys:
        if qk not in qmc:
            out.append((float("nan"), float("nan")))
            continue
        out.append((obs[nk], pull(obs[nk], obs[nek], qmc[qk])))
    return out


def plateau_step(series_vals, steps, tol=1.0):
    """First step whose pull stays within `tol` sigma of the FINAL pull for
    every later snapshot too (a true plateau, not a one-off crossing)."""
    final = series_vals[-1]
    for i, v in enumerate(series_vals):
        if all(abs(x - final) <= tol for x in series_vals[i:]):
            return steps[i]
    return None


def main():
    for cut, hx, hz, basis, stab_k, stab_qk, mag_k, mag_qk, op_k, op_qk, op_label in POINTS:
        qmc = load_qmc_point(hx, hz, basis)
        keys = [(stab_k, stab_k.replace("_mean", "_err"), stab_qk),
                (mag_k, mag_k.replace("_mean", "_err"), mag_qk),
                (op_k, op_k + "_err", op_qk),
                ("E0", "E_err", "energy")]
        labels = [stab_k.split("_")[0], mag_k[:2], op_label, "E/N"]

        base_path = os.path.join(ROOT, "results", "phaseB", cut, "L4",
                                 f"phaseB_dual_L4_hx{hx:g}_hz{hz:g}.json")
        base_obs = json.load(open(base_path))["observables"] if os.path.exists(base_path) else None

        snap_path = os.path.join(ROOT, "results", "phaseB_ablationA", cut, "L4",
                                 f"phaseB_ablationA_dual_L4_hx{hx:g}_hz{hz:g}.snapshots.json")
        series = json.load(open(snap_path))["series"]

        runb_path = os.path.join(ROOT, "results", "phaseB_ablationB", cut, "L4",
                                 f"phaseB_ablationB_dual_L4_hx{hx:g}_hz{hz:g}.json")
        runb_obs = json.load(open(runb_path))["observables"] if os.path.exists(runb_path) else None

        print(f"\n{'='*100}\n{cut} cut  hx={hx} hz={hz}  (order param = {op_label})\n{'='*100}")
        header = f"{'step':>6} | " + " | ".join(f"{l+' val':>12} {l+' pull':>9}" for l in labels)
        print(header)
        print("-" * len(header))

        if base_obs is not None:
            r = row(base_obs, qmc, keys)
            print(f"{'orig150':>6} | " + " | ".join(f"{v:>12.4f} {p:>+9.1f}" for v, p in r))
        for s in series:
            r = row(s, qmc, keys)
            print(f"{s['step']:>6} | " + " | ".join(f"{v:>12.4f} {p:>+9.1f}" for v, p in r))
        if runb_obs is not None:
            r = row(runb_obs, qmc, keys)
            print(f"{'RunB150':>6} | " + " | ".join(f"{v:>12.4f} {p:>+9.1f}" for v, p in r))

        # plateau on the diagnostic order parameter's pull series
        steps = [s["step"] for s in series]
        op_pulls = [row(s, qmc, keys)[2][1] for s in series]
        pstep = plateau_step(op_pulls, steps, tol=1.0)
        print(f"\n  order-param pull plateau (within 1.0 sigma of step-750): step >= {pstep}")

        # Run B vs Run A's own step-150 snapshot (matched training length, capacity isolated)
        step150 = next((s for s in series if s["step"] == 150), None)
        if step150 is not None and runb_obs is not None:
            op_a150 = row(step150, qmc, keys)[2][1]
            op_b150 = row(runb_obs, qmc, keys)[2][1]
            print(f"  order-param pull @150 steps: winner-arch={op_a150:+.1f}sigma  "
                  f"deeper-arch={op_b150:+.1f}sigma  "
                  f"({'capacity helps' if abs(op_b150) < abs(op_a150) else 'capacity does not help'})")


if __name__ == "__main__":
    main()
