"""Hyperparameter-tuning table: NQS runs vs QMC references, all observables.

NetKet-free post-processing (like plot_phase_diagram.py). Joins train.py artifacts
(or their eval_ckpt.py re-evaluations, preferred when present) against ParaToric
reference JSONs, per field point, and emits a config x observable table of
relative errors and pulls.

Observables compared against QMC: E0, <A_v>, <B_p>, Mx, Mz — the QMC JSONs carry
`star_x/plaquette_z/sigma_x/sigma_z` per chain block; blocks are exchangeable, so
the reference is the equal-weight block mean +- SEM (same reasoning as the energy
combine in paratoric_driver.py). O_FM and S2 have no QMC counterpart and are
reported as NQS-internal consistency columns only.

    python analysis/tuning_table.py --runs 'results/tune_rect/*/*.json' \
        --out_md results/tune_rect/tuning_table.md \
        --out_json results/tune_rect/tuning_table.json
"""
import argparse
import glob
import json
import math
import os

# QMC observable name -> nqs_observables key
QMC_TO_NQS = {"energy": "E0", "star_x": "A_v_mean", "plaquette_z": "B_p_mean",
              "sigma_x": "sx_mean", "sigma_z": "sz_mean"}
COMPARED = list(QMC_TO_NQS.values())

# (hx, hz) -> QMC reference JSONs. Runs with "chains" feed every observable;
# a chain-less (combined) file overrides the energy only. Extend as Phase-2
# references land in results/qmc_hx{X}_hz{Z}/.
QMC_REFS = {
    (0.2, 0.10): ["results/qmc_hx0.2_hz0.1/paratoric_L4.json"],
    (0.2, 0.20): ["results/qmc_hx0.2_hz0.2/paratoric_L4_beta12_x4_clean.json",
                  "results/qmc_hx0.2_hz0.2/paratoric_L4_combined.json"],
    # tune-rect corners (2026-08-05, sum-rule + beta-drift validated)
    (0.6, 0.10): ["results/qmc_hx0.6_hz0.1/paratoric_L4_beta12_x4_seed*.json",
                  "results/qmc_hx0.6_hz0.1/paratoric_L4_combined.json"],
    (0.2, 0.15): ["results/qmc_hx0.2_hz0.15/paratoric_L4_beta12_x4_seed*.json"],
    (0.6, 0.15): ["results/qmc_hx0.6_hz0.15/paratoric_L4_beta*_x4_seed*.json",
                  "results/qmc_hx0.6_hz0.15/paratoric_L4_combined.json"],
}


def qmc_reference(paths):
    """Pool chain blocks across `paths` -> {nqs_key: (mean, sem)}. A file without
    `chains` (a combined file) overrides E0 with its own (E, E_err)."""
    paths = [q for p in paths for q in (sorted(glob.glob(p)) or [p])]
    blocks, e_override = [], None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if d.get("chains"):
            blocks.extend(d["chains"])
        else:
            e_override = (float(d["E"]), float(d["E_err"]))
    ref = {}
    for qmc_key, nqs_key in QMC_TO_NQS.items():
        vals = [c[qmc_key][0] for c in blocks if qmc_key in c]
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            sem = math.sqrt(sum((v - m) ** 2 for v in vals)
                            / (len(vals) - 1) / len(vals))
            ref[nqs_key] = (m, sem)
    if e_override is not None:
        ref["E0"] = e_override
    return ref


def _config_cols(cfg):
    nh = cfg.get("noninv_hidden")
    return {
        "dual": bool(cfg.get("dual_basis", False)),
        "kernel": cfg.get("kernel_size"),
        "noninv": "x".join(map(str, nh)) if nh
                  else f"{cfg.get('n_noninv', 2)}x{cfg.get('noninv_channels', 4)}",
        "inv": "-".join(map(str, cfg.get("inv_hidden", ()))),
        "radius": cfg.get("radius_edge", 1.05),
        "diag_shift": cfg.get("diag_shift"),
        "dt": cfg.get("dt"),
        "n_samples": cfg.get("n_samples"),
        "seed": cfg.get("seed", 0),
    }


def load_row(json_path, eval_suffix):
    with open(json_path) as f:
        meta = json.load(f)
    if "config" not in meta or "observables" not in meta:
        return None                                  # aggregate/curve/eval file
    cfg, obs = meta["config"], meta["observables"]
    eval_path = json_path[:-len(".json")] + f"{eval_suffix}.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            obs = {**obs, **json.load(f)["observables"]}   # re-eval overrides
    curve = meta.get("curve") or {}
    steps = max((len(v) for v in curve.values() if isinstance(v, list)), default=0)
    row = {"name": meta.get("name"), "point": (float(cfg.get("hx", 0.0)),
                                               float(cfg.get("hz", 0.0))),
           **_config_cols(cfg), "n_params": meta.get("n_params"),
           "s_step": round(meta["runtime_s"] / steps, 2) if steps else None,
           "diverged": bool(meta.get("diverged")), "obs": obs}
    return row


def compare(row, ref):
    """Per-observable relative error + pull against the QMC reference."""
    out = {}
    for key in COMPARED:
        if key not in ref or row["obs"].get(key) is None:
            continue
        m, e = float(row["obs"][key]), float(row["obs"].get(
            {"E0": "E_err"}.get(key, key.replace("_mean", "_err")), 0.0) or 0.0)
        rm, re_ = ref[key]
        sig = math.sqrt(e * e + re_ * re_)
        out[key] = {"nqs": m, "nqs_err": e, "ref": rm, "ref_err": re_,
                    "rel": abs(m - rm) / abs(rm) if rm else float("nan"),
                    "pull": (m - rm) / sig if sig else float("nan")}
    return out


def score(row):
    """Composite ranking score (LOWER = better). USER-DEFINED physics judgment:
    how to weigh the energy pull/rel-err against stabilizers and magnetizations
    (and whether Vscore enters) shapes which config wins.

    TODO(user): implement (~5-10 lines). `row["cmp"]` holds per-observable
    {rel, pull, ...} for E0/A_v_mean/B_p_mean/sx_mean/sz_mean; row["obs"]["Vscore"]
    is available. Candidate shapes: max |pull| across observables (worst-case),
    energy-rel-err with a chi2 penalty for the rest, or a weighted sum.
    Until implemented, the table falls back to sorting by |pull_E| alone.
    """
    return None


def fmt_md(rows):
    hdr = ["name", "dual", "kernel", "noninv", "inv", "radius", "diag_shift", "dt",
           "n_samples", "n_params", "s/step", "Vscore",
           "relE", "pullE", "relA", "relB", "relMx", "relMz", "O_FM", "S2", "score"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c, o = r["cmp"], r["obs"]
        def rel(k): return f"{c[k]['rel']:.2e}" if k in c else "—"
        def pm(k, ek):
            return (f"{o[k]:.3f}({o.get(ek, 0) or 0:.3f})"
                    if isinstance(o.get(k), (int, float)) else "—")
        lines.append("| " + " | ".join(str(x) for x in [
            r["name"], int(r["dual"]), r["kernel"], r["noninv"], r["inv"],
            r["radius"], r["diag_shift"], r["dt"], r["n_samples"], r["n_params"],
            r["s_step"], f"{o.get('Vscore', float('nan')):.1e}",
            rel("E0"), f"{c['E0']['pull']:+.1f}" if "E0" in c else "—",
            rel("A_v_mean"), rel("B_p_mean"), rel("sx_mean"), rel("sz_mean"),
            pm("O_FM", "O_FM_err"), pm("S2", "S2_err"),
            f"{r['score']:.3g}" if r.get("score") is not None else "—"]) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, help="glob of train.py artifact JSONs")
    ap.add_argument("--eval_suffix", default=".eval65k")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(args.runs)):
        if p.endswith(f"{args.eval_suffix}.json") or p.endswith(".curve.json"):
            continue
        r = load_row(p, args.eval_suffix)
        if r is None:
            continue
        if r["diverged"]:
            print(f"[table] excluded (diverged): {r['name']}")
            continue
        refs = QMC_REFS.get(r["point"])
        if not refs:
            print(f"[table] no QMC ref for point {r['point']}: {r['name']}")
            continue
        r["cmp"] = compare(r, qmc_reference(refs))
        r["score"] = score(r)
        rows.append(r)

    rows.sort(key=lambda r: r["score"] if r.get("score") is not None
              else abs(r["cmp"].get("E0", {}).get("pull", float("inf"))))
    by_point = sorted({r["point"] for r in rows})
    md = []
    for pt in by_point:
        md += [f"### (hx, hz) = {pt}", "",
               fmt_md([r for r in rows if r["point"] == pt]), ""]
    text = "\n".join(md)
    print(text)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write(text + "\n")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=1, default=str)
    print(f"\n[table] {len(rows)} runs across {len(by_point)} points"
          + ("  (score(): user TODO — sorted by |pull_E|)"
             if rows and rows[0].get("score") is None else ""))
