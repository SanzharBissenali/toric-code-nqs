"""Re-evaluate saved train.py checkpoints with a larger sample budget.

One-run MC error at the campaign's n_samples=8192 (~4e-3 at L=4) cannot separate
close configs; E_err ~ 1/sqrt(n), so a single re-evaluation of the saved .mpack at
65536 samples sharpens the mean ~3x without retraining (log 2026-07-29 precedent).
Writes `{name}{suffix}.json` next to each artifact: the full `nqs_observables` set
(E0/A_v/B_p/Mx/Mz + errors) and, with --topological, the inline O_FM + S2.

Needs NetKet/JAX (rebuilds the VMC stack) -> run on the cluster GPU; export
JAX_COMPILATION_CACHE_DIR manually (extract-style invocations don't inherit it).

    python analysis/eval_ckpt.py --dir $PSCRATCH/tc_nqs/tune_rect/hx0.2_hz0.1 \
        --eval_samples 65536 --eval_chains 16 --topological
"""
import argparse
import glob
import json
import os

from tc3d.builders import build_state
from tc3d.fm import _load_weights
from tc3d.validation import nqs_observables, topological_observables


def eval_checkpoint(json_path, eval_samples, eval_chains, seed, topological):
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("diverged"):
        return None
    cfg = dict(meta["config"])
    cfg["n_samples"] = eval_samples
    if eval_chains:
        cfg["n_chains"] = eval_chains
    if seed is not None:
        cfg["seed"] = seed
    geo, hi, Ham, vs, xz = build_state(cfg)
    vs = _load_weights(vs, json_path)
    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz, dual=cfg.get("dual_basis", False))
    if topological:
        try:
            obs.update(topological_observables(vs, geo, cfg, hi=hi))
        except Exception as e:  # noqa: BLE001 — mirror train.py: never lose an eval
            obs["topological_error_msg"] = f"{type(e).__name__}: {e}"
    return {"name": meta.get("name"), "source_json": os.path.basename(json_path),
            "eval_samples": eval_samples, "eval_chains": eval_chains,
            "observables": obs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="directory of train.py artifacts")
    ap.add_argument("--glob", default="*.json", help="artifact filter within --dir")
    ap.add_argument("--eval_samples", type=int, default=65536)
    ap.add_argument("--eval_chains", type=int, default=16,
                    help="few LONG chains -> valid autocorrelation-corrected errors")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--topological", action="store_true",
                    help="also re-evaluate the inline O_FM + S2")
    ap.add_argument("--suffix", default=".eval65k")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.dir, args.glob))
                   if not p.endswith(f"{args.suffix}.json")
                   and not p.endswith(".curve.json"))
    for p in paths:
        out_path = p[:-len(".json")] + f"{args.suffix}.json"
        if args.skip_existing and os.path.exists(out_path):
            print(f"[eval] skip (exists): {os.path.basename(out_path)}")
            continue
        try:
            res = eval_checkpoint(p, args.eval_samples, args.eval_chains,
                                  args.seed, args.topological)
        except FileNotFoundError as e:  # no sibling .mpack (aggregates etc.)
            print(f"[eval] skip {os.path.basename(p)}: {e}")
            continue
        if res is None:
            print(f"[eval] skip (diverged): {os.path.basename(p)}")
            continue
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        o = res["observables"]
        print(f"[eval] {res['name']}: E={o['E0']:.4f}({o['E_err']:.4f}) "
              f"Vscore={o['Vscore']:.2e} -> {os.path.basename(out_path)}")
