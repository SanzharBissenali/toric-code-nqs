"""Re-evaluate train.py --snapshot_every mid-training weights snapshots.

Phase-B ablation A: for a run launched with --snapshot_every N (which writes
{name}.step{N}.mpack alongside the usual .ckpt.mpack, never overwritten — see
tc3d/train.py's `_write_checkpoint`), replay tc3d.validation.pooled_final_observables
— the SAME call train.py makes once at the end of training — once per saved
snapshot instead. This gives a (step, observables) series showing whether the
QMC-vs-NQS gap at a transition point closes with more training, without any
change to train.py's core loop.

Needs NetKet/JAX (rebuilds the VMC stack) -> run on the cluster GPU.

    python analysis/eval_snapshots.py --dir $PSCRATCH/tc_nqs/phaseB_ablationA/up/L4 \
        --glob 'phaseB_ablationA_dual_L4_hx0.2_hz0.26*.json' --rounds 8
"""
import argparse
import glob
import json
import os
import re

from tc3d.builders import build_state
from tc3d.io import load_weights
from tc3d.validation import build_eval_operators, pooled_final_observables

SNAPSHOT_RE = re.compile(r"\.step(\d+)\.mpack$")


def eval_run_snapshots(json_path, rounds, seed=None):
    with open(json_path) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    if seed is not None:
        cfg["seed"] = seed
    weights_base = json_path[:-len(".json")]
    snaps = sorted(
        (int(m.group(1)), p) for p in glob.glob(f"{weights_base}.step*.mpack")
        for m in [SNAPSHOT_RE.search(p)] if m
    )
    if not snaps:
        raise FileNotFoundError(
            f"no {weights_base}.step*.mpack snapshots found — was this run "
            "launched with --snapshot_every?")

    geo, hi, Ham, vs, xz_stabs = build_state(cfg)
    eval_ops = build_eval_operators(hi, geo, cfg, xz_stabs=xz_stabs)

    series = []
    for step, mpack_path in snaps:
        vs = load_weights(vs, mpack_path[:-len(".mpack")])
        obs = pooled_final_observables(vs, Ham, geo, cfg, xz_stabs=xz_stabs,
                                       rounds=rounds, eval_ops=eval_ops)
        obs["step"] = step
        series.append(obs)
        print(f"[eval_snapshots] step {step:4d}: E0={obs['E0']:+.4f}  "
              f"Vscore={obs['Vscore']:.2e}", flush=True)
    return {"name": meta.get("name"), "source_json": os.path.basename(json_path),
            "config": cfg, "series": series}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="directory of train.py artifacts")
    ap.add_argument("--glob", default="*.json", help="artifact filter within --dir")
    ap.add_argument("--rounds", type=int, default=8,
                    help="pooled eval rounds per snapshot (match the campaign's "
                         "--final_eval_rounds so statistics are comparable)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_suffix", default=".snapshots.json")
    args = ap.parse_args()

    for json_path in sorted(glob.glob(os.path.join(args.dir, args.glob))):
        if json_path.endswith(args.out_suffix):
            continue
        print(f"[eval_snapshots] {json_path}", flush=True)
        result = eval_run_snapshots(json_path, args.rounds, seed=args.seed)
        out_path = json_path[:-len(".json")] + args.out_suffix
        with open(out_path, "w") as f:
            from tc3d.fm import _json_nonfinite_safe
            json.dump(_json_nonfinite_safe(result), f, indent=2)
        print(f"[eval_snapshots] wrote {out_path}", flush=True)
