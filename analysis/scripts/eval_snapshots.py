"""Re-evaluate train.py --snapshot_every mid-training weights snapshots.

Phase-B ablation A: for a run launched with --snapshot_every N (which writes
{name}.step{N}.mpack alongside the usual .ckpt.mpack, never overwritten — see
tc3d/train.py's `_write_checkpoint`), replay tc3d.validation.pooled_final_observables
— the SAME call train.py makes once at the end of training — once per saved
snapshot instead. This gives a (step, observables) series showing whether the
QMC-vs-NQS gap at a transition point closes with more training, without any
change to train.py's core loop.

--topological additionally replays tc3d.validation.topological_observables per
snapshot (fm_sector, O_FM(+_err), S2(+_err), …) — train.py itself only calls it
when final_eval_rounds<=1, so Phase-B-style pooled runs (final_eval_rounds>1)
never get it; here it always runs when the flag is passed. Best-effort (never
raises; {} below L=4), matching the function's own contract. Saved configs
default fm_sector="auto" (`_auto_fm_sector`: hx>=hz -> magnetic), which flips
sector mid-cut — e.g. an hx=0.2 electric hz-sweep resolves magnetic once
hz drops below 0.2. Pass --fm_sector electric|magnetic to force one coherent
sector across the whole replayed cut.

A failed snapshot (partially-written .mpack, stale sampler shape, …) records
{"step": N, "error": ...} and the series continues; a failed run (missing
snapshots, corrupt config) is skipped and the remaining --glob matches proceed.

Needs NetKet/JAX (rebuilds the VMC stack) -> run on the cluster GPU.

    python analysis/scripts/eval_snapshots.py --dir $PSCRATCH/tc_nqs/phaseB_ablationA/up/L4 \
        --glob 'phaseB_ablationA_dual_L4_hx0.2_hz0.26*.json' --rounds 8 --topological
"""
import argparse
import glob
import json
import os
import re

from tc3d.builders import build_state
from tc3d.io import load_weights
from tc3d.validation import (
    build_eval_operators, pooled_final_observables, topological_observables)

SNAPSHOT_RE = re.compile(r"\.step(\d+)\.mpack$")


def eval_run_snapshots(json_path, rounds, seed=None, topological=False, fm_sector="auto"):
    with open(json_path) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    if seed is not None:
        cfg["seed"] = seed
    # --fm_sector overrides only the (separate) topological_observables call, on a
    # copy — pooled_final_observables and build_state never read fm_sector.
    topo_cfg = cfg if fm_sector == "auto" else {**cfg, "fm_sector": fm_sector}
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
        try:                                          # one bad snapshot (partial
            vs = load_weights(vs, mpack_path[:-len(".mpack")])   # .mpack, stale
            obs = pooled_final_observables(vs, Ham, geo, cfg, xz_stabs=xz_stabs,
                                           rounds=rounds, eval_ops=eval_ops)
            if topological:
                obs.update(topological_observables(vs, geo, topo_cfg, hi=hi))
            obs["step"] = step
            print(f"[eval_snapshots] step {step:4d}: E0={obs['E0']:+.4f}  "
                  f"Vscore={obs['Vscore']:.2e}", flush=True)
        except Exception as e:                        # noqa: BLE001 — shape, …) must
            obs = {"step": step, "error": f"{type(e).__name__}: {e}"}  # not abort the run
            print(f"[eval_snapshots] step {step:4d}: FAILED ({obs['error']})", flush=True)
        series.append(obs)
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
    ap.add_argument("--topological", action="store_true",
                    help="also compute fm_sector/O_FM/S2 per snapshot via "
                         "tc3d.validation.topological_observables (best-effort, "
                         "needs L>=4; runs even if the checkpoint used "
                         "final_eval_rounds>1)")
    ap.add_argument("--fm_sector", default="auto", choices=["auto", "electric", "magnetic"],
                    help="--topological only: override the checkpoint's saved "
                         "fm_sector (default auto = cfg's own value, which can "
                         "flip mid-cut via hx>=hz). Force one sector across a cut.")
    ap.add_argument("--out_suffix", default=".snapshots.json")
    args = ap.parse_args()

    for json_path in sorted(glob.glob(os.path.join(args.dir, args.glob))):
        if json_path.endswith(args.out_suffix) or json_path.endswith(".curve.json"):
            continue
        print(f"[eval_snapshots] {json_path}", flush=True)
        try:                                            # one bad run (missing snapshots,
            result = eval_run_snapshots(json_path, args.rounds, seed=args.seed,
                                        topological=args.topological,
                                        fm_sector=args.fm_sector)
        except Exception as e:                          # corrupt config, …) must not
            print(f"[eval_snapshots] {json_path}: FAILED ({type(e).__name__}: {e}) "
                  "— skipping", flush=True)             # abort the remaining --glob matches
            continue
        out_path = json_path[:-len(".json")] + args.out_suffix
        with open(out_path, "w") as f:
            from tc3d.fm import _json_nonfinite_safe
            json.dump(_json_nonfinite_safe(result), f, indent=2)
        print(f"[eval_snapshots] wrote {out_path}", flush=True)
