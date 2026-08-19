"""Bank a converged-but-unfinished sweep.py point as done, without training it out.

`tc3d.sweep`'s per-point skip check only looks for `{name}.json` — the file
`train.py` writes at full completion. A point whose energy/Vscore have already
plateaued well before `n_iter` is a GPU-hour waste to finish; this script
re-derives that exact final-JSON schema from the last periodic checkpoint
(`{name}.ckpt.mpack` + `{name}.curve.json`) so a resubmitted sweep sees the
point as finished and moves to the next field value.

Needs NetKet/JAX (rebuilds the VMC stack) -> run on the cluster GPU.

    python analysis/scripts/bank_point.py --curve_json $OUT/gridinv_..._eline.curve.json \
        --eval_samples 8192
"""
import argparse
import json
import os
import shutil

from tc3d.builders import build_state
from tc3d.fm import _json_nonfinite_safe, _load_weights
from tc3d.validation import nqs_observables

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--curve_json", required=True,
                    help="the point's {name}.curve.json (periodic checkpoint)")
    ap.add_argument("--eval_samples", type=int, default=None,
                    help="override n_samples for the final observable read "
                         "(default: keep the checkpoint's training value)")
    ap.add_argument("--eval_chains", type=int, default=None)
    args = ap.parse_args()

    with open(args.curve_json) as f:
        meta = json.load(f)
    cfg = dict(meta["config"])
    if args.eval_samples:
        cfg["n_samples"] = args.eval_samples
    if args.eval_chains:
        cfg["n_chains"] = args.eval_chains

    geo, hi, Ham, vs, xz = build_state(cfg)
    vs = _load_weights(vs, args.curve_json)          # resolves to {name}.ckpt.mpack
    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz, dual=cfg.get("dual_basis", False))

    name = meta["name"]
    weights_base = os.path.join(cfg["out_dir"], name)
    ckpt_mpack = weights_base + ".ckpt.mpack"
    shutil.copy(ckpt_mpack, weights_base + ".mpack")  # canonical name for downstream tools

    result = {
        "name": name, "config": cfg, "n_params": int(vs.n_parameters),
        "runtime_s": None, "observables": obs, "curve": meta["curve"],
        "weights": f"{weights_base}.mpack", "diverged": False,
        "early_stopped_at_step": meta.get("completed_steps"),
    }
    with open(f"{weights_base}.json", "w") as f:
        json.dump(_json_nonfinite_safe(result), f, indent=2)
    print(f"[bank] {name}: E={obs['E0']:.6f}({obs['E_err']:.2e}) "
          f"Vscore={obs['Vscore']:.2e} early-stopped @ step {meta.get('completed_steps')} "
          f"-> {weights_base}.json")
