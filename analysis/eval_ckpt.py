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
import math
import os

import numpy as np

from tc3d.builders import build_state
from tc3d.fm import (_bulk_cube, _load_weights, _pauli_product, fm_ratio,
                     magnetic_cube_edges, paratoric_fm_edges)
from tc3d.validation import nqs_observables, topological_observables


def paratoric_fm(vs, geo, hi, dual):
    """O_FM on ParaToric's exact loop geometry (see fm.paratoric_fm_edges).
    Physical σ^z string -> σ^x product in the dual (Hadamard-rotated) frame;
    both loops are short (<= 2R+1 edges), fine through vs.expect either way."""
    closed, open_ = paratoric_fm_edges(geo)
    pauli = "x" if dual else "z"
    return fm_ratio(vs, _pauli_product(hi, open_, pauli),
                    _pauli_product(hi, closed, pauli))


def paratoric_membrane_fm(vs, geo, dual):
    """O_FM X-membrane on the geometry our ParaToric patch hard-codes (centered
    R=L//2 cube, vertical=z; L>=5 — `_bulk_cube` raises at L=4 as intended).

    Evaluated SAMPLE-WISE, never as a NetKet operator: the membrane spans
    6(R+1)^2 = 54..96 edges and a LocalOperator product materializes a
    2^support matrix (the L=6 inline O_FM OOM). In the dual (Hadamard) frame
    the physical sigma^x membrane is a diagonal sigma^z product, so <M> is a
    plain mean of +-1 products over MCMC samples; errors from 16-way binning."""
    if not dual:
        raise NotImplementedError(
            "primal-frame membrane is off-diagonal in the sampling basis; "
            "use fm.fm_ratio_telescoped instead")
    closed, open_ = magnetic_cube_edges(geo, **_bulk_cube(geo), vertical=2)
    x = np.asarray(vs.samples)
    x = x.reshape(-1, x.shape[-1])

    def stats(edges):
        p = np.prod(x[:, list(edges)], axis=1)
        m = np.array([b.mean() for b in np.array_split(p, 16)])
        return float(m.mean()), float(m.std(ddof=1) / math.sqrt(len(m)))

    (o, oe), (c, ce) = stats(open_), stats(closed)
    val = o / math.sqrt(abs(c))
    err = math.hypot(oe / math.sqrt(abs(c)), o * ce / (2 * abs(c) ** 1.5))
    return val, err


def eval_checkpoint(json_path, eval_samples, eval_chains, seed, topological,
                    fm_paratoric=False, fm_membrane_paratoric=False):
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
    if fm_paratoric:
        try:
            val, err = paratoric_fm(vs, geo, hi, dual=cfg.get("dual_basis", False))
            obs["O_FM_paratoric"], obs["O_FM_paratoric_err"] = val, err
        except Exception as e:  # noqa: BLE001
            obs["O_FM_paratoric_error_msg"] = f"{type(e).__name__}: {e}"
    if fm_membrane_paratoric:
        try:
            val, err = paratoric_membrane_fm(vs, geo, dual=cfg.get("dual_basis", False))
            obs["O_FM_membrane_paratoric"] = val
            obs["O_FM_membrane_paratoric_err"] = err
        except Exception as e:  # noqa: BLE001
            obs["O_FM_membrane_paratoric_error_msg"] = f"{type(e).__name__}: {e}"
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
    ap.add_argument("--fm_paratoric", action="store_true",
                    help="also score O_FM on ParaToric's exact FM loop geometry "
                         "(compare against z-basis --fm QMC references)")
    ap.add_argument("--fm_membrane_paratoric", action="store_true",
                    help="also score the X-membrane O_FM on the patched-ParaToric "
                         "cube geometry (compare against x-basis --fm_membrane refs; "
                         "dual-basis checkpoints only, L>=5)")
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
                                  args.seed, args.topological, args.fm_paratoric,
                                  args.fm_membrane_paratoric)
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
