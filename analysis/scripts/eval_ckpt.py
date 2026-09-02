"""Re-evaluate saved train.py checkpoints with a larger sample budget.

One-run MC error at the campaign's n_samples=8192 (~4e-3 at L=4) cannot separate
close configs; E_err ~ 1/sqrt(n), so a single re-evaluation of the saved .mpack at
65536 samples sharpens the mean ~3x without retraining (log 2026-07-29 precedent).
Writes `{name}{suffix}.json` next to each artifact: the full `nqs_observables` set
(E0/A_v/B_p/Mx/Mz + errors) and, with --topological, the inline O_FM + S2.

Needs NetKet/JAX (rebuilds the VMC stack) -> run on the cluster GPU; export
JAX_COMPILATION_CACHE_DIR manually (extract-style invocations don't inherit it).

    python analysis/scripts/eval_ckpt.py --dir $PSCRATCH/tc_nqs/tune_rect/hx0.2_hz0.1 \
        --eval_samples 65536 --eval_chains 16 --topological
"""
import argparse
import glob
import json
import os

import numpy as np

from tc3d.builders import build_state
from tc3d.fm import (_load_weights, _pauli_product, fm_ratio, fm_ratio_sampled,
                     paratoric_fm_edges, paratoric_membrane_kwargs)
from tc3d.sign_frame import frame_eval_ops
from tc3d.validation import build_eval_operators, nqs_observables, topological_observables


def paratoric_fm(vs, geo, hi, dual, model="bosonic"):
    """O_FM on ParaToric's exact loop geometry (see fm.paratoric_fm_edges).
    Physical σ^z string -> σ^x product in the dual (Hadamard-rotated) frame;
    both loops are short (<= 2R+1 edges), fine through vs.expect either way."""
    if model == "fermionic":
        # A bare σ^z string anticommutes with the decorated B̃_p it crosses:
        # ⟨open⟩ ≡ 0 and ⟨closed⟩ ≈ 0 — noise/√noise silently recorded under
        # the bosonic key (2026-08-10 audit). Same guard as fm.py's paratoric
        # placement; use the dressed electric path when a fermionic FM exists.
        raise NotImplementedError(
            "ParaToric-geometry FM is bosonic-only (bare σ^z string is not a "
            "conserved-sector operator of the fermionic model)")
    closed, open_ = paratoric_fm_edges(geo)
    pauli = "x" if dual else "z"
    return fm_ratio(vs, _pauli_product(hi, open_, pauli),
                    _pauli_product(hi, closed, pauli))


def paratoric_membrane_fm(vs, geo, dual, R=None):
    """O_FM X-membrane for one ParaToric-convention family
    (`fm.paratoric_membrane_kwargs`): ``R=None`` → the growing corner-rule cube
    (matches the rewritten ParaToric patch edge-for-edge; L>=5), ``R=<int>`` →
    the fixed-size anchor cube (R=1 exists from L=4 up).

    Evaluated SAMPLE-WISE, never as a NetKet operator: the membrane spans
    6(R+1)^2 = 24..150 edges and a LocalOperator product materializes a
    2^support matrix (the L=6 inline O_FM OOM). In the dual (Hadamard) frame
    the physical sigma^x membrane is a diagonal sigma^z product — delegated to
    `fm.fm_ratio_sampled`, whose jackknife runs through the ASSEMBLED ratio:
    open/closed share samples and half the support, and the old independent-
    error hypot over-estimated the bar ~1.6x (2026-08-10 audit)."""
    if not dual:
        raise NotImplementedError(
            "primal-frame membrane is off-diagonal in the sampling basis; "
            "use fm.fm_ratio_telescoped instead")
    kw = paratoric_membrane_kwargs(geo, R)
    return fm_ratio_sampled(vs, geo, n_blocks=16, **kw)


def eval_checkpoint(json_path, eval_samples, eval_chains, seed, topological,
                    fm_paratoric=False, fm_membrane_paratoric=False,
                    fm_membrane_families=("pt", 1), chunk_size=None):
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("diverged"):
        return None
    cfg = dict(meta["config"])
    cfg["n_samples"] = eval_samples
    if chunk_size:
        # The checkpoint's training chunk_size is tuned for the TRAINING batch;
        # a true-65k eval at L=6 (N=540) OOMs the shared-QOS slice with it
        # (2026-08-11, jobs 56586392-395/56588971-974 — un-masked by the loader
        # fix: pre-fix "65k" evals silently ran at 8k and fit). Override here.
        cfg["chunk_size"] = int(chunk_size)
    if eval_chains:
        cfg["n_chains"] = eval_chains
    if seed is not None:
        cfg["seed"] = seed
    geo, hi, Ham, vs, xz = build_state(cfg)     # Ham is already framed (build_hamiltonian)
    vs = _load_weights(vs, json_path)
    mean_ops, _ = build_eval_operators(hi, geo, cfg, xz_stabs=xz)
    mean_ops = frame_eval_ops(mean_ops, cfg, geo)    # sign_frame: A_v/B~_p/M_x need S O S
    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz, dual=cfg.get("dual_basis", False),
                          mean_ops=mean_ops)
    if topological:
        try:
            obs.update(topological_observables(vs, geo, cfg, hi=hi))
        except Exception as e:  # noqa: BLE001 — mirror train.py: never lose an eval
            obs["topological_error_msg"] = f"{type(e).__name__}: {e}"
    if fm_paratoric:
        try:
            val, err = paratoric_fm(vs, geo, hi, dual=cfg.get("dual_basis", False),
                                    model=cfg.get("model", "bosonic"))
            obs["O_FM_paratoric"], obs["O_FM_paratoric_err"] = val, err
        except Exception as e:  # noqa: BLE001
            obs["O_FM_paratoric_error_msg"] = f"{type(e).__name__}: {e}"
    if fm_membrane_paratoric:
        # One entry per requested family: "pt" = growing corner-rule cube (L>=5),
        # <int> = fixed-size anchor cube. A family that doesn't exist at this L
        # (pt at L=4) records its skip reason instead of failing the eval.
        for fam in fm_membrane_families:
            tag = "pt" if fam == "pt" else f"R{int(fam)}"
            try:
                val, err = paratoric_membrane_fm(
                    vs, geo, dual=cfg.get("dual_basis", False),
                    R=None if fam == "pt" else int(fam))
                obs[f"O_FM_membrane_{tag}"] = val
                obs[f"O_FM_membrane_{tag}_err"] = err
            except Exception as e:  # noqa: BLE001
                obs[f"O_FM_membrane_{tag}_error_msg"] = f"{type(e).__name__}: {e}"
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
                         "cube geometries (compare against x-basis --fm_membrane/"
                         "--fm_membrane_r1 refs; dual-basis checkpoints only)")
    ap.add_argument("--fm_membrane_R", default="pt,1",
                    help="comma-separated membrane families to score: 'pt' = growing "
                         "corner-rule cube (L>=5), an int = fixed-size anchor cube "
                         "(1 needs L>=4). Default scores both; a family missing at "
                         "this L records a skip message instead of failing.")
    ap.add_argument("--chunk_size", type=int, default=None,
                    help="override the checkpoint's chunk_size for the eval "
                         "(smaller = less memory; e.g. 4096 for L=6 at 65k samples "
                         "on shared QOS). Default: keep the checkpoint's value.")
    ap.add_argument("--suffix", default=".eval65k")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.dir, args.glob))
                   if not p.endswith(f"{args.suffix}.json")
                   and not p.endswith(".curve.json"))
    fams = tuple(s if s == "pt" else int(s)
                 for s in (t.strip() for t in args.fm_membrane_R.split(","))
                 if s)
    for p in paths:
        out_path = p[:-len(".json")] + f"{args.suffix}.json"
        if args.skip_existing and os.path.exists(out_path):
            print(f"[eval] skip (exists): {os.path.basename(out_path)}")
            continue
        try:
            res = eval_checkpoint(p, args.eval_samples, args.eval_chains,
                                  args.seed, args.topological, args.fm_paratoric,
                                  args.fm_membrane_paratoric, fams,
                                  chunk_size=args.chunk_size)
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
