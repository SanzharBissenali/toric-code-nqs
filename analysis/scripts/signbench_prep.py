"""Learned-vs-gated sign-head benchmark: ED referee, per-arm ceilings, M-pre fit.

The 3D half of 2D-TC `docs/signhead_benchmark_plan.md` (fermionic TC, OBC box).
Per (h_x, h_z) point this writes what the three VMC arms (tc3d.train
--sign_arm mlp | mlp + --sign_mlp_init | twobranch) are scored against:

  ED referee   exact_diag_fermionic_{tag}.json + ed_vectors/gs_{tag}.npz in the
               layout analysis/scripts/eval_snapshots.py --exact --ed_vectors
               matches (ground state from sign_fidelity_ftc.ground_state;
               psi(all-up) > 0 gauge).
  ceilings     fidelity ceilings |psi_ED|^2-weight of what each arm's SIGN
               structure cannot represent (min over the free global sign):
                 T_head  1 - F_s(pt2)                   (the spec's T ceiling)
                 T_gate  what T = a A_triv + s A_top (A's positive, a signed)
                         can actually not reach: a > 0 takes either sign where
                         s = -1 and is positive where s = +1, so it misses
                         T_gate_plus = sum |psi|^2 [s = +1, psi < 0]; a < 0
                         misses T_gate_minus = sum |psi|^2 [s = -1, psi > 0];
                         T_gate = the smaller
                 M       0 -- the recovery feature map is injective (checked on
                         all 2^N configs), so (eps, x) determines sigma
                 plus    1 - F_s(+1), the positive-ansatz reference
  M-pre        the SignMLP m_theta fit to sign(psi_ED): |psi_ED|^2-weighted BCE
               over all 2^N configs, P(+) = sigmoid(2 m) (i.e. tanh m = 2P - 1);
               full-batch Adam. Saved as signmlp_{tag}.mpack for
               --sign_mlp_init; its achieved sign fidelity is reported against
               the 1 - 1e-4 capacity bar.

The pt2 head is the production per-config decoder (tc3d.sign_decoders,
tabulated over 2^N) -- the same table TwoBranchNet uses -- cross-checked
against results/fermionic_gate0/sign_table_pt2_{box}_OBC.npy when present.

    python analysis/scripts/signbench_prep.py --Lxyz 2 2 3 \\
        --hx 0.2 0.5 0.8 --hz 0.0 0.2 0.4 --out_dir $PSCRATCH/tc_nqs/fermionic_signbench
N <= 12 (L=2 OBC) is the only size allowed on the dev Mac; 2x2x3 runs on the cluster.
"""
import argparse
import json
import os
import time

import numpy as np

from sign_fidelity_ftc import ground_state
from tc3d.fermionic_decoration import fermionic_plaquettes
from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.sign_decoders import make_decoder_sign, recovery_features
from tc3d.sign_frame import sign_table


def box_tag(Lxyz, bc):
    return f"L{'x'.join(map(str, Lxyz))}_{bc}" if len(set(Lxyz)) > 1 else f"L{Lxyz[0]}_{bc}"


def ceilings(psi, s):
    """Sign-structure fidelity ceilings of the arms at one point (docstring)."""
    w = psi ** 2
    neg, pos = psi < 0, psi > 0
    miss = float(w[(s < 0) != neg].sum())
    gp = min(float(w[(s > 0) & neg].sum()), float(w[(s > 0) & pos].sum()))
    gm = min(float(w[(s < 0) & pos].sum()), float(w[(s < 0) & neg].sum()))
    return {"T_head": min(miss, 1.0 - miss),                     # global sign is free
            "T_gate": min(gp, gm), "T_gate_plus": gp, "T_gate_minus": gm,
            "M": 0.0,
            "plus": min(float(w[neg].sum()), float(w[pos].sum()))}


def pretrain_sign_mlp(feats, psi, hidden, steps, lr, seed, log_every=500):
    """Weighted-BCE fit of SignMLP to sign(psi): returns (params, stats)."""
    import jax
    import jax.numpy as jnp
    import optax
    from tc3d.networks import SignMLP

    jax.config.update("jax_enable_x64", True)
    w = jnp.asarray(psi ** 2 / np.sum(psi ** 2))
    y = jnp.asarray((psi > 0).astype(np.float64))
    f = jnp.asarray(feats)
    net = SignMLP(tuple(hidden))
    params = net.init(jax.random.PRNGKey(seed), f[:2])["params"]
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss_fn(p):
        z = 2.0 * net.apply({"params": p}, f)                  # logit of P(+)
        return jnp.sum(w * (jax.nn.softplus(z) - y * z))       # BCE with logits

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    def fidelity(p):
        m = np.asarray(net.apply({"params": p}, f))
        return float(np.sum(np.asarray(w) * ((m > 0) == (psi > 0)))), m

    t0 = time.time()
    for k in range(steps):
        params, state, loss = step(params, state)
        if (k + 1) % log_every == 0 or k == 0:
            F, _ = fidelity(params)
            print(f"    [pretrain] step {k + 1:5d}  loss={float(loss):.3e}  "
                  f"1-F_s={max(0.0, 1 - F):.3e}", flush=True)
    F, m = fidelity(params)
    t = np.abs(np.tanh(m))
    stats = {"steps": steps, "lr": lr, "seed": seed, "loss": float(loss),
             "F_s": F, "one_minus_F_s": max(0.0, 1.0 - F), "meets_1e-4_bar": bool(1.0 - F <= 1e-4),
             "weighted_mean_abs_tanh": float(np.sum(np.asarray(w) * t)),
             "runtime_s": time.time() - t0}
    return jax.tree_util.tree_map(np.asarray, params), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--Lxyz", type=int, nargs=3, default=[2, 2, 3])
    ap.add_argument("--bc", choices=["OBC"], default="OBC")
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--hx", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--hz", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--head", default="pt2", help="T-arm head (tc3d.sign_decoders kind)")
    ap.add_argument("--dense_max_N", type=int, default=14)
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--mlp_hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--pretrain_steps", type=int, default=6000)
    ap.add_argument("--pretrain_lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/fermionic_signbench")
    args = ap.parse_args()

    geo = ThreeD_ToricCodeGeometry(*args.Lxyz, bc=args.bc)
    stabs = fermionic_plaquettes(geo)
    N, dim = geo.N, 1 << geo.N
    if N > 20:
        raise SystemExit(f"N={N}: the benchmark enumerates 2^N configs (N <= 20)")
    box = box_tag(args.Lxyz, args.bc)
    os.makedirs(os.path.join(args.out_dir, "ed_vectors"), exist_ok=True)
    print(f"[geom] {box}: N={N}, {len(geo.vertex_all)} stars, {len(stabs)} plaquettes",
          flush=True)

    # --- the T head and the M features (geometry only) ---------------------
    t0 = time.time()
    s = sign_table(make_decoder_sign(args.head, geo, J=args.J), N).astype(np.int8)
    gates = []

    def gate(name, ok, detail=""):
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})
        print(f"[gate] {'ok  ' if ok else 'FAIL'} {name}  {detail}", flush=True)

    ref = f"results/fermionic_gate0/sign_table_{args.head}_{'x'.join(map(str, args.Lxyz))}_{args.bc}.npy"
    if os.path.exists(ref):
        gate(f"{args.head} table == {ref}", np.array_equal(np.load(ref), s))
    F, Gp, labels = recovery_features(geo, stabs)
    idx = np.arange(dim, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.uint8)
    fb = (bits.astype(np.int64) @ F.astype(np.int64)) % 2
    uniq = np.unique(np.packbits(fb.astype(np.uint8), axis=1), axis=0).shape[0]
    gate("recovery feature map injective on all 2^N (M ceiling = 0)", uniq == dim,
         f"{uniq}/{dim} distinct; F = {N} eps + {F.shape[1] - N} x bits "
         f"({sum(l[0] == 'plaq' for l in labels)} plaquette + "
         f"{sum(l[0] == 'star' for l in labels)} star)")
    feats = (1.0 - 2.0 * fb).astype(np.float64)
    del fb, bits
    print(f"[head] {args.head} table + features in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for hx in args.hx:
        for hz in args.hz:
            t0 = time.time()
            psi, E0, E1, deg, _H = ground_state(geo, stabs, hx, hz, args.J,
                                                args.dense_max_N)
            tag = f"{box}_hx{float(hx)}_hz{float(hz)}"
            with open(os.path.join(args.out_dir, f"exact_diag_fermionic_{tag}.json"), "w") as f:
                json.dump({"model": "fermionic", "Lx": geo.Lx, "Ly": geo.Ly, "Lz": geo.Lz,
                           "bc": args.bc, "N": N, "hx": float(hx), "hy": 0.0,
                           "hz": float(hz), "J": args.J, "E0": E0, "E1": E1,
                           "gap": E1 - E0, "gs_degeneracy": deg}, f, indent=2)
            np.savez_compressed(
                os.path.join(args.out_dir, "ed_vectors", f"gs_{tag}.npz"),
                psi=psi.astype(np.complex128), E0=E0, hx=hx, hy=0.0, hz=hz,
                Lxyz=np.asarray(args.Lxyz), bc=args.bc, N=N,
                basis_convention="arange(2^N); sigma_i = 1 - 2*bit_i(b); qubit index == bit position")
            gate(f"unique ground state at ({hx},{hz})", deg == 1 and E1 - E0 > 1e-3,
                 f"deg={deg} gap={E1 - E0:.3e}")    # else the referee vector is arbitrary
            c = ceilings(psi, s)
            row = {"hx": float(hx), "hz": float(hz), "E0": E0, "E1": E1, "gap": E1 - E0,
                   "gs_degeneracy": deg, "ceilings": c}
            print(f"  (hx={hx}, hz={hz}) E0={E0:.10f} gap={E1 - E0:.6f}  ceilings: "
                  + "  ".join(f"{k}={v:.3e}" for k, v in c.items())
                  + f"  [{time.time() - t0:.1f}s]", flush=True)
            if not args.no_pretrain:
                params, st = pretrain_sign_mlp(feats, psi, args.mlp_hidden,
                                               args.pretrain_steps, args.pretrain_lr,
                                               args.seed)
                from flax import serialization
                path = os.path.join(args.out_dir, f"signmlp_{tag}.mpack")
                with open(path, "wb") as f:
                    f.write(serialization.msgpack_serialize(params))
                st["path"] = os.path.basename(path)
                row["pretrain"] = st
                print(f"    [pretrain] 1-F_s={st['one_minus_F_s']:.3e} "
                      f"(bar 1e-4: {'met' if st['meets_1e-4_bar'] else 'NOT met'})  "
                      f"<|tanh m|>_w={st['weighted_mean_abs_tanh']:.4f}  "
                      f"{st['runtime_s']:.0f}s -> {path}", flush=True)
            rows.append(row)

    # gate-0 cross-check: the committed plane ceilings share (hx, hz) points
    g0 = f"results/fermionic_gate0/{'x'.join(map(str, args.Lxyz))}_{args.bc}_plane_gate0.json"
    if os.path.exists(g0):
        with open(g0) as f:
            ref_pts = {(p["hx"], p["hz"]): p for p in json.load(f)["points"]}
        for r in rows:
            p = ref_pts.get((r["hx"], r["hz"]))
            if p is not None:
                gate(f"E0 and 1-F_s({args.head}) at ({r['hx']},{r['hz']}) match gate-0",
                     abs(p["E0"] - r["E0"]) < 1e-9
                     and abs(p["one_minus_F_s"][args.head] - r["ceilings"]["T_head"]) < 1e-9,
                     f"E0 {r['E0']:.10f} vs {p['E0']:.10f}; "
                     f"{r['ceilings']['T_head']:.3e} vs {p['one_minus_F_s'][args.head]:.3e}")

    out = {"geometry": {"Lxyz": args.Lxyz, "bc": args.bc, "N": N,
                        "n_stars": len(geo.vertex_all), "n_plaquettes": len(stabs)},
           "head": args.head, "n_features": int(F.shape[1]),
           "feature_labels": [list(l) for l in labels],
           "mlp_hidden": args.mlp_hidden, "points": rows, "gates": gates}
    path = os.path.join(args.out_dir, f"signbench_prep_{box}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[out] {path}\nALL GATES {'PASS' if all(g['pass'] for g in gates) else 'FAIL'}")
    if not all(g["pass"] for g in gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
