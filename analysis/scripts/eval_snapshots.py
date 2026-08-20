"""Re-evaluate train.py --snapshot_every mid-training weights snapshots.

Phase-B ablation A: for a run launched with --snapshot_every N (which writes
{name}.step{N}.mpack alongside the usual .ckpt.mpack, never overwritten — see
tc3d/train.py's `_write_checkpoint`), replay tc3d.validation.pooled_final_observables
— the SAME call train.py makes once at the end of training — once per saved
snapshot instead. This gives a (step, observables) series showing whether the
QMC-vs-NQS gap at a transition point closes with more training, without any
change to train.py's core loop.

Needs NetKet/JAX (rebuilds the VMC stack) -> run on the cluster GPU.

    python analysis/scripts/eval_snapshots.py --dir $PSCRATCH/tc_nqs/phaseB_ablationA/up/L4 \
        --glob 'phaseB_ablationA_dual_L4_hx0.2_hz0.26*.json' --rounds 8

--exact (notes/fermionic_obc_l2_benchmark_plan.md Phase 3): for small systems
(geo.N <= 20 — the L=2 OBC fermionic benchmark, N=12) also contracts the NQS
over the FULL 2^N Hilbert space per snapshot: noise-free E/Var, per-qubit
sx/sz, per-vertex <A_v>, per-plaquette <B~_p>/<B_p>, and — with --ed_vectors —
fidelity + weighted sign/phase match against the matching dense-ED ground
state (analysis/scripts/ed_electric_line.py --bc OBC output). Runs locally
(no GPU needed at this size); local Mac is fine per CLAUDE.md (L=2 OBC, N=12).

    python analysis/scripts/eval_snapshots.py --dir results/fermionic_obc_L2 \
        --glob 'signhead_L2_OBC_hx0.0_hz0.0*.json' --exact \
        --ed_vectors results/fermionic_obc_L2/ed_vectors
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import jax.numpy as jnp

from tc3d.builders import build_state
from tc3d.io import load_weights
from tc3d.validation import build_eval_operators, pooled_final_observables
from tc3d.exact_diag import (hamiltonian_linop, expect_x_string, expect_z_string,
                             expect_xz_string, qubits_to_mask)

SNAPSHOT_RE = re.compile(r"\.step(\d+)\.mpack$")


# =============================================================================
# --exact: full-Hilbert-space contraction (small N only)
# =============================================================================

def exact_psi(vs, N, chunk=1024):
    """psi = exp(logpsi) over all 2^N computational-basis configs, normalized.

    Basis index i -> qubit bit i, spin_i = 1 - 2*bit_i(idx) -- the SAME
    convention as tc3d.exact_diag (z_string_eigvals: (basis>>i)&1) and
    analysis/scripts/prefit_phase_head.py's bits_r/Xr construction (line
    ~269: `bits_r = [(s >> i) & 1 for i in range(geo.N)]`, `Xr = 1-2*bits_r`).
    Chunked through vs._apply_fun so the (2^N, N) config array is never
    materialized as a JAX batch larger than `chunk` rows.
    """
    dim = 1 << N
    idx = np.arange(dim, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float64)
    X = 1.0 - 2.0 * bits
    variables = dict(vs.variables)
    logpsi = np.empty(dim, dtype=np.complex128)
    for i in range(0, dim, chunk):
        block = jnp.asarray(X[i:i + chunk])
        logpsi[i:i + chunk] = np.asarray(vs._apply_fun(variables, block))
    psi = np.exp(logpsi)
    norm = float(np.sqrt(np.sum(np.abs(psi) ** 2)))
    if not np.isfinite(norm) or norm == 0.0:
        raise FloatingPointError(f"exact psi failed to normalize (norm={norm}); N={N}")
    return psi / norm


def _ed_tag(cfg):
    """Filename tag matching analysis/scripts/ed_electric_line.py's --bc OBC output:
    gs_L{L}_{bc}_hx{hx}_hz{hz}.npz / exact_diag_{model}_L{L}_{bc}_hx{hx}_hz{hz}.json."""
    return (f"L{cfg['L']}_{cfg.get('bc', 'PBC')}_"
            f"hx{float(cfg.get('hx', 0.0))}_hz{float(cfg.get('hz', 0.0))}")


def find_ed_npz(ed_vectors_dir, cfg):
    path = os.path.join(ed_vectors_dir, f"gs_{_ed_tag(cfg)}.npz")
    return path if os.path.exists(path) else None


def find_ed_json(ed_vectors_dir, cfg):
    """The per-point exact_diag_*.json sits one directory up from ed_vectors/
    (see ed_electric_line.py: --out_dir holds it, --out_dir/ed_vectors holds the npz)."""
    parent = os.path.dirname(os.path.normpath(ed_vectors_dir))
    prefix = cfg.get("model", "fermionic")
    path = os.path.join(parent, f"exact_diag_{prefix}_{_ed_tag(cfg)}.json")
    return path if os.path.exists(path) else None


def _ed_startup_gate(psi_ed, H, json_path, tol=1e-9):
    """SELF-CHECK GATE: <psi_ed|H|psi_ed> must equal the E0 banked in the
    matching exact_diag_*.json to `tol` -- proves the basis/spin convention
    this module uses (exact_psi) lines up with the ED reference generator's
    (ed_electric_line.py's `hamiltonian_linop` + npz `basis_convention` field).
    A mismatch here means OUR convention is wrong (H is built the same way
    the ED script built it), never the physics."""
    with open(json_path) as f:
        ref = json.load(f)
    E_ed = float(np.real(np.vdot(psi_ed, H.matvec(psi_ed))))
    delta = abs(E_ed - float(ref["E0"]))
    ok = delta < tol
    print(f"[exact gate] {'ok  ' if ok else 'FAIL'} <psi_ed|H|psi_ed>={E_ed:.12f}  "
          f"ref E0={ref['E0']:.12f}  delta={delta:.3e}  ({os.path.basename(json_path)})",
          flush=True)
    if not ok:
        raise AssertionError(
            f"[exact gate] <psi_ed|H|psi_ed>={E_ed:.12f} != ref E0={ref['E0']:.12f} "
            f"(delta={delta:.3e} >= tol={tol:.1e}) at {json_path} -- basis/spin "
            "convention mismatch in eval_snapshots.py's exact-eval path")
    return E_ed


def prepare_exact_context(geo, cfg, xz_stabs, ed_vectors=None):
    """Per-run, per-(hx,hz) setup shared across all snapshots: builds H once,
    loads + gates the matching ED vector once. Aborts if geo.N > 20 (the
    --exact contract: full enumeration is 2^N, only tractable for small N)."""
    N = geo.N
    if N > 20:
        raise SystemExit(
            f"--exact requires geo.N <= 20 (got N={N}, dim=2^{N} states); "
            "this run is too large for full-Hilbert-space contraction")
    hx = float(cfg.get("hx", 0.0))
    hz = float(cfg.get("hz", 0.0))
    J = float(cfg.get("J", 1.0))
    H, basis = hamiltonian_linop(geo, hx=hx, hz=hz, J=J, xz_stabs=xz_stabs,
                                 dtype=np.complex128)
    ctx = {"geo": geo, "H": H, "basis": basis, "xz_stabs": xz_stabs}
    if ed_vectors:
        npz_path = find_ed_npz(ed_vectors, cfg)
        if npz_path is None:
            print(f"[exact] no ED npz match under {ed_vectors} for tag="
                  f"{_ed_tag(cfg)} -- fidelity/sign-match will be skipped", flush=True)
            return ctx
        d = np.load(npz_path)
        psi_ed = np.asarray(d["psi"], dtype=np.complex128)
        if psi_ed.shape[0] != basis.shape[0]:
            raise ValueError(f"ED vector dim {psi_ed.shape[0]} != 2^N={basis.shape[0]} "
                             f"at {npz_path}")
        json_path = find_ed_json(ed_vectors, cfg)
        if json_path is not None:
            _ed_startup_gate(psi_ed, H, json_path)
        else:
            print(f"[exact gate] SKIP <psi_ed|H|psi_ed> check -- no matching "
                  f"exact_diag_*.json next to {npz_path}", flush=True)
        ctx["psi_ed"] = psi_ed
        ctx["ed_npz"] = npz_path
    return ctx


def exact_observables(vs, ctx, chunk=1024):
    """Exact per-snapshot metrics on the full Hilbert space: E, Var, Vscore,
    per-qubit sx/sz (+ means), per-vertex <A_v> (+ mean), per-plaquette
    <B~_p>/<B_p> (+ mean); plus fidelity + weighted sign/phase match vs the
    ED ground state when `ctx` carries one (see prepare_exact_context)."""
    geo, H, basis, xz_stabs = ctx["geo"], ctx["H"], ctx["basis"], ctx["xz_stabs"]
    N = geo.N
    psi = exact_psi(vs, N, chunk=chunk)

    Hpsi = H.matvec(psi)
    E0 = float(np.real(np.vdot(psi, Hpsi)))
    E_var = float(np.real(np.vdot(Hpsi, Hpsi))) - E0 ** 2

    sx = [expect_x_string(psi, basis, 1 << i) for i in range(N)]
    sz = [expect_z_string(psi, basis, 1 << i, N) for i in range(N)]
    A_v = [expect_x_string(psi, basis, qubits_to_mask(v)) for v in geo.vertex_all]
    if xz_stabs:                                            # fermionic: B~_p (Z*X)
        B_p = [expect_xz_string(psi, basis, qubits_to_mask(z), qubits_to_mask(x), N)
               for z, x, _ in xz_stabs]
    else:                                                    # bosonic: B_p (Z-only)
        B_p = [expect_z_string(psi, basis, qubits_to_mask(p), N) for p in geo.plaq_all]

    out = {
        "N": N, "dim": int(basis.shape[0]),
        "E0": E0, "E_var": E_var,
        "Vscore": N * E_var / E0 ** 2 if E0 != 0 else float("nan"),
        "sx_mean": float(np.mean(sx)), "sx_per_qubit": sx,
        "sz_mean": float(np.mean(sz)), "sz_per_qubit": sz,
        "A_v_mean": float(np.mean(A_v)), "A_v_per_vertex": A_v,
        "B_p_mean": float(np.mean(B_p)), "B_p_per_plaq": B_p,
    }

    psi_ed = ctx.get("psi_ed")
    if psi_ed is not None:
        fidelity = float(np.abs(np.vdot(psi_ed, psi)) ** 2)
        # align the global phase on the config where |psi_ed * psi_nqs| peaks,
        # then the ED-weighted fraction with matching phase (cos(Delta) > 0)
        weight = np.abs(psi_ed * psi)
        k = int(np.argmax(weight))
        phase_offset = np.angle(psi[k]) - np.angle(psi_ed[k])
        delta = np.angle(psi) - np.angle(psi_ed) - phase_offset
        sign_match = float(np.sum(np.abs(psi_ed) ** 2 * (np.cos(delta) > 0)))
        out["ed_match"] = os.path.basename(ctx["ed_npz"])
        out["fidelity"] = fidelity
        out["sign_match_weighted"] = sign_match
    return out


def eval_run_snapshots(json_path, rounds, seed=None, exact=False, exact_only=False,
                       ed_vectors=None, exact_chunk=1024):
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
    eval_ops = None if exact_only else build_eval_operators(hi, geo, cfg, xz_stabs=xz_stabs)
    exact_ctx = prepare_exact_context(geo, cfg, xz_stabs, ed_vectors) if exact else None

    series = []
    for step, mpack_path in snaps:
        vs = load_weights(vs, mpack_path[:-len(".mpack")])
        obs = ({} if exact_only else
               pooled_final_observables(vs, Ham, geo, cfg, xz_stabs=xz_stabs,
                                        rounds=rounds, eval_ops=eval_ops))
        if not exact_only:
            print(f"[eval_snapshots] step {step:4d}: E0={obs['E0']:+.4f}  "
                  f"Vscore={obs['Vscore']:.2e}", flush=True)
        if exact:
            ex = exact_observables(vs, exact_ctx, chunk=exact_chunk)
            obs["exact"] = ex
            tail = (f"  fidelity={ex['fidelity']:.4f}  "
                    f"sign_match={ex['sign_match_weighted']:.6f}"
                    if "sign_match_weighted" in ex else "")
            print(f"[eval_snapshots] step {step:4d} [exact]: E0={ex['E0']:+.6f}  "
                  f"Vscore={ex['Vscore']:.2e}{tail}", flush=True)
        obs["step"] = step
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
    ap.add_argument("--out_suffix", default=".snapshots.json")
    ap.add_argument("--exact", action="store_true",
                    help="also contract the NQS over the FULL 2^N Hilbert space per "
                         "snapshot (requires geo.N <= 20); adds an 'exact' sub-dict "
                         "with noise-free E/Var/Vscore, per-qubit/vertex/plaquette "
                         "observables, and (with --ed_vectors) fidelity + weighted "
                         "sign/phase match vs the dense-ED ground state")
    ap.add_argument("--exact_only", action="store_true",
                    help="skip the MC-sampled pooled_final_observables path -- only "
                         "the 'exact' sub-dict is computed per snapshot (requires --exact)")
    ap.add_argument("--ed_vectors", default=None,
                    help="dir of ED ground-state npz files, e.g. "
                         "results/fermionic_obc_L2/ed_vectors (analysis/scripts/"
                         "ed_electric_line.py --bc OBC output); matched to each run's "
                         "config (L, bc, hx, hz)")
    ap.add_argument("--exact_chunk", type=int, default=1024,
                    help="forward-pass chunk size for the full 2^N enumeration")
    args = ap.parse_args()
    if args.exact_only and not args.exact:
        raise SystemExit("--exact_only requires --exact")

    for json_path in sorted(glob.glob(os.path.join(args.dir, args.glob))):
        if json_path.endswith(args.out_suffix) or json_path.endswith(".curve.json"):
            continue
        print(f"[eval_snapshots] {json_path}", flush=True)
        result = eval_run_snapshots(json_path, args.rounds, seed=args.seed,
                                    exact=args.exact, exact_only=args.exact_only,
                                    ed_vectors=args.ed_vectors,
                                    exact_chunk=args.exact_chunk)
        out_path = json_path[:-len(".json")] + args.out_suffix
        with open(out_path, "w") as f:
            from tc3d.fm import _json_nonfinite_safe
            json.dump(_json_nonfinite_safe(result), f, indent=2)
        print(f"[eval_snapshots] wrote {out_path}", flush=True)
