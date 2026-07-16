"""
Three_TC/renyi.py
─────────────────────────────────────────────────────────────────────────────
Second-Rényi-entropy S₂(A) = −ln Tr(ρ_A²) of a small fixed central patch, from
*trained* NQS checkpoints for the 3D toric code — an INDEPENDENT, local
transition locator to cross-check the Fredenhagen–Marcu pipeline (`fm.py`).

S₂ of a 4-qubit patch is a **local, susceptibility-like transition locator**: its
derivative dS₂/dh_z peaks at the pseudo-critical h_c(L). It is NOT a topological /
long-range-entanglement / TEE diagnostic — the patch is far too small to see the
constant topological term. Following arXiv:2405.17541 (4-qubit central square in 2D).

Pipeline (mirrors fm.py; one fixed L at a time, stack over L for FSS):

    checkpoints {name}.mpack + {name}.json
       │  fm.iter_matching_checkpoints  +  fm.load_vstate
       ▼
    renyi_sweep(dir, L, hx, field="hz")  → table  field, S₂ ± err  (+ per-plane)
       │  per checkpoint: vs.reset(); average Renyi2 over the 3 central-plaquette
       │  orientations (xy/xz/yz) on the same wavefunction
       ▼
    extract_s2_curve(...)  → JSON (raw S₂ curve; peak extraction is done in the
                             analysis notebook, analysis/s2_crossing.ipynb)

Patch = one **central unit plaquette** (4 coplanar edges = a single B_p), built by
`fm._bulk_square(geo, plane_axis, R=1)` → `fm.electric_loop_edges`. It is centered
and strictly interior for L>=4 (R=1 ≤ L−3), so it never touches the OBC surface.

Estimator = `netket.experimental.observable.Renyi2EntanglementEntropy` (SWAP-based
two-replica). NOTE: NetKet 3.16.1 returns S₂ in **bits** (−log₂ Tr ρ²); `_s2_of_state`
converts to **nats** (×ln2) so it matches the anchors/bound and the GF(2) check used
here. exp(S₂) amplifies the SWAP relative error, but a 4-qubit patch has S₂ ≤ 4 ln2 ≈
2.77 nats, so this is benign — no variance-reduction needed.

Exactly-solvable limits (checked locally, no ED — see `verify_s2_geometry`):
  • h_z = 0  (stabilizer ground state):  S₂ = 3 ln2  (only the patch's own B_p is a
    stabilizer supported entirely in A);
  • h_z → ∞  (trivial product state):    S₂ = 0.

Never run 3D ED/sweeps locally (CLAUDE.md): the S₂ *curves* run on the cluster GPU
node where the checkpoints live; only the GF(2) geometry unit test runs on the dev box.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import netket as nk
from netket.experimental.observable import Renyi2EntanglementEntropy

from Three_TC.fm import (
    PLANE_NORMAL, _bulk_square, electric_loop_edges,
    load_vstate, _struct_sig, _load_weights, _stat_err, _num,
    iter_matching_checkpoints,
)

LN2 = float(np.log(2.0))
# NetKet 3.16.1's Renyi2EntanglementEntropy returns S₂ = −log₂ Tr(ρ²) in BITS (verified
# empirically + in netket/experimental/observable/renyi2/expect.py: `-jnp.log2(...)`).
# Everything else here — the anchors below, the S₂ ≤ 4 ln2 bound, the GF(2) geometry
# check — is in NATS. `_s2_of_state` multiplies the estimator by this factor to convert
# bits → nats, so all reported/compared S₂ values share the nats convention.
BITS_TO_NATS = LN2
S2_EXACT_HZ0 = 3.0 * LN2      # central-plaquette patch on the stabilizer ground state (nats)
S2_EXACT_HZINF = 0.0          # trivial product state (all σ^z = +1)


# =============================================================================
# Patch geometry: the central unit plaquette (4 coplanar edges = one B_p)
# =============================================================================

def center_plaquette_edges(geo, plane_axis: int) -> List[int]:
    """The 4 edge-qubit indices of the bulk-centered unit plaquette with normal
    `plane_axis` (0/1/2 = yz/xz/xy). Reuses `fm._bulk_square(..., R=1)` so the patch
    is centered (corner ((L-2)//2,(L-2)//2), plane L//2) and strictly interior for
    L>=4. Returns them sorted."""
    kw = _bulk_square(geo, plane_axis, R=1)
    closed, _open = electric_loop_edges(geo, **kw)   # closed = the 4 sides of the R=1 square
    return sorted(int(e) for e in closed)


def patch_partitions(geo, planes: Sequence[str] = ("xy", "xz", "yz")
                     ) -> Dict[str, np.ndarray]:
    """{plane label -> np.array of the 4 patch edge indices} for the given orientations."""
    return {pl: np.array(center_plaquette_edges(geo, PLANE_NORMAL[pl]), dtype=int)
            for pl in planes}


# =============================================================================
# Exactly-solvable h_z=0 limit: stabilizer-state entropy via GF(2) (no ED)
# =============================================================================

def _gf2_rank(M) -> int:
    """Rank over GF(2) of a binary matrix (Gaussian elimination mod 2)."""
    M = (np.asarray(M).astype(np.uint8) & 1).copy()
    if M.size == 0:
        return 0
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        mask = M[:, c].astype(bool).copy()
        mask[r] = False
        M[mask] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def s2_stabilizer_exact(geo, A_edges: Sequence[int]) -> Dict[str, Any]:
    """Exact S₂(A) of the toric-code stabilizer ground state for region A (edge set).

    For a stabilizer state the entropy is flat-spectrum, so S₂ = S_vN =
    (|A| − |S_A|)·ln2, where |S_A| = number of independent stabilizers supported
    ENTIRELY in A. In symplectic (X|Z) form over the generators
    {A_v (X-type), B_p (Z-type)}, |S_A| = rank(G) − rank(G_B) (elements vanishing on
    the B columns), so S₂ = (|A| − (rank(G) − rank(G_B)))·ln2.

    Returns a dict (S₂ in nats + bits, ranks, |S_A|) — used by the geometry unit test.
    """
    N = int(geo.N)
    A = {int(e) for e in A_edges}
    B = [q for q in range(N) if q not in A]
    gens: List[np.ndarray] = []
    for v in geo.get_vertex_all_hetero():            # A_v — X-type
        row = np.zeros(2 * N, np.uint8)
        row[np.asarray(v, dtype=int)] = 1
        gens.append(row)
    for p in geo.plaq_all:                            # B_p — Z-type
        row = np.zeros(2 * N, np.uint8)
        row[N + np.asarray(p, dtype=int)] = 1
        gens.append(row)
    G = np.array(gens, np.uint8)
    bcols = [q for q in B] + [N + q for q in B]
    rank_G = _gf2_rank(G)
    rank_GB = _gf2_rank(G[:, bcols])
    n_in_A = rank_G - rank_GB                          # |S_A| independent stabilizers in A
    s2_bits = len(A) - n_in_A
    return {
        "N": N, "n_gens": len(gens), "N_A": len(A),
        "rank_G": rank_G, "rank_GB": rank_GB, "n_stab_in_A": n_in_A,
        "S2_bits": s2_bits, "S2_nats": s2_bits * LN2,
    }


def verify_s2_geometry(geo, planes: Sequence[str] = ("xy", "xz", "yz")
                       ) -> Dict[str, Any]:
    """Local (no-ED) unit test of the patch geometry + the exactly-solvable h_z=0 limit.

    Per plane: the 4 patch edges + their midpoint coords, N_A, whether the patch is
    strictly interior (no coord on an OBC face), and the exact stabilizer S₂. `ok` is
    True iff every plane is interior, has N_A=4, and gives S₂ = 3 ln2. Runs in the repo
    venv from `geo` alone (GF(2) linear algebra, no state vector)."""
    L = (geo.Lx, geo.Ly, geo.Lz)
    parts = patch_partitions(geo, planes)
    per: Dict[str, Any] = {}
    ok = True
    for pl, edges in parts.items():
        coords = [np.asarray(geo.arr_coord[int(e)], float).tolist() for e in edges]
        # interior: every coordinate strictly inside (0, L_axis-1) — off every OBC face
        interior = all(all(0.0 < c[ax] < L[ax] - 1 for ax in range(3)) for c in coords)
        ex = s2_stabilizer_exact(geo, edges)
        plane_ok = interior and ex["N_A"] == 4 and abs(ex["S2_nats"] - S2_EXACT_HZ0) < 1e-9
        ok &= plane_ok
        per[pl] = {"edges": [int(e) for e in edges], "coords": coords,
                   "interior": bool(interior), **ex, "ok": bool(plane_ok)}
    return {"L": L, "planes": list(planes), "S2_exact_hz0": S2_EXACT_HZ0,
            "S2_exact_hzinf": S2_EXACT_HZINF, "per_plane": per, "ok": bool(ok)}


# =============================================================================
# S₂ estimator over a checkpoint sweep (mirrors fm.fm_sweep)
# =============================================================================

def _build_renyi_obs(hi, parts: Dict[str, np.ndarray]
                     ) -> List[Tuple[str, Any]]:
    """One Renyi2EntanglementEntropy per orientation, labelled by plane."""
    return [(pl, Renyi2EntanglementEntropy(hi, partition=A)) for pl, A in parts.items()]


def _s2_of_state(vs, obs: List[Tuple[str, Any]]
                 ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """Average S₂ over the plane observables on the CURRENT samples of `vs`.

    Returns (S2, S2_err, per_plane). Per-plane errors are combined in quadrature and
    divided by n_planes (SE of the mean, treating orientations as independent — a mild
    approximation, since the three read off correlated samples; the authoritative h_c
    error is the notebook's peak-bootstrap over the S₂ curve, not this per-point bar).
    `_stat_err` falls back to √(var/n) when NetKet's autocorrelation error is NaN
    (short chains) — pair with few long chains (eval_chains≈16), as in fm.py.
    """
    n = int(getattr(vs, "n_samples", 0) or 0)
    per: Dict[str, Tuple[float, float]] = {}
    for pl, op in obs:
        st = vs.expect(op)
        # bits (NetKet) -> nats (this module's convention); the factor is exact and
        # linear, so it scales the error identically. See BITS_TO_NATS note above.
        per[pl] = (BITS_TO_NATS * float(np.real(st.mean)),
                   BITS_TO_NATS * _stat_err(st, n))
    vals = np.array([per[pl][0] for pl, _ in obs], float)
    errs = np.array([per[pl][1] for pl, _ in obs], float)
    s2 = float(vals.mean())
    s2e = float(np.sqrt(np.sum(errs ** 2)) / len(errs))
    return s2, s2e, per


def renyi_sweep(checkpoint_dir: str, *, field: str = "hz",
                L: Optional[int] = None, hx: Optional[float] = None,
                model: str = "bosonic", bc: Optional[str] = None,
                eval_samples: int = 8192, eval_chains: Optional[int] = None,
                planes: Sequence[str] = ("xy", "xz", "yz"),
                verbose: bool = True) -> Dict[str, np.ndarray]:
    """Score every matching checkpoint in `checkpoint_dir` for S₂(central patch), sorted
    by `field` (default "hz"). Mirrors `fm.fm_sweep`: shares the network/sampler/observables
    across the sweep (only the weights change per checkpoint) so JAX's compiled `expect`
    stays warm; a structural-config change forces a one-off rebuild.

    Returns a dict of equal-length arrays: field, S2, S2e, name, plus S2_<plane>/S2e_<plane>
    per orientation and a non-array "_meta" (planes, patch edges per plane, exact limits).
    """
    tmpl_sig = tmpl = None          # (geo, hi, vs, obs, meta)
    sweep_meta: Dict[str, Any] = {}
    rows = []
    for jp, cfg0, _doc in iter_matching_checkpoints(
            checkpoint_dir, L=L, hx=hx, model=model, bc=bc, verbose=verbose):
        t0 = time.perf_counter()
        sig = _struct_sig(cfg0)
        if tmpl is None or sig != tmpl_sig:            # first match, or a shape change
            _cfg, geo, hi, vs = load_vstate(jp, eval_samples=eval_samples,
                                            eval_chains=eval_chains)
            parts = patch_partitions(geo, planes)
            obs = _build_renyi_obs(hi, parts)
            vs, fallback = _ensure_renyi_sampler(vs, hi, obs[0][1], verbose=verbose)
            sweep_meta = {
                "planes": list(planes),
                "patch_edges": {pl: [int(x) for x in A] for pl, A in parts.items()},
                "S2_exact_hz0": S2_EXACT_HZ0, "S2_exact_hzinf": S2_EXACT_HZINF,
                "sampler_fallback": fallback,
            }
            tmpl_sig, tmpl = sig, (geo, hi, vs, obs, sweep_meta)
        else:                                          # reuse: swap weights only
            geo, hi, _vs, obs, sweep_meta = tmpl
            vs = _load_weights(_vs, jp)
        vs.reset()                                     # fresh samples for these weights
        S2, S2e, per = _s2_of_state(vs, obs)
        row = {"field": float(cfg0[field]), "S2": S2, "S2e": S2e,
               "name": cfg0.get("name", os.path.basename(jp)[:-5])}
        for pl, (v, e) in per.items():
            row[f"S2_{pl}"], row[f"S2e_{pl}"] = v, e
        rows.append(row)
        if verbose:
            spread = ("  planes={" + ", ".join(f"{pl}:{per[pl][0]:.3f}" for pl in per)
                      + "}" if len(per) > 1 else "")
            print(f"  {row['name']}: {field}={row['field']:.4g}  "
                  f"S2={S2:.4f}±{S2e:.4f} (3ln2={S2_EXACT_HZ0:.4f}){spread}  "
                  f"[{time.perf_counter() - t0:.1f}s]", flush=True)
    if not rows:
        raise ValueError(f"no checkpoints in {checkpoint_dir} match "
                         f"(L={L}, hx={hx}, model={model}, bc={bc})")
    rows.sort(key=lambda r: r["field"])
    out = {k: np.array([r[k] for r in rows],
                       dtype=object if k == "name" else float)
           for k in rows[0]}
    out["_meta"] = sweep_meta
    return out


def _ensure_renyi_sampler(vs, hi, obs0, *, verbose: bool = True):
    """Confirm the Renyi2 SWAP estimator runs with the checkpoint's own sampler; if not,
    rebuild `vs` on a plain MetropolisLocal (weights are sampler-shape-independent, per
    `load_vstate`). Returns (vs, fallback_used)."""
    try:
        _ = vs.expect(obs0)
        return vs, False
    except Exception as e:                             # noqa: BLE001 — surface, then fall back
        if verbose:
            print(f"  [renyi] estimator failed on the run's sampler "
                  f"({type(e).__name__}: {e}); rebuilding on MetropolisLocal", flush=True)
        n_chains = int(getattr(vs.sampler, "n_chains", 16) or 16)
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
        vs2 = nk.vqs.MCState(sampler, vs.model, n_samples=int(vs.n_samples))
        vs2.variables = vs.variables
        return vs2, True


# =============================================================================
# Per-L extraction → JSON (peak extraction lives in the notebook)
# =============================================================================

def _centered_fd(h: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float]]:
    """Convenience centered finite-difference dy/dh on the (sorted) grid midpoints."""
    h = np.asarray(h, float); y = np.asarray(y, float)
    hm = 0.5 * (h[1:] + h[:-1])
    dy = np.diff(y) / np.diff(h)
    return hm.tolist(), dy.tolist()


def extract_s2_curve(checkpoint_dir, *, L, hx, field="hz", model="bosonic", bc="OBC",
                     eval_samples=8192, eval_chains=None,
                     planes=("xy", "xz", "yz")) -> Dict[str, Any]:
    """renyi_sweep for one L → a JSON-serializable dict (raw S₂ curve, no logistic fit —
    the notebook does the smoothing-spline + finite-difference peak extraction)."""
    res = renyi_sweep(checkpoint_dir, field=field, L=L, hx=hx, model=model, bc=bc,
                      eval_samples=eval_samples, eval_chains=eval_chains, planes=planes)
    meta = res.get("_meta", {})
    hm, dS2 = _centered_fd(res["field"], res["S2"])
    rec = {
        "L": int(L), "hx": _num(hx), "field_name": field, "bc": bc, "model": model,
        "eval_samples": int(eval_samples), "patch": "center_plaquette", "N_A": 4,
        "planes": meta.get("planes", list(planes)),
        "patch_edges": meta.get("patch_edges", {}),
        "s2_exact_hz0": _num(meta.get("S2_exact_hz0", S2_EXACT_HZ0)),
        "s2_exact_hzinf": _num(meta.get("S2_exact_hzinf", S2_EXACT_HZINF)),
        "sampler_fallback": bool(meta.get("sampler_fallback", False)),
        "field": res["field"].tolist(), "S2": res["S2"].tolist(),
        "S2e": res["S2e"].tolist(), "names": [str(x) for x in res["name"]],
        "fd": {"h_mid": hm, "dS2dh": dS2},
    }
    s2_planes = {pl: res[f"S2_{pl}"].tolist() for pl in meta.get("planes", [])
                 if f"S2_{pl}" in res}
    if s2_planes:
        rec["S2_planes"] = s2_planes
        rec["S2e_planes"] = {pl: res[f"S2e_{pl}"].tolist() for pl in s2_planes}
    return rec


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Extract S2(field) of the central-plaquette patch for one L.")
    p.add_argument("--dir", required=True, help="checkpoint dir ({name}.json + .mpack)")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--hx", type=float, default=None,
                   help="fix hx (hz-sweep filters to this cut). OMIT for an hx-sweep "
                        "(--field hx): matches ALL hx in the dir.")
    p.add_argument("--field", default="hz", help="swept parameter: 'hz' or 'hx'")
    p.add_argument("--bc", default="OBC", choices=["OBC", "PBC"])
    p.add_argument("--model", default="bosonic", choices=["bosonic", "fermionic"])
    p.add_argument("--eval_samples", type=int, default=8192)
    p.add_argument("--eval_chains", type=int, default=16,
                   help="override n_chains at eval (default 16 = long chains → valid "
                        "error_of_mean; GPU runs saved 1024 → ~8 samples/chain → NaN).")
    p.add_argument("--planes", default="xy,xz,yz",
                   help="central-plaquette orientations to average (same wavefunction)")
    p.add_argument("--out", required=True, help="output JSON path")
    a = p.parse_args(argv)
    planes = tuple(s.strip() for s in a.planes.split(",") if s.strip())
    rec = extract_s2_curve(a.dir, L=a.L, hx=a.hx, field=a.field, model=a.model, bc=a.bc,
                           eval_samples=a.eval_samples, eval_chains=a.eval_chains,
                           planes=planes)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    fb = "  [sampler_fallback]" if rec.get("sampler_fallback") else ""
    print(f"[s2] L={a.L} hx={a.hx} planes={rec['planes']}: {len(rec['field'])} points "
          f"(3ln2={S2_EXACT_HZ0:.4f}){fb}  ->  {a.out}")


if __name__ == "__main__":
    main()
