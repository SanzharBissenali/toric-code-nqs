"""Locality/stencil compression of the fTC h=0 sign form (plan item 2).

Instead of gauge-fixing the raw GF(2) solution of prefit_phase_head.py, re-solve
the same class equations in a TRANSLATION-INVARIANT stencil basis: one unknown
per plaquette orientation (linear) and one per (orient_i, orient_j, displacement)
class within a cutoff (quadratic). Feasibility (residual 0 + fresh-sample
certificate) proves a local TI representative of the sign form exists.

The deploy test: the stencil solved at --solve sizes is applied VERBATIM at
--deploy sizes (no solving there) and certified on fresh sampled classes.
One size alone is NOT enough: its class space leaves gauge freedom that the
solver fixes size-specifically (range-1 tokens pass at L=3, coin-flip at L=4/5).

--vars picks the variables the form acts on: plaquette flux tokens (nonlocal
universally: t = Mx observes the analytically-local application form q(x) =
sum C_pq x_p x_q only through a linear map whose inverse is nonlocal) or raw
edge bits (Dehaene-De Moor guarantees SOME bit-quadratic form; locality is the
question under test).

Usage:
  python analysis/scripts/stencil_phase_head.py --solve 3 4 --cutoff 1.0 --deploy 5 --vars bits
"""
import argparse
import itertools
import json

import numpy as np

from tc3d.geometry import ThreeD_ToricCodeGeometry
from tc3d.fermionic_decoration import fermionic_plaquettes, _mask, _gf2_solve

_E = np.eye(3)


def orient_classes(period2):
    """Universal (size-independent) orientation-class list; index = class id.

    period2 enriches the class with the unit-cell parity (the 'spin structure'
    ansatz): translations are broken down to the even sublattice, so coefficients
    may depend on axis AND cell parity. Even L only (parity wraps inconsistently
    on odd tori)."""
    if not period2:
        return [(a,) for a in range(3)]
    return sorted((a, (px, py, pz)) for a in range(3)
                  for px in (0, 1) for py in (0, 1) for pz in (0, 1))


def build(L, variables="tokens", period2=False):
    """Stabilizer masks + per-variable (orientation-class, 2x-coordinate) metadata.

    variables='tokens': one variable per plaquette flux token (orientation =
    plaquette normal axis). 'bits': one variable per edge spin bit (orientation
    = edge axis, the half-integer coordinate component)."""
    geo = ThreeD_ToricCodeGeometry(L, L, L, bc="PBC")
    stabs = fermionic_plaquettes(geo)
    zxm = [(_mask(z), _mask(x)) for z, x, _ in stabs]
    stars = [_mask(v) for v in geo.vertex_all]
    cls_ix = {c: i for i, c in enumerate(orient_classes(period2))}

    def label(axis, coord2):
        if not period2:
            return cls_ix[(axis,)]
        return cls_ix[(axis, tuple((v >> 1) & 1 for v in coord2))]

    if variables == "tokens":
        meta = []                                # same loop order as fermionic_plaquettes
        for c in range(3):
            a, b = (d for d in range(3) if d != c)
            for ix in range(L):
                for iy in range(L):
                    for iz in range(L):
                        ctr2 = tuple(int(v) for v in
                                     (2 * (np.array([ix, iy, iz], float)
                                           + 0.5 * _E[a] + 0.5 * _E[b])).round())
                        meta.append((label(c, ctr2), ctr2))
        assert len(meta) == len(zxm)
    else:                                        # bits: invert the coord->idx map
        meta = [None] * geo.N
        for coord2, i in geo._coord_to_idx.items():
            axis = next(d for d in range(3) if coord2[d] & 1)
            c2 = tuple(int(v) for v in coord2)
            meta[int(i)] = (label(axis, c2), c2)
        assert all(m is not None for m in meta)
    return geo, zxm, stars, meta


def draw_class(rng, zxm, stars=None):
    """Random product of B~_p on |0..0>, sign tracked (see prefit_phase_head).

    With `stars` given, also XOR in a random star subset: signs are invariant,
    so this forces any solved BIT-variable form to be star-gauge-blind."""
    s, sg = 0, 1
    for k in np.nonzero(rng.integers(0, 2, size=len(zxm)))[0]:
        zb, xb = zxm[k]
        if bin(zb & s).count("1") & 1:
            sg = -sg
        s ^= xb
    if stars is not None:
        for k in np.nonzero(rng.integers(0, 2, size=len(stars)))[0]:
            s ^= stars[k]
    return s, sg


def token_vec(s, zxm):
    return np.array([bin(s & zb).count("1") & 1 for zb, _ in zxm], dtype=np.int8)


def bit_vec(s, N):
    return np.array([(s >> i) & 1 for i in range(N)], dtype=np.int8)


def disp2(meta, p, q, L):
    """Signed min-image displacement (2x units, components in (-L, L])."""
    d = [(meta[q][1][i] - meta[p][1][i]) % (2 * L) for i in range(3)]
    return tuple(v - 2 * L if v > L else v for v in d)


def pair_key(meta, p, q, L):
    """Canonical unordered stencil key (c_p, c_q, disp) for a plaquette pair."""
    k1 = (meta[p][0], meta[q][0], disp2(meta, p, q, L))
    k2 = (meta[q][0], meta[p][0], disp2(meta, q, p, L))
    return min(k1, k2)


def stencil_pairs(meta, L, cutoff, keys=None):
    """{key: [(p, q), ...]} over pairs within cutoff (lattice units, max-norm).

    With `keys` given, group only pairs whose key is in that set (deploy mode)."""
    out = {}
    for p, q in itertools.combinations(range(len(meta)), 2):
        d = disp2(meta, p, q, L)
        if max(abs(v) for v in d) > 2 * cutoff:
            continue
        k = pair_key(meta, p, q, L)
        if keys is not None and k not in keys:
            continue
        out.setdefault(k, []).append((p, q))
    return out


def features(t, meta, groups, key_ix, n_lin):
    """GF(2) feature bitmask of token vector t in the stencil basis."""
    r = 0
    for c in range(n_lin):
        if int(sum(int(t[p]) for p in range(len(meta)) if meta[p][0] == c)) & 1:
            r |= 1 << c
    for k, pairs in groups.items():
        if sum(int(t[p]) & int(t[q]) for p, q in pairs) & 1:
            r |= 1 << (n_lin + key_ix[k])
    return r


def apply_stencil(t, meta, groups_on, lin_on):
    """Evaluate the solved stencil (only coefficient-1 entries) on tokens t."""
    par = 0
    for c in lin_on:
        par ^= int(sum(int(t[p]) for p in range(len(meta)) if meta[p][0] == c)) & 1
    for pairs in groups_on.values():
        par ^= sum(int(t[p]) & int(t[q]) for p, q in pairs) & 1
    return par


ap = argparse.ArgumentParser()
ap.add_argument("--solve", type=int, nargs="+", default=[3],
                help="sizes whose class equations jointly constrain the stencil "
                     "(one size pins the form only on ITS class space — its gauge "
                     "freedom is fixed size-specifically; add sizes to kill it)")
ap.add_argument("--cutoff", type=float, default=1.0,
                help="stencil range in lattice units (max-norm on displacement)")
ap.add_argument("--train", type=int, default=4000, help="class samples PER solve size")
ap.add_argument("--cert", type=int, default=5000)
ap.add_argument("--deploy", type=int, nargs="*", default=[],
                help="apply the solved stencil verbatim at these held-out sizes")
ap.add_argument("--deploy_cert", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--vars", choices=["tokens", "bits"], default="tokens",
                help="variables the form acts on: plaquette flux tokens, or raw "
                     "edge bits (sampler then adds random star flips so the "
                     "form must be star-gauge-invariant)")
ap.add_argument("--period2", action="store_true",
                help="spin-structure ansatz: coefficients may depend on the "
                     "unit-cell parity (translations broken to the even "
                     "sublattice); even sizes only")
ap.add_argument("--dump", default=None, help="write the solved stencil JSON here")
args = ap.parse_args()

if args.period2 and any(L % 2 for L in list(args.solve) + list(args.deploy)):
    raise SystemExit("--period2 needs even sizes only (parity wraps "
                     "inconsistently on odd tori)")

def vec_of(s, zxm, meta):
    return token_vec(s, zxm) if args.vars == "tokens" else bit_vec(s, len(meta))

# shared unknown space: union of within-cutoff displacement classes over the
# solve sizes (keys are size-independent labels; no intra-support aliasing for
# cutoff < L/2 ... at L=3, cutoff 1 support members never alias mod 3)
sizes = {}
all_keys = set()
for L in sorted(set(args.solve) | set(args.deploy)):
    geo, zxm, stars, meta = build(L, args.vars, args.period2)
    groups = stencil_pairs(meta, L, args.cutoff)
    sizes[L] = (zxm, stars, meta, groups)
    if L in args.solve:
        all_keys |= set(groups)
key_ix = {k: i for i, k in enumerate(sorted(all_keys))}
n_lin = len(orient_classes(args.period2))
ncols = n_lin + len(all_keys)
print(f"solve sizes {sorted(args.solve)} over '{args.vars}'"
      f"{' period2' if args.period2 else ''}: stencil basis = "
      f"{n_lin} linear + {len(all_keys)} displacement classes "
      f"(cutoff {args.cutoff}) = {ncols} unknowns")

rng = np.random.default_rng(args.seed)
star_arg = lambda st: st if args.vars == "bits" else None
rows, targets, basis, contra = [], [], {}, 0
for L in sorted(args.solve):
    zxm, stars, meta, groups = sizes[L]
    for _ in range(args.train):
        s, sg = draw_class(rng, zxm, star_arg(stars))
        row = features(vec_of(s, zxm, meta), meta, groups, key_ix, n_lin)
        k = 0 if sg > 0 else 1
        rows.append(row); targets.append(k)
        aug = row | (k << ncols)
        while aug & ((1 << ncols) - 1):
            low = (aug & ((1 << ncols) - 1)) & -(aug & ((1 << ncols) - 1))
            c = low.bit_length() - 1
            if c not in basis:
                basis[c] = aug; aug = 0; break
            aug ^= basis[c]
        if aug:
            contra += 1
    print(f"  L={L}: +{args.train} class equations, cumulative rank {len(basis)}, "
          f"inconsistent rows {contra}")
if contra:
    raise SystemExit(f"NO translation-invariant stencil of range {args.cutoff} "
                     f"jointly reproduces the sign form at sizes {args.solve}")

sol = _gf2_solve(rows, targets, ncols)
resid = sum(1 for r, k in zip(rows, targets) if (bin(r & sol).count("1") & 1) != k)
lin_on = [c for c in range(n_lin) if (sol >> c) & 1]
on_keys = [k for k in sorted(all_keys) if (sol >> (n_lin + key_ix[k])) & 1]
print(f"solved: residual {resid}; linear coeffs on orientations {lin_on}; "
      f"{len(on_keys)} active displacement classes:")
for k in on_keys:
    print(f"    orient ({k[0]},{k[1]})  2*disp {tuple(k[2])}")

results = {}
for L in sorted(set(args.solve) | set(args.deploy)):
    zxm, stars, meta, groups = sizes[L]
    groups_on = {k: v for k, v in groups.items() if k in set(on_keys)}
    n = args.cert if L in args.solve else args.deploy_cert
    ok = 0
    for _ in range(n):
        s, sg = draw_class(rng, zxm, star_arg(stars))
        ok += int(apply_stencil(vec_of(s, zxm, meta), meta, groups_on, lin_on)
                  == (0 if sg > 0 else 1))
    results[L] = (ok, n)
    tag = "CERTIFICATE" if L in args.solve else "DEPLOY CERTIFICATE (held out)"
    print(f"{tag} L={L}: {ok}/{n} fresh samples correct")

if args.dump:
    with open(args.dump, "w") as f:
        json.dump({"solved_at": sorted(args.solve), "cutoff": args.cutoff,
                   "variables": args.vars, "period2": args.period2,
                   "orient_classes": [list(map(str, c)) for c in
                                      orient_classes(args.period2)],
                   "linear_on_orientations": lin_on,
                   "quad_keys_on": [{"orients": [int(k[0]), int(k[1])],
                                     "disp2": [int(v) for v in k[2]]}
                                    for k in on_keys],
                   "n_unknowns": ncols, "rank": len(basis), "residual": resid,
                   "certificates": {str(L): {"ok": o, "n": n,
                                             "held_out": L not in args.solve}
                                    for L, (o, n) in results.items()}}, f, indent=1)
    print(f"stencil dumped to {args.dump}")
