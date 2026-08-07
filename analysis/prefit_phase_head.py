"""Supervised phase pre-fit of ToricCNN_gridinv on the exact fTC h=0 sign map.

Dataset: exact BFS enumeration of the h=0 support (L=2 only) or sampled
stabilizer classes (--sampled, any L). Signs propagate by psi(s^star)=psi(s),
psi(s^x_p)=eps_p(s) psi(s), eps_p(s)=(-1)^{|dp cap s|}; the sign is a GF(2)
quadratic form over the plaquette flux tokens, solved exactly with
--analytic_head. SGD fit modes (Im log psi, loss 1 - sigma*cos b + amplitude
uniformity, manual Adam on conjugated grads) are kept for the failure-mode
ladder. Modes:
  --init random|trapped   (trapped = plateau checkpoint + symmetry-breaking noise)
  --holdout N             (exclude N token classes from the fit/solve)
  --sampled N             (dataset = random plaquette-subset class samples;
                           required for L>=3 where enumeration is impossible)
"""
import argparse
import json
import time
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp

from tc3d.builders import build_state
from tc3d.io import load_weights, save_model
from tc3d.fermionic_decoration import _mask

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=2)
ap.add_argument("--kernel", type=int, default=None,
                help="CNN kernel_size (default max(2, L-1); must match the "
                     "polish run's for --init_from compatibility)")
ap.add_argument("--init", choices=["random", "trapped"], default="random")
ap.add_argument("--trapped_ckpt", default=None,
                help="checkpoint base for --init trapped")
ap.add_argument("--holdout", type=int, default=0, help="# token classes held out")
ap.add_argument("--sampled", type=int, default=0,
                help="max class samples (0 = full BFS enumeration, L=2 only)")
ap.add_argument("--patience", type=int, default=500,
                help="stop sampling after this many draws without GF(2) rank gain")
ap.add_argument("--cert", type=int, default=10000,
                help="# fresh samples for the post-solve certificate (--sampled)")
ap.add_argument("--xcheck_enum", action="store_true",
                help="after a --sampled solve, cross-check against the full BFS "
                     "enumeration (L=2 only regression)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--amp_w", type=float, default=0.1)
ap.add_argument("--save", default=None, help="save fitted vstate to <base>.mpack")
ap.add_argument("--ckpt_every", type=int, default=100,
                help="also save <base>_s{step}.mpack every N steps")
ap.add_argument("--log", default=None, help="append per-step JSONL history here")
ap.add_argument("--expect", action="store_true", help="MCMC <H> of fitted state")
ap.add_argument("--phase_head", action="store_true",
                help="enable the token-quadratic phase head")
ap.add_argument("--analytic_head", action="store_true",
                help="GF(2)-solve the exact quadratic form from the training "
                     "classes and SET the head analytically (no SGD; implies "
                     "real trunk + head; skips the training loop)")
ap.add_argument("--real_trunk", action="store_true",
                help="zero the imaginary parts of all non-head params at init "
                     "(real trunk -> Im logpsi_trunk == 0 exactly)")
ap.add_argument("--fit_head_only", action="store_true",
                help="freeze the trunk: optimize only phase_lin/phase_quad "
                     "(linear model -> exact quadratic-form identification)")
ap.add_argument("--dump_form", default=None,
                help="dump the solved (l, Q) form to this JSON (--analytic_head)")
args = ap.parse_args()

if args.analytic_head and not args.phase_head:
    raise SystemExit("--analytic_head requires --phase_head")
if args.analytic_head and not args.real_trunk:
    args.real_trunk = True
    print("--analytic_head: forcing --real_trunk (head phases must be exact)")

kernel = args.kernel if args.kernel is not None else max(2, args.L - 1)
cfg = {"L": args.L, "model": "fermionic", "arch": "ToricCNN_gridinv", "bc": "PBC",
       "kernel_size": kernel, "noninv_hidden": [4, 8], "inv_hidden": [8, 8],
       "n_noninv": 2, "noninv_channels": 4, "n_samples": 8192, "n_chains": 16,
       "phase_head": args.phase_head}
geo, hi, Ham, vs, xz = build_state(cfg)

# ---- sign machinery over the stabilizer group -------------------------------
stars = [_mask(v) for v in geo.vertex_all]
zxm = [(_mask(z), _mask(x)) for z, x, _ in xz]
zedges = [z for z, _, _ in xz]
NP = len(zxm)
pair_idx = [(i, j) for i in range(NP) for j in range(i + 1, NP)]
pair_pos = {ij: m for m, ij in enumerate(pair_idx)}
ncols = NP + len(pair_idx)


def token_vec(s):
    """Plaquette flux tokens (z-parities on each dp) of state bitmask s."""
    return tuple(bin(s & zb).count("1") & 1 for zb, _ in zxm)


def gf2_row(tvec):
    """Feature bitmask (linear + upper-triangular pair columns) of a class."""
    on = [i for i, b in enumerate(tvec) if b]
    r = 0
    for i in on:
        r |= 1 << i
    for a in range(len(on)):
        for b in range(a + 1, len(on)):
            r |= 1 << (NP + pair_pos[(on[a], on[b])])
    return r


def draw_class(rng):
    """One uniform-random product of B~_p applied to |0..0>, sign tracked.

    Stars are omitted: every plaquette boundary overlaps a star evenly, so
    stars change neither tokens nor signs; stabilizers commute, so the
    application order is irrelevant."""
    s, sg = 0, 1
    for k in np.nonzero(rng.integers(0, 2, size=NP))[0]:
        zb, xb = zxm[k]
        if bin(zb & s).count("1") & 1:
            sg = -sg
        s ^= xb
    return s, sg


def gf2_absorb(basis, aug):
    """Online GF(2) elimination of an augmented row (coeffs | target<<ncols).

    Returns 'new' (rank +1), 'dep' (consistent), or 'contradiction' (the row
    space forces target parity != observed sign -> the token-quadratic
    hypothesis fails)."""
    coeff_mask = (1 << ncols) - 1
    while aug & coeff_mask:
        low = (aug & coeff_mask) & -(aug & coeff_mask)
        c = low.bit_length() - 1
        if c not in basis:
            basis[c] = aug
            return "new"
        aug ^= basis[c]
    return "dep" if aug == 0 else "contradiction"


def bfs_dataset():
    """Exact enumeration of the h=0 support with BFS signs (int64: L=2 only)."""
    if geo.N > 62:
        raise SystemExit(f"enumeration needs N<=62 bits (got N={geo.N}); "
                         "use --sampled")
    sign = {0: +1}
    q = deque([0])
    while q:
        s = q.popleft()
        for st in stars:
            t = s ^ st
            if t not in sign:
                sign[t] = sign[s]; q.append(t)
        for zb, xb in zxm:
            t = s ^ xb
            if t not in sign:
                sign[t] = sign[s] * (-1 if bin(zb & s).count("1") & 1 else +1)
                q.append(t)
    masks = np.array(list(sign.keys()), dtype=np.int64)
    yv = np.array([sign[int(m)] for m in masks], dtype=np.float64)
    bt = (masks[:, None] >> np.arange(geo.N)[None, :]) & 1
    tk = np.stack([bt[:, ze].sum(axis=1) & 1 for ze in zedges], axis=1)
    return bt, yv, tk


# ---- dataset ----------------------------------------------------------------
if args.sampled:
    rng_s = np.random.default_rng(args.seed)
    basis, seen = {}, {}                      # pivot col -> aug row ; class -> (rep, sign)
    contra = since_new = n_draw = 0
    for n_draw in range(1, args.sampled + 1):
        s, sg = draw_class(rng_s)
        tv = token_vec(s)
        prev = seen.get(tv)
        if prev is not None:
            if prev[1] != sg:
                contra += 1
            since_new += 1
        else:
            seen[tv] = (s, sg)
            res = gf2_absorb(basis, gf2_row(tv) | ((0 if sg > 0 else 1) << ncols))
            if res == "contradiction":
                contra += 1
            since_new = 0 if res == "new" else since_new + 1
        if n_draw % 1000 == 0:
            print(f"  sampled {n_draw}: {len(seen)} classes, rank {len(basis)}, "
                  f"contradictions {contra}", flush=True)
        if since_new >= args.patience:
            break
    print(f"sampling stopped after {n_draw} draws: {len(seen)} classes, "
          f"GF(2) rank {len(basis)} of {ncols} columns, contradictions {contra} "
          f"({'rank SATURATED' if since_new >= args.patience else 'budget exhausted'})")
    if contra:
        raise SystemExit("token-quadratic hypothesis FAILED at this L: the sign "
                         "is not a quadratic function of the flux tokens")
    reps = list(seen.items())
    y = np.array([sg for _, (_, sg) in reps], dtype=np.float64)
    bits = np.array([[(st >> i) & 1 for i in range(geo.N)]
                     for _, (st, _) in reps], dtype=np.int64)
    tokens = np.array([tv for tv, _ in reps], dtype=np.int64)
    class_id = np.arange(len(reps))
    n_class = len(reps)
    print(f"dataset: {n_class} sampled classes (one representative state each), "
          f"{int((y < 0).sum())} negative")
else:
    bits, y, tokens = bfs_dataset()
    class_id = np.unique(tokens, axis=0, return_inverse=True)[1]
    n_class = class_id.max() + 1
    print(f"dataset: {len(bits)} support states, {n_class} token classes, "
          f"{int((y < 0).sum())} negative")

X = (1.0 - 2.0 * bits).astype(np.float64)                    # bit 1 -> spin -1

train_mask = np.ones(len(X), bool)
if args.holdout:
    held = np.random.default_rng(1).choice(n_class, size=args.holdout, replace=False)
    train_mask = ~np.isin(class_id, held)
    print(f"holding out classes {sorted(held.tolist())} "
          f"({(~train_mask).sum()} states)")

# ---- init -------------------------------------------------------------------
if args.init == "trapped":
    if not args.trapped_ckpt:
        raise SystemExit("--init trapped needs --trapped_ckpt")
    vs = load_weights(vs, args.trapped_ckpt)
    rng = jax.random.PRNGKey(7)
    leaves, tree = jax.tree_util.tree_flatten(vs.parameters)
    keys = jax.random.split(rng, len(leaves))
    leaves = [l + 0.02 * jnp.std(jnp.abs(l)) *
              (jax.random.normal(k, l.shape) + 1j * jax.random.normal(k[::-1], l.shape))
              for l, k in zip(leaves, keys)]
    vs.parameters = jax.tree_util.tree_unflatten(tree, leaves)
    print("init: trapped checkpoint + 2% complex noise (symmetry breaking)")
else:
    print("init: random (cold)")

if args.real_trunk:
    params_r = {k: (val if k in ("phase_lin", "phase_quad")
                    else jax.tree_util.tree_map(
                        lambda l: jnp.real(l).astype(l.dtype), val))
                for k, val in vs.parameters.items()}
    # tiny REAL noise on the head breaks the b==0 saddle of the (even-in-b)
    # sign loss while keeping the trunk exactly phase-silent
    kr = jax.random.PRNGKey(11)
    k1, k2 = jax.random.split(kr)
    params_r["phase_lin"] = params_r["phase_lin"] + 0.03 * jax.random.normal(
        k1, params_r["phase_lin"].shape)
    params_r["phase_quad"] = params_r["phase_quad"] + 0.03 * jax.random.normal(
        k2, params_r["phase_quad"].shape)
    vs.parameters = params_r
    print("init: trunk Im zeroed (phase-silent); head seeded with 0.03 real noise")

apply_fun = vs._apply_fun
params0 = vs.parameters

def loss_fn(p, xb, yb):
    lp = apply_fun({"params": p}, xb)
    phase = jnp.mean(1.0 - yb * jnp.cos(jnp.imag(lp)))
    amp = jnp.var(jnp.real(lp))
    return phase + args.amp_w * amp, (phase, amp)

@jax.jit
def acc_fn(p, xb, yb):
    lp = apply_fun({"params": p}, xb)
    return jnp.mean((yb * jnp.cos(jnp.imag(lp))) > 0)

grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

# manual Adam on conj(grad) (Wirtinger descent for real loss of complex params)
m = jax.tree_util.tree_map(jnp.zeros_like, params0)
v = jax.tree_util.tree_map(lambda l: jnp.zeros(l.shape, jnp.float64), params0)

@jax.jit
def adam_step(p, m, v, g, t):
    b1, b2, eps = 0.9, 0.999, 1e-8
    gc = jax.tree_util.tree_map(jnp.conj, g)
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, gc)
    v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * jnp.abs(b) ** 2, v, gc)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
    p = jax.tree_util.tree_map(lambda a, b, c: a - args.lr * b / (jnp.sqrt(c) + eps),
                               p, mh, vh)
    return p, m, v

loss_eval = jax.jit(lambda p, xb, yb: loss_fn(p, xb, yb)[0])

if args.analytic_head:
    # ---- exact head: GF(2) solve over the TRAINING classes -------------------
    from tc3d.fermionic_decoration import _gf2_solve
    cls_seen = {}
    for c in range(n_class):
        members = np.where((class_id == c) & train_mask)[0]
        if len(members):
            cls_seen[c] = (tokens[members[0]], 0 if y[members[0]] > 0 else 1)
    rows = [gf2_row(tuple(int(b) for b in bvec)) for bvec, _ in cls_seen.values()]
    targets = [k for _, k in cls_seen.values()]
    if args.sampled and not args.holdout:
        # the online basis spans every absorbed equation — solve the small system
        rows_s = [r & ((1 << ncols) - 1) for r in basis.values()]
        targets_s = [(r >> ncols) & 1 for r in basis.values()]
    else:
        rows_s, targets_s = rows, targets
    print(f"analytic head: solving {len(rows_s)} equations "
          f"over {ncols} GF(2) columns ...", flush=True)
    sol = _gf2_solve(rows_s, targets_s, ncols)
    resid = sum(1 for r, k in zip(rows, targets)
                if (bin(r & sol).count("1") & 1) != k)
    print(f"analytic head: {len(rows)} class equations, GF(2) residual = {resid}")
    l = np.array([(sol >> i) & 1 for i in range(NP)], float)
    Q = np.zeros((NP, NP))
    for mm, (i, j) in enumerate(pair_idx):
        Q[i, j] = (sol >> (NP + mm)) & 1
    if args.dump_form:
        with open(args.dump_form, "w") as f:
            json.dump({"L": args.L, "NP": NP,
                       "plaq_order": "fermionic_plaquettes: normal axis c major, "
                                     "then ix, iy, iz",
                       "l": l.astype(int).tolist(), "Q": Q.astype(int).tolist(),
                       "n_equations": len(rows), "residual": resid}, f)
        print(f"solved form dumped to {args.dump_form}")
    # phi(t) = pi*[ sum l_i (1-t_i)/2 + sum_{i<j} Q_ij (1-t_i-t_j+t_i t_j)/4 ]
    th_lin = -np.pi * l / 2 - np.pi / 4 * (Q.sum(1) + Q.sum(0))
    th_quad = np.pi * Q / 4
    const = np.pi * (l.sum() / 2 + Q.sum() / 4)
    th_quad[0, 0] += const                       # t_0^2 = 1 carries the constant
    pr = dict(vs.parameters)
    pr["phase_lin"] = jnp.asarray(th_lin, jnp.complex128)
    pr["phase_quad"] = jnp.asarray(th_quad, jnp.complex128)
    vs.parameters = pr
    params0 = vs.parameters
    p = params0
    acc_all = float(acc_fn(p, jnp.asarray(X), jnp.asarray(y)))
    line = f"ANALYTIC: sign accuracy ({len(X)} dataset states) = {acc_all:.6f}"
    if args.holdout:
        line += (f"  HELDOUT acc = "
                 f"{float(acc_fn(p, jnp.asarray(X[~train_mask]), jnp.asarray(y[~train_mask]))):.6f}")
    print(line)
    if args.sampled and args.cert:
        # certificate: the solved form must be exact on FRESH samples
        rng_c = np.random.default_rng(args.seed + 1)
        ok = 0
        fresh = set()
        for _ in range(args.cert):
            s, sg = draw_class(rng_c)
            tv = token_vec(s)
            fresh.add(tv)
            ok += int((bin(gf2_row(tv) & sol).count("1") & 1) == (0 if sg > 0 else 1))
        unseen = sum(1 for t in fresh if t not in seen)
        print(f"CERTIFICATE: {ok}/{args.cert} fresh samples correct "
              f"({len(fresh)} distinct classes, {unseen} never sampled before)")
    if args.xcheck_enum and args.sampled:
        # L=2 regression: sampled solve must be exact on the FULL enumeration
        bt_e, y_e, tk_e = bfs_dataset()
        bad_cls = sum(1 for tv, sg in
                      {tuple(t): s for t, s in zip(tk_e.tolist(), y_e)}.items()
                      if tv in seen and seen[tv][1] != sg)
        X_e = (1.0 - 2.0 * bt_e).astype(np.float64)
        acc_e = float(acc_fn(p, jnp.asarray(X_e), jnp.asarray(y_e)))
        print(f"XCHECK vs enumeration: {len(X_e)} support states, "
              f"class-sign mismatches = {bad_cls}, CNN sign accuracy = {acc_e:.6f}")
    args.steps = 0                                # skip the SGD loop entirely


Xtr, ytr = X[train_mask], y[train_mask]
Xva = jnp.asarray(X[~train_mask]) if args.holdout else None
yva = jnp.asarray(y[~train_mask]) if args.holdout else None
rng = np.random.default_rng(0)
p = params0
logf = open(args.log, "a", buffering=1) if args.log else None
t0 = time.time()
for step in range(1, args.steps + 1):
    idx = rng.integers(0, len(Xtr), size=min(args.batch, len(Xtr)))
    (ltot, (lph, lam)), g = grad_fn(p, jnp.asarray(Xtr[idx]), jnp.asarray(ytr[idx]))
    if args.fit_head_only:
        g = {k: (val if k in ("phase_lin", "phase_quad")
                 else jax.tree_util.tree_map(jnp.zeros_like, val))
             for k, val in g.items()}
    p, m, v = adam_step(p, m, v, g, step)
    rec = {"step": step, "loss": float(ltot), "phase": float(lph), "amp": float(lam)}
    if args.holdout and (step % 50 == 0 or step == args.steps or step == 1):
        rec["val_loss"] = float(loss_eval(p, Xva, yva))
        rec["val_acc"] = float(acc_fn(p, Xva, yva))
        rec["train_acc"] = float(acc_fn(p, jnp.asarray(Xtr), jnp.asarray(ytr)))
    if logf:
        logf.write(json.dumps(rec) + "\n")
    if step % 10 == 0 or step <= 3 or step == args.steps:
        line = f"  step {step:4d}: loss={float(ltot):.5f}"
        if "val_loss" in rec:
            line += (f"  train_acc={rec['train_acc']:.4f}"
                     f"  VAL loss={rec['val_loss']:.5f} acc={rec['val_acc']:.4f}")
        print(line, flush=True)
    if args.save and args.ckpt_every and step % args.ckpt_every == 0:
        vs.parameters = p
        save_model(vs, f"{args.save}_s{step}", verbose=False)
if logf:
    logf.close()
print(f"fit done in {time.time()-t0:.1f}s")

# ---- final metrics ----------------------------------------------------------
acc_all = float(acc_fn(p, jnp.asarray(X), jnp.asarray(y)))
lp_all = apply_fun({"params": p}, jnp.asarray(X))
amp_std = float(jnp.std(jnp.real(lp_all)))
ok_row = np.asarray(y * np.cos(np.asarray(jnp.imag(lp_all))) > 0)
per_class_ok = all(ok_row[class_id == c].all() for c in range(n_class))
print(f"\nFINAL: sign accuracy ({len(X)} dataset states) = {acc_all:.6f}")
print(f"       every token class fully correct: {per_class_ok}")
print(f"       amplitude spread std(Re logpsi) = {amp_std:.4f}")

vs.parameters = p
if args.expect:
    vs.sample()
    print(f"       MCMC <H> of fitted state: {vs.expect(Ham)}   "
          f"(exact GS: {-4 * args.L ** 3})")
if args.save:
    save_model(vs, args.save)
