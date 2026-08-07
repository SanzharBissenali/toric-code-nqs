"""Supervised phase pre-fit of ToricCNN_gridinv on the exact fTC h=0 sign map (L=2).

Dataset: all 32,768 support states with exact BFS signs. Fit Im(log psi) to the
sign (loss 1 - sigma*cos b) + amplitude uniformity (var of Re log psi). Manual
Adam on conjugated grads (Wirtinger descent). Modes:
  --init random|trapped   (trapped = plateau checkpoint + symmetry-breaking noise)
  --holdout N             (exclude N of the 64 token classes from training)
"""
import argparse
import time
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp

from tc3d.builders import build_state
from tc3d.io import load_weights, save_model
from tc3d.fermionic_decoration import _mask

SCR = ("/private/tmp/claude-501/-Users-sanzhar123-Desktop-toric-code-nqs/"
       "88c57eca-122a-4c75-812c-21de18d04d1f/scratchpad")
TRAPPED = f"{SCR}/gridinv_fermionic_L2_PBC_hx0.0_hz0.0_n2x4_nh4-8_inv8-8_k2"

ap = argparse.ArgumentParser()
ap.add_argument("--init", choices=["random", "trapped"], default="random")
ap.add_argument("--holdout", type=int, default=0, help="# token classes held out")
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
args = ap.parse_args()

cfg = {"L": 2, "model": "fermionic", "arch": "ToricCNN_gridinv", "bc": "PBC",
       "kernel_size": 2, "noninv_hidden": [4, 8], "inv_hidden": [8, 8],
       "n_noninv": 2, "noninv_channels": 4, "n_samples": 8192, "n_chains": 16,
       "phase_head": args.phase_head}
geo, hi, Ham, vs, xz = build_state(cfg)

# ---- exact dataset: BFS over the stabilizer orbit --------------------------
stars = [_mask(v) for v in geo.vertex_all]
zxm = [(_mask(z), _mask(x)) for z, x, _ in xz]
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
y = np.array([sign[int(m)] for m in masks], dtype=np.float64)
bits = (masks[:, None] >> np.arange(geo.N)[None, :]) & 1
X = (1.0 - 2.0 * bits).astype(np.float64)                    # bit 1 -> spin -1
zedges = [z for z, _, _ in xz]
tokens = np.stack([bits[:, ze].sum(axis=1) & 1 for ze in zedges], axis=1)
class_id = np.unique(tokens, axis=0, return_inverse=True)[1]
n_class = class_id.max() + 1
print(f"dataset: {len(masks)} support states, {n_class} token classes, "
      f"{int((y < 0).sum())} negative")

train_mask = np.ones(len(masks), bool)
if args.holdout:
    held = np.random.default_rng(1).choice(n_class, size=args.holdout, replace=False)
    train_mask = ~np.isin(class_id, held)
    print(f"holding out classes {sorted(held.tolist())} "
          f"({(~train_mask).sum()} states)")

# ---- init -------------------------------------------------------------------
if args.init == "trapped":
    vs = load_weights(vs, TRAPPED)
    rng = jax.random.PRNGKey(7)
    noised = {}
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
    NP = tokens.shape[1]
    pair_idx = [(i, j) for i in range(NP) for j in range(i + 1, NP)]
    ncols = NP + len(pair_idx)
    cls_seen = {}
    for c in range(n_class):
        members = np.where((class_id == c) & train_mask)[0]
        if len(members):
            b = tokens[members[0]]
            cls_seen[c] = (b, 0 if y[members[0]] > 0 else 1)
    rows, targets = [], []
    for b, k in cls_seen.values():
        r = 0
        for i in range(NP):
            if b[i]: r |= 1 << i
        for m, (i, j) in enumerate(pair_idx):
            if b[i] and b[j]: r |= 1 << (NP + m)
        rows.append(r); targets.append(k)
    sol = _gf2_solve(rows, targets, ncols)
    resid = sum(1 for r, k in zip(rows, targets)
                if (bin(r & sol).count("1") & 1) != k)
    print(f"analytic head: {len(rows)} class equations, GF(2) residual = {resid}")
    l = np.array([(sol >> i) & 1 for i in range(NP)], float)
    Q = np.zeros((NP, NP))
    for m, (i, j) in enumerate(pair_idx):
        Q[i, j] = (sol >> (NP + m)) & 1
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
    line = f"ANALYTIC: sign accuracy (all {len(X)} support states) = {acc_all:.6f}"
    if args.holdout:
        line += (f"  HELDOUT acc = "
                 f"{float(acc_fn(p, jnp.asarray(X[~train_mask]), jnp.asarray(y[~train_mask]))):.6f}")
    print(line)
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
        import json as _json
        logf.write(_json.dumps(rec) + "\n")
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
per_class_ok = all(
    float(acc_fn(p, jnp.asarray(X[class_id == c]), jnp.asarray(y[class_id == c]))) == 1.0
    for c in range(n_class))
print(f"\nFINAL: sign accuracy (all 32768 support states) = {acc_all:.6f}")
print(f"       every token class fully correct: {per_class_ok}")
print(f"       amplitude spread std(Re logpsi) = {amp_std:.4f}")

vs.parameters = p
if args.expect:
    vs.sample()
    print(f"       MCMC <H> of fitted state: {vs.expect(Ham)}   (exact GS: -32)")
if args.save:
    save_model(vs, args.save)
