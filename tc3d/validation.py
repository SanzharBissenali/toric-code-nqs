"""
tc3d/validation.py
─────────────────────────────────────────────────────────────────────────────
End-of-training observables of a trained NQS (bosonic and fermionic), written
into the run JSON / W&B by `tc3d.train` and replayed on saved checkpoints by the
analysis evaluators (analysis/scripts/eval_ckpt.py, eval_snapshots.py, ...):

    nqs_observables           E0 (+ error, variance, Vscore), <A_v>, <B_p> (the
                              decorated B~_p for the fermionic model), <sigma_x>,
                              <sigma_z> (+ <sigma_y> when h_y != 0). Keys stay
                              physical in the dual basis.
    pooled_final_observables  the same, pooled over K sampling rounds, plus the
                              frozen ParaToric-convention FM operators
                              (`paratoric_order_parameters`).
    topological_observables   end-of-training O_FM + central-plaquette Rényi S2.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import netket as nk


# =============================================================================
# Observable operators (mean over sites / stabilizers, one expect call each)
# =============================================================================

def _mean_operators(hi, geo, xz_stabs=None, dual=False, hy=0.0):
    """Build (A_v_mean, B_mean, Mx, Mz, My) operators, each already divided by count.
    My is None unless hy != 0 (skips the complex-operator build/eval cost on the
    common hy=0 real-run path).

    A single summed operator per quantity gives the mean — and a proper MC error
    bar on that mean — in one `vs.expect` call. The plaquette operator B depends
    on the model:
        bosonic  (xz_stabs is None): B_p   = prod sigma^z over geo.plaq_all
        fermionic (xz_stabs given):  B~_p  = (prod sigma^z over z_edges)
                                             (prod sigma^x over x_edges)
    `dual=True` (Hadamard-conjugated run): the PHYSICAL observables are kept —
    each is built from the swapped Pauli letter so the returned operators (and
    downstream JSON keys) retain their basis-independent meaning. sigma_y does not
    swap under Hadamard (H sigma_y H = -sigma_y), so My instead carries a sign flip
    -- same `sgn_y` law as tc3d.hamiltonian.create_hamiltonian.
    """
    N, N_v = geo.N, len(geo.vertex_all)
    # representation of the physical Pauli in this run's sampling basis
    rep_x = nk.operator.spin.sigmaz if dual else nk.operator.spin.sigmax
    rep_z = nk.operator.spin.sigmax if dual else nk.operator.spin.sigmaz

    A_op = 0
    for v in geo.vertex_all:
        term = 1
        for i in v:
            if i == -1:
                continue
            term = term * rep_x(hi, int(i))
        A_op = A_op + term

    B_op = 0
    if xz_stabs is None:                                    # bosonic
        for p in geo.plaq_all:
            term = 1
            for i in p:
                if i == -1:
                    continue
                term = term * rep_z(hi, int(i))
            B_op = B_op + term
        n_B = len(geo.plaq_all)
    else:                                                   # fermionic
        assert not dual, "dual basis is bosonic-only"
        for z_edges, x_edges, _ in xz_stabs:
            term = 1
            for i in z_edges:
                if i != -1:
                    term = term * nk.operator.spin.sigmaz(hi, int(i))
            for i in x_edges:
                if i != -1:
                    term = term * nk.operator.spin.sigmax(hi, int(i))
            B_op = B_op + term
        n_B = len(xz_stabs)

    Mx = sum(rep_x(hi, i) for i in range(N))
    Mz = sum(rep_z(hi, i) for i in range(N))
    My = None
    if hy != 0:
        sgn_y = -1.0 if dual else 1.0
        My = sgn_y * sum(nk.operator.spin.sigmay(hi, i) for i in range(N)) / N

    return A_op / N_v, B_op / n_B, Mx / N, Mz / N, My


def _m(s):
    """(mean, err) of a netket Stats object, real part."""
    return float(np.real(s.mean)), float(np.real(s.error_of_mean))


def nqs_observables(vs, Ham, geo, xz_stabs=None, dual=False, hy=0.0,
                    mean_ops=None) -> Dict[str, float]:
    """Raw NQS expectation values + errors (NO exact reference needed).

    Keys mirror the colab_exact_diag.py reference schema (E0, A_v_mean, …) so the
    training pipeline can write them straight to its local JSON and W&B. The
    plaquette observable is B_p (bosonic) or the decorated B~_p (fermionic).
    `dual=True` keeps every key physical (constructors swapped, labels unchanged).
    `mean_ops` accepts a prebuilt `_mean_operators` tuple — the operators are
    field-independent, so multi-round callers hoist the ~3–5 s host-side
    construction out of their loop (2026-08-12 profiling). sy_mean/sy_err are
    added ONLY when hy != 0 — hy=0 runs must produce byte-identical observables
    JSONs (regression safety, and skips the complex-operator eval cost).
    """
    N = geo.N
    E = vs.expect(Ham)
    E_mean = float(np.real(E.mean))
    E_err = float(np.real(E.error_of_mean))
    E_var = float(np.real(E.variance))

    A_op, B_op, Mx_op, Mz_op, My_op = (mean_ops if mean_ops is not None else
                                       _mean_operators(vs.hilbert, geo,
                                                       xz_stabs=xz_stabs,
                                                       dual=dual, hy=hy))
    A_m, A_e = _m(vs.expect(A_op))
    B_m, B_e = _m(vs.expect(B_op))
    Mx_m, Mx_e = _m(vs.expect(Mx_op))
    Mz_m, Mz_e = _m(vs.expect(Mz_op))

    obs = {
        "E0": E_mean, "E_err": E_err, "E_var": E_var,
        "Vscore": N * E_var / E_mean**2 if E_mean != 0 else float("nan"),
        "A_v_mean": A_m, "A_v_err": A_e,
        "B_p_mean": B_m, "B_p_err": B_e,
        "sx_mean": Mx_m, "sx_err": Mx_e,
        "sz_mean": Mz_m, "sz_err": Mz_e,
    }
    if np.iscomplexobj(E.mean):
        # Complex ansatz (sign-full h_y, fermionic, or --force_complex): a
        # nonzero Im(E) is a sign/convention check, since H is Hermitian (mirrors
        # wandb_logger.log_step's per-step energy_im). error_of_mean is a single
        # real scalar covering the whole complex estimator (verified inline), not
        # separable into a real/imag pair, so there is no E_im_err to report here.
        # hy=0 REAL runs never hit this branch (E.mean stays real dtype) -> their
        # observables JSON key set is unchanged.
        obs["E_im"] = float(np.imag(E.mean))
    if hy != 0 and My_op is not None:
        obs["sy_mean"], obs["sy_err"] = _m(vs.expect(My_op))
    return obs


def paratoric_order_parameters(vs, geo, cfg, string_ops=None) -> Dict[str, float]:
    """The frozen cross-method FM operators on the CURRENT samples: Z-string +
    X-membrane families (pt = growing corner-rule cube, R1 = fixed anchor).

    Mirrors analysis/scripts/eval_ckpt.py's paratoric_fm/paratoric_membrane_fm (keys
    O_FM_paratoric, O_FM_membrane_pt, O_FM_membrane_R1 — keep in sync; the
    sample-wise membrane never builds a 2^support LocalOperator). Best-effort:
    a family missing at this L (string/pt below L=4/5) records its skip reason.
    Fermionic runs are skipped whole (the frozen convention is bosonic)."""
    from tc3d.fm import (_pauli_product, fm_ratio, fm_ratio_sampled,
                         paratoric_fm_edges, paratoric_membrane_kwargs)
    out: Dict[str, float] = {}
    if cfg.get("model", "bosonic") == "fermionic":
        return out
    dual = bool(cfg.get("dual_basis", False))
    hi = vs.hilbert
    try:
        if string_ops is not None:                         # hoisted by pooled eval
            open_op, closed_op = string_ops
        else:
            closed, open_ = paratoric_fm_edges(geo)
            pauli = "x" if dual else "z"
            open_op = _pauli_product(hi, open_, pauli)
            closed_op = _pauli_product(hi, closed, pauli)
        val, err, (den, den_err) = fm_ratio(vs, open_op, closed_op,
                                            return_den=True)
        out["O_FM_paratoric"], out["O_FM_paratoric_err"] = val, err
        # den travels with every ratio: analysis den-gates near-critical points
        # (audit 2026-08-11 MEDIUM b — QMC has den_z, NQS must match)
        out["O_FM_paratoric_den"], out["O_FM_paratoric_den_err"] = den, den_err
    except Exception as e:                                 # noqa: BLE001
        out["O_FM_paratoric_error_msg"] = f"{type(e).__name__}: {e}"
    if dual:                                               # diagonal only in dual frame
        for fam, tag in (("pt", "pt"), (1, "R1")):
            try:
                kw = paratoric_membrane_kwargs(geo, None if fam == "pt" else fam)
                # fm.fm_ratio_sampled == eval_ckpt's estimator: block jackknife
                # through the assembled ratio (open/closed share samples AND
                # half the support — independent-error hypot over-estimated by
                # ~1.6x; unified per the 2026-08-11 audit, MEDIUM c), pooled
                # NaN convention at <closed> <= 0, den fields for the gate.
                o, oe, (c, ce) = fm_ratio_sampled(
                    vs, geo, R=kw["R"], corner=kw["corner"],
                    vertical=kw["vertical"], return_den=True)
                out[f"O_FM_membrane_{tag}"] = o
                out[f"O_FM_membrane_{tag}_err"] = oe
                out[f"O_FM_membrane_{tag}_den"] = c
                out[f"O_FM_membrane_{tag}_den_err"] = ce
            except Exception as e:                         # noqa: BLE001
                out[f"O_FM_membrane_{tag}_error_msg"] = f"{type(e).__name__}: {e}"
    return out


def build_eval_operators(hi, geo, cfg, xz_stabs=None):
    """(mean_ops, string_ops) for the final observable block — FIELD-INDEPENDENT.

    Pure host-side NetKet assembly depending only on (hilbert, geometry, basis,
    model), so a batch runner builds it ONCE per chunk and hands it to every
    point: measured 1150 s per call inside a warm L=4 sweep process vs 46 s for
    the eight GPU eval rounds it feeds (2026-08-11 [t] instrumentation, job
    56641739_1). Best-effort on the string family, as before."""
    dual = bool(cfg.get("dual_basis", False))
    t_ops = time.time()
    mean_ops = _mean_operators(hi, geo, xz_stabs=xz_stabs, dual=dual,
                               hy=cfg.get("hy", 0.0))
    string_ops = None
    if cfg.get("model", "bosonic") != "fermionic":
        try:
            from tc3d.fm import _pauli_product, paratoric_fm_edges
            closed, open_ = paratoric_fm_edges(geo)
            pauli = "x" if dual else "z"
            string_ops = (_pauli_product(hi, open_, pauli),
                          _pauli_product(hi, closed, pauli))
        except Exception:                                  # noqa: BLE001
            string_ops = None      # per-round path records the skip reason
    print(f"[t] final-eval ops built in {time.time() - t_ops:.1f}s", flush=True)
    return mean_ops, string_ops


def pooled_final_observables(vs, Ham, geo, cfg, xz_stabs=None,
                             rounds: int = 1, eval_ops=None) -> Dict[str, float]:
    """End-of-training evaluation with K extra sampling ROUNDS through the
    already-compiled training kernels: rounds are contiguous segments of the
    SAME chains (state persists across vs.sample() calls), so K x n_samples is
    statistically identical to one big draw at equal chain count — 65k-equivalent
    statistics at training-budget memory, no recompile (BLOG 2026-08-11).

    Pooling: mean of round means; error = sqrt(sum e_k^2)/K (each round's error
    is autocorrelation-corrected within the round; at tau ~ 1 retained-sample
    units the cross-round correlation is negligible — certified by the sqrt(K)
    smoke). `E_err_scatter` (SEM of round means) is stored as the diagnostic:
    scatter >> pooled error means the rounds are NOT quasi-independent (tau
    blow-up near criticality) — trust the larger number and raise n_sweeps."""
    rounds = max(1, int(rounds))
    dual = bool(cfg.get("dual_basis", False))
    # Field-independent: a batch runner passes these in, built once per chunk.
    mean_ops, string_ops = (eval_ops if eval_ops is not None else
                            build_eval_operators(vs.hilbert, geo, cfg,
                                                 xz_stabs=xz_stabs))
    acc: Dict[str, List[float]] = {}
    for _ in range(rounds):
        vs.sample()                                        # force a fresh segment
        r = nqs_observables(vs, Ham, geo, xz_stabs=xz_stabs, dual=dual,
                            hy=cfg.get("hy", 0.0), mean_ops=mean_ops)
        r.update(paratoric_order_parameters(vs, geo, cfg, string_ops=string_ops))
        for k, v in r.items():
            if isinstance(v, (int, float)):
                acc.setdefault(k, []).append(float(v))
            else:
                acc.setdefault(k, v)                       # skip strings (error msgs)
    obs: Dict[str, float] = {}
    for k, vals in acc.items():
        if not isinstance(vals, list):
            obs[k] = vals
        elif k.endswith("_err"):
            obs[k] = float(np.sqrt(np.sum(np.square(vals))) / len(vals))
        else:
            obs[k] = float(np.mean(vals))
    if rounds > 1:
        e0 = acc.get("E0", [])
        obs["final_eval_rounds"] = rounds
        obs["E_err_scatter"] = float(np.std(e0, ddof=1) / np.sqrt(len(e0)))
    return obs


# =============================================================================
# Topological order parameters at end of training (O_FM + Rényi-S₂)
# =============================================================================
# Recorded alongside the magnetisations/stabilisers so a single trained checkpoint
# already carries its FM ratio and central-plaquette S₂ — no second GPU trip for the
# common case. This is a SINGLE-STATE mirror of the sweep extractors (`fm.fm_sweep`,
# `renyi.renyi_sweep`), reusing their operators/estimators verbatim. It is best-effort:
# the authoritative curves + h_c fits still come from the sweep extractors (full grid,
# few long chains, B3 health), and a timed-out run just re-extracts on an interactive
# node. Needs the bulk-centred placement, hence L>=4 (every field line this feeds uses
# L>=4). Never raises: each estimator is guarded so a heavy-tailed point can't lose a run.

def _auto_fm_sector(cfg) -> str:
    """Pick the sector whose stabiliser the DOMINANT field breaks: h_x → plaquettes
    (magnetic σ^x membrane), h_z → vertices (electric σ^z loop). Ties → magnetic."""
    hx = float(cfg.get("hx", 0.0) or 0.0)
    hz = float(cfg.get("hz", 0.0) or 0.0)
    return "magnetic" if hx >= hz else "electric"


def _eval_state(vs, *, n_chains: int = 16, n_samples: Optional[int] = None):
    """Best-effort eval clone of `vs`: same model + weights on a sampler with a few
    LONG chains (valid error bars for the FM/S₂ estimators, matching fm.load_vstate's
    `eval_chains`). Falls back to the trained `vs` if the reconfig fails — the estimators
    then lean on their variance/jackknife fallbacks (see `_stat_err`, `_jackknife_fm_ratio`)."""
    try:
        smp = vs.sampler
        try:
            smp = smp.replace(n_chains=int(n_chains))
        except Exception:                                  # sampler not .replace-able
            smp = nk.sampler.MetropolisLocal(vs.hilbert, n_chains=int(n_chains))
        ev = nk.vqs.MCState(smp, vs.model, n_samples=int(n_samples or vs.n_samples))
        ev.variables = vs.variables
        return ev
    except Exception:                                      # noqa: BLE001
        return vs


def topological_observables(vs, geo, cfg, *, hi=None) -> Dict[str, Any]:
    """End-of-training O_FM (sector the dominant field breaks) + central-plaquette S₂,
    as a flat dict merged into `observables`. Returns {} when disabled or L<4. See the
    section header for the single-state-vs-sweep caveat.

    Keys: fm_sector, fm_R, O_FM(+_err), S2(+_err); magnetic O_FM also carries the B3
    estimator-health pair (O_FM_b3_max_kurt, O_FM_b3_min_ess_frac). A failed estimator
    sets its value to None with an *_error_msg rather than raising.

    Fermionic model: the sector is forced to "electric" — a bare σ^z loop
    anticommutes with the decorated B~_p (identically-zero, noise-only O_FM), so
    O_FM is built from the dressed Z(z)·X(x) Wilson loop/string
    (`fm.dressed_electric_edges`, same bulk placement/R as bosonic); flagged
    with fm_dressed=True."""
    L = int(cfg.get("L", getattr(geo, "L", 0)) or 0)
    sector = cfg.get("fm_sector", "auto")
    if L < 4 or sector in (None, "none"):
        return {}
    if sector == "auto":
        sector = _auto_fm_sector(cfg)
    model = cfg.get("model", "bosonic")
    if model == "fermionic":
        sector = "electric"      # only the dressed σ^z sector exists (no membrane dressing)

    import tc3d.fm as fm
    import tc3d.renyi as renyi

    hi = hi if hi is not None else vs.hilbert
    planes = ("xy", "xz", "yz")
    dual = bool(cfg.get("dual_basis", False))
    aspect = float(cfg.get("topological_aspect", 0.5))
    R = max(1, min(round(L * aspect), L - 3))              # clamp loop side to the strict bulk
    ev = _eval_state(vs, n_chains=16, n_samples=cfg.get("n_samples"))
    out: Dict[str, Any] = {"fm_sector": sector, "fm_R": int(R)}
    if model == "fermionic":
        out["fm_dressed"] = True

    try:                                                   # ---- O_FM ----
        pairs, _meta = fm.build_loop_operators(
            geo, hi, sector, placement="bulk", planes=planes, R=R, dual=dual,
            model=model)
        ev.reset()
        # 3-way dispatch — MUST mirror fm.fm_sweep's exact branch order (fm.py
        # ~1475-1483): magnetic+dual returns (label, kwargs) SPECS, not NetKet
        # operator pairs, so `fm_ratio_avg`'s 3-tuple unpack raises ValueError on
        # them if the sampled-diagonal branch is skipped (was silently swallowed
        # to O_FM=None below — 2026-08-26 audit).
        if fm.uses_telescoped(sector, dual):               # off-diagonal σ^x membrane
            O, Oe, _per, diags = fm.fm_ratio_avg_telescoped(
                ev, geo, pairs, chunk=cfg.get("chunk_size"))
            faces = [f for d in diags.values()
                     for f in d["health_open"] + d["health_closed"]]
            out["O_FM_b3_max_kurt"] = max(
                (f["excess_kurtosis"] for f in faces), default=float("nan"))
            out["O_FM_b3_min_ess_frac"] = min(
                (f["ess_frac"] for f in faces), default=float("nan"))
        elif fm.uses_sampled_diagonal(sector, dual):       # dual membrane: ±1 products
            O, Oe, _per = fm.fm_ratio_avg_sampled(ev, geo, pairs)
        else:                                              # operator pairs via expect
            O, Oe, _per = fm.fm_ratio_avg(ev, pairs)
        out["O_FM"] = float(np.real(O))
        out["O_FM_err"] = float(np.real(Oe))
    except Exception as e:                                 # noqa: BLE001
        out["O_FM"] = None
        out["O_FM_error_msg"] = f"{type(e).__name__}: {e}"

    try:                                                   # ---- S₂ (central plaquette) ----
        parts = renyi.patch_partitions(geo, planes)
        robs = renyi._build_renyi_obs(hi, parts)
        evs, _fb = renyi._ensure_renyi_sampler(ev, hi, robs[0][1], verbose=False)
        evs.reset()
        s2, s2e, _per = renyi._s2_of_state(evs, robs)
        out["S2"] = float(s2)
        out["S2_err"] = float(s2e)
    except Exception as e:                                 # noqa: BLE001
        out["S2"] = None
        out["S2_error_msg"] = f"{type(e).__name__}: {e}"
    return out
