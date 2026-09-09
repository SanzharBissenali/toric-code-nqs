"""Transition locators + finite-size extrapolation for field cuts of the 3D toric code.

NetKet-free (numpy/scipy only). A *cut* fixes two of (h_x, h_y, h_z) and sweeps the
third; a *curve* is one observable vs the swept field at one system size L.

Consumed by `analysis/notebooks/transition_fss.ipynb` (cut -> per-L locators -> FSS ->
`results/transitions/<tag>.json`) and `analysis/notebooks/phase_diagram_3d_btc.ipynb`
(reads those JSONs). Two data lanes are supported:

* **runs**   — per-run final-state JSONs written by `tc3d.train` (`config` +
  `observables` dicts): `results/phaseB*/`, `results/hy_cuts_L4/`. Several runs at one
  (L, h) point (reruns, warm chains, seeds) -> the lowest-energy non-diverged run wins
  (recipe §B.4).
* **fm_json** — the pre-optimization campaign's aggregated `fm_L*.json` / `s2_L*.json`
  (`tc3d.fm` / `tc3d.renyi` output, mirrored in gitignored `data/tc_nqs/phase_h*`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, brentq

FIELDS = ("hx", "hy", "hz")
NU_3D_ISING = 0.62997          # Kos-Poland-Simmons-Duffin bootstrap; 1/nu = 1.5874
EXACT = {                      # thermodynamic anchors, J=1 convention (exact_benchmarks.REFERENCE)
    "hz_c(hx=0,hy=0)": 0.193869,   # 2nd order, (3+1)D Ising* via 3D-TFIM duality
    "hx_c(hz=0,hy=0)": 1.0,        # 1st order, self-duality to 4D Wegner Z2 gauge theory
}


# ----------------------------------------------------------------------------- data model
@dataclass
class Curve:
    L: int
    h: np.ndarray
    y: np.ndarray
    ye: np.ndarray
    obs: str                     # "O_FM" | "S2" | "E"
    src: str = ""                # provenance (dir / file pattern)
    names: list = field(default_factory=list)

    def clip(self, window):
        """Restrict to h in [lo, hi] (inclusive); None bounds are open."""
        lo, hi = window
        m = np.ones_like(self.h, bool)
        if lo is not None:
            m &= self.h >= lo - 1e-12
        if hi is not None:
            m &= self.h <= hi + 1e-12
        names = [n for n, k in zip(self.names, m) if k] if self.names else []
        return Curve(self.L, self.h[m], self.y[m], self.ye[m], self.obs, self.src, names)


def cut_tag(hy, fixed_name, fixed_val, sweep, lane=None):
    """'hy0.0_hx0.2_sweep-hz' (+ '@lane' for non-production data lanes)."""
    t = f"hy{hy:g}_{fixed_name}{fixed_val:g}_sweep-{sweep}"
    return t if lane in (None, "", "prod") else f"{t}@{lane}"


def register_cuts(*groups):
    """Merge (tag, spec) groups into one registry; raise on a duplicate tag instead of
    silently overwriting (two cuts that resolve to the same cut_tag would clobber each
    other's results/transitions record — give one of them a distinct lane=...)."""
    out = {}
    for g in groups:
        for tag, spec in (g.items() if isinstance(g, dict) else g):
            if tag in out:
                raise ValueError(f"duplicate cut tag {tag!r}: set lane=... on one of the two entries")
            out[tag] = spec
    return out


# ----------------------------------------------------------------------------- loaders
def _is_final_json(f: Path):
    return not f.name.endswith((".snapshots.json", ".curve.json", "finaleval_electric.json"))


def load_runs(dirs, sweep, fixed, obs, err=None, tol=1e-9, keep_diverged=False):
    """{L: Curve} from per-run final-state JSONs.

    fixed = {"hx": 0.2, "hy": 0.0}; sweep = "hz"; obs = key in `observables`
    ("O_FM_paratoric", "O_FM_membrane_R1", "S2", "E0"). Per (L, h): the lowest-energy
    non-diverged run that carries a finite `obs` wins. Energies of the winners are
    returned alongside as `cv.E` so the caller can plot E without a second pass.

    keep_diverged=True (chain / spinodal analysis): diverged runs are kept as candidates
    but never beat a non-diverged run; a point whose only runs diverged is included with
    that run's values and flagged. Every curve then carries `cv.diverged` (bool per point)
    and `cv.diverged_runs` ({h: [run stems]}, all diverged runs seen at that field).
    """
    err = err or ("E_err" if obs == "E0" else obs + "_err")
    best, div_runs = {}, {}                     # (L, h) -> (E0, y, ye, name, diverged)
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if not _is_final_json(f):
                continue
            j = json.loads(f.read_text())
            c, o = j.get("config"), j.get("observables")
            if not c or not o or o.get("E0") is None:
                continue
            dv = bool(j.get("diverged"))
            if dv and not keep_diverged:
                continue
            if any(abs(float(c.get(k, 0.0)) - v) > tol for k, v in fixed.items()):
                continue
            key = (int(c["L"]), round(float(c[sweep]), 6))
            if dv:
                div_runs.setdefault(key, []).append(f.stem)
            y = o.get(obs)
            if y is None or not np.isfinite(y):
                continue
            row = (float(o["E0"]), float(y), float(o.get(err) if o.get(err) is not None else np.nan), f.stem, dv)
            cur = best.get(key)
            # rank: non-diverged beats diverged; within a class the lower energy wins
            if cur is None or (cur[4], cur[0]) > (dv, row[0]):
                best[key] = row
    out = {}
    for L in sorted({k[0] for k in best} | {k[0] for k in div_runs}):
        pts = sorted((h, *best[(LL, h)]) for (LL, h) in best if LL == L)
        h = np.array([p[0] for p in pts])
        cv = Curve(L, h, np.array([p[2] for p in pts]), np.array([p[3] for p in pts]),
                   obs, src=";".join(str(d) for d in dirs), names=[p[4] for p in pts])
        cv.E = np.array([p[1] for p in pts])   # winners' energies (attribute, not a field)
        if keep_diverged:
            cv.diverged = np.array([p[5] for p in pts], bool)
            cv.diverged_runs = {hh: v for (LL, hh), v in sorted(div_runs.items()) if LL == L}
        out[L] = cv
    return out


def load_snapshot_s2(dirs, sweep, fixed, tol=1e-9, keep_diverged=False):
    """{L: Curve(S2)} from `*.snapshots.json` replay series (last snapshot per run);
    duplicates at one (L, h) -> the series whose last snapshot has the lowest energy.
    A series counts as diverged when its sibling final JSON (`<stem>.json`) says so;
    such series are dropped unless keep_diverged=True (then flagged in `cv.diverged`)."""
    best = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.snapshots.json")):
            j = json.loads(f.read_text())
            c = j.get("config", {})
            if any(abs(float(c.get(k, 0.0)) - v) > tol for k, v in fixed.items()):
                continue
            sib = f.with_name(f.name[: -len(".snapshots.json")] + ".json")
            dv = bool(json.loads(sib.read_text()).get("diverged")) if sib.exists() else False
            if dv and not keep_diverged:
                continue
            ser = [s for s in j.get("series", []) if "error" not in s and s.get("S2") is not None]
            if not ser:
                continue
            last = ser[-1]
            E = last.get("E0", last.get("energy", 0.0)) or 0.0
            key = (int(c["L"]), round(float(c[sweep]), 6))
            row = (float(E), float(last["S2"]), float(last.get("S2_err", np.nan)), f.stem, dv)
            cur = best.get(key)
            if cur is None or (cur[4], cur[0]) > (dv, row[0]):
                best[key] = row
    out = {}
    for L in sorted({k[0] for k in best}):
        pts = sorted((h, *best[(LL, h)]) for (LL, h) in best if LL == L)
        cv = Curve(L, np.array([p[0] for p in pts]), np.array([p[2] for p in pts]),
                   np.array([p[3] for p in pts]), "S2", src=";".join(map(str, dirs)),
                   names=[p[4] for p in pts])
        if keep_diverged:
            cv.diverged = np.array([p[5] for p in pts], bool)
        out[L] = cv
    return out


_FM_PAT = re.compile(r"fm_L(\d+)_h[xz][\d.]+(?:_([A-Za-z0-9.]+))?\.json$")
_S2_PAT = re.compile(r"s2_L(\d+)_h[xz][\d.]+_s2plaq\.json$")
_EN_PAT = re.compile(r"energy_L(\d+)_h[xz][\d.]+\.json$")


def load_fm_jsons(dirpath, tags=("bulkR1",)):
    """{L: Curve(O_FM)} from aggregated fm_L{L}_*_{tag}.json. `tags` is a preference
    order: the first tag present at a given L is used (mixed-family FSS is flagged by
    Curve.src, mirror plot_phase_diagram.load_curves' warning)."""
    tags = (tags,) if isinstance(tags, str) else tuple(tags)
    found = {}
    for f in sorted(Path(dirpath).glob("fm_L*.json")):
        m = _FM_PAT.match(f.name)
        if not m:
            continue
        L, tag = int(m.group(1)), (m.group(2) or "")
        if tag not in tags:
            continue
        rank = tags.index(tag)
        if L in found and found[L][0] <= rank:
            continue
        j = json.loads(f.read_text())
        cv = Curve(L, np.array(j["field"], float), np.array(j["O"], float),
                   np.array(j["Oe"], float), "O_FM", src=f"{f.name} [{tag}]", names=j.get("names", []))
        cv.h_c_stored = (j.get("h_c"), j.get("h_c_err"), j.get("h_c_fd"))
        found[L] = (rank, cv)
    return {L: v[1] for L, v in sorted(found.items())}


def load_s2_jsons(dirpath):
    out = {}
    for f in sorted(Path(dirpath).glob("s2_L*_s2plaq.json")):
        m = _S2_PAT.match(f.name)
        j = json.loads(f.read_text())
        out[int(m.group(1))] = Curve(int(m.group(1)), np.array(j["field"], float),
                                     np.array(j["S2"], float), np.array(j["S2e"], float),
                                     "S2", src=f.name, names=j.get("names", []))
    return dict(sorted(out.items()))


def load_energy_jsons(dirpath):
    out = {}
    for f in sorted(Path(dirpath).glob("energy_L*.json")):
        m = _EN_PAT.match(f.name)
        j = json.loads(f.read_text())
        out[int(m.group(1))] = Curve(int(m.group(1)), np.array(j["field"], float),
                                     np.array(j["E"], float), np.array(j["E_spread"], float),
                                     "E", src=f.name)
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------- fit models
def logistic(h, a, b, h0, w):
    """Plateau a -> a+b; inflection at h0 (= dO/dh extremum); width w."""
    with np.errstate(over="ignore"):
        return a + b / (1.0 + np.exp(-(h - h0) / w))


def dlogistic(h, a, b, h0, w):
    with np.errstate(over="ignore", invalid="ignore"):
        z = np.exp(-(h - h0) / w)
        return np.nan_to_num((b / w) * z / (1.0 + z) ** 2)


def richards(h, A, K, B, M, Q, nu):
    """Generalized logistic; asymmetric rise. Inflection M + ln(Q/nu)/B."""
    with np.errstate(over="ignore"):
        return A + (K - A) / (1.0 + Q * np.exp(-B * (h - M))) ** (1.0 / nu)


def richards_infl(p):
    A, K, B, M, Q, nu = p
    return M + np.log(Q / nu) / B


def richards_half(p):
    A, K, B, M, Q, nu = p
    return M + np.log(Q / (2 ** nu - 1.0)) / B


@dataclass
class Fit:
    method: str
    h_c: float
    h_c_err: float
    L: int = 0
    chi2red: float = np.nan
    n: int = 0
    popt: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def ok(self):
        return np.isfinite(self.h_c)


def _sigma(ye, n):
    """Weights: non-positive errors floor at the smallest positive one, non-finite
    errors ceiling at the largest (least trustworthy) — mirrors fm.fit_transition."""
    if ye is None:
        return None
    s = np.asarray(ye, float)
    pos = s[np.isfinite(s) & (s > 0)]
    if len(pos) == 0:
        return None
    return np.where(np.isfinite(s), np.where(s > 0, s, pos.min()), pos.max())


def _scale(err_mode, chi2red):
    """Error inflation: 'absolute' trusts the bars; 'scaled' = curve_fit's default
    (cov * chi2red); 'pdg' inflates only when chi2red > 1 (PDG scale factor)."""
    if not np.isfinite(chi2red):
        return 1.0
    return {"absolute": 1.0, "scaled": np.sqrt(chi2red),
            "pdg": max(1.0, np.sqrt(chi2red))}[err_mode]


def _grad_err(f, p, cov, h=1e-6):
    g = np.array([(f(p + h * e) - f(p - h * e)) / (2 * h) for e in np.eye(len(p))])
    return float(np.sqrt(max(g @ cov @ g, 0.0)))


def _half_rise_h(h, y):
    """First crossing of the mid-plateau level (linear interpolation) — the p0 for h0."""
    lo, hi = np.mean(y[:2]), np.mean(y[-2:])
    mid = 0.5 * (lo + hi)
    s = np.sign(y - mid)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return float(np.median(h))
    i = idx[0]
    return float(h[i] + (mid - y[i]) * (h[i + 1] - h[i]) / (y[i + 1] - y[i] + 1e-300))


def fit_logistic(curve: Curve, err_mode="pdg", window=(None, None)) -> Fit:
    """4-parameter logistic; h_c = inflection h0 (works for rising O_FM and falling S2)."""
    c = curve.clip(window)
    h, y = c.h, c.y
    keep = np.isfinite(h) & np.isfinite(y)
    h, y, ye = h[keep], y[keep], c.ye[keep]
    if len(y) < 5:
        return Fit("logistic", np.nan, np.nan, curve.L, n=len(y))
    p0 = [y[0], y[-1] - y[0], _half_rise_h(h, y), 0.1 * (h[-1] - h[0]) or 0.1]
    s = _sigma(ye, len(y))
    kw = dict(sigma=s, absolute_sigma=True) if s is not None else {}
    try:
        popt, pcov = curve_fit(logistic, h, y, p0=p0, maxfev=40000,
                               bounds=([-np.inf, -np.inf, h[0], 1e-4], [np.inf, np.inf, h[-1], np.inf]), **kw)
    except Exception as exc:                                     # noqa: BLE001
        return Fit("logistic", np.nan, np.nan, curve.L, n=len(y), extra={"error": str(exc)})
    dof = max(len(y) - 4, 1)
    chi2red = float(np.sum(((logistic(h, *popt) - y) / (s if s is not None else 1.0)) ** 2) / dof)
    err = float(np.sqrt(abs(pcov[2, 2]))) * _scale(err_mode, chi2red)
    return Fit("logistic", float(popt[2]), err, curve.L, chi2red, len(y), list(popt),
               extra={"width": float(popt[3]), "err_mode": err_mode})


def fit_richards(curve: Curve, err_mode="pdg", window=(None, None)) -> Fit:
    """6-parameter generalized logistic; h_c = inflection, extra['h_half'] = mid-rise."""
    c = curve.clip(window)
    h, y = c.h, c.y
    keep = np.isfinite(h) & np.isfinite(y)
    h, y, ye = h[keep], y[keep], c.ye[keep]
    if len(y) < 8:
        return Fit("richards", np.nan, np.nan, curve.L, n=len(y))
    span = h[-1] - h[0]
    rising = y[-1] > y[0]
    lo, hi = (min(y.min(), 0.0), max(y.max(), 0.0))
    p0 = [y[0], y[-1], 8.0 / span, _half_rise_h(h, y), 1.0, 1.0]
    bounds = ([lo - 0.5 * abs(hi - lo) - 0.2, lo - 0.5 * abs(hi - lo) - 0.2, 0.5 / span, h[0], 1e-3, 1e-2],
              [hi + 0.5 * abs(hi - lo) + 0.2, hi + 0.5 * abs(hi - lo) + 0.2, 300.0 / span, h[-1], 1e3, 20.0])
    s = _sigma(ye, len(y))
    kw = dict(sigma=s, absolute_sigma=True) if s is not None else {}
    try:
        popt, pcov = curve_fit(richards, h, y, p0=p0, bounds=bounds, maxfev=200000, **kw)
    except Exception as exc:                                     # noqa: BLE001
        return Fit("richards", np.nan, np.nan, curve.L, n=len(y), extra={"error": str(exc)})
    dof = max(len(y) - 6, 1)
    chi2red = float(np.sum(((richards(h, *popt) - y) / (s if s is not None else 1.0)) ** 2) / dof)
    sc = _scale(err_mode, chi2red)
    hi_, ei = richards_infl(popt), _grad_err(richards_infl, popt, pcov) * sc
    hh_, eh = richards_half(popt), _grad_err(richards_half, popt, pcov) * sc
    return Fit("richards", float(hi_), float(ei), curve.L, chi2red, len(y), list(popt),
               extra={"h_half": float(hh_), "h_half_err": float(eh), "nu": float(popt[5]),
                      "rising": bool(rising), "err_mode": err_mode})


def fit_fd_peak(curve: Curve, window=(None, None), n_boot=400, seed=0) -> Fit:
    """Model-free locator: finite-difference |dy/dh| on midpoints, parabola through the
    three midpoints around the maximum -> vertex. Error = bootstrap over Gaussian
    resampling of y within its bars (falls back to half the grid spacing)."""
    c = curve.clip(window)
    h, y, ye = c.h, c.y, np.where(np.isfinite(c.ye), c.ye, 0.0)
    if len(y) < 4:
        return Fit("fd_peak", np.nan, np.nan, curve.L, n=len(y))

    def vertex(yy):
        hm = 0.5 * (h[1:] + h[:-1])
        sl = np.abs(np.diff(yy) / np.diff(h))
        i = int(np.argmax(sl))
        if i == 0 or i == len(sl) - 1:
            return hm[i]
        x, v = hm[i - 1:i + 2], sl[i - 1:i + 2]
        a, b, _ = np.polyfit(x, v, 2)
        return float(np.clip(-b / (2 * a), x[0], x[-1])) if a < 0 else hm[i]

    h_c = vertex(y)
    rng = np.random.default_rng(seed)
    boots = np.array([vertex(y + rng.normal(0, 1, len(y)) * ye) for _ in range(n_boot)])
    step = float(np.median(np.diff(h)))
    # bootstrap sd, floored at a quarter grid step: a 3-point parabola cannot resolve
    # the vertex better than that, whatever the bars say
    err = max(float(np.std(boots)) if np.any(ye > 0) else 0.5 * step, 0.25 * step)
    return Fit("fd_peak", float(h_c), err, curve.L, np.nan, len(y),
               extra={"grid_step": step, "boot_sd": float(np.std(boots))})


def locate_all(curve: Curve, window=(None, None), err_mode="pdg"):
    """All locators on one curve -> {method: Fit}."""
    return {"logistic": fit_logistic(curve, err_mode, window),
            "richards": fit_richards(curve, err_mode, window),
            "fd_peak": fit_fd_peak(curve, window)}


def combine_default(fits: dict, s2_fit=None, central="richards"):
    """Default per-L marker policy: central value = the `central` locator's inflection
    ("richards" — asymmetric rise, the OBC curves are visibly skewed; falls back to
    "logistic" if Richards did not converge); error = PDG-inflated statistical error (+)
    in quadrature the half-spread of every other converged locator (logistic/Richards
    inflection, finite-difference peak, S2 inflection) as the locator systematic.
    Returns (h_c, err, {"stat": ..., "syst": ..., "central": ..., "spread_over": [...]})."""
    order = [central] + [m for m in ("richards", "logistic", "fd_peak") if m != central]
    main = next((fits[m] for m in order if m in fits and fits[m].ok()), None)
    if main is None:
        return np.nan, np.nan, {}
    others = [f.h_c for f in fits.values() if f.ok() and f is not main]
    if s2_fit is not None and s2_fit.ok():
        others.append(s2_fit.h_c)
    vals = np.array([main.h_c, *others])
    syst = 0.5 * (vals.max() - vals.min()) if len(vals) > 1 else 0.0
    err = float(np.hypot(main.h_c_err, syst))
    return float(main.h_c), err, {"stat": float(main.h_c_err), "syst": float(syst),
                                  "central": main.method, "spread_over": [float(v) for v in vals]}


def pairwise_crossings(fits: dict, window):
    """Crossing points of pairs of fitted logistic curves inside `window`
    -> {(L1, L2): h}. NaN where the two fits do not cross in the window."""
    Ls = sorted(L for L, f in fits.items() if f.ok() and f.method == "logistic")
    lo, hi = window
    out = {}
    for i, L1 in enumerate(Ls):
        for L2 in Ls[i + 1:]:
            p1, p2 = fits[L1].popt, fits[L2].popt
            g = lambda x: logistic(x, *p1) - logistic(x, *p2)     # noqa: E731
            xs = np.linspace(lo, hi, 400)
            v = g(xs)
            idx = np.where(np.sign(v[1:]) != np.sign(v[:-1]))[0]
            out[(L1, L2)] = float(brentq(g, xs[idx[0]], xs[idx[0] + 1])) if len(idx) else np.nan
    return out


# ----------------------------------------------------------------------------- FSS
def fss_fit(Ls, hc, hce, x=1.0, err_mode="pdg"):
    """Weighted linear fit h_c(L) = h_inf + a * L^-x at fixed exponent x."""
    Ls, hc, hce = map(lambda v: np.asarray(v, float), (Ls, hc, hce))
    keep = np.isfinite(hc) & np.isfinite(hce) & (hce > 0)
    Ls, hc, hce = Ls[keep], hc[keep], hce[keep]
    n = len(Ls)
    if n < 2:
        return dict(x=x, h_inf=np.nan, h_inf_err=np.nan, a=np.nan, a_err=np.nan,
                    chi2red=np.nan, n=n, Ls=Ls.tolist(), model="h_inf + a*L^-x")
    X = Ls ** (-x)
    A = np.vstack([np.ones(n), X]).T / hce[:, None]
    b = hc / hce
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    resid = (hc - (coef[0] + coef[1] * X)) / hce
    chi2red = float(np.sum(resid ** 2) / (n - 2)) if n > 2 else np.nan
    sc = _scale(err_mode, chi2red) if n > 2 else 1.0
    return dict(x=float(x), h_inf=float(coef[0]), h_inf_err=float(np.sqrt(cov[0, 0]) * sc),
                a=float(coef[1]), a_err=float(np.sqrt(cov[1, 1]) * sc), chi2red=chi2red,
                n=n, Ls=Ls.tolist(), model="h_inf + a*L^-x", err_mode=err_mode)


def fss_free(Ls, hc, hce, x0=1.0, bounds=(0.3, 4.0)):
    """h_c(L) = h_inf + a * L^-x with x free. Needs >= 4 sizes for a meaningful
    error (3 sizes: exactly determined, flagged)."""
    Ls, hc, hce = map(lambda v: np.asarray(v, float), (Ls, hc, hce))
    keep = np.isfinite(hc) & np.isfinite(hce) & (hce > 0)
    Ls, hc, hce = Ls[keep], hc[keep], hce[keep]
    n = len(Ls)
    if n < 3:
        return dict(x=np.nan, h_inf=np.nan, h_inf_err=np.nan, n=n, note="need >= 3 sizes")
    f0 = fss_fit(Ls, hc, hce, x0)
    model = lambda L, h_inf, a, x: h_inf + a * L ** (-x)          # noqa: E731
    try:
        popt, pcov = curve_fit(model, Ls, hc, p0=[f0["h_inf"], f0["a"], x0], sigma=hce,
                               absolute_sigma=True, bounds=([-np.inf, -np.inf, bounds[0]],
                                                            [np.inf, np.inf, bounds[1]]), maxfev=20000)
    except Exception as exc:                                     # noqa: BLE001
        return dict(x=np.nan, h_inf=np.nan, h_inf_err=np.nan, n=n, note=str(exc))
    chi2red = float(np.sum(((model(Ls, *popt) - hc) / hce) ** 2) / (n - 3)) if n > 3 else np.nan
    sc = _scale("pdg", chi2red) if n > 3 else 1.0
    e = np.sqrt(np.abs(np.diag(pcov))) * sc
    return dict(x=float(popt[2]), x_err=float(e[2]), h_inf=float(popt[0]), h_inf_err=float(e[0]),
                a=float(popt[1]), a_err=float(e[1]), chi2red=chi2red, n=n, Ls=Ls.tolist(),
                model="h_inf + a*L^-x (x free)", note="exactly determined" if n == 3 else "")


def fss_sweep(Ls, hc, hce, xs):
    """FSS at several fixed exponents; the spread of h_inf is the exponent systematic."""
    return [fss_fit(Ls, hc, hce, x) for x in xs]


# ----------------------------------------------------------------------------- records
def record_quality(spec, rows, fss):
    """Fit-quality gate used by the phase-diagram notebook: every size converged with h_c
    inside the interior of the fit window, logistic rise amplitude > 0.2 where an `amp`
    is recorded (rows without `amp`, e.g. energy-crossing records, are not judged on it),
    and >= 3 sizes with a finite extrapolation."""
    lo, hi = spec.get("window", (-np.inf, np.inf))
    pad = 0.1 * (hi - lo) if np.isfinite(hi - lo) else 0.0
    conv = [np.isfinite(r.get("h_c", np.nan)) and lo + pad < r["h_c"] < hi - pad for r in rows]
    amps = [abs(r["amp"]) > 0.2 for r in rows if r.get("amp") is not None]
    h_inf = fss.get("h_inf", np.nan)
    return {"n_L": len(rows), "all_converged": bool(all(conv)) if rows else False,
            "rise_ok": bool(all(amps)) if amps else True,
            "fss_ok": bool(len(rows) >= 3 and h_inf is not None and np.isfinite(h_inf))}


def make_record(tag, spec, rows, fss_main, sweep, free=None, notes=""):
    """Serializable transition record for results/transitions/<tag>.json.

    rows: per-L dicts {L, h_c, h_c_err, locators: {method: (h_c, err, chi2red)}, ...};
    any extra per-L keys (e.g. "secondary" locator dicts) pass through untouched.
    spec keys passed through: kind ("topo-trivial" default | "trivial-trivial"), obs,
    lane, window, x_main, xs, secondary. Adds syst_exponent (half-spread of h_inf over the
    exponent battery `sweep`) and the `quality` gate.
    """
    fixed_name, fixed_val = spec["fixed"]
    clean = lambda d: None if d is None else {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}  # noqa: E731
    hinfs = [s_["h_inf"] for s_ in sweep if s_.get("h_inf") is not None and np.isfinite(s_["h_inf"])]
    syst = 0.5 * (max(hinfs) - min(hinfs)) if len(hinfs) >= 2 else None
    rec = {
        "tag": tag, "hy": spec["hy"], "fixed": {fixed_name: fixed_val}, "sweep": spec["sweep"],
        "order": spec.get("order"), "obs": spec.get("obs"), "lane": spec.get("lane", "prod"),
        "kind": spec.get("kind", "topo-trivial"),
        "per_L": rows, "fss": clean(fss_main), "fss_sweep": [clean(s_) for s_ in sweep], "fss_free": clean(free),
        "syst_exponent": syst, "quality": record_quality(spec, rows, fss_main),
        "point": {"hy": spec["hy"], fixed_name: fixed_val}, "notes": notes,
    }
    for k in ("window", "x_main", "xs", "secondary"):
        if k in spec:
            rec[k] = list(spec[k]) if isinstance(spec[k], tuple) else spec[k]
    return rec


def save_record(rec, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1, default=_json_default) + "\n")
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def load_records(dirpath, hy=None, lane=None):
    """All transition JSONs in a dir, optionally filtered by hy and lane."""
    recs = [json.loads(f.read_text()) for f in sorted(Path(dirpath).glob("*.json"))]
    if hy is not None:
        recs = [r for r in recs if abs(r["hy"] - hy) < 1e-9]
    if lane is not None:
        recs = [r for r in recs if r.get("lane", "prod") == lane]
    return recs


# ----------------------------------------------------------------------------- plotting helpers
def plasma_by_L(Ls):
    """House palette: plasma keyed by L (0.15/0.5/0.8 for L=4/5/6; spread for other sets)."""
    import matplotlib.pyplot as plt
    fixed = {4: 0.15, 5: 0.5, 6: 0.8}
    if set(Ls) <= set(fixed):
        return {L: plt.cm.plasma(fixed[L]) for L in Ls}
    Ls = sorted(Ls)
    pos = np.linspace(0.12, 0.85, len(Ls)) if len(Ls) > 1 else [0.5]
    return {L: plt.cm.plasma(p) for L, p in zip(Ls, pos)}


def openax(ax):
    ax.spines[["top", "right"]].set_visible(False)
