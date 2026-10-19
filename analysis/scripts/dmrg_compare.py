"""Join final-state NQS run JSONs to the L=3 OBC DMRG reference and tabulate the gaps.

    python analysis/scripts/dmrg_compare.py --runs results/dmrg_L3_obc/btc results/dmrg_L3_obc/ftc \
        --ref results/dmrg_L3_obc/dmrg_L3_obc_reference.json [--model bosonic] [--csv out.csv] [--json out.json]
    python analysis/scripts/dmrg_compare.py --selftest

Run JSON schema: top-level name/config/observables (tc3d.train). Join key: model + (hx, hy, hz)
within 1e-9 against rows[].ours of the reference (already in tc3d conventions). Importable:
`load_table(runs, ref, model=None) -> list[dict]`. Observable deltas are NQS - DMRG.
"""
import argparse, csv, json, sys, tempfile
from pathlib import Path

TOL = 1e-9
_SKIP = (".curve", ".snapshots", "finaleval", ".fidelity", ".step")
OBS = (("sx", "sx_mean"), ("sy", "sy_mean"), ("sz", "sz_mean"), ("A", "A_v_mean"), ("B", "B_p_mean"))


def matches(row):
    """TODO(user): acceptance rule for 'NQS agrees with DMRG'; return True/False (None = undecided).

    `row` is one table dict (keys: dE, E_err, E_ref, rel, pull, dsx..dB, Vscore, diverged, ...).
    Candidate: abs(row["dE"]) < max(3 * row["E_err"], 1e-5 * abs(row["E_ref"]))
    """
    return None


def arm_label(cfg):
    if cfg.get("model", "bosonic") == "bosonic":
        return "dual" if cfg.get("dual_basis") else "primal"
    sf = cfg.get("sign_frame")
    if sf not in (None, "none"):
        return f"A:{sf}"
    return "B:anaC" if cfg.get("phase_head_frozen") else "C:blind"


def load_ref(path):
    return json.loads(Path(path).read_text())["rows"]


def _find_ref(rows, model, hx, hy, hz):
    for r in rows:
        o = r["ours"]
        if r["model"] == model and all(abs(o[k] - v) < TOL for k, v in (("hx", hx), ("hy", hy), ("hz", hz))):
            return o
    return None


def _run_files(d):
    for p in sorted(Path(d).rglob("*.json")):
        if not any(s in p.name for s in _SKIP):
            yield p


def _row(run, ref_rows):
    cfg, obs = run["config"], run["observables"]
    model = cfg.get("model", "bosonic")
    hx, hy, hz = (float(cfg.get(k, 0.0)) for k in ("hx", "hy", "hz"))
    ref = _find_ref(ref_rows, model, hx, hy, hz)
    row = {"model": model, "arm": arm_label(cfg), "hx": hx, "hy": hy, "hz": hz, "name": run.get("name"),
           "E_nqs": obs.get("E0"), "E_err": obs.get("E_err"), "E_err_scatter": obs.get("E_err_scatter"),
           "E_ref": ref["E"] if ref else None, "Vscore": obs.get("Vscore"),
           "diverged": bool(run.get("diverged", False))}
    row["dE"] = row["rel"] = row["pull"] = None
    if ref and row["E_nqs"] is not None:
        row["dE"] = row["E_nqs"] - ref["E"]
        row["rel"] = row["dE"] / abs(ref["E"])
        if row["E_err"]:
            row["pull"] = row["dE"] / row["E_err"]
    for short, key in OBS:
        row["d" + short] = obs[key] - ref[key] if ref and obs.get(key) is not None and key in ref else None
        row[short + "_nqs"] = obs.get(key)
    row["match"] = matches(row)
    return row


def load_table(runs, ref, model=None):
    """runs: dir/path or list of them; ref: reference JSON path. Rows sorted by (model, arm, hx, hy, hz).
    Runs without a reference row are kept with E_ref/dE = None."""
    runs = [runs] if isinstance(runs, (str, Path)) else list(runs)
    ref_rows = load_ref(ref)
    out = []
    for d in runs:
        for p in _run_files(d):
            try:
                run = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if not all(k in run for k in ("name", "config", "observables")):
                continue
            if model and run["config"].get("model", "bosonic") != model:
                continue
            out.append(_row(run, ref_rows))
    return sorted(out, key=lambda r: (r["model"], r["arm"], r["hx"], r["hy"], r["hz"]))


# (key, width, format spec); left-aligned text columns have a negative width
_COLS = [("model", -9, ""), ("arm", -8, ""), ("hx", 6, ".2f"), ("hy", 5, ".2f"), ("hz", 5, ".2f"),
         ("E_nqs", 11, ".5f"), ("E_err", 8, ".1e"), ("E_ref", 11, ".5f"), ("dE", 10, "+.2e"),
         ("rel", 9, "+.2e"), ("pull", 7, "+.1f"), ("dsx", 9, "+.1e"), ("dsy", 9, "+.1e"),
         ("dsz", 9, "+.1e"), ("dA", 9, "+.1e"), ("dB", 9, "+.1e"), ("Vscore", 9, ".1e"),
         ("diverged", 8, ""), ("match", 5, "")]


def _cell(k, v, spec):
    if k == "match":
        return "?" if v is None else ("Y" if v else "N")
    if k == "diverged":
        return "DIV" if v else "-"
    return "-" if v is None else format(v, spec)


def format_table(rows):
    lines = [" ".join(f"{k:{'<' if w < 0 else '>'}{abs(w)}}" for k, w, _ in _COLS)]
    for r in rows:
        lines.append(" ".join(f"{_cell(k, r.get(k), sp):{'<' if w < 0 else '>'}{abs(w)}}" for k, w, sp in _COLS))
    return "\n".join(lines)


def selftest():
    ref = {"rows": [{"model": "bosonic", "dmrg": {}, "ours": {"hx": -0.1, "hy": 0.0, "hz": 0.0, "E": -63.1048,
                     "sx_mean": 0.02, "sy_mean": -0.0, "sz_mean": 0.0, "A_v_mean": 0.99, "B_p_mean": 1.0}}]}
    run = {"name": "fake", "diverged": False, "config": {"model": "bosonic", "dual_basis": True, "hx": -0.1, "hy": 0.0, "hz": 0.0},
           "observables": {"E0": -63.1047, "E_err": 1e-4, "Vscore": 1e-3, "sx_mean": 0.0205, "sz_mean": 0.001,
                           "A_v_mean": 0.989, "B_p_mean": 1.0}}
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "ref.json").write_text(json.dumps(ref))
        (t / "r").mkdir()
        (t / "r" / "fake.json").write_text(json.dumps(run))
        (t / "r" / "fake.curve.json").write_text(json.dumps({"step": []}))            # must be skipped
        orphan = {**run, "name": "orphan", "config": {**run["config"], "hx": -0.3}}  # no reference row
        (t / "r" / "orphan.json").write_text(json.dumps(orphan))
        rows = load_table(t / "r", t / "ref.json")
        assert len(rows) == 2, rows
        r = next(x for x in rows if x["name"] == "fake")
        assert r["arm"] == "dual" and abs(r["dE"] - 1e-4) < 1e-12 and abs(r["pull"] - 1.0) < 1e-6
        assert abs(r["dsx"] - 5e-4) < 1e-12 and r["dsy"] is None and r["match"] is None
        assert next(x for x in rows if x["name"] == "orphan")["E_ref"] is None
        assert load_table(t / "r", t / "ref.json", model="fermionic") == []
        assert arm_label({"model": "fermionic", "phase_head_frozen": True}) == "B:anaC"
        assert arm_label({"model": "fermionic", "sign_frame": "x"}) == "A:x"
        assert arm_label({"model": "fermionic"}) == "C:blind" and arm_label({"model": "bosonic"}) == "primal"
        print(format_table(rows))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", metavar="DIR"); ap.add_argument("--ref")
    ap.add_argument("--model", choices=["bosonic", "fermionic"])
    ap.add_argument("--csv"); ap.add_argument("--json"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.runs and a.ref):
        ap.error("--runs and --ref are required")
    rows = load_table(a.runs, a.ref, a.model)
    print(format_table(rows) if rows else "no run JSONs found")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
