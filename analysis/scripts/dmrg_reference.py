"""Convert the collaborator's L=3 OBC DMRG pickles into a reference JSON in tc3d conventions.

DMRG convention (inferred from 2nd/3rd-order PT on all three axes, 2026-10-18):
    H_DMRG = -sum A_v - sum B_p + h . S,  S = sigma/2,  observables are <sigma> per edge.
tc3d convention: H = -sum A_v - sum B_p - h . sigma.  Hence
    hz_ours = +h/2, hy_ours = +h/2 (E even in these signs),
    hx_ours = -h/2 (the 8 weight-3 corner stars make E odd-sensitive at O(h^3), E3 = -/+ 3 h^3),
    <X>_ours = avgX, <Y>_ours = -avgY, <Z>_ours = -avgZ, <A>,<B>_ours = -avgA, -avgB.
Usage: python analysis/scripts/dmrg_reference.py DMRG_data -o results/dmrg_L3_obc/dmrg_L3_obc_reference.json
"""
import argparse, io, json, pickle, pickletools, sys
from pathlib import Path

_ALLOWED = {("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"), ("numpy", "dtype")}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_pickle(path):
    raw = Path(path).read_bytes()
    return _SafeUnpickler(io.BytesIO(raw)).load()


def convert(d, model):
    out = []
    for (hx, hy, hz), v in sorted(d.items()):
        v = [float(x) for x in v]
        row = {"model": model, "dmrg": {"hx": hx, "hy": hy, "hz": hz, "E": v[0], "avgX": v[1], "avgY": v[2], "avgZ": v[3]},
               "ours": {"hx": -hx / 2, "hy": hy / 2, "hz": hz / 2, "E": v[0], "sx_mean": v[1], "sy_mean": -v[2], "sz_mean": -v[3]}}
        if len(v) >= 6:
            row["dmrg"].update(avgA=v[4], avgB=v[5]); row["ours"].update(A_v_mean=-v[4], B_p_mean=-v[5])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dmrg_dir"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rows = []
    for model, fn in (("bosonic", "btc_measurements (1).pkl"), ("fermionic", "ftc_measurements (1).pkl")):
        rows += convert(load_pickle(Path(a.dmrg_dir) / fn), model)
    ref = {"source": "collaborator DMRG, L=3 OBC, N=54, 27 A_v + 36 B_p, E(h=0)=-63; received 2026-10-18",
           "convention": __doc__.strip(), "rows": rows}
    Path(a.out).write_text(json.dumps(ref, indent=1))
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
