"""
analysis/scripts/summarize_speed_bench.py
────────────────────────────────────────────────────────────────────────────
Tabulate bench_hy_speed.py JSONs (one per variant) into one markdown table:
per-stage median s/step, GPU peak memory, and the --microbench decomposition
(forward us/config, achieved TFLOP/s, E_loc-forward estimate, HLO census).

    python analysis/scripts/summarize_speed_bench.py results/speed_bench/*.json
"""
import json
import sys


def row(path):
    r = json.load(open(path))
    c = r["config"]
    lab, v = next(iter(r["variants"].items()))
    m = v["median_excl_compile"]
    mem = r.get("gpu_memory_stats") or {}
    peak = mem.get("peak_bytes_in_use")
    mb = r.get("microbench") or {}
    hlo = mb.get("hlo_census", {})
    return dict(
        L=c["L"], hy=c["hy"], impl=c.get("inv_impl", "conv"), dtype=c.get("compute_dtype") or "float64",
        solver=lab, chunk=c["chunk_size"], ns=c["n_samples"], n_params=r["n_params"],
        sample=m.get("sample"), grad=m.get("grad"), qgt=m.get("qgt"), total=m.get("total"),
        steps=len(v["timing_per_step"]) - 1,
        peak=None if peak is None else peak / 2**30,
        us=mb.get("fwd_us_per_config"), tf=mb.get("fwd_TFLOPs"),
        eloc=mb.get("eloc_forward_s_per_step_est"), conn=mb.get("get_conn_padded_s_per_step_est"),
        nconn=mb.get("n_conn"),
        hlo=f"conv {hlo.get('convolution(', '?')} / cudnn {hlo.get('cudnn', '?')} / dot {hlo.get(' dot(', '?')}",
    )


def f(x, p=1):
    return "—" if x is None else f"{x:.{p}f}"


def main(paths):
    rows = sorted((row(p) for p in paths), key=lambda r: (r["L"], r["hy"], r["dtype"], r["impl"], r["solver"], r["chunk"]))
    print("| L | hy | impl | compute | solver | chunk | n_s | n_params | sample | grad | qgt | **total s/step** | peak GiB | fwd us/cfg | TFLOP/s | E_loc fwd est | conn est | HLO (conv/cudnn/dot) |")
    print("|" + "---|" * 18)
    for r in rows:
        print(f"| {r['L']} | {r['hy']} | {r['impl']} | {r['dtype']} | {r['solver']} | {r['chunk']} | {r['ns']} | "
              f"{r['n_params']} | {f(r['sample'])} | {f(r['grad'])} | {f(r['qgt'], 2)} | **{f(r['total'])}** | "
              f"{f(r['peak'], 2)} | {f(r['us'], 2)} | {f(r['tf'], 2)} | {f(r['eloc'])} | {f(r['conn'], 2)} | {r['hlo']} |")


if __name__ == "__main__":
    main(sys.argv[1:])
