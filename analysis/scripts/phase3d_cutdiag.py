"""Tabulate both chain branches of one cut from the built viewer JSONs (campaign tick helper).

    python analysis/scripts/phase3d_cutdiag.py <viewer_dir> <plane> <cut_id> [obs] [L]
    e.g.  ... $VIEWER_DIR 1.4 electric_hx0            (obs defaults to the swept field's magnetization)
Columns per point: h, E0, obs, <B_p>, Vscore, guard rollbacks; '!' = above the h=0 bound or diverged,
'W' = the point wins the lower-energy comparison. Header: the cut's jump / net-crossing / loop locators.
"""
import json
import sys


def pick(d, L):
    d = (d or {}).get(str(L)) or {}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
            if k in ("h_c", "err", "obs", "ok", "lo", "hi", "merged")}


def main(argv):
    vdir, plane, cid = argv[:3]
    obs = argv[3] if len(argv) > 3 else None
    L = int(argv[4]) if len(argv) > 4 else 4
    with open(f"{vdir}/viewer_hy{plane}.json") as fh:
        cuts = json.load(fh)["cuts"]
    c = next((c for c in cuts if c["id"] == cid), None)
    if c is None:
        raise SystemExit(f"no cut {cid!r} on plane {plane}; have {[c['id'] for c in cuts]}")
    obs = obs or {"hz": "sz", "hx": "sx", "hy": "sy"}[c["sweep"]]
    print(f"{cid} {'first-order' if c.get('first_order') else ''} | jump {pick(c.get('jump'), L)} | "
          f"crossing {pick(c.get('crossing'), L)} | loop {pick(c.get('loop'), L)}")
    pts = [p for p in c["points"] if p["L"] == L]
    for br in ("up", "dn"):
        print(f"  {br}: h, E0, {obs}, B_p, Vscore, rollbacks")
        for p in sorted((p for p in pts if p["branch"].startswith(br)), key=lambda p: p["h"]):
            flag = "!" if (p.get("above_bound") or p.get("diverged")) else " "
            print(f"   {flag}{p['h']:<6g} {p['E0']:>10.3f} {p.get(obs, float('nan')):>7.3f} {p.get('B_p', float('nan')):>6.3f} "
                  f"{p.get('Vscore') or float('nan'):>7.3f} {p.get('n_rollbacks') or 0:>3} {'W' if p.get('winner') else ''}")


if __name__ == "__main__":
    main(sys.argv[1:])
