"""Embed per-plane viewer JSON (phase3d_status.py --export-viewer) into the viewer page.

    python analysis/scripts/phase3d_viewer_build.py OUT.html IN_hy0.0.json IN_hy0.2.json ...
    python analysis/scripts/phase3d_viewer_build.py --standalone OUT.html IN_*.json   # full HTML doc for a local browser

The template analysis/viewer/phase3d_viewer.html carries a `/*__DATA__*/[]` placeholder;
the built page is self-contained (data inline) and is what gets published as the Artifact.
`/*__EXTRA__*/{}` takes analysis/viewer/phase3d_extras.json (locators from outside
results/phase3d -- the hy_cuts_L4 h_x=0.2 / h_z=0.1 cuts and the h_y-axis point -- for the
phase-diagram view); `--extra FILE` overrides it.
"""
import json, sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "viewer" / "phase3d_viewer.html"
EXTRA = Path(__file__).resolve().parents[1] / "viewer" / "phase3d_extras.json"


SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n{head}</head>\n<body>\n{body}</body>\n</html>\n')


def build(out: Path, inputs, standalone=False, extra=EXTRA):
    planes = sorted((json.loads(Path(f).read_text()) for f in inputs),
                    key=lambda p: (isinstance(p["hy"], str), p["hy"] if not isinstance(p["hy"], str) else 0))   # y-cuts last
    esc = lambda o: json.dumps(o, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    extra_obj = json.loads(Path(extra).read_text()) if extra and Path(extra).exists() else {}
    html = TEMPLATE.read_text().replace("/*__DATA__*/[]", esc(planes), 1).replace("/*__EXTRA__*/{}", esc(extra_obj), 1)
    if standalone:                       # the Artifact host adds the skeleton itself; a local file needs it
        cut = html.index("</style>") + len("</style>")
        html = SKELETON.format(head=html[:cut], body=html[cut:])
    out.write_text(html)
    print(f"[viewer] {out}  planes={[p['hy'] for p in planes]}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    args = sys.argv[1:]
    standalone = "--standalone" in args
    extra = EXTRA
    if "--extra" in args:
        i = args.index("--extra"); extra = args[i + 1]; del args[i:i + 2]
    args = [a for a in args if a != "--standalone"]
    build(Path(args[0]), args[1:], standalone=standalone, extra=extra)
