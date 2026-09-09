"""Embed per-plane viewer JSON (phase3d_status.py --export-viewer) into the viewer page.

    python analysis/scripts/phase3d_viewer_build.py OUT.html IN_hy0.0.json IN_hy0.2.json ...
    python analysis/scripts/phase3d_viewer_build.py --standalone OUT.html IN_*.json   # full HTML doc for a local browser

The template analysis/viewer/phase3d_viewer.html carries a `/*__DATA__*/[]` placeholder;
the built page is self-contained (data inline) and is what gets published as the Artifact.
"""
import json, sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "viewer" / "phase3d_viewer.html"


SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n{head}</head>\n<body>\n{body}</body>\n</html>\n')


def build(out: Path, inputs, standalone=False):
    planes = sorted((json.loads(Path(f).read_text()) for f in inputs), key=lambda p: p["hy"])
    blob = json.dumps(planes, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/[]", blob, 1)
    if standalone:                       # the Artifact host adds the skeleton itself; a local file needs it
        cut = html.index("</style>") + len("</style>")
        html = SKELETON.format(head=html[:cut], body=html[cut:])
    out.write_text(html)
    print(f"[viewer] {out}  planes={[p['hy'] for p in planes]}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    args = sys.argv[1:]
    standalone = "--standalone" in args
    args = [a for a in args if a != "--standalone"]
    build(Path(args[0]), args[1:], standalone=standalone)
