"""Embed per-plane viewer JSON (phase3d_status.py --export-viewer) into the viewer page.

    python analysis/scripts/phase3d_viewer_build.py OUT.html IN_hy0.0.json IN_hy0.2.json ...

The template analysis/viewer/phase3d_viewer.html carries a `/*__DATA__*/[]` placeholder;
the built page is self-contained (data inline) and is what gets published as the Artifact.
"""
import json, sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "viewer" / "phase3d_viewer.html"


def build(out: Path, inputs):
    planes = sorted((json.loads(Path(f).read_text()) for f in inputs), key=lambda p: p["hy"])
    blob = json.dumps(planes, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/[]", blob, 1)
    out.write_text(html)
    print(f"[viewer] {out}  planes={[p['hy'] for p in planes]}  {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    build(Path(sys.argv[1]), sys.argv[2:])
