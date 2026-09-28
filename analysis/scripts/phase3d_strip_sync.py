"""Mirror the raw phase3d pull (gitignored archive) into the committed results/ tree.

A final JSON embeds its whole per-step learning curve (`curve`, ~260 KB at 500 steps), which
duplicates `<name>.curve.json` in data/tc_nqs/ and stays out of git (CLAUDE.md per-run commit
policy). pull_phase3d.sh rsyncs the raw plane JSONs into the archive; this writes each into
results/: finals minus `curve`, snapshot/finaleval JSONs byte-for-byte. The source mtime is kept
(phase3d_status stamps `data_as_of` from it), so only new or changed files are rewritten. A final
that does not parse (pulled mid-write) is skipped and retried on the next pull. Nothing is deleted:
parking is mirrored by `phase3d_reseed.py park` on BOTH trees, else the next pull restores the file.
Run per plane, as pull_phase3d.sh does:

    python analysis/scripts/phase3d_strip_sync.py data/archive/phase3d/hy1.0 results/phase3d/hy1.0
"""
import json
import os
import shutil
import sys
from pathlib import Path


def is_final(f: Path) -> bool:
    return (f.suffix == ".json" and f.name != "watch_state.json" and "finaleval" not in f.name
            and not f.name.endswith((".snapshots.json", ".curve.json")))


def strip_sync(raw, results) -> dict:
    raw, results = Path(raw), Path(results)
    n = {"stripped": 0, "copied": 0, "unchanged": 0, "unreadable": 0}
    for f in sorted(raw.rglob("*.json")):
        rel = f.relative_to(raw)
        if "wandb" in rel.parts or f.name.endswith(".curve.json"):
            continue                       # W&B run dirs; curves live in data/tc_nqs
        out = results / rel
        st = f.stat()
        if out.exists() and out.stat().st_mtime_ns == st.st_mtime_ns:
            n["unchanged"] += 1
            continue
        if is_final(f):
            try:
                d = json.loads(f.read_text())
            except json.JSONDecodeError:
                n["unreadable"] += 1       # train.py writes non-atomically: pulled mid-write, retry next pull
                continue
            d.pop("curve", None)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(d, indent=2))     # train.py's format; key order + floats round-trip
            n["stripped"] += 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)
            n["copied"] += 1
        os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns))
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(f"[strip-sync] {sys.argv[1]} -> {sys.argv[2]}: {strip_sync(*sys.argv[1:])}")
