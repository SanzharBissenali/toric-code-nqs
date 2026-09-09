"""Regroup phase3d campaign W&B runs by (hy, cut, L), read from each run's own
config.out_dir -- generalized from ~/wb_regroup.py (the hy_cuts-specific,
hardcoded up/right version). A FORCE sync resets a run's group (see
notes/transition_mapping_recipes.md #0), so this is meant to run after every
nersc/sync_wandb.sh (the phase3d_scrontab.txt cron pairs them).

    python nersc/wb_regroup.py
    PROJECT=tc3d-phase3d ENTITY=... python nersc/wb_regroup.py
"""
import os

import wandb

PROJECT = os.environ.get("PROJECT", "tc3d-phase3d")
ENTITY = os.environ.get("ENTITY", "models-california-institute-of-technology-caltech")


def group_for(out_dir):
    """out_dir = .../phase3d/hy{hy}/{cut}/L{L} -> group "hy{hy}/{cut}/L{L}"."""
    if "/phase3d/" not in out_dir:
        return None
    return out_dir.split("/phase3d/", 1)[1].rstrip("/")


def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", per_page=500)
    n = 0
    for r in runs:
        group = group_for((r.config or {}).get("out_dir", "") or "")
        if group and r.group != group:
            r.group = group
            r.update()
            n += 1
    print("regrouped:", n)


if __name__ == "__main__":
    main()
