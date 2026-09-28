#!/bin/bash
# Spot-check N random RUNNING p3d_ chain jobs (2-hourly tick, user 2026-09-23): the point being trained, its
# latest step / E / Vscore, the drift and spread of E over its last <=50 steps, and its guard rollbacks.
# A healthy link: drift ~0 (|drift| << spread), a few rollbacks at most.
#   bash nersc/phase3d_sample_jobs.sh [N=3]
N="${1:-3}"
squeue -u "$USER" -h -t R -o "%i %j" | grep " p3d_" | shuf -n "$N" | while read -r id name; do
  f="${REPO:-$HOME/toric-code-nqs}/slurm_logs/${name}-${id}.out"
  [ -f "$f" ] || { echo "SAMPLE $name ($id): no log at $f"; continue; }
  awk -v name="$name" '
    /^\[sweep\] \([0-9]+\/[0-9]+\) ===/ { pt = $2 " " $4; n = 0; rb = 0; next }
    /rollback #/ { rb++ }
    /^  step +[0-9]+\/[0-9]+:/ { n++; e[n] = $5; v = $NF; st = $2 }
    END {
      if (n < 4) { printf "SAMPLE %s: point %s, %d steps so far, rollbacks %d\n", name, pt, n, rb; exit }
      k = (n < 50) ? n : 50; h = int(k / 2); a = 0; b = 0; sd = 0
      for (i = n - k + 1; i <= n - k + h; i++) a += e[i]
      for (i = n - k + h + 1; i <= n; i++) b += e[i]
      a /= h; b /= (k - h)
      for (i = n - k + h + 1; i <= n; i++) sd += (e[i] - b) ^ 2
      printf "SAMPLE %s: point %s, step %s E=%.2f Vscore=%s | last-%d drift %+.2f sd %.2f | rollbacks %d\n",
             name, pt, st, e[n], v, k, b - a, sqrt(sd / (k - h)), rb
    }' "$f"
done
