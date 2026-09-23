#!/bin/zsh
# One campaign tick: pull -> STATUS/summary/viewer rebuild -> one PULL line + numeric checks
# (phase3d_tick_checks.py); then a cluster watch line + 3 random running-job samples (phase3d_sample_jobs.sh).
# Usage (zsh, from anywhere): DO_PULL=1 SINCE_MIN=35 bash analysis/scripts/phase3d_local_tick.sh
#   DO_PULL=0 -> cluster watch line only. Armed as a 30-min Monitor in the Claude session (re-arm on expiry).
W=/Users/sanzhar123/Desktop/toric-code-nqs-p3d/integration
R=/Users/sanzhar123/Desktop/toric-code-nqs/results/phase3d
DATA=/Users/sanzhar123/Desktop/toric-code-nqs/data/tc_nqs/phase3d
V="${VIEWER_DIR:-/tmp/phase3d_viewer}"          # built viewer + per-plane JSONs (publish $V/phase3d_viewer.html to the artifact)
cd $W || exit 1
mkdir -p $V
if [ "${DO_PULL:-1}" = "1" ]; then
  LOCAL_RESULTS=$R LOCAL_DATA=$DATA bash analysis/scripts/pull_phase3d.sh >/dev/null 2>&1 || echo "PULL FAILED $(date +%H:%M) (ssh/cert?)"
  n=$(find $R -name 'gridinv_dual_*.json' ! -name '*snapshots*' ! -name '*finaleval*' ! -name '*curve*' | wc -l | tr -d ' ')
  new=$(find $R -name 'gridinv_dual_*.json' ! -name '*snapshots*' ! -name '*finaleval*' ! -name '*curve*' -mmin -${SINCE_MIN:-35} | wc -l | tr -d ' ')
  .venv/bin/python analysis/scripts/phase3d_status.py --root $R --out $R/STATUS.md >/dev/null 2>&1
  .venv/bin/python analysis/scripts/phase3d_status.py --export-summary --root $R --out $R/summary.json >/dev/null 2>&1
  rm -f $V/viewer_hy*.json; planes=""
  for d in $R/hy*; do hy=${d#$R/hy}; .venv/bin/python analysis/scripts/phase3d_status.py --export-viewer $hy --root $R --curves-root $DATA --out $V/viewer_hy$hy.json >/dev/null 2>&1 && planes="$planes $hy"; done
  [ -d $R/ycuts ] && .venv/bin/python analysis/scripts/phase3d_status.py --export-viewer y --root $R --curves-root $DATA --out $V/viewer_hyy.json >/dev/null 2>&1 && planes="$planes y"
  .venv/bin/python analysis/scripts/phase3d_viewer_build.py $V/phase3d_viewer.html $V/viewer_hy*.json >/dev/null 2>&1 && cp $V/phase3d_viewer.html $R/viewer.html
  echo "PULL $(date +%H:%M): finals=$n new_last${SINCE_MIN:-35}m=$new planes=[$planes ]"
  .venv/bin/python analysis/scripts/phase3d_tick_checks.py --root $R --since-min ${SINCE_MIN:-35}   # numeric health + y-pol anchor gate
fi
ssh -o ConnectTimeout=30 perlmutter 'q=$(squeue -u sanzharb -h -o "%T %j" | grep -v "p3d-driver\|p3d-wandb"); echo "WATCH $(date +%H:%M): running=$(echo "$q" | grep -c ^RUNNING) pending=$(echo "$q" | grep -c ^PENDING) | failed_last6h=$(sacct -X -u sanzharb -S $(date -d "6 hours ago" +%Y-%m-%dT%H:%M) -s F,TO,OOM,NF -n -o JobID,JobName%22,State | grep -v "p3d-driver\|p3d-wandb" | awk "{print \$1\":\"\$2\":\"\$3}" | tr "\n" " ")"; find ~/toric-code-nqs -maxdepth 2 -name "p3d_*.out" -mmin -360 2>/dev/null | xargs -r grep -l "GENUINE DIVERGENCE\|CHAIN STOPPED" 2>/dev/null | xargs -r -n1 basename | sed "s/^/  DIV /"; for hy in y; do f=$PSCRATCH/tc_nqs/phase3d/driver_hy$hy.log; [ -f $f ] && echo "  driver hy$hy: $(grep "\[launch\].*job(s)\|\[watch\] flagged" $f | tail -2 | tr "\n" " ")"; done; bash ~/toric-code-nqs/nersc/phase3d_sample_jobs.sh 3' 2>&1 | grep -v "^$"
