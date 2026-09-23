#!/bin/bash
# Run every tests/test_*.py against THIS checkout (PYTHONPATH = repo root), one
# PASS/FAIL/SKIPPED line per file; exits nonzero if any test fails.
#   tests/run_all.sh [python]        (default: `python` on PATH)
# NB: the tests build small 3D lattices and a few JAX kernels -- run them where
# the project's compute rules allow (see CLAUDE.md), e.g. a cluster debug node.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${1:-python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOGS=$(mktemp -d)
cd "$ROOT/tests" || exit 1
n_pass=0; n_fail=0; n_skip=0
for t in test_*.py; do
  if [ "$t" = test_firstorder_fit.py ] && [ ! -f "$ROOT/analysis/scripts/transition_fit.py" ]; then
    echo "SKIPPED  $t (needs analysis/scripts/transition_fit.py)"; n_skip=$((n_skip + 1)); continue
  fi
  t0=$SECONDS
  if "$PY" -u "$t" > "$LOGS/$t.log" 2>&1; then
    echo "PASS     $t ($((SECONDS - t0)) s)"; n_pass=$((n_pass + 1))
  else
    echo "FAIL     $t ($((SECONDS - t0)) s) -- last lines of $LOGS/$t.log:"
    tail -n 15 "$LOGS/$t.log" | sed 's/^/    /'; n_fail=$((n_fail + 1))
  fi
done
echo "== $n_pass passed, $n_fail failed, $n_skip skipped (logs: $LOGS)"
[ "$n_fail" -eq 0 ]
