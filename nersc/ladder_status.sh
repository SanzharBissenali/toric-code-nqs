#!/bin/bash
# Compact live digest of the fermionic arch-ladder campaign: one line per run
# (last step + energy from its curve.json) plus alarm lines grepped from the
# tails of the Slurm logs. Intended for periodic remote polling.
DIR="${OUT_DIR:-$PSCRATCH/tc_nqs/fermionic_ladder}"
LOGDIR="${LOGDIR:-$HOME/toric-code-nqs-ladder}"

python3 - "$DIR" <<'EOF'
import json, glob, sys, os
d = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(d, 'ladder_*.curve.json'))):
    name = os.path.basename(p)[:-len('.curve.json')]
    try:
        c = json.load(open(p))['curve']
    except Exception as e:
        rows.append(f'{name}: UNREADABLE ({e})'); continue
    if not c.get('step'):
        rows.append(f'{name}: no steps yet'); continue
    E, s = c['energy'], c['step'][-1]
    im = c.get('energy_im') or [0.0]
    rows.append(f'{name}: step {s} E={E[-1]:.6f} imE={im[-1]:.1e}')
print('\n'.join(rows) if rows else 'no curves yet')
EOF

# alarms: NaN, guard activity, crashes, and the silent no-op-sign-head signature
for f in "$LOGDIR"/tc-gridinv-*.out; do
  [ -e "$f" ] || continue
  a=$(tail -n 300 "$f" | grep -E 'nan|NaN|Traceback|rollback|exceeded max_rollbacks|not found; cold start|DivergenceError' | tail -3)
  [ -n "$a" ] && printf 'ALARM %s:\n%s\n' "$(basename "$f")" "$a"
done
exit 0
