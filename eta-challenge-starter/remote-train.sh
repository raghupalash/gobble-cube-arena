#!/usr/bin/env bash
set -euo pipefail

VPS="${VPS:-anton}"
REMOTE_DIR="${REMOTE_DIR:-/home/palash/gobblecube-arena/eta-challenge-starter}"

# Sync code (py files + zone centroids csv)
echo "[sync] pushing code to $VPS:$REMOTE_DIR"
rsync -av \
  --include="*.py" \
  --include="*.csv" \
  --include="requirements.txt" \
  --exclude="*" \
  . "$VPS:$REMOTE_DIR/"

# Auto-increment log number
NEXT_N=$(ssh "$VPS" "
  max=0
  for f in /tmp/full_train_*.log; do
    [ -e \"\$f\" ] || continue
    n=\${f##*full_train_}
    n=\${n%.log}
    [ \"\$n\" -gt \"\$max\" ] 2>/dev/null && max=\$n
  done
  echo \$((max + 1))
")

LOG="/tmp/full_train_${NEXT_N}.log"
echo "[train] starting full run -> $LOG"
ssh "$VPS" "cd $REMOTE_DIR && nohup .venv/bin/python3 baseline.py --full > $LOG 2>&1 &"
echo "[train] started. to tail: ssh $VPS 'tail -f $LOG'"
echo "[tail]  or run: ./remote-train.sh tail"

echo "$VPS $REMOTE_DIR $LOG" > .remote-last-run
