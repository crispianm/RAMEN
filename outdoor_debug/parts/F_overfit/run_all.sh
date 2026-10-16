#!/bin/bash
# acquire GPU_LOCK per run (id F_overfit), release always
LOCK=/home/wg19671/.claude/jobs/a0d391cb/tmp/GPU_LOCK
PY=/home/wg19671/.venvs/ramen/bin/python
cd /home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/F_overfit
run() {
  for i in $(seq 1 60); do
    if [ ! -s "$LOCK" ]; then echo F_overfit > "$LOCK"; sleep 2; [ "$(cat $LOCK)" = F_overfit ] && break; fi
    sleep 60
  done
  [ "$(cat $LOCK)" = F_overfit ] || { echo "no lock for $*"; return 1; }
  PYTHONPATH=. $PY overfit.py "$@" 2>&1 | grep -v -i warn
  : > "$LOCK"
}
trap '[ "$(cat $LOCK)" = F_overfit ] && : > $LOCK' EXIT
if [ "$1" = smoke ]; then run vkitti 1 1 3; exit; fi
for kind in vkitti tartanair; do
  run $kind 1 0 300; run $kind 1 1 300; run $kind 10 0 600; run $kind 10 1 600
done
