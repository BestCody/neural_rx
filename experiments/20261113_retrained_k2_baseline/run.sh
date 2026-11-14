#!/usr/bin/env bash
# Retrained two-iteration baseline: ./run.sh start | status
# Training uses GPU 0 (learner) and GPU 1 (data and gradient replica); development checks run on GPU 1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
MODELS=$ROOT/models/retrained_k2_baseline_20261113
case "${1:-status}" in
  start)
    mkdir -p "$HERE/logs"
    tmux new-session -d -s k2_baseline "cd '$ROOT' && '$PY' -u '$HERE/run.py' >> '$HERE/logs/train.log' 2>&1"
    echo "started tmux session k2_baseline" ;;
  status)
    tmux has-session -t k2_baseline 2>/dev/null && echo "running" || echo "not running"
    [ -f "$MODELS/progress.json" ] && "$PY" -c "import json; r=json.load(open('$MODELS/progress.json')); print('update', r['update'], 'loss', round(r['loss'], 4), 'lr', r['learning_rate'])"
    for f in "$HERE"/results/development_*/results.json; do
      [ -f "$f" ] && "$PY" -c "import json; r=json.load(open('$f')); s=r['summary']['candidate_persistent']; print('development', r['update'], round(s['core_tbler'], 4))"
    done
    [ -f "$HERE/results/best.json" ] && cat "$HERE/results/best.json" || true ;;
esac
