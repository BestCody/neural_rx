#!/usr/bin/env bash
# Stop this sweep if its host memory approaches what a shared machine can spare.
LIMIT_KB=$((300*1024*1024))
while true; do
    pids=$(pgrep -f '[n]eural_rx.cli.sweep --gpu . run')
    [ -z "$pids" ] && { echo "$(date) no sweep processes"; exit 0; }
    total=$(ps -o rss= -p $pids | awk '{s+=$1} END {print s+0}')
    echo "$(date +%H:%M:%S) rss_gb=$((total/1024/1024)) files=$(find results/sweep_high -name '*.npz' | wc -l)"
    if [ "$total" -gt "$LIMIT_KB" ]; then echo "memory limit exceeded; stopping"; kill $pids; exit 1; fi
    sleep 60
done
