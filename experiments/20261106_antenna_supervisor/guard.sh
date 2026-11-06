#!/usr/bin/env bash
# Stop the antenna runs if their host memory approaches what a shared machine can spare.
cd "$(dirname "$0")"
LIMIT_KB=$((300*1024*1024))
while true; do
    pids=$(pgrep -f '[r]un.py (natural|controlled|schedule)')
    [ -z "$pids" ] && { echo "$(date) no antenna processes"; exit 0; }
    total=$(ps -o rss= -p $pids | awk '{s+=$1} END {print s+0}')
    echo "$(date +%H:%M:%S) rss_gb=$((total/1024/1024)) files=$(find results -name '*.npz' | wc -l)"
    if [ "$total" -gt "$LIMIT_KB" ]; then echo "memory limit exceeded; stopping"; kill $pids; exit 1; fi
    sleep 60
done
