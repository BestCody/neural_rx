#!/usr/bin/env bash
# End the run on Modal at a given time and continue it on this server's two GPUs.
#   ./bring_home.sh <Modal app id> <stop time HH:MM:SS> <log of the modal client> [run name on the volume, default run2]
# 1. waits until the time, then for the run's next status line (the run has just stored its state), and stops the app;
# 2. downloads the model directory, the development results and the logs from the volume nrx-run;
# 3. checks the newest checkpoint, and keeps only the development results that the checkpoint's schedule has already
#    used (a later one is recomputed here: a result names the checkpoint path of the machine that made it);
# 4. starts the training here (./run.sh start, which refuses while another training uses the GPUs).
# If step 2 or 3 fails, nothing on this server is started. Progress: logs/bring_home.log
# (On 2026-11-18 this script also stopped a parallel run with a teacher term in the loss before step 4; that run was
# no better and has been deleted.)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODAL=${MODAL:-$HOME/miniforge3/bin/modal}
PY=${PY:-$HOME/miniforge3/envs/neural_rx/bin/python}
APP=$1; AT=$2; CLIENT_LOG=$3; NAME=${4:-run2}
MODELS=$ROOT/models/pilot_memory_no_teacher_20261118
STAGE=$HERE/from_modal
mkdir -p "$HERE/logs"
LOG=$HERE/logs/bring_home.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
fail() { say "handover FAILED: $*"; exit 1; }
lines() { tr '\r' '\n' < "$CLIENT_LOG" | grep -c "h used, update"; }

[ -e "$MODELS" ] && fail "$MODELS already exists"
[ -e "$STAGE" ] && fail "$STAGE already exists"
wait=$(( $(date -d "$AT" +%s) - $(date +%s) ))
say "waiting $wait s until $AT, then stopping Modal app $APP"
[ "$wait" -gt 0 ] && sleep "$wait"
before=$(lines)
for _ in $(seq 75); do [ "$(lines)" -gt "$before" ] && break; sleep 2; done      # at most 150 s for the next status line
say "last status: $(tr '\r' '\n' < "$CLIENT_LOG" | grep "h used, update" | tail -1 | cut -c1-110)"
"$MODAL" app stop "$APP" --yes >> "$LOG" 2>&1 || say "stop command returned an error (the app may have ended already)"
sleep 20
say "Modal app stopped; downloading"

mkdir -p "$STAGE"
for part in model results logs; do
  "$MODAL" volume get nrx-run "$NAME/$part" "$STAGE/" >> "$LOG" 2>&1 || { [ "$part" = model ] && fail "download of $part"; say "no $part on the volume"; }
done
[ -f "$STAGE/model/latest.json" ] || fail "the downloaded model directory has no latest.json (see $STAGE)"
PYTHONPATH=$ROOT/src "$PY" - "$STAGE" >> "$LOG" 2>&1 <<'PYTHON' || fail "the downloaded checkpoint did not pass the check"
import json, shutil, sys
from pathlib import Path
from neural_rx.receiver.checkpoints import recover_latest
stage = Path(sys.argv[1])
for leftover in stage.rglob(".tmp-*"):
    shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
latest = recover_latest(stage/"model")                      # checks the newest checkpoint's files, corrects latest.json
manifest = json.loads((stage/"model"/latest["checkpoint"]/"manifest.json").read_text())
used = manifest["schedule"]["development_update"]
print("newest checkpoint", latest, "; schedule has used the development check of update", used)
(stage/"results_not_used").mkdir(exist_ok=True)
for result in sorted((stage/"results").glob("development_*")) if (stage/"results").is_dir() else []:
    if int(result.name.split("_")[1]) > used:
        result.rename(stage/"results_not_used"/result.name)
        print("recomputed here:", result.name)
(stage/"results"/"best.json").unlink(missing_ok=True)
PYTHON
say "checkpoint ok: $(cat "$STAGE/model/latest.json" | tr -d '\n ')"

mv "$STAGE/model" "$MODELS" || fail "could not place the model directory"
mkdir -p "$HERE/results"
for result in "$STAGE"/results/development_*; do [ -d "$result" ] && mv "$result" "$HERE/results/"; done
[ -d "$STAGE/logs" ] && mv "$STAGE/logs" "$HERE/logs_modal"

say "starting the training on this server"
"$HERE/run.sh" start >> "$LOG" 2>&1 || fail "run.sh start"
sleep 300
"$HERE/run.sh" status >> "$LOG" 2>&1
tmux has-session -t no_teacher_training 2>/dev/null || fail "the training is not running; see logs/train.log"
say "handover done"
