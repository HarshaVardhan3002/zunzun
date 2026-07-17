#!/usr/bin/env bash
# Batched probe sweep: Task 6 baseline + Task 8 wiring grid + audit candidates.
# One engine instance at a time, sequential. Logs progress to run_sweep.log.
set -u
cd "$(dirname "$0")/.."
LOG="spec/run_sweep.log"
: > "$LOG"
run() {
  local tag="$1"; shift
  echo "[$(date +%T)] START $tag $*" >> "$LOG"
  if ./spec_probe.sh "$tag" "$@" >> "$LOG" 2>&1; then
    echo "[$(date +%T)] DONE $tag" >> "$LOG"
  else
    echo "[$(date +%T)] FAIL $tag (exit $?)" >> "$LOG"
  fi
}
run base
run base2
run prenorm MTP_PRENORM=1
run swap    MTP_SWAP=1
run both    MTP_PRENORM=1 MTP_SWAP=1
run audit1  MTP_CHAINNORM=1
run audit2  MTP_MASK0=1
echo "[$(date +%T)] SWEEP_COMPLETE" >> "$LOG"
