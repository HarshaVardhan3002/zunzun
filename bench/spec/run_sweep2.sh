#!/usr/bin/env bash
# Sweep 2: confirm adopted default (chain-norm, no env) == audit1, legacy control,
# then Task-10 DRAFT depth sweep. Sequential, one engine instance.
set -u
cd "$(dirname "$0")/.."
LOG="spec/run_sweep2.log"
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
run default3
run rawchain MTP_RAWCHAIN=1
run d2 DRAFT=2
run d4 DRAFT=4
run d6 DRAFT=6
run d8 DRAFT=8
echo "[$(date +%T)] SWEEP2_COMPLETE" >> "$LOG"
