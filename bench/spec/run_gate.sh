#!/usr/bin/env bash
# Task-11 ship gate (runnable portion): greedy plain vs greedy speculative at the
# adopted default (chain-norm, DRAFT auto=4). Any text divergence must be a
# near-tie [gap] flip (MTP_DEBUG=1 logs top-2 gaps to stderr on mismatch).
set -u
cd "$(dirname "$0")/.."
LOG="spec/run_gate.log"
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
run gate_e DRAFT=0
run gate_d MTP_DEBUG=1
echo "[$(date +%T)] GATE_COMPLETE" >> "$LOG"
