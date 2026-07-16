#!/usr/bin/env bash
# Acceptance probe: fixed prompt, fixed seed, greedy, CPU-only. ~3 min warm.
#   ./spec_probe.sh <tag> [ENV=VAL ...]   e.g.: ./spec_probe.sh swap MTP_SWAP=1
# Appends the stats block to bench/spec/PROBES.md under the tag.
set -e
cd "$(dirname "$0")/../c"
TAG="${1:?usage: spec_probe.sh <tag> [ENV=VAL ...]}"; shift || true
OUT="../bench/spec/probe_${TAG}.out"; ERR="../bench/spec/probe_${TAG}.err"
env SNAP="${SNAP:-C:/Users/Von/Downloads/model}" PIPE=1 SEED=1 TEMP=0 NGEN=96 \
  PROMPT='[gMASK]<sop><|user|>Write a Python function that finds the shortest path in a weighted graph using Dijkstra with heapq. Output only code.<|assistant|><think></think>' \
  "$@" ./glm.exe >"$OUT" 2>"$ERR"
{ echo "## $TAG  ($(date +%F\ %T))  [$*]";
  grep -E "tokens in|speculation:|acceptance by depth" "$OUT"; echo; } >> ../bench/spec/PROBES.md
tail -n 4 "$OUT"
