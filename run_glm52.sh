#!/usr/bin/env bash
# Launch GLM-5.2 (744B) on zunzún, tuned for Strix Halo (Radeon 8060S + 128 GB).
# Run from an MSYS2 shell after `make HIP=1`.
#   ./run_glm52.sh                 # chat, GPU-accelerated, your Downloads model
#   ./run_glm52.sh /path/to/model  # different model dir
#   CPU=1 ./run_glm52.sh           # CPU-only (no iGPU tier)
set -e
MODEL="${1:-C:/Users/Von/Downloads/model}"
cd "$(dirname "$0")/c"
[ -x glm.exe ] || { echo "glm.exe not built — run: make HIP=1"; exit 1; }
if [ -z "$CPU" ]; then
  # dense on the 8060S only; expert mirrors OFF by default (they double-pay RAM:
  # each VRAM mirror keeps its RAM pin — measured net-negative on this box, see
  # bench/spec/run_gpu96.sh + PROBES.md 2026-07-17). CUDA_EXPERT_GB=n re-enables.
  export COLI_HIP=1 CUDA_DENSE="${CUDA_DENSE:-1}" CUDA_EXPERT_GB="${CUDA_EXPERT_GB:-0}"
fi
# Hard cap the CPU plan at 100 GB (unsplit BIOS mode: ~1 GB VRAM / rest system).
# Keeps experts + LRU inside a tight, predictable budget instead of auto=88% of
# MemAvailable; leaves real headroom for OS/apps/page cache. RAM_GB=n overrides.
export RAM_GB="${RAM_GB:-100}"
# Guard: a cap above physical RAM is a guaranteed OOM (seen on the 96/32 BIOS
# split 2026-07-17). If the box can't host RAM_GB+8, fall back to auto-budget.
TOT_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${TOT_GB:-0}" -gt 0 ] && [ "$TOT_GB" -lt $((${RAM_GB%.*}+8)) ]; then
  echo "run_glm52: RAM_GB=$RAM_GB won't fit in ${TOT_GB} GB physical — using auto budget" >&2
  unset RAM_GB
fi
export PIPE="${PIPE:-1}"   # overlap NVMe expert loads with matmul (+60% tok/s measured on this box; PIPE=0 reverts)
# PROFILE=coding ./run_glm52.sh — per-domain expert history: pre-pins YOUR coding hot set (any [A-Za-z0-9_-] name)
# --topp 0.7: adaptive expert top-p (~1.6x by cutting cold reads). MTP (int8) auto-detected.
exec python zun chat --model "$MODEL" --topp 0.7 "${@:2}"
