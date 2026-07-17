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
  # dense + hot experts on the 8060S (G1: mirror hot). Overridable, e.g.:
  #   CUDA_EXPERT_GB=6 RAM_GB=70 ./run_glm52.sh   # fit GPU mirrors + CPU side in the 127.5/0.5 split
  export COLI_HIP=1 CUDA_DENSE="${CUDA_DENSE:-1}" CUDA_EXPERT_GB="${CUDA_EXPERT_GB:-24}"
fi
export PIPE="${PIPE:-1}"   # overlap NVMe expert loads with matmul (+60% tok/s measured on this box; PIPE=0 reverts)
# PROFILE=coding ./run_glm52.sh — per-domain expert history: pre-pins YOUR coding hot set (any [A-Za-z0-9_-] name)
# --topp 0.7: adaptive expert top-p (~1.6x by cutting cold reads). MTP (int8) auto-detected.
exec python zun chat --model "$MODEL" --topp 0.7 "${@:2}"
