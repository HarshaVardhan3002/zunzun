#!/usr/bin/env bash
# Launch GLM-5.2 (744B) on colibrì, tuned for Strix Halo (Radeon 8060S + 128 GB).
# Run from an MSYS2 shell after `make HIP=1`.
#   ./run_glm52.sh                 # chat, GPU-accelerated, your Downloads model
#   ./run_glm52.sh /path/to/model  # different model dir
#   CPU=1 ./run_glm52.sh           # CPU-only (no iGPU tier)
set -e
MODEL="${1:-C:/Users/Von/Downloads/model}"
cd "$(dirname "$0")/c"
[ -x glm.exe ] || { echo "glm.exe not built — run: make HIP=1"; exit 1; }
if [ -z "$CPU" ]; then
  export COLI_HIP=1 CUDA_DENSE=1 CUDA_EXPERT_GB=24   # dense + hot experts on the 8060S (G1: mirror hot)
fi
# --topp 0.7: adaptive expert top-p (~1.6x by cutting cold reads). MTP (int8) auto-detected.
exec python coli chat --model "$MODEL" --topp 0.7 "${@:2}"
