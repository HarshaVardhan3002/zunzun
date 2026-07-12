#!/usr/bin/env bash
# Phase 0 baseline bench — Strix Halo 395+, Windows 11 / MSYS2 (MinGW gcc).
# Usage (MSYS2 shell):  ./phase0/bench.sh /c/path/to/glm52_i4
# Builds AVX2 + native(znver5) binaries, runs iobench + config matrix, writes CSV.
set -e
MODEL=${1:?usage: bench.sh <model_dir>}
NGEN=${NGEN:-32}; RUNS=${RUNS:-2}; WARMUP=${WARMUP:-2}; CAP=${CAP:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd); C=$ROOT/c; OUT=$ROOT/phase0
LOGS=$OUT/logs; CSV=$OUT/results.csv; mkdir -p "$LOGS"
Q="Explain why the sky is blue in exactly three sentences."
PROMPT="[gMASK]<sop><|user|>${Q}<|assistant|><think></think>"
SHARD=$(ls "$MODEL"/*.safetensors | head -1)

build(){ # $1=arch $2=suffix
  make -C "$C" clean >/dev/null; make -C "$C" glm ARCH="$1"
  cp "$C/glm.exe" "$C/glm_$2.exe"
  echo "vpdpbusd(VNNI) in glm_$2: $(objdump -d "$C/glm_$2.exe" | grep -c vpdpbusd)"
}

run1(){ # $1=binary $2=config-name $3=run-idx $4...=extra env
  local bin=$1 name=$2 idx=$3; shift 3
  local log="$LOGS/${name}_r${idx}.log"
  ( cd "$C" && env SNAP="$MODEL" PROMPT="$PROMPT" NGEN=$NGEN TEMP=0 SEED=42 "$@" \
      "./$bin" $CAP >"$log" 2>&1 ) || { echo "$name,r$idx,FAILED,,,,," >>"$CSV"; return; }
  # parse: "N token in X.XXs (Y.YY tok/s) | hit-rate expert Z.Z% | RSS R.RR GB"
  local s; s=$(grep -oP '\d+ token in [\d.]+s \([\d.]+ tok/s\) \| hit-rate expert [\d.]+% \| RSS [\d.]+ GB' "$log" | tail -1)
  local tok=$(echo "$s"|grep -oP '^\d+') secs=$(echo "$s"|grep -oP 'in \K[\d.]+')
  local tps=$(echo "$s"|grep -oP '\(\K[\d.]+') hit=$(echo "$s"|grep -oP 'expert \K[\d.]+')
  local rss=$(echo "$s"|grep -oP 'RSS \K[\d.]+') tfw=$(grep -oP '[\d.]+ tok/fw' "$log"|tail -1|grep -oP '^[\d.]+')
  echo "$name,r$idx,$tok,$secs,$tps,$hit,$rss,$tfw" >>"$CSV"
  echo "  $name r$idx: ${tps:-?} tok/s hit ${hit:-?}% rss ${rss:-?}GB tok/fw ${tfw:-}"
}

bench(){ # $1=binary $2=config-name $3...=extra env
  local bin=$1 name=$2; shift 2
  for i in $(seq 1 $RUNS); do run1 "$bin" "$name" "$i" "$@"; done
}

echo "== builds =="
build x86-64-v3 v3
build native native

echo "== iobench (19MB x 64, 8 threads) =="
make -C "$C" iobench.exe >/dev/null
( cd "$C" && ./iobench.exe "$SHARD" 19 64 8 0 | tee "$LOGS/iobench_buffered.log" )
( cd "$C" && ./iobench.exe "$SHARD" 19 64 8 1 | tee "$LOGS/iobench_direct.log" )  # O_DIRECT: no-op on Win, expect ~= buffered

echo "config,run,tok,secs,tps,hit%,rss_gb,tok_fw" >"$CSV"
echo "== warmup x$WARMUP (feeds AUTOPIN/.coli_usage + page cache, not recorded) =="
for i in $(seq 1 $WARMUP); do run1 glm_native.exe warmup "$i" >/dev/null || true; done
sed -i '/^warmup/d' "$CSV"

echo "== matrix (ngen=$NGEN, greedy, seed 42, ${RUNS}x each) =="
bench glm_v3.exe     A_v3_stock
bench glm_native.exe B_native
bench glm_native.exe C_native_topp  TOPP=0.7
bench glm_native.exe D_native_pilot TOPP=0.7 PILOT=1   # expect ~= C on Windows (fadvise no-op)
# NOTE: int4-IDOT-at-S=1 (g_i4s) is compile-time on x86, no env knob -> Phase 1 patch, not testable here.

echo "== done: $CSV =="; column -s, -t "$CSV" || cat "$CSV"
