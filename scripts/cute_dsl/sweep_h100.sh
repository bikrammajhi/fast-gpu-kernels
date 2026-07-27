#!/usr/bin/env bash
set -euo pipefail

KERNELS=(
  "4_wgmma_tma.py"
  "5_wgmma_tma_multistage.py"
  "6_wgmma_tma_multistage_epilogue.py"
  "7_wgmma_tma_multistage_epilogue_WS.py"
  "8_wgmma_tma_multistage_WS2.py"
  "9_wtmew_multicast.py"
)

SHAPES=(
  "128 128 128"
  "256 256 256"
  "512 512 512"
  "1024 1024 1024"
  "2048 2048 2048"
  "4096 4096 4096"
  "5120 5120 4096"
  "8192 8192 8192"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_DIR="$PROJECT_ROOT/kernels/cute_dsl/examples/sweep"
RESULTS_FILE="$SWEEP_DIR/results.tsv"
REFERENCE_FILE="$PROJECT_ROOT/kernels/cute_dsl/examples/benchmark.py"

mkdir -p "$SWEEP_DIR"
cp "$REFERENCE_FILE" "$SWEEP_DIR/benchmark_ref.py"

echo -e "kernel\tM\tN\tK\tTFLOPS\tms" > "$RESULTS_FILE"

for shape in "${SHAPES[@]}"; do
  read -r M N K <<< "$shape"
  echo "=== Shape: ${M}x${N}x${K} ==="

  for kernel in "${KERNELS[@]}"; do
    echo "  Running $kernel ..."
    src="$PROJECT_ROOT/kernels/cute_dsl/examples/$kernel"

    tmp="$SWEEP_DIR/${kernel%.py}_${M}_${N}_${K}.py"
    cp "$src" "$tmp"

    if grep -qE 'M, N, K = [0-9]' "$src"; then
      sed -i "s/^    M, N, K = .*/    M, N, K = $M, $N, $K/" "$tmp"
    elif grep -qE 'M, N, K = [0-9].*' "$src"; then
      sed -i "s/^    M, N, K = .*/    M, N, K = $M, $N, $K/" "$tmp"
    else
      python3 -c "
import re, sys
p = sys.argv[1]; m = int(sys.argv[2]); n = int(sys.argv[3]); k = int(sys.argv[4])
txt = open(p).read()
txt = re.sub(r'^(    M, N, K = ).*$', lambda m, mm=m, nn=n, kk=k: m.group(1) + f'{mm}, {nn}, {kk}', txt, flags=re.MULTILINE)
open(p, 'w').write(txt)
" "$tmp" "$M" "$N" "$K"
    fi

    out=$(modal run scripts/cute_dsl/run.py::main \
      --task "examples/sweep/${kernel%.py}_${M}_${N}_${K}.py" \
      --gpu H100 2>/dev/null || true)

    tflops=$(echo "$out" | grep -oP 'TFLOPS:\s+\K[0-9.]+' | head -1 || echo "N/A")
    ms=$(echo "$out" | grep -oP 'DURATION:\s+\K[0-9.]+' | head -1 || echo "N/A")

    echo -e "$kernel\t$M\t$N\t$K\t$tflops\t$ms" >> "$RESULTS_FILE"
    echo "      -> TFLOPS: $tflops, ms: $ms"
  done
done

echo ""
echo "=== Sweep complete. Results in $RESULTS_FILE ==="
