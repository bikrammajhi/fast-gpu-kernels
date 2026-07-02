#!/bin/bash
set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$BASE_DIR/../../.." && pwd)"
SCRIPTS="$ROOT_DIR/scripts"

for v in $(ls "$BASE_DIR"/matmul_v*.cu | sort -V); do
  vname=$(basename "$v" .cu)
  echo ""
  echo "========================================"
  echo "  $vname  8192x8192x8192"
  echo "========================================"

  # Create a temp benchmark that includes this version instead of v1
  tmp="$BASE_DIR/benchmark_8192_tmp.cu"
  sed "s|#include \"matmul_v1.cu\"|#include \"$vname.cu\"|" "$BASE_DIR/benchmark.cu" > "$tmp"

  modal run "$SCRIPTS/run.py" --task "kernels/cute/A100/benchmark_8192_tmp.cu" 2>&1 | \
    grep -E '(Device:|M.*N.*K.*CuTe|cubl|TF|ms|%|backend:|compile failed|Error|FAIL)'

  rm -f "$tmp"
done
