#!/usr/bin/env bash
# Chạy tuần tự nhiều thí nghiệm trên Kaggle (mỗi kernel một lúc): push -> wait -> pull.
# Dùng: tools/run_queue.sh "exp1 exp2 exp3" [ARGS chung, vd. "--tag v1"]
set -uo pipefail
cd "$(dirname "$0")/.."
EXPS=${1:?"danh sách thí nghiệm, cách nhau bằng dấu cách"}
EXTRA=${2:-""}
for exp in $EXPS; do
  echo "=== $(date +%T) $exp"
  make push EXP="$exp" ARGS="$EXTRA" || { echo "push lỗi: $exp"; continue; }
  sleep 60
  make wait EXP="$exp" POLL=30 >/dev/null 2>&1
  make status EXP="$exp"
  make pull EXP="$exp" | tail -3
done
echo "=== $(date +%T) xong"
