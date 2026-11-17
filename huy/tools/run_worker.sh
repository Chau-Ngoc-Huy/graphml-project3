#!/usr/bin/env bash
# Một "luồng" chạy tuần tự trên Kaggle: với mỗi thí nghiệm, push (thử lại khi Kaggle báo đủ 2 phiên GPU) → wait → pull.
# Kaggle cho tối đa 2 phiên GPU cùng lúc ⇒ chạy đúng 2 worker song song là tối đa mà không bị từ chối.
# Mục có tiền tố "pull:" (vd. pull:t1_qa_cn) = kernel đã đẩy từ trước, chỉ chờ xong rồi tải kết quả.
# Dùng: tools/run_worker.sh "pull:t1_qa_cn exp2 exp3" "--set data=mmxai-task1-bundle --tag kaggle-3seed"
set -uo pipefail
cd "$(dirname "$0")/.."
ITEMS=${1:?"danh sách thí nghiệm"}
EXTRA=${2:-""}
status() { make -s status EXP="$1" 2>&1 | tail -n 1; }
for item in $ITEMS; do
  exp=${item#pull:}
  echo "=== $(date +%T) $item"
  if [[ $item != pull:* ]]; then
    for try in $(seq 1 120); do
      out=$(make push EXP="$exp" ARGS="$EXTRA" 2>&1)
      if grep -q "successfully pushed" <<<"$out"; then echo "$(date +%T) pushed $exp (lần $try)"; break; fi
      echo "$(date +%T) push chưa được ($(grep -o 'Kernel push error.*' <<<"$out" | head -1)), thử lại sau 120s"
      sleep 120
    done
    sleep 60
  fi
  while status "$exp" | grep -qiE "running|queued"; do sleep 30; done
  echo "$(date +%T) $(status "$exp")"
  make pull EXP="$exp" 2>&1 | tail -n 3
done
echo "=== $(date +%T) worker xong"
