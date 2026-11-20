#!/usr/bin/env bash
# Chạy toàn bộ pipeline PSG trên máy, NHẸ (nice 19, 2 luồng, tuần tự). Giả định đã có data/psg_bundle (psg_to_bundle.py)
# và features (psg_extract_clip.py).
#   Lượt 1: mọi biến thể seed 0 (lưu model → UI)      Lượt 2: seed 1, 2 (không lưu model) → gộp 3 seed bằng tools/psg_table.py
# Dùng: tools/run_psg_local.sh [1|2|all]     log: logs/psg_*.log, tiến độ: logs/psg_queue.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
N="nice -n 19"
PASS=${1:-all}
EXPS="psg_clip psg_qa psg_qa_ablation_no_edge psg_qa_cn psg_qa_ablation_shuffle psg_qa_ablation_no_rel psg_qa_cn_no_sg psg_qa_cn_shuffle"
Q=logs/psg_queue.log
say() { echo "$(date '+%F %T') $*" | tee -a "$Q"; }

for x in huy_extra huy_extra_cn; do
  if [ ! -f "data/psg_bundle/$x/label_emb.npy" ]; then
    say "label_emb $x"
    $N $PY -I tools/extract_label_emb.py --bundle data/psg_bundle --extra "data/psg_bundle/$x" >> "$Q" 2>&1 || { say "LỖI label_emb $x"; exit 1; }
  fi
done

if [ ! -f logs/psg_smoke.ok ]; then
  say "smoke psg_qa_cn --limit 5"
  if $N $PY -m mmxai.run --exp psg_qa_cn --limit 5 --seed 0 --tag smoke --no-publish --set cpu_threads=2 > logs/psg_smoke.log 2>&1; then
    touch logs/psg_smoke.ok
  else
    say "LỖI smoke — xem logs/psg_smoke.log"; exit 1
  fi
fi

run() {  # run <exp> <tag> <args...>
  local exp=$1 tag=$2; shift 2
  if ls runs/"$exp"/*"$exp-$tag"*/metrics.json >/dev/null 2>&1; then
    say "bỏ qua $exp $tag (đã có)"; return
  fi
  say "bắt đầu $exp $tag"
  if $N $PY -m mmxai.run --exp "$exp" --tag "$tag" --set cpu_threads=2 "$@" > "logs/psg_${exp}_${tag}.log" 2>&1; then
    say "xong $exp $tag: $(grep -E 'test +full' "logs/psg_${exp}_${tag}.log" | tail -1 | sed 's/.*| test/test/')"
  else
    say "LỖI $exp $tag — xem logs/psg_${exp}_${tag}.log"
  fi
}

if [ "$PASS" = 1 ] || [ "$PASS" = all ]; then
  run psg_clip 3s                       # CLIP tất định: chạy luôn đủ 3 gallery + toàn pool
  for e in $EXPS; do [ "$e" = psg_clip ] || run "$e" s0 --seed 0; done
fi
if [ "$PASS" = 2 ] || [ "$PASS" = all ]; then
  for e in $EXPS; do
    [ "$e" = psg_clip ] && continue
    run "$e" s12 --set 'seeds=[1,2]' --set save_model=false
  done
fi
say "HẾT"
