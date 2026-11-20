#!/usr/bin/env bash
# Bước 3 của pipeline: huấn luyện + đánh giá mọi cấu hình QA-Graph (3 seed) của một bộ dữ liệu.
#   vg  (mặc định kaggle): 12 cấu hình của bảng chính thức vg_official (CLIP, Adapter, QA-Graph ± sg_rel/ConceptNet, GCN/GAT/SAGE,
#        xáo cạnh) — mỗi cấu hình một kernel Kaggle T4 chạy seed 0,1,2; 2 worker song song (Kaggle cho tối đa 2 phiên GPU).
#   psg (mặc định local): 8 cấu hình (tools/run_psg_local.sh: seed 0 lưu model, rồi seed 1,2), nice 19, 2 luồng, tuần tự.
# Dùng: tools/run_suite.sh vg|psg [kaggle|local]       log: logs/suite_<ds>_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
DS=${1:?"vg|psg"}
VG_A="vg_clip vg_qa_ablation_no_edge vg_qa vg_qa_ablation_no_rel vg_qa_cn vg_qa_cn_no_sg"
VG_B="vg_qa_cn_shuffle vg_qa_cn_gcn vg_qa_cn_sage vg_qa_cn_no_sg_gcn vg_qa_cn_no_sg_sage"
case "$DS:${2:-}" in
  vg:|vg:kaggle)
    ARGS="--set data=mmxai-vg-bundle --tag kaggle-3seed"
    tools/run_worker.sh "$VG_A" "$ARGS" > logs/suite_vg_A.log 2>&1 &
    tools/run_worker.sh "$VG_B" "$ARGS" > logs/suite_vg_B.log 2>&1 &
    wait
    echo "xong — log: logs/suite_vg_{A,B}.log; kết quả: runs/<exp>/*kaggle-3seed*/ (make pull đã chép metric sang results/)" ;;
  psg:|psg:local)
    tools/run_psg_local.sh all ;;
  psg:kaggle)
    echo "PSG trên Kaggle: make push EXP=<psg_*> ARGS='--set data=mmxai-psg-bundle --tag kaggle-3seed' (lượt chính thức chạy local)"; exit 1 ;;
  vg:local)
    for e in $VG_A $VG_B; do
      .venv/bin/python -m mmxai.run --exp "$e" --set cpu_threads=2 --tag local-3seed > "logs/suite_vg_$e.log" 2>&1 || echo "LỖI $e"
    done ;;
  *) echo "dùng: tools/run_suite.sh vg|psg [kaggle|local]"; exit 1 ;;
esac
