#!/usr/bin/env bash
# Bước 2 của pipeline: dựng phần bổ sung cho đồ thị trên một bundle (chạy trên máy, nhẹ: nice 19, 2 luồng).
#   VG : graph_extra    = 2000 nhãn (object_vocab_train) + vector CLIP chữ
#        graph_extra_cn = 2000 nhãn + 1000 khái niệm ConceptNet + cạnh ConceptNet (16 413) + vector
#   PSG: (bundle đã chứa 133 nhãn + 674 khái niệm, tools/psg_to_bundle.py) graph_extra + graph_extra_cn (3 743 cạnh)
# Đã kiểm chứng: dựng lại cho ra conceptnet.tsv / concepts_added.json / label_emb.npy GIỐNG HỆT bản đã dùng (lệch 0.0).
# Dùng: tools/build_extras.sh vg|psg [thư mục bundle]   (cần: make fetch → data/conceptnet/hf_en_simple/train.parquet)
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?"vg|psg"}
case $DS in
  vg)  B=${2:-data/vg_bundle}; ADD=1000 ;;
  psg) B=${2:-data/psg_bundle};   ADD=0 ;;
  *) echo "dataset phải là vg hoặc psg"; exit 1 ;;
esac
PY=.venv/bin/python
CN=data/conceptnet/hf_en_simple/train.parquet
export OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
N="nice -n 19"
[ -f "$B/data/images.jsonl" ] || { echo "Chưa có $B (make data-$DS hoặc make fetch-bundles)"; exit 1; }
[ -f "$CN" ] || { echo "Chưa có $CN (make fetch)"; exit 1; }

echo "[extras $DS] graph_extra: vector nhãn"
$N $PY -I tools/extract_label_emb.py --bundle "$B"
echo "[extras $DS] graph_extra_cn: cạnh ConceptNet (+$ADD khái niệm)"
$N $PY -I tools/fetch_conceptnet.py --bundle "$B" --assertions "$CN" --add-concepts "$ADD" --out "$B/graph_extra_cn"
$N $PY -I tools/extract_label_emb.py --bundle "$B" --extra "$B/graph_extra_cn"
echo "[extras $DS] xong → $B/graph_extra, $B/graph_extra_cn"
