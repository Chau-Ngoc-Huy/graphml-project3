#!/usr/bin/env bash
# Đưa bundle (sau bước extras) lên Kaggle Dataset riêng tư để các kernel huấn luyện (make push / make adamf) đọc được.
#   vg  → <KAGGLE_USER>/mmxai-task1-bundle      psg → <KAGGLE_USER>/mmxai-psg-bundle
# Lần đầu: tạo dataset; các lần sau: thêm phiên bản mới (kernel luôn đọc bản mới nhất).
# Dùng: tools/publish_bundle.sh vg|psg ["ghi chú phiên bản"]
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?"vg|psg"}; MSG=${2:-"cập nhật từ pipeline"}
case $DS in vg) B=data/vg_bundle; SLUG=mmxai-task1-bundle ;; psg) B=data/psg_bundle; SLUG=mmxai-psg-bundle ;; *) exit 1 ;; esac
K=.venv/bin/kaggle; [ -x $K ] || K=kaggle
USER_=$(sed -n 's/^KAGGLE_USER *[?:]*= *//p' local.mk 2>/dev/null | head -1)
[ -n "$USER_" ] || USER_=$($K config view | sed -n 's/.*username: *//p')
OUT=build/bundle_$DS; rm -rf "$OUT"; mkdir -p "$OUT"
# chỉ những gì pipeline đọc (không đóng gói shards / ảnh)
for x in data split features candidates graph_extra graph_extra_cn mmxai_eval.py psg_stats.json; do
  [ -e "$B/$x" ] && cp -r "$B/$x" "$OUT/"
done
cp src/mmxai_eval.py "$OUT/mmxai_eval.py"            # bản duy nhất trong repo
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
cat > "$OUT/dataset-metadata.json" <<EOF
{"title": "$SLUG", "id": "$USER_/$SLUG", "licenses": [{"name": "other"}],
 "subtitle": "Graph ML Project 3 bundle ($DS): data, split, CLIP features, graph extras"}
EOF
if $K datasets status "$USER_/$SLUG" >/dev/null 2>&1; then
  $K datasets version -p "$OUT" -m "$MSG" --dir-mode zip
else
  $K datasets create -p "$OUT" --dir-mode zip
fi
echo "→ https://www.kaggle.com/datasets/$USER_/$SLUG"
