#!/usr/bin/env bash
# Tái lập toàn bộ số liệu của báo cáo vào outputs/<TAG>/ (không ghi đè kết quả cũ).
#   ./run_experiments.sh            # chạy tất cả
#   TAG=lan2 DEVICE=cpu ./run_experiments.sh
#   PY=/opt/anaconda3/envs/ds/bin/python ./run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DEVICE=${DEVICE:-auto}
TAG=${TAG:-rerun-$(date +%Y%m%d-%H%M%S)}
OUT=outputs/$TAG
NANO=weights/yolo-nano/yolo11n-seg.pt
LARGE=weights/yolo-large/yolo11l-seg.pt
mkdir -p "$OUT"

echo "== 1/5 Kiểm thử"
$PY -m unittest discover -s tests -v

echo "== 2/5 Tốc độ O-QuickHull vs O-Graham"
$PY scripts/benchmark_hulls.py --output-dir "$OUT/benchmark/hull-speed"

echo "== 3/5 Đánh giá 100 ảnh Việt Nam"
$PY scripts/evaluate_vietnam_segmentation.py --model $NANO  --device "$DEVICE" --output-dir "$OUT/evaluation/vietnam-seg-100-nano"
$PY scripts/evaluate_vietnam_segmentation.py --model $LARGE --device "$DEVICE" --output-dir "$OUT/evaluation/vietnam-seg-100-large"

echo "== 4/5 Đánh giá 100 ảnh BDD100K"
$PY scripts/evaluate_bdd100k_segmentation.py --model $NANO  --device "$DEVICE" --output-dir "$OUT/evaluation/bdd100k-seg-100-nano"
$PY scripts/evaluate_bdd100k_segmentation.py --model $LARGE --device "$DEVICE" --output-dir "$OUT/evaluation/bdd100k-seg-100-large"

echo "== 5/5 Video demo (YOLO11l-seg + theo vết)"
if [ -f videos/vh1.mp4 ]; then
  $PY run_yolo11l_seg.py --device "$DEVICE" --output-dir "$OUT/vh1-large"
else
  echo "Bỏ qua: không có videos/vh1.mp4 (video không nằm trong repo)."
fi

echo "Xong: $OUT"
