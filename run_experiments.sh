#!/usr/bin/env bash
# Tái lập số liệu báo cáo vào outputs/<TAG>/.
# PY=python DEVICE=auto VIDEO=/path/video.mp4 TAG=lan-moi ./run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DEVICE=${DEVICE:-auto}
VIDEO=${VIDEO:-videos/traffic-night.mp4}
TAG=${TAG:-rerun-$(date +%Y%m%d-%H%M%S)}
OUT=outputs/$TAG
NANO=weights/yolo-nano/yolo11n-seg.pt
LARGE=weights/yolo-large/yolo11l-seg.pt
mkdir -p "$OUT"

echo "1/4 Kiểm thử"
"$PY" -m unittest discover -s tests -v

echo "2/4 So sánh hình dạng và đo tốc độ QuickHull, Graham"
"$PY" scripts/compare_shape_vertices.py --output-dir "$OUT/shape"
"$PY" scripts/benchmark_hulls.py --output-dir "$OUT/benchmark/hull-speed"

echo "3/4 Đánh giá 100 ảnh Việt Nam và tạo hình"
"$PY" scripts/evaluate_vietnam_segmentation.py --model "$NANO" --device "$DEVICE" --output-dir "$OUT/evaluation/vietnam-seg-100-nano"
"$PY" scripts/evaluate_vietnam_segmentation.py --model "$LARGE" --device "$DEVICE" --output-dir "$OUT/evaluation/vietnam-seg-100-large"
"$PY" scripts/compare_mask_hull.py --evaluation-dir "$OUT/evaluation" --device "$DEVICE" --output-dir "$OUT/evaluation/compare-mask-hull"
"$PY" scripts/make_report_figures.py --evaluation-dir "$OUT/evaluation" --speed-csv "$OUT/benchmark/hull-speed/synthetic.csv" --output-dir "$OUT/figures"

echo "4/4 Video demo với ByteTrack"
if [ -f "$VIDEO" ]; then
  "$PY" run_yolo11l_seg.py --input "$VIDEO" --algorithm ograham --device "$DEVICE" --output-dir "$OUT/video-large"
else
  echo "Bỏ qua video: đặt VIDEO=/duong/dan/video.mp4 để chạy."
fi

echo "Xong: $OUT"
