# Bao lồi trực giao cho phân đoạn phương tiện với YOLO11-seg

Mã nguồn đồ án cơ sở: YOLO11-seg cho mặt nạ từng xe (`car`, `truck`, `bus`, `motorcycle`), từ đường biên
mặt nạ dựng **bao lồi trực giao liên thông** bằng hai bản cài đặt (**bản đệ quy** và **O-Graham**), theo vết bằng IoU trên video.
Cả hai mô hình là trọng số **COCO huấn luyện sẵn của Ultralytics**, không fine-tune.

## Cài đặt

```bash
conda activate ds                      # hoặc môi trường Python 3.11 bất kỳ
pip install -r requirements.txt
# tải hai trọng số chính thức (nếu chưa có)
python -c "from ultralytics import YOLO; YOLO('weights/yolo-nano/yolo11n-seg.pt'); YOLO('weights/yolo-large/yolo11l-seg.pt')"
```

Mọi lệnh có `--device auto` (mặc định): tự chọn CUDA → Apple MPS → CPU. Có thể ép `--device cpu|mps|0`.

## Chạy nhanh

```bash
python run_yolo11l_seg.py               # video demo, large  -> outputs/vh1-large/
python run_yolo11n_seg.py               # video demo, nano   -> outputs/vh1-nano/
python evaluate_yolo11l_seg.py          # 100 ảnh Việt Nam, large -> outputs/evaluation/vietnam-seg-100-large/
python evaluate_yolo11n_seg.py          # 100 ảnh Việt Nam, nano  -> outputs/evaluation/vietnam-seg-100-nano/
python scripts/benchmark_hulls.py --output-dir outputs/benchmark/<ten>
python scripts/compare_mask_hull.py --datasets vietnam --evaluation-dir <thu-muc> --output-dir <thu-muc>/compare-mask-hull
python scripts/check_hull_uniqueness.py --output <tep>.json
python -m unittest discover -s tests -v
```

Tuỳ chọn video: `--input <video>`, `--algorithm quickhull|ograham|both`, `--max-frames 10`, `--overwrite`.
Script đánh giá **không ghi đè**: nếu thư mục kết quả đã có, truyền `--output-dir` mới.

Tái lập số liệu (kiểm thử, benchmark, 4 lượt đánh giá gồm cả BDD100K, video) vào `outputs/<TAG>/`:

```bash
PY=python DEVICE=auto TAG=lan2 ./run_experiments.sh     # ~15 phút trên Apple M1
```

## Dữ liệu

| Thư mục | Nội dung |
|---|---|
| `datasets/vietnam-seg-review-100/` | 100 ảnh giao thông Việt Nam (70 dễ / 20 trung bình / 10 khó), 540 nhãn polygon. Nguồn Roboflow, CC BY 4.0 |
| `datasets/bdd100k-seg-100/` | 100 ảnh BDD100K val và mặt nạ từng xe, theo điều khoản của Berkeley DeepDrive (phi thương mại). **Báo cáo hiện không dùng bộ này**, chỉ giữ để tham khảo; chạy bằng `scripts/evaluate_bdd100k_segmentation.py` |
| `Vietnam Traffic Vehicle Detectio.v1i.yolov11/` | Bộ nguồn Roboflow đầy đủ (dùng cho `scripts/prepare_vietnam_segmentation.py`) |
| `videos/vh1.mp4` | Video demo 434 khung hình (không có trong repo) |

## Kết quả đã chạy (số liệu trong báo cáo)

Bản rút gọn có trong repo ở `results/` (xem `results/README.md`). Bản đầy đủ có ảnh đối chiếu nằm ở `outputs/` trên máy đã chạy:

| Thư mục | Nội dung |
|---|---|
| `outputs/evaluation/final-2026-11-06/vietnam-seg-100-{nano,large}/` | Số liệu chính thức của báo cáo (nhãn theo quy ước bán tải là `truck`) |
| `outputs/evaluation/final-2026-11-06/compare-mask-hull/` | So mặt nạ với bao trên cùng tập xe, độ gọn, hình cắt cận |
| `outputs/evaluation/vietnam-seg-100-{nano,large}/` | Lượt cũ theo nhãn gốc: `metrics.json`, CSV theo lớp/độ khó/ảnh, ma trận nhầm lẫn, 100 ảnh đối chiếu |
| `outputs/evaluation/vietnam-seg-100-nano-vs-large/` | Báo cáo so sánh nano–large |
| `outputs/benchmark/hull-speed/` | Benchmark tốc độ: `synthetic.csv`, `summary.json`, `hull_speed.png`, `report.md` |
| `outputs/vh1-large/` | Video đầu ra hai thuật toán + CSV đặc trưng từng xe + JSON thống kê |

Mỗi `metrics.json` ghi SHA256 của trọng số đã dùng. Các lượt chạy trước ngày đổi tên ghi đường dẫn cũ
`weights/yolo-large/bestseg.pt`; SHA256 `cabe9004…` là cùng tệp với `yolo11l-seg.pt` hiện tại.

## Cấu trúc mã

- `src/vehicle_pipeline/quickhull.py`: bản đệ quy chọn cực trị, lấy cảm hứng từ O-Quickhull nhưng **không** phải Algorithm 1 của bài báo.
- `src/vehicle_pipeline/OGraham.py`: bản cài đặt O-Graham của nhóm.
- `src/vehicle_pipeline/hull_common.py`: chuẩn hoá điểm, gọi thuật toán, diện tích/chu vi/tỉ lệ lấp đầy.
- `src/vehicle_pipeline/iou_tracker.py`: theo vết IoU (dự đoán chuyển động, ghép Hungary theo lớp, hai ngưỡng tin cậy).
- `src/vehicle_pipeline/inference_common.py`: vòng lặp video.
- `scripts/evaluate_vietnam_segmentation.py`, `evaluate_bdd100k_segmentation.py`, `evaluation_metrics.py`: đánh giá
  6 đầu ra (hộp, vùng hình chữ nhật, bao lồi thường, mặt nạ YOLO, hai bao trực giao).
- `scripts/benchmark_hulls.py`: đo tốc độ có kiểm soát.
- `scripts/compare_mask_hull.py`, `make_report_figures.py`: so mặt nạ với bao trên cùng tập xe, độ gọn, hình của báo cáo.
- `scripts/check_hull_uniqueness.py`: kiểm tra bao duy nhất và trùng vùng chuẩn trên xe thật.
- `tests/`: 9 kiểm thử (kiểm tra bao theo định nghĩa; hai bản cài đặt cho cùng biên; đọc nhãn; ma trận nhầm lẫn).

## Repo GitHub

Repo gồm mã, hướng dẫn, bộ đánh giá 100 ảnh Việt Nam dùng trong báo cáo, bộ BDD100K để tham khảo và kết quả tham chiếu `results/`.
Không gồm trọng số (tải tự động bằng lệnh ở mục Cài đặt), video demo và `outputs/`.
Sau khi clone: cài đặt → `python -m unittest discover -s tests -v` → `python evaluate_yolo11n_seg.py` → so với `results/`.
