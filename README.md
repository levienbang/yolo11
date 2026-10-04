# YOLO11 segmentation — QuickHull / O-Graham

Chạy cá nhân với 4 lớp: `car`, `truck`, `bus`, `motorcycle`.
Hai thuật toán dùng chung mask YOLO; video có tracker IoU riêng.

## Chạy video

Trong thư mục này, bật môi trường Python rồi chọn **một** lệnh:

```bash
conda activate ds
python run_yolo11n_seg.py --device mps
```

Hoặc dùng weight large hiện có:

```bash
python run_yolo11_large_bestseg.py --device mps
```

Mỗi file mặc định chạy cả QuickHull và O-Graham trên `videos/vh1.mp4`.
Nano dùng `weights/yolo-nano/yolo11n-seg.pt`; large dùng `weights/yolo-large/bestseg.pt`.
Kết quả lưu riêng trong `outputs/vh1-nano-coco/` và `outputs/vh1-large-bestseg/`.

- Đổi video: `--input /duong/dan/video.mp4`.
- Chọn một thuật toán: `--algorithm quickhull` hoặc `--algorithm ograham`.
- Chạy lại, ghi đè output: `--overwrite`.
- Chạy CPU: `--device cpu`; NVIDIA: `--device 0`.
- Chạy thử ít frame: `--max-frames 10`.

## Đánh giá 100 ảnh Việt Nam

Chọn **một** lệnh:

```bash
python evaluate_yolo11n_seg.py --device mps
```

```bash
python evaluate_yolo11_large_bestseg.py --device mps
```

Cùng bộ `datasets/vietnam-seg-review-100/`: 70 dễ, 20 trung bình, 10 khó;
540 nhãn polygon, bỏ bicycle. Đánh giá mask và bao trước tracking.
Kết quả mới lưu riêng trong `outputs/evaluation/vietnam-seg-review-100-nano/`
và `outputs/evaluation/vietnam-seg-review-100-large/`.
Nếu thư mục đã tồn tại, thêm `--output-dir outputs/evaluation/ten-luot-moi`.

## Kết quả đã chạy

Báo cáo so sánh: `outputs/evaluation/vietnam-seg-review-100-nano-vs-large/report.md`.
Số liệu và 100 ảnh đối chiếu của từng model:

- Nano: `outputs/evaluation/vietnam-seg-review-100-nano-coco/`.
- Large: `outputs/evaluation/vietnam-seg-review-100-large-rerun/`.
- Video large trước đây: `outputs/vh1/`.

Nano là pretrained COCO chính thức. Metadata large cũng ghi COCO;
chưa xác minh đây là checkpoint fine-tune Việt Nam.
Bộ 100 ảnh được chọn có chủ đích; chưa xác minh độc lập với dữ liệu fine-tune.

## File bên trong

- `src/vehicle_pipeline/`: YOLO, tracker, QuickHull, O-Graham.
- `scripts/evaluate_vietnam_segmentation.py`: đánh giá dùng chung cho hai model.
- `scripts/evaluation_metrics.py`: tính IoU, precision/recall, AP và vẽ confusion.
- `scripts/prepare_vietnam_segmentation.py`: chọn/kiểm tra bộ ảnh từ nguồn Roboflow.
- `Vietnam Traffic Vehicle Detectio.v1i.yolov11/`: dataset nguồn Việt Nam.
- `tests/`: kiểm tra thuật toán bao và nhãn/matching.

Cài thư viện nếu cần: `pip install -r requirements.txt`.

## Repo GitHub

Commit đầu tiên chỉ gồm code và hướng dẫn. Weight, video, dataset và kết quả
đánh giá vẫn giữ ở máy local, chưa đưa lên repo.

Nano dùng weight pretrained chính thức; có thể tải bằng:

```bash
python -c "from ultralytics import YOLO; YOLO('weights/yolo-nano/yolo11n-seg.pt')"
```

Để chạy large hoặc đánh giá 100 ảnh, đặt weight và dataset vào các đường dẫn
được mô tả phía trên.
