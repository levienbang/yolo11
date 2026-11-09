# Phân đoạn phương tiện bằng YOLO11-seg và bao lồi trực giao

Đồ án nhận dạng bốn lớp `car`, `truck`, `bus`, `motorcycle`, rồi dựng bao lồi trực giao từ đường biên mặt nạ. Hai bản cài đặt của nhóm được gọi là **QuickHull** và **Graham**. Video dùng ByteTrack có sẵn trong Ultralytics để gắn ID.

[Báo cáo cuối — mã nguồn LaTeX](report/overleaf/main.tex) · [ZIP tải lên Overleaf](report/bao-cao.zip) · [Kết quả tham chiếu](results/README.md)

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/vehicle_pipeline/` | Hai thuật toán dựng bao, xử lý hình học và chạy video |
| `scripts/` | Chuẩn bị dữ liệu, đánh giá, đo tốc độ và tạo hình |
| `tests/` | Kiểm thử thuật toán và phần đánh giá |
| `datasets/vietnam-seg-review-100/` | 100 ảnh Việt Nam, 540 nhãn polygon dùng trong báo cáo |
| `results/` | JSON và CSV chứa số liệu đã dùng trong báo cáo |
| `report/` | Báo cáo LaTeX cuối cùng và ZIP Overleaf |
| `weights/`, `videos/`, `outputs/` | Trọng số, video và đầu ra chạy trên máy; không đưa lên Git |

`QuickHull` là bản chọn cực trị bằng đệ quy của nhóm, lấy ý tưởng từ O-Quickhull; đây không phải bản cài nguyên thuật toán trong bài báo. `Graham` là bản sắp xếp rồi quét theo bốn hướng. Mã tương ứng ở `quickhull.py` và `OGraham.py`.

## Cài đặt

Dùng Python 3.11, chạy các lệnh từ thư mục gốc của repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p weights/yolo-nano weights/yolo-large
python -c "from ultralytics import YOLO; YOLO('weights/yolo-nano/yolo11n-seg.pt'); YOLO('weights/yolo-large/yolo11l-seg.pt')"
```

Các kết quả chính dùng trọng số COCO huấn luyện sẵn. Thử nghiệm fine-tune của nhóm chưa cho cải thiện nên không dùng để tạo các bảng kết quả này.

## Chạy chương trình

```bash
# Video do người dùng cung cấp; kết quả dùng Graham và ByteTrack
python run_yolo11l_seg.py --input videos/traffic-night.mp4 --algorithm ograham --output-dir outputs/traffic-night-large

# So sánh cả hai thuật toán trên video
python run_yolo11l_seg.py --input videos/traffic-night.mp4 --algorithm both --output-dir outputs/traffic-night-both

# Đánh giá hai mô hình trên bộ Việt Nam
python evaluate_yolo11l_seg.py
python evaluate_yolo11n_seg.py

# Đo thời gian dựng bao
python scripts/benchmark_hulls.py --output-dir outputs/benchmark/lan-moi

# Kiểm thử
python -m unittest discover -s tests -v
```

Video không kèm theo repo; truyền đường dẫn video của bạn bằng `--input`. Có thể dùng `--max-frames 30` để chạy thử. `--device auto` mặc định chọn CUDA, Apple MPS hoặc CPU; có thể chỉ định `--device cpu`, `mps` hoặc `0`.

Mỗi lượt đánh giá nên dùng thư mục mới qua `--output-dir`. Với video, chỉ dùng `--overwrite` khi muốn thay đầu ra cũ.

Chạy kiểm thử, benchmark, đánh giá hai mô hình, tạo hình và video (nếu có):

```bash
TAG=ban-cuoi DEVICE=auto ./run_experiments.sh
# Có thể chọn Python và video:
PY=python VIDEO=/duong/dan/video.mp4 TAG=lan-moi ./run_experiments.sh
```

## Dữ liệu và kết quả

Bộ Việt Nam gồm 100 ảnh với 540 phương tiện: 179 car, 110 truck, 26 bus, 225 motorcycle. Nhãn polygon, thông tin nguồn và danh sách ảnh nằm trong thư mục dữ liệu; nguồn Roboflow theo giấy phép CC BY 4.0. Tệp `datasets/pickup_relabel.csv` lưu thay đổi nhãn để tái lập đúng số liệu.

Kết quả tham chiếu nằm ở `results/evaluation/` và `results/benchmark/`. Đây là kết quả trên ảnh dùng mô hình huấn luyện sẵn; hình minh hoạ video YOLO11-l, ByteTrack và Graham nằm trong `report/overleaf/figs/video_preview_night.jpg`. Các chỉ số tracking không được suy ra từ bảng đánh giá ảnh.

Các phép đo thời gian phụ thuộc máy. Trọng số, video, cache và toàn bộ ảnh đầu ra được giữ cục bộ, không đẩy lên GitHub.
