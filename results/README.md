# Kết quả tham chiếu (số liệu trong báo cáo)

Bản rút gọn của `outputs/` dùng trong báo cáo, để đối chiếu khi chạy lại. Không gồm ảnh đối chiếu từng ảnh
(`comparisons/`, `errors/`, `examples/`) và `prediction_records.json` — chạy lại script để sinh.

| Thư mục | Lệnh tạo lại |
|---|---|
| `evaluation/vietnam-seg-100-nano/` | `python evaluate_yolo11n_seg.py --output-dir outputs/evaluation/<ten>` |
| `evaluation/vietnam-seg-100-large/` | `python evaluate_yolo11l_seg.py --output-dir outputs/evaluation/<ten>` |
| `evaluation/bdd100k-seg-100-{nano,large}/` | `python scripts/evaluate_bdd100k_segmentation.py --model weights/yolo-{nano/yolo11n,large/yolo11l}-seg.pt --output-dir outputs/evaluation/<ten>` |
| `evaluation/vietnam-seg-100-nano-vs-large/` | So sánh tổng hợp hai lượt Việt Nam |
| `benchmark/hull-speed/` | `python scripts/benchmark_hulls.py --output-dir outputs/benchmark/<ten>` |
| `video/vh1-large.json` | `python run_yolo11l_seg.py` (cần `videos/vh1.mp4`, không có trong repo) |

Chạy trên Apple M1 (MPS), Ultralytics 8.4.37. Số liệu chất lượng tái lập được tới ~0,001 điểm %;
thời gian (ms) phụ thuộc máy, chỉ nên so tương đối giữa hai thuật toán trên cùng một máy.
