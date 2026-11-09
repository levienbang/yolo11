# Kết quả tham chiếu của báo cáo cuối

| Thư mục | Nội dung |
|---|---|
| `evaluation/vietnam-seg-100-large/` | Đánh giá YOLO11-l trên 100 ảnh, 540 nhãn Việt Nam |
| `evaluation/vietnam-seg-100-nano/` | Đánh giá YOLO11-n để so sánh và chọn mô hình |
| `evaluation/compare-mask-hull/` | So sánh mặt nạ và các vùng bao, hình cắt cận và ảnh toàn cảnh |
| `benchmark/hull-speed/` | Thời gian hai thuật toán trên điểm tổng hợp và đường biên xe thật |
| `video/` | Hình minh hoạ, thông tin đoạn video ban đêm đã chạy bằng YOLO11-l và ByteTrack |

Các thư mục đánh giá chỉ giữ bảng số liệu, JSON và hình tổng hợp. Chạy lại hai script `evaluate_yolo11l_seg.py`, `evaluate_yolo11n_seg.py` để sinh ảnh đối chiếu và bản ghi dự đoán đầy đủ trong `outputs/`.

Tạo lại hình so sánh từ kết quả đã lưu:

```bash
python scripts/make_report_figures.py --output-dir outputs/report-figures
```

So sánh mặt nạ với bao trên cùng tập xe (cần cả hai thư mục kết quả đánh giá):

```bash
python scripts/compare_mask_hull.py --evaluation-dir outputs/evaluation --output-dir outputs/compare-mask-hull
```

Các số liệu chính dùng trọng số huấn luyện sẵn, Ultralytics 8.4.37. Thời gian benchmark phụ thuộc máy; video minh hoạ không phải phép đo tốc độ toàn hệ thống. Đường dẫn tuyệt đối trong JSON ghi lại vị trí tại thời điểm chạy.
