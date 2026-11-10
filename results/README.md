# Số liệu dùng trong báo cáo

| Thư mục | Nội dung |
|---|---|
| `evaluation/vietnam-seg-100-large/` | Chất lượng YOLO11-l: JSON tổng hợp, CSV theo lớp, ảnh và ngưỡng độ tin cậy |
| `evaluation/vietnam-seg-100-nano/` | Kết quả YOLO11-n dùng để chọn mô hình |
| `evaluation/compare-mask-hull/` | Số liệu so sánh mặt nạ và vùng bao |
| `shape/` | IoU và số đỉnh của ba loại bao trên 540 xe, gồm CSV từng xe để kiểm tra lại |
| `benchmark/hull-speed/` | JSON tổng hợp và CSV thời gian dựng bao |

Các hình đã dùng chỉ giữ một bản trong `report/overleaf/figs/`. Không giữ ảnh kết quả lặp lại hay đầu ra chạy thử cũ.

Chạy lại đánh giá bằng `evaluate_yolo11l_seg.py`, `evaluate_yolo11n_seg.py`; đầu ra mới được ghi vào `outputs/`. Tạo hình từ số liệu tham chiếu:

```bash
python scripts/make_report_figures.py --output-dir outputs/report-figures
```

Số liệu chính dùng trọng số huấn luyện sẵn và Ultralytics 8.4.37. Thời gian benchmark phụ thuộc máy. Đường dẫn tuyệt đối trong JSON là thông tin tại thời điểm chạy.
