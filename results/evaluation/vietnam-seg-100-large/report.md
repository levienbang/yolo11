# Đánh giá 100 ảnh Việt Nam — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 540. Chọn 70 dễ (ít che khuất), 20 trung bình, 10 khó bằng xem ảnh.
Không dùng số lượng xe để định nghĩa dễ. Tất cả GT xe là polygon; loại toàn bộ ảnh có nhãn xe chỉ là box.

Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-large/bestseg.pt`.
Metadata training dataset: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.
Đây là checkpoint đang dùng trong pipeline local. Metadata COCO không chứng minh đây là bản fine-tune Việt Nam của người bạn; cần checkpoint đó để đánh giá riêng.
Không retrain. Đánh giá ảnh trước tracking: chưa đánh giá ID, đếm xe hoặc tốc độ Jetson.

## Kết quả

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP | Dice TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| box_detection | 72.9% | 69.8% | 71.3% | 71.8% | 55.5% | 85.8% | 92.0% |
| bbox_region | 61.1% | 58.5% | 59.8% | 57.8% | 21.8% | 67.6% | 80.3% |
| convex | 70.8% | 67.8% | 69.3% | 69.8% | 43.0% | 80.2% | 88.6% |
| yolo_mask | 74.7% | 71.5% | 73.0% | 74.0% | 54.7% | 84.4% | 91.2% |
| quickhull | 72.0% | 68.9% | 70.4% | 70.9% | 47.4% | 82.2% | 89.8% |
| ograham | 72.0% | 68.9% | 70.4% | 70.9% | 47.4% | 82.2% | 89.8% |

box_detection so box với box GT. bbox_region tô kín box so mask GT, chỉ là baseline vùng pixel.
Mask YOLO/QuickHull/O-Graham đều so cùng polygon GT raster hóa tại kích thước ảnh gốc.
mAP dùng matching đúng lớp, confidence-ordered, IoU 0.50:0.05:0.95, 101 điểm recall; không phải COCOeval chính thức.
IoU/Dice TP chỉ tính cặp đúng lớp có IoU ≥ 0.5: cần đọc cùng recall để thấy xe bỏ sót.

## Nhầm lớp

Confusion rows=GT, columns=prediction; ghép không xét lớp theo IoU box/mask, ngưỡng 0.5.
Background column=GT bỏ sót; background row=prediction không ghép được.
| Đầu ra | Car → truck | Truck → car | Accuracy trong cặp đã ghép |
|---|---:|---:|---:|
| box_detection | 17 | 4 | 92.5% |
| yolo_mask | 17 | 5 | 92.4% |
| quickhull | 17 | 5 | 92.2% |
| ograham | 17 | 5 | 92.2% |

## Bao lồi và mask YOLO

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.624 | 91.3% | 9.3% | 0.6% |
| ograham | 0.357 | 91.3% | 9.3% | 0.6% |

## Giới hạn và file

Bộ chọn có chủ đích, không đại diện ngẫu nhiên cho giao thông Việt Nam. Dễ là nhận xét trực quan tương đối.
Split nguồn và số lớp xem selection.json. Có thể dùng ảnh source train; chưa có dữ liệu fine-tune của người bạn để chứng minh không leakage.
Không tự đổi nhãn car/truck của nguồn, không suy đoán pickup/van. Bỏ bicycle ngoài phạm vi.
Ảnh cùng video có tương quan dù không trùng pixel; không coi đây là 100 cảnh độc lập.
AP bị giới hạn bởi confidence floor, NMS và max_det. Một số mask có thể thiếu/không sát vật thể ngay trong nguồn.
`class_metrics.csv`, `difficulty_metrics.csv`, `per_image.csv`, `metrics.json`, `prediction_records.json` chứa số liệu.
`comparisons/` có đủ 100 ảnh so GT / YOLO / QuickHull / O-Graham; `errors/` chứa ảnh có nhầm lớp qua ghép box.



(Ảnh ví dụ, `comparisons/`, `errors/` và `prediction_records.json` chỉ có trong `outputs/` ở máy chạy; chạy lại để sinh.)
