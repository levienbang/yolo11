# Đánh giá 100 ảnh Việt Nam — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 540. Chọn 70 dễ (ít che khuất), 20 trung bình, 10 khó bằng xem ảnh.
Không dùng số lượng xe để định nghĩa dễ. Tất cả GT xe là polygon; loại toàn bộ ảnh có nhãn xe chỉ là box.

Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-large/yolo11l-seg.pt`.
Metadata training dataset: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.
Mô hình dùng trực tiếp, không fine-tune trên dữ liệu Việt Nam; chỉ giữ 4 lớp phương tiện khi suy luận.
Không retrain. Đánh giá ảnh trước tracking: chưa đánh giá ID hoặc đếm xe.

## Kết quả

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP | Dice TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| box_detection | 73.1% | 70.0% | 71.5% | 72.3% | 56.0% | 85.8% | 92.0% |
| bbox_region | 61.3% | 58.7% | 60.0% | 58.3% | 21.8% | 67.5% | 80.2% |
| convex | 71.0% | 68.0% | 69.4% | 70.3% | 43.1% | 80.2% | 88.6% |
| yolo_mask | 74.9% | 71.7% | 73.2% | 74.6% | 55.1% | 84.5% | 91.2% |
| quickhull | 72.1% | 69.1% | 70.6% | 71.4% | 47.6% | 82.2% | 89.8% |
| ograham | 72.1% | 69.1% | 70.6% | 71.4% | 47.6% | 82.2% | 89.8% |

box_detection so box với box GT. bbox_region tô kín box so mask GT, chỉ là baseline vùng pixel.
Mask YOLO/QuickHull/O-Graham đều so cùng polygon GT raster hóa tại kích thước ảnh gốc.
mAP dùng matching đúng lớp, confidence-ordered, IoU 0.50:0.05:0.95, 101 điểm recall; không phải COCOeval chính thức.
IoU/Dice TP chỉ tính cặp đúng lớp có IoU ≥ 0.5: cần đọc cùng recall để thấy xe bỏ sót.

## Nhầm lớp

Confusion rows=GT, columns=prediction; ghép không xét lớp theo IoU box/mask, ngưỡng 0.5.
Background column=GT bỏ sót; background row=prediction không ghép được.
| Đầu ra | Car → truck | Truck → car | Accuracy trong cặp đã ghép |
|---|---:|---:|---:|
| box_detection | 12 | 8 | 92.8% |
| yolo_mask | 12 | 9 | 92.7% |
| quickhull | 12 | 9 | 92.4% |
| ograham | 12 | 9 | 92.4% |

## Bao lồi và mask YOLO

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.538 | 91.3% | 9.3% | 0.6% |
| ograham | 0.303 | 91.3% | 9.3% | 0.6% |

## Giới hạn và file

Bộ chọn có chủ đích, không đại diện ngẫu nhiên cho giao thông Việt Nam. Dễ là nhận xét trực quan tương đối.
Split nguồn và số lớp xem selection.json. Chưa xác minh đầy đủ lịch sử huấn luyện của checkpoint nên chưa loại trừ khả năng trùng dữ liệu đã học.
Quy ước xe bán tải là truck; danh sách nhãn đã đổi so với nguồn ở datasets/pickup_relabel.csv. Bỏ bicycle ngoài phạm vi.
Ảnh cùng video có tương quan dù không trùng pixel; không coi đây là 100 cảnh độc lập.
AP bị giới hạn bởi confidence floor, NMS và max_det. Một số mask có thể thiếu/không sát vật thể ngay trong nguồn.
`class_metrics.csv`, `difficulty_metrics.csv`, `per_image.csv`, `metrics.json`, `prediction_records.json` chứa số liệu.
`comparisons/` có đủ 100 ảnh so GT / YOLO / QuickHull / O-Graham; `errors/` chứa ảnh có nhầm lớp qua ghép box.

## Ví dụ

**easy_best — train01_00096_jpg.rf.10d074e826ed8e73076770d68ea793da.jpg**

![easy_best — train01_00096_jpg.rf.10d074e826ed8e73076770d68ea793da.jpg](examples/easy_best.jpg)

**easy_worst — train01_00268_jpg.rf.29d52d8ae7b61d80a7580e22788aa64c.jpg**

![easy_worst — train01_00268_jpg.rf.29d52d8ae7b61d80a7580e22788aa64c.jpg](examples/easy_worst.jpg)

**medium_best — test01_00015_jpg.rf.33bf8e352bd1cebf1fbc6bb752734fd8.jpg**

![medium_best — test01_00015_jpg.rf.33bf8e352bd1cebf1fbc6bb752734fd8.jpg](examples/medium_best.jpg)

**medium_worst — train03_00008_jpg.rf.8bad59f8a8b5fa73016ee9635dc92a6e.jpg**

![medium_worst — train03_00008_jpg.rf.8bad59f8a8b5fa73016ee9635dc92a6e.jpg](examples/medium_worst.jpg)

**hard_best — train02_00125_jpg.rf.c6b713822ff72511401c944128c4eeed.jpg**

![hard_best — train02_00125_jpg.rf.c6b713822ff72511401c944128c4eeed.jpg](examples/hard_best.jpg)

**hard_worst — train02_00210_jpg.rf.271bff88026605c71f96fb6cce1c33a7.jpg**

![hard_worst — train02_00210_jpg.rf.271bff88026605c71f96fb6cce1c33a7.jpg](examples/hard_worst.jpg)
