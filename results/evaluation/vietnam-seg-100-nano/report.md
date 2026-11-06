# Đánh giá 100 ảnh Việt Nam — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 540. Chọn 70 dễ (ít che khuất), 20 trung bình, 10 khó bằng xem ảnh.
Không dùng số lượng xe để định nghĩa dễ. Tất cả GT xe là polygon; loại toàn bộ ảnh có nhãn xe chỉ là box.

Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-nano/yolo11n-seg.pt`.
Metadata training dataset: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.
Mô hình dùng trực tiếp, không fine-tune trên dữ liệu Việt Nam; chỉ giữ 4 lớp phương tiện khi suy luận.
Không retrain. Đánh giá ảnh trước tracking: chưa đánh giá ID hoặc đếm xe.

## Kết quả

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP | Dice TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| box_detection | 65.3% | 58.0% | 61.4% | 63.5% | 46.4% | 84.2% | 90.9% |
| bbox_region | 56.6% | 50.2% | 53.2% | 53.4% | 19.7% | 67.4% | 80.1% |
| convex | 61.6% | 54.6% | 57.9% | 60.5% | 35.2% | 79.3% | 88.0% |
| yolo_mask | 65.1% | 57.8% | 61.2% | 63.5% | 43.8% | 82.8% | 90.2% |
| quickhull | 63.3% | 56.1% | 59.5% | 61.8% | 38.3% | 80.7% | 88.8% |
| ograham | 63.3% | 56.1% | 59.5% | 61.8% | 38.3% | 80.7% | 88.8% |

box_detection so box với box GT. bbox_region tô kín box so mask GT, chỉ là baseline vùng pixel.
Mask YOLO/QuickHull/O-Graham đều so cùng polygon GT raster hóa tại kích thước ảnh gốc.
mAP dùng matching đúng lớp, confidence-ordered, IoU 0.50:0.05:0.95, 101 điểm recall; không phải COCOeval chính thức.
IoU/Dice TP chỉ tính cặp đúng lớp có IoU ≥ 0.5: cần đọc cùng recall để thấy xe bỏ sót.

## Nhầm lớp

Confusion rows=GT, columns=prediction; ghép không xét lớp theo IoU box/mask, ngưỡng 0.5.
Background column=GT bỏ sót; background row=prediction không ghép được.
| Đầu ra | Car → truck | Truck → car | Accuracy trong cặp đã ghép |
|---|---:|---:|---:|
| box_detection | 10 | 9 | 89.7% |
| yolo_mask | 10 | 11 | 88.9% |
| quickhull | 10 | 11 | 88.9% |
| ograham | 10 | 11 | 88.9% |

## Bao lồi và mask YOLO

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.546 | 91.8% | 9.2% | 0.5% |
| ograham | 0.315 | 91.8% | 9.2% | 0.5% |

## Giới hạn và file

Bộ chọn có chủ đích, không đại diện ngẫu nhiên cho giao thông Việt Nam. Dễ là nhận xét trực quan tương đối.
Split nguồn và số lớp xem selection.json. Chưa xác minh đầy đủ lịch sử huấn luyện của checkpoint nên chưa loại trừ khả năng trùng dữ liệu đã học.
Quy ước xe bán tải là truck; danh sách nhãn đã đổi so với nguồn ở datasets/pickup_relabel.csv. Bỏ bicycle ngoài phạm vi.
Ảnh cùng video có tương quan dù không trùng pixel; không coi đây là 100 cảnh độc lập.
AP bị giới hạn bởi confidence floor, NMS và max_det. Một số mask có thể thiếu/không sát vật thể ngay trong nguồn.
`class_metrics.csv`, `difficulty_metrics.csv`, `per_image.csv`, `metrics.json`, `prediction_records.json` chứa số liệu.
`comparisons/` có đủ 100 ảnh so GT / YOLO / QuickHull / O-Graham; `errors/` chứa ảnh có nhầm lớp qua ghép box.

## Ví dụ

**easy_best — train02_00103_jpg.rf.5673a185cf8e4d82c9c1f25871cf8aaa.jpg**

![easy_best — train02_00103_jpg.rf.5673a185cf8e4d82c9c1f25871cf8aaa.jpg](examples/easy_best.jpg)

**easy_worst — test01_00022_jpg.rf.a438538f49dc27148175263c014f60d9.jpg**

![easy_worst — test01_00022_jpg.rf.a438538f49dc27148175263c014f60d9.jpg](examples/easy_worst.jpg)

**medium_best — test01_00015_jpg.rf.33bf8e352bd1cebf1fbc6bb752734fd8.jpg**

![medium_best — test01_00015_jpg.rf.33bf8e352bd1cebf1fbc6bb752734fd8.jpg](examples/medium_best.jpg)

**medium_worst — train02_00081_jpg.rf.f41231eb1c88ba25cc8bffcef5b243cf.jpg**

![medium_worst — train02_00081_jpg.rf.f41231eb1c88ba25cc8bffcef5b243cf.jpg](examples/medium_worst.jpg)

**hard_best — train01_00216_jpg.rf.b63102e0e673f98cc29d2dc4d7a8d00f.jpg**

![hard_best — train01_00216_jpg.rf.b63102e0e673f98cc29d2dc4d7a8d00f.jpg](examples/hard_best.jpg)

**hard_worst — train02_00210_jpg.rf.271bff88026605c71f96fb6cce1c33a7.jpg**

![hard_worst — train02_00210_jpg.rf.271bff88026605c71f96fb6cce1c33a7.jpg](examples/hard_worst.jpg)
