# Đánh giá 100 ảnh BDD100K (val) — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 1242 (car 1151, truck 63, bus 23, motorcycle 5).
GT là bitmask từng đối tượng của BDD100K ins_seg; bỏ bicycle. Không chia mức độ khó.
Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-large/bestseg.pt`; dữ liệu train theo metadata: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP |
|---|---:|---:|---:|---:|---:|---:|
| box_detection | 76.1% | 61.1% | 67.8% | 47.6% | 34.5% | 86.3% |
| bbox_region | 64.0% | 51.4% | 57.0% | 36.1% | 15.3% | 70.0% |
| convex | 71.3% | 57.2% | 63.5% | 40.7% | 24.7% | 78.1% |
| yolo_mask | 73.3% | 58.9% | 65.3% | 41.4% | 28.1% | 80.0% |
| quickhull | 72.9% | 58.5% | 64.9% | 41.3% | 27.7% | 79.0% |
| ograham | 72.9% | 58.5% | 64.9% | 41.3% | 27.7% | 79.0% |

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.849 | 96.7% | 4.3% | 0.4% |
| ograham | 0.370 | 96.7% | 4.3% | 0.4% |

Motorcycle chỉ có rất ít mẫu: AP lớp này không đáng tin. mAP là trung bình các lớp có GT; bộ tính local, không phải COCOeval.

