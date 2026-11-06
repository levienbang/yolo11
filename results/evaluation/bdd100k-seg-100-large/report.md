# Đánh giá 100 ảnh BDD100K (val) — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 1242 (car 1143, truck 71, bus 23, motorcycle 5).
GT là bitmask từng đối tượng của BDD100K ins_seg; bỏ bicycle. Không chia mức độ khó.
Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-large/yolo11l-seg.pt`; dữ liệu train theo metadata: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP |
|---|---:|---:|---:|---:|---:|---:|
| box_detection | 76.2% | 61.2% | 67.9% | 49.1% | 35.8% | 86.3% |
| bbox_region | 64.1% | 51.4% | 57.1% | 37.4% | 15.9% | 70.0% |
| convex | 71.4% | 57.3% | 63.6% | 42.1% | 25.7% | 78.1% |
| yolo_mask | 73.4% | 58.9% | 65.4% | 42.8% | 29.3% | 80.0% |
| quickhull | 73.0% | 58.6% | 65.0% | 42.7% | 28.8% | 79.0% |
| ograham | 73.0% | 58.6% | 65.0% | 42.7% | 28.8% | 79.0% |

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.729 | 96.7% | 4.3% | 0.4% |
| ograham | 0.265 | 96.7% | 4.3% | 0.4% |

Motorcycle chỉ có rất ít mẫu: AP lớp này không đáng tin. mAP là trung bình các lớp có GT; bộ tính local, không phải COCOeval.

**all_best — 981d9ee0-00000000.jpg**

![all_best — 981d9ee0-00000000.jpg](examples/all_best.jpg)

**all_worst — 7ed0f644-ac730001.jpg**

![all_worst — 7ed0f644-ac730001.jpg](examples/all_worst.jpg)