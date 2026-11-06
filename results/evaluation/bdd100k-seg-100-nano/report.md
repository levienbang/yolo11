# Đánh giá 100 ảnh BDD100K (val) — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 1242 (car 1143, truck 71, bus 23, motorcycle 5).
GT là bitmask từng đối tượng của BDD100K ins_seg; bỏ bicycle. Không chia mức độ khó.
Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-nano/yolo11n-seg.pt`; dữ liệu train theo metadata: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP |
|---|---:|---:|---:|---:|---:|---:|
| box_detection | 69.3% | 52.2% | 59.5% | 28.2% | 20.0% | 84.1% |
| bbox_region | 58.5% | 44.0% | 50.3% | 22.2% | 9.3% | 69.9% |
| convex | 64.8% | 48.8% | 55.7% | 26.3% | 15.2% | 77.2% |
| yolo_mask | 65.7% | 49.4% | 56.4% | 26.8% | 16.8% | 79.0% |
| quickhull | 65.2% | 49.1% | 56.0% | 26.5% | 16.6% | 78.2% |
| ograham | 65.2% | 49.1% | 56.0% | 26.5% | 16.6% | 78.2% |

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.706 | 96.6% | 4.9% | 0.3% |
| ograham | 0.296 | 96.6% | 4.9% | 0.3% |

Motorcycle chỉ có rất ít mẫu: AP lớp này không đáng tin. mAP là trung bình các lớp có GT; bộ tính local, không phải COCOeval.

**all_best — 981d9ee0-00000000.jpg**

![all_best — 981d9ee0-00000000.jpg](examples/all_best.jpg)

**all_worst — 7ed0f644-ac730001.jpg**

![all_worst — 7ed0f644-ac730001.jpg](examples/all_worst.jpg)