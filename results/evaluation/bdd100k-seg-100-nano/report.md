# Đánh giá 100 ảnh BDD100K (val) — YOLO và bao lồi trực giao

Ảnh: 100; nhãn xe: 1242 (car 1151, truck 63, bus 23, motorcycle 5).
GT là bitmask từng đối tượng của BDD100K ins_seg; bỏ bicycle. Không chia mức độ khó.
Model: `/Users/levienbang/Desktop/DACS/yolo11/weights/yolo-nano/yolo11n-seg.pt`; dữ liệu train theo metadata: `/ultralytics/ultralytics/cfg/datasets/coco.yaml`.

| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP |
|---|---:|---:|---:|---:|---:|---:|
| box_detection | 69.6% | 52.4% | 59.8% | 27.8% | 19.7% | 84.1% |
| bbox_region | 58.8% | 44.3% | 50.5% | 21.8% | 9.2% | 70.0% |
| convex | 65.1% | 49.0% | 55.9% | 25.9% | 14.9% | 77.2% |
| yolo_mask | 66.0% | 49.7% | 56.7% | 26.3% | 16.5% | 79.0% |
| quickhull | 65.6% | 49.4% | 56.3% | 26.1% | 16.3% | 78.2% |
| ograham | 65.6% | 49.4% | 56.3% | 26.1% | 16.3% | 78.2% |

| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |
|---|---:|---:|---:|---:|
| quickhull | 0.879 | 96.6% | 4.9% | 0.3% |
| ograham | 0.406 | 96.6% | 4.9% | 0.3% |

Motorcycle chỉ có rất ít mẫu: AP lớp này không đáng tin. mAP là trung bình các lớp có GT; bộ tính local, không phải COCOeval.

