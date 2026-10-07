# YOLO11 nano COCO so với bestseg large — 100 ảnh Việt Nam

Chỉ đánh giá car, truck, bus, motorcycle. Cùng 100 ảnh, 540 đối tượng (188 car, 101 truck, 26 bus, 225 motorcycle).
Nano: weights/yolo-nano/yolo11n-seg.pt, pretrained COCO chính thức. Large: weights/yolo-large/yolo11l-seg.pt (trước đây tên bestseg.pt) hiện có; metadata ghi COCO, chưa xác minh checkpoint fine-tune Việt Nam.
Nguồn nano: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt

Cấu hình chung: imgsz=960, MPS, confidence=0.20, AP confidence floor=0.001, NMS IoU=0.7, max_det=300, retina_masks=True. Giữ 4 lớp khi predict, không huấn luyện.

## QuickHull và O-Graham

| Model | Bao | TP | FP | FN | Precision | Recall | F1 | mAP50 | mAP50–95 | ms/bao |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n-seg COCO | quickhull | 304 | 175 | 236 | 63.47% | 56.30% | 59.67% | 61.51% | 38.13% | 0.543 |
| YOLO11n-seg COCO | ograham | 304 | 175 | 236 | 63.47% | 56.30% | 59.67% | 61.51% | 38.13% | 0.325 |
| YOLO11 large bestseg | quickhull | 372 | 145 | 168 | 71.95% | 68.89% | 70.39% | 70.87% | 47.44% | 0.585 |
| YOLO11 large bestseg | ograham | 372 | 145 | 168 | 71.95% | 68.89% | 70.39% | 70.87% | 47.44% | 0.344 |

## Mask YOLO trước khi dựng bao

| Model | TP | FP | FN | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n-seg COCO | 313 | 166 | 227 | 57.96% | 63.20% | 43.53% |
| YOLO11 large bestseg | 386 | 131 | 154 | 71.48% | 74.01% | 54.71% |

FN phía trên gồm bỏ sót, sai lớp và vùng IoU dưới ngưỡng. Không đồng nghĩa toàn bộ là không phát hiện được xe.

## Tách nhầm lớp và vùng không khớp

Ghép không xét lớp, confidence-ordered greedy, IoU >= 0.50. Cách ghép khác bảng TP/FP/FN đúng lớp nên không cộng chéo giữa các bảng.

| Model | Vùng | GT không ghép đủ | Đã ghép nhưng sai lớp | Prediction không ghép |
|---|---|---:|---:|---:|
| YOLO11n-seg COCO | yolo_mask | 199 | 37 | 138 |
| YOLO11n-seg COCO | quickhull | 208 | 36 | 147 |
| YOLO11n-seg COCO | ograham | 208 | 36 | 147 |
| YOLO11 large bestseg | yolo_mask | 131 | 31 | 108 |
| YOLO11 large bestseg | quickhull | 145 | 31 | 122 |
| YOLO11 large bestseg | ograham | 145 | 31 | 122 |

GT không ghép đủ có thể là không phát hiện, confidence thấp, vùng lệch hoặc ảnh hưởng ghép nhiều xe; không thể gọi tất cả là detector bỏ sót.

## Recall từng lớp — mask YOLO

| Lớp | Nano recall | Nano FN | Large recall | Large FN |
|---|---:|---:|---:|---:|
| car | 74.47% | 48 | 79.26% | 39 |
| truck | 59.41% | 41 | 68.32% | 32 |
| bus | 76.92% | 6 | 80.77% | 5 |
| motorcycle | 41.33% | 132 | 65.33% | 78 |

## Nhận xét và giới hạn

- Nano kém hơn large chủ yếu ở phát hiện/phân đoạn đủ khớp, đặc biệt xe máy. Nhầm lớp cũng tăng (37 so với 31 qua ghép mask không xét lớp).
- QuickHull và O-Graham có cùng kết quả chất lượng trong cả hai model. Bao làm giảm điểm segmentation so mask YOLO ở cả hai model.
- Nano có inference trung bình nhanh hơn trong hai lượt đo local; đây không phải phép benchmark tốc độ đã warm-up/lặp nhiều lần. Không suy rộng sang Jetson.
- ms/bao chỉ đo tính bao, không gồm suy luận YOLO hay raster hóa.
- mAP dùng bộ tính local 101 điểm, không phải COCOeval chính thức. P/R/FN ở confidence 0.20 và IoU 0.50.
- Bộ ảnh được chọn có chủ đích; có tương quan camera/video. Chưa xác minh độc lập với dữ liệu fine-tune. Không dùng kết quả này để quy toàn bộ chênh lệch cho fine-tuning hay kích thước model.

## Dữ liệu tái lập

- YOLO11n-seg COCO: model SHA256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`; kết quả `outputs/evaluation/vietnam-seg-100-nano`.
- YOLO11 large bestseg: model SHA256 `cabe90049795dfc9a370b7934d6dec7f6b9e44a20e573b0ff81b7e205512c872`; kết quả `outputs/evaluation/vietnam-seg-100-large`.
Manifest SHA256: 74db0812ba055f6d14915bf878ab9e85f1f2fb95473b9025df5a1c5919928c68
(Ảnh ví dụ, `comparisons/`, `errors/` và `prediction_records.json` chỉ có trong `outputs/` ở máy chạy; chạy lại để sinh.)
