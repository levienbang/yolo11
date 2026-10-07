# Benchmark tốc độ QuickHull và O-Graham

Script: `scripts/benchmark_hulls.py`. Mac MPS host, chỉ đo CPU dựng bao (không gồm YOLO/raster).
Mỗi điểm dữ liệu: 3 lượt warm-up, median 30 lượt. Contour thật: 540 xe Việt Nam, median 5 lượt/xe.

## Điểm tổng hợp (median ms)

| n | uniform QuickHull | uniform O-Graham | circle QuickHull | circle O-Graham |
|---:|---:|---:|---:|---:|
| 100 | 0.27 | 0.29 | 0.65 | 0.32 |
| 500 | 1.41 | 1.31 | 9.02 | 1.43 |
| 1000 | 2.21 | 2.35 | 31.2 | 2.80 |
| 2000 | 3.79 | 4.59 | 101.7 | 5.54 |
| 5000 | 9.14 | 11.66 | 404.6 | 12.7 |
| 10000 | 22.5 | 25.8 | 918.9 | 22.8 |

- Uniform (bao ít đỉnh, h ≈ 30–70): hai thuật toán gần như tương đương, tăng gần tuyến tính theo n.
- Circle (mọi điểm đều có thể là đỉnh, h lớn): QuickHull tăng gần bậc hai (n tăng 10× → ~100×),
  O-Graham vẫn gần n log n. Ở n=10000 O-Graham nhanh hơn ~40×.

## Contour xe thật (540 xe, median 165 điểm/xe, tối đa 1768)

| Thuật toán | median ms | mean ms | p95 ms | tổng ms |
|---|---:|---:|---:|---:|
| QuickHull | 0.648 | 1.509 | 6.35 | 815 |
| O-Graham | 0.458 | 0.658 | 1.98 | 356 |

O-Graham nhanh hơn ~1.4× (median theo từng xe); chênh lệch lớn hơn ở xe nhiều điểm biên (xem `hull_speed.png`).

## Tính đúng

Ở mọi n, hai bao có cùng tập đỉnh góc và raster hóa lệch 0 pixel. QuickHull đôi khi trả thêm
đỉnh thẳng hàng trên cạnh (ví dụ circle n=10000: 7397 so với 7238 đỉnh), không đổi hình dạng.

File: `synthetic.csv`, `summary.json`, `hull_speed.png`.
