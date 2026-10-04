# Bộ 100 ảnh đánh giá phương tiện Việt Nam

100 ảnh có nhãn polygon: 70 dễ, 20 trung bình, 10 khó.
4 lớp: 0 car, 1 truck, 2 bus, 3 motorcycle; tổng 540 đối tượng.

Nguồn: [Vietnam Traffic Vehicle Detectio — phan-ky-dat, version 1](https://universe.roboflow.com/phan-ky-dat/vietnam-traffic-vehicle-detectio/dataset/1).
Giấy phép nguồn: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Thông tin xuất dataset gốc được giữ trong README.dataset.txt và README.roboflow.txt.

Thay đổi từ nguồn: chọn 100 ảnh bằng kiểm tra trực quan, chỉ giữ ảnh có nhãn xe
polygon hợp lệ, bỏ bicycle và đổi ID lớp về thứ tự phía trên. Không vẽ mask từ box.
Ảnh và tọa độ polygon xe được giữ như nguồn.

- images/: 100 ảnh đã chọn.
- labels/: 100 file nhãn polygon tương ứng.
- manifest.csv: split nguồn, số xe, đường dẫn tương đối để đánh giá và hash ảnh.
- dataset.yaml: tên lớp và đường dẫn val tương đối với file YAML.
- contact_sheets/: ảnh tổng hợp đã dùng để kiểm tra trực quan.
- reviewed.csv, review_decisions.json, selection.json: thông tin chọn ảnh.

Các đường dẫn original_image_path/original_label_path trong manifest là thông tin
nguồn trên máy tạo bộ ảnh; script đánh giá dùng image_path/label_path tương đối.
Bộ chọn có chủ đích, chưa xác minh độc lập với dữ liệu fine-tune.

Chạy từ thư mục repo: python evaluate_yolo11n_seg.py --device mps.
Weight tải riêng; video và kết quả chạy không được đưa lên Git.
