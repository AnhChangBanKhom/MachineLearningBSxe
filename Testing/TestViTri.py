from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
# from detectron2.structures.boxes import Boxes

# Khởi tạo mô hình
model = YOLO('F:/Data_TT/runs/weights/best.pt')

# Đường dẫn đến hình ảnh đầu vào
# image_path = '/content/drive/MyDrive/TTTT/MHND_BienSoXe/trains/images/BSX-10-_jpg.rf.de40170f0a5997924c8581b88f05dc91.jpg'
image_path = 'F:/Data_TT/anh.jpg'

# Đọc hình ảnh
image = Image.open(image_path)

# # Dự đoán bằng cách gọi phương thức predict của mô hình
# results = model.predict(image)

# # Lặp qua từng kết quả dự đoán trong danh sách results
# for result in results:
#     # Hiển thị ảnh dự đoán
#     result.show()

#     # Lưu ảnh dự đoán
#     save_path = '/content/drive/MyDrive/TTTT/MHND_BienSoXe/train4/kq.jpg'
#     result.save(save_path)

# Dự đoán bằng cách gọi phương thức predict của mô hình
results = model.predict(image)
# print(results)

# Kiểm tra kết quả dự đoán
if results:
    # Lấy thông tin các khung bao và nhãn từ kết quả dự đoán
    boxes = results[0].boxes.xyxy  # Tọa độ khung bao
    classes = results[0].boxes.cls  # Các lớp (chỉ số) của các đối tượng
    names = results[0].names  # Tên các lớp đối tượng

    # Tạo danh sách các đối tượng cùng với tọa độ khung bao
    detected_objects = [(names[int(cls)], (x1, y1, x2, y2)) for cls, (x1, y1, x2, y2) in zip(classes, boxes)]
    # detected_objects.sort(key=lambda obj: (obj[1][1], obj[1][0]))

    # Chuyển đổi tọa độ YOLO thành tọa độ góc
    converted_labels = []
    for i in boxes:
        x1,x2,y1,y2 = i
        x_cent = (x1 + x2) / 2
        y_cent = (y1 + y2) / 2
        converted_labels.append((x_cent,y_cent))

    print(converted_labels)

    # Sắp xếp theo tọa độ trung bình
    sorted_objects = sorted(zip(detected_objects, converted_labels), key=lambda x: (x[1][1], x[1][0]))

    # In ra danh sách các đối tượng đã sắp xếp
    print("Danh sách các đối tượng đã sắp xếp theo tọa độ trung bình:")
    for obj in sorted_objects:
        print(obj[0][0], "tại tọa độ:", obj[1])


    # Hiển thị ảnh dự đoán
    results[0].show()

    # Lưu ảnh dự đoán
    save_path = 'F:/Data_TT/kq.jpg'
    results[0].save(save_path)
else:
    print("Không có dự đoán nào được thực hiện.")# t xử lý được cái biển số mà nó còn sai 1 số kí tự