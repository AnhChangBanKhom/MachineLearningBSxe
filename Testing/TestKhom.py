from ultralytics import YOLO
from PIL import Image

# Khởi tạo mô hình
model = YOLO('D:/TTTe/code/Version_Kq/best_epoch_19.pt')

# Đường dẫn đến hình ảnh đầu vào
image_path = 'D:\TTTe\code\AnhTest\KT (2).jpg'

# Đọc hình ảnh
image = Image.open(image_path)

# Dự đoán bằng cách gọi phương thức predict của mô hình
results = model.predict(image)

# Kiểm tra kết quả dự đoán
if results:
    # Lấy thông tin các khung bao và nhãn từ kết quả dự đoán
    boxes = results[0].boxes.xyxy  # Tọa độ khung bao
    classes = results[0].boxes.cls  # Các lớp (chỉ số) của các đối tượng
    names = results[0].names  # Tên các lớp đối tượng

    # Tạo danh sách các đối tượng cùng với tọa độ khung bao
    detected_objects = [(names[int(cls)], (x1, y1, x2, y2)) for cls, (x1, y1, x2, y2) in zip(classes, boxes)]

    # Chuyển đổi tọa độ YOLO thành tọa độ trung bình của các khung bao
    converted_labels = []
    for (x1, y1, x2, y2) in boxes:
        x_cent = (x1 + x2) / 2
        y_cent = (y1 + y2) / 2
        converted_labels.append((x_cent, y_cent))

    # Sắp xếp theo tọa độ trung bình từ trên xuống dưới (theo y) và từ trái sang phải (theo x)
    sorted_objects = sorted(zip(detected_objects, converted_labels), key=lambda x: (-x[1][1], x[1][0]))

    # In ra danh sách các đối tượng đã sắp xếp
    print("Danh sách các đối tượng đã sắp xếp theo tọa độ trung bình:")
    for obj in sorted_objects:
        print(obj[0][0], "tại tọa độ:", obj[1])

    # Hiển thị ảnh dự đoán
    results[0].show()

    # Lưu ảnh dự đoán
    save_path = 'D:/TTTe/code/KQ'
    results[0].save(save_path)
else:
    print("Không có dự đoán nào được thực hiện.")
