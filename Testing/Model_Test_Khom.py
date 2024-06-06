from ultralytics import YOLO
from PIL import Image

# Khởi tạo mô hình
model = YOLO('D:/TTTe/code/Version_Kq/best_epoch_19.pt')

# Đường dẫn đến hình ảnh đầu vào
image_path = 'D:/TTTe/code/AnhTest/KT.jpg'

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

    # Tính toán tọa độ trung bình theo trục y của toàn bộ biển số
    all_y_centers = [y for _, (x, y) in zip(detected_objects, converted_labels)]
    avg_y = sum(all_y_centers) / len(all_y_centers)

    # Phân loại các đối tượng vào phần trên hoặc phần dưới dựa trên tọa độ trung tâm của chúng
    upper_half = [(obj, center) for obj, center in zip(detected_objects, converted_labels) if center[1] <= avg_y]
    lower_half = [(obj, center) for obj, center in zip(detected_objects, converted_labels) if center[1] > avg_y]

    # Sắp xếp các đối tượng trong phần trên từ trái sang phải
    sorted_upper_half = sorted(upper_half, key=lambda x: x[1][0])

    # Sắp xếp các đối tượng trong phần dưới từ trái sang phải
    sorted_lower_half = sorted(lower_half, key=lambda x: x[1][0])

    # Kết hợp lại hai danh sách đã sắp xếp
    # sorted_objects = sorted_upper_half + sorted_lower_half
    sorted_objects = sorted_upper_half + sorted_lower_half

    # In ra danh sách các đối tượng đã sắp xếp
    print("In ra box")
    for obj in sorted_objects:
        if obj[0][0] == 'box':
            print("Khung biển số [", obj[0][0], "] mang tọa độ tại:", obj[1])
    print("\n")

    # In ra danh sách các đối tượng đã sắp xếp
    print("Danh sách các ký tự đã sắp xếp:")
    for obj in sorted_objects:
        if obj[0][0] != 'box':
                print(obj[0][0], "tại tọa độ:", obj[1])
            
    # result_list = [] 
    for obj in sorted_objects:
        if obj[0][0] != 'box':
            print(obj[0][0], end=' ')
            # result_list.append(obj[0][0]) 
            
    # Tên tệp tin để lưu trữ kết quả
    file_name = "D:/TTTe/code/KQ/BienSo.txt"

    # Mở tệp tin để ghi, nếu tệp không tồn tại, nó sẽ tự động tạo mới
    with open(file_name, "w") as file:
        for obj in sorted_objects:
            if obj[0][0] != 'box':
                value = obj[0][0]
                file.write(value + '')  # Ghi giá trị vào tệp tin

    # In ra thông báo khi việc ghi hoàn thành
    print("\nĐã lưu trữ vào tệp tin '{}'.".format(file_name))
    
    # Hiển thị ảnh dự đoán
    results[0].show()

    # Lưu ảnh dự đoán
    save_path = 'D:/TTTe/code/KQ'
    results[0].save(save_path)
else:
    print("Không có dự đoán nào được thực hiện.")
