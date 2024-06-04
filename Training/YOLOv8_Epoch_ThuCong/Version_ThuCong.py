# Do em chạy code trên colab nên em sẽ tổng hợp theo cấu trúc của colab


#dowload thư viện cần thiết
#pip install ultralytics

from ultralytics import YOLO
import os

# kết nối với Google Drive của tài khoảng
from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn tới file data.yaml
data_yaml = '/content/drive/MyDrive/ThucTap/MHND_BienSoXe.v1i.yolov8/data.yaml'

# Kiểm tra xem file data.yaml
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"File {data_yaml} đường dẫn sai")

# Khởi tạo mô hình YOLOv8
model = YOLO('yolov8x.pt') # phiên bản mới nhất trong YOLOv8

# Huấn luyện mô hình
model.train(data=data_yaml, epochs=10, imgsz=640, batch=16, name='yolov8x')

# Lưu mô hình đã huấn luyện
model.save('/content/drive/MyDrive/ThucTap/MHND_BienSoXe.v1i.yolov8/best_model.pt')

print("Mô hình đã huấn luyện và mô hình đã được lưu.")

