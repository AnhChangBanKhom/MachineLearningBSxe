#Kết nối với drive
from google.colab import drive
drive.mount('/content/drive')

#Tải các thư viện cần thiết
# pip install ultralytics
# pip install torchmetrics
# pip install pytorch_lightning
# pip install tensorboard matplotlib

#import thư viện
from ultralytics import YOLO
import yaml
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải tệp data.yaml
with open('/content/drive/MyDrive/ThucTap/Data/MHND_BienSoXe.v1i.yolov8/data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

#load model
model = YOLO('yolov8x.pt')  # Tải mô hình đã được huấn luyện trước

# Định nghĩa tham số
training_params = {
    'data': '/content/drive/MyDrive/ThucTap/Data/MHND_BienSoXe.v1i.yolov8/data.yaml',  # Đường dẫn tới data.yaml 
    'epochs': 100,  # Số lượng epochs
    'imgsz': 640,  # Kích thước ảnh
    'batch': 16,  # Kích thước batch
    'patience': 5,  # Số epoch dừng sớm 
    'model': 'yolov8x.pt',  # Mô hình đã được huấn luyện trước
    'save_period': 1,  # Lưu checkpoint sau mỗi epoch
    'project': '/content/drive/MyDrive/ThucTap/Checkpoints',  # Thư mục để lưu checkpoints
    'name': 'exp',  # Tên thư mục được tạo ra
    'exist_ok': True,  # Ghi đè dự án hiện có
    'device': '0'  # Thiết bị GPU, '0' cho GPU đầu tiên
}

# Huấn luyện mô hình và ghi log
logging.info("Bắt đầu quá trình huấn luyện")
history = model.train(**training_params)
logging.info("Kết thúc quá trình huấn luyện")

# Đánh giá mô hình
logging.info("Đánh giá mô hình")
metrics = model.val()
logging.info(f"Metrics: {metrics}")

# Thực hiện dự đoán với mô hình tốt nhất và lưu mô hình
logging.info("Dự đoán và lưu mô hình tốt nhất")
results = model.predict(source='/content/drive/MyDrive/ThucTap/Data/MHND_BienSoXe.v1i.yolov8/test/images/BSX-1070-_jpg.rf.987661d088bb41a6d4bccdaf193a140e.jpg', save=True)
model.save('/content/drive/MyDrive/ThucTap/Checkpoints/best_model.pt')


# Hiển thị log chi tiết
# logging.info("Kết quả huấn luyện:")
# logging.info(history)

# Hiển thị độ chính xác và lý do chọn epoch
# best_epoch = history['epoch'][-1]
# best_accuracy = max(history['results']['accuracy'])
# logging.info(f"Best Epoch: {best_epoch}")
# logging.info(f"Best Accuracy: {best_accuracy}")
# logging.info(f"Early Stopping đã kích hoạt tại epoch {best_epoch} với độ chính xác tốt nhất là {best_accuracy}")
