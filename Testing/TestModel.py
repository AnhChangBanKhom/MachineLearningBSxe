from ultralytics import YOLO
from PIL import Image
import os
import torch

model_path = 'D:/TTTe/code/Version_Kq/best_epoch.pt' #File kết quả huấn luyện
image_path = 'D:/TTTe/code/Data/MHND_BienSoXe.v1i.yolov8/test/images/BSX-2992-_jpg.rf.f35bec5709180211792876e2dcab16f7.jpg' #ảnh test
save_path = 'D:/TTTe/code/KQ/kq2.jpg' #Vị trí lưu ảnh kết quả

# Kiểm tra mô hình
if not os.path.exists(model_path):
    print("không tồn tại mô hình.")
else:
    # Tải tệp mô hình bằng PyTorch
    try:
        model = torch.load(model_path, map_location='cpu')
        print("Tải mô hình thành công bằng")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")


    # Khởi tạo mô hình YOLO
    try:
        model = YOLO(model_path)

        # Hình ảnh test
        image = Image.open(image_path)

        # Dự đoán bằng cách gọi mô hình
        results = model.predict(image)

        # Kiểm tra kết quả dự đoán
        if isinstance(results, list) and len(results) > 0:
            for result in results:
                result.show()

                result.save(save_path)
        else:
            print("Không có dự đoán nào được thực hiện.")
    except Exception as e:
        print(f"Lỗi khi sử dụng mô hình YOLO: {e}")
