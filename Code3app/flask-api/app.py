from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Nhận dữ liệu ảnh từ yêu cầu POST
        image_data = request.files['image']
        # Lưu ảnh vào một tệp tạm thời
        image_path = '/tmp/temp_image.jpg'  # Đường dẫn đến tệp tạm thời
        image_data.save(image_path)
        # Gọi file Python riêng của bạn để xử lý ảnh
        process = subprocess.Popen(['python', '../code/Code2app/xuly2.py', image_path], stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            return jsonify({'error': error.decode()}), 500
        else:
            return jsonify({'result': output.decode()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
