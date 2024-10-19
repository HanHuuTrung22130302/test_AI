from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Tải mô hình
model = load_model('saved_model/cat_bird_dog_classifier_v2.h5')  # Đảm bảo tên file đúng

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Chuyển đổi file thành định dạng mà load_img có thể xử lý
    img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Chia cho 255 để chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

    # Dự đoán
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]

    # Định nghĩa nhãn cho các lớp
    labels = {0: 'bird', 1: 'cat', 2: 'dog'}  # Thêm chó vào đây
    predicted_class = labels[class_index]
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
