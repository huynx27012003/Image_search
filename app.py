# app.py
from flask import Flask, request, render_template, jsonify
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
from search import search_similar_images
from database import connect_mongodb

from config import DEFAULT_WEIGHTS

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Tạo thư mục uploads nếu cần
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Khởi tạo kết nối MongoDB
collection = connect_mongodb()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Đọc ảnh từ request
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        query_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Chuyển ảnh truy vấn thành Base64 để trả về client
        _, buffer = cv2.imencode('.jpg', query_image)
        query_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Lấy các đặc trưng đã chọn từ form (nếu có)
        selected = request.form.getlist('features') or list(DEFAULT_WEIGHTS.keys())
        # Lấy trọng số từng đặc trưng, mặc định từ config
        weights = {
            f: float(request.form.get(f'weight_{f}', DEFAULT_WEIGHTS[f]))
            for f in DEFAULT_WEIGHTS
        }

        # Tìm kiếm ảnh tương tự
        results = search_similar_images(query_image, limit=5, selected_features=selected, weights=weights)    # Chuẩn bị response JSON với ảnh mã hóa base64
        response = {
            'query_image': query_image_b64,
            'results': []
        }
        
        for idx, res in enumerate(results):
            # Lấy ảnh từ DB dưới dạng Base64 (nếu có)
            # Giả sử bạn có trường 'image_base64' trong DB
            db_result = collection.find_one({'filename': res.get('filename')})
            
            image_data = ""
            if db_result and 'image_base64' in db_result:
                # Nếu ảnh được lưu dưới dạng Base64 trong DB, lấy trực tiếp
                image_data = db_result['image_base64']
            else:
                # Nếu không, đọc file từ đường dẫn và convert sang Base64
                filepath = res.get('filepath', '')
                if filepath and os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        _, buffer = cv2.imencode('.jpg', img)
                        image_data = base64.b64encode(buffer).decode('utf-8')
            
            response['results'].append({
                'rank': idx + 1,
                'image_name': res.get('filename', ''),
                'species': os.path.basename(os.path.dirname(res.get('filepath', ''))),
                'score': res.get('score', 0),
                'image_data': image_data
            })

        return jsonify(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
