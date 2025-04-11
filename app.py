from flask import Flask, request, render_template, jsonify
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
from search import search_similar_images
from config import *

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Đọc ảnh truy vấn
        query_image = cv2.imread(file_path)
        
        # Lấy danh sách các feature được chọn từ form checkbox
        selected_features = request.form.getlist('selected_features')
        # Nếu không chọn gì, sử dụng mặc định tất cả
        if not selected_features:
            selected_features = ['hsv', 'hog', 'lbp', 'sift']
        
        # Tìm kiếm với các feature đã chọn
        results = search_similar_images(query_image, limit=3, selected_features=selected_features)
        
        # Chuẩn bị phản hồi JSON
        response = []
        for i, result in enumerate(results):
            response.append({
                'rank': i + 1,
                'image_name': result['image_name'],
                'species': result['species'],
                'score': float(result['score']),
                'image_data': encode_image_to_base64(result['image_path'])
            })
        
        query_image_data = encode_image_to_base64(file_path)
        return jsonify({
            'query_image': query_image_data,
            'results': response
        })
    
    return jsonify({'error': 'Invalid file'})

if __name__ == '__main__':
    app.run(debug=True)
