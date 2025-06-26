import cv2
import numpy as np
from pymongo import MongoClient
from feature_extraction import extract_features, combine_features
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, DEFAULT_WEIGHTS
from scipy.spatial.distance import euclidean

def connect_mongodb():
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME][COLLECTION_NAME]

def search_similar_images(query_image, limit=5, selected_features=None, weights=None):
    """
    Tìm kiếm ảnh tương tự với ảnh query.
    
    Args:
        query_image: Ảnh cần tìm kiếm
        limit: Số lượng kết quả tối đa
        selected_features: Danh sách các đặc trưng được chọn để sử dụng
        weights: Trọng số cho từng loại đặc trưng (mặc định lấy từ config)
    
    Returns:
        Danh sách các ảnh gần nhất kèm thông tin và điểm số
    """
    # Bước 1: Trích xuất đặc trưng của ảnh query
    features = extract_features(query_image)
    
    # Bước 2: Kết hợp các đặc trưng đã chuẩn hóa
    weights = weights or DEFAULT_WEIGHTS

    if selected_features:
        filtered_weights = {k: weights.get(k, DEFAULT_WEIGHTS.get(k, 0)) 
                           for k in selected_features if k in weights or k in DEFAULT_WEIGHTS}
        total = sum(filtered_weights.values())
        if total > 0:
            weights = {k: v/total for k, v in filtered_weights.items()}
        else:
            weights = {k: 1.0/len(filtered_weights) if filtered_weights else 0 
                      for k in filtered_weights}

    combined_features = combine_features(features, weights)
    
    # Bước 3: Tìm kiếm trong database
    collection = connect_mongodb()
    results = []
    cursor = collection.find()
    
    for doc in cursor:
        db_features = np.array(doc['features'])
        distance = euclidean(combined_features, db_features)
        result = {
            'filename': doc.get('filename', ''),
            'filepath': doc.get('filepath', ''),
            'dimensions': doc.get('dimensions', {}),
            'score': 1.0 / (1.0 + distance)  # Chuyển khoảng cách thành độ tương đồng
        }
        results.append(result)
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results[:limit]