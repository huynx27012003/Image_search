import cv2
import numpy as np
from pymongo import MongoClient
from sklearn.neighbors import KDTree
import pickle
import os
from feature_extraction import extract_features, combine_features
from config import *

def load_models():
    """
    Load required models for feature extraction, bao gồm visual codebook, PCA (nếu cần) và hog_pca.
    """
    models = {}
    
    try:
        # Load visual codebook
        codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(codebook_path, 'rb') as f:
            models['codebook'] = pickle.load(f)
        
        # Load PCA tổng hợp nếu cần (cho SIFT/BoW, HSV, LBP nếu có)
        pca_path = os.path.join(MODELS_PATH, 'pca_model.pkl')
        with open(pca_path, 'rb') as f:
            models['pca'] = pickle.load(f)
        
        # Load mô hình hog_pca
        hog_pca_path = os.path.join(MODELS_PATH, 'hog_pca_model.pkl')
        with open(hog_pca_path, 'rb') as f:
            models['hog_pca'] = pickle.load(f)
        
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def combine_stored_features(doc, selected_features):
    """
    Kết hợp các vector đặc trưng đã lưu trong DB theo danh sách selected_features.
    Đầu tiên, lấy dictionary 'features' đã lưu trong tài liệu, sau đó ghép nối các vector tương ứng.
    """
    vects = []
    features = doc.get("features", {})
    
    if "hsv" in selected_features and "hsv" in features:
        vects.append(np.array(features["hsv"]))
    if "hog" in selected_features and "hog" in features:
        vects.append(np.array(features["hog"]))
    if "lbp" in selected_features and "lbp" in features:
        vects.append(np.array(features["lbp"]))
    if "sift" in selected_features and "sift_bow" in features:
        vects.append(np.array(features["sift_bow"]))
    
    if vects:
        return np.concatenate(vects)
    else:
        return np.array([])

def search_images_vector_search(query_image, limit=3, selected_features=None):
    """
    Thử sử dụng vector search nếu index khả dụng, ngược lại chuyển sang KD-tree.
    Ở đây ta vẫn sử dụng truy vấn từ vector kết hợp đầy đủ.
    """
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    models = load_models()
    if models is None:
        return []
    
    features_dict = extract_features(query_image)
    # Khi sử dụng vector search, ta vẫn tính query vector dựa trên full kết hợp theo selected_features:
    combined_features = combine_features(features_dict, hog_pca=models.get('hog_pca'), selected_features=selected_features)
    
    try:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "combined_features",
                    "queryVector": combined_features.tolist(),
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "image_path": 1,
                    "image_name": 1,
                    "species": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ])
        return list(results)
    except Exception as e:
        print(f"Vector search failed: {e}")
        print("Falling back to KD-tree search")
        return search_images_kdtree(query_image, limit, selected_features)

def search_images_kdtree(query_image, limit=3, selected_features=None):
    """
    Tìm kiếm sử dụng KD-tree, xây dựng tree trên các vector được tạo từ các feature lưu riêng,
    sao cho nếu người dùng chỉ chọn tập con các feature, chiều của vector query và stored đều khớp.
    """
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    models = load_models()
    if models is None:
        return []
    
    features_dict = extract_features(query_image)
    # Tính query vector theo selected_features (với hog_pca áp dụng cho HOG)
    combined_query = combine_features(features_dict, hog_pca=models.get('hog_pca'), selected_features=selected_features)
    
    all_docs = list(collection.find({}))
    
    stored_vectors = []
    for doc in all_docs:
        vec = combine_stored_features(doc, selected_features)
        stored_vectors.append(vec)
    
    stored_vectors = np.array(stored_vectors)
    
    try:
        kdtree = KDTree(stored_vectors)
        distances, indices = kdtree.query(combined_query.reshape(1, -1), k=limit)
    except Exception as e:
        print(f"KDTree query error: {e}")
        return []
    
    results = []
    for i in range(min(limit, len(indices[0]))):
        idx = indices[0][i]
        doc = all_docs[idx]
        similarity = 1.0 / (1.0 + distances[0][i])
        results.append({
            "image_path": doc["image_path"],
            "image_name": doc["image_name"],
            "species": doc["species"],
            "score": similarity
        })
    
    return results

def search_similar_images(query_image, limit=3, selected_features=None):
    """
    Tìm kiếm ảnh tương tự với chiến lược fallback:
    - Thử vector search trước
    - Nếu không có kết quả, sử dụng KD-tree
    selected_features: list chứa tên của các feature cần sử dụng, ví dụ: ['hsv', 'hog']
    """
   
    return search_images_kdtree(query_image, limit, selected_features)
