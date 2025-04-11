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
    Load required models for feature extraction
    """
    models = {}
    
    try:
        # Load visual codebook
        codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(codebook_path, 'rb') as f:
            models['codebook'] = pickle.load(f)
        
        # Load PCA model
        pca_path = os.path.join(MODELS_PATH, 'pca_model.pkl')
        with open(pca_path, 'rb') as f:
            models['pca'] = pickle.load(f)
        
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def search_images_vector_search(query_image, limit=3, selected_features=None):
    # Connect to MongoDB
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    models = load_models()
    if models is None:
        return []
    
    features_dict = extract_features(query_image)
    combined_features = combine_features(features_dict, models['pca'], selected_features=selected_features)
    
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
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    models = load_models()
    if models is None:
        return []
    
    features_dict = extract_features(query_image)
    combined_features = combine_features(features_dict, models['pca'], selected_features=selected_features)
    
    all_docs = list(collection.find({}, {
        "_id": 1,
        "image_path": 1,
        "image_name": 1,
        "species": 1,
        "combined_features": 1
    }))
    
    feature_vectors = np.array([doc["combined_features"] for doc in all_docs])
    kdtree = KDTree(feature_vectors)
    distances, indices = kdtree.query(combined_features.reshape(1, -1), k=limit)
    
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
    Search for similar images with fallback strategy.
    selected_features: list chứa tên của các feature cần sử dụng, ví dụ: ['hsv', 'hog']
    """
    try:
        # Thử vector search trước
        results = search_images_vector_search(query_image, limit, selected_features)
        # Nếu không có kết quả, sử dụng KD-tree
        if not results:
            results = search_images_kdtree(query_image, limit, selected_features)
        return results
    except Exception as e:
        print(f"Search error: {e}")
        # Last resort: sử dụng KD-tree
        return search_images_kdtree(query_image, limit, selected_features)

    """
    Search for similar images with fallback strategy
    """
    try:
        # Try vector search first
        results = search_images_vector_search(query_image, limit)
        
        # If results are empty, fall back to KD-tree
        if not results:
            results = search_images_kdtree(query_image, limit)
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        # Last resort: fall back to KD-tree
        return search_images_kdtree(query_image, limit)