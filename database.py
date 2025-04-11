from pymongo import MongoClient
from datetime import datetime
import cv2
import os
import numpy as np
import pickle
from feature_extraction import extract_features, combine_features
from config import *

def connect_mongodb():
    """
    Connect to MongoDB
    """
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

def create_vector_index(collection):
    """
    Create vector index for MongoDB Atlas vector search
    """
    try:
        collection.create_index(
            [("combined_features", "vector")],
            {
                "name": "vector_index",
                "vectorOptions": {
                    "dimensions": 300,  # Lưu ý: dimension ở đây tương ứng với kết quả kết hợp (với HOG đã giảm xuống 300)
                    "similarity": "cosine"
                }
            }
        )
        print("Vector index created successfully")
        return True
    except Exception as e:
        print(f"Failed to create vector index: {e}")
        print("Vector search may not be available")
        return False

def store_image_features(image_folder, rebuild=False):
    """
    Extract features from images and store them in MongoDB
    """
    collection = connect_mongodb()
    
    if rebuild or collection.count_documents({}) == 0:
        if rebuild:
            collection.delete_many({})
        
        # Load mô hình hog_pca
        hog_pca_path = os.path.join(MODELS_PATH, 'hog_pca_model.pkl')
        try:
            with open(hog_pca_path, 'rb') as f:
                hog_pca = pickle.load(f)
        except Exception as e:
            print(f"Error loading hog_pca: {e}")
            hog_pca = None

        # Load visual codebook and other model nếu cần (đã được lưu riêng)
        try:
            # Load visual codebook
            codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
            with open(codebook_path, 'rb') as f:
                codebook = pickle.load(f)
            
            # Nếu có PCA tổng hợp cho những feature ngoài HOG (nếu bạn vẫn cần)
            pca_path = os.path.join(MODELS_PATH, 'pca_model.pkl')
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
        except Exception as e:
            print(f"Error loading additional models: {e}")
            pca = None
        
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Processing {len(image_paths)} images...")
        count = 0
        
        for img_path in image_paths:
            image_name = os.path.basename(img_path)
            species = image_name.split('_')[0] if '_' in image_name else 'unknown'
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image: {img_path}")
                continue
            
            features_dict = extract_features(image)

            # Giảm chiều HOG bằng hog_pca nếu có
            if 'hog' in features_dict and hog_pca is not None:
                try:
                    features_dict['hog'] = hog_pca.transform(features_dict['hog'].reshape(1, -1))[0]
                except Exception as e:
                    print(f"[WARN] HOG PCA transform failed for {image_name}: {e}")
            
            # Kết hợp các feature để tạo vector tìm kiếm
            combined_features = combine_features(features_dict, hog_pca=None)  # đã giảm trước nên không cần truyền hog_pca nữa
            
            # Lưu vào MongoDB
            document = {
                'image_path': img_path,
                'image_name': image_name,
                'species': species,
                'features': {
                    'hsv': features_dict['hsv'].tolist(),
                    'hog': features_dict['hog'].tolist(),
                    'lbp': features_dict['lbp'].tolist(),
                    'sift_bow': features_dict['sift_bow'].tolist()
                },
                'combined_features': combined_features.tolist(),
                'created_at': datetime.now()
            }
            
            collection.insert_one(document)
            count += 1
            
            if count % 10 == 0:
                print(f"Processed {count}/{len(image_paths)} images")
        
        print(f"Successfully stored features for {count} images in MongoDB")
        create_vector_index(collection)
        
        return True
    else:
        print(f"Collection already contains {collection.count_documents({})} documents")
        print("Use rebuild=True to rebuild the database")
        return False
