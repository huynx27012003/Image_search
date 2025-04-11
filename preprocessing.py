import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from feature_extraction import extract_features, combine_features
from config import *

def create_visual_codebook(image_folder, sample_size=50):
    """
    Create a visual codebook from SIFT descriptors of sample images
    """
    print("Creating visual codebook...")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_paths) > sample_size:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    all_descriptors = []
    sift = cv2.SIFT_create()
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, IMAGE_SIZE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        print(f"Clustering {len(all_descriptors)} descriptors...")
        kmeans = KMeans(n_clusters=SIFT_CODEBOOK_SIZE, random_state=42, verbose=0)
        kmeans.fit(all_descriptors)
        
        codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(codebook_path, 'wb') as f:
            pickle.dump(kmeans, f)
        
        print(f"Visual codebook created and saved to {codebook_path}")
        return kmeans
    else:
        print("Failed to collect SIFT descriptors")
        return None

def train_pca(image_folder, sample_size=100):
    """
    Train PCA on combined features from sample images
    """
    print("Training PCA...")
    
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_paths) > sample_size:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    try:
        codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(codebook_path, 'rb') as f:
            codebook = pickle.load(f)
    except:
        print("Visual codebook not found. Creating one...")
        codebook = create_visual_codebook(image_folder)
        if codebook is None:
            return None
    
    all_features = []
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        features_dict = extract_features(image)
        combined = combine_features(features_dict)
        all_features.append(combined)
    
    if not all_features:
        print("Failed to collect features for PCA")
        return None
    
    all_features = np.vstack(all_features)
    
    # Sửa tại đây: sử dụng all_features thay cho biến không tồn tại
    n_components = min(250, all_features.shape[0], all_features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(all_features)
    
    pca_path = os.path.join(MODELS_PATH, 'pca_model.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    
    print(f"PCA model trained and saved to {pca_path}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca

def train_hog_pca(image_folder, sample_size=100, n_components=300):
    """
    Huấn luyện PCA riêng cho đặc trưng HOG:
    - Thu thập vector HOG từ tập ảnh
    - Huấn luyện PCA với số thành phần là n_components (điều chỉnh nếu vượt quá)
    """
    print("Training hog PCA...")
    hog_features_list = []
    
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_paths) > sample_size:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, IMAGE_SIZE)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Tính vector HOG
        from skimage.feature import hog
        hog_feature = hog(gray_image,
                          orientations=HOG_ORIENTATIONS, 
                          pixels_per_cell=HOG_PIXELS_PER_CELL,
                          cells_per_block=HOG_CELLS_PER_BLOCK, 
                          block_norm='L2-Hys',
                          visualize=False)
        hog_features_list.append(hog_feature)
    
    if not hog_features_list:
        print("Không thu thập được đặc trưng HOG cho huấn luyện PCA")
        return None
    
    hog_features_array = np.vstack(hog_features_list)
    n_samples, n_features = hog_features_array.shape
    # Số thành phần tối đa có thể là min(n_samples, n_features)
    n_components_adjusted = min(n_components, n_samples, n_features)
    if n_components_adjusted < n_components:
        print(f"Điều chỉnh n_components từ {n_components} xuống {n_components_adjusted} "
              f"vì dữ liệu HOG chỉ có n_samples={n_samples}, n_features={n_features}")
    
    from sklearn.decomposition import PCA
    hog_pca = PCA(n_components=n_components_adjusted)
    hog_pca.fit(hog_features_array)
    
    hog_pca_path = os.path.join(MODELS_PATH, 'hog_pca_model.pkl')
    with open(hog_pca_path, 'wb') as f:
        pickle.dump(hog_pca, f)
    
    print(f"Hog PCA model trained and saved to {hog_pca_path}")
    return hog_pca

def preprocess_dataset(image_folder):
    """
    Preprocess the entire dataset by creating the visual codebook, training PCA,
    and training the hog_pca model.
    """
    # Tạo visual codebook cho SIFT
    codebook = create_visual_codebook(image_folder)
    
    # Huấn luyện PCA cho các feature kết hợp (nếu cần)
    pca = train_pca(image_folder)
    
    # Huấn luyện hog_pca riêng cho HOG
    hog_pca = train_hog_pca(image_folder)
    
    return codebook, pca
