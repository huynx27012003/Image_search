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
    
    # Get all image paths
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # If there are too many images, sample them
    if len(image_paths) > sample_size:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    # Collect all SIFT descriptors
    all_descriptors = []
    sift = cv2.SIFT_create()
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Resize and convert to grayscale
        image = cv2.resize(image, IMAGE_SIZE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    
    # Concatenate all descriptors
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        
        # Cluster descriptors using K-means
        print(f"Clustering {len(all_descriptors)} descriptors...")
        kmeans = KMeans(n_clusters=SIFT_CODEBOOK_SIZE, random_state=42, verbose=0)
        kmeans.fit(all_descriptors)
        
        # Save the codebook
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
    
    # Get all image paths
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # If there are too many images, sample them
    if len(image_paths) > sample_size:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:sample_size]
    
    # Load visual codebook
    try:
        codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(codebook_path, 'rb') as f:
            codebook = pickle.load(f)
    except:
        print("Visual codebook not found. Creating one...")
        codebook = create_visual_codebook(image_folder)
        if codebook is None:
            return None
    
    # Collect combined features
    all_features = []
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Extract features
        features_dict = extract_features(image)
        
        # Combine features
        combined = combine_features(features_dict)
        all_features.append(combined)
    
    if not all_features:
        print("Failed to collect features for PCA")
        return None
    
    # Stack features
    all_features = np.vstack(all_features)
    
    # Train PCA
    n_components = min(250, all_features.shape[0], all_features.shape[1])
    pca = PCA(n_components=n_components)



    pca.fit(all_features)
    
    # Save PCA model
    pca_path = os.path.join(MODELS_PATH, 'pca_model.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    
    print(f"PCA model trained and saved to {pca_path}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca

def preprocess_dataset(image_folder):
    """
    Preprocess the entire dataset by creating the visual codebook and training PCA
    """
    # Create visual codebook
    codebook = create_visual_codebook(image_folder)
    
    # Train PCA
    pca = train_pca(image_folder)
    
    return codebook, pca