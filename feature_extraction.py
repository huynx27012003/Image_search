import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import os
from config import *

def compute_bow_features(descriptors, visual_codebook):
    """
    Compute Bag of Visual Words features from SIFT descriptors
    """
    if descriptors is None:
        # Return empty histogram if no descriptors found
        return np.zeros(len(visual_codebook.cluster_centers_))
    
    # Predict cluster for each descriptor
    cluster_ids = visual_codebook.predict(descriptors)
    
    # Create histogram of cluster IDs
    histogram = np.zeros(len(visual_codebook.cluster_centers_))
    for cluster_id in cluster_ids:
        histogram[cluster_id] += 1
    
    # Normalize histogram
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    return histogram

def extract_features(image):
    """
    Extract all features from an image.
    Đối với đặc trưng màu (HSV) sẽ chỉ lấy màu từ vùng con chim, 
    được xác định dựa trên contour lớn nhất sau khi phát hiện biên bằng Canny.
    Các đặc trưng khác (HOG, LBP, SIFT/BoW) được tính trên toàn bộ ảnh.
    """
    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)
    
    # ---------------------------
    # 1. HSV Color Features lấy từ vùng con chim
    # ---------------------------
    # Chuyển ảnh sang grayscale và làm mịn để phát hiện biên
    gray_for_edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_for_edge, (5, 5), 0)
    
    # Phát hiện biên bằng Canny
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    
    # Tìm contour từ ảnh biên
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu tìm được contour, chọn contour lớn nhất giả sử đó là vùng con chim
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray_for_edge)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        # Nếu không tìm được, sử dụng toàn bộ ảnh
        mask = None

    # Chuyển ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính histogram HSV dựa trên mask (nếu có)
    if mask is not None:
        hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, 
                                HSV_BINS, [0, 180, 0, 256, 0, 256])
    else:
        hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, 
                                HSV_BINS, [0, 180, 0, 256, 0, 256])
    hsv_features = cv2.normalize(hsv_hist, hsv_hist).flatten()
    
    # ---------------------------
    # 2. HOG Features
    # ---------------------------
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, 
                       orientations=HOG_ORIENTATIONS, 
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK, 
                       block_norm='L2-Hys',
                       visualize=False)
    
    # ---------------------------
    # 3. LBP Features
    # ---------------------------
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_features = lbp_hist.astype('float32') / np.sum(lbp_hist)
    
    # ---------------------------
    # 4. SIFT Features with BoW
    # ---------------------------
    try:
        # Load the visual codebook
        visual_codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(visual_codebook_path, 'rb') as f:
            codebook = pickle.load(f)
        
        # Extract SIFT descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        
        # Compute BoW features
        bow_features = compute_bow_features(descriptors, codebook)
    except Exception as e:
        print(f"Error in SIFT/BoW extraction: {e}")
        # Return empty array if codebook not available yet
        bow_features = np.array([])
    
    # Return individual feature sets
    return {
        'hsv': hsv_features,
        'hog': hog_features,
        'lbp': lbp_features,
        'sift_bow': bow_features
    }

def combine_features(features_dict, pca=None, hsv_weight=2.0, selected_features=None):
    if selected_features is None:
        selected_features = ['hsv', 'hog', 'lbp', 'sift']

    feature_list = []

    if 'hsv' in selected_features and len(features_dict.get('hsv', [])) > 0:
        feature_list.append(hsv_weight * features_dict['hsv'])
    if 'hog' in selected_features and len(features_dict.get('hog', [])) > 0:
        feature_list.append(features_dict['hog'])
    if 'lbp' in selected_features and len(features_dict.get('lbp', [])) > 0:
        feature_list.append(features_dict['lbp'])
    if 'sift' in selected_features and len(features_dict.get('sift_bow', [])) > 0:
        feature_list.append(features_dict['sift_bow'])

    if feature_list:
        combined = np.concatenate(feature_list)
    else:
        combined = np.array([])

    # Chỉ áp dụng PCA nếu vector có đúng số chiều mà PCA kỳ vọng
    if pca is not None:
        try:
            combined = pca.transform(combined.reshape(1, -1))[0]
        except ValueError as e:
            print(f"[WARN] PCA skipped: {e}")
            # Bỏ qua PCA nếu số chiều không phù hợp
            pass

    return combined
