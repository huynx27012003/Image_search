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
        return np.zeros(len(visual_codebook.cluster_centers_))
    
    cluster_ids = visual_codebook.predict(descriptors)
    
    histogram = np.zeros(len(visual_codebook.cluster_centers_))
    for cluster_id in cluster_ids:
        histogram[cluster_id] += 1
    
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    return histogram

def extract_features(image):
    """
    Extract all features from an image.
    Đối với HSV, chỉ lấy màu từ vùng con chim (dựa trên contour lớn nhất).
    Các đặc trưng khác (HOG, LBP, SIFT/BoW) được tính trên toàn bộ ảnh.
    """
    image = cv2.resize(image, IMAGE_SIZE)
    
    # HSV features - dùng histogram 2D Hue + Saturation (16x8 = 128 chiều)
    gray_for_edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_for_edge, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = None
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # lọc contour bé
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray_for_edge)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is not None:
        hsv_hist = cv2.calcHist([hsv_image], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    else:
        hsv_hist = cv2.calcHist([hsv_image], [0, 1], None, [16, 8], [0, 180, 0, 256])
    
    hsv_features = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # HOG features
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, 
                       orientations=HOG_ORIENTATIONS, 
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK, 
                       block_norm='L2-Hys',
                       visualize=False)
    
    # LBP features
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_features = lbp_hist.astype('float32') / np.sum(lbp_hist)
    
    # SIFT features with BoW
    try:
        visual_codebook_path = os.path.join(MODELS_PATH, 'visual_codebook.pkl')
        with open(visual_codebook_path, 'rb') as f:
            codebook = pickle.load(f)
        
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        bow_features = compute_bow_features(descriptors, codebook)
    except Exception as e:
        print(f"Error in SIFT/BoW extraction: {e}")
        bow_features = np.array([])

    return {
        'hsv': hsv_features,
        'hog': hog_features,
        'lbp': lbp_features,
        'sift_bow': bow_features
    }

def combine_features(features_dict, hog_pca=None, selected_features=None):
    vects = []

    # Nếu không truyền gì thì mặc định lấy tất cả feature
    if not selected_features:
        selected_features = ['hsv', 'hog', 'lbp', 'sift']

    if 'hsv' in selected_features:
        vects.append(features_dict['hsv'])
    if 'hog' in selected_features:
        hog_vec = features_dict['hog']
        if hog_pca is not None:
            try:
                hog_vec = hog_pca.transform(hog_vec.reshape(1, -1))[0]
            except Exception as e:
                print(f"[WARN] HOG PCA transform failed in combine_features: {e}")
        vects.append(hog_vec)

    if 'lbp' in selected_features:
        vects.append(features_dict['lbp'])
    if 'sift' in selected_features:
        vects.append(features_dict['sift_bow'])

    return np.concatenate(vects)
