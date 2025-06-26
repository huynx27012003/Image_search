# config.py
import os

# MongoDB configuration
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "bird_image_db"
COLLECTION_NAME = "bird_images"

# Data paths
DATASET_PATH = "dataset/birds"
MODELS_PATH = "models"


# Feature extraction parameters
IMAGE_SIZE = (256, 256)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 8 * 3
REDUCED_DIM = 100    # số chiều sau khi giảm
REDUCER_PATH = "models/pca_reducer.pkl"
SCALERS_PATH = "models/feature_scalers.pkl"

# Default feature weights (sum to 1) for updated features: area_ratio, hsv_grid, hog, lbp
DEFAULT_WEIGHTS = {
    'silhouette': 0.167,  # 1/6
    'elongation': 0.167,  # 1/6
    'hsv_grid': 0.333,    # 1/3
    'lbp': 0.333          # 1/3
}

# Create directories if they don't exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)