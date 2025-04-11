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
HSV_BINS = (8, 3, 3)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 3
LBP_POINTS = 8 * 3
SIFT_CODEBOOK_SIZE = 256

# PCA parameters
PCA_COMPONENTS = 128

# Create directories if they don't exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)