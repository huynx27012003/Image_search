import cv2
import os
import sys
import matplotlib.pyplot as plt
from preprocessing import preprocess_dataset
from database import store_image_features
from search import search_similar_images
from config import *

def show_images(query_image, results):
    """
    Display query image and results using matplotlib
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    for i, result in enumerate(results[:3]):
        img_path = result['image_path']
        similarity = result['score']
        species = result['species']

        img = cv2.imread(img_path)
        if img is not None:
            axes[i+1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"Result {i+1}\nSpecies: {species}\nScore: {similarity:.4f}")
            axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def initialize_system(rebuild=False):
    """
    Initialize the system by preprocessing dataset and storing features
    """
    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) < 10:
        print(f"[ERROR] Dataset not found or too small in {DATASET_PATH}")
        print("Please add at least 10 bird images to the dataset directory")
        return False

    print("[INFO] Creating visual codebook and training PCA...")
    codebook, pca = preprocess_dataset(DATASET_PATH)
    if codebook is None or pca is None:
        print("[ERROR] Failed to preprocess dataset")
        return False

    print(f"[INFO] Storing features in MongoDB (rebuild={rebuild})...")
    success = store_image_features(DATASET_PATH, rebuild=rebuild)
    return success

def search_with_image(query_image_path):
    """
    Search for similar images to the provided query image
    """
    if not os.path.exists(query_image_path):
        print(f"[ERROR] Query image not found: {query_image_path}")
        return None

    query_image = cv2.imread(query_image_path)
    if query_image is None:
        print(f"[ERROR] Could not read query image: {query_image_path}")
        return None

    results = search_similar_images(query_image, limit=3)
    return query_image, results

def main():
    """
    Main function
    """
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "init":
            rebuild = False
            if len(sys.argv) > 2 and sys.argv[2] == "rebuild":
                rebuild = True

            print("[INFO] Initializing the system...")
            if initialize_system(rebuild=rebuild):
                print("[SUCCESS] System initialized successfully")
            else:
                print("[FAILED] Failed to initialize the system")
            return

        elif os.path.isfile(command):
            query_image_path = command
            result = search_with_image(query_image_path)
            if result:
                query_image, results = result
                show_images(query_image, results)
            return

    print("Usage:")
    print("  python main.py init                # Train model only, keep old database")
    print("  python main.py init rebuild        # Rebuild database and retrain everything")
    print("  python main.py <image_path>        # Search similar images")

if __name__ == "__main__":
    main()
