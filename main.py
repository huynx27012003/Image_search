# main.py
import os
import sys
import cv2
from database import store_image_features, insert_image_features

def initialize_system(rebuild=False):
    # return store_image_features("dataset/birds", rebuild)
    return insert_image_features("dataset/chim")

def show_images(query_image, results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 5))
    axes[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Query')
    axes[0].axis('off')
    for i, res in enumerate(results):
        img = cv2.imread(res['image_path'])
        axes[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"{res['species']}\n{res['score']:.3f}")
        axes[i + 1].axis('off')
    plt.show()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'init':
        rebuild = (len(sys.argv) > 2 and sys.argv[2] == 'rebuild')
        print("Starting database initialization...")
        success = initialize_system(rebuild)
        print('Init success' if success else 'Init failed')
        return

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        from search import search_similar_images
        path = sys.argv[1]
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read image: {path}")
            return
        results = search_similar_images(img, limit=5)
        show_images(img, results)
        return

    print("Usage: python main.py init [rebuild] OR python main.py <image_path>")

if __name__ == '__main__':
    main()