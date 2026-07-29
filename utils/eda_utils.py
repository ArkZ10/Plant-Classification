"""Dataset exploration: resolution check, per-class counts, image sizes."""

import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image


def check_uniform_resolution(root_path):
    """Return the set of unique (width, height) resolutions under `root_path`."""
    resolution_set = set()
    for root, _, files in os.walk(root_path):
        for image_file in files:
            image_path = os.path.join(root, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                resolution_set.add((image.shape[1], image.shape[0]))
            else:
                print(f"Failed to read image: {image_path}")
    return resolution_set


def count_images_per_class(root_path, classes=None):
    """Count images per class subfolder under `root_path`."""
    class_names = classes or os.listdir(root_path)
    return {
        class_name: len(os.listdir(os.path.join(root_path, class_name)))
        for class_name in class_names
    }


def plot_class_distribution(class_counts, save_path=None):
    """Bar chart of image counts per class."""
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Number of Images in Each Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.show()


def get_image_sizes(root_dir):
    """Return a list of (file_path, (width, height)) for every image under `root_dir`."""
    image_sizes = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        image_sizes.append((file_path, img.size))
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
    return image_sizes
