import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_sift_features(image_dir, output_dir):
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the directory
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # Save the keypoints and descriptors
        if descriptors is not None:
            feature_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_sift.npz")
            np.savez_compressed(feature_file, keypoints=keypoints, descriptors=descriptors)
        else:
            print(f"No descriptors found for image: {image_path}")

if __name__ == "__main__":
    image_dir = 'data/processed/cam1'  # Path to the directory containing images
    output_dir = 'data/features/cam1'  # Path to the directory where features will be saved
    extract_sift_features(image_dir, output_dir)
