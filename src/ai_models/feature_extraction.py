import cv2
import os
import numpy as np

def extract_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

def process_images(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            keypoints, descriptors = extract_sift_features(image_path)
            if descriptors is not None:
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.npz')
                np.savez(output_path, keypoints=keypoints, descriptors=descriptors)
                print(f"Extracted SIFT features from {image_path} and saved to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract SIFT features from images.')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save SIFT features.')

    args = parser.parse_args()
    process_images(args.image_dir, args.output_dir)
