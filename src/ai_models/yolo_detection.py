import torch
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import argparse

def detect_objects(model, img_path):
    img = Image.open(img_path)
    results = model(img)
    return results

def process_images(image_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(image_dir):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                detections = detect_objects(model, img_path)
                # Convert detections to a format suitable for saving
                detection_data = {
                    "boxes": detections.xyxy[0].cpu().numpy(),
                    "labels": detections.names,
                    "scores": detections.xyxyn[0][:, 4].cpu().numpy()
                }
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.npz')
                np.savez(output_path, **detection_data)
                print(f"Processed {img_path}, saved to {output_path}")
            except UnidentifiedImageError:
                print(f"Error: Could not identify image {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect objects in images using YOLOv5.')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save detection results.')

    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    process_images(args.image_dir, args.output_dir, model)
