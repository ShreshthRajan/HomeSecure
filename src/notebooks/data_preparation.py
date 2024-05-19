# src/notebooks/data_preparation.py
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import cv2

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to homesecure
VIDEO_DIR = os.path.join(BASE_DIR, 'data/raw')
FRAME_DIR = os.path.join(BASE_DIR, 'data/processed')
ANNOTATION_DIR = os.path.join(BASE_DIR, 'data/annotations')
CALIBRATION_DIR = os.path.join(BASE_DIR, 'calibrations')

# Create directories if they don't exist
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)

def extract_frames(video_path, output_dir, interval=5):
    print(f"Processing video: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't read video stream from file {video_path}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        print(f"Error: Couldn't get frame rate for video {video_path}")
        return

    count = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}, frame rate: {frame_rate}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_id % int(frame_rate * interval) == 0:
            frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1

        # Print progress every 100 frames
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{frame_count} frames...")

    cap.release()
    print(f"Extracted {count} frames from {video_path}")

def read_calibration_file(calibration_file):
    tree = ET.parse(calibration_file)
    root = tree.getroot()
    
    camera_matrix = None
    distortion_coefficients = None
    rvec = None
    tvec = None

    for elem in root:
        if elem.tag == 'camera_matrix':
            camera_matrix = np.fromstring(elem[3].text, sep=' ').reshape((3, 3))
        elif elem.tag == 'distortion_coefficients':
            distortion_coefficients = np.fromstring(elem[3].text, sep=' ').reshape((5, 1))
        elif elem.tag == 'rvec':
            rvec = np.fromstring(elem.text, sep=' ').reshape((3, 1))
        elif elem.tag == 'tvec':
            tvec = np.fromstring(elem.text, sep=' ').reshape((3, 1))

    return camera_matrix, distortion_coefficients, rvec, tvec

def undistort_image(image, camera_matrix, distortion_coefficients):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

def process_annotations(annotation_file, grid_size=(480, 1440), origin=(-3.0, -9.0), spacing=0.025):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    processed_annotations = []
    for item in annotations:
        position_id = item['positionID']
        x = origin[0] + spacing * (position_id % grid_size[0])
        y = origin[1] + spacing * (position_id // grid_size[0])
        processed_annotations.append({
            'id': item['id'],
            'x': x,
            'y': y,
            'bbox': item['bbox']
        })

    return processed_annotations

def main():
    # Extract frames from cam1 and cam2
    videos = ['cam1.mp4', 'cam2.mp4']
    for video in videos:
        video_path = os.path.join(VIDEO_DIR, video)
        output_dir = os.path.join(FRAME_DIR, os.path.splitext(video)[0])
        os.makedirs(output_dir, exist_ok=True)
        extract_frames(video_path, output_dir, interval=5)

    # Apply calibration to each frame for cam1 and cam2
    extrinsic_files = ['extr_CVLab1.xml', 'extr_CVLab2.xml']
    intrinsic_files = {
        'cam1': ['intr_CVLab1.xml', 'intr_CVLab1.xml'],
        'cam2': ['intr_CVLab2.xml', 'intr_CVLab2.xml']
    }

    for idx, (extrinsic_file, cam_name) in enumerate(zip(extrinsic_files, ['cam1', 'cam2'])):
        extrinsic_path = os.path.join(CALIBRATION_DIR, extrinsic_file)
        intrinsic_original_path = os.path.join(CALIBRATION_DIR, 'intrinsic_original', intrinsic_files[cam_name][0])
        intrinsic_zero_path = os.path.join(CALIBRATION_DIR, 'intrinsic_zero', intrinsic_files[cam_name][1])

        camera_matrix, distortion_coefficients, _, _ = read_calibration_file(intrinsic_original_path)
        frame_dir = os.path.join(FRAME_DIR, cam_name)
        for root, _, files in os.walk(frame_dir):
            for file in files:
                frame_path = os.path.join(root, file)
                frame = cv2.imread(frame_path)
                undistorted_frame = undistort_image(frame, camera_matrix, distortion_coefficients)
                cv2.imwrite(frame_path, undistorted_frame)

    # Process annotations
    annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.json')]
    for annotation_file in annotation_files:
        annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)
        processed_annotations = process_annotations(annotation_path)
        processed_annotation_path = os.path.join(ANNOTATION_DIR, f"processed_{annotation_file}")
        with open(processed_annotation_path, 'w') as f:
            json.dump(processed_annotations, f)

if __name__ == '__main__':
    main()
