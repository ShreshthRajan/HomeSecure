# Multi Modal Deep CNN w/ Feature Extraction for Gesture Recognition 

Data: https://drive.google.com/drive/folders/1Xs9Re6tVkx6sLczYPWzHU355RjT9M7zm

## Overview

Designed to enhance the analysis of security footage. Leveraging deep CNNs for object detection, feature extraction, and multi-modal integration, I aim to find improvement in comprehension and interaction with video content. This project integrates YOLO for object detection, SIFT for detailed feature extraction, and a custom deep CNN for object recognition. Employs GPT-4’s multi-modal capabilities for comprehensive question answering about the video content. The goal is to assess the viability of layering large language models with high dimensional extracted features against regular GPT4 multi modal capability. 

## Key Components

### 1. YOLO (You Only Look Once) for Object Detection

YOLO for real-time object compression. processes each frame of the input video to identify and label objects. This detection pipeline involves:

- **Frame Decomposition:** Sequentially extracting frames from the video stream.
- **Detection Algorithm:** Applying the YOLO model to each frame to detect objects.
- **Annotation Storage:** Recording the detected objects' labels and bounding boxes for subsequent stages.

### 2. Deep Convolutional Neural Network (CNN) for Object Recognition

deep CNN is trained using the labeled frames from the YOLO output. The CNN is architected to optimize object recognition accuracy. 

- **Data Curation:** Organized the detected objects and corresponding frames into training and validation datasets.
- **Model Architecture:** Defined a deep CNN structure optimized for feature extraction and classification.
- **Feature Extraction:** Producing high-dimensional feature maps that encapsulate object characteristics.

### 3. SIFT for lower-level Feature Extraction

SIFT extracts invariant features from the frames. This method captures descriptors about textures and patterns, regardless of scale and rotation. The goal of this layer is capture data not found in the CNN. 

- **Key Point Detection:** Identifying distinctive locations within each frame.
- **Descriptor Calculation:** Computing the local image gradients around each key point to generate a descriptor vector.
- **Feature Compilation:** Aggregating the key points and descriptors for use in multi-modal integration.

### 4. Multi-Modal Integration with GPT-4

Multi-modal integration uses high dimensional output from the prevoius layers to provide a comprehensive representation of each frame. Additional modality includes a layer of raw video data in future iterations. GPT-4 processes these features for advanced question answering. 

- **Feature Fusion:** Combining high-level CNN features with SIFT descriptors into a unified representation.
- **GPT-4 Processing:** Utilizing GPT-4’s multi-modal capabilities to interpret the fused features. 
- **Question Answering:** Enabling sophisticated querying of the video content, including object identification, interaction understanding, and event recognition.

## Implementation

### Backend (`app.py`)

The backend is implemented using Flask to handle video uploads and processing. 

- **Video Upload Endpoint:** Accepts video files from users.
- **Frame Extraction:** Decomposes videos into individual frames for processing.
- **Object Detection:** Executes YOLO on frames to generate object annotations.
- **Feature Extraction:** Applies deep CNN and SIFT algorithms to extract features.
- **Multi-Modal Integration:** Combines features and processes them with GPT-4 for question answering.

### Frontend (`index.html`, `app.js`, `style.css`)

The frontend is designed to provide an interactive interface for video upload and querying. 

- **Upload Interface:** Facilitates user video uploads.
- **Chatbot Interface:** Allows users to interact with the system via a conversational agent.
- **Results Display:** Visualizes detected objects, extracted features, and GPT-4's responses.

## Usage

1. **Upload a Video:** Users upload a video through the web interface.
2. **Processing:** The backend processes the video, extracting frames, detecting objects, and performing feature extraction.
3. **Interaction:** Users query the system using the chatbot interface.
4. **Results:** The system responds with detailed answers based on the combined multi-modal features processed by GPT-4.

