# cnn_feature_extraction.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing import image

def extract_cnn_features(image_dir, model_path, output_dir):
    model = load_model(model_path)
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)  # Adjust to the second last layer

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        features = feature_model.predict(img_array)

        feature_file = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_cnn_features.npz")
        np.savez_compressed(feature_file, features=features)

if __name__ == "__main__":
    image_dir = 'data/processed/cam1'
    model_path = 'models/deep_cnn_model.h5'
    output_dir = 'data/features/cam1'
    extract_cnn_features(image_dir, model_path, output_dir)
