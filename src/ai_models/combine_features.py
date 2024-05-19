import numpy as np
import os
from tqdm import tqdm

def combine_features(cnn_feature_dir, sift_feature_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(cnn_feature_dir)):
        if file_name.endswith('_cnn_features.npz'):
            base_name = file_name.replace('_cnn_features.npz', '')
            sift_file = os.path.join(sift_feature_dir, f"{base_name}_sift.npz")

            if os.path.exists(sift_file):
                cnn_features = np.load(os.path.join(cnn_feature_dir, file_name))['features']
                sift_features = np.load(sift_file)['descriptors']
                
                combined_features = np.concatenate((cnn_features.flatten(), sift_features.flatten()), axis=0)
                combined_file = os.path.join(output_dir, f"{base_name}_combined_features.npz")
                np.savez_compressed(combined_file, features=combined_features)

if __name__ == "__main__":
    cnn_feature_dir = 'data/features/cam1'
    sift_feature_dir = 'data/features/cam1'
    output_dir = 'data/combined_features/cam1'
    combine_features(cnn_feature_dir, sift_feature_dir, output_dir)
