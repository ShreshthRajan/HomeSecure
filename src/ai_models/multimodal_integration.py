import openai
import json
import numpy as np
import os

openai.api_key = 'YOUR_OPENAI_API_KEY'

def generate_prompt(image_features, question):
    prompt = f"Analyze the following image features and answer the question:\n\nFeatures: {image_features}\n\nQuestion: {question}\n\nAnswer:"
    return prompt

def ask_gpt4(question, image_features):
    prompt = generate_prompt(image_features, question)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def load_features(feature_dir):
    features = []
    for root, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith('.npz'):
                data = np.load(os.path.join(root, file), allow_pickle=True)
                features.append({
                    "file": file,
                    "keypoints": data['keypoints'],
                    "descriptors": data['descriptors']
                })
    return features

def main():
    feature_dir = 'data/sift_features'
    features = load_features(feature_dir)
    
    question = "How many people are wearing red shirts in the video?"
    
    image_features = json.dumps(features)
    
    answer = ask_gpt4(question, image_features)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()
