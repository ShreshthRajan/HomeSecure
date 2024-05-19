from langchain import LangChain, PromptTemplate, OpenAI
import numpy as np
import os

# Load combined features
def load_combined_features(combined_feature_dir):
    features = []
    for file_name in os.listdir(combined_feature_dir):
        if file_name.endswith('_combined_features.npz'):
            file_path = os.path.join(combined_feature_dir, file_name)
            features.append(np.load(file_path)['features'])
    return np.array(features)

# Define the question-answering function
def answer_questions_with_gpt4(features, questions):
    # Initialize GPT-4 model using LangChain
    llm = OpenAI(model="gpt-4", api_key="openai_api_key")
    chain = LangChain(llm)
    
    # Prepare the prompt template
    prompt_template = PromptTemplate("Given the features {features}, answer the following questions: {questions}")
    
    for feature in features:
        # Convert features to string to input into GPT-4
        feature_str = ', '.join(map(str, feature.tolist()))
        for question in questions:
            prompt = prompt_template.format(features=feature_str, questions=question)
            response = chain(prompt)
            print(f"Question: {question}\nAnswer: {response}\n")

if __name__ == "__main__":
    combined_feature_dir = 'data/combined_features/cam1'
    features = load_combined_features(combined_feature_dir)
    questions = ["What objects are present?", "Describe the scene.", "Identify any unusual activities."]
    
    answer_questions_with_gpt4(features, questions)
