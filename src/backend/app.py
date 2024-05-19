from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import openai
import subprocess
import json
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

openai.api_key = 'MY KEY'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video file 
        subprocess.call(['python', 'src/ai_models/data_preparation.py', filepath, 'data/annotations/cam1'])
        subprocess.call(['python', 'src/ai_models/yolo_detection.py', '--image-dir', 'data/annotations/cam1', '--output-dir', 'data/processed/cam1'])
        subprocess.call(['python', 'src/ai_models/deep_cnn.py'])
        subprocess.call(['python', 'src/ai_models/feature_extraction.py'])
        subprocess.call(['python', 'src/ai_models/combine_features.py'])

        return jsonify({'success': 'File uploaded and processed'}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    # Call GPT4
    response = openai.Completion.create(
        model= "  ",
        prompt=user_input,
        max_tokens=150
    )
    
    answer = response.choices[0].text.strip()
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)
