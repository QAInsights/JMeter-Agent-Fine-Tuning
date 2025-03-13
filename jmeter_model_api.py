#!/usr/bin/env python
# coding: utf-8

"""
JMeter Model API Server

This script creates a Flask API server that serves the trained JMeter RoBERTa model.
It provides endpoints to predict JMeter elements based on user queries.
"""

import os
import torch
import json
import shutil
import numpy as np
from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Global variables to store model, tokenizer, and label encoder
model = None
tokenizer = None
label_encoder = None

def load_model(model_path='./results/best_model'):
    """
    Load the trained RoBERTa model, tokenizer, and label encoder
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tuple of (model, tokenizer, label_encoder)
    """
    global model, tokenizer, label_encoder
    
    # Check if the custom model files exist
    custom_model_file = os.path.join(model_path, 'JMeter Model.safetensors')
    custom_label_encoder_file = os.path.join(model_path, 'Label Encoder.pkl')
    custom_config_file = os.path.join(model_path, 'JMeter config.json')
    
    # Check for standard files as fallback
    standard_label_encoder_file = os.path.join(model_path, 'label_encoder.pkl')
    
    # Load the tokenizer
    try:
        # Try to load custom tokenizer if it exists
        vocab_file = os.path.join(model_path, 'JMeter vocab.json')
        merges_file = os.path.join(model_path, 'JMeter Merges.txt')
        tokenizer_config = os.path.join(model_path, 'JMeter Tokenizer Config.json')
        
        if os.path.exists(vocab_file) and os.path.exists(merges_file):
            print("Loading custom tokenizer...")
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        else:
            print("Loading base tokenizer...")
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    except Exception as e:
        print(f"Error loading custom tokenizer: {str(e)}")
        print("Loading base tokenizer instead...")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Load the label encoder
    if os.path.exists(custom_label_encoder_file):
        print(f"Loading custom label encoder from {custom_label_encoder_file}...")
        with open(custom_label_encoder_file, 'rb') as f:
            label_encoder = pickle.load(f)
    elif os.path.exists(standard_label_encoder_file):
        print(f"Loading standard label encoder from {standard_label_encoder_file}...")
        with open(standard_label_encoder_file, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        raise FileNotFoundError(f"Could not find label encoder file in {model_path}")
    
    # Get the number of labels
    num_labels = len(label_encoder.classes_)
    print(f"Number of labels: {num_labels}")
    
    # Try to load the model from the saved path
    try:
        if os.path.exists(custom_model_file) and os.path.exists(custom_config_file):
            print(f"Loading custom model from {custom_model_file}...")
            # Create a temporary config file with the correct name
            config_dict = {}
            with open(custom_config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Update the config with the correct number of labels
            config_dict['num_labels'] = num_labels
            
            # Create a temporary directory to load the model from
            temp_dir = os.path.join(model_path, 'temp_model')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy the model file to the temporary directory with the standard name
            shutil.copy(custom_model_file, os.path.join(temp_dir, 'model.safetensors'))
            
            # Save the config to the temporary directory
            with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
                json.dump(config_dict, f)
            
            # Load the model from the temporary directory
            model = RobertaForSequenceClassification.from_pretrained(temp_dir, num_labels=num_labels)
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
        else:
            print(f"Loading model from {model_path}...")
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    except (OSError, IOError) as e:
        # If model not found, load the base model and warn the user
        print(f"Warning: Could not load model from {model_path}. Error: {str(e)}")
        print("Loading base RoBERTa model instead. This will not provide accurate predictions.")
        print("Please train the model using jmeter_roberta_training.py first.")
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    
    model.eval()  # Set model to evaluation mode
    
    return model, tokenizer, label_encoder

def predict_jmeter_element(query):
    """
    Predict the JMeter element for a given query
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with prediction results
    """
    global model, tokenizer, label_encoder
    
    # Ensure model is loaded
    if model is None or tokenizer is None or label_encoder is None:
        load_model()
    
    # Implement keyword matching as a fallback for when we don't have a trained model
    # This helps ensure we return meaningful results even without a fine-tuned model
    query_lower = query.lower()
    
    # Map common keywords to element types
    keyword_mapping = {
        "http": ["org.apache.jmeter.protocol.http", "HTTPSamplerProxy"],
        "web": ["org.apache.jmeter.protocol.http", "HTTPSamplerProxy"],
        "ftp": ["org.apache.jmeter.protocol.ftp", "FTPSampler"],
        "jdbc": ["org.apache.jmeter.protocol.jdbc", "JDBCSampler"],
        "sql": ["org.apache.jmeter.protocol.jdbc", "JDBCSampler"],
        "thread": ["org.apache.jmeter.threads", "ThreadGroup"],
        "timer": ["org.apache.jmeter.timers", "ConstantTimer"],
        "assertion": ["org.apache.jmeter.assertions", "ResponseAssertion"],
        "csv": ["org.apache.jmeter.config", "CSVDataSet"]
    }
    
    # Tokenize the input
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply keyword matching to adjust logits if we're using the base model
    # This is a fallback mechanism for when we don't have a fine-tuned model
    if hasattr(model, '_name_or_path') and model._name_or_path == 'roberta-base':
        print(f"Using keyword matching for query: '{query}'")
        
        # Create a dictionary to store the keyword match scores
        keyword_scores = {}
        
        # Check for keyword matches
        for keyword, element_info in keyword_mapping.items():
            if keyword in query_lower:
                # Find indices of classes that match this keyword
                for i, class_name in enumerate(label_encoder.classes_):
                    if element_info[0] in class_name and element_info[1] in class_name:
                        keyword_scores[i] = keyword_scores.get(i, 0) + 1
        
        # Adjust logits based on keyword matches
        if keyword_scores:
            # Create a tensor of zeros
            adjusted_logits = torch.zeros_like(logits)
            
            # Set values for matched keywords
            for idx, score in keyword_scores.items():
                adjusted_logits[0, idx] = score * 5.0  # Boost the score
            
            # Use the adjusted logits
            logits = adjusted_logits
    
    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    # Get the top predictions (up to 5 or the number of classes, whichever is smaller)
    top_k = min(5, len(label_encoder.classes_))
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    
    # Convert to numpy for easier handling
    top_k_probs = top_k_probs.cpu().numpy().tolist()
    top_k_indices = top_k_indices.cpu().numpy().tolist()
    
    # Get the class names for the top-k predictions
    top_k_classes = [label_encoder.inverse_transform([idx])[0] for idx in top_k_indices]
    
    # Parse the top prediction
    predicted_class = top_k_indices[0]
    element_info = label_encoder.inverse_transform([predicted_class])[0]
    parts = element_info.split(',')
    
    # Create result dictionary
    result = {
        "query": query,
        "prediction": {
            "class_name": parts[0].strip(),
            "gui_class": parts[1].strip(),
            "version": parts[2].strip(),
            "is_deprecated": parts[3].strip().lower() == "true"
        },
        "confidence": float(top_k_probs[0]),
        "alternatives": []
    }
    
    # Add alternatives
    for i in range(1, len(top_k_classes)):
        alt_parts = top_k_classes[i].split(',')
        result["alternatives"].append({
            "class_name": alt_parts[0].strip(),
            "gui_class": alt_parts[1].strip(),
            "version": alt_parts[2].strip(),
            "is_deprecated": alt_parts[3].strip().lower() == "true",
            "confidence": float(top_k_probs[i])
        })
    
    return result

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict JMeter element
    
    Expected JSON input:
    {
        "query": "add a http request"
    }
    """
    # Get the request data
    data = request.json
    
    # Check if query is provided
    if 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    # Get the query
    query = data['query']
    
    # Get the prediction
    result = predict_jmeter_element(query)
    
    # Return the result
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

def save_label_encoder(label_encoder, model_path='./results/best_model'):
    """
    Save the label encoder to disk
    
    Args:
        label_encoder: The label encoder to save
        model_path: Path to save the label encoder
    """
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

if __name__ == '__main__':
    # Ensure the model directory exists
    os.makedirs('./results/best_model', exist_ok=True)
    
    # Check if we need to save the label encoder from the training data
    if not os.path.exists('./results/best_model/label_encoder.pkl'):
        print("Label encoder not found. Creating from training data...")
        # Read the CSV file
        df = pd.read_csv('data/elements_v1.csv', header=None)
        df.columns = ['query', 'class_name', 'gui_class', 'version', 'is_deprecated']
        
        # Create element_info column
        df['element_info'] = df.apply(lambda row: f"{row['class_name']},{row['gui_class']},{row['version']},{row['is_deprecated']}", axis=1)
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(df['element_info'])
        
        # Save the label encoder
        save_label_encoder(label_encoder)
        print("Label encoder saved.")
    
    # Load the model
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    
    # Start the server
    print("Starting API server...")
    # Use port 8080 instead of 5000 to avoid conflicts with AirPlay on macOS
    app.run(host='0.0.0.0', port=8080, debug=False)
