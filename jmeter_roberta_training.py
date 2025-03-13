#!/usr/bin/env python
# coding: utf-8

# # JMeter Element Finder using RoBERTa
# 
# This notebook trains a RoBERTa model to match user queries like "add a web request" to the corresponding JMeter element information.

# ## Setup and Dependencies

# In[ ]:

# Install required packages
!pip install transformers datasets pandas scikit-learn torch

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    RobertaConfig
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import os
from datasets import Dataset as HFDataset

# ## Data Preparation

# In[ ]:

# Upload the CSV file if running in Colab
from google.colab import files
uploaded = files.upload()  # Upload elements_v1.csv

# Read the CSV file
df = pd.read_csv('elements_v1.csv', header=None)
df.columns = ['query', 'class_name', 'gui_class', 'version', 'is_deprecated']

# Display the first few rows to verify data
print(f"Dataset shape: {df.shape}")
df.head()

# ## Preprocess Data

# In[ ]:

# Create a unique identifier for each JMeter element (combination of all columns except query)
df['element_info'] = df.apply(lambda row: f"{row['class_name']},{row['gui_class']},{row['version']},{row['is_deprecated']}", axis=1)

# Encode the element_info as labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['element_info'])

# Generate additional training examples with variations
def generate_variations(query):
    """Generate variations of the query to improve model robustness"""
    variations = []
    
    # Remove "add a" or "add an" prefix
    if query.startswith("add a "):
        variations.append(query[6:])
    elif query.startswith("add an "):
        variations.append(query[7:])
    
    # Add "create" variations
    if query.startswith("add a "):
        variations.append("create a " + query[6:])
        variations.append("create " + query[6:])
    elif query.startswith("add an "):
        variations.append("create an " + query[7:])
        variations.append("create " + query[7:])
    
    # Add "insert" variations
    if query.startswith("add a "):
        variations.append("insert a " + query[6:])
        variations.append("insert " + query[6:])
    elif query.startswith("add an "):
        variations.append("insert an " + query[7:])
        variations.append("insert " + query[7:])
    
    # Add "I want" variations
    if query.startswith("add a "):
        variations.append("I want a " + query[6:])
        variations.append("I need a " + query[6:])
    elif query.startswith("add an "):
        variations.append("I want an " + query[7:])
        variations.append("I need an " + query[7:])
    
    # Add specific variations for FTP requests
    if "ftp" in query.lower():
        base_term = query[6:] if query.startswith("add a ") else query[7:] if query.startswith("add an ") else query
        ftp_variations = [
            f"add ftp {base_term.replace('ftp ', '').replace('ftp', '')}",
            f"add an ftp {base_term.replace('ftp ', '').replace('ftp', '')}",
            f"create ftp {base_term.replace('ftp ', '').replace('ftp', '')}",
            f"setup ftp {base_term.replace('ftp ', '').replace('ftp', '')}",
            f"configure ftp {base_term.replace('ftp ', '').replace('ftp', '')}",
            f"add a file transfer protocol {base_term.replace('ftp ', '').replace('ftp', '')}",
            f"ftp {base_term.replace('ftp ', '').replace('ftp', '')}"
        ]
        variations.extend([v.strip() for v in ftp_variations if v.strip()])
    
    # Add variations to handle "sampler" and "request" interchangeably
    new_variations = []
    for var in [query] + variations:
        # Convert "request" to "sampler"
        if " request" in var.lower():
            new_variations.append(var.lower().replace(" request", " sampler"))
        
        # Convert "sampler" to "request"
        if " sampler" in var.lower():
            new_variations.append(var.lower().replace(" sampler", " request"))
    
    # Add the new variations
    variations.extend(new_variations)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in variations:
        if var.lower() not in seen:
            seen.add(var.lower())
            unique_variations.append(var)
    
    return unique_variations

# Create an expanded dataset with variations
expanded_data = []
for _, row in df.iterrows():
    # Add the original query
    expanded_data.append({
        'query': row['query'],
        'element_info': row['element_info'],
        'label': row['label']
    })
    
    # Add variations
    variations = generate_variations(row['query'])
    for var in variations:
        expanded_data.append({
            'query': var,
            'element_info': row['element_info'],
            'label': row['label']
        })

# Convert to DataFrame
expanded_df = pd.DataFrame(expanded_data)
print(f"Original dataset size: {len(df)}")
print(f"Expanded dataset size: {len(expanded_df)}")

# Split the data into training and validation sets
train_df, val_df = train_test_split(expanded_df, test_size=0.2, random_state=42, stratify=expanded_df['label'])

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# ## Create PyTorch Dataset

# In[ ]:

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create a custom dataset class
class JMeterDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.queries = queries
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Convert DataFrames to Hugging Face Datasets
train_dataset = HFDataset.from_pandas(train_df)
val_dataset = HFDataset.from_pandas(val_df)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['query'], padding='max_length', truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# ## Train the Model

# In[ ]:

# Get the number of labels
num_labels = len(label_encoder.classes_)
print(f"Number of unique JMeter elements: {num_labels}")

# Initialize the RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=num_labels
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Define a compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# ## Evaluate the Model

# In[ ]:

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# ## Save the Model and Metadata

# In[ ]:

# Save the model
model_save_path = "./jmeter_roberta_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save the label encoder classes for later use
import pickle
with open(f"{model_save_path}/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(f"Model saved to {model_save_path}")

# ## Test the Model with Sample Queries

# In[ ]:

# Function to predict JMeter element for a query
def predict_jmeter_element(query, model, tokenizer, label_encoder):
    # Tokenize the input
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Decode the prediction
    element_info = label_encoder.inverse_transform([predicted_class])[0]
    return element_info

# Function to analyze prediction for FTP requests
def analyze_prediction(query, model, tokenizer, label_encoder):
    # Tokenize the input
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Decode the prediction
    element_info = label_encoder.inverse_transform([predicted_class])[0]
    
    # Detailed analysis for FTP requests
    print("\nDetailed Analysis for FTP Request:")
    print(f"Query: '{query}'")
    print(f"Predicted JMeter Element: {element_info}")
    parts = element_info.split(',')
    print(f"Class Name: {parts[0]}")
    print(f"GUI Class: {parts[1]}")
    print(f"Version: {parts[2]}")
    print(f"Is Deprecated: {parts[3]}")

# Test with some sample queries
test_queries = [
    "add a web request",
    "add a http sampler",  # Testing sampler/request interchangeability
    "add a ftp request",
    "add a ftp sampler",   # Testing sampler/request interchangeability
    "create a ftp request",
    "I need a ftp request",
    "ftp request",
    "ftp sampler",         # Testing sampler/request interchangeability
    "add a jdbc request",
    "add a jdbc sampler",  # Testing sampler/request interchangeability
    "add a timer",
    "add a thread group",
    "add a listener",
    "add a controller",
    "add a config element",
    "add a post processor",
    "add a pre processor",
    "add a assertion",
    "add a sampler",
    "add a request"        # Testing generic sampler/request interchangeability
]

# Test the model on sample queries
print("\nTesting model on sample queries:")
for query in test_queries:
    element_info = predict_jmeter_element(query, model, tokenizer, label_encoder)
    print(f"Query: '{query}' → {element_info.split(',')[0]}")
    
    # For FTP requests, run the detailed analysis to debug
    if "ftp" in query.lower():
        analyze_prediction(query, model, tokenizer, label_encoder)

# ## Create a Simple Interface for Testing

# In[ ]:

# Create a simple interface for testing
def interactive_prediction():
    print("JMeter Element Finder - Interactive Mode")
    print("Enter 'quit' to exit")
    
    while True:
        query = input("\nEnter your query (e.g., 'add a web request'): ")
        if query.lower() == 'quit':
            break
            
        element_info = predict_jmeter_element(query, model, tokenizer, label_encoder)
        parts = element_info.split(',')
        
        print("\nPredicted JMeter Element:")
        print(f"Class Name: {parts[0]}")
        print(f"GUI Class: {parts[1]}")
        print(f"Version: {parts[2]}")
        print(f"Is Deprecated: {parts[3]}")

# Run the interactive prediction
interactive_prediction()
