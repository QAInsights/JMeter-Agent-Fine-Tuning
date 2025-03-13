#!/usr/bin/env python
# coding: utf-8

"""
Create a label encoder for the JMeter Model API
This script creates a label encoder from the training data and saves it to disk.
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_label_encoder(csv_path='data/elements_v1.csv', output_path='results/best_model/label_encoder.pkl'):
    """
    Create a label encoder from the training data and save it to disk
    
    Args:
        csv_path: Path to the CSV file with training data
        output_path: Path to save the label encoder
    """
    print(f"Reading data from {csv_path}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['query', 'class_name', 'gui_class', 'version', 'is_deprecated']
    
    # Create element_info column
    df['element_info'] = df.apply(lambda row: f"{row['class_name']},{row['gui_class']},{row['version']},{row['is_deprecated']}", axis=1)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['element_info'])
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the label encoder
    with open(output_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Label encoder saved to {output_path}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Print a few examples
    print("\nExample classes:")
    for i, class_name in enumerate(label_encoder.classes_[:5]):
        print(f"  {i}: {class_name}")

if __name__ == "__main__":
    create_label_encoder()
