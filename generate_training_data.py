#!/usr/bin/env python3
"""
Script to convert JMeter elements CSV to JSONL training data format.
"""

import csv
import json
import os

def generate_jsonl_from_csv(csv_path, output_path):
    """
    Reads a CSV file with JMeter element data and generates a JSONL file for training.
    
    Args:
        csv_path: Path to the CSV file containing JMeter elements
        output_path: Path to save the generated JSONL file
    """
    system_prompt = "You are an helpful JMeter assistant and agent. You are given a JMeter test plan and you need to generate the correct component's class name,it's GUI class name, JMeter supported version, and is it deprecated or not."
    
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Create or overwrite the output file
        with open(output_path, 'w') as jsonl_file:
            for row in csv_reader:
                # Skip empty rows
                if not row or len(row) < 2:
                    continue
                
                # Parse the row data
                element_name = row[0].strip()
                
                # Check if we have enough columns for class name and GUI class name
                if len(row) < 3:
                    continue
                
                # Extract the class information
                class_name = row[1].strip()
                gui_class_name = row[2].strip()
                
                # Format the user query
                if element_name.startswith("add a "):
                    user_query = element_name
                else:
                    user_query = f"add a {element_name}"
                
                # Check if version and deprecated status are available
                version = "5.4.3"  # Default version
                deprecated = "false"  # Default deprecated status
                
                if len(row) >= 4 and row[3].strip():
                    version = row[3].strip()
                
                if len(row) >= 5 and row[4].strip():
                    deprecated = row[4].strip()
                
                # Create the JSON object
                json_obj = {
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": f"{class_name},{gui_class_name}, {version}, {deprecated}"}
                    ]
                }
                
                # Write to the JSONL file
                jsonl_file.write(json.dumps(json_obj) + "\n")

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "elements.csv")
    output_path = os.path.join(script_dir, "data", "generated_training.jsonl")
    
    # Generate the JSONL file
    generate_jsonl_from_csv(csv_path, output_path)
    
    print(f"JSONL file generated at: {output_path}")
