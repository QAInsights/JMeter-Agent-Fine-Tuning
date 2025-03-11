#!/usr/bin/env python3
"""
Script to format the first column of the elements.csv file by adding spaces between words.
"""

import csv
import re
import os

def add_spaces_to_first_column(input_file, output_file):
    """
    Read the CSV file and add spaces between words in the first column.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the formatted CSV file
    """
    rows = []
    
    # Common word endings to look for
    word_endings = [
        'request', 'sampler', 'controller', 'config', 'assertion', 'timer', 'listener',
        'preprocessor', 'postprocessor', 'visualizer', 'extractor', 'manager', 'defaults',
        'parameters', 'variable', 'configuration', 'assert', 'group', 'report'
    ]
    
    # Read the input file
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if not row:
                rows.append(row)
                continue
                
            # Get the first column
            first_column = row[0].strip()
            
            # Skip if it already starts with "add a"
            if first_column.startswith('add a '):
                rows.append(row)
                continue
            
            # If it already has spaces, keep it as is
            if ' ' in first_column:
                rows.append(row)
                continue
            
            # Format the element name by adding spaces between words
            formatted = first_column
            
            # First try to identify common word endings and add spaces before them
            for ending in word_endings:
                if ending in formatted and not formatted == ending:
                    formatted = formatted.replace(ending, f' {ending}')
            
            # Then use regex for camelCase and other patterns
            formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', formatted)
            formatted = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', formatted)
            formatted = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', formatted)
            
            # Handle special cases
            formatted = formatted.replace('jsr 223', 'jsr223')
            formatted = formatted.replace('jsr223', 'jsr 223')
            formatted = formatted.replace('xpath 2', 'xpath2')
            formatted = formatted.replace('xpath2', 'xpath 2')
            formatted = formatted.replace('jmes path', 'jmespath')
            formatted = formatted.replace('jmes ', 'jmespath ')
            formatted = formatted.replace('ldap ext', 'ldapext')
            formatted = formatted.replace('ldapext', 'ldap ext')
            formatted = formatted.replace('user parameters', 'userparameters')
            formatted = formatted.replace('userparameters', 'user parameters')
            formatted = formatted.replace('compare assert', 'compareassert')
            formatted = formatted.replace('compareassert', 'compare assert')
            formatted = formatted.replace('random variable', 'randomvariable')
            formatted = formatted.replace('randomvariable', 'random variable')
            
            # Clean up any double spaces
            while '  ' in formatted:
                formatted = formatted.replace('  ', ' ')
            
            # Update the row with the formatted name
            row[0] = formatted.strip()
            rows.append(row)
    
    # Write to the output file
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "data", "elements.csv")
    output_file = os.path.join(script_dir, "data", "elements_formatted.csv")
    
    # Format the CSV file
    add_spaces_to_first_column(input_file, output_file)
    
    print(f"Formatted CSV file generated at: {output_file}")
