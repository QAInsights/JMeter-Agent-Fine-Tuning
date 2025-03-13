#!/usr/bin/env python
# coding: utf-8

"""
Test script for the JMeter Model Flask API

This script sends test queries to the Flask API and displays the results.
It helps verify that the API is working correctly and the model is making
accurate predictions.
"""

import requests
import json
import time

def check_health():
    """
    Check if the API server is healthy
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        response = requests.get("http://localhost:8080/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def test_api(query):
    """
    Test the JMeter Model API with a query
    
    Args:
        query: The query to test
        
    Returns:
        The API response
    """
    url = "http://localhost:8080/predict"
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the API server. Make sure it's running."

def print_prediction(result):
    """
    Print a formatted prediction result
    
    Args:
        result: The prediction result from the API
    """
    if isinstance(result, str) and result.startswith("Error"):
        print(result)
        return
    
    print(f"Predicted JMeter Element: {result['prediction']['class_name']}")
    print(f"GUI Class: {result['prediction']['gui_class']}")
    print(f"Version: {result['prediction']['version']}")
    print(f"Is Deprecated: {result['prediction']['is_deprecated']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    if result['alternatives']:
        print("\nAlternative predictions:")
        for i, alt in enumerate(result['alternatives'][:3], 1):
            print(f"  {i}. {alt['class_name']} (Confidence: {alt['confidence']:.4f})")

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "add a http request",
        "add a http sampler",  # Testing sampler/request interchangeability
        "add a ftp request",
        "add a ftp sampler",   # Testing sampler/request interchangeability
        "add a jdbc request",
        "add a thread group"
    ]
    
    # Check if the API is running
    if not check_health():
        print("Error: API server is not running. Start it with './deploy_api.sh start'")
        exit(1)
    
    print("Testing JMeter Model Flask API...\n")
    
    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: '{query}'")
        print(f"{'-' * 50}")
        
        # Test the API
        start_time = time.time()
        result = test_api(query)
        elapsed_time = time.time() - start_time
        
        # Print the result
        print_prediction(result)
        print(f"Response time: {elapsed_time:.3f} seconds")
    
    print(f"\n{'=' * 50}")
    print("All tests completed!")
