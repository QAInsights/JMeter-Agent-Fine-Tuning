# JMeter Element Finder using RoBERTa

This project trains a RoBERTa model to match natural language queries to JMeter elements and exposes it as a Flask API. For example, when a user asks for "add a web request", the model will return the corresponding JMeter element information.

## Project Components

1. **Model Training** (`jmeter_roberta_training.py`): Fine-tunes a RoBERTa model on JMeter element data
2. **Flask API** (`jmeter_model_api.py`): Serves the trained model via a REST API
3. **Deployment Script** (`deploy_api.sh`): Manages the API server (start, stop, status)
4. **Test Script** (`test_api.py`): Tests the API with sample queries

## How to Use the Flask API

### Prerequisites

- Python 3.x
- Virtual environment (recommended)

### Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv jmeter_venv
   source jmeter_venv/bin/activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Starting the API Server

Use the deployment script to manage the API server:

```bash
# Start the server
./deploy_api.sh start

# Check server status
./deploy_api.sh status

# Stop the server
./deploy_api.sh stop

# Restart the server
./deploy_api.sh restart

# Test the API with sample queries
./deploy_api.sh test
```

### API Endpoints

1. **Predict JMeter Element** - `POST /predict`
   - Request body: `{"query": "add a http request"}`
   - Response: JSON with prediction details

2. **Health Check** - `GET /health`
   - Response: `{"status": "healthy"}`

### Example API Usage

Using curl:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "add a http request"}'
```

Using Python requests:
```python
import requests

url = "http://localhost:8080/predict"
payload = {"query": "add a http request"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result)
```

## Model Training Details

### How to Train in Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Upload the following files:
   - `elements_v1.csv` - The training data
   - `jmeter_roberta_training.py` - The training script

4. Run the following command in a Colab cell to execute the script:
   ```python
   %run jmeter_roberta_training.py
   ```

### What the Training Script Does

1. Loads and preprocesses the JMeter element data
2. Generates additional training examples with variations of the queries
3. Fine-tunes a RoBERTa model to match queries to JMeter elements
4. Evaluates the model performance
5. Saves the trained model

### Model Details

- Base model: `roberta-base`
- Training epochs: 8 (increased from 5)
- Batch size: 16
- The model is fine-tuned to classify user queries into the appropriate JMeter element information

## Example Queries

The model can handle various phrasings of requests, such as:
- "add a http request" / "add a http sampler" (interchangeable)
- "create a web request"
- "add a ftp request" / "add a ftp sampler" (interchangeable)
- "I need a jdbc request"
- "insert a timer"

## API Response Format

For each query, the API returns:

```json
{
  "query": "add a http request",
  "prediction": {
    "class_name": "org.apache.jmeter.protocol.http.sampler.HTTPSamplerProxy",
    "gui_class": "org.apache.jmeter.protocol.http.control.gui.HttpTestSampleGui",
    "version": "5.4.3",
    "is_deprecated": false
  },
  "confidence": 0.9876,
  "alternatives": [
    {
      "class_name": "org.apache.jmeter.protocol.http.sampler.HTTPSampler",
      "gui_class": "org.apache.jmeter.protocol.http.control.gui.HttpTestSampleGui",
      "version": "5.4.3",
      "is_deprecated": true,
      "confidence": 0.0123
    }
  ]
}
```
