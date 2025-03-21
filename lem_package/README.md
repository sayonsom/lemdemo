# LEM Model

A Python package for the Large Event Model (LEM), a system designed to process, analyze, and predict timeseries events from home appliances.

## Installation

```bash
pip install lem-model
```

Or install from source:

```bash
git clone https://github.com/yourusername/lem-model.git
cd lem-model
pip install -e .
```

## Usage

### Python API

```python
from lem_package.inference import load_resources, preprocess_event, get_event_embedding, suggest_device_action

# Load resources
model, device_encoder, capability_encoder, state_encoder, historical_embeddings, historical_actions = load_resources()

# Example event
event = {
    "device": "ac_unit",
    "capability": "power",
    "attributes": {
        "state": "ON"
    },
    "timestamp": "2023-01-01T08:00:00"
}

# Preprocess event
features = preprocess_event(event, device_encoder, capability_encoder, state_encoder)
print(f"Preprocessed features: {features}")

# Example event sequence
events = [
    {
        "device": "light",
        "capability": "power",
        "attributes": {
            "state": "ON"
        },
        "timestamp": "2023-01-01T08:00:00"
    },
    {
        "device": "door",
        "capability": "lock",
        "attributes": {
            "state": "UNLOCKED"
        },
        "timestamp": "2023-01-01T08:01:00"
    }
]

# Generate embedding
embedding = get_event_embedding(events, model)
print(f"Embedding: {embedding}")

# Suggest action
action = suggest_device_action(events, historical_embeddings, historical_actions, model)
print(f"Suggested action: {action}")
```

### Command-Line Interface

The package provides a command-line interface for common operations:

```bash
# Preprocess an event
lem-cli preprocess '{"device": "ac_unit", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}'

# Generate an embedding for a sequence of events
lem-cli embed '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]'

# Save the embedding to a file
lem-cli embed '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]' -o embedding.pt

# Suggest a device action based on recent events
lem-cli suggest '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]'
```

### FastAPI REST API

The package provides a FastAPI-based REST API for serving the model:

```bash
# Start the API server
lem-api
```

The API server provides the following endpoints:

- `GET /health`: Health check endpoint
- `POST /preprocess`: Preprocess an event
- `POST /embed`: Generate an embedding for a sequence of events
- `POST /suggest`: Suggest a device action based on recent events
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation (ReDoc)

Example usage with curl:

```bash
# Health check
curl -X GET http://localhost:8000/health

# Preprocess an event
curl -X POST http://localhost:8000/preprocess \
  -H "Content-Type: application/json" \
  -d '{"device": "ac_unit", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}'

# Generate an embedding for a sequence of events
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]'

# Suggest a device action based on recent events
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '[{"device": "light", "capability": "power", "attributes": {"state": "ON"}, "timestamp": "2023-01-01T08:00:00"}, {"device": "door", "capability": "lock", "attributes": {"state": "UNLOCKED"}, "timestamp": "2023-01-01T08:01:00"}]'
```

## Docker

You can run the LEM model in a Docker container:

```bash
# Build the Docker image
docker build -t lem-model -f lem_package/Dockerfile .

# Run the API server
docker run -p 8000:8000 lem-model
```

## Deployment to Google Cloud Platform (GCP)

The package includes a script to deploy the LEM model to Google Cloud Run:

1. Edit the `deploy_to_gcp.sh` script to set your GCP project ID and preferred region:
   ```bash
   PROJECT_ID="your-gcp-project-id"  # Replace with your GCP project ID
   REGION="us-central1"  # Replace with your preferred region
   ```

2. Run the deployment script:
   ```bash
   cd lem_package
   ./deploy_to_gcp.sh
   ```

3. The script will:
   - Build the Docker image
   - Push it to Google Container Registry
   - Deploy it to Google Cloud Run
   - Output the URL of your deployed API

4. Access your API:
   - Health check: `curl https://your-service-url/health`
   - API documentation: `https://your-service-url/docs`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 