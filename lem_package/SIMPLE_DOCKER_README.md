# Simplified Docker Setup for LEM API

This directory contains a simplified Docker setup for the Large Event Model (LEM) API. This setup is designed to be easier to deploy and test, with more robust error handling and simplified code.

## Files

- `Dockerfile.simple`: A simplified Dockerfile that builds a Docker image for the LEM API.
- `simple_api.py`: A simplified FastAPI implementation for the LEM API.
- `requirements.txt`: A list of Python dependencies required by the API.
- `test_simple_docker.sh`: A script to build and run the Docker container.

## Prerequisites

- Docker installed on your system
- Python 3.8 or higher
- Required model files in the project root directory:
  - `event_embedding_model.pt`
  - `device_encoder.pkl`
  - `capability_encoder.pkl`
  - `state_encoder.pkl`
  - `historical_event_embeddings.pt`
  - `historical_actions.json`

## Getting Started

1. Make sure all required model files are in the project root directory.
2. Run the test script:

```bash
./test_simple_docker.sh
```

This will:
- Check if all required model files are present
- Build the Docker image
- Run the Docker container
- Test the health endpoint

## API Endpoints

The API provides the following endpoints:

### Health Check

```
GET /health
```

Returns the health status of the API and information about loaded resources.

Example response:
```json
{
  "status": "healthy",
  "message": "LEM API is running",
  "resources_loaded": true,
  "known_devices": ["ac_unit", "fridge", "smart_light", "smart_lock", "smart_tv", "smartphone", "washer"],
  "known_capabilities": ["brightness_control", "door_status", "incoming_call", "lock_control", "power", "temperature_control", "volume_control"],
  "known_states": ["BRIGHT", "CALL_RECEIVED", "COOL_HIGH", "COOL_MEDIUM", "DIM", "ENERGY_SAVER", "HIGH", "LOCKED", "LOW", "MEDIUM", "MUTE", "NO_CALL", "OFF", "ON", "OPEN", "UNLOCKED"]
}
```

### Preprocess Event

```
POST /preprocess
```

Preprocesses an event by encoding device, capability, and state.

Example request:
```json
{
  "device": "smart_light",
  "capability": "brightness_control",
  "state": "BRIGHT"
}
```

Example response:
```json
[
  {
    "device_encoded": 2,
    "capability_encoded": 0,
    "state_encoded": 0,
    "features": [2.0, 0.0, 0.0]
  }
]
```

### Suggest Action

```
POST /suggest
```

Suggests an action based on recent events.

Example request:
```json
[
  {
    "device": "smart_light",
    "capability": "brightness_control",
    "state": "BRIGHT"
  }
]
```

Example response:
```json
{
  "action": "{\"device\": \"smartphone\", \"capability\": \"incoming_call\", \"state\": \"off\"}",
  "confidence": 0.95
}
```

## Docker Commands

- View logs: `docker logs lem-api-simple`
- Stop the container: `docker stop lem-api-simple`
- Remove the container: `docker rm lem-api-simple`

## Troubleshooting

If you encounter issues:

1. Check if all required model files are present in the project root directory.
2. Check the Docker logs for error messages: `docker logs lem-api-simple`
3. Make sure the Docker container is running: `docker ps`
4. If the API is in a degraded state, check the health endpoint for more information.

## Known Issues

- The API may show warnings about unpickling estimators from different scikit-learn versions. This is expected and should not affect functionality.
- The model loading is designed to be flexible and handle different formats, which may result in some non-critical error messages in the logs. 