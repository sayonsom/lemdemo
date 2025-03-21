# Testing LEM API Locally with Docker

This guide will walk you through the process of testing your LEM API locally using Docker before deploying to Google Cloud Platform (GCP).

## Prerequisites

1. **Docker**: Ensure Docker is installed and running on your machine.
   - Download from: https://docs.docker.com/get-docker/
   - Verify installation with: `docker --version`

2. **Required Model Files**: Ensure you have the following model files in your project root directory:
   - `event_embedding_model.pt`
   - `device_encoder.pkl`
   - `capability_encoder.pkl`
   - `state_encoder.pkl`
   - `historical_event_embeddings.pt`
   - `historical_actions.json`

3. **jq**: For pretty-printing JSON responses (optional but recommended).
   - Install on macOS: `brew install jq`
   - Install on Ubuntu: `sudo apt-get install jq`

## Testing Steps

### 1. Check Required Files

First, make sure all the required model files are present:

```bash
cd lem_package
python check_deployment_files.py
```

If all files are present, you'll see a success message. If any files are missing, the script will tell you which ones.

### 2. Build and Run Docker Container

We've provided a script that automates the process of building and running the Docker container:

```bash
cd lem_package
./test_local_docker.sh
```

This script will:
1. Check if all required model files are present
2. Stop and remove any existing container with the same name
3. Build the Docker image
4. Run the container
5. Test the health endpoint to verify the API is working

If successful, you'll see a message indicating that the container is running and the API is available at http://localhost:8000.

### 3. Test API Endpoints

Once the container is running, you can test the API endpoints using the provided script:

```bash
cd lem_package
./test_api_endpoints.sh
```

This script will test the following endpoints:
- `GET /health`: Health check endpoint
- `POST /preprocess`: Preprocess an event
- `POST /embed`: Generate an embedding for a sequence of events
- `POST /suggest`: Suggest a device action based on recent events

### 4. Access Interactive API Documentation

You can also access the interactive API documentation at:
```
http://localhost:8000/docs
```

This provides a Swagger UI where you can explore and test all API endpoints.

### 5. Manually Test Endpoints

If you prefer to test the endpoints manually, you can use curl:

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

### 6. View Container Logs

If you encounter any issues, you can view the container logs:

```bash
docker logs lem-api-local
```

### 7. Stop and Remove the Container

When you're done testing, you can stop and remove the container:

```bash
docker stop lem-api-local
docker rm lem-api-local
```

## Troubleshooting

### Container Fails to Start

If the container fails to start, check the logs:

```bash
docker logs lem-api-local
```

Common issues include:
- Missing model files
- Port conflicts (another service is using port 8000)
- Insufficient memory or CPU resources

### API Endpoints Return Errors

If the API endpoints return errors, check:
1. The format of your request data
2. The container logs for error messages
3. That all model files are correctly loaded

### Docker Build Fails

If the Docker build fails, check:
1. That all required files are in the correct locations
2. That you have sufficient disk space
3. That your Docker installation is working correctly

## Next Steps

Once you've verified that your LEM API works correctly in the local Docker container, you can proceed to deploy it to Google Cloud Platform using the `deploy_to_gcp.sh` script.

```bash
cd lem_package
./deploy_to_gcp.sh
```

This will deploy your API to Google Cloud Run, making it accessible over the internet. 