# Deploying Simplified LEM API to Google Cloud Platform

This guide will walk you through the process of deploying your simplified LEM API to Google Cloud Platform (GCP) using Google Cloud Run.

## Prerequisites

1. **Google Cloud SDK**: Make sure you have the Google Cloud SDK installed on your machine.
   - Download from: https://cloud.google.com/sdk/docs/install
   - Verify installation with: `gcloud --version`

2. **Docker**: Ensure Docker is installed and running on your machine.
   - Download from: https://docs.docker.com/get-docker/
   - Verify installation with: `docker --version`

3. **Required Model Files**: Ensure you have the following model files in your project root directory:
   - `event_embedding_model.pt`
   - `device_encoder.pkl`
   - `capability_encoder.pkl`
   - `state_encoder.pkl`
   - `historical_event_embeddings.pt`
   - `historical_actions.json`

4. **GCP Project**: You need a Google Cloud Platform project with billing enabled.
   - The deployment script is configured to use the project ID: `lem-gpet-v`
   - If you need to use a different project ID, edit the `PROJECT_ID` variable in the `deploy_to_gcp_simple.sh` script.

## Deployment Steps

### 1. Test Locally First

Before deploying to GCP, it's recommended to test your API locally using Docker:

```bash
cd lem_package
./test_simple_docker.sh
```

This will build and run the Docker container locally, allowing you to verify that everything works as expected.

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
```

Follow the prompts to authenticate with your Google account that has access to the GCP project.

### 3. Run the Deployment Script

```bash
cd lem_package
./deploy_to_gcp_simple.sh
```

The script will:
1. Check if all required model files are present
2. Authenticate with Google Cloud
3. Configure Docker to use gcloud credentials
4. Build the Docker image using the simplified Dockerfile
5. Push the image to Google Container Registry
6. Deploy the image to Google Cloud Run
7. Output the URL where your API is accessible

### 4. Verify Deployment

Once deployment is complete, you'll receive a URL where your API is hosted. You can verify it's working with:

```bash
# Replace YOUR_SERVICE_URL with the URL from the deployment output
curl YOUR_SERVICE_URL/health
```

You should receive a response like:
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

### 5. Access API Documentation

Your API documentation is available at:
```
YOUR_SERVICE_URL/docs
```

This provides an interactive Swagger UI where you can test all API endpoints.

## Testing the Deployed API

You can test the deployed API using curl:

### Health Check
```bash
curl -X GET YOUR_SERVICE_URL/health
```

### Preprocess Event
```bash
curl -X POST YOUR_SERVICE_URL/preprocess \
  -H "Content-Type: application/json" \
  -d '{"device": "smart_light", "capability": "brightness_control", "state": "BRIGHT"}'
```

### Suggest Action
```bash
curl -X POST YOUR_SERVICE_URL/suggest \
  -H "Content-Type: application/json" \
  -d '[{"device": "smart_light", "capability": "brightness_control", "state": "BRIGHT"}]'
```

## Troubleshooting

### Deployment Failures

If deployment to Cloud Run fails:
1. Verify you have the necessary permissions in your GCP project
2. Check if the service account has the required roles
3. Review the error messages from the `gcloud run deploy` command

### API Not Responding

If the API is deployed but not responding:
1. Check the Cloud Run logs in the GCP Console
2. Verify the model files were correctly included in the Docker image
3. Try redeploying with increased memory allocation if needed

## Monitoring and Management

### Viewing Logs

You can view the logs of your deployed service in the GCP Console:
1. Go to Cloud Run in the GCP Console
2. Click on your service (`lem-api-simple`)
3. Click on "Logs" tab

### Updating the Deployment

If you need to update your deployment:
1. Make your changes to the code
2. Run the deployment script again:
```bash
./deploy_to_gcp_simple.sh
```

### Deleting the Deployment

If you want to delete the deployment:
```bash
gcloud run services delete lem-api-simple --platform managed --region us-central1
```

## Cost Management

Cloud Run charges based on the resources your service uses. To manage costs:
1. Configure minimum and maximum instances
2. Set appropriate memory and CPU allocations
3. Monitor your usage in the GCP Console

For more information on Cloud Run pricing, visit: https://cloud.google.com/run/pricing 