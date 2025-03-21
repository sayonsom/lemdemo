# Deploying LEM API to Google Cloud Platform

This guide will walk you through the process of deploying your LEM API to Google Cloud Platform (GCP) using Google Cloud Run.

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

## Deployment Steps

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
```

Follow the prompts to authenticate with your Google account that has access to the GCP project.

### 2. Set Up Docker to Use GCP Credentials

```bash
gcloud auth configure-docker
```

This allows Docker to push images to Google Container Registry.

### 3. Run the Deployment Script

The deployment script has been configured with your project details:
- Project ID: `lem-gpet-v`
- Region: `us-central1`
- Service name: `lem-api`

To deploy, run:

```bash
cd lem_package
chmod +x deploy_to_gcp.sh  # Make sure the script is executable
./deploy_to_gcp.sh
```

The script will:
1. Build a Docker image of your LEM API
2. Push the image to Google Container Registry
3. Deploy the image to Google Cloud Run
4. Output the URL where your API is accessible

### 4. Verify Deployment

Once deployment is complete, you'll receive a URL where your API is hosted. You can verify it's working with:

```bash
# Replace YOUR_SERVICE_URL with the URL from the deployment output
curl YOUR_SERVICE_URL/health
```

You should receive a response like:
```json
{"status": "ok"}
```

### 5. Access API Documentation

Your API documentation is available at:
```
YOUR_SERVICE_URL/docs
```

This provides an interactive Swagger UI where you can test all API endpoints.

## Troubleshooting

### Image Build Failures

If the Docker image fails to build:
1. Check that all required model files are in the correct location
2. Verify your Docker installation is working correctly
3. Check for any error messages in the build output

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

## Additional Configuration

### Scaling Configuration

By default, Cloud Run will automatically scale based on traffic. You can configure minimum and maximum instances:

```bash
gcloud run services update lem-api \
  --min-instances=0 \
  --max-instances=10 \
  --region=us-central1
```

### Memory and CPU Allocation

If your model requires more resources, you can update the deployment:

```bash
gcloud run services update lem-api \
  --memory=4Gi \
  --cpu=2 \
  --region=us-central1
```

### Custom Domain

To use a custom domain with your API:
1. Go to the Cloud Run console
2. Select your service
3. Go to "Domain Mappings"
4. Follow the instructions to map your custom domain 