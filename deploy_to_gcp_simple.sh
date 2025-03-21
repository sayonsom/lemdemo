#!/bin/bash
# Script to build and deploy the simplified LEM model to Google Cloud Run

# Configuration
PROJECT_ID="lem-gpet-v1"  # User's GCP project ID
REPO_NAME="lem-repo"  # Artifact Registry repository name
IMAGE_NAME="lem-model-simple"
REGION="us-central1"  # Default region, can be changed if needed
SERVICE_NAME="lem-api-simple"
IMAGE_TAG="us-central1-docker.pkg.dev/lem-gpet-v1/lem-repo/lem-model-simple:latest"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install it from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if required model files are present
echo "Checking if all required model files are present..."
python check_deployment_files.py
if [ $? -ne 0 ]; then
    echo "Error: Missing required model files. Please check the error message above."
    exit 1
fi

# Ensure user is logged in to gcloud
echo "Checking gcloud authentication..."
gcloud auth print-access-token &> /dev/null || gcloud auth login

# Set the project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Configure Docker to use gcloud as a credential helper for Artifact Registry
echo "Configuring Docker to use gcloud credentials for Artifact Registry..."
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build the Docker image using the simplified Dockerfile
echo "Building Docker image..."
docker build -t $IMAGE_TAG -f Dockerfile.simple ..

# Push the image to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push $IMAGE_TAG

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --port 8000

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo "Deployment complete!"
echo "Your LEM API is now available at: $SERVICE_URL"
echo "Try it out with: curl $SERVICE_URL/health"
echo "API documentation is available at: $SERVICE_URL/docs" 