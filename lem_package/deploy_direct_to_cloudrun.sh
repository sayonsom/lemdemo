#!/bin/bash
# Script to deploy the LEM API directly to Cloud Run without using Artifact Registry

# Configuration
PROJECT_ID="lem-gpet-v1"  # User's GCP project ID
SERVICE_NAME="lem-api-direct"
REGION="us-central1"  # Default region, can be changed if needed

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install"
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

# Enable required APIs if not already enabled
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com

# Deploy directly to Cloud Run using source code
echo "Deploying directly to Cloud Run..."
cd ..  # Move to the parent directory
gcloud run deploy $SERVICE_NAME \
    --source . \
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