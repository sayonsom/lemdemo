#!/bin/bash
# Script to build and run the LEM API Docker container locally

# Configuration
IMAGE_NAME="lem-model-local"
CONTAINER_NAME="lem-api-local"
PORT=8000

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

# Stop and remove existing container if it exists
echo "Stopping and removing existing container if it exists..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build the Docker image
echo "Building Docker image..."
cd ..
docker build -t $IMAGE_NAME -f lem_package/Dockerfile .

# Run the container
echo "Running the container..."
docker run --name $CONTAINER_NAME -p $PORT:8000 -d $IMAGE_NAME

# Check if container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container is running!"
    echo "Your LEM API is now available at: http://localhost:$PORT"
    echo "Try it out with: curl http://localhost:$PORT/health"
    echo "API documentation is available at: http://localhost:$PORT/docs"
    
    # Wait for the API to start up
    echo "Waiting for the API to start up..."
    sleep 5
    
    # Test the health endpoint
    echo "Testing the health endpoint..."
    curl -s http://localhost:$PORT/health
    echo -e "\n"
else
    echo "Error: Container failed to start."
    echo "Check the logs with: docker logs $CONTAINER_NAME"
    exit 1
fi

echo "To view logs: docker logs $CONTAINER_NAME"
echo "To stop the container: docker stop $CONTAINER_NAME"
echo "To remove the container: docker rm $CONTAINER_NAME" 