#!/bin/bash
# Script to run the FastAPI server locally for testing

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "Error: uvicorn is not installed. Please install it with: pip install uvicorn"
    exit 1
fi

# Run the server
echo "Starting FastAPI server on http://localhost:8000..."
echo "API documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
uvicorn lem_package.api:app --host 0.0.0.0 --port 8000 --reload 