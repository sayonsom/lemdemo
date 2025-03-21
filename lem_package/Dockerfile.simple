FROM python:3.8-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY lem_package/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for model files
RUN mkdir -p /app/models /app/encoders /app/data

# Copy model files
COPY event_embedding_model.pt /app/models/
COPY device_encoder.pkl /app/encoders/
COPY capability_encoder.pkl /app/encoders/
COPY state_encoder.pkl /app/encoders/
COPY historical_event_embeddings.pt /app/data/
COPY historical_actions.json /app/data/

# Copy the API code
COPY lem_package/simple_api.py /app/api.py

# Expose port for the API
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 