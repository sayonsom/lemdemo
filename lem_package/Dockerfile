FROM python:3.8-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.95.0 \
    uvicorn>=0.21.0 \
    pydantic>=1.10.7 \
    scikit-learn>=1.0.0 \
    joblib>=1.0.0 \
    numpy>=1.20.0

# Create directories for model files
RUN mkdir -p /app/lem_package/models /app/lem_package/encoders /app/lem_package/data

# Copy package files
COPY lem_package/ /app/lem_package/

# Copy the app.py file
COPY lem_package/app.py /app/app.py

# Copy model files
COPY event_embedding_model.pt /app/lem_package/models/
COPY device_encoder.pkl /app/lem_package/encoders/
COPY capability_encoder.pkl /app/lem_package/encoders/
COPY state_encoder.pkl /app/lem_package/encoders/
COPY historical_event_embeddings.pt /app/lem_package/data/
COPY historical_actions.json /app/lem_package/data/

# Install PyTorch separately to avoid timeout issues
RUN pip install --no-cache-dir torch==1.13.1

# Expose port for API
EXPOSE 8000

# Set entrypoint to Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 