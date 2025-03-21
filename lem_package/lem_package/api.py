"""
FastAPI implementation for the Large Event Model (LEM).
"""

import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from .inference import load_resources, preprocess_event, get_event_embedding, suggest_device_action

app = FastAPI(
    title="Large Event Model API",
    description="API for processing, analyzing, and predicting timeseries events from home appliances",
    version="0.1.0"
)

# Load resources on startup
model, device_encoder, capability_encoder, state_encoder, historical_embeddings, historical_actions = load_resources()

class Event(BaseModel):
    device: str
    capability: str
    attributes: Dict[str, Any]
    timestamp: str

class EventFeatures(BaseModel):
    features: List[float]

class Embedding(BaseModel):
    embedding: List[List[float]]

class Action(BaseModel):
    action: Dict[str, Any]

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@app.post("/preprocess", response_model=EventFeatures)
def preprocess(event: Event):
    """
    Preprocess an event.
    """
    try:
        # Preprocess event
        features = preprocess_event(event.dict(), device_encoder, capability_encoder, state_encoder)
        
        # Convert tensor to list for JSON serialization
        features_list = features.tolist()
        
        return {"features": features_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing event: {str(e)}")

@app.post("/embed", response_model=Embedding)
def embed(events: List[Event]):
    """
    Generate an embedding for a sequence of events.
    """
    try:
        # Convert Pydantic models to dictionaries
        event_dicts = [event.dict() for event in events]
        
        # Generate embedding
        embedding = get_event_embedding(event_dicts, model)
        
        # Convert tensor to list for JSON serialization
        embedding_list = embedding.tolist()
        
        return {"embedding": embedding_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.post("/suggest", response_model=Action)
def suggest(events: List[Event]):
    """
    Suggest a device action based on recent events.
    """
    try:
        # Convert Pydantic models to dictionaries
        event_dicts = [event.dict() for event in events]
        
        # Suggest action
        action = suggest_device_action(event_dicts, historical_embeddings, historical_actions, model)
        
        return {"action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error suggesting action: {str(e)}")

def run_server(host='0.0.0.0', port=8000, debug=False):
    """
    Run the FastAPI server using Uvicorn.
    """
    uvicorn.run("lem_package.api:app", host=host, port=port, reload=debug)

if __name__ == '__main__':
    run_server(debug=True) 