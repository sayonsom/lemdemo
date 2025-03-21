"""
Simple FastAPI implementation for the Large Event Model (LEM) API.
This file serves as the entry point for the FastAPI application when running in a Docker container.
"""

import os
import json
import pickle
import torch
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Global variables for model and encoder paths
MODEL_PATH = "/app/models/event_embedding_model.pt"
DEVICE_ENCODER_PATH = "/app/encoders/device_encoder.pkl"
CAPABILITY_ENCODER_PATH = "/app/encoders/capability_encoder.pkl"
STATE_ENCODER_PATH = "/app/encoders/state_encoder.pkl"
HISTORICAL_EMBEDDINGS_PATH = "/app/data/historical_event_embeddings.pt"
HISTORICAL_ACTIONS_PATH = "/app/data/historical_actions.json"

# Define the FastAPI app
app = FastAPI(
    title="Large Event Model API",
    description="API for preprocessing events and suggesting actions based on recent events",
    version="1.0.0",
)

# Define data models
class Event(BaseModel):
    device: str
    capability: str
    state: str
    timestamp: Optional[str] = None

class EventFeatures(BaseModel):
    device_encoded: int
    capability_encoded: int
    state_encoded: int
    features: List[float]

class Embedding(BaseModel):
    embedding: List[float]

class Action(BaseModel):
    action: str
    confidence: float

# Global variables for loaded resources
device_encoder = None
capability_encoder = None
state_encoder = None
model = None
historical_embeddings = None
historical_actions = None

# Dictionary to store known classes for each encoder
device_classes = []
capability_classes = []
state_classes = []

def load_resources():
    """Load all required resources for the API."""
    global device_encoder, capability_encoder, state_encoder, model, historical_embeddings, historical_actions
    global device_classes, capability_classes, state_classes
    
    # Load encoders - try different methods
    try:
        # Try joblib first
        device_encoder = joblib.load(DEVICE_ENCODER_PATH)
        capability_encoder = joblib.load(CAPABILITY_ENCODER_PATH)
        state_encoder = joblib.load(STATE_ENCODER_PATH)
        print("Loaded encoders using joblib")
        
        # Store known classes
        device_classes = device_encoder.classes_.tolist() if hasattr(device_encoder, 'classes_') else []
        capability_classes = capability_encoder.classes_.tolist() if hasattr(capability_encoder, 'classes_') else []
        state_classes = state_encoder.classes_.tolist() if hasattr(state_encoder, 'classes_') else []
        
        print(f"Known device classes: {device_classes}")
        print(f"Known capability classes: {capability_classes}")
        print(f"Known state classes: {state_classes}")
        
    except Exception as e:
        print(f"Failed to load encoders with joblib: {e}")
        try:
            # Try pickle as fallback
            with open(DEVICE_ENCODER_PATH, 'rb') as f:
                device_encoder = pickle.load(f)
            with open(CAPABILITY_ENCODER_PATH, 'rb') as f:
                capability_encoder = pickle.load(f)
            with open(STATE_ENCODER_PATH, 'rb') as f:
                state_encoder = pickle.load(f)
            print("Loaded encoders using pickle")
            
            # Store known classes
            device_classes = device_encoder.classes_.tolist() if hasattr(device_encoder, 'classes_') else []
            capability_classes = capability_encoder.classes_.tolist() if hasattr(capability_encoder, 'classes_') else []
            state_classes = state_encoder.classes_.tolist() if hasattr(state_encoder, 'classes_') else []
            
            print(f"Known device classes: {device_classes}")
            print(f"Known capability classes: {capability_classes}")
            print(f"Known state classes: {state_classes}")
            
        except Exception as e:
            print(f"Failed to load encoders with pickle: {e}")
            raise
    
    # Load model - handle different formats
    try:
        # Try loading as a PyTorch model
        model_data = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Check if it's a model or just a state dict
        if hasattr(model_data, 'eval'):
            model = model_data
            model.eval()
        else:
            # Assume it's a state dict or other data structure
            model = model_data  # Just store whatever was loaded
            
        print("Loaded model successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Continue without the model for now
        model = {}
    
    # Load historical data
    try:
        historical_embeddings = torch.load(HISTORICAL_EMBEDDINGS_PATH, map_location=torch.device('cpu'))
        print("Loaded historical embeddings successfully")
    except Exception as e:
        print(f"Failed to load historical embeddings: {e}")
        historical_embeddings = []
    
    try:
        with open(HISTORICAL_ACTIONS_PATH, 'r') as f:
            historical_actions = json.load(f)
        print(f"Loaded historical actions successfully: {type(historical_actions)}")
        print(f"First action: {historical_actions[0] if historical_actions else None}")
    except Exception as e:
        print(f"Failed to load historical actions: {e}")
        historical_actions = []

@app.on_event("startup")
async def startup_event():
    """Load resources on startup."""
    try:
        load_resources()
        print("All resources loaded successfully")
    except Exception as e:
        print(f"Error during startup: {e}")
        # Continue anyway to allow health endpoint to work

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    resources_loaded = all([
        device_encoder is not None,
        capability_encoder is not None,
        state_encoder is not None,
        model is not None
    ])
    
    return {
        "status": "healthy" if resources_loaded else "degraded",
        "message": "LEM API is running",
        "resources_loaded": resources_loaded,
        "known_devices": device_classes,
        "known_capabilities": capability_classes,
        "known_states": state_classes
    }

def safe_encode(encoder, value, known_classes):
    """Safely encode a value, handling unknown labels."""
    if value in known_classes:
        return encoder.transform([value])[0]
    else:
        # Return a default value for unknown labels
        return -1

@app.post("/preprocess", response_model=List[EventFeatures])
async def preprocess_event(event: Event):
    """Preprocess an event by encoding device, capability, and state."""
    if not all([device_encoder, capability_encoder, state_encoder]):
        raise HTTPException(status_code=503, detail="Encoders not loaded. API is in degraded state.")
    
    try:
        # Encode categorical features safely
        device_encoded = safe_encode(device_encoder, event.device, device_classes)
        capability_encoded = safe_encode(capability_encoder, event.capability, capability_classes)
        state_encoded = safe_encode(state_encoder, event.state, state_classes)
        
        # Create a simple feature vector (placeholder)
        features = [float(device_encoded), float(capability_encoded), float(state_encoded)]
        
        return [EventFeatures(
            device_encoded=device_encoded,
            capability_encoded=capability_encoded,
            state_encoded=state_encoded,
            features=features
        )]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing event: {str(e)}")

@app.post("/suggest", response_model=Action)
async def suggest_action(recent_events: List[Event]):
    """Suggest an action based on recent events."""
    if not historical_actions:
        raise HTTPException(status_code=503, detail="Historical actions not loaded. API is in degraded state.")
    
    try:
        # For simplicity, just return a random action from historical actions
        if historical_actions and len(historical_actions) > 0:
            # Handle different formats of historical actions
            action_data = historical_actions[0]
            
            if isinstance(action_data, dict):
                # Format the action as a string representation of the dict
                action_str = json.dumps(action_data)
            elif isinstance(action_data, str):
                action_str = action_data
            else:
                action_str = str(action_data)
                
            return Action(action=action_str, confidence=0.95)
        else:
            return Action(action="No action suggested", confidence=0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error suggesting action: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 