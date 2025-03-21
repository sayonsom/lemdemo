"""
Inference functions for the Large Event Model (LEM).
"""

import os
import json
import torch
import joblib
from datetime import datetime
import numpy as np
from .model import EventTransformerModel, LSTMEmbeddingModel

# Default paths for model and encoder files
# Check if running in Docker container
if os.path.exists('/app/lem_package/models'):
    # Docker paths
    DEFAULT_MODEL_PATH = '/app/lem_package/models/event_embedding_model.pt'
    DEFAULT_DEVICE_ENCODER_PATH = '/app/lem_package/encoders/device_encoder.pkl'
    DEFAULT_CAPABILITY_ENCODER_PATH = '/app/lem_package/encoders/capability_encoder.pkl'
    DEFAULT_STATE_ENCODER_PATH = '/app/lem_package/encoders/state_encoder.pkl'
    DEFAULT_HISTORICAL_EMBEDDINGS_PATH = '/app/lem_package/data/historical_event_embeddings.pt'
    DEFAULT_HISTORICAL_ACTIONS_PATH = '/app/lem_package/data/historical_actions.json'
else:
    # Local development paths
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'event_embedding_model.pt')
    DEFAULT_DEVICE_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoders', 'device_encoder.pkl')
    DEFAULT_CAPABILITY_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoders', 'capability_encoder.pkl')
    DEFAULT_STATE_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoders', 'state_encoder.pkl')
    DEFAULT_HISTORICAL_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'historical_event_embeddings.pt')
    DEFAULT_HISTORICAL_ACTIONS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'historical_actions.json')

# Global variables to store loaded resources
_model = None
_device_encoder = None
_capability_encoder = None
_state_encoder = None
_historical_embeddings = None
_historical_actions = None

def load_resources(
    model_path=DEFAULT_MODEL_PATH,
    device_encoder_path=DEFAULT_DEVICE_ENCODER_PATH,
    capability_encoder_path=DEFAULT_CAPABILITY_ENCODER_PATH,
    state_encoder_path=DEFAULT_STATE_ENCODER_PATH,
    historical_embeddings_path=DEFAULT_HISTORICAL_EMBEDDINGS_PATH,
    historical_actions_path=DEFAULT_HISTORICAL_ACTIONS_PATH
):
    """
    Load all necessary resources for inference.
    
    Args:
        model_path: Path to the model file
        device_encoder_path: Path to the device encoder file
        capability_encoder_path: Path to the capability encoder file
        state_encoder_path: Path to the state encoder file
        historical_embeddings_path: Path to the historical embeddings file
        historical_actions_path: Path to the historical actions file
        
    Returns:
        Tuple of (model, device_encoder, capability_encoder, state_encoder, 
                 historical_embeddings, historical_actions)
    """
    global _model, _device_encoder, _capability_encoder, _state_encoder, _historical_embeddings, _historical_actions
    
    # Load model if not already loaded
    if _model is None:
        try:
            # First try to load with the Transformer architecture
            _model = EventTransformerModel(input_dim=7, embed_dim=128)
            _model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            _model.eval()
        except Exception as e:
            print(f"Warning: Could not load Transformer model: {str(e)}")
            # Fall back to LSTM model
            try:
                _model = LSTMEmbeddingModel(input_dim=8)
                _model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                _model.eval()
            except Exception as e:
                print(f"Warning: Could not load LSTM model either: {str(e)}")
                print("The saved model architecture doesn't match any of our model architectures.")
                print("Will use feature vectors directly for similarity.")
                _model = None
    
    # Load encoders if not already loaded
    if _device_encoder is None:
        _device_encoder = joblib.load(device_encoder_path)
    
    if _capability_encoder is None:
        _capability_encoder = joblib.load(capability_encoder_path)
    
    if _state_encoder is None:
        _state_encoder = joblib.load(state_encoder_path)
    
    # Load historical data if not already loaded
    if _historical_embeddings is None:
        _historical_embeddings = torch.load(historical_embeddings_path, map_location=torch.device('cpu'))
    
    if _historical_actions is None:
        with open(historical_actions_path, 'r') as f:
            _historical_actions = json.load(f)
    
    return _model, _device_encoder, _capability_encoder, _state_encoder, _historical_embeddings, _historical_actions

def preprocess_event(event, device_encoder=None, capability_encoder=None, state_encoder=None):
    """
    Convert event data to numerical features using encoders.
    
    Args:
        event: Dictionary containing event data
        device_encoder: LabelEncoder for device names
        capability_encoder: LabelEncoder for capability names
        state_encoder: LabelEncoder for state values
        
    Returns:
        Tensor of numerical features
    """
    # Load encoders if not provided
    if device_encoder is None or capability_encoder is None or state_encoder is None:
        _, device_encoder, capability_encoder, state_encoder, _, _ = load_resources()
    
    try:
        # Try to encode with existing encoders
        device = device_encoder.transform([event["device"]])[0]
        capability = capability_encoder.transform([event["capability"]])[0]
        state = state_encoder.transform([event["attributes"]["state"]])[0]
        
        # Create feature vector
        features = [
            device, capability, state,
            # Add time-based features
            float(datetime.fromisoformat(event["timestamp"]).hour) / 24.0,
            float(datetime.fromisoformat(event["timestamp"]).minute) / 60.0,
            float(datetime.fromisoformat(event["timestamp"]).weekday()) / 7.0,
            1.0  # Bias term
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    except (ValueError, KeyError) as e:
        print(f"\nWarning: Encountered unknown label: {str(e)}")
        print("Using fallback encoding for unknown values.")
        
        # Use fallback values for unknown labels
        device_val = 0  # Default to first device
        capability_val = 0  # Default to first capability
        state_val = 0  # Default to first state
        
        # Try to encode known values
        try:
            if event["device"] in device_encoder.classes_:
                device_val = device_encoder.transform([event["device"]])[0]
        except:
            pass
            
        try:
            if event["capability"] in capability_encoder.classes_:
                capability_val = capability_encoder.transform([event["capability"]])[0]
        except:
            pass
            
        try:
            if event["attributes"]["state"] in state_encoder.classes_:
                state_val = state_encoder.transform([event["attributes"]["state"]])[0]
        except:
            pass
        
        # Create feature vector with fallback values
        features = [
            device_val, capability_val, state_val,
            # Add time-based features
            float(datetime.fromisoformat(event["timestamp"]).hour) / 24.0,
            float(datetime.fromisoformat(event["timestamp"]).minute) / 60.0,
            float(datetime.fromisoformat(event["timestamp"]).weekday()) / 7.0,
            1.0  # Bias term
        ]
        
        return torch.tensor(features, dtype=torch.float32)

def get_event_embedding(events, model=None):
    """
    Generate an embedding for a sequence of events.
    
    Args:
        events: List of event dictionaries
        model: Pre-loaded model (optional)
        
    Returns:
        Embedding tensor
    """
    # Load model if not provided
    if model is None:
        model, _, _, _, _, _ = load_resources()
    
    # Preprocess events
    processed_events = [preprocess_event(event) for event in events]
    
    # Stack tensors and add batch dimension
    tensor_input = torch.stack(processed_events).unsqueeze(0)
    
    # Generate embedding
    with torch.no_grad():
        if model is not None:
            embedding = model(tensor_input)
        else:
            # If no model is available, use mean of feature vectors as embedding
            embedding = tensor_input.mean(dim=1)
    
    return embedding

def suggest_device_action(recent_events, historical_embeddings=None, historical_actions=None, model=None):
    """
    Suggest a device action based on recent events.
    
    Args:
        recent_events: List of recent event dictionaries
        historical_embeddings: Tensor of historical embeddings (optional)
        historical_actions: List of historical actions (optional)
        model: Pre-loaded model (optional)
        
    Returns:
        Dictionary containing the suggested action
    """
    # Load resources if not provided
    if model is None or historical_embeddings is None or historical_actions is None:
        model, _, _, _, historical_embeddings, historical_actions = load_resources()
    
    # Get embedding for recent events
    embedding = get_event_embedding(recent_events, model)
    
    # Calculate cosine similarity with historical embeddings
    similarities = torch.nn.functional.cosine_similarity(embedding, historical_embeddings)
    
    # Find most similar pattern
    most_similar_idx = torch.argmax(similarities).item()
    
    # Return suggested action
    return historical_actions[most_similar_idx] 