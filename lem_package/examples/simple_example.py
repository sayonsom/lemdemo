#!/usr/bin/env python3
"""
Simple example of using the LEM package.
"""

import json
from lem_package.inference import load_resources, preprocess_event, get_event_embedding, suggest_device_action

def main():
    """
    Main function to demonstrate the LEM package.
    """
    # Load resources
    print("Loading resources...")
    model, device_encoder, capability_encoder, state_encoder, historical_embeddings, historical_actions = load_resources()
    
    # Example event
    event = {
        "device": "ac_unit",
        "capability": "power",
        "attributes": {
            "state": "ON"
        },
        "timestamp": "2023-01-01T08:00:00"
    }
    
    # Preprocess event
    print("\nPreprocessing event...")
    features = preprocess_event(event, device_encoder, capability_encoder, state_encoder)
    print(f"Preprocessed features: {features}")
    
    # Example event sequence
    events = [
        {
            "device": "light",
            "capability": "power",
            "attributes": {
                "state": "ON"
            },
            "timestamp": "2023-01-01T08:00:00"
        },
        {
            "device": "door",
            "capability": "lock",
            "attributes": {
                "state": "UNLOCKED"
            },
            "timestamp": "2023-01-01T08:01:00"
        }
    ]
    
    # Generate embedding
    print("\nGenerating embedding...")
    embedding = get_event_embedding(events, model)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")
    
    # Suggest action
    print("\nSuggesting action...")
    action = suggest_device_action(events, historical_embeddings, historical_actions, model)
    print(f"Suggested action: {json.dumps(action, indent=2)}")

if __name__ == '__main__':
    main() 