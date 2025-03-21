import json
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import joblib
import os
import argparse

# Define a Transformer-based model
class EventTransformerModel(nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Ensure input has the right shape
        if x.shape[-1] != 7:
            # Pad or truncate to 7 features
            if x.shape[-1] < 7:
                # Pad with zeros
                padding = torch.zeros(*x.shape[:-1], 7 - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[..., :7]
                
        x = self.input_proj(x)
        embeddings = self.transformer_encoder(x)
        pooled_embedding = embeddings.mean(dim=1)
        return self.output_proj(pooled_embedding)

# Load encoders
device_encoder = joblib.load('device_encoder.pkl')
capability_encoder = joblib.load('capability_encoder.pkl')
state_encoder = joblib.load('state_encoder.pkl')

# Preprocess a single event
def preprocess_event(event):
    """Convert event data to numerical features using encoders"""
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

# Function to get embedding from events
def get_event_embedding(events, model):
    """Process a sequence of events and return their embedding"""
    # Preprocess each event
    processed = [preprocess_event(event) for event in events]
    
    # Stack the tensors
    if processed:
        tensor_input = torch.stack(processed).unsqueeze(0)
    else:
        return torch.zeros((1, 128), dtype=torch.float32)  # Default size
    
    # Get embedding from model
    with torch.no_grad():
        embedding = model(tensor_input)
    return embedding

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Transformer-based Event Embedding Demo')
    parser.add_argument('--device', type=str, help='Specify a device for a custom event')
    parser.add_argument('--capability', type=str, help='Specify a capability for a custom event')
    parser.add_argument('--state', type=str, help='Specify a state for a custom event')
    parser.add_argument('--top', type=int, default=1, help='Show top N most similar actions')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode to enter custom events')
    args = parser.parse_args()
    
    # Check if saved model exists
    model = EventTransformerModel(input_dim=7, embed_dim=128)
    if os.path.exists('transformer_model.pt'):
        try:
            # Try to load the saved model
            model.load_state_dict(torch.load('transformer_model.pt', weights_only=False))
            model.eval()
            print("Loaded saved Transformer model from transformer_model.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model instead...")
            model.eval()
            # Save the new model
            torch.save(model.state_dict(), 'transformer_model.pt')
            print("Saved new Transformer model to transformer_model.pt")
    else:
        model.eval()
        # Save the model
        torch.save(model.state_dict(), 'transformer_model.pt')
        print("Saved new Transformer model to transformer_model.pt")
        
    # Check if saved embeddings exist
    if os.path.exists('transformer_embeddings.pt') and os.path.exists('transformer_actions.json'):
        try:
            # Load saved embeddings and actions
            historical_embeddings = torch.load('transformer_embeddings.pt', weights_only=False)
            with open('transformer_actions.json', 'r') as f:
                historical_actions = json.load(f)
            print(f"Loaded saved embeddings (shape: {historical_embeddings.shape}) and {len(historical_actions)} actions")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("Will create new embeddings...")
            historical_embeddings = None
            historical_actions = None
    else:
        historical_embeddings = None
        historical_actions = None
        
    # Load dataset
    with open('event_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    # Get recent events
    if args.interactive:
        # Interactive mode - prompt user for custom events
        print("\n=== Interactive Mode ===")
        print("Enter custom events (up to 5). Press Enter with empty device name to finish.")
        
        custom_events = []
        max_events = 5
        
        # List available devices, capabilities, and states for reference
        available_devices = ["smart_light", "smart_tv", "smart_thermostat", "smart_lock", "smartphone", "washer", "dryer"]
        available_capabilities = {
            "smart_light": ["power", "brightness_control", "color_control"],
            "smart_tv": ["power", "volume_control", "channel_control"],
            "smart_thermostat": ["temperature_control", "mode_control"],
            "smart_lock": ["lock_control"],
            "smartphone": ["incoming_call", "app_usage"],
            "washer": ["power", "cycle_control", "door_control"],
            "dryer": ["power", "cycle_control", "door_control"]
        }
        available_states = {
            "power": ["ON", "OFF"],
            "brightness_control": ["BRIGHT", "DIM", "OFF"],
            "color_control": ["RED", "GREEN", "BLUE", "WHITE"],
            "volume_control": ["LOUD", "QUIET", "MUTE"],
            "channel_control": ["CHANNEL_1", "CHANNEL_2", "CHANNEL_3"],
            "temperature_control": ["COOL", "WARM", "HOT"],
            "mode_control": ["ECO", "NORMAL", "BOOST"],
            "lock_control": ["LOCKED", "UNLOCKED"],
            "incoming_call": ["CALL_RECEIVED", "NO_CALL"],
            "app_usage": ["ACTIVE", "INACTIVE"],
            "cycle_control": ["WASH", "RINSE", "SPIN", "DRY"],
            "door_control": ["OPEN", "CLOSED"]
        }
        
        print("\nAvailable devices:", ", ".join(available_devices))
        
        for i in range(max_events):
            print(f"\nEvent {i+1}/{max_events}:")
            device = input("Device (or Enter to finish): ").strip()
            if not device:
                break
                
            # Show available capabilities for the selected device
            if device in available_capabilities:
                print(f"Available capabilities for {device}: {', '.join(available_capabilities[device])}")
            
            capability = input("Capability: ").strip()
            
            # Show available states for the selected capability
            if capability in available_states:
                print(f"Available states for {capability}: {', '.join(available_states[capability])}")
                
            state = input("State: ").strip()
            
            custom_event = {
                "device": device,
                "capability": capability,
                "attributes": {"state": state},
                "timestamp": datetime.now().isoformat()
            }
            custom_events.append(custom_event)
        
        if custom_events:
            # Use custom events
            if len(custom_events) < 5:
                # Fill remaining slots with events from dataset
                remaining = 5 - len(custom_events)
                recent_events = dataset[0][-remaining:] + custom_events
            else:
                recent_events = custom_events
            print("\nUsing your custom events as context:")
        else:
            # No custom events entered, use default
            recent_events = dataset[0][-5:]  # Last 5 events as context
            print("\nUsing these recent events as context:")
    elif args.device and args.capability and args.state:
        # Single custom event from command line
        custom_event = {
            "device": args.device,
            "capability": args.capability,
            "attributes": {"state": args.state},
            "timestamp": datetime.now().isoformat()
        }
        
        # Use the last 4 events from the dataset and add the custom event
        recent_events = dataset[0][-4:] + [custom_event]
        print("\nUsing context with your custom event:")
    else:
        # Use default events from dataset
        recent_events = dataset[0][-5:]  # Last 5 events as context
        print("\nUsing these recent events as context:")
    
    for i, event in enumerate(recent_events):
        print(f"{i+1}. Device: {event['device']}, Capability: {event['capability']}, State: {event['attributes']['state']}")
    
    # Generate embeddings for recent events
    event_embedding = get_event_embedding(recent_events, model)
    print(f"\nGenerated embedding shape: {event_embedding.shape}")
    
    # Generate historical embeddings
    print("\nGenerating historical embeddings...")
    historical_data = dataset[:10]  # Use first 10 sequences
    historical_events = [event for sequence in historical_data for event in sequence]
    
    # Process in batches
    all_embeddings = []
    batch_size = 5
    for i in range(0, len(historical_events), batch_size):
        if i + batch_size <= len(historical_events):
            event_group = historical_events[i:i+batch_size]
            all_embeddings.append(get_event_embedding(event_group, model))
    
    # Stack historical embeddings
    historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, 128))
    print(f"Historical embeddings shape: {historical_embeddings.shape}")
    
    # Save historical embeddings
    torch.save(historical_embeddings, 'transformer_embeddings.pt')
    print("Saved historical embeddings to transformer_embeddings.pt")
    
    # Create historical actions
    historical_actions = []
    for i in range(historical_embeddings.shape[0]):
        # Create a dummy action
        action = {
            "device": "smart_light",
            "capability": "power",
            "state": "on" if i % 2 == 0 else "off"
        }
        historical_actions.append(action)
    
    # Save historical actions
    with open('transformer_actions.json', 'w') as f:
        json.dump(historical_actions, f)
    print(f"Saved {len(historical_actions)} historical actions to transformer_actions.json")
    
    # Calculate similarity
    similarities = nn.CosineSimilarity(dim=1)(
        event_embedding, 
        historical_embeddings.squeeze(1)
    )
    
    # Get top N most similar actions
    top_n = min(args.top, len(similarities))
    top_indices = torch.topk(similarities, top_n).indices.tolist()
    top_similarities = torch.topk(similarities, top_n).values.tolist()
    
    print("\n=== Prediction Results ===")
    for i, (idx, similarity) in enumerate(zip(top_indices, top_similarities)):
        suggested_action = historical_actions[idx]
        print(f"\nTop {i+1} (Similarity: {similarity:.4f}):")
        print(f"Recommendation: Set {suggested_action['device']} {suggested_action['capability']} to {suggested_action['state']}")

if __name__ == "__main__":
    main() 