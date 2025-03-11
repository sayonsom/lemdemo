import json
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np
import joblib
import os
import argparse

# Define model architectures
class EventEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        embeddings = self.transformer(x)
        return embeddings.mean(dim=1)

# Define a Transformer-based model
class EventTransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=2):
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

# Define a model that matches the saved architecture (LSTM-based)
class LSTMEmbeddingModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32):  # Use 8 dimensions to match saved model
        super().__init__()
        # Use Sequential for encoder to match 'encoder.0.weight' in saved model
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        # LSTM with hidden_dim=32 to match saved model dimensions
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Ensure input has the right shape
        if x.shape[-1] != 8:
            # Pad or truncate to 8 features
            if x.shape[-1] < 8:
                # Pad with zeros
                padding = torch.zeros(*x.shape[:-1], 8 - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[..., :8]
        
        x = self.encoder(x)
        x, _ = self.lstm(x)
        # Take the last output for classification/embedding
        return x[:, -1, :]

# Load trained embedding model
try:
    # First try to load with the Transformer architecture
    if os.path.exists('event_embedding_model.pt'):
        try:
            model = EventTransformerModel(input_dim=7, embed_dim=128)
            model.load_state_dict(torch.load('event_embedding_model.pt'))
            print("Loaded pre-trained Transformer embedding model.")
            model.eval()
        except Exception as e:
            print(f"Warning: Could not load Transformer model: {str(e)}")
            # Fall back to LSTM model
            try:
                model = LSTMEmbeddingModel(input_dim=8)
                model.load_state_dict(torch.load('event_embedding_model.pt'))
                print("Loaded pre-trained LSTM embedding model.")
                model.eval()
            except Exception as e:
                print(f"Warning: Could not load LSTM model either: {str(e)}")
                print("The saved model architecture doesn't match any of our model architectures.")
                print("Will use feature vectors directly for similarity.")
                model = None
    else:
        # No saved model, use default Transformer architecture
        model = EventTransformerModel(input_dim=7, embed_dim=128)
        print("Warning: Pre-trained model not found. Using untrained Transformer model.")
        model.eval()
except Exception as e:
    print(f"Warning: Could not initialize embedding model: {str(e)}")
    print("Will use feature vectors directly for similarity.")
    model = None

# Load trained encoders
device_encoder = joblib.load('device_encoder.pkl')
capability_encoder = joblib.load('capability_encoder.pkl')
state_encoder = joblib.load('state_encoder.pkl')

# Define device-specific capabilities and states
DEVICE_CAPABILITIES = {
    'ac_unit': ['power', 'temperature_control'],
    'fridge': ['power', 'door_status', 'temperature_control'],
    'smart_tv': ['power', 'volume_control'],
    'smartphone': ['power', 'incoming_call'],
    'washer': ['power', 'door_status']
}

CAPABILITY_STATES = {
    'power': ['ON', 'OFF'],
    'temperature_control': ['COOL_HIGH', 'COOL_MEDIUM', 'COOL_LOW', 'ENERGY_SAVER_ON', 'ENERGY_SAVER_OFF'],
    'door_status': ['OPEN', 'CLOSED'],
    'volume_control': ['HIGH', 'MEDIUM', 'LOW', 'MUTE'],
    'incoming_call': ['CALL_RECEIVED', 'NO_CALL', 'CALENDAR_ALERT']
}

# Function to get relevant capabilities for a device
def get_device_capabilities(device):
    """Get the relevant capabilities for a specific device"""
    # Remove np.str_ prefix if present
    if isinstance(device, str) and device.startswith('np.str_('):
        device = device.split("'")[1]
    
    # Convert to string if it's a numpy string
    if hasattr(device, 'item'):
        device = device.item()
    
    # Return capabilities for this device or all if not found
    return DEVICE_CAPABILITIES.get(device, list(CAPABILITY_STATES.keys()))

# Function to get relevant states for a capability
def get_capability_states(capability):
    """Get the relevant states for a specific capability"""
    # Remove np.str_ prefix if present
    if isinstance(capability, str) and capability.startswith('np.str_('):
        capability = capability.split("'")[1]
    
    # Convert to string if it's a numpy string
    if hasattr(capability, 'item'):
        capability = capability.item()
    
    # Return states for this capability or all if not found
    return CAPABILITY_STATES.get(capability, [])

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
def get_event_embedding(events):
    """Process a sequence of events and return their embedding"""
    # Preprocess each event
    processed = [preprocess_event(event) for event in events]
    
    # Stack the tensors instead of creating a new tensor
    if processed:
        # Check if processed contains tensors
        if isinstance(processed[0], torch.Tensor):
            # Stack the tensors along a new dimension
            tensor_input = torch.stack(processed).unsqueeze(0)
        else:
            # If they're not tensors (old behavior), convert to tensor
            tensor_input = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
    else:
        # Handle empty list case
        return torch.zeros((1, 1, 7), dtype=torch.float32)  # Default size with 7 features
    
    # Get embedding from model
    try:
        # If model is None, don't even try to use it
        if model is None:
            # Return the processed features directly as the embedding
            return tensor_input
        
        # If using LSTM model, ensure we have 8 features (pad if needed)
        if isinstance(model, LSTMEmbeddingModel) and tensor_input.shape[-1] == 7:
            # Add a bias feature (1.0) to make it 8 dimensions
            bias = torch.ones(*tensor_input.shape[:-1], 1, device=tensor_input.device)
            tensor_input = torch.cat([tensor_input, bias], dim=-1)
            
        with torch.no_grad():
            embedding = model(tensor_input)
        return embedding
    except (NameError, AttributeError, TypeError) as e:
        # If model is not defined or has issues, return the input as the embedding
        print(f"Warning: Could not use model for embedding: {str(e)}")
        print("Using input features directly as embedding.")
        # Return the processed features directly as the embedding
        # Ensure it has the expected shape [1, seq_len, features]
        if len(tensor_input.shape) == 3:
            return tensor_input
        else:
            return tensor_input.unsqueeze(0)

# Simulate LEM inference by embedding similarity (demo)
def suggest_device_action(recent_events, historical_embeddings, historical_actions):
    recent_embedding = get_event_embedding(recent_events)

    # Calculate similarity (cosine similarity)
    cosine_similarity = nn.CosineSimilarity(dim=1)
    similarities = cosine_similarity(recent_embedding, historical_embeddings)

    # Find most similar historical event
    most_similar_idx = torch.argmax(similarities).item()
    suggested_action = historical_actions[most_similar_idx]

    return most_similar_idx, similarities[most_similar_idx].item()

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='LEM Demo - Learning Event Model for Smart Home Automation')
    parser.add_argument('--device', type=str, help='Specify a device to query (e.g., "light", "thermostat")')
    parser.add_argument('--state', type=str, help='Specify a state for the device (e.g., "on", "off")')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    # Load recent events for demo
    with open('event_dataset.json', 'r') as f:
        dataset = json.load(f)

    # Check if historical data exists, if not, create dummy data
    if not os.path.exists('historical_event_embeddings.pt') and not os.path.exists('historical_event_embeddings_new.pt'):
        print("Creating dummy historical embeddings...")
        # Process first 10 sequences as historical data
        historical_data = dataset[:10]
        # Flatten the sequences
        historical_events = [event for sequence in historical_data for event in sequence]
        # Get embeddings for each event
        all_embeddings = []
        for i in range(0, len(historical_events), 5):
            if i + 5 <= len(historical_events):
                event_group = historical_events[i:i+5]
                all_embeddings.append(get_event_embedding(event_group))
        
        historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, 64))
        torch.save(historical_embeddings, 'historical_event_embeddings.pt')
        
        # Create dummy actions
        historical_actions = []
        for i in range(len(all_embeddings)):
            # Create a dummy action based on the last event in each group
            if i*5+4 < len(historical_events):
                last_event = historical_events[i*5+4]
                action = {
                    "device": last_event['device'],
                    "capability": last_event['capability'],
                    "state": "on" if last_event['attributes']['state'] == "off" else "off"
                }
                historical_actions.append(action)
        
        # Save dummy actions
        with open('historical_actions.json', 'w') as f:
            json.dump(historical_actions, f)
    else:
        # Load historical embeddings - prefer the new ones if available
        if os.path.exists('historical_event_embeddings_new.pt'):
            historical_embeddings = torch.load('historical_event_embeddings_new.pt')
            print("Loaded new historical embeddings that match the current model.")
            print(f"Historical embeddings shape: {historical_embeddings.shape}")
        else:
            historical_embeddings = torch.load('historical_event_embeddings.pt')
            print("Loaded original historical embeddings.")
            print(f"Historical embeddings shape: {historical_embeddings.shape}")
        
        # Load historical actions - prefer the new ones if available
        if os.path.exists('historical_actions_new.json'):
            historical_actions = json.load(open('historical_actions_new.json', 'r'))
            print(f"Loaded {len(historical_actions)} new historical actions.")
        else:
            historical_actions = json.load(open('historical_actions.json', 'r'))
            print(f"Loaded {len(historical_actions)} original historical actions.")
        
        # Ensure we have enough historical actions
        if len(historical_actions) < historical_embeddings.shape[0]:
            print(f"Warning: Not enough historical actions ({len(historical_actions)}) for embeddings ({historical_embeddings.shape[0]})")
            print("Creating additional dummy actions...")
            
            # Create additional actions
            additional_actions = []
            for i in range(historical_embeddings.shape[0] - len(historical_actions)):
                action = {
                    "device": "smart_light",
                    "capability": "power",
                    "state": "on" if i % 2 == 0 else "off"
                }
                additional_actions.append(action)
            
            # Extend the historical actions
            historical_actions.extend(additional_actions)
            
            # Save the updated historical actions
            with open('historical_actions_new.json', 'w') as f:
                json.dump(historical_actions, f)
            print(f"Created and saved {len(historical_actions)} total historical actions.")

    # Interactive mode
    if args.interactive:
        print("\n=== LEM Interactive Demo ===")
        print("This demo allows you to simulate smart home events and get predictions.")
        
        # Get available options from encoders
        available_devices = list(device_encoder.classes_)
        
        print("\nAvailable devices:", available_devices)
        print("\nNote: For each device, you'll see relevant capabilities and states.")
        
        while True:
            print("\nOptions:")
            print("1. Use last 5 events from dataset")
            print("2. Create a custom event")
            print("3. Show available devices")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                recent_events = dataset[0][-5:]  # Last 5 events as context
                print("\nUsing these recent events as context:")
                for i, event in enumerate(recent_events):
                    print(f"{i+1}. Device: {event['device']}, Capability: {event['capability']}, State: {event['attributes']['state']}")
            
            elif choice == '2':
                # Create a custom event
                recent_events = []
                num_events = int(input("\nHow many events do you want to create (1-5)? "))
                num_events = min(max(1, num_events), 5)  # Ensure between 1 and 5
                
                for i in range(num_events):
                    print(f"\nEvent {i+1}:")
                    
                    # Show available devices
                    print("Available devices:", available_devices)
                    device = input("Device: ")
                    
                    # Show device-specific capabilities
                    device_capabilities = get_device_capabilities(device)
                    print(f"Available capabilities for {device}:", device_capabilities)
                    capability = input("Capability: ")
                    
                    # Show capability-specific states
                    capability_states = get_capability_states(capability)
                    print(f"Available states for {capability}:", capability_states)
                    state = input("State: ")
                    
                    event = {
                        "device": device,
                        "capability": capability,
                        "attributes": {"state": state},
                        "timestamp": datetime.now().isoformat()
                    }
                    recent_events.append(event)
                
                print("\nCreated these events as context:")
                for i, event in enumerate(recent_events):
                    print(f"{i+1}. Device: {event['device']}, Capability: {event['capability']}, State: {event['attributes']['state']}")
            
            elif choice == '3':
                print("\nAvailable devices:", available_devices)
                
                # Show capabilities and states for each device
                for device in available_devices:
                    device_str = device
                    if hasattr(device, 'item'):
                        device_str = device.item()
                    
                    print(f"\nDevice: {device_str}")
                    device_capabilities = get_device_capabilities(device)
                    print(f"  Capabilities: {device_capabilities}")
                    
                    for capability in device_capabilities:
                        capability_states = get_capability_states(capability)
                        print(f"    {capability} states: {capability_states}")
                
                continue
            
            elif choice == '4':
                print("Exiting demo. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Process the events and make prediction
            if len(historical_embeddings) > 0 and historical_embeddings.numel() > 0:
                event_embedding = get_event_embedding(recent_events)
                
                print("\nOriginal shapes - Event embedding:", event_embedding.shape, "Historical embeddings:", historical_embeddings.shape)
                
                # Fix dimensionality issue - flatten tensors if needed
                if len(event_embedding.shape) == 3:  # [1, seq_len, features]
                    event_embedding = event_embedding.squeeze(0)  # Remove batch dimension
                
                if len(event_embedding.shape) == 2:  # [seq_len, features]
                    # If it's a sequence, take the mean across the sequence dimension
                    event_embedding = torch.mean(event_embedding, dim=0)  # Now [features]
                
                # Handle historical embeddings
                if len(historical_embeddings.shape) == 3:  # [batch, seq_len, features]
                    # Take mean across sequence dimension for each batch
                    historical_embeddings = torch.mean(historical_embeddings, dim=1)  # Now [batch, features]
                
                print("After reshaping - Event embedding:", event_embedding.shape, "Historical embeddings:", historical_embeddings.shape)
                
                # Check for dimension mismatch and handle it
                if event_embedding.shape[-1] != historical_embeddings.shape[-1]:
                    print(f"Dimension mismatch detected: {event_embedding.shape[-1]} vs {historical_embeddings.shape[-1]}")
                    
                    # If model is not None, regenerate historical embeddings using the same model
                    if model is not None:
                        print("Regenerating historical embeddings using the loaded model...")
                        # Load historical data
                        with open('event_dataset.json', 'r') as f:
                            historical_data = json.load(f)[:10]  # Use first 10 sequences
                        
                        # Flatten the sequences
                        historical_events = [event for sequence in historical_data for event in sequence]
                        
                        # Process events in batches to reduce warnings
                        all_embeddings = []
                        batch_size = 5
                        
                        # Disable print temporarily to avoid too many warnings
                        import sys
                        original_stdout = sys.stdout
                        sys.stdout = open(os.devnull, 'w')
                        
                        try:
                            for i in range(0, len(historical_events), batch_size):
                                if i + batch_size <= len(historical_events):
                                    event_group = historical_events[i:i+batch_size]
                                    all_embeddings.append(get_event_embedding(event_group))
                        finally:
                            # Restore print
                            sys.stdout.close()
                            sys.stdout = original_stdout
                        
                        print(f"Processed {len(all_embeddings)} historical event groups")
                        
                        # Stack the embeddings
                        historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, event_embedding.shape[-1]))
                        
                        # Save the new historical embeddings for future use
                        torch.save(historical_embeddings, 'historical_event_embeddings_new.pt')
                        print("Saved new historical embeddings for future use.")
                        
                        # Reshape if needed
                        if len(historical_embeddings.shape) == 3:
                            historical_embeddings = torch.mean(historical_embeddings, dim=1)
                        
                        print("New historical embeddings shape:", historical_embeddings.shape)
                        
                        # Create new historical actions to match the number of embeddings
                        historical_actions = []
                        for i in range(historical_embeddings.shape[0]):
                            # Create a dummy action
                            action = {
                                "device": "smart_light",
                                "capability": "power",
                                "state": "on" if i % 2 == 0 else "off"
                            }
                            historical_actions.append(action)
                        
                        # Save the new historical actions
                        with open('historical_actions_new.json', 'w') as f:
                            json.dump(historical_actions, f)
                        print(f"Created and saved {len(historical_actions)} new historical actions.")
                    # If model is None, use feature vectors directly
                    else:
                        print("Regenerating historical embeddings with the same dimensions...")
                        # Load historical data
                        with open('event_dataset.json', 'r') as f:
                            historical_data = json.load(f)[:10]  # Use first 10 sequences
                        
                        # Flatten the sequences
                        historical_events = [event for sequence in historical_data for event in sequence]
                        
                        # Process events in batches to reduce warnings
                        all_embeddings = []
                        batch_size = 5
                        
                        # Disable print temporarily to avoid too many warnings
                        import sys
                        original_stdout = sys.stdout
                        sys.stdout = open(os.devnull, 'w')
                        
                        try:
                            for i in range(0, len(historical_events), batch_size):
                                if i + batch_size <= len(historical_events):
                                    event_group = historical_events[i:i+batch_size]
                                    all_embeddings.append(get_event_embedding(event_group))
                        finally:
                            # Restore print
                            sys.stdout.close()
                            sys.stdout = original_stdout
                        
                        print(f"Processed {len(all_embeddings)} historical event groups")
                        
                        # Stack the embeddings
                        historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, event_embedding.shape[-1]))
                        
                        # Reshape if needed
                        if len(historical_embeddings.shape) == 3:
                            historical_embeddings = torch.mean(historical_embeddings, dim=1)
                        
                        print("New historical embeddings shape:", historical_embeddings.shape)
                        
                        # Create new historical actions to match the number of embeddings
                        historical_actions = []
                        for i in range(historical_embeddings.shape[0]):
                            # Create a dummy action
                            action = {
                                "device": "smart_light",
                                "capability": "power",
                                "state": "on" if i % 2 == 0 else "off"
                            }
                            historical_actions.append(action)
                        
                        # Save the new historical actions
                        with open('historical_actions_new.json', 'w') as f:
                            json.dump(historical_actions, f)
                        print(f"Created and saved {len(historical_actions)} new historical actions.")
                
                # Ensure both tensors are 2D for proper comparison
                event_embedding_expanded = event_embedding.unsqueeze(0)  # [1, features]
                print("Final comparison shapes:", event_embedding_expanded.shape, historical_embeddings.shape)
                
                # Check dimensions again before calculating similarity
                if event_embedding_expanded.shape[-1] == historical_embeddings.shape[-1]:
                    # Ensure we have enough historical actions
                    if len(historical_actions) < historical_embeddings.shape[0]:
                        print(f"Warning: Not enough historical actions ({len(historical_actions)}) for embeddings ({historical_embeddings.shape[0]})")
                        print("Creating additional dummy actions...")
                        
                        # Create additional actions
                        additional_actions = []
                        for i in range(historical_embeddings.shape[0] - len(historical_actions)):
                            action = {
                                "device": "smart_light",
                                "capability": "power",
                                "state": "on" if i % 2 == 0 else "off"
                            }
                            additional_actions.append(action)
                        
                        # Extend the historical actions
                        historical_actions.extend(additional_actions)
                        
                        # Save the updated historical actions
                        with open('historical_actions_new.json', 'w') as f:
                            json.dump(historical_actions, f)
                        print(f"Created and saved {len(historical_actions)} total historical actions.")
                    
                    try:
                        similarities = nn.CosineSimilarity(dim=1)(
                            event_embedding_expanded, 
                            historical_embeddings
                        )
                        
                        most_similar_idx = torch.argmax(similarities).item()
                        
                        # Ensure the index is within bounds
                        if most_similar_idx >= len(historical_actions):
                            print(f"Warning: most_similar_idx ({most_similar_idx}) is out of bounds for historical_actions (length {len(historical_actions)})")
                            most_similar_idx = most_similar_idx % len(historical_actions)
                            print(f"Using index {most_similar_idx} instead")
                        
                        suggested_action = historical_actions[most_similar_idx]
                        
                        print("\n=== Prediction Results ===")
                        print("Suggested Action:", suggested_action)
                        print(f"Recommendation: Set {suggested_action['device']} {suggested_action['capability']} to {suggested_action['state']}")
                    except Exception as e:
                        print(f"Error during similarity calculation: {str(e)}")
                        print("Shapes - Event:", event_embedding_expanded.shape, "Historical:", historical_embeddings.shape)
                        print("Historical actions length:", len(historical_actions))
                else:
                    print("\nError: Dimension mismatch still exists after correction attempt.")
                    print("Cannot calculate similarity between tensors of different dimensions.")
            else:
                print("No historical data available for comparison.")
            
            input("\nPress Enter to continue...")
    
    # Command line mode with specific device/state
    elif args.device:
        # Create a custom event with the specified device and state
        custom_event = {
            "device": args.device,
            "capability": "switch" if not args.state else "state",
            "attributes": {"state": args.state if args.state else "on"},
            "timestamp": datetime.now().isoformat()
        }
        
        # Use the last 4 events from the dataset and add the custom event
        recent_events = dataset[0][-4:] + [custom_event]
        
        print("\nUsing context with your custom event:")
        for i, event in enumerate(recent_events):
            print(f"{i+1}. Device: {event['device']}, Capability: {event['capability']}, State: {event['attributes']['state']}")
        
        # Process the events and make prediction
        if len(historical_embeddings) > 0 and historical_embeddings.numel() > 0:
            event_embedding = get_event_embedding(recent_events)
            
            print("\nOriginal shapes - Event embedding:", event_embedding.shape, "Historical embeddings:", historical_embeddings.shape)
            
            # Fix dimensionality issue - flatten tensors if needed
            if len(event_embedding.shape) == 3:  # [1, seq_len, features]
                event_embedding = event_embedding.squeeze(0)  # Remove batch dimension
            
            if len(event_embedding.shape) == 2:  # [seq_len, features]
                # If it's a sequence, take the mean across the sequence dimension
                event_embedding = torch.mean(event_embedding, dim=0)  # Now [features]
            
            # Handle historical embeddings
            if len(historical_embeddings.shape) == 3:  # [batch, seq_len, features]
                # Take mean across sequence dimension for each batch
                historical_embeddings = torch.mean(historical_embeddings, dim=1)  # Now [batch, features]
            
            print("After reshaping - Event embedding:", event_embedding.shape, "Historical embeddings:", historical_embeddings.shape)
            
            # Check for dimension mismatch and handle it
            if event_embedding.shape[-1] != historical_embeddings.shape[-1]:
                print(f"Dimension mismatch detected: {event_embedding.shape[-1]} vs {historical_embeddings.shape[-1]}")
                
                # If model is not None, regenerate historical embeddings using the same model
                if model is not None:
                    print("Regenerating historical embeddings using the loaded model...")
                    # Load historical data
                    with open('event_dataset.json', 'r') as f:
                        historical_data = json.load(f)[:10]  # Use first 10 sequences
                    
                    # Flatten the sequences
                    historical_events = [event for sequence in historical_data for event in sequence]
                    
                    # Process events in batches to reduce warnings
                    all_embeddings = []
                    batch_size = 5
                    
                    # Disable print temporarily to avoid too many warnings
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    try:
                        for i in range(0, len(historical_events), batch_size):
                            if i + batch_size <= len(historical_events):
                                event_group = historical_events[i:i+batch_size]
                                all_embeddings.append(get_event_embedding(event_group))
                    finally:
                        # Restore print
                        sys.stdout.close()
                        sys.stdout = original_stdout
                    
                    print(f"Processed {len(all_embeddings)} historical event groups")
                    
                    # Stack the embeddings
                    historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, event_embedding.shape[-1]))
                    
                    # Save the new historical embeddings for future use
                    torch.save(historical_embeddings, 'historical_event_embeddings_new.pt')
                    print("Saved new historical embeddings for future use.")
                    
                    # Reshape if needed
                    if len(historical_embeddings.shape) == 3:
                        historical_embeddings = torch.mean(historical_embeddings, dim=1)
                    
                    print("New historical embeddings shape:", historical_embeddings.shape)
                    
                    # Create new historical actions to match the number of embeddings
                    historical_actions = []
                    for i in range(historical_embeddings.shape[0]):
                        # Create a dummy action
                        action = {
                            "device": "smart_light",
                            "capability": "power",
                            "state": "on" if i % 2 == 0 else "off"
                        }
                        historical_actions.append(action)
                    
                    # Save the new historical actions
                    with open('historical_actions_new.json', 'w') as f:
                        json.dump(historical_actions, f)
                    print(f"Created and saved {len(historical_actions)} new historical actions.")
                # If model is None, use feature vectors directly
                else:
                    print("Regenerating historical embeddings with the same dimensions...")
                    # Load historical data
                    with open('event_dataset.json', 'r') as f:
                        historical_data = json.load(f)[:10]  # Use first 10 sequences
                    
                    # Flatten the sequences
                    historical_events = [event for sequence in historical_data for event in sequence]
                    
                    # Process events in batches to reduce warnings
                    all_embeddings = []
                    batch_size = 5
                    
                    # Disable print temporarily to avoid too many warnings
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    try:
                        for i in range(0, len(historical_events), batch_size):
                            if i + batch_size <= len(historical_events):
                                event_group = historical_events[i:i+batch_size]
                                all_embeddings.append(get_event_embedding(event_group))
                    finally:
                        # Restore print
                        sys.stdout.close()
                        sys.stdout = original_stdout
                    
                    print(f"Processed {len(all_embeddings)} historical event groups")
                    
                    # Stack the embeddings
                    historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, event_embedding.shape[-1]))
                    
                    # Reshape if needed
                    if len(historical_embeddings.shape) == 3:
                        historical_embeddings = torch.mean(historical_embeddings, dim=1)
                    
                    print("New historical embeddings shape:", historical_embeddings.shape)
                    
                    # Create new historical actions to match the number of embeddings
                    historical_actions = []
                    for i in range(historical_embeddings.shape[0]):
                        # Create a dummy action
                        action = {
                            "device": "smart_light",
                            "capability": "power",
                            "state": "on" if i % 2 == 0 else "off"
                        }
                        historical_actions.append(action)
                    
                    # Save the new historical actions
                    with open('historical_actions_new.json', 'w') as f:
                        json.dump(historical_actions, f)
                    print(f"Created and saved {len(historical_actions)} new historical actions.")
                
                # Ensure both tensors are 2D for proper comparison
                event_embedding_expanded = event_embedding.unsqueeze(0)
                
                # Check dimensions again before calculating similarity
                if event_embedding_expanded.shape[-1] == historical_embeddings.shape[-1]:
                    similarities = nn.CosineSimilarity(dim=1)(
                        event_embedding_expanded, 
                        historical_embeddings
                    )
                    
                    most_similar_idx = torch.argmax(similarities).item()
                    
                    # Ensure the index is within bounds
                    if most_similar_idx >= len(historical_actions):
                        print(f"Warning: most_similar_idx ({most_similar_idx}) is out of bounds for historical_actions (length {len(historical_actions)})")
                        most_similar_idx = most_similar_idx % len(historical_actions)
                        print(f"Using index {most_similar_idx} instead")
                    
                    suggested_action = historical_actions[most_similar_idx]
                    
                    print("\n=== Prediction Results ===")
                    print("Suggested Action:", suggested_action)
                    print(f"Recommendation: Set {suggested_action['device']} {suggested_action['capability']} to {suggested_action['state']}")
                else:
                    print("\nError: Dimension mismatch still exists after correction attempt.")
                    print("Cannot calculate similarity between tensors of different dimensions.")
            else:
                print("No historical data available for comparison.")
        
        else:
            print("No historical data available for comparison.")
    
    # Default mode - use dataset events
    else:
        recent_events = dataset[0][-5:]  # Last 5 events as context
        
        print("\nUsing these recent events as context:")
        for i, event in enumerate(recent_events):
            print(f"{i+1}. Device: {event['device']}, Capability: {event['capability']}, State: {event['attributes']['state']}")
        
        # Suggest action based on embedding similarity
        if len(historical_embeddings) > 0 and historical_embeddings.numel() > 0:
            event_embedding = get_event_embedding(recent_events)
            
            print("\nOriginal shapes - Event embedding:", event_embedding.shape, "Historical embeddings:", historical_embeddings.shape)
            
            # Fix dimensionality issue - flatten tensors if needed
            if len(event_embedding.shape) == 3:  # [1, seq_len, features]
                event_embedding = event_embedding.squeeze(0)  # Remove batch dimension
            
            if len(event_embedding.shape) == 2:  # [seq_len, features]
                # If it's a sequence, take the mean across the sequence dimension
                event_embedding = torch.mean(event_embedding, dim=0)  # Now [features]
            
            # Handle historical embeddings
            if len(historical_embeddings.shape) == 3:  # [batch, seq_len, features]
                # Take mean across sequence dimension for each batch
                historical_embeddings = torch.mean(historical_embeddings, dim=1)  # Now [batch, features]
            
            print("After reshaping - Event embedding:", event_embedding.shape, "Historical embeddings:", historical_embeddings.shape)
            
            # Check for dimension mismatch and handle it
            if event_embedding.shape[-1] != historical_embeddings.shape[-1]:
                print(f"Dimension mismatch detected: {event_embedding.shape[-1]} vs {historical_embeddings.shape[-1]}")
                
                # If model is not None, regenerate historical embeddings using the same model
                if model is not None:
                    print("Regenerating historical embeddings using the loaded model...")
                    # Load historical data
                    with open('event_dataset.json', 'r') as f:
                        historical_data = json.load(f)[:10]  # Use first 10 sequences
                    
                    # Flatten the sequences
                    historical_events = [event for sequence in historical_data for event in sequence]
                    
                    # Process events in batches to reduce warnings
                    all_embeddings = []
                    batch_size = 5
                    
                    # Disable print temporarily to avoid too many warnings
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    try:
                        for i in range(0, len(historical_events), batch_size):
                            if i + batch_size <= len(historical_events):
                                event_group = historical_events[i:i+batch_size]
                                all_embeddings.append(get_event_embedding(event_group))
                    finally:
                        # Restore print
                        sys.stdout.close()
                        sys.stdout = original_stdout
                    
                    print(f"Processed {len(all_embeddings)} historical event groups")
                    
                    # Stack the embeddings
                    historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, event_embedding.shape[-1]))
                    
                    # Save the new historical embeddings for future use
                    torch.save(historical_embeddings, 'historical_event_embeddings_new.pt')
                    print("Saved new historical embeddings for future use.")
                    
                    # Reshape if needed
                    if len(historical_embeddings.shape) == 3:
                        historical_embeddings = torch.mean(historical_embeddings, dim=1)
                    
                    print("New historical embeddings shape:", historical_embeddings.shape)
                    
                    # Create new historical actions to match the number of embeddings
                    historical_actions = []
                    for i in range(historical_embeddings.shape[0]):
                        # Create a dummy action
                        action = {
                            "device": "smart_light",
                            "capability": "power",
                            "state": "on" if i % 2 == 0 else "off"
                        }
                        historical_actions.append(action)
                    
                    # Save the new historical actions
                    with open('historical_actions_new.json', 'w') as f:
                        json.dump(historical_actions, f)
                    print(f"Created and saved {len(historical_actions)} new historical actions.")
                # If model is None, use feature vectors directly
                else:
                    print("Regenerating historical embeddings with the same dimensions...")
                    # Load historical data
                    with open('event_dataset.json', 'r') as f:
                        historical_data = json.load(f)[:10]  # Use first 10 sequences
                    
                    # Flatten the sequences
                    historical_events = [event for sequence in historical_data for event in sequence]
                    
                    # Process events in batches to reduce warnings
                    all_embeddings = []
                    batch_size = 5
                    
                    # Disable print temporarily to avoid too many warnings
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    try:
                        for i in range(0, len(historical_events), batch_size):
                            if i + batch_size <= len(historical_events):
                                event_group = historical_events[i:i+batch_size]
                                all_embeddings.append(get_event_embedding(event_group))
                    finally:
                        # Restore print
                        sys.stdout.close()
                        sys.stdout = original_stdout
                    
                    print(f"Processed {len(all_embeddings)} historical event groups")
                    
                    # Stack the embeddings
                    historical_embeddings = torch.stack(all_embeddings) if all_embeddings else torch.zeros((1, event_embedding.shape[-1]))
                    
                    # Reshape if needed
                    if len(historical_embeddings.shape) == 3:
                        historical_embeddings = torch.mean(historical_embeddings, dim=1)
                    
                    print("New historical embeddings shape:", historical_embeddings.shape)
                    
                    # Create new historical actions to match the number of embeddings
                    historical_actions = []
                    for i in range(historical_embeddings.shape[0]):
                        # Create a dummy action
                        action = {
                            "device": "smart_light",
                            "capability": "power",
                            "state": "on" if i % 2 == 0 else "off"
                        }
                        historical_actions.append(action)
                    
                    # Save the new historical actions
                    with open('historical_actions_new.json', 'w') as f:
                        json.dump(historical_actions, f)
                    print(f"Created and saved {len(historical_actions)} new historical actions.")
                
                # Ensure both tensors are 2D for proper comparison
                event_embedding_expanded = event_embedding.unsqueeze(0)
                
                # Check dimensions again before calculating similarity
                if event_embedding_expanded.shape[-1] == historical_embeddings.shape[-1]:
                    similarities = nn.CosineSimilarity(dim=1)(
                        event_embedding_expanded, 
                        historical_embeddings
                    )
                    
                    most_similar_idx = torch.argmax(similarities).item()
                    
                    # Ensure the index is within bounds
                    if most_similar_idx >= len(historical_actions):
                        print(f"Warning: most_similar_idx ({most_similar_idx}) is out of bounds for historical_actions (length {len(historical_actions)})")
                        most_similar_idx = most_similar_idx % len(historical_actions)
                        print(f"Using index {most_similar_idx} instead")
                    
                    suggested_action = historical_actions[most_similar_idx]
                    
                    print("\n=== Prediction Results ===")
                    print("Suggested Action:", suggested_action)
                    print(f"Recommendation: Set {suggested_action['device']} {suggested_action['capability']} to {suggested_action['state']}")
                else:
                    print("\nError: Dimension mismatch still exists after correction attempt.")
                    print("Cannot calculate similarity between tensors of different dimensions.")
            else:
                print("No historical data available for comparison.")
        else:
            print("No historical data available for comparison.")
