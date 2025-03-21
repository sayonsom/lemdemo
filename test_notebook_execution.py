import json
import torch
import numpy as np
from datetime import datetime
import joblib
import time

print("Testing notebook functionality with real data...")

try:
    # Load encoders
    device_encoder = joblib.load('device_encoder.pkl')
    capability_encoder = joblib.load('capability_encoder.pkl')
    state_encoder = joblib.load('state_encoder.pkl')
    print("Successfully loaded encoders")
except Exception as e:
    print(f"Error loading encoders: {e}")
    exit(1)

# Define the LargeEventModel class (similar to notebook)
class LargeEventModel(torch.nn.Module):
    def __init__(self, input_dim=7, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, embed_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_proj = torch.nn.Linear(embed_dim, embed_dim)
        
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

# Preprocess event function (from notebook)
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

# The rewritten find_similar_actions function
def find_similar_actions(embedding, historical_embeddings, historical_actions, top_n=3):
    """Find the most similar actions based on embedding similarity"""
    # Convert to numpy for simplicity
    if isinstance(embedding, torch.Tensor):
        # Remove batch dimension if present
        if len(embedding.shape) > 1 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        # Convert to numpy array
        embedding_np = embedding.detach().cpu().numpy()
    else:
        embedding_np = embedding
    
    if isinstance(historical_embeddings, torch.Tensor):
        historical_embeddings_np = historical_embeddings.detach().cpu().numpy()
    else:
        historical_embeddings_np = historical_embeddings
    
    # Handle 3D historical embeddings
    if len(historical_embeddings_np.shape) == 3:
        print(f"Original historical embeddings shape: {historical_embeddings_np.shape}")
        # Reshape (n, 1, d) to (n, d)
        historical_embeddings_np = historical_embeddings_np.squeeze(1)
        print(f"Reshaped historical embeddings: {historical_embeddings_np.shape}")
    
    # Ensure both are 2D arrays
    if len(embedding_np.shape) == 1:
        embedding_np = embedding_np.reshape(1, -1)
        print(f"Reshaped embedding: {embedding_np.shape}")
    
    # Print shapes for debugging
    print(f"Embedding shape: {embedding_np.shape}")
    print(f"Historical embeddings shape: {historical_embeddings_np.shape}")
    
    # Compute dot product
    dot_product = np.dot(embedding_np, historical_embeddings_np.T)
    print(f"Dot product shape: {dot_product.shape}")
    
    # Compute norms
    embedding_norm = np.linalg.norm(embedding_np, axis=1)
    historical_norm = np.linalg.norm(historical_embeddings_np, axis=1)
    print(f"Embedding norm shape: {embedding_norm.shape}")
    print(f"Historical norm shape: {historical_norm.shape}")
    
    # Compute similarity
    similarity = dot_product / (embedding_norm.reshape(-1, 1) * historical_norm)
    similarity = similarity.flatten()  # Ensure it's a flat array
    print(f"Similarity shape: {similarity.shape}")
    
    # Get top indices
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    print(f"Top indices: {top_indices}")
    
    # Get corresponding actions and similarities
    top_actions = [historical_actions[i] for i in top_indices]
    top_similarities = [similarity[i] for i in top_indices]
    
    return list(zip(top_actions, top_similarities))

try:
    # Load the model, historical embeddings, and actions
    print("\nLoading model and historical data...")
    model = LargeEventModel(input_dim=7, embed_dim=128)
    model.load_state_dict(torch.load('transformer_model.pt'))
    model.eval()
    print("Model loaded successfully")
    
    historical_embeddings = torch.load('transformer_embeddings.pt')
    print(f"Historical embeddings shape: {historical_embeddings.shape}")
    
    with open('transformer_actions.json', 'r') as f:
        historical_actions = json.load(f)
    print(f"Loaded historical embeddings (shape: {historical_embeddings.shape}) and {len(historical_actions)} actions")
    
    # Load event dataset (for testing)
    with open('event_dataset.json', 'r') as f:
        dataset = json.load(f)
    print(f"Loaded event dataset with {len(dataset)} sequences")
    
    # Get a sample sequence
    recent_events = dataset[0][-5:]  # Last 5 events from first sequence
    print(f"Using a sample sequence with {len(recent_events)} events")
    
    # Get embedding
    embedding = get_event_embedding(recent_events, model)
    print(f"Generated embedding for sample sequence, shape: {embedding.shape}")
    
    # Test the find_similar_actions function (this is what would fail previously)
    print("\nTesting find_similar_actions function...")
    similar_actions = find_similar_actions(embedding, historical_embeddings, historical_actions, top_n=3)
    
    # Display results
    print("\nRecommended actions found successfully!")
    for i, (action, similarity) in enumerate(similar_actions):
        print(f"\nRecommendation {i+1} (similarity: {similarity:.4f}):")
        for key, value in action.items():
            print(f"  {key}: {value}")
    
    # Test with custom events
    print("\nTesting with custom events...")
    custom_events = [
        {
            "device": "smart_light",
            "capability": "power",
            "attributes": {"state": "ON"},
            "timestamp": datetime.now().isoformat()
        },
        {
            "device": "smart_thermostat",
            "capability": "temperature_control", 
            "attributes": {"state": "WARM"},
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    custom_embedding = get_event_embedding(custom_events, model)
    print(f"Custom embedding shape: {custom_embedding.shape}")
    similar_actions = find_similar_actions(custom_embedding, historical_embeddings, historical_actions, top_n=3)
    
    print("\nRecommendations for custom events:")
    for i, (action, similarity) in enumerate(similar_actions):
        print(f"\nRecommendation {i+1} (similarity: {similarity:.4f}):")
        for key, value in action.items():
            print(f"  {key}: {value}")
    
    print("\nAll tests completed successfully!")

except Exception as e:
    import traceback
    print(f"\nError during execution: {e}")
    traceback.print_exc() 