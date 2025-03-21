import json
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime

print("Testing the rewritten find_similar_actions function...")

# Define the Large Event Model (LEM)
class LargeEventModel(nn.Module):
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

# Function to find similar actions based on embeddings
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
        # Handle extra dimension - squeeze the middle dimension if it's 1
        if len(historical_embeddings.shape) == 3 and historical_embeddings.shape[1] == 1:
            historical_embeddings = historical_embeddings.squeeze(1)
        historical_embeddings_np = historical_embeddings.detach().cpu().numpy()
    else:
        historical_embeddings_np = historical_embeddings
    
    # Ensure both are 2D arrays
    if len(embedding_np.shape) == 1:
        embedding_np = embedding_np.reshape(1, -1)
    
    if len(historical_embeddings_np.shape) == 1:
        historical_embeddings_np = historical_embeddings_np.reshape(1, -1)
    
    # Compute dot product
    dot_product = np.dot(embedding_np, historical_embeddings_np.T)
    
    # Compute norms
    embedding_norm = np.linalg.norm(embedding_np, axis=1)
    historical_norm = np.linalg.norm(historical_embeddings_np, axis=1)
    
    # Compute similarity
    similarity = dot_product / (embedding_norm.reshape(-1, 1) * historical_norm)
    similarity = similarity.flatten()  # Ensure it's a flat array
    
    # Get top indices
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    
    # Get corresponding actions and similarities
    top_actions = [historical_actions[i] for i in top_indices]
    top_similarities = [similarity[i] for i in top_indices]
    
    return list(zip(top_actions, top_similarities))

try:
    # Load the model
    model = LargeEventModel(input_dim=7, embed_dim=128)
    model.load_state_dict(torch.load('transformer_model.pt'))
    model.eval()
    print("Loaded Large Event Model")
    
    # Load saved embeddings and actions
    historical_embeddings = torch.load('transformer_embeddings.pt')
    with open('transformer_actions.json', 'r') as f:
        historical_actions = json.load(f)
    print(f"Loaded historical embeddings shape: {historical_embeddings.shape}, actions: {len(historical_actions)}")
    
    # Create a test embedding
    test_embedding = torch.rand(1, 128)
    print(f"Created test embedding with shape: {test_embedding.shape}")
    
    # Print the shapes after squeezing
    if len(historical_embeddings.shape) == 3 and historical_embeddings.shape[1] == 1:
        historical_embeddings_squeezed = historical_embeddings.squeeze(1)
        print(f"Squeezed historical embeddings shape: {historical_embeddings_squeezed.shape}")
    
    # Find similar actions
    similar_actions = find_similar_actions(test_embedding, historical_embeddings, historical_actions, top_n=3)
    print(f"Found {len(similar_actions)} similar actions")
    
    # Print the top action
    action, similarity = similar_actions[0]
    print(f"\nTop recommendation (similarity: {similarity:.4f}):")
    for key, value in action.items():
        print(f"  {key}: {value}")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error during test: {str(e)}") 