import json
import torch
import numpy as np
from datetime import datetime

# Load the data
historical_embeddings = torch.load('transformer_embeddings.pt')
with open('transformer_actions.json', 'r') as f:
    historical_actions = json.load(f)

print(f"Historical embeddings shape: {historical_embeddings.shape}")
print(f"Number of historical actions: {len(historical_actions)}")

# Create a random test embedding
test_embedding = torch.randn(1, 128)

# Simplified find_similar_actions function
def find_similar_actions(embedding, historical_embeddings, historical_actions, top_n=3):
    # Convert tensors to numpy arrays
    embedding_np = embedding.detach().cpu().numpy()
    
    # Handle the extra dimension in historical_embeddings
    if len(historical_embeddings.shape) == 3 and historical_embeddings.shape[1] == 1:
        historical_embeddings = historical_embeddings.squeeze(1)
    historical_embeddings_np = historical_embeddings.detach().cpu().numpy()
    
    # Compute cosine similarity
    dot_product = np.dot(embedding_np, historical_embeddings_np.T)
    embedding_norm = np.linalg.norm(embedding_np, axis=1)
    historical_norm = np.linalg.norm(historical_embeddings_np, axis=1)
    
    similarity = dot_product / (embedding_norm.reshape(-1, 1) * historical_norm)
    similarity = similarity.flatten()
    
    # Get top indices and corresponding actions
    top_indices = np.argsort(similarity)[-top_n:][::-1]
    top_actions = [historical_actions[i] for i in top_indices]
    top_similarities = [similarity[i] for i in top_indices]
    
    return list(zip(top_actions, top_similarities))

# Test the function
similar_actions = find_similar_actions(test_embedding, historical_embeddings, historical_actions, top_n=3)

# Display results
print(f"\nFound {len(similar_actions)} similar actions")

for i, (action, similarity) in enumerate(similar_actions):
    print(f"\nRecommendation {i+1} (similarity: {similarity:.4f}):")
    for key, value in action.items():
        print(f"  {key}: {value}")

print("\nTest completed successfully!") 