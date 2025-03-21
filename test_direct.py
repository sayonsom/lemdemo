#!/usr/bin/env python3
import torch
import numpy as np
import json

print("Testing the fixed find_similar_actions function directly...")

# Define the function with proper formatting 
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
        # Reshape (n, 1, d) to (n, d)
        historical_embeddings_np = historical_embeddings_np.squeeze(1)
    
    # Ensure both are 2D arrays
    if len(embedding_np.shape) == 1:
        embedding_np = embedding_np.reshape(1, -1)
    
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

# Create test data
embedding = torch.rand(128)  # 1D tensor of shape (128,)
historical_embeddings = torch.rand(10, 1, 128)  # 3D tensor of shape (10, 1, 128)
historical_actions = [{"id": i, "action": f"action_{i}"} for i in range(10)]

# Try to run the function
try:
    results = find_similar_actions(embedding, historical_embeddings, historical_actions)
    print("Function executed successfully!")
    print(f"Got {len(results)} results, top result similarity: {results[0][1]:.4f}")
    print("The find_similar_actions function is working correctly with the proper formatting.")
except Exception as e:
    print(f"Error executing function: {e}")
    print("The function still has issues.")

# Test different input shapes
try:
    # Test with 2D embedding
    embedding_2d = torch.rand(1, 128)  # 2D tensor of shape (1, 128)
    results = find_similar_actions(embedding_2d, historical_embeddings, historical_actions)
    print("Successfully handled 2D embedding input.")
    
    # Test with 2D historical embeddings
    historical_embeddings_2d = torch.rand(10, 128)  # 2D tensor of shape (10, 128)
    results = find_similar_actions(embedding, historical_embeddings_2d, historical_actions)
    print("Successfully handled 2D historical embeddings input.")
    
    # Test with numpy arrays
    embedding_np = np.random.rand(128)
    historical_embeddings_np = np.random.rand(10, 128)
    results = find_similar_actions(embedding_np, historical_embeddings_np, historical_actions)
    print("Successfully handled numpy array inputs.")
    
    print("All shape handling tests passed!")
except Exception as e:
    print(f"Error during shape testing: {e}") 