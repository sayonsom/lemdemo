import torch
import numpy as np
import json

print("Testing the rewritten find_similar_actions function...")

# Create test data
embedding = torch.randn(128)  # Random embedding vector
historical_embeddings = torch.randn(10, 128)  # 10 random historical embeddings

# Load actual actions for testing
try:
    with open('transformer_actions.json', 'r') as f:
        historical_actions = json.load(f)
    print(f"Loaded {len(historical_actions)} historical actions")
except Exception as e:
    print(f"Error loading actions: {e}")
    # Create dummy actions if file not found
    historical_actions = [{"device": f"device_{i}", "action": f"action_{i}"} for i in range(10)]

# Rewritten find_similar_actions function
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
    
    # Ensure both are 2D arrays
    if len(embedding_np.shape) == 1:
        embedding_np = embedding_np.reshape(1, -1)
    
    if len(historical_embeddings_np.shape) == 1:
        historical_embeddings_np = historical_embeddings_np.reshape(1, -1)
    
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

# Test with different shapes of embeddings
print("\nTest 1: Standard embedding (1D tensor)")
result1 = find_similar_actions(embedding, historical_embeddings, historical_actions)
print(f"Result1 has {len(result1)} recommendations")

print("\nTest 2: Batched embedding (2D tensor with batch_size=1)")
batched_embedding = embedding.unsqueeze(0)  # Add batch dimension
result2 = find_similar_actions(batched_embedding, historical_embeddings, historical_actions)
print(f"Result2 has {len(result2)} recommendations")

print("\nTest 3: Using numpy arrays instead of tensors")
embedding_np = embedding.detach().cpu().numpy()
historical_embeddings_np = historical_embeddings.detach().cpu().numpy()
result3 = find_similar_actions(embedding_np, historical_embeddings_np, historical_actions)
print(f"Result3 has {len(result3)} recommendations")

print("\nAll tests completed successfully!") 