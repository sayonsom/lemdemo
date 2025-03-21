import json
import torch
import numpy as np

# Load data
historical_embeddings = torch.load("transformer_embeddings.pt")
with open("transformer_actions.json", "r") as f:
    historical_actions = json.load(f)

print(f"Historical embeddings shape: {historical_embeddings.shape}")
print(f"Number of actions: {len(historical_actions)}")

# Create test embedding
embedding = torch.randn(1, 128)

# Convert to numpy
embedding_np = embedding.detach().cpu().numpy()
historical_embeddings_np = historical_embeddings.squeeze(1).detach().cpu().numpy()

# Compute dot product
dot_product = np.dot(embedding_np, historical_embeddings_np.T)

# Compute norms
embedding_norm = np.linalg.norm(embedding_np, axis=1)
historical_norm = np.linalg.norm(historical_embeddings_np, axis=1)

# Compute similarity
similarity = dot_product / (embedding_norm.reshape(-1, 1) * historical_norm)
similarity = similarity.flatten()

# Get top indices
top_n = 3
top_indices = np.argsort(similarity)[-top_n:][::-1]

# Get corresponding actions and similarities
top_actions = [historical_actions[i] for i in top_indices]
top_similarities = [similarity[i] for i in top_indices]

# Create results
results = list(zip(top_actions, top_similarities))

print(f"Found {len(results)} similar actions")
print("\nTop recommendation:")
action, sim = results[0]
print(f"Similarity score: {sim:.4f}")
for key, value in action.items():
    print(f"  {key}: {value}")
print("\nTest completed successfully!") 