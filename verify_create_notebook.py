#!/usr/bin/env python3
import nbformat as nbf
import numpy as np
import json
import torch
import os

print("Testing notebook creation with fixed find_similar_actions function...")

# Verify our solution works
def test_find_similar_actions():
    """Test the fixed find_similar_actions function with synthetic data"""
    # Create test tensors
    embedding = torch.randn(1, 128)
    historical_embeddings = torch.randn(10, 1, 128)  # 3D shape like in the real data
    historical_actions = [{"device": f"device_{i}", "state": f"state_{i}"} for i in range(10)]
    
    # Call the function with the synthetic data
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
        results = find_similar_actions(embedding, historical_embeddings, historical_actions)
        print(f"Function returned {len(results)} results successfully")
        return True
    except Exception as e:
        print(f"Error testing function: {e}")
        return False

# Run the test
if test_find_similar_actions():
    print("Function test passed successfully!")
else:
    print("Function test failed!")
    exit(1)

# Now update the create_notebook.py file with our fixed function
try:
    with open("create_notebook.py", "r") as f:
        create_notebook_code = f.read()
    
    # Find the existing find_similar_actions function
    start_marker = "def find_similar_actions(embedding, historical_embeddings, historical_actions, top_n=3):"
    end_marker = "return list(zip(top_actions, top_similarities))"
    
    start_idx = create_notebook_code.find(start_marker)
    if start_idx == -1:
        print("Could not find find_similar_actions function in create_notebook.py")
        exit(1)
    
    end_idx = create_notebook_code.find(end_marker, start_idx)
    if end_idx == -1:
        print("Could not find the end of find_similar_actions function")
        exit(1)
    
    # Include the return statement in the replaced portion
    end_idx += len(end_marker)
    
    # The new function implementation
    new_function = """def find_similar_actions(embedding, historical_embeddings, historical_actions, top_n=3):
    \"\"\"Find the most similar actions based on embedding similarity\"\"\"
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
    
    return list(zip(top_actions, top_similarities))"""
    
    # Replace the function in the code
    updated_code = create_notebook_code[:start_idx] + new_function + create_notebook_code[end_idx:]
    
    # Write the updated code back to the file
    with open("create_notebook.py", "w") as f:
        f.write(updated_code)
    
    print("Successfully updated the find_similar_actions function in create_notebook.py")
    
    # Run the create_notebook.py script to generate the updated notebook
    print("\nGenerating the updated notebook...")
    exit_code = os.system("python create_notebook.py")
    
    if exit_code == 0:
        print("\nNotebook generated successfully!")
        print("The fixed find_similar_actions function has been incorporated into the notebook.")
    else:
        print(f"\nError generating notebook: exit code {exit_code}")
        exit(1)
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    exit(1)

print("\nVerification complete - all tests passed!") 