import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load preprocessed data
data_dict = torch.load('processed_event_dataset.pt')
dataset = data_dict['tensor']  # Extract the tensor from the dictionary
sequence_lengths = data_dict['sequence_lengths']  # Extract the sequence lengths

print(f"Loaded dataset with shape: {dataset.shape}")
print(f"Sequence lengths range from {min(sequence_lengths)} to {max(sequence_lengths)}")

# Use a subset of the data to reduce memory usage
MAX_SEQUENCES = 100  # Adjust based on your memory constraints
MAX_SEQ_LENGTH = 200  # Truncate sequences to this length

if dataset.shape[0] > MAX_SEQUENCES:
    print(f"Using only the first {MAX_SEQUENCES} sequences to reduce memory usage")
    dataset = dataset[:MAX_SEQUENCES]
    sequence_lengths = sequence_lengths[:MAX_SEQUENCES]

if dataset.shape[1] > MAX_SEQ_LENGTH:
    print(f"Truncating sequences to length {MAX_SEQ_LENGTH} to reduce memory usage")
    dataset = dataset[:, :MAX_SEQ_LENGTH, :]

print(f"Reduced dataset shape: {dataset.shape}")

# Create a simpler embedding model
class SimpleEventEmbeddingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)  # For reconstruction
        
    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Process with LSTM
        _, (hidden, _) = self.lstm(encoded)
        # Return the hidden state as the embedding
        return hidden.squeeze(0)
    
    def reconstruct(self, embedding, seq_length):
        # Expand the embedding to match sequence length
        expanded = embedding.unsqueeze(1).expand(-1, seq_length, -1)
        # Decode back to input space
        return self.decoder(expanded)

# Create dataset and dataloader for batching
tensor_dataset = TensorDataset(dataset)
batch_size = 16  # Adjust based on your memory constraints
dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
input_dim = dataset.shape[2]  # Feature dimension (8)
model = SimpleEventEmbeddingModel(input_dim=input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop with batching
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    
    for batch in dataloader:
        batch_data = batch[0]  # Extract data from TensorDataset
        optimizer.zero_grad()
        
        # Get embeddings for the batch
        embeddings = model(batch_data)
        
        # Reconstruct the input from the embeddings
        reconstructed = model.reconstruct(embeddings, batch_data.shape[1])
        
        # Calculate reconstruction loss
        loss = criterion(reconstructed, batch_data)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'event_embedding_model.pt')
print("Model trained and saved.")

# Generate and save embeddings for the entire dataset
model.eval()
with torch.no_grad():
    all_embeddings = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_embeddings = model(batch).cpu().numpy()
        all_embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    np.save('event_embeddings.npy', all_embeddings)

print(f"Generated embeddings with shape {all_embeddings.shape}")
print("Embeddings saved to event_embeddings.npy")
