import json
import torch
import joblib
from datetime import datetime
import numpy as np

# Load encoders
device_encoder = joblib.load('device_encoder.pkl')
capability_encoder = joblib.load('capability_encoder.pkl')
state_encoder = joblib.load('state_encoder.pkl')
capability_group_encoder = joblib.load('capability_group_encoder.pkl')

capability_to_group = {
    'temperature_control': 'thermal',
    'door_status': 'mechanical',
    'power': 'mechanical',
    'incoming_call': 'audio',
    'volume_control': 'audio',
    'brightness_control': 'visual',
    'screen_state': 'visual',
    'humidity_control': 'thermal',
    'motion_detection': 'security',
    'lock_control': 'security'
}

# Load synthetic dataset
with open('event_dataset.json', 'r') as file:
    dataset = json.load(file)

def preprocess_event(event):
    timestamp = datetime.fromisoformat(event["timestamp"].rstrip("Z"))
    timestamp_features = [timestamp.hour, timestamp.minute, timestamp.second]

    device = device_encoder.transform([event["device"]])[0]
    capability = capability_encoder.transform([event["capability"]])[0]
    state = state_encoder.transform([event["attributes"]["state"]])[0]
    measurement = event["attributes"].get("measurement", 0)

    capability_group = capability_to_group.get(event["capability"], "other")
    capability_group_encoded = capability_group_encoder.transform([capability_group])[0]

    return timestamp_features + [device, capability, capability_group_encoded, state, measurement]

# Preprocess entire dataset
processed_sequences = []
for sequence in dataset:
    processed_sequence = [preprocess_event(event) for event in sequence]
    processed_sequences.append(processed_sequence)

# Get the length of each sequence and the feature dimension
sequence_lengths = [len(seq) for seq in processed_sequences]
max_length = max(sequence_lengths)
feature_dim = len(processed_sequences[0][0])  # Dimension of each event vector

print(f"Sequences have varying lengths from {min(sequence_lengths)} to {max_length}")
print(f"Feature dimension: {feature_dim}")

# Pad sequences to the same length
padded_sequences = []
for seq in processed_sequences:
    # Create a padded sequence filled with zeros
    padded_seq = np.zeros((max_length, feature_dim))
    # Fill in the actual sequence data
    padded_seq[:len(seq)] = seq
    padded_sequences.append(padded_seq)

# Convert to tensor
tensor_data = torch.tensor(padded_sequences, dtype=torch.float32)
print(f"Final tensor shape: {tensor_data.shape}")

# Save the tensor and sequence lengths
torch.save({
    'tensor': tensor_data,
    'sequence_lengths': sequence_lengths  # Save original lengths for later use
}, 'processed_event_dataset.pt')

print("Preprocessing completed with capability groups, tensor data saved.")
