# Large Event Model (LEM)

## Overview

The Large Event Model (LEM) is a system designed to process, analyze, and predict timeseries events from home appliances. Similar to how Large Language Models (LLMs) process and generate text, LEM is specialized for understanding and predicting patterns in home automation events. The system uses transformer-based architectures to learn from historical event sequences and make predictions about future device states and actions.

## System Architecture

The LEM system consists of several key components:

### 1. Data Representation

Events from home appliances are represented as structured data with the following key attributes:
- **Device**: The appliance generating the event (e.g., ac_unit, fridge, smart_tv)
- **Capability**: The function or feature of the device (e.g., power, temperature_control, volume_control)
- **State**: The current state of the capability (e.g., ON, OFF, COOL_HIGH)
- **Timestamp**: When the event occurred
- **Attributes**: Additional metadata including measurements and state information

### 2. Encoders

The system uses several encoders to convert categorical data into numerical representations:
- **Device Encoder**: Converts device names to numerical IDs
- **Capability Encoder**: Converts capability names to numerical IDs
- **State Encoder**: Converts state values to numerical IDs
- **Capability Group Encoder**: Groups similar capabilities (e.g., thermal, mechanical, audio)

### 3. Event Vectorization

Events are converted into feature vectors with the following components:
- Device ID
- Capability ID
- State ID
- Time-based features (hour, minute, weekday)
- Capability group ID
- Measurements (when available)
- Bias term

### 4. Neural Network Models

The system implements several neural network architectures:

#### EventTransformerModel
A transformer-based model that processes sequences of events:
- Input projection layer
- Transformer encoder with multi-head attention
- Output projection layer

#### EventEmbeddingModel
A simpler transformer-based model for generating embeddings:
- Input projection layer
- Transformer encoder
- Mean pooling for sequence representation

#### LSTMEmbeddingModel
An LSTM-based model for sequence processing:
- Linear encoder
- LSTM layer
- Linear decoder for reconstruction

#### SimpleEventEmbeddingModel
A lightweight model for training embeddings:
- Linear encoder with ReLU activation
- LSTM layer
- Linear decoder for reconstruction

### 5. Inference System

The system uses similarity-based inference to predict actions:
- Embeds recent event sequences
- Compares with historical embeddings using cosine similarity
- Suggests actions based on the most similar historical patterns

## Data Flow

1. **Data Collection**: Home appliance events are collected and stored in JSON format
2. **Preprocessing**: Events are converted to numerical feature vectors
3. **Embedding**: Event sequences are embedded using transformer or LSTM models
4. **Training**: Models learn patterns from historical event sequences
5. **Inference**: New event sequences are compared with historical patterns to predict actions

## Key Files and Their Functions

### Core System Files

- **lem_demo.py**: Main demonstration of the LEM system, including model definitions, preprocessing, and inference
- **transformer_demo.py**: Implementation of the transformer-based model for event processing
- **events2vector.py**: Converts event data to numerical feature vectors
- **train_embeddings.py**: Trains embedding models on preprocessed event data
- **train_encoders.py**: Creates and trains encoders for categorical data

### Data Generation and Processing

- **build_event_corpus.py**: Generates synthetic event data for training and testing
- **analyze_dataset.py**: Analyzes the event dataset for statistics and patterns

### Model and Data Files

- **transformer_model.pt**: Saved transformer model weights
- **event_embedding_model.pt**: Saved embedding model weights
- **device_encoder.pkl**: Encoder for device names
- **capability_encoder.pkl**: Encoder for capability names
- **state_encoder.pkl**: Encoder for state values
- **capability_group_encoder.pkl**: Encoder for capability groups
- **event_dataset.json**: Dataset of event sequences
- **processed_event_dataset.pt**: Preprocessed and vectorized event data
- **historical_event_embeddings.pt**: Embeddings of historical event sequences

## Event Scenarios

The system models several realistic home automation scenarios:

1. **Morning Routine**: Light turns on, door unlocks, fridge opens, AC adjusts
2. **Evening Relaxation**: Lights dim, TV turns on, AC adjusts, phone quiets
3. **Party Time**: AC cools, TV volume increases, fridge opens, lights brighten
4. **Night Alert**: Phone receives call, TV mutes, door locks, lights turn off
5. **Lazy Afternoon**: Washer runs, TV plays quietly, AC in energy saver mode

## Technical Implementation Details

### Feature Engineering

Events are represented as 7-dimensional feature vectors:
- Device ID (encoded)
- Capability ID (encoded)
- State ID (encoded)
- Hour of day (normalized to 0-1)
- Minute of hour (normalized to 0-1)
- Day of week (normalized to 0-1)
- Bias term (1.0)

### Model Training

The embedding models are trained using:
- Reconstruction loss (MSE)
- Adam optimizer
- Batch processing for memory efficiency
- Sequence padding for uniform length

### Inference Process

1. Recent events are preprocessed and embedded
2. Cosine similarity is calculated with historical embeddings
3. The most similar historical pattern is identified
4. The corresponding action is suggested

## Usage Examples

### Training New Embeddings

```python
# Load preprocessed data
data_dict = torch.load('processed_event_dataset.pt')
dataset = data_dict['tensor']

# Initialize model
model = SimpleEventEmbeddingModel(input_dim=8)

# Train model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    # Training code...

# Save model
torch.save(model.state_dict(), 'event_embedding_model.pt')
```

### Making Predictions

```python
# Load recent events
recent_events = [...]  # List of recent events

# Preprocess events
processed_events = [preprocess_event(event) for event in recent_events]
tensor_input = torch.stack(processed_events).unsqueeze(0)

# Get embedding
with torch.no_grad():
    embedding = model(tensor_input)

# Calculate similarity with historical patterns
similarities = cosine_similarity(embedding, historical_embeddings)

# Find most similar pattern
most_similar_idx = torch.argmax(similarities).item()
suggested_action = historical_actions[most_similar_idx]
```

## Installation and Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Generate synthetic data:
   ```
   python build_event_corpus.py
   ```
4. Train encoders:
   ```
   python train_encoders.py
   ```
5. Process event data:
   ```
   python events2vector.py
   ```
6. Train embeddings:
   ```
   python train_embeddings.py
   ```
7. Run the demo:
   ```
   python lem_demo.py
   ```

## Future Enhancements

1. **Real-time Processing**: Enhance the system to process streaming event data
2. **Multi-modal Integration**: Incorporate data from different sensor types
3. **Contextual Awareness**: Add environmental context (weather, occupancy)
4. **Personalization**: Adapt to individual user preferences and patterns
5. **Anomaly Detection**: Identify unusual patterns that may indicate issues
6. **Explainable Predictions**: Provide reasoning for suggested actions
7. **Federated Learning**: Learn from multiple homes while preserving privacy

## Conclusion

The Large Event Model (LEM) represents a novel approach to home automation by applying transformer-based architectures to timeseries event data. By learning patterns from historical events, the system can predict and suggest actions based on current conditions, enabling more intelligent and responsive home environments.

Similar to how Large Language Models revolutionized text processing, LEM has the potential to transform how we interact with and automate our homes by understanding the complex relationships between devices, capabilities, states, and time.

## Requirements

The project requires the following dependencies:
- Python 3.8+
- PyTorch 2.6.0
- NumPy 2.2.3
- scikit-learn 1.6.1
- joblib 1.4.2

For a complete list of dependencies, see `requirements.txt`. 