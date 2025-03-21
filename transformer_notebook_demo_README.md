# Large Event Model (LEM) Demo Notebook

This Jupyter notebook demonstrates how to load and use the Large Event Model (LEM) for event embeddings and action recommendations in smart home environments. It provides a step-by-step walkthrough of the model's functionality and capabilities.

## Prerequisites

Before running the notebook, ensure you have the following dependencies installed:

```bash
pip install torch numpy joblib matplotlib scikit-learn ipython jupyter nbformat
```

For the LLM comparison section, you'll also need:

```bash
pip install transformers
```

## Required Files

The notebook requires the following files to be in the same directory:

- `transformer_model.pt` - The trained LEM model
- `transformer_embeddings.pt` - Pre-computed historical embeddings
- `transformer_actions.json` - Action data corresponding to the embeddings
- `event_dataset.json` - Dataset of event sequences
- `device_encoder.pkl` - Encoder for device categories
- `capability_encoder.pkl` - Encoder for capability categories
- `state_encoder.pkl` - Encoder for state categories

## Running the Notebook

To run the notebook, use the following command in your terminal:

```bash
jupyter notebook transformer_notebook_demo.ipynb
```

Or if you're using JupyterLab:

```bash
jupyter lab transformer_notebook_demo.ipynb
```

## Notebook Contents

The notebook is organized into the following sections:

1. **Setup and Dependencies** - Import necessary libraries
2. **Large Event Model (LEM) Definition** - Define the LEM architecture
3. **Load Encoders and Preprocessing Functions** - Load the encoders and define functions for event preprocessing
4. **Load LEM and Historical Data** - Load the pre-trained model and historical data
5. **Load Event Dataset** - Load and explore the event dataset
6. **Find Similar Actions Based on Event Sequences** - Demonstrate how to get embeddings and find similar actions
7. **Example Events for Custom Testing** - Pre-defined event sequences for testing
8. **Create Custom Events and Get Recommendations** - Show how to create custom events and get recommendations
9. **Benchmark Different Event Sequences** - Measure inference time for various sequence lengths
10. **Compare with LLM** - Compare LEM performance with a general-purpose LLM
11. **Conclusion** - Summary of what was demonstrated

## Customizing Event Sequences

In Section 8 of the notebook, you can customize the event sequence by modifying the `custom_events` list. Use the available devices, capabilities, and states shown in Section 3 to create meaningful event sequences.

Example:
```python
custom_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("smart_thermostat", "temperature_control", "WARM"),
    create_custom_event("smartphone", "app_usage", "ACTIVE"),
]
```

## LLM Comparison

Section 10 compares the specialized LEM with a general-purpose Large Language Model. To run this section, you'll need:
- The `transformers` library installed
- For Llama models, a Hugging Face access token may be required
- Sufficient memory to load the LLM (the notebook includes fallbacks to smaller models)

## Troubleshooting

If you encounter any issues:

1. Ensure all required files are in the correct directory
2. Check that you have installed all the required dependencies
3. Verify that the model is compatible with your version of PyTorch
4. Contact Sayonsom Chanda at +91-93304-77432 (sayonsom.c@samsung.com) for assistance