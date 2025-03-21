"""
Large Event Model (LEM) - A system for processing, analyzing, and predicting timeseries events from home appliances.
"""

from .model import EventTransformerModel, EventEmbeddingModel, LSTMEmbeddingModel
from .inference import preprocess_event, get_event_embedding, suggest_device_action

__version__ = "0.1.0" 