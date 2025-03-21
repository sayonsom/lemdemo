import json
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import joblib
import os

print("Loading encoders...")
device_encoder = joblib.load('device_encoder.pkl')
capability_encoder = joblib.load('capability_encoder.pkl')
state_encoder = joblib.load('state_encoder.pkl')

print("\nAvailable devices:", device_encoder.classes_)
print("\nAvailable capabilities:", capability_encoder.classes_)
print("\nAvailable states:", state_encoder.classes_)

def preprocess_event(event):
    """Convert event data to numerical features using encoders"""
    try:
        # Try to encode with existing encoders
        device = device_encoder.transform([event["device"]])[0]
        capability = capability_encoder.transform([event["capability"]])[0]
        state = state_encoder.transform([event["attributes"]["state"]])[0]
        
        # Create feature vector
        features = [
            device, capability, state,
            # Add time-based features
            float(datetime.fromisoformat(event["timestamp"]).hour) / 24.0,
            float(datetime.fromisoformat(event["timestamp"]).minute) / 60.0,
            float(datetime.fromisoformat(event["timestamp"]).weekday()) / 7.0,
            1.0  # Bias term
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    except (ValueError, KeyError) as e:
        print(f"\nWarning: Encountered unknown label: {str(e)}")
        print("Using fallback encoding for unknown values.")
        
        # Use fallback values for unknown labels
        device_val = 0  # Default to first device
        capability_val = 0  # Default to first capability
        state_val = 0  # Default to first state
        
        # Try to encode known values
        try:
            if event["device"] in device_encoder.classes_:
                device_val = device_encoder.transform([event["device"]])[0]
        except:
            pass
            
        try:
            if event["capability"] in capability_encoder.classes_:
                capability_val = capability_encoder.transform([event["capability"]])[0]
        except:
            pass
            
        try:
            if event["attributes"]["state"] in state_encoder.classes_:
                state_val = state_encoder.transform([event["attributes"]["state"]])[0]
        except:
            pass
        
        # Create feature vector with fallback values
        features = [
            device_val, capability_val, state_val,
            # Add time-based features
            float(datetime.fromisoformat(event["timestamp"]).hour) / 24.0,
            float(datetime.fromisoformat(event["timestamp"]).minute) / 60.0,
            float(datetime.fromisoformat(event["timestamp"]).weekday()) / 7.0,
            1.0  # Bias term
        ]
        
        return torch.tensor(features, dtype=torch.float32)

def create_custom_event(device, capability, state):
    return {
        "device": device,
        "capability": capability,
        "attributes": {"state": state},
        "timestamp": datetime.now().isoformat()
    }

# PROBLEMATIC EVENTS - These are the ones causing warnings
problematic_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("smart_therm", "temperature_control", "WARM"),  # 'smart_therm' is wrong, should be 'smart_thermostat'
    create_custom_event("smartphone", "app_usage", "ACTIVE"),  # 'app_usage' is not recognized
]

# FIXED EVENTS - Here's how they should look
fixed_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("smart_thermostat", "temperature_control", "COOL_MEDIUM"),  # Fixed device name
    create_custom_event("smartphone", "incoming_call", "CALL_RECEIVED"),  # Fixed capability and state
]

# Process events and show the warnings
print("\n\nTESTING PROBLEMATIC EVENTS (You'll see warnings):")
for i, event in enumerate(problematic_events):
    print(f"\nProcessing event {i+1}: {event['device']} - {event['capability']} - {event['attributes']['state']}")
    processed = preprocess_event(event)

print("\n\nTESTING FIXED EVENTS (Should work without warnings):")
for i, event in enumerate(fixed_events):
    print(f"\nProcessing event {i+1}: {event['device']} - {event['capability']} - {event['attributes']['state']}")
    processed = preprocess_event(event)

# Check if "smart_thermostat" is actually in the encoder
print("\n\nDEBUGGING DEVICE NAMES:")
device_names = list(device_encoder.classes_)
print(f"Is 'smart_thermostat' in device encoder? {'smart_thermostat' in device_names}")
print(f"Is 'smart_therm' in device encoder? {'smart_therm' in device_names}")
print(f"Available device names: {device_names}")

# Check if WARM is in the states
print("\n\nDEBUGGING STATE NAMES:")
state_names = list(state_encoder.classes_)
print(f"Is 'WARM' in state encoder? {'WARM' in state_names}")
print(f"Available state names for temperature_control: {state_names}")

print("\n\nRECOMMENDED CUSTOM EVENTS FOR YOUR NOTEBOOK:")
print("custom_events = [")
print("    create_custom_event(\"smart_light\", \"power\", \"ON\"),")
print("    create_custom_event(\"smart_thermostat\", \"temperature_control\", \"COOL_MEDIUM\"),")  # Use one of the available state values
print("    create_custom_event(\"smartphone\", \"incoming_call\", \"CALL_RECEIVED\"),")
print("]") 