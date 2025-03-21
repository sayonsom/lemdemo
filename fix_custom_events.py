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

print("\nAVAILABLE VALID VALUES:")
print("\nDevice names:", list(device_encoder.classes_))
print("\nCapabilities:", list(capability_encoder.classes_))
print("\nStates:", list(state_encoder.classes_))

def create_custom_event(device, capability, state):
    return {
        "device": device,
        "capability": capability,
        "attributes": {"state": state},
        "timestamp": datetime.now().isoformat()
    }

# Current problematic custom_events (causing warnings)
problematic_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("smart_therm", "temperature_control", "WARM"),  # Device not found
    create_custom_event("smartphone", "app_usage", "ACTIVE"),  # Capability not found
]

# Corrected custom_events using only values from the encoders
correct_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("ac_unit", "temperature_control", "COOL_MEDIUM"),  # Using valid device and state
    create_custom_event("smartphone", "incoming_call", "CALL_RECEIVED"),  # Using valid capability and state
]

print("\n\n=== CURRENT CUSTOM EVENTS (WITH PROBLEMS) ===")
for i, event in enumerate(problematic_events):
    print(f"Event {i+1}: {event['device']} - {event['capability']} - {event['attributes']['state']}")

print("\n\n=== CORRECTED CUSTOM EVENTS (WILL WORK WITHOUT WARNINGS) ===")
for i, event in enumerate(correct_events):
    print(f"Event {i+1}: {event['device']} - {event['capability']} - {event['attributes']['state']}")

print("\n\n=== RECOMMENDATION FOR YOUR NOTEBOOK ===")
print("Replace your current custom_events with this:")
print("custom_events = [")
print("    create_custom_event(\"smart_light\", \"power\", \"ON\"),")
print("    create_custom_event(\"ac_unit\", \"temperature_control\", \"COOL_MEDIUM\"),")
print("    create_custom_event(\"smartphone\", \"incoming_call\", \"CALL_RECEIVED\"),")
print("]")

print("\n\n=== EXPLANATION ===")
print("1. 'smart_therm' was not recognized because it's not in the device list.")
print("   - Valid devices you can use are:", list(device_encoder.classes_))
print("   - We replaced it with 'ac_unit' which supports temperature_control")
print("\n2. 'app_usage' was not recognized because it's not in the capability list.")
print("   - Valid capabilities you can use are:", list(capability_encoder.classes_))
print("   - We replaced it with 'incoming_call' for smartphone")
print("\n3. 'WARM' was not recognized because it's not in the state list.")
print("   - Valid states you can use are:", list(state_encoder.classes_))
print("   - We replaced it with 'COOL_MEDIUM' for temperature_control") 