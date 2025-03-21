#!/usr/bin/env python3
import json
from datetime import datetime

# Original events (with errors)
print("ORIGINAL EVENTS (WITH ERRORS):")
problematic_events = [
    {
        "device": "smart_light",
        "capability": "power",
        "attributes": {"state": "ON"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smart_therm",  # ERROR: Not a recognized device
        "capability": "temperature_control",
        "attributes": {"state": "WARM"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smartphone",
        "capability": "app_usage",  # ERROR: Not a recognized capability
        "attributes": {"state": "ACTIVE"},
        "timestamp": datetime.now().isoformat()
    }
]

# Print original events
for i, event in enumerate(problematic_events):
    print(f"\nEvent {i+1}:")
    print(f"  Device: {event['device']}")
    print(f"  Capability: {event['capability']}")
    print(f"  State: {event['attributes']['state']}")

# Fixed events
print("\n\nFIXED EVENTS:")
fixed_events = [
    {
        "device": "smart_light",
        "capability": "power",
        "attributes": {"state": "ON"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smart_thermostat",  # FIXED: Use the recognized device name
        "capability": "temperature_control",
        "attributes": {"state": "WARM"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smartphone",
        "capability": "incoming_call",  # FIXED: Use a recognized capability
        "attributes": {"state": "ACTIVE"},  # Note: "ACTIVE" might be recognized as "CALL_RECEIVED"
        "timestamp": datetime.now().isoformat()
    }
]

# Print fixed events
for i, event in enumerate(fixed_events):
    print(f"\nEvent {i+1}:")
    print(f"  Device: {event['device']}")
    print(f"  Capability: {event['capability']}")
    print(f"  State: {event['attributes']['state']}")

print("\n\nSUGGESTED CUSTOM EVENTS FOR NOTEBOOK:")
print('''
# Create a custom event sequence with valid device names and capabilities
custom_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("smart_thermostat", "temperature_control", "WARM"), 
    create_custom_event("smartphone", "incoming_call", "CALL_RECEIVED"),
]
''') 