import json
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open('event_dataset.json', 'r') as f:
    dataset = json.load(f)

devices = set()
capabilities = set()
states = set()

# Explicit capability to group mapping
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

capability_groups = set(capability_to_group.values())

# Gather unique categories from dataset
for sequence in dataset:
    for event in sequence:
        devices.add(event['device'])
        capabilities.add(event['capability'])
        states.add(event['attributes']['state'])

# Create encoders
device_encoder = LabelEncoder().fit(sorted(list(devices)))
capability_encoder = LabelEncoder().fit(sorted(list(capabilities)))
state_encoder = LabelEncoder().fit(sorted(list(states)))
capability_group_encoder = LabelEncoder().fit(sorted(list(capability_groups)))

# Save encoders
joblib.dump(device_encoder, 'device_encoder.pkl')
joblib.dump(capability_encoder, 'capability_encoder.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(capability_group_encoder, 'capability_group_encoder.pkl')

print("Encoders (including capability groups) saved successfully.")
