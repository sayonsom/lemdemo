import json
import random
from datetime import datetime, timedelta

# Clearly define devices, capabilities, states, and groups
devices = {
    'ac_unit': {
        'capability': 'temperature_control',
        'states': ['OFF', 'ENERGY_SAVER', 'COOL_LOW', 'COOL_MEDIUM', 'COOL_HIGH'],
        'measurement_range': (16, 30),
        'group': 'thermal'
    },
    'fridge': {
        'capability': 'door_status',
        'states': ['OPEN', 'CLOSED', 'ENERGY_SAVER_ON', 'ENERGY_SAVER_OFF'],
        'group': 'thermal'
    },
    'washer': {
        'capability': 'power',
        'states': ['ON', 'OFF'],
        'group': 'mechanical'
    },
    'smartphone': {
        'capability': 'incoming_call',
        'states': ['CALL_RECEIVED', 'NO_CALL', 'CALENDAR_ALERT'],
        'group': 'audio'
    },
    'smart_tv': {
        'capability': 'volume_control',
        'states': ['MUTE', 'LOW', 'MEDIUM', 'HIGH', 'SLEEP_TIMER_ON', 'SLEEP_TIMER_OFF'],
        'group': 'audio_visual'
    },
    'smart_light': {
        'capability': 'brightness_control',
        'states': ['OFF', 'DIM', 'MEDIUM', 'BRIGHT'],
        'group': 'visual'
    },
    'smart_lock': {
        'capability': 'lock_control',
        'states': ['LOCKED', 'UNLOCKED'],
        'group': 'security'
    }
}

# Realistic scenario generator for event clusters
def generate_event(device_name, state, timestamp, measurement=None):
    capability = devices[device_name]['capability']
    event = {
        "timestamp": timestamp.isoformat() + "Z",
        "device": device_name,
        "capability": capability,
        "attributes": {"state": state}
    }
    if measurement:
        event["attributes"]["measurement"] = measurement
    return event

# Clustered event generation simulating realistic interactions
def generate_realistic_event_cluster(start_time):
    cluster_events = []
    current_time = start_time

    # Randomly select a persona-driven scenario
    scenario = random.choice([
        'morning_routine', 'evening_relaxation', 'party_time', 'night_alert', 'lazy_afternoon'
    ])

    if scenario == 'morning_routine':
        # Short burst of events (wake up)
        cluster_events.extend([
            generate_event('smart_light', 'BRIGHT', current_time),
            generate_event('smart_lock', 'UNLOCKED', current_time + timedelta(seconds=30)),
            generate_event('fridge', 'OPEN', current_time + timedelta(minutes=1)),
            generate_event('ac_unit', 'ENERGY_SAVER', current_time + timedelta(minutes=2))
        ])

    elif scenario == 'evening_relaxation':
        # Evening cluster (TV and lights dim)
        cluster_events.extend([
            generate_event('smart_light', 'DIM', current_time),
            generate_event('smart_tv', 'MEDIUM', current_time + timedelta(seconds=15)),
            generate_event('ac_unit', 'COOL_MEDIUM', current_time + timedelta(seconds=30), measurement=22),
            generate_event('smartphone', 'NO_CALL', current_time + timedelta(minutes=5))
        ])

    elif scenario == 'party_time':
        # Active party scenario
        cluster_events.extend([
            generate_event('ac_unit', 'COOL_HIGH', current_time, measurement=18),
            generate_event('smart_tv', 'HIGH', current_time + timedelta(seconds=20)),
            generate_event('fridge', 'OPEN', current_time + timedelta(seconds=45)),
            generate_event('smart_light', 'BRIGHT', current_time + timedelta(seconds=60))
        ])

    elif scenario == 'night_alert':
        # Security and audio-visual cluster
        cluster_events.extend([
            generate_event('smartphone', 'CALL_RECEIVED', current_time),
            generate_event('smart_tv', 'MUTE', current_time + timedelta(seconds=5)),
            generate_event('smart_lock', 'LOCKED', current_time + timedelta(seconds=10)),
            generate_event('smart_light', 'OFF', current_time + timedelta(seconds=20))
        ])

    elif scenario == 'lazy_afternoon':
        # Infrequent events, mostly quiet
        cluster_events.extend([
            generate_event('washer', 'ON', current_time),
            generate_event('smart_tv', 'LOW', current_time + timedelta(minutes=1)),
            generate_event('ac_unit', 'ENERGY_SAVER', current_time + timedelta(minutes=5))
        ])

    # Return generated cluster
    return cluster_events

# Main event corpus generation function
def generate_event_sequence(clusters_per_day=10, days=30):
    events = []
    current_time = datetime.utcnow() - timedelta(days=days)

    for _ in range(days):
        for _ in range(clusters_per_day):
            # Generate a cluster of events
            cluster = generate_realistic_event_cluster(current_time)
            events.extend(cluster)

            # Short intervals between events in cluster
            current_time = cluster[-1]['timestamp']
            current_time = datetime.fromisoformat(current_time.rstrip('Z')) + timedelta(minutes=random.randint(1, 5))

        # Long delay (night time or inactivity)
        current_time += timedelta(hours=random.randint(6, 12))

    # Sort events chronologically
    events.sort(key=lambda x: x["timestamp"])
    return events

# Save to dataset
def generate_and_save_dataset(filename='event_dataset.json', clusters_per_day=10, days=30):
    dataset = []
    for _ in range(500):  # generating 500 sequences
        sequence = generate_event_sequence(clusters_per_day, days)
        dataset.append(sequence)

    with open(filename, 'w') as file:
        json.dump(dataset, file, indent=4)

    print("Comprehensive synthetic dataset generated successfully.")

# Run script
if __name__ == '__main__':
    generate_and_save_dataset()
