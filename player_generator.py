import json
import random
from datetime import datetime, timedelta

# Define IoT devices, capabilities, and diverse attributes
devices = {
    'smart_light': {
        'capability': 'brightness_control',
        'attributes': ['brightness', 'color_temp', 'power_state'],
        'states': {
            'brightness': ['OFF', 'DIM', 'MEDIUM', 'BRIGHT'],
            'color_temp': ['WARM', 'NEUTRAL', 'COOL'],
            'power_state': ['ON', 'OFF']
        },
        'group': 'visual'
    },
    'smart_tv': {
        'capability': 'video_audio_control',
        'attributes': ['volume', 'brightness', 'power_state'],
        'states': {
            'volume': ['MUTE', 'LOW', 'MEDIUM', 'HIGH'],
            'brightness': ['LOW', 'MEDIUM', 'HIGH'],
            'power_state': ['ON', 'OFF']
        },
        'group': 'audio_visual'
    },
    'smartphone': {
        'capability': 'notifications',
        'attributes': ['incoming_call', 'app_alert', 'screen_state'],
        'states': {
            'incoming_call': ['CALL_RECEIVED', 'NO_CALL'],
            'app_alert': ['WHATSAPP', 'EMAIL', 'NONE'],
            'screen_state': ['ON', 'OFF']
        },
        'group': 'communication'
    },
    'ac_unit': {
        'capability': 'temperature_control',
        'attributes': ['temperature', 'power_state', 'fan_speed'],
        'states': {
            'temperature': ['COOL_LOW', 'COOL_MEDIUM', 'COOL_HIGH'],
            'power_state': ['ON', 'OFF'],
            'fan_speed': ['LOW', 'MEDIUM', 'HIGH']
        },
        'group': 'thermal'
    },
    'fridge': {
        'capability': 'storage_status',
        'attributes': ['door_state', 'temperature', 'energy_mode'],
        'states': {
            'door_state': ['OPEN', 'CLOSED'],
            'temperature': ['NORMAL', 'COOLER'],
            'energy_mode': ['ENERGY_SAVER_ON', 'ENERGY_SAVER_OFF']
        },
        'group': 'thermal'
    },
    'smart_lock': {
        'capability': 'security_control',
        'attributes': ['lock_state', 'alarm_state'],
        'states': {
            'lock_state': ['LOCKED', 'UNLOCKED'],
            'alarm_state': ['ARMED', 'DISARMED']
        },
        'group': 'security'
    },
    'washer': {
        'capability': 'wash_cycle',
        'attributes': ['power_state', 'mode'],
        'states': {
            'power_state': ['ON', 'OFF'],
            'mode': ['QUICK_WASH', 'HEAVY_WASH', 'SPIN_DRY']
        },
        'group': 'mechanical'
    }
}

# Human activity-based event clustering
time_based_scenarios = {
    "morning_routine": (6, 9),
    "leaving_home": (9, 10),
    "cooking_dinner": (17, 19),
    "watching_tv": (19, 21),
    "returning_home": (18, 20),
    "bedtime_routine": (22, 24)
}

# Generate an event with multiple attributes
def generate_event(device_name, timestamp):
    device_info = devices[device_name]
    capability = device_info['capability']
    attributes = device_info['attributes']
    
    # Assign state values to multiple attributes
    attributes_data = {}
    for attr in attributes:
        attributes_data[attr] = random.choice(device_info['states'][attr])

    return {
        "timestamp": timestamp.isoformat() + "Z",
        "device": device_name,
        "capability": capability,
        "attributes": attributes_data
    }

# Generate realistic events per second for a full day
def generate_player_file(start_time, duration_hours=24):
    events = []
    current_time = start_time

    while current_time < start_time + timedelta(hours=duration_hours):
        hour = current_time.hour
        burst_mode = False

        # Determine active scenario for bursts of IoT activities
        for scenario, (start, end) in time_based_scenarios.items():
            if start <= hour < end:
                burst_mode = True
                break

        if burst_mode:
            # Simulate multiple devices operating at the same time
            num_devices = random.randint(3, 5)
            selected_devices = random.sample(list(devices.keys()), num_devices)

            for device in selected_devices:
                events.append(generate_event(device, current_time))

            # Small delay within the burst
            current_time += timedelta(seconds=random.randint(1, 3))
        else:
            # Normal background activity
            device = random.choice(list(devices.keys()))
            events.append(generate_event(device, current_time))
            current_time += timedelta(seconds=1)

        # Occasionally simulate long gaps for inactivity (e.g., work hours, sleeping)
        if random.random() < 0.05:
            current_time += timedelta(minutes=random.randint(10, 60))

    return events

# Save the generated player file
def save_player_file(events, filename="player_file.json"):
    with open(filename, 'w') as f:
        json.dump(events, f, indent=4)
    print(f"Player file '{filename}' generated successfully with {len(events)} events.")

# Run the script
if __name__ == '__main__':
    start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    player_events = generate_player_file(start_time)
    save_player_file(player_events)
