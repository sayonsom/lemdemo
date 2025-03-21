"""
This script shows how to fix the cross-device scenario warnings in your notebook
by replacing the unrecognized 'ACTIVE' state with the correct 'CALL_RECEIVED' state.
"""

print("=== CURRENT PROBLEMATIC CODE IN NOTEBOOK ===")
print("""
# Example: Incoming Call During TV Watching
call_tv_events = [
    {
        "device": "smart_tv",
        "capability": "power",
        "attributes": {"state": "ON"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smart_tv",
        "capability": "volume_control",
        "attributes": {"state": "MEDIUM"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smartphone",
        "capability": "incoming_call",
        "attributes": {"state": "ACTIVE"},  # This state is causing the warning
        "timestamp": datetime.now().isoformat()
    }
]
""")

print("\n\n=== FIXED CODE TO USE IN NOTEBOOK ===")
print("""
# Example: Incoming Call During TV Watching
call_tv_events = [
    {
        "device": "smart_tv",
        "capability": "power",
        "attributes": {"state": "ON"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smart_tv",
        "capability": "volume_control",
        "attributes": {"state": "MEDIUM"},
        "timestamp": datetime.now().isoformat()
    },
    {
        "device": "smartphone",
        "capability": "incoming_call",
        "attributes": {"state": "CALL_RECEIVED"},  # Fixed state value
        "timestamp": datetime.now().isoformat()
    }
]
""")

print("\n\n=== HOW TO IMPLEMENT THIS FIX ===")
print("1. Find the 'call_tv_events' list in your notebook")
print("2. Replace 'ACTIVE' with 'CALL_RECEIVED' in the smartphone event")
print("3. Run the cell again")
print("4. This should eliminate the warning about unknown label 'ACTIVE'")

print("\n\n=== EXPLANATION ===")
print("The valid states for the 'incoming_call' capability are:")
print("- 'CALL_RECEIVED'")
print("- 'NO_CALL'")
print("- 'CALENDAR_ALERT'")
print("\nThe state 'ACTIVE' is not recognized by the encoder, which causes the warning.")
print("By changing it to 'CALL_RECEIVED', you're using a valid state value that the model recognizes.") 