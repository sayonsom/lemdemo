"""
This script shows the necessary changes to fix the custom events in your Jupyter notebook
to avoid the warnings about unknown labels such as 'smart_therm' and 'app_usage'.
"""

print("=== CURRENT PROBLEMATIC CODE IN NOTEBOOK ===")
print("""
custom_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("smart_therm", "temperature_control", "WARM"),  # These values cause errors
    create_custom_event("smartphone", "app_usage", "ACTIVE"),  # These values cause errors
]
""")

print("\n\n=== FIXED CODE TO USE IN NOTEBOOK ===")
print("""
custom_events = [
    create_custom_event("smart_light", "power", "ON"),
    create_custom_event("ac_unit", "temperature_control", "COOL_MEDIUM"),  # Fixed values
    create_custom_event("smartphone", "incoming_call", "CALL_RECEIVED"),  # Fixed values
]
""")

print("\n\n=== HOW TO IMPLEMENT THIS FIX ===")
print("1. Find cell [8] in your notebook (or whichever cell contains the `custom_events` list)")
print("2. Replace the problematic code with the fixed code shown above")
print("3. Run the cell again")
print("4. This should eliminate the warnings about unknown labels")

print("\n\n=== EXPLANATION OF CHANGES ===")
print("1. Replaced 'smart_therm' with 'ac_unit' - The device 'smart_therm' is not recognized,")
print("   but 'ac_unit' is a valid device that supports temperature control")
print("2. Replaced 'WARM' with 'COOL_MEDIUM' - The state 'WARM' is not recognized,")
print("   but 'COOL_MEDIUM' is a valid state for temperature control")
print("3. Replaced 'app_usage' with 'incoming_call' - The capability 'app_usage' is not recognized,")
print("   but 'incoming_call' is a valid capability for smartphone")
print("4. Replaced 'ACTIVE' with 'CALL_RECEIVED' - The state 'ACTIVE' is not recognized,")
print("   but 'CALL_RECEIVED' is a valid state for incoming_call") 