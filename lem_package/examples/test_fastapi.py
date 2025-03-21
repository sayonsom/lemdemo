#!/usr/bin/env python3
"""
Test script for the FastAPI implementation of the LEM API.
"""

import requests
import json
import argparse

def main():
    """
    Main function to test the FastAPI implementation.
    """
    parser = argparse.ArgumentParser(description='Test the LEM FastAPI implementation')
    parser.add_argument('--host', default='http://localhost:8000', help='Host URL of the API')
    args = parser.parse_args()
    
    base_url = args.host
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Test preprocess endpoint
    print("Testing preprocess endpoint...")
    event = {
        "device": "ac_unit",
        "capability": "power",
        "attributes": {
            "state": "ON"
        },
        "timestamp": "2023-01-01T08:00:00"
    }
    response = requests.post(f"{base_url}/preprocess", json=event)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()
    
    # Test embed endpoint
    print("Testing embed endpoint...")
    events = [
        {
            "device": "light",
            "capability": "power",
            "attributes": {
                "state": "ON"
            },
            "timestamp": "2023-01-01T08:00:00"
        },
        {
            "device": "door",
            "capability": "lock",
            "attributes": {
                "state": "UNLOCKED"
            },
            "timestamp": "2023-01-01T08:01:00"
        }
    ]
    response = requests.post(f"{base_url}/embed", json=events)
    print(f"Status code: {response.status_code}")
    print(f"Response shape: {len(response.json()['embedding'])}x{len(response.json()['embedding'][0])}")
    print()
    
    # Test suggest endpoint
    print("Testing suggest endpoint...")
    response = requests.post(f"{base_url}/suggest", json=events)
    print(f"Status code: {response.status_code}")
    print(f"Suggested action: {json.dumps(response.json()['action'], indent=2)}")

if __name__ == '__main__':
    main() 