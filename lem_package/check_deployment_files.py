#!/usr/bin/env python3
"""
Script to check if all required model files are present before deployment.
"""

import os
import sys

def main():
    """
    Main function to check if all required model files are present.
    """
    # Required model files
    required_files = [
        'event_embedding_model.pt',
        'device_encoder.pkl',
        'capability_encoder.pkl',
        'state_encoder.pkl',
        'historical_event_embeddings.pt',
        'historical_actions.json'
    ]
    
    # Check if we're in the lem_package directory
    if os.path.basename(os.getcwd()) == 'lem_package':
        # We need to check in the parent directory
        parent_dir = os.path.dirname(os.getcwd())
        check_dir = parent_dir
    else:
        # We're already in the parent directory
        check_dir = os.getcwd()
    
    # Check if all required files exist
    missing_files = []
    for file in required_files:
        file_path = os.path.join(check_dir, file)
        if not os.path.isfile(file_path):
            missing_files.append(file)
    
    # Print results
    if missing_files:
        print("ERROR: The following required model files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease make sure all required model files are in the project root directory before deployment.")
        sys.exit(1)
    else:
        print("SUCCESS: All required model files are present.")
        print("You can proceed with deployment.")
        sys.exit(0)

if __name__ == '__main__':
    main() 