#!/usr/bin/env python3
"""
Script to prepare the LEM package with the necessary model files.
"""

import os
import shutil

def main():
    """
    Main function to prepare the package.
    """
    # Create directories
    os.makedirs('lem_package/models', exist_ok=True)
    os.makedirs('lem_package/encoders', exist_ok=True)
    os.makedirs('lem_package/data', exist_ok=True)
    
    # Copy model files
    shutil.copy('../event_embedding_model.pt', 'lem_package/models/')
    
    # Copy encoder files
    shutil.copy('../device_encoder.pkl', 'lem_package/encoders/')
    shutil.copy('../capability_encoder.pkl', 'lem_package/encoders/')
    shutil.copy('../state_encoder.pkl', 'lem_package/encoders/')
    
    # Copy data files
    shutil.copy('../historical_event_embeddings.pt', 'lem_package/data/')
    shutil.copy('../historical_actions.json', 'lem_package/data/')
    
    print("Package prepared successfully!")

if __name__ == '__main__':
    main() 