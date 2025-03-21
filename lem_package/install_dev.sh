#!/bin/bash
# Script to install the LEM package in development mode

# Run the prepare_package.py script
echo "Preparing package..."
python prepare_package.py

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Installation complete!" 