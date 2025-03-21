#!/usr/bin/env python3
import subprocess
import sys

print("Regenerating the transformer notebook properly...")

# Simply run the create_notebook.py script to regenerate the notebook with the correct implementation
try:
    # Execute the create_notebook.py script
    result = subprocess.run(['python', 'create_notebook.py'], 
                          capture_output=True, 
                          text=True, 
                          check=True)
    
    # Print the output
    print(result.stdout)
    print("Notebook regenerated successfully!")
    
except subprocess.CalledProcessError as e:
    print(f"Error executing create_notebook.py: {e}")
    print(f"Error output: {e.stderr}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1) 