#!/bin/bash

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements if requirements.txt exists
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi