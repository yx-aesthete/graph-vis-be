#!/usr/bin/env bash

# Install system dependencies
apt-get update && apt-get install -y graphviz graphviz-dev

# Set environment variables for include and library paths
export CPATH=/usr/include/graphviz/
export LIBRARY_PATH=/usr/lib64/graphviz/

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the gunicorn server
gunicorn app:app
