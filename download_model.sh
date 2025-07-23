#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Install gdown if it's not already installed
pip install --no-cache-dir gdown

# Make a directory
mkdir -p /app/example_models
cd /app/example_models

gdown --folder https://drive.google.com/drive/folders/1RWfl_mU-0kAR92kXVZlHvNadIiok90cE
