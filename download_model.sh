#!/bin/bash
set -e
FROM python:3.9-slim

# Install dependencies
RUN pip install gdown

# Make a directory
WORKDIR /app/example_models
# Download multiple files from Google Drive
# Replace FILE_ID_x with your actual file IDs
RUN gdown --folders https://drive.google.com/drive/folders/1RWfl_mU-0kAR92kXVZlHvNadIiok90cE
