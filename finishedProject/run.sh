#!/bin/bash

echo "Starting ML GUI Image Classifier..."
echo

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda and try again"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "classifier.py" ]; then
    echo "Error: classifier.py not found"
    echo "Please run this script from the finishedProject directory"
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "ml_gui_env"; then
    echo "Creating conda environment 'ml_gui_env'..."
    conda create -n ml_gui_env python=3.9 -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment"
        exit 1
    fi
fi

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml_gui_env

# Install requirements using conda-forge when possible
echo "Installing required packages..."
conda install -c conda-forge tensorflow flask pillow numpy -y
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

echo
echo "Starting the application..."
echo "The GUI will open in your default web browser"
echo "Press Ctrl+C to stop the application"
echo

python classifier.py