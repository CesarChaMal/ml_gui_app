#!/bin/bash

echo "Starting ML GUI Image Classifier..."
echo

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "classifier.py" ]; then
    echo "Error: classifier.py not found"
    echo "Please run this script from the finishedProject directory"
    exit 1
fi

# Check if model file exists
if [ ! -f "baseline_mariya.keras" ]; then
    echo "Warning: baseline_mariya.keras model file not found"
    echo "The application may not work properly without the trained model"
    echo
fi

# Install requirements
echo "Installing required packages..."
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

echo
echo "Starting the application..."
echo "The GUI will open in your default web browser at http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the application"
echo

$PYTHON_CMD classifier.py