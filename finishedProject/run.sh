#!/bin/bash

echo "Starting ML GUI Image Classifier..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.8+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if we're in the correct directory
if [ ! -f "classifier.py" ]; then
    echo "Error: classifier.py not found"
    echo "Please run this script from the finishedProject directory"
    exit 1
fi

# Check if model file exists
if [ ! -f "baseline_mariya.keras" ]; then
    echo "Error: baseline_mariya.keras model file not found"
    echo "Please ensure the model file is in the current directory"
    exit 1
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
echo "The GUI will open in your default web browser"
echo "Press Ctrl+C to stop the application"
echo

$PYTHON_CMD classifier.py