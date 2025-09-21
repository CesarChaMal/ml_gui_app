@echo off
echo Starting ML GUI Image Classifier...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "classifier.py" (
    echo Error: classifier.py not found
    echo Please run this script from the finishedProject directory
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "baseline_mariya.keras" (
    echo Error: baseline_mariya.keras model file not found
    echo Please ensure the model file is in the current directory
    pause
    exit /b 1
)

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Starting the application...
echo The GUI will open in your default web browser
echo Press Ctrl+C to stop the application
echo.

python classifier.py

pause