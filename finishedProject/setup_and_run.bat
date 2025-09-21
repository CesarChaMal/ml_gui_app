@echo off
echo ML GUI Image Classifier - Setup and Run
echo =======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
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

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "baseline_mariya.keras" (
    echo Warning: baseline_mariya.keras model file not found
    echo The application may not work properly without the trained model
    echo.
)

echo.
echo Setup complete! Starting the application...
echo The GUI will open in your default web browser at http://127.0.0.1:5000
echo Press Ctrl+C to stop the application
echo.

python classifier.py

echo.
echo Application stopped.
pause