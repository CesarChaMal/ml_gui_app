@echo off
echo ML GUI Image Classifier - Conda Environment Setup
echo =================================================
echo.

REM Check if conda is installed
conda --version >nul 2>&1
if errorlevel 1 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda and try again
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

REM Create conda environment if it doesn't exist
conda info --envs | findstr "ml_gui_env" >nul
if errorlevel 1 (
    echo Creating conda environment 'ml_gui_env'...
    conda create -n ml_gui_env python=3.9 -y
    if errorlevel 1 (
        echo Error: Failed to create conda environment
        pause
        exit /b 1
    )
)

REM Activate conda environment
echo Activating conda environment...
call conda activate ml_gui_env

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
echo Starting the application...
echo The GUI will open in your default web browser
echo Press Ctrl+C to stop the application
echo.

python classifier.py

echo.
echo Application stopped.
pause