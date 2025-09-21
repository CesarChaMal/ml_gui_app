# ML GUI Image Classifier - Usage Guide

## Quick Start

### Option 1: Simple Run (Windows)
```bash
run.bat
```

### Option 2: Complete Setup with Virtual Environment (Windows)
```bash
setup_and_run.bat
```

### Option 3: Conda Environment (Windows)
```bash
run_with_conda.bat
```

### Option 4: Unix/Linux/macOS
```bash
chmod +x run.sh
./run.sh
```

## Manual Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-step Installation

1. **Navigate to the project directory:**
   ```bash
   cd finishedProject
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Unix/Linux/macOS: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python classifier.py
   ```

## Using the Application

1. **Access the GUI:** Open your web browser and go to `http://127.0.0.1:5000`

2. **Upload an image:** Click on the file selector and choose a PNG image

3. **View results:** The application will display:
   - The uploaded image
   - Predicted class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
   - Confidence percentage

## Troubleshooting

### Common Issues

**"baseline_mariya.keras not found"**
- Ensure the model file is in the finishedProject directory
- The model file should be included with the project

**"Module not found" errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `python -m pip install --upgrade pip`

**Port already in use**
- Close any other applications using port 5000
- Or modify the port in classifier.py by changing `app.run()` to `app.run(port=5001)`

**Virtual environment issues**
- Deactivate and recreate: `deactivate`, then `python -m venv venv`
- Make sure you're using the correct Python version

### System Requirements
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** ~2GB for dependencies
- **OS:** Windows 10+, macOS 10.14+, or Linux

## Features
- Real-time image classification
- Support for 10 different classes
- Confidence scoring
- User-friendly web interface
- Drag-and-drop image upload

## Model Information
- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 32x32 RGB images
- **Classes:** 10 (CIFAR-10 dataset categories)
- **Framework:** TensorFlow/Keras