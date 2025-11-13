# CLAUDE.md - AI Assistant Guide

This document provides comprehensive guidance for AI assistants working with the Machine Learning GUI App codebase.

Last Updated: 2025-11-13

---

## Project Overview

**Purpose**: Educational machine learning project providing a beginner-friendly web-based image classifier with real-time predictions for CIFAR-10 dataset categories.

**Primary Functionality**:
- Web-based GUI for uploading images and receiving instant classification predictions
- Educational Jupyter notebooks for training custom CNN models
- Pre-trained model ready for immediate use
- Comprehensive documentation for learners

**Target Audience**: Beginners learning machine learning, computer vision, and GUI development

**Classification Categories**: 10 CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## Codebase Structure

### Directory Layout

```
ml_gui_app/
├── .git/                           # Git repository data
├── .gitignore                      # Git ignore rules
├── README.md                       # Main project documentation
├── THEORY.md                       # ML concepts and CNN theory
├── USAGE.md                        # Detailed usage instructions
├── CLAUDE.md                       # This file - AI assistant guide
│
├── starterFiles/                   # Learning materials (starting point)
│   ├── NeuralNetworkBuilder.ipynb          # Comprehensive training tutorial (259KB)
│   ├── NeuralNetworkQuickBuilder.ipynb     # Quick training workflow (247KB)
│   ├── requirements.txt                     # Python dependencies (214 packages)
│   ├── baseline_mariya.keras                # Pre-trained CNN model (1.9MB)
│   ├── logo.png                            # Application logo
│   ├── placeholder_image.png               # Default placeholder
│   ├── wireframe.png                       # UI wireframe design
│   └── demo_images/                        # 24 test images (PNG)
│
└── finishedProject/                # Complete working implementation
    ├── classifier.py                        # Main GUI application (100 lines)
    ├── NeuralNetworkBuilder.ipynb          # Same as starterFiles version
    ├── NeuralNetworkQuickBuilder.ipynb     # Same as starterFiles version
    ├── requirements.txt                     # Same dependencies
    ├── baseline_mariya.keras                # Pre-trained model
    ├── run.sh                              # Unix/Linux/macOS launcher
    ├── run.bat                             # Windows quick launcher
    ├── run_with_conda.bat                  # Windows Conda launcher
    ├── setup_and_run.bat                   # Windows full setup script
    ├── USAGE.md                            # Project-specific usage guide
    ├── info.txt                            # Conda setup commands
    ├── logo.png                            # Application logo
    ├── placeholder_image.png               # Default placeholder
    ├── wireframe.png                       # UI wireframe
    └── demo_images/                        # Test images
```

### Key Files and Their Purposes

#### Core Application Files

**`finishedProject/classifier.py`** (100 lines)
- Primary application entry point
- Implements Taipy GUI web interface
- Contains prediction logic
- Key components:
  - Line 11-22: `class_names` dictionary mapping indices to CIFAR-10 classes
  - Line 25: Model loading (`baseline_mariya.keras`)
  - Line 28-53: `predict_image()` function (preprocessing + inference)
  - Line 62-75: Taipy GUI layout definition
  - Line 77-95: `on_change()` callback for file uploads
  - Line 98-101: Application initialization and launch

**`finishedProject/baseline_mariya.keras`** (1.9MB)
- Pre-trained CNN model (HDF5 format)
- Architecture: Custom CNN with 158,314 parameters
- Input: 32x32 RGB images
- Output: 10-class probabilities
- Test accuracy: ~65-70%

#### Training Notebooks

**`NeuralNetworkBuilder.ipynb`** (Comprehensive)
- Full educational tutorial with theory
- Step-by-step model building process
- Visualization and analysis tools
- Dataset exploration
- Training with detailed explanations

**`NeuralNetworkQuickBuilder.ipynb`** (Streamlined)
- Fast-track training workflow
- Minimal explanations
- Direct model creation and training
- Quick iteration for experimentation

#### Documentation Files

**`README.md`** (155 lines)
- Project overview and features
- Quick start guide
- Technical stack information
- Model performance metrics
- Troubleshooting guide

**`THEORY.md`** (246 lines)
- Introduction to machine learning
- CNN architecture explanations
- CIFAR-10 dataset details
- Training process theory
- Evaluation metrics
- Further learning resources

**`USAGE.md`** (257 lines, root level)
- Prerequisites and installation
- Running the application
- Development workflow
- Model customization
- Advanced usage examples

**`finishedProject/USAGE.md`** (105 lines)
- Multiple launch methods
- Platform-specific instructions
- Quick troubleshooting
- Features overview

#### Configuration Files

**`requirements.txt`** (214 packages)
- Complete dependency list
- Critical packages:
  - `taipy==3.0.0` (GUI framework)
  - `tensorflow==2.14.0` (ML framework)
  - `keras==2.14.0` (High-level API)
  - `Pillow==10.1.0` (Image processing)
  - `numpy==1.26.2` (Numerical computing)
  - `matplotlib==3.8.1` (Visualization)

**`.gitignore`**
- Excludes: `__pycache__/`, `*.keras`, `*.h5`, `.ipynb_checkpoints`
- Ignores virtual environments and IDE files
- Prevents large model files from being committed

#### Launch Scripts

**`run.sh`** (Unix/Linux/macOS)
- Auto-detects Python command
- Validates environment
- Installs dependencies
- Launches application

**`run.bat`** (Windows quick)
- Validates Python installation
- Checks required files
- Runs classifier

**`setup_and_run.bat`** (Windows full)
- Creates virtual environment
- Upgrades pip
- Installs all requirements
- Launches application

**`run_with_conda.bat`** (Windows Conda)
- Creates Conda environment
- Installs dependencies
- Runs classifier

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|----------|
| **Language** | Python | 3.8+ | Primary development language |
| **GUI Framework** | Taipy | 3.0.0 | Web-based user interface |
| **ML Framework** | TensorFlow | 2.14.0 | Deep learning and model training |
| **Neural Network API** | Keras | 2.14.0 | High-level model definition |
| **Image Processing** | Pillow (PIL) | 10.1.0 | Image loading and manipulation |
| **Numerical Computing** | NumPy | 1.26.2 | Array operations and normalization |
| **Visualization** | Matplotlib | 3.8.1 | Data visualization and plotting |
| **Web Server** | Flask | 3.0.0 | Underlying Taipy web server |
| **Notebooks** | Jupyter | 1.0.0 | Interactive development |

### Key Framework Details

**Taipy GUI**
- Markdown-like syntax for UI definition
- Reactive state management
- Built-in components: file_selector, image, indicator
- Callback-based event handling
- Automatic browser launch

**TensorFlow/Keras**
- Model format: `.keras` (HDF5-based)
- API: Sequential and Functional
- Common layers: Conv2D, MaxPooling2D, Dense, Dropout
- Optimizer: Adam (default)
- Loss function: Categorical crossentropy

---

## Development Workflows

### Running the Application

**Standard Launch**:
```bash
cd finishedProject
python classifier.py
```
- Opens browser at `http://localhost:5000`
- Hot reload enabled (`use_reloader=True`)
- Model summary printed to console

**With Virtual Environment** (Recommended):
```bash
cd finishedProject
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
python classifier.py
```

**Platform-Specific Scripts**:
- Unix/Linux/macOS: `bash run.sh`
- Windows: `run.bat` or `setup_and_run.bat`
- Conda: `run_with_conda.bat`

### Training a New Model

**Quick Training** (NeuralNetworkQuickBuilder.ipynb):
1. Open notebook in Jupyter/JupyterLab
2. Run all cells sequentially
3. Model saved as `baseline_mariya.keras`
4. Copy to `finishedProject/` directory
5. Test in GUI application

**Comprehensive Training** (NeuralNetworkBuilder.ipynb):
1. Follow tutorial sections with explanations
2. Experiment with hyperparameters
3. Visualize training progress
4. Evaluate on test set
5. Save and deploy model

### Model Customization

**Architecture Modifications**:
```python
# Example: Adding more convolutional layers
model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
# ... add more layers
```

**Training Parameters**:
```python
# Adjust epochs, batch size, learning rate
model.fit(
    x_train, y_train,
    epochs=20,              # Number of training iterations
    batch_size=32,          # Samples per gradient update
    validation_split=0.2    # Portion for validation
)
```

**Saving Models**:
```python
# Modern Keras format (recommended)
model.save('my_model.keras')

# Update classifier.py line 25:
model = models.load_model("my_model.keras")
```

### Extending to New Classes

**Steps to Add New Categories**:
1. Prepare labeled dataset (images + labels)
2. Update `class_names` dictionary in `classifier.py`
3. Modify notebook to load new dataset
4. Retrain model with new data
5. Update GUI text/layout if needed
6. Test with new images

---

## Key Conventions

### Code Style

**Python**:
- Comments above code blocks explaining purpose
- Descriptive variable names (e.g., `path_to_img`, `top_prob`)
- Function docstrings with Args and Returns sections
- PEP 8 style (generally followed)
- Line length: Not strictly enforced

**Naming Conventions**:
- Variables: `snake_case` (e.g., `img_path`, `class_names`)
- Functions: `snake_case` (e.g., `predict_image`, `on_change`)
- Constants: `UPPER_CASE` (not used in current code)
- Classes: `PascalCase` (not defined in classifier.py)

### File Naming

- Scripts: `lowercase.py` or `snake_case.py`
- Notebooks: `PascalCase.ipynb`
- Models: `descriptive_name.keras`
- Images: `lowercase.png` or `snake_case.png`
- Documentation: `UPPERCASE.md`

### Model Conventions

**Input Requirements**:
- Format: RGB images (3 channels)
- Size: 32x32 pixels (resized if different)
- Type: PNG files (enforced in GUI)
- Normalization: Pixel values divided by 255 (0-1 range)

**Output Format**:
- 10-element probability array
- Values sum to 1.0
- Index corresponds to `class_names` dictionary
- Highest probability selected as prediction

### GUI Conventions

**State Variables** (Global):
- `content`: File selector content (uploaded file path)
- `img_path`: Currently displayed image path
- `prob`: Prediction confidence (0-100 integer)
- `pred`: Prediction text (string)

**Callback Pattern**:
```python
def on_change(state, var_name, var_val):
    if var_name == "content":
        # Handle file upload
        # Update state.prob, state.pred, state.img_path
```

**Taipy Syntax**:
- `<|{variable}|component|properties|>`
- File selector: `<|{content}|file_selector|extensions=.png|>`
- Image display: `<|{img_path}|image|>`
- Indicator: `<|{prob}|indicator|min=0|max=100|>`

---

## Testing and Quality Assurance

### Current Testing Approach

**Manual Testing**:
- 24 demo images provided in `demo_images/` directory
- Visual inspection of predictions
- Confidence score validation
- GUI interaction testing

**Model Evaluation**:
- Test set: 10,000 CIFAR-10 images
- Metrics: Accuracy (~69-70%), loss
- Confusion matrix visualization (in notebooks)
- Per-class performance analysis

**No Automated Testing**:
- No pytest or unittest files
- No CI/CD pipelines
- No integration tests
- No unit tests for functions

### Quality Assurance Recommendations

**For AI Assistants Implementing Tests**:
1. Create `tests/` directory
2. Add `test_classifier.py` for unit tests:
   - Test `predict_image()` with sample images
   - Test model loading
   - Test image preprocessing pipeline
3. Add `pytest` to requirements.txt
4. Consider integration tests for GUI callbacks

**Manual Testing Checklist**:
- [ ] Application launches without errors
- [ ] Model loads successfully
- [ ] File selector accepts PNG files
- [ ] Predictions displayed correctly
- [ ] Confidence scores in 0-100 range
- [ ] Images displayed properly
- [ ] Hot reload works in development

---

## Common Tasks and Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Upgrade pip (recommended)
pip install --upgrade pip
```

### Running the Application

```bash
# Standard launch
cd finishedProject
python classifier.py

# With specific Python version
python3.8 classifier.py

# Background mode (Linux/macOS)
nohup python classifier.py &

# Check if running
ps aux | grep classifier
```

### Working with Notebooks

```bash
# Install Jupyter (if not in requirements.txt)
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Launch JupyterLab (modern interface)
jupyter lab

# Convert notebook to Python script
jupyter nbconvert --to script NeuralNetworkBuilder.ipynb
```

### Model Management

```bash
# List model files
ls -lh *.keras

# Check model size
du -h baseline_mariya.keras

# Backup model
cp baseline_mariya.keras baseline_mariya_backup.keras

# Remove old models (if cleaning up)
rm old_model.keras
```

### Git Workflows

```bash
# Current branch
git branch

# Create feature branch
git checkout -b feature/new-model

# Stage changes
git add classifier.py

# Commit with message
git commit -m "Update prediction threshold"

# Push to remote
git push -u origin claude/claude-md-mhy2yo0ewqmzki45-018zoe5ZM3nH3HJGS7giKS4L

# View status
git status

# View recent commits
git log --oneline -5
```

### Troubleshooting Commands

```bash
# Check Python version
python --version

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Test model loading
python -c "from tensorflow.keras import models; m = models.load_model('baseline_mariya.keras'); print('Model loaded')"

# Check port availability
netstat -an | grep 5000  # Linux/macOS
netstat -an | findstr 5000  # Windows

# Kill process on port 5000 (if needed)
lsof -ti:5000 | xargs kill -9  # Linux/macOS
```

---

## Important Considerations for AI Assistants

### When Modifying Code

1. **Always Read First**: Use Read tool before editing any file
2. **Preserve Formatting**: Maintain existing indentation and style
3. **Test Changes**: Verify modifications don't break functionality
4. **Document Changes**: Update comments and docstrings
5. **Check Dependencies**: Ensure new imports are in requirements.txt

### File Operations

**Prefer Editing Over Creating**:
- ALWAYS edit existing files rather than creating new ones
- Only create new files if explicitly required
- Never create unnecessary documentation files
- Avoid creating README files unless requested

**Critical Files** (Handle with Care):
- `classifier.py`: Core application logic
- `baseline_mariya.keras`: Pre-trained model (1.9MB)
- `requirements.txt`: Dependency list
- `.gitignore`: Git exclusions

### Model-Related Operations

**Loading Models**:
```python
# Correct approach
from tensorflow.keras import models
model = models.load_model("baseline_mariya.keras")
```

**Prediction Pipeline**:
1. Load image: `Image.open(path)`
2. Convert to RGB: `img.convert("RGB")`
3. Resize: `img.resize((32, 32))`
4. Convert to array: `np.asarray(img)`
5. Normalize: `data / 255`
6. Predict: `model.predict(np.array([data])[:1])`

**Common Issues**:
- Model file not in current directory → Use absolute paths
- Wrong image size → Always resize to 32x32
- Normalization forgotten → Predictions will be incorrect
- RGBA images → Convert to RGB before processing

### GUI Modifications

**State Management**:
- Global variables: `content`, `img_path`, `prob`, `pred`
- Update via `state.variable_name = value`
- Changes automatically trigger re-render

**Adding Components**:
```python
# Example: Add new button
<|button_label|button|on_action=function_name|>

# Example: Add text input
<|{variable}|input|>

# Example: Add slider
<|{variable}|slider|min=0|max=100|>
```

**Callback Handling**:
- All state changes trigger `on_change()`
- Check `var_name` to identify which variable changed
- Update multiple state variables as needed

### Performance Considerations

**Image Processing**:
- PIL (Pillow) is efficient for single images
- For batch processing, consider using NumPy arrays
- Avoid loading large images unnecessarily

**Model Inference**:
- Current inference time: <100ms per image
- Model runs on CPU by default
- GPU acceleration possible with CUDA-enabled TensorFlow

**Memory Usage**:
- Model size in memory: ~2MB
- Keep image sizes small (32x32 after resize)
- Clear large arrays when done

### Security Considerations

**File Uploads**:
- Current implementation: Accepts only `.png` files
- No validation of file contents (could be security risk)
- No size limits enforced (could cause DoS)

**Recommendations for Improvements**:
```python
# Add file size validation
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
if os.path.getsize(path) > MAX_FILE_SIZE:
    raise ValueError("File too large")

# Verify file is actually an image
try:
    img = Image.open(path)
    img.verify()
except:
    raise ValueError("Invalid image file")
```

**Model Security**:
- `.keras` files can execute code during loading
- Only load models from trusted sources
- Consider scanning uploaded models for malicious code

### Documentation Updates

**When to Update Docs**:
- New features added → Update README.md
- API changes → Update code comments
- Architecture changes → Update THEORY.md
- Usage changes → Update USAGE.md
- AI assistant guidance changes → Update CLAUDE.md (this file)

**Documentation Style**:
- Clear, concise language
- Code examples for complex concepts
- Tables for structured information
- Emojis used sparingly (educational context)
- Screenshots/diagrams encouraged

### Common Pitfalls to Avoid

1. **Don't assume file paths**: Always check current directory
2. **Don't skip normalization**: Images must be normalized (0-1)
3. **Don't forget RGB conversion**: Some PNGs have alpha channel
4. **Don't modify global state directly**: Use callback pattern
5. **Don't create files unnecessarily**: Prefer editing existing files
6. **Don't ignore errors**: Handle exceptions gracefully
7. **Don't push large files to Git**: Use .gitignore for models
8. **Don't use absolute paths in code**: Relative paths preferred

### Helpful Debugging Strategies

**Print Debugging**:
```python
# Check image shape
print(f"Image shape: {data.shape}")  # Should be (32, 32, 3)

# Check normalization
print(f"Pixel range: {data.min()} to {data.max()}")  # Should be 0-1

# Check prediction probabilities
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum()}")  # Should be ~1.0
```

**Model Debugging**:
```python
# View model architecture
model.summary()

# Check input/output shapes
print(model.input_shape)   # Should be (None, 32, 32, 3)
print(model.output_shape)  # Should be (None, 10)

# Test with random input
test_input = np.random.rand(1, 32, 32, 3)
test_output = model.predict(test_input)
print(test_output.shape)   # Should be (1, 10)
```

**GUI Debugging**:
```python
# Log state changes
def on_change(state, var_name, var_val):
    print(f"Changed: {var_name} = {var_val}")
    # ... rest of function
```

---

## Quick Reference

### File Paths (Relative to Repository Root)

```
finishedProject/classifier.py           # Main application
finishedProject/baseline_mariya.keras   # Pre-trained model
finishedProject/requirements.txt        # Dependencies
finishedProject/run.sh                  # Unix launcher
finishedProject/run.bat                 # Windows launcher
finishedProject/demo_images/            # Test images
README.md                               # Project overview
THEORY.md                               # ML theory guide
USAGE.md                                # Usage instructions
CLAUDE.md                               # This file
```

### Key Functions (classifier.py)

```python
predict_image(model, path_to_img)  # Line 28-53: Prediction pipeline
on_change(state, var_name, var_val) # Line 77-95: GUI callback
```

### Key Variables (classifier.py)

```python
class_names     # Line 11-22: Index to class name mapping
model           # Line 25: Loaded Keras model
content         # Line 56: File selector content
img_path        # Line 57: Current image path
prob            # Line 58: Prediction confidence (0-100)
pred            # Line 59: Prediction text
```

### Dependencies (Core)

```
taipy==3.0.0
tensorflow==2.14.0
keras==2.14.0
Pillow==10.1.0
numpy==1.26.2
matplotlib==3.8.1
```

### Useful Links

- Live Demo: https://classifier.taipy.cloud/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Taipy Framework: https://github.com/Avaiga/taipy
- TensorFlow: https://github.com/tensorflow/tensorflow

---

## Version History

- **2025-11-13**: Initial creation with comprehensive codebase analysis
  - Documented project structure and architecture
  - Added development workflows and conventions
  - Included troubleshooting and debugging guidance
  - Provided AI assistant best practices

---

## Notes for Future Updates

**Areas to Expand**:
- Add automated testing framework guidance
- Document CI/CD pipeline setup
- Include Docker containerization instructions
- Add performance optimization techniques
- Expand security hardening recommendations
- Document cloud deployment options (AWS, Azure, GCP)
- Add contribution guidelines for community

**Watch for Changes In**:
- Taipy framework updates (breaking changes possible)
- TensorFlow version upgrades
- Python version requirements
- Model architecture improvements
- GUI enhancements

---

**Remember**: This is an educational project. Prioritize code clarity and documentation over complex optimizations. The goal is to help learners understand machine learning concepts and GUI development.