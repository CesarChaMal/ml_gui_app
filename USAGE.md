# Usage Guide

## Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see [Installation](#installation))

### Installation
```bash
pip install taipy tensorflow pillow numpy matplotlib
```

### Running the Application
```bash
cd finishedProject
python classifier.py
```

The web interface will open at `http://localhost:5000`

## Using the GUI Application

### Step 1: Launch Application
- Run `python classifier.py`
- Wait for model to load (displays architecture summary)
- Browser opens automatically to the web interface

### Step 2: Upload Image
- Click "Browse" or drag-and-drop a PNG image
- Supported format: PNG files only
- Recommended: Clear images of objects from CIFAR-10 classes

### Step 3: View Results
- **Prediction**: Shows detected object class
- **Confidence**: Percentage indicator (0-100%)
- **Image Preview**: Displays uploaded image

### Supported Classes
The model can classify these 10 categories:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐎 Horse
- 🚢 Ship
- 🚛 Truck

## Development Workflow

### Training Your Own Model

#### Option 1: Quick Training
```bash
jupyter notebook NeuralNetworkQuickBuilder.ipynb
```
- Minimal setup for fast experimentation
- Basic CNN architecture
- Essential training steps

#### Option 2: Comprehensive Training  
```bash
jupyter notebook NeuralNetworkBuilder.ipynb
```
- Detailed explanations and theory
- Data visualization and analysis
- Advanced training techniques
- Model evaluation and testing

### Model Customization

#### Modify Architecture
Edit the `build_model()` function in the notebook:
```python
def build_model():
    model = Sequential()
    # Add your layers here
    model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)))
    # ... more layers
    return model
```

#### Adjust Training Parameters
```python
# Training configuration
epochs = 20           # Number of training iterations
batch_size = 32       # Samples per batch
learning_rate = 0.001 # Optimizer learning rate
```

#### Save Custom Model
```python
model.save('my_custom_model.keras')
```

Then update `classifier.py`:
```python
model = models.load_model("my_custom_model.keras")
```

## File Structure

```
ml_gui_app/
├── finishedProject/
│   ├── classifier.py                    # Main GUI application
│   ├── NeuralNetworkBuilder.ipynb       # Comprehensive training notebook
│   ├── NeuralNetworkQuickBuilder.ipynb  # Quick training notebook
│   ├── baseline_mariya.keras            # Pre-trained model
│   ├── logo.png                         # Application logo
│   └── placeholder_image.png            # Default image
├── THEORY.md                            # ML theory and concepts
├── USAGE.md                             # This usage guide
├── README.md                            # Project overview
└── .gitignore                           # Git ignore rules
```

## Troubleshooting

### Common Issues

#### Model Loading Error
```
Error: No such file or directory: 'baseline_mariya.keras'
```
**Solution**: Ensure you're in the `finishedProject` directory when running the app.

#### Import Error
```
ModuleNotFoundError: No module named 'taipy'
```
**Solution**: Install required packages:
```bash
pip install taipy tensorflow pillow numpy
```

#### Low Prediction Accuracy
**Possible Causes**:
- Image doesn't match CIFAR-10 classes
- Image quality is poor
- Object is not clearly visible

**Solutions**:
- Use clear, well-lit images
- Ensure object fills most of the frame
- Try different angles or images

#### GUI Not Loading
**Check**:
- Port 5000 is not in use
- Firewall allows local connections
- Browser supports modern web standards

### Performance Tips

#### Faster Predictions
- Keep the application running (model stays loaded)
- Use smaller image files
- Close unnecessary browser tabs

#### Better Accuracy
- Use high-quality images
- Ensure good lighting and contrast
- Center the object in the frame
- Avoid cluttered backgrounds

## Advanced Usage

### Batch Processing
For processing multiple images programmatically:

```python
from classifier import predict_image, model

images = ['image1.png', 'image2.png', 'image3.png']
results = []

for img_path in images:
    prob, pred = predict_image(model, img_path)
    results.append({
        'image': img_path,
        'prediction': pred,
        'confidence': round(prob * 100, 2)
    })

print(results)
```

### Custom Preprocessing
Modify the `predict_image` function for different input formats:

```python
def predict_image_custom(model, img_array):
    """Accept numpy array instead of file path"""
    # Ensure correct shape and type
    if img_array.shape != (32, 32, 3):
        img_array = cv2.resize(img_array, (32, 32))
    
    # Normalize
    data = img_array.astype(np.float32) / 255.0
    
    # Predict
    probs = model.predict(np.array([data])[:1])
    return probs.max(), class_names[np.argmax(probs)]
```

### Integration with Other Applications
The classifier can be integrated into larger applications:

```python
# As a module
from classifier import predict_image, model, class_names

# In your application
def classify_user_image(image_path):
    confidence, prediction = predict_image(model, image_path)
    return {
        'class': prediction,
        'confidence': confidence,
        'all_classes': list(class_names.values())
    }
```

## Contributing

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

### Improving the Model
1. Experiment with different architectures
2. Try data augmentation techniques
3. Implement transfer learning
4. Share results and insights

## Support

For questions and issues:
- Check this usage guide
- Review the theory documentation
- Examine the notebook examples
- Test with known working images

## Next Steps

After mastering the basic application:
1. **Experiment** with different model architectures
2. **Learn** about transfer learning and pre-trained models
3. **Explore** other datasets beyond CIFAR-10
4. **Deploy** to cloud platforms for wider access
5. **Integrate** with mobile or web applications