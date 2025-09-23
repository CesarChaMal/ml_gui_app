# Machine Learning Image Classifier GUI Application
# Built with Taipy GUI framework for CIFAR-10 dataset classification

# Import required libraries
from taipy.gui import Gui          # Web GUI framework
from tensorflow.keras import models # Pre-trained model loading
from PIL import Image              # Image processing
import numpy as np                 # Numerical operations

# CIFAR-10 class mapping - converts model output indices to readable labels
class_names = {
    0: 'airplane',
    1: 'automobile', 
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# Load pre-trained CNN model and display architecture summary
model = models.load_model("baseline_mariya.keras")
model.summary()

def predict_image(model, path_to_img):
    """
    Process uploaded image and make prediction using trained model
    
    Args:
        model: Loaded Keras model
        path_to_img: File path to uploaded image
        
    Returns:
        tuple: (confidence_probability, predicted_class_name)
    """
    # Load and preprocess image
    img = Image.open(path_to_img)     # Open image file
    img = img.convert("RGB")          # Ensure RGB format (3 channels)
    img = img.resize((32, 32))        # Resize to model input size (32x32)
    data = np.asarray(img)            # Convert to numpy array
    data = data / 255                 # Normalize pixel values to 0-1 range
    
    # Make prediction
    probs = model.predict(np.array([data])[:1])  # Get prediction probabilities
    
    # Extract results
    top_prob = probs.max()                       # Highest confidence score
    top_pred = class_names[np.argmax(probs)]     # Corresponding class name
    
    return top_prob, top_pred

# Initialize GUI state variables
content = ""                          # File selector content
img_path = "placeholder_image.png"    # Default image path
prob = 0                             # Prediction confidence (0-100)
pred = ""                            # Prediction text

# Taipy GUI page layout using markdown-like syntax
index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    """
    Callback function triggered when GUI state changes
    Handles file upload and updates prediction results
    
    Args:
        state: Current GUI state object
        var_name: Name of changed variable
        var_val: New value of changed variable
    """
    # Process new image upload
    if var_name == "content":
        # Get prediction from uploaded image
        top_prob, top_pred = predict_image(model, var_val)
        
        # Update GUI state with results
        state.prob = round(top_prob * 100)      # Convert to percentage
        state.pred = "this is a " + top_pred    # Format prediction text
        state.img_path = var_val                # Update displayed image path

# Create and run Taipy GUI application
app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)  # Enable hot reload for development