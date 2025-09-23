# Machine Learning GUI App

🚀 **A beginner-friendly machine learning image classifier with an intuitive web interface**

Classify images of animals and vehicles using a trained Convolutional Neural Network (CNN) with real-time predictions and confidence scores.

![Demo Screenshot](https://github.com/MariyaSha/ml_gui_app/assets/32107652/4925650b-9ee5-4b55-ab7c-415b772762c1)

## 🎯 Features

- **Web-based GUI** built with Taipy framework
- **Real-time image classification** for 10 CIFAR-10 categories
- **Confidence scoring** with visual indicators
- **Drag-and-drop** image upload
- **Educational notebooks** for model training and customization
- **Pre-trained model** ready to use

## 🏗️ Project Structure

```
ml_gui_app/
├── finishedProject/
│   ├── classifier.py                    # 🖥️ Main GUI application
│   ├── NeuralNetworkBuilder.ipynb       # 📚 Comprehensive training guide
│   ├── NeuralNetworkQuickBuilder.ipynb  # ⚡ Quick training notebook
│   └── baseline_mariya.keras            # 🧠 Pre-trained CNN model
├── THEORY.md                            # 🎓 ML concepts and theory
├── USAGE.md                             # 📖 Detailed usage instructions
└── README.md                            # 📋 This overview
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ml_gui_app

# Install dependencies
pip install taipy tensorflow pillow numpy matplotlib
```

### Run Application
```bash
cd finishedProject
python classifier.py
```

The web interface opens automatically at `http://localhost:5000`

## 🎮 How to Use

1. **Launch** the application
2. **Upload** a PNG image using the file selector
3. **View** the prediction and confidence score
4. **Experiment** with different images!

### Supported Classes
✈️ Airplane • 🚗 Automobile • 🐦 Bird • 🐱 Cat • 🦌 Deer • 🐕 Dog • 🐸 Frog • 🐎 Horse • 🚢 Ship • 🚛 Truck

## 📚 Learning Resources

### For Beginners
- **[USAGE.md](USAGE.md)** - Step-by-step usage guide
- **NeuralNetworkQuickBuilder.ipynb** - Fast-track model training

### For Deep Learning
- **[THEORY.md](THEORY.md)** - Comprehensive ML theory guide
- **NeuralNetworkBuilder.ipynb** - Detailed training with explanations

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|----------|
| **GUI Framework** | [Taipy](https://github.com/Avaiga/taipy) | Web-based user interface |
| **ML Framework** | [TensorFlow/Keras](https://github.com/tensorflow) | Neural network training & inference |
| **Image Processing** | [Pillow](https://github.com/python-pillow/Pillow) | Image loading and preprocessing |
| **Numerical Computing** | [NumPy](https://github.com/numpy/numpy) | Array operations and normalization |
| **Visualization** | [Matplotlib](https://matplotlib.org/) | Training plots and data visualization |

## 🎯 Model Performance

- **Architecture**: Custom CNN with 158K parameters
- **Dataset**: CIFAR-10 (60,000 images, 10 classes)
- **Accuracy**: ~65-70% on test set
- **Input Size**: 32x32 RGB images
- **Inference Time**: <100ms per image

## 🔧 Customization

### Train Your Own Model
```python
# Modify architecture in notebook
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
# ... add more layers

# Train with custom parameters
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Save and use in GUI
model.save('my_custom_model.keras')
```

### Extend to New Classes
1. Prepare labeled dataset
2. Update `class_names` dictionary
3. Retrain model with new data
4. Update GUI accordingly

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure you're in `finishedProject/` directory |
| Import errors | Run `pip install -r requirements.txt` |
| Low accuracy | Use clear images matching CIFAR-10 classes |
| GUI not loading | Check port 5000 availability |

## 🌟 Live Demo

Try the application online: [classifier.taipy.cloud](https://classifier.taipy.cloud/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Credits

- **Neural Pattern**: [Freepik](https://freepik.com)
- **Brain Icon**: [Flaticon](https://flaticon.com)
- **CIFAR-10 Dataset**: [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Taipy Framework**: [Avaiga](https://github.com/Avaiga/taipy)

## 📞 Support

For questions, issues, or contributions:
- 📖 Check [USAGE.md](USAGE.md) for detailed instructions
- 🎓 Review [THEORY.md](THEORY.md) for ML concepts
- 💡 Open an issue for bugs or feature requests
- 🌟 Star the repository if you find it helpful!

---

**Happy Learning! 🎉**
