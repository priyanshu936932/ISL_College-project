# Indian Sign Language (ISL) Gesture Recognition

A deep learning project that recognizes 7 different Indian Sign Language gestures using 3D CNN (Convolutional Neural Networks) on video data.

## 📋 Project Overview

This project uses spatiotemporal CNNs to recognize the following ISL gestures:
- **Alive**: Indian sign for "alive"
- **Bad**: Indian sign for "bad"
- **Female**: Indian sign for "female"
- **Good**: Indian sign for "good"
- **Happy**: Indian sign for "happy"
- **Long**: Indian sign for "long"
- **Male**: Indian sign for "male"

### Dataset
- **Video Format**: MP4 and MOV files
- **Augmentation**: Data augmented with brightness, flip, rotation, and zoom-shift variations
- **Frames per Video**: 30 frames
- **Frame Size**: 224 x 224 pixels
- **Total Videos**: Multiple instances per gesture class

## 🏗️ Project Structure

```
isl-organised-main/
├── main.ipynb                          # Legacy notebook (deprecated)
├── ISL_Training_Pipeline.ipynb         # ✅ Main training notebook
├── ISL_Training.py                     # ✅ Command-line training script
├── ISL_Inference.py                    # ✅ Testing & inference utility
├── feature_extraction/
│   └── feature_extraction.py           # Video frame extraction module
├── mediapipe/
│   ├── mediapipe.py                    # MediaPipe skeleton extraction
│   └── mp.py                           # Updated MediaPipe utilities
├── models/
│   └── model.py                        # ✅ 3D CNN model architecture
├── videos/                             # Video dataset
│   ├── alive/                          # Gesture videos
│   ├── bad/
│   ├── female/
│   ├── good/
│   ├── happy/
│   ├── long/
│   └── male/
└── webapp/                             # Web interface
    ├── server.py                       # Flask server
    ├── app.js                          # Frontend
    └── index.html
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow keras numpy opencv-python scikit-learn matplotlib seaborn mediapipe
```

### Option 1: Run Training Pipeline (Recommended)
Using the Jupyter Notebook (interactive):
```bash
jupyter notebook ISL_Training_Pipeline.ipynb
```

### Option 2: Run Training Script (Command Line)
```bash
python ISL_Training.py
```

The training pipeline will:
1. ✅ Extract frames from all video files
2. ✅ Organize data by gesture class
3. ✅ Split into train/validation/test sets (70/10/20)
4. ✅ Build and train a 3D CNN model
5. ✅ Save the trained model
6. ✅ Generate evaluation metrics and visualizations

## 🔍 Model Architecture

```
3D CNN Model
├── Conv3D Block 1: 32 filters
├── Conv3D Block 2: 64 filters
├── Conv3D Block 3: 128 filters
├── Conv3D Block 4: 256 filters (with Global Average Pooling)
├── Dense Layer 1: 512 units (ReLU)
├── Dense Layer 2: 256 units (ReLU)
└── Output Layer: 7 units (Softmax)

Features:
- Batch Normalization after each Conv3D layer
- Dropout for regularization (0.3-0.5)
- L2 regularization on dense layers
- Early stopping & learning rate reduction callbacks
```

## 📊 Training Details

- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 4
- **Epochs**: Up to 100 (with early stopping)
- **Input Shape**: (30, 224, 224, 3) - 30 frames of 224x224 RGB images

## 📈 Expected Performance

After training on your ISL gesture dataset:
- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-90%
- **Test Accuracy**: 80-90%

*Actual results depend on video quality, lighting conditions, and dataset size*

## 🎯 Testing & Inference

### Test on Video Files
```bash
python ISL_Inference.py
```

Or use the inference utility in Python:
```python
from ISL_Inference import ISLGestureRecognizer

# Initialize
recognizer = ISLGestureRecognizer('isl_gesture_model_final.h5')

# Predict on video
result = recognizer.predict_video('path/to/video.mp4')
print(f"Gesture: {result['gesture']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Real-time Webcam Recognition
```python
recognizer.predict_webcam()  # Press 'q' to quit
```

## 📁 Output Files After Training

- **isl_gesture_model_final.h5**: Final trained model
- **isl_gesture_model_best.h5**: Best model during training (highest validation accuracy)
- **model_info.json**: Model configuration and gesture mapping
- **training_history.png**: Accuracy and loss curves
- **confusion_matrix.png**: Detailed confusion matrix visualization
- **X.npy**: Extracted features (video frames)
- **y.npy**: Corresponding labels

## 🔧 Customization

### Adjust Training Parameters
Edit `ISL_Training_Pipeline.ipynb` or `ISL_Training.py`:

```python
# Change number of epochs
epochs=100  # Increase for more training

# Change batch size
batch_size=4  # Increase for faster but less stable training

# Adjust input frames
NUM_FRAMES = 30  # More frames = more temporal information
```

### Change Model Architecture
Edit `models/model.py`:
- Adjust filter sizes
- Add/remove layers
- Change dropout rates
- Modify regularization

## 📊 Performance Optimization Tips

1. **Increase Video Diversity**: More varied gestures improve generalization
2. **Augmentation**: Use more augmented versions of videos
3. **Longer Training**: Train for more epochs (with early stopping)
4. **Better Quality Videos**: Higher resolution and good lighting
5. **Data Normalization**: Ensure consistent preprocessing

## 🌐 Web Interface

The `webapp` folder contains a Flask web server for real-time gesture recognition:

```bash
cd webapp
python server.py
# Visit http://localhost:5000
```

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
batch_size=2

# Reduce input image size
TARGET_SIZE = (160, 160)

# Reduce number of frames
NUM_FRAMES = 15
```

### Poor Accuracy
- Ensure videos have good lighting and clear hand visibility
- Add more training data
- Try different model architectures
- Increase training epochs
- Check for class imbalance

### Video Processing Slow
- Use pre-extracted features (X.npy, y.npy)
- Reduce frame resolution
- Process videos in parallel

## 📚 References

- [TensorFlow 3D CNN](https://www.tensorflow.org/tutorials/video)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)
- [Sign Language Recognition Literature](https://arxiv.org/search/?query=sign+language+recognition&searchtype=all&abstracts=show&order=-announced_date_first&size=50)

## 📝 License

This project is open source and available for educational and research purposes.

## 👥 Contributing

To improve this project:
1. Add more gesture classes
2. Improve model architecture
3. Optimize for mobile deployment
4. Create better web interface
5. Add more augmentation techniques

## 📧 Support

For issues or questions, please check:
- Video file formats (supported: mp4, MOV, avi)
- Dataset directory structure
- TensorFlow/Keras versions compatibility
- GPU availability for faster training

---

**Note**: Training time depends on your hardware. With GPU: 5-15 minutes. With CPU: 30-60 minutes.
