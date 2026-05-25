# 🤟 ISL Recognition Web Application

A beautiful, modern web application for real-time Indian Sign Language (ISL) recognition using AI.

## ✨ Features

- **Real-time Camera Feed** - Live webcam integration with smooth video processing
- **MediaPipe Integration** - Advanced hand and pose landmark detection
- **AI-Powered Recognition** - Deep learning model for accurate sign classification
- **Beautiful UI/UX** - Modern, responsive design with smooth animations
- **Prediction History** - Track recent predictions with confidence scores
- **Keyboard Shortcuts** - Quick controls for better user experience

## 🎯 Recognized Signs

The system can recognize four basic ISL signs:
- 🌅 Morning (Good Morning)
- ☀️ Afternoon (Good Afternoon)
- 🌆 Evening (Good Evening)
- 🌙 Night (Good Night)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Webcam

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install flask flask-cors numpy opencv-python tensorflow mediapipe
   ```

2. **Start the backend server:**
   ```bash
   cd webapp
   python server.py
   ```

   The server will start on `http://localhost:5000`

3. **Open the web application:**
   - Simply open `index.html` in your web browser
   - Or use a simple HTTP server:
     ```bash
     python -m http.server 8000
     ```
   - Then navigate to `http://localhost:8000`

## 📖 How to Use

1. **Start Camera** - Click the "Start Camera" button to activate your webcam
2. **Position Yourself** - Make sure your hands are clearly visible in the frame
3. **Record Gesture** - Click "Record Gesture" or press `Spacebar` to start recording
4. **Perform Sign** - Execute your sign language gesture while recording (30 frames)
5. **View Results** - The AI will analyze and display the predicted sign with confidence

## ⌨️ Keyboard Shortcuts

- `Spacebar` - Start/Stop recording gesture
- `Escape` - Stop camera

## 🛠️ Technical Details

### Frontend
- **HTML5** - Semantic markup with modern features
- **CSS3** - Advanced styling with animations and gradients
- **JavaScript (ES6+)** - Async/await, modern APIs
- **MediaPipe** - Real-time hand and pose tracking
- **TensorFlow.js** - Optional client-side ML support

### Backend
- **Flask** - Lightweight Python web framework
- **OpenCV** - Image processing and computer vision
- **TensorFlow/Keras** - Deep learning model inference
- **MediaPipe** - Landmark extraction from video frames
- **NumPy** - Numerical computations

### Model Architecture
- **Type**: 3D Spatiotemporal CNN
- **Input**: 30 frames of 224x224 RGB images
- **Output**: Classification into 4 sign categories
- **Training**: Based on skeleton videos from MediaPipe landmarks

## 📁 File Structure

```
webapp/
├── index.html          # Main HTML file
├── style.css           # Styling and animations
├── app.js              # Frontend JavaScript logic
├── server.py           # Flask backend server
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## 🔧 Configuration

### Backend API URL

If you're running the backend on a different port or host, update the `API_URL` in `app.js`:

```javascript
const API_URL = 'http://localhost:5000/predict';
```

### Model Path

The backend looks for the trained model at `../my_sign_model.h5`. Update `MODEL_PATH` in `server.py` if your model is elsewhere:

```python
MODEL_PATH = '../my_sign_model.h5'
```

## 🎨 Customization

### Adding More Signs

1. Train your model with additional sign classes
2. Update the `SIGNS` array in `server.py`
3. Add corresponding icons in `app.js` (`signIcons` object)
4. Add sign cards in `index.html`

### Changing Theme Colors

Edit the CSS variables and gradients in `style.css`:
- Background: `.bg-animation` and `.shape` classes
- Primary color: Gradient in buttons and headers
- Accent colors: Card borders and highlights

## 🐛 Troubleshooting

### Camera not working
- Check browser permissions for camera access
- Ensure no other application is using the webcam
- Try refreshing the page

### Backend connection failed
- Verify the Flask server is running
- Check the console for error messages
- Ensure CORS is enabled in `server.py`

### Low FPS or lag
- Close unnecessary browser tabs
- Reduce MediaPipe model complexity
- Use a more powerful computer

### Model not found
- The app will run in "demo mode" with simulated predictions
- Train the model using `main.ipynb` in the parent directory
- Save the model as `my_sign_model.h5`

## 🌟 Demo Mode

If the trained model is not available, the application runs in demo mode:
- Backend simulates predictions with random signs
- Confidence scores are randomly generated (75-99%)
- All UI features work normally
- Great for testing the interface

## 📱 Browser Compatibility

- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari
- ❌ Internet Explorer (Not supported)

## 🔒 Privacy

- All video processing happens locally
- Frames are sent to localhost backend only
- No data is stored or transmitted externally
- Camera feed is not recorded or saved

## 📝 License

This project is developed for educational purposes as a college project.

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## 📧 Contact

For questions or feedback, please reach out to the development team.

---

**Powered by TensorFlow & MediaPipe** | Made with ❤️
