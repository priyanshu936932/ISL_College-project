# 🚀 Quick Start Guide - ISL Recognition Web App

Get up and running in less than 5 minutes!

## 📋 Prerequisites

- ✅ Python 3.8 or higher
- ✅ Web browser (Chrome recommended)
- ✅ Webcam
- ✅ Internet connection (for loading external libraries)

## 🎯 Option 1: Automatic Launch (Recommended)

### For macOS/Linux:
```bash
cd webapp
./run.sh
```

### For Windows:
```cmd
cd webapp
run.bat
```

That's it! The script will:
1. Check Python installation
2. Install dependencies automatically
3. Start the backend server
4. Open the web app in your browser

---

## 🛠️ Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
cd webapp
pip install -r requirements.txt
```

Or install individually:
```bash
pip install flask flask-cors numpy opencv-python tensorflow mediapipe
```

### Step 2: Start Backend Server

```bash
python server.py
```

You should see:
```
ISL Recognition Backend Server
Server starting on http://localhost:5000
```

### Step 3: Open Web Application

**Method A: Direct file**
- Simply double-click `index.html` in your file browser

**Method B: Local server (recommended)**
```bash
# In a new terminal window
cd webapp
python -m http.server 8000
```
Then open: `http://localhost:8000`

---

## 📖 Using the Application

### 1️⃣ Start Camera
Click the **"Start Camera"** button to activate your webcam. You should see:
- Live video feed
- Status changes to "Online"
- FPS counter starts updating

### 2️⃣ Record Gesture
Click **"Record Gesture"** or press **Spacebar** to begin recording:
- Red recording indicator appears
- Frame counter shows progress (0/30 → 30/30)
- Recording stops automatically after 30 frames

### 3️⃣ Perform Sign
While recording is active:
- Position yourself clearly in frame
- Keep hands visible
- Perform your sign language gesture smoothly
- The system captures 30 frames (~1-2 seconds)

### 4️⃣ View Results
After recording completes:
- Prediction appears with confidence score
- Corresponding sign card highlights
- Result is added to prediction history

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Spacebar` | Start/Stop recording gesture |
| `Escape` | Stop camera |

---

## 🎨 Visual Features

### What You'll See:

1. **Animated Background** - Beautiful gradient shapes floating
2. **Live Camera Feed** - Real-time video with MediaPipe landmarks
3. **Prediction Overlay** - Large, clear prediction display with confidence bar
4. **Recording Indicator** - Pulsing red dot when recording
5. **Status Bar** - Real-time FPS, status, and frame count
6. **Sign Cards** - Interactive cards that highlight when detected
7. **Prediction History** - Recent predictions with timestamps

### Landmarks Visualization:
- 🔴 **Red** - Pose landmarks (body)
- 🔵 **Blue** - Left hand landmarks
- 🟢 **Green** - Right hand landmarks

---

## 🔍 Recognized Signs

The system recognizes **4 ISL signs**:

| Sign | Icon | Meaning |
|------|------|---------|
| Morning | 🌅 | Good Morning |
| Afternoon | ☀️ | Good Afternoon |
| Evening | 🌆 | Good Evening |
| Night | 🌙 | Good Night |

---

## ✅ Testing the System

### Quick Test (Demo Mode):
If you don't have a trained model:
1. The app runs in demo mode automatically
2. Predictions are simulated (random)
3. All features work normally
4. Great for testing the UI!

### With Trained Model:
1. Train the model using `main.ipynb` in parent directory
2. Save as `my_sign_model.h5` in project root
3. Restart the backend server
4. Get real predictions!

---

## 🐛 Troubleshooting

### ❌ Camera not working
**Solution:**
- Check browser permissions (allow camera access)
- Close other apps using webcam
- Refresh the page
- Try a different browser

### ❌ Backend connection failed
**Solution:**
- Ensure Flask server is running on port 5000
- Check terminal for error messages
- Verify firewall allows localhost connections

### ❌ Dependencies installation fails
**Solution:**
```bash
# Try upgrading pip first
pip install --upgrade pip

# Then install dependencies
pip install -r requirements.txt

# For TensorFlow issues on macOS with Apple Silicon:
pip install tensorflow-macos tensorflow-metal
```

### ❌ Low FPS / Lag
**Solution:**
- Close unnecessary browser tabs
- Reduce browser window size
- Use Chrome (better WebGL performance)
- Check CPU usage

### ⚠️ Model not found warning
**This is normal!**
- App runs in demo mode
- Predictions are simulated
- Train model to get real predictions

---

## 🎓 Tips for Best Results

### Camera Setup:
- ✅ Good lighting
- ✅ Plain background
- ✅ Keep hands in frame
- ✅ Position camera at chest level
- ❌ Avoid backlighting
- ❌ Don't move too fast

### Performing Signs:
- 🎯 Clear, deliberate movements
- 🎯 Complete the full gesture
- 🎯 Hold the sign briefly
- 🎯 Practice consistency

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Dual-core 2.0 GHz | Quad-core 2.5+ GHz |
| RAM | 4 GB | 8 GB+ |
| Browser | Chrome 90+ | Chrome 100+ |
| Webcam | 480p | 720p+ |
| Internet | For CDN libraries | Stable connection |

---

## 🎯 Next Steps

### For Users:
1. Practice with different lighting conditions
2. Try all four sign gestures
3. Check prediction accuracy
4. Review history of predictions

### For Developers:
1. Train model with more data
2. Add more sign classes
3. Customize UI colors/theme
4. Optimize for mobile devices
5. Add text-to-speech output

---

## 📱 Mobile Support

Currently optimized for desktop. Mobile features coming soon!

For now, mobile users can:
- View the interface (responsive design)
- Limited camera functionality
- Better on tablets than phones

---

## 🆘 Need Help?

1. **Check the full README.md** for detailed documentation
2. **Review server logs** in the terminal
3. **Browser console** for frontend errors (F12)
4. **Test in demo mode** to isolate issues

---

## 🎉 Success Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed successfully
- [ ] Backend server running on port 5000
- [ ] Web app opened in browser
- [ ] Camera permission granted
- [ ] Live video feed visible
- [ ] MediaPipe landmarks showing
- [ ] Recording creates 30 frames
- [ ] Predictions appear with confidence
- [ ] History updates correctly

---

**Ready to recognize signs! 🤟**

For more details, see [README.md](README.md)
