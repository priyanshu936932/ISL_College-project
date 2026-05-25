#!/bin/bash

# ISL Recognition Web App Launcher
# This script starts the backend server and opens the web app

echo "=========================================="
echo "  ISL Recognition Web Application"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import flask, flask_cors, numpy, cv2, tensorflow, mediapipe" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing."
    echo "Installing required packages..."
    pip3 install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please run:"
        echo "   pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "✅ All dependencies installed"
echo ""

# Start the backend server in background
echo "Starting backend server..."
python3 server.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Backend server started (PID: $SERVER_PID)"
    echo "   Server URL: http://localhost:5000"
else
    echo "❌ Failed to start backend server"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Opening web application..."
echo "=========================================="
echo ""

# Open the web app in default browser
if command -v open &> /dev/null; then
    # macOS
    open index.html
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open index.html
elif command -v start &> /dev/null; then
    # Windows
    start index.html
else
    echo "Please open index.html manually in your browser"
fi

echo ""
echo "✅ Application is ready!"
echo ""
echo "📖 Instructions:"
echo "   1. Click 'Start Camera' to begin"
echo "   2. Click 'Record Gesture' or press Spacebar"
echo "   3. Perform your sign language gesture"
echo "   4. View the prediction results"
echo ""
echo "⌨️  Keyboard shortcuts:"
echo "   Spacebar - Start/Stop recording"
echo "   Escape   - Stop camera"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""

# Wait for user to stop
trap "echo ''; echo 'Stopping server...'; kill $SERVER_PID 2>/dev/null; echo '✅ Server stopped'; exit 0" INT

# Keep script running
wait $SERVER_PID
