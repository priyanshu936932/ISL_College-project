@echo off
REM ISL Recognition Web App Launcher for Windows

echo ==========================================
echo   ISL Recognition Web Application
echo ==========================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo + Python found
python --version
echo.

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import flask, flask_cors, numpy, cv2, tensorflow, mediapipe" >nul 2>nul

if %errorlevel% neq 0 (
    echo ! Some dependencies are missing.
    echo Installing required packages...
    pip install -r requirements.txt

    if %errorlevel% neq 0 (
        echo X Failed to install dependencies. Please run:
        echo    pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo + All dependencies installed
echo.

REM Start the backend server
echo Starting backend server...
start "ISL Backend Server" python server.py

REM Wait for server to start
timeout /t 3 /nobreak >nul

echo + Backend server started
echo   Server URL: http://localhost:5000
echo.

echo ==========================================
echo   Opening web application...
echo ==========================================
echo.

REM Open the web app in default browser
start index.html

echo.
echo + Application is ready!
echo.
echo Instructions:
echo    1. Click 'Start Camera' to begin
echo    2. Click 'Record Gesture' or press Spacebar
echo    3. Perform your sign language gesture
echo    4. View the prediction results
echo.
echo Keyboard shortcuts:
echo    Spacebar - Start/Stop recording
echo    Escape   - Stop camera
echo.
echo To stop the server, close the backend server window
echo.
pause
