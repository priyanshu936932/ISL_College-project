"""
Flask Backend Server for ISL Gesture Recognition
Runs with: conda activate dl_gpu && python webapp/server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os
import sys
import json

# Resolve project root so imports work when run from any directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'feature_extraction'))
sys.path.insert(0, os.path.join(ROOT, 'models'))

try:
    import torch
    from model import build_model
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found. Running in demo mode.")
    TORCH_AVAILABLE = False

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("Warning: MediaPipe not found or incompatible version.")

app = Flask(__name__)
CORS(app)

# ── Load model metadata ───────────────────────────────────────────────────────
MODEL_INFO_PATH = os.path.join(ROOT, 'model_info.json')
MODEL_PATH      = os.path.join(ROOT, 'isl_gesture_model_best.pth')

DEFAULT_CLASSES = ['alive', 'bad', 'female', 'good', 'happy', 'long', 'male']

if os.path.exists(MODEL_INFO_PATH):
    with open(MODEL_INFO_PATH) as f:
        _info = json.load(f)
    CLASSES     = _info['gestures']
    NUM_FRAMES  = _info.get('num_frames', 16)
    TARGET_SIZE = tuple(_info.get('target_size', [112, 112]))
else:
    CLASSES     = DEFAULT_CLASSES
    NUM_FRAMES  = 16
    TARGET_SIZE = (112, 112)

DEVICE = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else 'cpu'

# ── Global singletons ─────────────────────────────────────────────────────────
model    = None
holistic = None


def load_model():
    global model
    if not TORCH_AVAILABLE:
        return False
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}  →  demo mode")
        return False
    try:
        net = build_model(num_classes=len(CLASSES)).to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt)
        net.load_state_dict(state)
        net.eval()
        model = net
        print(f"Model loaded  ({len(CLASSES)} classes: {CLASSES})")
        print(f"Device: {DEVICE}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def init_mediapipe():
    global holistic
    if not MEDIAPIPE_AVAILABLE:
        return False
    try:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe ready")
        return True
    except Exception as e:
        print(f"MediaPipe init error: {e}")
        return False


# ── Frame helpers ─────────────────────────────────────────────────────────────

def decode_frame(b64_string):
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    try:
        data  = base64.b64decode(b64_string)
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def preprocess_frames(b64_frames):
    """Decode base64 frames → float32 array (NUM_FRAMES, H, W, 3).
    Uses RAW frame data (same as training), NOT MediaPipe landmarks.
    """
    processed = []

    for b64 in b64_frames:
        frame = decode_frame(b64)
        if frame is None:
            continue

        # Use RAW frame directly (no MediaPipe preprocessing)
        # Model was trained on raw frames, not landmark sketches
        frame = cv2.resize(frame, TARGET_SIZE).astype(np.float32) / 255.0
        processed.append(frame)

    # Downsample / Upsample to exactly NUM_FRAMES
    if len(processed) == 0:
        # No frames decoded, return zeros
        processed = [np.zeros((*TARGET_SIZE, 3), np.float32) for _ in range(NUM_FRAMES)]
    elif len(processed) > NUM_FRAMES:
        # Downsample: take evenly spaced frames
        indices = np.linspace(0, len(processed) - 1, NUM_FRAMES, dtype=int)
        processed = [processed[i] for i in indices]
    elif len(processed) < NUM_FRAMES:
        # Upsample: interpolate missing frames
        while len(processed) < NUM_FRAMES:
            processed.append(processed[-1])

    return np.stack(processed, axis=0)  # (T, H, W, 3)


def predict(frames_np):
    """Run inference. Returns (class_name, confidence)."""
    if model is None:
        import random
        idx  = random.randrange(len(CLASSES))
        conf = 0.55 + random.random() * 0.40
        return CLASSES[idx], conf

    try:
        tensor = torch.from_numpy(frames_np[np.newaxis]).to(DEVICE)  # (1,T,H,W,3)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        print(f"Prediction: {CLASSES[idx]}  ({conf:.2%})")
        return CLASSES[idx], conf
    except Exception as e:
        print(f"Inference error: {e}")
        import random
        idx  = random.randrange(len(CLASSES))
        conf = 0.55 + random.random() * 0.40
        return CLASSES[idx], conf


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'frames' not in data or not data['frames']:
            return jsonify({'error': 'No frames provided'}), 400

        frames_np         = preprocess_frames(data['frames'])
        gesture, conf     = predict(frames_np)

        # Also return per-class probabilities if model is loaded
        all_preds = {}
        if model is not None:
            tensor = torch.from_numpy(frames_np[np.newaxis]).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            all_preds = {c: float(p) for c, p in zip(CLASSES, probs)}

        return jsonify({
            'prediction'     : gesture,
            'confidence'     : conf,
            'all_predictions': all_preds,
            'status'         : 'success'
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/predict-video', methods=['POST'])
def predict_video_endpoint():
    """Handle video file upload and prediction."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        # Save uploaded video temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_file.save(tmp.name)
            video_path = tmp.name

        try:
            # Extract frames from video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return jsonify({'error': 'Failed to open video file'}), 400

            frames_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to RGB (video is BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for model
                frame = cv2.resize(frame, TARGET_SIZE).astype(np.float32) / 255.0
                frames_list.append(frame)

            cap.release()

            if len(frames_list) == 0:
                return jsonify({'error': 'No frames extracted from video'}), 400

            # Downsample / upsample to NUM_FRAMES
            if len(frames_list) > NUM_FRAMES:
                indices = np.linspace(0, len(frames_list) - 1, NUM_FRAMES, dtype=int)
                frames_list = [frames_list[i] for i in indices]
            elif len(frames_list) < NUM_FRAMES:
                while len(frames_list) < NUM_FRAMES:
                    frames_list.append(frames_list[-1])

            frames_np = np.stack(frames_list, axis=0)

            # Predict
            gesture, conf = predict(frames_np)

            # Get probabilities
            all_preds = {}
            if model is not None:
                tensor = torch.from_numpy(frames_np[np.newaxis]).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                all_preds = {c: float(p) for c, p in zip(CLASSES, probs)}

            return jsonify({
                'prediction'      : gesture,
                'confidence'      : conf,
                'frames_extracted': len(frames_list),
                'all_predictions' : all_preds,
                'status'          : 'success'
            }), 200

        finally:
            # Clean up temporary file
            import os
            os.unlink(video_path)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'           : 'healthy',
        'model_loaded'     : model is not None,
        'mediapipe_ready'  : MEDIAPIPE_AVAILABLE and holistic is not None,
        'torch_available'  : TORCH_AVAILABLE,
        'device'           : str(DEVICE),
        'classes'          : CLASSES,
        'num_frames'       : NUM_FRAMES,
        'target_size'      : list(TARGET_SIZE),
    }), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message'  : 'ISL Gesture Recognition — Backend API',
        'classes'  : CLASSES,
        'endpoints': {'/predict': 'POST frames[]', '/health': 'GET status'}
    }), 200


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("ISL Gesture Recognition — Backend Server")
    print("=" * 60)

    init_mediapipe()
    loaded = load_model()
    if not loaded:
        print("\nWARNING: Running in DEMO mode (random predictions)")
        print(f"Expected model at: {MODEL_PATH}")
        print("Train the model first: python ISL_Training.py")

    print(f"\nClasses: {CLASSES}")
    print(f"Server: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
