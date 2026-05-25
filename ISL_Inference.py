"""
ISL Gesture Recognition — Inference and Testing Utilities
Run with:  conda activate dl_gpu && python ISL_Inference.py
"""

import cv2
import numpy as np
import torch
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'feature_extraction'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'models'))

from feature_extraction import extract_video_frames
from model import build_model

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ISLGestureRecognizer:
    """Recognizes ISL gestures from video files or webcam feed."""

    def __init__(self, model_path='isl_gesture_model_final.pth',
                 model_info_path='model_info.json'):

        if not os.path.isabs(model_path):
            model_path = os.path.join(SCRIPT_DIR, model_path)
        if not os.path.isabs(model_info_path):
            model_info_path = os.path.join(SCRIPT_DIR, model_info_path)

        # Load class metadata
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                info = json.load(f)
            self.gestures    = info['gestures']
            self.num_classes = info['num_classes']
            self.num_frames  = info.get('num_frames', 30)
            self.target_size = tuple(info.get('target_size', [64, 64]))
        else:
            print(f"Warning: {model_info_path} not found — using defaults.")
            self.gestures    = ['alive', 'bad', 'female', 'good', 'happy', 'long', 'male']
            self.num_classes = len(self.gestures)
            self.num_frames  = 30
            self.target_size = (64, 64)

        # Load model weights
        self.model = build_model(num_classes=self.num_classes).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # Support both bare state_dict and wrapped checkpoint
        state = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"Model loaded: {model_path}")
        print(f"Device      : {DEVICE}")
        print(f"Gestures    : {', '.join(self.gestures)}")
        print(f"Input       : {self.num_frames} frames @ {self.target_size[0]}x{self.target_size[1]}")

    def predict_video(self, video_path, confidence_threshold=0.5):
        """Predict gesture from a video file."""
        print(f"\nProcessing: {video_path}")

        frames = extract_video_frames(video_path, self.num_frames, self.target_size)
        if frames is None:
            print("Error: could not read video.")
            return None

        # (T, H, W, C) -> (1, T, H, W, C)
        tensor = torch.from_numpy(frames.astype(np.float32)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted  = int(np.argmax(probs))
        confidence = float(probs[predicted])

        print(f"Predicted : {self.gestures[predicted]}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        if confidence < confidence_threshold:
            print("Low confidence — result may be uncertain.")

        print("\nAll predictions:")
        for gesture, prob in zip(self.gestures, probs):
            print(f"  {gesture:15s}: {prob:.4f}")

        return {
            'gesture'        : self.gestures[predicted],
            'confidence'     : confidence,
            'all_predictions': {g: float(p) for g, p in zip(self.gestures, probs)}
        }

    def predict_webcam(self):
        """Real-time gesture recognition from webcam. Press 'q' to quit."""
        print("\nStarting webcam inference (press 'q' to quit)...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        frame_buffer = []
        frame_count  = 0
        last_text    = ""

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = holistic.process(image_rgb)

                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS)

                resized = cv2.resize(frame, self.target_size).astype(np.float32) / 255.0
                frame_buffer.append(resized)
                if len(frame_buffer) > self.num_frames:
                    frame_buffer.pop(0)

                frame_count += 1
                if len(frame_buffer) == self.num_frames and frame_count % self.num_frames == 0:
                    arr    = np.stack(frame_buffer).astype(np.float32)
                    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = self.model(tensor)
                        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    predicted  = int(np.argmax(probs))
                    confidence = float(probs[predicted])
                    last_text  = f"{self.gestures[predicted]} ({confidence:.2f})"

                if last_text:
                    cv2.putText(frame, last_text, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                cv2.imshow('ISL Gesture Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam inference ended.")


def main():
    print("=" * 80)
    print("ISL GESTURE RECOGNITION — INFERENCE")
    print("=" * 80)

    recognizer = ISLGestureRecognizer()

    # Test on a few videos from the first gesture class
    test_dir = os.path.join(SCRIPT_DIR, "videos", recognizer.gestures[0])
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir)
                 if f.lower().endswith(('.mp4', '.mov', '.avi'))][:3]
        for fname in files:
            recognizer.predict_video(os.path.join(test_dir, fname))
            print("-" * 60)
    else:
        print(f"No test videos found at {test_dir}")

    # Uncomment to test webcam:
    # recognizer.predict_webcam()


if __name__ == "__main__":
    main()
