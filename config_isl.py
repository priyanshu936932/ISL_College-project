# ISL Gesture Recognition Configuration

import json
import os

# Dataset Configuration
DATASET_CONFIG = {
    "video_folder": "videos",
    "gestures": ["alive", "bad", "female", "good", "happy", "long", "male"],
    "num_classes": 7,
    "video_extensions": [".mp4", ".MOV", ".avi", ".mov"]
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    "num_frames": 30,
    "frame_height": 224,
    "frame_width": 224,
    "normalize": True,
    "output_features": "X.npy",
    "output_labels": "y.npy"
}

# Model Configuration
MODEL_CONFIG = {
    "input_shape": (30, 224, 224, 3),
    "num_classes": 7,
    "architecture": "3D_CNN",
    "optimizer": {
        "type": "Adam",
        "learning_rate": 0.0001
    },
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"]
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 4,
    "epochs": 100,
    "validation_split": 0.1,
    "test_split": 0.2,
    "random_state": 42,
    "verbose": 1,
    "callbacks": {
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 10,
            "restore_best_weights": True
        },
        "reduce_lr": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-7
        },
        "model_checkpoint": {
            "monitor": "val_accuracy",
            "save_best_only": True
        }
    }
}

# MediaPipe Configuration
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 1,
    "enable_segmentation": False,
    "refine_face_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

# Output Configuration
OUTPUT_CONFIG = {
    "model_file": "isl_gesture_model_final.h5",
    "best_model_file": "isl_gesture_model_best.h5",
    "model_info_file": "model_info.json",
    "training_history_file": "training_history.png",
    "confusion_matrix_file": "confusion_matrix.png",
    "features_file": "X.npy",
    "labels_file": "y.npy"
}

# Data Split Configuration
DATA_SPLIT_CONFIG = {
    "train_percentage": 0.70,
    "validation_percentage": 0.10,
    "test_percentage": 0.20
}

def save_config(filename="config_isl.json"):
    """Save configuration to JSON file."""
    config = {
        "dataset": DATASET_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "mediapipe": MEDIAPIPE_CONFIG,
        "output": OUTPUT_CONFIG,
        "data_split": DATA_SPLIT_CONFIG
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filename}")

def load_config(filename="config_isl.json"):
    """Load configuration from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"Config file {filename} not found. Using defaults.")
        return None

if __name__ == "__main__":
    save_config()
