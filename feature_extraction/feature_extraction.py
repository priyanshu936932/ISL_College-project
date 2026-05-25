import cv2
import os
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".MOV", ".avi", ".mov", ".MP4", ".AVI")

# Kinetics-400 normalisation — required by the pretrained R(2+1)D-18 backbone
_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD  = np.array([0.22803,  0.22145, 0.216989], dtype=np.float32)


def extract_video_frames(video_path, num_frames=16, target_size=(112, 112)):
    """
    Extract `num_frames` evenly-spaced frames from a video file.
    Returns float32 array of shape (num_frames, H, W, 3) normalised with
    Kinetics-400 mean/std (required for the pretrained R2+1D-18 backbone).
    Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    # Sample evenly-spaced frame indices
    indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int))

    frames = []
    count  = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count in indices:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - _MEAN) / _STD   # Kinetics normalisation
            frames.append(frame)
        count += 1

    cap.release()

    # Pad with zeros if video is shorter than num_frames
    while len(frames) < num_frames:
        frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float32))

    return np.array(frames[:num_frames], dtype=np.float32)  # (T, H, W, 3)


def get_video_paths_and_labels(videos_dir):
    """
    Scan the videos directory and return paths, integer labels, and class names.
    Does NOT load any video data — safe to call on large datasets.

    Returns: (paths, labels, class_names)
      paths       — list of absolute video file paths
      labels      — list of integer class indices (matching class_names)
      class_names — sorted list of gesture folder names
    """
    class_names = sorted([
        name for name in os.listdir(videos_dir)
        if os.path.isdir(os.path.join(videos_dir, name)) and not name.startswith('.')
    ])

    paths, labels = [], []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(videos_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(VIDEO_EXTENSIONS):
                paths.append(os.path.join(class_dir, fname))
                labels.append(idx)

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Total videos found: {len(paths)}")
    return paths, labels, class_names
