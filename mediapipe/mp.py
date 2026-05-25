import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Holistic and Drawing
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def use_mediapipe(dataset="videos", output_prefix="skeleton"):
    """
    Extract skeleton landmarks from videos using MediaPipe Holistic.
    Processes all video files from organized dataset structure.
    
    Args:
        dataset: Path to dataset folder (default: "videos")
        output_prefix: Prefix for output folder (default: "skeleton")
    """
    
    if not os.path.exists(dataset):
        print(f"Error: Dataset path '{dataset}' not found!")
        return
    
    # Create output directory
    output_root = dataset + "_" + output_prefix
    os.makedirs(output_root, exist_ok=True)
    
    total_videos = 0
    processed_videos = 0
    
    # Iterate through gesture classes
    for className in sorted(os.listdir(dataset)):
        classPath = os.path.join(dataset, className)
        
        if not os.path.isdir(classPath):
            continue
        
        outputPath = os.path.join(output_root, className)
        os.makedirs(outputPath, exist_ok=True)
        
        print(f"\nProcessing class: {className}")
        
        # Iterate through videos
        for video in sorted(os.listdir(classPath)):
            if not video.endswith((".MOV", ".mp4", ".avi", ".mov")):
                continue
            
            total_videos += 1
            input_video_path = os.path.join(classPath, video)
            output_video_path = os.path.join(outputPath, os.path.splitext(video)[0] + "_skeleton.mp4")
            
            try:
                print(f"  Processing: {video}...", end=" ")
                
                # Open video
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    print("❌ Failed to open")
                    continue
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                frame_count = 0
                
                # Initialize Holistic model
                with mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as holistic:
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Create white background for skeleton
                        skeleton_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
                        
                        # Convert BGR to RGB
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process the frame
                        results = holistic.process(image_rgb)
                        
                        # Draw landmarks on skeleton frame
                        mp_drawing.draw_landmarks(skeleton_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                        mp_drawing.draw_landmarks(skeleton_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(skeleton_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(skeleton_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        
                        # Write frame to output video
                        out.write(skeleton_frame)
                        frame_count += 1
                    
                    cap.release()
                    out.release()
                    
                processed_videos += 1
                print(f"✅ ({frame_count} frames)")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
        
        print(f"Class '{className}' completed")
    
    print("\n" + "="*80)
    print(f"✅ MediaPipe skeleton extraction complete!")
    print(f"   Total videos: {total_videos}")
    print(f"   Successfully processed: {processed_videos}")
    print(f"   Output directory: {output_root}")
    print("="*80)


def extract_skeleton_features(video_path, num_frames=30, target_size=(224, 224)):
    """
    Extract frames from skeleton video.
    
    Args:
        video_path: Path to skeleton video
        num_frames: Number of frames to extract
        target_size: Frame size (width, height)
    
    Returns:
        numpy array of frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    
    frames = []
    count = 0
    
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0
            frames.append(frame)
        count += 1
    
    cap.release()
    
    # Pad if fewer frames
    while len(frames) < num_frames:
        frames.append(np.zeros((target_size[0], target_size[1], 3)))
    
    return np.array(frames[:num_frames])


if __name__ == "__main__":
    # Extract skeleton from all videos
    use_mediapipe(dataset="videos", output_prefix="skeleton")
    print("\n✅ Skeleton extraction complete! Use 'skeleton_videos' folder for training.")



