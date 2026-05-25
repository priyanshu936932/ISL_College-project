#!/usr/bin/env python
"""
ISL Gesture Recognition - Setup Verification Script
Run this before training to ensure everything is configured correctly
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.7+"""
    print("\n📌 Python Version Check")
    print("-" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ ERROR: Python 3.7+ required!")
        return False
    print("✅ Python version OK")
    return True

def check_packages():
    """Check if required packages are installed"""
    print("\n📌 Dependencies Check")
    print("-" * 60)
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'mediapipe': 'MediaPipe'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20s} installed")
        except ImportError:
            print(f"❌ {name:20s} NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Install missing packages:")
        print("   pip install -r requirements_ISL.txt")
    
    return all_installed

def check_dataset_structure():
    """Check if video dataset is organized correctly"""
    print("\n📌 Dataset Structure Check")
    print("-" * 60)
    
    expected_gestures = ['alive', 'bad', 'female', 'good', 'happy', 'long', 'male']
    videos_dir = 'videos'
    
    if not os.path.exists(videos_dir):
        print(f"❌ ERROR: '{videos_dir}/' folder not found!")
        print("   Create 'videos/' folder with gesture subfolders")
        return False
    
    print(f"✅ Found '{videos_dir}/' folder")
    
    all_exist = True
    total_videos = 0
    
    for gesture in expected_gestures:
        gesture_path = os.path.join(videos_dir, gesture)
        if os.path.exists(gesture_path):
            video_files = [f for f in os.listdir(gesture_path) 
                          if f.endswith(('.mp4', '.MOV', '.avi', '.mov'))]
            count = len(video_files)
            total_videos += count
            
            if count > 0:
                print(f"✅ {gesture:10s}: {count:3d} videos")
            else:
                print(f"⚠️  {gesture:10s}: NO videos found")
                all_exist = False
        else:
            print(f"❌ {gesture:10s}: Folder missing")
            all_exist = False
    
    print(f"\nTotal videos found: {total_videos}")
    
    if total_videos < 20:
        print("⚠️  WARNING: Very few videos. More data improves accuracy.")
    
    return all_exist and total_videos > 0

def check_model_files():
    """Check if trained models exist"""
    print("\n📌 Model Files Check")
    print("-" * 60)
    
    files_to_check = {
        'models/model.py': 'Model architecture',
        'feature_extraction/feature_extraction.py': 'Feature extraction',
        'mediapipe/mp.py': 'MediaPipe utilities',
        'ISL_Training.py': 'Training script',
        'ISL_Inference.py': 'Inference script'
    }
    
    all_exist = True
    for filepath, description in files_to_check.items():
        if os.path.exists(filepath):
            print(f"✅ {description:30s} ({filepath})")
        else:
            print(f"❌ {description:30s} ({filepath}) - MISSING")
            all_exist = False
    
    return all_exist

def check_disk_space():
    """Check available disk space"""
    print("\n📌 Disk Space Check")
    print("-" * 60)
    
    try:
        import shutil
        disk_space = shutil.disk_usage('.')
        available_gb = disk_space.free / (1024**3)
        
        print(f"Available disk space: {available_gb:.2f} GB")
        
        if available_gb < 2:
            print("⚠️  WARNING: Less than 2GB free. Training may fail.")
            return False
        else:
            print("✅ Sufficient disk space")
            return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def check_gpu():
    """Check if GPU is available"""
    print("\n📌 GPU Check")
    print("-" * 60)
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   Device {i}: {gpu}")
            print("⚡ GPU training enabled - much faster!")
            return True
        else:
            print("⚠️  No GPU detected")
            print("ℹ️  Training will use CPU (slower but still works)")
            return True
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return True

def generate_report():
    """Run all checks and generate report"""
    print("\n" + "="*60)
    print("   ISL GESTURE RECOGNITION - SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_packages),
        ("Dataset Structure", check_dataset_structure),
        ("Model Files", check_model_files),
        ("Disk Space", check_disk_space),
        ("GPU Available", check_gpu)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"⚠️  Error during {check_name}: {e}")
            results[check_name] = False
    
    # Final report
    print("\n" + "="*60)
    print("   VERIFICATION REPORT")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:30s}: {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to train.")
        print("\n🚀 Next steps:")
        print("   python ISL_Training.py")
        print("   or")
        print("   jupyter notebook ISL_Training_Pipeline.ipynb")
        return True
    else:
        print("\n⚠️  Please fix the above issues before training.")
        print("\n📚 For help, see ISL_README.md or QUICKSTART.md")
        return False

def main():
    """Main function"""
    try:
        success = generate_report()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
