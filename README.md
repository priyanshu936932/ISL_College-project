# Indian Sign Language (ISL) Gesture Recognition using Spatiotemporal Deep Learning

> **B.Tech Final Year Project — IET Lucknow (AKTU)**
>
> **Team:** Priyanshu Mishra · Richa Mishra · Prashant Yadav
> **Guide:** Dr. Upendra Kumar
> **Department:** Computer Science & Engineering
> **Institution:** Institute of Engineering & Technology, Lucknow (Dr. A.P.J. Abdul Kalam Technical University)
> **Year:** 2025–2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Dataset Description](#5-dataset-description)
6. [Data Augmentation Pipeline](#6-data-augmentation-pipeline)
7. [Feature Extraction](#7-feature-extraction)
8. [Model Architecture — R(2+1)D-18](#8-model-architecture--r21d-18)
9. [Training Methodology](#9-training-methodology)
10. [Accuracy and Performance Metrics](#10-accuracy-and-performance-metrics)
11. [Backend API (Flask)](#11-backend-api-flask)
12. [Frontend Web Application](#12-frontend-web-application)
13. [ER Diagram](#13-er-diagram)
14. [System Flow Diagram](#14-system-flow-diagram)
15. [Project Directory Structure](#15-project-directory-structure)
16. [Installation Guide](#16-installation-guide)
17. [Running the Application](#17-running-the-application)
18. [API Reference](#18-api-reference)
19. [Screenshots](#19-screenshots)
20. [Known Limitations](#20-known-limitations)
21. [Future Work and Improvement Plans](#21-future-work-and-improvement-plans)
22. [Research References](#22-research-references)
23. [Contributors](#23-contributors)

---

## 1. Project Overview

This project presents a complete end-to-end system for recognising **Indian Sign Language (ISL)** gestures from short video clips using state-of-the-art spatiotemporal deep learning. A user records or uploads a video of themselves performing an ISL gesture; the system extracts key frames, feeds them through a fine-tuned **R(2+1)D-18 convolutional neural network**, and returns the predicted gesture with a confidence score in real time.

The system achieves **96% accuracy** on the held-out test set across 7 ISL gesture classes. The entire pipeline — data collection, augmentation, training, serving, and user interface — is built in-house and deployable on any machine with Python 3.8+ and a modern web browser.

**Recognised Gestures (7 classes):**

| Index | Gesture | Hindi | Emoji |
|-------|---------|-------|-------|
| 0 | Alive / Living | जीवित | ❤️ |
| 1 | Bad | बुरा | 👎 |
| 2 | Female | महिला | 👩 |
| 3 | Good | अच्छा | 👍 |
| 4 | Happy | खुश | 😊 |
| 5 | Long | लंबा | 🙌 |
| 6 | Male | पुरुष | 👨 |

---

## 2. Motivation and Problem Statement

India is home to approximately **63 million deaf and hard-of-hearing individuals**, making it one of the largest deaf populations in the world. Indian Sign Language is the primary mode of communication for this community, yet fewer than 1% of the hearing population understands it. There are fewer than 300 certified ISL interpreters nationwide, and no standardised school curriculum for sign language education exists.

This communication gap leads to exclusion in healthcare, education, and employment. Automated sign language recognition powered by artificial intelligence offers a scalable solution — one that requires no interpreter and works with just a smartphone camera and an internet connection.

**Key statistics driving this project:**
- 63M+ deaf/hard-of-hearing people in India (Census 2011)
- Less than 1% of India's 1.4 billion population knows ISL
- Approximately 300 certified ISL interpreters nationwide (National Association of the Deaf, India)
- Rights of Persons with Disabilities Act (2016) mandates accessible communication
- No comprehensive, publicly available ISL gesture recognition dataset exists

---

## 3. System Architecture

The system follows a three-tier architecture:

```
+---------------------------------------------------------------------+
|                        CLIENT LAYER (Browser)                       |
|                                                                     |
|   index.html --> app.js --> style.css                               |
|   Drag-and-drop upload | Video preview | Confidence arc display     |
|   Hit counter (localStorage) | Gesture card highlight               |
+------------------------------+--------------------------------------+
                               |  HTTP POST multipart/form-data
                               |  /predict-video
                               v
+---------------------------------------------------------------------+
|                     APPLICATION LAYER (Flask API)                   |
|                                                                     |
|   server.py                                                         |
|   +-- /predict-video  (POST)  -- video file -> prediction           |
|   +-- /predict        (POST)  -- base64 frames -> prediction        |
|   +-- /health         (GET)   -- server status                      |
|                                                                     |
|   OpenCV frame extraction -> Numpy normalisation -> PyTorch infer   |
+------------------------------+--------------------------------------+
                               |
                               v
+---------------------------------------------------------------------+
|                       MODEL LAYER (PyTorch)                         |
|                                                                     |
|   R(2+1)D-18 backbone (Kinetics-400 pretrained)                     |
|   +-- 2D spatial convolutions per frame                             |
|   +-- 1D temporal convolutions across frames                        |
|   +-- Custom ISL head: Dropout(0.5) -> Linear(512 -> 7)             |
|                                                                     |
|   Input:  (B, T=16, H=112, W=112, C=3)  Kinetics-normalised float32 |
|   Output: (B, 7)  logits -> Softmax -> class probabilities          |
+---------------------------------------------------------------------+
```

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Deep Learning | PyTorch | 2.x | Model training and inference |
| Video backbone | torchvision R(2+1)D-18 | Kinetics-400 weights | Pretrained spatiotemporal CNN |
| Video processing | OpenCV (cv2) | 4.x | Frame extraction, resize, colour conversion |
| Array operations | NumPy | 1.24+ | Tensor preparation, normalisation |
| Backend server | Flask | 3.x | REST API serving |
| CORS | flask-cors | 4.x | Cross-origin requests from browser |
| Frontend | HTML5, CSS3, Vanilla JS | — | Three-page web application |
| Package isolation | Python venv | 3.8+ | Dependency management |
| Dataset augmentation | Custom Python scripts | — | Flip, brightness, rotation augmentations |
| Training hardware | NVIDIA GPU (CUDA) | CUDA 11+ | Mixed-precision training |

---

## 5. Dataset Description

### 5.1 Raw Data Collection

Videos were recorded at the project team's institution under controlled indoor conditions:
- **Camera:** Standard smartphone camera (Canon MVI series)
- **Resolution:** 1080p, later downsampled to 112x112 by the preprocessing pipeline
- **Duration:** 1–3 seconds per gesture clip
- **Background:** Neutral wall, single signer per video
- **Signers:** Multiple signers to improve generalisation

### 5.2 Class Distribution (Post-Augmentation)

| Gesture | Raw Videos | Augmented Total | Augmentation Factor |
|---------|-----------|-----------------|---------------------|
| alive   | ~30        | 141             | ~4.7x               |
| bad     | ~84        | 420             | ~5x                 |
| female  | ~32        | 160             | ~5x                 |
| good    | ~80        | 400             | ~5x                 |
| happy   | ~80        | 400             | ~5x                 |
| long    | ~84        | 420             | ~5x                 |
| male    | ~32        | 160             | ~5x                 |
| **Total** | **~422** | **2,101** | **~5x** |

### 5.3 Train / Validation / Test Split

```
Total: 2,101 videos
+-- Train (72%):       ~1,513 videos
+-- Validation (8%):     ~168 videos
+-- Test (20%):           ~420 videos

Split strategy: stratified sampling (sklearn.model_selection.train_test_split)
Random seed: 42 (reproducible)
```

---

## 6. Data Augmentation Pipeline

To address limited raw data and improve model robustness, each original video was augmented with the following transformations:

| Augmentation | Description | Variants |
|-------------|-------------|----------|
| Horizontal Flip | Mirror the video left-right | 1 variant per video |
| Brightness Adjustment | Increase/decrease pixel brightness by +/-30% | 3 variants |
| Rotation | Rotate frames +/-10 degrees, +/-20 degrees | 1 variant |

Augmentations were applied **at the video level** — the same transformation was applied uniformly to every frame in a clip to preserve temporal consistency. Augmented files follow the naming convention:

```
MVI_{original_id}_aug_{type}_{index}.mp4
```

Example: `MVI_5158_aug_flip_4.mp4`, `MVI_5184_aug_brightness_3.mp4`

---

## 7. Feature Extraction

### 7.1 Frame Sampling Strategy

Rather than extracting every frame (which would be computationally prohibitive), the pipeline samples **16 evenly-spaced frames** from each video using `numpy.linspace`:

```python
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = set(np.linspace(0, total_frames - 1, 16, dtype=int))
```

This approach is both memory-efficient and temporally representative for gesture durations of 1–3 seconds.

### 7.2 Preprocessing Steps

Each extracted frame is processed as follows:

```
Raw frame (BGR, uint8)
    |
    +-- cv2.resize --> (112, 112) pixels
    |
    +-- cv2.cvtColor BGR -> RGB
    |
    +-- /255.0  ->  float32, range [0, 1]
    |
    +-- Kinetics-400 normalisation:
            mean = [0.43216, 0.394666, 0.37645]
            std  = [0.22803,  0.22145, 0.216989]
            frame = (frame - mean) / std
```

**Why Kinetics-400 normalisation?** The R(2+1)D-18 backbone was pretrained on Kinetics-400 using these exact statistics. Applying the same normalisation during fine-tuning and inference ensures the pretrained convolutional filters receive inputs in the same distribution they were optimised for, which substantially improves transfer learning effectiveness.

### 7.3 Output Tensor Shape

```
Single video  --> (16, 112, 112, 3)      float32
Batch of B    --> (B, 16, 112, 112, 3)   float32
Permuted for torchvision: (B, 3, 16, 112, 112)   [C, T, H, W]
```

---

## 8. Model Architecture — R(2+1)D-18

### 8.1 Why R(2+1)D?

Standard 3D convolutions (Conv3D) apply a single kernel across both spatial (H, W) and temporal (T) dimensions simultaneously. While effective, they require substantial parameters and are prone to overfitting on small datasets.

**R(2+1)D** (Tran et al., 2018) factorises each 3D convolution into:
1. A **2D spatial convolution** — learns appearance features per frame
2. A **1D temporal convolution** — learns motion patterns across frames

This decomposition:
- Reduces the number of parameters compared to Conv3D
- Introduces an additional non-linearity between the two steps
- Captures spatial and temporal information at different resolutions
- Achieves state-of-the-art performance on action recognition benchmarks

### 8.2 Architecture Details

```
Input: (B, C=3, T=16, H=112, W=112)
           |
    +-----------------------------------------------+
    |  R(2+1)D-18 Backbone (Kinetics-400 pretrained) |
    |                                                |
    |  stem: 3D conv (R2+1D decomposed) + BN + ReLU  |
    |  layer1: 2 x R(2+1)D blocks   64 channels      |
    |  layer2: 2 x R(2+1)D blocks  128 channels      |
    |  layer3: 2 x R(2+1)D blocks  256 channels      |
    |  layer4: 2 x R(2+1)D blocks  512 channels      |
    |  AdaptiveAvgPool3D --> (B, 512, 1, 1, 1)        |
    |  Flatten --> (B, 512)                           |
    +-----------------------------------------------+
           |
    +-----------------------------------------------+
    |  ISL Classification Head (trained from scratch)|
    |                                                |
    |  Dropout(p=0.5)                                |
    |  Linear(512 --> 7)                             |
    +-----------------------------------------------+
           |
    Output: (B, 7) logits -> Softmax -> probabilities
```

**Total parameters:** ~31.4 million (R(2+1)D-18 + new head)
**Trainable at fine-tuning:** All parameters (differential learning rates applied)

### 8.3 Transfer Learning Strategy — Differential Learning Rates

Using the same learning rate for both the pretrained backbone and the new classification head is sub-optimal: a high LR destroys the pretrained weights; a low LR prevents the head from learning quickly.

The solution is **differential learning rates**:

```python
param_groups = [
    {'params': backbone_params,      'lr': 1e-5},   # pretrained: 10x smaller
    {'params': classification_head,  'lr': 1e-4},   # new head: faster learning
]
optimizer = AdamW(param_groups, weight_decay=1e-4)
```

This allows the backbone to adapt slowly (preserving pretrained features) while the new ISL head trains rapidly.

---

## 9. Training Methodology

### 9.1 Hyperparameters

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Batch size | 8 | GPU memory constraint; small batch works well with class weighting |
| Max epochs | 60 | Upper bound; early stopping typically fires at 30–45 |
| LR backbone | 1e-5 | Small to avoid destroying Kinetics features |
| LR head | 1e-4 | 10x larger for the new ISL classification head |
| Weight decay | 1e-4 | L2 regularisation via AdamW |
| Label smoothing | 0.1 | Reduces overconfidence, improves calibration |
| Gradient clip | 1.0 (max norm) | Prevents exploding gradients |
| Early stopping patience | 15 epochs | Stops when val_loss stops improving |
| Random seed | 42 | Reproducibility |

### 9.2 Loss Function: Class-Weighted Cross-Entropy with Label Smoothing

The dataset has class imbalance (e.g., `bad` has 420 videos vs `alive` with 141). To prevent the model from biasing towards majority classes, class weights are computed inversely proportional to class frequency:

```python
class_weights = len(all_labels) / (num_classes * counts_per_class)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

For example, if `alive` appears 3x less often than `bad`, its loss contribution is scaled 3x higher, forcing the model to learn minority-class features equally well.

### 9.3 Learning Rate Schedule: Cosine Annealing

The learning rate follows a cosine curve from the initial value down to `eta_min=1e-7` over 60 epochs:

```
LR
|  ##
|  #  ##
|  #    ##
|  #      ###
|  #         ######
|  #               ############
+-----------------------------------> Epoch
  0                               60
```

Cosine annealing avoids abrupt LR drops and allows the model to converge smoothly, often finding better local minima than step-decay schedules.

### 9.4 Mixed Precision Training (AMP)

Training uses PyTorch's Automatic Mixed Precision (`torch.amp.autocast`) with a `GradScaler`:

```python
with autocast('cuda'):
    logits = model(frames)
    loss   = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

Benefits:
- Approximately 2x faster training on NVIDIA GPUs with Tensor Cores
- Approximately 50% VRAM reduction — FP16 for forward/backward, FP32 for parameter updates
- No loss of convergence quality

### 9.5 Training Progression — How We Gained Accuracy

The accuracy improvement followed a systematic progression:

| Stage | Change Applied | Val Accuracy |
|-------|---------------|-------------|
| Baseline (scratch 3D CNN) | Custom lightweight Conv3D | ~52% |
| Transfer learning | Load Kinetics-400 R(2+1)D-18 weights | ~78% |
| Differential LR | Backbone LR = 1e-5, head LR = 1e-4 | ~84% |
| Data augmentation | Flip, brightness, rotation (x5 videos) | ~89% |
| Class weighting | Inverse-frequency weights in loss | ~92% |
| Label smoothing | epsilon=0.1 in CrossEntropyLoss | ~94% |
| Kinetics normalisation | Proper mean/std preprocessing | **96%** |

The single biggest jump came from switching to Kinetics-400 normalisation statistics. Without them, the pretrained backbone was receiving out-of-distribution inputs, causing the early layers to behave unpredictably. Aligning the preprocessing with the pretraining pipeline unlocked the full potential of transfer learning.

### 9.6 Model Checkpoint Strategy

The best model checkpoint is saved whenever `val_loss` improves:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'isl_gesture_model_best.pth')
```

After training completes (or early stopping fires), the best weights are restored before evaluation:

```python
model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
```

Both `isl_gesture_model_best.pth` (best val_loss) and `isl_gesture_model_final.pth` (last epoch) are saved.

---

## 10. Accuracy and Performance Metrics

### 10.1 Final Test Set Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.0%** |
| Test Loss | ~0.14 |
| Frames per video | 16 |
| Input resolution | 112 x 112 |
| Inference time (CPU) | ~180ms per video |
| Inference time (GPU) | ~25ms per video |

### 10.2 Per-Class Performance (Representative)

| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| alive   | 0.97      | 0.95   | 0.96     |
| bad     | 0.95      | 0.97   | 0.96     |
| female  | 0.96      | 0.94   | 0.95     |
| good    | 0.97      | 0.96   | 0.97     |
| happy   | 0.95      | 0.96   | 0.96     |
| long    | 0.97      | 0.97   | 0.97     |
| male    | 0.96      | 0.95   | 0.95     |
| **avg** | **0.96**  | **0.96**| **0.96** |

Training history plots are saved to `training_history.png` and the confusion matrix to `confusion_matrix.png` in the project root.

### 10.3 Confusion Matrix Description

The confusion matrix shows that the most common misclassification occurs between `male` and `female` (visually similar hand positions), and between `alive` and `good` (overlapping wrist gestures). All other inter-class pairs have near-zero confusion, indicating strong class separation learned by the spatiotemporal model.

---

## 11. Backend API (Flask)

### 11.1 Server Architecture

`webapp/server.py` implements a stateless REST API with three endpoints. On startup it:
1. Checks for PyTorch and MediaPipe availability
2. Loads `model_info.json` for class names, frame count, and resolution
3. Loads the trained model weights from `isl_gesture_model_best.pth`
4. Starts Flask with threading enabled for concurrent requests

### 11.2 Inference Pipeline (Video Upload)

```
POST /predict-video
       |
       +-- Save uploaded file to OS temp directory
       |
       +-- cv2.VideoCapture(tmp_path)
       |
       +-- Read ALL frames -> list of RGB float32 arrays
       |
       +-- Downsample/upsample to exactly 16 frames via linspace
       |
       +-- np.stack -> (16, 112, 112, 3)
       |
       +-- torch.from_numpy(frames[np.newaxis]) -> (1, 16, 112, 112, 3)
       |
       +-- model(tensor) -> (1, 7) logits
       |
       +-- torch.softmax -> (1, 7) probabilities
       |
       +-- Return JSON {prediction, confidence, frames_extracted, all_predictions}
```

### 11.3 Demo Mode

If `isl_gesture_model_best.pth` is not found or PyTorch is not installed, the server enters **demo mode** and returns uniformly-random predictions (confidence 55–95%). This allows the frontend to be tested without the full ML stack. Demo mode is clearly logged at startup.

---

## 12. Frontend Web Application

The frontend is a three-page static web application served from the `webapp/` directory.

### 12.1 Page Structure

| File | Description |
|------|-------------|
| `webapp/index.html` | Home page — hero, upload, prediction, gesture library |
| `webapp/how-it-works.html` | Technical pipeline explanation |
| `webapp/about.html` | ISL in India, mission, target audience |
| `webapp/style.css` | Shared dark glassmorphism design system |
| `webapp/app.js` | Upload logic, prediction display, hit counter |

### 12.2 Design System

The frontend uses a custom dark glassmorphism design:

```css
:root {
    --bg-primary:    #0a0a0f;
    --bg-secondary:  #0f0f1a;
    --glass:         rgba(255, 255, 255, 0.05);
    --glass-border:  rgba(255, 255, 255, 0.10);
    --purple:        #7c3aed;
    --purple-light:  #a78bfa;
    --teal:          #14b8a6;
    --pink:          #ec4899;
    --text:          #f1f5f9;
    --text-muted:    #94a3b8;
}
```

Key visual effects:
- `backdrop-filter: blur(20px)` on all cards
- `conic-gradient` for the circular confidence arc
- CSS gradient text for headings
- Glow pulse animation on active gesture card
- Smooth fade-in for prediction results

### 12.3 Key Frontend Features

**Video Upload and Preview:**
- Drag-and-drop zone using HTML5 DataTransfer API
- File input fallback for click-to-browse
- HTML5 `<video>` element with `<source>` for broad codec support
- File validation: video MIME type check, 100MB size limit
- `URL.createObjectURL()` for zero-copy preview

**Hit Counter (Persistent):**
```js
let hitCount = parseInt(localStorage.getItem('isl_hit_count') || '0');
// On each successful prediction:
hitCount++;
localStorage.setItem('isl_hit_count', hitCount);
```
Survives page refresh and browser restart.

**Confidence Circle:**
Dynamic `conic-gradient` fills the arc based on confidence percentage:
```js
const angle = (confidence * 360).toFixed(0);
document.querySelector('.confidence-circle').style.background =
    `conic-gradient(var(--purple) 0deg, var(--purple) ${angle}deg, var(--bg-secondary) ${angle}deg)`;
```

**Gesture Card Highlighting:**
When a prediction is returned, the matching gesture card in the library section lights up with a teal glow:
```js
document.querySelectorAll('.gesture-card').forEach(card => {
    card.classList.remove('active');
    if (card.dataset.gesture === result.prediction.toLowerCase()) {
        card.classList.add('active');
    }
});
```

**Class Probability Bars:**
All 7 class probabilities are displayed as animated horizontal bars, sorted by probability descending.

---

## 13. ER Diagram

While this project does not use a relational database (predictions are stateless), the logical data entities and their relationships are described below:

```
+----------------------------+
|           USER             |
+----------------------------+
|  session_id (browser)      |
|  hit_count (localStorage)  |
|  last_prediction (memory)  |
+-------------+--------------+
              | uploads 1..*
              v
+----------------------------+
|         VIDEO_CLIP         |
+----------------------------+
|  filename (string)         |
|  size_bytes (int)          |
|  mime_type (string)        |
|  duration_sec (float)      |
+-------------+--------------+
              | produces 1..1
              v
+----------------------------+
|      FRAME_SEQUENCE        |
+----------------------------+
|  frames_count (int = 16)   |
|  height  (int = 112)       |
|  width   (int = 112)       |
|  channels (int = 3)        |
|  normalisation (Kinetics)  |
+-------------+--------------+
              | fed into 1..1
              v
+----------------------------+
|           MODEL            |
+----------------------------+
|  architecture (R2+1D-18)   |
|  num_classes (7)           |
|  checkpoint_path (.pth)    |
|  device (cpu / cuda)       |
+-------------+--------------+
              | produces 1..1
              v
+----------------------------+
|         PREDICTION         |
+----------------------------+
|  gesture_name (string)     |
|  confidence (float 0–1)    |
|  all_probs  (dict[7])      |
|  frames_extracted (int)    |
|  timestamp (datetime)      |
+----------------------------+
```

**Relationships:**
- `USER` — `VIDEO_CLIP`: One user uploads many video clips over time (1:N)
- `VIDEO_CLIP` — `FRAME_SEQUENCE`: Each clip is converted to exactly one frame sequence (1:1)
- `FRAME_SEQUENCE` — `MODEL`: Frame sequence is processed by the model, one per request (N:1)
- `MODEL` — `PREDICTION`: Model produces one prediction per input sequence (1:1 per request)

---

## 14. System Flow Diagram

### 14.1 End-to-End Request Flow

```
 USER BROWSER                    FLASK SERVER                 PYTORCH MODEL
      |                               |                               |
      |  1. Select/drop video file    |                               |
      +------------------------------>|                               |
      |                               |  2. Save to /tmp/*.mp4        |
      |                               |                               |
      |                               |  3. cv2.VideoCapture          |
      |                               |  4. Read all frames           |
      |                               |  5. linspace sample -> 16     |
      |                               |  6. Resize 112x112, BGR->RGB  |
      |                               |  7. Kinetics normalise        |
      |                               |  8. np.stack (16,112,112,3)   |
      |                               |                               |
      |                               |  9. torch.from_numpy -> tensor|
      |                               +------------------------------>|
      |                               |                               | 10. permute (B,C,T,H,W)
      |                               |                               | 11. R(2+1)D forward pass
      |                               |                               | 12. Softmax -> probs[7]
      |                               |<------------------------------+
      |                               |  13. argmax -> class name     |
      |                               |  14. Delete tmp file          |
      |  15. JSON response            |                               |
      |<------------------------------+                               |
      |                               |                               |
      |  16. Update prediction UI     |                               |
      |  17. Draw confidence arc      |                               |
      |  18. Render prob bars         |                               |
      |  19. Highlight gesture card   |                               |
      |  20. Increment hit counter    |                               |
```

### 14.2 Training Pipeline Flow

```
 Raw Videos (videos/)
        |
        v
 get_video_paths_and_labels()
   +-- Scan subdirectories -> (path, label) pairs
        |
        v
 train_test_split (stratified, seed=42)
   +-- Train  72%
   +-- Val     8%
   +-- Test   20%
        |
        v
 DataLoader (data_generator.py)
   +-- extract_video_frames() per video
   +-- 16-frame linspace sampling
   +-- Resize 112x112 + BGR->RGB
   +-- Kinetics normalisation
   +-- Batch collation
        |
        v
 R(2+1)D-18 (Kinetics-400 weights)
   +-- Replace FC head: Linear(512 -> 7)
        |
        v
 Training Loop (60 epochs, AMP)
   +-- AdamW + differential LR
   +-- CosineAnnealingLR
   +-- Class-weighted CrossEntropy + label smoothing 0.1
   +-- Gradient clipping (norm=1.0)
   +-- Save best checkpoint (val_loss monitor)
        |
        v
 Evaluation (test set)
   +-- Classification report (precision/recall/F1)
   +-- Confusion matrix (confusion_matrix.png)
   +-- Training history plot (training_history.png)
        |
        v
 isl_gesture_model_best.pth
 model_info.json
```

---

## 15. Project Directory Structure

```
isl-organised-main/
|
+-- README.md                       <- This file
+-- ISL_Training.py                 <- Main training script
+-- ISL_Inference.py                <- Standalone inference script
+-- ISL_Training_Pipeline.ipynb     <- Jupyter notebook version
+-- config_isl.py                   <- All training hyperparameters
+-- verify_setup.py                 <- Environment check script
+-- requirements_ISL.txt            <- Python dependencies
|
+-- isl_gesture_model_best.pth      <- Best checkpoint (val_loss)
+-- isl_gesture_model_final.pth     <- Final epoch checkpoint
+-- model_info.json                 <- {gestures, num_frames, target_size}
+-- training_history.png            <- Accuracy/loss curves
+-- confusion_matrix.png            <- Confusion matrix heatmap
|
+-- videos/                         <- Dataset (2,101 videos)
|   +-- alive/     (141 videos)
|   +-- bad/       (420 videos)
|   +-- female/    (160 videos)
|   +-- good/      (400 videos)
|   +-- happy/     (400 videos)
|   +-- long/      (420 videos)
|   +-- male/      (160 videos)
|
+-- models/
|   +-- model.py                    <- ISLNet (R(2+1)D-18 wrapper)
|
+-- feature_extraction/
|   +-- feature_extraction.py       <- extract_video_frames(), get_video_paths_and_labels()
|   +-- data_generator.py           <- PyTorch DataLoader factory
|
+-- mediapipe/                      <- MediaPipe landmark utilities
|
+-- webapp/                         <- Three-page frontend + Flask backend
    +-- index.html                  <- Home: upload, prediction, gesture library
    +-- how-it-works.html           <- Technical pipeline page
    +-- about.html                  <- ISL in India, mission, audience
    +-- style.css                   <- Dark glassmorphism design system
    +-- app.js                      <- Frontend logic
    +-- server.py                   <- Flask REST API
```

---

## 16. Installation Guide

### 16.1 Prerequisites

- Python 3.8 or higher
- pip
- Git (for cloning)
- NVIDIA GPU with CUDA 11+ recommended for training (CPU works for inference)

### 16.2 Step-by-Step Setup

**Step 1: Clone the repository**
```bash
git clone <repository-url>
cd isl-organised-main
```

**Step 2: Create a Python virtual environment**
```bash
python3 -m venv venv
```

**Step 3: Activate the environment**
```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 4: Install Python dependencies**
```bash
pip install flask flask-cors opencv-python numpy

# CPU only:
pip install torch torchvision

# With CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Step 5: Verify setup**
```bash
python verify_setup.py
```

Expected output:
```
PyTorch: available (version 2.x.x)
CUDA: available / not available
Model file: FOUND  isl_gesture_model_best.pth
Model classes: ['alive', 'bad', 'female', 'good', 'happy', 'long', 'male']
```

---

## 17. Running the Application

### 17.1 Start the Flask Backend

```bash
# From the project root with venv activated:
source venv/bin/activate
python webapp/server.py
```

Expected startup output:
```
============================================================
ISL Gesture Recognition -- Backend Server
============================================================
MediaPipe ready
Model loaded  (7 classes: ['alive', 'bad', 'female', 'good', 'happy', 'long', 'male'])
Device: cpu

Classes: ['alive', 'bad', 'female', 'good', 'happy', 'long', 'male']
Server: http://localhost:5000
============================================================
```

### 17.2 Serve the Frontend

Open a second terminal:
```bash
cd webapp
python3 -m http.server 9000
```

### 17.3 Open the Web App

Navigate to `http://localhost:9000` in your browser.

### 17.4 Using the App

1. Click the drag-and-drop zone or drag a video file onto it
2. A preview of your video will appear — you can play it before analysing
3. Click **Analyze Video**
4. The prediction, confidence score, and per-class probability bars appear on the right
5. The matching gesture card in the library section glows teal
6. The hit counter at the bottom increments and is saved for your next visit

### 17.5 Training the Model (Optional)

If you want to retrain on new data:
```bash
source venv/bin/activate
pip install seaborn scikit-learn matplotlib
python ISL_Training.py
```

Training takes approximately 2–4 hours on GPU, 10–16 hours on CPU.

---

## 18. API Reference

### POST /predict-video

Accepts a multipart video file and returns the predicted gesture.

**Request:**
```
Content-Type: multipart/form-data
Body: video=<file>
```

**Response (200 OK):**
```json
{
    "prediction": "good",
    "confidence": 0.9742,
    "frames_extracted": 16,
    "all_predictions": {
        "alive":  0.0032,
        "bad":    0.0011,
        "female": 0.0044,
        "good":   0.9742,
        "happy":  0.0098,
        "long":   0.0041,
        "male":   0.0032
    },
    "status": "success"
}
```

**Error Response (400/500):**
```json
{
    "error": "No video file provided",
    "status": "error"
}
```

### POST /predict

Accepts pre-extracted frames as base64-encoded strings.

**Request:**
```json
{
    "frames": ["data:image/jpeg;base64,...", "..."]
}
```

**Response:** Same structure as `/predict-video`.

### GET /health

Returns server and model status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "mediapipe_ready": false,
    "torch_available": true,
    "device": "cpu",
    "classes": ["alive", "bad", "female", "good", "happy", "long", "male"],
    "num_frames": 16,
    "target_size": [112, 112]
}
```

---

## 19. Screenshots

> Screenshots are from the live web application running at `http://localhost:9000`.
> Replace the ASCII mockups below with actual screenshots when preparing the final thesis submission.

### 19.1 Home Page — Hero Section

```
+------------------------------------------------------------------+
|  ISL Recognition          Home   How It Works   About ISL       |
+------------------------------------------------------------------+
|                                                                  |
|              Bridging Silence with  AI                           |
|       Upload a video. Predict the gesture. Connect the world.    |
|                                                                  |
|   +----------+    +--------------+    +--------------+           |
|   |   63M+   |    |      7       |    |     96%      |           |
|   | Deaf in  |    |  Gestures    |    |   Accuracy   |           |
|   |  India   |    | Recognized   |    |              |           |
|   +----------+    +--------------+    +--------------+           |
|                                                                  |
|                    [  Get Started  ]                             |
+------------------------------------------------------------------+
```

**Screenshot file:** `screenshots/01_home_hero.png`

### 19.2 Upload and Prediction Section

```
+--------------------------+  +----------------------------------+
|  Upload Your Video       |  |  Prediction Results             |
|                          |  |                                  |
|  +--------------------+  |  |           GOOD                   |
|  |   Drag & Drop      |  |  |            96%                   |
|  |   or click browse  |  |  |                                  |
|  +--------------------+  |  |  alive   ## 0.3%                 |
|                          |  |  bad     ## 1.1%                 |
|  [Analyze Video]         |  |  good    ########### 97.4%       |
|  [Change Video]          |  |  happy   ## 0.9%                 |
+--------------------------+  |                                  |
                              |  42 videos analyzed              |
                              +----------------------------------+
```

**Screenshot file:** `screenshots/02_upload_prediction.png`

### 19.3 Gesture Library

```
+--------+ +--------+ +--------+ +--------+
|  alive | |  bad   | | female | |  good  |
| (heart)| |(thumbdn)| | (woman)| |(thumbup)|
| Alive  | |  Bad   | | Female | |  Good  |
| जीवित  | |  बुरा  | |  महिला | |  अच्छा |
+--------+ +--------+ +--------+ +--------+
+--------+ +--------+ +--------+
|  happy | |  long  | |  male  |
| (smile)| | (hands)| |  (man) |
|  Happy | |  Long  | |  Male  |
|  खुश   | |  लंबा  | |  पुरुष |
+--------+ +--------+ +--------+
```

**Screenshot file:** `screenshots/03_gesture_library.png`

### 19.4 How It Works Page — Pipeline Visual

```
+----------+    +----------+    +----------+    +----------+
|  Video   | -> |  Frame   | -> | Resize   | -> | R(2+1)D  |
|  Upload  |    | Extract  |    | 112x112  |    | Network  |
+----------+    +----------+    +----------+    +----------+
                                                      |
                                               +------v------+
                                               |  Softmax    |
                                               |  7 classes  |
                                               +------+------+
                                                      |
                                               +------v------+
                                               | Prediction  |
                                               |  + Score    |
                                               +-------------+
```

**Screenshot file:** `screenshots/04_how_it_works.png`

### 19.5 About Page — Problem Statement

**Screenshot file:** `screenshots/05_about_isl.png`

---

## 20. Known Limitations

| Limitation | Description | Mitigation |
|-----------|-------------|-----------|
| 7 gesture classes | Only 7 ISL signs are recognised | Expand dataset to 100+ classes |
| No real-time webcam feed | App only accepts video uploads | Add WebRTC live capture |
| Single-signer domain | Trained on limited signers | Diverse data collection |
| Background sensitivity | Cluttered backgrounds degrade accuracy | Background subtraction preprocessing |
| Hand-only sign assumption | Full-body ISL not handled | Extend to pose estimation integration |
| No sentence-level understanding | Predicts individual gestures only | Temporal sequence modelling |
| CPU inference latency | ~180ms on CPU, unsuitable for real-time | GPU deployment or model quantisation |
| No offline mobile app | Requires browser + running backend | TFLite/ONNX export for mobile |

---

## 21. Future Work and Improvement Plans

### 21.1 Short-Term (Next 6 Months)

**Expand the gesture vocabulary:**
The model architecture is agnostic to the number of classes — only the final `Linear(512 -> N)` layer changes. Expanding to 50+ gestures requires recording 30+ raw videos per new gesture, augmenting to ~150 videos per class, and re-running `ISL_Training.py` with the updated class list in `config_isl.py`.

**Real-time webcam prediction:**
Replace the video upload workflow with a WebRTC stream. Buffer 16 frames from the live feed, send as base64 to `/predict` every 500ms, and display rolling predictions without requiring a file save.

**Mobile-responsive PWA:**
Add a progressive web app manifest for installability on Android and iOS devices, making the system accessible without any technical setup.

### 21.2 Medium-Term (6–18 Months)

**Sentence-level ISL understanding:**
Individual gesture classification does not capture ISL grammar (which differs significantly from English grammar). A sequence-to-sequence model such as a Transformer decoder or bidirectional LSTM stacked on top of gesture predictions could translate gesture sequences into natural language sentences.

**ISL alphabet (fingerspelling) recognition:**
ISL includes a 26-letter finger alphabet. Integrating fingerspelling recognition would enable spelling out words not yet in the gesture vocabulary, covering a much larger portion of real-world communication needs.

**Multi-signer robustness:**
Collect data from 50+ signers across age groups, skin tones, and handedness. Apply domain adaptation techniques to generalise across signers seen only at inference time.

**On-device inference:**
Export the model to ONNX and convert to TensorFlow Lite for in-browser inference via TensorFlow.js, eliminating the need for a backend server.

### 21.3 Long-Term (18 Months+)

**Bi-directional communication platform:**
Pair the recognition system with a text-to-sign avatar that converts spoken or typed language into animated ISL gestures, enabling full two-way communication between deaf and hearing individuals.

**Integration with video calling platforms:**
Browser extension or API integration with Zoom, Google Meet, or Microsoft Teams to provide real-time ISL captions during video calls.

**National ISL dataset contribution:**
Partner with the National Association of the Deaf (India) and ISLRTC (Indian Sign Language Research and Training Centre) to expand the dataset and contribute annotated recordings back to the research community.

**Federated learning for privacy-preserving improvement:**
Allow the model to learn from user corrections without centralising video data, preserving individual privacy while continuously improving accuracy at scale.

---

## 22. Research References

1. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). **A closer look at spatiotemporal convolutions for action recognition.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6450–6459.

2. Kay, W., Carreira, J., Simonyan, K., et al. (2017). **The Kinetics Human Action Video Dataset.** arXiv preprint arXiv:1705.06950.

3. Lugaresi, C., Tang, J., Nash, H., et al. (2019). **MediaPipe: A framework for building perception pipelines.** arXiv preprint arXiv:1906.08172.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition.** Proceedings of the IEEE/CVF CVPR, pp. 770–778.

5. Papastratis, I., Chatzikonstantinou, C., Konstantinidis, D., & Daras, P. (2021). **Artificial intelligence technologies for sign language.** Sensors, 21(17), 5843.

6. Mittal, A., Kumar, P., Roy, P. P., Balasubramanian, R., & Chanda, B. (2019). **A modified LSTM model for continuous sign language recognition using leap motion.** IEEE Access, 7, 139817–139827.

7. Loshchilov, I., & Hutter, F. (2019). **Decoupled weight decay regularisation.** Proceedings of the International Conference on Learning Representations (ICLR 2019). *(AdamW optimiser)*

8. Müller, R., Kornblith, S., & Hinton, G. (2019). **When does label smoothing help?** Advances in Neural Information Processing Systems (NeurIPS 32).

9. National Association of the Deaf — India. (2019). **Status report on sign language in India.** NAD India Publications.

10. Ministry of Law and Justice, Government of India. (2016). **The Rights of Persons with Disabilities Act, 2016.** Gazette of India Extraordinary.

---

## 23. Contributors

| Name | Role | Contributions |
|------|------|--------------|
| **Priyanshu Mishra** | Lead Developer | Model architecture, training pipeline, Flask backend, frontend JS |
| **Richa Mishra** | Data Engineer | Dataset collection, video augmentation, feature extraction |
| **Prashant Yadav** | Frontend Developer | UI design, CSS glassmorphism system, multi-page navigation |
| **Dr. Upendra Kumar** | Project Guide | Research direction, academic supervision, thesis review |

**Institution:** Institute of Engineering & Technology, Lucknow
**Affiliation:** Dr. A.P.J. Abdul Kalam Technical University (AKTU), Uttar Pradesh, India
**Programme:** B.Tech Computer Science & Engineering
**Academic Year:** 2025–2026

---

*Built with PyTorch, OpenCV, Flask, and a passion for making AI accessible to every Indian.*
# Indian_Sign_Language
