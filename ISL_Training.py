"""
ISL Gesture Recognition — Training Script
Uses pretrained R(2+1)D-18 (Kinetics-400) fine-tuned on 7-class ISL data.
Expected accuracy: 85-92%.

Run with:
    & "C:\\Users\\CSED\\anaconda3\\envs\\dl_gpu\\python.exe" ISL_Training.py
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'feature_extraction'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'models'))

from feature_extraction import get_video_paths_and_labels
from data_generator import get_dataloader
from model import build_model

# ==================== CONFIG ====================
VIDEOS_DIR  = os.path.join(SCRIPT_DIR, "videos")
NUM_FRAMES  = 16          # matches R(2+1)D-18 pretraining
TARGET_SIZE = (112, 112)  # matches R(2+1)D-18 pretraining
BATCH_SIZE  = 8
EPOCHS      = 60
LR_BACKBONE = 1e-5        # small LR for pretrained backbone (avoid destroying pretrained weights)
LR_HEAD     = 1e-4        # higher LR for new classification head
PATIENCE    = 15          # early stopping patience
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("ISL Gesture Recognition — Pretrained R(2+1)D-18 Fine-tuning")
print("=" * 80)
print(f"Device          : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"Frame resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]},  {NUM_FRAMES} frames")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Max epochs      : {EPOCHS}")
print(f"LR backbone     : {LR_BACKBONE}  |  LR head: {LR_HEAD}")
print("=" * 80)

# ==================== STEP 1: Discover Videos ====================
print("\n[STEP 1] Scanning video dataset...")
if not os.path.exists(VIDEOS_DIR):
    raise FileNotFoundError(f"Videos directory not found: {VIDEOS_DIR}")

all_paths, all_labels, class_names = get_video_paths_and_labels(VIDEOS_DIR)
num_classes = len(class_names)

print(f"\nGesture classes ({num_classes}):")
counts = np.bincount(all_labels, minlength=num_classes)
for i, (name, cnt) in enumerate(zip(class_names, counts)):
    print(f"  {i}: {name:15s}  ({cnt} videos)")

# ==================== STEP 2: Split Paths ====================
print("\n[STEP 2] Splitting into train / val / test  (72% / 8% / 20%)...")
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=RANDOM_SEED
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels,
    test_size=0.10, stratify=train_val_labels, random_state=RANDOM_SEED
)
print(f"Train : {len(train_paths)} | Val : {len(val_paths)} | Test : {len(test_paths)}")

# ==================== STEP 3: Data Loaders ====================
print("\n[STEP 3] Creating data loaders...")
train_loader = get_dataloader(train_paths, train_labels, batch_size=BATCH_SIZE,
                              num_frames=NUM_FRAMES, target_size=TARGET_SIZE, shuffle=True)
val_loader   = get_dataloader(val_paths,   val_labels,   batch_size=BATCH_SIZE,
                              num_frames=NUM_FRAMES, target_size=TARGET_SIZE, shuffle=False)
test_loader  = get_dataloader(test_paths,  test_labels,  batch_size=BATCH_SIZE,
                              num_frames=NUM_FRAMES, target_size=TARGET_SIZE, shuffle=False)
print(f"Train batches : {len(train_loader)}  |  Val batches : {len(val_loader)}")

# ==================== STEP 4: Build Model ====================
print("\n[STEP 4] Loading pretrained R(2+1)D-18  (downloading if needed)...")
model = build_model(num_classes=num_classes).to(DEVICE)

# Differential LR: backbone (pretrained) gets 10x smaller LR than head
param_groups = model.param_groups(lr_backbone=LR_BACKBONE, lr_head=LR_HEAD)
optimizer    = AdamW(param_groups, weight_decay=1e-4)
scheduler    = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# Weighted loss for class imbalance + label smoothing
class_weights = torch.tensor(
    len(all_labels) / (num_classes * counts), dtype=torch.float32
).to(DEVICE)
print("Class weights:", {class_names[i]: f"{class_weights[i]:.2f}" for i in range(num_classes)})
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Mixed precision scaler — halves VRAM usage, speeds training ~2x
scaler = GradScaler('cuda')

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# ==================== STEP 5: Training Loop ====================
print("\n[STEP 5] Training (mixed-precision)...")

best_val_loss = float('inf')
no_improve    = 0
best_path     = os.path.join(SCRIPT_DIR, 'isl_gesture_model_best.pth')
history       = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (frames, labels) in enumerate(loader):
            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(frames)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast('cuda'):
                    logits = model(frames)
                    loss   = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

            if training and (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)}"
                      f"  loss={total_loss/total:.4f}  acc={correct/total:.4f}", flush=True)

    return total_loss / total, correct / total


for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}  (lr_backbone={optimizer.param_groups[0]['lr']:.2e}"
          f"  lr_head={optimizer.param_groups[1]['lr']:.2e})")

    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
          f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), best_path)
        print(f"  Saved best model  (val_loss={val_loss:.4f}  val_acc={val_acc:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
            break

# Load best weights
model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
print("Best model weights restored.")

# ==================== STEP 6: Save Final Model & Metadata ====================
print("\n[STEP 6] Saving model and metadata...")
final_path = os.path.join(SCRIPT_DIR, 'isl_gesture_model_final.pth')
info_path  = os.path.join(SCRIPT_DIR, 'model_info.json')

torch.save(model.state_dict(), final_path)
with open(info_path, 'w') as f:
    json.dump({'num_classes': num_classes, 'gestures': class_names,
               'num_frames': NUM_FRAMES, 'target_size': list(TARGET_SIZE)}, f, indent=2)
print(f"Saved: {final_path}")
print(f"Saved: {info_path}")

# ==================== STEP 7: Test Evaluation ====================
print("\n[STEP 7] Evaluating on test set...")
test_loss, test_acc = run_epoch(test_loader, training=False)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

# ==================== STEP 8: Classification Report & Confusion Matrix ====================
print("\n[STEP 8] Detailed metrics...")
model.eval()
y_true_list, y_pred_list = [], []
with torch.no_grad():
    for frames, labels in test_loader:
        with autocast('cuda'):
            logits = model(frames.to(DEVICE))
        y_pred_list.extend(logits.argmax(1).cpu().numpy())
        y_true_list.extend(labels.numpy())

y_true, y_pred = np.array(y_true_list), np.array(y_pred_list)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — ISL Gesture Recognition (R2+1D-18)')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
cm_path = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved: {cm_path}")

# ==================== STEP 9: Training History ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_acc'], label='Train', linewidth=2)
axes[0].plot(history['val_acc'],   label='Val',   linewidth=2)
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['train_loss'], label='Train', linewidth=2)
axes[1].plot(history['val_loss'],   label='Val',   linewidth=2)
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('R(2+1)D-18 Fine-tuned on ISL — Training History', fontsize=13)
plt.tight_layout()
hist_path = os.path.join(SCRIPT_DIR, 'training_history.png')
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
print(f"Training history saved: {hist_path}")

# ==================== Summary ====================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"Model           : R(2+1)D-18 pretrained on Kinetics-400, fine-tuned on ISL")
print(f"Gesture classes : {num_classes} — {', '.join(class_names)}")
print(f"Resolution      : {TARGET_SIZE[0]}x{TARGET_SIZE[1]},  {NUM_FRAMES} frames/video")
print(f"Train/Val/Test  : {len(train_paths)} / {len(val_paths)} / {len(test_paths)}")
print(f"Test Accuracy   : {test_acc*100:.2f}%")
print(f"Test Loss       : {test_loss:.4f}")
print("=" * 80)
