import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from feature_extraction import extract_video_frames


class ISLDataset(Dataset):
    """PyTorch Dataset that loads ISL gesture videos from disk one at a time."""

    def __init__(self, video_paths, labels, num_frames=30, target_size=(64, 64)):
        self.video_paths = video_paths
        self.labels      = labels
        self.num_frames  = num_frames
        self.target_size = target_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = extract_video_frames(
            self.video_paths[idx], self.num_frames, self.target_size
        )
        if frames is None:
            frames = np.zeros(
                (self.num_frames, self.target_size[0], self.target_size[1], 3),
                dtype=np.float32
            )
        # frames: (T, H, W, C) float32
        return torch.from_numpy(frames.astype(np.float32)), int(self.labels[idx])


def get_dataloader(paths, labels, batch_size=8, num_frames=30,
                   target_size=(64, 64), shuffle=True, num_workers=0):
    dataset = ISLDataset(paths, labels, num_frames, target_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
