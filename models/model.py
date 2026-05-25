import torch
import torch.nn as nn
import torchvision.models.video as video_models


class ISLNet(nn.Module):
    """
    ISL Gesture Recognition using pretrained R(2+1)D-18.
    Pretrained on Kinetics-400 (a large 400-class video action dataset),
    then fine-tuned on the 7-class ISL gesture dataset.

    R(2+1)D decomposes each 3D convolution into a 2D spatial conv followed by
    a 1D temporal conv — capturing spatial appearance and motion separately.
    This is more efficient and generalises better than plain Conv3d.

    Input: (B, T, H, W, C)  float32, normalised with Kinetics-400 mean/std.
    """

    # Kinetics-400 normalisation — must match pretraining
    MEAN = [0.43216, 0.394666, 0.37645]
    STD  = [0.22803,  0.22145, 0.216989]

    def __init__(self, num_classes=7):
        super().__init__()

        weights = video_models.R2Plus1D_18_Weights.KINETICS400_V1
        backbone = video_models.r2plus1d_18(weights=weights)

        # Replace the Kinetics-400 head (512 → 400) with ISL head (512 → num_classes)
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        # x: (B, T, H, W, C) → (B, C, T, H, W)  required by torchvision video models
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return self.backbone(x)

    def param_groups(self, lr_backbone=1e-5, lr_head=1e-4):
        """Differential LR: small LR for pretrained backbone, higher for new head."""
        head_ids = {id(p) for p in self.backbone.fc.parameters()}
        backbone_params = [p for p in self.backbone.parameters() if id(p) not in head_ids]
        return [
            {'params': backbone_params,          'lr': lr_backbone},
            {'params': list(self.backbone.fc.parameters()), 'lr': lr_head},
        ]


def build_model(num_classes=7):
    return ISLNet(num_classes=num_classes)
