import torch
import torch.nn as nn


class CollisionCNN(nn.Module):
    """
    CoordConv collision classifier.

    Input:  (B, 6, 32, 32)
              ch 0  alpha_i / 255
              ch 1  alpha_j / 255
              ch 2  dx / 32  (constant broadcast)
              ch 3  dy / 32  (constant broadcast)
              ch 4  x coordinate grid [0, 1]
              ch 5  y coordinate grid [0, 1]

    Output: (B,) raw logit — use BCEWithLogitsLoss, no sigmoid here.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # → (B, 32, 16, 16)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # → (B, 64, 8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # → (B, 4096)
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)
