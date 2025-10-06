import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRespCNN(nn.Module):
    """
    A lightweight CNN for respiratory sound spectrograms.
    Input: (B, 1, F, T)
    Output: 4-class logits
    """
    def __init__(self, n_classes: int = 4):
        super(SimpleRespCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
        )

        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten (B, 64)
        x = self.classifier(x)     # (B, n_classes)
        return x
