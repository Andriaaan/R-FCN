import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels=512, n_anchors=9):
        super(RPN, self).__init__()

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(mid_channels, n_anchors * 2, kernel_size=1)  # класи: об'єкт/фон
        self.bbox_reg = nn.Conv2d(mid_channels, n_anchors * 4, kernel_size=1)  # 4 координати на якір

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv(x))
        objectness = self.objectness(x)  # [B, 18, H, W] — 2 класи * 9 якорів
        bbox_deltas = self.bbox_reg(x)  # [B, 36, H, W] — 4 координати * 9 якорів
        return objectness, bbox_deltas

    def _initialize_weights(self):
        for m in [self.conv, self.objectness, self.bbox_reg]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
