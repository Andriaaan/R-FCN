import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

def psroi_pooling(features, rois, output_size, spatial_scale, num_classes, group_size):
    """
    PSRoI Pooling на основі roi_align + reshaping.
    Args:
        features: [B, C, H, W] — фічмапа з PS-sensitive картами
        rois: [N, 5] — (batch_idx, x1, y1, x2, y2)
        output_size: (k, k) — сітка для pooling (типово 7x7)
        spatial_scale: масштабує координати bbox до фічмапи
        num_classes: кількість класів (включаючи фон)
        group_size: k (output_size)

    Returns:
        scores: [N, num_classes] — логіти класів
    """
    k = group_size
    if len(features.size()) != 4:
        raise ValueError(f"Expected features to have 4 dimensions (B, C, H, W), but got {features.size()}")
    N, C, H, W = features.shape
    assert C == num_classes * k * k, f"Feature map channels ({C}) ≠ num_classes × k²"

    if len(rois.size()) != 2 or rois.size(1) != 5:
        raise ValueError(f"Expected rois to have shape [N, 5], but got {rois.size()}")
    # PS: розділимо features → [num_classes, k, k]
    pooled = roi_align(features, rois, output_size=output_size, spatial_scale=spatial_scale, aligned=True)
    pooled = pooled.view(rois.shape[0], num_classes, k, k, output_size[0], output_size[1])

    # Вибираємо по діагоналі [i-th region → i-th PS map] → середнє значення
    #output = pooled.mean(dim=[3, 4, 5])  # [N, num_classes]
    pooled = pooled.mean(dim=[2, 3])  # Середнє по k x k
    return pooled