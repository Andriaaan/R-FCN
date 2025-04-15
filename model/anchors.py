import torch
import itertools
import numpy as np


def generate_base_anchors(scales, ratios):
    """
    Генерує базові якірні бокси (в центрі (0,0)) для всіх комбінацій scale × ratio
    """
    anchors = []
    for scale, ratio in itertools.product(scales, ratios):
        w = scale * np.sqrt(1.0 / ratio)
        h = scale * np.sqrt(ratio)
        anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def generate_anchors(feature_size, stride, scales, ratios, device="cpu"):
    """
    Створює якірні бокси по всій фічмапі
    - feature_size: (H, W)
    - stride: скільки пікселів у оригінальному зображенні відповідає 1 позиції фічмапи
    """
    base_anchors = generate_base_anchors(scales, ratios).to(device)  # [A, 4]
    A = base_anchors.size(0)
    H, W = feature_size

    shift_x = torch.arange(0, W * stride, step=stride, device=device)
    shift_y = torch.arange(0, H * stride, step=stride, device=device)
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)

    anchors = (base_anchors[None, :, :] + shifts[:, None, :]).reshape(-1, 4)
    return anchors  # [H * W * A, 4]


if __name__ == "__main__":
    anchors = generate_anchors(
        feature_size=(14, 14),
        stride=16,
        scales=[8, 16, 32],
        ratios=[0.5, 1.0, 2.0],
        device="cpu"
    )
    print("Anchor shape:", anchors.shape)  # [14×14×9, 4] = [1764, 4]