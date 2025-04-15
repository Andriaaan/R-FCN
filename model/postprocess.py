import torch
from torchvision.ops import nms


def postprocess_proposals(proposals, scores, score_thresh=0.05, nms_thresh=0.5, max_dets=100):
    """
    Постобробка для пропозицій:
    1. Фільтрація по ймовірності.
    2. Non-Maximum Suppression (NMS).

    Args:
        proposals: [N, 4] — координати боксів (x1, y1, x2, y2)
        scores: [N, num_classes] — ймовірності класів
        score_thresh: float — поріг ймовірності
        nms_thresh: float — поріг для NMS
        max_dets: int — макс. кількість результатів

    Returns:
        filtered_boxes: [M, 4]
        filtered_scores: [M]
        filtered_labels: [M]
    """
    # Фільтрація боксів за максимальним скором
    max_scores, _ = scores.max(dim=1)
    keep = max_scores > score_thresh
    proposals = proposals[keep]
    scores = scores[keep]

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    num_classes = scores.shape[1]
    device = proposals.device

    for class_idx in range(num_classes):
        class_scores = scores[:, class_idx]
        score_mask = class_scores > score_thresh

        if score_mask.sum() == 0:
            continue

        class_scores = class_scores[score_mask]
        class_proposals = proposals[score_mask]

        keep_indices = nms(class_proposals, class_scores, nms_thresh)

        filtered_boxes.append(class_proposals[keep_indices])
        filtered_scores.append(class_scores[keep_indices])
        filtered_labels.append(
            torch.full((len(keep_indices),), class_idx, dtype=torch.long, device=device)
        )

    if len(filtered_boxes) == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), dtype=torch.long,
                                                                                                 device=device)

    filtered_boxes = torch.cat(filtered_boxes, dim=0)
    filtered_scores = torch.cat(filtered_scores, dim=0)
    filtered_labels = torch.cat(filtered_labels, dim=0)

    # Сортування та обрізка до max_dets
    if len(filtered_boxes) > max_dets:
        sorted_indices = filtered_scores.argsort(descending=True)[:max_dets]
        filtered_boxes = filtered_boxes[sorted_indices]
        filtered_scores = filtered_scores[sorted_indices]
        filtered_labels = filtered_labels[sorted_indices]

    return filtered_boxes, filtered_scores, filtered_labels
