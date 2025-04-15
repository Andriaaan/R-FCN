import torch
from torchvision.ops import box_iou


def match_rois_to_gt(rois, targets, iou_thresh_fg=0.5, iou_thresh_bg=0.1):
    """
    Розмітити кожен RoI як об'єкт (1...C) або фон (0).
    Args:
        rois: [N, 5] — (batch_idx, x1, y1, x2, y2)
        targets: список з B словників: { 'boxes': [...], 'labels': [...] }
    Returns:
        matched_labels: [N] — integer labels (0 = background)
    """
    matched_labels = []

    for i in range(len(targets)):
        roi_batch = rois[rois[:, 0] == i][:, 1:]  # [N_i, 4]
        gt_boxes = targets[i]['boxes'].to(roi_batch.device)  # [M_i, 4]
        gt_labels = targets[i]['labels'].to(roi_batch.device)  # [M_i]

        if len(gt_boxes) == 0:
            matched_labels.append(torch.zeros((roi_batch.shape[0],), dtype=torch.long, device=roi_batch.device))
            continue

        ious = box_iou(roi_batch, gt_boxes)  # [N_i, M_i]
        max_ious, gt_ids = ious.max(dim=1)  # [N_i]

        labels = torch.where(
            max_ious >= iou_thresh_fg,
            gt_labels[gt_ids],  # об'єкт
            torch.zeros_like(gt_ids)  # фон
        )
        matched_labels.append(labels)

    return torch.cat(matched_labels, dim=0)  # [N]
