import torch
from torchvision.ops import box_iou


def match_rois_to_gt(rois, targets, iou_thresh_fg=0.5, iou_thresh_bg=0.1):
    matched_labels = []
    matched_boxes = []

    for i in range(len(targets)):
        roi_batch = rois[rois[:, 0] == i][:, 1:]  # [N_i, 4]
        gt_boxes = targets[i]['boxes'].to(roi_batch.device)  # [M_i, 4]
        gt_labels = targets[i]['labels'].to(roi_batch.device)  # [M_i]

        if len(gt_boxes) == 0:
            matched_labels.append(torch.zeros((roi_batch.shape[0],), dtype=torch.long, device=roi_batch.device))
            matched_boxes.append(torch.zeros((roi_batch.shape[0], 4), dtype=torch.float, device=roi_batch.device))
            continue

        ious = box_iou(roi_batch, gt_boxes)
        max_ious, gt_ids = ious.max(dim=1)

        labels = torch.where(
            max_ious >= iou_thresh_fg,
            gt_labels[gt_ids],
            torch.zeros_like(gt_ids)
        )
        boxes = gt_boxes[gt_ids]

        matched_labels.append(labels)
        matched_boxes.append(boxes)

    return None, torch.cat(matched_labels, dim=0), torch.cat(matched_boxes, dim=0)
