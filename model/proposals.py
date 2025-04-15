import torch
import torchvision
from torchvision.ops import nms


def apply_deltas_to_anchors(deltas, anchors):
    """
    Застосовує регресії bbox'ів до anchor box'ів.
    """
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes_to_image(boxes, image_size):
    """
    Обрізає координати боксу, щоб вони не виходили за межі зображення.
    """
    h, w = image_size
    boxes[:, 0].clamp_(min=0, max=w)
    boxes[:, 1].clamp_(min=0, max=h)
    boxes[:, 2].clamp_(min=0, max=w)
    boxes[:, 3].clamp_(min=0, max=h)
    return boxes


def remove_small_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep.nonzero(as_tuple=True)[0]


def generate_proposals(objectness, bbox_deltas, anchors, image_size,
                       pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.7, min_size=16):
    """
    Основна функція для генерації пропозицій (proposal boxes) з RPN.
    """
    B, _, H, W = objectness.shape
    A = anchors.shape[0] // (H * W)

    proposals = []
    for i in range(B):
        obj = objectness[i].permute(1, 2, 0).reshape(-1, 2)
        scores = torch.softmax(obj, dim=1)[:, 1]  # ймовірність об'єкта

        deltas = bbox_deltas[i].permute(1, 2, 0).reshape(-1, 4)

        boxes = apply_deltas_to_anchors(deltas, anchors)

        boxes = clip_boxes_to_image(boxes, image_size)
        keep = remove_small_boxes(boxes, min_size)
        boxes = boxes[keep]
        scores = scores[keep]

        scores, idx = scores.sort(descending=True)
        idx = idx[:pre_nms_top_n]
        boxes = boxes[idx]
        scores = scores[:pre_nms_top_n]

        keep = nms(boxes, scores, nms_thresh)
        keep = keep[:post_nms_top_n]

        proposals.append(boxes[keep])

    return proposals  # список з B елементів [num_props_i, 4]

