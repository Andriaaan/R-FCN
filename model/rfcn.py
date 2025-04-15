import torch
import torch.nn as nn
from model.backbone import build_backbone
from model.psroi_pool import psroi_pooling
from model.rpn import RPN
from model.anchors import generate_anchors
from model.proposals import generate_proposals
from model.matcher import match_rois_to_gt


class RFCN(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True, n_classes=21, n_anchors=9):
        """
        n_classes: включає фон (background = 0), VOC має 20 класів + 1 фон = 21
        """
        super(RFCN, self).__init__()

        # Backbone (ResNet до C5)
        self.backbone, out_channels = build_backbone(backbone_name, pretrained)

        # RPN
        self.rpn = RPN(in_channels=out_channels, n_anchors=n_anchors)

        self.anchor_stride = 16
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.anchor_scales = [8, 16, 32]

        # TODO: PSRoI pooling
        self.psroi_pool = None  # Поки заглушка

        self.num_classes = num_classes

        self.k = 3

        # TODO: Classification + bbox heads
        self.cls_head = nn.Conv2d(256, num_classes * self.k * self.k, kernel_size=1)
        self.bbox_head = None

    def forward(self, images, targets=None):
        B, _, H, W = images.shape

        # 1. Backbone → Features
        features = self.backbone(images)  # [B, C, H', W']

        # 2. RPN → objectness + bbox_deltas
        objectness, bbox_deltas = self.rpn_head(features)

        # 3. Generate anchors (на одне зображення — однакові)
        feat_size = features.shape[-2:]  # (H', W')
        device = images.device
        anchors = generate_anchors(
            feature_size=feat_size,
            stride=self.anchor_stride,
            scales=self.anchor_scales,
            ratios=self.anchor_ratios,
            device=device
        )

        # 4. Generate proposals (list of B tensors)
        proposals = generate_proposals(
            objectness=objectness,
            bbox_deltas=bbox_deltas,
            anchors=anchors,
            image_size=(H, W),
            pre_nms_top_n=6000,
            post_nms_top_n=1000,
            nms_thresh=0.7,
            min_size=16,
        )

        # --- Підготовка RoIs ---
        rois = []
        for i, props in enumerate(proposals):
            batch_idx = torch.full((props.shape[0], 1), i, dtype=torch.float32, device=props.device)
            rois.append(torch.cat([batch_idx, props], dim=1))
        rois = torch.cat(rois, dim=0)

        # --- Класифікаційна фічмапа ---
        cls_maps = self.cls_head(features)  # [B, C, H', W']

        # --- PSRoI Pooling ---
        scores = psroi_pooling(
            features=cls_maps,
            rois=rois,
            output_size=(self.k, self.k),
            spatial_scale=1.0 / self.anchor_stride,
            num_classes=self.num_classes,
            group_size=self.k
        )  # [N, num_classes]

        if self.training and targets is not None:
            matched_labels = match_rois_to_gt(rois, targets)
            loss_cls = F.cross_entropy(scores, matched_labels)
            return loss_cls

        # --- Інференс ---
        probs = F.softmax(scores, dim=1)
        return proposals, probs
