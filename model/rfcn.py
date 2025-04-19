import torch
import torch.nn as nn
from model.backbone import build_backbone
from model.postprocess import postprocess_proposals
from model.psroi_pool import psroi_pooling
from model.rpn import RPN
from model.anchors import generate_anchors
from model.proposals import generate_proposals
from model.matcher import match_rois_to_gt


class BBoxHead(nn.Module):
    def __init__(self, in_channels, num_classes, k=7):
        """
        Вхід: фічемапа з розмірністю in_channels.
        Вихід: тензор розміру [B, num_classes*4*k*k, H, W],
               який буде подаватися в PSRoI Pooling.
        """
        super(BBoxHead, self).__init__()
        self.k = k
        self.num_classes = num_classes
        # Тут вихідний канал = num_classes * 4 * k * k,
        # оскільки для кожного класу потрібно 4 регресійні значення і кожен регіон має k x k позиційно чутливих блоків.
        self.conv = nn.Conv2d(in_channels, num_classes * 4 * k * k, kernel_size=1)

    def forward(self, x):
        # x має форму [B, in_channels, H, W]
        # Результат: [B, num_classes*4*k*k, H, W]
        return self.conv(x)

class RFCN(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True, n_classes=21, n_anchors=9):
        """
        n_classes: включає фон (background = 0), VOC має 20 класів + 1 фон = 21
        """
        super(RFCN, self).__init__()

        # Backbone (ResNet до C5)
        self.backbone, out_channels = build_backbone(backbone_name, pretrained)

        # RPN
        self.rpn_head = RPN(in_channels=out_channels, n_anchors=n_anchors)

        self.anchor_stride = 16
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.anchor_scales = [8, 16, 32]

        self.psroi_pool = psroi_pooling

        self.num_classes = num_classes

        self.k = 7  # Розмір PSRoI pooling (k x k)

        self.cls_head = nn.Conv2d(2048, num_classes * self.k * self.k, kernel_size=1)
        self.bbox_head = BBoxHead(in_channels=out_channels, num_classes=num_classes, k=self.k)

    def forward(self, images, targets=None):
        B, _, H, W = images.shape

        # --- Backbone ---
        features = self.backbone(images)

        # --- RPN ---
        objectness, bbox_deltas_rpn = self.rpn_head(features)

        # Generate anchors
        feat_size = features.shape[-2:]
        device = images.device
        anchors = generate_anchors(
            feature_size=feat_size,
            stride=self.anchor_stride,
            scales=self.anchor_scales,
            ratios=self.anchor_ratios,
            device=device
        )

        # Generate proposals
        proposals = generate_proposals(
            objectness=objectness,
            bbox_deltas=bbox_deltas_rpn,
            anchors=anchors,
            image_size=(H, W),
            pre_nms_top_n=6000,
            post_nms_top_n=1000,
            nms_thresh=0.7,
            min_size=16,
        )

        # Підготовка RoIs
        rois = []
        for i, props in enumerate(proposals):
            batch_idx = torch.full((props.shape[0], 1), i, dtype=torch.float32, device=props.device)
            rois.append(torch.cat([batch_idx, props], dim=1))
        rois = torch.cat(rois, dim=0)  # [N, 5]

        # Класифікаційна карта
        cls_maps = self.cls_head(features)  # [B, C, H', W']

        # --- Класифікаційна гілка ---
        cls_maps = self.cls_head(features)  # [B, num_classes*k*k, H', W']
        cls_scores = psroi_pooling(
            features=cls_maps,
            rois=rois,
            output_size=(self.k, self.k),
            spatial_scale=1.0 / self.anchor_stride,
            num_classes=self.num_classes,
            group_size=self.k
        )  # [N, num_classes, k, k]
        cls_scores = cls_scores.mean(dim=[2, 3])  # [N, num_classes]

        # --- Регістрійна гілка ---
        bbox_maps = self.bbox_head(features)  # [B, num_classes*4*k*k, H', W']
        bbox_deltas = psroi_pooling(
            features=bbox_maps,
            rois=rois,
            output_size=(self.k, self.k),
            spatial_scale=1.0 / self.anchor_stride,
            num_classes=self.num_classes * 4,  # тут 4 значення на клас
            group_size=self.k
        )  # [N, num_classes*4, k, k]
        bbox_deltas = bbox_deltas.mean(dim=[2, 3])  # [N, num_classes*4]

        if self.training or targets is not None:
            return cls_scores, bbox_deltas, rois, targets
        else:
            # Постобробка
            filtered_boxes, filtered_scores, filtered_labels = postprocess_proposals(
                proposals=rois[:, 1:],  # координати боксів
                scores=cls_scores,
                score_thresh=0.05,
                nms_thresh=0.5,
                max_dets=100
            )
            return filtered_boxes, filtered_scores, filtered_labels
