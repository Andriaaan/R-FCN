import torch.nn as nn
import torchvision.models as models


def build_backbone(name='resnet50', pretrained=True, freeze_layers=True):
    resnet = getattr(models, name)(pretrained=pretrained)

    # Вирізаємо FC-шари
    layers = list(resnet.children())[:-2]
    backbone = nn.Sequential(*layers)

    if freeze_layers:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, 2048  # вихідний розмір останнього шару
