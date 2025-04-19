import torch
import torch.nn.functional as F
from torchvision import transforms
from Dataset.voc_dataset import VOCDataset
from model.matcher import match_rois_to_gt
from model.rfcn import RFCN

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Для збереження графіків без GUI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


def compute_losses(scores, bbox_deltas, rois, targets, num_classes, device):
    # --- Підготовка GT ---
    _, gt_labels, gt_bboxes = match_rois_to_gt(
        rois, targets, iou_thresh_fg=0.5, iou_thresh_bg=0.1
    )

    # --- Класифікаційна втрата ---
    cls_loss = F.cross_entropy(scores, gt_labels.to(device))

    # --- BBox регресія ---
    N = bbox_deltas.size(0)
    bbox_deltas = bbox_deltas.view(N, num_classes, 4)
    idx = torch.arange(N, device=device)

    # Вибираємо bbox дельти лише для GT класу
    pred_deltas = bbox_deltas[idx, gt_labels]

    bbox_loss = F.smooth_l1_loss(pred_deltas, gt_bboxes.to(device))

    return cls_loss, bbox_loss


def train_model(model, dataset, num_epochs=30, lr=1e-4, batch_size=2, device='cuda', save_dir='checkpoints'):
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    best_loss = float('inf')
    cls_losses, bbox_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0

        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, targets) in loop:
            images_tensor = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            scores, bbox_deltas, rois, targets = model(images_tensor, targets)
            cls_loss, bbox_loss = compute_losses(scores, bbox_deltas, rois, targets, model.num_classes, device)
            loss = cls_loss + bbox_loss
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_bbox_loss += bbox_loss.item()

        epoch_cls_loss /= len(dataloader)
        epoch_bbox_loss /= len(dataloader)
        cls_losses.append(epoch_cls_loss)
        bbox_losses.append(epoch_bbox_loss)

        print(f"Epoch {epoch+1}: Classification Loss = {epoch_cls_loss:.4f}, BBox Loss = {epoch_bbox_loss:.4f}")

        # Збереження найкращої моделі
        total_loss = epoch_cls_loss + epoch_bbox_loss
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("✅ Best model saved!")

    # --- Побудова графіків втрат ---
    plt.figure(figsize=(10, 5))
    plt.plot(cls_losses, label="Classification Loss")
    plt.plot(bbox_losses, label="BBox Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.show()


class ResizeWithBoxes:
    def __init__(self, size):
        self.size = size  # (height, width)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, target):
        # Початкові розміри
        orig_w, orig_h = img.size
        new_h, new_w = self.size

        # Ресайз зображення
        img = img.resize((new_w, new_h))

        # Масштабування bbox-ів
        boxes = target['boxes']
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= scale_x  # xmin, xmax
        boxes[:, [1, 3]] *= scale_y  # ymin, ymax
        target['boxes'] = boxes

        # Конвертація зображення в тензор
        img = self.to_tensor(img)

        return img, target


transform = ResizeWithBoxes((224, 224))
# Без аугментацій на перший раз
dataset = VOCDataset(
    root_dir='Dataset/VOCdevkit/VOC2007',
    image_set='train',
    transforms=transform

)

model = RFCN(num_classes=21, backbone_name='resnet50', pretrained=True)
train_model(model, dataset, num_epochs=30, lr=1e-4, batch_size=2)
