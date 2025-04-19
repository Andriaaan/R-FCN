import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time

from Dataset.voc_dataset import VOCDataset
from model.rfcn import RFCN


def collate_fn(batch):
    return tuple(zip(*batch))


train_transforms = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
])

val_transforms = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
])

train_dataset = VOCDataset(root_dir='Dataset/VOCdevkit/VOC2007', image_set='trainval',
                           transforms=lambda img, target: (train_transforms(img), target))
val_dataset = VOCDataset(root_dir='Dataset/VOCdevkit/VOC2007', image_set='test',
                         transforms=lambda img, target: (val_transforms(img), target))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RFCN(num_classes=21).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

cls_criterion = torch.nn.CrossEntropyLoss()

bbox_criterion = torch.nn.SmoothL1Loss()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for images, targets in tqdm(loader, desc="Training"):
        images = [img.to(device) for img in images]
        optimizer.zero_grad()
        images_tensor = torch.stack(images)  # Assumes all images are the same size

        # Forward pass
        filtered_boxes, scores, pred_labels = model(images_tensor, targets)

        if scores.numel() == 0:  # Handle empty predictions
            continue

        # Classification loss
        cls_labels = torch.cat([t['labels'] for t in targets]).to(device)
        print("scores shape:", scores.shape)
        print("cls_labels shape:", cls_labels.shape)
        if scores.ndim == 1:
            scores = scores.view(-1, 21)

        cls_loss = cls_criterion(scores, cls_labels)

        # Bounding box regression loss
        bbox_targets = torch.cat([t['boxes'] for t in targets]).to(device)
        bbox_loss = bbox_criterion(filtered_boxes.view(-1, 4), bbox_targets.view(-1, 4))

        # Total loss
        loss = cls_loss + bbox_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []

    for images, targets in tqdm(loader, desc="Validation"):
        images = [img.to(device) for img in images]
        images_tensor = torch.stack(images)

        boxes, scores, labels = model(images_tensor)

        all_preds.append((boxes, scores, labels))
        # Тут можна реалізувати підрахунок метрик (наприклад, IoU, mAP)

    return all_preds


num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    val_preds = validate(model, val_loader, device)
