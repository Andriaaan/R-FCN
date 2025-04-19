import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from Dataset.voc_dataset import VOCDataset
from model.rfcn import RFCN
import torchvision.transforms as T
import matplotlib
matplotlib.use('TkAgg')  # Для збереження графіків без GUI


@torch.no_grad()
def evaluate_model(model, val_loader, num_classes, device):
    model.eval()
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    count = 0

    for images, targets in tqdm(val_loader, desc="Validation"):
        images_tensor = torch.stack(images).to(device)
        scores, bbox_deltas, rois, targets = model(images_tensor, targets)

        from main import compute_losses  # або імпортуй звідки в тебе визначено
        cls_loss, bbox_loss = compute_losses(scores, bbox_deltas, rois, targets, num_classes, device)

        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        count += 1

    avg_cls_loss = total_cls_loss / count
    avg_bbox_loss = total_bbox_loss / count

    print(f"Validation: Classification Loss = {avg_cls_loss:.4f}, BBox Loss = {avg_bbox_loss:.4f}")
    return avg_cls_loss, avg_bbox_loss


@torch.no_grad()
def visualize_predictions(model, dataset, device, num_images=3):
    model.eval()

    for idx in range(num_images):
        image, target = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        scores, bbox_deltas, rois, _ = model(image_tensor, [target])

        probs = F.softmax(scores, dim=1)
        confs, labels = probs.max(dim=1)
        keep = (labels > 0) & (confs > 0.5)

        pred_boxes = rois[keep][:, 1:].cpu()
        pred_labels = labels[keep].cpu()
        pred_scores = confs[keep].cpu()

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image.permute(1, 2, 0).cpu())

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"Class {label}: {score:.2f}", color='red', fontsize=10)

        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 + 10, f"GT {label}", color='green', fontsize=10)

        plt.title(f"Image {idx}")
        plt.axis('off')
        plt.show()


def load_validation_data():
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

    dataset = VOCDataset(
        root_dir='Dataset/VOCdevkit/VOC2007',
        image_set='test',
        transforms=transform,

    )
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    return dataset, val_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Завантаження валідаційних даних
    val_dataset, val_loader = load_validation_data()

    # Ініціалізація моделі
    num_classes = 21  # Pascal VOC має 20 класів + background
    model = RFCN(num_classes=num_classes).to(device)

    # Завантаження найкращої моделі
    checkpoint_path = "checkpoints/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Завантажено чекпоінт: {checkpoint_path}")

    # Валідація
    #evaluate_model(model, val_loader, num_classes, device)

    # Візуалізація результатів
    visualize_predictions(model, val_dataset, device, num_images=3)


if __name__ == "__main__":
    main()
