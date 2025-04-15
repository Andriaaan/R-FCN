import torch
from model.rfcn import RFCN
from model.matcher import match_rois_to_gt
from model.postprocess import postprocess_proposals

# Ініціалізація моделі
num_classes = 21  # Для Pascal VOC (20 класів + фон)
model = RFCN(num_classes)

# Створення випадкових даних
batch_size = 2
images = torch.randn(batch_size, 3, 224, 224)  # Два зображення 224x224
targets = [
    {'boxes': torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]]), 'labels': torch.tensor([1, 2])},  # для першого зображення
    {'boxes': torch.tensor([[60, 60, 170, 170]]), 'labels': torch.tensor([1])}  # для другого зображення
]

# Тест на прогін через модель
model.eval()  # Потрібно для інференсу

with torch.no_grad():
    # Assume model returns lists of proposals and probs per image
    proposals_list, probs_list,  = model(images)

    # Process each image's proposals and scores separately
    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_labels = []

    for proposals, probs in zip(proposals_list, probs_list):
        filtered_boxes, filtered_scores, filtered_labels = postprocess_proposals(
            proposals=proposals,
            scores=probs,
            score_thresh=0.05,
            nms_thresh=0.5,
            max_dets=100
        )
        all_filtered_boxes.append(filtered_boxes)
        all_filtered_scores.append(filtered_scores)
        all_filtered_labels.append(filtered_labels)

    # Now all_filtered_* contain results for each image in the batch
    print(f"Filtered Boxes: {all_filtered_boxes}")
    print(f"Filtered Scores: {all_filtered_scores}")
    print(f"Filtered Labels: {all_filtered_labels}")
