import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transforms=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms

        image_set_path = os.path.join(root_dir, 'ImageSets/Main', f'{image_set}.txt')
        with open(image_set_path) as f:
            self.ids = [line.strip() for line in f]

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        ann_path = os.path.join(self.annotation_dir, f'{image_id}.xml')

        img = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_voc_xml(ann_path)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            img, target = self.transforms(img, target)
            #img = self.transforms(img)

        return img, target

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text.lower().strip()
            if label not in VOC_CLASSES:
                continue
            label_idx = VOC_CLASSES.index(label) + 1  # 0 is reserved for background

            bbox = obj.find('bndbox')
            box = [
                float(bbox.find('xmin').text),
                float(bbox.find('ymin').text),
                float(bbox.find('xmax').text),
                float(bbox.find('ymax').text)
            ]
            boxes.append(box)
            labels.append(label_idx)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels