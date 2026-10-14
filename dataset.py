import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
import matplotlib.pyplot as plt
import random

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

"""Dataset utilities for COCO-style object detection training.

This module not contain any data augmentation from augmentation.
....
"""

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def visualize_sample(dataset: Dataset, idx=None):
    """Display one dataset sample with bounding boxes."""
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    img, target = dataset[idx]

    img = img.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    for box in target['boxes']:
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

    plt.title(f"Image {idx} - {len(target['boxes'])} boxes")
    plt.axis('off')
    plt.show()


class CustomDataset(Dataset):
    """A COCO-format dataset for object detection."""

    def __init__(self,
                 root: str | Path,
                 ann_file: str | Path,
                 category_name_to_label: dict[str, int] | None = None
    ) -> None:
        self.root = resolve_path(root)
        self.ann_file = self._resolve_ann_file(ann_file)

        with self.ann_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images: list[dict] = coco["images"]
        self.categories: list[dict] = coco["categories"]

        if category_name_to_label is None:
            self.cat2label: dict[int, int | None] = {
                cat["id"]: i + 1 for i, cat in enumerate(self.categories)
            }
        else:
            self.cat2label = {
                cat["id"]: category_name_to_label.get(cat["name"])
                for cat in self.categories
            }

        self.ann_map: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

    def _resolve_ann_file(self, ann_file: str | Path) -> Path:
        ann_path = Path(ann_file).expanduser()
        if ann_path.is_absolute():
            return ann_path

        candidate = self.root / ann_path
        if candidate.is_file():
            return candidate

        return resolve_path(ann_path)

    def __len__(self):
        return len(self.images)

    def load_image(self, file_name):
        path = self.root / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Image referenced by COCO annotation not found: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        meta = self.images[idx]
        img_id = meta["id"]

        image = self.load_image(meta["file_name"])
        anns = self.ann_map.get(img_id, [])

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            label = self.cat2label.get(ann["category_id"])
            if label is None:
                continue
            # Torchvision detection models reserve label 0 for background.
            boxes.append([x, y, x + w, y + h])
            labels.append(label)

        image_t = to_tensor(image)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(img_id),
            "area": area,
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        return image_t, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloader(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """ Build the training and validation data loaders """
    ds_cfg = cfg["dataset"]
    dl_cfg = cfg["dataset"]["dataloader"]

    train_ds = CustomDataset(
        root=ds_cfg["train"],
        ann_file=ds_cfg["ann_file"],
    )

    train_category_labels = {
        category["name"]: index + 1
        for index, category in enumerate(train_ds.categories)
    }

    val_ds = CustomDataset(
        root=ds_cfg["valid"],
        ann_file=ds_cfg["ann_file"],
        category_name_to_label=train_category_labels,
    )

    num_workers = dl_cfg["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


# if __name__ == "__main__":
#     """Test the dataset and visualization"""
#     from config import CONFIG

#     config = CONFIG
#     dataset = CustomDataset(
#         root=config["dataset"]["train"],
#         ann_file=config["dataset"]["ann_file"],
#     )

#     idx = random.randint(0, len(dataset) - 1)
#     img, target = dataset[idx]

#     print(f"Total samples : {len(dataset)}")
#     print(f"Sample index  : {idx}")
#     print(f"Image ID      : {dataset.images[idx]['id']}")
#     print(f"Image shape   : {img.shape}")
#     print(f"Target keys   : {target.keys()}")
#     print(f"Num boxes     : {len(target['boxes'])}")
#     print(f"Sample boxes  : {target['boxes'][:5]}")

#     visualize_sample(dataset, idx=idx)