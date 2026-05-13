"""Datasets for the LCDTC-only full system."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


STATE_NAMES = ["empty", "little", "half", "much", "full"]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class LCDTCDetectionDataset(Dataset):
    """Torchvision-detection style dataset built from LCDTC box annotations."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 800,
        annotation_name: str | None = None,
        train: bool | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.train = (split == "train") if train is None else train
        ann_name = annotation_name or (
            "instances_cls.json"
            if split == "val"
            else "instances_train2017.json"
        )
        self.annotation_name = ann_name
        ann_path = self.data_dir / "annotations" / ann_name
        img_dir = self.data_dir / "images" / ("train2017" if split == "train" else "val2017")

        coco = _load_json(ann_path)
        self.image_lookup = {
            int(img["id"]): {
                "path": img_dir / img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
            }
            for img in coco["images"]
        }
        self.anns_by_image: Dict[int, List[Dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        self.image_ids = sorted(self.image_lookup.keys())
        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]
        self.color_jitter = (
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)
            if self.train
            else None
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resize_boxes(
        self, boxes: torch.Tensor, old_w: int, old_h: int, new_w: int, new_h: int
    ) -> torch.Tensor:
        scale_x = new_w / max(old_w, 1)
        scale_y = new_h / max(old_h, 1)
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return boxes

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        meta = self.image_lookup[image_id]
        image = _open_rgb(meta["path"])
        old_w, old_h = image.size
        anns = self.anns_by_image.get(image_id, [])

        boxes = []
        states = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            raw_cat = int(ann["category_id"])
            if "instances_cls" in self.annotation_name:
                states.append(-1)
            else:
                states.append(raw_cat if raw_cat < 5 else -1)

        boxes_t = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_t = torch.ones((boxes_t.shape[0],), dtype=torch.int64)
        states_t = (
            torch.tensor(states, dtype=torch.long)
            if states
            else torch.zeros((0,), dtype=torch.long)
        )

        target_size = self.image_size
        image = TF.resize(
            image,
            [target_size, target_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        boxes_t = self._resize_boxes(boxes_t, old_w, old_h, target_size, target_size)
        if boxes_t.numel() > 0:
            valid = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])
            boxes_t = boxes_t[valid]
            labels_t = labels_t[valid]
            states_t = states_t[valid]

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            if boxes_t.numel() > 0:
                x1 = boxes_t[:, 0].clone()
                x2 = boxes_t[:, 2].clone()
                boxes_t[:, 0] = target_size - x2
                boxes_t[:, 2] = target_size - x1
                valid = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])
                boxes_t = boxes_t[valid]
                labels_t = labels_t[valid]
                states_t = states_t[valid]

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image_t = TF.to_tensor(image)
        area = (
            (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0)
            * (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
        )
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id]),
            "area": area,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
            "states": states_t,
            "orig_size": torch.tensor([old_h, old_w], dtype=torch.int64),
        }
        return image_t, target


class LCDTCStateDataset(Dataset):
    """Bottle ROI classifier dataset from LCDTC box annotations."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 320,
        context: float = 0.15,
        train: bool | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.context = context
        self.train = (split == "train") if train is None else train

        ann_name = "instances_train2017.json" if split == "train" else "instances_val2017.json"
        ann_path = self.data_dir / "annotations" / ann_name
        img_dir = self.data_dir / "images" / ("train2017" if split == "train" else "val2017")

        coco = _load_json(ann_path)
        image_lookup = {int(img["id"]): img_dir / img["file_name"] for img in coco["images"]}
        records = []
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            records.append(
                {
                    "image_path": image_lookup[int(ann["image_id"])],
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "state": int(ann["category_id"]),
                }
            )
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records
        self.color_jitter = (
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)
            if self.train
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _expand_box(self, box, width: int, height: int):
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        pad_x = int(round(bw * self.context))
        pad_y = int(round(bh * self.context))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        return x1, y1, x2, y2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image = _open_rgb(record["image_path"])
        box = self._expand_box(record["bbox"], image.width, image.height)
        image = image.crop(box)
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image_t = TF.to_tensor(image)
        state = int(record["state"])
        return {
            "image": image_t,
            "state_label": torch.tensor(state, dtype=torch.long),
            "binary_label": torch.tensor(0 if state == 0 else 1, dtype=torch.float32),
        }
