"""LCDTC dataset loader for COCO-format annotations."""

import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Category mapping: combined category id -> (cls_id, content_id)
# 0: bottleempty  -> cls=0, content=0
# 1: bottlelittle -> cls=0, content=1
# 2: bottlehalf   -> cls=0, content=2
# 3: bottlemuch   -> cls=0, content=3
# 4: bottlefill   -> cls=0, content=4
CAT2CONTENT = {
    0: (0, 0),  # bottle, empty
    1: (0, 1),  # bottle, little
    2: (0, 2),  # bottle, half
    3: (0, 3),  # bottle, much
    4: (0, 4),  # bottle, fill
}
CONTENT_NAMES = ["empty", "little", "half", "much", "full"]


class LCDTCDataset(Dataset):
    """LCDTC dataset with mosaic augmentation and standard transforms.

    Returns per image:
      - img: (3, H, W) float32 tensor in [0, 1]
      - targets: (N, 7) tensor: [cls_id, content_id, cx, cy, w, h, orig_cat]
        where cx,cy,w,h are normalized to [0,1]
    """

    def __init__(self, data_dir, split="train", img_size=640, augment=True):
        self.img_size = img_size
        self.augment = augment and split == "train"

        ann_file = os.path.join(data_dir, "annotations",
                                "instances_train2017.json" if split == "train"
                                else "instances_val2017.json")
        img_dir = os.path.join(data_dir, "images",
                               "train2017" if split == "train" else "val2017")

        with open(ann_file) as f:
            coco = json.load(f)

        # Build image lookup
        self.imgs = {}
        for img_info in coco["images"]:
            self.imgs[img_info["id"]] = os.path.join(img_dir, img_info["file_name"])

        # Build annotation lookup: image_id -> list of annotations
        self.anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            cls_id, content_id = CAT2CONTENT[cat_id]
            bbox = ann["bbox"]  # [x, y, w, h]
            # Convert to cx, cy, w, h
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            w, h = bbox[2], bbox[3]
            self.anns.setdefault(img_id, []).append(
                [cls_id, content_id, cx, cy, w, h, cat_id]
            )

        self.img_ids = list(self.imgs.keys())
        print(f"[LCDTC] {split}: {len(self.img_ids)} images, "
              f"{sum(len(v) for v in self.anns.values())} annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.imgs[img_id]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        targets = self.anns.get(img_id, [])
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 7))

        if targets.numel() > 0:
            # Normalize box coords to [0, 1]
            targets[:, 2] /= orig_w  # cx
            targets[:, 3] /= orig_h  # cy
            targets[:, 4] /= orig_w  # w
            targets[:, 5] /= orig_h  # h

        # Resize to img_size
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, targets, torch.tensor([orig_h, orig_w])


def lcdtc_collate(batch):
    """Custom collate: images stacked, targets as list, sizes stacked."""
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    sizes = torch.stack([b[2] for b in batch])
    return imgs, targets, sizes


def get_loader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        drop_last=shuffle, collate_fn=lcdtc_collate,
    )
