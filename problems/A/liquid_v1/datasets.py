"""Datasets for the v1 bottle liquid training stack.

This module intentionally supports the two datasets currently under
`problems/A/datasets` even though their layouts differ:

1. LCDTC (COCO-style detection annotations)
2. Segmenting Transparent Objects in the Wild (image/mask pairs, often zipped)
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


STATE_NAMES = ["empty", "little", "half", "much", "full"]


@dataclass(frozen=True)
class SampleRecord:
    """Canonical sample descriptor used by both datasets."""

    image_key: str
    mask_key: Optional[str]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    state_label: int
    binary_label: int
    has_bottle_mask: bool
    has_liquid_mask: bool
    task_id: int


def _safe_open_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _safe_open_mask(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def _expand_box(
    box: Tuple[int, int, int, int], width: int, height: int, context: float
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return None
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * context))
    pad_y = int(round(bh * context))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _crop_image_and_mask(
    image: Image.Image,
    mask: Optional[Image.Image],
    box: Optional[Tuple[int, int, int, int]],
    context: float,
) -> Tuple[Image.Image, Optional[Image.Image]]:
    if box is None:
        return image, mask
    expanded = _expand_box(box, image.width, image.height, context)
    if expanded is None:
        return image, mask
    x1, y1, x2, y2 = expanded
    image = image.crop((x1, y1, x2, y2))
    if mask is not None:
        mask = mask.crop((x1, y1, x2, y2))
    return image, mask


class BaseBottleDataset(Dataset):
    """Shared preprocessing and output format."""

    def __init__(
        self,
        image_size: int = 320,
        train: bool = True,
        context: float = 0.12,
        color_jitter: float = 0.15,
        flip_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.context = context
        self.flip_prob = flip_prob
        self.color_jitter = (
            ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(color_jitter, 0.05),
            )
            if train and color_jitter > 0
            else None
        )

    def _augment(
        self, image: Image.Image, bottle_mask: Optional[Image.Image], liquid_mask: Optional[Image.Image]
    ) -> Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image]]:
        if self.train and random.random() < self.flip_prob:
            image = TF.hflip(image)
            if bottle_mask is not None:
                bottle_mask = TF.hflip(bottle_mask)
            if liquid_mask is not None:
                liquid_mask = TF.hflip(liquid_mask)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if bottle_mask is not None:
            bottle_mask = TF.resize(
                bottle_mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
        if liquid_mask is not None:
            liquid_mask = TF.resize(
                liquid_mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
        return image, bottle_mask, liquid_mask

    def _to_tensor(
        self,
        image: Image.Image,
        bottle_mask: Optional[Image.Image],
        liquid_mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_t = TF.to_tensor(image)
        bottle_t = (
            (TF.to_tensor(bottle_mask) > 0.0).float()
            if bottle_mask is not None
            else torch.zeros(1, self.image_size, self.image_size)
        )
        liquid_t = (
            (TF.to_tensor(liquid_mask) > 0.0).float()
            if liquid_mask is not None
            else torch.zeros(1, self.image_size, self.image_size)
        )
        return image_t, bottle_t, liquid_t


class LCDTCCropDataset(BaseBottleDataset):
    """Crop bottle ROI from LCDTC and use liquid state labels."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 320,
        context: float = 0.15,
        train: Optional[bool] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(
            image_size=image_size,
            train=(split == "train") if train is None else train,
            context=context,
        )
        self.data_dir = Path(data_dir)
        ann_name = "instances_train2017.json" if split == "train" else "instances_val2017.json"
        ann_path = self.data_dir / "annotations" / ann_name
        img_dir = self.data_dir / "images" / ("train2017" if split == "train" else "val2017")

        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        image_lookup = {
            int(img_info["id"]): img_dir / img_info["file_name"] for img_info in coco["images"]
        }

        records: List[SampleRecord] = []
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            bbox = (
                int(round(x)),
                int(round(y)),
                int(round(x + w)),
                int(round(y + h)),
            )
            state_label = int(ann["category_id"])
            records.append(
                SampleRecord(
                    image_key=str(image_lookup[int(ann["image_id"])]),
                    mask_key=None,
                    bbox_xyxy=bbox,
                    state_label=state_label,
                    binary_label=0 if state_label == 0 else 1,
                    has_bottle_mask=False,
                    has_liquid_mask=False,
                    task_id=0,
                )
            )

        if max_samples is not None:
            records = records[:max_samples]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image = _safe_open_image(Path(record.image_key))
        image, _ = _crop_image_and_mask(image, None, record.bbox_xyxy, self.context)
        image, bottle_mask, liquid_mask = self._augment(image, None, None)
        image_t, bottle_t, liquid_t = self._to_tensor(image, bottle_mask, liquid_mask)
        return {
            "image": image_t,
            "bottle_mask": bottle_t,
            "liquid_mask": liquid_t,
            "state_label": torch.tensor(record.state_label, dtype=torch.long),
            "binary_label": torch.tensor(record.binary_label, dtype=torch.float32),
            "has_bottle_mask": torch.tensor(0, dtype=torch.bool),
            "has_liquid_mask": torch.tensor(0, dtype=torch.bool),
            "has_state_label": torch.tensor(1, dtype=torch.bool),
            "task_id": torch.tensor(record.task_id, dtype=torch.long),
        }


class _ZipReader:
    """Lazily open zip archives inside each worker."""

    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        self._zip_file: Optional[zipfile.ZipFile] = None

    @property
    def zip_file(self) -> zipfile.ZipFile:
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path)
        return self._zip_file

    def read_image(self, key: str, mode: str) -> Image.Image:
        with self.zip_file.open(key, "r") as f:
            data = f.read()
        with Image.open(io.BytesIO(data)) as img:
            return img.convert(mode)


class TransparentObjectSegDataset(BaseBottleDataset):
    """Transparent object segmentation dataset with zip/extracted support."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        difficulty: str = "all",
        image_size: int = 384,
        context: float = 0.08,
        train: Optional[bool] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(
            image_size=image_size,
            train=(split == "train") if train is None else train,
            context=context,
            color_jitter=0.2,
        )
        self.data_dir = Path(data_dir)
        self.split = split
        self.difficulty = difficulty
        self.reader: Optional[_ZipReader] = None

        extracted_root = self.data_dir / split
        zip_path = self.data_dir / f"{split}.zip"

        if extracted_root.exists():
            self.mode = "dir"
            self.root = extracted_root
        elif zip_path.exists():
            self.mode = "zip"
            self.root = zip_path
            self.reader = _ZipReader(zip_path)
        else:
            raise FileNotFoundError(
                f"Could not find extracted directory or zip archive for split={split!r} in {self.data_dir}"
            )

        self.records = self._build_records(max_samples=max_samples)

    def _build_records(self, max_samples: Optional[int]) -> List[SampleRecord]:
        if self.mode == "dir":
            records = self._build_records_from_dir()
        else:
            records = self._build_records_from_zip()
        if max_samples is not None:
            records = records[:max_samples]
        return records

    def _iter_validation_subsets(self) -> Iterable[str]:
        if self.split == "train":
            yield "train"
            return
        if self.difficulty == "all":
            yield from ("easy", "hard")
        else:
            yield self.difficulty

    def _build_records_from_dir(self) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        if self.split == "train":
            image_dir = self.root / "images"
            mask_dir = self.root / "masks"
            for mask_path in sorted(mask_dir.glob("*_mask.png")):
                stem = mask_path.name.replace("_mask.png", "")
                image_path = image_dir / f"{stem}.jpg"
                if not image_path.exists():
                    image_path = image_dir / f"{stem}.png"
                if not image_path.exists():
                    continue
                records.append(
                    SampleRecord(
                        image_key=str(image_path),
                        mask_key=str(mask_path),
                        bbox_xyxy=None,
                        state_label=-1,
                        binary_label=-1,
                        has_bottle_mask=True,
                        has_liquid_mask=False,
                        task_id=1,
                    )
                )
            return records

        for subset in self._iter_validation_subsets():
            image_dir = self.root / subset / "images"
            mask_dir = self.root / subset / "masks"
            for mask_path in sorted(mask_dir.glob("*_mask.png")):
                stem = mask_path.name.replace("_mask.png", "")
                image_path = image_dir / f"{stem}.jpg"
                if not image_path.exists():
                    image_path = image_dir / f"{stem}.png"
                if not image_path.exists():
                    continue
                records.append(
                    SampleRecord(
                        image_key=str(image_path),
                        mask_key=str(mask_path),
                        bbox_xyxy=None,
                        state_label=-1,
                        binary_label=-1,
                        has_bottle_mask=True,
                        has_liquid_mask=False,
                        task_id=1,
                    )
                )
        return records

    def _build_records_from_zip(self) -> List[SampleRecord]:
        assert self.reader is not None
        names = self.reader.zip_file.namelist()
        records: List[SampleRecord] = []

        def add_record(image_key: str, mask_key: str) -> None:
            records.append(
                SampleRecord(
                    image_key=image_key,
                    mask_key=mask_key,
                    bbox_xyxy=None,
                    state_label=-1,
                    binary_label=-1,
                    has_bottle_mask=True,
                    has_liquid_mask=False,
                    task_id=1,
                )
            )

        if self.split == "train":
            mask_keys = sorted(
                name
                for name in names
                if name.startswith("train/masks/") and name.endswith("_mask.png")
            )
            for mask_key in mask_keys:
                stem = Path(mask_key).name.replace("_mask.png", "")
                image_key = f"train/images/{stem}.jpg"
                if image_key not in names:
                    image_key = f"train/images/{stem}.png"
                if image_key not in names:
                    continue
                add_record(image_key, mask_key)
            return records

        subsets = list(self._iter_validation_subsets())
        for subset in subsets:
            prefix = f"{self.split}/{subset}/masks/"
            mask_keys = sorted(
                name for name in names if name.startswith(prefix) and name.endswith("_mask.png")
            )
            for mask_key in mask_keys:
                stem = Path(mask_key).name.replace("_mask.png", "")
                image_key = f"{self.split}/{subset}/images/{stem}.jpg"
                if image_key not in names:
                    image_key = f"{self.split}/{subset}/images/{stem}.png"
                if image_key not in names:
                    continue
                add_record(image_key, mask_key)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_and_mask(self, record: SampleRecord) -> Tuple[Image.Image, Image.Image]:
        if self.mode == "dir":
            image = _safe_open_image(Path(record.image_key))
            mask = _safe_open_mask(Path(record.mask_key or ""))
            return image, mask
        assert self.reader is not None
        image = self.reader.read_image(record.image_key, "RGB")
        mask = self.reader.read_image(record.mask_key or "", "L")
        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image, bottle_mask = self._load_image_and_mask(record)
        mask_np = np.array(bottle_mask)
        bbox = _bbox_from_mask(mask_np)
        image, bottle_mask = _crop_image_and_mask(
            image, bottle_mask, bbox, self.context
        )
        image, bottle_mask, liquid_mask = self._augment(image, bottle_mask, None)
        image_t, bottle_t, liquid_t = self._to_tensor(image, bottle_mask, liquid_mask)
        return {
            "image": image_t,
            "bottle_mask": bottle_t,
            "liquid_mask": liquid_t,
            "state_label": torch.tensor(-1, dtype=torch.long),
            "binary_label": torch.tensor(-1.0, dtype=torch.float32),
            "has_bottle_mask": torch.tensor(1, dtype=torch.bool),
            "has_liquid_mask": torch.tensor(0, dtype=torch.bool),
            "has_state_label": torch.tensor(0, dtype=torch.bool),
            "task_id": torch.tensor(record.task_id, dtype=torch.long),
        }
