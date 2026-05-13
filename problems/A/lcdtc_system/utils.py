"""Utilities and metrics for the LCDTC-only full system."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou

from .datasets import STATE_NAMES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_jsonl(path: str | Path, record: Dict) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_binary_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.long()
    gt = gt.long()
    acc = (pred == gt).float().mean().item()
    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_macro_f1(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 5) -> float:
    pred = pred.long()
    gt = gt.long()
    total = 0.0
    for cls in range(num_classes):
        tp = ((pred == cls) & (gt == cls)).sum().item()
        fp = ((pred == cls) & (gt != cls)).sum().item()
        fn = ((pred != cls) & (gt == cls)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        total += f1
    return total / num_classes


def compute_ap(detections, gt_by_image, iou_thresh: float = 0.5) -> float:
    n_gt = sum(gt.shape[0] for gt in gt_by_image.values())
    if n_gt == 0 or not detections:
        return 0.0

    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    tp = torch.zeros(len(detections), dtype=torch.float32)
    fp = torch.zeros(len(detections), dtype=torch.float32)
    matched = {
        img_idx: torch.zeros(gt.shape[0], dtype=torch.bool)
        for img_idx, gt in gt_by_image.items()
    }

    for i, (img_idx, _score, det_box) in enumerate(detections):
        gt_boxes = gt_by_image.get(img_idx)
        if gt_boxes is None or gt_boxes.numel() == 0:
            fp[i] = 1
            continue
        iou = box_iou(det_box.unsqueeze(0), gt_boxes).squeeze(0).cpu()
        best_iou, best_idx = iou.max(dim=0)
        if best_iou.item() >= iou_thresh and not matched[img_idx][best_idx]:
            tp[i] = 1
            matched[img_idx][best_idx] = True
        else:
            fp[i] = 1

    tp_cum = torch.cumsum(tp, 0)
    fp_cum = torch.cumsum(fp, 0)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    return ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()


def compute_detection_ap50(predictions, targets, score_thresh: float = 0.05) -> float:
    detections = []
    gt_by_image = {}
    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        gt_by_image[img_idx] = target["boxes"].cpu()
        if pred["boxes"].numel() == 0:
            continue
        mask = pred["scores"] >= score_thresh
        boxes = pred["boxes"][mask].cpu()
        scores = pred["scores"][mask].cpu()
        for j in range(boxes.shape[0]):
            detections.append((img_idx, float(scores[j].item()), boxes[j]))
    return compute_ap(detections, gt_by_image, iou_thresh=0.5)


def compute_state_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    acc = (pred == gt).float().mean().item()
    macro_f1 = compute_macro_f1(pred, gt, num_classes=len(STATE_NAMES))
    binary_metrics = compute_binary_metrics((pred > 0).long(), (gt > 0).long())
    return {
        "state_acc": acc,
        "state_macro_f1": macro_f1,
        "binary_acc": binary_metrics["acc"],
        "binary_f1": binary_metrics["f1"],
    }
