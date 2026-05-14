"""Multitask training entrypoint for the v2 bottle liquid model.

Key differences from v1:
  - Uses LiquidV2Net (CNN + Transformer hybrid)
  - Higher decoder_dim default (256) to match transformer enc_dim
  - Slightly lower LR for stable transformer training
  - Same loss functions and multi-dataset training loop as v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Allow running as `python -m liquid_v2.train` from problems/A/
# by adding the parent directory (problems/A/) to sys.path.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from liquid_v1.datasets import (
    LCDTCCropDataset,
    STATE_NAMES,
    TransparentObjectSegDataset,
)
from liquid_v2.model import LiquidV2Net


# ═══════════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════════


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════════
# Loss functions (same as v1)
# ═══════════════════════════════════════════════════════════════════════════════════


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def masked_bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def compute_area_ratio_loss(
    bottle_logits: torch.Tensor,
    liquid_logits: torch.Tensor,
    state_labels: torch.Tensor,
) -> torch.Tensor:
    bottle = torch.sigmoid(bottle_logits).flatten(1)
    liquid = torch.sigmoid(liquid_logits).flatten(1)
    ratio = liquid.sum(dim=1) / bottle.sum(dim=1).clamp(min=1e-6)

    lower = torch.tensor([0.0, 0.01, 0.18, 0.45, 0.72], device=ratio.device)
    upper = torch.tensor([0.03, 0.18, 0.45, 0.72, 1.05], device=ratio.device)
    l = lower[state_labels]
    u = upper[state_labels]
    return (F.relu(l - ratio) + F.relu(ratio - u)).mean()


# ═══════════════════════════════════════════════════════════════════════════════════
# Loss aggregation
# ═══════════════════════════════════════════════════════════════════════════════════


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: Dict[str, torch.Tensor] = {}

    has_state = batch["has_state_label"]
    if has_state.any():
        idx = has_state.nonzero(as_tuple=True)[0]
        state_labels = batch["state_label"][idx]
        binary_labels = batch["binary_label"][idx]
        losses["state"] = F.cross_entropy(
            outputs["state_logits"][idx],
            state_labels,
            label_smoothing=args.label_smoothing,
        )
        losses["binary"] = F.binary_cross_entropy_with_logits(
            outputs["binary_logits"][idx], binary_labels
        )
        if args.area_prior_weight > 0:
            losses["area"] = compute_area_ratio_loss(
                outputs["bottle_logits"][idx],
                outputs["liquid_logits"][idx],
                state_labels,
            )

    has_bottle = batch["has_bottle_mask"]
    if has_bottle.any():
        idx = has_bottle.nonzero(as_tuple=True)[0]
        losses["bottle"] = masked_bce_dice_loss(
            outputs["bottle_logits"][idx], batch["bottle_mask"][idx]
        )

    has_liquid = batch["has_liquid_mask"]
    if has_liquid.any():
        idx = has_liquid.nonzero(as_tuple=True)[0]
        losses["liquid"] = masked_bce_dice_loss(
            outputs["liquid_logits"][idx], batch["liquid_mask"][idx]
        )

    total = torch.tensor(0.0, device=outputs["binary_logits"].device)
    if "state" in losses:
        total = total + args.state_loss_weight * losses["state"]
    if "binary" in losses:
        total = total + args.binary_loss_weight * losses["binary"]
    if "area" in losses:
        total = total + args.area_prior_weight * losses["area"]
    if "bottle" in losses:
        total = total + args.bottle_mask_loss_weight * losses["bottle"]
    if "liquid" in losses:
        total = total + args.liquid_mask_loss_weight * losses["liquid"]

    loss_dict = {k: float(v.detach().item()) for k, v in losses.items()}
    loss_dict["total"] = float(total.detach().item())
    return total, loss_dict


# ═══════════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════════


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def cycle(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_lcdtc(
    model: LiquidV2Net,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    n = 0
    binary_correct = 0
    state_correct = 0
    binary_tp = 0
    binary_fp = 0
    binary_fn = 0
    state_preds = []
    state_gts = []

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["image"])
        binary_pred = (torch.sigmoid(outputs["binary_logits"]) >= 0.5).long()
        state_pred = outputs["state_logits"].argmax(dim=1)
        state_gt = batch["state_label"]
        binary_gt = batch["binary_label"].long()

        n += state_gt.numel()
        binary_correct += (binary_pred == binary_gt).sum().item()
        state_correct += (state_pred == state_gt).sum().item()
        binary_tp += ((binary_pred == 1) & (binary_gt == 1)).sum().item()
        binary_fp += ((binary_pred == 1) & (binary_gt == 0)).sum().item()
        binary_fn += ((binary_pred == 0) & (binary_gt == 1)).sum().item()
        state_preds.append(state_pred.cpu())
        state_gts.append(state_gt.cpu())

    state_preds_t = torch.cat(state_preds)
    state_gts_t = torch.cat(state_gts)
    macro_f1 = 0.0
    for cls in range(len(STATE_NAMES)):
        tp = ((state_preds_t == cls) & (state_gts_t == cls)).sum().item()
        fp = ((state_preds_t == cls) & (state_gts_t != cls)).sum().item()
        fn = ((state_preds_t != cls) & (state_gts_t == cls)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        macro_f1 += f1
    macro_f1 /= len(STATE_NAMES)

    binary_prec = binary_tp / max(binary_tp + binary_fp, 1)
    binary_rec = binary_tp / max(binary_tp + binary_fn, 1)
    binary_f1 = (
        0.0
        if binary_prec + binary_rec == 0
        else 2 * binary_prec * binary_rec / (binary_prec + binary_rec)
    )

    return {
        "binary_acc": binary_correct / max(n, 1),
        "state_acc": state_correct / max(n, 1),
        "binary_f1": binary_f1,
        "state_macro_f1": macro_f1,
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Dataset construction
# ═══════════════════════════════════════════════════════════════════════════════════


def build_datasets(args: argparse.Namespace):
    lcd_train = LCDTCCropDataset(
        args.lcdtc_root,
        split="train",
        image_size=args.lcd_image_size,
        context=args.lcd_context,
        max_samples=args.max_lcd_train_samples,
    )
    lcd_val = LCDTCCropDataset(
        args.lcdtc_root,
        split="val",
        image_size=args.lcd_image_size,
        context=args.lcd_context,
        train=False,
        max_samples=args.max_lcd_val_samples,
    )

    seg_train = None
    seg_val = None
    if not args.disable_trans_seg:
        seg_train = TransparentObjectSegDataset(
            args.trans_seg_root,
            split="train",
            image_size=args.seg_image_size,
            context=args.seg_context,
            max_samples=args.max_seg_train_samples,
        )
        seg_val = TransparentObjectSegDataset(
            args.trans_seg_root,
            split="validation",
            difficulty="all",
            image_size=args.seg_image_size,
            context=args.seg_context,
            train=False,
            max_samples=args.max_seg_val_samples,
        )
    return lcd_train, lcd_val, seg_train, seg_val


# ═══════════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════════


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Datasets ----
    lcd_train, lcd_val, seg_train, _seg_val = build_datasets(args)
    lcd_loader = make_loader(
        lcd_train, args.lcd_batch, shuffle=True, workers=args.workers
    )
    lcd_val_loader = make_loader(
        lcd_val, args.lcd_batch, shuffle=False, workers=args.workers
    )
    seg_loader = (
        make_loader(seg_train, args.seg_batch, shuffle=True, workers=args.workers)
        if seg_train is not None
        else None
    )
    seg_iter = cycle(seg_loader) if seg_loader is not None else None

    # ---- Model ----
    model = LiquidV2Net(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        decoder_dim=args.decoder_dim,
        num_states=len(STATE_NAMES),
        dropout=args.dropout,
        enc_dim=args.enc_dim,
        enc_heads=args.enc_heads,
        enc_layers=args.enc_layers,
        cls_heads=args.cls_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model params: {n_params / 1e6:.2f}M")
    print(f"LCDTC train/val: {len(lcd_train)} / {len(lcd_val)}")
    if seg_train is not None:
        print(f"Transparent seg train: {len(seg_train)}")

    # ---- Optimizer ----
    # Use lower LR for transformer components
    transformer_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "transformer_enc" in name or "cls_head" in name:
            transformer_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": transformer_params, "lr": args.lr * args.transformer_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- Training state ----
    best_metric = -1.0
    history_path = Path(args.output_dir) / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meter = {
            "total": 0.0,
            "state": 0.0,
            "binary": 0.0,
            "area": 0.0,
            "bottle": 0.0,
            "liquid": 0.0,
        }
        steps = 0
        lcd_iter = iter(lcd_loader)

        for lcd_batch in lcd_iter:
            steps += 1
            optimizer.zero_grad(set_to_none=True)

            lcd_batch = move_batch(lcd_batch, device)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                lcd_outputs = model(lcd_batch["image"])
                total_loss, loss_dict = compute_losses(lcd_outputs, lcd_batch, args)

            scaler.scale(total_loss).backward()

            if seg_iter is not None:
                seg_batch = move_batch(next(seg_iter), device)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    seg_outputs = model(seg_batch["image"])
                    seg_loss, seg_loss_dict = compute_losses(
                        seg_outputs, seg_batch, args
                    )
                    seg_loss = seg_loss * args.seg_task_weight
                scaler.scale(seg_loss).backward()
                loss_dict["bottle"] = loss_dict.get("bottle", 0.0) + seg_loss_dict.get(
                    "bottle", 0.0
                )
                loss_dict["liquid"] = loss_dict.get("liquid", 0.0) + seg_loss_dict.get(
                    "liquid", 0.0
                )
                loss_dict["total"] += float(seg_loss.detach().item())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key in meter:
                meter[key] += loss_dict.get(key, 0.0)

            if steps % args.print_interval == 0 or steps == len(lcd_loader):
                avg = {k: meter[k] / max(steps, 1) for k in meter}
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{epoch}/{args.epochs}] step={steps}/{len(lcd_loader)} "
                    f"loss={avg['total']:.4f} state={avg['state']:.4f} "
                    f"binary={avg['binary']:.4f} bottle={avg['bottle']:.4f} "
                    f"area={avg['area']:.4f} lr={lr:.6f}"
                )

        scheduler.step()
        train_time = time.time() - t0

        metrics = evaluate_lcdtc(model, lcd_val_loader, device)
        print(
            f"  Val state_acc={metrics['state_acc']:.4f} "
            f"state_macro_f1={metrics['state_macro_f1']:.4f} "
            f"binary_acc={metrics['binary_acc']:.4f} "
            f"binary_f1={metrics['binary_f1']:.4f} "
            f"time={train_time:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train": {k: meter[k] / max(steps, 1) for k in meter},
            "val": metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": train_time,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        }
        torch.save(ckpt, Path(args.output_dir) / "last.pth")

        score = metrics["state_macro_f1"] + 0.5 * metrics["binary_f1"]
        if score > best_metric:
            best_metric = score
            torch.save(ckpt, Path(args.output_dir) / "best.pth")
            print(f"  -> Saved best checkpoint (score={best_metric:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    default_datasets = here / "datasets"

    parser = argparse.ArgumentParser("Bottle liquid multitask training v2")

    # ---- Paths ----
    parser.add_argument(
        "--lcdtc-root",
        default=str(default_datasets / "LCDTC"),
        help="Path to extracted LCDTC dataset root",
    )
    parser.add_argument(
        "--trans-seg-root",
        default=str(default_datasets / "Segmenting_Transparent_Objects"),
        help="Path to transparent segmentation dataset root",
    )
    parser.add_argument("--output-dir", default=str(here / "output_liquid_v2"))

    # ---- Training ----
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # ---- Model ----
    parser.add_argument(
        "--backbone", default="resnet34", choices=["resnet34", "resnet50"]
    )
    parser.add_argument("--decoder-dim", type=int, default=192)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    # Transformer encoder
    parser.add_argument("--enc-dim", type=int, default=256)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=2)
    # Cross-attention classifier
    parser.add_argument("--cls-heads", type=int, default=4)

    # ---- Data ----
    parser.add_argument("--lcd-image-size", type=int, default=320)
    parser.add_argument("--seg-image-size", type=int, default=384)
    parser.add_argument("--lcd-batch", type=int, default=40)
    parser.add_argument("--seg-batch", type=int, default=8)
    parser.add_argument("--lcd-context", type=float, default=0.15)
    parser.add_argument("--seg-context", type=float, default=0.08)

    # ---- Optimizer ----
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--transformer-lr-scale", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")

    # ---- Loss weights ----
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=0.5)
    parser.add_argument("--bottle-mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--liquid-mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--area-prior-weight", type=float, default=0.05)
    parser.add_argument("--seg-task-weight", type=float, default=0.6)

    # ---- Misc ----
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--disable-trans-seg", action="store_true")
    parser.add_argument("--max-lcd-train-samples", type=int, default=None)
    parser.add_argument("--max-lcd-val-samples", type=int, default=None)
    parser.add_argument("--max-seg-train-samples", type=int, default=None)
    parser.add_argument("--max-seg-val-samples", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
