"""Train Liquid V5 ROI-first state classifier.

V5 is designed for the next training run when there is no time for broad
ablation.  It keeps the best-performing local direction simple:
  - crop each LCDTC bottle ROI from the annotation box;
  - train a dedicated state classifier;
  - use ordinal/ratio losses only as auxiliary regularisation.
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
from typing import Dict, Iterable, Optional

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from lcdtc_system.datasets import LCDTCStateDataset, STATE_NAMES
from liquid_v5.model import LiquidV5Net


RATIO_CENTERS = torch.tensor([0.0, 0.12, 0.38, 0.68, 0.95], dtype=torch.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    smoothing: float,
    neighbor_smoothing: float,
) -> torch.Tensor:
    """Label smoothing plus adjacent-class mass for ordered labels."""
    targets = torch.full(
        (labels.numel(), num_classes),
        smoothing / num_classes,
        device=labels.device,
    )
    main_mass = 1.0 - smoothing
    targets.scatter_add_(1, labels[:, None], torch.full_like(labels[:, None].float(), main_mass))

    if neighbor_smoothing > 0:
        adjusted = targets * (1.0 - neighbor_smoothing)
        for offset in (-1, 1):
            neighbor = labels + offset
            valid = (neighbor >= 0) & (neighbor < num_classes)
            if valid.any():
                adjusted[valid, neighbor[valid]] += neighbor_smoothing * 0.5
            invalid = ~valid
            if invalid.any():
                adjusted[invalid, labels[invalid]] += neighbor_smoothing * 0.5
        targets = adjusted / adjusted.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return targets


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1)
    if class_weights is not None:
        sample_weights = (soft_targets * class_weights.to(logits.device)).sum(dim=1)
        loss = loss * sample_weights
    return loss.mean()


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        labels,
        reduction="none",
        weight=class_weights.to(logits.device) if class_weights is not None else None,
        label_smoothing=label_smoothing,
    )
    pt = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    return (((1.0 - pt).clamp(min=1e-6) ** gamma) * ce).mean()


def ordinal_loss(
    ordinal_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    levels = torch.arange(num_classes - 1, device=labels.device)
    targets = (labels[:, None] > levels[None, :]).float()
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


def ordinal_violation_loss(ordinal_logits: torch.Tensor) -> torch.Tensor:
    if ordinal_logits.shape[1] < 2:
        return torch.tensor(0.0, device=ordinal_logits.device)
    return F.relu(ordinal_logits[:, 1:] - ordinal_logits[:, :-1]).mean()


def distance_penalty_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Expected ordinal distance. Cheaply discourages far-away mistakes."""
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(logits.shape[1], device=logits.device).float()
    dist = (classes[None, :] - labels[:, None].float()).abs()
    return (probs * dist).sum(dim=1).mean()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    class_weights: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, float]]:
    labels = batch["state_label"]
    binary = batch["binary_label"]
    ratio_targets = RATIO_CENTERS.to(labels.device)[labels]

    if args.focal_gamma > 0:
        state_loss = focal_cross_entropy(
            outputs["state_logits"],
            labels,
            gamma=args.focal_gamma,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        soft_targets = make_soft_targets(
            labels,
            len(STATE_NAMES),
            smoothing=args.label_smoothing,
            neighbor_smoothing=args.neighbor_smoothing,
        )
        state_loss = soft_cross_entropy(outputs["state_logits"], soft_targets, class_weights)

    binary_loss = F.binary_cross_entropy_with_logits(outputs["binary_logits"], binary)
    ratio_loss = F.smooth_l1_loss(outputs["ratio_pred"], ratio_targets)
    ord_loss = ordinal_loss(
        outputs["ordinal_logits"],
        labels,
        num_classes=len(STATE_NAMES),
        label_smoothing=args.ordinal_smoothing,
    )
    rank_loss = ordinal_violation_loss(outputs["ordinal_logits"])
    dist_loss = distance_penalty_loss(outputs["state_logits"], labels)

    total = (
        args.state_loss_weight * state_loss
        + args.binary_loss_weight * binary_loss
        + args.ratio_loss_weight * ratio_loss
        + args.ordinal_loss_weight * ord_loss
        + args.rank_loss_weight * rank_loss
        + args.distance_loss_weight * dist_loss
    )
    return total, {
        "total": float(total.detach().item()),
        "state": float(state_loss.detach().item()),
        "binary": float(binary_loss.detach().item()),
        "ratio": float(ratio_loss.detach().item()),
        "ordinal": float(ord_loss.detach().item()),
        "rank": float(rank_loss.detach().item()),
        "distance": float(dist_loss.detach().item()),
    }


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.long()
    gt = gt.long()
    acc = (pred == gt).float().mean().item()
    per_class = {}
    macro_f1 = 0.0
    for cls, name in enumerate(STATE_NAMES):
        tp = ((pred == cls) & (gt == cls)).sum().item()
        fp = ((pred == cls) & (gt != cls)).sum().item()
        fn = ((pred != cls) & (gt == cls)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        per_class[f"{name}_precision"] = prec
        per_class[f"{name}_recall"] = rec
        per_class[f"{name}_f1"] = f1
        macro_f1 += f1
    macro_f1 /= len(STATE_NAMES)

    binary_pred = (pred > 0).long()
    binary_gt = (gt > 0).long()
    binary_acc = (binary_pred == binary_gt).float().mean().item()
    tp = ((binary_pred == 1) & (binary_gt == 1)).sum().item()
    fp = ((binary_pred == 1) & (binary_gt == 0)).sum().item()
    fn = ((binary_pred == 0) & (binary_gt == 1)).sum().item()
    b_prec = tp / max(tp + fp, 1)
    b_rec = tp / max(tp + fn, 1)
    binary_f1 = 0.0 if b_prec + b_rec == 0 else 2 * b_prec * b_rec / (b_prec + b_rec)

    return {
        "state_acc": acc,
        "state_macro_f1": macro_f1,
        "binary_acc": binary_acc,
        "binary_f1": binary_f1,
        **per_class,
    }


@torch.no_grad()
def evaluate(
    model: LiquidV5Net,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
) -> Dict[str, float]:
    model.eval()
    preds = []
    gts = []
    ratio_mae_sum = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["state_label"].to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["state_logits"]
        ratio_pred = outputs["ratio_pred"]
        if tta:
            flip_outputs = model(torch.flip(images, dims=[3]))
            logits = (logits + flip_outputs["state_logits"]) * 0.5
            ratio_pred = (ratio_pred + flip_outputs["ratio_pred"]) * 0.5
        pred = logits.argmax(dim=1)
        preds.append(pred.cpu())
        gts.append(labels.cpu())
        ratio_targets = RATIO_CENTERS.to(device)[labels]
        ratio_mae_sum += (ratio_pred - ratio_targets).abs().sum().item()
        n += labels.numel()

    pred_t = torch.cat(preds)
    gt_t = torch.cat(gts)
    metrics = compute_metrics(pred_t, gt_t)
    metrics["ratio_mae"] = ratio_mae_sum / max(n, 1)
    return metrics


def class_counts(dataset: LCDTCStateDataset) -> torch.Tensor:
    counts = torch.zeros(len(STATE_NAMES), dtype=torch.long)
    for record in dataset.records:
        counts[int(record["state"])] += 1
    return counts


def make_class_weights(counts: torch.Tensor, half_boost: float) -> torch.Tensor:
    inv = counts.sum().float() / counts.float().clamp(min=1.0)
    weights = inv / inv.mean()
    if half_boost > 0:
        weights[2] *= half_boost
    return weights / weights.mean()


def make_train_loader(
    dataset: LCDTCStateDataset,
    args: argparse.Namespace,
    counts: torch.Tensor,
) -> DataLoader:
    if not args.class_balanced_sampler:
        return DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.workers > 0,
        )

    sample_weights = []
    inv = counts.sum().float() / counts.float().clamp(min=1.0)
    for record in dataset.records:
        w = float(inv[int(record["state"])].item())
        if int(record["state"]) == 2:
            w *= args.half_sampler_boost
        sample_weights.append(w)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = LCDTCStateDataset(
        args.lcdtc_root,
        split="train",
        image_size=args.image_size,
        context=args.context,
        train=True,
        max_samples=args.max_train_samples,
    )
    val_set = LCDTCStateDataset(
        args.lcdtc_root,
        split="val",
        image_size=args.image_size,
        context=args.context,
        train=False,
        max_samples=args.max_val_samples,
    )

    counts = class_counts(train_set)
    class_weights = (
        make_class_weights(counts, args.half_class_boost).to(device)
        if args.use_class_weights
        else None
    )

    train_loader = make_train_loader(train_set, args, counts)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = LiquidV5Net(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        num_states=len(STATE_NAMES),
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
    ).to(device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = min(args.warmup_epochs, args.epochs)

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        denom = max(args.epochs - warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs + 1) / denom
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp and device.type == "cuda")

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: LiquidV5Net backbone={args.backbone}")
    print(f"Train/val crops: {len(train_set)} / {len(val_set)}")
    print(f"Class counts: {dict(zip(STATE_NAMES, counts.tolist()))}")
    if class_weights is not None:
        print(f"Class weights: {[round(v, 3) for v in class_weights.detach().cpu().tolist()]}")
    print(f"AMP: {'off' if args.no_amp else 'on'}  TTA val: {'on' if args.val_tta else 'off'}")

    best_score = -1.0
    best_epoch = 0
    history_path = output_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meter = {
            "total": 0.0,
            "state": 0.0,
            "binary": 0.0,
            "ratio": 0.0,
            "ordinal": 0.0,
            "rank": 0.0,
            "distance": 0.0,
        }
        steps = 0
        for step, batch in enumerate(train_loader, start=1):
            steps += 1
            images = batch["image"].to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                outputs = model(images)
                loss, loss_dict = compute_losses(outputs, batch, args, class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key in meter:
                meter[key] += loss_dict.get(key, 0.0)

            if step % args.print_interval == 0 or step == len(train_loader):
                avg = {k: meter[k] / max(steps, 1) for k in meter}
                print(
                    f"[{epoch}/{args.epochs}] step={step}/{len(train_loader)} "
                    f"loss={avg['total']:.4f} state={avg['state']:.4f} "
                    f"bin={avg['binary']:.4f} ratio={avg['ratio']:.4f} "
                    f"ord={avg['ordinal']:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()
        dt = time.time() - t0
        train_avg = {k: meter[k] / max(steps, 1) for k in meter}
        metrics = evaluate(model, val_loader, device, tta=args.val_tta)
        score = metrics["state_macro_f1"] + 0.5 * metrics["binary_f1"]
        print(
            f"  Val state_acc={metrics['state_acc']:.4f} "
            f"state_macro_f1={metrics['state_macro_f1']:.4f} "
            f"binary_acc={metrics['binary_acc']:.4f} binary_f1={metrics['binary_f1']:.4f} "
            f"half_f1={metrics['half_f1']:.4f} ratio_mae={metrics['ratio_mae']:.4f} "
            f"time={dt:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train": train_avg,
            "val": metrics,
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "time": dt,
        }
        save_jsonl(history_path, record)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "score": score,
        }
        torch.save(ckpt, output_dir / "last.pth")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(ckpt, output_dir / "best.pth")
            print(f"  -> Saved best checkpoint (score={best_score:.4f}, epoch={best_epoch})")

    print(f"Best score={best_score:.4f} at epoch {best_epoch}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    default_datasets = here / "datasets"
    parser = argparse.ArgumentParser("Liquid V5 ROI-first state training")

    parser.add_argument("--lcdtc-root", default=str(default_datasets / "LCDTC"))
    parser.add_argument("--output-dir", default=str(here / "output_liquid_v5"))

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-interval", type=int, default=20)

    parser.add_argument(
        "--backbone",
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "convnext_tiny", "efficientnet_v2_s"],
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--context", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=512)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--neighbor-smoothing", type=float, default=0.08)
    parser.add_argument("--ordinal-smoothing", type=float, default=0.03)
    parser.add_argument("--focal-gamma", type=float, default=0.0)

    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=0.2)
    parser.add_argument("--ratio-loss-weight", type=float, default=0.2)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.15)
    parser.add_argument("--rank-loss-weight", type=float, default=0.02)
    parser.add_argument("--distance-loss-weight", type=float, default=0.03)

    parser.add_argument("--use-class-weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--half-class-boost", type=float, default=1.5)
    parser.add_argument("--class-balanced-sampler", action="store_true", default=True)
    parser.add_argument("--no-class-balanced-sampler", dest="class_balanced_sampler", action="store_false")
    parser.add_argument("--half-sampler-boost", type=float, default=1.5)

    parser.add_argument("--val-tta", action="store_true", default=True)
    parser.add_argument("--no-val-tta", dest="val_tta", action="store_false")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

