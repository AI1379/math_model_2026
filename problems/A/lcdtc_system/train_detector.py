"""Train a bottle detector on LCDTC."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets import LCDTCDetectionDataset, detection_collate_fn
from .models import build_detector
from .utils import compute_detection_ap50, save_jsonl, seed_everything


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch: int):
    model.train()
    meter = 0.0
    n = 0
    t0 = time.time()
    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        meter += float(loss.detach().item())
        n += 1
        if step % args.print_interval == 0 or step == len(loader):
            print(
                f"[{epoch}/{args.epochs}] step={step}/{len(loader)} "
                f"loss={meter / max(n, 1):.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )
    return meter / max(n, 1), time.time() - t0


@torch.no_grad()
def evaluate(model, loader, device, score_thresh: float):
    model.eval()
    preds = []
    tgts = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds.extend([{k: v.cpu() for k, v in out.items()} for out in outputs])
        tgts.extend([{k: v.cpu() for k, v in tgt.items()} for tgt in targets])
    return {"ap50": compute_detection_ap50(preds, tgts, score_thresh=score_thresh)}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("LCDTC detector training")
    parser.add_argument("--data-root", default=str(root / "datasets" / "LCDTC"))
    parser.add_argument("--output-dir", default=str(root / "output_lcdtc_detector"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--detector", default="fasterrcnn_mobilenet_v3_large_fpn")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = LCDTCDetectionDataset(
        args.data_root,
        split="train",
        image_size=args.image_size,
        train=True,
        max_samples=args.max_train_samples,
    )
    val_set = LCDTCDetectionDataset(
        args.data_root,
        split="val",
        image_size=args.image_size,
        annotation_name="instances_cls.json",
        train=False,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    model = build_detector(
        name=args.detector,
        pretrained=not args.no_pretrained,
        num_classes=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_ap50 = -1.0
    history_path = Path(args.output_dir) / "history.jsonl"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train/val images: {len(train_set)} / {len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, dt = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args, epoch
        )
        scheduler.step()
        metrics = evaluate(model, val_loader, device, args.score_thresh)
        print(f"  Val ap50={metrics['ap50']:.4f} time={dt:.1f}s")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": metrics,
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
        }
        torch.save(ckpt, Path(args.output_dir) / "last.pth")
        if metrics["ap50"] > best_ap50:
            best_ap50 = metrics["ap50"]
            torch.save(ckpt, Path(args.output_dir) / "best.pth")
            print(f"  -> Saved best detector (ap50={best_ap50:.4f})")


if __name__ == "__main__":
    main()
