"""Train ROI liquid state classifier on LCDTC."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .datasets import LCDTCStateDataset
from .models import build_state_classifier
from .utils import compute_state_metrics, save_jsonl, seed_everything


def evaluate(model, loader, device):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["state_label"].to(device)
            out = model(images)
            preds.append(out["state_logits"].argmax(dim=1).cpu())
            gts.append(gt.cpu())
    pred_t = torch.cat(preds)
    gt_t = torch.cat(gts)
    return compute_state_metrics(pred_t, gt_t)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("LCDTC state classifier training")
    parser.add_argument("--data-root", default=str(root / "datasets" / "LCDTC"))
    parser.add_argument("--output-dir", default=str(root / "output_lcdtc_state"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--context", type=float, default=0.15)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--binary-loss-weight", type=float, default=0.3)
    parser.add_argument("--print-interval", type=int, default=20)
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
    train_set = LCDTCStateDataset(
        args.data_root,
        split="train",
        image_size=args.image_size,
        context=args.context,
        train=True,
        max_samples=args.max_train_samples,
    )
    val_set = LCDTCStateDataset(
        args.data_root,
        split="val",
        image_size=args.image_size,
        context=args.context,
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
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_state_classifier(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        num_states=5,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_score = -1.0
    history_path = Path(args.output_dir) / "history.jsonl"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train/val crops: {len(train_set)} / {len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = 0.0
        n = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            state_gt = batch["state_label"].to(device)
            binary_gt = batch["binary_label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(images)
                state_loss = F.cross_entropy(
                    out["state_logits"], state_gt, label_smoothing=args.label_smoothing
                )
                binary_loss = F.binary_cross_entropy_with_logits(out["binary_logits"], binary_gt)
                loss = state_loss + args.binary_loss_weight * binary_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            meter += float(loss.detach().item())
            n += 1
            if step % args.print_interval == 0 or step == len(train_loader):
                print(
                    f"[{epoch}/{args.epochs}] step={step}/{len(train_loader)} "
                    f"loss={meter / max(n, 1):.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"  Val state_acc={metrics['state_acc']:.4f} state_macro_f1={metrics['state_macro_f1']:.4f} "
            f"binary_acc={metrics['binary_acc']:.4f} binary_f1={metrics['binary_f1']:.4f} time={dt:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": meter / max(n, 1),
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
        score = metrics["state_macro_f1"] + 0.5 * metrics["binary_f1"]
        if score > best_score:
            best_score = score
            torch.save(ckpt, Path(args.output_dir) / "best.pth")
            print(f"  -> Saved best state model (score={best_score:.4f})")


if __name__ == "__main__":
    main()
