"""Evaluate detector + state classifier end-to-end on LCDTC val."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from .datasets import LCDTCDetectionDataset
from .infer import expand_box, load_detector, load_state_model
from .utils import compute_ap


def build_ground_truths(targets):
    gt_category = {}
    gt_state = {cid: {} for cid in range(5)}
    gt_joint = {cid: {} for cid in range(5)}
    for img_idx, target in enumerate(targets):
        boxes = target["boxes"].cpu()
        states = target["states"].cpu()
        gt_category[img_idx] = boxes
        for cid in range(5):
            mask = states == cid
            gt_state[cid][img_idx] = boxes[mask]
            gt_joint[cid][img_idx] = boxes[mask]
    return gt_category, gt_state, gt_joint


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("Evaluate LCDTC full pipeline")
    parser.add_argument("--data-root", default=str(root / "datasets" / "LCDTC"))
    parser.add_argument("--detector-ckpt", default=str(root / "output_lcdtc_detector" / "best.pth"))
    parser.add_argument("--state-ckpt", default=str(root / "output_lcdtc_state" / "best.pth"))
    parser.add_argument("--det-thresh", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, det_args = load_detector(args.detector_ckpt, device)
    state_model, state_args = load_state_model(args.state_ckpt, device)
    dataset = LCDTCDetectionDataset(
        args.data_root,
        split="val",
        image_size=int(det_args.get("image_size", 800)),
        annotation_name="instances_val2017.json",
        train=False,
        max_samples=args.max_samples,
    )

    det_size = int(det_args.get("image_size", 800))
    cls_size = int(state_args.get("image_size", 320))
    context = float(state_args.get("context", 0.15))

    all_targets = []
    det_for_apc = []
    det_for_apt = {cid: [] for cid in range(5)}
    det_for_apct = {cid: [] for cid in range(5)}

    for img_idx in range(len(dataset)):
        image_t, target = dataset[img_idx]
        all_targets.append(target)
        with torch.no_grad():
            pred = detector([image_t.to(device)])[0]

        image_path = dataset.image_lookup[dataset.image_ids[img_idx]]["path"]
        orig = Image.open(image_path).convert("RGB")
        sx = orig.width / det_size
        sy = orig.height / det_size

        if pred["boxes"].numel() == 0:
            continue
        keep = pred["scores"] >= args.det_thresh
        boxes = pred["boxes"][keep].cpu()
        scores = pred["scores"][keep].cpu()

        for box, score in zip(boxes, scores):
            box_orig = box.clone()
            box_orig[[0, 2]] *= sx
            box_orig[[1, 3]] *= sy
            crop_box = expand_box(box_orig.tolist(), orig.width, orig.height, context)
            crop = orig.crop(crop_box)
            crop = TF.resize(crop, [cls_size, cls_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            crop_t = TF.to_tensor(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                out = state_model(crop_t)
                probs = torch.softmax(out["state_logits"], dim=1)[0].cpu()
                state_id = int(probs.argmax().item())
                state_prob = float(probs[state_id].item())
            det_for_apc.append((img_idx, float(score.item()), box))
            det_for_apt[state_id].append((img_idx, state_prob, box))
            det_for_apct[state_id].append((img_idx, float(score.item() * state_prob), box))

    gt_category, gt_state, gt_joint = build_ground_truths(all_targets)
    apc = compute_ap(det_for_apc, gt_category, iou_thresh=0.5)
    apt = sum(compute_ap(det_for_apt[cid], gt_state[cid], 0.5) for cid in range(5)) / 5.0
    apct = sum(compute_ap(det_for_apct[cid], gt_joint[cid], 0.5) for cid in range(5)) / 5.0
    mapct = sum(
        sum(compute_ap(det_for_apct[cid], gt_joint[cid], 0.5 + 0.05 * i) for cid in range(5)) / 5.0
        for i in range(10)
    ) / 10.0
    print(f"APc@0.5={apc:.4f} APt@0.5={apt:.4f} APct@0.5={apct:.4f} mAPct={mapct:.4f}")


if __name__ == "__main__":
    main()
