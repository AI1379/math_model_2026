"""End-to-end inference: detector + ROI state classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from .datasets import STATE_NAMES
from .models import build_detector, build_state_classifier


def load_detector(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    model = build_detector(
        name=args.get("detector", "fasterrcnn_mobilenet_v3_large_fpn"),
        pretrained=False,
        num_classes=2,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args


def load_state_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    model = build_state_classifier(
        backbone=args.get("backbone", "resnet18"),
        pretrained=False,
        num_states=5,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args


def expand_box(box, width: int, height: int, context: float):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(bw * context))
    pad_y = int(round(bh * context))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def predict_single(
    image_path: Path,
    detector,
    state_model,
    detector_args,
    state_args,
    device: torch.device,
    det_thresh: float,
):
    orig = Image.open(image_path).convert("RGB")
    det_size = int(detector_args.get("image_size", 800))
    det_img = TF.resize(orig, [det_size, det_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    det_tensor = TF.to_tensor(det_img).to(device)
    with torch.no_grad():
        pred = detector([det_tensor])[0]
    if pred["boxes"].numel() == 0:
        return []

    mask = pred["scores"] >= det_thresh
    boxes = pred["boxes"][mask].cpu()
    scores = pred["scores"][mask].cpu()
    sx = orig.width / det_size
    sy = orig.height / det_size

    outputs = []
    crop_size = int(state_args.get("image_size", 320))
    context = float(state_args.get("context", 0.15))

    for box, score in zip(boxes, scores):
        box = box.clone()
        box[[0, 2]] *= sx
        box[[1, 3]] *= sy
        crop_box = expand_box(box.tolist(), orig.width, orig.height, context)
        crop = orig.crop(crop_box)
        crop = TF.resize(crop, [crop_size, crop_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        crop_t = TF.to_tensor(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = state_model(crop_t)
            probs = torch.softmax(out["state_logits"], dim=1)[0]
            state_id = int(probs.argmax().item())
            binary_prob = float(torch.sigmoid(out["binary_logits"])[0].item())

        outputs.append(
            {
                "bbox_xyxy": [float(v) for v in box.tolist()],
                "det_score": float(score.item()),
                "state_id": state_id,
                "state_name": STATE_NAMES[state_id],
                "state_prob": float(probs[state_id].item()),
                "binary_prob": binary_prob,
                "joint_score": float(score.item() * probs[state_id].item()),
            }
        )

    outputs.sort(key=lambda x: x["joint_score"], reverse=True)
    return outputs


def draw_predictions(image_path: Path, preds: List[Dict], out_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for pred in preds:
        x1, y1, x2, y2 = pred["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=3)
        text = f"{pred['state_name']} det={pred['det_score']:.2f} state={pred['state_prob']:.2f}"
        draw.text((x1 + 4, max(0, y1 - 14)), text, fill=(255, 64, 64))
    image.save(out_path)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("LCDTC full-system inference")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--detector-ckpt", default=str(root / "output_lcdtc_detector" / "best.pth"))
    parser.add_argument("--state-ckpt", default=str(root / "output_lcdtc_state" / "best.pth"))
    parser.add_argument("--output-dir", default=str(root / "output_lcdtc_infer"))
    parser.add_argument("--det-thresh", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    detector, detector_args = load_detector(args.detector_ckpt, device)
    state_model, state_args = load_state_model(args.state_ckpt, device)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    else:
        images = [input_path]

    all_results = []
    for image_path in images:
        preds = predict_single(
            image_path,
            detector,
            state_model,
            detector_args,
            state_args,
            device,
            args.det_thresh,
        )
        all_results.append({"image": str(image_path), "predictions": preds})
        out_image = Path(args.output_dir) / f"{image_path.stem}_pred.jpg"
        draw_predictions(image_path, preds[:3], out_image)
        print(f"{image_path.name}: {preds[:1]}")

    with (Path(args.output_dir) / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
