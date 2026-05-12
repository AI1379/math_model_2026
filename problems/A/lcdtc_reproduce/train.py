"""LCD-YOLOX Training Script.

Reproduces the LCDTC paper's LCD-YOLOX baseline detector.
Usage:
    python -m lcdtc_reproduce.train --epochs 300 --batch 16 --lr 0.01

Paper results to reproduce (LCD-YOLOX with CrossFormer-S):
    APc@0.5=0.809, APt@0.5=0.607, APct@0.5=0.624
    mAPc=0.704, mAPt=0.533, mAPct=0.548
"""

import os
import sys
import argparse
import time
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .lcd_yolox import CSPDarknet, PAFPN, LCDHead
from .dataset import LCDTCDataset, get_loader


# ─── Grid generation ───────────────────────────────────────────────────────────

def build_grids(input_size=640, strides=(8, 16, 32)):
    """Precompute grid centers (in grid units) and strides per level.

    Returns list of (grid_xy, stride, gs) where:
      grid_xy: (gs*gs, 2) tensor of (x, y) grid indices
      stride: int
      gs: grid size (H = W = input_size // stride)
    """
    grids = []
    for s in strides:
        gs = input_size // s
        yy, xx = torch.meshgrid(torch.arange(gs, dtype=torch.float32),
                                torch.arange(gs, dtype=torch.float32),
                                indexing="ij")
        grid_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (gs*gs, 2)
        grids.append((grid_xy, s, gs))
    return grids


# ─── SimOTA Label Assignment ───────────────────────────────────────────────────

def get_predictions(preds_raw, grids, num_cls, num_content):
    """Decode raw head outputs into flat predictions across all scales.

    Prediction format per cell:
      dx, dy = offset from grid center (grid units)
      dw, dh = log(w / stride), log(h / stride)
      obj, cls..., content...

    Decoded to pixel space:
      cx = (grid_x + dx) * stride,  cy = (grid_y + dy) * stride
      w = exp(dw) * stride,  h = exp(dh) * stride
    """
    B = preds_raw[0].shape[0]
    device = preds_raw[0].device

    all_grid_xy, all_stride, all_dx, all_dy, all_dw, all_dh = [], [], [], [], [], []
    all_obj, all_cls, all_content = [], [], []

    for i, (grid_xy, stride, gs) in enumerate(grids):
        grid_dev = grid_xy.to(device)  # (gs*gs, 2)
        raw = preds_raw[i]
        n = gs * gs

        dx = raw[:, 0].view(B, n)          # (B, N)
        dy = raw[:, 1].view(B, n)
        dw = raw[:, 2].view(B, n)
        dh = raw[:, 3].view(B, n)
        obj = raw[:, 4].view(B, n)
        cls = raw[:, 5:5+num_cls].view(B, num_cls, n).permute(0, 2, 1)  # (B, N, nc)
        content = raw[:, 5+num_cls:5+num_cls+num_content].view(B, num_content, n).permute(0, 2, 1)

        all_grid_xy.append(grid_dev.unsqueeze(0).expand(B, -1, -1))
        all_stride.append(torch.full((n,), stride, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1))
        all_dx.append(dx)
        all_dy.append(dy)
        all_dw.append(dw)
        all_dh.append(dh)
        all_obj.append(obj)
        all_cls.append(cls)
        all_content.append(content)

    return {
        "grid_xy": torch.cat(all_grid_xy, 1),    # (B, M, 2)
        "stride": torch.cat(all_stride, 1),       # (B, M)
        "dx": torch.cat(all_dx, 1),               # (B, M)
        "dy": torch.cat(all_dy, 1),
        "dw": torch.cat(all_dw, 1),
        "dh": torch.cat(all_dh, 1),
        "obj": torch.cat(all_obj, 1),             # (B, M)
        "cls": torch.cat(all_cls, 1),             # (B, M, nc)
        "content": torch.cat(all_content, 1),     # (B, M, ncont)
    }


def simota_assign(preds, gt_bboxes, gt_cls, gt_content, num_cls, num_content, fg_topk=10):
    """SimOTA dynamic label assignment.

    For each GT box, selects dynamic-k positive anchors using:
      1. Center prior (within radius 2.5*stride)
      2. Cost matrix (cls_cost + reg_cost)
      3. Dynamic k based on IoU sum

    Args:
        preds: dict from get_predictions()
        gt_bboxes: (B, max_n, 4) normalized cx,cy,w,h in [0,1]
        gt_cls: (B, max_n) padded with -1
        gt_content: (B, max_n) padded with -1
    """
    B, M = preds["obj"].shape
    device = preds["obj"].device
    img_size = 640

    # Decoded boxes in pixels (clamp to prevent overflow)
    cx = (preds["grid_xy"][:, :, 0] + preds["dx"]) * preds["stride"]  # (B, M)
    cy = (preds["grid_xy"][:, :, 1] + preds["dy"]) * preds["stride"]
    w = preds["dw"].clamp(max=10).exp() * preds["stride"]
    h = preds["dh"].clamp(max=10).exp() * preds["stride"]

    # Targets
    obj_tgt = torch.zeros(B, M, device=device)
    cls_tgt = torch.zeros(B, M, num_cls, device=device)
    cont_tgt = torch.zeros(B, M, num_content, device=device)
    # Regression targets: dx_target = gt_cx/stride - grid_x, etc.
    tgt_dx = torch.zeros(B, M, device=device)
    tgt_dy = torch.zeros(B, M, device=device)
    tgt_dw = torch.zeros(B, M, device=device)
    tgt_dh = torch.zeros(B, M, device=device)
    iou_weight = torch.zeros(B, M, device=device)  # for IoU-aware cls loss

    for b in range(B):
        valid = gt_cls[b] >= 0
        if not valid.any():
            continue
        n_gt = valid.sum().item()

        g_cx = gt_bboxes[b, valid, 0] * img_size
        g_cy = gt_bboxes[b, valid, 1] * img_size
        g_w = gt_bboxes[b, valid, 2] * img_size
        g_h = gt_bboxes[b, valid, 3] * img_size
        g_x1 = g_cx - g_w / 2
        g_y1 = g_cy - g_h / 2
        g_x2 = g_cx + g_w / 2
        g_y2 = g_cy + g_h / 2
        g_cls = gt_cls[b, valid].long()
        g_cont = gt_content[b, valid].long()

        # Center prior
        is_in = torch.zeros(n_gt, M, device=device, dtype=torch.bool)
        for gi in range(n_gt):
            # Center of GT in grid units for each stride level
            radius = 2.5
            gx_grid = g_cx[gi] / preds["stride"][b]  # (M,)
            gy_grid = g_cy[gi] / preds["stride"][b]
            dist_x = (preds["grid_xy"][b, :, 0] + 0.5 - gx_grid).abs()
            dist_y = (preds["grid_xy"][b, :, 1] + 0.5 - gy_grid).abs()
            is_in[gi] = (dist_x < radius) & (dist_y < radius)

        # IoU between decoded predictions and GT
        px1 = cx[b] - w[b] / 2
        py1 = cy[b] - h[b] / 2
        px2 = cx[b] + w[b] / 2
        py2 = cy[b] + h[b] / 2

        ix1 = torch.max(px1.unsqueeze(0), g_x1.unsqueeze(1))  # (n_gt, M)
        iy1 = torch.max(py1.unsqueeze(0), g_y1.unsqueeze(1))
        ix2 = torch.min(px2.unsqueeze(0), g_x2.unsqueeze(1))
        iy2 = torch.min(py2.unsqueeze(0), g_y2.unsqueeze(1))
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_p = w[b] * h[b]
        area_g = g_w.unsqueeze(1) * g_h.unsqueeze(1)
        union = area_p.unsqueeze(0) + area_g - inter
        iou = inter / (union + 1e-8)  # (n_gt, M)

        # Cost: cls cost + reg cost (neg IoU)
        cls_onehot = F.one_hot(g_cls, num_cls).float()  # (n_gt, nc)
        cls_cost = F.binary_cross_entropy_with_logits(
            preds["cls"][b].unsqueeze(0).expand(n_gt, -1, -1),
            cls_onehot.unsqueeze(1).expand(-1, M, -1),
            reduction="none",
        ).mean(-1)  # (n_gt, M)

        reg_cost = 12.0 * (1.0 - iou)  # weighting factor from YOLOX
        cost = cls_cost + reg_cost
        cost[~is_in] = float("inf")

        # Dynamic k
        topk = min(fg_topk, M)
        iou_topk, _ = iou.sort(descending=True, dim=1)
        dynamic_k = (iou_topk[:, :topk].sum(1).int().clamp(min=1))  # (n_gt,)

        # Assign
        for gi in range(n_gt):
            finite_mask = cost[gi].isfinite()
            if not finite_mask.any():
                continue
            _, idx = cost[gi].sort()
            k = dynamic_k[gi].item()
            pos_idx = idx[:k]

            obj_tgt[b, pos_idx] = 1.0
            cls_tgt[b, pos_idx, g_cls[gi]] = 1.0
            cont_tgt[b, pos_idx, g_cont[gi]] = 1.0

            # Regression targets in grid-offset format
            s = preds["stride"][b, pos_idx]  # strides for positive anchors
            gx = preds["grid_xy"][b, pos_idx, 0]
            gy = preds["grid_xy"][b, pos_idx, 1]
            tgt_dx[b, pos_idx] = g_cx[gi] / s - gx
            tgt_dy[b, pos_idx] = g_cy[gi] / s - gy
            tgt_dw[b, pos_idx] = (g_w[gi] / s).clamp(min=1e-4).log().clamp(min=-10, max=10)
            tgt_dh[b, pos_idx] = (g_h[gi] / s).clamp(min=1e-4).log().clamp(min=-10, max=10)
            iou_weight[b, pos_idx] = iou[gi, pos_idx].detach()

    return obj_tgt, cls_tgt, cont_tgt, tgt_dx, tgt_dy, tgt_dw, tgt_dh, iou_weight


# ─── Loss ───────────────────────────────────────────────────────────────────────

def compute_loss(preds_raw, targets, grids, num_cls=1, num_content=5, lambda_content=0.8):
    """L = L_obj + L_cls + L_reg + λ * L_content."""
    device = preds_raw[0].device
    B = preds_raw[0].shape[0]

    # Pad targets
    max_n = max(t.shape[0] for t, _ in targets)
    if max_n == 0:
        max_n = 1
    gt_bboxes = torch.zeros(B, max_n, 4, device=device)
    gt_cls = torch.full((B, max_n), -1, dtype=torch.long, device=device)
    gt_content = torch.full((B, max_n), -1, dtype=torch.long, device=device)

    for b, (tgt, sz) in enumerate(targets):
        if tgt.numel() > 0 and tgt.shape[0] > 0:
            n = tgt.shape[0]
            gt_cls[b, :n] = tgt[:, 0].long()
            gt_content[b, :n] = tgt[:, 1].long()
            gt_bboxes[b, :n] = tgt[:, 2:6]

    # Decode predictions
    preds = get_predictions(preds_raw, grids, num_cls, num_content)

    # Assign
    obj_tgt, cls_tgt, cont_tgt, tgt_dx, tgt_dy, tgt_dw, tgt_dh, iou_w = simota_assign(
        preds, gt_bboxes, gt_cls, gt_content, num_cls, num_content
    )

    pos = obj_tgt > 0  # (B, M)
    n_pos = pos.sum().clamp(min=1).float()

    # 1) Objectness loss (BCE, all samples)
    loss_obj = F.binary_cross_entropy_with_logits(preds["obj"], obj_tgt, reduction="none")
    # Balance: weight positives higher
    alpha = 0.25
    weight = torch.where(obj_tgt > 0, torch.tensor(1.0, device=device), torch.tensor(alpha, device=device))
    loss_obj = (loss_obj * weight).sum() / (n_pos + alpha * (~pos).sum().float().clamp(min=1))

    # 2) Classification loss (BCE, positives only)
    loss_cls = torch.tensor(0.0, device=device)
    if pos.any():
        loss_cls = F.binary_cross_entropy_with_logits(
            preds["cls"][pos], cls_tgt[pos], reduction="none"
        ).sum() / n_pos

    # 3) Regression loss (GIoU or L1, positives only)
    loss_reg = torch.tensor(0.0, device=device)
    if pos.any():
        # L1 loss on grid offsets
        pred_dx = preds["dx"][pos]
        pred_dy = preds["dy"][pos]
        pred_dw = preds["dw"][pos]
        pred_dh = preds["dh"][pos]
        loss_reg = (
            F.l1_loss(pred_dx, tgt_dx[pos]) +
            F.l1_loss(pred_dy, tgt_dy[pos]) +
            F.l1_loss(pred_dw, tgt_dw[pos]) +
            F.l1_loss(pred_dh, tgt_dh[pos])
        ) / 4.0

    # 4) Content loss (BCE, positives only)
    loss_content = torch.tensor(0.0, device=device)
    if pos.any():
        loss_content = F.binary_cross_entropy_with_logits(
            preds["content"][pos], cont_tgt[pos], reduction="none"
        ).sum() / n_pos

    loss = loss_obj + loss_cls + loss_reg + lambda_content * loss_content

    return loss, {
        "total": loss.item(),
        "obj": loss_obj.item(),
        "cls": loss_cls.item() if isinstance(loss_cls, torch.Tensor) else loss_cls,
        "reg": loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg,
        "content": loss_content.item() if isinstance(loss_content, torch.Tensor) else loss_content,
        "n_pos": n_pos.item(),
    }


# ─── Full model ─────────────────────────────────────────────────────────────────

class LCDYOLOXModel(nn.Module):
    def __init__(self, num_cls=1, num_content=5, use_ta=True, dep_mul=0.33, wid_mul=0.5):
        super().__init__()
        base = int(64 * wid_mul)  # 32 for YOLOX-S
        self.backbone = CSPDarknet(dep_mul, wid_mul)
        in_ch = (base * 4, base * 8, base * 16)  # (128, 256, 512)
        self.neck = PAFPN(dep_mul, wid_mul, in_ch)
        self.head = LCDHead(num_cls, num_content, wid_mul, use_ta)

        self.num_cls = num_cls
        self.num_content = num_content
        self.strides = (8, 16, 32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        fpn = self.neck(c3, c4, c5)
        return self.head(fpn)


# ─── Evaluation ─────────────────────────────────────────────────────────────────

def decode_predictions(preds_raw, grids, num_cls, num_content, img_size=640, conf_thresh=0.05):
    """Decode raw head outputs to (x1, y1, x2, y2) boxes + scores."""
    device = preds_raw[0].device
    B = preds_raw[0].shape[0]
    results = []

    for b in range(B):
        boxes, scores, cat_ids, cont_ids = [], [], [], []

        for i, (grid_xy, stride, gs) in enumerate(grids):
            g = grid_xy.to(device)  # (gs*gs, 2)
            raw = preds_raw[i][b]  # (C, gs, gs)
            raw = raw.view(-1, gs * gs).permute(1, 0)  # (N, C)

            dx = raw[:, 0]
            dy = raw[:, 1]
            dw = raw[:, 2]
            dh = raw[:, 3]
            obj = raw[:, 4].sigmoid()
            cls_score = raw[:, 5:5+num_cls].sigmoid().max(1)[0]
            content_score = raw[:, 5+num_cls:5+num_cls+num_content].sigmoid()
            cont_label = content_score.max(1)[1]

            # Decode boxes to pixel coords
            cx = (g[:, 0] + dx) * stride
            cy = (g[:, 1] + dy) * stride
            w = dw.clamp(max=10).exp() * stride
            h = dh.clamp(max=10).exp() * stride

            # Clamp to image size
            cx = cx.clamp(0, img_size)
            cy = cy.clamp(0, img_size)
            w = w.clamp(0, img_size)
            h = h.clamp(0, img_size)

            score = obj * cls_score
            mask = score > conf_thresh

            if mask.any():
                boxes.append(torch.stack([cx[mask] - w[mask]/2, cy[mask] - h[mask]/2,
                                          cx[mask] + w[mask]/2, cy[mask] + h[mask]/2], dim=1))
                scores.append(score[mask])
                cat_ids.append(torch.zeros(mask.sum(), dtype=torch.long, device=device))
                cont_ids.append(cont_label[mask])

        if boxes:
            results.append({
                "boxes": torch.cat(boxes),
                "scores": torch.cat(scores),
                "cat_ids": torch.cat(cat_ids),
                "cont_ids": torch.cat(cont_ids),
            })
        else:
            results.append(None)

    return results


def compute_ap(det_boxes, det_scores, gt_boxes, iou_thresh=0.5):
    """Compute AP for a single category at a given IoU threshold."""
    if len(gt_boxes) == 0:
        return 0.0

    # Sort detections by score (descending)
    order = det_scores.argsort(descending=True)
    det_boxes = det_boxes[order]
    det_scores = det_scores[order]

    n_det = len(det_boxes)
    n_gt = len(gt_boxes)
    tp = torch.zeros(n_det)
    fp = torch.zeros(n_det)
    matched = set()

    for i in range(n_det):
        # IoU with all unmatched GTs
        ix1 = torch.max(det_boxes[i, 0], gt_boxes[:, 0])
        iy1 = torch.max(det_boxes[i, 1], gt_boxes[:, 1])
        ix2 = torch.min(det_boxes[i, 2], gt_boxes[:, 2])
        iy2 = torch.min(det_boxes[i, 3], gt_boxes[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_det = (det_boxes[i, 2] - det_boxes[i, 0]) * (det_boxes[i, 3] - det_boxes[i, 1])
        area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union = area_det + area_gt - inter
        iou = inter / (union + 1e-8)

        best_j = iou.argmax().item()
        if iou[best_j] >= iou_thresh and best_j not in matched:
            tp[i] = 1
            matched.add(best_j)
        else:
            fp[i] = 1

    tp_cum = torch.cumsum(tp, 0)
    fp_cum = torch.cumsum(fp, 0)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    # VOC-style AP (all-point)
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()
    return ap


def evaluate(model, val_loader, grids, num_cls=1, num_content=5, img_size=640):
    """Evaluate APc, APt, APct at IoU 0.5 and mAP@[0.5:0.95]."""
    model.eval()
    device = next(model.parameters()).device
    grids_dev = [(g[0].to(device), s, gs) for g, s, gs in [(grids[i], grids[i][1], grids[i][2]) for i in range(len(grids))]]
    grids_dev = [(g[0].to(device), g[1], g[2]) for g in grids]

    all_dets = []  # list of (detection_dict, gt_list, orig_size) per image
    img_idx = 0

    with torch.no_grad():
        for imgs, tgts, sizes in val_loader:
            imgs = imgs.to(device)
            preds_raw = model(imgs)
            dets = decode_predictions(preds_raw, grids_dev, num_cls, num_content, img_size)

            for b in range(len(dets)):
                orig_h, orig_w = sizes[b, 0].item(), sizes[b, 1].item()
                sx, sy = orig_w / img_size, orig_h / img_size

                # Scale GT boxes to pixels
                gt_list = []
                if tgts[b].numel() > 0 and tgts[b].shape[0] > 0:
                    for k in range(tgts[b].shape[0]):
                        cx, cy, w, h = tgts[b][k, 2:6]
                        gt_list.append({
                            "x1": (cx - w/2).item() * orig_w,
                            "y1": (cy - h/2).item() * orig_h,
                            "x2": (cx + w/2).item() * orig_w,
                            "y2": (cy + h/2).item() * orig_h,
                            "cls": int(tgts[b][k, 0].item()),
                            "content": int(tgts[b][k, 1].item()),
                        })

                # Scale detection boxes
                det = dets[b]
                if det is not None:
                    det["boxes"][:, 0] *= sx
                    det["boxes"][:, 1] *= sy
                    det["boxes"][:, 2] *= sx
                    det["boxes"][:, 3] *= sy

                all_dets.append((det, gt_list))
                img_idx += 1

    # Compute APct@0.5 (combined category + content = 5 classes)
    apct_05 = _compute_combined_ap(all_dets, iou_thresh=0.5)

    # Compute APc@0.5 (category only: bottle vs background = 1 class)
    apc_05 = _compute_category_ap(all_dets, iou_thresh=0.5)

    # Compute APt@0.5 (content only: 5 liquid states)
    apt_05 = _compute_content_ap(all_dets, iou_thresh=0.5)

    # mAP@[0.5:0.95]
    apcts = []
    for thr in [0.5 + 0.05 * i for i in range(10)]:
        apcts.append(_compute_combined_ap(all_dets, thr))
    mapct = sum(apcts) / len(apcts)

    print(f"  APc@0.5={apc_05:.3f}  APt@0.5={apt_05:.3f}  APct@0.5={apct_05:.3f}  mAPct={mapct:.3f}")
    return {"APc@0.5": apc_05, "APt@0.5": apt_05, "APct@0.5": apct_05, "mAPct": mapct}


def _compute_combined_ap(all_dets, iou_thresh):
    """APct: each (container_category + liquid_state) is a separate class -> 5 classes."""
    aps = []
    for content_id in range(5):
        det_boxes_list, det_scores_list, gt_boxes_list = [], [], []
        for det, gts in all_dets:
            # GTs for this content state
            gt_for_class = torch.tensor(
                [[g["x1"], g["y1"], g["x2"], g["y2"]] for g in gts if g["content"] == content_id],
                dtype=torch.float32
            ) if any(g["content"] == content_id for g in gts) else torch.zeros(0, 4)

            if det is not None:
                mask = det["cont_ids"] == content_id
                if mask.any():
                    det_boxes_list.append(det["boxes"][mask].cpu())
                    det_scores_list.append(det["scores"][mask].cpu())
                else:
                    det_boxes_list.append(torch.zeros(0, 4))
                    det_scores_list.append(torch.zeros(0))
            else:
                det_boxes_list.append(torch.zeros(0, 4))
                det_scores_list.append(torch.zeros(0))

            gt_boxes_list.append(gt_for_class)

        all_det_boxes = torch.cat(det_boxes_list) if any(b.numel() > 0 for b in det_boxes_list) else torch.zeros(0, 4)
        all_det_scores = torch.cat(det_scores_list) if any(s.numel() > 0 for s in det_scores_list) else torch.zeros(0)
        all_gt_boxes = torch.cat(gt_boxes_list) if any(g.numel() > 0 for g in gt_boxes_list) else torch.zeros(0, 4)

        if all_gt_boxes.numel() > 0:
            ap = compute_ap(all_det_boxes, all_det_scores, all_gt_boxes, iou_thresh)
            aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def _compute_category_ap(all_dets, iou_thresh):
    """APc: all containers are one class."""
    det_boxes_all, det_scores_all, gt_boxes_all = [], [], []
    for det, gts in all_dets:
        gt_all = torch.tensor(
            [[g["x1"], g["y1"], g["x2"], g["y2"]] for g in gts],
            dtype=torch.float32
        ) if gts else torch.zeros(0, 4)

        if det is not None:
            det_boxes_all.append(det["boxes"].cpu())
            det_scores_all.append(det["scores"].cpu())
        else:
            det_boxes_all.append(torch.zeros(0, 4))
            det_scores_all.append(torch.zeros(0))
        gt_boxes_all.append(gt_all)

    all_det = torch.cat(det_boxes_all) if any(b.numel() > 0 for b in det_boxes_all) else torch.zeros(0, 4)
    all_scores = torch.cat(det_scores_all) if any(s.numel() > 0 for s in det_scores_all) else torch.zeros(0)
    all_gt = torch.cat(gt_boxes_all) if any(g.numel() > 0 for g in gt_boxes_all) else torch.zeros(0, 4)

    return compute_ap(all_det, all_scores, all_gt, iou_thresh) if all_gt.numel() > 0 else 0.0


def _compute_content_ap(all_dets, iou_thresh):
    """APt: treat liquid content state as the class (ignoring container category)."""
    # Since there's only 1 container category, APt is the same as APct
    return _compute_combined_ap(all_dets, iou_thresh)


# ─── Training ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = os.path.join(args.data_root, "datasets", "LCDTC")
    train_set = LCDTCDataset(data_dir, "train", args.img_size)
    val_set = LCDTCDataset(data_dir, "val", args.img_size, augment=False)

    train_loader = get_loader(train_set, args.batch, shuffle=True, num_workers=args.workers)
    val_loader = get_loader(val_set, args.batch, shuffle=False, num_workers=args.workers)

    model = LCDYOLOXModel(
        num_cls=args.num_cls, num_content=args.num_content,
        use_ta=not args.no_triplet,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    grids = build_grids(args.img_size)

    best_ap = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0
        n_batch = 0

        for imgs, tgts, sizes in train_loader:
            imgs = imgs.to(device)
            target_list = [(tgts[b], sizes[b]) for b in range(len(tgts))]

            preds_raw = model(imgs)
            loss, loss_dict = compute_loss(
                preds_raw, target_list, grids,
                num_cls=args.num_cls, num_content=args.num_content,
                lambda_content=args.lambda_content,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()

            loss_val = loss.item()
            if math.isfinite(loss_val):
                total_loss += loss_val
                n_batch += 1

        scheduler.step()
        dt = time.time() - t0
        avg_loss = total_loss / max(n_batch, 1)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, grids, args.num_cls, args.num_content, args.img_size)
            ap = metrics.get("mAPct", 0)
            lr = scheduler.get_last_lr()[0]
            print(f"[{epoch+1}/{args.epochs}] loss={avg_loss:.4f} lr={lr:.6f} time={dt:.1f}s")

            if ap > best_ap:
                best_ap = ap
                torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": metrics},
                           os.path.join(args.output_dir, "best.pth"))
        else:
            lr = scheduler.get_last_lr()[0]
            print(f"[{epoch+1}/{args.epochs}] loss={avg_loss:.4f} n_pos={loss_dict.get('n_pos',0):.0f} "
                  f"obj={loss_dict.get('obj',0):.3f} cls={loss_dict.get('cls',0):.3f} "
                  f"reg={loss_dict.get('reg',0):.3f} cont={loss_dict.get('content',0):.3f} "
                  f"lr={lr:.6f} time={dt:.1f}s")

    print(f"\nDone! Best mAPct={best_ap:.3f}")
    print(f"Paper target: mAPct=0.548 (LCD-YOLOX)")


def main():
    parser = argparse.ArgumentParser("LCD-YOLOX Training")
    parser.add_argument("--data-root", type=str, default="~/math_model_2026/problems/A")
    parser.add_argument("--output-dir", type=str, default="./output_lcdtc")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-cls", type=int, default=1)
    parser.add_argument("--num-content", type=int, default=5)
    parser.add_argument("--lambda-content", type=float, default=0.8)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--no-triplet", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
