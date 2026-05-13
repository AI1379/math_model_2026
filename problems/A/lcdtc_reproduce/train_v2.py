"""LCD-YOLOX v2 Training Script with pretrained ResNet50 + augmentation.

Key improvements:
  - ResNet50 pretrained backbone (ImageNet)
  - Mosaic + flip + color augmentation
  - Linear warmup + cosine schedule
  - GIoU regression loss
  - EMA for stable evaluation

Paper targets (LCD-YOLOX, CrossFormer-S):
  mAPct=0.548, APc@0.5=0.809, APt@0.5=0.607
"""

import os
import sys
import argparse
import time
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model_v2 import LCDYOLOXModelV2, ModelEMA
from .dataset import LCDTCDataset


# ─── Grid generation ───────────────────────────────────────────────────────────

def build_grids(input_size=640, strides=(8, 16, 32)):
    grids = []
    for s in strides:
        gs = input_size // s
        yy, xx = torch.meshgrid(torch.arange(gs, dtype=torch.float32),
                                torch.arange(gs, dtype=torch.float32),
                                indexing="ij")
        grid_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        grids.append((grid_xy, s, gs))
    return grids


# ─── SimOTA Label Assignment ───────────────────────────────────────────────────

def get_predictions(preds_raw, grids, num_cls, num_content):
    B = preds_raw[0].shape[0]
    device = preds_raw[0].device
    all_grid_xy, all_stride, all_dx, all_dy, all_dw, all_dh = [], [], [], [], [], []
    all_obj, all_cls, all_content = [], [], []

    for i, (grid_xy, stride, gs) in enumerate(grids):
        grid_dev = grid_xy.to(device)
        raw = preds_raw[i]
        n = gs * gs

        dx = raw[:, 0].view(B, n)
        dy = raw[:, 1].view(B, n)
        dw = raw[:, 2].view(B, n)
        dh = raw[:, 3].view(B, n)
        obj = raw[:, 4].view(B, n)
        cls = raw[:, 5:5+num_cls].view(B, num_cls, n).permute(0, 2, 1)
        content = raw[:, 5+num_cls:5+num_cls+num_content].view(B, num_content, n).permute(0, 2, 1)

        all_grid_xy.append(grid_dev.unsqueeze(0).expand(B, -1, -1))
        all_stride.append(torch.full((n,), stride, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1))
        all_dx.append(dx); all_dy.append(dy); all_dw.append(dw); all_dh.append(dh)
        all_obj.append(obj); all_cls.append(cls); all_content.append(content)

    return {
        "grid_xy": torch.cat(all_grid_xy, 1),
        "stride": torch.cat(all_stride, 1),
        "dx": torch.cat(all_dx, 1), "dy": torch.cat(all_dy, 1),
        "dw": torch.cat(all_dw, 1), "dh": torch.cat(all_dh, 1),
        "obj": torch.cat(all_obj, 1),
        "cls": torch.cat(all_cls, 1),
        "content": torch.cat(all_content, 1),
    }


def simota_assign(preds, gt_bboxes, gt_cls, gt_content, num_cls, num_content, fg_topk=10):
    B, M = preds["obj"].shape
    device = preds["obj"].device
    img_size = 640

    cx = (preds["grid_xy"][:, :, 0] + preds["dx"]) * preds["stride"]
    cy = (preds["grid_xy"][:, :, 1] + preds["dy"]) * preds["stride"]
    w = preds["dw"].clamp(max=10).exp() * preds["stride"]
    h = preds["dh"].clamp(max=10).exp() * preds["stride"]

    obj_tgt = torch.zeros(B, M, device=device)
    cls_tgt = torch.zeros(B, M, num_cls, device=device)
    cont_tgt = torch.zeros(B, M, num_content, device=device)
    tgt_dx = torch.zeros(B, M, device=device)
    tgt_dy = torch.zeros(B, M, device=device)
    tgt_dw = torch.zeros(B, M, device=device)
    tgt_dh = torch.zeros(B, M, device=device)
    iou_weight = torch.zeros(B, M, device=device)

    for b in range(B):
        valid = gt_cls[b] >= 0
        if not valid.any():
            continue
        n_gt = valid.sum().item()

        g_cx = gt_bboxes[b, valid, 0] * img_size
        g_cy = gt_bboxes[b, valid, 1] * img_size
        g_w = gt_bboxes[b, valid, 2] * img_size
        g_h = gt_bboxes[b, valid, 3] * img_size
        g_x1 = g_cx - g_w / 2; g_y1 = g_cy - g_h / 2
        g_x2 = g_cx + g_w / 2; g_y2 = g_cy + g_h / 2
        g_cls = gt_cls[b, valid].long()
        g_cont = gt_content[b, valid].long()

        # Center prior
        is_in = torch.zeros(n_gt, M, device=device, dtype=torch.bool)
        for gi in range(n_gt):
            radius = 2.5
            gx_grid = g_cx[gi] / preds["stride"][b]
            gy_grid = g_cy[gi] / preds["stride"][b]
            dist_x = (preds["grid_xy"][b, :, 0] + 0.5 - gx_grid).abs()
            dist_y = (preds["grid_xy"][b, :, 1] + 0.5 - gy_grid).abs()
            is_in[gi] = (dist_x < radius) & (dist_y < radius)

        # IoU
        px1 = cx[b] - w[b] / 2; py1 = cy[b] - h[b] / 2
        px2 = cx[b] + w[b] / 2; py2 = cy[b] + h[b] / 2

        ix1 = torch.max(px1.unsqueeze(0), g_x1.unsqueeze(1))
        iy1 = torch.max(py1.unsqueeze(0), g_y1.unsqueeze(1))
        ix2 = torch.min(px2.unsqueeze(0), g_x2.unsqueeze(1))
        iy2 = torch.min(py2.unsqueeze(0), g_y2.unsqueeze(1))
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_p = w[b] * h[b]
        area_g = g_w.unsqueeze(1) * g_h.unsqueeze(1)
        union = area_p.unsqueeze(0) + area_g - inter
        iou = inter / (union + 1e-8)

        # Cost
        cls_onehot = F.one_hot(g_cls, num_cls).float()
        cls_cost = F.binary_cross_entropy_with_logits(
            preds["cls"][b].unsqueeze(0).expand(n_gt, -1, -1),
            cls_onehot.unsqueeze(1).expand(-1, M, -1),
            reduction="none",
        ).mean(-1)
        reg_cost = 3.0 * (1.0 - iou)
        cost = cls_cost + reg_cost
        cost[~is_in] = float("inf")

        # Dynamic k
        topk = min(fg_topk, M)
        iou_topk, _ = iou.sort(descending=True, dim=1)
        dynamic_k = iou_topk[:, :topk].sum(1).int().clamp(min=1)

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

            s = preds["stride"][b, pos_idx]
            gx = preds["grid_xy"][b, pos_idx, 0]
            gy = preds["grid_xy"][b, pos_idx, 1]
            tgt_dx[b, pos_idx] = g_cx[gi] / s - gx
            tgt_dy[b, pos_idx] = g_cy[gi] / s - gy
            tgt_dw[b, pos_idx] = (g_w[gi] / s).clamp(min=1e-4).log().clamp(min=-10, max=10)
            tgt_dh[b, pos_idx] = (g_h[gi] / s).clamp(min=1e-4).log().clamp(min=-10, max=10)
            iou_weight[b, pos_idx] = iou[gi, pos_idx].detach()

    return obj_tgt, cls_tgt, cont_tgt, tgt_dx, tgt_dy, tgt_dw, tgt_dh, iou_weight


# ─── GIoU Loss ──────────────────────────────────────────────────────────────────

def giou_loss(pred_boxes, tgt_boxes):
    """Compute GIoU loss between predicted and target boxes (all in pixel coords)."""
    px1, py1 = pred_boxes[:, 0], pred_boxes[:, 1]
    px2, py2 = pred_boxes[:, 2], pred_boxes[:, 3]
    tx1, ty1 = tgt_boxes[:, 0], tgt_boxes[:, 1]
    tx2, ty2 = tgt_boxes[:, 2], tgt_boxes[:, 3]

    # Intersection
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Areas
    area_p = (px2 - px1) * (py2 - py1)
    area_t = (tx2 - tx1) * (ty2 - ty1)
    union = area_p + area_t - inter

    # Enclosing box
    cx1 = torch.min(px1, tx1); cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2); cy2 = torch.max(py2, ty2)
    enc_area = (cx2 - cx1) * (cy2 - cy1)

    iou = inter / (union + 1e-7)
    giou = iou - (enc_area - union) / (enc_area + 1e-7)
    return 1.0 - giou


# ─── Loss ───────────────────────────────────────────────────────────────────────

def compute_loss(preds_raw, targets, grids, num_cls=1, num_content=5, lambda_content=0.8):
    device = preds_raw[0].device
    B = preds_raw[0].shape[0]

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

    preds = get_predictions(preds_raw, grids, num_cls, num_content)

    obj_tgt, cls_tgt, cont_tgt, tgt_dx, tgt_dy, tgt_dw, tgt_dh, iou_w = simota_assign(
        preds, gt_bboxes, gt_cls, gt_content, num_cls, num_content
    )

    pos = obj_tgt > 0
    n_pos = pos.sum().clamp(min=1).float()

    # 1) Objectness (BCE with focal weighting)
    loss_obj = F.binary_cross_entropy_with_logits(preds["obj"], obj_tgt, reduction="none")
    alpha = 0.25
    weight = torch.where(obj_tgt > 0, torch.tensor(1.0, device=device), torch.tensor(alpha, device=device))
    loss_obj = (loss_obj * weight).sum() / (n_pos + alpha * (~pos).sum().float().clamp(min=1))

    # 2) Classification (BCE, positives only)
    loss_cls = torch.tensor(0.0, device=device)
    if pos.any():
        loss_cls = F.binary_cross_entropy_with_logits(
            preds["cls"][pos], cls_tgt[pos], reduction="none"
        ).sum() / n_pos

    # 3) Regression (GIoU, positives only)
    loss_reg = torch.tensor(0.0, device=device)
    if pos.any():
        # Decode positive predictions to pixel boxes
        p_dx = preds["dx"][pos]
        p_dy = preds["dy"][pos]
        p_dw = preds["dw"][pos].clamp(max=10)
        p_dh = preds["dh"][pos].clamp(max=10)

        stride_p = preds["stride"][pos]
        gx_p = preds["grid_xy"][pos, 0]
        gy_p = preds["grid_xy"][pos, 1]

        p_cx = (gx_p + p_dx) * stride_p
        p_cy = (gy_p + p_dy) * stride_p
        p_w = p_dw.exp() * stride_p
        p_h = p_dh.exp() * stride_p

        pred_boxes = torch.stack([p_cx - p_w/2, p_cy - p_h/2, p_cx + p_w/2, p_cy + p_h/2], dim=1)

        # Decode GT targets to pixel boxes
        t_dx = tgt_dx[pos]; t_dy = tgt_dy[pos]
        t_dw = tgt_dw[pos]; t_dh = tgt_dh[pos]
        t_cx = (gx_p + t_dx) * stride_p
        t_cy = (gy_p + t_dy) * stride_p
        t_w = t_dw.exp() * stride_p
        t_h = t_dh.exp() * stride_p
        tgt_boxes = torch.stack([t_cx - t_w/2, t_cy - t_h/2, t_cx + t_w/2, t_cy + t_h/2], dim=1)

        loss_reg = giou_loss(pred_boxes, tgt_boxes).mean()

    # 4) Content (BCE, positives only)
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


# ─── Evaluation ─────────────────────────────────────────────────────────────────

def decode_predictions(preds_raw, grids, num_cls, num_content, img_size=640, conf_thresh=0.001):
    device = preds_raw[0].device
    B = preds_raw[0].shape[0]
    results = []

    for b in range(B):
        boxes, scores, cat_ids, cont_ids = [], [], [], []

        for i, (grid_xy, stride, gs) in enumerate(grids):
            g = grid_xy.to(device)
            raw = preds_raw[i][b].view(-1, gs * gs).permute(1, 0)

            dx = raw[:, 0]; dy = raw[:, 1]
            dw = raw[:, 2]; dh = raw[:, 3]
            obj = raw[:, 4].sigmoid()
            cls_score = raw[:, 5:5+num_cls].sigmoid().max(1)[0]
            content_score = raw[:, 5+num_cls:5+num_cls+num_content].sigmoid()
            cont_label = content_score.max(1)[1]

            cx = (g[:, 0] + dx) * stride
            cy = (g[:, 1] + dy) * stride
            w = dw.clamp(max=10).exp() * stride
            h = dh.clamp(max=10).exp() * stride

            cx = cx.clamp(0, img_size); cy = cy.clamp(0, img_size)
            w = w.clamp(1, img_size); h = h.clamp(1, img_size)

            score = obj * cls_score
            mask = score > conf_thresh

            if mask.any():
                boxes.append(torch.stack([cx[mask]-w[mask]/2, cy[mask]-h[mask]/2,
                                          cx[mask]+w[mask]/2, cy[mask]+h[mask]/2], dim=1))
                scores.append(score[mask])
                cat_ids.append(torch.zeros(mask.sum(), dtype=torch.long, device=device))
                cont_ids.append(cont_label[mask])

        if boxes:
            all_boxes = torch.cat(boxes)
            all_scores = torch.cat(scores)
            # NMS per content class
            keep_list = []
            for cid in range(num_content):
                cids = torch.cat(cont_ids)
                cmask = cids == cid
                if cmask.any():
                    cboxes = all_boxes[cmask]
                    cscores = all_scores[cmask]
                    keep = torchvision_nms(cboxes, cscores, 0.65)
                    idx_orig = cmask.nonzero(as_tuple=True)[0][keep]
                    keep_list.append(idx_orig)
            if keep_list:
                keep_idx = torch.cat(keep_list)
                results.append({
                    "boxes": all_boxes[keep_idx],
                    "scores": all_scores[keep_idx],
                    "cat_ids": torch.cat(cat_ids)[keep_idx] if cat_ids else torch.tensor([]),
                    "cont_ids": torch.cat(cont_ids)[keep_idx] if cont_ids else torch.tensor([]),
                })
            else:
                results.append(None)
        else:
            results.append(None)
    return results


def torchvision_nms(boxes, scores, iou_thresh):
    """NMS using torchvision."""
    from torchvision.ops import nms
    return nms(boxes, scores, iou_thresh)


def compute_ap(det_boxes, det_scores, gt_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return 0.0
    if len(det_boxes) == 0:
        return 0.0

    order = det_scores.argsort(descending=True)
    det_boxes = det_boxes[order]
    det_scores = det_scores[order]

    n_det = len(det_boxes)
    n_gt = len(gt_boxes)
    tp = torch.zeros(n_det)
    fp = torch.zeros(n_det)
    matched = set()

    for i in range(n_det):
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
            tp[i] = 1; matched.add(best_j)
        else:
            fp[i] = 1

    tp_cum = torch.cumsum(tp, 0)
    fp_cum = torch.cumsum(fp, 0)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()
    return ap


def evaluate(model, val_loader, grids, num_cls=1, num_content=5, img_size=640):
    model.eval()
    device = next(model.parameters()).device
    grids_dev = [(grid_xy.to(device), stride, gs) for grid_xy, stride, gs in grids]

    all_dets = []

    with torch.no_grad():
        for imgs, tgts, sizes in val_loader:
            imgs = imgs.to(device)
            preds_raw = model(imgs)
            dets = decode_predictions(preds_raw, grids_dev, num_cls, num_content, img_size)

            for b in range(len(dets)):
                orig_h, orig_w = sizes[b, 0].item(), sizes[b, 1].item()
                sx, sy = orig_w / img_size, orig_h / img_size

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

                det = dets[b]
                if det is not None:
                    det["boxes"][:, 0] *= sx
                    det["boxes"][:, 1] *= sy
                    det["boxes"][:, 2] *= sx
                    det["boxes"][:, 3] *= sy

                all_dets.append((det, gt_list))

    apct_05 = _compute_combined_ap(all_dets, 0.5)
    apc_05 = _compute_category_ap(all_dets, 0.5)
    apt_05 = _compute_content_ap(all_dets, 0.5)

    apcts = [_compute_combined_ap(all_dets, 0.5 + 0.05 * i) for i in range(10)]
    mapct = sum(apcts) / len(apcts)

    print(f"  APc@0.5={apc_05:.3f}  APt@0.5={apt_05:.3f}  APct@0.5={apct_05:.3f}  mAPct={mapct:.3f}")
    return {"APc@0.5": apc_05, "APt@0.5": apt_05, "APct@0.5": apct_05, "mAPct": mapct}


def _compute_combined_ap(all_dets, iou_thresh):
    aps = []
    for content_id in range(5):
        det_boxes_list, det_scores_list, gt_boxes_list = [], [], []
        for det, gts in all_dets:
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

        all_det = torch.cat(det_boxes_list) if any(b.numel() > 0 for b in det_boxes_list) else torch.zeros(0, 4)
        all_scores = torch.cat(det_scores_list) if any(s.numel() > 0 for s in det_scores_list) else torch.zeros(0)
        all_gt = torch.cat(gt_boxes_list) if any(g.numel() > 0 for g in gt_boxes_list) else torch.zeros(0, 4)
        aps.append(compute_ap(all_det, all_scores, all_gt, iou_thresh))
    return sum(aps) / len(aps) if aps else 0.0


def _compute_category_ap(all_dets, iou_thresh):
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
    return _compute_combined_ap(all_dets, iou_thresh)


# ─── Augmented Dataset ─────────────────────────────────────────────────────────

import cv2
import numpy as np

class AugmentedLCDTCDataset(LCDTCDataset):
    """LCDTCDataset with mosaic + flip + color augmentation."""

    def __init__(self, data_dir, split="train", img_size=640):
        super().__init__(data_dir, split, img_size, augment=False)
        self.do_augment = (split == "train")

    def __getitem__(self, idx):
        if self.do_augment and random.random() < 0.5:
            return self._load_mosaic(idx)
        return self._load_single(idx)

    def _load_single(self, idx, target_size=None):
        if target_size is None:
            target_size = self.img_size
        img_id = self.img_ids[idx]
        img = cv2.imread(self.imgs[img_id])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        targets = self.anns.get(img_id, [])
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 7))

        if targets.numel() > 0:
            targets[:, 2] /= orig_w
            targets[:, 3] /= orig_h
            targets[:, 4] /= orig_w
            targets[:, 5] /= orig_h

        img = cv2.resize(img, (target_size, target_size))

        if self.do_augment:
            img, targets = self._augment(img, targets)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, targets, torch.tensor([orig_h, orig_w])

    def _load_mosaic(self, idx):
        """Mosaic augmentation: combine 4 random images."""
        s = self.img_size
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]

        # Mosaic canvas: 2s x 2s
        yc, xc = (random.randint(s // 2, s * 3 // 2), random.randint(s // 2, s * 3 // 2))
        mosaic_img = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
        all_targets = []

        for i, sub_idx in enumerate(indices):
            sub_img_id = self.img_ids[sub_idx]
            img = cv2.imread(self.imgs[sub_img_id])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Place in quadrant
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, s * 2 - xc), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, s * 2 - yc)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, s * 2 - xc), min(h, s * 2 - yc)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Adjust targets
            sub_targets = self.anns.get(sub_img_id, [])
            if sub_targets:
                st = torch.tensor(sub_targets, dtype=torch.float32)
                # Convert cx,cy,w,h from pixel to normalized
                st[:, 2] /= w; st[:, 3] /= h; st[:, 4] /= w; st[:, 5] /= h

                # Scale to mosaic coords
                pad_w = x2a - x1a; pad_h = y2a - y1a
                scale_w = pad_w / w; scale_h = pad_h / h

                st[:, 2] = st[:, 2] * scale_w * w + (x1a / (s * 2))  # normalize cx
                st[:, 3] = st[:, 3] * scale_h * h + (y1a / (s * 2))  # normalize cy
                st[:, 4] = st[:, 4] * scale_w  # normalize w
                st[:, 5] = st[:, 5] * scale_h  # normalize h

                # Clip to valid region
                cx = st[:, 2]; cy = st[:, 3]
                bw = st[:, 4]; bh = st[:, 5]
                valid = (cx - bw/2 < 1) & (cy - bh/2 < 1) & (cx + bw/2 > 0) & (cy + bh/2 > 0)
                all_targets.append(st[valid])

        # Crop/resize to target size
        # Random crop from mosaic
        x1 = random.randint(0, max(s, 1))
        y1 = random.randint(0, max(s, 1))
        x2, y2 = x1 + s, y1 + s
        mosaic_img = mosaic_img[y1:y2, x1:x2]

        # Adjust targets
        if all_targets:
            targets = torch.cat(all_targets)
            # Shift and scale
            targets[:, 2] = (targets[:, 2] * s * 2 - x1) / s
            targets[:, 3] = (targets[:, 3] * s * 2 - y1) / s
            targets[:, 4] = targets[:, 4] * 2  # because mosaic is 2s
            targets[:, 5] = targets[:, 5] * 2
            # Clip
            valid = (targets[:, 2] > 0) & (targets[:, 3] > 0) & (targets[:, 2] < 1) & (targets[:, 3] < 1)
            targets = targets[valid]
        else:
            targets = torch.zeros((0, 7))

        orig_h, orig_w = s, s

        if self.do_augment:
            mosaic_img, targets = self._augment(mosaic_img, targets)

        img = torch.from_numpy(mosaic_img).permute(2, 0, 1).float() / 255.0
        return img, targets, torch.tensor([640.0, 640.0])  # fake orig size for mosaic

    def _augment(self, img, targets):
        """Random flip + color jitter."""
        # Horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            if targets.numel() > 0:
                targets[:, 2] = 1.0 - targets[:, 2]  # flip cx

        # Color jitter (HSV)
        if random.random() < 0.3:
            img = img.astype(np.float32)
            # Brightness
            img *= random.uniform(0.7, 1.3)
            # Contrast
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img, targets


def lcdtc_collate(batch):
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


# ─── Training ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = os.path.join(args.data_root, "datasets", "LCDTC")
    train_set = AugmentedLCDTCDataset(data_dir, "train", args.img_size)
    val_set = LCDTCDataset(data_dir, "val", args.img_size, augment=False)

    train_loader = get_loader(train_set, args.batch, shuffle=True, num_workers=args.workers)
    val_loader = get_loader(val_set, args.batch, shuffle=False, num_workers=args.workers)

    model = LCDYOLOXModelV2(
        num_cls=args.num_cls, num_content=args.num_content,
        use_ta=not args.no_triplet, pretrained=True, fpn_dim=256,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M (trainable: {n_trainable/1e6:.2f}M)")

    # Optimizer: only train non-frozen params
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=5e-4
    )

    # Warmup + Cosine
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = ModelEMA(model, decay=0.9998)

    grids = build_grids(args.img_size)
    best_ap = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        # Unfreeze backbone BN after warmup for better adaptation
        # (keep them frozen for stability with small batch size)

        t0 = time.time()
        total_loss = 0
        n_batch = 0
        loss_sum = {"obj": 0, "cls": 0, "reg": 0, "content": 0}

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
            ema.update(model)

            loss_val = loss.item()
            if math.isfinite(loss_val):
                total_loss += loss_val
                n_batch += 1
                for k in loss_sum:
                    loss_sum[k] += loss_dict.get(k, 0)

        scheduler.step()
        dt = time.time() - t0
        avg_loss = total_loss / max(n_batch, 1)

        if (epoch + 1) % args.print_interval == 0:
            lr = scheduler.get_last_lr()[0]
            avg_detail = {k: f"{v/max(n_batch,1):.3f}" for k, v in loss_sum.items()}
            print(f"[{epoch+1}/{args.epochs}] loss={avg_loss:.4f} {avg_detail} lr={lr:.6f} time={dt:.1f}s")

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            ema.apply(model)
            metrics = evaluate(model, val_loader, grids, args.num_cls, args.num_content, args.img_size)
            ema.restore(model)

            ap = metrics.get("mAPct", 0)
            lr = scheduler.get_last_lr()[0]
            print(f"[{epoch+1}/{args.epochs}] loss={avg_loss:.4f} lr={lr:.6f} time={dt:.1f}s")

            if ap > best_ap:
                best_ap = ap
                torch.save({
                    "epoch": epoch, "model": model.state_dict(), "ema": ema.ema,
                    "best_mAPct": best_ap, "metrics": metrics,
                }, os.path.join(args.output_dir, "best.pth"))
                print(f"  -> Saved best model (mAPct={best_ap:.3f})")

    print(f"\nDone! Best mAPct={best_ap:.3f}")
    print(f"Paper target: mAPct=0.548 (LCD-YOLOX, CrossFormer-S)")
    print(f"Note: We use ResNet50 backbone instead of CrossFormer-S, so results may differ.")


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running as script too
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lcdtc_reproduce.train_v2 import train as _train

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--num-cls", type=int, default=1)
    p.add_argument("--num-content", type=int, default=5)
    p.add_argument("--lambda-content", type=float, default=0.8)
    p.add_argument("--no-triplet", action="store_true")
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--print-interval", type=int, default=5)
    p.add_argument("--output-dir", default="./output_lcdtc_v2")
    train(p.parse_args())
