"""Model builders for the LCDTC-only full system."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights


def _safe_construct(builder, *args, **kwargs):
    try:
        return builder(*args, **kwargs)
    except Exception as exc:
        print(f"[lcdtc_system] pretrained weights unavailable, fallback to random init: {exc}")
        fallback = copy.deepcopy(kwargs)
        for key in list(fallback.keys()):
            if "weight" in key:
                fallback[key] = None
        return builder(*args, **fallback)


def build_detector(
    name: str = "fasterrcnn_mobilenet_v3_large_fpn",
    pretrained: bool = True,
    num_classes: int = 2,
) -> nn.Module:
    if name == "fasterrcnn_mobilenet_v3_large_fpn":
        weights = (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        )
        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = _safe_construct(
            fasterrcnn_mobilenet_v3_large_fpn,
            weights=weights,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=3,
        )
    elif name == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
        model = _safe_construct(
            fasterrcnn_resnet50_fpn,
            weights=weights,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=3,
        )
    else:
        raise ValueError(f"Unsupported detector: {name}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class LCDTCStateClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, num_states: int = 5):
        super().__init__()
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = _safe_construct(resnet18, weights=weights)
            feat_dim = 512
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = _safe_construct(resnet34, weights=weights)
            feat_dim = 512
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = _safe_construct(resnet50, weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported classifier backbone: {backbone}")

        self.features = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(0.2)
        self.state_head = nn.Linear(feat_dim, num_states)
        self.binary_head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor):
        feat = self.features(x).flatten(1)
        feat = self.dropout(feat)
        return {
            "state_logits": self.state_head(feat),
            "binary_logits": self.binary_head(feat).squeeze(1),
        }


def build_state_classifier(
    backbone: str = "resnet18",
    pretrained: bool = True,
    num_states: int = 5,
) -> nn.Module:
    return LCDTCStateClassifier(backbone=backbone, pretrained=pretrained, num_states=num_states)
