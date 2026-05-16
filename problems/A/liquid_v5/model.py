"""Liquid v5 model: ROI-first state classifier with ordinal auxiliary heads.

This version intentionally prioritises the strongest local signal from the
LCDTC experiments: classify cropped bottle ROIs directly, and keep segmentation
out of the critical path.  The model still respects the ordered nature of the
labels via a continuous fill-ratio head and an optional ordinal head, but the
main decision remains a standard 5-way softmax.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_v2_s,
    resnet18,
    resnet34,
    resnet50,
)


def _safe_build(builder, weights):
    if weights is None:
        return builder(weights=None)
    try:
        return builder(weights=weights)
    except Exception as exc:
        print(f"[LiquidV5Net] pretrained weights unavailable, fallback to random init: {exc}")
        return builder(weights=None)


class LiquidV5Net(nn.Module):
    """Bottle ROI classifier.

    Outputs:
      - state_logits: five-class liquid-level logits.
      - binary_logits: empty vs non-empty auxiliary output.
      - ratio_pred: continuous fill-ratio prediction in [0, 1].
      - ordinal_logits: K-1 logits for P(y > k), used only as auxiliary loss.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        num_states: int = 5,
        dropout: float = 0.25,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_states = num_states

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = _safe_build(resnet18, weights)
            feat_dim = base.fc.in_features
            self.features = nn.Sequential(*list(base.children())[:-1])
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = _safe_build(resnet34, weights)
            feat_dim = base.fc.in_features
            self.features = nn.Sequential(*list(base.children())[:-1])
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = _safe_build(resnet50, weights)
            feat_dim = base.fc.in_features
            self.features = nn.Sequential(*list(base.children())[:-1])
        elif backbone == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = _safe_build(convnext_tiny, weights)
            feat_dim = base.classifier[-1].in_features
            self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
        elif backbone == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base = _safe_build(efficientnet_v2_s, weights)
            feat_dim = base.classifier[-1].in_features
            self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.neck = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_head = nn.Linear(hidden_dim, num_states)
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.ratio_head = nn.Linear(hidden_dim, 1)
        self.ordinal_head = nn.Linear(hidden_dim, num_states - 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        emb = self.neck(feat)
        return {
            "state_logits": self.state_head(emb),
            "binary_logits": self.binary_head(emb).squeeze(-1),
            "ratio_pred": torch.sigmoid(self.ratio_head(emb)).squeeze(-1),
            "ordinal_logits": self.ordinal_head(emb),
        }

