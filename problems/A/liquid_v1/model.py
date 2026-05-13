"""Shared-backbone multitask model for bottle liquid recognition."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True) -> None:
        super().__init__()
        if backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = _build_resnet_with_fallback(resnet34, weights)
            out_channels = [64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = _build_resnet_with_fallback(resnet50, weights)
            out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict(
            {
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "c5": c5,
            }
        )


def _build_resnet_with_fallback(builder, weights):
    if weights is None:
        return builder(weights=None)
    try:
        return builder(weights=weights)
    except Exception as exc:
        print(f"[BottleLiquidNet] pretrained weights unavailable, fallback to random init: {exc}")
        return builder(weights=None)


class SegDecoder(nn.Module):
    def __init__(self, in_channels, out_channels: int = 192) -> None:
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channels, out_channels)
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels * 4, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = self.fpn(feats)
        target_size = p["c2"].shape[-2:]
        fused = torch.cat(
            [
                p["c2"],
                F.interpolate(p["c3"], size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(p["c4"], size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(p["c5"], size=target_size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return self.fuse(fused)


class BottleLiquidNet(nn.Module):
    """Shared encoder with segmentation heads and state classifiers."""

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        decoder_dim: int = 192,
        num_states: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)
        self.decoder = SegDecoder(self.encoder.out_channels, decoder_dim)

        self.bottle_head = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )
        self.liquid_head = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

        cls_dim = decoder_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, decoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(decoder_dim, 1)
        self.state_head = nn.Linear(decoder_dim, num_states)

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(mask_logits)
        weight = mask / mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (feat * weight).sum(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        dec = self.decoder(feats)

        bottle_logits = self.bottle_head(dec)
        liquid_logits = self.liquid_head(dec)
        bottle_logits_up = F.interpolate(
            bottle_logits, size=input_size, mode="bilinear", align_corners=False
        )
        liquid_logits_up = F.interpolate(
            liquid_logits, size=input_size, mode="bilinear", align_corners=False
        )

        global_feat = F.adaptive_avg_pool2d(dec, 1).flatten(1)
        bottle_feat = self._masked_pool(dec, bottle_logits)
        liquid_feat = self._masked_pool(dec, liquid_logits)
        cls_feat = self.classifier(torch.cat([global_feat, bottle_feat, liquid_feat], dim=1))

        return {
            "bottle_logits": bottle_logits_up,
            "liquid_logits": liquid_logits_up,
            "binary_logits": self.binary_head(cls_feat).squeeze(1),
            "state_logits": self.state_head(cls_feat),
        }
