"""Liquid v4 model: v3 + ordinal regression + segmentation gating + ratio prediction.

Key improvements over v3:
  - Ordinal state classifier (K-1 binary logits with rank-consistency constraint).
  - Bottle→Liquid spatial cross-gate (bottle_mask modulates liquid features).
  - Fill-ratio regression head for continuous self-consistency signal.
  - Segmentation ratio injected as extra classifier input.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork


# ═══════════════════════════════════════════════════════════════════════════════════
# Basic blocks
# ═══════════════════════════════════════════════════════════════════════════════════

class ConvNormAct(nn.Module):
    """Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ═══════════════════════════════════════════════════════════════════════════════════
# Triplet Attention (LCDTC paper, Section 4.3)
# ═══════════════════════════════════════════════════════════════════════════════════

class TripletAttention(nn.Module):
    """Cross-dimensional triplet attention."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        p = kernel_size // 2
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=p, bias=False),
            nn.BatchNorm2d(1),
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=p, bias=False),
            nn.BatchNorm2d(1),
        )
        self.conv_b3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=p, bias=False),
            nn.BatchNorm2d(1),
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _zpool(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w = x.permute(0, 3, 2, 1)
        a1 = self.sigmoid(self.conv_b1(self._zpool(x_w)))
        x1 = (a1 * x_w).permute(0, 3, 2, 1)

        x_h = x.permute(0, 2, 3, 1)
        a2 = self.sigmoid(self.conv_b2(self._zpool(x_h)))
        x2 = (a2 * x_h).permute(0, 3, 1, 2)

        a3 = self.sigmoid(self.conv_b3(self._zpool(x)))
        x3 = a3 * x

        return (x1 + x2 + x3) / 3.0


# ═══════════════════════════════════════════════════════════════════════════════════
# ResNet Encoder
# ═══════════════════════════════════════════════════════════════════════════════════

def _build_resnet_with_fallback(builder, weights):
    if weights is None:
        return builder(weights=None)
    try:
        return builder(weights=weights)
    except Exception as exc:
        print(
            f"[LiquidV4Net] pretrained weights unavailable, "
            f"fallback to random init: {exc}"
        )
        return builder(weights=None)


class ResNetEncoder(nn.Module):
    """ResNet34/50 backbone outputting C2-C5 feature maps."""

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

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
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
        return OrderedDict({"c2": c2, "c3": c3, "c4": c4, "c5": c5})


# ═══════════════════════════════════════════════════════════════════════════════════
# Transformer Encoder
# ═══════════════════════════════════════════════════════════════════════════════════

class LearnablePosEmbed2D(nn.Module):
    """Learnable 2D positional encoding for image feature patches."""

    def __init__(self, dim: int, max_h: int = 128, max_w: int = 128) -> None:
        super().__init__()
        self.row_embed = nn.Parameter(torch.empty(max_h, dim // 2))
        self.col_embed = nn.Parameter(torch.empty(max_w, dim // 2))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

    def forward(self, h: int, w: int) -> torch.Tensor:
        r = self.row_embed[:h].unsqueeze(1).expand(-1, w, -1)
        c = self.col_embed[:w].unsqueeze(0).expand(h, -1, -1)
        return torch.cat([r, c], dim=-1).flatten(0, 1)


class TransformerEncoderBlock(nn.Module):
    """Lightweight transformer encoder over image feature patches."""

    def __init__(
        self,
        in_dim: int = 192,
        enc_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = ConvNormAct(in_dim, enc_dim, k=3, s=2)
        self.pos_enc = LearnablePosEmbed2D(enc_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=num_heads,
            dim_feedforward=int(enc_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(enc_dim, in_dim, k=3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        x_down = self.input_proj(x)
        _, _, Hd, Wd = x_down.shape

        tokens = x_down.flatten(2).transpose(1, 2)
        pos = self.pos_enc(Hd, Wd).unsqueeze(0).to(tokens.device)
        tokens = self.encoder(tokens + pos)

        feat_2d = tokens.transpose(1, 2).reshape(B, -1, Hd, Wd)
        feat_2d = self.output_proj(feat_2d)
        return feat_2d, tokens


# ═══════════════════════════════════════════════════════════════════════════════════
# Ordinal Cross-Attention Classifier
# ═══════════════════════════════════════════════════════════════════════════════════

class OrdinalCrossAttentionClassifier(nn.Module):
    """Classification via learnable state queries + ordinal regression head.

    Compared to v2/v3 CrossAttentionClassifier:
      - state_head outputs K-1 ordinal logits (P(y > k)) instead of K independent classes.
        Predicted class = count(sigmoid(logits) > 0.5), yielding 0..K-1.
      - ratio_head predicts continuous fill_ratio for self-consistency training.
      - seg_ratio (from bottle/liquid masks) injected as auxiliary input.
    """

    def __init__(
        self,
        feat_dim: int = 192,
        enc_dim: int = 256,
        num_states: int = 5,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_ordinal = num_states - 1  # K-1 binary classifiers

        # Learnable state prototype queries (one per class, as before)
        self.state_queries = nn.Parameter(torch.empty(num_states, enc_dim))
        nn.init.trunc_normal_(self.state_queries, std=0.02)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            enc_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(enc_dim)

        # Cross-attention: queries attend to image tokens
        self.cross_attn = nn.MultiheadAttention(
            enc_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(enc_dim)

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, enc_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim * 2, enc_dim),
        )
        self.ffn_norm = nn.LayerNorm(enc_dim)

        # Shared feature projector
        # Input: state_feat(enc_dim) + global(feat_dim) + bottle(feat_dim)
        #        + liquid(feat_dim) + seg_ratio(1)
        cls_in = enc_dim + feat_dim * 3 + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, feat_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(feat_dim, 1)
        # K-1 ordinal logits: P(y > 0), P(y > 1), ..., P(y > K-2)
        self.state_head = nn.Linear(feat_dim, self.num_ordinal)
        # Fill-ratio regression (auxiliary, for consistency with segmentation)
        self.ratio_head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        global_feat: torch.Tensor,
        bottle_feat: torch.Tensor,
        liquid_feat: torch.Tensor,
        seg_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, enc_dim) patch tokens from transformer encoder.
            global_feat: (B, feat_dim) global average pool.
            bottle_feat: (B, feat_dim) mask-weighted bottle features.
            liquid_feat: (B, feat_dim) mask-weighted liquid features.
            seg_ratio: (B, 1) fill_ratio from segmentation masks.
        Returns:
            binary_logits: (B,),  state_logits: (B, K-1),  ratio_pred: (B, 1).
        """
        B = tokens.shape[0]

        q = self.state_queries.unsqueeze(0).expand(B, -1, -1)
        q = self.self_norm(q + self.self_attn(q, q, q)[0])
        q = self.cross_norm(q + self.cross_attn(q, tokens, tokens)[0])
        attended = self.ffn_norm(q + self.ffn(q))  # (B, num_states, enc_dim)

        state_feat = attended.mean(dim=1)

        cls_feat = self.classifier(
            torch.cat(
                [state_feat, global_feat, bottle_feat, liquid_feat, seg_ratio], dim=-1
            )
        )
        binary_logits = self.binary_head(cls_feat).squeeze(-1)
        state_logits = self.state_head(cls_feat)   # (B, K-1) ordinal
        ratio_pred = self.ratio_head(cls_feat)     # (B, 1)
        return binary_logits, state_logits, ratio_pred


# ═══════════════════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════════════════

class LiquidV4Net(nn.Module):
    """CNN + Transformer hybrid for bottle liquid recognition (v4).

    Improvements over v3:
      - Bottle→Liquid spatial cross-gate: bottle probability spatially modulates
        liquid features, enforcing the physical prior that liquid ⊂ bottle.
      - OrdinalCrossAttentionClassifier: K-1 ordinal logits replace K-class softmax.
      - Fill-ratio regression for self-consistency between seg and cls.
      - seg_ratio (from masks) injected as an explicit classifier input.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        decoder_dim: int = 192,
        num_states: int = 5,
        dropout: float = 0.2,
        enc_dim: int = 256,
        enc_heads: int = 8,
        enc_layers: int = 2,
        cls_heads: int = 4,
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.num_states = num_states

        # ---- CNN backbone ----
        self.encoder = ResNetEncoder(backbone, pretrained)

        # ---- FPN ----
        self.fpn = FeaturePyramidNetwork(
            list(self.encoder.out_channels), decoder_dim
        )

        self.fpn_fuse = nn.Sequential(
            ConvNormAct(decoder_dim * 4, decoder_dim),
            ConvNormAct(decoder_dim, decoder_dim),
        )

        # ---- Triplet Attention ----
        self.triplet_attn = TripletAttention(kernel_size=7)

        # ---- Transformer Encoder ----
        self.transformer_enc = TransformerEncoderBlock(
            in_dim=decoder_dim,
            enc_dim=enc_dim,
            num_heads=enc_heads,
            num_layers=enc_layers,
            dropout=dropout,
        )

        # ---- Segmentation heads (with Bottle→Liquid cross-gate) ----
        self.bottle_feat_conv = ConvNormAct(decoder_dim, decoder_dim)
        self.bottle_out = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.liquid_feat_conv = ConvNormAct(decoder_dim, decoder_dim)
        # Bottle probability → gate map that modulates liquid features spatially
        self.bottle_to_liquid_gate = nn.Sequential(
            nn.Conv2d(1, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.Sigmoid(),
        )
        self.liquid_out = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # ---- Ordinal Classification head ----
        self.cls_head = OrdinalCrossAttentionClassifier(
            feat_dim=decoder_dim,
            enc_dim=enc_dim,
            num_states=num_states,
            num_heads=cls_heads,
            dropout=dropout,
        )

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(mask_logits)
        weight = mask / mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (feat * weight).sum(dim=(2, 3))

    @torch.no_grad()
    def _compute_seg_ratio(
        self, bottle_logits: torch.Tensor, liquid_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute fill_ratio from segmentation masks for classifier input."""
        b = torch.sigmoid(bottle_logits).flatten(1)
        l = torch.sigmoid(liquid_logits).flatten(1)
        # Clamp liquid to bottle region (hard constraint)
        l = l * (b > 0.5).float()
        ratio = l.sum(dim=1) / b.sum(dim=1).clamp(min=1e-6)
        return ratio.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]

        # ---- CNN + FPN ----
        feats = self.encoder(x)
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
        fused = self.fpn_fuse(fused)

        # ---- Triplet Attention ----
        fused = self.triplet_attn(fused)

        # ---- Transformer Encoder ----
        encoded_2d, tokens = self.transformer_enc(fused)

        # ---- Segmentation with Bottle→Liquid cross-gate ----
        # Bottle branch
        bottle_feat = self.bottle_feat_conv(encoded_2d)
        bottle_logits = self.bottle_out(bottle_feat)

        # Liquid branch gated by bottle probability
        bottle_prob = torch.sigmoid(bottle_logits.detach())  # (B, 1, H, W), no grad
        gate = self.bottle_to_liquid_gate(bottle_prob)       # (B, dec_dim, H, W)
        liquid_feat = self.liquid_feat_conv(encoded_2d)
        liquid_feat = liquid_feat * gate                     # spatial gate
        liquid_logits = self.liquid_out(liquid_feat)

        bottle_up = F.interpolate(
            bottle_logits, size=input_size, mode="bilinear", align_corners=False
        )
        liquid_up = F.interpolate(
            liquid_logits, size=input_size, mode="bilinear", align_corners=False
        )

        # ---- Classification ----
        global_feat = F.adaptive_avg_pool2d(encoded_2d, 1).flatten(1)
        bottle_feat_pool = self._masked_pool(encoded_2d, bottle_logits)
        liquid_feat_pool = self._masked_pool(encoded_2d, liquid_logits)
        seg_ratio = self._compute_seg_ratio(bottle_logits, liquid_logits)

        binary_logits, state_logits, ratio_pred = self.cls_head(
            tokens, global_feat, bottle_feat_pool, liquid_feat_pool, seg_ratio
        )

        return {
            "bottle_logits": bottle_up,
            "liquid_logits": liquid_up,
            "binary_logits": binary_logits,
            "state_logits": state_logits,    # (B, K-1) ordinal logits
            "ratio_pred": ratio_pred,        # (B, 1)
        }
