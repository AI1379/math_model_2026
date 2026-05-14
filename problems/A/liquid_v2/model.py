"""Liquid v2 model: CNN + Transformer hybrid for bottle liquid recognition.

Key improvements over v1:
  - Triplet Attention (LCDTC paper) for cross-dimensional feature enhancement.
  - Transformer Encoder with learnable 2D positional encoding for global receptive field.
  - Learnable class prototype queries + cross-attention (Trans2Seg-inspired) replacing
    simple masked pooling for classification.
  - Multi-head self-attention among state prototypes before cross-attending to features.
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
    """Cross-dimensional triplet attention.

    Three parallel branches capture C-H, C-W, and H-W (spatial) interactions
    via tensor rotation and residual attention weighting.
    """

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
        # Branch 1: C-H interaction (rotate along W axis)
        x_w = x.permute(0, 3, 2, 1)
        a1 = self.sigmoid(self.conv_b1(self._zpool(x_w)))
        x1 = (a1 * x_w).permute(0, 3, 2, 1)

        # Branch 2: C-W interaction (rotate along H axis)
        x_h = x.permute(0, 2, 3, 1)
        a2 = self.sigmoid(self.conv_b2(self._zpool(x_h)))
        x2 = (a2 * x_h).permute(0, 3, 1, 2)

        # Branch 3: H-W spatial interaction
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
            f"[LiquidV2Net] pretrained weights unavailable, "
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
# Transformer Encoder (Trans2Seg-inspired global self-attention)
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
        r = self.row_embed[:h].unsqueeze(1).expand(-1, w, -1)   # (H, W, D/2)
        c = self.col_embed[:w].unsqueeze(0).expand(h, -1, -1)   # (H, W, D/2)
        return torch.cat([r, c], dim=-1).flatten(0, 1)           # (H*W, D)


class TransformerEncoderBlock(nn.Module):
    """Lightweight transformer encoder over image feature patches.

    Projects features to *enc_dim*, applies self-attention at reduced spatial
    resolution, then projects back.  Returns both the 2D feature map and the
    flat token sequence (the latter is used by the cross-attention classifier).
    """

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
        # Strided projection for efficiency (H/4 -> H/8)
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

        # Upsample back to original spatial size + project to in_dim
        self.output_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(enc_dim, in_dim, k=3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        x_down = self.input_proj(x)
        _, _, Hd, Wd = x_down.shape

        tokens = x_down.flatten(2).transpose(1, 2)                 # (B, Hd*Wd, enc_dim)
        pos = self.pos_enc(Hd, Wd).unsqueeze(0).to(tokens.device)  # (1, Hd*Wd, enc_dim)
        tokens = self.encoder(tokens + pos)                         # self-attention

        feat_2d = tokens.transpose(1, 2).reshape(B, -1, Hd, Wd)    # (B, enc_dim, Hd, Wd)
        feat_2d = self.output_proj(feat_2d)                         # (B, in_dim, H, W)
        return feat_2d, tokens


# ═══════════════════════════════════════════════════════════════════════════════════
# Cross-Attention Classifier (Trans2Seg-inspired learnable prototypes)
# ═══════════════════════════════════════════════════════════════════════════════════

class CrossAttentionClassifier(nn.Module):
    """Classification via learnable state queries + multi-head cross-attention.

    Inspired by Trans2Seg's learnable class prototypes: each of the *num_states*
    queries learns a prototypical feature pattern for one liquid level.  Queries
    first interact via self-attention, then cross-attend to image patch tokens.
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

        # Learnable state prototype queries
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

        # Final classifier
        cls_in = enc_dim + feat_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, feat_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(feat_dim, 1)
        self.state_head = nn.Linear(feat_dim, num_states)

    def forward(
        self,
        tokens: torch.Tensor,
        global_feat: torch.Tensor,
        bottle_feat: torch.Tensor,
        liquid_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, enc_dim) – patch tokens from transformer encoder.
            global_feat: (B, feat_dim).
            bottle_feat: (B, feat_dim).
            liquid_feat: (B, feat_dim).
        Returns:
            binary_logits: (B,),  state_logits: (B, num_states).
        """
        B = tokens.shape[0]

        # Expand queries for batch
        q = self.state_queries.unsqueeze(0).expand(B, -1, -1)

        # Self-attention among state prototypes
        q = self.self_norm(q + self.self_attn(q, q, q)[0])

        # Cross-attend to image patches
        q = self.cross_norm(q + self.cross_attn(q, tokens, tokens)[0])

        # Feed-forward
        attended = self.ffn_norm(q + self.ffn(q))  # (B, num_states, enc_dim)

        state_feat = attended.mean(dim=1)  # aggregate prototypes

        cls_feat = self.classifier(
            torch.cat([state_feat, global_feat, bottle_feat, liquid_feat], dim=-1)
        )
        return self.binary_head(cls_feat).squeeze(-1), self.state_head(cls_feat)


# ═══════════════════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════════════════

class LiquidV2Net(nn.Module):
    """CNN + Transformer hybrid for bottle liquid recognition.

    Pipeline
    --------
    ResNet Encoder -> FPN (multi-scale fusion) -> TripletAttention
        -> TransformerEncoder (global self-attention)
           ├── Segmentation heads (bottle mask, liquid mask)
           └── CrossAttentionClassifier (binary + 5-class state)
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

        # ---- CNN backbone ----
        self.encoder = ResNetEncoder(backbone, pretrained)

        # ---- FPN ----
        self.fpn = FeaturePyramidNetwork(
            list(self.encoder.out_channels), decoder_dim
        )

        # FPN fusion: concat all scales -> project
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

        # ---- Segmentation heads ----
        self.bottle_head = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )
        self.liquid_head = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

        # ---- Classification head ----
        self.cls_head = CrossAttentionClassifier(
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

        # ---- Segmentation ----
        bottle_logits = self.bottle_head(encoded_2d)
        liquid_logits = self.liquid_head(encoded_2d)
        bottle_up = F.interpolate(
            bottle_logits, size=input_size, mode="bilinear", align_corners=False
        )
        liquid_up = F.interpolate(
            liquid_logits, size=input_size, mode="bilinear", align_corners=False
        )

        # ---- Classification ----
        global_feat = F.adaptive_avg_pool2d(encoded_2d, 1).flatten(1)
        bottle_feat = self._masked_pool(encoded_2d, bottle_logits)
        liquid_feat = self._masked_pool(encoded_2d, liquid_logits)
        binary_logits, state_logits = self.cls_head(
            tokens, global_feat, bottle_feat, liquid_feat
        )

        return {
            "bottle_logits": bottle_up,
            "liquid_logits": liquid_up,
            "binary_logits": binary_logits,
            "state_logits": state_logits,
        }
