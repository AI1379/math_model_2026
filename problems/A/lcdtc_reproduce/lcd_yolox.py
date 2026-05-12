"""Complete LCD-YOLOX model with Triplet Attention content head.

Based on YOLOX architecture adapted for the LCDTC paper:
  CSPDarknet backbone → PAFPN neck → (cls + reg + obj + content) heads

The content head predicts 5 liquid states: empty, little, half, much, full.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .triplet_attention import TripletAttention


# ─── Building blocks ───────────────────────────────────────────────────────────

class Conv(nn.Module):
    """Conv2d + BN + SiLU."""
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        h = int(c2 * e)
        self.cv1 = Conv(c1, h, 1, 1)
        self.cv2 = Conv(h, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        h = int(c2 * e)
        self.cv1 = Conv(c1, h, 1, 1)
        self.cv2 = Conv(c1, h, 1, 1)
        self.cv3 = Conv(2 * h, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(h, h, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        h = c1 // 2
        self.cv1 = Conv(c1, h, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(ki, 1, ki // 2) for ki in k])
        self.cv2 = Conv(h * 4, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [p(x) for p in self.pools], 1))


# ─── CSPDarknet ────────────────────────────────────────────────────────────────

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul=0.33, wid_mul=0.5):
        super().__init__()
        d = max(round(dep_mul * 3), 1)
        base = int(64 * wid_mul)  # 32 for YOLOX-S

        self.stem = Conv(3, base, 3, 2)                            # -> 32
        self.dark2 = nn.Sequential(
            Conv(base, base * 2, 3, 2),                            # 32 -> 64
            CSPBlock(base * 2, base * 2, d),
        )
        self.dark3 = nn.Sequential(
            Conv(base * 2, base * 4, 3, 2),                        # 64 -> 128
            CSPBlock(base * 4, base * 4, d * 3),
        )
        self.dark4 = nn.Sequential(
            Conv(base * 4, base * 8, 3, 2),                        # 128 -> 256
            CSPBlock(base * 8, base * 8, d * 3),
        )
        self.dark5 = nn.Sequential(
            Conv(base * 8, base * 16, 3, 2),                       # 256 -> 512
            SPP(base * 16, base * 16),
            CSPBlock(base * 16, base * 16, d, shortcut=False),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        c3 = self.dark3(x)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)
        return c3, c4, c5


# ─── PAFPN ─────────────────────────────────────────────────────────────────────

class PAFPN(nn.Module):
    def __init__(self, dep_mul=0.33, wid_mul=0.5, in_ch=(128, 256, 512)):
        super().__init__()
        d = max(round(dep_mul * 3), 1)
        h = int(256 * wid_mul)  # hidden dim: 128 for YOLOX-S

        self.up = nn.Upsample(None, 2, "nearest")
        self.d3 = Conv(in_ch[2], h, 1, 1)
        self.d4 = Conv(in_ch[1], h, 1, 1)
        self.d5 = Conv(in_ch[0], h, 1, 1)

        self.sm3 = CSPBlock(h * 2, h, d, False)
        self.sm2 = CSPBlock(h * 2, h, d, False)

        self.down3 = Conv(h, h, 3, 2)
        self.sm4 = CSPBlock(h * 2, h, d, False)
        self.down4 = Conv(h, h, 3, 2)
        self.sm5 = CSPBlock(h * 2, h, d, False)

    def forward(self, c3, c4, c5):
        p5 = self.d3(c5)
        p4 = self.sm3(torch.cat([self.up(p5), self.d4(c4)], 1))
        p3 = self.sm2(torch.cat([self.up(p4), self.d5(c3)], 1))

        n4 = self.sm4(torch.cat([self.down3(p3), p4], 1))
        n5 = self.sm5(torch.cat([self.down4(n4), p5], 1))
        return p3, n4, n5


# ─── Detection Head (cls + reg + obj + content) ────────────────────────────────

class LCDHead(nn.Module):
    def __init__(self, num_cls=1, num_content=5, wid_mul=0.5, use_ta=True):
        super().__init__()
        h = int(256 * wid_mul)  # 128 for YOLOX-S
        self.num_cls = num_cls
        self.num_content = num_content
        self.use_ta = use_ta

        # Per-scale branches
        self._make_branches(h, use_ta)

    def _make_branches(self, h, use_ta):
        self.cls_conv = nn.ModuleList([nn.Sequential(Conv(h, h, 3), Conv(h, h, 3)) for _ in range(3)])
        self.reg_conv = nn.ModuleList([nn.Sequential(Conv(h, h, 3), Conv(h, h, 3)) for _ in range(3)])

        content_layers = []
        for _ in range(3):
            layers = [Conv(h, h, 3), Conv(h, h, 3)]
            if use_ta:
                layers.append(TripletAttention(7))
            content_layers.append(nn.Sequential(*layers))
        self.content_conv = nn.ModuleList(content_layers)

        self.cls_pred = nn.ModuleList([nn.Conv2d(h, self.num_cls, 1) for _ in range(3)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(h, 4, 1) for _ in range(3)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(h, 1, 1) for _ in range(3)])
        self.content_pred = nn.ModuleList([nn.Conv2d(h, self.num_content, 1) for _ in range(3)])

    def forward(self, fpn_out):
        """Returns raw predictions before decode. Shape: list of (B, C, H, W)."""
        outputs = []
        for i, feat in enumerate(fpn_out):
            cls_f = self.cls_conv[i](feat)
            reg_f = self.reg_conv[i](feat)
            content_f = self.content_conv[i](feat)

            reg = self.reg_pred[i](reg_f)             # (B, 4, H, W)
            obj = self.obj_pred[i](reg_f)              # (B, 1, H, W)
            cls = self.cls_pred[i](cls_f)              # (B, num_cls, H, W)
            content = self.content_pred[i](content_f)  # (B, num_content, H, W)

            # Concat: [reg(4), obj(1), cls(num_cls), content(num_content)]
            outputs.append(torch.cat([reg, obj, cls, content], dim=1))
        return outputs
