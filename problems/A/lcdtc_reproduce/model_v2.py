"""LCD-YOLOX v2: Pre-trained ResNet50 backbone + proper augmentation.

Key improvements over v1:
  1. ResNet50 backbone pre-trained on ImageNet
  2. Mosaic + flip + color augmentation
  3. Linear warmup + cosine schedule
  4. IoU-based regression loss (GIoU)
  5. EMA for stable evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .triplet_attention import TripletAttention


# ─── Building blocks ───────────────────────────────────────────────────────────

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        h = int(c2 * e)
        self.cv1 = Conv(c1, h, 1, 1)
        self.cv2 = Conv(c1, h, 1, 1)
        self.cv3 = Conv(2 * h, c2, 1, 1)
        self.m = nn.Sequential(*[
            nn.Sequential(Conv(h, h, 3), Conv(h, h, 3))
            for _ in range(n)
        ])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ─── Pre-trained ResNet50 backbone ─────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """ResNet50 from torchvision, outputs c3/c4/c5 feature maps."""
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 ch, stride 4
        self.layer2 = resnet.layer2  # 512 ch, stride 8
        self.layer3 = resnet.layer3  # 1024 ch, stride 16
        self.layer4 = resnet.layer4  # 2048 ch, stride 32

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)   # (B, 512, H/8,  W/8)
        c4 = self.layer3(c3)  # (B, 1024, H/16, W/16)
        c5 = self.layer4(c4)  # (B, 2048, H/32, W/32)
        return c3, c4, c5


# ─── FPN + PAN neck ─────────────────────────────────────────────────────────────

class PAFPN(nn.Module):
    def __init__(self, in_ch=(512, 1024, 2048), out_ch=256):
        super().__init__()
        self.up = nn.Upsample(None, 2, "nearest")

        # Lateral convs (reduce channels to out_ch)
        self.lc3 = Conv(in_ch[0], out_ch, 1)
        self.lc4 = Conv(in_ch[1], out_ch, 1)
        self.lc5 = Conv(in_ch[2], out_ch, 1)

        # Top-down merge
        self.td4 = CSPBlock(out_ch * 2, out_ch, 1, False)
        self.td3 = CSPBlock(out_ch * 2, out_ch, 1, False)

        # Bottom-up merge
        self.down4 = Conv(out_ch, out_ch, 3, 2)
        self.bu4 = CSPBlock(out_ch * 2, out_ch, 1, False)
        self.down5 = Conv(out_ch, out_ch, 3, 2)
        self.bu5 = CSPBlock(out_ch * 2, out_ch, 1, False)

    def forward(self, c3, c4, c5):
        p5 = self.lc5(c5)
        p4 = self.td4(torch.cat([self.up(p5), self.lc4(c4)], 1))
        p3 = self.td3(torch.cat([self.up(p4), self.lc3(c3)], 1))

        n4 = self.bu4(torch.cat([self.down4(p3), p4], 1))
        n5 = self.bu5(torch.cat([self.down5(n4), p5], 1))
        return p3, n4, n5


# ─── Detection Head (cls + reg + obj + content) ────────────────────────────────

class LCDHead(nn.Module):
    def __init__(self, num_cls=1, num_content=5, hidden=256, use_ta=True):
        super().__init__()
        self.num_cls = num_cls
        self.num_content = num_content

        self.cls_conv = nn.ModuleList([nn.Sequential(Conv(hidden, hidden, 3), Conv(hidden, hidden, 3)) for _ in range(3)])
        self.reg_conv = nn.ModuleList([nn.Sequential(Conv(hidden, hidden, 3), Conv(hidden, hidden, 3)) for _ in range(3)])

        content_layers = []
        for _ in range(3):
            layers = [Conv(hidden, hidden, 3), Conv(hidden, hidden, 3)]
            if use_ta:
                layers.append(TripletAttention(7))
            content_layers.append(nn.Sequential(*layers))
        self.content_conv = nn.ModuleList(content_layers)

        self.cls_pred = nn.ModuleList([nn.Conv2d(hidden, num_cls, 1) for _ in range(3)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(hidden, 4, 1) for _ in range(3)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(hidden, 1, 1) for _ in range(3)])
        self.content_pred = nn.ModuleList([nn.Conv2d(hidden, num_content, 1) for _ in range(3)])

    def forward(self, fpn_out):
        outputs = []
        for i, feat in enumerate(fpn_out):
            cls_f = self.cls_conv[i](feat)
            reg_f = self.reg_conv[i](feat)
            content_f = self.content_conv[i](feat)

            reg = self.reg_pred[i](reg_f)
            obj = self.obj_pred[i](reg_f)
            cls = self.cls_pred[i](cls_f)
            content = self.content_pred[i](content_f)

            outputs.append(torch.cat([reg, obj, cls, content], dim=1))
        return outputs


# ─── Full model ─────────────────────────────────────────────────────────────────

class LCDYOLOXModelV2(nn.Module):
    def __init__(self, num_cls=1, num_content=5, use_ta=True, pretrained=True, fpn_dim=256):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.neck = PAFPN(in_ch=(512, 1024, 2048), out_ch=fpn_dim)
        self.head = LCDHead(num_cls, num_content, fpn_dim, use_ta)

        self.num_cls = num_cls
        self.num_content = num_content
        self.strides = (8, 16, 32)

        # Freeze BN in backbone (pretrained, large batch norm stats are good)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self._init_new_layers()

    def _init_new_layers(self):
        """Initialize only the new (non-backbone) layers."""
        for m in self.neck.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        fpn = self.neck(c3, c4, c5)
        return self.head(fpn)


# ─── EMA ────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average for stable evaluation."""
    def __init__(self, model, decay=0.9998):
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        for k, v in model.state_dict().items():
            self.ema[k] = self.decay * self.ema[k] + (1 - self.decay) * v.clone().detach()

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema)

    def restore(self, model):
        model.load_state_dict(self.backup)
        del self.backup
