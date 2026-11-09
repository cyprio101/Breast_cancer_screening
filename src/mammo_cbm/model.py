"""Concept-bottleneck network and Grad-CAM."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .imaging import IMG


class CBM(nn.Module):
    """Backbone + one softmax head per concept group. heads = {"margins": ["CIRCUMSCRIBED", ...], ...}."""

    def __init__(self, heads, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        d = self.backbone.num_features
        self.heads = nn.ModuleDict({k: nn.Linear(d, len(v)) for k, v in heads.items()})

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        g = self.features(x).mean((2, 3))
        return {k: h(g) for k, h in self.heads.items()}, g


def gradcam(model, x, head, cls):
    """Grad-CAM for one concept class. Post-hoc and approximate: shows where the network looked, not proof."""
    model.eval()
    with torch.enable_grad():
        model.zero_grad()
        f = model.features(x[None])
        f.retain_grad()
        model.heads[head](f.mean((2, 3)))[0, cls].backward()
        cam = F.relu((f.grad.mean((2, 3), keepdim=True) * f).sum(1))[0].detach()
    cam = F.interpolate(cam[None, None], (IMG, IMG), mode="bilinear", align_corners=False)[0, 0]
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).numpy()
