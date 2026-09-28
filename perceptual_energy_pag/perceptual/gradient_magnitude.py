"""Differentiable gradient magnitude with fixed Sobel convolutions.

    Gx = (x * Kx) / 8,  Gy = (x * Ky) / 8,  Kx = [[-1,0,1],[-2,0,2],[-1,0,1]],  Ky = Kx^T
    GM(x) = sqrt(Gx^2 + Gy^2 + eps)
Replicate padding (zero padding would create an artificial edge at the border of a non-black background).
The /8 makes GM the intensity change per pixel for a linear ramp. Kx, Ky sum to zero => GM is exactly invariant
to additive brightness offsets (up to clipping).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GMConfig


class GradientMagnitude(nn.Module):
    def __init__(self, cfg: GMConfig = GMConfig()):
        super().__init__()
        self.cfg = cfg
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) * cfg.sobel_scale
        self.register_buffer("kernel", torch.stack([kx, kx.t()])[:, None])  # [2,1,3,3]

    def forward(self, x, return_parts=False):
        g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode=self.cfg.padding), self.kernel)
        gm = torch.sqrt(g[:, :1] ** 2 + g[:, 1:] ** 2 + self.cfg.eps)
        if return_parts:
            return gm, {"gx": g[:, :1], "gy": g[:, 1:]}
        return gm
