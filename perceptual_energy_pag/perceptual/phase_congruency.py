"""Differentiable phase congruency (PC) with a log-Gabor filter bank, in PyTorch.

Follows the structure of Kovesi's phasecong3, with every non-smooth operation replaced by a smooth or a.e.-
differentiable counterpart so that PC(x) can be back-propagated to the input image.

Filters (frequency domain, f = radial frequency in cycles/pixel, theta = frequency orientation):
    radial   G_s(f)      = exp( -(log(f / f_s))^2 / (2 log(sigma_onf)^2) ),  f_s = 1 / (min_wavelength * mult^s),
                           G_s(0) = 0 (zero DC => exactly invariant to additive brightness offsets)
    lowpass  L(f)        = 1 / (1 + (f / 0.45)^(2 * 15))                    (suppresses corner frequencies)
    angular  A_o(theta)  = exp( -dtheta_o^2 / (2 sigma_theta^2) ),  sigma_theta = pi / norient / d_theta_on_sigma
    A_o covers one half-plane of orientations only, so ifft2(FFT(x) * G_s L A_o) is complex:
        e_so = Re(.)  (even-symmetric response),  o_so = Im(.)  (odd-symmetric response)

Per scale s / orientation o:
    A_so = sqrt(e_so^2 + o_so^2 + tiny)                                     (local amplitude)
Per orientation o (sums over scales):
    sumE = sum_s e_so,  sumO = sum_s o_so,  sumA = sum_s A_so,  maxA = max_s A_so
    mean-phase unit vector:  (phiE, phiO) = (sumE, sumO) / (sqrt(sumE^2 + sumO^2) + eps)
    local energy (Kovesi's phase-deviation form):
        En_o = sum_s [ e_so phiE + o_so phiO - |e_so phiO - o_so phiE| ]      |.| -> smooth abs
    noise threshold (Rayleigh model on the smallest scale, mean instead of median to keep gradients spread):
        tau = mean_pixels(A_0o) / sqrt(pi/2)
        totalTau = tau (1 - (1/mult)^nscale) / (1 - 1/mult)
        T_o = totalTau sqrt(pi/2) + k totalTau sqrt((4 - pi)/2)
    soft threshold:  softplus_T(z) = (T/b) log(1 + exp(b z / T)),  b = softplus_beta_rel     (~ max(z, 0))
        (scaled by T so that PC stays invariant to global contrast scaling, like the hard threshold)
    frequency-spread weight:
        width_o = (sumA / (maxA + eps) - 1) / (nscale - 1),   W_o = 1 / (1 + exp(g (cutoff - width_o)))
Final map:
    PC(x) = sum_o W_o softplus_T(En_o - T_o) / (sum_o sumA_o + eps)                   in [0, ~1]

Boundary handling: FFT => periodic boundary. Backgrounds are uniform and objects never touch the border, so
wrap-around artefacts are limited to the (weak) noise/texture.
"""
import math

import torch
import torch.nn as nn

from config import PCConfig


def _smooth_abs(z, d):
    return torch.sqrt(z * z + d * d) - d


class PhaseCongruency(nn.Module):
    def __init__(self, image_size=64, cfg: PCConfig = PCConfig()):
        super().__init__()
        self.cfg = cfg
        n = image_size
        fy = torch.fft.fftfreq(n, dtype=torch.float64)
        fx = torch.fft.fftfreq(n, dtype=torch.float64)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(FX ** 2 + FY ** 2)
        radius[0, 0] = 1.0  # avoid log(0); DC is zeroed below
        theta = torch.atan2(-FY, FX)

        lowpass = 1.0 / (1.0 + (radius / cfg.lowpass_cutoff) ** (2 * cfg.lowpass_order))
        radial = []
        for s in range(cfg.nscale):
            fo = 1.0 / (cfg.min_wavelength * cfg.mult ** s)
            lg = torch.exp(-(torch.log(radius / fo)) ** 2 / (2 * math.log(cfg.sigma_onf) ** 2)) * lowpass
            lg[0, 0] = 0.0
            radial.append(lg)
        sigma_theta = math.pi / cfg.norient / cfg.d_theta_on_sigma
        angular = []
        for o in range(cfg.norient):
            ang = o * math.pi / cfg.norient
            ds = torch.sin(theta) * math.cos(ang) - torch.cos(theta) * math.sin(ang)
            dc = torch.cos(theta) * math.cos(ang) + torch.sin(theta) * math.sin(ang)
            dtheta = torch.abs(torch.atan2(ds, dc))
            angular.append(torch.exp(-dtheta ** 2 / (2 * sigma_theta ** 2)))
        filt = torch.stack([torch.stack([r * a for a in angular]) for r in radial])  # [S,O,H,W]
        self.register_buffer("filters", filt.float())

    def forward(self, x, return_parts=False):
        """x: [B,1,H,W] -> PC map [B,1,H,W]."""
        c = self.cfg
        X = torch.fft.fft2(x[:, 0])                                     # [B,H,W]
        EO = torch.fft.ifft2(X[:, None, None] * self.filters[None])     # [B,S,O,H,W]
        e, o = EO.real, EO.imag
        A = torch.sqrt(e * e + o * o + 1e-12)

        sumE, sumO, sumA = e.sum(1), o.sum(1), A.sum(1)                 # [B,O,H,W]
        maxA = A.max(1).values
        xen = torch.sqrt(sumE * sumE + sumO * sumO + 1e-12) + c.eps
        phiE, phiO = sumE / xen, sumO / xen
        energy = (e * phiE[:, None] + o * phiO[:, None]
                  - _smooth_abs(e * phiO[:, None] - o * phiE[:, None], c.abs_delta)).sum(1)   # [B,O,H,W]

        tau = A[:, 0].mean(dim=(-2, -1)) / math.sqrt(math.pi / 2)       # [B,O]
        total_tau = tau * (1 - (1 / c.mult) ** c.nscale) / (1 - 1 / c.mult)
        T = total_tau * math.sqrt(math.pi / 2) + c.k * total_tau * math.sqrt((4 - math.pi) / 2)
        T = (T + 1e-6)[..., None, None]
        b = c.softplus_beta_rel
        en_thr = (T / b) * nn.functional.softplus(b * (energy - T) / T)

        width = (sumA / (maxA + c.eps) - 1) / (c.nscale - 1)
        W = torch.sigmoid(c.g * (width - c.cutoff))
        pc = (W * en_thr).sum(1) / (sumA.sum(1) + c.eps)                 # [B,H,W]
        if return_parts:
            return pc[:, None], {"energy": energy, "T": T, "W": W, "sumA": sumA}
        return pc[:, None]
