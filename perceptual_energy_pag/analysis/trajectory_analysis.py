"""Gradient-flow trajectory comparison (section 12).

From the same x0:
    classifier ascent        x^f_{t+1} = x^f_t + eta * v / ||v||,          v = grad_x f_y
    perceptual-energy descent x^E_{t+1} = x^E_t - eta * grad E / ||grad E||,  E = E_edge + E_contrast
Unclipped (pure flow in R^N, so the two fields are compared without projection effects).
Per step: cos(grad f_y, -grad E_y) evaluated on BOTH paths, ||x^f_t - x^E_t||, and the normalised distance
d_t / (eta t), which is 0 for identical paths, ~sqrt(2) for orthogonal straight paths and 2 for opposite ones.
"""
import numpy as np
import pandas as pd
import torch

from training.common import input_gradient


def _unit(g):
    return g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)


def _cos(a, b):
    a, b = a.flatten(1), b.flatten(1)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12))


def energy_descent_path(energy, x0, y, steps, eta):
    xs = [x0.clone()]
    x = x0.clone()
    for _ in range(steps):
        g = energy.get_total_force(x, y)
        x = x + eta * _unit(g)
        xs.append(x.clone())
    return torch.stack(xs, 1)  # [B,T+1,1,H,W]


def classifier_ascent_path(model, x0, y, steps, eta):
    xs = [x0.clone()]
    x = x0.clone()
    for _ in range(steps):
        v, _ = input_gradient(model, x, y)
        x = x + eta * _unit(v)
        xs.append(x.clone())
    return torch.stack(xs, 1)


def evaluate_paths(model, energy, pf, pe, y, eta, name):
    rows = []
    B, T1 = pf.shape[:2]
    for t in range(T1):
        xf, xe = pf[:, t], pe[:, t]
        vf, lf = input_gradient(model, xf, y)
        ve, le = input_gradient(model, xe, y)
        gf, ge = energy.get_total_force(xf, y), energy.get_total_force(xe, y)
        with torch.no_grad():
            Ef = sum(energy.energies(xf, y))
            Ee = sum(energy.energies(xe, y))
        dist = (xf - xe).flatten(1).norm(dim=1)
        cf, ce = _cos(vf, gf), _cos(ve, ge)
        for b in range(B):
            rows.append({"model": name, "image": b, "label": int(y[b]), "step": t,
                         "cos_on_classifier_path": cf[b].item(), "cos_on_energy_path": ce[b].item(),
                         "dist": dist[b].item(), "norm_dist": dist[b].item() / (eta * t) if t > 0 else np.nan,
                         "f_y_classifier_path": lf[b, y[b]].item(), "f_y_energy_path": le[b, y[b]].item(),
                         "E_classifier_path": Ef[b].item(), "E_energy_path": Ee[b].item()})
    return pd.DataFrame(rows)
