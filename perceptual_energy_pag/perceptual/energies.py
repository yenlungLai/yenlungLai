"""Class-conditioned perceptual energies and the forces obtained from them by autograd.

    mu_PC,y = E_{x:y}[PC(x)],  mu_GM,y = E_{x:y}[GM(x)]          (prototype split: centred, roughly aligned)
    E_edge(x,y)     = 1/2 || PC(x) - mu_PC,y ||_2^2
    E_contrast(x,y) = 1/2 || GM(x) - mu_GM,y ||_2^2
    g_e(x,y) = -grad_x E_edge(x,y),   g_c(x,y) = -grad_x E_contrast(x,y)     (torch.autograd, never hand-designed)
Controls:
    g_pixel(x,y) = r_y - x  with r_y the class-mean image (Ganz-style pixel-distance force)
"""
import torch
import torch.nn as nn

from perceptual.gradient_magnitude import GradientMagnitude
from perceptual.phase_congruency import PhaseCongruency


class PerceptualEnergy(nn.Module):
    def __init__(self, pc: PhaseCongruency, gm: GradientMagnitude, n_classes=3, image_size=64):
        super().__init__()
        self.pc, self.gm = pc, gm
        self.register_buffer("mu_pc", torch.zeros(n_classes, 1, image_size, image_size))
        self.register_buffer("mu_gm", torch.zeros(n_classes, 1, image_size, image_size))
        self.register_buffer("mu_pix", torch.zeros(n_classes, 1, image_size, image_size))

    @torch.no_grad()
    def fit_prototypes(self, images, labels, batch=256):
        pcs, gms = [], []
        for i in range(0, len(images), batch):
            pcs.append(self.pc(images[i:i + batch]))
            gms.append(self.gm(images[i:i + batch]))
        pcs, gms = torch.cat(pcs), torch.cat(gms)
        for y in range(self.mu_pc.shape[0]):
            m = labels == y
            self.mu_pc[y] = pcs[m].mean(0)
            self.mu_gm[y] = gms[m].mean(0)
            self.mu_pix[y] = images[m].mean(0)

    # -------------------------------------------------------------- energies
    def edge_energy(self, x, y):
        return 0.5 * ((self.pc(x) - self.mu_pc[y]) ** 2).sum(dim=(1, 2, 3))

    def contrast_energy(self, x, y):
        return 0.5 * ((self.gm(x) - self.mu_gm[y]) ** 2).sum(dim=(1, 2, 3))

    def energies(self, x, y):
        """Both energies from one PC/GM evaluation. Returns (E_edge, E_contrast) each [B]."""
        pc, gm = self.pc(x), self.gm(x)
        return (0.5 * ((pc - self.mu_pc[y]) ** 2).sum(dim=(1, 2, 3)),
                0.5 * ((gm - self.mu_gm[y]) ** 2).sum(dim=(1, 2, 3)))

    # -------------------------------------------------------------- forces (autograd)
    def _force(self, fn, x, y):
        with torch.enable_grad():
            xx = x.detach().clone().requires_grad_(True)
            e = fn(xx, y)
            (g,) = torch.autograd.grad(e.sum(), xx)  # samples are independent -> per-sample gradients
        return -g.detach()

    def get_edge_force(self, x, y):
        return self._force(self.edge_energy, x, y)

    def get_contrast_force(self, x, y):
        return self._force(self.contrast_energy, x, y)

    def get_total_force(self, x, y):
        return self._force(lambda a, b: self.edge_energy(a, b) + self.contrast_energy(a, b), x, y)

    def get_pixel_force(self, x, y):
        return (self.mu_pix[y] - x).detach()
