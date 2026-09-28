"""Model C: PAG-like gradient-alignment training on CLEAN images only.

    L = L_CE + lambda * L_PAG,   L_PAG = 1 - cos( grad_x f_y(x), g_target(x,y) )
    g_target = -grad_x [E_edge(x,y) + E_contrast(x,y)]   (true-class prototypes)
The target does not depend on the model, so it is precomputed once for the whole training set. The classifier
input gradient is taken with create_graph=True so that L_PAG is differentiated through it (double backprop).
"""
import torch
import torch.nn.functional as F

from training.common import input_gradient, train_model


@torch.no_grad()
def precompute_targets(energy, x, y, batch=250):
    out = []
    for i in range(0, len(x), batch):
        out.append(energy.get_total_force(x[i:i + batch], y[i:i + batch]))
    return torch.cat(out)


def make_pag_loss(cfg):
    lam = cfg.train.pag_lambda

    def pag_loss(model, xb, yb, extras, gen):
        (g_t,) = extras
        xb = xb.clone().requires_grad_(True)
        v, logits = input_gradient(model, xb, yb, create_graph=True)
        ce = F.cross_entropy(logits, yb)
        cos = F.cosine_similarity(v.flatten(1), g_t.flatten(1), dim=1, eps=1e-12)
        l_pag = (1 - cos).mean()
        loss = ce + lam * l_pag
        return loss, {"loss": loss.item(), "ce": ce.item(), "cos": cos.mean().item(),
                      "acc": (logits.argmax(1) == yb).float().mean().item()}

    return pag_loss


def train_pag(x_train, y_train, targets, cfg):
    return train_model(make_pag_loss(cfg), x_train, y_train, cfg, "PAG", extra_tensors=[targets])
