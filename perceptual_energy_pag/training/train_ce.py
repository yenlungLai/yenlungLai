"""Model A: standard cross-entropy, L = L_CE."""
import torch.nn.functional as F

from training.common import train_model


def ce_loss(model, xb, yb, extras, gen):
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    return loss, {"loss": loss.item(), "acc": (logits.argmax(1) == yb).float().mean().item()}


def train_ce(x_train, y_train, cfg):
    return train_model(ce_loss, x_train, y_train, cfg, "CE")
