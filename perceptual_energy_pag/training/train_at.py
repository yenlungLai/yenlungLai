"""Model B: PGD adversarial training (Madry et al.), L = L_CE(x + delta*), delta* = PGD_{L-inf, eps}(x).
Adversarial examples only (no clean term) once eps is ramped up. Schedule: at_clean_epochs with eps=0, then a
linear eps/alpha ramp over at_eps_ramp_epochs, then full eps. Attack parameters come from cfg.train.at_*."""
import torch.nn.functional as F

from training.common import pgd_attack, train_model


def make_at_loss(cfg):
    t = cfg.train

    def at_loss(model, xb, yb, extras, gen):
        ramp = min(1.0, max(0.0, (at_loss.epoch + 1 - t.at_clean_epochs) / max(t.at_eps_ramp_epochs, 1)))
        eps, alpha = t.at_eps * ramp, t.at_alpha * ramp
        xa = pgd_attack(model, xb, yb, eps, alpha, t.at_steps, t.at_random_start, gen) if eps > 0 else xb
        logits = model(xa)
        loss = F.cross_entropy(logits, yb)
        return loss, {"loss": loss.item(), "adv_acc": (logits.argmax(1) == yb).float().mean().item(), "eps255": eps * 255}

    at_loss.epoch = 0
    return at_loss


def train_at(x_train, y_train, cfg):
    return train_model(make_at_loss(cfg), x_train, y_train, cfg, "AT", epochs=cfg.train.at_epochs)
