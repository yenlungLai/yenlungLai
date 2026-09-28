"""Shared training / evaluation utilities: PGD attack, generic training loop, accuracy evaluation."""
import json
import os
import time

import torch
import torch.nn.functional as F

from models.small_cnn import SmallCNN


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True, generator=None):
    """L-inf PGD on the cross-entropy loss, pixel range [0,1]."""
    was_training = model.training
    model.eval()
    x = x.detach()
    if random_start:
        noise = torch.rand(x.shape, generator=generator, dtype=x.dtype) * 2 - 1
        x_adv = (x + eps * noise).clamp(0, 1)
    else:
        x_adv = x.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (g,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    model.train(was_training)
    return x_adv.detach()


def input_gradient(model, x, y, create_graph=False):
    """v(x,y) = grad_x f_y(x), f_y = pre-softmax logit of class y. Per-sample (samples are independent)."""
    with torch.enable_grad():
        xx = x if create_graph else x.detach().clone()
        if not xx.requires_grad:
            xx.requires_grad_(True)
        logits = model(xx)
        f = logits.gather(1, y[:, None]).squeeze(1)
        (v,) = torch.autograd.grad(f.sum(), xx, create_graph=create_graph)
    return v, logits


@torch.no_grad()
def accuracy(model, x, y, batch=500):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x)


def robust_accuracy(model, x, y, eps, alpha, steps, batch=250, seed=0):
    gen = torch.Generator().manual_seed(seed)
    correct = 0
    for i in range(0, len(x), batch):
        xa = pgd_attack(model, x[i:i + batch], y[i:i + batch], eps, alpha, steps, True, gen)
        with torch.no_grad():
            correct += (model(xa).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x)


def train_model(loss_fn, x_train, y_train, cfg, name, extra_tensors=None, log_every=1, epochs=None):
    """Generic loop. loss_fn(model, xb, yb, extras_b, gen) -> (loss, dict_of_logged_scalars)."""
    epochs = epochs or cfg.train.epochs
    torch.manual_seed(cfg.train.model_seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = (len(x_train) + cfg.train.batch_size - 1) // cfg.train.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(cfg.train.model_seed + 1)
    history = []
    for ep in range(epochs):
        loss_fn.epoch = ep
        model.train()
        perm = torch.randperm(len(x_train), generator=gen)
        t0, agg, nb = time.time(), {}, 0
        for i in range(0, len(x_train), cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]
            extras = None if extra_tensors is None else [t[idx] for t in extra_tensors]
            loss, logs = loss_fn(model, x_train[idx], y_train[idx], extras, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            nb += 1
        rec = {"epoch": ep + 1, "time_s": round(time.time() - t0, 1), **{k: v / nb for k, v in agg.items()}}
        history.append(rec)
        if (ep + 1) % log_every == 0:
            print(f"  [{name}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in rec.items()), flush=True)
    return model, history


def save_checkpoint(model, history, path, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history, "meta": meta}, path)
    with open(path.replace(".pt", "_history.json"), "w") as f:
        json.dump({"history": history, "meta": meta}, f, indent=2)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SmallCNN()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck
