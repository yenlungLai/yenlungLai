"""Adversarial vulnerability decomposition (section 13).

For PGD examples x' = x + delta (true class y):
    dE_edge = E_edge(x',y) - E_edge(x,y),   dE_contrast = E_contrast(x',y) - E_contrast(x,y)
Because the two energies are in different units, each is also reported
    relative : dE / E(x)
    vs random: dE / mean dE under a random-sign perturbation of the SAME L-inf radius (eps * sign(N(0,1))).
The random control matters: any high-frequency perturbation raises GM (hence E_contrast) almost everywhere.
Also reports how the attack direction aligns with the forces: cos(delta, -g_e), cos(delta, -g_c)
(positive = the attack pushes AGAINST the force, i.e. it climbs that energy).
"""
import numpy as np
import pandas as pd
import torch

from training.common import pgd_attack


def _cos(a, b):
    a, b = a.flatten(1), b.flatten(1)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def attack_energy_changes(model, energy, x, y, eps, alpha, steps, name, seed=0, batch=200):
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for s in range(0, len(x), batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        xa = pgd_attack(model, xb, yb, eps, alpha, steps, True, gen)
        xr = (xb + eps * torch.randn(xb.shape, generator=gen).sign()).clamp(0, 1)
        with torch.no_grad():
            p0, pa, pr = model(xb).argmax(1), model(xa).argmax(1), model(xr).argmax(1)
            e0, c0 = energy.energies(xb, yb)
            ea, ca = energy.energies(xa, yb)
            er, cr = energy.energies(xr, yb)
        ge, gc = energy.get_edge_force(xb, yb), energy.get_contrast_force(xb, yb)
        d = xa - xb
        cde, cdc = _cos(d, -ge), _cos(d, -gc)
        for k in range(len(xb)):
            rows.append({"model": name, "eps": eps, "steps": steps, "index": s + k, "label": int(yb[k]),
                         "clean_correct": int(p0[k] == yb[k]), "adv_correct": int(pa[k] == yb[k]),
                         "rand_correct": int(pr[k] == yb[k]),
                         "E_edge": e0[k].item(), "E_contrast": c0[k].item(),
                         "dE_edge": (ea[k] - e0[k]).item(), "dE_contrast": (ca[k] - c0[k]).item(),
                         "dE_edge_rand": (er[k] - e0[k]).item(), "dE_contrast_rand": (cr[k] - c0[k]).item(),
                         "cos_delta_neg_ge": cde[k].item(), "cos_delta_neg_gc": cdc[k].item(),
                         "delta_l2": d[k].norm().item()})
    df = pd.DataFrame(rows)
    df["success"] = ((df.clean_correct == 1) & (df.adv_correct == 0)).astype(int)
    df["rel_dE_edge"] = df.dE_edge / df.E_edge
    df["rel_dE_contrast"] = df.dE_contrast / df.E_contrast
    df["rel_dE_edge_rand"] = df.dE_edge_rand / df.E_edge
    df["rel_dE_contrast_rand"] = df.dE_contrast_rand / df.E_contrast
    return df


def summarise_attacks(df):
    rows = []
    for (model, eps, steps), d in df.groupby(["model", "eps", "steps"], sort=False):
        cc = d[d.clean_correct == 1]
        succ = cc[cc.success == 1]
        r = {"model": model, "eps": eps, "eps_255": round(eps * 255, 1), "steps": steps,
             "clean_acc": d.clean_correct.mean(), "adv_acc": d.adv_correct.mean(),
             "attack_success_rate": cc.success.mean() if len(cc) else np.nan, "n_success": len(succ)}
        for grp, dd in [("succ", succ), ("all", cc)]:
            if len(dd) == 0:
                continue
            r[f"{grp}_rel_dE_edge"] = dd.rel_dE_edge.mean()
            r[f"{grp}_rel_dE_contrast"] = dd.rel_dE_contrast.mean()
            r[f"{grp}_rel_dE_edge_rand"] = dd.rel_dE_edge_rand.mean()
            r[f"{grp}_rel_dE_contrast_rand"] = dd.rel_dE_contrast_rand.mean()
            r[f"{grp}_edge_attack_over_random"] = dd.dE_edge.mean() / (dd.dE_edge_rand.mean() + 1e-12)
            r[f"{grp}_contrast_attack_over_random"] = dd.dE_contrast.mean() / (dd.dE_contrast_rand.mean() + 1e-12)
            r[f"{grp}_cos_delta_neg_ge"] = dd.cos_delta_neg_ge.mean()
            r[f"{grp}_cos_delta_neg_gc"] = dd.cos_delta_neg_gc.mean()
        rows.append(r)
    return pd.DataFrame(rows)
