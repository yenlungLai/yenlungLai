"""Stage 1: do PC and GM respond to the physical factors we claim they respond to?

For controlled pairs (x, x') differing in ONE generative factor:
    D_PC = ||PC(x) - PC(x')||_2,   D_GM = ||GM(x) - GM(x')||_2
Normalisations (so PC and GM, which live in different units, can be compared):
    rel   : D_PC / ||PC(x)||_2  and  D_GM / ||GM(x)||_2
    floor : D / median(D under a pure noise re-sample of the same image)   (operator's own variability floor)
    selectivity = log2(rel_PC / rel_GM)   (>0: PC responds relatively more; <0: GM responds relatively more)
Also reports the change in the class-conditioned energies and a nearest-prototype classification sanity check.
"""
import numpy as np
import pandas as pd
import torch

from config import CLASSES
from data import transformations as T

LEVELS = {  # small / medium / large (medium is used in the headline table)
    "contrast": [1.1, 1.25, 1.5],
    "brightness": [0.02, 0.05, 0.10],
    "brightness_gain(pixel)": [1.1, 1.25, 1.5],
    "position": [1.0, 2.0, 4.0],
    "rotation": [5.0, 10.0, 20.0],
    "texture": [0.02, 0.05, 0.10],
    "size": [1.05, 1.10, 1.20],
}


def _pairs(metas, images, image_size):
    """Yield (transformation, level_name, level_value, base_index, x_prime)."""
    for i, m in enumerate(metas):
        x = images[i, 0]
        for j, off in enumerate([1, 2]):
            tgt = CLASSES[(m["label"] + off) % 3]
            yield "shape", f"->{off}", tgt, i, T.change_shape(m, tgt, image_size)[0]
        yield "noise_resample", "medium", 0, i, T.resample_noise(m, m["noise_seed"] + 99991, image_size)[0]
        for name, vals in LEVELS.items():
            for lvl, v in zip(["small", "medium", "large"], vals):
                if name == "contrast":
                    xp = T.change_contrast(m, v, image_size)[0]
                elif name == "brightness":
                    xp = T.change_brightness(m, v, image_size)[0]
                elif name == "brightness_gain(pixel)":
                    xp = T.brightness_gain(x, v)
                elif name == "position":
                    xp = T.change_position(m, v / np.sqrt(2), v / np.sqrt(2), image_size)[0]
                elif name == "rotation":
                    if m["class"] == "circle":  # rotation of a circle is not a physical change
                        continue
                    xp = T.change_rotation(m, v, image_size)[0]
                elif name == "texture":
                    xp = T.change_texture(m, m["texture_strength"] + v, image_size)[0]
                elif name == "size":
                    xp = T.change_size(m, v, image_size)[0]
                yield name, lvl, v, i, xp


@torch.no_grad()
def transformation_sensitivity(energy, images, labels, metas, image_size=64, batch=256):
    x = torch.tensor(images)
    y = torch.tensor(labels)
    pc0, gm0 = energy.pc(x), energy.gm(x)
    ee0, ec0 = energy.energies(x, y)
    recs = list(_pairs(metas, images, image_size))
    rows = []
    for s in range(0, len(recs), batch):
        chunk = recs[s:s + batch]
        idx = torch.tensor([r[3] for r in chunk])
        xp = torch.tensor(np.stack([r[4] for r in chunk]))[:, None]
        pc1, gm1 = energy.pc(xp), energy.gm(xp)
        ee1, ec1 = energy.energies(xp, y[idx])
        d_pc = (pc1 - pc0[idx]).flatten(1).norm(dim=1)
        d_gm = (gm1 - gm0[idx]).flatten(1).norm(dim=1)
        d_px = (xp - x[idx]).flatten(1).norm(dim=1)
        for k, r in enumerate(chunk):
            i = r[3]
            rows.append({
                "transformation": r[0], "level": r[1], "value": r[2], "base_index": i,
                "class": metas[i]["class"],
                "D_PC": d_pc[k].item(), "D_GM": d_gm[k].item(), "D_pixel": d_px[k].item(),
                "norm_PC": pc0[i].norm().item(), "norm_GM": gm0[i].norm().item(),
                "dE_edge": (ee1[k] - ee0[i]).item(), "dE_contrast": (ec1[k] - ec0[i]).item(),
                "E_edge": ee0[i].item(), "E_contrast": ec0[i].item(),
            })
    df = pd.DataFrame(rows)
    df["rel_PC"] = df.D_PC / df.norm_PC
    df["rel_GM"] = df.D_GM / df.norm_GM
    floor = df[df.transformation == "noise_resample"]
    df["floor_PC"] = df.D_PC / floor.D_PC.median()
    df["floor_GM"] = df.D_GM / floor.D_GM.median()
    sel = np.log2((df.rel_PC + 1e-12) / (df.rel_GM + 1e-12))
    df["selectivity_log2"] = np.where((df.rel_PC < 1e-6) & (df.rel_GM < 1e-6), np.nan, sel)  # 0/0 -> undefined
    df["rel_dE_edge"] = df.dE_edge / df.E_edge
    df["rel_dE_contrast"] = df.dE_contrast / df.E_contrast
    return df


def summarise_sensitivity(df, n_boot=2000, seed=0):
    from analysis.stats import bootstrap_ci
    rows = []
    sel = df[(df.level == "medium") | (df.transformation == "shape")]
    order = ["shape", "contrast", "brightness", "brightness_gain(pixel)", "position", "rotation", "texture", "size",
             "noise_resample"]
    for t in order:
        d = sel[sel.transformation == t]
        if len(d) == 0:
            continue
        r = {"transformation": t, "n": len(d), "mean_D_pixel": d.D_pixel.mean(),
             "mean_D_PC": d.D_PC.mean(), "mean_D_GM": d.D_GM.mean()}
        for col in ["rel_PC", "rel_GM", "floor_PC", "floor_GM", "selectivity_log2", "rel_dE_edge", "rel_dE_contrast"]:
            est, lo, hi = bootstrap_ci(d[col].values, np.mean, n_boot, seed)
            r[col], r[col + "_lo"], r[col + "_hi"] = est, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


@torch.no_grad()
def nearest_prototype_accuracy(energy, images, labels, batch=256):
    """Sanity check: is argmin_y E(x, y) the true class? (chance = 1/3)"""
    x, y = torch.tensor(images), torch.tensor(labels)
    Es = {"E_edge": [], "E_contrast": [], "E_total": [], "pixel": []}
    for s in range(0, len(x), batch):
        xb = x[s:s + batch]
        pc, gm = energy.pc(xb), energy.gm(xb)
        ee = 0.5 * ((pc[:, None] - energy.mu_pc[None]) ** 2).sum(dim=(2, 3, 4))
        ec = 0.5 * ((gm[:, None] - energy.mu_gm[None]) ** 2).sum(dim=(2, 3, 4))
        ep = 0.5 * ((xb[:, None] - energy.mu_pix[None]) ** 2).sum(dim=(2, 3, 4))
        Es["E_edge"].append(ee)
        Es["E_contrast"].append(ec)
        Es["E_total"].append(ee + ec)
        Es["pixel"].append(ep)
    return {k: float((torch.cat(v).argmin(1) == y).float().mean()) for k, v in Es.items()}
