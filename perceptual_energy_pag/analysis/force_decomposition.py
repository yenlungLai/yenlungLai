"""Main experiment: decompose classifier input gradients into perceptual forces.

    v(x,y) = grad_x f_y(x)
    min_{a_e,a_c} || v - a_e g_e - a_c g_c ||^2     (unconstrained LS and NNLS a_e,a_c >= 0)
    vhat = a_e g_e + a_c g_c
Metrics (per sample):
    A      = cos(v, vhat)
    R_res  = ||v - vhat|| / ||v||
    R2     = 1 - ||v - vhat||^2 / ||v||^2       (>= 0 for any LS fit because a=0 is feasible; reported unclipped)
    rho_e  = |a_e| ||g_e|| / (|a_e| ||g_e|| + |a_c| ||g_c|| + eps),  rho_c = 1 - rho_e
All metrics are invariant to rescaling v, g_e or g_c, so norm differences cannot drive them.

Controls (same samples, paired):
    random  : two random Gaussian directions (2-regressor LS; chance R2 ~ 2/4096)
    signscr : g_e, g_c with pixel signs randomly flipped (same spatial magnitude profile, destroyed direction)
    wrong   : g_e, g_c computed with a WRONG-class prototype (mean over both wrong classes)
    pixel   : g_pixel = r_y - x (class-mean image), alone and together with g_e, g_c
Zero-parameter alignments cos(v, g) are reported for g_e, g_c, g_e+g_c, g_pixel, random, sign-scrambled, wrong.
"""
import numpy as np
import pandas as pd
import torch

from training.common import input_gradient

EPS = 1e-12


def _flat(t):
    return t.flatten(1).double()


def cos(a, b):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + EPS)


def lstsq(v, G, ridge=1e-10):
    """Batched LS: v [B,N], G [B,N,k] -> a [B,k] (normal equations, relative ridge for stability)."""
    GtG = G.transpose(1, 2) @ G
    scale = GtG.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
    I = torch.eye(G.shape[2], dtype=G.dtype)[None]
    return torch.linalg.solve(GtG + ridge * scale * I, (G.transpose(1, 2) @ v[..., None]))[..., 0]


def nnls2(v, g1, g2):
    """Exact 2-variable NNLS by enumerating the KKT candidates {both>0, only a1, only a2, none}."""
    G = torch.stack([g1, g2], 2)
    a_full = lstsq(v, G)
    a1 = ((v * g1).sum(1) / ((g1 * g1).sum(1) + EPS)).clamp(min=0)
    a2 = ((v * g2).sum(1) / ((g2 * g2).sum(1) + EPS)).clamp(min=0)
    z = torch.zeros_like(a1)
    cands = torch.stack([a_full, torch.stack([a1, z], 1), torch.stack([z, a2], 1), torch.stack([z, z], 1)], 1)
    feasible = (cands >= 0).all(2)
    res = ((v[:, None] - (cands[..., None] * G.transpose(1, 2)[:, None]).sum(2)) ** 2).sum(2)
    res = torch.where(feasible, res, torch.full_like(res, float("inf")))
    best = res.argmin(1)
    return cands[torch.arange(len(v)), best]


def fit_metrics(v, G, a):
    vhat = (G * a[:, None, :]).sum(2)
    r = v - vhat
    vn = v.norm(dim=1)
    return {"cos": cos(v, vhat), "rres": r.norm(dim=1) / (vn + EPS), "r2": 1 - (r ** 2).sum(1) / (vn ** 2 + EPS)}, vhat


def rho_edge(a_e, a_c, ge, gc):
    ne = a_e.abs() * ge.norm(dim=1)
    nc = a_c.abs() * gc.norm(dim=1)
    den = ne + nc
    return torch.where(den > 1e-30, ne / (den + 1e-300), torch.full_like(den, float("nan")))


# ------------------------------------------------------------------ model-independent fields (cached per split)
@torch.no_grad()
def perceptual_fields(energy, x, y, seed, batch=200):
    """Forces for ALL classes + controls. Deterministic random controls (paired across models)."""
    n_cls = energy.mu_pc.shape[0]
    ge_all, gc_all, Ee_all, Ec_all = [], [], [], []
    for c in range(n_cls):
        ge, gc, ee, ec = [], [], [], []
        for s in range(0, len(x), batch):
            xb = x[s:s + batch]
            yc = torch.full((len(xb),), c, dtype=torch.long)
            ge.append(energy.get_edge_force(xb, yc))
            gc.append(energy.get_contrast_force(xb, yc))
            e1, e2 = energy.energies(xb, yc)
            ee.append(e1)
            ec.append(e2)
        ge_all.append(torch.cat(ge))
        gc_all.append(torch.cat(gc))
        Ee_all.append(torch.cat(ee))
        Ec_all.append(torch.cat(ec))
    gen = torch.Generator().manual_seed(seed)
    shape = x.shape
    pcs, gms = [], []
    for s in range(0, len(x), batch):
        pcs.append(energy.pc(x[s:s + batch]))
        gms.append(energy.gm(x[s:s + batch]))
    return {
        "pc": torch.cat(pcs), "gm": torch.cat(gms),
        "ge": torch.stack(ge_all, 1), "gc": torch.stack(gc_all, 1),         # [B,C,1,H,W]
        "E_edge": torch.stack(Ee_all, 1), "E_contrast": torch.stack(Ec_all, 1),  # [B,C]
        "gpix": energy.get_pixel_force(x, y),
        "rand1": torch.randn(shape, generator=gen), "rand2": torch.randn(shape, generator=gen),
        "sign1": torch.randint(0, 2, shape, generator=gen).float() * 2 - 1,
        "sign2": torch.randint(0, 2, shape, generator=gen).float() * 2 - 1,
    }


def classifier_gradients(model, x, y, batch=250):
    vs, logits = [], []
    for s in range(0, len(x), batch):
        v, lg = input_gradient(model, x[s:s + batch], y[s:s + batch])
        vs.append(v)
        logits.append(lg.detach())
    return torch.cat(vs), torch.cat(logits)


def decompose(model, fields, x, y, model_name, split, metas=None):
    """Per-sample decomposition table + tensors needed for figures."""
    v_img, logits = classifier_gradients(model, x, y)
    B = len(x)
    ar = torch.arange(B)
    v = _flat(v_img)
    ge, gc = _flat(fields["ge"][ar, y]), _flat(fields["gc"][ar, y])
    gsum = ge + gc
    gpix = _flat(fields["gpix"])
    rand1, rand2 = _flat(fields["rand1"]), _flat(fields["rand2"])
    s1, s2 = _flat(fields["sign1"]), _flat(fields["sign2"])

    out = {"model": model_name, "split": split, "index": np.arange(B), "label": y.numpy(),
           "pred": logits.argmax(1).numpy(), "logit_y": logits[ar, y].numpy()}
    out["correct"] = (out["pred"] == out["label"]).astype(int)
    out["v_norm"], out["ge_norm"], out["gc_norm"] = v.norm(dim=1).numpy(), ge.norm(dim=1).numpy(), gc.norm(dim=1).numpy()

    # --- main fits
    G = torch.stack([ge, gc], 2)
    a_ls = lstsq(v, G)
    m_ls, vhat_ls = fit_metrics(v, G, a_ls)
    a_nn = nnls2(v, ge, gc)
    m_nn, vhat_nn = fit_metrics(v, G, a_nn)
    for pre, a, m in [("ls", a_ls, m_ls), ("nnls", a_nn, m_nn)]:
        out[f"{pre}_a_e"], out[f"{pre}_a_c"] = a[:, 0].numpy(), a[:, 1].numpy()
        for k in ["cos", "rres", "r2"]:
            out[f"{pre}_{k}"] = m[k].numpy()
        re = rho_edge(a[:, 0], a[:, 1], ge, gc)
        out[f"{pre}_rho_e"], out[f"{pre}_rho_c"] = re.numpy(), (1 - re).numpy()

    # --- zero-parameter alignments
    out["cos_ge"], out["cos_gc"], out["cos_gsum"] = cos(v, ge).numpy(), cos(v, gc).numpy(), cos(v, gsum).numpy()
    out["cos_gpix"] = cos(v, gpix).numpy()
    g_rand = rand1 / rand1.norm(dim=1, keepdim=True) * gsum.norm(dim=1, keepdim=True)  # norm-matched random force
    out["cos_grand"] = cos(v, g_rand).numpy()
    out["cos_gsum_signscr"] = cos(v, gsum * s1).numpy()
    out["r2_gsum"] = out["cos_gsum"] ** 2
    out["r2_pix"] = out["cos_gpix"] ** 2

    # --- controls, 2-regressor LS
    Gr = torch.stack([rand1, rand2], 2)
    out["r2_ls_random2"] = fit_metrics(v, Gr, lstsq(v, Gr))[0]["r2"].numpy()
    Gs = torch.stack([ge * s1, gc * s2], 2)
    out["r2_ls_signscr"] = fit_metrics(v, Gs, lstsq(v, Gs))[0]["r2"].numpy()
    n_cls = fields["ge"].shape[1]
    r2w, cw, rhow = [], [], []
    for off in range(1, n_cls):
        yw = (y + off) % n_cls
        gew, gcw = _flat(fields["ge"][ar, yw]), _flat(fields["gc"][ar, yw])
        Gw = torch.stack([gew, gcw], 2)
        aw = lstsq(v, Gw)
        r2w.append(fit_metrics(v, Gw, aw)[0]["r2"])
        cw.append(cos(v, gew + gcw))
        aw_nn = nnls2(v, gew, gcw)
        r2w_nn = fit_metrics(v, Gw, aw_nn)[0]["r2"]
        rhow.append(r2w_nn)
    out["r2_ls_wrong"] = torch.stack(r2w).mean(0).numpy()
    out["r2_nnls_wrong"] = torch.stack(rhow).mean(0).numpy()
    out["cos_gsum_wrong"] = torch.stack(cw).mean(0).numpy()
    G3 = torch.stack([ge, gc, gpix], 2)
    out["r2_ls_e_c_pix"] = fit_metrics(v, G3, lstsq(v, G3))[0]["r2"].numpy()

    # --- post-hoc: spatial co-localisation (sign-agnostic). Does gradient MASS sit where the maps / forces are,
    # regardless of direction? Baseline: the same maps from a different image (index shifted by one).
    av = v.abs()
    shift = torch.roll(ar, 1)
    pcf, gmf = _flat(fields["pc"]), _flat(fields["gm"])
    out["coloc_absv_pc"] = cos(av, pcf).numpy()
    out["coloc_absv_gm"] = cos(av, gmf).numpy()
    out["coloc_absv_absgsum"] = cos(av, gsum.abs()).numpy()
    out["coloc_absv_pc_shuffled"] = cos(av, pcf[shift]).numpy()
    out["coloc_absv_gm_shuffled"] = cos(av, gmf[shift]).numpy()
    out["coloc_absv_const"] = cos(av, torch.ones_like(av)).numpy()  # = gradient spread (1/sqrt(N) .. 1)

    # --- local scaling beta (section 11): v ~ beta * g
    out["beta"] = ((v * gsum).sum(1) / ((gsum * gsum).sum(1) + EPS)).numpy()
    out["beta_e"] = ((v * ge).sum(1) / ((ge * ge).sum(1) + EPS)).numpy()
    out["beta_c"] = ((v * gc).sum(1) / ((gc * gc).sum(1) + EPS)).numpy()
    out["E_edge"] = fields["E_edge"][ar, y].numpy()
    out["E_contrast"] = fields["E_contrast"][ar, y].numpy()
    out["E_total"] = out["E_edge"] + out["E_contrast"]

    df = pd.DataFrame(out)
    if metas is not None:
        for k in ["contrast", "size", "noise_std", "texture_strength", "foreground_intensity", "background_intensity"]:
            df[k] = [m[k] for m in metas]
    tensors = {"v": v_img, "vhat_ls": vhat_ls.float().view_as(v_img), "vhat_nnls": vhat_nn.float().view_as(v_img),
               "ge": fields["ge"][ar, y], "gc": fields["gc"][ar, y]}
    return df, tensors


SUMMARY_COLS = ["ls_r2", "nnls_r2", "ls_cos", "nnls_cos", "ls_rres", "nnls_rres", "ls_rho_e", "nnls_rho_e",
                "cos_ge", "cos_gc", "cos_gsum", "cos_gpix", "cos_grand", "cos_gsum_signscr", "cos_gsum_wrong",
                "r2_ls_random2", "r2_ls_signscr", "r2_ls_wrong", "r2_nnls_wrong", "r2_pix", "r2_gsum",
                "r2_ls_e_c_pix", "coloc_absv_pc", "coloc_absv_gm", "coloc_absv_absgsum", "coloc_absv_pc_shuffled",
                "coloc_absv_gm_shuffled", "coloc_absv_const"]


def summarise(df, n_boot, seed):
    from analysis.stats import bootstrap_ci
    rows = []
    for (model, split), d in df.groupby(["model", "split"], sort=False):
        r = {"model": model, "split": split, "n": len(d)}
        for c in SUMMARY_COLS:
            est, lo, hi = bootstrap_ci(d[c].values, np.mean, n_boot, seed)
            r[c], r[c + "_lo"], r[c + "_hi"] = est, lo, hi
            r[c + "_median"] = float(np.nanmedian(d[c].values))
        r["nnls_both_zero_frac"] = float(((d.nnls_a_e == 0) & (d.nnls_a_c == 0)).mean())
        rows.append(r)
    return pd.DataFrame(rows)
