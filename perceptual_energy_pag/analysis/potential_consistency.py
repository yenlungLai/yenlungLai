"""Potential-energy consistency (section 10) and local scaling beta (section 11).

If grad f_y ~ -beta grad E_y then along a path x(t):  Delta f_y ~ -beta Delta E_y.
One-factor trajectories x(t) (shape morph, contrast, brightness, texture, position, rotation) are generated from
the generative factors; for each we compute f_y(x(t)) for every model and E_y(x(t)) = E_edge + E_contrast
(true class y of the base image; for the shape morph y stays the source class).

Statistics:
    per-trajectory  corr_t( f_y(x(t)), -E_y(x(t)) )           (11 points; inflated by shared monotone trends)
    pooled          corr( Delta f_y, -Delta E_y )             (consecutive steps, pooled over trajectories and
                                                               factors; the headline number, grouped bootstrap
                                                               over base images)
beta:
    beta(x) = v^T g / (||g||^2 + eps),  g = -grad E_y  (and separately for g_e, g_c)
    across-sample regression per class  f_y = -beta_y E_y + C_y   (R^2 reported; only meaningful if beta stable)
"""
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

from analysis.stats import pearson
from data.transformations import TRAJECTORY_FACTORS, trajectory


def build_trajectories(metas, labels, energy, n_per_class, n_steps, image_size=64):
    """Model-independent part: images + energies along each trajectory."""
    chosen = []
    for c in range(3):
        idx = [i for i in range(len(metas)) if labels[i] == c][:n_per_class]
        chosen += idx
    rows, imgs = [], []
    for i in chosen:
        m = metas[i]
        for fac in TRAJECTORY_FACTORS:
            if fac == "rotation" and m["class"] == "circle":
                continue
            ts, xs = trajectory(m, fac, n_steps, image_size)
            for k, (t, xi) in enumerate(zip(ts, xs)):
                rows.append({"base_index": i, "label": int(labels[i]), "class": m["class"], "factor": fac,
                             "step": k, "t": float(t)})
                imgs.append(xi)
    df = pd.DataFrame(rows)
    x = torch.tensor(np.stack(imgs))[:, None]
    y = torch.tensor(df.label.values)
    ee, ec = [], []
    with torch.no_grad():
        for s in range(0, len(x), 256):
            a, b = energy.energies(x[s:s + 256], y[s:s + 256])
            ee.append(a)
            ec.append(b)
    df["E_edge"] = torch.cat(ee).numpy()
    df["E_contrast"] = torch.cat(ec).numpy()
    df["E_total"] = df.E_edge + df.E_contrast
    return df, x, y


@torch.no_grad()
def add_model_logits(df, x, y, model, name):
    out = []
    for s in range(0, len(x), 500):
        out.append(model(x[s:s + 500]).gather(1, y[s:s + 500, None])[:, 0])
    d = df.copy()
    d["model"] = name
    d["f_y"] = torch.cat(out).numpy()
    return d


def _deltas(d):
    d = d.sort_values(["base_index", "factor", "step"])
    g = d.groupby(["base_index", "factor"], sort=False)
    out = d.copy()
    for c in ["f_y", "E_total", "E_edge", "E_contrast"]:
        out["d_" + c] = g[c].diff()
    return out.dropna(subset=["d_f_y"])


def _suff(sub, xcol, ycol, bases):
    """Per-base sufficient statistics [n_bases, 6] for Pearson corr(x, y): n, Sx, Sy, Sxx, Syy, Sxy."""
    out = np.zeros((len(bases), 6))
    pos = {b: i for i, b in enumerate(bases)}
    xs, ys, bs = sub[xcol].values, sub[ycol].values, sub.base_index.map(pos).values
    for j, arr in enumerate([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys]):
        out[:, j] = np.bincount(bs, weights=arr, minlength=len(bases))
    return out


def _corr_from_suff(S):
    n, sx, sy, sxx, syy, sxy = [S[..., j] for j in range(6)]
    den = np.sqrt(np.maximum(n * sxx - sx ** 2, 0) * np.maximum(n * syy - sy ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n * sxy - sx * sy) / den


def _boot_weights(n_groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot)


def _ci(est, boot):
    boot = boot[np.isfinite(boot)]
    return float(est), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def consistency_stats(traj_df, n_boot=2000, seed=0):
    """Returns (per-trajectory table, pooled summary table). Pooled corr uses a grouped (base-image) bootstrap."""
    per_traj = []
    for (model, fac, bi), d in traj_df.groupby(["model", "factor", "base_index"], sort=False):
        per_traj.append({"model": model, "factor": fac, "base_index": bi, "class": d["class"].iloc[0],
                         "corr_f_negE": pearson(d.f_y, -d.E_total),
                         "corr_f_negEedge": pearson(d.f_y, -d.E_edge),
                         "corr_f_negEcontrast": pearson(d.f_y, -d.E_contrast),
                         "f_range": d.f_y.max() - d.f_y.min(), "E_range": d.E_total.max() - d.E_total.min()})
    per_traj = pd.DataFrame(per_traj)

    rows = []
    for model, dm in traj_df.groupby("model", sort=False):
        dd = _deltas(dm)
        for c in ["d_E_total", "d_E_edge", "d_E_contrast"]:
            dd["neg_" + c] = -dd[c]
        for fac in ["ALL"] + TRAJECTORY_FACTORS:
            sub = dd if fac == "ALL" else dd[dd.factor == fac]
            if len(sub) == 0:
                continue
            bases = np.sort(sub.base_index.unique())
            W = _boot_weights(len(bases), n_boot, seed)
            r = {"model": model, "factor": fac, "n_deltas": len(sub)}
            for col, key in [("d_E_total", "corr_df_negdE"), ("d_E_edge", "corr_df_negdEedge"),
                             ("d_E_contrast", "corr_df_negdEcontrast")]:
                S = _suff(sub, "d_f_y", "neg_" + col, bases)
                r[key], r[key + "_lo"], r[key + "_hi"] = _ci(_corr_from_suff(S.sum(0)), _corr_from_suff(W @ S))
            r["spearman_df_negdE"] = float(sps.spearmanr(sub.d_f_y, -sub.d_E_total).correlation)
            pt = per_traj[(per_traj.model == model) & ((per_traj.factor == fac) if fac != "ALL" else True)]
            r["median_traj_corr_f_negE"] = float(np.nanmedian(pt.corr_f_negE))
            r["mean_traj_corr_f_negE"] = float(np.nanmean(pt.corr_f_negE))
            rows.append(r)
    return per_traj, pd.DataFrame(rows)


def paired_consistency_difference(traj_df, model_a, model_b, n_boot=2000, seed=0):
    """Grouped-bootstrap CI for corr_A(df,-dE) - corr_B(df,-dE), resampling base images jointly (paired)."""
    da = _deltas(traj_df[traj_df.model == model_a])
    db = _deltas(traj_df[traj_df.model == model_b])
    for d in (da, db):
        d["neg_d_E_total"] = -d["d_E_total"]
    rows = []
    for fac in ["ALL"] + TRAJECTORY_FACTORS:
        sa = da if fac == "ALL" else da[da.factor == fac]
        sb = db if fac == "ALL" else db[db.factor == fac]
        if len(sa) == 0:
            continue
        bases = np.sort(sa.base_index.unique())
        W = _boot_weights(len(bases), n_boot, seed)
        Sa, Sb = _suff(sa, "d_f_y", "neg_d_E_total", bases), _suff(sb, "d_f_y", "neg_d_E_total", bases)
        est = _corr_from_suff(Sa.sum(0)) - _corr_from_suff(Sb.sum(0))
        e, lo, hi = _ci(est, _corr_from_suff(W @ Sa) - _corr_from_suff(W @ Sb))
        rows.append({"comparison": f"{model_a} - {model_b}", "factor": fac, "diff_corr": e, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def beta_statistics(per_sample):
    """per_sample: decomposition table (aligned split) with beta, logit_y, E_total columns."""
    rows = []
    for (model, label), d in per_sample.groupby(["model", "label"], sort=False):
        rows.append(_beta_row(model, str(label), d))
    for model, d in per_sample.groupby("model", sort=False):
        rows.append(_beta_row(model, "all", d))
    return pd.DataFrame(rows)


def _beta_row(model, label, d):
    b = d.beta.values
    lr = sps.linregress(d.E_total.values, d.logit_y.values)  # f = slope E + C ; slope = -beta_y
    return {"model": model, "class": label, "n": len(d),
            "beta_mean": b.mean(), "beta_median": np.median(b), "beta_std": b.std(),
            "beta_cv": b.std() / (abs(b.mean()) + 1e-12), "beta_frac_positive": (b > 0).mean(),
            "beta_iqr_over_median": (np.quantile(b, .75) - np.quantile(b, .25)) / (abs(np.median(b)) + 1e-12),
            "beta_e_mean": d.beta_e.mean(), "beta_e_cv": d.beta_e.std() / (abs(d.beta_e.mean()) + 1e-12),
            "beta_c_mean": d.beta_c.mean(), "beta_c_cv": d.beta_c.std() / (abs(d.beta_c.mean()) + 1e-12),
            "reg_beta": -lr.slope, "reg_C": lr.intercept, "reg_R2": lr.rvalue ** 2, "reg_p": lr.pvalue}
