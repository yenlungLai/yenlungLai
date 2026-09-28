"""Publication figures. Model colours are fixed per entity (never by rank): CE blue, AT orange, PAG aqua."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_COLORS = {"CE": "#2a78d6", "AT": "#eb6834", "PAG": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = "RdBu_r"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sym(ax, img, title):
    m = np.abs(img).max() + 1e-12
    ax.imshow(img, cmap=DIVERGING, vmin=-m, vmax=m)
    ax.set_title(title)
    ax.axis("off")


# -------------------------------------------------------------------------------- stage 1
def fig_energy_validation(summary, path):
    order = list(summary.transformation)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (col, title) in zip(axes[:2], [("rel", "Relative change  ||ΔM|| / ||M(x)||"),
                                            ("floor", "Change in noise-floor units")]):
        for k, (op, c) in enumerate([("PC", "#4a3aa7"), ("GM", "#eda100")]):
            v = summary[f"{col}_{op}"].values
            lo, hi = summary[f"{col}_{op}_lo"].values, summary[f"{col}_{op}_hi"].values
            ax.barh(y + (k - 0.5) * 0.38, v, height=0.36, color=c, label=op,
                    xerr=[v - lo, hi - v], error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.set_title(title)
        ax.set_xscale("log")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    s = summary.selectivity_log2.values
    axes[2].barh(y, s, color=["#4a3aa7" if v > 0 else "#eda100" for v in s], height=0.6)
    axes[2].axvline(0, color=INK2, lw=0.8)
    axes[2].set_title("Selectivity log2(rel PC / rel GM)\n>0: PC responds more,  <0: GM responds more")
    fig.suptitle("Stage 1 - which operator responds to which physical change (medium level, 95% bootstrap CI)",
                 fontsize=10, y=1.06)
    _save(fig, path)


def fig_prototypes(energy, examples, path):
    """examples: tensor [k,1,H,W] of sample images."""
    import torch
    with torch.no_grad():
        pcs, gms = energy.pc(examples), energy.gm(examples)
    n = energy.mu_pc.shape[0]
    fig, ax = plt.subplots(3, n + len(examples), figsize=(1.6 * (n + len(examples)), 5))
    names = ["circle", "square", "triangle"]
    for c in range(n):
        ax[0, c].imshow(energy.mu_pix[c, 0], cmap="gray", vmin=0, vmax=1)
        ax[0, c].set_title(f"r_y: {names[c]}")
        ax[1, c].imshow(energy.mu_pc[c, 0], cmap="magma")
        ax[1, c].set_title("mu_PC,y")
        ax[2, c].imshow(energy.mu_gm[c, 0], cmap="viridis")
        ax[2, c].set_title("mu_GM,y")
    for j in range(len(examples)):
        ax[0, n + j].imshow(examples[j, 0], cmap="gray", vmin=0, vmax=1)
        ax[0, n + j].set_title("sample")
        ax[1, n + j].imshow(pcs[j, 0], cmap="magma")
        ax[1, n + j].set_title("PC(x)")
        ax[2, n + j].imshow(gms[j, 0], cmap="viridis")
        ax[2, n + j].set_title("GM(x)")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle("Class prototypes (centred/aligned prototype split) and example PC / GM maps", fontsize=10)
    _save(fig, path)


# -------------------------------------------------------------------------------- figure 1
def fig_decomposition_examples(model_name, x, pc, gm, tensors, idx, df, path):
    cols = ["image", "PC(x)", "GM(x)", "∇x f_y (v)", "edge force g_e", "contrast force g_c",
            "reconstruction v̂ (LS)", "residual v - v̂"]
    fig, ax = plt.subplots(len(idx), len(cols), figsize=(1.55 * len(cols), 1.7 * len(idx)))
    for r, i in enumerate(idx):
        v = tensors["v"][i, 0].numpy()
        vh = tensors["vhat_ls"][i, 0].numpy()
        ax[r, 0].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[r, 0].axis("off")
        ax[r, 1].imshow(pc[i, 0], cmap="magma")
        ax[r, 1].axis("off")
        ax[r, 2].imshow(gm[i, 0], cmap="viridis")
        ax[r, 2].axis("off")
        _sym(ax[r, 3], v, "")
        _sym(ax[r, 4], tensors["ge"][i, 0].numpy(), "")
        _sym(ax[r, 5], tensors["gc"][i, 0].numpy(), "")
        m = np.abs(v).max()
        ax[r, 6].imshow(vh, cmap=DIVERGING, vmin=-m, vmax=m)
        ax[r, 6].axis("off")
        ax[r, 7].imshow(v - vh, cmap=DIVERGING, vmin=-m, vmax=m)
        ax[r, 7].axis("off")
        row = df.iloc[i]
        ax[r, 0].text(-4, 32, f"R²={row.ls_r2:.2f}\nρe={row.ls_rho_e:.2f}", ha="right", va="center", fontsize=7.5,
                      color=INK)
    for c, t in enumerate(cols):
        ax[0, c].set_title(t, fontsize=8)
    fig.suptitle(f"Figure 1 - force decomposition, {model_name} "
                 "(v, v̂ and residual share one symmetric colour scale; forces scaled individually)", fontsize=9.5)
    _save(fig, path)


# -------------------------------------------------------------------------------- figures 2, 3
def _violin(ax, data, labels, colors):
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for b, c in zip(parts["bodies"], colors):
        b.set_facecolor(c)
        b.set_edgecolor(c)
        b.set_alpha(0.35)
    parts["cmedians"].set_color(INK)
    ax.boxplot(data, widths=0.12, showfliers=False, medianprops={"color": INK},
               boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    ax.set_xticks(range(1, len(labels) + 1), labels)


def fig_r2(df, path, split="test_aligned"):
    d = df[df.split == split]
    models = list(dict.fromkeys(d.model))
    cols = [c for m in models for c in [MODEL_COLORS[m]]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (col, title) in zip(axes, [("ls_r2", "R² force (unconstrained LS)"), ("nnls_r2", "R² force (NNLS)"),
                                       ("r2_ls_wrong", "Control: R² with WRONG-class prototype (LS)")]):
        _violin(ax, [d[d.model == m][col].values for m in models], models, cols)
        ax.set_title(title)
    rnd = d.r2_ls_random2.mean()
    for ax in axes:
        ax.axhline(rnd, color=INK2, ls=":", lw=1)
    axes[0].text(0.55, rnd, " random 2-direction control", va="bottom", fontsize=7, color=INK2)
    fig.suptitle(f"Figure 2 - explained gradient energy R²_force per model ({split})", fontsize=10)
    _save(fig, path)


def fig_contributions(df, path, split="test_aligned"):
    d = df[df.split == split]
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _violin(axes[0], [d[d.model == m].ls_rho_e.dropna().values for m in models], models,
            [MODEL_COLORS[m] for m in models])
    axes[0].set_title("ρ_edge (LS);  ρ_contrast = 1 - ρ_edge")
    _violin(axes[1], [d[d.model == m].nnls_rho_e.dropna().values for m in models], models,
            [MODEL_COLORS[m] for m in models])
    axes[1].set_title("ρ_edge (NNLS; samples with a_e=a_c=0 excluded)")
    for m in models:
        dm = d[d.model == m]
        axes[2].scatter(dm.cos_ge, dm.cos_gc, s=9, alpha=0.5, color=MODEL_COLORS[m], label=m, linewidths=0)
    axes[2].axhline(0, color=INK2, lw=0.8)
    axes[2].axvline(0, color=INK2, lw=0.8)
    axes[2].set_xlabel("cos(v, g_e)")
    axes[2].set_ylabel("cos(v, g_c)")
    axes[2].set_title("Zero-parameter alignment with each force")
    axes[2].legend(markerscale=2)
    fig.suptitle(f"Figure 3 - edge vs contrast contribution ({split})", fontsize=10)
    _save(fig, path)


def fig_controls(summary, path, split="test_aligned"):
    d = summary[summary.split == split]
    keys = [("cos_ge", "g_e"), ("cos_gc", "g_c"), ("cos_gsum", "g_e+g_c"), ("cos_gpix", "g_pixel"),
            ("cos_gsum_wrong", "wrong-class\ng_e+g_c"), ("cos_gsum_signscr", "sign-scrambled\ng_e+g_c"),
            ("cos_grand", "random\n(norm-matched)")]
    models = list(d.model)
    x = np.arange(len(keys))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for k, m in enumerate(models):
        r = d[d.model == m].iloc[0]
        v = np.array([r[c] for c, _ in keys])
        lo = np.array([r[c + "_lo"] for c, _ in keys])
        hi = np.array([r[c + "_hi"] for c, _ in keys])
        ax.bar(x + (k - (len(models) - 1) / 2) * w, v, width=w * 0.92, color=MODEL_COLORS[m], label=m,
               yerr=[v - lo, hi - v], error_kw={"elinewidth": 0.8, "ecolor": INK2})
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [k for _, k in keys])
    ax.set_ylabel("mean cos(∇x f_y, g)")
    ax.legend()
    ax.set_title(f"Controls - zero-parameter cosine between classifier gradient and candidate forces ({split}, 95% CI)")
    _save(fig, path)


# -------------------------------------------------------------------------------- figure 4
def fig_potential(traj_df, pooled, path):
    from data.transformations import TRAJECTORY_FACTORS
    models = list(dict.fromkeys(traj_df.model))
    fig, axes = plt.subplots(2, len(TRAJECTORY_FACTORS), figsize=(2.3 * len(TRAJECTORY_FACTORS), 5.2))
    for j, fac in enumerate(TRAJECTORY_FACTORS):
        d = traj_df[traj_df.factor == fac]
        # z-score within each (model, base) trajectory so logits and energies share one axis
        for m in models:
            dm = d[d.model == m].copy()
            g = dm.groupby("base_index")
            dm["zf"] = (dm.f_y - g.f_y.transform("mean")) / (g.f_y.transform("std") + 1e-9)
            mean = dm.groupby("t").zf.mean()
            axes[0, j].plot(mean.index, mean.values, color=MODEL_COLORS[m], label=f"f_y {m}")
        de = d[d.model == models[0]].copy()
        g = de.groupby("base_index")
        de["zE"] = -(de.E_total - g.E_total.transform("mean")) / (g.E_total.transform("std") + 1e-9)
        me = de.groupby("t").zE.mean()
        axes[0, j].plot(me.index, me.values, color=INK, ls="--", lw=1.5, label="-E_y")
        axes[0, j].set_title(fac)
        if j == 0:
            axes[0, j].set_ylabel("z-scored within trajectory")
        for m in models:
            p = pooled[(pooled.model == m) & (pooled.factor == fac)]
            if len(p):
                axes[1, j].bar(models.index(m), p.corr_df_negdE.iloc[0], color=MODEL_COLORS[m],
                               yerr=[[p.corr_df_negdE.iloc[0] - p.corr_df_negdE_lo.iloc[0]],
                                     [p.corr_df_negdE_hi.iloc[0] - p.corr_df_negdE.iloc[0]]],
                               error_kw={"elinewidth": 0.8, "ecolor": INK2})
        axes[1, j].set_xticks(range(len(models)), models)
        axes[1, j].axhline(0, color=INK2, lw=0.8)
        axes[1, j].set_ylim(-1, 1)
        if j == 0:
            axes[1, j].set_ylabel("corr(Δf_y, -ΔE_y)")
        axes[0, j].set_xlabel("t")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Figure 4 - potential consistency: logit vs negative energy along one-factor trajectories "
                 "(top: mean z-scored curves; bottom: pooled step-wise correlation, 95% grouped-bootstrap CI)",
                 fontsize=9.5)
    _save(fig, path)


def fig_delta_scatter(traj_df, path):
    from analysis.potential_consistency import _deltas
    models = list(dict.fromkeys(traj_df.model))
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.3), sharey=False)
    axes = np.atleast_1d(axes)
    factors = list(dict.fromkeys(traj_df.factor))
    fc = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for ax, m in zip(axes, models):
        dd = _deltas(traj_df[traj_df.model == m])
        for k, f in enumerate(factors):
            s = dd[dd.factor == f]
            ax.scatter(-s.d_E_total, s.d_f_y, s=5, alpha=0.45, color=fc[k % len(fc)], label=f, linewidths=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(m)
        ax.set_xlabel("-ΔE_y (per step)")
        ax.set_ylabel("Δf_y (per step)")
    axes[0].legend(fontsize=7, markerscale=2)
    fig.suptitle("Figure 4b - step-wise logit change vs negative energy change, coloured by factor", fontsize=10)
    _save(fig, path)


# -------------------------------------------------------------------------------- figure 5
def fig_trajectories(flow_df, path):
    models = list(dict.fromkeys(flow_df.model))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for m in models:
        d = flow_df[flow_df.model == m].groupby("step")
        for ax, col in zip(axes, ["cos_on_classifier_path", "cos_on_energy_path", "norm_dist"]):
            mu, sd = d[col].mean(), d[col].std()
            ax.plot(mu.index, mu.values, color=MODEL_COLORS[m], label=m)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=MODEL_COLORS[m], alpha=0.12, linewidth=0)
    axes[0].set_title("cos(∇f_y, -∇E_y) along classifier-ascent path")
    axes[1].set_title("cos(∇f_y, -∇E_y) along energy-descent path")
    axes[2].set_title("||x^f_t - x^E_t|| / (η t)\n0 identical, √2 orthogonal, 2 opposite")
    axes[2].axhline(np.sqrt(2), color=INK2, ls=":", lw=1)
    for ax in axes:
        ax.set_xlabel("step t")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[0].legend()
    fig.suptitle("Figure 5 - classifier gradient flow vs perceptual-energy descent (mean ± sd over images)",
                 fontsize=10)
    _save(fig, path)


def fig_image_sequences(paths_f, path_e, model_names, path, every=8, image=0):
    """paths_f: dict model -> [B,T+1,1,H,W]; path_e: [B,T+1,1,H,W]."""
    T1 = path_e.shape[1]
    steps = list(range(0, T1, every))
    rows = [("energy descent", path_e)] + [(f"{m} ascent", paths_f[m]) for m in model_names]
    fig, ax = plt.subplots(len(rows), len(steps) + 1, figsize=(1.35 * (len(steps) + 1), 1.45 * len(rows)))
    x0 = path_e[image, 0, 0].numpy()
    for r, (name, p) in enumerate(rows):
        for c, t in enumerate(steps):
            ax[r, c].imshow(p[image, t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(f"t={t}", fontsize=8)
        diff = p[image, -1, 0].numpy() - x0
        m = np.abs(diff).max() + 1e-12
        ax[r, -1].imshow(diff, cmap=DIVERGING, vmin=-m, vmax=m)
        ax[r, -1].axis("off")
        ax[r, 0].text(-6, 32, name, ha="right", va="center", fontsize=8)
    ax[0, -1].set_title("x_T - x_0", fontsize=8)
    fig.suptitle("Figure 5b - image sequences along each flow", fontsize=10)
    _save(fig, path)


# -------------------------------------------------------------------------------- figure 6
def fig_attacks(att_df, path, eps=None):
    d = att_df if eps is None else att_df[np.isclose(att_df.eps, eps)]
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(3.7 * len(models), 3.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        dm = d[(d.model == m) & (d.clean_correct == 1)]
        ax.scatter(dm.rel_dE_edge_rand, dm.rel_dE_contrast_rand, s=7, color="#c3c2b7", alpha=0.6,
                   label="random ±ε", linewidths=0)
        f = dm[dm.success == 0]
        s = dm[dm.success == 1]
        ax.scatter(f.rel_dE_edge, f.rel_dE_contrast, s=8, facecolors="none", edgecolors=MODEL_COLORS[m],
                   alpha=0.6, label="PGD failed", linewidths=0.6)
        ax.scatter(s.rel_dE_edge, s.rel_dE_contrast, s=8, color=MODEL_COLORS[m], alpha=0.8,
                   label="PGD success", linewidths=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_title(f"{m}  (success {s.shape[0]}/{dm.shape[0]})")
        ax.set_xlabel("ΔE_edge / E_edge")
    axes[0].set_ylabel("ΔE_contrast / E_contrast")
    axes[0].legend(fontsize=7, markerscale=1.5)
    e = "" if eps is None else f", ε={eps * 255:.0f}/255"
    fig.suptitle(f"Figure 6 - attack decomposition into energy changes (true-class energies{e})", fontsize=10)
    _save(fig, path)
