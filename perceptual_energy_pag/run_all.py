"""Staged experiment runner.

    python run_all.py --stage 1          # dataset + PC/GM validation only
    python run_all.py --stage 2          # train CE + AT, force decomposition
    python run_all.py --stage 3          # potential consistency, beta, gradient-flow trajectories
    python run_all.py --stage 4          # PAG training (+ re-run stage 2/3 analyses with PAG included)
    python run_all.py --stage 5          # adversarial energy decomposition, stronger attacks
    python run_all.py --stage all
    add --quick for a tiny smoke test (writes to results_quick/)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import config as C  # noqa: E402
from analysis import adversarial_analysis as AA  # noqa: E402
from analysis import energy_validation as EV  # noqa: E402
from analysis import force_decomposition as FD  # noqa: E402
from analysis import potential_consistency as PCons  # noqa: E402
from analysis import trajectory_analysis as TA  # noqa: E402
from analysis.stats import fmt_ci, paired_comparison  # noqa: E402
from data.synthetic_dataset import clip_fraction, load_or_generate  # noqa: E402
from perceptual.energies import PerceptualEnergy  # noqa: E402
from perceptual.gradient_magnitude import GradientMagnitude  # noqa: E402
from perceptual.phase_congruency import PhaseCongruency  # noqa: E402
from training.common import accuracy, load_checkpoint, robust_accuracy, save_checkpoint  # noqa: E402
from visualization import plots as P  # noqa: E402

MODELS = ["CE", "AT", "PAG"]


class Ctx:
    """Everything the stages share: config, data tensors, energy, output paths."""

    def __init__(self, quick):
        self.cfg = C.quick_config() if quick else C.Config()
        torch.set_num_threads(self.cfg.num_threads)
        torch.manual_seed(self.cfg.global_seed)
        np.random.seed(self.cfg.global_seed)
        self.out = C.RESULTS + ("_quick" if quick else "")
        self.fig = os.path.join(self.out, "figures")
        self.ckpt = os.path.join(self.out, "checkpoints")
        for d in (self.out, self.fig, self.ckpt):
            os.makedirs(d, exist_ok=True)
        self.cfg.save(os.path.join(self.out, "config.json"))
        t = time.time()
        self.data = load_or_generate(self.cfg.data, tag="quick" if quick else "full")
        self.t = {k: (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in self.data.items()}
        self.metas = {k: v[2] for k, v in self.data.items()}
        print(f"data ready ({time.time() - t:.1f}s): " + ", ".join(f"{k}={len(v[1])}" for k, v in self.data.items()))
        self.energy = PerceptualEnergy(PhaseCongruency(self.cfg.data.image_size, self.cfg.pc),
                                       GradientMagnitude(self.cfg.gm))
        self.energy.fit_prototypes(*self.t["proto"])
        self.energy.eval()

    def path(self, name):
        return os.path.join(self.out, name)

    def summary(self, stage, lines):
        text = "\n".join(lines)
        print("\n" + "=" * 100 + f"\nSTAGE {stage} SUMMARY\n" + "=" * 100 + "\n" + text + "\n")
        with open(self.path(f"stage{stage}_summary.md"), "w") as f:
            f.write(f"# Stage {stage} summary (auto-generated numbers)\n\n" + text + "\n")

    def available_models(self):
        return [m for m in MODELS if os.path.exists(os.path.join(self.ckpt, f"{m}.pt"))]

    def load(self, m):
        return load_checkpoint(os.path.join(self.ckpt, f"{m}.pt"))[0]


# ============================================================================================ STAGE 1
def stage1(ctx):
    cfg = ctx.cfg
    lines = []
    all_meta = pd.DataFrame([m for k in ctx.metas for m in ctx.metas[k]])
    all_meta.to_csv(ctx.path("dataset_metadata.csv"), index=False)
    for k, (x, y) in ctx.t.items():
        lines.append(f"- split `{k}`: n={len(y)}, class counts={np.bincount(y.numpy()).tolist()}, "
                     f"clipped-pixel fraction={clip_fraction(x.numpy()):.5f}")
    P.fig_prototypes(ctx.energy, ctx.t["test_aligned"][0][:4], os.path.join(ctx.fig, "fig0_prototypes.png"))

    n_base = min(300, len(ctx.t["test_aligned"][1]))
    imgs, labels = ctx.data["test_aligned"][0][:n_base], ctx.data["test_aligned"][1][:n_base]
    t = time.time()
    sens = EV.transformation_sensitivity(ctx.energy, imgs, labels, ctx.metas["test_aligned"][:n_base],
                                         cfg.data.image_size)
    sens.to_csv(ctx.path("transformation_sensitivity_pairs.csv"), index=False)
    summ = EV.summarise_sensitivity(sens, n_boot=min(cfg.eval.bootstrap_resamples, 2000))
    summ.to_csv(ctx.path("transformation_sensitivity.csv"), index=False)
    lev = sens.groupby(["transformation", "level"])[["rel_PC", "rel_GM", "selectivity_log2"]].mean().reset_index()
    lev.to_csv(ctx.path("transformation_sensitivity_by_level.csv"), index=False)
    P.fig_energy_validation(summ, os.path.join(ctx.fig, "fig0_energy_validation.png"))
    print(f"sensitivity computed on {n_base} base images ({time.time() - t:.1f}s)")

    acc = {s: EV.nearest_prototype_accuracy(ctx.energy, ctx.data[s][0], ctx.data[s][1])
           for s in ["test_aligned", "test"]}
    with open(ctx.path("nearest_prototype_accuracy.json"), "w") as f:
        json.dump(acc, f, indent=2)

    lines += ["", f"Transformation sensitivity (medium level; {n_base} base images; mean [95% CI]):", "",
              "| Transformation | ΔPC (rel) | ΔGM (rel) | ΔPC (noise-floor units) | ΔGM (noise-floor units) | "
              "selectivity log2(PC/GM) |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r.transformation} | {fmt_ci(r.rel_PC, r.rel_PC_lo, r.rel_PC_hi)} | "
                     f"{fmt_ci(r.rel_GM, r.rel_GM_lo, r.rel_GM_hi)} | {r.floor_PC:.2f} | {r.floor_GM:.2f} | "
                     f"{fmt_ci(r.selectivity_log2, r.selectivity_log2_lo, r.selectivity_log2_hi, 2)} |")
    g = summ.set_index("transformation")
    lines += ["", "Observations (rule-based from the numbers above):"]
    for tname, expect in [("shape", "PC"), ("contrast", "GM")]:
        s = g.loc[tname, "selectivity_log2"]
        who = "PC" if s > 0 else "GM"
        lines.append(f"- {tname}: relatively larger response from **{who}** (selectivity {s:+.2f}); "
                     f"hypothesised: {expect} -> {'consistent' if who == expect else 'NOT consistent'}.")
    lines.append(f"- brightness offset: rel ΔPC={g.loc['brightness', 'rel_PC']:.4f}, rel ΔGM="
                 f"{g.loc['brightness', 'rel_GM']:.4f}. NOTE both operators are offset-invariant BY CONSTRUCTION "
                 "(zero-DC log-Gabor, zero-sum Sobel); nonzero values come only from clipping.")
    lines.append(f"- brightness gain (pixel x k): rel ΔPC={g.loc['brightness_gain(pixel)', 'rel_PC']:.4f}, rel ΔGM="
                 f"{g.loc['brightness_gain(pixel)', 'rel_GM']:.4f}.")
    lines.append(f"- noise re-sample floor: rel ΔPC={g.loc['noise_resample', 'rel_PC']:.4f}, "
                 f"rel ΔGM={g.loc['noise_resample', 'rel_GM']:.4f}.")
    lines.append(f"- nearest-prototype accuracy (chance 0.333): " + json.dumps(acc))
    ctx.summary(1, lines)


# ============================================================================================ STAGE 2 / 4 training
def train_models(ctx, names, retrain=False):
    from training.train_at import train_at
    from training.train_ce import train_ce
    from training.train_pag import precompute_targets, train_pag
    x, y = ctx.t["train"]
    cfg = ctx.cfg
    for m in names:
        p = os.path.join(ctx.ckpt, f"{m}.pt")
        if os.path.exists(p) and not retrain:
            print(f"{m}: checkpoint exists, skipping training")
            continue
        t = time.time()
        meta = {"name": m, "config": cfg.to_dict()}
        if m == "CE":
            model, hist = train_ce(x, y, cfg)
        elif m == "AT":
            model, hist = train_at(x, y, cfg)
        elif m == "PAG":
            tt = time.time()
            targets = precompute_targets(ctx.energy, x, y)
            print(f"PAG targets precomputed ({time.time() - tt:.1f}s)")
            model, hist = train_pag(x, y, targets, cfg)
        meta["train_time_s"] = round(time.time() - t, 1)
        save_checkpoint(model, hist, p, meta)
        print(f"{m}: trained in {meta['train_time_s']}s")


def evaluate_accuracy(ctx, names):
    e = ctx.cfg.eval
    rows = []
    for m in names:
        model = ctx.load(m)
        r = {"model": m}
        for s in ["test", "test_aligned"]:
            x, y = ctx.t[s]
            r[f"clean_acc_{s}"] = accuracy(model, x, y)
            r[f"pgd{e.pgd_steps}_acc_{s}"] = robust_accuracy(model, x, y, e.pgd_eps, e.pgd_alpha, e.pgd_steps)
        rows.append(r)
        print(f"  {m}: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "model"))
    return pd.DataFrame(rows)


# ============================================================================================ STAGE 2 analysis
def force_analysis(ctx, names):
    cfg = ctx.cfg
    fields = {s: FD.perceptual_fields(ctx.energy, *ctx.t[s], seed=cfg.eval.control_seed)
              for s in ["test_aligned", "test"]}
    dfs, tens = [], {}
    for m in names:
        model = ctx.load(m)
        for s in ["test_aligned", "test"]:
            df, tt = FD.decompose(model, fields[s], *ctx.t[s], m, s, ctx.metas[s])
            dfs.append(df)
            if s == "test_aligned":
                tens[m] = tt
    per = pd.concat(dfs, ignore_index=True)
    per.to_csv(ctx.path("per_sample_metrics.csv"), index=False)
    summ = FD.summarise(per, cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_seed)
    summ.to_csv(ctx.path("force_summary.csv"), index=False)

    comps = []
    for s in ["test_aligned", "test"]:
        d = per[per.split == s]
        for other in [m for m in names if m != "CE"]:
            a, b = d[d.model == other], d[d.model == "CE"]
            for col in ["ls_r2", "nnls_r2", "ls_rho_e", "nnls_rho_e", "ls_cos", "cos_gsum", "cos_ge", "cos_gc",
                        "r2_ls_wrong", "coloc_absv_pc", "coloc_absv_gm", "coloc_absv_absgsum"]:
                r = paired_comparison(a[col].values, b[col].values, cfg.eval.bootstrap_resamples,
                                      cfg.eval.bootstrap_seed, other, "CE")
                comps.append({"split": s, "metric": col, **r})
    comps = pd.DataFrame(comps)
    comps.to_csv(ctx.path("comparisons.csv"), index=False)

    # figures
    x, y = ctx.t["test_aligned"]
    with torch.no_grad():
        pc, gm = ctx.energy.pc(x[:64]), ctx.energy.gm(x[:64])
    idx = [int(np.where(y.numpy() == c)[0][0]) for c in range(3)] + [int(np.where(y.numpy() == 1)[0][1])]
    for m in names:
        dm = per[(per.model == m) & (per.split == "test_aligned")].reset_index(drop=True)
        P.fig_decomposition_examples(m, x, pc, gm, tens[m], idx, dm, os.path.join(ctx.fig, f"fig1_decomposition_{m}.png"))
    P.fig_r2(per, os.path.join(ctx.fig, "fig2_r2_force.png"))
    P.fig_r2(per, os.path.join(ctx.fig, "fig2b_r2_force_randomsplit.png"), split="test")
    P.fig_contributions(per, os.path.join(ctx.fig, "fig3_contributions.png"))
    P.fig_controls(summ, os.path.join(ctx.fig, "fig_controls_cosines.png"))
    return per, summ, comps


def main_table(ctx, acc, summ, extra=None):
    e = ctx.cfg.eval
    rows = []
    for _, a in acc.iterrows():
        s = summ[(summ.model == a.model) & (summ.split == "test_aligned")].iloc[0]
        r = {"Model": a.model, "Clean Acc (test)": a.clean_acc_test, "Clean Acc (aligned)": a.clean_acc_test_aligned,
             f"PGD{e.pgd_steps} Acc eps=8/255 (test)": a[f"pgd{e.pgd_steps}_acc_test"],
             f"PGD{e.pgd_steps} Acc eps=8/255 (aligned)": a[f"pgd{e.pgd_steps}_acc_test_aligned"]}
        for c, lab in [("ls_cos", "Cos(v,vhat) LS"), ("ls_r2", "R2 force LS"), ("nnls_r2", "R2 force NNLS"),
                       ("ls_rho_e", "Edge contribution rho_e LS"), ("nnls_rho_e", "Edge contribution rho_e NNLS"),
                       ("r2_ls_wrong", "R2 LS wrong-proto (control)"), ("r2_ls_random2", "R2 LS random (control)"),
                       ("r2_pix", "R2 pixel force (control)")]:
            r[lab] = s[c]
            r[lab + " CI"] = f"[{s[c + '_lo']:.4f}, {s[c + '_hi']:.4f}]"
        r["Contrast contribution rho_c LS"] = 1 - s["ls_rho_e"]
        rows.append(r)
    t = pd.DataFrame(rows)
    if extra is not None:
        t = t.merge(extra, on="Model", how="left")
    t.to_csv(ctx.path("metrics.csv"), index=False)
    return t


def stage2(ctx, retrain=False, names=("CE", "AT")):
    names = list(names)
    train_models(ctx, names, retrain)
    print("evaluating accuracy / robustness ...")
    acc = evaluate_accuracy(ctx, names)
    acc.to_csv(ctx.path("accuracy.csv"), index=False)
    print("force decomposition ...")
    per, summ, comps = force_analysis(ctx, names)
    table = main_table(ctx, acc, summ)
    lines = ["Main comparison (primary split = test_aligned; means with 95% bootstrap CI):", "",
             "| Model | Clean Acc | PGD Acc (8/255) | Cos(v,v̂) LS | R² force LS | R² force NNLS | ρ_edge LS | "
             "ρ_contrast LS | R² wrong-proto | R² random |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, a in acc.iterrows():
        s = summ[(summ.model == a.model) & (summ.split == "test_aligned")].iloc[0]
        lines.append(f"| {a.model} | {a.clean_acc_test:.3f} | {a[f'pgd{ctx.cfg.eval.pgd_steps}_acc_test']:.3f} | "
                     f"{s.ls_cos:.3f} | {fmt_ci(s.ls_r2, s.ls_r2_lo, s.ls_r2_hi)} | "
                     f"{fmt_ci(s.nnls_r2, s.nnls_r2_lo, s.nnls_r2_hi)} | {fmt_ci(s.ls_rho_e, s.ls_rho_e_lo, s.ls_rho_e_hi)} | "
                     f"{1 - s.ls_rho_e:.3f} | {s.r2_ls_wrong:.4f} | {s.r2_ls_random2:.5f} |")
    lines += ["", "Paired comparisons vs CE (mean difference [95% CI], Wilcoxon one-sided p for >):", ""]
    for _, c in comps.iterrows():
        if c.metric in ("ls_r2", "nnls_r2", "ls_rho_e", "nnls_rho_e", "cos_gsum", "r2_ls_wrong", "coloc_absv_pc"):
            lines.append(f"- [{c.split}] {c.comparison} {c.metric}: {fmt_ci(c.mean_diff, c.mean_diff_ci_lo, c.mean_diff_ci_hi, 4)}"
                         f"  frac>0={c.frac_a_gt_b:.2f}  p={c.wilcoxon_p_greater:.2e}")
    lines += ["", "Zero-parameter cosines (test_aligned):", ""]
    for _, s in summ[summ.split == "test_aligned"].iterrows():
        lines.append(f"- {s.model}: cos(v,g_e)={s.cos_ge:.4f}, cos(v,g_c)={s.cos_gc:.4f}, cos(v,g_e+g_c)={s.cos_gsum:.4f}, "
                     f"cos(v,g_pixel)={s.cos_gpix:.4f}, wrong-proto={s.cos_gsum_wrong:.4f}, "
                     f"sign-scrambled={s.cos_gsum_signscr:.4f}, random={s.cos_grand:.4f}")
    lines += ["", "Post-hoc sign-agnostic co-localisation cos(|v|, map) (test_aligned; shuffled = map of another image):", ""]
    for _, s in summ[summ.split == "test_aligned"].iterrows():
        lines.append(f"- {s.model}: |v|~PC {s.coloc_absv_pc:.3f} (shuffled {s.coloc_absv_pc_shuffled:.3f}), "
                     f"|v|~GM {s.coloc_absv_gm:.3f} (shuffled {s.coloc_absv_gm_shuffled:.3f}), "
                     f"|v|~|g_e+g_c| {s.coloc_absv_absgsum:.3f}, |v|~const {s.coloc_absv_const:.3f}")
    ctx.summary(2 if "PAG" not in names else 4, lines)
    return per, summ, comps, table


# ============================================================================================ STAGE 3
def stage3(ctx, names=None, tag=""):
    cfg = ctx.cfg
    names = names or ctx.available_models()
    print("building one-factor trajectories ...")
    t = time.time()
    base_df, xt, yt = PCons.build_trajectories(ctx.metas["test_aligned"], ctx.data["test_aligned"][1], ctx.energy,
                                               cfg.eval.n_traj_per_class, cfg.eval.traj_steps, cfg.data.image_size)
    print(f"  {len(base_df)} trajectory images ({time.time() - t:.1f}s)")
    traj = pd.concat([PCons.add_model_logits(base_df, xt, yt, ctx.load(m), m) for m in names], ignore_index=True)
    traj.to_csv(ctx.path("potential_trajectories_raw.csv"), index=False)
    per_traj, pooled = PCons.consistency_stats(traj, min(cfg.eval.bootstrap_resamples, 2000), cfg.eval.bootstrap_seed)
    per_traj.to_csv(ctx.path("potential_per_trajectory.csv"), index=False)
    pooled.to_csv(ctx.path("potential_consistency.csv"), index=False)
    diffs = pd.concat([PCons.paired_consistency_difference(traj, m, "CE", min(cfg.eval.bootstrap_resamples, 2000),
                                                           cfg.eval.bootstrap_seed) for m in names if m != "CE"],
                      ignore_index=True) if len(names) > 1 else pd.DataFrame()
    diffs.to_csv(ctx.path("potential_consistency_differences.csv"), index=False)
    P.fig_potential(traj, pooled, os.path.join(ctx.fig, "fig4_potential_consistency.png"))
    P.fig_delta_scatter(traj, os.path.join(ctx.fig, "fig4b_delta_scatter.png"))

    per = pd.read_csv(ctx.path("per_sample_metrics.csv"))
    per = per[(per.split == "test_aligned") & per.model.isin(names)]
    beta = PCons.beta_statistics(per)
    beta.to_csv(ctx.path("beta_statistics.csv"), index=False)

    print("gradient-flow trajectories ...")
    x, y = ctx.t["test_aligned"]
    idx = torch.cat([torch.where(y == c)[0][:cfg.eval.flow_images_per_class] for c in range(3)])
    x0, y0 = x[idx], y[idx]
    pe = TA.energy_descent_path(ctx.energy, x0, y0, cfg.eval.flow_steps, cfg.eval.flow_eta)
    flows, pfs = [], {}
    for m in names:
        model = ctx.load(m)
        pf = TA.classifier_ascent_path(model, x0, y0, cfg.eval.flow_steps, cfg.eval.flow_eta)
        pfs[m] = pf
        flows.append(TA.evaluate_paths(model, ctx.energy, pf, pe, y0, cfg.eval.flow_eta, m))
    flow = pd.concat(flows, ignore_index=True)
    flow.to_csv(ctx.path("trajectory_flow.csv"), index=False)
    np.savez_compressed(ctx.path("trajectory_image_sequences.npz"), energy_path=pe.numpy(), labels=y0.numpy(),
                        **{f"classifier_path_{m}": pfs[m].numpy() for m in names})
    P.fig_trajectories(flow, os.path.join(ctx.fig, "fig5_trajectories.png"))
    for i in range(min(3, len(idx))):
        img = i * cfg.eval.flow_images_per_class
        P.fig_image_sequences(pfs, pe, names, os.path.join(ctx.fig, f"fig5b_image_sequences_{i}.png"),
                              every=max(1, cfg.eval.flow_steps // 5), image=img)

    lines = ["Pooled step-wise correlation corr(Δf_y, -ΔE_y) [95% grouped bootstrap CI over base images]:", "",
             "| Factor | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for fac in pooled.factor.unique():
        cells = []
        for m in names:
            r = pooled[(pooled.model == m) & (pooled.factor == fac)]
            cells.append(fmt_ci(r.corr_df_negdE.iloc[0], r.corr_df_negdE_lo.iloc[0], r.corr_df_negdE_hi.iloc[0])
                         if len(r) else "-")
        lines.append(f"| {fac} | " + " | ".join(cells) + " |")
    lines += ["", "Edge-only / contrast-only energy (factor=ALL): corr(Δf,-ΔE_edge), corr(Δf,-ΔE_contrast):"]
    for m in names:
        r = pooled[(pooled.model == m) & (pooled.factor == "ALL")].iloc[0]
        lines.append(f"- {m}: edge {fmt_ci(r.corr_df_negdEedge, r.corr_df_negdEedge_lo, r.corr_df_negdEedge_hi)}, "
                     f"contrast {fmt_ci(r.corr_df_negdEcontrast, r.corr_df_negdEcontrast_lo, r.corr_df_negdEcontrast_hi)}; "
                     f"median per-trajectory corr(f,-E)={r.median_traj_corr_f_negE:.3f}")
    if len(diffs):
        lines += ["", "Paired differences in pooled corr vs CE:"]
        for _, r in diffs[diffs.factor == "ALL"].iterrows():
            lines.append(f"- {r.comparison}: {fmt_ci(r.diff_corr, r.lo, r.hi)}")
    lines += ["", "Local scaling β (test_aligned, all classes):", "",
              "| Model | mean β | median β | CV(β) | frac β>0 | regression f=-βE+C: β_reg | R² |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in beta[beta["class"] == "all"].iterrows():
        lines.append(f"| {r.model} | {r.beta_mean:.4g} | {r.beta_median:.4g} | {r.beta_cv:.2f} | {r.beta_frac_positive:.2f} | "
                     f"{r.reg_beta:.4g} | {r.reg_R2:.3f} |")
    lines += ["", "Per-class regression R²: " + "; ".join(f"{r.model}/{r['class']}: β_reg={r.reg_beta:.3g}, R²={r.reg_R2:.3f}, CV(β)={r.beta_cv:.2f}"
                                                        for _, r in beta[beta['class'] != 'all'].iterrows())]
    lines += ["", "Gradient-flow (mean over images; final step and average over steps):"]
    for m in names:
        d = flow[flow.model == m]
        last = d[d.step == d.step.max()]
        lines.append(f"- {m}: mean cos on classifier path={d.cos_on_classifier_path.mean():.3f}, on energy path="
                     f"{d.cos_on_energy_path.mean():.3f}; final displacement cosine={last.cos_displacement.mean():.3f} "
                     f"(0 = orthogonal); f_y change along energy-descent path="
                     f"{(last.f_y_energy_path.values - d[d.step == 0].f_y_energy_path.values).mean():.3f}; "
                     f"E change along classifier-ascent path="
                     f"{(last.E_classifier_path.values - d[d.step == 0].E_classifier_path.values).mean():.3f}")
    ctx.summary(f"3{tag}", lines)


# ============================================================================================ STAGE 4
def stage4(ctx, retrain=False):
    comps_path, fs_path = ctx.path("comparisons.csv"), ctx.path("force_summary.csv")
    if os.path.exists(comps_path) and os.path.exists(fs_path):
        c, fs = pd.read_csv(comps_path), pd.read_csv(fs_path)
        c = c[(c.split == "test_aligned") & (c.comparison == "AT - CE")].set_index("metric")
        at = fs[(fs.split == "test_aligned") & (fs.model == "AT")].iloc[0]
        crit = {
            "R²_LS(AT) > R²_LS(CE), CI excludes 0": c.loc["ls_r2", "mean_diff_ci_lo"] > 0,
            "R²_NNLS(AT) > R²_NNLS(CE), CI excludes 0 (forces enter with the hypothesised + sign)":
                c.loc["nnls_r2", "mean_diff_ci_lo"] > 0,
            "cos(v_AT, g_e+g_c) > 0 (gradient descends the energy)": at.cos_gsum_lo > 0,
            "true-class prototype beats wrong-class prototype for AT": at.ls_r2 > at.r2_ls_wrong,
        }
        print("STAGE-4 GATE ('only if results are promising'):")
        for k, v in crit.items():
            print(f"  [{'x' if v else ' '}] {k}")
        print(f"  -> {'PROMISING' if all(crit.values()) else 'NOT PROMISING'}. PAG is trained anyway as a POSITIVE "
              "CONTROL for the measurement pipeline (does the decomposition detect alignment when it is imposed?), "
              "not as discovery evidence.")
    per, summ, comps, table = stage2(ctx, retrain, names=("CE", "AT", "PAG"))
    stage3(ctx, ["CE", "AT", "PAG"], tag="_with_PAG")


# ============================================================================================ STAGE 5
def stage5(ctx):
    cfg = ctx.cfg.eval
    names = ctx.available_models()
    x, y = ctx.t["test_aligned"]
    dfs = []
    configs = [(cfg.pgd_eps, cfg.pgd_steps)] + [(e, cfg.strong_pgd_steps) for e in cfg.strong_pgd_eps]
    for m in names:
        model = ctx.load(m)
        for eps, steps in configs:
            t = time.time()
            dfs.append(AA.attack_energy_changes(model, ctx.energy, x, y, eps, cfg.pgd_alpha if steps == cfg.pgd_steps
                                                else eps / 8, steps, m, seed=cfg.control_seed))
            print(f"  {m} eps={eps * 255:.0f}/255 steps={steps}: {time.time() - t:.1f}s")
    att = pd.concat(dfs, ignore_index=True)
    att.to_csv(ctx.path("attack_energy_changes.csv"), index=False)
    summ = AA.summarise_attacks(att)
    summ.to_csv(ctx.path("attack_summary.csv"), index=False)
    main = att[np.isclose(att.eps, cfg.pgd_eps) & (att.steps == cfg.pgd_steps)]
    P.fig_attacks(main, os.path.join(ctx.fig, "fig6_attack_decomposition.png"))
    P.fig_attacks(att[np.isclose(att.eps, 16 / 255)], os.path.join(ctx.fig, "fig6b_attack_decomposition_eps16.png"))

    # update metrics.csv with strong-attack robust accuracy
    if os.path.exists(ctx.path("metrics.csv")):
        met = pd.read_csv(ctx.path("metrics.csv"))
        for eps in cfg.strong_pgd_eps:
            col = f"PGD{cfg.strong_pgd_steps} Acc eps={eps * 255:.0f}/255 (aligned)"
            met[col] = met.Model.map(summ[np.isclose(summ.eps, eps) & (summ.steps == cfg.strong_pgd_steps)]
                                     .set_index("model").adv_acc)
        met.to_csv(ctx.path("metrics.csv"), index=False)

    lines = ["Attack decomposition (test_aligned, true-class energies; relative change ΔE/E; mean over initially "
             "correct samples; 'succ' = successful attacks only):", "",
             "| Model | ε | steps | success rate | ΔEdge/E (succ) | ΔContrast/E (succ) | ΔEdge/E random | ΔContrast/E random | "
             "edge attack/random | contrast attack/random | cos(δ,-g_e) | cos(δ,-g_c) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summ.iterrows():
        g = lambda k: r.get(f"succ_{k}", np.nan)
        lines.append(f"| {r.model} | {r.eps_255:.0f}/255 | {r.steps} | {r.attack_success_rate:.3f} | {g('rel_dE_edge'):.4f} | "
                     f"{g('rel_dE_contrast'):.4f} | {g('rel_dE_edge_rand'):.4f} | {g('rel_dE_contrast_rand'):.4f} | "
                     f"{g('edge_attack_over_random'):.2f} | {g('contrast_attack_over_random'):.2f} | "
                     f"{g('cos_delta_neg_ge'):.4f} | {g('cos_delta_neg_gc'):.4f} |")
    ctx.summary(5, lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    ctx = Ctx(args.quick)
    stages = ["1", "2", "3", "4", "5"] if args.stage == "all" else args.stage.split(",")
    for s in stages:
        t = time.time()
        {"1": lambda: stage1(ctx), "2": lambda: stage2(ctx, args.retrain), "3": lambda: stage3(ctx, ["CE", "AT"]),
         "4": lambda: stage4(ctx, args.retrain), "5": lambda: stage5(ctx)}[s]()
        print(f"[stage {s} done in {time.time() - t:.1f}s]")


if __name__ == "__main__":
    main()
