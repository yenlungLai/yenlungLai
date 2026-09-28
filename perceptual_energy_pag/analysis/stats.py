"""Bootstrap confidence intervals and paired tests."""
import numpy as np
from scipy import stats


def bootstrap_ci(values, fn=np.mean, n=10000, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    boot = fn(v[idx], axis=1) if fn in (np.mean, np.median) else np.array([fn(v[i]) for i in idx])
    return float(fn(v)), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def paired_comparison(a, b, n=10000, seed=0, label_a="A", label_b="B"):
    """Paired comparison of per-sample metric a vs b (same samples). Tests a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    mean, lo, hi = bootstrap_ci(d, np.mean, n, seed)
    med, mlo, mhi = bootstrap_ci(d, np.median, n, seed + 1)
    try:
        w = stats.wilcoxon(a, b, alternative="greater")
        w_p = float(w.pvalue)
    except ValueError:
        w_p = np.nan
    t = stats.ttest_rel(a, b, alternative="greater")
    return {"comparison": f"{label_a} - {label_b}", "n": int(len(d)),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": mean, "mean_diff_ci_lo": lo, "mean_diff_ci_hi": hi,
            "median_diff": med, "median_diff_ci_lo": mlo, "median_diff_ci_hi": mhi,
            "frac_a_gt_b": float((d > 0).mean()),
            "wilcoxon_p_greater": w_p, "paired_t_p_greater": float(t.pvalue)}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def grouped_bootstrap(stat_fn, groups, n=2000, seed=0, alpha=0.05):
    """Resample whole groups (e.g. base images) with replacement. stat_fn(list_of_group_ids) -> float."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    est = stat_fn(groups)
    boot = np.array([stat_fn(groups[rng.integers(0, len(groups), len(groups))]) for _ in range(n)])
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return est, np.nan, np.nan
    return float(est), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def fmt_ci(est, lo, hi, p=3):
    return f"{est:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"
