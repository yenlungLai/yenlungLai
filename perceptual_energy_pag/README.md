# Are classifier input gradients sums of perceptual "forces"?

A controlled PyTorch experiment testing the hypothesis

    ∇x f_y(x) ≈ a_e(x) g_e(x,y) + a_c(x) g_c(x,y),
    g_e = -∇x E_edge,   E_edge(x,y)     = ½‖PC(x) − μ_PC,y‖²   (phase congruency, "structure")
    g_c = -∇x E_contrast, E_contrast(x,y) = ½‖GM(x) − μ_GM,y‖²  (Sobel gradient magnitude, "contrast")

for a standard cross-entropy CNN (CE), a PGD adversarially trained CNN (AT), and a PAG-style gradient-aligned CNN
(PAG), all with the same architecture, on a synthetic dataset where every generative factor is known.

RESULTS_PLACEHOLDER

## Reproducing

```bash
pip install -r requirements.txt
python run_all.py --stage 1     # dataset + PC/GM validation              (~1 min, CPU)
python run_all.py --stage 2     # train CE + AT, force decomposition      (~1 h, CPU; AT dominates)
python run_all.py --stage 3     # potential consistency, beta, flows      (~5 min)
python run_all.py --stage 4     # PAG training + re-run 2/3 with PAG      (~30 min)
python run_all.py --stage 5     # attack energy decomposition             (~15 min)
python run_all.py --stage all --quick   # 2-minute smoke test of the whole pipeline (results_quick/)
```

Everything is deterministic given `config.py` (dataset seeds 1001–1004, model seed 0, control seed 11,
bootstrap seed 7). The dataset is regenerated from seeds into `data_cache/` (git-ignored); every sample's
generative factors are saved in `results/dataset_metadata.csv`. `results/config.json` is the full configuration
of the run that produced the committed results (architecture, optimiser, learning rate, epochs, attack
parameters, PAG λ, PC parameters, preprocessing).

## What was built

| Component | File | Notes |
|---|---|---|
| Synthetic dataset | `data/synthetic_dataset.py` | 64×64 grayscale circle / square / triangle, **area-matched** shapes; independent size, x/y, rotation, foreground & background intensity, contrast (pivot = image mean), brightness offset, Gaussian noise (fixed per-sample field), sinusoidal object texture. Each image is a pure function `render(meta)` of its factors. Splits: train 5000 and test 1000 (fully random pose), `test_aligned` 600 and `proto` 900 (centred ±1.5 px, rotation ±8°). |
| Single-factor pairs / trajectories | `data/transformations.py` | `change_shape`, `morph_shape` (signed-distance interpolation), `change_contrast`, `change_brightness`, `change_position`, `change_rotation`, `change_texture`, `change_size`, `resample_noise`; they operate on the factors, so exactly one factor (and the identical noise field) changes. `brightness_gain` is a pixel-space scaling, labelled as such. |
| Phase congruency | `perceptual/phase_congruency.py` | Differentiable Kovesi-style PC: 4 scales × 6 orientations log-Gabor bank in the Fourier domain, even/odd responses, phase-deviation local energy, Rayleigh noise threshold (mean-based), soft (T-scaled softplus) threshold, frequency-spread weighting. All formulas are in the module docstring. |
| Gradient magnitude | `perceptual/gradient_magnitude.py` | Fixed Sobel convolutions, replicate padding, `sqrt(Gx²+Gy²+ε)`. |
| Energies & forces | `perceptual/energies.py` | Prototypes μ_PC,y, μ_GM,y and class-mean image r_y from the `proto` split; `get_edge_force`, `get_contrast_force`, `get_total_force` all by `torch.autograd.grad` of the energy; `get_pixel_force = r_y − x`. |
| Model | `models/small_cnn.py` | Same for CE / AT / PAG (≈60k params). |
| Training | `training/train_ce.py`, `train_at.py`, `train_pag.py` | PAG uses `create_graph=True` double back-prop on `1 − cos(∇x f_y, g_target)`, clean images only. |
| Analyses | `analysis/*.py` | energy validation, force decomposition (LS + exact 2-variable NNLS + controls), potential consistency + β, gradient-flow trajectories, attack energy decomposition, bootstrap / paired tests. |
| Figures | `visualization/plots.py` | → `results/figures/` |
