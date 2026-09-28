"""Central experiment configuration. Everything that affects results lives here and is dumped to results/config.json."""
from dataclasses import dataclass, field, asdict
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
CHECKPOINTS = os.path.join(RESULTS, "checkpoints")
DATA_CACHE = os.path.join(ROOT, "data_cache")

CLASSES = ["circle", "square", "triangle"]


@dataclass
class DataConfig:
    image_size: int = 64
    n_train: int = 5000
    n_test: int = 1000            # fully randomised nuisance factors (accuracy / robustness)
    n_test_aligned: int = 600     # centred + roughly aligned (primary force-decomposition set)
    n_proto: int = 900            # centred + aligned, disjoint from test; used ONLY to build prototypes
    seed_train: int = 1001
    seed_test: int = 1002
    seed_test_aligned: int = 1003
    seed_proto: int = 1004
    # nuisance ranges (uniform); chosen so that clipping to [0,1] is rare
    size_range: tuple = (8.0, 13.0)            # circle radius; square/triangle are area-matched to the circle
    center_range: tuple = (24.0, 40.0)         # pixel coordinates of object centre (random splits)
    aligned_center: float = 31.5
    aligned_center_jitter: float = 1.5         # +- px (aligned splits)
    aligned_rotation_jitter: float = 8.0       # +- deg (aligned splits)
    foreground_range: tuple = (0.50, 0.72)
    background_range: tuple = (0.22, 0.42)
    contrast_range: tuple = (0.75, 1.25)       # multiplicative, pivot = mean of the pre-noise image
    brightness_range: tuple = (-0.06, 0.06)    # additive offset
    noise_range: tuple = (0.0, 0.03)           # std of additive Gaussian noise (fixed per-sample noise field)
    texture_strength_range: tuple = (0.0, 0.06)
    texture_freq_range: tuple = (0.08, 0.20)   # cycles / pixel, sinusoidal grating on the object only


@dataclass
class PCConfig:
    """Log-Gabor phase-congruency parameters (defaults follow Kovesi's phasecong3)."""
    nscale: int = 4
    norient: int = 6
    min_wavelength: float = 3.0
    mult: float = 2.1
    sigma_onf: float = 0.55
    d_theta_on_sigma: float = 1.2
    k: float = 2.0                  # noise threshold = mean + k * sigma of the Rayleigh noise-energy model
    cutoff: float = 0.5             # frequency-spread weighting
    g: float = 10.0
    eps: float = 1e-4
    softplus_beta_rel: float = 10.0  # soft threshold sharpness, relative to T (keeps PC contrast-invariant)
    abs_delta: float = 1e-4          # smooth |z| = sqrt(z^2+d^2)-d
    lowpass_cutoff: float = 0.45
    lowpass_order: int = 15


@dataclass
class GMConfig:
    eps: float = 1e-6
    sobel_scale: float = 1.0 / 8.0   # Sobel output /8 -> units of intensity per pixel
    padding: str = "replicate"


@dataclass
class TrainConfig:
    arch: str = ("SmallCNN: fixed input affine (x-0.345)/0.107 -> [conv3x3-BN-ReLU]x2(16,32) -> maxpool2 -> "
                 "conv3x3-BN-ReLU(64) -> maxpool2 -> conv3x3-BN-ReLU(64) -> global average pool -> linear(3)")
    optimizer: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 20
    lr_schedule: str = "cosine"
    model_seed: int = 0
    # adversarial training (PGD, L-inf, pixel range [0,1])
    at_eps: float = 8 / 255
    at_alpha: float = 2 / 255
    at_steps: int = 7
    at_random_start: bool = True
    at_epochs: int = 25
    at_clean_epochs: int = 3        # eps = 0 (plain CE) for the first 3 epochs: from-scratch AT stalled at chance
    at_eps_ramp_epochs: int = 6     # then eps (and alpha) ramp linearly to at_eps over 6 epochs, full eps after
    at_attack_mode: str = "PGD examples generated with the model in eval mode (BN running statistics)"
    # PAG-like training (clean images only)
    pag_lambda: float = 1.0
    pag_target: str = "-(grad E_edge + grad E_contrast) wrt x, class-mean (centred/aligned) prototypes of TRUE class"


@dataclass
class EvalConfig:
    pgd_eps: float = 8 / 255
    pgd_alpha: float = 2 / 255
    pgd_steps: int = 20
    pgd_random_start: bool = True
    strong_pgd_eps: tuple = (4 / 255, 8 / 255, 16 / 255)
    strong_pgd_steps: int = 50
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 7
    control_seed: int = 11
    # potential-consistency trajectories
    n_traj_per_class: int = 30
    traj_steps: int = 11
    # gradient-flow trajectories
    flow_images_per_class: int = 4
    flow_steps: int = 40
    flow_eta: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    pc: PCConfig = field(default_factory=PCConfig)
    gm: GMConfig = field(default_factory=GMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    global_seed: int = 0
    num_threads: int = 4
    preprocessing: str = ("grayscale float32 in [0,1], 64x64, no augmentation; images rendered from generative "
                          "factors then clipped to [0,1]; inside the model a FIXED affine (x-0.345)/0.107 (training-set "
                          "mean/std). Attacks, forces and gradients are all taken w.r.t. the [0,1] image. "
                          "Per-image standardisation was rejected because it hard-codes contrast/offset invariance.")

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def quick_config():
    """Tiny config for smoke-testing the pipeline end to end."""
    c = Config()
    c.data.n_train, c.data.n_test, c.data.n_test_aligned, c.data.n_proto = 300, 90, 60, 90
    c.train.epochs = 1
    c.train.at_epochs, c.train.at_clean_epochs, c.train.at_eps_ramp_epochs = 2, 1, 1
    c.eval.bootstrap_resamples = 200
    c.eval.n_traj_per_class = 2
    c.eval.flow_images_per_class = 1
    c.eval.flow_steps = 5
    c.eval.pgd_steps = 3
    c.eval.strong_pgd_steps = 3
    return c
