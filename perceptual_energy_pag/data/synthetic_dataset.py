"""Synthetic grayscale shape dataset with fully recorded generative factors.

Every image is a deterministic function of its metadata dict (``render(meta)``), so any single factor can be
changed while all others -- including the exact per-pixel noise field -- stay fixed.

Rendering pipeline (all in float64, then cast to float32):
    1. object-frame coordinates:  p = R(-rotation) (u - cx, v - cy)
    2. signed distance d(p) of the shape (negative inside); a shape morph uses d = (1-t) d_a + t d_b
    3. anti-aliased mask m = clip(0.5 - d, 0, 1)                       (1-pixel linear ramp)
    4. texture T = sin(2 pi f ((u-cx) cos(phi) + (v-cy) sin(phi)) + phase), applied to the object only
    5. raw = B + (F - B) m + s_tex T m                                   (F/B = foreground/background intensity)
    6. contrast about the pre-noise mean mu = mean(raw):  img = mu + c (raw - mu)
    7. brightness offset: img += delta
    8. noise: img += sigma * N_seed   (N_seed is a fixed standard-normal field drawn from noise_seed)
    9. clip to [0, 1]
Shapes are area-matched: circle radius r; square half-side r*sqrt(pi)/2; equilateral triangle side
r*sqrt(4 pi / sqrt 3).  A shape change therefore does not change the object's area / mean brightness.
"""
import json
import os

import numpy as np

from config import CLASSES, DATA_CACHE, DataConfig


# ---------------------------------------------------------------- signed distance functions (negative inside)
def _sdf(cls, px, py, size):
    if cls == "circle":
        return np.sqrt(px ** 2 + py ** 2) - size
    if cls == "square":
        h = size * np.sqrt(np.pi) / 2.0
        qx, qy = np.abs(px) - h, np.abs(py) - h
        outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        inside = np.minimum(np.maximum(qx, qy), 0)
        return outside + inside
    if cls == "triangle":  # equilateral, centroid at origin (Inigo Quilez' exact SDF)
        side = size * np.sqrt(4 * np.pi / np.sqrt(3))
        r, k = side / 2.0, np.sqrt(3.0)
        x = np.abs(px) - r
        y = py + r / k
        cond = x + k * y > 0
        x2 = np.where(cond, (x - k * y) / 2.0, x)
        y2 = np.where(cond, (-k * x - y) / 2.0, y)
        x2 = x2 - np.clip(x2, -2 * r, 0)
        return -np.sqrt(x2 ** 2 + y2 ** 2) * np.sign(y2)
    raise ValueError(cls)


def _noise_field(seed, n):
    return np.random.default_rng(int(seed)).standard_normal((n, n))


def render(meta, image_size=64):
    """Render one image (float32, HxW, in [0,1]) from its generative factors."""
    n = image_size
    v, u = np.mgrid[0:n, 0:n].astype(np.float64)
    dx, dy = u - meta["x"], v - meta["y"]
    th = np.deg2rad(meta["rotation"])
    px = np.cos(th) * dx + np.sin(th) * dy
    py = -np.sin(th) * dx + np.cos(th) * dy

    d = _sdf(meta["class"], px, py, meta["size"])
    t = meta.get("morph_t", 0.0)
    if meta.get("morph_to") and t > 0:
        d = (1 - t) * d + t * _sdf(meta["morph_to"], px, py, meta["size"])
    m = np.clip(0.5 - d, 0.0, 1.0)

    phi = np.deg2rad(meta["texture_orientation"])
    tex = np.sin(2 * np.pi * meta["texture_freq"] * (dx * np.cos(phi) + dy * np.sin(phi)) + meta["texture_phase"])

    F, B = meta["foreground_intensity"], meta["background_intensity"]
    raw = B + (F - B) * m + meta["texture_strength"] * tex * m
    mu = raw.mean()
    img = mu + meta["contrast"] * (raw - mu)
    img = img + meta["brightness"]
    img = img + meta["noise_std"] * _noise_field(meta["noise_seed"], n)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- factor sampling
def sample_meta(rng, label, cfg: DataConfig, aligned: bool):
    U = lambda r: float(rng.uniform(*r))
    if aligned:
        j = cfg.aligned_center_jitter
        x = cfg.aligned_center + float(rng.uniform(-j, j))
        y = cfg.aligned_center + float(rng.uniform(-j, j))
        rot = float(rng.uniform(-cfg.aligned_rotation_jitter, cfg.aligned_rotation_jitter))
    else:
        x, y = U(cfg.center_range), U(cfg.center_range)
        rot = float(rng.uniform(0, 360))
    return {
        "class": CLASSES[label],
        "label": int(label),
        "x": x,
        "y": y,
        "rotation": rot,
        "size": U(cfg.size_range),
        "foreground_intensity": U(cfg.foreground_range),
        "background_intensity": U(cfg.background_range),
        "contrast": U(cfg.contrast_range),
        "brightness": U(cfg.brightness_range),
        "noise_std": U(cfg.noise_range),
        "noise_seed": int(rng.integers(0, 2 ** 31 - 1)),
        "texture_strength": U(cfg.texture_strength_range),
        "texture_freq": U(cfg.texture_freq_range),
        "texture_orientation": float(rng.uniform(0, 180)),
        "texture_phase": float(rng.uniform(0, 2 * np.pi)),
        "morph_to": None,
        "morph_t": 0.0,
    }


def generate_split(n, seed, cfg: DataConfig, aligned=False, split_name="train"):
    """Class-balanced split. Returns images [n,1,H,W] float32, labels [n] int64, list of metadata dicts."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % len(CLASSES)
    rng.shuffle(labels)
    metas, imgs = [], np.zeros((n, 1, cfg.image_size, cfg.image_size), np.float32)
    for i, lab in enumerate(labels):
        m = sample_meta(rng, int(lab), cfg, aligned)
        m.update({"split": split_name, "index": i, "split_seed": seed})
        metas.append(m)
        imgs[i, 0] = render(m, cfg.image_size)
    return imgs, labels.astype(np.int64), metas


def load_or_generate(cfg: DataConfig, tag="full"):
    """Generate (or load cached) train / test / test_aligned / proto splits."""
    os.makedirs(DATA_CACHE, exist_ok=True)
    specs = {
        "train": (cfg.n_train, cfg.seed_train, False),
        "test": (cfg.n_test, cfg.seed_test, False),
        "test_aligned": (cfg.n_test_aligned, cfg.seed_test_aligned, True),
        "proto": (cfg.n_proto, cfg.seed_proto, True),
    }
    out = {}
    for name, (n, seed, aligned) in specs.items():
        path = os.path.join(DATA_CACHE, f"{tag}_{name}.npz")
        mpath = os.path.join(DATA_CACHE, f"{tag}_{name}_metadata.json")
        if os.path.exists(path) and os.path.exists(mpath):
            z = np.load(path)
            with open(mpath) as f:
                metas = json.load(f)
            out[name] = (z["images"], z["labels"], metas)
            continue
        imgs, labels, metas = generate_split(n, seed, cfg, aligned, name)
        np.savez_compressed(path, images=imgs, labels=labels)
        with open(mpath, "w") as f:
            json.dump(metas, f)
        out[name] = (imgs, labels, metas)
    return out


def clip_fraction(images):
    return float(((images <= 0.0) | (images >= 1.0)).mean())
