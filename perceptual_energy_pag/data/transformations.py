"""Controlled single-factor transformations.

Each function takes a sample's metadata (the generative factors of image x) and returns (x', meta') where exactly
ONE generative factor differs.  Operating on the factors rather than on pixels is what makes the change exact:
e.g. a shape change keeps position, size, rotation, intensities, texture phase and the identical noise field.
The one exception is ``brightness_gain`` which is a pixel-space photometric scaling (x' = clip(k x)); it is not a
generative factor and is labelled as such.
"""
import copy

import numpy as np

from config import CLASSES
from data.synthetic_dataset import render


def _apply(meta, image_size=64, **changes):
    m = copy.deepcopy(meta)
    m.update(changes)
    return render(m, image_size), m


def change_shape(meta, new_class, image_size=64):
    return _apply(meta, image_size, **{"class": new_class, "label": CLASSES.index(new_class), "morph_to": None,
                                       "morph_t": 0.0})


def morph_shape(meta, target_class, t, image_size=64):
    """Signed-distance interpolation between the sample's shape (t=0) and target_class (t=1)."""
    return _apply(meta, image_size, morph_to=target_class, morph_t=float(t))


def change_contrast(meta, factor, image_size=64):
    return _apply(meta, image_size, contrast=meta["contrast"] * factor)


def set_contrast(meta, value, image_size=64):
    return _apply(meta, image_size, contrast=float(value))


def change_brightness(meta, delta, image_size=64):
    return _apply(meta, image_size, brightness=meta["brightness"] + delta)


def change_position(meta, dx, dy, image_size=64):
    return _apply(meta, image_size, x=meta["x"] + dx, y=meta["y"] + dy)


def change_rotation(meta, dtheta, image_size=64):
    return _apply(meta, image_size, rotation=meta["rotation"] + dtheta)


def change_texture(meta, strength, image_size=64):
    return _apply(meta, image_size, texture_strength=float(strength))


def change_size(meta, factor, image_size=64):
    return _apply(meta, image_size, size=meta["size"] * factor)


def resample_noise(meta, new_seed, image_size=64):
    """Same image, new noise realisation: the intrinsic operator-variability floor."""
    return _apply(meta, image_size, noise_seed=int(new_seed))


def brightness_gain(image, k):
    """Pixel-space multiplicative brightness scaling (NOT a single generative factor)."""
    return np.clip(image * k, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- gradual one-factor trajectories x(t)
TRAJECTORY_FACTORS = ["shape", "contrast", "brightness", "texture", "position", "rotation"]


def trajectory(meta, factor, n_steps=11, image_size=64):
    """Returns (params t [n_steps], images [n_steps,H,W]) where only `factor` varies with t."""
    if factor == "shape":
        target = CLASSES[(meta["label"] + 1) % len(CLASSES)]
        ts = np.linspace(0, 1, n_steps)
        imgs = [morph_shape(meta, target, t, image_size)[0] for t in ts]
    elif factor == "contrast":
        ts = np.linspace(0.5, 1.5, n_steps)
        imgs = [set_contrast(meta, t, image_size)[0] for t in ts]
    elif factor == "brightness":
        ts = np.linspace(-0.15, 0.15, n_steps)
        imgs = [_apply(meta, image_size, brightness=float(t))[0] for t in ts]
    elif factor == "texture":
        ts = np.linspace(0.0, 0.12, n_steps)
        imgs = [change_texture(meta, t, image_size)[0] for t in ts]
    elif factor == "position":
        ts = np.linspace(0.0, 8.0, n_steps)
        imgs = [change_position(meta, t, 0.0, image_size)[0] for t in ts]
    elif factor == "rotation":
        ts = np.linspace(0.0, 40.0, n_steps)
        imgs = [change_rotation(meta, t, image_size)[0] for t in ts]
    else:
        raise ValueError(factor)
    return np.asarray(ts, np.float64), np.stack(imgs).astype(np.float32)
