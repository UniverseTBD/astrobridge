"""grz -> RGB composite, for feeding a Legacy Survey cutout to something that expects an ordinary
picture (Qwen's own native vision pathway — see inference/compare.py) rather than raw per-band
flux (AION's pathway, which wants the bands kept separate — see encoders/aion_image.py).

Same conversion the standalone test_subjects_review.ipynb notebook already uses, factored out
here so it's not duplicated between that notebook and inference/compare.py.
"""
from __future__ import annotations

import numpy as np
from astropy.visualization import make_lupton_rgb
from PIL import Image


def grz_to_rgb(bands: np.ndarray, q: float = 8, stretch: float = 0.5) -> Image.Image:
    """`bands`: (3, H, W) array in g, r, z order (configs/modalities.yaml's declared band order —
    see captioner/test_subjects/'s image_*.npy for real examples). Lupton's asinh composite maps
    the reddest band to R and bluest to B, so g/r/z -> B/G/R, not straight g/r/z -> R/G/B.
    """
    if bands.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) in g, r, z order; got shape {bands.shape}.")
    g, r, z = bands
    rgb_array = make_lupton_rgb(z, r, g, Q=q, stretch=stretch)
    return Image.fromarray(rgb_array)
