from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image


CUBE_PATH = Path(
    "outputs/galsyn/"
    "galsyn_tng50_phalo_45_493093_lsst_photo.fits"
)

OUTPUT_PATH = Path("outputs/galsyn/lsst_gri_rgb.png")


def clean(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).copy()
    image[~np.isfinite(image)] = 0
    image[image < 0] = 0
    return image


with fits.open(CUBE_PATH, memmap=True) as hdul:
    red = clean(hdul["DUST_LSST_I"].data)
    green = clean(hdul["DUST_LSST_R"].data)
    blue = clean(hdul["DUST_LSST_G"].data)

stack = np.stack([red, green, blue])
positive = stack[stack > 0]

if positive.size == 0:
    raise ValueError("All selected LSST bands contain no positive flux.")

scale = np.percentile(positive, 99.5)

with np.errstate(divide="ignore", invalid="ignore"):
    rgb = make_lupton_rgb(
        red / scale,
        green / scale,
        blue / scale,
        stretch=0.25,
        Q=10,
    )

rgb = np.nan_to_num(rgb, nan=0, posinf=255, neginf=0)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.flipud(rgb)).save(OUTPUT_PATH)

print(f"Saved {OUTPUT_PATH}")