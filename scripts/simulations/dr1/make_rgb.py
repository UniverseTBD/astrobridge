from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image


CUBE_PATH = Path("./data/TNG50-11-10_8sqarcmin/dust0-S18K13/angle1/individual_datacubes/galsyn_tng50_phalo_45_493093.fits.gz")
OUTPUT_PATH = Path("outputs/dr1/rgb.png")

BLUE_FILTER = "JWST_NIRCAM_F090W"
GREEN_FILTER = "JWST_NIRCAM_F150W"
RED_FILTER = "JWST_NIRCAM_F200W"


def clean(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32).copy()
    result[~np.isfinite(result)] = 0
    result[result < 0] = 0
    return result


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(CUBE_PATH, memmap=True) as cube:
        red = clean(cube[f"DUST_{RED_FILTER}"].data)
        green = clean(cube[f"DUST_{GREEN_FILTER}"].data)
        blue = clean(cube[f"DUST_{BLUE_FILTER}"].data)

    # One shared scale preserves relative band brightness better than
    # independently normalizing each channel.
    positive = np.concatenate(
        [
            red[red > 0],
            green[green > 0],
            blue[blue > 0],
        ]
    )

    if positive.size == 0:
        raise ValueError("All three bands are empty or non-positive.")

    scale = np.percentile(positive, 99.7)

    rgb = make_lupton_rgb(
        red / scale,
        green / scale,
        blue / scale,
        stretch=0.15,
        Q=8,
    )

    Image.fromarray(np.flipud(rgb)).save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()