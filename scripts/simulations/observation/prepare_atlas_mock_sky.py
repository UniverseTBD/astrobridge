"""Place an Atlas z=0 surface-brightness map at a chosen mock redshift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.io import fits


def center_crop(image: np.ndarray, pixels: int) -> np.ndarray:
    if pixels > min(image.shape):
        raise ValueError(f"Crop of {pixels} pixels exceeds image shape {image.shape}.")
    y0 = (image.shape[0] - pixels) // 2
    x0 = (image.shape[1] - pixels) // 2
    return image[y0 : y0 + pixels, x0 : x0 + pixels]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas_files", type=Path, help="Atlas files directory.")
    parser.add_argument("output", type=Path, help="Derived ideal sky FITS.")
    parser.add_argument("--observer", default="O1")
    parser.add_argument("--redshift", type=float, required=True)
    parser.add_argument("--crop-kpc", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.redshift <= 0:
        raise ValueError("A positive mock redshift is required for angular sampling.")
    kpc_per_arcsec = Planck18.kpc_proper_per_arcmin(args.redshift).to_value(u.kpc / u.arcsec)
    input_scale_arcsec = 0.1 / kpc_per_arcsec
    crop_pixels = round(args.crop_kpc / 0.1)
    dimming = (1 + args.redshift) ** -3
    primary = fits.PrimaryHDU()
    primary.header["OBSSOURCE"] = "TNG50-SKIRT Atlas"
    primary.header["MOCKZ"] = (args.redshift, "Assigned mock observation redshift")
    primary.header["PIXSIZE"] = (input_scale_arcsec, "Input angular scale [arcsec/pixel]")
    primary.header["CROPKPC"] = (args.crop_kpc, "Centered physical crop [kpc]")
    primary.header["SBDIM"] = (dimming, "Applied I_nu cosmological dimming")
    primary.header["COSMO"] = "Planck18"
    hdus = [primary]
    inputs: dict[str, str] = {}
    for band in ("u", "g", "r", "i", "z", "y"):
        source = args.atlas_files / f"TNG000008_{args.observer}_LSST_{band}.fits"
        if not source.is_file():
            raise FileNotFoundError(source)
        with fits.open(source, memmap=False) as atlas:
            data = center_crop(np.asarray(atlas[0].data, dtype=np.float64), crop_pixels) * dimming
        header = fits.Header()
        header["EXTNAME"] = f"DUST_LSST_{band.upper()}"
        header["BUNIT"] = "MJy/sr"
        header["PIXSIZE"] = (input_scale_arcsec, "Input angular scale [arcsec/pixel]")
        hdus.append(fits.ImageHDU(data=data, header=header))
        inputs[band] = str(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(args.output, overwrite=True)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps({
            "source": "TNG50-SKIRT Atlas", "observer": args.observer,
            "mock_redshift": args.redshift, "crop_kpc": args.crop_kpc,
            "input_pixel_scale_arcsec": input_scale_arcsec,
            "surface_brightness_dimming": "(1+z)^-3", "dimming_factor": dimming,
            "cosmology": "Planck18", "inputs": inputs,
            "caveat": "No bandpass K-correction is applied; at z=0.03 this is a small approximation.",
        }, indent=2), encoding="utf-8"
    )
    print(f"Wrote derived Atlas sky cube: {args.output}")


if __name__ == "__main__":
    main()
