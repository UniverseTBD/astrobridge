"""Apply a configured analytic LSST PSF, detector-pixel integration, and optional noise.

The input must contain ideal, surface-brightness image planes (for example the
DR1 ``DUST_LSST_*`` extensions).  The source product is never edited: this
script writes a separate FITS file and a provenance sidecar.  Background and noise are selected entirely by the supplied observation profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Observation profile must be a YAML mapping.")
    return config


def source_pixel_scale_arcsec(header: fits.Header, override: float | None) -> float:
    if override is not None:
        return override
    value = header.get("PIXSIZE")
    if value is None:
        raise ValueError("Input has no PIXSIZE header; pass --input-pixel-scale-arcsec.")
    return float(value)


def footprint_weights(
    input_pixels: int,
    *,
    input_pixel_scale_arcsec: float,
    output_pixel_scale_arcsec: float,
) -> np.ndarray:
    """Return 1-D area weights from ideal pixels to detector footprints."""
    input_width = input_pixels * input_pixel_scale_arcsec
    output_pixels = int(np.ceil(input_width / output_pixel_scale_arcsec))
    output_width = output_pixels * output_pixel_scale_arcsec
    output_edges = (
        np.arange(output_pixels + 1) * output_pixel_scale_arcsec
        - (output_width - input_width) / 2
    )
    input_edges = np.arange(input_pixels + 1) * input_pixel_scale_arcsec
    lower = np.maximum(output_edges[:-1, None], input_edges[None, :-1])
    upper = np.minimum(output_edges[1:, None], input_edges[None, 1:])
    overlap = np.clip(upper - lower, 0.0, None)
    return overlap / output_pixel_scale_arcsec


def convolve_and_resample(
    image: np.ndarray,
    *,
    fwhm_arcsec: float,
    input_pixel_scale_arcsec: float,
    output_pixel_scale_arcsec: float,
) -> np.ndarray:
    """Convolve and integrate ideal surface brightness over detector pixels.

    The output grid is centred on the input field and uses exact rectangular
    overlap areas, including fractional ideal pixels at detector edges. Values
    stay in surface-brightness units; integrated flux needs pixel solid angle.
    """
    if input_pixel_scale_arcsec <= 0 or output_pixel_scale_arcsec <= 0:
        raise ValueError("Pixel scales must be positive.")
    sigma_pixels = fwhm_arcsec / (2.354820045 * input_pixel_scale_arcsec)
    valid = np.isfinite(image)
    filled = np.where(valid, image, 0.0)
    weights = gaussian_filter(valid.astype(float), sigma=sigma_pixels, mode="constant")
    blurred = gaussian_filter(filled, sigma=sigma_pixels, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        blurred = np.where(weights > 1e-8, blurred / weights, np.nan)
    row_weights = footprint_weights(
        blurred.shape[0],
        input_pixel_scale_arcsec=input_pixel_scale_arcsec,
        output_pixel_scale_arcsec=output_pixel_scale_arcsec,
    )
    column_weights = footprint_weights(
        blurred.shape[1],
        input_pixel_scale_arcsec=input_pixel_scale_arcsec,
        output_pixel_scale_arcsec=output_pixel_scale_arcsec,
    )
    return np.matmul(np.matmul(row_weights, blurred), column_weights.T)


ARCSEC2_PER_SR = 206264.80624709636**2


def abmag_to_electrons(
    abmag: float | np.ndarray,
    *,
    zeropoint_abmag_1e_per_s: float,
    exposure_seconds: float,
    area_arcsec2: float,
) -> np.ndarray:
    """Convert an AB magnitude per square arcsecond to electrons per pixel."""
    flux_jy = 3631.0 * np.power(10.0, -0.4 * np.asarray(abmag)) * area_arcsec2
    return exposure_seconds * flux_jy / 3631.0 * np.power(10.0, 0.4 * zeropoint_abmag_1e_per_s)


def surface_brightness_to_electrons(
    surface_brightness_mjy_sr: np.ndarray,
    *,
    zeropoint_abmag_1e_per_s: float,
    exposure_seconds: float,
    pixel_scale_arcsec: float,
) -> np.ndarray:
    """Convert MJy/sr to expected source electrons in one output pixel."""
    area_arcsec2 = pixel_scale_arcsec**2
    flux_jy = (
        np.clip(np.nan_to_num(surface_brightness_mjy_sr, nan=0.0), 0.0, None)
        * 1.0e6
        * area_arcsec2
        / ARCSEC2_PER_SR
    )
    return exposure_seconds * flux_jy / 3631.0 * np.power(10.0, 0.4 * zeropoint_abmag_1e_per_s)


def residual_background_e(
    shape: tuple[int, int],
    *,
    band: str,
    sky_e: float,
    profile: dict[str, Any],
) -> np.ndarray:
    """Return an optional residual background in electrons per output pixel."""
    residual = profile["background"].get("residual", {"model": "none"})
    model = residual["model"]
    if model == "none":
        return np.zeros(shape, dtype=np.float64)
    if model != "polynomial":
        raise ValueError(f"Unsupported residual background model: {model}")
    coefficients = residual["coefficients_fraction_of_sky"][band]
    if len(coefficients) != 6:
        raise ValueError("Polynomial residual needs six coefficients: 1, x, y, x2, xy, y2.")
    y, x = np.indices(shape, dtype=np.float64)
    x = 2 * x / max(shape[1] - 1, 1) - 1
    y = 2 * y / max(shape[0] - 1, 1) - 1
    c0, cx, cy, cxx, cxy, cyy = (float(value) for value in coefficients)
    return sky_e * (c0 + cx * x + cy * y + cxx * x**2 + cxy * x * y + cyy * y**2)


def add_single_visit_noise(
    image_mjy_sr: np.ndarray,
    *,
    band: str,
    profile: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw/science electrons, variance, sky, and total background."""
    photometry = profile["photometry"]
    noise = profile["noise"]
    background = profile["background"]
    coadd_visits = int(noise.get("coadd_visits", 1))
    exposure_seconds = (
        coadd_visits
        * float(noise["exposures_per_visit"])
        * float(noise["exposure_seconds_per_exposure"])
    )
    pixel_scale = float(profile["detector"]["pixel_scale_arcsec"])
    area_arcsec2 = pixel_scale**2
    zeropoint = float(photometry["instrumental_zeropoint_abmag_1e_per_s"][band])
    source_e = surface_brightness_to_electrons(
        image_mjy_sr,
        zeropoint_abmag_1e_per_s=zeropoint,
        exposure_seconds=exposure_seconds,
        pixel_scale_arcsec=pixel_scale,
    )
    sky_e = float(
        abmag_to_electrons(
            float(background["sky_abmag_per_arcsec2"][band]),
            zeropoint_abmag_1e_per_s=zeropoint,
            exposure_seconds=exposure_seconds,
            area_arcsec2=area_arcsec2,
        )
    )
    dark_e = float(noise["dark_current_e_per_s_per_pixel"]) * exposure_seconds
    residual_e = residual_background_e(source_e.shape, band=band, sky_e=sky_e, profile=profile)
    background_e = sky_e + dark_e + residual_e
    expected_e = source_e + background_e
    poisson_e = rng.poisson(expected_e)
    read_sigma_e = np.sqrt(
        coadd_visits * float(noise["exposures_per_visit"])
    ) * float(noise["read_noise_e_per_exposure"])
    raw_e = poisson_e + rng.normal(0.0, read_sigma_e, size=source_e.shape)
    science_e = raw_e - background_e
    variance_e2 = expected_e + read_sigma_e**2
    sky_map_e = np.full(source_e.shape, sky_e, dtype=np.float64)
    background_map_e = np.full(source_e.shape, background_e, dtype=np.float64)
    return raw_e, science_e, variance_e2, sky_map_e, background_map_e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Ideal DR1 FITS cube.")
    parser.add_argument("output", type=Path, help="New PSF-only FITS product.")
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/observation/lsst_median_zenith_v1.yaml"),
        help="Observation profile YAML.",
    )
    parser.add_argument(
        "--dust-state",
        choices=("DUST", "NODUST"),
        default="DUST",
        help="Which ideal DR1 image set to process.",
    )
    parser.add_argument(
        "--input-pixel-scale-arcsec",
        type=float,
        help="Override input PIXSIZE when its FITS header lacks it.",
    )
    parser.add_argument(
        "--seed", type=int, help="Override the profile random seed for this sample."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_yaml(args.profile)
    if args.seed is not None:
        profile["noise"]["random_seed"] = args.seed
    psf = profile["psf"]
    detector = profile["detector"]
    if psf["model"] != "circular_gaussian":
        raise ValueError("Only psf.model=circular_gaussian is implemented.")
    background_model = profile["background"]["model"]
    noise_model = profile["noise"]["model"]
    noisy = background_model != "none" or noise_model != "none"
    if noisy and (
        background_model != "uniform_dark_sky"
        or noise_model != "poisson_plus_gaussian_read"
    ):
        raise ValueError("Unsupported background/noise model combination.")

    band_fwhm = psf["geometric_fwhm_arcsec"]
    rng = np.random.default_rng(profile["noise"].get("random_seed")) if noisy else None
    output_scale = float(detector["pixel_scale_arcsec"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_hdus: list[fits.ImageHDU | fits.PrimaryHDU] = []

    with fits.open(args.input, memmap=False) as source:
        source_scale = source_pixel_scale_arcsec(source[0].header, args.input_pixel_scale_arcsec)
        primary = fits.PrimaryHDU(header=source[0].header.copy())
        primary.header["OBSPROF"] = (profile["profile"]["id"], "Observation profile")
        primary.header["PSFMODEL"] = (psf["model"], "PSF model")
        primary.header["FWHMTYPE"] = (psf["fwhm_definition"], "PSF FWHM definition")
        primary.header["INPIXSCL"] = (source_scale, "Ideal pixel scale [arcsec/pixel]")
        primary.header["PIXSIZE"] = (output_scale, "Observed pixel scale [arcsec/pixel]")
        primary.header["RESAMP"] = ("FOOTPRINT", "Exact detector-pixel footprint integration")
        primary.header["NOISEMOD"] = (noise_model, "Configured noise model")
        primary.header["NCOADD"] = (int(profile["noise"].get("coadd_visits", 1)), "Visits coadded")
        output_hdus.append(primary)

        for band in ("u", "g", "r", "i", "z", "y"):
            extension_name = f"{args.dust_state}_LSST_{band.upper()}"
            if extension_name not in source:
                raise KeyError(f"Missing expected DR1 extension: {extension_name}")
            observed = convolve_and_resample(
                np.asarray(source[extension_name].data, dtype=np.float64),
                fwhm_arcsec=float(band_fwhm[band]),
                input_pixel_scale_arcsec=source_scale,
                output_pixel_scale_arcsec=output_scale,
            )
            header = source[extension_name].header.copy()
            header["EXTNAME"] = f"OBS_LSST_{band.upper()}"
            header["OBSPROF"] = profile["profile"]["id"]
            header["PSFFWHM"] = (float(band_fwhm[band]), "Geometric FWHM [arcsec]")
            header["PIXSIZE"] = (output_scale, "Observed pixel scale [arcsec/pixel]")
            header["RESAMP"] = ("FOOTPRINT", "Exact detector-pixel footprint integration")
            if not noisy:
                output_hdus.append(fits.ImageHDU(data=observed, header=header))
                continue
            raw_e, science_e, variance_e2, sky_e, background_e = add_single_visit_noise(
                observed, band=band, profile=profile, rng=rng
            )
            header["BUNIT"] = "electron"
            header["NOISEMOD"] = noise_model
            header["RNGSEED"] = int(profile["noise"]["random_seed"])
            header["NCOADD"] = (int(profile["noise"].get("coadd_visits", 1)), "Visits coadded")
            header["BGSUB"] = (True, "Expected uniform sky and dark current removed")
            raw_header = header.copy()
            raw_header["EXTNAME"] = f"RAW_LSST_{band.upper()}"
            raw_header["BGSUB"] = (False, "Raw detector counts include background")
            output_hdus.append(fits.ImageHDU(data=raw_e, header=raw_header))
            output_hdus.append(fits.ImageHDU(data=science_e, header=header))
            variance_header = header.copy()
            variance_header["EXTNAME"] = f"VAR_LSST_{band.upper()}"
            variance_header["BUNIT"] = "electron2"
            output_hdus.append(fits.ImageHDU(data=variance_e2, header=variance_header))
            sky_header = header.copy()
            sky_header["EXTNAME"] = f"SKY_LSST_{band.upper()}"
            output_hdus.append(fits.ImageHDU(data=sky_e, header=sky_header))
            background_header = header.copy()
            background_header["EXTNAME"] = f"BKG_LSST_{band.upper()}"
            output_hdus.append(fits.ImageHDU(data=background_e, header=background_header))
            mask_header = header.copy()
            mask_header["EXTNAME"] = f"MASK_LSST_{band.upper()}"
            mask_header["BUNIT"] = "bitmask"
            output_hdus.append(
                fits.ImageHDU(data=np.asarray(~np.isfinite(observed), dtype=np.uint16), header=mask_header)
            )

    fits.HDUList(output_hdus).writeto(args.output, overwrite=True)
    operations = ["gaussian_psf_convolution", "exact_rectangular_detector_footprint_integration"]
    if noisy:
        operations += [
            "mjy_per_sr_to_expected_electrons",
            "uniform_dark_sky_background", "raw_detector_counts",
            "poisson_source_sky_dark_noise",
            "gaussian_read_noise",
            "exact_background_subtraction",
        ]
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "product": str(args.output),
                "observation_profile": profile,
                "input_pixel_scale_arcsec": source_scale,
                "output_pixel_scale_arcsec": output_scale,
                "dust_state": args.dust_state,
                "operations": operations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote LSST observation product: {args.output}")
    print(f"Wrote provenance: {sidecar}")


if __name__ == "__main__":
    main()
