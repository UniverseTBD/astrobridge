"""Spectrum plotting and spectral line visualization utilities."""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

logger = logging.getLogger(__name__)

# Canonical rest-frame optical, UV, and near-IR spectral lines (wavelength in Angstroms).
# Grouped by type for distinct visual styling.
REST_FRAME_LINES: list[dict[str, Any]] = [
    # Hydrogen Lyman / Balmer series
    {"name": r"$\mathrm{Ly\alpha}$", "rest_wave": 1215.67, "type": "emission", "color": "#2ca02c"},
    {"name": r"$\mathrm{H\delta}$", "rest_wave": 4101.74, "type": "balmer", "color": "#1f77b4"},
    {"name": r"$\mathrm{H\gamma}$", "rest_wave": 4340.47, "type": "balmer", "color": "#1f77b4"},
    {"name": r"$\mathrm{H\beta}$", "rest_wave": 4861.33, "type": "balmer", "color": "#1f77b4"},
    {"name": r"$\mathrm{H\alpha}$", "rest_wave": 6562.82, "type": "balmer", "color": "#d62728"},
    # Nebular emission lines (AGN / Star-forming)
    {"name": r"$[\mathrm{O\,II}]$", "rest_wave": 3727.09, "type": "forbidden", "color": "#9467bd"},
    {"name": r"$[\mathrm{O\,III}]\,4959$", "rest_wave": 4958.91, "type": "forbidden", "color": "#8c564b"},
    {"name": r"$[\mathrm{O\,III}]\,5007$", "rest_wave": 5006.84, "type": "forbidden", "color": "#8c564b"},
    {"name": r"$[\mathrm{O\,I}]\,6300$", "rest_wave": 6300.30, "type": "forbidden", "color": "#e377c2"},
    {"name": r"$[\mathrm{N\,II}]\,6583$", "rest_wave": 6583.45, "type": "forbidden", "color": "#bcbd22"},
    {"name": r"$[\mathrm{S\,II}]\,6716$", "rest_wave": 6716.44, "type": "forbidden", "color": "#17becf"},
    {"name": r"$[\mathrm{S\,II}]\,6731$", "rest_wave": 6730.82, "type": "forbidden", "color": "#17becf"},
    # UV / Quasar lines
    {"name": r"$\mathrm{C\,IV}$", "rest_wave": 1549.06, "type": "uv", "color": "#ff7f0e"},
    {"name": r"$\mathrm{C\,III]}$", "rest_wave": 1908.73, "type": "uv", "color": "#ff7f0e"},
    {"name": r"$\mathrm{Mg\,II}$", "rest_wave": 2798.75, "type": "uv", "color": "#e377c2"},
    {"name": r"$\mathrm{He\,II}$", "rest_wave": 4685.70, "type": "emission", "color": "#2ca02c"},
    # Stellar absorption features
    {"name": r"$\mathrm{Ca\,II\,K}$", "rest_wave": 3933.66, "type": "absorption", "color": "#7f7f7f"},
    {"name": r"$\mathrm{Ca\,II\,H}$", "rest_wave": 3968.47, "type": "absorption", "color": "#7f7f7f"},
    {"name": r"$\mathrm{G\text{-}band}$", "rest_wave": 4304.40, "type": "absorption", "color": "#7f7f7f"},
    {"name": r"$\mathrm{Mg\,I\,b}$", "rest_wave": 5175.40, "type": "absorption", "color": "#7f7f7f"},
    {"name": r"$\mathrm{Na\,I\,D}$", "rest_wave": 5892.94, "type": "absorption", "color": "#7f7f7f"},
]

_PLOT_LOCK = threading.Lock()

def plot_spectrum(
    spectrum_dict: dict,
    redshift: float | None = None,
    dataset_name: str = "sdss",
    dpi: int = 150,
) -> bytes:
    """Render a 1D astronomical spectrum into PNG image bytes with redshifted line overlays.

    Args:
        spectrum_dict: Dictionary with 'lambda', 'flux', and optionally 'mask'/'ivar'.
        redshift: Measured spectroscopic redshift (z). If provided, rest-frame lines
            are redshifted to observed wavelengths and marked with labeled dashed lines.
        dataset_name: Name of the survey/dataset (e.g., 'sdss' or 'desi').
        dpi: DPI resolution for the generated PNG image.

    Returns:
        PNG image bytes.

    Raises:
        ValueError: If no valid spectral data points exist in the input dictionary.
    """
    raw_wave = spectrum_dict.get("lambda")
    raw_flux = spectrum_dict.get("flux")

    if raw_wave is None or raw_flux is None:
        raise ValueError("Spectrum dictionary must contain 'lambda' and 'flux'.")

    wave = np.asarray(raw_wave, dtype=np.float64)
    flux = np.asarray(raw_flux, dtype=np.float64)

    # Filter invalid points (unobserved padding wave <= 0 or NaNs/Infs).
    valid = (wave > 0) & np.isfinite(wave) & np.isfinite(flux)

    # Apply mask if present.
    raw_mask = spectrum_dict.get("mask")
    if raw_mask is not None:
        mask = np.asarray(raw_mask, dtype=bool)
        if len(mask) == len(wave):
            valid = valid & (~mask)

    if np.sum(valid) < 10:
        raise ValueError(
            f"Insufficient valid spectral points ({np.sum(valid)} valid out of {len(wave)})."
        )

    wave_valid = wave[valid]
    flux_valid = flux[valid]

    # Robust Y-axis bounds using percentile clipping to prevent cosmic rays/bad pixels
    # from compressing the visible spectral features.
    p_low = float(np.percentile(flux_valid, 0.5))
    p_high = float(np.percentile(flux_valid, 99.5))
    flux_range = max(p_high - p_low, 1e-4)

    ymin = p_low - 0.08 * flux_range
    # Leave extra headroom at the top for vertical line labels
    ymax = p_high + 0.35 * flux_range

    with _PLOT_LOCK:
        fig = Figure(figsize=(10, 4.5), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot spectral flux curve
        ax.plot(
            wave_valid,
            flux_valid,
            color="#203a43",
            linewidth=0.85,
            alpha=0.95,
            label="Flux",
        )

        w_min = float(wave_valid.min())
        w_max = float(wave_valid.max())

        # Add redshifted rest-frame spectral lines if redshift is available
        if redshift is not None and not np.isnan(redshift) and redshift > -0.01:
            z_factor = 1.0 + redshift
            label_y_top = ymax - 0.04 * (ymax - ymin)

            for line in REST_FRAME_LINES:
                obs_wave = line["rest_wave"] * z_factor
                if w_min <= obs_wave <= w_max:
                    color = line.get("color", "#7f7f7f")
                    ax.axvline(
                        obs_wave,
                        color=color,
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.6,
                    )
                    ax.text(
                        obs_wave,
                        label_y_top,
                        line["name"],
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="center",
                        fontsize=7.5,
                        color=color,
                        alpha=0.9,
                        clip_on=True,
                    )

        # Formatting axes & labels
        ax.set_xlim(w_min, w_max)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"Observed Wavelength ($\mathrm{\AA}$)", fontsize=10)
        ax.set_ylabel(
            r"Flux ($10^{-17}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)",
            fontsize=10,
        )

        z_str = f"z = {redshift:.4f}" if redshift is not None and not np.isnan(redshift) else "z = N/A"
        ax.set_title(
            f"1D Optical/Near-IR Spectrum ({z_str})",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, linestyle=":", alpha=0.35, color="#888888")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        
    buf.seek(0)
    return buf.getvalue()
