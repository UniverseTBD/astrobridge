"""Strategy 2: Multimodal spectra image captioning.

Plots 1D optical/near-IR spectra (annotated with redshifted rest-frame lines)
and prompts Gemini with the rendered image and measured redshift to generate
captions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from spectra_captioning.data.grouping import get_closest_observation
from spectra_captioning.data.plotting import plot_spectrum
from spectra_captioning.models.gemini import GeminiClient
from spectra_captioning.strategies.base import (
    CaptionResult,
    CaptionStrategy,
    register_strategy,
)
from spectra_captioning.utils import get_qualitative_distance

logger = logging.getLogger(__name__)

# Locate the prompts directory relative to this file.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@register_strategy
class SpectraImageStrategy(CaptionStrategy):
    """Generate captions by inspecting plotted 1D spectra and redshift."""

    def __init__(self, gemini_client: GeminiClient):
        self._client = gemini_client
        self._env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            keep_trailing_newline=True,
        )
        self._template = self._env.get_template("spectra_image.jinja2")

    @property
    def strategy_name(self) -> str:
        return "spectra_image_v3"

    def generate_caption(
        self, object_key: str, group_df: pd.DataFrame, dataset: str, config: dict
    ) -> CaptionResult:
        """Generate a caption from the object's plotted spectrum and redshift.

        Steps:
        1. Select the closest observation row for this object ID.
        2. Extract the spectral array dictionary and redshift from that row.
        3. Plot the spectrum with redshifted line overlays into PNG image bytes.
        4. Render the multimodal Jinja2 prompt template.
        5. Send the prompt and plot image to Gemini.
        6. Return a CaptionResult.
        """
        obs_row = get_closest_observation(group_df, object_key)
        if obs_row is None:
            logger.warning("Object %s has no matching observation row.", object_key)
            return CaptionResult(
                caption="INSUFFICIENT_SPECTRAL_DATA",
                prompt_used="(no observation available)",
            )

        spectrum_dict = obs_row.get("spectrum")
        if not (isinstance(spectrum_dict, dict) and "lambda" in spectrum_dict and "flux" in spectrum_dict):
            logger.warning(
                "Object %s has no valid spectrum dictionary in closest observation row.", object_key
            )
            return CaptionResult(
                caption="INSUFFICIENT_SPECTRAL_DATA",
                prompt_used="(no spectrum available)",
            )

        redshift: float | None = None
        z_val = obs_row.get("Z")
        if z_val is not None and not pd.isna(z_val):
            try:
                redshift = float(z_val)
            except (ValueError, TypeError):
                redshift = None

        try:
            image_bytes = plot_spectrum(
                spectrum_dict,
                redshift=redshift,
                dataset_name=dataset,
            )
        except Exception as exc:
            logger.warning(
                "Failed to render spectrum plot for object %s: %s", object_key, exc
            )
            return CaptionResult(
                caption="INSUFFICIENT_SPECTRAL_DATA",
                prompt_used=f"(plot rendering failure: {exc})",
            )

        # Optionally save plot to disk if configured
        if config.get("captioning", {}).get("save_plots", False):
            plot_dir = Path(config.get("captioning", {}).get("output_dir", "output")) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"{object_key}_{dataset}.png"
            plot_path.write_bytes(image_bytes)
            logger.debug("Saved spectrum plot to %s", plot_path)

        redshift_description = None
        if redshift is not None:
            redshift_description = get_qualitative_distance(redshift)

        prompt = self._template.render(
            redshift=redshift,
            redshift_description=redshift_description,
        )

        logger.debug(
            "Generating multimodal caption for object %s (dataset=%s, z=%s, image=%d bytes)...",
            object_key,
            dataset,
            f"{redshift:.4f}" if redshift is not None else "N/A",
            len(image_bytes),
        )

        response = self._client.generate(prompt, images=[image_bytes])

        logger.debug(
            "Multimodal caption generated for %s: %d tokens used.",
            object_key,
            response.total_tokens,
        )

        return CaptionResult.from_gemini_response(response, prompt_used=prompt)
