"""Strategy 3: Combined (Plot + Quotes) captioning.

Plots 1D optical/near-IR spectra and extracts paper quotes. Prompts Gemini
with the rendered image, the measured redshift, and the quotes to generate
captions. The plot acts as the ground truth, while quotes provide enrichment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from spectra_captioning.data.grouping import extract_quotes, get_closest_observation
from spectra_captioning.data.plotting import plot_spectrum
from spectra_captioning.data.preprocessing import clean_quotes
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
class CombinedStrategy(CaptionStrategy):
    """Generate captions by inspecting plotted 1D spectra enriched with quotes."""

    def __init__(self, gemini_client: GeminiClient):
        self._client = gemini_client
        self._env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            keep_trailing_newline=True,
        )
        self._template = self._env.get_template("combined.jinja2")

    @property
    def strategy_name(self) -> str:
        return "combined_v2"

    def _get_template_context(
        self,
        object_key: str,
        group_df: pd.DataFrame,
        redshift: float | None,
        redshift_description: str | None,
        cleaned_quotes: list[str],
    ) -> dict:
        """Construct the template variables for rendering the prompt."""
        return {
            "redshift": redshift,
            "redshift_description": redshift_description,
            "quotes": cleaned_quotes,
        }

    def generate_caption(
        self, object_key: str, group_df: pd.DataFrame, dataset: str, config: dict
    ) -> CaptionResult:
        """Generate a caption using both the plot and literature quotes.

        Steps:
        1. Extract and clean quotes from the group dataframe.
        2. Select the closest observation row for this object ID to get spectra data.
        3. Plot the spectrum with redshifted line overlays into PNG image bytes.
        4. Render the combined Jinja2 prompt template with redshift and quotes.
        5. Send the prompt and plot image to Gemini.
        6. Return a CaptionResult.
        """
        # 1. Extract Quotes
        all_quotes = extract_quotes(group_df)
        cleaned_quotes = clean_quotes(all_quotes)

        # 2. Extract Spectral Data
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

        # 3. Plot Spectrum
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
            plot_path = plot_dir / f"{object_key}_{dataset}_combined.png"
            plot_path.write_bytes(image_bytes)
            logger.debug("Saved spectrum plot to %s", plot_path)

        redshift_description = None
        if redshift is not None:
            redshift_description = get_qualitative_distance(redshift)

        # 4. Render Prompt
        context = self._get_template_context(
            object_key, group_df, redshift, redshift_description, cleaned_quotes
        )
        prompt = self._template.render(**context)

        logger.debug(
            "Generating combined caption for object %s (z=%s, %d quotes, image=%d bytes)...",
            object_key,
            f"{redshift:.4f}" if redshift is not None else "N/A",
            len(cleaned_quotes),
            len(image_bytes),
        )

        # 5. Call Gemini
        response = self._client.generate(prompt, images=[image_bytes])

        logger.debug(
            "Combined caption generated for %s: %d tokens used.",
            object_key,
            response.total_tokens,
        )

        return CaptionResult.from_gemini_response(response, prompt_used=prompt)
