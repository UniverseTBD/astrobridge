"""Strategy 4: Combined Ground Truth captioning.

An extension of the CombinedStrategy that incorporates authoritative
ground-truth metadata (classification, subclass, redshift, and measured emission lines).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from spectra_captioning.data.grouping import get_closest_observation
from spectra_captioning.models.gemini import GeminiClient
from spectra_captioning.strategies.base import register_strategy
from spectra_captioning.strategies.combined import CombinedStrategy

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

# Mapping for common emission line names
_LINE_MAPPINGS = {
    "LYALPHA": "Lyα",
    "NV_1240": "N V 1240",
    "OI_1304": "O I 1304",
    "SILIV_1396": "Si IV 1396",
    "CIV_1549": "C IV 1549",
    "HEII_1640": "He II 1640",
    "ALIII_1857": "Al III 1857",
    "SILIII_1892": "Si III 1892",
    "CIII_1908": "C III] 1908",
    "MGII_2796": "Mg II 2796",
    "MGII_2803": "Mg II 2803",
    "NEV_3346": "[Ne V] 3346",
    "NEV_3426": "[Ne V] 3426",
    "OII_3726": "[O II] 3726",
    "OII_3729": "[O II] 3729",
    "NEIII_3869": "[Ne III] 3869",
    "H6": "Hδ",
    "HEPSILON": "Hε",
    "HDELTA": "Hδ",
    "HGAMMA": "Hγ",
    "OIII_4363": "[O III] 4363",
    "HEI_4471": "He I 4471",
    "HEII_4686": "He II 4686",
    "HBETA": "Hβ",
    "OIII_4959": "[O III] 4959",
    "OIII_5007": "[O III] 5007",
    "NII_5755": "[N II] 5755",
    "HEI_5876": "He I 5876",
    "OI_6300": "[O I] 6300",
    "SIII_6312": "[S III] 6312",
    "NII_6548": "[N II] 6548",
    "HALPHA": "Hα",
    "NII_6584": "[N II] 6584",
    "SII_6716": "[S II] 6716",
    "SII_6731": "[S II] 6731",
    "ARIII_7135": "[Ar III] 7135",
    "OII_7320": "[O II] 7320",
    "OII_7330": "[O II] 7330",
    "SIII_9069": "[S III] 9069",
    "SIII_9532": "[S III] 9532",
}

def format_line_name(name: str) -> str:
    """Format an uppercase line name from the CSV into a readable string."""
    # Handle broad lines like HALPHA_BROAD
    is_broad = False
    base_name = name
    if name.endswith("_BROAD"):
        is_broad = True
        base_name = name.replace("_BROAD", "")
        
    mapped = _LINE_MAPPINGS.get(base_name, base_name)
    if is_broad:
        return f"{mapped} (broad)"
    return mapped


@register_strategy
class CombinedGroundTruthStrategy(CombinedStrategy):
    """Generate captions by inspecting plotted 1D spectra enriched with quotes and metadata."""

    def __init__(self, gemini_client: GeminiClient):
        super().__init__(gemini_client)
        self._template = self._env.get_template("combined_ground_truth.jinja2")

        types_path = _DATA_DIR / "extracted_types.csv"
        lines_path = _DATA_DIR / "extracted_emission_lines.csv"

        if types_path.exists():
            self._types_df = pd.read_csv(types_path)
            # Make sure missing values are properly converted to None
            self._types_df = self._types_df.where(pd.notnull(self._types_df), None)
            self._types_df = self._types_df.drop_duplicates(subset=["wiki_entity_id"])
            self._types_dict = self._types_df.set_index("wiki_entity_id").to_dict("index")
            logger.debug(f"Loaded {len(self._types_dict)} classification rows from {types_path.name}")
        else:
            self._types_dict = {}
            logger.warning("extracted_types.csv not found")

        if lines_path.exists():
            self._lines_df = pd.read_csv(lines_path)
            
            # Extract list of (LINE_NAME, SNR) tuples for each wiki_entity_id
            self._lines_dict = {}
            for wiki_id, group in self._lines_df.groupby("wiki_entity_id"):
                lines = []
                for _, row in group.iterrows():
                    lines.append((row["LINE_NAME"], row["SNR"]))
                
                # Sort by SNR descending
                lines.sort(key=lambda x: x[1], reverse=True)
                
                # Format strings including SNR
                formatted_lines = [
                    f"{format_line_name(lname)} (SNR: {snr:.1f})" 
                    for lname, snr in lines
                ]
                self._lines_dict[wiki_id] = formatted_lines
            logger.debug(f"Loaded emission lines for {len(self._lines_dict)} objects from {lines_path.name}")
        else:
            self._lines_dict = {}
            logger.warning("extracted_emission_lines.csv not found")

    @property
    def strategy_name(self) -> str:
        return "combined_ground_truth_v1"

    def _get_template_context(
        self,
        object_key: str,
        group_df: pd.DataFrame,
        redshift: float | None,
        redshift_description: str | None,
        cleaned_quotes: list[str],
    ) -> dict:
        """Construct the template variables for rendering the prompt."""
        context = super()._get_template_context(
            object_key, group_df, redshift, redshift_description, cleaned_quotes
        )
        
        type_info = self._types_dict.get(object_key)
        class_phrase = None
        subclass_phrase = None
        if type_info:
            c = type_info.get("class")
            if pd.notna(c) and c is not None:
                class_phrase = c if c != "QSO" else "Quasar"
                
            sc = type_info.get("subclass")
            if pd.notna(sc) and sc is not None:
                subclass_phrase = sc

        obs_row = get_closest_observation(group_df, object_key)
        survey = obs_row.get("survey") if obs_row is not None else None

        desi_lines = None
        if survey == "desi":
            desi_lines = self._lines_dict.get(object_key, [])

        context.update({
            "class_phrase": class_phrase,
            "subclass_phrase": subclass_phrase,
            "desi_lines": desi_lines,
        })
        return context
