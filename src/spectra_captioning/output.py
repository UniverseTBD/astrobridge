"""Output serialization — build and write JSONL records."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from spectra_captioning.data.grouping import extract_quotes, get_closest_observation
from spectra_captioning.strategies.base import CaptionResult

logger = logging.getLogger(__name__)


def build_output_record(
    object_key: str,
    group_df: pd.DataFrame,
    result: CaptionResult,
    strategy_name: str,
    model: str,
    dataset: str,
    config: dict,
) -> dict:
    """Build the full output record for one captioned object.

    This record contains everything needed for downstream use:
    provenance, coordinates, the caption itself, thought summaries,
    and token usage.
    """
    obs_row = get_closest_observation(group_df, object_key)
    
    if obs_row is not None:
        ra_val = obs_row.get("ra_mentions")
        if ra_val is None or pd.isna(ra_val): ra_val = obs_row.get("ra", 0.0)
        dec_val = obs_row.get("dec_mentions")
        if dec_val is None or pd.isna(dec_val): dec_val = obs_row.get("dec", 0.0)
        source = obs_row.get("survey", dataset)
    else:
        ra_val = group_df.iloc[0].get("ra_mentions", group_df.iloc[0].get("ra", 0.0))
        dec_val = group_df.iloc[0].get("dec_mentions", group_df.iloc[0].get("dec", 0.0))
        source = dataset
    
    all_quotes = extract_quotes(group_df)
    
    return {
        "object_key": object_key,
        "dataset_source": source,
        "ra": float(ra_val),
        "dec": float(dec_val),
        "strategy": strategy_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Input summary.
        "input": {
            "num_quotes": len(all_quotes),
            "num_mentions": group_df["mention_id"].nunique() if "mention_id" in group_df else 1,
        },
        # Caption output.
        "output": {
            "caption": result.caption,
            "thought_summaries": result.thought_summaries,
            "is_insufficient": result.caption.strip() == "INSUFFICIENT_SPECTRAL_DATA",
        },
        # Token usage.
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "thought_tokens": result.thought_tokens,
            "total_tokens": result.total_tokens,
        },
        # Provenance for reproducibility.
        "provenance": {
            "crossmatch_radius_arcsec": config["crossmatch"]["radius_arcsec"],
            "prompt_template": strategy_name,
        },
    }


def append_to_jsonl(record: dict, output_path: Path) -> None:
    """Append a single JSON record to a JSONL file.

    Creates the file and parent directories if they don't exist.
    Each record is a single line — this makes it safe to stop and
    resume without losing completed work.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Appended record for %s to %s", record.get("object_key"), output_path)


def generate_output_filename(
    strategy: str, model: str
) -> str:
    """Generate a descriptive output filename.

    Format: ``captions_{strategy}_{model}_{timestamp}.jsonl``
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filesystem.
    safe_model = model.replace("/", "_").replace(".", "_")
    return f"captions_{strategy}_{safe_model}_{ts}.jsonl"
