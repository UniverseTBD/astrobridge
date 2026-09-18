"""Loader + RA/Dec crossmatch for `astronolan/gz-decals-embeddings` — the real Galaxy Zoo DECaLS
crowd-vote catalog (24,162 objects, confirmed live: every decision-tree question's `_fraction`
AND `_debiased` columns, plus `.mask` reliability flags, plus `ra`/`dec`/`iauname`). This is what
`eval/metrics/vote_fraction_scoring.py` scores against — that module only does pure math over an
already-joined row; this file is where the join with `galaxy10-aion`'s own `ra`/`dec` happens.

Same author ("astronolan") as `astronolan/galaxy10-aion`, but a genuinely different dataset with
no shared id column — `galaxy10-aion` has `Galaxy10_DECals_index`, this one has `iauname`/
`object_id`, neither of which cross-references the other directly. Only `ra`/`dec` are common to
both, so the join is a small-radius sky coordinate crossmatch (same kind of join
`configs/data.yaml`'s `join.fallback_radius_arcsec` already uses elsewhere in this project for a
different pair of tables) via `astropy.coordinates.SkyCoord`, not a plain merge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics.vote_fraction_scoring import _RULES

GZ_DECALS_VOTES_HF_PATH = "astronolan/gz-decals-embeddings"

# Every debiased column any Galaxy10 class rule actually reads, plus its `.mask` twin — derived
# from _RULES directly (not hand-copied) so a rule change here automatically pulls whatever
# columns it now needs, rather than silently missing one.
_VOTE_COLUMNS = sorted({
    col for alternatives_list in _RULES.values() for alternatives in alternatives_list for col, _, _ in alternatives
})
_MASK_COLUMNS = [f"{col}.mask" for col in _VOTE_COLUMNS]
_ID_COLUMNS = ["ra", "dec", "iauname"]


def _parquet_paths(hf_path: str = GZ_DECALS_VOTES_HF_PATH) -> list[str]:
    from huggingface_hub import list_repo_files

    files = sorted(
        f for f in list_repo_files(hf_path, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No 'data/*.parquet' files found in {hf_path!r} — check the repo layout hasn't changed.")
    return files


def load_vote_fractions(hf_path: str = GZ_DECALS_VOTES_HF_PATH, cache_dir: Path | None = None) -> pd.DataFrame:
    """Only downloads/reads the columns `vote_fraction_scoring._RULES` actually needs (plus
    `ra`/`dec`/`iauname`) — not all ~230 columns in the real schema, most of which
    (`_fraction`-without-`_debiased`, embeddings, photometry) this eval bench never touches.
    """
    from huggingface_hub import hf_hub_download

    import pyarrow.parquet as pq

    columns = _ID_COLUMNS + _VOTE_COLUMNS + _MASK_COLUMNS
    frames = []
    for f in _parquet_paths(hf_path):
        local_path = hf_hub_download(
            repo_id=hf_path, filename=f, repo_type="dataset", cache_dir=str(cache_dir) if cache_dir else None,
        )
        # Not every shard necessarily carries every mask column with data (an all-null mask
        # column would still be present in the schema) — read defensively with the columns that
        # are actually present in this shard, then reindex to the full set so concat aligns.
        available = [c for c in columns if c in pq.ParquetFile(local_path).schema_arrow.names]
        table = pq.read_table(local_path, columns=available)
        frames.append(table.to_pandas(ignore_metadata=True).reindex(columns=columns))
    return pd.concat(frames, ignore_index=True, sort=False)


def crossmatch_to_sample(sample: pd.DataFrame, votes: pd.DataFrame, radius_arcsec: float = 1.0) -> pd.DataFrame:
    """For each row in `sample` (must have `ra`/`dec` columns, degrees), finds the nearest
    `votes` row within `radius_arcsec` and returns `sample` with the vote/mask columns joined in.
    Rows with no match within the radius get `NaN` for the debiased vote columns and `True`
    (masked/unreliable — never a plain `NaN`, since these are real bool columns and "no crossmatch
    at all" is at least as unreliable as "matched but low vote count") for every `.mask` column.
    Both cases already make `vote_fraction_scoring.class_margin` treat that column as unscoreable
    — no separate handling needed downstream for "did this object even crossmatch."
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    sample_coords = SkyCoord(ra=sample["ra"].to_numpy() * u.deg, dec=sample["dec"].to_numpy() * u.deg)
    vote_coords = SkyCoord(ra=votes["ra"].to_numpy() * u.deg, dec=votes["dec"].to_numpy() * u.deg)

    idx, sep2d, _ = sample_coords.match_to_catalog_sky(vote_coords)
    within_radius = sep2d.arcsec <= radius_arcsec

    joined_columns = _VOTE_COLUMNS + _MASK_COLUMNS + ["iauname"]
    matched = votes.iloc[idx][joined_columns].reset_index(drop=True)
    # Built via np.where into a fresh column, not `.loc[...] = value` in place — a mask column
    # can arrive as non-bool dtype (e.g. `load_vote_fractions`'s defensive `reindex` fills an
    # entirely-missing column with float NaN), and assigning `True` into a float64 column raises
    # `TypeError: Invalid value 'True' for dtype 'float64'` rather than silently coercing. Confirmed
    # real by hitting exactly this in testing, not a hypothetical.
    for col in _VOTE_COLUMNS + ["iauname"]:
        matched[col] = matched[col].where(within_radius, other=None)
    for col in _MASK_COLUMNS:
        matched[col] = pd.Series(np.where(within_radius, matched[col].astype(object), True), index=matched.index)

    result = sample.reset_index(drop=True).copy()
    for col in joined_columns:
        result[col] = matched[col]
    result["_crossmatch_separation_arcsec"] = sep2d.arcsec
    result["_crossmatched"] = within_radius
    return result
