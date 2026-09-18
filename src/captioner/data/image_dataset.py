"""Loaders for the image tier's two data shapes — neither is the `datasets.load_dataset`-with-an-
`image`-column shape this codebase originally assumed.

**Caption source** (`load_image_captions_table`): `caption_fused` by default — the dataset's own
final editor pass, which takes the three earlier drafts (blind, properties, literature) as
proposals rather than ground truth and re-examines the pixels to resolve disagreements. The
no-name/no-designation rule is enforced and re-checked on every stage, so it is no more
leak-prone than `caption_blind`. Used directly for the image tier in 01_generate_captions.py
rather than re-derived via our own keyword decomposition. The three earlier stages
(`caption_blind`, `caption_properties`, `caption_literature`) remain available; `caption_field`
selects between them so switching is a config change, not a code change.

(Was `gapatron/legacy_survey_south_images_captions`'s per-object `*_captions.json` files with
`caption_blind`; the new repo is the same author's superset with the fused caption added.)

**Pixel source** (`load_image_flux_identity_table` / `load_image_flux_pixels`):
`legacy_south_all_images.parquet`, a *separate* file in the same repo — confirmed schema via
`pyarrow.parquet.ParquetFile(...).schema`. 2,399 rows, each one an ALREADY crossmatched pair: a
`target_object_id_target` (= AstroBridge-Data's own `object_id` — a direct join key, no
coordinate crossmatch needed for this subset) plus `object_id_legacy`/`ra_legacy`/`dec_legacy`
(the Legacy Survey side's own identity, real decimal degrees) plus `image_legacy`: a list of
per-band structs (`band`, `flux`, `mask`, `ivar`, `psf_fwhm`, `scale`), each `flux`/`mask`/`ivar`
a 2D (H, W) array. This is real calibrated flux — what AION's LegacySurveyImage actually needs,
unlike the RGB PNGs above. `rgb_legacy` (bytes/path) is also present but unused, same reason.
This table's own `ra`/`dec` are what manifest.py actually joins on — real decimal degrees, no
coordinate parsing needed, unlike the caption source above.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FLUX_PARQUET_FILENAME = "legacy_south_all_images.parquet"
IMAGE_CAPTION_COLUMN = "caption_fused"


def load_image_captions_table(
    hf_path: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
    caption_field: str = "caption_fused",
    surveys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per object: `object_id`, `caption` (from `caption_fused`). Only these two columns
    are read from the parquet shards — the flux columns in the same file are ignored (the AION
    pixel source stays `legacy_south_all_images.parquet`). scripts/01_generate_captions.py looks
    captions up by object id; identity/coordinates for the manifest come from the flux parquet.
    A row with an empty `caption_fused` is dropped — nothing to caption the image tier with.
    """
    from huggingface_hub import list_repo_files

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet_files = sorted(
        f for f in list_repo_files(hf_path, repo_type="dataset", revision=revision)
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No data/*.parquet files in {hf_path!r} — check the repo layout and that gated "
            "access has been granted."
        )

    frames = []
    for f in parquet_files:
        local = hf_hub_download(
            repo_id=hf_path, filename=f, repo_type="dataset", revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        table = pq.read_table(local, columns=["object_id", IMAGE_CAPTION_COLUMN])
        frames.append(table.to_pandas(ignore_metadata=True))

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={IMAGE_CAPTION_COLUMN: "caption"})
    df["caption"] = df["caption"].astype("string").str.strip()
    df = df.dropna(subset=["object_id", "caption"])
    df = df[df["caption"].str.len() > 0].drop_duplicates(subset="object_id").reset_index(drop=True)
    return df


def _download_flux_parquet(
    hf_path: str, revision: str | None, filename: str, cache_dir: Path | None
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=hf_path,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def _read_parquet_columns(path: str, columns: list[str]) -> pd.DataFrame:
    """Confirmed against a real run: `pd.read_parquet(path, columns=[...])` on this file raises
    inside pyarrow's pandas-metadata dtype restoration —
    `ValueError: format number 1 of "nested<band: [string], flux: [...], ...>" is not recognized`
    — even for `image_legacy`, a column NOT in the requested subset. The file's embedded pandas
    metadata describes that column's dtype in a form `numpy.dtype()` can't parse, and pyarrow
    tries to apply it regardless of column projection. We don't need that metadata-driven dtype
    restoration for these plain identity columns — reading via `pyarrow.parquet` directly and
    passing `ignore_metadata=True` to `to_pandas()` skips it and lets Arrow's own types drive
    dtype inference instead, which is what actually avoids the crash.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return table.to_pandas(ignore_metadata=True)


def load_image_flux_identity_table(
    hf_path: str,
    revision: str | None = None,
    filename: str = FLUX_PARQUET_FILENAME,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Lightweight identity columns only from the flux parquet — NOT `image_legacy` (the ~1.7GB
    nested flux/ivar/mask arrays) or `rgb_legacy`; see load_image_flux_pixels() for those.
    `object_id` here is `target_object_id_target`, i.e. AstroBridge-Data's own id — this table's
    rows are already-crossmatched pairs, so this becomes a direct join key in manifest.py rather
    than needing coordinate-based matching.
    """
    local_path = _download_flux_parquet(hf_path, revision, filename, cache_dir)
    df = _read_parquet_columns(
        local_path,
        columns=["target_object_id_target", "object_id_legacy", "ra_legacy", "dec_legacy", "_dist_arcsec"],
    )
    df = df.rename(columns={"target_object_id_target": "object_id", "ra_legacy": "ra", "dec_legacy": "dec"})
    df["has_image"] = True
    return df


def load_image_flux_pixels(
    hf_path: str,
    revision: str | None = None,
    filename: str = FLUX_PARQUET_FILENAME,
    cache_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """object_id (= target_object_id_target, matching load_image_flux_identity_table) -> the
    per-band list from `image_legacy` (each entry: band/flux/mask/ivar/psf_fwhm/scale), for
    scripts/02_cache_embeddings.py's batch loader. Loaded once into memory (~1.7GB at today's row
    count) — reasonable for 2,399 rows; revisit if the crossmatch grows much larger.
    """
    local_path = _download_flux_parquet(hf_path, revision, filename, cache_dir)
    df = _read_parquet_columns(local_path, columns=["target_object_id_target", "image_legacy"])
    return dict(zip(df["target_object_id_target"], df["image_legacy"]))
