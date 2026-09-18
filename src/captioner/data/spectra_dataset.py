"""Loader for `UniverseTBD/AstroBridge-Data`, working around a real schema-union failure in
`datasets.load_dataset(hf_path, split="train")` against this repo — confirmed against a real
run, not a guess:

    datasets.table.CastError: Couldn't cast ... because column names don't match

The repo's `observations/spectra/` folder holds four separate parquet files, one per crossmatch
source (DESI-only, DESI+SDSS, DESI+SDSS-subset, SDSS-only). `datasets`'s automatic multi-file
loading tries to cast every file into one strict Arrow schema inferred up front, and fails,
because the SDSS-crossmatched files carry extra `survey`/`survey_metadata` columns the DESI-only
file doesn't have. The core columns this codebase actually reads (`object_id`,
`ra_spectra`/`dec_spectra`, `spectrum`, `Z`/`ZERR`/`ZWARN`, `mention_summary`/`evidence_quotes`/
`arxiv_id`, etc.) are present in every file — only the SDSS-specific extras differ.

Loading each file separately with pandas and `pd.concat`ing sidesteps the cast entirely:
`pd.concat` fills a column missing from one file with NaN for those rows instead of requiring an
exact schema match across all files, which is exactly the flexibility `datasets` doesn't give us
here.

Separately: these files' `spectrum` column is a nested struct (flux/ivar/lsf_sigma/lambda/mask —
the same shape of thing as the image side's `image_legacy`, which hit a real crash reading via
plain `pd.read_parquet`: pyarrow's pandas-metadata-driven dtype restoration chokes on a nested
struct dtype string `numpy.dtype()` can't parse). Reading via `pyarrow.parquet` directly with
`ignore_metadata=True` avoids that class of failure — see data/image_dataset.py's
`_read_parquet_columns` for the full explanation; used here defensively for the same reason.

A third real issue, also confirmed against a real run: the four files are NOT disjoint by
`object_id` — a target can appear in more than one crossmatch-source file (e.g. `..._subset...`
looks like a literal subset of the non-subset file, and a target with both DESI and SDSS spectra
plausibly appears in more than one of the survey-specific files too). `pd.concat` alone leaves
duplicate `object_id` rows in the result, which breaks every downstream `set_index("object_id")`
lookup — one of them (`DataFrame.to_dict(orient="index")` in 01_generate_captions.py) raises
`ValueError: DataFrame index must be unique for orient='index'` outright; others (e.g. a plain
`Series.to_dict()`) would have silently kept whichever duplicate happened to sort last instead of
erroring, which is worse. `load_spectra_table` now deduplicates by `object_id` before returning —
preferring the row with the smallest `_dist_arcsec` (the crossmatch quality metric already in
the data) when duplicates disagree on it, since that's an objective tie-break rather than an
arbitrary one based on file processing order.

A fourth real issue, confirmed against the live repo, and the reason `_canonical_object_id`
exists: the four files spell the *same* `object_id` three different ways — plain digit strings,
int64, and a stringified Python bytes repr of a space-padded fixed-width FITS char field (the
literal text `b'    462849895556999168'`). Canonicalizing is what lets the deduplication below
actually collapse them; see that function's docstring for the two failures it caused.

Separately: `load_gemini_spectra_captions` reads a *different*, teammate-produced dataset
(`ibrahimhabibeg/spectra_captions_dataset`) — a JSONL file of LLM-generated (Gemini) captions,
each grounded purely in one object's spectrum (no object names/coordinates in the prompt, per
inspecting real rows), analogous to what `caption_blind` is for images: a pre-vetted,
modality-restricted caption source, preferred over decomposing `mention_summary` ourselves.
Confirmed real shape: one JSON object per line, `object_key` (join key) plus
`output.caption`/`output.is_insufficient`/`output.thought_summaries` (verbose chain-of-thought,
discarded — never used as caption text). Rows with `is_insufficient=True` (the model itself
flagged not enough evidence) are dropped. `object_key` joins against `wiki_entity_id` — confirmed
present directly on `UniverseTBD/AstroBridge-Data` rows (alongside `object_id`), so no extra
crossmatch file is needed; `01_generate_captions.py` builds the `wiki_entity_id -> object_id`
map from the `spectra_df` it already loads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from captioner.utils.logging import get_logger

logger = get_logger(__name__)

SPECTRA_DIR_PREFIX = "observations/spectra/"


def _read_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pandas(ignore_metadata=True)


def _attach_survey_column(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Populates `survey` ("desi"/"sdss" per row) — required by aion_spectrum.py to route each
    object to AION's matching DESISpectrum/SDSSSpectrum modality class (see that file's
    docstring for why the distinction is real and not optional). A filename containing only one
    survey name (e.g. `desi_crossmatch_...parquet`) determines every row's survey directly; a
    mixed filename (e.g. `desi_sdss_crossmatch_...parquet`) must rely on that file's own
    per-row `survey` column instead — inventing a value would silently mislabel real spectra.
    """
    stem = filename.lower()
    has_desi, has_sdss = "desi" in stem, "sdss" in stem
    df = df.copy()

    if has_desi and not has_sdss:
        df["survey"] = "desi"
    elif has_sdss and not has_desi:
        df["survey"] = "sdss"
    elif "survey" in df.columns:
        df["survey"] = df["survey"].str.lower()
        if df["survey"].isna().any():
            raise ValueError(
                f"{filename!r} mixes DESI/SDSS in its filename and has a `survey` column, but "
                f"{int(df['survey'].isna().sum())} rows have no value in it — no safe default "
                "to fall back to for which survey those rows came from."
            )
    else:
        reason = (
            "mixes DESI/SDSS in its filename"
            if (has_desi and has_sdss)
            else "has neither 'desi' nor 'sdss' in its filename"
        )
        raise ValueError(
            f"{filename!r} {reason} and has no `survey` column to determine each row's survey "
            "origin from — check the real schema for this file before adding it to "
            "SPECTRA_DIR_PREFIX's file discovery. No safe default to guess from here."
        )
    return df


def load_spectra_table(
    hf_path: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
    files: list[str] | None = None,
) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download, list_repo_files

    if files is None:
        selected = [
            f
            for f in list_repo_files(hf_path, repo_type="dataset", revision=revision)
            if f.startswith(SPECTRA_DIR_PREFIX) and f.endswith(".parquet")
        ]
        if not selected:
            raise FileNotFoundError(
                f"No parquet files found under {SPECTRA_DIR_PREFIX!r} in {hf_path!r} — check the "
                "repo layout hasn't changed."
            )
    else:
        selected = [str(f) for f in files]
        logger.info(f"Reading {len(selected)} pinned spectra file(s): {selected}")

    frames = []
    for f in selected:
        local_path = hf_hub_download(
            repo_id=hf_path,
            filename=f,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        frames.append(_attach_survey_column(_read_parquet(local_path), Path(f).name))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _canonicalize_object_ids(combined)
    combined = _propagate_upstream_split(combined)
    return _deduplicate_by_object_id(combined)


_BYTES_REPR_RE = re.compile(r"""b(['"])(?P<value>.*)\1""", re.DOTALL)


def _canonical_object_id(value) -> str:
    """One object, one id string — whichever source file the row came from.

    Confirmed against the real repo, not a guess: the four spectra parquet files spell the same
    `object_id` three different ways. `desi_crossmatch_...` stores plain digit strings
    (`'39632951360620188'`); `desi_sdss_subset_crossmatch_...` stores int64
    (`462849895556999168`); and `sdss_crossmatch_...` / `desi_sdss_crossmatch_...` store a
    *stringified Python bytes repr* of a space-padded fixed-width FITS char field — the literal
    text `b'    462849895556999168'`. All 15,198 rows across the four files reduce to 2,178 real
    objects, every one present under two spellings whose ra/dec/name/wiki_entity_id agree exactly,
    so collapsing them loses nothing.

    Left unnormalized those spellings are distinct keys, so `_deduplicate_by_object_id` cannot
    collapse them and the manifest ends up with two rows per object, which `manifest.py`'s
    `astype(str)` then freezes. Two real failures follow. `make cache` dies with
    `KeyError: '299516890357721088'`: the manifest asks for the stringified id while
    02_cache_embeddings.py's `spectra_df.set_index("object_id").to_dict()` is still keyed by the
    raw int. And every spectra object appears twice under ids that look unrelated, so
    `assign_splits` can put one physical object in both train and val without tripping its own
    per-object uniqueness assertion — a leak that would quietly flatter val loss.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace").strip()
    text = str(value).strip()
    match = _BYTES_REPR_RE.fullmatch(text)
    return match.group("value").strip() if match else text


def _canonicalize_object_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Applies `_canonical_object_id` to the whole column, before deduplication runs.

    A null `object_id` is rejected rather than canonicalized: `str(nan)` is `'nan'`, so a file
    missing the column entirely (pd.concat fills it with NaN) would otherwise collapse every one
    of its rows onto a single bogus object rather than failing.
    """
    n_missing = int(df["object_id"].isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} of {len(df)} spectra rows have no `object_id` — every row needs one to "
            "be keyed, joined and cached by. Check whether a source file under "
            f"{SPECTRA_DIR_PREFIX!r} is missing the column altogether."
        )

    df = df.copy()
    canonical = df["object_id"].map(_canonical_object_id)
    n_rewritten = int((canonical != df["object_id"].astype(str)).sum())
    n_spellings, n_objects = df["object_id"].nunique(), canonical.nunique()
    df["object_id"] = canonical

    if n_rewritten or n_spellings != n_objects:
        logger.info(
            f"Canonicalized object_id across the source files: {n_rewritten}/{len(df)} rows were "
            f"stored as a stringified bytes repr (b'...') or padded, and {n_spellings} distinct "
            f"raw spellings resolve to {n_objects} distinct objects. Without this the same object "
            "reaches the manifest under two unrelated-looking ids."
        )
    return df


UPSTREAM_SPLIT_COLUMN = "split"


def _propagate_upstream_split(df: pd.DataFrame) -> pd.DataFrame:
    if UPSTREAM_SPLIT_COLUMN not in df.columns:
        return df

    df = df.copy()
    resolved = df[UPSTREAM_SPLIT_COLUMN].groupby(df["object_id"]).first()
    filled = df["object_id"].map(resolved)
    n_recovered = int(filled.notna().sum() - df[UPSTREAM_SPLIT_COLUMN].notna().sum())
    df[UPSTREAM_SPLIT_COLUMN] = filled

    logger.info(
        f"Upstream {UPSTREAM_SPLIT_COLUMN!r} labels: {int(filled.notna().sum())}/{len(df)} rows "
        f"covering {int(resolved.notna().sum())}/{resolved.size} objects "
        f"({n_recovered} rows filled in from another file's row for the same object)."
    )
    return df


def _deduplicate_by_object_id(df: pd.DataFrame) -> pd.DataFrame:
    n_dupe_rows = int(df["object_id"].duplicated(keep=False).sum())
    if n_dupe_rows == 0:
        return df

    n_dupe_objects = df.loc[df["object_id"].duplicated(keep=False), "object_id"].nunique()
    if "_dist_arcsec" in df.columns:
        # Prefer the closest crossmatch when the same object appears in more than one source
        # file — an objective tie-break already present in the data, not an arbitrary one based
        # on which file happened to be listed/processed first.
        df = df.sort_values("_dist_arcsec", na_position="last")
        tie_break = "smallest _dist_arcsec"
    else:
        tie_break = "first occurrence (no _dist_arcsec column to break ties on)"

    deduped = df.drop_duplicates(subset="object_id", keep="first").reset_index(drop=True)
    logger.warning(
        f"{n_dupe_rows} rows across {n_dupe_objects} object_ids are not one-row-per-object — a "
        f"target with more than one spectrum (DESI and SDSS both observed it), and, when more "
        f"than one file is read, the same target repeated across files — kept one row per "
        f"object_id, tie-broken "
        f"by {tie_break}. This affects which mention_summary/spectrum version each duplicated "
        f"object ends up with; worth spot-checking a few of them if caption quality looks off "
        f"for objects that had duplicates."
    )
    return deduped


def load_gemini_spectra_captions(
    hf_path: str, filename: str, revision: str | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    """One row per object with a usable Gemini-generated spectrum caption: `wiki_entity_id`,
    `caption`. Drops rows where the model itself flagged `is_insufficient=True` (not enough
    evidence to caption confidently) and rows missing `object_key`/`caption`. Everything else in
    the record (`thought_summaries`, `usage`, `provenance`, top-level `ra`/`dec`/`model`/etc.) is
    metadata we don't need here and is discarded.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=hf_path,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    rows = []
    n_insufficient = 0
    n_missing = 0
    with open(local_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            object_key = rec.get("object_key")
            output = rec.get("output") or {}
            if output.get("is_insufficient"):
                n_insufficient += 1
                continue
            caption = output.get("caption")
            if not object_key or not caption:
                n_missing += 1
                continue
            rows.append({"wiki_entity_id": object_key, "caption": caption})

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} usable Gemini spectra captions from {filename} "
        f"({n_insufficient} marked is_insufficient, {n_missing} missing object_key/caption, "
        "both skipped)."
    )
    return df


def trimmed_spectrum_arrays(spectrum) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One object's `(flux, wavelength, ivar, mask)` with the wavelength grid made STRICTLY
    INCREASING — drop `wavelength <= 0`, then sort ascending.

    This is not defensive tidying, it is the difference between AION seeing the spectrum and
    AION seeing noise. AstroBridge-Data pads `spectrum.lambda` with a `-1` sentinel at the end of
    the array on 94% of SDSS rows (282/300 sampled), so the raw grid looks like
    `[3789.7, 3790.5, ..., 9219.3, -1, -1, -1, ...]`. AION's codec projects onto its own latent
    grid via `LatentSpectralGrid.to_latent` -> `interp1d` -> `torch.searchsorted(wavelength,
    target)`, and `searchsorted` REQUIRES a sorted array — on unsorted input its result is
    undefined, so every spectrum interpolated to near-identical garbage and AION's VQ tokenizer
    collapsed. Measured on 700 real SDSS spectra: a quasar and an early-type galaxy shared 99.6%
    of their 273 token codes, and a linear class probe on the resulting embedding scored 0.356
    against a 0.401 majority baseline (i.e. below chance). With this trim: 9.8% shared codes and
    0.660 linear / 0.693 MLP. See DIAGNOSIS.md and diag/fix_test.py for the full before/after.

    The sentinel positions are already `mask=True` in the source data, but the mask never reaches
    the interpolation — AION's codec uses it only for the flux normalization factor — so masking
    alone does not help. The rows have to go.
    """
    flux = np.asarray(spectrum["flux"], dtype=np.float32)
    wavelength = np.asarray(spectrum["lambda"], dtype=np.float32)
    ivar = np.asarray(spectrum["ivar"], dtype=np.float32)
    mask = np.asarray(spectrum["mask"], dtype=bool)

    keep = np.isfinite(wavelength) & (wavelength > 0)
    flux, wavelength, ivar, mask = flux[keep], wavelength[keep], ivar[keep], mask[keep]

    order = np.argsort(wavelength, kind="stable")
    return flux[order], wavelength[order], ivar[order], mask[order]


def spectrum_group_key(spectrum) -> int:
    """Shard-partition key: the object's TRIMMED length. Objects only share a shard when they
    need no cross-object padding at all.

    Without this, `cache_modality` sharded spectra in plain manifest order and every shard mixed
    DESI (7781 samples) with SDSS (~3845-3883) — 84.7% of cached objects (1844/2178) were padded
    out to the longest member. Measured effect on the embedding was small (3.4%), but it makes an
    object's embedding depend on which shard it happened to land in, which is a reproducibility
    hazard and exactly the kind of thing that hid the sentinel bug above.
    """
    return int(len(trimmed_spectrum_arrays(spectrum)[0]))
