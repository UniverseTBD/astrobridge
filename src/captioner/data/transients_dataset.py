"""Loader for `BuildNg/astrobridge-transients-dataset` — 987 ZTF BTS supernovae with accepted
alert photometry, keyed by ZTF id.

**A fourth id namespace.** `object_id` here is a ZTF designation (`ZTF18AADMSSD`, always 12 chars),
which shares no namespace with AstroBridge-Data's numeric `object_id`, the Legacy Survey's
`object_id_legacy`, or the Gemini captions' `wiki_entity_id`. These objects join to nothing — they
are appended to the manifest as new rows, never merged (see data/manifest.py).

**The host image is deliberately not loaded.** `image_flux`, `display_image`, `host_image_caption`
and `host_image_id` are debugging artifacts for this dataset and must not reach training, so this
module never reads those columns. Transient objects therefore carry `has_lightcurve=True` and
`has_image=False`, and contribute only the lightcurve tier.

**`atcat_*` is the model-ready form, `lc_*` is the physical form.** The dataset ships both:
`lc_flux_njy` in AB nanojansky, and `atcat_flux` in SNANA FLUXCAL at zero point 27.5. The ratio is
exactly 10^((31.40-27.5)/2.5) = 36.308, i.e. the conversion was done deliberately to feed ATCAT
(`light-curve/atcat`, arXiv:2511.00614), whose input contract these columns match one-for-one:
`atcat_flux`/`atcat_flux_error`/`atcat_band_id` (1=g, 2=r) / `atcat_use` (validity mask). We consume
the `atcat_*` form and use `lc_mjd` for the time axis; the `lc_flux_njy` columns are not read.

**`atcat_band_id = 0` is an excluded-i-band sentinel, but ATCAT reads 0 as u-band.** Those points
are only harmless because they also carry `atcat_use = False`. Every consumer must therefore mask by
`atcat_use`; `prepare_lightcurve_arrays` below does, and `load_transients_table` asserts that
`sum(atcat_use) == atcat_length` per row so a future revision that breaks that alignment fails
loudly rather than silently mislabelling photometry as u-band.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from captioner.utils.logging import get_logger

logger = get_logger(__name__)

DATA_DIR_PREFIX = "data/"

# ATCAT's exported ONNX has a static sequence length of 243 (only `batch` is a symbolic axis), so
# this is a property of the model, not a tunable.
ATCAT_SEQ_LEN = 243

# Columns actually consumed. The host-image columns are excluded on purpose — see module docstring.
LIGHTCURVE_COLUMNS = [
    "object_id",
    "class_label",
    "lc_mjd",
    "atcat_flux",
    "atcat_flux_error",
    "atcat_band_id",
    "atcat_use",
    "atcat_length",
]
CAPTION_COLUMNS = ["object_id", "transient_caption"]


def _read_parquet_columns(path: str, columns: list[str]) -> pd.DataFrame:
    """Reads via `pyarrow.parquet` with `ignore_metadata=True` rather than `pd.read_parquet`.

    Same defence as data/image_dataset.py's `_read_parquet_columns` (see that docstring for the full
    explanation): this repo also carries a nested `Image` feature (`display_image`), and pyarrow's
    pandas-metadata-driven dtype restoration chokes on nested struct dtype strings even for columns
    outside the requested projection.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return table.to_pandas(ignore_metadata=True)


def _list_parquet_files(hf_path: str, revision: str | None) -> list[str]:
    from huggingface_hub import list_repo_files

    files = [
        f
        for f in list_repo_files(hf_path, repo_type="dataset", revision=revision)
        if f.startswith(DATA_DIR_PREFIX) and f.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {DATA_DIR_PREFIX!r} in {hf_path!r} — check the repo "
            "layout hasn't changed."
        )
    return sorted(files)


def _download_and_read(
    hf_path: str, revision: str | None, cache_dir: Path | None, columns: list[str]
) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    frames = []
    for f in _list_parquet_files(hf_path, revision):
        local_path = hf_hub_download(
            repo_id=hf_path,
            filename=f,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        frames.append(_read_parquet_columns(local_path, columns))
    return pd.concat(frames, ignore_index=True, sort=False)


def _deduplicate_by_object_id(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Defensive, mirroring data/spectra_dataset.py. The four shards are documented as disjoint
    ("deterministic, balanced shards of one object-ordered dataset"), but a duplicated key would
    turn a downstream `set_index("object_id")` into a silent wrong-row lookup, so it is checked
    rather than assumed.
    """
    n_dupe = int(df["object_id"].duplicated(keep=False).sum())
    if n_dupe == 0:
        return df
    logger.warning(
        f"{n_dupe} duplicated object_id rows in {label} across shards — keeping the first "
        "occurrence of each. The shards are documented as disjoint, so this is worth checking."
    )
    return df.drop_duplicates(subset="object_id", keep="first").reset_index(drop=True)


def _assert_accepted_counts(df: pd.DataFrame) -> None:
    """`atcat_length` must equal the number of accepted (`atcat_use=True`) points.

    This is the invariant that makes the `atcat_band_id = 0` excluded-i sentinel safe: an excluded
    point is u-band as far as ATCAT is concerned, and only the `atcat_use` mask keeps it out. If the
    two ever disagree, masking by `atcat_use` would no longer be equivalent to "the accepted
    photometry", so this fails loudly instead of feeding ATCAT mislabelled bands.
    """
    bad = []
    for _, row in df.iterrows():
        n_used = int(np.count_nonzero(np.asarray(row["atcat_use"], dtype=bool)))
        if n_used != int(row["atcat_length"]):
            bad.append((row["object_id"], n_used, int(row["atcat_length"])))
    if bad:
        head = ", ".join(f"{oid}: use={u} length={n}" for oid, u, n in bad[:5])
        raise ValueError(
            f"{len(bad)} object(s) have sum(atcat_use) != atcat_length ({head}"
            f"{', ...' if len(bad) > 5 else ''}). The `atcat_use` mask is what keeps the "
            "`atcat_band_id = 0` excluded-i sentinel from being read as u-band by ATCAT, so this "
            "misalignment must be understood before the data is encoded."
        )


def load_transients_table(
    hf_path: str, revision: str | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    """One row per ZTF object: identity, class label, and the light-curve arrays. Never reads the
    host-image columns (see module docstring).
    """
    df = _download_and_read(hf_path, revision, cache_dir, LIGHTCURVE_COLUMNS)
    df = _deduplicate_by_object_id(df, "transients table")
    _assert_accepted_counts(df)
    df["has_lightcurve"] = True
    logger.info(f"Loaded {len(df)} transients from {hf_path}")
    return df


def load_transient_captions(
    hf_path: str, revision: str | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    """One row per object: `object_id`, `transient_caption`. `host_image_caption` is deliberately
    not read — the host image is excluded from training, so there is no image tier to ground it in.
    """
    df = _download_and_read(hf_path, revision, cache_dir, CAPTION_COLUMNS)
    df = _deduplicate_by_object_id(df, "transient captions")
    df = df.dropna(subset=["object_id", "transient_caption"]).reset_index(drop=True)
    return df


def _stable_seed(seed: int, object_id: str) -> int:
    """Deterministic across processes and runs, unlike Python's salted `hash()`. Re-running the
    cache therefore selects exactly the same points for a downsampled object.
    """
    digest = hashlib.sha256(f"{seed}:{object_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def prepare_lightcurve_arrays(
    mjd,
    flux,
    flux_err,
    band_id,
    use,
    *,
    object_id: str = "",
    seq_len: int = ATCAT_SEQ_LEN,
    detection_window_days: float = 30.0,
    detection_snr: float = 5.0,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Turns one object's raw arrays into ATCAT's five fixed-length inputs.

    Returns `({"flux","flux_err","time","mask","channel_index"}, info)`, every array shaped
    `(seq_len,)`. Deliberately numpy-only (no torch) so the selection logic — the part with real
    failure modes — is unit-testable without a GPU stack.

    Steps, in this order, following the model card's preprocessing section:

    1. **Keep only accepted points** (`atcat_use`). This is what keeps the `band_id = 0`
       excluded-i sentinel out of ATCAT, which would otherwise read it as u-band.
    2. **Trim to the detection window** — `detection_window_days` before the first S/N >
       `detection_snr` point through the last one. Not optional: the model card states training was
       filtered this way and that "data far outside this window is out of distribution". It is also
       what makes the length cap tractable — measured on the real data, 16 objects exceed 243
       accepted points but only 5 still do after this trim, because the long ones are multi-year ZTF
       baselines wrapped around a ~10-week outburst (one has 660 points over 2,974 days with a
       102-day detection window holding 16 of them).
    3. **If still over `seq_len`, select `seq_len` points uniformly at random**, seeded stably per
       object. Deliberately *not* a fixed stride and deliberately *not* truncation: truncation would
       delete the decline phase the captions describe (on the worst object, first-243-of-364 drops
       the last third of an 800-day span). Deliberately *not* SNR- or peak-weighted either —
       hand-picking the cleanest epochs would hand ATCAT an unnaturally clean curve, pushing input
       away from the naturally-sampled light curves it trained on. Verified on the five affected
       objects: random selection preserves g/r composition to within one percentage point.
    4. **Shift time to start at zero**, in days, and right-zero-pad to `seq_len` with `mask = 0`.
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    band_id = np.asarray(band_id, dtype=np.int64)
    use = np.asarray(use, dtype=bool)

    lengths = {len(mjd), len(flux), len(flux_err), len(band_id), len(use)}
    if len(lengths) != 1:
        raise ValueError(
            f"object={object_id!r}: light-curve arrays disagree on length "
            f"(mjd={len(mjd)}, flux={len(flux)}, flux_err={len(flux_err)}, "
            f"band_id={len(band_id)}, use={len(use)}). They are documented as aligned."
        )

    n_accepted = int(np.count_nonzero(use))
    if n_accepted == 0:
        raise ValueError(
            f"object={object_id!r} has no accepted photometry (atcat_use is all False) — nothing "
            "to encode. It should have been excluded upstream."
        )

    keep = use.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(flux_err > 0, flux / flux_err, 0.0)
    detected = keep & np.isfinite(snr) & (snr > detection_snr)
    if detected.any():
        lo = float(mjd[detected].min()) - float(detection_window_days)
        hi = float(mjd[detected].max())
        keep = keep & (mjd >= lo) & (mjd <= hi)
    else:
        # No point clears the detection threshold. Rather than invent a window, keep every accepted
        # point and let step 3 handle the length.
        logger.warning(
            f"object={object_id!r}: no accepted point exceeds S/N {detection_snr}; skipping the "
            "detection-window trim and keeping all accepted photometry."
        )

    idx = np.flatnonzero(keep)
    n_in_window = int(idx.size)
    if n_in_window == 0:
        raise ValueError(
            f"object={object_id!r}: detection-window trim removed every point, which should be "
            "impossible (the window is built from accepted points). Check the input arrays."
        )

    if n_in_window > seq_len:
        rng = np.random.default_rng(_stable_seed(seed, object_id))
        idx = rng.choice(idx, size=seq_len, replace=False)

    n_selected = int(idx.size)

    # MJD order — the source is documented as MJD-ordered, but sorting makes that explicit rather
    # than assumed, and random selection above returns points in arbitrary order.
    idx = idx[np.argsort(mjd[idx], kind="stable")]

    sel_time = mjd[idx]
    sel_time = sel_time - sel_time.min()

    out_flux = np.zeros(seq_len, dtype=np.float64)
    out_err = np.zeros(seq_len, dtype=np.float64)
    out_time = np.zeros(seq_len, dtype=np.float64)
    out_mask = np.zeros(seq_len, dtype=np.int64)
    out_band = np.zeros(seq_len, dtype=np.int64)

    out_flux[:n_selected] = flux[idx]
    out_err[:n_selected] = flux_err[idx]
    out_time[:n_selected] = sel_time
    out_mask[:n_selected] = 1
    out_band[:n_selected] = band_id[idx]

    arrays = {
        "flux": out_flux,
        "flux_err": out_err,
        "time": out_time,
        "mask": out_mask,
        "channel_index": out_band,
    }
    info = {
        "n_accepted": n_accepted,
        "n_in_window": n_in_window,
        "n_selected": n_selected,
        "downsampled": int(n_selected < n_in_window),
    }
    return arrays, info
