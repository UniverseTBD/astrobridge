"""Loader for `BuildNg/astrobridge-yse-test-dataset-v2` — a real, held-out SN-typing benchmark.
Confirmed directly against the live HF repo, not assumed: 266 rows, every `object_id` unique, and
**zero overlap** with `object_id`s in the training set (`BuildNg/astrobridge-transients-dataset`)
— genuinely disjoint, not accidentally-reused training data. Same schema as the training set
(`object_id, host_image_id, lc_mjd, lc_flux_njy, lc_flux_error_njy, lc_band, atcat_flux,
atcat_flux_error, atcat_band_id, atcat_use, atcat_length, image_flux, image_bands, image_modality,
display_image, class_label`), same 3-class taxonomy — real confirmed counts on this eval set:
`SN Ia` 180, `SN II` 71, `SN Ibc` 15 (same imbalance pattern as training).

Two tracks, from the same underlying dataset:
  - **lightcurve-only** (`load_lightcurve_table`): reuses `captioner.data.transients_dataset`'s
    private loading helpers unchanged, just pointed at this repo instead — same column names,
    same `atcat_*` convention, same invariant checks (`_assert_accepted_counts`), so nothing
    dataset-shape-specific needed reinventing here.
  - **lightcurve + host image** (`load_host_image_table`): genuinely new — the training loader's
    own module docstring calls the host-image columns "debugging artifacts" and never reads them
    for training, so there was no existing loader to extend. `image_modality` is confirmed
    uniformly `LegacySurveyImage` across all 266 rows, matching what `AionImageEncoder` expects,
    but `image_flux`'s actual band count/order has NOT been separately verified here (same-org
    dataset as the training set, so less likely to have an undocumented-band surprise than
    `astronolan/galaxy10-aion` did, but check `build_raw_inputs_with_image`'s shape assertion
    output on a real run before trusting it blindly).

**Base-model comparison, mirroring the image track's `image_rgb` pattern**: a raw lightcurve array
genuinely isn't something an out-of-the-box vision-language model can consume directly (still
true), but `render_lightcurve_plot` below turns one object's `atcat_*` photometry into an ordinary
flux-vs-time PNG scatter plot, which the base model's native-vision pathway *can* consume, same as
`eval.datasets.image_galaxy10.decode_rgb_image` feeds the image track's base side. This is a real,
independent rendering, not a proxy for what the equipped side sees (which reads the raw
`atcat_flux`/`atcat_flux_error`/`atcat_band_id`/`atcat_use` arrays directly, never a plot).

**Deterministic classification (`SN_REASONING_PROMPT`/`SN_CANDIDATES`)**: earlier digit-code and
free-text prompting both showed real compliance failures — a bare digit collapses to one symbol
regardless of the object (confirmed live: base emitted `"2"` on 28/30 objects, equipped never
emitted `"1"` once), and free text needs fuzzy parsing that can fail outright (confirmed live on
v5: equipped's parse rate collapsed to 53% under the chat template). `SN_REASONING_PROMPT` sidesteps
both by never asking for a label at all — the model reasons freely, then `eval/backend.py`'s
`.classify(...)` scores `SN_CANDIDATES`'s three exact label strings as teacher-forced continuations
and takes the argmax, which is always one of the three by construction: no parsing, no unparseable
output, fully deterministic given the same weights.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from captioner.data.transients_dataset import (
    LIGHTCURVE_COLUMNS,
    _assert_accepted_counts,
    _deduplicate_by_object_id,
    _download_and_read,
    prepare_lightcurve_arrays,
)

YSE_HF_PATH = "BuildNg/astrobridge-yse-test-dataset-v2"

# Real, confirmed values of the `class_label` column — see module docstring. Order matters for
# eval/metrics/caption_to_label.py's tie-break behaviour; kept in sync with that file's
# SN_TYPE_SYNONYMS key order deliberately.
SN_LABELS = ["SN Ia", "SN II", "SN Ibc"]

HOST_IMAGE_COLUMNS = ["object_id", "class_label", "image_flux", "image_bands", "image_modality"]

# Digit-code label mapping, SN_LABELS order — kept selectable (`--answer-format digit_code`)
# alongside the default `logprob_argmax` path so the two can be compared on the same objects.
SN_CLASS_CODES: dict[str, str] = {str(i): label for i, label in enumerate(SN_LABELS)}
SN_CLASS_CODE_LEGEND = "\n".join(f"{code}={name}" for code, name in SN_CLASS_CODES.items())
SN_CLASS_CODE_PROMPT = (
    "Supernova light curve classifier. Output ONLY the digit code, nothing else.\n"
    f"{SN_CLASS_CODE_LEGEND}\n"
    "Light curve class code:"
)

# --- Deterministic classification (the default path, see module docstring) --------------------
#
# Deliberately contains no digits, letters, or a request to name a class at all — asking for that
# is exactly what corrupted the digit-code and free-text paths (see module docstring). The model
# only ever has to reason about the observation; `eval/backend.py`'s `.classify(...)` handles
# turning that reasoning into a label by scoring `SN_CANDIDATES` as its continuation.
SN_REASONING_PROMPT = (
    "Examine this supernova light curve: note its total duration, the peak brightness reached in "
    "each band, how quickly it declines after peak, and whether there is any gap in the "
    "observations. Based on the evidence above, the classification is:"
)

# Leading space on each candidate: these are scored as a direct continuation of
# SN_REASONING_PROMPT's answer, not as a standalone sentence — verified against the real Qwen3.5-9B
# tokenizer that " SN Ia"/" SN II"/" SN Ibc" all share the same first token (`' SN'`), so
# first-token scoring is impossible and full-sequence scoring (`score_completions`/
# `score_completions_qwen_native`) is mandatory, not a refinement.
SN_CANDIDATES = [f" {label}" for label in SN_LABELS]


def load_lightcurve_table(
    hf_path: str = YSE_HF_PATH, revision: str | None = None, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per object: identity, `class_label`, and the raw `atcat_*` light-curve arrays —
    exactly `captioner.data.transients_dataset.load_transients_table`'s shape, reimplemented here
    (not imported) only because that function hardcodes the log message "transients from
    {hf_path}"; the underlying read/dedupe/invariant-check logic is reused directly, unchanged.
    """
    df = _download_and_read(hf_path, revision, cache_dir, LIGHTCURVE_COLUMNS)
    df = _deduplicate_by_object_id(df, "YSE eval lightcurve table")
    _assert_accepted_counts(df)
    return df


def load_host_image_table(
    hf_path: str = YSE_HF_PATH, revision: str | None = None, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per object: `object_id`, `class_label`, and the raw host-image columns
    (`image_flux`, `image_bands`, `image_modality`) — genuinely new, see module docstring for why
    no existing loader covers this.
    """
    df = _download_and_read(hf_path, revision, cache_dir, HOST_IMAGE_COLUMNS)
    return _deduplicate_by_object_id(df, "YSE eval host-image table")


# 1=g, 2=r per captioner.data.transients_dataset's module docstring (`atcat_band_id`); 0 is the
# excluded-i sentinel, always masked out by `atcat_use` below before it's ever plotted.
_BAND_PLOT_STYLE = {1: {"color": "tab:green", "label": "g-band"}, 2: {"color": "tab:red", "label": "r-band"}}


def render_lightcurve_plot(
    row: pd.Series,
    *,
    x_axis_label: str = "Days since first detection",
) -> Image.Image:
    """Renders one `load_lightcurve_table` row's `atcat_*` photometry as a flux-vs-time PNG
    scatter/errorbar plot — the base model's `{"image": <PIL.Image>}` input (`eval.backend`'s
    base-side contract). Masks by `atcat_use` first (the excluded-i sentinel at `atcat_band_id=0`
    is only safe to drop via this mask), then plots g/r separately by color so the model has some
    chance of reading band information off the picture the same way it would off two colored
    curves in a real light-curve plot.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless — this runs inside eval collection loops/Modal containers, never a GUI session
    import matplotlib.pyplot as plt

    mjd = np.asarray(row["lc_mjd"], dtype=float)
    flux = np.asarray(row["atcat_flux"], dtype=float)
    flux_err = np.asarray(row["atcat_flux_error"], dtype=float)
    band_id = np.asarray(row["atcat_band_id"], dtype=int)
    use = np.asarray(row["atcat_use"], dtype=bool)
    mjd, flux, flux_err, band_id = mjd[use], flux[use], flux_err[use], band_id[use]

    order = np.argsort(mjd)
    mjd, flux, flux_err, band_id = mjd[order], flux[order], flux_err[order], band_id[order]
    t0 = mjd[0] if len(mjd) else 0.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    any_plotted = False
    for bid, style in _BAND_PLOT_STYLE.items():
        m = band_id == bid
        if np.any(m):
            ax.errorbar(mjd[m] - t0, flux[m], yerr=flux_err[m], fmt="o", markersize=4, capsize=2, **style)
            any_plotted = True
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Flux (SNANA FLUXCAL, zp=27.5)")
    # Deliberately no object_id in the title — these are real, catalogued ZTF/YSE supernovae, so a
    # title naming the object hands the base model (native vision, reads text baked into the image
    # like any other pixels) the answer for free: it can transcribe the name and recognise/recall
    # the object instead of reading the photometry. Confirmed live: 256/266 base captions echoed
    # their own object_id back. Equipped has no equivalent leak — it consumes numeric arrays, never
    # a name — so a titled plot made the two sides not comparable.
    ax.set_title("Light curve")
    if any_plotted:
        ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def stratified_sample(
    table: pd.DataFrame, n_total: int, seed: int, min_per_class: int = 2, label_col: str = "class_label",
) -> pd.DataFrame:
    """Same "largest remainder" apportionment algorithm as `eval.datasets.image_galaxy10.
    stratified_sample` — one `numpy.random.default_rng(seed)`, consumed in `SN_LABELS`' fixed
    order, so the same seed always draws the same objects; raises rather than silently returning a
    bigger-or-smaller-than-requested sample if `n_total` is infeasible.
    """
    rng = np.random.default_rng(seed)
    groups = {label: table.index[table[label_col] == label].to_numpy() for label in SN_LABELS}
    sizes = {label: len(idx) for label, idx in groups.items()}
    total_available = sum(sizes.values())
    if n_total > total_available:
        raise ValueError(f"n_total={n_total} exceeds the {total_available} objects available across all {len(SN_LABELS)} classes.")
    min_feasible = sum(min(min_per_class, sizes[label]) for label in SN_LABELS)
    if n_total < min_feasible:
        raise ValueError(
            f"n_total={n_total} is below {min_feasible}, the minimum needed to give every class at "
            f"least min_per_class={min_per_class} (capped by classes with fewer available, e.g. "
            f"SN Ibc has only {sizes.get('SN Ibc', 0)}). Raise n_total or lower min_per_class — "
            "returning a silently-larger-than-requested sample here would break the 'exactly "
            "n_total, reproducibly' contract."
        )

    raw_share = {label: n_total * sizes[label] / total_available for label in SN_LABELS}
    target = {label: min(sizes[label], max(min_per_class, int(np.floor(raw_share[label])))) for label in SN_LABELS}

    by_largest_remainder = sorted(SN_LABELS, key=lambda label: raw_share[label] - np.floor(raw_share[label]), reverse=True)
    by_smallest_remainder = list(reversed(by_largest_remainder))

    while sum(target.values()) > n_total:
        progressed = False
        for label in by_smallest_remainder:
            if sum(target.values()) <= n_total:
                break
            if target[label] > min_per_class:
                target[label] -= 1
                progressed = True
        if not progressed:
            break

    while sum(target.values()) < n_total:
        progressed = False
        for label in by_largest_remainder:
            if sum(target.values()) >= n_total:
                break
            if target[label] < sizes[label]:
                target[label] += 1
                progressed = True
        if not progressed:
            break

    sampled_indices = []
    for label in SN_LABELS:
        chosen = rng.choice(groups[label], size=target[label], replace=False)
        sampled_indices.extend(chosen.tolist())

    return table.loc[sampled_indices].reset_index(drop=True)


def balanced_sample(
    table: pd.DataFrame, per_class: int, seed: int, label_col: str = "class_label",
) -> pd.DataFrame:
    """Exactly `per_class` objects from EACH class — an equal-bucket draw, not the
    population-proportional one `stratified_sample` does. Deterministic given `seed`. Raises if any
    class has fewer than `per_class` objects rather than silently returning an unbalanced sample.
    """
    rng = np.random.default_rng(seed)
    groups = {label: table.index[table[label_col] == label].to_numpy() for label in SN_LABELS}
    short = {label: len(idx) for label, idx in groups.items() if len(idx) < per_class}
    if short:
        raise ValueError(
            f"balanced_sample(per_class={per_class}) needs {per_class} of every class, but "
            f"{short} — lower per_class to at most {min(len(idx) for idx in groups.values())}."
        )

    sampled_indices = []
    for label in SN_LABELS:
        chosen = rng.choice(groups[label], size=per_class, replace=False)
        sampled_indices.extend(chosen.tolist())
    return table.loc[sampled_indices].reset_index(drop=True)


def build_raw_inputs_lightcurve(row: pd.Series, cfg, seed: int = 0) -> dict:
    """Turns one `load_lightcurve_table` row into the `{"lightcurve": {...}}` shape
    `captioner.inference.generate_caption` expects — identical preprocessing
    (detection-window trim / seeded downsample / pad) the training cache used, via
    `prepare_lightcurve_arrays`, so this eval track sees the model under the same input
    distribution it was trained on, not a distribution shift introduced by the eval code itself.
    """
    lc_modality = cfg.modalities.lightcurve
    lc_kwargs = lc_modality.encoder.get("kwargs", {})
    arrays, _info = prepare_lightcurve_arrays(
        row["lc_mjd"], row["atcat_flux"], row["atcat_flux_error"], row["atcat_band_id"], row["atcat_use"],
        object_id=str(row["object_id"]),
        seq_len=int(lc_modality.max_tokens),
        detection_window_days=float(lc_kwargs.get("detection_window_days", 30.0)),
        detection_snr=float(lc_kwargs.get("detection_snr", 5.0)),
        seed=int(lc_kwargs.get("subsample_seed", seed)),
    )
    return {"lightcurve": {k: torch.from_numpy(v[None, :]) for k, v in arrays.items()}}


def build_raw_inputs_with_image(row_lc: pd.Series, row_image: pd.Series, cfg, seed: int = 0) -> dict:
    """Same as `build_raw_inputs_lightcurve`, plus the host image keyed under `"image"` —
    `row_image["image_flux"]`/`row_image["image_bands"]` must already have been resolved to a
    `(n_bands, H, W)` array in the order `cfg.modalities.image.encoder.kwargs.bands` expects
    (currently `[DES-G, DES-R, DES-Z]`) before calling this; this function does NOT reorder or
    validate bands itself — do that once, explicitly, against a real sample of rows (see module
    docstring's "not yet verified" note), not silently inside this per-row helper.
    """
    raw_inputs = build_raw_inputs_lightcurve(row_lc, cfg, seed=seed)
    image_array = np.asarray(row_image["image_flux"], dtype=np.float32)
    expected_bands = len(cfg.modalities.image.encoder.kwargs.bands)
    if image_array.shape[0] != expected_bands:
        raise ValueError(
            f"object={row_image['object_id']!r}: image_flux has {image_array.shape[0]} bands, "
            f"expected {expected_bands} ({list(cfg.modalities.image.encoder.kwargs.bands)}) — "
            "resolve the real band order/identity before building raw_inputs, don't guess here."
        )
    raw_inputs["image"] = {"pixel_values": torch.from_numpy(image_array).unsqueeze(0)}
    return raw_inputs
