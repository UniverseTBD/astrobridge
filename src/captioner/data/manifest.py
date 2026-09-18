"""Builds manifest.parquet: one row per object, with modality-availability flags, tier, and
split — plus manifest_stats.json (tier histogram, join diagnostics). No join/count is ever
hardcoded; everything here is a runtime fact (§1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from captioner.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_UPSTREAM_SPLIT_COLUMN = "split_upstream"


def _load_spectra_table(cfg: DictConfig) -> pd.DataFrame:
    """Uses data/spectra_dataset.py, not `datasets.load_dataset` — the latter fails against the
    real repo's multiple, non-identically-schemaed parquet files (confirmed against a real run,
    see that module's docstring).
    """
    from captioner.data.spectra_dataset import UPSTREAM_SPLIT_COLUMN, load_spectra_table

    files = cfg.sources.spectra.get("files")
    df = load_spectra_table(
        cfg.sources.spectra.hf_path,
        revision=cfg.sources.spectra.get("revision"),
        files=list(files) if files else None,
    )
    df = df.rename(columns={"ra_spectra": "ra", "dec_spectra": "dec"})
    if UPSTREAM_SPLIT_COLUMN in df.columns:
        df = df.rename(columns={UPSTREAM_SPLIT_COLUMN: MANIFEST_UPSTREAM_SPLIT_COLUMN})
    df["has_spectra"] = True
    return df


def _load_image_table(cfg: DictConfig) -> pd.DataFrame:
    """Sources identity from `legacy_south_all_images.parquet`, not the caption-only RGB dataset
    (see data/image_dataset.py) — this table's rows are objects that already have real
    per-band calibrated flux (usable for AION), and its `object_id` is AstroBridge-Data's own id
    (a direct join key below — see build_manifest's "object_id" join path), not a coordinate
    match. The caption dataset (gapatron/astrobridge-image-captions) is still used for `caption_fused` text in
    scripts/01_generate_captions.py, just not for identity here.
    """
    from captioner.data.image_dataset import load_image_flux_identity_table

    return load_image_flux_identity_table(
        cfg.sources.image.hf_path,
        revision=cfg.sources.image.get("revision"),
    )


def _load_transients_table(cfg: DictConfig) -> pd.DataFrame | None:
    """`BuildNg/astrobridge-transients-dataset` — ZTF light curves, appended not joined.

    Returns None when the source is not configured, so a config predating this modality still
    builds a manifest unchanged.
    """
    if "transients" not in cfg.sources:
        return None
    from captioner.data.transients_dataset import load_transients_table

    return load_transients_table(
        cfg.sources.transients.hf_path,
        revision=cfg.sources.transients.get("revision"),
    )


def _modality_names_from_flags(df: pd.DataFrame) -> list[str]:
    """Modality names derived from the frame's own `has_<name>` columns rather than from
    configs/modalities.yaml, because scripts/00_build_manifest.py loads only base+data — the
    registry is not in scope here. Adding a modality therefore needs no change in this module.
    """
    return sorted(c[len("has_") :] for c in df.columns if c.startswith("has_"))


def _crossmatch_coords(left: pd.DataFrame, right: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    left_coord = SkyCoord(ra=left["ra"].to_numpy() * u.deg, dec=left["dec"].to_numpy() * u.deg)
    right_coord = SkyCoord(ra=right["ra"].to_numpy() * u.deg, dec=right["dec"].to_numpy() * u.deg)
    idx, sep2d, _ = left_coord.match_to_catalog_sky(right_coord)
    within = sep2d.arcsec <= radius_arcsec

    matched = left.copy()
    matched["_match_idx"] = np.where(within, idx, -1)
    matched["_sep_arcsec"] = sep2d.arcsec
    return matched


def _assert_unique_object_id(df: pd.DataFrame, label: str) -> None:
    """A join key that isn't actually unique silently turns `pd.merge` into a cross-product for
    every duplicated key, inflating row counts in a way that's easy to miss (a real failure mode
    hit in production — see spectra_dataset.py's deduplication). Loaders are expected to
    deduplicate themselves; this is the last-resort net that fails loudly instead of letting a
    future data source quietly corrupt the manifest the same way.
    """
    if "object_id" not in df.columns:
        return
    dupes = df["object_id"][df["object_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{label} has {dupes.nunique()} duplicated object_id values ({len(dupes)} rows) — "
            "the loader for this table should have deduplicated before returning. Merging with "
            "a non-unique key would silently inflate the manifest's row count."
        )


def build_manifest(cfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    spectra_df = _load_spectra_table(cfg)
    image_df = _load_image_table(cfg)
    _assert_unique_object_id(spectra_df, "spectra table")
    _assert_unique_object_id(image_df, "image table")

    join_key = cfg.join.key
    have_key_both = join_key in spectra_df.columns and join_key in image_df.columns

    if have_key_both:
        logger.info(f"Joining on {join_key}")
        merged_key = pd.merge(
            spectra_df, image_df, on=join_key, how="outer", suffixes=("_spec", "_img")
        )
        join_method = join_key
    elif "object_id" in image_df.columns:
        # legacy_south_all_images.parquet's object_id is target_object_id_target — already
        # AstroBridge-Data's own id (these rows are pre-crossmatched by the dataset authors) —
        # a direct, authoritative join, no coordinate fuzz-matching needed for this subset.
        overlap = image_df["object_id"].isin(spectra_df["object_id"]).mean()
        logger.info(f"Direct object_id join: {overlap:.1%} of image-table object_ids matched a spectra object_id")
        if overlap < 0.5:
            logger.warning(
                "Less than half of the image table's object_id values matched AstroBridge-Data's "
                "object_id — the assumption that legacy_south_all_images.parquet's "
                "target_object_id_target shares AstroBridge-Data's id namespace may not hold for "
                "this data revision. Verify before trusting the resulting joint tier."
            )
        merged_key = pd.merge(
            spectra_df, image_df, on="object_id", how="outer", suffixes=("_spec", "_img")
        )
        join_method = "object_id"
    else:
        logger.warning(
            f"{join_key} not present in both tables; falling back to coordinate crossmatch "
            f"at {cfg.join.fallback_radius_arcsec} arcsec"
        )
        matched = _crossmatch_coords(spectra_df, image_df, cfg.join.fallback_radius_arcsec)
        joint_mask = matched["_match_idx"] >= 0
        joint = matched[joint_mask].copy()
        joint_image_cols = image_df.iloc[joint["_match_idx"].to_numpy()].reset_index(drop=True)
        joint = joint.reset_index(drop=True)
        for c in joint_image_cols.columns:
            if c not in joint.columns:
                joint[c] = joint_image_cols[c]

        spec_only = matched[~joint_mask].copy()
        image_only_mask = ~image_df.index.isin(matched.loc[joint_mask, "_match_idx"])
        image_only = image_df[image_only_mask].copy()

        merged_key = pd.concat([joint, spec_only, image_only], ignore_index=True, sort=False)
        join_method = f"coord@{cfg.join.fallback_radius_arcsec}arcsec"

    if "object_id" not in merged_key.columns:
        merged_key["object_id"] = merged_key.index.astype(str)

    # Transients are APPENDED, never merged: a ZTF designation shares no namespace with
    # AstroBridge-Data's object_id, the Legacy Survey's object_id_legacy, or the Gemini captions'
    # wiki_entity_id, so there is nothing to join on. They arrive with has_lightcurve=True and pick
    # up has_image/has_spectra=False from the fill below.
    transients_df = _load_transients_table(cfg)
    if transients_df is not None:
        _assert_unique_object_id(transients_df, "transients table")
        collisions = set(transients_df["object_id"]) & set(merged_key["object_id"])
        if collisions:
            raise ValueError(
                f"{len(collisions)} transient object_id(s) collide with the image/spectra manifest "
                f"(e.g. {sorted(collisions)[:3]}). ZTF designations are meant to be a disjoint "
                "namespace; appending them would silently create duplicate manifest rows."
            )
        merged_key = pd.concat([merged_key, transients_df], ignore_index=True, sort=False)
        logger.info(f"Appended {len(transients_df)} transient objects (lightcurve-only)")

    # Force the two original flags into existence so a source producing no rows still yields a
    # well-formed manifest — what the previous `.get(..., False)` calls provided.
    for name in ("spectra", "image"):
        if f"has_{name}" not in merged_key.columns:
            merged_key[f"has_{name}"] = False

    # Confirmed real: spectra's object_id can be a genuine Python int (DESI's numeric target ids)
    # while image/transients object_ids are strings (Legacy Survey brick-style names like
    # '0001m057-6125', ZTF designations). After any of the merge/concat paths above — including
    # the transients append — pandas can end up holding both kinds in one `object`-dtype column.
    # pyarrow then infers a single Arrow type from a sample and fails the moment it hits a value
    # that doesn't fit ("Could not convert '0001m057-6125' ... tried to convert to int64").
    # object_id is only ever used as an opaque lookup key downstream (dict/index, never
    # arithmetic), so normalizing to a uniform string here — once, after every path has already
    # run — is always safe and loses no real information.
    merged_key["object_id"] = merged_key["object_id"].astype(str)

    modality_names = _modality_names_from_flags(merged_key)
    for name in modality_names:
        col = f"has_{name}"
        merged_key[col] = merged_key[col].fillna(False).infer_objects(copy=False).astype(bool)

    # `joint` = more than one modality present, derived from the flag columns rather than a
    # hardcoded has_spectra-and-has_image pair, so adding a modality needs no change here. The two
    # labels are deliberately unchanged, so tier_histogram consumers and existing tests keep working.
    flag_cols = [f"has_{n}" for n in modality_names]
    merged_key["tier"] = np.where(merged_key[flag_cols].sum(axis=1) >= 2, "joint", "single")

    # Drop heavy nested/array columns before writing — the raw `spectrum` struct (flux/ivar/
    # mask/lambda, thousands of floats per object) and whatever the image dataset's pixel-cutout
    # field is called. Nothing downstream reads these back out of manifest.parquet:
    # 02_cache_embeddings.py reloads the raw HF datasets itself for encoding, and everything else
    # only needs scalars/metadata. Keeping them here would silently duplicate the full raw
    # dataset size into a second on-disk copy for no reason.
    heavy_columns = [
        c
        for c in (
            "spectrum",
            "image",
            # The transients' light-curve arrays: 02_cache_embeddings.py reloads them from the
            # source, so duplicating them into manifest.parquet buys nothing. `atcat_length` and
            # `class_label` are scalars and are kept for diagnostics / eval slicing.
            "lc_mjd",
            "atcat_flux",
            "atcat_flux_error",
            "atcat_band_id",
            "atcat_use",
        )
        if c in merged_key.columns
    ]
    if heavy_columns:
        logger.info(f"Dropping heavy array columns from manifest.parquet: {heavy_columns}")

    leading = ["object_id"] + flag_cols + ["tier"]
    manifest = merged_key[
        leading + [c for c in merged_key.columns if c not in leading and c not in heavy_columns]
    ].copy()

    stats = _compute_stats(manifest, join_method, cfg)
    return manifest, stats


def _compute_stats(manifest: pd.DataFrame, join_method: str, cfg: DictConfig) -> dict:
    tier_hist = manifest["tier"].value_counts().to_dict()
    n_joint = int(tier_hist.get("joint", 0))
    n_total = len(manifest)

    modality_names = _modality_names_from_flags(manifest)

    def _only(name: str) -> int:
        mask = manifest[f"has_{name}"].copy()
        for other in modality_names:
            if other != name:
                mask &= ~manifest[f"has_{other}"]
        return int(mask.sum())

    # Generic per-combination counts, so a lightcurve-only object is not silently invisible the way
    # it would be to the two hardcoded `n_*_only` keys below (which are kept for compatibility).
    availability_histogram: dict[str, int] = {}
    for present in (
        tuple(m for m in modality_names if bool(row[f"has_{m}"])) for _, row in manifest.iterrows()
    ):
        key = "+".join(present) if present else "none"
        availability_histogram[key] = availability_histogram.get(key, 0) + 1

    stats = {
        "join_method": join_method,
        "n_total_objects": n_total,
        "modalities": modality_names,
        "n_spectra_only": _only("spectra") if "spectra" in modality_names else 0,
        "n_image_only": _only("image") if "image" in modality_names else 0,
        "n_joint": n_joint,
        "tier_histogram": {k: int(v) for k, v in tier_hist.items()},
        "availability_histogram": availability_histogram,
    }

    if n_joint < cfg.sanity.min_joint_objects:
        logger.warning(
            f"Only {n_joint} joint objects (< configured min_joint_objects="
            f"{cfg.sanity.min_joint_objects}). Val/test are drawn from joint objects only — "
            "eval sets may be very thin. Not failing the build; proceeding."
        )
    return stats


STRATIFY_NA = "(none)"


def _draw_per_tier(
    manifest: pd.DataFrame,
    object_ids,
    frac_val: float,
    frac_test: float,
    rng,
    keys: list[str] | None = None,
    tier_fraction_overrides: dict[str, tuple[float, float]] | None = None,
) -> tuple[set, set]:
    """Draw val/test proportionally within every stratum, where a stratum is one combination of
    `keys` (default `["tier"]`, i.e. the original behaviour).

    Stratifying on `tier` alone is not enough once a tier mixes populations the model has to
    generalise across. Confirmed real on this dataset: the spectra tier holds 1,844 SDSS spectra
    (~3,860 samples, SDSSSpectrum codec) and 334 DESI (7,781 samples, DESISpectrum codec — a
    genuinely different AION internal domain), and `split_upstream` put 331 of the 334 DESI in
    test. v5 and v6 therefore trained on 2 DESI spectra and were scored on a spectra test set
    that is 72% DESI. Adding `survey` to the keys splits each survey 80/10/10 on its own.
    It does the same for the image tier's legacy-south (1,620) vs legacy-north (653), which
    differ in cutout size and in whether ivar/mask/i-band exist at all.

    NaN is a real stratum here, not missing data — light-curve objects have no `survey` — so it
    is filled rather than dropped. `groupby` would otherwise silently drop all 987 of them from
    the draw and leave them entirely in train.

    `tier_fraction_overrides`: `{tier_name: (frac_val, frac_test)}`, checked per stratum group
    against that group's `tier` value — lets one tier use different val/test fractions than the
    rest (e.g. spectra has no separate external held-out test set anywhere in this repo, unlike
    image/lightcurve which eval against their own dedicated test datasets, so spectra's `test`
    fraction is set to 0.0 here rather than carving out held-out data with nowhere to use it).
    Stratification itself (which strata get formed) is unaffected — only the val/test fractions
    applied within each group change.
    """
    keys = list(keys or ["tier"])
    tier_fraction_overrides = tier_fraction_overrides or {}
    val: set = set()
    test: set = set()
    pool = manifest[manifest["object_id"].isin(set(object_ids))]
    if pool.empty:
        return val, test
    grouping = [pool[k].astype(object).where(pool[k].notna(), STRATIFY_NA) for k in keys]
    for _, group in pool.groupby(grouping, sort=True):
        group_tier = group["tier"].iloc[0] if "tier" in group.columns and len(group) else None
        gv, gt = tier_fraction_overrides.get(group_tier, (frac_val, frac_test))
        ids = group["object_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_val = int(round(len(ids) * float(gv)))
        n_test = int(round(len(ids) * float(gt)))
        val.update(ids[:n_val])
        test.update(ids[n_val : n_val + n_test])
    return val, test


def assign_splits(manifest: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Object-level 80/10/10. Val/test are drawn from joint objects only *when there are enough
    of them* (>= sanity.min_joint_objects) — that's the spec's original intent, so joint-tier
    eval is honest once milestone 2 (joint training) is real. Below that threshold — including
    the current 0-joint-object state, a real consequence of the object_id/target_object_id_target
    namespace mismatch, not a milestone-1-scope non-issue — falling back to joint-only would
    leave val/test completely empty. An empty val set doesn't just mean "no eval": evaluate_loss
    returns a fake 0.0 every epoch, which early-stopping reads as "no improvement" from epoch 2
    onward, silently truncating training after `patience` epochs regardless of real progress.
    So below the threshold, val/test are drawn from ALL objects instead (stratified by tier so
    single-image and single-spectra objects are both represented), with a loud warning — this
    keeps milestone-1 single-modality training and eval meaningful without needing the joint
    crossmatch fixed first.

    `cfg.splits.policy` makes that choice explicit rather than threshold-inferred: 'auto' (the
    default, and the behaviour described above), 'stratified' (always all tiers) or 'joint_only'
    (always joint). It is pinned to 'stratified' in configs/data.yaml so that adding a source which
    happens to push the joint count past min_joint_objects cannot silently redefine what val/test
    mean mid-project.
    """
    rng = np.random.default_rng(int(cfg.splits.seed))
    manifest = manifest.copy()
    manifest["split"] = "train"

    stratify_by = [
        k for k in list(cfg.splits.get("stratify_by", ["tier"])) if k in manifest.columns
    ] or ["tier"]

    # Per-tier val/test fraction overrides — e.g. spectra has no separate external held-out test
    # set anywhere in this repo (unlike image/lightcurve, which eval against their own dedicated
    # test datasets: Galaxy10's own test parquet, the YSE test set), so carving out a `test` slice
    # for it holds data back from training with nothing that will ever score it. See
    # `_draw_per_tier`'s docstring for the mechanics; stratification (survey within spectra) is
    # unaffected, only the fractions are.
    tier_fraction_overrides = {
        tier: (float(entry.get("val", cfg.splits.val)), float(entry.get("test", cfg.splits.test)))
        for tier, entry in dict(cfg.splits.get("tier_overrides", {})).items()
    }

    joint_ids = manifest.loc[manifest["tier"] == "joint", "object_id"].to_numpy()
    min_joint = int(cfg.sanity.min_joint_objects) if "sanity" in cfg else 0

    policy = str(cfg.splits.get("policy", "auto")) if "splits" in cfg else "auto"
    if policy not in ("auto", "stratified", "joint_only"):
        raise ValueError(
            f"splits.policy={policy!r} is not recognised; expected one of "
            "'auto', 'stratified', 'joint_only'."
        )

    if policy == "joint_only":
        use_joint_only = True
    elif policy == "stratified":
        use_joint_only = False
    else:
        use_joint_only = len(joint_ids) >= min_joint and len(joint_ids) > 0

    upstream = (
        manifest[MANIFEST_UPSTREAM_SPLIT_COLUMN]
        if bool(cfg.splits.get("honor_upstream", True))
        and MANIFEST_UPSTREAM_SPLIT_COLUMN in manifest.columns
        else None
    )
    n_unlabelled = len(manifest) if upstream is None else int(upstream.isna().sum())

    if use_joint_only:
        if len(joint_ids) == 0 and upstream is None:
            raise ValueError(
                "splits.policy='joint_only' but the manifest has no joint-tier objects, so val and "
                "test would be empty — which makes evaluate_loss return a fake 0.0 every epoch and "
                "silently truncates training via early stopping. Use 'stratified' until the joint "
                "crossmatch produces real joint objects."
            )
        eligible_ids = set(joint_ids)
    else:
        if policy == "auto":
            logger.warning(
                f"Only {len(joint_ids)} joint objects (< min_joint_objects={min_joint}) — val/test "
                "would be empty if drawn from joint-tier only, which silently breaks early stopping "
                "(fake 0.0 val loss every epoch). Falling back to drawing val/test from ALL tiers, "
                "stratified, until the joint crossmatch produces enough real joint objects."
            )
        else:
            logger.info(
                f"splits.policy='stratified': drawing val/test from all tiers, stratified "
                f"({len(joint_ids)} joint objects present; threshold not consulted)."
            )
        eligible_ids = set(manifest["object_id"])

    val_ids: set = set()
    test_ids: set = set()

    if upstream is None:
        val_ids, test_ids = _draw_per_tier(
            manifest, eligible_ids, cfg.splits.val, cfg.splits.test, rng, stratify_by,
            tier_fraction_overrides,
        )
    else:
        # NOTE: upstream-honouring path is dead code as configured — `honor_upstream: false` is
        # pinned in configs/data.yaml — but kept correct: a tier with a `test`-fraction override
        # of 0.0 would still gain test objects here from the source's OWN split labels, since
        # those are taken as given rather than drawn. Revisit if honor_upstream is ever flipped.
        test_ids |= set(manifest.loc[upstream == "test", "object_id"])

        carved, _ = _draw_per_tier(
            manifest, manifest.loc[upstream == "train", "object_id"], cfg.splits.val, 0.0, rng,
            stratify_by, tier_fraction_overrides,
        )
        val_ids |= carved

        unlabelled = set(manifest.loc[upstream.isna(), "object_id"]) & eligible_ids
        drawn_val, drawn_test = _draw_per_tier(
            manifest, unlabelled, cfg.splits.val, cfg.splits.test, rng, stratify_by,
            tier_fraction_overrides,
        )
        val_ids |= drawn_val
        test_ids |= drawn_test

        n_labelled = int(upstream.notna().sum())
        logger.info(
            f"Honouring the source's own split labels for {n_labelled}/{len(manifest)} objects "
            f"({int((upstream == 'test').sum())} test, {int((upstream == 'train').sum())} train); "
            f"carved {len(carved)} val out of the upstream-train pool and drew "
            f"{len(drawn_val)} val / {len(drawn_test)} test from the {n_unlabelled} objects with "
            "no upstream label."
        )

    manifest.loc[manifest["object_id"].isin(val_ids), "split"] = "val"
    manifest.loc[manifest["object_id"].isin(test_ids), "split"] = "test"

    assert manifest.groupby("object_id")["split"].nunique().max() <= 1, (
        "An object_id was assigned to more than one split — this is a correctness bug."
    )
    return manifest


def write_manifest(cfg: DictConfig) -> None:
    manifest, stats = build_manifest(cfg)
    manifest = assign_splits(manifest, cfg)

    split_hist = manifest["split"].value_counts().to_dict()
    stats["split_histogram"] = {k: int(v) for k, v in split_hist.items()}

    # Hard fail rather than warn: an empty train or test split silently breaks training (no data)
    # or makes evaluate_loss return a fake 0.0 every epoch and truncate training via early stop.
    # This is the "final training run" gate — must not proceed on a broken split.
    for required in ("train", "test"):
        if int(split_hist.get(required, 0)) == 0:
            raise ValueError(
                f"Split {required!r} is empty ({split_hist}). Check the upstream `split` column in "
                "the spectra parquet (configs/data.yaml sources.spectra.files) and "
                "cfg.splits.honor_upstream / policy."
            )

    n_upstream = (
        int(manifest[MANIFEST_UPSTREAM_SPLIT_COLUMN].notna().sum())
        if MANIFEST_UPSTREAM_SPLIT_COLUMN in manifest.columns
        else 0
    )
    honored = bool(cfg.splits.get("honor_upstream", True)) and n_upstream > 0
    stats["split_source"] = "upstream+seeded_draw" if honored else "seeded_draw"
    stats["n_upstream_labelled_objects"] = n_upstream if honored else 0
    if honored:
        upstream_hist = manifest[MANIFEST_UPSTREAM_SPLIT_COLUMN].value_counts().to_dict()
        stats["upstream_split_histogram"] = {str(k): int(v) for k, v in upstream_hist.items()}

    for subset_name, count in stats["tier_histogram"].items():
        if count < cfg.sanity.min_per_subset:
            logger.warning(
                f"Tier {subset_name!r} has only {count} objects "
                f"(< min_per_subset={cfg.sanity.min_per_subset})."
            )

    out_dir = Path(cfg.manifest.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(cfg.manifest.parquet, index=False)
    Path(cfg.manifest.stats).write_text(json.dumps(stats, indent=2))

    logger.info(f"Wrote manifest with {len(manifest)} rows to {cfg.manifest.parquet}")
    logger.info(f"Tier histogram: {stats['tier_histogram']}")
    logger.info(f"Split histogram: {stats['split_histogram']} (source: {stats['split_source']})")
    # Per-survey balance, logged every run. The whole reason v7 re-draws splits is that this was
    # never printed: `split_upstream` put 331/334 DESI spectra in test and nothing surfaced it.
    if "survey" in manifest.columns:
        balance = (
            manifest.assign(survey=manifest["survey"].fillna("(none)"))
            .groupby(["survey", "split"], sort=True)
            .size()
            .unstack(fill_value=0)
        )
        for survey, row in balance.iterrows():
            counts = {k: int(v) for k, v in row.items()}
            total = sum(counts.values())
            train_frac = counts.get("train", 0) / total if total else 0.0
            flag = "  <-- SKEWED" if total >= 50 and train_frac < 0.5 else ""
            logger.info(f"  split balance [{survey:<14}] {counts} train_frac={train_frac:.2f}{flag}")
