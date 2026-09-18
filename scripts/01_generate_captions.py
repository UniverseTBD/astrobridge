#!/usr/bin/env python
"""Decomposes literature-derived text into per-tier captions and runs the leakage-validation
CI gate (§4, §9 step 3). Gate: zero leakage violations, plus claim-survival rate reported, plus
a sample written out for the required manual read of 50 captions spanning both tiers.

Three caption sources, kept deliberately separate:
  - spectra: a teammate-produced, spectrum-grounded LLM caption set (Gemini —
    UniverseTBD/AstroBridge-Data's captions/spectra/*.jsonl, see data/spectra_dataset.py:
    load_gemini_spectra_captions). Covers a subset of objects (2,021 as of the configured
    filename, v2). Deliberate scope choice: spectra-tier captions are restricted to ONLY this set
    for now — objects with spectra but no Gemini caption get no spectra-tier caption at all,
    same asymmetry the image side already has (not every has_image object has a caption_fused
    match either). No fallback to mention_summary decomposition for the rest.
  - image: gapatron's own pre-vetted caption field, `caption_fused` unless
    configs/data.yaml's `sources.image.caption_field` says otherwise (data/image_dataset.py) —
    same reasoning as spectra above: pre-vetted, modality-restricted, preferred over decomposing
    text ourselves. Looked up by the image source's own id (`object_id_legacy` on the manifest
    row), since joint-tier rows are keyed by the spectra side's object_id instead.
  - relational/joint claims (dormant — milestone 1 doesn't train on the joint tier, see
    configs/modalities.yaml's dropout weights): still decomposed from AstroBridge-Data's
    `mention_summary` via decompose_object, since Gemini's spectra captions and gapatron's image
    captions are both single-modality-only and can't produce a claim that needs both together.
    Pure single-modality claims from this decomposition are always dropped in favor of the two
    pre-vetted sources above — decompose_object's only live output now is relational claims, so
    `claim_survival_rate` in the report below is really measuring "how much of mention_summary
    produces joint claims," not overall claim survival — a low number here is expected, not a
    sign of something broken.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from captioner.data.captions import (
    Caption,
    Claim,
    compose_captions,
    decompose_object,
    claim_survival_rate,
    _split_sentences,
)
from captioner.data.image_dataset import load_image_captions_table
from captioner.data.spectra_dataset import load_gemini_spectra_captions, load_spectra_table
from captioner.data.transients_dataset import load_transient_captions
from captioner.eval.claims import claim_kind_histogram, validate_all
from captioner.utils.config import load_config
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config("base", "data")
    manifest = pd.read_parquet(cfg.manifest.parquet)

    # AstroBridge-Data carries the literature-derived fields directly.
    spectra_ds = load_spectra_table(
        cfg.sources.spectra.hf_path,
        revision=cfg.sources.spectra.get("revision"),
        files=list(cfg.sources.spectra.get("files") or []) or None,
    )
    text_by_object = spectra_ds.set_index("object_id")[
        ["mention_summary", "evidence_quotes", "arxiv_id"]
    ].to_dict(orient="index")

    # Gemini spectra captions are keyed by wiki_entity_id, not our canonical object_id — build
    # the mapping from the same spectra_ds row (both columns confirmed present on AstroBridge-Data).
    wiki_to_object_id = spectra_ds.dropna(subset=["wiki_entity_id"]).set_index("wiki_entity_id")["object_id"].to_dict()
    gemini_captions_df = load_gemini_spectra_captions(
        cfg.sources.spectra_captions.hf_path,
        cfg.sources.spectra_captions.filename,
        revision=cfg.sources.spectra_captions.get("revision"),
    )
    n_gemini_unmatched_wiki_id = 0
    spectra_gemini_caption_by_object: dict[str, str] = {}
    for _, grow in gemini_captions_df.iterrows():
        oid = wiki_to_object_id.get(grow["wiki_entity_id"])
        if oid is None:
            n_gemini_unmatched_wiki_id += 1
            continue
        spectra_gemini_caption_by_object[oid] = grow["caption"]

    image_captions_df = load_image_captions_table(
        cfg.sources.image.hf_path,
        revision=cfg.sources.image.get("revision"),
        caption_field=cfg.sources.image.get("caption_field", "caption_fused"),
        surveys=list(cfg.sources.image.get("surveys") or []) or None,
    )
    # load_image_captions_table normalises whichever caption_field was selected to the column
    # name `caption_blind` (see data/image_dataset.py), so the lookup here does not change when
    # the stage does. train-v2 names that column `caption` instead — do not "fix" this to match.
    image_caption_by_object = image_captions_df.set_index("object_id")["caption_blind"].to_dict()

    # Transients ship their own caption, used RAW here (no keyword veto — see module docstring
    # and RETRAIN_SKETCH.md B3). It is not pre-vetted like caption_fused / the Gemini captions,
    # and it does assert spectroscopic type / redshift / designation — kept on purpose now.
    transient_caption_by_object: dict[str, str] = {}
    if "transients" in cfg.sources:
        transient_captions_df = load_transient_captions(
            cfg.sources.transients.hf_path, revision=cfg.sources.transients.get("revision")
        )
        transient_caption_by_object = (
            transient_captions_df.set_index("object_id")["transient_caption"].to_dict()
        )

    all_captions: list[Caption] = []
    n_source_sentences = 0
    n_surviving = 0
    n_image_from_gapatron = 0
    n_image_available = 0
    n_spectra_from_gemini = 0
    n_spectra_available = 0
    n_lightcurve_available = 0
    n_lightcurve_kept = 0
    n_no_usable_text = 0
    class_label_by_object: dict[str, str] = {}

    for _, row in manifest.iterrows():
        object_id = row["object_id"]
        # Derived from the manifest's own has_<modality> columns rather than a hardcoded tuple, so
        # adding a modality needs no change here.
        available = frozenset(
            c[len("has_") :] for c in manifest.columns if c.startswith("has_") and bool(row[c])
        )
        if not available:
            continue

        text_row = text_by_object.get(object_id)
        claims: list[Claim] = []

        if text_row is not None:
            claims = decompose_object(
                object_id=object_id,
                mention_summary=text_row.get("mention_summary") or "",
                evidence_quotes=text_row.get("evidence_quotes"),
                arxiv_id=text_row.get("arxiv_id"),
                available_modalities=available,
                generator=cfg.captions.generator,
            )
            n_source_sentences += len(_split_sentences(text_row.get("mention_summary") or ""))
            n_surviving += len(claims)

            # Pure single-modality claims are always dropped in favor of the pre-vetted sources
            # below (Gemini for spectra, caption_fused for image) — only relational/joint claims
            # from this decomposition are ever used. See module docstring.
            claims = [c for c in claims if len(c.supporting) > 1]

        if "image" in available:
            # manifest's canonical object_id is AstroBridge-Data's id (from
            # target_object_id_target); the caption parquet is keyed by the Legacy Survey's
            # own naming (object_id_legacy) — different namespace, so look up by that instead.
            n_image_available += 1
            image_lookup_id = row.get("object_id_legacy") or object_id
            fused = image_caption_by_object.get(image_lookup_id)
            if fused:
                claims.append(
                    Claim(
                        text=fused,
                        supporting=frozenset({"image"}),
                        kind="observation",
                        provenance=f"gapatron:{object_id}",
                    )
                )
                n_image_from_gapatron += 1

        if "spectra" in available:
            n_spectra_available += 1
            gemini_caption = spectra_gemini_caption_by_object.get(object_id)
            if gemini_caption:
                claims.append(
                    Claim(
                        text=gemini_caption,
                        supporting=frozenset({"spectra"}),
                        kind="observation",
                        provenance=f"gemini:{object_id}",
                    )
                )
                n_spectra_from_gemini += 1

        if "lightcurve" in available:
            n_lightcurve_available += 1
            raw_caption = transient_caption_by_object.get(object_id)
            if raw_caption:
                # Used raw, same as image/spectra above — NO keyword veto. The transient caption
                # asserts spectroscopic type / redshift / designation; we now keep that on
                # purpose (RETRAIN_SKETCH.md B3). See scripts docstring.
                claims.append(
                    Claim(
                        text=raw_caption,
                        supporting=frozenset({"lightcurve"}),
                        kind="observation",
                        provenance=f"transient_lc:{object_id}",
                    )
                )
                n_lightcurve_kept += 1

        if "class_label" in row and not pd.isna(row.get("class_label")):
            class_label_by_object[object_id] = row["class_label"]

        if not claims:
            n_no_usable_text += 1
            continue

        captions = compose_captions(object_id, claims, available)
        all_captions.extend(captions)

    if n_lightcurve_available:
        logger.info(
            f"Lightcurve tier: kept {n_lightcurve_kept}/{n_lightcurve_available} transient "
            "captions RAW (no keyword veto). These assert spectroscopic type / redshift / "
            "designation — read outputs/captions/manual_review_sample.jsonl before training."
        )

    violations = validate_all(all_captions)
    survival_rate = claim_survival_rate(n_source_sentences, n_surviving)
    kind_hist = claim_kind_histogram(all_captions)

    image_caption_match_rate = n_image_from_gapatron / n_image_available if n_image_available else None
    if image_caption_match_rate is not None and image_caption_match_rate < 0.5:
        logger.warning(
            f"Only {image_caption_match_rate:.1%} of image-available objects "
            f"({n_image_from_gapatron}/{n_image_available}) found a caption match. Captions and "
            "flux now come from the same rows of the same dataset "
            f"({cfg.sources.image.hf_path}), so this should be ~100% — anything less means the "
            "manifest's object_id_legacy has drifted from the image dataset (a stale manifest "
            "against a newer dataset revision), or rows were dropped for having no caption text "
            f"in {cfg.sources.image.get('caption_field', 'caption_fused')!r}."
        )

    spectra_caption_match_rate = n_spectra_from_gemini / n_spectra_available if n_spectra_available else None
    if n_gemini_unmatched_wiki_id > 0:
        logger.warning(
            f"{n_gemini_unmatched_wiki_id}/{len(gemini_captions_df)} Gemini captions' "
            "wiki_entity_id had no match in AstroBridge-Data's wiki_entity_id column — those "
            "captions were dropped rather than silently misattributed. If this is a large "
            "fraction, the id namespace assumption may not hold as cleanly as the spot-check "
            "suggested; worth a closer look."
        )

    report = {
        "n_objects_processed": len({c.object_id for c in all_captions}),
        "n_captions": len(all_captions),
        "n_leakage_violations": len(violations),
        "claim_survival_rate": survival_rate,
        "claim_kind_histogram": kind_hist,
        "n_image_captions_from_gapatron_fused": n_image_from_gapatron,
        "n_image_available": n_image_available,
        "image_caption_match_rate": image_caption_match_rate,
        "n_spectra_captions_from_gemini": n_spectra_from_gemini,
        "n_spectra_available": n_spectra_available,
        "spectra_caption_match_rate": spectra_caption_match_rate,
        "n_gemini_captions_total": len(gemini_captions_df),
        "n_gemini_unmatched_wiki_id": n_gemini_unmatched_wiki_id,
        "n_lightcurve_available": n_lightcurve_available,
        "n_lightcurve_captions_kept": n_lightcurve_kept,
        "n_objects_with_no_usable_text": n_no_usable_text,
        "generator": cfg.captions.generator,
    }

    out_dir = Path(cfg.captions.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions_df = pd.DataFrame(
        [
            {
                "object_id": c.object_id,
                "subset": sorted(c.subset),
                "text": c.text,
                "n_claims": len(c.claims),
                "source": c.source,
                "generator": c.generator,
                # Structured metadata for eval slicing only — never concatenated into caption text.
                "class_label": class_label_by_object.get(c.object_id),
            }
            for c in all_captions
        ]
    )
    captions_df.to_parquet(cfg.captions.parquet, index=False)
    Path(cfg.captions.report).write_text(json.dumps(report, indent=2))

    # Sample 50 captions spanning both tiers for the required manual read (§9 step 3 gate).
    rng = random.Random(0)
    single = [c for c in all_captions if len(c.subset) == 1]
    joint = [c for c in all_captions if len(c.subset) > 1]
    sample = rng.sample(single, min(25, len(single))) + rng.sample(joint, min(25, len(joint)))
    sample_path = out_dir / "manual_review_sample.jsonl"
    with open(sample_path, "w") as f:
        for c in sample:
            f.write(json.dumps({"object_id": c.object_id, "subset": sorted(c.subset), "text": c.text}) + "\n")

    logger.info(f"Captions: {report}")
    logger.info(f"Manual review sample written to {sample_path}")

    if violations:
        logger.error(f"{len(violations)} leakage violations found — gate failed. Examples:")
        for v in violations[:10]:
            logger.error(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
