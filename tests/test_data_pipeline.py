"""Does training data assembly produce correct, leak-free batches?

Covers, in order: `build_manifest` (joining spectra/image/transients sources into one object-level
table), `assign_splits` (train/val/test assignment — the single most leak-prone step in this
pipeline), `CaptionerDataset` (per-example subset sampling and caption lookup), and `collate_batch`
(padding/masking into a model-ready batch). Each test traces back to a real production bug —
see each docstring.

Ported and consolidated from Infonioknight/Astrobridge-Train's test suite (test_manifest.py,
test_splits.py, test_assign_splits.py, test_dataset_caption_subset_sampling.py,
test_dataset_drops_uncaptionable_objects.py, test_degenerate_batch_guard.py, test_collate.py).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from captioner.data.cache import cache_dir_for
from captioner.data.collate import collate_batch
from captioner.data.dataset import CaptionerDataset
from captioner.data.manifest import assign_splits, build_manifest
from captioner.encoders.registry import encoder_hash, encoder_spec
from captioner.train.loop import _is_degenerate_batch
from tests.conftest import make_prompt_cfg

# --- build_manifest ---------------------------------------------------------------------------


def _manifest_cfg():
    return OmegaConf.create(
        {
            "sources": {"spectra": {"hf_path": "x", "split": "train"}, "image": {"hf_path": "y"}},
            "join": {"key": "_healpix_29", "fallback_radius_arcsec": 1.0},
            "sanity": {"min_joint_objects": 500, "min_per_subset": 50},
        }
    )


def _manifest_cfg_with_transients():
    cfg = _manifest_cfg()
    cfg.sources.transients = {"hf_path": "t", "revision": None}
    return cfg


def test_object_id_join():
    """Prefers an object_id join over coordinate crossmatching when available. An object present
    in only one source stays `tier="single"` with `has_<other>=False` instead of vanishing."""
    spectra_df = pd.DataFrame(
        {
            "object_id": ["a", "b", "c"], "ra_spectra": [1.0, 2.0, 3.0], "dec_spectra": [1.0, 2.0, 3.0],
            "spectrum": [{"flux": [1.0]}] * 3, "has_spectra": [True, True, True],
        }
    )
    # "z" has no matching spectra object_id — must not silently vanish or crash.
    image_df = pd.DataFrame(
        {
            "object_id": ["a", "z"], "object_id_legacy": ["0001m057-6125", "9999p000-0000"],
            "ra": [10.0, 30.0], "dec": [-5.0, -7.0], "has_image": [True, True],
        }
    )
    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ):
        manifest, stats = build_manifest(_manifest_cfg())

    assert stats["join_method"] == "object_id"
    by_id = manifest.set_index("object_id")
    assert by_id.loc["a", "tier"] == "joint"
    assert by_id.loc["b", "tier"] == "single"
    assert by_id.loc["z", "has_spectra"] == False  # noqa: E712
    assert by_id.loc["a", "object_id_legacy"] == "0001m057-6125"


def test_object_id_stringified(tmp_path):
    """object_id must be `str` everywhere even when sources mix Python `int` (DESI) and string
    ids (Legacy Survey, ZTF) — a leftover mixed-dtype column crashes `to_parquet`."""
    spectra_df = pd.DataFrame(
        {"object_id": [39627939041510701, 39633282643528688], "ra_spectra": [1.0, 2.0],
         "dec_spectra": [1.0, 2.0], "has_spectra": [True, True]}
    )
    image_df = pd.DataFrame({"object_id": [39627939041510701], "ra": [1.0], "dec": [1.0], "has_image": [True]})
    transients_df = pd.DataFrame(
        {"object_id": ["0001m057-6125"], "class_label": ["SN Ia"], "lc_mjd": [[1.0]],
         "atcat_length": [1], "has_lightcurve": [True]}
    )
    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ), patch("captioner.data.manifest._load_transients_table", return_value=transients_df):
        manifest, _ = build_manifest(_manifest_cfg_with_transients())

    assert manifest["object_id"].apply(type).eq(str).all()
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)  # the actual real-world failure mode
    assert set(manifest["object_id"]) == {"39627939041510701", "39633282643528688", "0001m057-6125"}


def test_transients_appended():
    """A ZTF designation shares no namespace with the rest of the manifest, so transients must
    arrive as NEW rows carrying has_lightcurve only — never merged onto an existing row."""
    spectra_df = pd.DataFrame({"object_id": ["a"], "ra_spectra": [1.0], "dec_spectra": [1.0], "has_spectra": [True]})
    image_df = pd.DataFrame({"object_id": ["a"], "ra": [1.0], "dec": [1.0], "has_image": [True]})
    transients_df = pd.DataFrame(
        {"object_id": ["ZTF0000000001", "ZTF0000000002"], "class_label": ["SN Ia", "SN II"],
         "lc_mjd": [[1.0], [2.0]], "atcat_length": [1, 1], "has_lightcurve": [True, True]}
    )
    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ), patch("captioner.data.manifest._load_transients_table", return_value=transients_df):
        manifest, stats = build_manifest(_manifest_cfg_with_transients())

    assert len(manifest) == 3
    by_id = manifest.set_index("object_id")
    assert by_id.loc["ZTF0000000001", "has_lightcurve"] == True  # noqa: E712
    assert by_id.loc["ZTF0000000001", "tier"] == "single"
    assert by_id.loc["a", "tier"] == "joint"


def test_colliding_ids_raise():
    """A genuine object_id collision among transients must raise, not silently duplicate rows."""
    spectra_df = pd.DataFrame({"object_id": ["dup"], "ra_spectra": [1.0], "dec_spectra": [1.0], "has_spectra": [True]})
    image_df = pd.DataFrame({"object_id": ["dup"], "ra": [1.0], "dec": [1.0], "has_image": [True]})
    transients_df = pd.DataFrame({"object_id": ["dup"], "atcat_length": [1], "has_lightcurve": [True]})
    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ), patch("captioner.data.manifest._load_transients_table", return_value=transients_df):
        with pytest.raises(ValueError, match="collide"):
            build_manifest(_manifest_cfg_with_transients())


# --- assign_splits -----------------------------------------------------------------------------


def _split_manifest(n_joint: int = 40, n_single: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_joint):
        rows.append({"object_id": f"joint_{i}", "tier": "joint", "has_image": True, "has_spectra": True})
    for i in range(n_single):
        rows.append({"object_id": f"single_{i}", "tier": "single", "has_image": (i % 2 == 0), "has_spectra": (i % 2 == 1)})
    return pd.DataFrame(rows)


def _split_cfg():
    return OmegaConf.create({"splits": {"val": 0.1, "test": 0.1, "seed": 0}})


def test_no_split_leak():
    """The core invariant: the same object_id can never land in two different splits."""
    result = assign_splits(_split_manifest(), _split_cfg())
    assert result.groupby("object_id")["split"].nunique().max() == 1


def test_val_test_joint_only():
    """By default, val/test are drawn only from joint-tier objects (every modality present)."""
    result = assign_splits(_split_manifest(), _split_cfg())
    non_train = result[result["split"] != "train"]
    assert (non_train["tier"] == "joint").all()


def test_stratified_fallback():
    """With too few joint objects (a real production incident — an id-namespace mismatch left 0
    joint objects), val/test must be drawn from all tiers instead of left empty. An empty val set
    makes evaluate_loss silently return 0.0 every epoch, which early stopping reads as done."""
    manifest = _split_manifest(n_joint=0, n_single=60)
    cfg = OmegaConf.create({"splits": {"val": 0.1, "test": 0.1, "seed": 0}, "sanity": {"min_joint_objects": 500}})
    result = assign_splits(manifest, cfg)
    assert (result["split"] == "val").sum() > 0
    assert (result["split"] == "test").sum() > 0


def test_upstream_test_honored():
    """An object the source already labeled `split_upstream="test"` must land in our test split,
    not get redrawn by our own seeded shuffle."""
    rows = (
        [{"object_id": f"up_train_{i}", "tier": "single", "split_upstream": "train"} for i in range(100)]
        + [{"object_id": f"up_test_{i}", "tier": "single", "split_upstream": "test"} for i in range(20)]
    )
    cfg = OmegaConf.create({"splits": {"policy": "stratified", "honor_upstream": True, "val": 0.1, "test": 0.1, "seed": 0},
                            "sanity": {"min_joint_objects": 500}})
    out = assign_splits(pd.DataFrame(rows), cfg).set_index("object_id")
    assert (out.loc[[f"up_test_{i}" for i in range(20)], "split"] == "test").all()


def test_survey_stratification():
    """Regression test for the v5/v6 bug: the upstream split put 331/334 DESI spectra in test,
    leaving the model trained on almost no DESI. Stratifying by `[tier, survey]` (not just tier,
    which can't see DESI vs SDSS within the spectra tier) keeps each survey at ~80/10/10 on its
    own instead of one survey nearly disappearing from train."""
    rows = (
        [{"object_id": f"sdss_{i}", "tier": "single", "survey": "sdss"} for i in range(1800)]
        + [{"object_id": f"desi_{i}", "tier": "single", "survey": "desi"} for i in range(334)]
        + [{"object_id": f"lc_{i}", "tier": "single", "survey": np.nan} for i in range(500)]
    )
    cfg = OmegaConf.create(
        {"splits": {"val": 0.1, "test": 0.1, "seed": 1337, "policy": "stratified", "honor_upstream": False,
                    "stratify_by": ["tier", "survey"]}, "sanity": {"min_joint_objects": 500}}
    )
    out = assign_splits(pd.DataFrame(rows), cfg)
    for survey in ["sdss", "desi"]:
        sub = out[out["survey"] == survey]
        frac = (sub["split"] == "train").mean()
        assert 0.75 < frac < 0.85, f"{survey} train fraction {frac:.2f} is not ~0.8"
        assert (sub["split"] == "test").sum() > 0, f"{survey} has no test objects"
    lc = out[out["survey"].isna()]  # light curves have no survey; must still be drawn, not all-train
    assert 0.75 < (lc["split"] == "train").mean() < 0.85


def _tier_override_manifest(n_per_tier: int = 100):
    rows = []
    for tier in ("spectra", "image", "lightcurve"):
        for i in range(n_per_tier):
            rows.append({"object_id": f"{tier}_{i}", "tier": tier, "split_upstream": None})
    return pd.DataFrame(rows)


def _tier_override_cfg(**overrides):
    splits = {"policy": "stratified", "honor_upstream": False, "val": 0.2, "test": 0.1, "seed": 0,
              "tier_overrides": {"spectra": {"val": 0.2, "test": 0.0}}}
    splits.update(overrides)
    return OmegaConf.create({"splits": splits, "sanity": {"min_joint_objects": 500}})


def test_spectra_no_test():
    """Spectra has no separate external held-out test set, unlike image/lightcurve — so its
    `test` fraction is overridden to 0.0 rather than holding data back from training with
    nowhere to use it."""
    out = assign_splits(_tier_override_manifest(), _tier_override_cfg())
    spectra = out[out["tier"] == "spectra"]
    assert (spectra["split"] != "test").all()
    assert (spectra["split"] == "val").sum() == 20


def test_override_is_scoped():
    """The spectra-only override must not leak into image/lightcurve's normal val/test fractions."""
    out = assign_splits(_tier_override_manifest(), _tier_override_cfg())
    for tier in ("image", "lightcurve"):
        sub = out[out["tier"] == tier]
        assert (sub["split"] == "val").sum() == 20
        assert (sub["split"] == "test").sum() == 10


# --- CaptionerDataset ----------------------------------------------------------------------------


def _sample_subset(available, weights, captioned, seed=0):
    fake_self = type("Fake", (), {"subset_weights": weights})()
    rng = np.random.default_rng(seed)
    return CaptionerDataset._sample_target_subset(fake_self, available, rng, captioned)


def test_subset_needs_caption():
    """The sampler must never pick a subset with no caption, even if the underlying data is
    available. Confirmed real gap: spectra-tier captions cover only a subset of has_spectra=True
    objects. Picking an uncaptioned subset anyway produces an empty training target — enough of
    those in one micro-batch and every label is IGNORE_INDEX, so cross-entropy divides 0/0 into
    NaN, which then poisons the optimizer's state for good."""
    weights = [(frozenset({"spectra"}), 0.5), (frozenset({"image", "spectra"}), 0.5)]
    captioned = {frozenset({"image", "spectra"})}
    for seed in range(50):
        shown = _sample_subset(frozenset({"image", "spectra"}), weights, captioned, seed)
        assert shown != frozenset({"spectra"})


def test_sampler_empty_fallback():
    """If nothing available is captioned, the sampler returns an empty selection, not a guess."""
    weights = [(frozenset({"spectra"}), 1.0)]
    assert _sample_subset(frozenset({"spectra"}), weights, captioned=set()) == frozenset()


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, return_tensors="pt"):
        ids = [ord(c) % 50 for c in text] or [0]
        return {"input_ids": torch.tensor([ids])}


def _write_dataset_cache(cache_root: Path, modality: str, modality_cfg, object_ids: list[str]) -> None:
    spec = encoder_spec(modality_cfg)
    spec_hash = encoder_hash(spec)
    target_dir = cache_dir_for(cache_root, modality, spec_hash)
    target_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ones((len(object_ids), 2, 3), dtype=np.float16)
    np.save(target_dir / "shard_00000.npy", arr)
    rows = [{"object_id": oid, "shard_file": "shard_00000.npy", "row_in_shard": i, "n_tokens": 2, "out_dim": 3}
            for i, oid in enumerate(object_ids)]
    pd.DataFrame(rows or [], columns=["object_id", "shard_file", "row_in_shard", "n_tokens", "out_dim"]).to_parquet(
        target_dir / "index.parquet", index=False
    )


@pytest.fixture
def _modalities_cfg_for_dataset():
    return OmegaConf.create(
        {
            "modalities": {
                "image": {"out_dim": 3, "max_tokens": 2, "encoder": {"impl": "fake", "hf_path": "x/img", "revision": None, "kwargs": {}}},
                "spectra": {"out_dim": 3, "max_tokens": 2, "encoder": {"impl": "fake", "hf_path": "x/spec", "revision": None, "kwargs": {}}},
            },
            "dropout": {"subset_weights": [
                {"subset": ["image"], "weight": 0.34}, {"subset": ["spectra"], "weight": 0.33},
                {"subset": ["image", "spectra"], "weight": 0.33},
            ]},
        }
    )


def test_uncaptioned_object_dropped(tmp_path, _modalities_cfg_for_dataset):
    """An object with no captioned subset at all is dropped once at construction time —
    otherwise it sits in the split forever producing empty-prompt/empty-caption examples on
    every visit, every epoch: pure dead weight."""
    manifest = pd.DataFrame([
        {"object_id": "keep", "has_image": False, "has_spectra": True, "split": "train"},
        {"object_id": "drop", "has_image": False, "has_spectra": True, "split": "train"},
    ])
    captions = pd.DataFrame([{"object_id": "keep", "subset": ["spectra"], "text": "a real caption"}])
    _write_dataset_cache(tmp_path, "image", _modalities_cfg_for_dataset.modalities.image, [])
    _write_dataset_cache(tmp_path, "spectra", _modalities_cfg_for_dataset.modalities.spectra, ["keep", "drop"])

    ds = CaptionerDataset(manifest, captions, _modalities_cfg_for_dataset, tmp_path, "train", _FakeTokenizer(), make_prompt_cfg())
    assert list(ds.manifest["object_id"]) == ["keep"]


def test_getitem_shape(tmp_path, _modalities_cfg_for_dataset):
    """A real `__getitem__` call returns the expected keys, and the caption target ends with the
    tokenized stop sequence (`<|im_end|>`) — without that, the model never learns to stop."""
    manifest = pd.DataFrame([{"object_id": "a", "has_image": False, "has_spectra": True, "split": "train"}])
    captions = pd.DataFrame([{"object_id": "a", "subset": ["spectra"], "text": "a real caption"}])
    _write_dataset_cache(tmp_path, "image", _modalities_cfg_for_dataset.modalities.image, [])
    _write_dataset_cache(tmp_path, "spectra", _modalities_cfg_for_dataset.modalities.spectra, ["a"])

    ds = CaptionerDataset(manifest, captions, _modalities_cfg_for_dataset, tmp_path, "train", _FakeTokenizer(), make_prompt_cfg())
    item = ds[0]

    assert set(item) == {"object_id", "shown", "modality_arrays", "pre_ids", "post_ids", "caption_ids"}
    suffix = ds._caption_suffix_ids
    assert torch.equal(item["caption_ids"][-len(suffix):], suffix)


# --- degenerate-batch guard ----------------------------------------------------------------------


def _batch_with_mask(caption_attn_mask: torch.Tensor) -> dict:
    return {"caption_attn_mask": caption_attn_mask, "object_id": ["a", "b"], "shown": [frozenset(), frozenset()]}


def test_zero_mask_degenerate():
    """A batch where every caption is fully masked (0 real label positions) must be flagged
    degenerate — this is the second line of defense on top of the sampler-level check above, in
    case anything else ever slips a fully-masked batch through to the loss."""
    assert _is_degenerate_batch(_batch_with_mask(torch.zeros(2, 5, dtype=torch.long))) is True


def test_one_token_is_enough():
    """Just one real (unmasked) caption token anywhere in the batch makes it usable."""
    mask = torch.zeros(2, 5, dtype=torch.long)
    mask[0, 0] = 1
    assert _is_degenerate_batch(_batch_with_mask(mask)) is False


# --- collate_batch -------------------------------------------------------------------------------


def _collate_example(shown_image: bool, shown_spectra: bool, image_T: int = 3, spectra_T: int = 2):
    return {
        "object_id": "obj",
        "shown": frozenset({"image"} if shown_image else set()) | frozenset({"spectra"} if shown_spectra else set()),
        "modality_arrays": {
            "image": np.random.randn(image_T, 8).astype(np.float32) if shown_image else None,
            "spectra": np.random.randn(spectra_T, 6).astype(np.float32) if shown_spectra else None,
        },
        "pre_ids": torch.tensor([1, 2, 3]), "post_ids": torch.tensor([6, 7]), "caption_ids": torch.tensor([4, 5]),
    }


def test_absent_modality_masked():
    """A modality an example doesn't have gets a mask that's entirely True (excluded), not a
    zero-content vector that looks like real (but blank) data."""
    examples = [_collate_example(shown_image=True, shown_spectra=False)]
    batch = collate_batch(examples, ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0)
    assert torch.all(batch["modality_batch"]["spectra"]["mask"])
    assert not torch.all(batch["modality_batch"]["image"]["mask"])


def test_caption_mask_accuracy():
    """When captions of different lengths are padded to the same width, only the real tokens are
    marked attend-to — padding must be excluded from the loss."""
    ex_short = _collate_example(True, True); ex_short["caption_ids"] = torch.tensor([9])
    ex_long = _collate_example(True, True); ex_long["caption_ids"] = torch.tensor([9, 9, 9])
    batch = collate_batch([ex_short, ex_long], ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0)
    assert batch["caption_attn_mask"][0].tolist() == [1, 0, 0]
    assert batch["caption_attn_mask"][1].tolist() == [1, 1, 1]


def test_pre_post_masked_separately():
    """The system-prompt text (pre) and instruction text (post) pad and mask independently per
    example — each example's mask reflects its own lengths, never mixed up across the batch."""
    ex_a = _collate_example(True, True); ex_a["pre_ids"] = torch.tensor([1, 2]); ex_a["post_ids"] = torch.tensor([6, 7, 8, 9])
    ex_b = _collate_example(True, True); ex_b["pre_ids"] = torch.tensor([1, 2, 3, 4]); ex_b["post_ids"] = torch.tensor([6, 7])
    batch = collate_batch([ex_a, ex_b], ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0)
    assert batch["pre_attn_mask"][0].tolist() == [1, 1, 0, 0]
    assert batch["post_attn_mask"][1].tolist() == [1, 1, 0, 0]
