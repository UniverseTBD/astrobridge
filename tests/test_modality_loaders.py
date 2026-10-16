"""Does each modality's raw-data loader actually match what the real upstream repos contain?

This is the file that would have caught the `sources.image`/`sources.image_captions` split
regression in `scripts/01_generate_captions.py`: `load_image_captions_table`'s output column
contract is pinned directly below, not just documented in a comment. The spectra tests cover the
real multi-file, multi-spelling object_id mess AstroBridge-Data actually has — each one traces to
a production incident, see its docstring.

Ported and consolidated from Infonioknight/Astrobridge-Train's test suite (test_image_dataset.py,
test_spectra_dataset.py, test_gemini_spectra_captions.py), with the Gemini-caption tests adapted
to this repo's own (stricter) `load_gemini_spectra_captions`/`load_spectra_captions_jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from captioner.data.image_dataset import IMAGE_CAPTION_COLUMN, load_image_captions_table
from captioner.data.spectra_dataset import (
    _canonical_object_id,
    _canonicalize_object_ids,
    _deduplicate_by_object_id,
    load_gemini_spectra_captions,
    load_spectra_table,
)

# --- image captions: load_image_captions_table --------------------------------------------------


def _write_image_shard(tmp_path, rows, name="data/train-00000-of-00001.parquet"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def _patch_image_hub(tmp_path, monkeypatch):
    import huggingface_hub

    shards: dict[str, str] = {}

    def _set(rows_by_file: dict[str, list[dict]]):
        for name, rows in rows_by_file.items():
            shards[name] = str(_write_image_shard(tmp_path, rows, name))

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **kw: list(shards.keys()))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: shards[filename])
    return _set


def test_caption_column_name(_patch_image_hub):
    """load_image_captions_table always normalizes the configured caption field down to a column
    literally named `caption` — the exact contract the sources.image/sources.image_captions
    regression broke by reading the stale name `caption_blind` instead."""
    _patch_image_hub({
        "data/train-00000-of-00001.parquet": [
            {"object_id": "a", IMAGE_CAPTION_COLUMN: "A compact source.", "flux_g": [1, 2]},
            {"object_id": "b", IMAGE_CAPTION_COLUMN: "An extended disk.", "flux_g": [3, 4]},
        ],
    })
    df = load_image_captions_table("gapatron/astrobridge-image-captions")
    assert list(df.columns) == ["object_id", "caption"]
    assert dict(zip(df["object_id"], df["caption"])) == {"a": "A compact source.", "b": "An extended disk."}


def test_missing_caption_dropped(_patch_image_hub):
    """A row with no caption text is dropped, not kept as an empty string."""
    _patch_image_hub({
        "data/train-00000-of-00001.parquet": [
            {"object_id": "a", IMAGE_CAPTION_COLUMN: "A galaxy."},
            {"object_id": "b", IMAGE_CAPTION_COLUMN: None},
        ],
    })
    df = load_image_captions_table("irrelevant/repo")
    assert list(df["object_id"]) == ["a"]


def test_shards_concatenated(_patch_image_hub):
    """Captions split across multiple parquet shards are all read and combined."""
    _patch_image_hub({
        "data/train-00000-of-00002.parquet": [{"object_id": "a", IMAGE_CAPTION_COLUMN: "One."}],
        "data/train-00001-of-00002.parquet": [{"object_id": "b", IMAGE_CAPTION_COLUMN: "Two."}],
    })
    df = load_image_captions_table("gapatron/astrobridge-image-captions")
    assert set(df["object_id"]) == {"a", "b"}


def test_no_shards_raises(monkeypatch):
    """Pointing this loader at a repo with no data/*.parquet (like the pixel-only image repo)
    raises FileNotFoundError instead of silently returning nothing."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **kw: ["README.md"])
    with pytest.raises(FileNotFoundError, match="parquet"):
        load_image_captions_table("gapatron/astrobridge-image-captions")


# --- spectra data: load_spectra_table -------------------------------------------------------------


def _fake_spectra_repo(tmp_path, filenames):
    return (
        patch("huggingface_hub.list_repo_files", return_value=[f"observations/spectra/{n}" for n in filenames]),
        patch("huggingface_hub.hf_hub_download", side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name)),
    )


def test_mismatched_columns_ok(tmp_path):
    """AstroBridge-Data's real multi-file layout: different crossmatch-source parquet files
    carry different columns, which `datasets.load_dataset` can't handle (it demands one strict
    schema across every file). pd.concat must fill missing columns with NaN instead of erroring."""
    pd.DataFrame({"object_id": ["a", "b"], "mention_summary": ["x", "y"], "Z": [0.1, 0.2]}).to_parquet(tmp_path / "desi_only.parquet")
    pd.DataFrame({
        "object_id": ["c", "d"], "mention_summary": ["z", "w"], "Z": [0.3, 0.4],
        "survey_metadata_extra": ["stuff", "stuff"],  # extra column absent from the other file
    }).to_parquet(tmp_path / "sdss_only.parquet")

    listed, download = _fake_spectra_repo(tmp_path, ["desi_only.parquet", "sdss_only.parquet"])
    with listed, download:
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert len(df) == 4
    by_id = df.set_index("object_id")
    assert pd.isna(by_id.loc["a", "survey_metadata_extra"])
    assert by_id.loc["a", "survey"] == "desi"  # single-survey filenames determine `survey` directly
    assert by_id.loc["c", "survey"] == "sdss"


def test_dedup_prefers_closest():
    """When an object appears more than once, the closest crossmatch (smallest _dist_arcsec)
    wins the tie-break, not an arbitrary row."""
    df = pd.DataFrame({"object_id": ["x", "x", "a"], "mention_summary": ["worse match", "better match", "only one"], "_dist_arcsec": [5.0, 0.05, 0.1]})
    deduped = _deduplicate_by_object_id(df)
    assert deduped["object_id"].is_unique
    assert deduped.set_index("object_id").loc["x", "mention_summary"] == "better match"


def test_canonical_id_collapse():
    """The three real spellings of one object_id across AstroBridge-Data's files — a plain digit
    string, a raw int64, and a stringified Python bytes repr of a padded FITS field — must all
    collapse to the same string. This is what prevents the same physical object from reaching the
    manifest under ids that look unrelated, which let a train/val split leak past its own
    per-object uniqueness check."""
    assert _canonical_object_id("b'    462849895556999168'") == "462849895556999168"
    assert _canonical_object_id(462849895556999168) == "462849895556999168"
    assert _canonical_object_id("462849895556999168") == "462849895556999168"
    assert _canonical_object_id(b"    462849895556999168") == "462849895556999168"
    assert _canonical_object_id("b1234") == "b1234"  # merely starts with "b" — not a bytes repr


def test_missing_id_rejected():
    """A file missing the object_id column entirely must raise, not silently collapse every row
    onto one bogus object — str(nan) == 'nan', so this has to be checked explicitly."""
    df = pd.DataFrame({"object_id": ["a", None], "mention_summary": ["x", "y"]})
    with pytest.raises(ValueError, match="object_id"):
        _canonicalize_object_ids(df)


def test_three_spellings_one_row(tmp_path):
    """Full integration check, the exact production failure: the same object arrived as a plain
    digit string, an int64, and a stringified bytes repr across three different files. Loaded
    together they must end up as exactly one row, under one id, with the right survey tag
    (DESI/SDSS routing matters downstream for which AION codec reads it)."""
    pd.DataFrame({"object_id": ["39632951360620188"], "mention_summary": ["desi text"], "_dist_arcsec": [0.5]}).to_parquet(tmp_path / "desi_crossmatch.parquet")
    pd.DataFrame({"object_id": ["b'    462849895556999168'"], "mention_summary": ["sdss text"], "_dist_arcsec": [0.4]}).to_parquet(tmp_path / "sdss_crossmatch.parquet")
    pd.DataFrame({
        "object_id": [39632951360620188, 462849895556999168], "mention_summary": ["desi text again", "sdss text again"],
        "survey": ["DESI", "SDSS"], "_dist_arcsec": [0.1, 0.9],
    }).to_parquet(tmp_path / "desi_sdss_subset_crossmatch.parquet")

    listed, download = _fake_spectra_repo(tmp_path, ["desi_crossmatch.parquet", "sdss_crossmatch.parquet", "desi_sdss_subset_crossmatch.parquet"])
    with listed, download:
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert set(df["object_id"]) == {"39632951360620188", "462849895556999168"}
    assert df["object_id"].is_unique
    by_id = df.set_index("object_id")
    assert by_id.loc["39632951360620188", "survey"] == "desi"
    assert by_id.loc["462849895556999168", "survey"] == "sdss"


# --- spectra captions: load_gemini_spectra_captions -----------------------------------------------


def _gemini_record(object_key: str | None, caption: str | None, is_insufficient: bool = False) -> dict:
    return {
        "object_key": object_key, "dataset_source": "sdss", "ra": 139.5, "dec": 4.4,
        "strategy": "combined_v1", "model": "gemini-3.7-flash",
        "output": {
            "caption": caption,
            "thought_summaries": ["long chain-of-thought text that must never end up as a caption"],
            "is_insufficient": is_insufficient,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    }


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_drops_bad_rows(tmp_path):
    """Rows flagged is_insufficient, missing a caption, or missing an object key are all
    dropped — none should end up as a training target."""
    path = tmp_path / "captions.jsonl"
    _write_jsonl(path, [
        _gemini_record("gmw_00000046", "A real caption about the spectrum."),
        _gemini_record("gmw_00000057", "Another real caption.", is_insufficient=True),
        _gemini_record("gmw_00000099", None),
        _gemini_record(None, "orphan"),
    ])
    df = load_gemini_spectra_captions(local_path=path)
    assert len(df) == 1
    assert df.iloc[0]["wiki_entity_id"] == "gmw_00000046"
    assert "thought_summaries" not in df.columns


def test_sentinel_caption_dropped(tmp_path):
    """The literal string "INSUFFICIENT_SPECTRAL_DATA" as a caption value is treated the same as
    is_insufficient=True — captioner's own loader never checks for this sentinel at all, so if it
    shows up in the real data there, it could be trained on as if it were real caption text."""
    path = tmp_path / "captions.jsonl"
    _write_jsonl(path, [
        _gemini_record("gmw_1", "A real caption."),
        _gemini_record("gmw_2", "INSUFFICIENT_SPECTRAL_DATA"),
    ])
    df = load_gemini_spectra_captions(local_path=path)
    assert list(df["wiki_entity_id"]) == ["gmw_1"]


def test_conflicting_captions_raise(tmp_path):
    """Two different caption texts for the same object is a data integrity problem — it must
    raise, not silently pick one."""
    path = tmp_path / "captions.jsonl"
    _write_jsonl(path, [_gemini_record("gmw_1", "Caption A."), _gemini_record("gmw_1", "Caption B.")])
    with pytest.raises(ValueError, match="[Cc]onflicting"):
        load_gemini_spectra_captions(local_path=path)


def test_identical_dupes_merge(tmp_path):
    """The same object with the same caption text twice collapses to one row without raising."""
    path = tmp_path / "captions.jsonl"
    _write_jsonl(path, [_gemini_record("gmw_1", "Same caption."), _gemini_record("gmw_1", "Same caption.")])
    df = load_gemini_spectra_captions(local_path=path)
    assert len(df) == 1


def test_local_path_no_download(tmp_path, monkeypatch):
    """A supplied local_path is used as-is and never triggers a network download, even if
    hf_path/filename are also given."""
    import huggingface_hub

    def _boom(*a, **k):
        raise AssertionError("should not download when local_path is supplied")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    path = tmp_path / "captions.jsonl"
    _write_jsonl(path, [_gemini_record("gmw_1", "hi")])
    df = load_gemini_spectra_captions(hf_path="x", filename="y", local_path=path)
    assert len(df) == 1


def test_missing_source_raises():
    """Calling with neither local_path nor both hf_path and filename raises a clear ValueError."""
    with pytest.raises(ValueError, match="local_path"):
        load_gemini_spectra_captions()
