from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from captioner.data.spectra_captioning import (
    STRATEGIES,
    UPSTREAM_REVISION,
    generate_spectra_captions,
    prepare_groups,
)
from captioner.data.spectra_dataset import load_gemini_spectra_captions
from captioner.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def caption_record(key="gmw_a", caption="A spectrum with strong emission lines.", **output):
    return {"object_key": key, "output": {"caption": caption, **output}}


@pytest.fixture
def observations():
    wavelength = np.linspace(3600.0, 9800.0, 160)
    return pd.DataFrame([
        {
            "wiki_entity_id": f"gmw_{suffix}",
            "object_id": f"observation_{suffix}",
            "survey": survey,
            "ra_mentions": 150.0,
            "dec_mentions": 2.0,
            "Z": 0.1,
            "mention_id": f"mention_{suffix}",
            "mention_summary": "",
            "arxiv_id": None,
            "evidence_quotes": np.array([
                {"quote": "The spectrum displays prominent emission lines above the continuum."}
            ]),
            "spectrum": {
                "lambda": wavelength,
                "flux": np.linspace(1.0, 2.0, 160),
                "ivar": np.ones(160),
                "mask": np.zeros(160, dtype=bool),
            },
        }
        for suffix, survey in (("a", "desi"), ("b", "sdss"))
    ])


@pytest.fixture
def generation_config():
    return {
        "captioning": {
            "strategy": "combined_v2",
            "model": "offline-test-model",
            "thinking_level": "low",
            "thinking_summaries": "auto",
        },
        "crossmatch": {"radius_arcsec": 1.0},
    }


@pytest.fixture
def client_factory():
    pytest.importorskip("spectra_captioning", reason="Optional upstream captioning dependency")
    from spectra_captioning.models.gemini import GeminiResponse

    class StubGemini:
        def __init__(self, responses=None):
            self.responses = list(responses or [])
            self.calls = []

        def generate(self, prompt, images=None):
            self.calls.append((prompt, images))
            result = self.responses.pop(0) if self.responses else "A spectrum with strong emission lines."
            if isinstance(result, Exception):
                raise result
            return GeminiResponse(text=result, input_tokens=10, output_tokens=8, total_tokens=18)

    return StubGemini


def test_local_captions_take_precedence_and_filter_unusable_rows(tmp_path, monkeypatch):
    def no_download(**kwargs):
        pytest.fail("Local captions must not trigger a Hugging Face download")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_download)
    path = write_records(tmp_path / "captions.jsonl", [
        caption_record(caption="  A spectrum with strong emission lines.  "),
        caption_record(),
        caption_record("gmw_insufficient", is_insufficient=True),
        caption_record("gmw_sentinel", caption=" INSUFFICIENT_SPECTRAL_DATA "),
        caption_record("gmw_blank", caption=" "),
        {"object_key": "gmw_missing"},
    ])
    result = load_gemini_spectra_captions("unused/repo", "unused.jsonl", local_path=path)
    assert result.to_dict("records") == [{
        "wiki_entity_id": "gmw_a", "caption": "A spectrum with strong emission lines.",
    }]
    with pytest.raises(FileNotFoundError):
        load_gemini_spectra_captions(local_path=tmp_path / "missing.jsonl")


def test_remote_captions_keep_existing_download_contract(tmp_path, monkeypatch):
    path = write_records(tmp_path / "remote.jsonl", [caption_record()])
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    result = load_gemini_spectra_captions("org/data", "captions.jsonl", "revision", tmp_path)
    assert len(result) == 1
    assert calls == [{
        "repo_id": "org/data", "filename": "captions.jsonl", "repo_type": "dataset",
        "revision": "revision", "cache_dir": str(tmp_path),
    }]


def test_empty_caption_source_keeps_column_schema(tmp_path):
    path = write_records(tmp_path / "captions.jsonl", [caption_record(is_insufficient=True)])
    result = load_gemini_spectra_captions(local_path=path)
    assert result.empty
    assert list(result.columns) == ["wiki_entity_id", "caption"]


@pytest.mark.parametrize("record,match", [
    (caption_record(caption="A contradictory caption."), "Conflicting captions"),
    ({"object_key": "gmw_b", "output": "bad"}, "Expected an output object"),
    (caption_record("gmw_b", caption=["bad"]), "Expected a caption string"),
    (["bad"], "Expected a spectra caption object"),
])
def test_invalid_records_fail_with_line_number(tmp_path, record, match):
    path = write_records(tmp_path / "captions.jsonl", [caption_record(), record])
    with pytest.raises(ValueError, match=match) as error:
        load_gemini_spectra_captions(local_path=path)
    assert f"{path}:2" in str(error.value)


def test_malformed_json_has_actionable_error(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"object_key":')
    with pytest.raises(ValueError, match="Invalid spectra caption JSON"):
        load_gemini_spectra_captions(local_path=path)


def test_groups_keep_all_mentions_and_normalize_parquet_quotes(tmp_path, observations):
    extra = observations.iloc[[0]].copy()
    extra["mention_id"] = "another_mention"
    frame = pd.concat([observations.iloc[[1]], extra, observations.iloc[[0]]], ignore_index=True)
    path = tmp_path / "spectra.parquet"
    frame.to_parquet(path)
    groups = prepare_groups(pd.read_parquet(path), limit=1)
    assert len(groups) == 1
    key, group = groups[0]
    assert key == "gmw_a"
    assert len(group) == 2
    assert isinstance(group.iloc[0]["evidence_quotes"], list)
    assert len(prepare_groups(observations, limit=0)) == 2
    assert prepare_groups(observations, object_ids=["gmw_b"])[0][0] == "gmw_b"
    with pytest.raises(ValueError, match="Unknown wiki_entity_id"):
        prepare_groups(observations, object_ids=["unknown"])
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_groups(observations, limit=-1)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_real_upstream_strategies_roundtrip(
    tmp_path, observations, generation_config, client_factory, strategy,
):
    config = copy.deepcopy(generation_config)
    config["captioning"]["strategy"] = strategy
    if strategy == "combined_ground_truth_v1":
        (tmp_path / "extracted_types.csv").write_text("wiki_entity_id,class,subclass\ngmw_a,QSO,BROADLINE\n")
        (tmp_path / "extracted_emission_lines.csv").write_text("wiki_entity_id,LINE_NAME,SNR\ngmw_a,OIII_5007,12.0\n")
        config["captioning"]["metadata_dir"] = str(tmp_path)
    client = client_factory()
    output = tmp_path / "captions.jsonl"
    stats = generate_spectra_captions(prepare_groups(observations, limit=1), output, config, client=client)
    record = json.loads(output.read_text())
    assert stats == {"selected": 1, "skipped": 0, "generated": 1, "insufficient": 0}
    assert record["object_key"] == "gmw_a"
    assert record["dataset_source"] == "desi"
    assert record["strategy"] == strategy
    assert record["input"]["num_quotes"] == 1
    assert record["usage"]["total_tokens"] == 18
    assert record["provenance"]["spectra_captioning_revision"] == UPSTREAM_REVISION
    assert load_gemini_spectra_captions(local_path=output).iloc[0]["wiki_entity_id"] == "gmw_a"
    prompt, images = client.calls[0]
    if strategy == "quotes_only_v3":
        assert images is None
    else:
        assert images[0].startswith(b"\x89PNG")
    if strategy == "combined_ground_truth_v1":
        assert "Quasar" in prompt
        assert "[O III] 5007" in prompt


def test_interrupt_and_resume_only_generates_pending_objects(
    tmp_path, observations, generation_config, client_factory, monkeypatch,
):
    groups = prepare_groups(observations)
    output = tmp_path / "captions.jsonl"
    first = client_factory(["First caption.", RuntimeError("Simulated API failure")])
    with pytest.raises(RuntimeError, match="Simulated API failure"):
        generate_spectra_captions(groups, output, generation_config, client=first)
    assert len(output.read_text().splitlines()) == 1
    resumed = client_factory(["Second caption."])
    stats = generate_spectra_captions(groups, output, generation_config, resume=True, client=resumed)
    assert stats["skipped"] == stats["generated"] == 1
    assert len(resumed.calls) == 1
    assert len(load_gemini_spectra_captions(local_path=output)) == 2
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    stats = generate_spectra_captions(groups, output, generation_config, resume=True)
    assert stats["skipped"] == 2
    assert stats["generated"] == 0


def test_resume_handles_last_record_without_newline(
    tmp_path, observations, generation_config, client_factory,
):
    groups = prepare_groups(observations)
    output = tmp_path / "captions.jsonl"
    generate_spectra_captions(groups[:1], output, generation_config, client=client_factory())
    output.write_text(output.read_text().rstrip())
    generate_spectra_captions(groups, output, generation_config, resume=True, client=client_factory())
    assert len(load_gemini_spectra_captions(local_path=output)) == 2


def test_existing_output_and_changed_inputs_are_rejected(
    tmp_path, observations, generation_config, client_factory,
):
    groups = prepare_groups(observations, limit=1)
    output = tmp_path / "captions.jsonl"
    generate_spectra_captions(groups, output, generation_config, client=client_factory())
    original = output.read_bytes()
    unused = client_factory()
    with pytest.raises(FileExistsError):
        generate_spectra_captions(groups, output, generation_config, client=unused)
    changed_config = copy.deepcopy(generation_config)
    changed_config["captioning"]["model"] = "different-model"
    with pytest.raises(ValueError, match="generation settings differ"):
        generate_spectra_captions(groups, output, changed_config, resume=True, client=unused)
    observations.at[0, "Z"] = 0.2
    with pytest.raises(ValueError, match="Input changed"):
        generate_spectra_captions(prepare_groups(observations, limit=1), output, generation_config, resume=True, client=unused)
    assert not unused.calls
    assert output.read_bytes() == original


def test_ground_truth_metadata_must_be_explicit(tmp_path, observations, generation_config):
    generation_config["captioning"]["strategy"] = "combined_ground_truth_v1"
    with pytest.raises(ValueError, match="requires --metadata-dir"):
        generate_spectra_captions(prepare_groups(observations), tmp_path / "captions.jsonl", generation_config)
    assert not (tmp_path / "captions.jsonl").exists()


def test_cli_dry_run_uses_local_input_without_api(tmp_path, observations, monkeypatch, capsys):
    from scripts.spectra.caption_spectra import main

    path = tmp_path / "input.parquet"
    output = tmp_path / "captions.jsonl"
    observations.to_parquet(path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["--input", str(path), "--output", str(output), "--limit", "1", "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["object_ids"] == ["gmw_a"]
    assert not output.exists()


def test_generated_captions_reach_training_parquet(
    tmp_path, observations, generation_config, client_factory, monkeypatch,
):
    output = tmp_path / "upstream.jsonl"
    generate_spectra_captions(prepare_groups(observations), output, generation_config, client=client_factory())
    cfg = load_config("base", "data", argv=[])
    del cfg.sources["transients"]
    cfg.sources.spectra_captions.local_path = str(output)
    cfg.manifest.parquet = str(tmp_path / "manifest.parquet")
    cfg.captions.out_dir = str(tmp_path / "training")
    cfg.captions.parquet = str(tmp_path / "training/captions.parquet")
    cfg.captions.report = str(tmp_path / "training/report.json")
    pd.DataFrame([
        {"object_id": row.object_id, "has_spectra": True, "has_image": False}
        for row in observations.itertuples()
    ]).to_parquet(cfg.manifest.parquet)
    spec = importlib.util.spec_from_file_location("generate_training_captions", ROOT / "scripts/01_generate_captions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_config", lambda *args: cfg)
    monkeypatch.setattr(module, "load_spectra_table", lambda *args, **kwargs: observations)
    monkeypatch.setattr(module, "load_image_captions_table", lambda *args, **kwargs: pd.DataFrame(columns=["object_id", "caption_blind"]))
    module.main()
    training = pd.read_parquet(cfg.captions.parquet)
    assert set(training["object_id"]) == {"observation_a", "observation_b"}
    assert all(list(subset) == ["spectra"] for subset in training["subset"])
    assert set(training["text"]) == {"A spectrum with strong emission lines."}
    report = json.loads(Path(cfg.captions.report).read_text())
    assert report["spectra_caption_match_rate"] == 1.0
    assert report["spectra_captions_source"]["local_path"] == str(output)
