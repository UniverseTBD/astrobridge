# AstroBridge — Connecting Astronomy with Multimodal LLMs

> **Official code repository for the paper:** **The Great Chat in the Sky: Aligning Exotic Astronomical Data to Language with AstroBridge**

**AstroBridge** is a multimodal large language model for astrophysics capable of reading and reasoning over **astronomical images**, **1D optical spectra**, and **multi-band time-series light curves**.

It bridges frozen domain-specific foundation encoders into a frozen/LoRA-adapted **`Qwen/Qwen3.5-9B`** language decoder through a shared 64-query **Q-Former** fusion stack:
- **Images & 1D Spectra**: Encoded via [**AION**](https://huggingface.co/polymathic-ai/aion-base) (`polymathic-ai/aion-base`, 768-dim token sequences).
- **Light Curves**: Encoded via [**ATCAT**](https://huggingface.co/light-curve/atcat) (`light-curve/atcat`, 384-dim temporal sequences).
- **Language Decoder**: [**Qwen 3.5 9B**](https://huggingface.co/Qwen/Qwen3.5-9B) (`Qwen/Qwen3.5-9B`) with per-modality learned query projections, a 4-layer cross-attention Q-Former, and Stage-2 LoRA adapters ($r=16, \alpha=32$).

---

## Published Artifacts on Hugging Face

### Trained Model
- **AstroBridge (Paper Release)**: [`UniverseTBD/AstroBridge`](https://huggingface.co/UniverseTBD/AstroBridge)

### Upstream Encoders & Base LLM
- **AION Base**: [`polymathic-ai/aion-base`](https://huggingface.co/polymathic-ai/aion-base) *(open access, not gated)*
- **ATCAT Light Curve Encoder**: [`light-curve/atcat`](https://huggingface.co/light-curve/atcat) *(open access, not gated)*
- **Base LLM**: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) *(open access, not gated)*

### Training & Evaluation Datasets
All AstroBridge training observations, pre-generated captions, held-out spectra/light-curve benchmarks, and reference evaluation outputs are consolidated in [`UniverseTBD/AstroBridge-Data`](https://huggingface.co/datasets/UniverseTBD/AstroBridge-Data):
- **Spectra Observations & Gemini Captions**: `observations/spectra/spectra.parquet` & `captions/spectra/spectra_captions.jsonl`
- **Galaxy Image Observations & Fused Captions**: `observations/images/legacy_south_all_images.parquet` & `captions/images/part-*.parquet`
- **ZTF BTS Supernova Light Curves & Captions**: `observations/lightcurves/part-*.parquet`
- **Held-out Spectra Benchmark, Model Captions & Frontier Judge Results**: `evals/spectra/{benchmarks,captions,results}/`
- **Held-out YSE Light Curve Benchmark**: `evals/lightcurves/yse/part-*.parquet`
- **External Galaxy10 & Galaxy Zoo DECaLS Image Benchmarks**: [`astronolan/galaxy10-aion`](https://huggingface.co/datasets/astronolan/galaxy10-aion) & [`astronolan/gz-decals-embeddings`](https://huggingface.co/datasets/astronolan/gz-decals-embeddings)

---

## 0. Setup & Prerequisites

### Environment Installation
Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (or any `pyproject.toml`-compatible package manager) and install the project in editable mode:

```bash
make install
# Equivalent to: uv pip install -e ".[dev]"
```

### Credentials & Access Verification
1. **Hugging Face Token (`HF_TOKEN`)**: All upstream model weights (`polymathic-ai/aion-base`, `light-curve/atcat`, `Qwen/Qwen3.5-9B`) are open and not gated. Setting `HF_TOKEN` (or logging in via CLI) is recommended for higher Hugging Face Hub download rate limits, accessing any gated dataset repos, and publishing trained checkpoints (`make publish`):
   ```bash
   hf auth login
   # or: export HF_TOKEN=hf_...
   ```
2. **Gemini API Key (`GEMINI_API_KEY`)**: Only required if you are generating new training captions from scratch or running LLM-as-a-judge evaluation steps. Add it to a local `.env` file or export it in your shell:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

Verify that your environment can reach all upstream models and datasets before launching long jobs:

```bash
make check-access
```

---

## 1. Caption Generation

> **Note**: Pre-generated training captions for all three modalities are already hosted on Hugging Face and are automatically pulled during `make captions` (Step 2). You only need to run the commands below if you want to regenerate captions from scratch using the Gemini API.

### Spectra Captions
Generates structured descriptions from 1D spectra plots and metadata via Gemini (`src/spectra_captioning/`). By default, [`configs/spectra_captioning.yaml`](configs/spectra_captioning.yaml) sets `limit: 10` as a safety cap; passing `--limit 0` disables that cap and processes all objects in the dataset:

```bash
# Dry-run on 5 objects (no API calls):
make spectra-captions ARGS="--limit 5 --dry-run"

# Full generation across all objects:
make spectra-captions ARGS="--limit 0"
```
Outputs JSONL files to `outputs/captions/spectra/`.

### Light Curve Captions
Generates descriptions for ZTF Bright Transient Survey (BTS) light curves via Gemini (`scripts/lightcurves/caption_lightcurves.py`):

```bash
# Dry-run preflight check:
make lightcurve-captions ARGS="--dry-run --limit 5"
```
Outputs JSONL ledgers and consolidated parquet files to `outputs/lightcurve_captions/`.

### Image Captions
Pre-generated fused image captions (`caption_fused`) are hosted in [`UniverseTBD/AstroBridge-Data`](https://huggingface.co/datasets/UniverseTBD/AstroBridge-Data) under `captions/images/part-*.parquet`.

---

## 2. Data Pipeline, Training & Inference

All configuration files live in [`configs/`](configs/) (`base.yaml`, `data.yaml`, `modalities.yaml`, `model.yaml`, `stage1.yaml`, `stage2.yaml`) and can be overridden via CLI `key=value` arguments through `ARGS="..."`.

### Step 2.1 — Build Master Manifest
Crossmatches objects across modalities using HEALPix ($\text{NSIDE}=2^{29}$, $1.0''$ fallback radius) and assigns stratified `train`/`val`/`test` splits:

```bash
make manifest
```
*Output*: `outputs/manifest/manifest.parquet` and `outputs/manifest/manifest_stats.json`.

### Step 2.2 — Assemble Training Captions
Downloads and aligns captions across all three modalities against the master manifest:

```bash
make captions
# Or override with a locally generated spectra captions file:
# make captions SPECTRA_CAPTIONS=outputs/captions/spectra/captions.jsonl
```
*Output*: `outputs/captions/captions.parquet` and `outputs/captions/decomposition_report.json`.

### Step 2.3 — Pre-cache Encoder Embeddings
Runs the frozen domain encoders (**AION** for images and spectra, **ATCAT** for light curves) once over the dataset and writes sharded WebDataset `.tar` archives to disk so training does not recompute encoder forward passes:

```bash
make cache
```
*Output*: `outputs/cache/{image,spectra,lightcurve}/<spec_hash>/{train,val,test}/shard-*.tar`.

### Step 2.4 — Stage 1 Training (Fusion Stack Only)
Freezes the Qwen 3.5 9B LLM and trains only the modality projections and 64-query Q-Former fusion stack:

```bash
make stage1
```
*Output*: `outputs/checkpoints/stage1/best/` and `outputs/checkpoints/stage1/last/`.

### Step 2.5 — Stage 2 Training (Fusion Stack + LoRA Adapters)
Initializes from `stage1/best`, attaches LoRA adapters ($r=16, \alpha=32$) to the Qwen attention projections, and jointly trains the fusion stack and LoRA weights:

```bash
make stage2
```
*Output*: `outputs/checkpoints/stage2/best/` (contains `fusion_stack.pt`, `lora_adapter/`, and `train_state.pt`).

### Step 2.6 — Groundedness & Sensitivity Evaluation
Verifies that the trained model genuinely conditions on the astronomical inputs via three checks per modality (observation swap sensitivity, modality ablation degradation, and caption diversity):

```bash
make eval CKPT=outputs/checkpoints/stage1/best
make eval CKPT=outputs/checkpoints/stage2/best
```
*Output*: `outputs/eval/groundedness.json`.

### Step 2.7 — Publish & Run Single-Object Inference
Upload a trained checkpoint to Hugging Face Hub:

```bash
make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-model
```

Run interactive inference on custom `.npy`/`.npz` observations:

```bash
make infer \
  CKPT=outputs/checkpoints/stage2/best \
  LORA=outputs/checkpoints/stage2/best/lora_adapter \
  SPECTRUM=path/to/spectrum.npz \
  SURVEY=desi \
  QUESTION="Describe the key spectral features and object type."
```

---

## 3. Replicating Paper Evaluations

### 3.1 Spectra Evaluation Suite (Table 1a)
Evaluates AstroBridge against vision and text baseline LLMs on the **held-out spectra test benchmark** hosted in [`UniverseTBD/AstroBridge-Data`](https://huggingface.co/datasets/UniverseTBD/AstroBridge-Data) under `evals/spectra/benchmarks/` (`all.parquet`, `source.parquet`, `subclass.parquet`, `distance.parquet`, and `emission_lines.parquet`, automatically downloaded to `data/benchmarks/` on first run) across 4 astrophysical tasks (`source`, `subclass`, `distance`, `emission_lines`) configured in [`eval_configs/suites/paper_controls.yaml`](eval_configs/suites/paper_controls.yaml):
- **Phase 1 (`--phase 1`)**: Generates or pulls cached model captions (`evals/spectra/captions/*.jsonl` -> `eval_results/cached_captions/*.jsonl`) for the held-out spectra test objects.
- **Phase 2 (`--phase 2`)**: Uses a frontier judge (`gemini-4-argon` via `GEMINI_API_KEY`) to extract structured predictions from captions and computes accuracy, emission-line Jaccard index, and 10,000-iteration paired bootstrap 95% confidence intervals (reference results hosted in `evals/spectra/results/`).

```bash
# Run both Phase 1 (caption generation/pull) and Phase 2 (frontier judge + metrics):
make eval-spectra

# Run only Phase 2 scoring over existing/downloaded captions:
make eval-spectra PHASE=2

# Run for a single model or task:
make eval-spectra PHASE=2 ARGS="--model astrobridge --task source"
```
*Output*: `eval_results/paper_evals/paper_controls/` (per-model JSONLs, `summary_metrics.json`, `summary_table.csv`, and `comparison_table.csv`).

---

### 3.2 Light Curve Evaluation Suite (Table 1b & Appendix D)
Evaluates supernova classification (`SN Ia`, `SN II`, `SN Ibc`) on the held-out 266-object **Young Supernova Experiment (YSE)** benchmark (`evals/lightcurves/yse/` in [`UniverseTBD/AstroBridge-Data`](https://huggingface.co/datasets/UniverseTBD/AstroBridge-Data)).

1. **Prepare immutable YSE benchmark artifacts** (manifests, flux tables, rendered plots, and checksums):
   ```bash
   make eval-lc-prep
   ```
   *Output*: `outputs/benchmark_v8/manifest.json`, `charts/`, and `tables/`.

2. **Run generative classification routes** (Table 1b):
   ```bash
   # AstroBridge (free-form caption prefix -> classification):
   make eval-lc-generate ROUTE=astrobridge_v8_prefix

   # Control baselines (rendered chart or numerical table):
   make eval-lc-generate ROUTE=qwen35_chart
   make eval-lc-generate ROUTE=qwen35_table
   make eval-lc-generate ROUTE=gemma4_12b_chart
   make eval-lc-generate ROUTE=gemma4_12b_table
   make eval-lc-generate ROUTE=gemma4_31b_chart
   make eval-lc-generate ROUTE=gemma4_31b_table
   ```
   *Output*: `outputs/benchmark_v8/generations/<route>.jsonl`.

3. **Run frozen representation probes** (Appendix D — linear & MLP heads over ATCAT and Q-Former embeddings):
   ```bash
   make eval-lc-heads
   ```
   *Output*: `outputs/benchmark_v8/representations/` and `outputs/benchmark_v8/heads/`.

4. **Score all light curve routes & compute bootstrap CIs** (accuracy, macro F1, balanced accuracy):
   ```bash
   make eval-lc-score
   ```
   *Output*: `outputs/benchmark_v8/scores/summary.json`, `summary.csv`, and `bootstrap.json`.

5. **Audit physical light curve caption claims** (Appendix D claim precision audit):
   ```bash
   make eval-lc-audit ARGS="--route astrobridge_v8_prefix --scope manual20"
   ```
   *Output*: `outputs/benchmark_v8/claims_audit/`.

---

### 3.3 Image Evaluation Suite (Table 2 & Appendix E)
Evaluates galaxy morphology descriptions and classification on the held-out **Galaxy10 DECaLS** test partition (`astronolan/galaxy10-aion`):

1. **Generate paired test-partition captions** for AstroBridge (`equipped`) and unmodified Qwen 3.5 9B (`base`) under an identical token budget (Table 2 input):
   ```bash
   make eval-image-captions SIDE=equipped N=200 SEED=0
   make eval-image-captions SIDE=base N=200 SEED=0
   ```
   *Output*: `outputs/eval/raw_generations/gz10_images.json` and `outputs/eval/raw_generations/gz10_images_base.json`.

2. **Collect and score morphology classification labels** (Appendix E — hard accuracy, coarse morphology group score, and Galaxy Zoo DECaLS crowd-vote-fraction soft score):
   ```bash
   make collect-image-labels N=150 SEED=0
   make score-image-eval IN=outputs/eval/raw_generations/galaxy10_logprob_argmax_seed0_n150.json
   make score-image-eval-debiased IN=outputs/eval/raw_generations/galaxy10_logprob_argmax_seed0_n150.json
   ```
   See [`eval/README.md`](eval/README.md) for detailed documentation on the morphology classification metrics.

---

## Repository Structure

```text
astrobridge/
├── configs/                  # Training, data, modality, and model YAML configs
├── eval/                     # Light curve (v8 benchmark) & image evaluation package
│   ├── datasets/             # Galaxy10 DECaLS, GZ vote fractions, and YSE loaders
│   ├── metrics/              # Classification, group, soft vote-fraction, and LLM judge scorers
│   ├── runners/              # Image and light curve evaluation runners
│   └── benchmark_*.py        # Reproducible YSE v8 benchmark protocol, heads, and claim audit
├── eval_configs/             # Spectra evaluation configs (suites, tasks, responders, frontier judge)
├── eval_scripts/             # Spectra evaluation runners, metric calculators, and bootstrap CIs
├── scripts/                  # Numbered pipeline entrypoints
│   ├── 00_build_manifest.py  # Crossmatch & train/val/test split generation
│   ├── 01_generate_captions.py # Assemble multimodal captions parquet
│   ├── 02_cache_embeddings.py  # Pre-encode observations into WebDataset tar shards
│   ├── 03_train_stage1.py    # Stage 1 Q-Former training
│   ├── 04_eval.py            # Modality swap/ablation groundedness check
│   ├── 05_train_stage2.py    # Stage 2 Q-Former + LoRA training
│   ├── 06_publish_model.py   # Upload checkpoint to Hugging Face Hub
│   ├── 07_infer.py           # CLI inference on single objects
│   ├── 10..15_*.py           # YSE v8 light curve benchmark suite
│   ├── lightcurves/          # Gemini caption generator for ZTF light curves
│   └── spectra/              # Gemini caption generator CLI for 1D spectra
├── src/
│   ├── captioner/            # Core AstroBridge model, encoders (AION/ATCAT), data & training loop
│   ├── evals/                # Spectra evaluation framework & frontier judge integration
│   └── spectra_captioning/   # Vendored 1D spectra plotting & Gemini captioning strategies
└── tests/                    # Fast CPU unit & integration test suite
```
