# Astrobridge -- Connecting astronomy with multimodal LLMs

## Info

something something fine-tune on Qwen 3.5 9B with AION encoder for spectra and image modals and ATCAT for lightcurve inputs.

--- 

## Pipeline

The data creation and training pipeline requires a huggingface account to agree and access these repositories:

- https://huggingface.co/polymathic-ai/aion-base
- https://huggingface.co/Qwen/Qwen3.5-9B

The pipeline also requires a huggingface token. You can obtain this by following the steps after running:

```
hf auth login
```

Additionally, we have prepared code for you to either generate captions yourself or utilize our existing repositories for each modality.

- https://huggingface.co/datasets/gapatron/legacy_survey_south_images_captions
- link to lcs
- link to spectra

### 0. Setting up dev environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) or any other `pyproject.toml` compatible package manager and run:

```
uv pip install -e ".[dev]"
```

You can confirm the environment works by running the tests:

```
make tests
```

To check access, running this command will inform you of missing access/permissions before running anything else:

```
make check-access
```

### 1. Captioning data
First, we need to obtain the captions for all the datapoints used to train and validate the model. This can be done by either using a Gemini key or by using our existing repository as explained above.

If moving forward with generating independent captions via Gemini, ensure `GEMINI_API_KEY` is set in `.env` and run the following to generate all captions for the spectra data:

```
make spectra-captions ARGS="--limit 0"
```

For images:

```
aaa
```

For light curves:

```
aaa
```

### 2. Set up data

Create the master `manifest.parquet` dataset for all modalities and delegate the train/val split via:

```
make manifest
```

Then, create the master `captions.parquet` dataset for all modalities:

```
make captions
```

### 3. Encode the modalities

Retrieve encoded outputs of the data by inputting into their respective encoders -- spectra + images into AION/lightcurves into ATCAT:

```
make cache
```


### 4. Train the fusion stack

Freeze the model, and train the fusion stack by running:

```
make stage1
```

After this completes, check the fusion stack training by performing 3 tests on each modality -- swapping an observation for another observation, removing the observation, and checking unique-ness of each caption by running:

```
make eval CKPT=outputs/checkpoints/stage1/best
```

### 5. Additionally train the LoRA adapters

The weights are still frozen, and the fusion stack is trained once more but with the addition of the LoRA adapters by running:

```
make stage2
```

After this stage completes, the same check as step 3. can be performed by running:

```
make eval CKPT=outputs/checkpoints/stage2/best
```

### 6. Evaluating the model

something something test set from 3 modalities:

- spectra
- images
- lightcurve

