# stage1/stage2 run through `accelerate launch` so the same command works on one GPU or several
# on one node — accelerate no-ops down to plain single-process/single-GPU if unconfigured.
# Override on the command line, e.g.: make stage1 ACCELERATE_CONFIG=configs/my_cluster.yaml
ACCELERATE_CONFIG ?= configs/accelerate_ddp.yaml

.PHONY: test manifest captions cache stage1 eval stage2 install check-access publish infer eval-lightcurve collect-image-labels score-image-eval score-image-eval-debiased

install:
	uv pip install -e ".[dev]"

test:
	pytest -q tests/

check-access:
	python scripts/check_access.py

manifest:
	python scripts/00_build_manifest.py

captions:
	python scripts/01_generate_captions.py

cache:
	python scripts/02_cache_embeddings.py

stage1:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/03_train_stage1.py

eval:
	@test -n "$(CKPT)" || (echo "Usage: make eval CKPT=outputs/checkpoints/stage1/best" && exit 1)
	python scripts/04_eval.py --checkpoint-dir $(CKPT)

stage2:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/05_train_stage2.py

publish:
	@test -n "$(CKPT)" || (echo "Usage: make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-captioner-v1" && exit 1)
	@test -n "$(REPO)" || (echo "Usage: make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-captioner-v1" && exit 1)
	python scripts/06_publish_model.py --checkpoint-dir $(CKPT) --repo-id $(REPO)

infer:
	@test -n "$(CKPT)" || (echo "Usage: make infer CKPT=outputs/checkpoints/stage2/best QUESTION='What kind of object is this?' [LORA=...] [IMAGE=cutout.npy] [SPECTRUM=spectrum.npz] [SURVEY=desi] [LIGHTCURVE=lc.npz]" && exit 1)
	@test -n "$(QUESTION)" || (echo "Usage: make infer CKPT=outputs/checkpoints/stage2/best QUESTION='What kind of object is this?' [LORA=...] [IMAGE=cutout.npy] [SPECTRUM=spectrum.npz] [SURVEY=desi] [LIGHTCURVE=lc.npz]" && exit 1)
	python scripts/07_infer.py --checkpoint-dir $(CKPT) --question "$(QUESTION)" \
		$(if $(LORA),--lora-dir $(LORA)) \
		$(if $(IMAGE),--image-npy $(IMAGE)) \
		$(if $(SPECTRUM),--spectrum-npz $(SPECTRUM)) \
		$(if $(SURVEY),--spectrum-survey $(SURVEY)) \
		$(if $(LIGHTCURVE),--lightcurve-npz $(LIGHTCURVE))

# See eval/README.md — TRACK/BACKEND/LIMIT are optional, defaulting per eval/runners/*.py's own
# argparse defaults (lightcurve_only / base_only, local, no limit).
eval-lightcurve:
	uv run python -m eval.runners.run_lightcurve_eval \
		$(if $(TRACK),--track $(TRACK)) \
		$(if $(BACKEND),--backend $(BACKEND)) \
		$(if $(LIMIT),--limit $(LIMIT))

# Two steps, deliberately not one target — see eval/README.md. Step 1 is the expensive/billed
# one (N/SEED/BACKEND optional, default to collect_image_labels.py's own argparse defaults);
# step 2 needs the exact output path step 1 printed (IN=outputs/eval/raw_generations/...).
collect-image-labels:
	uv run python -m eval.runners.collect_image_labels \
		$(if $(N),--n $(N)) \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(BACKEND),--backend $(BACKEND))

score-image-eval:
	@test -n "$(IN)" || (echo "Usage: make score-image-eval IN=outputs/eval/raw_generations/galaxy10_seed0_n150.json" && exit 1)
	uv run python -m eval.runners.score_image_eval --in $(IN)

# Separate, parked script — needs a real network crossmatch, meaningfully slower than
# score-image-eval above, so it's not bundled into that target automatically.
score-image-eval-debiased:
	@test -n "$(IN)" || (echo "Usage: make score-image-eval-debiased IN=outputs/eval/raw_generations/galaxy10_seed0_n150.json" && exit 1)
	uv run python -m eval.runners.score_image_eval_debiased --in $(IN)
