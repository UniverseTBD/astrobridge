# stage1/stage2 run through `accelerate launch` so the same command works on one GPU or several
# on one node — accelerate no-ops down to plain single-process/single-GPU if unconfigured.
# Override on the command line, e.g.: make stage1 ACCELERATE_CONFIG=configs/my_cluster.yaml
ACCELERATE_CONFIG ?= configs/accelerate_ddp.yaml

.PHONY: install test check-access \
	spectra-captions lightcurve-captions \
	manifest captions cache stage1 eval stage2 publish infer \
	eval-spectra \
	eval-lc-prep eval-lc-generate eval-lc-score eval-lc-heads eval-lc-audit eval-lightcurve \
	eval-image-captions collect-image-labels score-image-eval score-image-eval-debiased

# --- 0. Environment & Access Setup ---

install:
	uv pip install -e ".[dev]"

test:
	pytest -q tests/

check-access:
	python scripts/check_access.py

# --- 1. Caption Generation (Optional: pre-generated captions are downloaded in `make captions`) ---

spectra-captions:
	python scripts/spectra/caption_spectra.py $(ARGS)

lightcurve-captions:
	python scripts/lightcurves/caption_lightcurves.py $(ARGS)

# --- 2. Data Preparation, Training & Inference ---

manifest:
	python scripts/00_build_manifest.py $(ARGS)

captions:
	python scripts/01_generate_captions.py $(if $(SPECTRA_CAPTIONS),"sources.spectra_captions.local_path=$(SPECTRA_CAPTIONS)") $(ARGS)

cache:
	python scripts/02_cache_embeddings.py $(ARGS)

stage1:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/03_train_stage1.py $(ARGS)

eval:
	@test -n "$(CKPT)" || (echo "Usage: make eval CKPT=outputs/checkpoints/stage1/best" && exit 1)
	python scripts/04_eval.py --checkpoint-dir $(CKPT) $(ARGS)

stage2:
	accelerate launch --config_file $(ACCELERATE_CONFIG) scripts/05_train_stage2.py $(ARGS)

publish:
	@test -n "$(CKPT)" || (echo "Usage: make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-captioner-v1" && exit 1)
	@test -n "$(REPO)" || (echo "Usage: make publish CKPT=outputs/checkpoints/stage2/best REPO=your-org/astrobridge-captioner-v1" && exit 1)
	python scripts/06_publish_model.py --checkpoint-dir $(CKPT) --repo-id $(REPO) $(ARGS)

infer:
	@test -n "$(CKPT)" || (echo "Usage: make infer CKPT=outputs/checkpoints/stage2/best QUESTION='What kind of object is this?' [LORA=...] [IMAGE=cutout.npy] [SPECTRUM=spectrum.npz] [SURVEY=desi] [LIGHTCURVE=lc.npz]" && exit 1)
	@test -n "$(QUESTION)" || (echo "Usage: make infer CKPT=outputs/checkpoints/stage2/best QUESTION='What kind of object is this?' [LORA=...] [IMAGE=cutout.npy] [SPECTRUM=spectrum.npz] [SURVEY=desi] [LIGHTCURVE=lc.npz]" && exit 1)
	python scripts/07_infer.py --checkpoint-dir $(CKPT) --question "$(QUESTION)" \
		$(if $(LORA),--lora-dir $(LORA)) \
		$(if $(IMAGE),--image-npy $(IMAGE)) \
		$(if $(SPECTRUM),--spectrum-npz $(SPECTRUM)) \
		$(if $(SURVEY),--spectrum-survey $(SURVEY)) \
		$(if $(LIGHTCURVE),--lightcurve-npz $(LIGHTCURVE))

# --- 3. Paper Evaluations ---

# 3a. Spectra Evaluation Suite (Table 1a)
eval-spectra:
	python eval_scripts/run_paper_captions.py --phase $(or $(PHASE),all) --suite $(or $(SUITE),paper_controls) $(ARGS)

# 3b. Light Curve Benchmark Suite (Table 1b & Appendix D)
eval-lc-prep:
	python scripts/10_prepare_benchmark.py $(ARGS)

eval-lc-generate:
	@test -n "$(ROUTE)" || (echo "Usage: make eval-lc-generate ROUTE=astrobridge_v8_prefix [ARGS=...]" && exit 1)
	python scripts/11_run_generative_route.py --route $(ROUTE) $(ARGS)

eval-lc-score:
	python scripts/14_score_benchmark.py $(ARGS)

eval-lc-heads:
	python scripts/12_extract_representations.py $(ARGS)
	python scripts/13_train_head.py $(ARGS)

eval-lc-audit:
	python scripts/15_audit_caption_claims.py $(ARGS)

eval-lightcurve:
	python -m eval.runners.run_lightcurve_eval \
		$(if $(TRACK),--track $(TRACK)) \
		$(if $(BACKEND),--backend $(BACKEND)) \
		$(if $(LIMIT),--limit $(LIMIT))

# 3c. Image Evaluation Suite (Table 2 & Appendix E)
eval-image-captions:
	python -m eval.runners.collect_image_captions \
		--side $(or $(SIDE),equipped) \
		$(if $(N),--n $(N)) \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(BACKEND),--backend $(BACKEND)) \
		$(ARGS)

collect-image-labels:
	python -m eval.runners.collect_image_labels \
		$(if $(N),--n $(N)) \
		$(if $(SEED),--seed $(SEED)) \
		$(if $(BACKEND),--backend $(BACKEND))

score-image-eval:
	@test -n "$(IN)" || (echo "Usage: make score-image-eval IN=outputs/eval/raw_generations/galaxy10_seed0_n150.json" && exit 1)
	python -m eval.runners.score_image_eval --in $(IN)

score-image-eval-debiased:
	@test -n "$(IN)" || (echo "Usage: make score-image-eval-debiased IN=outputs/eval/raw_generations/galaxy10_seed0_n150.json" && exit 1)
	python -m eval.runners.score_image_eval_debiased --in $(IN)

