"""THE single editable module for "how do I call the model" — swap local GPU vs. a rented Modal
GPU by changing `kind="local"` to `kind="modal"` in a runner's `get_backend(...)` call, nothing
else. Every `eval/runners/*.py` script and `eval/metrics/*.py` module is written against
`EvalBackend`'s plain `.generate(...)` interface and never imports `modal`, `transformers`, or
`captioner.inference` directly — this file is the only place those live.

**Two sides, loaded one at a time, never both simultaneously**: "equipped" (our AION encoders +
fusion stack + LoRA-adapted Qwen — `captioner.inference.load_inference_model_from_hub` +
`generate_caption`) and "base" (plain, genuinely out-of-the-box Qwen, no LoRA, native vision
pathway — `captioner.inference.load_qwen_native_vision_model` + `generate_qwen_native_vision_
answer`). A runner that needs both calls `get_backend(..., side="equipped")`, finishes the whole
dataset, lets that backend/model get garbage-collected, then calls `get_backend(...,
side="base")` — the same VRAM-bounding sequencing `inference/compare.py` used on the (now
deleted) `modal-inference` branch, just moved up into the runner's control flow instead of being
buried inside one Modal function body. Not every runner needs both sides: the lightcurve track
only ever uses "equipped" (a raw lightcurve array isn't something a plain out-of-the-box
vision-language model can consume at all — there is no meaningful "base" comparison for it), the
image track uses both.

**Why `.generate()` takes a `raw_inputs` dict, not per-modality kwargs**: matches `generate_
caption`'s existing contract exactly on the equipped side (`{"image": {"pixel_values": ...}}`
etc. — see `captioner.inference.generate_caption`'s docstring for the full per-modality shape).
The base side's contract is narrower and different in kind — a plain vision-language model only
ever consumes one rendered picture, never raw AION-shaped tensors — so a base-backend's
`raw_inputs` is always exactly `{"image": <PIL.Image>}`. This asymmetry is real, not
overlooked: a base backend raises `KeyError` loudly if asked for anything else, rather than
pretending to support modalities it structurally cannot.

**Why `.generate()` returns a string (caption text), never a label**: keeps this file about model
invocation only. Turning that caption into a predicted class belongs to
`eval/metrics/caption_to_label.py` — this is also what lets a future caption-quality eval reuse
this file completely unchanged.
"""
import gc
from dataclasses import dataclass
from typing import Callable, Literal

from omegaconf import DictConfig

Side = Literal["equipped", "base"]


@dataclass
class EvalBackend:
    """Thin wrapper around one already-loaded model (or one already-deployed Modal function) for
    exactly one side. Callers never construct this directly — use `get_backend(...)`.
    """

    side: Side
    _generate: Callable[[dict, str, int, str | None], str]
    _generate_batch: Callable[[list, str, int, str | None], list[str]] | None = None
    _classify: Callable[[dict, str, list[str], int], dict] | None = None

    def generate_batch(
        self, raw_inputs_list: list, question: str, max_new_tokens: int = 128, system: str | None = None,
    ) -> list[str]:
        """`generate(...)` over B objects in one forward pass, answers in input order. Falls back
        to looping `generate(...)` when a backend has no batched path wired, so callers can always
        use it. See `captioner.inference.generate_captions_batched` for why batching wins here."""
        if self._generate_batch is None:
            return [self._generate(r, question, max_new_tokens, system) for r in raw_inputs_list]
        return self._generate_batch(raw_inputs_list, question, max_new_tokens, system)

    def generate(self, raw_inputs: dict, question: str, max_new_tokens: int = 128, system: str | None = None) -> str:
        """`system`: only meaningful for `side="equipped"` (overrides `configs/model.yaml`'s
        `prompt.system_variants[0]` default — see `captioner.inference.generate_caption`'s
        `system` param). Silently ignored on `side="base"`, which has no equivalent concept —
        Qwen's own native chat template doesn't expose a system-prompt override through
        `generate_qwen_native_vision_answer`.
        """
        return self._generate(raw_inputs, question, max_new_tokens, system)

    def classify(
        self, raw_inputs: dict, context: str, candidates: list[str], max_reasoning_tokens: int = 150,
    ) -> dict:
        """Deterministic classification: generates free-form reasoning against `context`, then
        scores each of `candidates` as its teacher-forced continuation and returns
        `{"reasoning": str, "logprobs": {candidate: float}}` — argmax over `logprobs` is always
        one of `candidates` by construction, no parsing involved. See
        `captioner.inference.score_completions`/`score_completions_qwen_native`'s docstrings for
        the mechanism on each side.
        """
        if self._classify is None:
            raise NotImplementedError(f"classify() is not wired for side={self.side!r} backend.")
        return self._classify(raw_inputs, context, candidates, max_reasoning_tokens)


def get_backend(
    kind: Literal["local", "modal"],
    side: Side,
    cfg: DictConfig,
    *,
    repo_id: str | None = None,
    device: str = "cuda",
    modality_names: list[str] | None = None,
    modal_app_name: str = "astrobridge-eval-backend",
    enable_thinking: bool | None = None,
) -> EvalBackend:
    """The one function every runner calls. `repo_id`/`modality_names` only matter for
    `side="equipped"`; ignored (but harmless to pass) for `side="base"`. `enable_thinking` is the
    reverse — only matters for `side="base"` — see `generate_qwen_native_vision_answer`'s
    docstring for what it actually does and why it defaults to `None` (the model's own default,
    thinking ON) rather than being force-disabled here.
    """
    if kind == "local":
        return _local_backend(
            side, cfg, repo_id=repo_id, device=device, modality_names=modality_names, enable_thinking=enable_thinking,
        )
    if kind == "modal":
        return _modal_backend(
            side, modal_app_name=modal_app_name, repo_id=repo_id, modality_names=modality_names,
            enable_thinking=enable_thinking,
        )
    raise ValueError(f"kind={kind!r} not recognised — expected 'local' or 'modal'.")


# --- local backend ------------------------------------------------------------------------


def _local_backend(
    side: Side,
    cfg: DictConfig,
    *,
    repo_id: str | None,
    device: str,
    modality_names: list[str] | None,
    enable_thinking: bool | None = None,
) -> EvalBackend:
    from captioner.inference import (
        generate_caption,
        generate_captions_batched,
        generate_qwen_native_vision_answers_batched,
        generate_qwen_native_vision_answer,
        load_inference_model_from_hub,
        load_qwen_native_vision_model,
        score_completions,
        score_completions_qwen_native,
    )

    if side == "equipped":
        if repo_id is None:
            raise ValueError("repo_id is required for side='equipped' (the published model to load).")
        model, tokenizer, encoders = load_inference_model_from_hub(
            cfg, repo_id, device=device, modality_names=modality_names,
        )
        out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
        max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

        def _generate(raw_inputs: dict, question: str, max_new_tokens: int, system: str | None = None) -> str:
            return generate_caption(
                model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt, device,
                raw_inputs, max_new_tokens=max_new_tokens, question=question, system=system,
            )

        def _classify(raw_inputs: dict, context: str, candidates: list[str], max_reasoning_tokens: int) -> dict:
            reasoning = generate_caption(
                model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt, device,
                raw_inputs, max_new_tokens=max_reasoning_tokens, question=context,
            )
            logprobs = score_completions(
                model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt, device,
                raw_inputs, candidates, question=context, reasoning=reasoning,
            )
            return {"reasoning": reasoning, "logprobs": logprobs}

        def _generate_batch(raw_inputs_list: list, question: str, max_new_tokens: int, system: str | None = None) -> list[str]:
            return generate_captions_batched(
                model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt, device,
                raw_inputs_list, max_new_tokens=max_new_tokens, question=question, system=system,
            )

        return EvalBackend(side="equipped", _generate=_generate, _classify=_classify, _generate_batch=_generate_batch)

    # side == "base"
    vision_model, processor = load_qwen_native_vision_model(cfg, device=device)

    def _generate(raw_inputs: dict, question: str, max_new_tokens: int, system: str | None = None) -> str:
        if "image" not in raw_inputs or len(raw_inputs) != 1:
            raise KeyError(
                f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys "
                f"{list(raw_inputs)}. There is no native-vision equivalent for other modalities."
            )
        return generate_qwen_native_vision_answer(
            vision_model, processor, device, question, raw_inputs["image"], max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

    def _classify(raw_inputs: dict, context: str, candidates: list[str], max_reasoning_tokens: int) -> dict:
        if "image" not in raw_inputs or len(raw_inputs) != 1:
            raise KeyError(
                f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys {list(raw_inputs)}."
            )
        image = raw_inputs["image"]
        reasoning = generate_qwen_native_vision_answer(
            vision_model, processor, device, context, image, max_new_tokens=max_reasoning_tokens,
            enable_thinking=False,
        )
        logprobs = score_completions_qwen_native(
            vision_model, processor, device, context, image, candidates, reasoning=reasoning,
        )
        return {"reasoning": reasoning, "logprobs": logprobs}

    def _generate_batch(raw_inputs_list: list, question: str, max_new_tokens: int, system: str | None = None) -> list[str]:
        for raw in raw_inputs_list:
            if "image" not in raw or len(raw) != 1:
                raise KeyError(f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys {list(raw)}.")
        return generate_qwen_native_vision_answers_batched(
            vision_model, processor, device, question, [r["image"] for r in raw_inputs_list],
            max_new_tokens=max_new_tokens, enable_thinking=enable_thinking,
        )

    return EvalBackend(side="base", _generate=_generate, _classify=_classify, _generate_batch=_generate_batch)


def free_local_backend(backend: EvalBackend) -> None:
    """Call between `get_backend(..., side="equipped")` and `get_backend(..., side="base")` (or
    vice versa) in the SAME process, so the first model's VRAM is actually released before the
    second one loads — otherwise both stay resident simultaneously, doubling peak VRAM for no
    reason (each side is only ever used for the whole dataset in turn, never interleaved).
    `backend` itself is not reusable after this call.
    """
    import torch

    del backend._generate
    backend._classify = None
    backend._generate_batch = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- modal backend -------------------------------------------------------------------------
#
# Reuses the exact pattern proven on the (now deleted) `modal-inference` branch's
# `inference/modal_app.py`/`compare.py`: `uv_sync()` (installs from this repo's real
# pyproject.toml/uv.lock — a hand-picked pip_install list once silently missed `pandas`, a
# transitive import) + `add_local_python_source("captioner")` + `add_local_dir("configs",
# remote_path="/configs")` (required because `add_local_python_source` mounts the package at
# `/root/captioner/...`, skipping the local `src/` layer, and `utils/config.py`'s `CONFIG_DIR`
# path-climbing lands on exactly `/configs` from that remote layout).
#
# `modal deploy`/`modal run` look for a variable literally named `app` by default (confirmed
# live: naming it `modal_app` fails deploy with "module 'backend' has no attribute 'app'") — so
# the actual `modal.App` instance below is named `app`, not `modal_app`, despite this section's
# comments still saying "modal_app" in prose for clarity against `inference/modal_app.py`'s name.

import modal  # noqa: E402  (kept below the docstring/local-backend code on purpose — this is the
                # one place Modal-specific imports belong; nothing above this line touches modal)

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_python_source("captioner", "eval")
    .add_local_dir("configs", remote_path="/configs")
)

app = modal.App("astrobridge-eval-backend")

hf_cache_volume = modal.Volume.from_name("astrobridge-hf-cache", create_if_missing=True)

# Real, confirmed-live published repo. Bumped to v5 (image + spectra + lightcurve, 64-query
# Q-Former, chat-template prompting) from the earlier v3_qwen default.
DEFAULT_MODEL_REPO_ID = "UniverseTBD/astrobridge-model-v7"

# A100, not L4 — bumped per explicit request for faster real eval runs.
# Pinned to the 40GB A100 explicitly: bare "A100" is Modal's 40GB variant anyway, but naming it
# means a future Modal default change can't silently move us to the pricier 80GB card. 40GB is
# ample — ~18GB of Qwen weights plus tens of MB per batched sequence.
_MODAL_GPU_KW = {"gpu": "A100-40GB"}


_MODAL_CLS_KW = dict(
    cpu=2.0,
    memory=16384,
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=1800,
    scaledown_window=300,  # keep a container warm for 5 min of idle time between calls in a run
    **_MODAL_GPU_KW,
)


@app.cls(**_MODAL_CLS_KW)
class _EquippedModel:
    """Warm-container equivalent of the old `_equipped_infer`/`_equipped_classify` functions: the
    model loads once in `@modal.enter()` when the container starts, then every `.infer.remote(...)`
    /`.classify.remote(...)` call against the SAME `(repo_id, modality_names)` pair reuses that
    already-loaded model — the model is NOT reloaded per object. Confirmed real, not a style
    preference: a live run collecting 45 objects timed out after 28+ minutes still on the base
    side under the old per-call-reload design, because each of the ~90 total calls (45 objects x 2
    sides) reloaded the full model from scratch.

    `repo_id`/`modality_names_json` are `modal.parameter()`s, not plain constructor args — Modal
    keys warm containers by their exact parameter values, so calling with a different repo_id/
    modality set gets its own container rather than silently reusing one loaded for a different
    model. `modality_names` is JSON-encoded because `modal.parameter()` only supports simple
    scalar types, not `list[str] | None`.
    """

    repo_id: str = modal.parameter(default=DEFAULT_MODEL_REPO_ID)
    modality_names_json: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        import json

        import torch

        from captioner.inference import load_inference_model_from_hub
        from captioner.utils.config import load_config

        self.cfg = load_config("base", "data", "modalities", "model", "stage2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        modality_names = json.loads(self.modality_names_json) if self.modality_names_json else None

        self.model, self.tokenizer, self.encoders = load_inference_model_from_hub(
            self.cfg, self.repo_id, device=self.device, modality_names=modality_names,
        )
        hf_cache_volume.commit()
        self.out_dims = {n: int(c.out_dim) for n, c in self.cfg.modalities.items()}
        self.max_tokens = {n: int(c.max_tokens) for n, c in self.cfg.modalities.items()}

    @modal.method()
    def infer(self, raw_inputs_serialized: dict, question: str, max_new_tokens: int = 128, system: str | None = None) -> str:
        from captioner.inference import generate_caption

        return generate_caption(
            self.model, self.tokenizer, self.encoders, self.out_dims, self.max_tokens, self.cfg.prompt,
            self.device, raw_inputs_serialized, max_new_tokens=max_new_tokens, question=question, system=system,
        )

    @modal.method()
    def infer_batch(
        self, raw_inputs_list: list, question: str, max_new_tokens: int = 128, system: str | None = None,
    ) -> list[str]:
        """B objects per remote call instead of one — see
        `captioner.inference.generate_captions_batched` for why this is the large win, and it also
        removes B-1 Modal round trips."""
        from captioner.inference import generate_captions_batched

        return generate_captions_batched(
            self.model, self.tokenizer, self.encoders, self.out_dims, self.max_tokens, self.cfg.prompt,
            self.device, raw_inputs_list, max_new_tokens=max_new_tokens, question=question, system=system,
        )

    @modal.method()
    def fusion_prefix(self, raw_inputs_list: list) -> list:
        """Diagnostic, not part of the caption/classify surface: returns the raw fusion-stack
        prefix embeddings (n_queries x d_llm, flattened) for each object in `raw_inputs_list`, no
        LLM decode involved. Exists to sanity-check the encoder in isolation — if prefix vectors
        for genuinely different light curves (or a real object vs pure random-noise input) come
        out near-identical, the bug is upstream of the LLM/training-target truncation entirely
        (a dead/collapsed encoder), which the truncation-bug finding alone cannot rule out since
        it only ever observed behavior through a decoder trained on truncated targets.
        """
        import torch

        from captioner.inference import _build_modality_batch

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        out = []
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            for raw in raw_inputs_list:
                mb = _build_modality_batch(self.encoders, self.out_dims, self.max_tokens, raw, self.device)
                prefix = self.model.fusion_stack(mb)  # (1, n_queries, d_llm)
                out.append(prefix.float().reshape(-1).cpu().tolist())
        return out

    @modal.method()
    def classify(
        self, raw_inputs_serialized: dict, context: str, candidates: list[str], max_reasoning_tokens: int = 150,
    ) -> dict:
        from captioner.inference import generate_caption, score_completions

        reasoning = generate_caption(
            self.model, self.tokenizer, self.encoders, self.out_dims, self.max_tokens, self.cfg.prompt,
            self.device, raw_inputs_serialized, max_new_tokens=max_reasoning_tokens, question=context,
        )
        logprobs = score_completions(
            self.model, self.tokenizer, self.encoders, self.out_dims, self.max_tokens, self.cfg.prompt,
            self.device, raw_inputs_serialized, candidates, question=context, reasoning=reasoning,
        )
        return {"reasoning": reasoning, "logprobs": logprobs}


@app.cls(**_MODAL_CLS_KW)
class _BaseModel:
    """Warm-container equivalent of the old `_base_infer`/`_base_classify` functions — see
    `_EquippedModel`'s docstring for why this matters. No parameters: this side is always the same
    plain, never-LoRA'd Qwen3.5-9B regardless of which equipped repo is under test.
    """

    @modal.enter()
    def load(self):
        import torch

        from captioner.inference import load_qwen_native_vision_model
        from captioner.utils.config import load_config

        self.cfg = load_config("base", "data", "modalities", "model", "stage2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_model, self.processor = load_qwen_native_vision_model(self.cfg, device=self.device)
        hf_cache_volume.commit()

    @modal.method()
    def infer(self, image_bytes: bytes, question: str, max_new_tokens: int = 128, enable_thinking: bool | None = None) -> str:
        import io

        from PIL import Image

        from captioner.inference import generate_qwen_native_vision_answer

        image = Image.open(io.BytesIO(image_bytes))
        return generate_qwen_native_vision_answer(
            self.vision_model, self.processor, self.device, question, image, max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

    @modal.method()
    def infer_batch(
        self, image_bytes_list: list, question: str, max_new_tokens: int = 128, enable_thinking: bool | None = None,
    ) -> list[str]:
        import io

        from PIL import Image

        from captioner.inference import generate_qwen_native_vision_answers_batched

        images = [Image.open(io.BytesIO(b)) for b in image_bytes_list]
        return generate_qwen_native_vision_answers_batched(
            self.vision_model, self.processor, self.device, question, images,
            max_new_tokens=max_new_tokens, enable_thinking=enable_thinking,
        )

    @modal.method()
    def classify(self, image_bytes: bytes, context: str, candidates: list[str], max_reasoning_tokens: int = 150) -> dict:
        """`enable_thinking=False` for the reasoning step too (not just scoring) — see
        `score_completions_qwen_native`'s docstring for why an open `<think>` block breaks the
        "reasoning immediately followed by a scored candidate" contract this depends on.
        """
        import io

        from PIL import Image

        from captioner.inference import generate_qwen_native_vision_answer, score_completions_qwen_native

        image = Image.open(io.BytesIO(image_bytes))
        reasoning = generate_qwen_native_vision_answer(
            self.vision_model, self.processor, self.device, context, image, max_new_tokens=max_reasoning_tokens,
            enable_thinking=False,
        )
        logprobs = score_completions_qwen_native(
            self.vision_model, self.processor, self.device, context, image, candidates, reasoning=reasoning,
        )
        return {"reasoning": reasoning, "logprobs": logprobs}


def _modal_backend(
    side: Side, *, modal_app_name: str, repo_id: str | None = None, modality_names: list[str] | None = None,
    enable_thinking: bool | None = None,
) -> EvalBackend:
    """Talks to an already-`modal deploy`ed app — never defines `@app.cls` itself (that lives
    above, once, shared by every runner). `modal deploy eval/backend.py` must have been run at
    least once before this works. Redeploy again after any change to `_EquippedModel`/`_BaseModel`
    — a running deployment keeps serving whatever code it was deployed with until `modal deploy`
    is re-run.
    """
    import json

    if side == "equipped":
        cls = modal.Cls.from_name(modal_app_name, "_EquippedModel")
        instance = cls(
            repo_id=repo_id or DEFAULT_MODEL_REPO_ID,
            modality_names_json=json.dumps(modality_names) if modality_names is not None else "",
        )

        def _generate(raw_inputs: dict, question: str, max_new_tokens: int, system: str | None = None) -> str:
            return instance.infer.remote(raw_inputs, question, max_new_tokens, system)

        def _classify(raw_inputs: dict, context: str, candidates: list[str], max_reasoning_tokens: int) -> dict:
            return instance.classify.remote(raw_inputs, context, candidates, max_reasoning_tokens)

        def _generate_batch(raw_inputs_list: list, question: str, max_new_tokens: int, system: str | None = None) -> list[str]:
            return instance.infer_batch.remote(raw_inputs_list, question, max_new_tokens, system)

        return EvalBackend(side="equipped", _generate=_generate, _classify=_classify, _generate_batch=_generate_batch)

    cls = modal.Cls.from_name(modal_app_name, "_BaseModel")
    instance = cls()

    def _to_png_bytes(image) -> bytes:
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _generate(raw_inputs: dict, question: str, max_new_tokens: int, system: str | None = None) -> str:
        if "image" not in raw_inputs or len(raw_inputs) != 1:
            raise KeyError(
                f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys "
                f"{list(raw_inputs)}."
            )
        return instance.infer.remote(_to_png_bytes(raw_inputs["image"]), question, max_new_tokens, enable_thinking)

    def _classify(raw_inputs: dict, context: str, candidates: list[str], max_reasoning_tokens: int) -> dict:
        if "image" not in raw_inputs or len(raw_inputs) != 1:
            raise KeyError(
                f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys {list(raw_inputs)}."
            )
        return instance.classify.remote(_to_png_bytes(raw_inputs["image"]), context, candidates, max_reasoning_tokens)

    def _generate_batch(raw_inputs_list: list, question: str, max_new_tokens: int, system: str | None = None) -> list[str]:
        for raw in raw_inputs_list:
            if "image" not in raw or len(raw) != 1:
                raise KeyError(f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys {list(raw)}.")
        return instance.infer_batch.remote(
            [_to_png_bytes(r["image"]) for r in raw_inputs_list], question, max_new_tokens, enable_thinking,
        )

    return EvalBackend(side="base", _generate=_generate, _classify=_classify, _generate_batch=_generate_batch)
