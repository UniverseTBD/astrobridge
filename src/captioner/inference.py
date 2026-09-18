"""Single-object, no-cache inference: raw image/spectrum arrays in, caption out — for a
brand-new object that was never part of the manifest/embedding-cache pipeline. Distinct from
data/dataset.py's CaptionerDataset, which only reads pre-cached embeddings keyed by object_id.

Runs the same three stages training does (encoders -> FusionStack -> LLM), just live instead of
cached, and with the same autocast(bf16) wrapping eval/groundedness.py needs to reconcile the
fp32 FusionStack against the bf16 LLM (see that module's docstring for why that's required).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from captioner.encoders.registry import build_encoder
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm
from captioner.publish import filter_missing_lora_keys
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.prompt import build_wrapper_text, human_readable_subset


def load_inference_model(
    cfg: DictConfig,
    checkpoint_dir: Path,
    lora_dir: Path | None,
    device: str = "cuda",
    modality_names: list[str] | None = None,
) -> tuple[Captioner, Any, dict]:
    """`checkpoint_dir` needs middle.pt (e.g. outputs/checkpoints/stage2/best, or stage1's).
    `lora_dir` is the matching lora/ subfolder — pass None for a stage-1-only checkpoint.

    `modality_names`: which encoders to actually build — defaults to every modality in
    `cfg.modalities` (`None`), but pass an explicit subset (e.g. `["image"]`) when a caller only
    ever uses some of them. Confirmed real cost, not just tidiness: building an encoder means
    downloading and loading its own model weights (AION for image/spectra, ATCAT for light
    curve) — building all three when only one is ever used wastes real time and, on a rented GPU,
    real money.

    `lora_dir` holds this repo's own checkpoint format (a raw filtered state_dict at
    lora/adapter.pt — see train/checkpoint.py's save_checkpoint), NOT PEFT's own
    save_pretrained format (adapter_config.json + safetensors). `PeftModel.from_pretrained`
    expects the latter, so the LoRA config is reconstructed here from configs/stage2.yaml and
    the state dict loaded manually — same approach scripts/06_publish_model.py uses to convert
    into the standard format for publishing.
    """
    llm, tokenizer = build_llm(cfg)
    if lora_dir is not None:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=float(cfg.lora.dropout),
            target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        lora_state = torch.load(Path(lora_dir) / "adapter.pt", map_location="cpu", weights_only=False)
        missing, _unexpected = llm.load_state_dict(lora_state, strict=False)
        real_missing = filter_missing_lora_keys(missing)
        if real_missing:
            raise RuntimeError(
                f"LoRA state from {Path(lora_dir) / 'adapter.pt'} didn't fully load onto a "
                f"freshly LoRA-wrapped {cfg.llm.name} — missing keys: {real_missing[:10]}. "
                "This means either the checkpoint doesn't match configs/stage2.yaml's current "
                "lora.* settings, or the checkpoint is corrupt."
            )
    d_llm = get_llm_hidden_size(llm)

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
        adapter_target_norm=llm_embedding_norm(llm),
    )
    fusion_stack.load_state_dict(
        torch.load(Path(checkpoint_dir) / "middle.pt", map_location="cpu", weights_only=False)
    )

    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(device)
    model.eval()

    names = list(cfg.modalities) if modality_names is None else modality_names
    encoders = {name: build_encoder(name, cfg.modalities[name], device=device) for name in names}
    return model, tokenizer, encoders


def load_inference_model_from_hub(
    cfg: DictConfig, repo_id: str, device: str = "cuda", modality_names: list[str] | None = None
) -> tuple[Captioner, Any, dict]:
    """Loads a model *published* via scripts/06_publish_model.py (e.g. from `make publish`),
    as opposed to `load_inference_model` above which reads this repo's own internal training-
    checkpoint layout directly off disk.

    `modality_names`: see load_inference_model's docstring — same meaning, same real cost
    reason for passing an explicit subset rather than always building every encoder.

    The published repo is genuinely a different format, not just a different location: publish
    converts the raw lora/adapter.pt state_dict into PEFT's own save_pretrained layout
    (adapter_config.json + adapter_model.safetensors), specifically so it loads the standard
    way. That's why this function can call `PeftModel.from_pretrained(llm, repo_id)` directly,
    unlike `load_inference_model`, which has to reconstruct the LoraConfig by hand because the
    local checkpoint format doesn't carry one.

    `middle.pt` (the fusion stack) isn't a PEFT/HF concept, so it's just downloaded as a plain
    file via `hf_hub_download` and loaded the same way as everywhere else in this codebase.
    """
    from huggingface_hub import hf_hub_download
    from peft import PeftModel

    llm, tokenizer = build_llm(cfg)
    llm = PeftModel.from_pretrained(llm, repo_id)
    d_llm = get_llm_hidden_size(llm)

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
        adapter_target_norm=llm_embedding_norm(llm),
    )
    middle_path = hf_hub_download(repo_id=repo_id, filename="middle.pt")
    fusion_stack.load_state_dict(torch.load(middle_path, map_location="cpu", weights_only=False))

    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(device)
    model.eval()

    names = list(cfg.modalities) if modality_names is None else modality_names
    encoders = {name: build_encoder(name, cfg.modalities[name], device=device) for name in names}
    return model, tokenizer, encoders


def _build_modality_batch(
    encoders: dict,
    modality_out_dims: dict[str, int],
    modality_max_tokens: dict[str, int],
    raw_inputs: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Shared by `generate_caption` and `score_completions` so both build the exact same
    modality batch from the same `raw_inputs` — a modality absent from `raw_inputs` is true
    exclusion (all-True mask), never a zero-content placeholder (§6).
    """
    modality_batch: dict[str, dict[str, torch.Tensor]] = {}
    for name, out_dim in modality_out_dims.items():
        T_m = modality_max_tokens[name]
        tokens = torch.zeros((1, T_m, out_dim), dtype=torch.float32, device=device)
        mask = torch.ones((1, T_m), dtype=torch.bool, device=device)  # True = pad/absent by default
        if name in raw_inputs:
            # AION's real token count depends on the actual input (image pixel dims / spectrum
            # length), not just the `num_encoder_tokens` the encoder was built with — confirmed
            # real, not a bug in the encoder: a 384x384 image or a ~7800-sample spectrum can both
            # produce fewer raw tokens than max_tokens. data/collate.py already pads/truncates to
            # max_tokens per-object during training for exactly this reason; mirror that here
            # instead of requiring an exact match (which training never assumed either).
            raw_tokens = encoders[name].encode(raw_inputs[name]).to(torch.float32)  # (1, T_raw, out_dim)
            n = min(raw_tokens.shape[1], T_m)
            tokens[:, :n] = raw_tokens[:, :n].to(device)
            mask[:, :n] = False
        modality_batch[name] = {"tokens": tokens, "mask": mask}
    return modality_batch


def _resolve_system_instruction(prompt_cfg, shown: frozenset[str], question: str | None, system: str | None) -> tuple[str, str]:
    """The deterministic default `generate_caption`/`score_completions` both pin: training
    samples across every `system_variants`/`instruction_variants` entry, inference always pins
    the first of each unless the caller overrides.
    """
    system = system if system is not None else str(prompt_cfg.system_variants[0])
    instruction = question if question is not None else str(prompt_cfg.instruction_variants[0])
    if "{modalities}" in instruction:
        instruction = instruction.format(modalities=human_readable_subset(shown))
    return system, instruction


@torch.no_grad()
def _stop_token_ids(tokenizer) -> list[int]:
    """The ids `generate()` must stop on for the equipped path.

    `Qwen3_5Config` carries NO `eos_token_id` (checked live — it raises AttributeError), and we call
    `model.llm.generate()` directly rather than going through the model's own generation_config, so
    without this the caption never terminates: the model emits `<|im_end|>` at the true end of its
    turn, generation keeps going to `max_new_tokens`, and it hallucinates a whole second user/
    assistant turn. `skip_special_tokens=True` then strips the markers, leaving bare "user" /
    "assistant" / "<think>" text welded onto the caption. That corrupted 73% of a v7 run before this
    was caught, and 12/266 of the earlier v5 run. The base side never had it because Qwen's own
    generation_config supplies the stop token.
    """
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return sorted(i for i in ids if i is not None and i >= 0)


def generate_caption(
    model: Captioner,
    tokenizer,
    encoders: dict,
    modality_out_dims: dict[str, int],
    modality_max_tokens: dict[str, int],
    prompt_cfg,
    device: str,
    raw_inputs: dict[str, dict[str, Any]],
    max_new_tokens: int = 128,
    question: str | None = None,
    system: str | None = None,
) -> str:
    """`raw_inputs`: {modality_name: encoder-specific batch dict}, only for modalities actually
    present — e.g. {"image": {"pixel_values": ...}} or
    {"spectra": {"flux": ..., "ivar": ..., "mask": ..., "wavelength": ..., "survey": [...]}}, or
    {"lightcurve": {"flux": ..., "flux_err": ..., "time": ..., "mask": ..., "channel_index": ...}}
    (build that one with data/transients_dataset.py's `prepare_lightcurve_arrays`, so live inference
    applies exactly the same detection-window trim and padding the cache was built with).
    See each encoder's `encode()` docstring (encoders/aion_image.py, aion_spectrum.py) for the
    exact field contract. A modality absent from `raw_inputs` is treated as not shown at all —
    true exclusion (all-True mask), never a zero-content placeholder (§6).

    Builds the same chat-template structure training used (`configs/model.yaml` prompt block):
    `wrapper_pre` (with `system`) -> observation vectors -> `wrapper_post` (with the instruction)
    -> generate. `question` fills the instruction slot; default is `prompt_cfg.instruction_
    variants[0]`. `system` defaults to `prompt_cfg.system_variants[0]`. Both defaults are the
    deterministic choice — training samples across all variants, inference pins one.
    """
    if not raw_inputs:
        raise ValueError("raw_inputs is empty — at least one modality must be provided.")

    shown = frozenset(raw_inputs.keys())
    modality_batch = _build_modality_batch(encoders, modality_out_dims, modality_max_tokens, raw_inputs, device)

    system, instruction = _resolve_system_instruction(prompt_cfg, shown, question, system)
    pre_text, post_text = build_wrapper_text(prompt_cfg, system, instruction)
    pre_ids = tokenizer(pre_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    post_ids = tokenizer(post_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        prefix = model.fusion_stack(modality_batch)
        embed_fn = model.llm.get_input_embeddings()
        inputs_embeds = torch.cat([embed_fn(pre_ids), prefix, embed_fn(post_ids)], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        gen = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=_stop_token_ids(tokenizer),
        )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


@torch.no_grad()
def generate_captions_batched(
    model: Captioner,
    tokenizer,
    encoders: dict,
    modality_out_dims: dict[str, int],
    modality_max_tokens: dict[str, int],
    prompt_cfg,
    device: str,
    raw_inputs_list: list[dict[str, dict[str, Any]]],
    max_new_tokens: int = 128,
    question: str | None = None,
    system: str | None = None,
) -> list[str]:
    """`generate_caption` over B objects in one forward pass — same prompt construction, same
    greedy decoding, one caption per input, in order.

    Why: single-sequence decode on a 9B model is memory-bandwidth-bound, not compute-bound. Each
    decode step reads the full ~18GB of weights from HBM to emit ONE token, so the GPU idles on
    memory. Batching reads those weights once per step and emits B tokens — nearly free until the
    batch is large enough to become compute-bound. VRAM cost is small: only 8 of Qwen3.5-9B's 32
    layers are full attention (the other 24 are linear-attention with fixed-size state), so the KV
    cache is tens of MB per sequence against ~18GB of fixed weights.

    **Left padding, not right.** Training pads right (loss is masked per position, so trailing pad
    is harmless). Generation must pad LEFT: every sequence has to END at the same index, or the
    "next token" position differs per row and the model continues from the wrong place — which
    yields subtly wrong captions rather than an error. `tests/test_inference.py` asserts the last
    mask column is real for every row; A/B a few objects against `generate_caption` after changing
    anything here.
    """
    B = len(raw_inputs_list)
    if B == 0:
        return []
    if any(not raw for raw in raw_inputs_list):
        raise ValueError("a raw_inputs entry is empty — every object needs at least one modality.")

    modality_batch: dict[str, dict[str, torch.Tensor]] = {}
    for name, out_dim in modality_out_dims.items():
        T_m = modality_max_tokens[name]
        tokens = torch.zeros((B, T_m, out_dim), dtype=torch.float32, device=device)
        mask = torch.ones((B, T_m), dtype=torch.bool, device=device)  # True = pad/absent
        for i, raw in enumerate(raw_inputs_list):
            if name in raw:
                raw_tokens = encoders[name].encode(raw[name]).to(torch.float32)  # (1, T_raw, out_dim)
                n = min(raw_tokens.shape[1], T_m)
                tokens[i, :n] = raw_tokens[0, :n].to(device)
                mask[i, :n] = False
        modality_batch[name] = {"tokens": tokens, "mask": mask}

    pre_ids_list, post_ids_list = [], []
    for raw in raw_inputs_list:
        sys_i, instr_i = _resolve_system_instruction(prompt_cfg, frozenset(raw.keys()), question, system)
        pre_text, post_text = build_wrapper_text(prompt_cfg, sys_i, instr_i)
        pre_ids_list.append(tokenizer(pre_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device))
        post_ids_list.append(tokenizer(post_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device))

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        prefix = model.fusion_stack(modality_batch)  # (B, n_queries, d_llm)
        embed_fn = model.llm.get_input_embeddings()
        seqs = [
            torch.cat([embed_fn(pre_ids_list[i])[0], prefix[i], embed_fn(post_ids_list[i])[0]], dim=0)
            for i in range(B)
        ]
        L = max(s.shape[0] for s in seqs)
        inputs_embeds = torch.zeros((B, L, seqs[0].shape[-1]), dtype=seqs[0].dtype, device=device)
        attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            inputs_embeds[i, L - s.shape[0]:] = s
            attention_mask[i, L - s.shape[0]:] = 1

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        gen = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=_stop_token_ids(tokenizer),
        )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)



@torch.no_grad()
def score_completions(
    model: Captioner,
    tokenizer,
    encoders: dict,
    modality_out_dims: dict[str, int],
    modality_max_tokens: dict[str, int],
    prompt_cfg,
    device: str,
    raw_inputs: dict[str, dict[str, Any]],
    candidates: list[str],
    question: str | None = None,
    system: str | None = None,
    reasoning: str = "",
) -> dict[str, float]:
    """Deterministic classification without asking the model to emit a label at all: scores each
    of `candidates` (fixed strings, e.g. `[" SN Ia", " SN II", " SN Ibc"]`) as a teacher-forced
    continuation of the same chat-template prompt `generate_caption` would build, and returns each
    candidate's length-normalised mean log-prob. Argmax over a fixed candidate set is always a
    valid label by construction — no parsing, no unparseable output, fully deterministic given the
    same weights.

    `reasoning`: free-form text (typically produced by a prior `generate_caption(..., question=
    the same `question`)` call) appended to the end of `post_text`, i.e. inside the assistant
    turn, before the candidate is scored as its continuation — this is what lets the model reason
    about the observation before being scored on the label, without the reasoning itself counting
    toward any candidate's score.

    Reuses `Captioner.forward`'s existing teacher-forcing contract unchanged: it already returns
    HF's mean cross-entropy over exactly the target span (`caption_ids`), so `-loss` is already the
    length-normalised mean log-prob of the candidate — the same reason this avoids penalising a
    multi-token candidate (e.g. `" SN Ibc"`, 3 tokens) against a shorter one (`" SN Ia"`, 2 tokens).
    """
    if not raw_inputs:
        raise ValueError("raw_inputs is empty — at least one modality must be provided.")
    if not candidates:
        raise ValueError("candidates is empty — nothing to score.")

    shown = frozenset(raw_inputs.keys())
    modality_batch = _build_modality_batch(encoders, modality_out_dims, modality_max_tokens, raw_inputs, device)

    system, instruction = _resolve_system_instruction(prompt_cfg, shown, question, system)
    pre_text, post_text = build_wrapper_text(prompt_cfg, system, instruction)
    post_text = post_text + reasoning
    caption_suffix = str(prompt_cfg.caption_suffix)

    pre_ids = tokenizer(pre_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    post_ids = tokenizer(post_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scores: dict[str, float] = {}
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        for candidate in candidates:
            caption_ids = tokenizer(
                candidate + caption_suffix, add_special_tokens=False, return_tensors="pt",
            )["input_ids"].to(device)
            outputs = model(modality_batch, pre_ids, post_ids, caption_ids)
            scores[candidate] = -float(outputs.loss.item())
    return scores


def load_qwen_native_vision_model(cfg: DictConfig, device: str = "cuda"):
    """Loads a genuinely separate, plain Qwen3.5 instance for generate_qwen_native_vision_answer
    — NOT reusable with the model load_inference_model_from_hub produces, and this isn't a style
    choice, it's confirmed real: `AutoModelForCausalLM` (what build_llm uses) resolves to
    `Qwen3_5ForCausalLM` for this model_type — a genuinely different, text-only Python class from
    `Qwen3_5ForConditionalGeneration` (what `AutoModelForImageTextToText`/`AutoModelForMultimodal
    LM` resolve to), confirmed by direct inspection of transformers' real model-mapping tables,
    not assumed. `Qwen3_5ForCausalLM` structurally has no vision tower at all — no amount of
    unwrapping LoRA/PEFT changes that, since the limitation is in the base class itself.

    This does mean a second, real set of weights gets loaded (same underlying language-model
    weights as the text-only class — `Qwen3_5ForConditionalGeneration`'s language_model submodule
    is built from the same checkpoint's `config.text_config`, just with a vision tower attached
    alongside it — but a second copy in memory all the same). inference/compare.py loads this
    sequentially, after freeing the text-only model, specifically to keep peak VRAM within one
    GPU's budget rather than needing both loaded simultaneously.

    Returns (model, processor). No LoRA involved at any point — this model is never PEFT-wrapped,
    so there's nothing to disable; it answers purely on Qwen's own pretraining.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    trust_remote_code = bool(cfg.llm.get("trust_remote_code", False))
    processor = AutoProcessor.from_pretrained(cfg.llm.name, trust_remote_code=trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.llm.name,
        torch_dtype=torch.bfloat16 if cfg.llm.dtype == "bfloat16" else torch.float32,
        attn_implementation=cfg.llm.attn_impl,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def generate_qwen_native_vision_answer(
    model, processor, device: str, question: str, image, max_new_tokens: int = 128,
    enable_thinking: bool | None = None,
) -> str:
    """The "plain Qwen, genuinely out of the box" side of a comparison against generate_caption's
    fully-equipped answer — routed through Qwen's own native multimodal pathway instead of this
    project's AION encoders + fusion stack. Nothing this project trained touches this call.

    `model`: from load_qwen_native_vision_model — a plain `Qwen3_5ForConditionalGeneration`
    instance, never PEFT-wrapped, so there's no LoRA to disable here at all (unlike an earlier
    version of this function, which mistakenly tried to reuse the text-only LoRA-wrapped model —
    see load_qwen_native_vision_model's docstring for why that doesn't work).

    `image`: a PIL.Image — build one from a raw grz array with utils/image.py's grz_to_rgb first.
    Confirmed real, not guessed: Qwen's chat_template.jinja checks for an "image" key in each
    content block (`'image' in item or 'image_url' in item or item.type == 'image'`), matching
    the {"type": "image", "image": ...} shape used below.

    `enable_thinking`: `Qwen/Qwen3.5-9B`'s real chat template has a built-in reasoning mode —
    confirmed by reading the template directly: `add_generation_prompt` normally opens an empty
    `<think>\n` block that the model fills with step-by-step reasoning before ever reaching an
    answer, which is exactly why free-form questions ("what class is this?") were getting
    truncated mid-reasoning even at generous `max_new_tokens`. Passing `enable_thinking=False`
    closes that block immediately (`<think>\n\n</think>\n\n`), forcing a direct answer — confirmed
    live via `apply_chat_template(..., enable_thinking=False)`. Left as `None` (the template's own
    default, thinking ON) unless the caller opts out, since "genuinely out of the box" for a
    general caption/comparison use case means Qwen's actual default behavior, not a modification
    — callers that specifically need a short, direct answer (e.g. eval/prompt_playground.py's
    classification prompts) should pass `enable_thinking=False` explicitly.
    """
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    chat_template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        **chat_template_kwargs,
    ).to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


@torch.no_grad()
def generate_qwen_native_vision_answers_batched(
    model, processor, device: str, question: str, images: list, max_new_tokens: int = 128,
    enable_thinking: bool | None = None,
) -> list[str]:
    """Batched `generate_qwen_native_vision_answer` — same single-turn `{image, question}` message
    through Qwen's own chat template, B images at once, answers in input order.

    `padding_side` is forced to "left" for the duration and restored after: generation needs every
    sequence to end at the same index (see `generate_captions_batched`), while the processor's
    tokenizer defaults to right padding for training.
    """
    if not images:
        return []

    messages = [
        [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": question}]}]
        for im in images
    ]
    chat_template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}

    previous_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True, **chat_template_kwargs,
        ).to(device)
    finally:
        processor.tokenizer.padding_side = previous_side

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    n_prompt = inputs["input_ids"].shape[-1]
    return [processor.decode(g[n_prompt:], skip_special_tokens=True) for g in gen]



@torch.no_grad()
def score_completions_qwen_native(
    model, processor, device: str, question: str, image, candidates: list[str],
    reasoning: str = "", enable_thinking: bool | None = False,
) -> dict[str, float]:
    """Base-side counterpart to `score_completions`: scores each of `candidates` as a
    teacher-forced continuation of Qwen's own chat template, with `reasoning` (typically a prior
    `generate_qwen_native_vision_answer(..., question)` call's output) preceding the candidate in
    the assistant's turn — the model reasons freely, then is scored on the label alone, without
    the reasoning itself counting toward any candidate's score.

    Each candidate's full sequence (context + reasoning + candidate) is built via one fresh
    `apply_chat_template(..., continue_final_message=True)` call, not by manually concatenating
    token ids onto an already-tokenized context: `Qwen3_5ForConditionalGeneration.forward`
    recomputes 3D (mRoPE) position ids from `input_ids`/`attention_mask`/the image-grid tensors
    every call (confirmed live — hand-splicing extra ids onto a previously tokenized+forwarded
    context raises `IndexError` deep inside `get_rope_index`, a real shape mismatch, not a
    theoretical concern), so every forward pass here gets a self-consistently retokenized sequence
    instead. `SN_CANDIDATES`'s leading space (`" SN Ia"`, not `"SN Ia"`) is what makes this safe
    from the BPE-boundary issue that hand-splicing was meant to avoid — a leading-space token is a
    clean BPE boundary regardless of what precedes it, so retokenizing the joined text still
    produces the same trailing token count as tokenizing the candidate alone.

    `enable_thinking` defaults to `False` here (unlike `generate_qwen_native_vision_answer`,
    which defaults to `None`/thinking-on): scoring a fixed candidate as the very next tokens after
    `reasoning` only makes sense if the template isn't about to reopen an empty `<think>` block in
    between, so the "genuinely out of the box" default doesn't apply to this function.
    """
    if not candidates:
        raise ValueError("candidates is empty — nothing to score.")

    chat_template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}

    def _tokenize_assistant_turn(text: str):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]
        return processor.apply_chat_template(
            messages, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt",
            **chat_template_kwargs,
        ).to(device)

    n_context = _tokenize_assistant_turn(reasoning)["input_ids"].shape[1]

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scores: dict[str, float] = {}
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        for candidate in candidates:
            full_inputs = _tokenize_assistant_turn(reasoning + candidate)
            input_ids = full_inputs["input_ids"]
            candidate_ids = input_ids[:, n_context:]
            model_kwargs = {k: v for k, v in full_inputs.items() if k != "input_ids"}
            outputs = model(input_ids=input_ids, **model_kwargs)
            # logits at position i predict token i+1: positions [n_context-1, end-1) predict the
            # candidate span [n_context, end).
            candidate_logits = outputs.logits[:, n_context - 1:-1, :]
            log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
            scores[candidate] = float(token_log_probs.mean().item())
    return scores
