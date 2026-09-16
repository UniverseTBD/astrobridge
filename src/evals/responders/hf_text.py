from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import (
    BaseResponder,
    GeneratedCaption,
    resolve_caption_prompt,
    resolve_system_prompt,
)
from .sample import SpectrumSample
from .utils import (
    subsample_spectrum,
    format_spectrum_text,
    clean_and_extract_caption,
    has_explicit_caption,
)


class HFTextResponder(BaseResponder):
    def __init__(self, config: dict, device: str):
        assert "hf_model_id" in config and config["hf_model_id"], (
            "Missing 'hf_model_id' in config"
        )
        self._model_id = config["hf_model_id"]
        self.caption_prompt = resolve_caption_prompt(
            config, "caption_generation/text_baseline.jinja2"
        )
        self.system_prompt = resolve_system_prompt(
            config, "caption_generation/system_caption.jinja2"
        )
        self._num_points = config.get("num_points", 100)
        self._max_tokens = config.get("max_tokens", 2048)
        self._fallback_max_tokens = config.get("fallback_max_tokens", 256)
        self._repetition_penalty = config.get("repetition_penalty", 1.1)
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        if device.startswith("cuda"):
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                device_map={"": device},
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        elif device in ("cpu", "mps"):
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        self._model.eval()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "HFTextResponder",
            "model_id": self._model_id,
            "num_points": self._num_points,
            "caption_prompt": self.caption_prompt,
            "system_prompt": self.system_prompt,
            "max_tokens": self._max_tokens,
            "fallback_max_tokens": self._fallback_max_tokens,
            "repetition_penalty": self._repetition_penalty,
        }

    def generate_captions(
        self,
        samples: List[SpectrumSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        prompt = prompt_override or self.caption_prompt

        messages_batch = []
        for sample in samples:
            w_str, f_str = subsample_spectrum(
                sample.wavelength, sample.flux, num_points=self._num_points
            )
            spectrum_text = format_spectrum_text(w_str, f_str, self._num_points)
            full_prompt = f"{prompt}\n\n{spectrum_text}"

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": full_prompt})
            messages_batch.append(messages)

        texts = []
        for msgs in messages_batch:
            try:
                t = self._tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                if len(msgs) > 1 and msgs[0]["role"] == "system":
                    merged_user = f"{msgs[0]['content']}\n\n{msgs[1]['content']}"
                    t = self._tokenizer.apply_chat_template(
                        [{"role": "user", "content": merged_user}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    raise
            texts.append(t)

        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        inputs = self._tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self._max_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self._repetition_penalty and self._repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(self._repetition_penalty)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        raw_responses = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        fallback_indices = [
            i for i, raw in enumerate(raw_responses) if not has_explicit_caption(raw)
        ]
        fallback_triggered = {i: False for i in range(len(samples))}

        if fallback_indices and self._fallback_max_tokens > 0:
            suffix_text = "\n\nFinal answer below.\nCAPTION: "
            fb_texts = []
            for i in fallback_indices:
                orig_msgs = messages_batch[i]
                cont_msgs = list(orig_msgs) + [
                    {
                        "role": "assistant",
                        "content": f"{raw_responses[i]}{suffix_text}",
                    }
                ]
                try:
                    t = self._tokenizer.apply_chat_template(
                        cont_msgs, tokenize=False, continue_final_message=True
                    )
                except Exception:
                    if len(cont_msgs) > 2 and cont_msgs[0]["role"] == "system":
                        merged_user = f"{cont_msgs[0]['content']}\n\n{cont_msgs[1]['content']}"
                        merged_cont = [
                            {"role": "user", "content": merged_user},
                            cont_msgs[-1],
                        ]
                        t = self._tokenizer.apply_chat_template(
                            merged_cont, tokenize=False, continue_final_message=True
                        )
                    else:
                        raise
                fb_texts.append(t)

            try:
                self._tokenizer.padding_side = "left"
                fb_inputs = self._tokenizer(fb_texts, return_tensors="pt", padding=True)
                fb_inputs = {k: v.to(self._device) for k, v in fb_inputs.items()}

                fb_gen_kwargs = {
                    "max_new_tokens": self._fallback_max_tokens,
                    "do_sample": False,
                    "pad_token_id": self._tokenizer.pad_token_id,
                }
                if self._repetition_penalty and self._repetition_penalty != 1.0:
                    fb_gen_kwargs["repetition_penalty"] = float(self._repetition_penalty)

                with torch.no_grad():
                    fb_out = self._model.generate(
                        **fb_inputs,
                        **fb_gen_kwargs,
                    )
                fb_input_len = fb_inputs["input_ids"].shape[1]
                fb_new_tokens = fb_out[:, fb_input_len:]
                fb_raw_responses = self._tokenizer.batch_decode(
                    fb_new_tokens, skip_special_tokens=True
                )
                for k, i in enumerate(fallback_indices):
                    raw_responses[i] = (raw_responses[i] + suffix_text + fb_raw_responses[k]).strip()
                    fallback_triggered[i] = True
            except Exception as e:
                print(f"Batched fallback generation error: {e}")

        captions = []
        for i, (sample, raw) in enumerate(zip(samples, raw_responses)):
            cleaned = clean_and_extract_caption(raw)
            captions.append(
                GeneratedCaption(
                    sample_id=sample.sample_id,
                    caption=cleaned if cleaned else raw.strip(),
                    responder_type="hf_text",
                    model_id=self._model_id,
                    caption_prompt=prompt,
                    survey=sample.survey,
                    metadata={
                        "raw_output": raw,
                        "fallback_triggered": fallback_triggered[i],
                    },
                )
            )
        return captions
