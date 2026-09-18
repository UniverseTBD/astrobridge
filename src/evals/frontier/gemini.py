from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional
from tqdm import tqdm

from google import genai
from google.genai import types

from .base import FrontierModel, FrontierResponse


class GeminiFrontierModel(FrontierModel):
    def __init__(self, config: dict):
        self._model_name = config.get("gemini_model", "gemini-2.5-flash")
        self._num_workers = config.get("gemini_num_workers", config.get("num_workers", 8))
        self._temperature = config.get("gemini_temperature", config.get("temperature", 0.0))
        self._max_tokens = config.get("max_tokens", 1024)
        self._fallback_max_tokens = config.get("fallback_max_tokens", 128)
        self._max_retries = config.get("max_retries", 3)
        self._client = genai.Client()

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "GeminiFrontierModel",
            "model_name": self._model_name,
            "num_workers": self._num_workers,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        delay = 1.0
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise last_exception

    def predict(
        self,
        prompt: str,
        parse_fn: Callable[[str], Any],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> FrontierResponse:
        gen_config = types.GenerateContentConfig(
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        chat = None
        raw_text = ""
        try:
            chat = self._client.chats.create(model=self._model_name, config=gen_config)
            response = self._call_with_retry(lambda: chat.send_message(prompt))
            raw_text = response.text if response and response.text else ""
        except Exception as e:
            print(f"Gemini API error: {e}")
            raw_text = ""

        parsed = parse_fn(raw_text)
        forced_fallback = False

        if parsed is None and fallback_tag and chat:
            try:
                forced_fallback = True
                fallback_config = types.GenerateContentConfig(
                    max_output_tokens=self._fallback_max_tokens,
                    temperature=self._temperature,
                )
                fallback_prompt = (
                    f"Please continue your previous response and output only the "
                    f"final answer starting with the indicator '{fallback_tag.strip()}'"
                )
                fb_response = self._call_with_retry(
                    lambda: chat.send_message(fallback_prompt, config=fallback_config)
                )
                fb_text = fb_response.text if fb_response and fb_response.text else ""
                raw_text = (raw_text + "\n" + fb_text).strip()
                parsed = parse_fn(raw_text)
            except Exception as e:
                print(f"Gemini API fallback error: {e}")

        return FrontierResponse(
            raw_text=raw_text,
            parsed=parsed,
            forced_fallback=forced_fallback,
            metadata={"model": self._model_name},
        )

    def predict_all(
        self,
        prompts: List[str],
        parse_fn: Callable[[str], Any] | List[Callable[[str], Any]],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[FrontierResponse]:
        responses: List[Optional[FrontierResponse]] = [None] * len(prompts)
        is_fn_list = isinstance(parse_fn, list)

        def worker(idx: int, p: str):
            fn = parse_fn[idx] if is_fn_list else parse_fn
            return idx, self.predict(p, fn, fallback_tag=fallback_tag, system_prompt=system_prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx = {
                executor.submit(worker, idx, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(prompts),
                desc=f"Frontier Model ({self._model_name})",
                leave=False,
            ):
                idx, resp = future.result()
                responses[idx] = resp

        return [r for r in responses if r is not None]

