"""Google Gemini client wrapper using the Interactions API.

Uses ``client.interactions.create()`` (the recommended API) to generate
content.  Extracts thought summaries (plural — there can be multiple
thought steps) and usage metadata from the interaction response.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Structured response from a Gemini interaction."""

    # The final output text (the caption).
    text: str

    # Summaries from each thought step (can be empty if thinking is
    # not supported or not enabled).
    thought_summaries: list[str] = field(default_factory=list)

    # Token usage.
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int | None = None
    total_tokens: int = 0


class GeminiClient:
    """Wrapper around Google GenAI APIs supporting text and multimodal inputs.

    Prefers the Interactions API for all workflows (text-only and multimodal).
    Images are uploaded via the Files API and referenced in the interaction
    input.  Falls back to ``models.generate_content`` if the Interactions API
    is unavailable.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_level: str = "low",
        thinking_summaries: str = "auto",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.thinking_level = thinking_level
        self.thinking_summaries = thinking_summaries

    def generate(
        self, prompt: str, images: list[bytes] | None = None
    ) -> GeminiResponse:
        """Generate content from a prompt and optional image bytes.

        Always attempts the Interactions API first (supports both text-only
        and multimodal inputs with thinking).  Falls back to
        ``models.generate_content`` if the Interactions API fails.
        """
        try:
            return self._generate_via_interactions(prompt, images=images)
        except Exception as exc:
            logger.warning(
                "Interactions API failed (%s); falling back to generate_content.",
                exc,
            )
            return self._generate_via_generate_content(prompt, images=images)

    # ------------------------------------------------------------------
    # Interactions API (preferred for all workflows)
    # ------------------------------------------------------------------

    def _generate_via_interactions(
        self, prompt: str, images: list[bytes] | None = None
    ) -> GeminiResponse:
        generation_config: dict = {}

        # Only set thinking params if we intend to use them.
        if self.thinking_level:
            generation_config["thinking_level"] = self.thinking_level
        if self.thinking_summaries:
            generation_config["thinking_summaries"] = self.thinking_summaries

        # Upload images via the Files API if provided.
        uploaded_files: list[Any] = []
        if images:
            for i, img_bytes in enumerate(images):
                with tempfile.NamedTemporaryFile(
                    suffix=".png", prefix=f"spectrum_{i}_", delete=False
                ) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = Path(tmp.name)
                try:
                    uploaded = self.client.files.upload(
                        file=tmp_path,
                        config={"mime_type": "image/png"},
                    )
                    uploaded_files.append(uploaded)
                finally:
                    tmp_path.unlink(missing_ok=True)

        # Build the input: multimodal list of typed content objects, or plain text string.
        interaction_input: str | list[dict[str, Any]] = prompt
        if uploaded_files:
            interaction_input = [
                {"type": "image", "uri": f.uri, "mime_type": "image/png"}
                for f in uploaded_files
            ] + [{"type": "text", "text": prompt}]

        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=interaction_input,
                generation_config=generation_config if generation_config else None,
            )

            # Extract thought summaries and output text from steps.
            thought_summaries: list[str] = []
            output_text = ""

            for step in interaction.steps:
                if step.type == "thought":
                    if step.summary:
                        for content_block in step.summary:
                            if content_block.type == "text" and content_block.text:
                                thought_summaries.append(content_block.text)
                elif step.type == "model_output":
                    if step.content:
                        for content_block in step.content:
                            if content_block.type == "text" and content_block.text:
                                output_text += content_block.text

            # Extract usage metadata.
            usage = interaction.usage
            return GeminiResponse(
                text=output_text.strip(),
                thought_summaries=thought_summaries,
                input_tokens=getattr(usage, "total_input_tokens", 0) or 0,
                output_tokens=getattr(usage, "total_output_tokens", 0) or 0,
                thought_tokens=getattr(usage, "total_thought_tokens", None),
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        finally:
            # Clean up uploaded files from the Files API.
            for f in uploaded_files:
                try:
                    self.client.files.delete(name=f.name)
                except Exception as del_exc:
                    logger.debug("Failed to delete uploaded file %s: %s", f.name, del_exc)

    # ------------------------------------------------------------------
    # Fallback: generate_content
    # ------------------------------------------------------------------

    def _generate_via_generate_content(
        self, prompt: str, images: list[bytes] | None = None
    ) -> GeminiResponse:
        from google.genai import types

        contents: list[Any] = [prompt]
        if images:
            for img in images:
                contents.append(types.Part.from_bytes(data=img, mime_type="image/png"))

        config_kwargs: dict[str, Any] = {}
        if self.thinking_level:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            except Exception:
                pass

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        thought_summaries: list[str] = []
        text = response.text or ""

        # Extract thought summaries from candidate parts if present
        if response.candidates:
            cand = response.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "thought", False) and getattr(part, "text", None):
                        thought_summaries.append(part.text)

        usage = response.usage_metadata
        return GeminiResponse(
            text=text.strip(),
            thought_summaries=thought_summaries,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            thought_tokens=getattr(usage, "candidates_thought_token_count", None)
            or getattr(usage, "thought_token_count", None),
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )
