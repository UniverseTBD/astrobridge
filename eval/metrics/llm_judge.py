"""Gemini-as-judge classification: turns a free-form caption/description into one of a fixed set
of labels by asking a small, cheap LLM to read it and pick the best match — the automated
counterpart to a human reading the caption and judging which class it describes.

Why this exists (see `eval/metrics/caption_to_label.py`'s module docstring and the real evidence
behind it): a live manual read of 30 real Galaxy10 captions found the regex/keyword extractor
(`predict_label_from_free_text`) badly undercounts real signal — captions like "distinctly
asymmetric, disturbed morphology... blue knot..." clearly describe Disturbed Galaxies, but only
match if the exact right words are in the synonym list. An LLM judge generalizes past exact
wording the way a human reader does, without needing an ever-growing hand-maintained synonym
list.

Deliberately a *small, cheap* model (`gemini-2.5-flash` by default) — this task is reading
comprehension over a short paragraph and picking one of ~10 options, not something that needs a
frontier model. Uses Gemini's structured-output mode (`response_schema`) so the answer is always
exactly one of the valid indices — no free-text parsing of the judge's own answer either.
"""
from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, create_model

from captioner.utils.logging import get_logger

load_dotenv()  # picks up GEMINI_API_KEY from a local .env — gitignored, never committed

logger = get_logger(__name__)

# Confirmed live: "gemini-2.5-flash" now 404s for new API keys ("no longer available to new
# users" — the API's own error names gemini-3.6-flash as the replacement). Update this again if
# it goes stale the same way; this task (read a paragraph, pick 1 of ~10 options) never needs a
# frontier model, just whatever the current cheap/fast tier is.
DEFAULT_JUDGE_MODEL = "gemini-3.6-flash"

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert astronomer. You will be given a free-form description of an astronomical "
    "object and a numbered list of candidate classes. Read the description and pick the single "
    "class index that best matches what it describes. The description was NOT written with this "
    "exact taxonomy in mind, so use your own judgement about which class the described physical "
    "properties correspond to — do not require an exact wording match."
)


def _answer_schema(n_classes: int) -> type[BaseModel]:
    """One `Literal["0", ..., str(n_classes-1)]` field — Gemini's structured-output mode enforces
    this at generation time, so the response is always a valid index, never free text to parse.
    """
    index_literal = Literal[tuple(str(i) for i in range(n_classes))]  # type: ignore[valid-type]
    return create_model("LLMJudgeAnswer", classification=(index_literal, ...))


def build_judge_prompt(caption: str, label_vocabulary: list[str]) -> str:
    class_list = "\n".join(f"  {i}: {name}" for i, name in enumerate(label_vocabulary))
    return (
        f"<classes>\n{class_list}\n</classes>\n"
        f"<description>\n{caption}\n</description>\n"
        "Which class index best matches this description?"
    )


def judge_caption(
    caption: str,
    label_vocabulary: list[str],
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
) -> str | None:
    """Returns one of `label_vocabulary`'s entries, or `None` if the caption is empty/whitespace
    or the API call fails (rate limit, network error, etc — the caller decides how to treat a
    `None`, same convention as every other predictor in `caption_to_label.py`: counted as
    wrong/unparsed downstream, never silently dropped).
    """
    if not caption or not caption.strip():
        return None

    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    schema = _answer_schema(len(label_vocabulary))
    config = GenerateContentConfig(
        system_instruction=_JUDGE_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=schema,
    )

    try:
        response = client.models.generate_content(
            model=model, contents=build_judge_prompt(caption, label_vocabulary), config=config,
        )
        parsed = response.parsed
        if parsed is None:
            return None
        answer = schema(**parsed) if isinstance(parsed, dict) else parsed
        return label_vocabulary[int(answer.classification)]
    except Exception as e:
        # Logged, not silently swallowed — a systematic failure (bad model name, bad API key,
        # every call rate-limited) must be visible as a stream of warnings, not just a
        # mysteriously 0% parse rate in the final report with no clue why.
        logger.warning(f"judge_caption failed: {type(e).__name__}: {e}")
        return None
