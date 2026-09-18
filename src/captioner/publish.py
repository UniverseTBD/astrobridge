"""Pure logic for scripts/06_publish_model.py, kept separate from the HF Hub / real-weights
orchestration so it's testable without a GPU or network access.
"""
from __future__ import annotations

import json


def filter_missing_lora_keys(missing: list[str]) -> list[str]:
    """`PeftModel.load_state_dict(..., strict=False)` returns every key it didn't find a match
    for — most of those are expected (base model buffers, etc). Only missing keys that are
    themselves LoRA parameters indicate the checkpoint's adapter didn't fully load.
    """
    return [k for k in missing if "lora_" in k]


def build_model_card(
    base_model_name: str,
    repo_id: str,
    state: dict,
    eval_report: dict | None,
    notes: str | None = None,
) -> str:
    """`notes` is free-form markdown appended under "Release notes" — for the things a JSON gate
    report cannot say, like "this version's test split is not comparable with the previous one's".
    Pass it with 06_publish_model.py's --notes-file.
    """
    eval_section = json.dumps(eval_report, indent=2) if eval_report else "Not yet run — see `make eval`."
    notes_section = f"\n## Release notes\n\n{notes.strip()}\n" if notes else ""
    return f"""---
base_model: {base_model_name}
tags: [astrobridge, captioner, lora, multimodal]
---

# AstroBridge Captioner

n-modality (image + spectra) astronomy captioner. LoRA adapter + fusion stack trained on top of
a frozen `{base_model_name}`. The base model itself is NOT included here — load it fresh from
`{base_model_name}` and apply this adapter on top.

## How to load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "{base_model_name}", dtype=torch.bfloat16, trust_remote_code=True
)
llm = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# middle.pt (fusion stack: projectors/modality_identity/qformer/adapter) needs the captioner
# package's FusionStack class to reload — see captioner/model/captioner.py and
# captioner/train/stage1.py's run_stage1 for how it's constructed and wired to the LLM.
```

## Training info
- config_hash: {state.get("config_hash")}
- quantization: {state.get("quantization")}
- git_sha: {state.get("git_sha")}
- tier_histogram: {json.dumps(state.get("tier_histogram"))}

## Eval (groundedness gate)
```json
{eval_section}
```
{notes_section}"""
