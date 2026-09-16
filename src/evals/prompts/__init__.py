from __future__ import annotations

from pathlib import Path
from typing import Any
import jinja2

PROMPTS_DIR = Path(__file__).resolve().parent

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,
)


def get_template(template_name: str) -> jinja2.Template:
    return _env.get_template(template_name)


def render_prompt(template_name: str, **kwargs: Any) -> str:
    return get_template(template_name).render(**kwargs).strip()
