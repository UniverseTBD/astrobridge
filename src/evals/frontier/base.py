from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


@dataclass
class FrontierResponse:
    raw_text: str
    parsed: Any
    forced_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrontierModel(Protocol):
    def predict(
        self,
        prompt: str,
        parse_fn: Callable[[str], Any],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> FrontierResponse:
        ...

    def predict_all(
        self,
        prompts: List[str],
        parse_fn: Callable[[str], Any] | List[Callable[[str], Any]],
        fallback_tag: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[FrontierResponse]:
        ...

    def get_config(self) -> Dict[str, Any]:
        ...
