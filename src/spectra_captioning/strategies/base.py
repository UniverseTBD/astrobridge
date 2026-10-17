"""Abstract base class for captioning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

from spectra_captioning.models.gemini import GeminiResponse


@dataclass
class CaptionResult:
    """Result of generating a caption for a single object."""

    # The generated caption text.
    caption: str

    # Thought summaries from the model (may be empty).
    thought_summaries: list[str] = field(default_factory=list)

    # Token usage.
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int | None = None
    total_tokens: int = 0

    # The full prompt that was sent to the model.
    prompt_used: str = ""

    @classmethod
    def from_gemini_response(
        cls, response: GeminiResponse, prompt_used: str
    ) -> CaptionResult:
        """Create a CaptionResult from a GeminiResponse."""
        return cls(
            caption=response.text,
            thought_summaries=response.thought_summaries,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            thought_tokens=response.thought_tokens,
            total_tokens=response.total_tokens,
            prompt_used=prompt_used,
        )


class CaptionStrategy(ABC):
    """Abstract base class for captioning strategies.

    Each strategy defines how to generate a caption from an object's crossmatch
    data. Strategies are pluggable — new ones can be added by subclassing and
    registering in the strategy registry.
    """

    @abstractmethod
    def generate_caption(
        self, object_key: str, group_df: pd.DataFrame, dataset: str, config: dict
    ) -> CaptionResult:
        """Generate a caption for the given object.

        Args:
            object_key: Unique identifier for the object (e.g. wiki_entity_id).
            group_df: The subset of the crossmatch dataframe for this object.
            dataset: The dataset name (e.g., 'sdss' or 'desi').
            config: The full application configuration dictionary.

        Returns:
            A :class:`CaptionResult` with the generated caption.
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Unique identifier for this strategy (e.g. ``"quotes_only_v1"``)."""
        ...


# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------

_REGISTRY: dict[str, type[CaptionStrategy]] = {}


def register_strategy(cls: type[CaptionStrategy]) -> type[CaptionStrategy]:
    """Class decorator that registers a strategy by its name."""
    # Instantiate briefly to get the name.
    name = cls.strategy_name.fget(cls)  # type: ignore[attr-defined]
    # Fallback: use a temporary instance-less approach.
    if name is None:
        name = cls.__name__
    _REGISTRY[name] = cls
    return cls


def get_strategy(name: str) -> type[CaptionStrategy]:
    """Look up a registered strategy by name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY) or "(none)"
        raise ValueError(
            f"Unknown strategy {name!r}. Available: {available}"
        )
    return _REGISTRY[name]
