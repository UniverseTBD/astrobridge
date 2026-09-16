from __future__ import annotations

from typing import Any, Dict

from .base import FrontierModel, FrontierResponse
def get_frontier_model(config: Dict[str, Any]) -> FrontierModel:
    from .mock import MockFrontierModel

    model_type = config.get("frontier_type", "gemini")
    if model_type == "gemini":
        from .gemini import GeminiFrontierModel
        return GeminiFrontierModel(config)
    elif model_type == "mock":
        return MockFrontierModel(config)
    else:
        raise ValueError(
            f"Unknown frontier model type '{model_type}'. Available: ['gemini', 'mock']"
        )


def MockFrontierJudge(config: Dict[str, Any]) -> FrontierModel:
    from .mock import MockFrontierModel
    return MockFrontierModel(config)

get_frontier_judge = get_frontier_model

__all__ = [
    "FrontierModel",
    "FrontierResponse",
    "GeminiFrontierModel",
    "MockFrontierModel",
    "MockFrontierJudge",
    "get_frontier_model",
    "get_frontier_judge",
]
