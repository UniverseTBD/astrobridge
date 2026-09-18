from __future__ import annotations

from typing import Any

__all__ = [
    "load_benchmark_dataset",
    "load_all_benchmark_spectra",
    "BenchmarkDataManager",
    "get_task",
    "Task",
    "get_responder",
    "SpectrumSample",
    "get_frontier_model",
    "FrontierModel",
    "compute_caption_metrics",
]


def __getattr__(name: str) -> Any:
    if name in ("load_benchmark_dataset", "load_all_benchmark_spectra", "BenchmarkDataManager"):
        from . import data
        return getattr(data, name)
    if name in ("get_task", "Task"):
        from . import tasks
        return getattr(tasks, name)
    if name in ("get_responder", "SpectrumSample"):
        from . import responders
        return getattr(responders, name)
    if name in ("get_frontier_model", "FrontierModel"):
        from . import frontier
        return getattr(frontier, name)
    if name == "compute_caption_metrics":
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
