from __future__ import annotations

from typing import Any

from .base import Task, parse_multi_choice, parse_single_choice
from .distance import DistanceTask
from .emission_lines import EmissionLineTask
from .source import SourceTask
from .subclass import SubclassTask

TASK_REGISTRY = {
    "distance": DistanceTask,
    "source": SourceTask,
    "subclass": SubclassTask,
    "emission_lines": EmissionLineTask,
}


def get_task(task_type: str, **kwargs: Any) -> Task:
    if task_type not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_type}'. Available tasks: {sorted(list(TASK_REGISTRY.keys()))}"
        )
    return TASK_REGISTRY[task_type](**kwargs)


__all__ = [
    "Task",
    "DistanceTask",
    "SourceTask",
    "SubclassTask",
    "EmissionLineTask",
    "get_task",
    "parse_single_choice",
    "parse_multi_choice",
]
