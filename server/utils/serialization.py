"""Serialization helpers."""
from __future__ import annotations

from typing import Any
import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert numpy types to JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return value.item()
    return value
