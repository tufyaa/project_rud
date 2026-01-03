from __future__ import annotations

from typing import Any
import numpy as np
from fastapi import HTTPException, status


def validate_texts(texts: list[str]) -> list[str]:
    errors: list[str] = []
    if not texts:
        errors.append("texts must not be empty")
    cleaned: list[str] = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            errors.append(f"text at index {idx} is empty")
        else:
            cleaned.append(text.strip())
    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"errors": errors})
    return cleaned


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": ["text must not be empty"]},
        )
    return text.strip()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return value.item()
    return value
