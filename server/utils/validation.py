"""Validation utilities."""
from __future__ import annotations

from fastapi import HTTPException, status


def validate_texts(texts: list[str]) -> list[str]:
    """Validate list of texts and return cleaned list."""
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
    """Validate a single text input."""
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": ["text must not be empty"]},
        )
    return text.strip()
