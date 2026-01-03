"""Normalization and tokenization utilities."""
from __future__ import annotations

import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def tokenize(text: str, lower: bool = True, min_token_len: int = 2) -> list[str]:
    """Tokenize text using regex for Latin/Cyrillic tokens."""
    tokens = TOKEN_PATTERN.findall(text)
    if lower:
        tokens = [token.lower() for token in tokens]
    if min_token_len > 1:
        tokens = [token for token in tokens if len(token) >= min_token_len]
    return tokens


def tokenize_many(texts: Iterable[str], lower: bool = True, min_token_len: int = 2) -> list[list[str]]:
    """Tokenize multiple texts."""
    return [tokenize(text, lower=lower, min_token_len=min_token_len) for text in texts]
