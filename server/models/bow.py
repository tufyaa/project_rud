"""Bag-of-words model using NumPy."""
from __future__ import annotations

import numpy as np

from server.preprocessing.normalize import tokenize_many


def build_bow(texts: list[str], lower: bool = True, min_token_len: int = 2) -> dict[str, object]:
    """Build bag-of-words vocabulary and counts matrix."""
    tokenized = tokenize_many(texts, lower=lower, min_token_len=min_token_len)
    vocab = sorted({token for tokens in tokenized for token in tokens})
    vocab_index = {token: idx for idx, token in enumerate(vocab)}

    counts = np.zeros((len(texts), len(vocab)), dtype=np.int32)
    for doc_idx, tokens in enumerate(tokenized):
        for token in tokens:
            token_idx = vocab_index[token]
            counts[doc_idx, token_idx] += 1

    return {"vocabulary": vocab, "counts": counts}
