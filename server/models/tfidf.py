"""TF-IDF model using NumPy."""
from __future__ import annotations

import numpy as np

from server.models.bow import build_bow


def build_tfidf(
    texts: list[str],
    lower: bool = True,
    min_token_len: int = 2,
    smooth_idf: bool = True,
    normalize: str = "l2",
) -> dict[str, object]:
    """Build TF-IDF matrix and IDF values."""
    bow = build_bow(texts, lower=lower, min_token_len=min_token_len)
    counts = bow["counts"].astype(np.float64)
    vocab = bow["vocabulary"]

    doc_lengths = counts.sum(axis=1)
    doc_lengths[doc_lengths == 0] = 1.0
    tf = counts / doc_lengths[:, None]

    df = (counts > 0).sum(axis=0)
    n_docs = counts.shape[0]
    if smooth_idf:
        idf = np.log((1 + n_docs) / (1 + df)) + 1
    else:
        idf = np.log(n_docs / np.maximum(df, 1))

    tfidf = tf * idf
    if normalize == "l2":
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf = tfidf / norms

    return {"vocabulary": vocab, "tfidf": tfidf, "idf": idf}
