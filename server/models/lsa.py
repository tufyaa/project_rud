"""Latent Semantic Analysis via TruncatedSVD."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD

from server.models.tfidf import build_tfidf


def build_lsa(
    texts: list[str],
    lower: bool = True,
    min_token_len: int = 2,
    n_components: int = 2,
    n_top_terms: int = 10,
) -> dict[str, object]:
    """Build LSA components and document embeddings."""
    tfidf_data = build_tfidf(
        texts,
        lower=lower,
        min_token_len=min_token_len,
        smooth_idf=True,
        normalize="l2",
    )
    tfidf = tfidf_data["tfidf"]
    vocab = tfidf_data["vocabulary"]

    n_components = min(n_components, tfidf.shape[1]) if tfidf.shape[1] > 0 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    doc_embeddings = svd.fit_transform(tfidf)
    components = svd.components_

    top_terms: list[list[str]] = []
    for comp in components:
        top_indices = np.argsort(np.abs(comp))[::-1][:n_top_terms]
        top_terms.append([vocab[idx] for idx in top_indices])

    return {
        "vocabulary": vocab,
        "doc_embeddings": doc_embeddings,
        "components": components,
        "top_terms": top_terms,
    }
