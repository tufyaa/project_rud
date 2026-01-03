from __future__ import annotations

import re
import numpy as np
from sklearn.decomposition import TruncatedSVD

TOKEN_PATTERN = r"[A-Za-zА-Яа-яЁё]+"


def tokenize(text: str, lower: bool = True, min_token_len: int = 2) -> list[str]:
    tokens = list(re.findall(TOKEN_PATTERN, text))
    if lower:
        tokens = [token.lower() for token in tokens]
    if min_token_len > 1:
        tokens = [token for token in tokens if len(token) >= min_token_len]
    return tokens


def tokenize_many(texts: list[str], lower: bool = True, min_token_len: int = 2) -> list[list[str]]:
    return [tokenize(text, lower=lower, min_token_len=min_token_len) for text in texts]


def build_bow(texts: list[str], lower: bool = True, min_token_len: int = 2) -> dict[str, object]:
    tokenized = tokenize_many(texts, lower=lower, min_token_len=min_token_len)
    vocab = sorted({token for tokens in tokenized for token in tokens})
    vocab_index = {token: idx for idx, token in enumerate(vocab)}

    counts = np.zeros((len(texts), len(vocab)), dtype=np.int32)
    for doc_idx, tokens in enumerate(tokenized):
        for token in tokens:
            token_idx = vocab_index[token]
            counts[doc_idx, token_idx] += 1

    return {"vocabulary": vocab, "counts": counts}


def build_tfidf(
    texts: list[str],
    lower: bool = True,
    min_token_len: int = 2,
    smooth_idf: bool = True,
    normalize: str = "l2",
) -> dict[str, object]:
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


def build_lsa(
    texts: list[str],
    lower: bool = True,
    min_token_len: int = 2,
    n_components: int = 2,
    n_top_terms: int = 10,
) -> dict[str, object]:
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
