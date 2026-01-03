"""NLTK operations for Russian text."""
from __future__ import annotations

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer


def ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are available."""
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "averaged_perceptron_tagger_ru": "taggers/averaged_perceptron_tagger_ru",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


def tokenize_text(text: str) -> dict[str, list[str]]:
    """Return sentences and tokens using NLTK Russian models."""
    ensure_nltk_resources()
    sentences = sent_tokenize(text, language="russian")
    tokens = word_tokenize(text, language="russian")
    return {"sentences": sentences, "tokens": tokens}


def stem_tokens(tokens: list[str]) -> list[str]:
    """Stem tokens with SnowballStemmer."""
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(token) for token in tokens]


def pos_tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Return POS tags for tokens using NLTK."""
    ensure_nltk_resources()
    return nltk.pos_tag(tokens, lang="rus")
