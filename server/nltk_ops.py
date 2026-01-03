from __future__ import annotations

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
import pymorphy2

_SEGMENTER = Segmenter()
_EMBEDDING = NewsEmbedding()
_NER_TAGGER = NewsNERTagger(_EMBEDDING)
_MORPH = pymorphy2.MorphAnalyzer()


def ensure_nltk_resources() -> None:
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
    ensure_nltk_resources()
    sentences = sent_tokenize(text, language="russian")
    tokens = word_tokenize(text, language="russian")
    return {"sentences": sentences, "tokens": tokens}


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = SnowballStemmer("russian")
    return [stemmer.stem(token) for token in tokens]


def pos_tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    ensure_nltk_resources()
    return nltk.pos_tag(tokens, lang="rus")


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [_MORPH.parse(token)[0].normal_form for token in tokens]


def extract_entities(text: str) -> list[dict[str, str]]:
    doc = Doc(text)
    doc.segment(_SEGMENTER)
    doc.tag_ner(_NER_TAGGER)
    entities: list[dict[str, str]] = []
    for span in doc.spans:
        entities.append({"text": span.text, "type": span.type})
    return entities
