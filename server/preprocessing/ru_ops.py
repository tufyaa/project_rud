"""Russian-specific operations (lemmatization, NER)."""
from __future__ import annotations

from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
import pymorphy2

_segmenter = Segmenter()
_embedding = NewsEmbedding()
_ner_tagger = NewsNERTagger(_embedding)
_morph = pymorphy2.MorphAnalyzer()


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens with pymorphy2."""
    return [_morph.parse(token)[0].normal_form for token in tokens]


def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract named entities with Natasha."""
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_ner(_ner_tagger)
    entities: list[dict[str, str]] = []
    for span in doc.spans:
        entities.append({"text": span.text, "type": span.type})
    return entities
